//! OpenAI Responses transport for ChatGPT OAuth and AWS Bedrock Mantle.
//!
//! ChatGPT credentials may come from Codex's `auth.json` or OpenClaw's
//! `auth-profiles.json`; expired tokens are refreshed and written atomically.
//! Bedrock Mantle uses a bearer token from the configured environment variable.

use anyhow::{anyhow, bail, Context, Result};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use fs2::FileExt;
use futures::StreamExt;
use genai::chat::{ChatMessage, ChatRole, ContentPart, MessageContent, Tool, ToolCall};
use reqwest::StatusCode;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, Mutex};

use crate::config::RetryConfig;
use crate::retry::{self, Retryable};

const DEFAULT_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const AUTH_CLAIM: &str = "https://api.openai.com/auth";

#[derive(Debug)]
enum ResponsesAuth {
    ChatgptOauth { credentials_file: PathBuf },
    BedrockBearer { env_var: String },
}

#[derive(Debug)]
pub struct OpenAIResponsesClient {
    http: reqwest::Client,
    base_url: String,
    auth: ResponsesAuth,
    retry: RetryConfig,
    credential_lock: Mutex<()>,
}

#[derive(Debug, Clone)]
struct Credentials {
    access: String,
    refresh: String,
    account_id: String,
    expires_ms: Option<u64>,
}

#[derive(Debug)]
pub struct ResponsesOutcome {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub output_items: Vec<Value>,
    pub input_tokens: Option<i32>,
    pub output_tokens: Option<i32>,
}

#[derive(Debug)]
pub enum ResponsesStreamEvent {
    TextDelta(String),
    ToolCall(ToolCall),
    OutputItem(Value),
    Completed {
        input_tokens: Option<i32>,
        output_tokens: Option<i32>,
    },
    Error(String),
}

impl OpenAIResponsesClient {
    pub fn chatgpt_oauth(credentials_file: PathBuf, base_url: Option<&str>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url
                .unwrap_or(DEFAULT_BASE_URL)
                .trim_end_matches('/')
                .to_string(),
            auth: ResponsesAuth::ChatgptOauth { credentials_file },
            retry: RetryConfig::default(),
            credential_lock: Mutex::new(()),
        }
    }

    pub fn bedrock_bearer(env_var: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            auth: ResponsesAuth::BedrockBearer {
                env_var: env_var.into(),
            },
            retry: RetryConfig::default(),
            credential_lock: Mutex::new(()),
        }
    }

    pub fn with_retry_config(mut self, retry: RetryConfig) -> Self {
        self.retry = retry;
        self
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn converse(
        &self,
        model: &str,
        messages: &[ChatMessage],
        continuation_items: &[Value],
        system_prompt: Option<&str>,
        tools: Option<&[Tool]>,
        reasoning: Option<&str>,
        service_tier: Option<&str>,
    ) -> Result<ResponsesOutcome> {
        let response = self
            .start_request(
                model,
                messages,
                continuation_items,
                system_prompt,
                tools,
                reasoning,
                service_tier,
            )
            .await?;
        let mut parser = CodexEventParser::default();
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        let mut output_items = Vec::new();
        let mut input_tokens = None;
        let mut output_tokens = None;
        let mut completed = false;
        let mut stream = response.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("reading OpenAI Responses stream")?;
            for event in parser.feed(&chunk)? {
                match event {
                    ResponsesStreamEvent::TextDelta(delta) => text.push_str(&delta),
                    ResponsesStreamEvent::ToolCall(call) => tool_calls.push(call),
                    ResponsesStreamEvent::OutputItem(item) => output_items.push(item),
                    ResponsesStreamEvent::Completed {
                        input_tokens: input,
                        output_tokens: output,
                    } => {
                        input_tokens = input;
                        output_tokens = output;
                        completed = true;
                    }
                    ResponsesStreamEvent::Error(error) => bail!("{error}"),
                }
            }
        }
        parser.finish()?;
        if !completed {
            bail!("OpenAI Responses stream ended before response.completed");
        }

        Ok(ResponsesOutcome {
            text,
            tool_calls,
            output_items,
            input_tokens,
            output_tokens,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn converse_stream(
        &self,
        model: &str,
        messages: &[ChatMessage],
        continuation_items: &[Value],
        system_prompt: Option<&str>,
        tools: Option<&[Tool]>,
        reasoning: Option<&str>,
        service_tier: Option<&str>,
        tx: mpsc::Sender<ResponsesStreamEvent>,
    ) {
        if let Err(error) = self
            .converse_stream_inner(
                model,
                messages,
                continuation_items,
                system_prompt,
                tools,
                reasoning,
                service_tier,
                &tx,
            )
            .await
        {
            let _ = tx
                .send(ResponsesStreamEvent::Error(format!("{error:#}")))
                .await;
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn converse_stream_inner(
        &self,
        model: &str,
        messages: &[ChatMessage],
        continuation_items: &[Value],
        system_prompt: Option<&str>,
        tools: Option<&[Tool]>,
        reasoning: Option<&str>,
        service_tier: Option<&str>,
        tx: &mpsc::Sender<ResponsesStreamEvent>,
    ) -> Result<()> {
        let response = self
            .start_request(
                model,
                messages,
                continuation_items,
                system_prompt,
                tools,
                reasoning,
                service_tier,
            )
            .await?;
        let mut parser = CodexEventParser::default();
        let mut completed = false;
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("reading OpenAI Responses stream")?;
            for event in parser.feed(&chunk)? {
                match &event {
                    ResponsesStreamEvent::Completed { .. } => completed = true,
                    ResponsesStreamEvent::Error(_) => {
                        let _ = tx.send(event).await;
                        return Ok(());
                    }
                    _ => {}
                }
                if tx.send(event).await.is_err() {
                    return Ok(());
                }
            }
        }
        parser.finish()?;
        if !completed {
            bail!("OpenAI Responses stream ended before response.completed");
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn start_request(
        &self,
        model: &str,
        messages: &[ChatMessage],
        continuation_items: &[Value],
        system_prompt: Option<&str>,
        tools: Option<&[Tool]>,
        reasoning: Option<&str>,
        service_tier: Option<&str>,
    ) -> Result<reqwest::Response> {
        let body = build_request_body(
            model,
            messages,
            continuation_items,
            system_prompt,
            tools,
            reasoning,
            service_tier,
        )?;
        let url = if self.base_url.ends_with("/responses") {
            self.base_url.clone()
        } else {
            format!("{}/responses", self.base_url)
        };
        let mut force_oauth_refresh = false;
        let mut oauth_refresh_attempted = false;
        let mut attempt = 1;
        loop {
            let mut request = self
                .http
                .post(&url)
                .header("Accept", "text/event-stream")
                .header("User-Agent", concat!("ai-cli/", env!("CARGO_PKG_VERSION")))
                .json(&body);
            request = match &self.auth {
                ResponsesAuth::ChatgptOauth { .. } => {
                    let credentials = self.credentials(force_oauth_refresh).await?;
                    force_oauth_refresh = false;
                    request
                        .bearer_auth(&credentials.access)
                        .header("chatgpt-account-id", &credentials.account_id)
                        .header("originator", "ai-cli")
                        .header("OpenAI-Beta", "responses=experimental")
                }
                ResponsesAuth::BedrockBearer { env_var } => {
                    let bearer = std::env::var(env_var).with_context(|| {
                        format!("Bedrock bearer environment variable {env_var} is not set")
                    })?;
                    request.bearer_auth(bearer)
                }
            };
            let response = match request.send().await {
                Ok(response) => response,
                Err(error) => {
                    if attempt < self.retry.max_attempts
                        && matches!(retry::classify_reqwest(&error), Retryable::Yes { .. })
                    {
                        let delay = retry::effective_delay(&self.retry, attempt, None);
                        eprintln!(
                            "[retry] {} attempt {}/{} after {:?}",
                            model, attempt, self.retry.max_attempts, delay
                        );
                        tokio::time::sleep(delay).await;
                        attempt += 1;
                        continue;
                    }
                    return Err(error).with_context(|| format!("POST {url}"));
                }
            };

            if response.status().is_success() {
                return Ok(response);
            }
            if response.status() == StatusCode::UNAUTHORIZED
                && matches!(self.auth, ResponsesAuth::ChatgptOauth { .. })
                && !oauth_refresh_attempted
            {
                oauth_refresh_attempted = true;
                force_oauth_refresh = true;
                continue;
            }
            let status = response.status();
            if retry::is_retryable_status(status) && attempt < self.retry.max_attempts {
                let retry_after = response
                    .headers()
                    .get(reqwest::header::RETRY_AFTER)
                    .and_then(|value| value.to_str().ok())
                    .and_then(retry::parse_retry_after);
                let delay = retry::effective_delay(&self.retry, attempt, retry_after);
                eprintln!(
                    "[retry] {} attempt {}/{} after {:?}",
                    model, attempt, self.retry.max_attempts, delay
                );
                tokio::time::sleep(delay).await;
                attempt += 1;
                continue;
            }
            let text = response.text().await.unwrap_or_default();
            return Err(responses_http_error(status, &text));
        }
    }

    async fn credentials(&self, force_refresh: bool) -> Result<Credentials> {
        let _guard = self.credential_lock.lock().await;
        let ResponsesAuth::ChatgptOauth { credentials_file } = &self.auth else {
            bail!("ChatGPT OAuth credentials requested for a non-OAuth client");
        };
        let _file_guard = lock_credential_file(credentials_file).await?;
        let document = read_auth_document(credentials_file)?;
        let mut credentials = extract_credentials(&document)?;
        let now_ms = unix_ms()?;
        if !credentials_need_refresh(credentials.expires_ms, now_ms, force_refresh) {
            return Ok(credentials);
        }

        let mut refreshed = match refresh_credentials(
            &self.http,
            TOKEN_URL,
            &credentials.refresh,
            &credentials.account_id,
            &self.retry,
            Some(credentials_file),
        )
        .await
        {
            Ok(refreshed) => refreshed,
            Err(refresh_error) => {
                let latest = extract_credentials(&read_auth_document(credentials_file)?)?;
                if latest.access == credentials.access && latest.refresh == credentials.refresh {
                    return Err(refresh_error);
                }
                let latest_now_ms = unix_ms()?;
                if !credentials_need_refresh(latest.expires_ms, latest_now_ms, false) {
                    return Ok(latest);
                }
                credentials = latest;
                refresh_credentials(
                    &self.http,
                    TOKEN_URL,
                    &credentials.refresh,
                    &credentials.account_id,
                    &self.retry,
                    Some(credentials_file),
                )
                .await
                .context("refreshing from concurrently updated OAuth credentials")?
            }
        };

        // Preserve edits made by other credential owners while the network
        // refresh was in flight. If they rotated the token set, use their
        // fresh data or refresh from their newer refresh token.
        let latest_document = read_auth_document(credentials_file)?;
        let latest = extract_credentials(&latest_document)?;
        if latest.access != credentials.access || latest.refresh != credentials.refresh {
            let latest_now_ms = unix_ms()?;
            let latest_is_fresh = match latest.expires_ms {
                None => true,
                Some(expires) => expires > latest_now_ms.saturating_add(60_000),
            };
            if latest_is_fresh {
                return Ok(latest);
            }
            credentials = latest;
            refreshed = refresh_credentials(
                &self.http,
                TOKEN_URL,
                &credentials.refresh,
                &credentials.account_id,
                &self.retry,
                Some(credentials_file),
            )
            .await?;
        }

        let mut before_write_document = read_auth_document(credentials_file)?;
        let before_write = extract_credentials(&before_write_document)?;
        if before_write.access != credentials.access || before_write.refresh != credentials.refresh
        {
            bail!("OAuth credentials changed during refresh; retry the request");
        }
        store_refreshed_oauth(&mut before_write_document, &refreshed)?;
        write_auth_document(credentials_file, &before_write_document)?;
        Ok(refreshed)
    }
}

fn credentials_need_refresh(expires_ms: Option<u64>, now_ms: u64, force_refresh: bool) -> bool {
    force_refresh || expires_ms.is_some_and(|expires| expires <= now_ms.saturating_add(60_000))
}

async fn lock_credential_file(path: &Path) -> Result<fs::File> {
    let parent = path
        .parent()
        .with_context(|| format!("OAuth credential path has no parent: {}", path.display()))?
        .to_path_buf();
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("oauth");
    let lock_path = parent.join(format!(".{name}.lock"));
    tokio::task::spawn_blocking(move || {
        fs::create_dir_all(parent)?;
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        options.mode(0o600);
        let file = options
            .open(&lock_path)
            .with_context(|| format!("opening OAuth credential lock {}", lock_path.display()))?;
        file.lock_exclusive()
            .with_context(|| format!("locking OAuth credentials {}", lock_path.display()))?;
        Ok(file)
    })
    .await
    .context("joining OAuth credential lock task")?
}

fn build_request_body(
    model: &str,
    messages: &[ChatMessage],
    continuation_items: &[Value],
    system_prompt: Option<&str>,
    tools: Option<&[Tool]>,
    reasoning: Option<&str>,
    service_tier: Option<&str>,
) -> Result<Value> {
    let (instructions, mut input) = convert_messages(messages, system_prompt)?;
    input.extend_from_slice(continuation_items);
    let mut body = json!({
        "model": model,
        "store": false,
        "stream": true,
        "instructions": instructions,
        "input": input,
        "text": { "verbosity": "low" },
        "include": ["reasoning.encrypted_content"],
        "tool_choice": "auto",
        "parallel_tool_calls": true
    });
    if let Some(reasoning) = reasoning {
        body["reasoning"] = json!({ "effort": reasoning, "summary": "auto" });
    }
    if let Some(service_tier) = service_tier {
        body["service_tier"] = json!(normalize_service_tier(service_tier)?);
    }
    if let Some(tools) = tools {
        body["tools"] = Value::Array(
            tools
                .iter()
                .map(|tool| {
                    json!({
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.schema.clone().unwrap_or_else(|| json!({
                            "type": "object",
                            "properties": {}
                        })),
                        "strict": false
                    })
                })
                .collect(),
        );
    }
    Ok(body)
}

const RESPONSES_ITEM_PREFIX: &str = "ai-cli:openai-responses-output-items:v1:";

pub(crate) fn assistant_message_with_output_items(
    text: &str,
    tool_calls: &[ToolCall],
    output_items: &[Value],
) -> ChatMessage {
    let mut parts = Vec::new();
    if !output_items.is_empty() {
        let encoded = serde_json::to_string(output_items)
            .expect("serde_json::Value output items always serialize");
        parts.push(ContentPart::ThoughtSignature(format!(
            "{RESPONSES_ITEM_PREFIX}{encoded}"
        )));
    }
    if !text.is_empty() {
        parts.push(ContentPart::Text(text.to_string()));
    }
    parts.extend(tool_calls.iter().cloned().map(ContentPart::ToolCall));
    ChatMessage::assistant(MessageContent::from_parts(parts))
}

fn stored_output_items(message: &ChatMessage) -> Result<Option<Vec<Value>>> {
    for part in message.content.parts() {
        let ContentPart::ThoughtSignature(signature) = part else {
            continue;
        };
        let Some(encoded) = signature.strip_prefix(RESPONSES_ITEM_PREFIX) else {
            continue;
        };
        let items =
            serde_json::from_str(encoded).context("parsing stored Responses output items")?;
        return Ok(Some(items));
    }
    Ok(None)
}

fn convert_messages(
    messages: &[ChatMessage],
    system_prompt: Option<&str>,
) -> Result<(String, Vec<Value>)> {
    let mut instructions = Vec::new();
    let mut seen_instructions = HashSet::new();
    if let Some(system) = system_prompt.filter(|s| !s.is_empty()) {
        seen_instructions.insert(system.to_string());
        instructions.push(system.to_string());
    }
    let mut input = Vec::new();
    let mut message_index = 0usize;

    for message in messages {
        if matches!(message.role, ChatRole::System) {
            let text = message.content.texts().join("");
            if !text.is_empty() && seen_instructions.insert(text.clone()) {
                instructions.push(text);
            }
            continue;
        }

        match message.role {
            ChatRole::User => {
                let text = message.content.texts().join("");
                if !text.is_empty() {
                    input.push(json!({
                        "role": "user",
                        "content": [{ "type": "input_text", "text": text }]
                    }));
                }
            }
            ChatRole::Assistant => {
                if let Some(items) = stored_output_items(message)? {
                    input.extend(items);
                } else {
                    let text = message.content.texts().join("");
                    if !text.is_empty() {
                        input.push(json!({
                            "type": "message",
                            "id": format!("msg_{message_index}"),
                            "role": "assistant",
                            "status": "completed",
                            "content": [{
                                "type": "output_text",
                                "text": text,
                                "annotations": []
                            }]
                        }));
                    }
                    for call in message.content.tool_calls() {
                        input.push(json!({
                            "type": "function_call",
                            "call_id": call.call_id,
                            "name": call.fn_name,
                            "arguments": serde_json::to_string(&call.fn_arguments)?
                        }));
                    }
                }
            }
            ChatRole::Tool => {
                for part in message.content.parts() {
                    if let ContentPart::ToolResponse(response) = part {
                        input.push(json!({
                            "type": "function_call_output",
                            "call_id": response.call_id,
                            "output": response.content
                        }));
                    }
                }
            }
            ChatRole::System => {}
        }
        message_index += 1;
    }
    Ok((instructions.join("\n\n"), input))
}

fn normalize_service_tier(value: &str) -> Result<&'static str> {
    match value.to_ascii_lowercase().as_str() {
        "fast" | "priority" => Ok("priority"),
        "default" | "auto" | "standard" => Ok("default"),
        "flex" => Ok("flex"),
        other => bail!("unsupported OpenAI service tier: {other}"),
    }
}

#[derive(Default)]
struct CodexEventParser {
    buffer: Vec<u8>,
}

impl CodexEventParser {
    fn feed(&mut self, bytes: &[u8]) -> Result<Vec<ResponsesStreamEvent>> {
        self.buffer.extend_from_slice(bytes);
        let mut events = Vec::new();
        while let Some((index, separator_len)) = next_sse_separator(&self.buffer) {
            let block = self.buffer[..index].to_vec();
            self.buffer.drain(..index + separator_len);
            let block =
                std::str::from_utf8(&block).context("OpenAI Responses SSE event was not UTF-8")?;
            let data = block
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(str::trim)
                .collect::<Vec<_>>()
                .join("\n");
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            let value: Value = serde_json::from_str(&data)
                .with_context(|| format!("parsing OpenAI Responses SSE event: {data}"))?;
            events.extend(parse_codex_event(&value)?);
        }
        Ok(events)
    }

    fn finish(&self) -> Result<()> {
        if self.buffer.iter().all(u8::is_ascii_whitespace) {
            Ok(())
        } else {
            bail!("OpenAI Responses stream ended with an incomplete SSE event")
        }
    }
}

fn next_sse_separator(buffer: &[u8]) -> Option<(usize, usize)> {
    let lf = buffer.windows(2).position(|window| window == b"\n\n");
    let crlf = buffer.windows(4).position(|window| window == b"\r\n\r\n");
    match (lf, crlf) {
        (Some(a), Some(b)) if a <= b => Some((a, 2)),
        (Some(_), Some(b)) => Some((b, 4)),
        (Some(a), None) => Some((a, 2)),
        (None, Some(b)) => Some((b, 4)),
        (None, None) => None,
    }
}

fn parse_codex_event(value: &Value) -> Result<Vec<ResponsesStreamEvent>> {
    match value["type"].as_str().unwrap_or_default() {
        "response.output_text.delta" | "response.refusal.delta" => Ok(value["delta"]
            .as_str()
            .map(|delta| vec![ResponsesStreamEvent::TextDelta(delta.to_string())])
            .unwrap_or_default()),
        "response.output_item.done" => {
            let item = &value["item"];
            let mut events = vec![ResponsesStreamEvent::OutputItem(item.clone())];
            if item["type"] == "function_call" {
                let arguments = item["arguments"].as_str().unwrap_or("{}");
                events.push(ResponsesStreamEvent::ToolCall(ToolCall {
                    call_id: item["call_id"].as_str().unwrap_or_default().to_string(),
                    fn_name: item["name"].as_str().unwrap_or_default().to_string(),
                    fn_arguments: serde_json::from_str(arguments)
                        .with_context(|| format!("parsing tool arguments: {arguments}"))?,
                    thought_signatures: None,
                }));
            }
            Ok(events)
        }
        "response.completed" | "response.done" => {
            let usage = &value["response"]["usage"];
            Ok(vec![ResponsesStreamEvent::Completed {
                input_tokens: usage["input_tokens"].as_i64().map(|v| v as i32),
                output_tokens: usage["output_tokens"].as_i64().map(|v| v as i32),
            }])
        }
        "response.incomplete" => {
            let reason = value["response"]["incomplete_details"]["reason"]
                .as_str()
                .unwrap_or("unknown reason");
            Ok(vec![ResponsesStreamEvent::Error(format!(
                "OpenAI Responses request was incomplete: {reason}"
            ))])
        }
        "response.failed" => {
            let message = value["response"]["error"]["message"]
                .as_str()
                .unwrap_or("OpenAI Responses request failed");
            Ok(vec![ResponsesStreamEvent::Error(message.to_string())])
        }
        "error" => {
            let message = value["message"]
                .as_str()
                .unwrap_or("OpenAI Responses error");
            Ok(vec![ResponsesStreamEvent::Error(message.to_string())])
        }
        _ => Ok(Vec::new()),
    }
}

fn read_auth_document(path: &Path) -> Result<Value> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("reading OAuth credentials {}", path.display()))?;
    serde_json::from_str(&text)
        .with_context(|| format!("parsing OAuth credentials {}", path.display()))
}

fn extract_credentials(document: &Value) -> Result<Credentials> {
    if let Some(key) = openclaw_profile_key(document)? {
        let profile = &document["profiles"][key];
        return credentials_from_fields(
            profile["access"].as_str(),
            profile["refresh"].as_str(),
            profile["accountId"].as_str(),
            profile["expires"].as_u64(),
        );
    }

    let tokens = &document["tokens"];
    credentials_from_fields(
        tokens["access_token"].as_str(),
        tokens["refresh_token"].as_str(),
        tokens["account_id"].as_str(),
        tokens["access_token"].as_str().and_then(jwt_expiration_ms),
    )
}

fn openclaw_profile_key(document: &Value) -> Result<Option<String>> {
    let Some(profiles) = document["profiles"].as_object() else {
        return Ok(None);
    };
    let matches: Vec<&String> = profiles
        .iter()
        .filter_map(|(key, profile)| (profile["provider"] == "openai-codex").then_some(key))
        .collect();
    match matches.as_slice() {
        [] => bail!("no openai-codex OAuth profile found"),
        [key] => Ok(Some((*key).clone())),
        _ => bail!(
            "multiple openai-codex OAuth profiles found; use a credentials file with one profile"
        ),
    }
}

fn credentials_from_fields(
    access: Option<&str>,
    refresh: Option<&str>,
    account_id: Option<&str>,
    expires_ms: Option<u64>,
) -> Result<Credentials> {
    let access = access.context("OAuth access token is missing")?.to_string();
    let refresh = refresh
        .context("OAuth refresh token is missing")?
        .to_string();
    let account_id = account_id
        .map(str::to_string)
        .or_else(|| account_id_from_jwt(&access))
        .context("ChatGPT account ID is missing")?;
    let expires_ms = expires_ms.or_else(|| jwt_expiration_ms(&access));
    Ok(Credentials {
        access,
        refresh,
        account_id,
        expires_ms,
    })
}

async fn refresh_credentials(
    http: &reqwest::Client,
    token_url: &str,
    refresh: &str,
    fallback_account_id: &str,
    retry_config: &RetryConfig,
    credentials_file: Option<&Path>,
) -> Result<Credentials> {
    let mut attempt = 1;
    let body = loop {
        let response = match http
            .post(token_url)
            .form(&[
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh),
                ("client_id", CLIENT_ID),
            ])
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) => {
                if attempt < retry_config.max_attempts && error.is_connect() {
                    let delay = retry::effective_delay(retry_config, attempt, None);
                    eprintln!(
                        "[retry] ChatGPT OAuth refresh attempt {}/{} after {:?}",
                        attempt, retry_config.max_attempts, delay
                    );
                    tokio::time::sleep(delay).await;
                    ensure_refresh_token_unchanged(credentials_file, refresh)?;
                    attempt += 1;
                    continue;
                }
                return Err(error).context("refreshing ChatGPT OAuth token");
            }
        };
        let status = response.status();
        if is_retryable_oauth_refresh_status(status) && attempt < retry_config.max_attempts {
            let retry_after = response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
                .and_then(retry::parse_retry_after);
            let delay = retry::effective_delay(retry_config, attempt, retry_after);
            eprintln!(
                "[retry] ChatGPT OAuth refresh attempt {}/{} after {:?}",
                attempt, retry_config.max_attempts, delay
            );
            tokio::time::sleep(delay).await;
            ensure_refresh_token_unchanged(credentials_file, refresh)?;
            attempt += 1;
            continue;
        }
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(responses_http_error(status, &body))
                .context("refreshing ChatGPT OAuth token");
        }
        break body;
    };
    let value: Value = serde_json::from_str(&body).context("parsing OAuth refresh response")?;
    let access = value["access_token"]
        .as_str()
        .context("OAuth refresh response omitted access_token")?
        .to_string();
    let refresh = value["refresh_token"]
        .as_str()
        .unwrap_or(refresh)
        .to_string();
    let expires_ms = value["expires_in"]
        .as_u64()
        .map(|seconds| unix_ms().unwrap_or_default() + seconds * 1_000)
        .or_else(|| jwt_expiration_ms(&access));
    let account_id =
        account_id_from_jwt(&access).unwrap_or_else(|| fallback_account_id.to_string());
    Ok(Credentials {
        access,
        refresh,
        account_id,
        expires_ms,
    })
}

fn is_retryable_oauth_refresh_status(status: StatusCode) -> bool {
    // Refresh grants may rotate the token. Avoid replaying after ambiguous
    // timeout/gateway failures where the first request may have succeeded.
    matches!(status.as_u16(), 425 | 429)
}

fn ensure_refresh_token_unchanged(credentials_file: Option<&Path>, expected: &str) -> Result<()> {
    let Some(path) = credentials_file else {
        return Ok(());
    };
    let latest = extract_credentials(&read_auth_document(path)?)?;
    if latest.refresh != expected {
        bail!("credentials changed during OAuth refresh backoff");
    }
    Ok(())
}

fn store_refreshed_oauth(document: &mut Value, refreshed: &Credentials) -> Result<()> {
    if let Some(key) = openclaw_profile_key(document)? {
        let profile = document["profiles"]
            .get_mut(&key)
            .context("openai-codex OAuth profile disappeared")?;
        profile["access"] = json!(refreshed.access);
        profile["refresh"] = json!(refreshed.refresh);
        profile["accountId"] = json!(refreshed.account_id);
        if let Some(expires) = refreshed.expires_ms {
            profile["expires"] = json!(expires);
        }
        return Ok(());
    }
    document["auth_mode"] = json!("chatgpt");
    document["tokens"]["access_token"] = json!(refreshed.access);
    document["tokens"]["refresh_token"] = json!(refreshed.refresh);
    document["tokens"]["account_id"] = json!(refreshed.account_id);
    Ok(())
}

fn write_auth_document(path: &Path, document: &Value) -> Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("OAuth credential path has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".{}.tmp-{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("auth"),
        std::process::id()
    ));
    let result = (|| -> Result<()> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        let mut file = options
            .open(&temporary)
            .with_context(|| format!("creating {}", temporary.display()))?;
        serde_json::to_writer_pretty(&mut file, document)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn account_id_from_jwt(token: &str) -> Option<String> {
    let payload = decode_jwt_payload(token)?;
    payload[AUTH_CLAIM]["chatgpt_account_id"]
        .as_str()
        .map(str::to_string)
}

fn jwt_expiration_ms(token: &str) -> Option<u64> {
    decode_jwt_payload(token)?["exp"]
        .as_u64()
        .map(|seconds| seconds * 1_000)
}

fn decode_jwt_payload(token: &str) -> Option<Value> {
    let payload = token.split('.').nth(1)?;
    let decoded = URL_SAFE_NO_PAD.decode(payload).ok()?;
    serde_json::from_slice(&decoded).ok()
}

fn unix_ms() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system time is before Unix epoch")?
        .as_millis() as u64)
}

fn responses_http_error(status: StatusCode, body: &str) -> anyhow::Error {
    let message = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|value| value["error"]["message"].as_str().map(str::to_string))
        .unwrap_or_else(|| body.chars().take(500).collect());
    anyhow!(
        "OpenAI Responses endpoint returned HTTP {}: {}",
        status.as_u16(),
        message
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RetryConfig;
    use genai::chat::{MessageContent, ToolResponse};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[test]
    fn service_tier_aliases_map_to_priority() {
        assert_eq!(normalize_service_tier("fast").unwrap(), "priority");
        assert_eq!(normalize_service_tier("priority").unwrap(), "priority");
        assert!(normalize_service_tier("express").is_err());
    }

    #[test]
    fn request_preserves_max_reasoning() {
        let body = build_request_body(
            "gpt-5.6-sol",
            &[ChatMessage::user("hello")],
            &[],
            None,
            None,
            Some("max"),
            Some("fast"),
        )
        .unwrap();
        assert_eq!(body["reasoning"]["effort"], "max");
        assert_eq!(body["service_tier"], "priority");
    }

    #[test]
    fn request_serializes_tool_strictness_as_boolean() {
        let body = build_request_body(
            "gpt-5.6-sol",
            &[ChatMessage::user("hello")],
            &[],
            None,
            Some(&[crate::search::web_search_tool()]),
            Some("max"),
            None,
        )
        .unwrap();
        assert_eq!(body["tools"][0]["strict"], false);
    }

    #[test]
    fn forced_refresh_handles_tokens_without_expiry() {
        assert!(!credentials_need_refresh(None, 1_000, false));
        assert!(credentials_need_refresh(None, 1_000, true));
        assert!(credentials_need_refresh(Some(61_000), 1_000, false));
        assert!(!credentials_need_refresh(Some(61_001), 1_000, false));
    }

    #[test]
    fn ambiguous_openclaw_profiles_are_rejected() {
        let document = json!({
            "profiles": {
                "first": { "provider": "openai-codex" },
                "second": { "provider": "openai-codex" }
            }
        });
        assert!(openclaw_profile_key(&document)
            .unwrap_err()
            .to_string()
            .contains("multiple openai-codex"));
    }

    #[test]
    fn sse_parser_handles_split_chunks() {
        let mut parser = CodexEventParser::default();
        assert!(parser
            .feed(b"data: {\"type\":\"response.output_text.")
            .unwrap()
            .is_empty());
        let events = parser.feed(b"delta\",\"delta\":\"ok\"}\n\n").unwrap();
        assert!(matches!(
            events.as_slice(),
            [ResponsesStreamEvent::TextDelta(text)] if text == "ok"
        ));
    }

    #[test]
    fn sse_parser_preserves_utf8_split_inside_character() {
        let event = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"🙂\"}\n\n";
        let emoji_start = event.find('🙂').unwrap();
        let bytes = event.as_bytes();
        let mut parser = CodexEventParser::default();
        assert!(parser.feed(&bytes[..emoji_start + 2]).unwrap().is_empty());
        let events = parser.feed(&bytes[emoji_start + 2..]).unwrap();
        assert!(matches!(
            events.as_slice(),
            [ResponsesStreamEvent::TextDelta(text)] if text == "🙂"
        ));
    }

    #[test]
    fn sse_parser_surfaces_incomplete_responses() {
        let mut parser = CodexEventParser::default();
        let events = parser
            .feed(
                b"data: {\"type\":\"response.incomplete\",\"response\":{\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n\n",
            )
            .unwrap();
        assert!(matches!(
            events.as_slice(),
            [ResponsesStreamEvent::Error(message)] if message.contains("max_output_tokens")
        ));
    }

    #[test]
    fn sse_parser_preserves_refusal_text() {
        let mut parser = CodexEventParser::default();
        let events = parser
            .feed(b"data: {\"type\":\"response.refusal.delta\",\"delta\":\"cannot comply\"}\n\n")
            .unwrap();
        assert!(matches!(
            events.as_slice(),
            [ResponsesStreamEvent::TextDelta(text)] if text == "cannot comply"
        ));
    }

    #[test]
    fn sse_parser_rejects_unterminated_event_at_eof() {
        let mut parser = CodexEventParser::default();
        parser
            .feed(b"data: {\"type\":\"response.output_text.delta\"")
            .unwrap();
        assert!(parser.finish().is_err());
    }

    #[test]
    fn request_appends_raw_continuation_items_in_order() {
        let reasoning = json!({
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "opaque"
        });
        let tool_output = json!({
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "result"
        });
        let body = build_request_body(
            "gpt-5.6-sol",
            &[ChatMessage::user("hello")],
            &[reasoning.clone(), tool_output.clone()],
            None,
            None,
            Some("max"),
            None,
        )
        .unwrap();
        let input = body["input"].as_array().unwrap();
        assert_eq!(input[input.len() - 2], reasoning);
        assert_eq!(input[input.len() - 1], tool_output);
    }

    #[test]
    fn follow_up_replays_stored_output_items_verbatim() {
        let reasoning = json!({
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "opaque"
        });
        let function_call = json!({
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "web_search",
            "arguments": "{\"query\":\"rust\"}"
        });
        let message_item = json!({
            "type": "message",
            "id": "msg_real",
            "role": "assistant",
            "status": "completed",
            "content": [{
                "type": "output_text",
                "text": "done",
                "annotations": []
            }]
        });
        let tool_turn = assistant_message_with_output_items(
            "searching",
            &[ToolCall {
                call_id: "call_1".to_string(),
                fn_name: "web_search".to_string(),
                fn_arguments: json!({"query": "rust"}),
                thought_signatures: None,
            }],
            &[reasoning.clone(), function_call.clone()],
        );
        let answer = assistant_message_with_output_items(
            "done",
            &[],
            &[reasoning.clone(), message_item.clone()],
        );
        let body = build_request_body(
            "gpt-5.6-sol",
            &[
                ChatMessage::user("hello"),
                tool_turn,
                ChatMessage::from(ToolResponse::new("call_1", "result")),
                answer,
                ChatMessage::user("follow up"),
            ],
            &[],
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let input = body["input"].as_array().unwrap();
        assert_eq!(
            input,
            &vec![
                json!({
                    "role": "user",
                    "content": [{ "type": "input_text", "text": "hello" }]
                }),
                reasoning.clone(),
                function_call,
                json!({
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "result"
                }),
                reasoning,
                message_item,
                json!({
                    "role": "user",
                    "content": [{ "type": "input_text", "text": "follow up" }]
                }),
            ]
        );
    }

    #[test]
    fn malformed_stored_output_items_are_rejected() {
        let message = ChatMessage::assistant(MessageContent::from_parts(vec![
            ContentPart::ThoughtSignature(format!("{RESPONSES_ITEM_PREFIX}not-json")),
            ContentPart::Text("visible".to_string()),
        ]));
        let error =
            build_request_body("gpt-5.6-sol", &[message], &[], None, None, None, None).unwrap_err();
        assert!(error.to_string().contains("stored Responses output items"));
    }

    #[tokio::test]
    async fn transient_http_status_is_retried_before_streaming() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            for attempt in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = vec![0; 8192];
                let _ = socket.read(&mut request).await.unwrap();
                if attempt == 0 {
                    socket
                        .write_all(
                            b"HTTP/1.1 429 Too Many Requests\r\nContent-Length: 0\r\nRetry-After: 0\r\nConnection: close\r\n\r\n",
                        )
                        .await
                        .unwrap();
                } else {
                    let body = concat!(
                        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n",
                        "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n"
                    );
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    socket.write_all(response.as_bytes()).await.unwrap();
                }
            }
        });
        let client =
            OpenAIResponsesClient::bedrock_bearer("PATH", format!("http://{address}/openai/v1"))
                .with_retry_config(RetryConfig {
                    max_attempts: 2,
                    base_ms: 1,
                    max_delay_ms: 1,
                });

        let outcome = client
            .converse(
                "openai.gpt-5.6-sol",
                &[ChatMessage::user("hello")],
                &[],
                None,
                None,
                Some("high"),
                None,
            )
            .await
            .unwrap();
        server.await.unwrap();
        assert_eq!(outcome.text, "ok");
        assert_eq!(outcome.input_tokens, Some(1));
        assert_eq!(outcome.output_tokens, Some(1));
    }

    #[tokio::test]
    async fn transient_oauth_refresh_status_is_retried() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            for attempt in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = vec![0; 8192];
                let _ = socket.read(&mut request).await.unwrap();
                if attempt == 0 {
                    socket
                        .write_all(
                            b"HTTP/1.1 429 Too Many Requests\r\nContent-Length: 0\r\nRetry-After: 0\r\nConnection: close\r\n\r\n",
                        )
                        .await
                        .unwrap();
                } else {
                    let body = r#"{"access_token":"new-access","refresh_token":"new-refresh","expires_in":3600}"#;
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    socket.write_all(response.as_bytes()).await.unwrap();
                }
            }
        });
        let retry = RetryConfig {
            max_attempts: 2,
            base_ms: 1,
            max_delay_ms: 1,
        };

        let credentials = refresh_credentials(
            &reqwest::Client::new(),
            &format!("http://{address}/oauth/token"),
            "old-refresh",
            "account",
            &retry,
            None,
        )
        .await
        .unwrap();
        server.await.unwrap();
        assert_eq!(credentials.access, "new-access");
        assert_eq!(credentials.refresh, "new-refresh");
        assert_eq!(credentials.account_id, "account");
    }

    #[test]
    fn oauth_refresh_retries_only_unambiguous_statuses() {
        for code in [425, 429] {
            assert!(is_retryable_oauth_refresh_status(
                StatusCode::from_u16(code).unwrap()
            ));
        }
        for code in [408, 500, 502, 503, 504] {
            assert!(!is_retryable_oauth_refresh_status(
                StatusCode::from_u16(code).unwrap()
            ));
        }
    }

    #[tokio::test]
    async fn oauth_refresh_stops_if_credentials_rotate_during_backoff() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        use tokio::time::{sleep, timeout, Duration};

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let request_count = Arc::new(AtomicUsize::new(0));
        let server_count = Arc::clone(&request_count);
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            server_count.fetch_add(1, Ordering::SeqCst);
            let mut request = vec![0; 8192];
            let _ = socket.read(&mut request).await.unwrap();
            socket
                .write_all(
                    b"HTTP/1.1 429 Too Many Requests\r\nContent-Length: 0\r\nRetry-After: 1\r\nConnection: close\r\n\r\n",
                )
                .await
                .unwrap();
            if timeout(Duration::from_millis(200), listener.accept())
                .await
                .is_ok()
            {
                server_count.fetch_add(1, Ordering::SeqCst);
            }
        });

        let directory = std::env::temp_dir().join(format!(
            "ai-cli-oauth-retry-{}-{}",
            std::process::id(),
            unix_ms().unwrap()
        ));
        fs::create_dir_all(&directory).unwrap();
        let credentials_file = directory.join("auth.json");
        fs::write(
            &credentials_file,
            serde_json::to_vec(&json!({
                "tokens": {
                    "access_token": "old-access",
                    "refresh_token": "old-refresh",
                    "account_id": "account"
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let writer_path = credentials_file.clone();
        let writer = tokio::spawn(async move {
            sleep(Duration::from_millis(10)).await;
            fs::write(
                writer_path,
                serde_json::to_vec(&json!({
                    "tokens": {
                        "access_token": "new-access",
                        "refresh_token": "new-refresh",
                        "account_id": "account"
                    }
                }))
                .unwrap(),
            )
            .unwrap();
        });
        let retry = RetryConfig {
            max_attempts: 2,
            base_ms: 1,
            max_delay_ms: 50,
        };

        let error = refresh_credentials(
            &reqwest::Client::new(),
            &format!("http://{address}/oauth/token"),
            "old-refresh",
            "account",
            &retry,
            Some(&credentials_file),
        )
        .await
        .unwrap_err();
        writer.await.unwrap();
        server.await.unwrap();
        fs::remove_dir_all(directory).unwrap();
        assert!(error
            .to_string()
            .contains("credentials changed during OAuth refresh backoff"));
        assert_eq!(request_count.load(Ordering::SeqCst), 1);
    }
}
