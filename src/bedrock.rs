//! Direct client for AWS Bedrock's native Converse / ConverseStream APIs.
//!
//! genai's `AdapterKind::Anthropic` speaks Anthropic's native wire protocol
//! (POST `{base_url}/messages`). Bedrock does not serve that protocol — a
//! mismatched request yields HTTP 200 with `UnknownOperationException`,
//! which presents as a silent empty response. This module bypasses genai
//! entirely for Bedrock-hosted Claude models by calling
//! `POST /model/{id}/converse[-stream]` directly with bearer-token auth
//! (`AWS_BEARER_TOKEN_BEDROCK`).
//!
//! Streaming uses AWS's binary Event Stream framing — the frame decoder
//! lives in `bedrock_eventstream`; this module converts those raw frames
//! into `BedrockStreamEvent`s (TextDelta, ToolUseStart, ToolUseInputDelta,
//! ContentBlockStop, MessageStop, Metadata).
use crate::bedrock_eventstream::{Frame, FrameDecoder};
use anyhow::{anyhow, bail, Context, Result};
use futures::stream::StreamExt;
use genai::chat::{ChatMessage, ChatRole, ContentPart, MessageContent, Tool, ToolCall};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

/// Detect whether a base_url points at AWS Bedrock. Used by the provider
/// layer to decide between genai and this direct client — and therefore
/// whether the Bedrock bearer token gets sent, so match the *host* rather
/// than substrings anywhere in the URL. Covers standard, FIPS
/// (`bedrock-runtime-fips.`), GovCloud, China (`.amazonaws.com.cn`) and
/// PrivateLink (`vpce-….bedrock-runtime.<region>.vpce.amazonaws.com`)
/// endpoints: some DNS label must be exactly `bedrock-runtime[-fips]` and
/// the host must be under an AWS domain.
pub fn is_bedrock_base_url(url: &str) -> bool {
    let Ok(parsed) = reqwest::Url::parse(url) else {
        return false;
    };
    let Some(host) = parsed.host_str() else {
        return false;
    };
    let host = host.to_ascii_lowercase();
    let aws_domain = host.ends_with(".amazonaws.com") || host.ends_with(".amazonaws.com.cn");
    aws_domain
        && host
            .split('.')
            .any(|label| label == "bedrock-runtime" || label == "bedrock-runtime-fips")
}

/// Extract the Bedrock host (scheme + authority only) from a possibly
/// path-suffixed base_url. E.g.
/// "https://bedrock-runtime.us-east-1.amazonaws.com/v1/" ->
/// "https://bedrock-runtime.us-east-1.amazonaws.com".
fn bedrock_host(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if let Some(scheme_end) = trimmed.find("://") {
        let rest = &trimmed[scheme_end + 3..];
        let host_end = rest.find('/').unwrap_or(rest.len());
        format!("{}{}", &trimmed[..scheme_end + 3], &rest[..host_end])
    } else {
        trimmed.to_string()
    }
}

pub struct BedrockClient {
    http: reqwest::Client,
    host: String,
    auth_token: String,
}

impl BedrockClient {
    /// Build a client from a base_url and the env var holding the bearer token.
    /// Returns `None` if the env var is unset, so the caller can fall back.
    pub fn from_env(base_url: &str, env_var: &str) -> Option<Self> {
        let auth_token = std::env::var(env_var).ok()?;
        if auth_token.is_empty() {
            return None;
        }
        // Without timeouts a stalled connection hangs forever on the paths
        // that have no cancel token (non-streaming and one-shot streaming).
        // The read timeout is per read, so long streamed answers are fine
        // as long as bytes keep arriving; it's generous because a
        // non-streaming Converse call sends nothing until it's done.
        let http = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(15))
            .read_timeout(std::time::Duration::from_secs(300))
            .build()
            .ok()?;
        Some(Self {
            http,
            host: bedrock_host(base_url),
            auth_token,
        })
    }

    /// Invoke the Converse API (non-streaming) and return the assistant's
    /// text, any tool calls, and usage.
    pub async fn converse(
        &self,
        model_id: &str,
        messages: &[ChatMessage],
        system_prompt: Option<&str>,
        tools: Option<&[Tool]>,
    ) -> Result<BedrockOutcome> {
        let body = build_converse_request(messages, system_prompt, tools)?;

        let url = format!(
            "{}/model/{}/converse",
            self.host,
            urlencode_path_segment(model_id)
        );

        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.auth_token)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("POST {}", url))?;

        if !resp.status().is_success() {
            return Err(http_error(model_id, resp).await);
        }

        let text = resp
            .text()
            .await
            .with_context(|| "reading Bedrock response body")?;

        let parsed: ConverseResponse = serde_json::from_str(&text).with_context(|| {
            format!(
                "parsing Bedrock Converse response (first 200 chars: {:?})",
                text.chars().take(200).collect::<String>()
            )
        })?;

        let mut assistant_text = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        for block in &parsed.output.message.content {
            if let Some(t) = block.text.as_deref() {
                assistant_text.push_str(t);
            }
            if let Some(tu) = &block.tool_use {
                tool_calls.push(ToolCall {
                    call_id: tu.tool_use_id.clone(),
                    fn_name: tu.name.clone(),
                    fn_arguments: tu.input.clone(),
                    thought_signatures: None,
                });
            }
        }

        Ok(BedrockOutcome {
            text: assistant_text,
            tool_calls,
            input_tokens: parsed.usage.as_ref().and_then(ConverseUsage::total_input),
            output_tokens: parsed.usage.as_ref().and_then(|u| u.output_tokens),
            stop_reason: parsed.stop_reason,
        })
    }

    /// Invoke the ConverseStream API and emit semantic events over a
    /// channel. Returns when the stream ends. Request-level failures (HTTP
    /// status, transport, framing) come back as `Err` so the caller can
    /// decide whether to retry; server-side exception frames that arrive
    /// mid-stream also end it as a typed `Err` ([`BedrockStreamException`]).
    pub async fn converse_stream(
        &self,
        model_id: &str,
        messages: &[ChatMessage],
        system_prompt: Option<&str>,
        tools: Option<&[Tool]>,
        tx: mpsc::Sender<BedrockStreamEvent>,
    ) -> Result<()> {
        let body = build_converse_request(messages, system_prompt, tools)?;

        let url = format!(
            "{}/model/{}/converse-stream",
            self.host,
            urlencode_path_segment(model_id)
        );

        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.auth_token)
            .header("Accept", "application/vnd.amazon.eventstream")
            .json(&body)
            .send()
            .await
            .with_context(|| format!("POST {}", url))?;

        if !resp.status().is_success() {
            return Err(http_error(model_id, resp).await);
        }

        let mut decoder = FrameDecoder::new();
        let mut body_stream = resp.bytes_stream();
        let mut saw_message_stop = false;

        while let Some(chunk) = body_stream.next().await {
            let chunk = chunk.with_context(|| "reading Bedrock ConverseStream response body")?;
            decoder.feed(&chunk);

            // Exception/error frames come back as Err (typed, so the caller
            // can retry transient ones like throttling) and end the stream.
            while let Some(frame) = decoder.next_frame()? {
                for ev in frame_to_events(frame)? {
                    saw_message_stop |= matches!(ev, BedrockStreamEvent::MessageStop { .. });
                    // If the receiver dropped, stop reading more bytes.
                    if tx.send(ev).await.is_err() {
                        return Ok(());
                    }
                }
            }
        }

        // A connection closed early (proxy timeout, network drop) ends the
        // body cleanly from reqwest's point of view. Without this check a
        // half-finished answer would be reported as complete. Once
        // messageStop has arrived the answer *is* complete; a partial frame
        // after it can only be the trailing usage metadata, so it's dropped
        // rather than failing a finished answer.
        if !saw_message_stop {
            return Err(anyhow::Error::new(BedrockStreamTruncated {
                mid_frame: !decoder.is_empty(),
            }));
        }
        Ok(())
    }

    /// GET a path on the Bedrock *control plane* of this client's region
    /// (`bedrock.<region>…` rather than `bedrock-runtime.<region>…`), with
    /// the same bearer token, and parse the JSON body. Used for the model
    /// catalog (`/foundation-models`, `/inference-profiles`).
    pub async fn control_plane_get(&self, path_and_query: &str) -> Result<Value> {
        let Some(host) = control_plane_host(&self.host) else {
            bail!(
                "can't derive the Bedrock control-plane endpoint from {}",
                self.host
            );
        };
        let url = format!("{}{}", host, path_and_query);
        let resp = self
            .http
            .get(&url)
            .bearer_auth(&self.auth_token)
            // The catalog is small; don't let a stalled endpoint hang the
            // model picker for the 300s read timeout.
            .timeout(std::time::Duration::from_secs(20))
            .send()
            .await
            .with_context(|| format!("GET {}", url))?;
        if !resp.status().is_success() {
            return Err(http_error("catalog", resp).await);
        }
        resp.json::<Value>()
            .await
            .with_context(|| format!("parsing JSON from {}", url))
    }
}

/// The control-plane origin for a runtime origin: the label
/// `bedrock-runtime[-fips]` becomes `bedrock[-fips]`, so
/// `https://bedrock-runtime.us-east-1.amazonaws.com` →
/// `https://bedrock.us-east-1.amazonaws.com`. `None` for hosts with no such
/// label (a PrivateLink endpoint serves only the runtime API).
fn control_plane_host(runtime_origin: &str) -> Option<String> {
    let (scheme, host) = runtime_origin.split_once("://")?;
    if host.split('.').next()?.starts_with("vpce-") {
        return None;
    }
    let mut replaced = false;
    let labels: Vec<&str> = host
        .split('.')
        .map(|label| match label {
            "bedrock-runtime" if !replaced => {
                replaced = true;
                "bedrock"
            }
            "bedrock-runtime-fips" if !replaced => {
                replaced = true;
                "bedrock-fips"
            }
            other => other,
        })
        .collect();
    replaced.then(|| format!("{}://{}", scheme, labels.join(".")))
}

/// Human-readable note for stop reasons that mean the answer is incomplete.
/// `end_turn`, `tool_use` and `stop_sequence` are normal and return `None`.
pub fn stop_reason_note(stop_reason: Option<&str>) -> Option<&'static str> {
    match stop_reason? {
        "max_tokens" => Some("response truncated: the model hit its output token limit"),
        "model_context_window_exceeded" => {
            Some("response truncated: the conversation exceeded the model's context window")
        }
        "guardrail_intervened" => Some("response stopped by a Bedrock guardrail"),
        "content_filtered" => Some("response stopped by content filtering"),
        _ => None,
    }
}

/// A server-side exception delivered inside a ConverseStream (the HTTP
/// status was 200). Typed so the retry layer can retry the transient kinds.
#[derive(Debug)]
pub struct BedrockStreamException {
    pub kind: String,
    pub message: String,
}

impl BedrockStreamException {
    /// Exceptions worth retrying: capacity and transient server faults.
    /// Validation/access errors will fail the same way again.
    pub fn is_transient(&self) -> bool {
        matches!(
            self.kind.as_str(),
            "throttlingException"
                | "serviceUnavailableException"
                | "internalServerException"
                | "modelStreamErrorException"
                | "ThrottlingException"
                | "ServiceUnavailableException"
                | "InternalServerException"
                | "ModelStreamErrorException"
        )
    }
}

impl std::fmt::Display for BedrockStreamException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Bedrock stream exception ({}): {}",
            self.kind, self.message
        )
    }
}

impl std::error::Error for BedrockStreamException {}

/// The ConverseStream body ended before `messageStop` — the connection was
/// cut (proxy timeout, network drop). Retryable while nothing has been shown.
#[derive(Debug)]
pub struct BedrockStreamTruncated {
    pub mid_frame: bool,
}

impl std::fmt::Display for BedrockStreamTruncated {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.mid_frame {
            write!(
                f,
                "Bedrock stream ended mid-frame (connection closed early)"
            )
        } else {
            write!(
                f,
                "Bedrock stream ended before messageStop (connection closed early)"
            )
        }
    }
}

impl std::error::Error for BedrockStreamTruncated {}

/// A non-2xx response from Bedrock. Kept as a typed error (rather than a
/// formatted string) so the retry layer can see the status code and any
/// `Retry-After` hint — 429 throttling and 503s are routine on Bedrock.
#[derive(Debug)]
pub struct BedrockHttpError {
    pub status: reqwest::StatusCode,
    pub retry_after: Option<std::time::Duration>,
    pub message: String,
}

impl std::fmt::Display for BedrockHttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HTTP {}: {}", self.status.as_u16(), self.message)
    }
}

impl std::error::Error for BedrockHttpError {}

async fn http_error(model_id: &str, resp: reqwest::Response) -> anyhow::Error {
    let status = resp.status();
    let retry_after = resp
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(crate::retry::parse_retry_after);
    let body = resp.text().await.unwrap_or_default();
    anyhow::Error::new(BedrockHttpError {
        status,
        retry_after,
        message: truncate_for_error(&body),
    })
    .context(format!("Bedrock {} request failed", model_id))
}

/// Shape returned from the non-streaming `converse()` call.
pub struct BedrockOutcome {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub input_tokens: Option<i32>,
    pub output_tokens: Option<i32>,
    pub stop_reason: Option<String>,
}

/// Semantic events produced by the streaming `converse_stream()` call.
/// These normalize Bedrock's on-the-wire event vocabulary into something
/// the query layer can consume without caring about frame framing.
#[derive(Debug, Clone)]
pub enum BedrockStreamEvent {
    /// Plain text delta for a text content block.
    TextDelta(String),
    /// Start of a tool-use content block. The `input` arrives as deltas
    /// that must be concatenated and JSON-parsed at ContentBlockStop time.
    ToolUseStart {
        block_index: usize,
        tool_use_id: String,
        name: String,
    },
    /// A partial JSON fragment of the tool input for `block_index`.
    ToolUseInputDelta {
        block_index: usize,
        input_fragment: String,
    },
    /// A content block (text or tool-use) has ended. The consumer finalizes
    /// any tool-use input accumulated for this index.
    ContentBlockStop { block_index: usize },
    /// Model's stopReason (end_turn, tool_use, max_tokens, etc.).
    MessageStop { stop_reason: Option<String> },
    /// Usage metadata (tokens) arriving near end of stream.
    Metadata {
        input_tokens: Option<i32>,
        output_tokens: Option<i32>,
    },
}

/// Decode a single Event Stream frame into zero or more semantic events.
/// Server-reported failures are returned as errors (a typed
/// [`BedrockStreamException`] for exception frames), which ends the stream.
fn frame_to_events(frame: Frame) -> Result<Vec<BedrockStreamEvent>> {
    // :message-type is "event" for data, "exception" for modeled server
    // errors (name in :exception-type, JSON `{"message": ...}` payload) and
    // "error" for unmodeled ones (:error-code / :error-message headers).
    match frame.message_type() {
        Some("exception") => {
            let message = serde_json::from_slice::<Value>(&frame.payload)
                .ok()
                .and_then(|v| v.get("message")?.as_str().map(String::from))
                .unwrap_or_else(|| String::from_utf8_lossy(&frame.payload).into_owned());
            return Err(anyhow::Error::new(BedrockStreamException {
                kind: frame
                    .exception_type()
                    .or_else(|| frame.event_type())
                    .unwrap_or("?")
                    .to_string(),
                message: truncate_for_error(&message),
            }));
        }
        Some("error") => {
            return Err(anyhow!(
                "Bedrock stream error ({}): {}",
                frame.error_code().unwrap_or("?"),
                truncate_for_error(frame.error_message().unwrap_or("")),
            ));
        }
        Some("event") | None => {}
        Some(other) => bail!("unexpected event-stream message type {:?}", other),
    }

    let event_type = frame.event_type().unwrap_or("");
    // Only parse payloads for event kinds we act on, so a future event kind
    // with a non-JSON payload is ignored instead of aborting the stream.
    if !matches!(
        event_type,
        "contentBlockStart" | "contentBlockDelta" | "contentBlockStop" | "messageStop" | "metadata"
    ) {
        return Ok(Vec::new());
    }
    // Empty payload is legal for some frame types; skip JSON parsing then.
    let payload: Value = if frame.payload.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&frame.payload).with_context(|| {
            format!(
                "parsing {} payload as JSON (first 200 bytes: {:?})",
                event_type,
                String::from_utf8_lossy(&frame.payload[..frame.payload.len().min(200)])
            )
        })?
    };

    match event_type {
        "contentBlockStart" => {
            let block_index = payload["contentBlockIndex"].as_u64().unwrap_or(0) as usize;
            if let Some(tu) = payload["start"].get("toolUse") {
                let tool_use_id = tu["toolUseId"].as_str().unwrap_or("").to_string();
                let name = tu["name"].as_str().unwrap_or("").to_string();
                Ok(vec![BedrockStreamEvent::ToolUseStart {
                    block_index,
                    tool_use_id,
                    name,
                }])
            } else {
                // Text content block start — nothing semantic to emit.
                Ok(Vec::new())
            }
        }
        "contentBlockDelta" => {
            let block_index = payload["contentBlockIndex"].as_u64().unwrap_or(0) as usize;
            let delta = &payload["delta"];
            if let Some(text) = delta["text"].as_str() {
                Ok(vec![BedrockStreamEvent::TextDelta(text.to_string())])
            } else if let Some(fragment) = delta["toolUse"]["input"].as_str() {
                Ok(vec![BedrockStreamEvent::ToolUseInputDelta {
                    block_index,
                    input_fragment: fragment.to_string(),
                }])
            } else {
                Ok(Vec::new())
            }
        }
        "contentBlockStop" => {
            let block_index = payload["contentBlockIndex"].as_u64().unwrap_or(0) as usize;
            Ok(vec![BedrockStreamEvent::ContentBlockStop { block_index }])
        }
        "messageStop" => {
            let stop_reason = payload["stopReason"].as_str().map(String::from);
            Ok(vec![BedrockStreamEvent::MessageStop { stop_reason }])
        }
        "metadata" => {
            let usage = &payload["usage"];
            Ok(vec![BedrockStreamEvent::Metadata {
                input_tokens: total_input_tokens(
                    usage["inputTokens"].as_i64(),
                    usage["cacheReadInputTokens"].as_i64(),
                    usage["cacheWriteInputTokens"].as_i64(),
                ),
                output_tokens: usage["outputTokens"].as_i64().map(|v| v as i32),
            }])
        }
        _ => Ok(Vec::new()),
    }
}

/// Cap an error body for display. Truncates on a char boundary: slicing at
/// a raw byte offset panics when a multi-byte character (CJK, emoji, an
/// HTML page's typographic quotes) straddles the limit.
fn truncate_for_error(s: &str) -> String {
    const MAX: usize = 500;
    if s.len() <= MAX {
        return s.to_string();
    }
    let mut end = MAX;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

/// URL-encode a path segment, preserving characters that Bedrock's router
/// expects verbatim (`.`, `:`, `-`). `reqwest::Url` would double-encode
/// these; simpler to hand-encode here since model IDs are ASCII-safe.
fn urlencode_path_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' | b':' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

// =============================================================================
// Converse request/response DTOs (only the fields we produce or consume)
// =============================================================================

#[derive(Serialize)]
struct ConverseRequest {
    messages: Vec<BedrockMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<Vec<BedrockSystemBlock>>,
    #[serde(rename = "toolConfig", skip_serializing_if = "Option::is_none")]
    tool_config: Option<BedrockToolConfig>,
}

#[derive(Serialize)]
struct BedrockMessage {
    role: &'static str,
    content: Vec<BedrockContentOut>,
}

/// Content block variants we serialize. Only one field is set per instance;
/// serde will emit the non-None one. Bedrock rejects requests that include
/// multiple keys in a single content object.
#[derive(Serialize, Default)]
struct BedrockContentOut {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(rename = "toolUse", skip_serializing_if = "Option::is_none")]
    tool_use: Option<BedrockToolUseOut>,
    #[serde(rename = "toolResult", skip_serializing_if = "Option::is_none")]
    tool_result: Option<BedrockToolResultOut>,
}

#[derive(Serialize)]
struct BedrockToolUseOut {
    #[serde(rename = "toolUseId")]
    tool_use_id: String,
    name: String,
    input: Value,
}

#[derive(Serialize)]
struct BedrockToolResultOut {
    #[serde(rename = "toolUseId")]
    tool_use_id: String,
    content: Vec<BedrockToolResultContent>,
}

#[derive(Serialize)]
struct BedrockToolResultContent {
    text: String,
}

#[derive(Serialize)]
struct BedrockSystemBlock {
    text: String,
}

#[derive(Serialize)]
struct BedrockToolConfig {
    tools: Vec<BedrockToolSpec>,
}

#[derive(Serialize)]
struct BedrockToolSpec {
    #[serde(rename = "toolSpec")]
    tool_spec: BedrockToolSpecInner,
}

#[derive(Serialize)]
struct BedrockToolSpecInner {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(rename = "inputSchema")]
    input_schema: BedrockInputSchema,
}

#[derive(Serialize)]
struct BedrockInputSchema {
    json: Value,
}

#[derive(Deserialize)]
struct ConverseResponse {
    output: ConverseOutput,
    #[serde(default)]
    usage: Option<ConverseUsage>,
    #[serde(default, rename = "stopReason")]
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
struct ConverseOutput {
    message: ConverseMessage,
}

#[derive(Deserialize)]
struct ConverseMessage {
    #[serde(default)]
    content: Vec<ConverseContent>,
}

#[derive(Deserialize)]
struct ConverseContent {
    #[serde(default)]
    text: Option<String>,
    #[serde(default, rename = "toolUse")]
    tool_use: Option<ConverseToolUseIn>,
}

#[derive(Deserialize)]
struct ConverseToolUseIn {
    #[serde(rename = "toolUseId")]
    tool_use_id: String,
    name: String,
    #[serde(default)]
    input: Value,
}

#[derive(Deserialize)]
struct ConverseUsage {
    #[serde(default, rename = "inputTokens")]
    input_tokens: Option<i32>,
    #[serde(default, rename = "cacheReadInputTokens")]
    cache_read_input_tokens: Option<i32>,
    #[serde(default, rename = "cacheWriteInputTokens")]
    cache_write_input_tokens: Option<i32>,
    #[serde(default, rename = "outputTokens")]
    output_tokens: Option<i32>,
}

impl ConverseUsage {
    fn total_input(&self) -> Option<i32> {
        total_input_tokens(
            self.input_tokens.map(i64::from),
            self.cache_read_input_tokens.map(i64::from),
            self.cache_write_input_tokens.map(i64::from),
        )
    }
}

/// The prompt's full size. Bedrock leaves cached tokens out of
/// `inputTokens`: a model that caches prompts automatically (Grok does)
/// reports `inputTokens: 0, cacheReadInputTokens: 38` for a 38-token prompt,
/// which showed as "0 in". `None` only when Bedrock sent none of the three.
fn total_input_tokens(
    input: Option<i64>,
    cache_read: Option<i64>,
    cache_write: Option<i64>,
) -> Option<i32> {
    if input.is_none() && cache_read.is_none() && cache_write.is_none() {
        return None;
    }
    let total = input.unwrap_or(0) + cache_read.unwrap_or(0) + cache_write.unwrap_or(0);
    Some(i32::try_from(total).unwrap_or(i32::MAX))
}

fn build_converse_request(
    messages: &[ChatMessage],
    system_prompt: Option<&str>,
    tools: Option<&[Tool]>,
) -> Result<ConverseRequest> {
    // Bedrock puts the system prompt in its own field (not a role). Pull any
    // genai `ChatRole::System` messages out and fold them in with the
    // caller-provided system_prompt.
    let mut system_texts: Vec<String> = Vec::new();
    if let Some(s) = system_prompt {
        if !s.is_empty() {
            system_texts.push(s.to_string());
        }
    }

    let mut out_messages = Vec::with_capacity(messages.len());
    for msg in messages {
        match msg.role {
            ChatRole::System => {
                // Callers may pass the same prompt both as `system_prompt`
                // and as a System message in the history; send it once.
                if let Some(t) = content_to_text(&msg.content) {
                    if !system_texts.contains(&t) {
                        system_texts.push(t);
                    }
                }
            }
            ChatRole::User => {
                let blocks = convert_user_content(&msg.content)?;
                push_turn(&mut out_messages, "user", blocks);
            }
            ChatRole::Assistant => {
                let blocks = convert_assistant_content(&msg.content)?;
                push_turn(&mut out_messages, "assistant", blocks);
            }
            ChatRole::Tool => {
                // genai uses a synthetic "tool" role for tool responses;
                // Bedrock expects these as a user turn with toolResult blocks.
                let blocks = convert_tool_content(&msg.content)?;
                push_turn(&mut out_messages, "user", blocks);
            }
        }
    }

    let system = if system_texts.is_empty() {
        None
    } else {
        Some(
            system_texts
                .into_iter()
                .map(|text| BedrockSystemBlock { text })
                .collect(),
        )
    };

    let tool_config = tools.and_then(|ts| {
        if ts.is_empty() {
            None
        } else {
            Some(BedrockToolConfig {
                tools: ts
                    .iter()
                    .map(|t| BedrockToolSpec {
                        tool_spec: BedrockToolSpecInner {
                            name: t.name.clone(),
                            description: t.description.clone(),
                            input_schema: BedrockInputSchema {
                                json: t
                                    .schema
                                    .clone()
                                    // Bedrock requires a valid JSON schema; if
                                    // the caller provided none, send an empty
                                    // object schema ("any input shape").
                                    .unwrap_or_else(|| serde_json::json!({"type": "object"})),
                            },
                        },
                    })
                    .collect(),
            })
        }
    });

    Ok(ConverseRequest {
        messages: out_messages,
        system,
        tool_config,
    })
}

/// Append a turn, merging it into the previous message when the role
/// repeats. Converse requires strictly alternating user/assistant turns,
/// but genai emits one `Tool` message per tool response — so a model that
/// makes two tool calls in one turn would otherwise produce two consecutive
/// user turns, which Bedrock rejects with a ValidationException.
fn push_turn(out: &mut Vec<BedrockMessage>, role: &'static str, blocks: Vec<BedrockContentOut>) {
    if blocks.is_empty() {
        return;
    }
    match out.last_mut() {
        Some(prev) if prev.role == role => prev.content.extend(blocks),
        _ => out.push(BedrockMessage {
            role,
            content: blocks,
        }),
    }
}

fn content_to_text(content: &MessageContent) -> Option<String> {
    let mut text = String::new();
    for part in content.parts() {
        if let ContentPart::Text(t) = part {
            text.push_str(t);
        } else {
            return None;
        }
    }
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// User turns support text and toolResult (genai's tool-response mode — see
/// ChatRole::Tool handling above — is a separate path; this function stays
/// lenient in case a ToolResponse sneaks in via a user-tagged message).
fn convert_user_content(content: &MessageContent) -> Result<Vec<BedrockContentOut>> {
    let mut out = Vec::new();
    for part in content.parts() {
        match part {
            ContentPart::Text(t) if !t.is_empty() => out.push(BedrockContentOut {
                text: Some(t.clone()),
                ..Default::default()
            }),
            ContentPart::ToolResponse(tr) => {
                out.push(BedrockContentOut {
                    tool_result: Some(BedrockToolResultOut {
                        tool_use_id: tr.call_id.clone(),
                        content: vec![BedrockToolResultContent {
                            text: tr.content.clone(),
                        }],
                    }),
                    ..Default::default()
                });
            }
            ContentPart::Text(_) => {}
            ContentPart::ThoughtSignature(_) => {} // drop; Bedrock has no wire equivalent
            other => {
                return Err(anyhow!(
                    "Bedrock routing: unsupported user content part: {:?}",
                    std::mem::discriminant(other)
                ));
            }
        }
    }
    Ok(out)
}

/// Assistant turns support text and toolUse content blocks.
fn convert_assistant_content(content: &MessageContent) -> Result<Vec<BedrockContentOut>> {
    let mut out = Vec::new();
    for part in content.parts() {
        match part {
            ContentPart::Text(t) if !t.is_empty() => out.push(BedrockContentOut {
                text: Some(t.clone()),
                ..Default::default()
            }),
            ContentPart::ToolCall(tc) => {
                out.push(BedrockContentOut {
                    tool_use: Some(BedrockToolUseOut {
                        tool_use_id: tc.call_id.clone(),
                        name: tc.fn_name.clone(),
                        input: tc.fn_arguments.clone(),
                    }),
                    ..Default::default()
                });
            }
            ContentPart::Text(_) => {}
            ContentPart::ThoughtSignature(_) => {} // drop; Bedrock has no wire equivalent
            other => {
                return Err(anyhow!(
                    "Bedrock routing: unsupported assistant content part: {:?}",
                    std::mem::discriminant(other)
                ));
            }
        }
    }
    Ok(out)
}

/// `ChatRole::Tool` messages from genai carry ToolResponse parts. In Bedrock
/// these ride on a user turn with toolResult blocks.
fn convert_tool_content(content: &MessageContent) -> Result<Vec<BedrockContentOut>> {
    let mut out = Vec::new();
    for part in content.parts() {
        if let ContentPart::ToolResponse(tr) = part {
            out.push(BedrockContentOut {
                tool_result: Some(BedrockToolResultOut {
                    tool_use_id: tr.call_id.clone(),
                    content: vec![BedrockToolResultContent {
                        text: tr.content.clone(),
                    }],
                }),
                ..Default::default()
            });
        } else {
            return Err(anyhow!(
                "Bedrock routing: tool-role message contained non-ToolResponse content"
            ));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_prompt_tokens_count_as_input() {
        let usage: ConverseUsage = serde_json::from_str(
            r#"{"cacheReadInputTokens":38,"cacheWriteInputTokens":0,"inputTokens":0,"outputTokens":62}"#,
        )
        .unwrap();
        assert_eq!(usage.total_input(), Some(38));
        let plain: ConverseUsage = serde_json::from_str(r#"{"inputTokens":13}"#).unwrap();
        assert_eq!(plain.total_input(), Some(13));
        let none: ConverseUsage = serde_json::from_str(r#"{"outputTokens":5}"#).unwrap();
        assert_eq!(none.total_input(), None);
        assert_eq!(total_input_tokens(Some(5), Some(10), Some(2)), Some(17));
    }

    #[test]
    fn control_plane_host_swaps_the_runtime_label() {
        assert_eq!(
            control_plane_host("https://bedrock-runtime.us-east-1.amazonaws.com").as_deref(),
            Some("https://bedrock.us-east-1.amazonaws.com")
        );
        assert_eq!(
            control_plane_host("https://bedrock-runtime-fips.us-gov-west-1.amazonaws.com")
                .as_deref(),
            Some("https://bedrock-fips.us-gov-west-1.amazonaws.com")
        );
        assert_eq!(
            control_plane_host("https://bedrock-runtime.cn-north-1.amazonaws.com.cn").as_deref(),
            Some("https://bedrock.cn-north-1.amazonaws.com.cn")
        );
        // PrivateLink endpoints only serve the runtime API.
        assert_eq!(
            control_plane_host("https://vpce-0abc.bedrock-runtime.us-east-1.vpce.amazonaws.com"),
            None
        );
        assert_eq!(control_plane_host("https://example.com"), None);
    }

    #[test]
    fn detects_bedrock_url() {
        assert!(is_bedrock_base_url(
            "https://bedrock-runtime.us-east-1.amazonaws.com/v1/"
        ));
        assert!(is_bedrock_base_url(
            "https://bedrock-runtime.us-gov-east-1.amazonaws.com/"
        ));
        assert!(!is_bedrock_base_url("https://api.anthropic.com/v1/"));
        assert!(!is_bedrock_base_url("https://openrouter.ai/api/v1"));
    }

    #[test]
    fn strips_path_from_host() {
        assert_eq!(
            bedrock_host("https://bedrock-runtime.us-east-1.amazonaws.com/v1/"),
            "https://bedrock-runtime.us-east-1.amazonaws.com"
        );
        assert_eq!(
            bedrock_host("https://bedrock-runtime.us-east-1.amazonaws.com"),
            "https://bedrock-runtime.us-east-1.amazonaws.com"
        );
    }

    #[test]
    fn encodes_inference_profile_ids() {
        // Dots and colons must survive (they're path-segment safe).
        assert_eq!(
            urlencode_path_segment("us.anthropic.claude-opus-4-7"),
            "us.anthropic.claude-opus-4-7"
        );
        assert_eq!(
            urlencode_path_segment("anthropic.claude-3-5-sonnet-20241022-v2:0"),
            "anthropic.claude-3-5-sonnet-20241022-v2:0"
        );
    }

    #[test]
    fn builds_basic_request() {
        let msgs = vec![ChatMessage::user("hello")];
        let req = build_converse_request(&msgs, None, None).unwrap();
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["messages"][0]["role"], "user");
        assert_eq!(json["messages"][0]["content"][0]["text"], "hello");
        assert!(json.get("system").is_none_or(|v| v.is_null()));
        assert!(json.get("toolConfig").is_none_or(|v| v.is_null()));
    }

    #[test]
    fn folds_system_prompts() {
        let msgs = vec![ChatMessage::system("be brief"), ChatMessage::user("hi")];
        let req = build_converse_request(&msgs, Some("also: polite"), None).unwrap();
        let json = serde_json::to_value(&req).unwrap();
        let sys = json["system"].as_array().unwrap();
        assert_eq!(sys.len(), 2);
        // order: explicit system_prompt arg first, then ChatRole::System messages
        assert_eq!(sys[0]["text"], "also: polite");
        assert_eq!(sys[1]["text"], "be brief");
        // user message survives unchanged
        assert_eq!(json["messages"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn encodes_assistant_tool_calls_and_tool_responses() {
        use genai::chat::{ToolCall, ToolResponse};
        let tool_call = ToolCall {
            call_id: "c1".into(),
            fn_name: "web_search".into(),
            fn_arguments: serde_json::json!({"query": "rust"}),
            thought_signatures: None,
        };
        let msgs = vec![
            ChatMessage::user("search for rust info"),
            ChatMessage::from(vec![tool_call.clone()]),
            ChatMessage::from(ToolResponse::new("c1", "search results here")),
        ];
        let req = build_converse_request(&msgs, None, None).unwrap();
        let json = serde_json::to_value(&req).unwrap();
        let messages = json["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "user");
        // Assistant turn carries the toolUse block.
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"][0]["toolUse"]["toolUseId"], "c1");
        assert_eq!(messages[1]["content"][0]["toolUse"]["name"], "web_search");
        assert_eq!(
            messages[1]["content"][0]["toolUse"]["input"]["query"],
            "rust"
        );
        // genai's ChatRole::Tool becomes a user turn with toolResult.
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(messages[2]["content"][0]["toolResult"]["toolUseId"], "c1");
        assert_eq!(
            messages[2]["content"][0]["toolResult"]["content"][0]["text"],
            "search results here"
        );
    }

    #[test]
    fn includes_tool_config_when_tools_provided() {
        use genai::chat::Tool;
        let tool = Tool::new("web_search")
            .with_description("Search the web.")
            .with_schema(serde_json::json!({
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }));
        let msgs = vec![ChatMessage::user("find something")];
        let req = build_converse_request(&msgs, None, Some(&[tool])).unwrap();
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(
            json["toolConfig"]["tools"][0]["toolSpec"]["name"],
            "web_search"
        );
        assert_eq!(
            json["toolConfig"]["tools"][0]["toolSpec"]["description"],
            "Search the web."
        );
        assert_eq!(
            json["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"]["type"],
            "object"
        );
    }

    #[test]
    fn omits_tool_config_when_tools_empty() {
        let msgs = vec![ChatMessage::user("hi")];
        let req = build_converse_request(&msgs, None, Some(&[])).unwrap();
        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("toolConfig").is_none_or(|v| v.is_null()));
    }

    #[test]
    fn parses_non_streaming_tool_use_response() {
        // Payload matches what we observed from real Bedrock responses
        // during the streaming probe (tooluse_... id, input object).
        let json = r#"{
            "output": {"message": {"role": "assistant", "content": [
                {"toolUse": {
                    "toolUseId": "tooluse_abc123",
                    "name": "multiply",
                    "input": {"a": 7, "b": 13}
                }}
            ]}},
            "usage": {"inputTokens": 790, "outputTokens": 84, "totalTokens": 874},
            "stopReason": "tool_use"
        }"#;
        let parsed: ConverseResponse = serde_json::from_str(json).unwrap();
        let tc = parsed.output.message.content[0].tool_use.as_ref().unwrap();
        assert_eq!(tc.tool_use_id, "tooluse_abc123");
        assert_eq!(tc.name, "multiply");
        assert_eq!(tc.input["a"], 7);
        assert_eq!(tc.input["b"], 13);
    }

    #[test]
    fn frame_to_events_handles_text_delta() {
        let payload = br#"{"contentBlockIndex":0,"delta":{"text":"hello"}}"#;
        let mut headers = std::collections::HashMap::new();
        headers.insert(":event-type".to_string(), "contentBlockDelta".to_string());
        headers.insert(":message-type".to_string(), "event".to_string());
        let frame = Frame {
            headers,
            payload: payload.to_vec(),
        };
        let events = frame_to_events(frame).unwrap();
        assert_eq!(events.len(), 1);
        match &events[0] {
            BedrockStreamEvent::TextDelta(t) => assert_eq!(t, "hello"),
            other => panic!("expected TextDelta, got {:?}", other),
        }
    }

    #[test]
    fn frame_to_events_handles_tool_use_start_and_deltas() {
        // contentBlockStart with toolUse
        let start_payload =
            br#"{"contentBlockIndex":0,"start":{"toolUse":{"name":"mul","toolUseId":"tu_x"}}}"#;
        let mut h = std::collections::HashMap::new();
        h.insert(":event-type".to_string(), "contentBlockStart".to_string());
        h.insert(":message-type".to_string(), "event".to_string());
        let frame = Frame {
            headers: h,
            payload: start_payload.to_vec(),
        };
        let events = frame_to_events(frame).unwrap();
        match &events[0] {
            BedrockStreamEvent::ToolUseStart {
                block_index,
                tool_use_id,
                name,
            } => {
                assert_eq!(*block_index, 0);
                assert_eq!(tool_use_id, "tu_x");
                assert_eq!(name, "mul");
            }
            other => panic!("expected ToolUseStart, got {:?}", other),
        }

        // contentBlockDelta with toolUse.input fragment
        let delta_payload = br#"{"contentBlockIndex":0,"delta":{"toolUse":{"input":"{\"a\": 1"}}}"#;
        let mut h = std::collections::HashMap::new();
        h.insert(":event-type".to_string(), "contentBlockDelta".to_string());
        h.insert(":message-type".to_string(), "event".to_string());
        let frame = Frame {
            headers: h,
            payload: delta_payload.to_vec(),
        };
        let events = frame_to_events(frame).unwrap();
        match &events[0] {
            BedrockStreamEvent::ToolUseInputDelta {
                block_index,
                input_fragment,
            } => {
                assert_eq!(*block_index, 0);
                assert_eq!(input_fragment, r#"{"a": 1"#);
            }
            other => panic!("expected ToolUseInputDelta, got {:?}", other),
        }
    }

    #[test]
    fn frame_to_events_surfaces_exceptions() {
        let mut h = std::collections::HashMap::new();
        h.insert(":event-type".to_string(), "throttlingException".to_string());
        h.insert(":message-type".to_string(), "exception".to_string());
        let frame = Frame {
            headers: h,
            payload: b"rate limited".to_vec(),
        };
        let msg = frame_to_events(frame).unwrap_err().to_string();
        assert!(msg.contains("throttlingException"), "got: {}", msg);
        assert!(msg.contains("rate limited"), "got: {}", msg);
    }

    fn frame(headers: &[(&str, &str)], payload: &[u8]) -> Frame {
        Frame {
            headers: headers
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            payload: payload.to_vec(),
        }
    }

    #[test]
    fn stream_metadata_counts_cached_prompt_tokens() {
        let events = frame_to_events(frame(
            &[(":message-type", "event"), (":event-type", "metadata")],
            br#"{"usage":{"inputTokens":0,"cacheReadInputTokens":38,"outputTokens":64}}"#,
        ))
        .unwrap();
        assert!(matches!(
            events.as_slice(),
            [BedrockStreamEvent::Metadata {
                input_tokens: Some(38),
                output_tokens: Some(64),
            }]
        ));
    }

    #[test]
    fn detects_fips_govcloud_and_china_endpoints_but_not_lookalikes() {
        assert!(is_bedrock_base_url(
            "https://bedrock-runtime-fips.us-gov-west-1.amazonaws.com"
        ));
        assert!(is_bedrock_base_url(
            "https://bedrock-runtime.cn-north-1.amazonaws.com.cn/"
        ));
        // The bearer token must not be sent to a host that merely mentions
        // Bedrock somewhere in its URL.
        assert!(!is_bedrock_base_url(
            "https://proxy.example.com/bedrock-runtime.us-east-1.amazonaws.com/"
        ));
        assert!(!is_bedrock_base_url(
            "https://bedrock-runtime.us-east-1.amazonaws.com.evil.example/"
        ));
        assert!(!is_bedrock_base_url("not a url"));
    }

    #[test]
    fn identical_system_prompts_are_sent_once() {
        let msgs = vec![ChatMessage::system("be brief"), ChatMessage::user("hi")];
        let req = build_converse_request(&msgs, Some("be brief"), None).unwrap();
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["system"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn parallel_tool_results_merge_into_one_user_turn() {
        // Two tool calls in one assistant turn yield two genai Tool
        // messages; Converse needs them in a single user turn or it rejects
        // the request for not alternating roles.
        use genai::chat::{ToolCall, ToolResponse};
        let call = |id: &str, q: &str| ToolCall {
            call_id: id.into(),
            fn_name: "web_search".into(),
            fn_arguments: serde_json::json!({ "query": q }),
            thought_signatures: None,
        };
        let msgs = vec![
            ChatMessage::user("compare a and b"),
            ChatMessage::from(vec![call("c1", "a"), call("c2", "b")]),
            ChatMessage::from(ToolResponse::new("c1", "about a")),
            ChatMessage::from(ToolResponse::new("c2", "about b")),
        ];
        let req = build_converse_request(&msgs, None, None).unwrap();
        let json = serde_json::to_value(&req).unwrap();
        let messages = json["messages"].as_array().unwrap();
        let roles: Vec<&str> = messages
            .iter()
            .map(|m| m["role"].as_str().unwrap())
            .collect();
        assert_eq!(roles, ["user", "assistant", "user"]);
        let results = messages[2]["content"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["toolResult"]["toolUseId"], "c1");
        assert_eq!(results[1]["toolResult"]["toolUseId"], "c2");
    }

    #[test]
    fn truncate_for_error_never_splits_a_character() {
        // 499 ASCII bytes, then a 3-byte character straddling byte 500.
        let s = format!("{}€€", "a".repeat(499));
        let out = truncate_for_error(&s);
        assert!(out.ends_with('…'));
        assert!(out.starts_with(&"a".repeat(499)));
        assert_eq!(truncate_for_error("short"), "short");
    }

    #[test]
    fn exception_frames_use_exception_type_header_and_message() {
        let f = frame(
            &[
                (":message-type", "exception"),
                (":exception-type", "throttlingException"),
            ],
            br#"{"message":"Too many requests, please wait."}"#,
        );
        let err = frame_to_events(f).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("throttlingException"), "got: {}", msg);
        assert!(msg.contains("Too many requests"), "got: {}", msg);
        assert!(!msg.contains('{'), "raw JSON leaked: {}", msg);
        // Typed, and classified as transient so a stream that is throttled
        // before any output gets retried.
        let exception = err.downcast_ref::<BedrockStreamException>().unwrap();
        assert!(exception.is_transient());
        assert!(matches!(
            crate::retry::classify(&err),
            crate::retry::Retryable::Yes { .. }
        ));
    }

    #[test]
    fn validation_exceptions_are_not_retried() {
        let f = frame(
            &[
                (":message-type", "exception"),
                (":exception-type", "validationException"),
            ],
            br#"{"message":"bad input"}"#,
        );
        let err = frame_to_events(f).unwrap_err();
        assert!(matches!(
            crate::retry::classify(&err),
            crate::retry::Retryable::No
        ));
    }

    #[test]
    fn truncated_streams_are_retryable() {
        let err = anyhow::Error::new(BedrockStreamTruncated { mid_frame: false });
        assert!(err.to_string().contains("before messageStop"));
        assert!(matches!(
            crate::retry::classify(&err),
            crate::retry::Retryable::Yes { .. }
        ));
    }

    #[test]
    fn detects_privatelink_endpoints() {
        assert!(is_bedrock_base_url(
            "https://vpce-0abc-123.bedrock-runtime.us-east-1.vpce.amazonaws.com"
        ));
        assert!(!is_bedrock_base_url("https://bedrock-runtime.example.com/"));
    }

    #[test]
    fn error_frames_are_surfaced_not_ignored() {
        let f = frame(
            &[
                (":message-type", "error"),
                (":error-code", "InternalFailure"),
                (":error-message", "something broke"),
            ],
            b"",
        );
        let msg = frame_to_events(f).unwrap_err().to_string();
        assert!(msg.contains("InternalFailure"), "got: {}", msg);
        assert!(msg.contains("something broke"), "got: {}", msg);
    }

    #[test]
    fn unknown_event_types_with_non_json_payloads_are_ignored() {
        let f = frame(
            &[
                (":message-type", "event"),
                (":event-type", "someFutureEvent"),
            ],
            b"\x00\x01 not json",
        );
        assert!(frame_to_events(f).unwrap().is_empty());
    }

    #[test]
    fn stop_reason_notes_only_flag_incomplete_answers() {
        assert!(stop_reason_note(Some("max_tokens")).is_some());
        assert!(stop_reason_note(Some("guardrail_intervened")).is_some());
        assert!(stop_reason_note(Some("model_context_window_exceeded")).is_some());
        assert!(stop_reason_note(Some("end_turn")).is_none());
        assert!(stop_reason_note(Some("tool_use")).is_none());
        assert!(stop_reason_note(None).is_none());
    }

    #[test]
    fn parses_stop_reason_from_converse_response() {
        let json = r#"{
            "output": {"message": {"content": [{"text": "partial"}], "role": "assistant"}},
            "stopReason": "max_tokens"
        }"#;
        let parsed: ConverseResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.stop_reason.as_deref(), Some("max_tokens"));
    }

    #[test]
    fn parses_converse_response() {
        let json = r#"{
            "output": {"message": {"content": [{"text": "Hello there!"}], "role": "assistant"}},
            "usage": {"inputTokens": 13, "outputTokens": 5, "totalTokens": 18},
            "stopReason": "end_turn"
        }"#;
        let parsed: ConverseResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            parsed.output.message.content[0].text.as_deref(),
            Some("Hello there!")
        );
        assert_eq!(parsed.usage.as_ref().unwrap().input_tokens, Some(13));
        assert_eq!(parsed.usage.as_ref().unwrap().output_tokens, Some(5));
    }
}
