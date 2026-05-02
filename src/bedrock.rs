//! Direct client for AWS Bedrock's native Converse API.
//!
//! genai's `AdapterKind::Anthropic` speaks Anthropic's native wire protocol
//! (POST `{base_url}/messages`). Bedrock does not serve that protocol — a
//! mismatched request yields HTTP 200 with `UnknownOperationException`,
//! which presents as a silent empty response. This module bypasses genai
//! entirely for Bedrock-hosted Claude models by calling
//! `POST /model/{id}/converse` directly with bearer-token auth
//! (`AWS_BEARER_TOKEN_BEDROCK`, introduced 2025).
//!
//! Currently non-streaming only: the full response is delivered in one shot.
//! Real streaming (ConverseStream) uses AWS's binary Event Stream framing and
//! is a follow-up.
use anyhow::{anyhow, Context, Result};
use genai::chat::{ChatMessage, ChatRole, ContentPart, MessageContent};
use serde::{Deserialize, Serialize};

/// Detect whether a base_url points at AWS Bedrock. Used by the provider
/// layer to decide between genai and this direct client.
pub fn is_bedrock_base_url(url: &str) -> bool {
    url.contains("bedrock-runtime.") && url.contains("amazonaws.com")
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
        Some(Self {
            http: reqwest::Client::new(),
            host: bedrock_host(base_url),
            auth_token,
        })
    }

    /// Invoke the Converse API and return the assistant's text plus usage.
    pub async fn converse(
        &self,
        model_id: &str,
        messages: &[ChatMessage],
        system_prompt: Option<&str>,
    ) -> Result<BedrockOutcome> {
        let body = build_converse_request(messages, system_prompt)?;

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

        let status = resp.status();
        let text = resp
            .text()
            .await
            .with_context(|| "reading Bedrock response body")?;

        if !status.is_success() {
            return Err(anyhow!(
                "Bedrock {} returned HTTP {}: {}",
                model_id,
                status.as_u16(),
                truncate_for_error(&text)
            ));
        }

        let parsed: ConverseResponse = serde_json::from_str(&text).with_context(|| {
            format!(
                "parsing Bedrock Converse response (first 200 chars: {:?})",
                text.chars().take(200).collect::<String>()
            )
        })?;

        let assistant_text = parsed
            .output
            .message
            .content
            .iter()
            .filter_map(|c| c.text.as_deref())
            .collect::<Vec<_>>()
            .join("");

        Ok(BedrockOutcome {
            text: assistant_text,
            input_tokens: parsed.usage.as_ref().and_then(|u| u.input_tokens),
            output_tokens: parsed.usage.as_ref().and_then(|u| u.output_tokens),
        })
    }
}

pub struct BedrockOutcome {
    pub text: String,
    pub input_tokens: Option<i32>,
    pub output_tokens: Option<i32>,
}

fn truncate_for_error(s: &str) -> String {
    const MAX: usize = 500;
    if s.len() <= MAX {
        s.to_string()
    } else {
        format!("{}…", &s[..MAX])
    }
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
}

#[derive(Serialize)]
struct BedrockMessage {
    role: &'static str,
    content: Vec<BedrockContentOut>,
}

#[derive(Serialize)]
struct BedrockContentOut {
    text: String,
}

#[derive(Serialize)]
struct BedrockSystemBlock {
    text: String,
}

#[derive(Deserialize)]
struct ConverseResponse {
    output: ConverseOutput,
    #[serde(default)]
    usage: Option<ConverseUsage>,
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
}

#[derive(Deserialize)]
struct ConverseUsage {
    #[serde(default, rename = "inputTokens")]
    input_tokens: Option<i32>,
    #[serde(default, rename = "outputTokens")]
    output_tokens: Option<i32>,
}

fn build_converse_request(
    messages: &[ChatMessage],
    system_prompt: Option<&str>,
) -> Result<ConverseRequest> {
    // Bedrock puts the system prompt in its own field (not a role). Pull any
    // genai `ChatRole::System` messages out and fold them in with the
    // caller-provided system_prompt. Assistant tool-call messages and tool
    // responses are currently unsupported (we error out so the caller can
    // decide whether to degrade or fail loudly).
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
                if let Some(t) = content_to_text(&msg.content) {
                    system_texts.push(t);
                }
            }
            ChatRole::User | ChatRole::Assistant => {
                let text = content_to_text(&msg.content).ok_or_else(|| {
                    anyhow!("Bedrock routing does not yet support non-text content (tool calls, images, etc.)")
                })?;
                out_messages.push(BedrockMessage {
                    role: match msg.role {
                        ChatRole::User => "user",
                        ChatRole::Assistant => "assistant",
                        _ => unreachable!(),
                    },
                    content: vec![BedrockContentOut { text }],
                });
            }
            ChatRole::Tool => {
                return Err(anyhow!(
                    "Bedrock routing does not yet support tool messages (web search etc.)"
                ));
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

    Ok(ConverseRequest {
        messages: out_messages,
        system,
    })
}

fn content_to_text(content: &MessageContent) -> Option<String> {
    // Only emit if the content is plain text (possibly multi-part).
    // If any non-text parts exist, return None so the caller can error.
    let mut text = String::new();
    for part in content.parts() {
        match part {
            ContentPart::Text(t) => text.push_str(t),
            _ => return None,
        }
    }
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let req = build_converse_request(&msgs, None).unwrap();
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["messages"][0]["role"], "user");
        assert_eq!(json["messages"][0]["content"][0]["text"], "hello");
        assert!(json.get("system").is_none_or(|v| v.is_null()));
    }

    #[test]
    fn folds_system_prompts() {
        let msgs = vec![
            ChatMessage::system("be brief"),
            ChatMessage::user("hi"),
        ];
        let req = build_converse_request(&msgs, Some("also: polite")).unwrap();
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
    fn rejects_tool_messages() {
        use genai::chat::{ToolCall, ToolResponse};
        let tool_call = ToolCall {
            call_id: "c1".into(),
            fn_name: "web_search".into(),
            fn_arguments: serde_json::json!({"query": "x"}),
            thought_signatures: None,
        };
        let msgs = vec![
            ChatMessage::user("hi"),
            ChatMessage::from(vec![tool_call.clone()]),
            ChatMessage::from(ToolResponse::new("c1", "result")),
        ];
        assert!(build_converse_request(&msgs, None).is_err());
    }

    #[test]
    fn parses_converse_response() {
        let json = r#"{
            "output": {"message": {"content": [{"text": "Hello there!"}], "role": "assistant"}},
            "usage": {"inputTokens": 13, "outputTokens": 5, "totalTokens": 18},
            "stopReason": "end_turn"
        }"#;
        let parsed: ConverseResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.output.message.content[0].text.as_deref(), Some("Hello there!"));
        assert_eq!(parsed.usage.as_ref().unwrap().input_tokens, Some(13));
        assert_eq!(parsed.usage.as_ref().unwrap().output_tokens, Some(5));
    }
}
