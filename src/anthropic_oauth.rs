//! Unofficial support for authenticating the Anthropic provider against a
//! Claude Pro/Max subscription instead of a metered API key.
//!
//! Claude Code stores an OAuth credential at `~/.claude/.credentials.json`
//! after `claude login` with a Claude.ai account (as opposed to entering an
//! API key). That token is scoped to ride the subscription's included usage.
//! This module replays it against the same `/v1/messages` endpoint using
//! `Authorization: Bearer` + `anthropic-beta: oauth-2025-04-20` in place of
//! `x-api-key`, mirroring what a handful of open-source Claude Code proxies
//! do. It is not a documented Anthropic API: the header names, endpoint
//! behavior, and gating logic here are inferred from public reverse
//! engineering, not from an Anthropic spec, and can change without notice.
//! Anthropic's usage terms scope this credential to Anthropic's own clients,
//! so using it here carries real risk of the token or account being flagged.

use anyhow::{bail, Context, Result};
use genai::resolver::AuthData;
use genai::Headers;
use serde::Deserialize;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1/";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const OAUTH_BETA: &str = "oauth-2025-04-20";

#[derive(Debug, Deserialize)]
struct CredentialsFile {
    #[serde(rename = "claudeAiOauth")]
    claude_ai_oauth: Option<OauthBlock>,
}

#[derive(Debug, Deserialize)]
struct OauthBlock {
    #[serde(rename = "accessToken")]
    access_token: String,
    #[serde(rename = "expiresAt")]
    expires_at: u64,
}

fn credentials_path() -> Result<PathBuf> {
    let home = dirs::home_dir().context("could not determine home directory")?;
    Ok(home.join(".claude").join(".credentials.json"))
}

fn read_access_token() -> Result<String> {
    let path = credentials_path()?;
    let content = std::fs::read_to_string(&path).with_context(|| {
        format!(
            "Claude Code credentials not found at {:?} — run `claude` once and log in \
             with your Pro/Max account (not an API key) first",
            path
        )
    })?;
    let parsed: CredentialsFile = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse Claude Code credentials at {:?}", path))?;
    let oauth = parsed.claude_ai_oauth.context(
        "No claudeAiOauth block in Claude Code credentials — log in via `claude` with a \
         Pro/Max account (not an API key) first",
    )?;

    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    if oauth.expires_at <= now_ms {
        bail!(
            "Claude Code subscription token has expired — run `claude` once (it refreshes \
             the token automatically) and retry"
        );
    }

    Ok(oauth.access_token)
}

/// Cheap availability check so `subscription`-mode providers can be silently
/// skipped at query time, the same way providers with no API key are —
/// rather than showing a panel that then fails mid-query.
pub fn is_available() -> bool {
    read_access_token().is_ok()
}

/// Build the `AuthData` that replays the Claude Code subscription token
/// against Anthropic's Messages endpoint. `base_url` overrides the default
/// endpoint if the provider config sets one.
pub fn subscription_auth_data(base_url: Option<&str>) -> Result<AuthData> {
    let token = read_access_token()?;

    let base = base_url.unwrap_or(DEFAULT_BASE_URL);
    let base = if base.ends_with('/') {
        base.to_string()
    } else {
        format!("{base}/")
    };
    let url = format!("{base}messages");

    let headers = Headers::from(vec![
        ("authorization".to_string(), format!("Bearer {token}")),
        ("anthropic-version".to_string(), ANTHROPIC_VERSION.to_string()),
        ("anthropic-beta".to_string(), OAUTH_BETA.to_string()),
    ]);

    Ok(AuthData::RequestOverride { url, headers })
}
