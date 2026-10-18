//! `ai doctor`: a one-shot health check of config, credentials and network
//! reachability, in the spirit of `brew doctor` / `flutter doctor`.
//!
//! The motivating case is a work laptop whose DNS filter (Cisco Umbrella)
//! answers `api.anthropic.com`, `api.openai.com` and `api.x.ai` with its own
//! block-page servers in 146.112.0.0/16, which then present a Cisco-issued
//! TLS certificate. Queries die with an opaque TLS or connect error while AWS
//! Bedrock stays reachable. `ai doctor` names the cause directly and points
//! at the provider that does work.
//!
//! Layout: everything that decides *what* to report (auth-source detection,
//! probe planning, HTTP-status and transport-error classification, the
//! filtered-IP check, rendering) is pure and unit-tested. `run` only does the
//! I/O: reading the environment, DNS lookups, and one free request per
//! provider (list models / key info; never a billed call).
//!
//! Secrets: key values are never printed. The report says at most where a
//! key comes from and how long it is. Response bodies are never shown either,
//! because some providers echo a masked fragment of the key in their 401
//! message, and a configured `base_url` is shown with any credentials in it
//! (userinfo, `?key=…`) redacted.

use crate::anthropic_oauth;
use crate::bedrock::is_bedrock_base_url;
use crate::config::{AuthMode, Config, ModelTier, ProviderConfig, SearchConfig};
use anyhow::{Context, Result};
use colored::Colorize;
use futures::future::join_all;
use reqwest::Url;
use std::fmt;
use std::io::{IsTerminal, Write};
use std::net::IpAddr;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Short timeouts: a probe that hangs is itself the diagnosis, and the whole
/// report should come back in seconds, not after the 90s a stuck query takes.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const DNS_TIMEOUT: Duration = Duration::from_secs(5);

const ANTHROPIC_VERSION: &str = "2023-06-01";
const BUILD_COMMIT: &str = env!("AI_CLI_GIT_HASH");

/// Cisco Umbrella answers DNS for blocked domains with one of its block-page
/// servers, all inside Cisco's 146.112.0.0/16. No AI provider API lives there.
const UMBRELLA_BLOCK_RANGE: &str = "146.112.0.0/16";

/// Env vars reqwest reads to route requests through a proxy.
const PROXY_VARS: [&str; 6] = [
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
];

const LANGFUSE_VARS: [&str; 3] = [
    "LANGFUSE_HOST",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
];

// =============================================================================
// Entry point
// =============================================================================

/// Run all checks and print the report. Exits the process with status 1 when
/// a connectivity probe fails or the config file doesn't load.
pub async fn run(offline: bool) -> Result<()> {
    let env = ProcessEnv;
    let loaded = load_config();
    let entries = provider_entries(&loaded.config, &env);

    // Print the static half first so the user sees something while the
    // network probes run.
    print!(
        "{}",
        render(&[system_section(&loaded, &env), providers_section(&entries)])
    );
    let _ = std::io::stdout().flush();

    let mut targets = Vec::new();
    let mut not_probed = Vec::new();
    for entry in entries.iter().filter(|e| e.enabled) {
        match probe_target(entry, &env) {
            Ok(target) => targets.push(target),
            Err(reason) => not_probed.push((entry.name.clone(), reason.to_string())),
        }
    }

    let results = if offline || targets.is_empty() {
        Vec::new()
    } else {
        let progress = std::io::stderr().is_terminal();
        if progress {
            eprint!(
                "{}",
                format!("probing {} provider(s)…", targets.len()).dimmed()
            );
        }
        let results = probe_all(targets).await;
        if progress {
            // Erase the progress line before the report continues.
            eprint!("\r\x1b[2K");
        }
        results?
    };

    let summary = summarize(&results);
    print!(
        "{}",
        render(&[
            connectivity_section(&results, &not_probed, offline),
            search_section(&loaded.config.search, &env),
            telemetry_section(&env),
            summary_section(&loaded.state, offline, &summary),
        ])
    );
    let _ = std::io::stdout().flush();

    if should_fail(&loaded.state, &summary) {
        std::process::exit(1);
    }
    Ok(())
}

// =============================================================================
// Environment access (behind a trait so the report can be built in tests)
// =============================================================================

trait Environment {
    /// The value of an environment variable, if set (possibly blank).
    fn var(&self, name: &str) -> Option<String>;
    /// Whether Claude Code's subscription OAuth token exists and is unexpired.
    fn subscription_token_available(&self) -> bool;
}

struct ProcessEnv;

impl Environment for ProcessEnv {
    fn var(&self, name: &str) -> Option<String> {
        std::env::var(name).ok()
    }

    fn subscription_token_available(&self) -> bool {
        anthropic_oauth::is_available()
    }
}

/// The value if it has any non-whitespace content. A whitespace-only value
/// can't be a valid key, so it counts as missing.
fn non_blank(value: Option<String>) -> Option<String> {
    value.filter(|v| !v.trim().is_empty())
}

/// A credential value on its way to a probe request. Deliberately has no
/// `Display`, and its `Debug` is redacted, so it can't leak into the report.
#[derive(Clone)]
struct Secret(String);

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Secret(<redacted>)")
    }
}

// =============================================================================
// Report model and rendering
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Ok,
    Info,
    Warn,
    Fail,
}

impl Status {
    fn symbol(self) -> String {
        match self {
            Status::Ok => "✓".green().bold().to_string(),
            Status::Info => "•".blue().to_string(),
            Status::Warn => "!".yellow().bold().to_string(),
            Status::Fail => "✗".red().bold().to_string(),
        }
    }
}

/// One line of the report, with optional indented detail lines under it.
#[derive(Debug, Clone)]
struct Check {
    status: Status,
    text: String,
    details: Vec<String>,
}

impl Check {
    fn new(status: Status, text: impl Into<String>) -> Self {
        Check {
            status,
            text: text.into(),
            details: Vec::new(),
        }
    }

    fn detail(mut self, line: impl Into<String>) -> Self {
        self.details.push(line.into());
        self
    }
}

#[derive(Debug)]
struct Section {
    title: &'static str,
    checks: Vec<Check>,
}

fn render(sections: &[Section]) -> String {
    let mut out = String::new();
    for section in sections {
        out.push_str(&format!("{}\n", section.title.bold()));
        for check in &section.checks {
            out.push_str(&format!("  {} {}\n", check.status.symbol(), check.text));
            for line in &check.details {
                out.push_str(&format!("      {}\n", line));
            }
        }
        out.push('\n');
    }
    out
}

fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        let prefix: String = text.chars().take(max_chars).collect();
        format!("{prefix}…")
    }
}

/// Words that mark a `name = value` line as holding a credential.
const SECRET_WORDS: [&str; 4] = ["key", "token", "secret", "password"];

/// Further words that mark a URL query parameter as a credential: signed
/// URLs (`sig`, `X-Amz-Signature`, `X-Amz-Credential`), `auth`, `pass`,
/// `pwd`. Too broad for whole lines, but over-matching a parameter name
/// only hides a harmless value.
const SECRET_PARAM_WORDS: [&str; 5] = ["sig", "auth", "credential", "pass", "pwd"];

/// Best-effort scrub of anything key-like from text we didn't write. A TOML
/// parse error quotes the offending line of the config file, and that line
/// could hold an `api_key`, a key pasted on a line of its own, or a
/// `base_url` with credentials in it.
fn redact_secrets(text: &str) -> String {
    text.lines().map(redact_line).collect::<Vec<_>>().join("\n")
}

fn redact_line(line: &str) -> String {
    let lower = line.to_lowercase();
    if SECRET_WORDS.iter().any(|word| lower.contains(word)) {
        if let Some(eq) = line.find('=') {
            return format!("{}= <redacted>", &line[..eq]);
        }
    }
    line.split(' ')
        .map(|word| {
            if looks_like_secret(word) {
                "<redacted>".to_string()
            } else if word.contains("://") {
                redact_url(word)
            } else {
                word.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// A URL for display, with credentials replaced by `<redacted>`: the
/// userinfo (`user:pass@`), the values of secret-looking query or fragment
/// parameters (`?api_key=…`, `&sig=…`), and any parameter value that looks
/// like a pasted key. Works on the text as written, so the rest shows
/// exactly as configured and a URL that doesn't parse is scrubbed too.
fn redact_url(url: &str) -> String {
    let scheme_end = url.find("://").map_or(0, |i| i + 3);
    let (scheme, rest) = url.split_at(scheme_end);

    // The userinfo ends at the last `@` of the authority, which runs to the
    // next `/`, `\`, `?` or `#`, as URL parsers read it. A URL that doesn't
    // parse may have one of those unencoded in its password, so there the
    // last `@` anywhere ends it.
    let authority_len = if Url::parse(url).is_ok() {
        rest.find(['/', '\\', '?', '#']).unwrap_or(rest.len())
    } else {
        rest.len()
    };
    let rest = match rest[..authority_len].rfind('@') {
        Some(at) => format!("<redacted>{}", &rest[at..]),
        None => rest.to_string(),
    };

    let (rest, fragment) = match rest.split_once('#') {
        Some((rest, fragment)) => (rest, Some(fragment)),
        None => (rest.as_str(), None),
    };
    let mut out = match rest.split_once('?') {
        Some((path, query)) => format!("{scheme}{path}?{}", redact_url_params(query)),
        None => format!("{scheme}{rest}"),
    };
    if let Some(fragment) = fragment {
        out.push('#');
        out.push_str(&redact_url_params(fragment));
    }
    out
}

/// `redact_url` for an `a=1&b=2` query or fragment.
fn redact_url_params(params: &str) -> String {
    params
        .split('&')
        .map(|pair| match pair.split_once('=') {
            Some((name, value))
                if !value.is_empty() && (is_secret_param(name) || looks_like_secret(value)) =>
            {
                format!("{name}=<redacted>")
            }
            None if looks_like_secret(pair) => "<redacted>".to_string(),
            _ => pair.to_string(),
        })
        .collect::<Vec<_>>()
        .join("&")
}

fn is_secret_param(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    SECRET_WORDS
        .iter()
        .chain(SECRET_PARAM_WORDS.iter())
        .any(|word| name.contains(word))
}

/// Heuristic for a pasted API key or token: 20+ chars from the key/base64
/// alphabet mixing upper case, lower case and digits. Model names are all
/// lower case and paths contain `.` or `\`, so neither trips it.
fn looks_like_secret(word: &str) -> bool {
    let word = word.trim_matches(|c: char| matches!(c, '"' | '\'' | '`' | ',' | ';'));
    word.len() >= 20
        && word
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '+' | '/' | '='))
        && word.chars().any(|c| c.is_ascii_uppercase())
        && word.chars().any(|c| c.is_ascii_lowercase())
        && word.chars().any(|c| c.is_ascii_digit())
}

// =============================================================================
// System: version, OS, config file, proxy
// =============================================================================

/// `Config::config_path()` returns a bare `PathBuf` today, and a pending
/// config-loading change makes it fallible (`Result<PathBuf>`). Accepting
/// either keeps this module compiling whichever of the two lands first.
trait IntoPathResult {
    fn into_path_result(self) -> Result<PathBuf>;
}

impl IntoPathResult for PathBuf {
    fn into_path_result(self) -> Result<PathBuf> {
        Ok(self)
    }
}

impl IntoPathResult for Result<PathBuf> {
    fn into_path_result(self) -> Result<PathBuf> {
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ConfigState {
    Loaded,
    /// No file: the CLI runs on built-in defaults.
    Missing,
    /// The file exists but failed to load; holds the redacted error chain.
    Invalid(String),
}

struct LoadedConfig {
    path: Result<PathBuf, String>,
    state: ConfigState,
    /// The parsed config, or built-in defaults when missing or invalid.
    config: Config,
    /// A config at `~/.config/ai-cli/config.toml` that this build doesn't read
    /// (on macOS `dirs::config_dir()` is `~/Library/Application Support`).
    stray: Option<PathBuf>,
}

fn load_config() -> LoadedConfig {
    let path = Config::config_path()
        .into_path_result()
        .map_err(|e| redact_secrets(&format!("{e:#}")));
    let exists = path.as_ref().is_ok_and(|p| p.exists());
    let (state, config) = match Config::load() {
        Ok(config) if exists => (ConfigState::Loaded, config),
        Ok(config) => (ConfigState::Missing, config),
        Err(e) => (
            ConfigState::Invalid(redact_secrets(&format!("{e:#}"))),
            Config::default(),
        ),
    };
    let stray = match (&path, &state) {
        (Ok(path), ConfigState::Missing) => dirs::home_dir()
            .map(|home| home.join(".config").join("ai-cli").join("config.toml"))
            .filter(|alt| alt != path && alt.exists()),
        _ => None,
    };
    LoadedConfig {
        path,
        state,
        config,
        stray,
    }
}

fn system_section(loaded: &LoadedConfig, env: &dyn Environment) -> Section {
    let mut checks = vec![
        Check::new(
            Status::Info,
            format!(
                "ai-cli {} (build {}) on {}/{}",
                env!("CARGO_PKG_VERSION"),
                BUILD_COMMIT,
                std::env::consts::OS,
                std::env::consts::ARCH
            ),
        ),
        config_check(loaded),
    ];
    if let Some(check) = proxy_check(env) {
        checks.push(check);
    }
    Section {
        title: "System",
        checks,
    }
}

fn config_check(loaded: &LoadedConfig) -> Check {
    let path = match &loaded.path {
        Ok(path) => path.display().to_string(),
        Err(e) => format!("<could not resolve: {e}>"),
    };
    match &loaded.state {
        ConfigState::Loaded => {
            Check::new(Status::Ok, format!("config: {path}")).detail("found and parsed")
        }
        ConfigState::Missing => match &loaded.stray {
            Some(stray) => Check::new(Status::Warn, format!("config: {path} not found"))
                .detail(format!(
                    "found {} instead, but this build reads the path above; move it there",
                    stray.display()
                ))
                .detail("using built-in defaults until then"),
            None => Check::new(Status::Info, format!("config: {path} not found"))
                .detail("using built-in defaults; run `ai --init` to create one"),
        },
        ConfigState::Invalid(err) => {
            let mut check = Check::new(Status::Fail, format!("config: {path} failed to load"));
            for line in err.lines() {
                check = check.detail(line);
            }
            check.detail("`ai` exits with this error until the file is fixed; checks below use built-in defaults")
        }
    }
}

/// Presence only: proxy URLs can carry credentials (`http://user:pass@…`).
fn proxy_check(env: &dyn Environment) -> Option<Check> {
    let set: Vec<&str> = PROXY_VARS
        .iter()
        .copied()
        .filter(|var| non_blank(env.var(var)).is_some())
        .collect();
    if set.is_empty() {
        return None;
    }
    Some(Check::new(
        Status::Info,
        format!("proxy: {} set; HTTP requests go through it", set.join(", ")),
    ))
}

// =============================================================================
// Providers: enabled, models, auth source, endpoint, Bedrock routing
// =============================================================================

/// Which wire protocol a provider speaks, which decides how to probe it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderKind {
    Anthropic,
    OpenAiCompatible,
    Gemini,
    OpenRouter,
    Ollama,
    /// `base_url` points at `bedrock-runtime`: served by our native Converse client.
    BedrockConverse,
    Unknown,
}

fn provider_kind(name: &str, cfg: &ProviderConfig) -> ProviderKind {
    if cfg.base_url.as_deref().is_some_and(is_bedrock_base_url) {
        return ProviderKind::BedrockConverse;
    }
    match name {
        "anthropic" => ProviderKind::Anthropic,
        "openai" | "xai" => ProviderKind::OpenAiCompatible,
        "google" => ProviderKind::Gemini,
        "openrouter" => ProviderKind::OpenRouter,
        "ollama" => ProviderKind::Ollama,
        // A custom provider with its own base_url is almost always an
        // OpenAI-compatible server (vLLM, LiteLLM, Bedrock's OpenAI API, …).
        _ if cfg.base_url.is_some() => ProviderKind::OpenAiCompatible,
        _ => ProviderKind::Unknown,
    }
}

/// genai's built-in endpoint for each provider we know by name.
fn default_endpoint(name: &str) -> Option<&'static str> {
    match name {
        "anthropic" => Some("https://api.anthropic.com/v1/"),
        "openai" => Some("https://api.openai.com/v1/"),
        "xai" => Some("https://api.x.ai/v1/"),
        "google" => Some("https://generativelanguage.googleapis.com/v1beta/"),
        "openrouter" => Some("https://openrouter.ai/api/v1"),
        "ollama" => Some("http://localhost:11434"),
        _ => None,
    }
}

/// The endpoint that requests go to, and whether it came from `base_url`.
fn endpoint_for(name: &str, cfg: &ProviderConfig) -> Option<(String, bool)> {
    match cfg
        .base_url
        .as_deref()
        .map(str::trim)
        .filter(|u| !u.is_empty())
    {
        Some(url) => Some((url.to_string(), true)),
        None => default_endpoint(name).map(|url| (url.to_string(), false)),
    }
}

/// True for any AWS Bedrock endpoint: the `bedrock-runtime` hosts served by
/// the native Converse client, and Bedrock's OpenAI-compatible hosts (e.g.
/// `bedrock-mantle.<region>.api.aws`).
fn is_bedrock_endpoint(url: &str) -> bool {
    if is_bedrock_base_url(url) {
        return true;
    }
    Url::parse(url)
        .ok()
        .and_then(|u| u.host_str().map(str::to_ascii_lowercase))
        .is_some_and(|host| {
            host.contains("bedrock")
                && (host.ends_with(".amazonaws.com") || host.ends_with(".api.aws"))
        })
}

/// The env var a provider conventionally reads its key from.
fn conventional_key_env(name: &str, kind: ProviderKind) -> Option<&'static str> {
    if kind == ProviderKind::BedrockConverse {
        return Some("AWS_BEARER_TOKEN_BEDROCK");
    }
    match name {
        "anthropic" => Some("ANTHROPIC_API_KEY"),
        "openai" => Some("OPENAI_API_KEY"),
        "xai" => Some("XAI_API_KEY"),
        "google" => Some("GEMINI_API_KEY"),
        "openrouter" => Some("OPENROUTER_API_KEY"),
        _ => None,
    }
}

/// Where a provider's credentials come from. Carries lengths, never values.
#[derive(Debug, Clone, PartialEq, Eq)]
enum AuthSource {
    /// Ollama: a local server that needs no key.
    NotRequired,
    /// Anthropic `auth_mode = "subscription"` (Claude Code's OAuth token).
    Subscription {
        available: bool,
    },
    /// `auth_mode = "subscription"` on a provider that can't use it.
    SubscriptionUnsupported,
    /// `api_key` written directly in the config file.
    ConfigKey {
        len: usize,
    },
    ConfigKeyBlank,
    EnvVar {
        var: String,
        len: usize,
    },
    EnvBlank {
        var: String,
    },
    EnvUnset {
        vars: Vec<String>,
    },
    /// Neither `api_key` nor `api_key_env` is configured. `unreferenced` names
    /// the conventional env var when it is set but the config doesn't use it.
    NotConfigured {
        unreferenced: Option<String>,
    },
}

impl AuthSource {
    fn usable(&self) -> bool {
        matches!(
            self,
            AuthSource::NotRequired
                | AuthSource::Subscription { available: true }
                | AuthSource::ConfigKey { .. }
                | AuthSource::EnvVar { .. }
        )
    }

    fn describe(&self) -> String {
        match self {
            AuthSource::NotRequired => "none required (local server)".to_string(),
            AuthSource::Subscription { available: true } => {
                "Claude subscription (Claude Code OAuth token found)".to_string()
            }
            AuthSource::Subscription { available: false } => {
                "Claude subscription, but no valid Claude Code OAuth token (run `claude` and log in with a Pro/Max account)"
                    .to_string()
            }
            AuthSource::SubscriptionUnsupported => {
                "auth_mode = \"subscription\" only works for the anthropic provider".to_string()
            }
            AuthSource::ConfigKey { len } => {
                format!("config api_key (set, {len} chars); prefer api_key_env")
            }
            AuthSource::ConfigKeyBlank => "config api_key is blank".to_string(),
            AuthSource::EnvVar { var, len } => format!("env {var} (set, {len} chars)"),
            AuthSource::EnvBlank { var } => format!("env {var} is set but blank"),
            AuthSource::EnvUnset { vars } => format!("env {} not set", vars.join(" / ")),
            AuthSource::NotConfigured {
                unreferenced: Some(var),
            } => format!(
                "no api_key or api_key_env configured ({var} is set but unused; add api_key_env = \"{var}\")"
            ),
            AuthSource::NotConfigured { unreferenced: None } => {
                "no api_key or api_key_env configured".to_string()
            }
        }
    }
}

/// Work out where a provider's credentials come from, mirroring the
/// precedence the query path uses (`Config::get_api_key`: config `api_key`,
/// then `api_key_env`, with `GOOGLE_API_KEY` as the fallback for
/// `GEMINI_API_KEY`). Bedrock `bedrock-runtime` providers read the token from
/// `api_key_env` only (`BedrockClient::from_env`), so `api_key` is ignored
/// for them. Returns the key alongside so the probe can use it.
fn resolve_auth(
    name: &str,
    cfg: &ProviderConfig,
    env: &dyn Environment,
) -> (AuthSource, Option<Secret>) {
    if name == "ollama" {
        return (AuthSource::NotRequired, None);
    }
    if cfg.auth_mode == AuthMode::Subscription {
        if name != "anthropic" {
            return (AuthSource::SubscriptionUnsupported, None);
        }
        let available = env.subscription_token_available();
        return (AuthSource::Subscription { available }, None);
    }

    let kind = provider_kind(name, cfg);
    if kind != ProviderKind::BedrockConverse {
        if let Some(key) = &cfg.api_key {
            if key.trim().is_empty() {
                return (AuthSource::ConfigKeyBlank, None);
            }
            let len = key.chars().count();
            return (AuthSource::ConfigKey { len }, Some(Secret(key.clone())));
        }
    }

    let Some(var) = cfg.api_key_env.as_deref() else {
        let unreferenced = conventional_key_env(name, kind)
            .filter(|var| non_blank(env.var(var)).is_some())
            .map(str::to_string);
        return (AuthSource::NotConfigured { unreferenced }, None);
    };

    let mut candidates = vec![var.to_string()];
    if var == "GEMINI_API_KEY" && kind != ProviderKind::BedrockConverse {
        candidates.push("GOOGLE_API_KEY".to_string());
    }
    // First var that is set wins, even if blank: that's what the query path
    // does, so a blank GEMINI_API_KEY shadows a valid GOOGLE_API_KEY there too.
    for var in &candidates {
        if let Some(value) = env.var(var) {
            if value.trim().is_empty() {
                return (AuthSource::EnvBlank { var: var.clone() }, None);
            }
            let len = value.chars().count();
            return (
                AuthSource::EnvVar {
                    var: var.clone(),
                    len,
                },
                Some(Secret(value)),
            );
        }
    }
    (AuthSource::EnvUnset { vars: candidates }, None)
}

#[derive(Debug)]
struct ProviderEntry {
    name: String,
    aliases: Vec<String>,
    enabled: bool,
    fast_model: String,
    smart_model: String,
    kind: ProviderKind,
    endpoint: Option<(String, bool)>,
    auth: AuthSource,
    secret: Option<Secret>,
}

impl ProviderEntry {
    /// The provider name plus any aliases that differ from it.
    fn label(&self) -> String {
        let aliases: Vec<&str> = self
            .aliases
            .iter()
            .map(String::as_str)
            .filter(|alias| !alias.eq_ignore_ascii_case(&self.name))
            .collect();
        if aliases.is_empty() {
            self.name.clone()
        } else {
            format!("{} ({})", self.name, aliases.join(", "))
        }
    }

    fn is_bedrock(&self) -> bool {
        self.kind == ProviderKind::BedrockConverse
            || self
                .endpoint
                .as_ref()
                .is_some_and(|(url, _)| is_bedrock_endpoint(url))
    }
}

fn provider_entries(config: &Config, env: &dyn Environment) -> Vec<ProviderEntry> {
    let mut providers: Vec<(&String, &ProviderConfig)> = config.providers.iter().collect();
    providers.sort_by(|a, b| a.0.cmp(b.0));
    providers
        .into_iter()
        .map(|(name, cfg)| {
            let (auth, secret) = resolve_auth(name, cfg, env);
            ProviderEntry {
                name: name.clone(),
                aliases: cfg.aliases.clone(),
                enabled: cfg.enabled,
                fast_model: cfg.get_model(&ModelTier::Fast).to_string(),
                smart_model: cfg.get_model(&ModelTier::Smart).to_string(),
                kind: provider_kind(name, cfg),
                endpoint: endpoint_for(name, cfg),
                auth,
                secret,
            }
        })
        .collect()
}

fn or_none(model: &str) -> &str {
    if model.is_empty() {
        "(none)"
    } else {
        model
    }
}

fn provider_check(entry: &ProviderEntry) -> Check {
    let (status, text) = if !entry.enabled {
        (Status::Info, format!("{}: disabled", entry.label()))
    } else if entry.auth.usable() {
        (Status::Ok, entry.label())
    } else {
        (
            Status::Warn,
            format!(
                "{}: no usable credentials, skipped at query time",
                entry.label()
            ),
        )
    };
    let endpoint = match &entry.endpoint {
        // A base_url can carry credentials (`https://user:pass@…`, `?key=…`).
        Some((url, true)) => format!("{} (base_url)", redact_url(url)),
        Some((url, false)) => format!("{url} (default)"),
        None => "unknown (no base_url and no built-in default)".to_string(),
    };
    let bedrock = if entry.kind == ProviderKind::BedrockConverse {
        "yes (native Converse API)"
    } else if entry.is_bedrock() {
        "yes (OpenAI-compatible API)"
    } else {
        "no"
    };
    Check::new(status, text)
        .detail(format!(
            "models:   fast {}, smart {}",
            or_none(&entry.fast_model),
            or_none(&entry.smart_model)
        ))
        .detail(format!("auth:     {}", entry.auth.describe()))
        .detail(format!("endpoint: {endpoint}"))
        .detail(format!("bedrock:  {bedrock}"))
}

fn providers_section(entries: &[ProviderEntry]) -> Section {
    let mut checks: Vec<Check> = entries.iter().map(provider_check).collect();
    if checks.is_empty() {
        checks.push(Check::new(Status::Warn, "no providers configured"));
    }
    Section {
        title: "Providers",
        checks,
    }
}

// =============================================================================
// Connectivity probes: planning (pure) and execution (network)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbeAuth {
    None,
    /// Anthropic: `x-api-key` plus `anthropic-version`.
    XApiKey,
    Bearer,
    /// Gemini: `x-goog-api-key` header, never `?key=` in the URL.
    GoogApiKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbeMode {
    /// A free authenticated call that must return 2xx.
    ExpectSuccess,
    /// Reachability only: any HTTP response over a verified TLS connection
    /// passes. Used where the only authenticated calls are billed (Bedrock)
    /// or where we don't replay the credential (subscription OAuth).
    Reachability,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ProbeRequest {
    url: String,
    auth: ProbeAuth,
    mode: ProbeMode,
}

fn url_join(base: &str, path: &str) -> String {
    format!("{}/{}", base.trim_end_matches('/'), path)
}

/// Ollama's native API lives at the server root; strip the `/v1` of its
/// OpenAI-compatible base URL if the config (or routing) added one.
fn ollama_root(endpoint: &str) -> &str {
    let trimmed = endpoint.trim_end_matches('/');
    trimmed.strip_suffix("/v1").unwrap_or(trimmed)
}

/// The cheapest request that proves a provider works: a list-models (or
/// key-info) call, which no provider bills for.
fn probe_request(kind: ProviderKind, endpoint: &str, auth: &AuthSource) -> Option<ProbeRequest> {
    let (url, auth, mode) = match kind {
        ProviderKind::Anthropic if matches!(auth, AuthSource::Subscription { .. }) => (
            url_join(endpoint, "models"),
            ProbeAuth::None,
            ProbeMode::Reachability,
        ),
        ProviderKind::Anthropic => (
            url_join(endpoint, "models"),
            ProbeAuth::XApiKey,
            ProbeMode::ExpectSuccess,
        ),
        ProviderKind::OpenAiCompatible => (
            url_join(endpoint, "models"),
            ProbeAuth::Bearer,
            ProbeMode::ExpectSuccess,
        ),
        ProviderKind::Gemini => (
            url_join(endpoint, "models?pageSize=1"),
            ProbeAuth::GoogApiKey,
            ProbeMode::ExpectSuccess,
        ),
        ProviderKind::OpenRouter => (
            url_join(endpoint, "key"),
            ProbeAuth::Bearer,
            ProbeMode::ExpectSuccess,
        ),
        ProviderKind::Ollama => (
            url_join(ollama_root(endpoint), "api/tags"),
            ProbeAuth::None,
            ProbeMode::ExpectSuccess,
        ),
        ProviderKind::BedrockConverse => {
            let origin = Url::parse(endpoint).ok()?.origin();
            if !origin.is_tuple() {
                return None;
            }
            (
                format!("{}/", origin.ascii_serialization()),
                ProbeAuth::None,
                ProbeMode::Reachability,
            )
        }
        ProviderKind::Unknown => return None,
    };
    Some(ProbeRequest { url, auth, mode })
}

#[derive(Debug)]
struct ProbeTarget {
    provider: String,
    request: ProbeRequest,
    secret: Option<Secret>,
    bedrock: bool,
    /// Shown under a passing reachability probe: what it did *not* verify.
    note: Option<&'static str>,
    /// The proxy the request goes through (credentials redacted), or `None`
    /// for a direct connection.
    proxy: Option<String>,
}

fn probe_target(entry: &ProviderEntry, env: &dyn Environment) -> Result<ProbeTarget, &'static str> {
    if !entry.auth.usable() {
        return Err("no usable credentials");
    }
    let Some((endpoint, _)) = &entry.endpoint else {
        return Err("no known endpoint");
    };
    let Some(request) = probe_request(entry.kind, endpoint, &entry.auth) else {
        return Err("no probe for this provider type");
    };
    let note = match (entry.kind, request.mode) {
        (ProviderKind::BedrockConverse, _) => {
            Some("token present but not validated: doctor never makes a billed Bedrock call")
        }
        (_, ProbeMode::Reachability) => {
            Some("OAuth token present but not validated: only reachability is checked")
        }
        _ => None,
    };
    let proxy = Url::parse(&request.url)
        .ok()
        .and_then(|url| proxy_for(&url, env));
    Ok(ProbeTarget {
        provider: entry.name.clone(),
        request,
        secret: entry.secret.clone(),
        bedrock: entry.is_bedrock(),
        note,
        proxy,
    })
}

/// How a probe ended.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ProbeOutcome {
    Success {
        status: u16,
    },
    /// Reachability probe got an HTTP response (any status).
    Reachable {
        status: u16,
    },
    AuthFailed {
        status: u16,
    },
    NotFound,
    RateLimited,
    ServerError {
        status: u16,
    },
    UnexpectedStatus {
        status: u16,
    },
    Tls {
        detail: String,
    },
    Dns {
        detail: String,
    },
    Timeout,
    ConnectRefused,
    ConnectFailed {
        detail: String,
    },
    InvalidUrl {
        detail: String,
    },
    Other {
        detail: String,
    },
    /// DNS sent the host to a network filter's block page and the request
    /// connected directly, so it went there. Whatever HTTP status came back
    /// (with the filter's root CA trusted, the block page can complete TLS
    /// and answer 2xx), this is not the provider.
    Filtered {
        http: Option<u16>,
    },
}

impl ProbeOutcome {
    fn status(&self) -> Status {
        match self {
            ProbeOutcome::Success { .. } | ProbeOutcome::Reachable { .. } => Status::Ok,
            // Reachable and authenticated, just not right now.
            ProbeOutcome::RateLimited | ProbeOutcome::ServerError { .. } => Status::Warn,
            _ => Status::Fail,
        }
    }

    /// Failures of the network path itself (as opposed to the provider
    /// answering and saying no). These are what a filter or firewall causes.
    fn is_network_failure(&self) -> bool {
        matches!(
            self,
            ProbeOutcome::Tls { .. }
                | ProbeOutcome::Dns { .. }
                | ProbeOutcome::Timeout
                | ProbeOutcome::ConnectRefused
                | ProbeOutcome::ConnectFailed { .. }
                | ProbeOutcome::Filtered { .. }
        )
    }

    fn describe(&self) -> String {
        match self {
            ProbeOutcome::Success { status } => format!("OK (HTTP {status})"),
            ProbeOutcome::Reachable { status } => format!("reachable (HTTP {status})"),
            ProbeOutcome::AuthFailed { status } => {
                format!("authentication failed (HTTP {status})")
            }
            ProbeOutcome::NotFound => "endpoint not found (HTTP 404)".to_string(),
            ProbeOutcome::RateLimited => "rate limited (HTTP 429)".to_string(),
            ProbeOutcome::ServerError { status } => format!("server error (HTTP {status})"),
            ProbeOutcome::UnexpectedStatus { status } => format!("unexpected HTTP {status}"),
            ProbeOutcome::Tls { .. } => "TLS error".to_string(),
            ProbeOutcome::Dns { .. } => "DNS lookup failed".to_string(),
            ProbeOutcome::Timeout => "timed out".to_string(),
            ProbeOutcome::ConnectRefused => "connection refused".to_string(),
            ProbeOutcome::ConnectFailed { .. } => "connection failed".to_string(),
            ProbeOutcome::InvalidUrl { .. } => "invalid endpoint URL".to_string(),
            ProbeOutcome::Other { .. } => "request failed".to_string(),
            ProbeOutcome::Filtered { http: Some(status) } => {
                format!("blocked by a network filter (its block page answered HTTP {status})")
            }
            ProbeOutcome::Filtered { http: None } => "blocked by a network filter".to_string(),
        }
    }

    /// The HTTP status behind this outcome, if a response came back.
    fn http_status(&self) -> Option<u16> {
        match self {
            ProbeOutcome::Success { status }
            | ProbeOutcome::Reachable { status }
            | ProbeOutcome::AuthFailed { status }
            | ProbeOutcome::ServerError { status }
            | ProbeOutcome::UnexpectedStatus { status } => Some(*status),
            ProbeOutcome::NotFound => Some(404),
            ProbeOutcome::RateLimited => Some(429),
            ProbeOutcome::Filtered { http } => *http,
            _ => None,
        }
    }

    fn detail(&self) -> Option<&str> {
        match self {
            ProbeOutcome::Tls { detail }
            | ProbeOutcome::Dns { detail }
            | ProbeOutcome::ConnectFailed { detail }
            | ProbeOutcome::InvalidUrl { detail }
            | ProbeOutcome::Other { detail } => Some(detail.as_str()),
            _ => None,
        }
    }

    fn hint(&self, local: bool) -> Option<&'static str> {
        match self {
            ProbeOutcome::AuthFailed { .. } => {
                Some("the credentials were rejected: check the key is current and belongs to this provider")
            }
            ProbeOutcome::NotFound => Some("the endpoint doesn't serve this path: check base_url"),
            ProbeOutcome::RateLimited => Some("the key works but is being rate limited right now"),
            ProbeOutcome::ServerError { .. } => {
                Some("the provider returned a server error; usually transient")
            }
            ProbeOutcome::Tls { .. } => Some(
                "TLS verification failed: an intercepting proxy or network filter (e.g. Cisco Umbrella, Zscaler) may be in the way",
            ),
            ProbeOutcome::Dns { .. } => Some("check your network connection and DNS"),
            // Some setups (e.g. WSL2) time out instead of refusing a closed
            // local port, so both point at the local server.
            ProbeOutcome::Timeout | ProbeOutcome::ConnectRefused if local => {
                Some("nothing answered: is the local server (`ollama serve`) running?")
            }
            ProbeOutcome::Timeout => {
                Some("no response in time: a firewall or proxy may be dropping the traffic")
            }
            ProbeOutcome::ConnectRefused => {
                Some("the port is closed or a firewall is rejecting the connection")
            }
            _ => None,
        }
    }
}

/// Map the HTTP status of an authenticated probe to an outcome.
fn classify_http_status(status: u16) -> ProbeOutcome {
    match status {
        200..=299 => ProbeOutcome::Success { status },
        401 | 403 => ProbeOutcome::AuthFailed { status },
        404 => ProbeOutcome::NotFound,
        429 => ProbeOutcome::RateLimited,
        500..=599 => ProbeOutcome::ServerError { status },
        _ => ProbeOutcome::UnexpectedStatus { status },
    }
}

fn classify_response(status: u16, mode: ProbeMode) -> ProbeOutcome {
    match mode {
        ProbeMode::ExpectSuccess => classify_http_status(status),
        ProbeMode::Reachability => ProbeOutcome::Reachable { status },
    }
}

/// What matters about a failed request, lifted out of `reqwest::Error` so the
/// classification itself is a pure, testable function.
#[derive(Debug, Default)]
struct TransportFacts {
    timeout: bool,
    connect: bool,
    builder: bool,
    io_kind: Option<std::io::ErrorKind>,
    /// Every message in the error's source chain, joined, for keyword matching.
    chain: String,
    /// The innermost message (the root cause), shown to the user.
    root: String,
}

fn transport_facts(err: &reqwest::Error) -> TransportFacts {
    // reqwest ends its own messages with "for url (…)", and the query of a
    // configured base_url can carry a key.
    let scrub = |message: String| match err.url() {
        Some(url) => message.replace(url.as_str(), &redact_url(url.as_str())),
        None => message,
    };
    let mut messages = vec![scrub(err.to_string())];
    let mut io_kind = None;
    let mut source = std::error::Error::source(err);
    while let Some(e) = source {
        if io_kind.is_none() {
            io_kind = e.downcast_ref::<std::io::Error>().map(std::io::Error::kind);
        }
        messages.push(scrub(e.to_string()));
        source = e.source();
    }
    TransportFacts {
        timeout: err.is_timeout(),
        connect: err.is_connect(),
        builder: err.is_builder(),
        io_kind,
        root: messages.last().cloned().unwrap_or_default(),
        chain: messages.join(": "),
    }
}

fn classify_transport(facts: &TransportFacts) -> ProbeOutcome {
    use std::io::ErrorKind;

    let chain = facts.chain.to_lowercase();
    let mentions = |needles: &[&str]| needles.iter().any(|n| chain.contains(n));
    let root = truncate(&facts.root, 200);

    if facts.builder {
        ProbeOutcome::Other { detail: root }
    } else if facts.timeout || facts.io_kind == Some(ErrorKind::TimedOut) {
        ProbeOutcome::Timeout
    } else if mentions(&[
        "dns error",
        "failed to lookup address",
        "name or service not known",
        "nodename nor servname",
        "no such host",
        "temporary failure in name resolution",
    ]) {
        ProbeOutcome::Dns { detail: root }
    } else if facts.io_kind == Some(ErrorKind::ConnectionRefused)
        || mentions(&["connection refused"])
    {
        ProbeOutcome::ConnectRefused
    } else if mentions(&["certificate", "tls", "ssl", "handshake"]) {
        ProbeOutcome::Tls { detail: root }
    } else if facts.connect {
        ProbeOutcome::ConnectFailed { detail: root }
    } else {
        ProbeOutcome::Other { detail: root }
    }
}

fn classify_reqwest_error(err: &reqwest::Error) -> ProbeOutcome {
    classify_transport(&transport_facts(err))
}

/// True for addresses Cisco Umbrella uses for its block pages
/// (146.112.0.0/16), including their IPv4-mapped IPv6 form.
fn is_filtered_ip(ip: IpAddr) -> bool {
    let v4 = match ip {
        IpAddr::V4(v4) => v4,
        IpAddr::V6(v6) => match v6.to_ipv4_mapped() {
            Some(v4) => v4,
            None => return false,
        },
    };
    let [a, b, _, _] = v4.octets();
    a == 146 && b == 112
}

fn is_local_host(host: &str) -> bool {
    let host = host.to_ascii_lowercase();
    host == "localhost"
        || host.ends_with(".localhost")
        || host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

// Proxy routing. The probe client keeps reqwest's default proxy handling, as
// the query path's clients do, so a probe takes the route a query would.
// Through a proxy, the proxy resolves the host, so our local DNS answer only
// describes a direct connection. `proxy_for` predicts the route from the
// environment by the rules reqwest applies (hyper-util's proxy `Matcher`).
// Not mirrored: the OS proxy settings reqwest also reads on macOS and Windows
// when these vars are unset. With only those configured, a probe counts as
// direct.

/// The proxy reqwest sends a request for `url` through, for display (with
/// credentials redacted), or `None` when it connects directly.
fn proxy_for(url: &Url, env: &dyn Environment) -> Option<String> {
    // Under CGI a request header can set `HTTP_PROXY` ("httpoxy"), so
    // reqwest ignores every proxy var there.
    if env.var("REQUEST_METHOD").is_some() {
        return None;
    }
    // The first var that is set wins, even when its value is blank or
    // unusable (then `ALL_PROXY` is tried).
    let first_set = |names: [&str; 2]| names.into_iter().find_map(|name| env.var(name));
    let scheme_vars = match url.scheme() {
        "https" => ["HTTPS_PROXY", "https_proxy"],
        "http" => ["HTTP_PROXY", "http_proxy"],
        _ => return None,
    };
    let proxy = first_set(scheme_vars)
        .filter(|value| is_usable_proxy(value))
        .or_else(|| first_set(["ALL_PROXY", "all_proxy"]).filter(|value| is_usable_proxy(value)))?;
    let no_proxy = first_set(["NO_PROXY", "no_proxy"]).unwrap_or_default();
    if no_proxy_matches(&no_proxy, url.host_str()?) {
        return None;
    }
    Some(redact_url(&proxy))
}

/// Whether reqwest accepts `value` as a proxy: an `http`, `https` or SOCKS
/// URL, or a bare `host:port` (taken as `http`). An approximation of its URI
/// parsing, which also rejects whitespace and a missing host.
fn is_usable_proxy(value: &str) -> bool {
    let (scheme, rest) = value.split_once("://").unwrap_or(("http", value));
    let known_scheme = scheme.eq_ignore_ascii_case("http")
        || scheme.eq_ignore_ascii_case("https")
        || matches!(scheme, "socks4" | "socks4a" | "socks5" | "socks5h");
    known_scheme
        && !rest.is_empty()
        && !rest.starts_with('/')
        && !value.contains(char::is_whitespace)
}

/// Whether a `NO_PROXY` list exempts `host`, by reqwest's (curl-derived)
/// rules: comma-separated entries; an IP host matches an IP or CIDR entry;
/// a domain matches `*`, or an entry naming it or a parent domain, with or
/// without a leading dot. Entries are compared as written (a parsed URL's
/// host is already lower case).
fn no_proxy_matches(no_proxy: &str, host: &str) -> bool {
    let mut entries = no_proxy
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty());
    let host = host.trim_start_matches('[').trim_end_matches(']');
    match host.parse::<IpAddr>() {
        Ok(ip) => {
            entries.any(|entry| entry.parse::<IpAddr>() == Ok(ip) || cidr_contains(entry, ip))
        }
        Err(_) => entries.any(|entry| {
            entry == "*"
                || entry == host
                || entry.strip_prefix('.') == Some(host)
                || host
                    .strip_suffix(entry)
                    .is_some_and(|rest| entry.starts_with('.') || rest.ends_with('.'))
        }),
    }
}

/// Whether `ip` lies in the CIDR block `entry` (`10.0.0.0/8`, `fd00::/8`).
fn cidr_contains(entry: &str, ip: IpAddr) -> bool {
    let Some((net, len)) = entry.split_once('/') else {
        return false;
    };
    let (Ok(net), Ok(len)) = (net.parse::<IpAddr>(), len.parse::<u32>()) else {
        return false;
    };
    match (net, ip) {
        (IpAddr::V4(net), IpAddr::V4(ip)) if len <= 32 => {
            let mask = u32::MAX.checked_shl(32 - len).unwrap_or(0);
            u32::from(net) & mask == u32::from(ip) & mask
        }
        (IpAddr::V6(net), IpAddr::V6(ip)) if len <= 128 => {
            let mask = u128::MAX.checked_shl(128 - len).unwrap_or(0);
            u128::from(net) & mask == u128::from(ip) & mask
        }
        _ => false,
    }
}

fn format_ips(ips: &[IpAddr]) -> String {
    const SHOWN: usize = 4;
    let mut out = ips
        .iter()
        .take(SHOWN)
        .map(IpAddr::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    if ips.len() > SHOWN {
        out.push_str(&format!(" (+{} more)", ips.len() - SHOWN));
    }
    out
}

#[derive(Debug, Clone)]
struct ProbeResult {
    provider: String,
    host: String,
    dns: Result<Vec<IpAddr>, String>,
    outcome: ProbeOutcome,
    elapsed: Duration,
    bedrock: bool,
    local: bool,
    note: Option<&'static str>,
    /// See `ProbeTarget::proxy`.
    proxy: Option<String>,
}

impl ProbeResult {
    /// Our local DNS answer points into the block range. It only decides the
    /// outcome for a direct connection; see `settle_outcome`.
    fn dns_filtered(&self) -> bool {
        self.dns
            .as_ref()
            .is_ok_and(|ips| ips.iter().any(|ip| is_filtered_ip(*ip)))
    }

    /// A public (non-Bedrock, non-local) API that the network path blocks.
    fn looks_blocked(&self) -> bool {
        let filtered_direct = self.proxy.is_none() && self.dns_filtered();
        !self.bedrock && !self.local && (filtered_direct || self.outcome.is_network_failure())
    }
}

async fn resolve_host(host: &str, port: u16) -> Result<Vec<IpAddr>, String> {
    if let Ok(ip) = host.parse::<IpAddr>() {
        return Ok(vec![ip]);
    }
    match tokio::time::timeout(DNS_TIMEOUT, tokio::net::lookup_host((host, port))).await {
        Err(_) => Err(format!("lookup timed out after {}s", DNS_TIMEOUT.as_secs())),
        Ok(Err(e)) => Err(e.to_string()),
        Ok(Ok(addrs)) => {
            let mut ips: Vec<IpAddr> = addrs.map(|addr| addr.ip()).collect();
            ips.sort();
            ips.dedup();
            if ips.is_empty() {
                Err("no addresses returned".to_string())
            } else {
                Ok(ips)
            }
        }
    }
}

/// Combine the DNS result with the HTTP outcome. On a direct connection, a
/// failed DNS lookup is the root cause of any network failure after it, and
/// a filter-redirected lookup means whatever answered wasn't the provider —
/// even if it answered 2xx (the block page completes TLS when the filter's
/// root CA is trusted). Through a proxy, the proxy resolves the host itself,
/// so our lookup says nothing about where the request went and the HTTP
/// outcome stands.
fn settle_outcome(
    dns: &Result<Vec<IpAddr>, String>,
    local: bool,
    proxied: bool,
    outcome: ProbeOutcome,
) -> ProbeOutcome {
    if proxied {
        return outcome;
    }
    let filtered = dns
        .as_ref()
        .is_ok_and(|ips| ips.iter().any(|ip| is_filtered_ip(*ip)));
    match (dns, outcome) {
        (Err(detail), outcome) if outcome.is_network_failure() => ProbeOutcome::Dns {
            detail: detail.clone(),
        },
        (_, outcome) if filtered && !local => ProbeOutcome::Filtered {
            http: outcome.http_status(),
        },
        (_, outcome) => outcome,
    }
}

async fn run_probe(client: &reqwest::Client, target: ProbeTarget) -> ProbeResult {
    let started = Instant::now();
    let mut result = ProbeResult {
        provider: target.provider,
        host: String::new(),
        dns: Err("not attempted".to_string()),
        outcome: ProbeOutcome::Other {
            detail: String::new(),
        },
        elapsed: Duration::ZERO,
        bedrock: target.bedrock,
        local: false,
        note: target.note,
        proxy: target.proxy,
    };

    let url = match Url::parse(&target.request.url) {
        Ok(url) => url,
        Err(e) => {
            result.outcome = ProbeOutcome::InvalidUrl {
                detail: e.to_string(),
            };
            return result;
        }
    };
    let host = url
        .host_str()
        .unwrap_or_default()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .to_string();
    let port = url.port_or_known_default().unwrap_or(443);
    result.local = is_local_host(&host);
    result.dns = resolve_host(&host, port).await;
    result.host = host;

    // The HTTP request runs even when our DNS lookup failed: with a proxy
    // configured, the proxy resolves the host, not us.
    let mut request = client.get(url);
    if let Some(Secret(key)) = &target.secret {
        request = match target.request.auth {
            ProbeAuth::None => request,
            ProbeAuth::XApiKey => request
                .header("x-api-key", key)
                .header("anthropic-version", ANTHROPIC_VERSION),
            ProbeAuth::Bearer => request.bearer_auth(key),
            ProbeAuth::GoogApiKey => request.header("x-goog-api-key", key),
        };
    }
    let outcome = match request.send().await {
        Ok(response) => classify_response(response.status().as_u16(), target.request.mode),
        Err(err) => classify_reqwest_error(&err),
    };

    result.outcome = settle_outcome(&result.dns, result.local, result.proxy.is_some(), outcome);
    result.elapsed = started.elapsed();
    result
}

async fn probe_all(targets: Vec<ProbeTarget>) -> Result<Vec<ProbeResult>> {
    let client = reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(REQUEST_TIMEOUT)
        // Report the status the endpoint actually returns, not wherever a
        // redirect lands.
        .redirect(reqwest::redirect::Policy::none())
        // Proxies stay at reqwest's defaults, which `proxy_for` predicts.
        .user_agent(concat!("ai-cli/", env!("CARGO_PKG_VERSION"), " (doctor)"))
        .build()
        .context("building the HTTP client for connectivity probes")?;
    Ok(join_all(targets.into_iter().map(|t| run_probe(&client, t))).await)
}

fn probe_check(result: &ProbeResult, width: usize) -> Check {
    let mut check = Check::new(
        result.outcome.status(),
        format!(
            "{:<width$}  {}  ({} ms)",
            result.provider,
            result.outcome.describe(),
            result.elapsed.as_millis()
        ),
    );
    if let Ok(ips) = &result.dns {
        check = check.detail(format!("dns: {} → {}", result.host, format_ips(ips)));
    }
    if let Some(proxy) = &result.proxy {
        check = check.detail(format!("proxy: {proxy}"));
    }
    if result.dns_filtered() {
        check = check.detail(if result.proxy.is_some() {
            format!(
                "local DNS resolves {} into Cisco Umbrella's block range ({}); the request went through the proxy, which resolves the host itself",
                result.host, UMBRELLA_BLOCK_RANGE
            )
        } else {
            format!(
                "DNS appears redirected by a network filter (Cisco Umbrella): {} resolves into {}",
                result.host, UMBRELLA_BLOCK_RANGE
            )
        });
    }
    if let Some(detail) = result.outcome.detail() {
        check = check.detail(format!("error: {detail}"));
    }
    if let Some(hint) = result.outcome.hint(result.local) {
        check = check.detail(hint);
    }
    if let (Some(note), Status::Ok) = (result.note, result.outcome.status()) {
        check = check.detail(note);
    }
    check
}

fn connectivity_section(
    results: &[ProbeResult],
    not_probed: &[(String, String)],
    offline: bool,
) -> Section {
    let mut checks = Vec::new();
    if offline {
        checks.push(Check::new(Status::Info, "skipped (--offline)"));
    } else {
        let width = results.iter().map(|r| r.provider.len()).max().unwrap_or(0);
        checks.extend(results.iter().map(|r| probe_check(r, width)));
        if !not_probed.is_empty() {
            let list = not_probed
                .iter()
                .map(|(name, why)| format!("{name} ({why})"))
                .collect::<Vec<_>>()
                .join(", ");
            checks.push(Check::new(Status::Info, format!("not probed: {list}")));
        }
        if checks.is_empty() {
            checks.push(Check::new(Status::Info, "no enabled providers to probe"));
        }
    }
    Section {
        title: "Connectivity",
        checks,
    }
}

// =============================================================================
// Web search and telemetry (presence only)
// =============================================================================

/// Which of `vars` web search takes its key from.
#[derive(Debug)]
enum SearchKey<'a> {
    Set(&'a str),
    /// The chosen var is whitespace-only. `shadowed` lists the vars after it
    /// that have a real value but are never reached.
    Blank {
        var: &'a str,
        shadowed: Vec<&'a str>,
    },
    Unset,
}

/// Pick the search key var the way the runtime does
/// (`GeminiSearchClient::from_env`, `TavilyClient::from_env`): the first var
/// with a non-empty value wins. Unlike provider keys, a whitespace-only value
/// counts as set there: it's sent as the key and rejected, and it shadows
/// the vars after it, so it can't just be skipped here.
fn search_key<'a>(env: &dyn Environment, vars: &[&'a str]) -> SearchKey<'a> {
    let chosen = vars.iter().enumerate().find_map(|(i, var)| {
        env.var(var)
            .filter(|value| !value.is_empty())
            .map(|value| (i, *var, value.trim().is_empty()))
    });
    match chosen {
        None => SearchKey::Unset,
        Some((_, var, false)) => SearchKey::Set(var),
        Some((i, var, true)) => SearchKey::Blank {
            var,
            shadowed: vars[i + 1..]
                .iter()
                .copied()
                .filter(|later| non_blank(env.var(later)).is_some())
                .collect(),
        },
    }
}

fn search_section(search: &SearchConfig, env: &dyn Environment) -> Section {
    let primary = search.provider.to_lowercase();
    let backends: [(&str, Vec<&str>); 2] = [
        (
            "gemini",
            vec![
                search.gemini.api_key_env.as_str(),
                search.gemini.fallback_api_key_env.as_str(),
            ],
        ),
        ("tavily", vec![search.tavily.api_key_env.as_str()]),
    ];

    let mut checks = Vec::new();
    if !backends.iter().any(|(name, _)| *name == primary) {
        checks.push(Check::new(
            Status::Warn,
            format!(
                "unknown search provider \"{}\" in [search] (expected gemini or tavily); `ai -w` will fail",
                search.provider
            ),
        ));
    }
    for (name, vars) in &backends {
        let is_primary = *name == primary;
        let role = if is_primary { "primary" } else { "fallback" };
        checks.push(match search_key(env, vars) {
            SearchKey::Set(var) => Check::new(Status::Ok, format!("{name} ({role}): {var} set")),
            SearchKey::Blank { var, shadowed } => {
                let check = Check::new(
                    Status::Warn,
                    format!(
                        "{name} ({role}): {var} is set but blank; searches send it as the key and are rejected"
                    ),
                );
                if shadowed.is_empty() {
                    check
                } else {
                    check.detail(format!(
                        "{} is ignored while {var} is set; unset {var} to use it",
                        shadowed.join(", ")
                    ))
                }
            }
            SearchKey::Unset if is_primary => Check::new(
                Status::Warn,
                format!(
                    "{name} ({role}): {} not set; `ai -w` needs it",
                    vars.join(" / ")
                ),
            ),
            SearchKey::Unset => Check::new(
                Status::Info,
                format!("{name} ({role}): {} not set; no fallback", vars.join(" / ")),
            ),
        });
    }
    Section {
        title: "Web search",
        checks,
    }
}

fn telemetry_section(env: &dyn Environment) -> Section {
    let disabled = env
        .var("LANGFUSE_ENABLED")
        .is_some_and(|v| v.eq_ignore_ascii_case("false") || v == "0");
    let (set, missing): (Vec<&str>, Vec<&str>) = LANGFUSE_VARS
        .iter()
        .copied()
        .partition(|var| non_blank(env.var(var)).is_some());

    let check = if disabled {
        Check::new(
            Status::Info,
            "Langfuse telemetry disabled (LANGFUSE_ENABLED=false)",
        )
    } else if missing.is_empty() {
        Check::new(
            Status::Ok,
            format!("Langfuse telemetry on ({} set)", set.join(", ")),
        )
    } else if set.is_empty() {
        Check::new(
            Status::Info,
            "Langfuse telemetry off (LANGFUSE_* not set; optional)",
        )
    } else {
        Check::new(
            Status::Warn,
            format!(
                "Langfuse telemetry off: {} not set ({} set)",
                missing.join(", "),
                set.join(", ")
            ),
        )
    };
    Section {
        title: "Telemetry",
        checks: vec![check],
    }
}

// =============================================================================
// Summary
// =============================================================================

#[derive(Debug, Default, PartialEq, Eq)]
struct Summary {
    probed: usize,
    ok: usize,
    warned: usize,
    failed: usize,
    /// A public API failed in a way a network filter causes.
    public_blocked: bool,
    /// Bedrock-hosted providers whose probe passed.
    bedrock_ok: Vec<String>,
}

fn summarize(results: &[ProbeResult]) -> Summary {
    let count = |status: Status| {
        results
            .iter()
            .filter(|r| r.outcome.status() == status)
            .count()
    };
    Summary {
        probed: results.len(),
        ok: count(Status::Ok),
        warned: count(Status::Warn),
        failed: count(Status::Fail),
        public_blocked: results.iter().any(ProbeResult::looks_blocked),
        bedrock_ok: results
            .iter()
            .filter(|r| r.bedrock && r.outcome.status() == Status::Ok)
            .map(|r| r.provider.clone())
            .collect(),
    }
}

fn should_fail(state: &ConfigState, summary: &Summary) -> bool {
    matches!(state, ConfigState::Invalid(_)) || summary.failed > 0
}

fn summary_section(state: &ConfigState, offline: bool, s: &Summary) -> Section {
    let mut checks = Vec::new();
    if matches!(state, ConfigState::Invalid(_)) {
        checks.push(Check::new(Status::Fail, "config file failed to load"));
    }
    checks.push(if offline {
        Check::new(Status::Info, "connectivity not checked (--offline)")
    } else if s.probed == 0 {
        Check::new(
            Status::Warn,
            "no providers probed: none are enabled with usable credentials",
        )
    } else if s.failed > 0 {
        Check::new(
            Status::Fail,
            format!("{} of {} provider probes failed", s.failed, s.probed),
        )
    } else if s.warned > 0 {
        Check::new(
            Status::Warn,
            format!(
                "{} of {} provider probes passed, {} with warnings",
                s.ok, s.probed, s.warned
            ),
        )
    } else {
        Check::new(
            Status::Ok,
            format!("all {} provider probes passed", s.probed),
        )
    });

    if s.public_blocked && !s.bedrock_ok.is_empty() {
        let use_it = s
            .bedrock_ok
            .iter()
            .map(|name| format!("`ai -p {name}`"))
            .collect::<Vec<_>>()
            .join(" or ");
        checks.push(
            Check::new(
                Status::Info,
                "hint: public APIs look blocked on this network; Bedrock is reachable — consider using the Bedrock provider here",
            )
            .detail(format!("e.g. {use_it}")),
        );
    } else if s.public_blocked {
        checks.push(Check::new(
            Status::Info,
            "hint: public AI APIs look blocked or unreachable from this network (see Connectivity above)",
        ));
    }
    Section {
        title: "Summary",
        checks,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::net::Ipv4Addr;

    #[test]
    fn filtered_dns_fails_the_probe_even_on_http_2xx() {
        // With the filter's root CA trusted, the Umbrella block page can
        // complete TLS and answer 200 — that must not count as OK.
        let umbrella = Ok(vec![IpAddr::V4(Ipv4Addr::new(146, 112, 61, 104))]);
        let outcome = settle_outcome(
            &umbrella,
            false,
            false,
            ProbeOutcome::Success { status: 200 },
        );
        assert!(matches!(
            outcome,
            ProbeOutcome::Filtered { http: Some(200) }
        ));
        assert_eq!(outcome.status(), Status::Fail);
        assert!(outcome.is_network_failure());

        let normal = Ok(vec![IpAddr::V4(Ipv4Addr::new(104, 18, 2, 115))]);
        let outcome = settle_outcome(&normal, false, false, ProbeOutcome::Success { status: 200 });
        assert_eq!(outcome.status(), Status::Ok);

        let failed_dns: Result<Vec<IpAddr>, String> = Err("nxdomain".into());
        let outcome = settle_outcome(&failed_dns, false, false, ProbeOutcome::Timeout);
        assert!(matches!(outcome, ProbeOutcome::Dns { .. }));
    }

    #[test]
    fn proxied_probe_keeps_its_http_outcome_despite_filtered_dns() {
        // The proxy resolves the host itself, so a filtered local answer
        // says nothing about where the request went.
        let umbrella = Ok(vec![IpAddr::V4(Ipv4Addr::new(146, 112, 61, 104))]);
        let outcome = settle_outcome(
            &umbrella,
            false,
            true,
            ProbeOutcome::Success { status: 200 },
        );
        assert_eq!(outcome, ProbeOutcome::Success { status: 200 });
        // Nor is a failed local lookup the cause of a proxied failure.
        let failed_dns: Result<Vec<IpAddr>, String> = Err("nxdomain".into());
        let outcome = settle_outcome(&failed_dns, false, true, ProbeOutcome::Timeout);
        assert_eq!(outcome, ProbeOutcome::Timeout);

        // The report passes, and still shows the filtered local answer.
        let result = ProbeResult {
            proxy: Some("http://<redacted>@proxy.example:8080".into()),
            ..probe_result(
                "openai",
                ProbeOutcome::Success { status: 200 },
                &["146.112.61.104"],
                false,
            )
        };
        let summary = summarize(std::slice::from_ref(&result));
        assert_eq!(summary.ok, 1);
        assert!(!summary.public_blocked);
        assert!(!should_fail(&ConfigState::Loaded, &summary));
        let check = probe_check(&result, 6);
        assert_eq!(check.status, Status::Ok);
        assert!(
            check
                .details
                .contains(&"proxy: http://<redacted>@proxy.example:8080".to_string()),
            "{:?}",
            check.details
        );
        assert!(
            check.details.iter().any(|d| d.contains(
                "local DNS resolves openai.example into Cisco Umbrella's block range (146.112.0.0/16); the request went through the proxy"
            )),
            "{:?}",
            check.details
        );
        assert!(!check.details.iter().any(|d| d.starts_with("DNS appears")));
    }

    fn route(url: &str, vars: &[(&str, &str)]) -> Option<String> {
        proxy_for(&Url::parse(url).expect("valid URL"), &FakeEnv::new(vars))
    }

    #[test]
    fn proxy_choice_follows_reqwest_env_rules() {
        let https = "https://api.openai.com/v1/models";
        let http = "http://localhost:11434/api/tags";
        assert_eq!(route(https, &[]), None);

        // Each scheme has its own var, upper or lower case.
        let via = |url: &str, vars: &[(&str, &str)]| route(url, vars).unwrap_or_default();
        assert_eq!(
            via(https, &[("HTTPS_PROXY", "http://p:3128")]),
            "http://p:3128"
        );
        assert_eq!(
            via(https, &[("https_proxy", "http://p:3128")]),
            "http://p:3128"
        );
        assert_eq!(route(http, &[("HTTPS_PROXY", "http://p:3128")]), None);
        assert_eq!(via(http, &[("http_proxy", "p:3128")]), "p:3128");
        // ALL_PROXY covers both when the scheme's own var doesn't.
        assert_eq!(
            via(https, &[("ALL_PROXY", "socks5h://p:1080")]),
            "socks5h://p:1080"
        );
        assert_eq!(
            via(http, &[("all_proxy", "socks5h://p:1080")]),
            "socks5h://p:1080"
        );
        // A set-but-blank HTTPS_PROXY hides https_proxy, not ALL_PROXY.
        let blank = ("HTTPS_PROXY", "");
        assert_eq!(
            route(https, &[blank, ("https_proxy", "http://p:3128")]),
            None
        );
        assert_eq!(
            via(https, &[blank, ("ALL_PROXY", "http://q:3128")]),
            "http://q:3128"
        );
        // An unusable value is ignored, and CGI turns proxies off.
        assert_eq!(route(https, &[("HTTPS_PROXY", "ftp://p:21")]), None);
        assert_eq!(
            route(
                https,
                &[("HTTPS_PROXY", "http://p:3128"), ("REQUEST_METHOD", "GET")]
            ),
            None
        );
        // Credentials never reach the report.
        assert_eq!(
            via(https, &[("HTTPS_PROXY", "http://user:hunter2@p:3128")]),
            "http://<redacted>@p:3128"
        );
    }

    #[test]
    fn no_proxy_follows_reqwest_rules() {
        let proxied = |url: &str, no_proxy: &str| {
            route(
                url,
                &[("ALL_PROXY", "http://p:3128"), ("NO_PROXY", no_proxy)],
            )
            .is_some()
        };
        // A domain entry covers it and its subdomains, leading dot or not.
        assert!(!proxied("https://openai.com/v1", "openai.com"));
        assert!(!proxied("https://api.openai.com/v1", "openai.com"));
        assert!(!proxied("https://api.openai.com/v1", ".openai.com"));
        assert!(!proxied("https://openai.com/v1", ".openai.com"));
        assert!(proxied("https://notopenai.com/v1", "openai.com"));
        assert!(proxied("https://openai.com/v1", "api.openai.com"));
        // Comma-separated with optional spaces; `*` covers every domain.
        assert!(!proxied("https://api.x.ai/v1", "localhost , api.x.ai"));
        assert!(!proxied("https://api.x.ai/v1", "*"));
        // No implicit exemption for localhost.
        assert!(proxied("http://localhost:11434", ""));
        // IP hosts match IP and CIDR entries only (not `*`).
        assert!(!proxied("http://10.1.2.3:8000/v1", "10.0.0.0/8"));
        assert!(proxied("http://11.1.2.3:8000/v1", "10.0.0.0/8"));
        assert!(!proxied("http://127.0.0.1:11434", "127.0.0.1"));
        assert!(!proxied("http://[::1]:11434", "::1"));
        assert!(!proxied("http://[fd00::5]:8000", "fd00::/8"));
        assert!(proxied("http://127.0.0.1:11434", "*"));
        // Lowercase no_proxy counts when NO_PROXY is unset.
        let vars = [("HTTPS_PROXY", "http://p:3128"), ("no_proxy", "x.ai")];
        assert_eq!(route("https://api.x.ai/v1", &vars), None);
    }

    #[test]
    fn probe_targets_record_the_proxy_they_use() {
        let env = FakeEnv::new(&[
            ("ANTHROPIC_API_KEY", "sk-ant-test"),
            ("OPENAI_API_KEY", "sk-test"),
            ("HTTPS_PROXY", "http://user:hunter2@proxy.example:8080"),
            ("NO_PROXY", "localhost,.openai.com"),
        ]);
        let entries = provider_entries(&Config::default(), &env);
        let target = |name: &str| {
            let entry = entries.iter().find(|e| e.name == name).expect(name);
            probe_target(entry, &env).expect("probeable")
        };
        assert_eq!(
            target("anthropic").proxy.as_deref(),
            Some("http://<redacted>@proxy.example:8080")
        );
        assert_eq!(target("openai").proxy, None);
    }

    struct FakeEnv {
        vars: HashMap<String, String>,
        subscription: bool,
    }

    impl FakeEnv {
        fn new(pairs: &[(&str, &str)]) -> Self {
            FakeEnv {
                vars: pairs
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
                subscription: false,
            }
        }
    }

    impl Environment for FakeEnv {
        fn var(&self, name: &str) -> Option<String> {
            self.vars.get(name).cloned()
        }

        fn subscription_token_available(&self) -> bool {
            self.subscription
        }
    }

    /// A provider config built from the shipped defaults, so these tests
    /// don't break when `ProviderConfig` gains a field.
    fn provider(api_key_env: Option<&str>, api_key: Option<&str>) -> ProviderConfig {
        let mut cfg = Config::default()
            .providers
            .remove("openai")
            .expect("default config has openai");
        cfg.api_key_env = api_key_env.map(str::to_string);
        cfg.api_key = api_key.map(str::to_string);
        cfg
    }

    fn ip(s: &str) -> IpAddr {
        s.parse().expect("valid IP literal")
    }

    fn probe_result(
        provider: &str,
        outcome: ProbeOutcome,
        ips: &[&str],
        bedrock: bool,
    ) -> ProbeResult {
        ProbeResult {
            provider: provider.to_string(),
            host: format!("{provider}.example"),
            dns: Ok(ips.iter().copied().map(ip).collect()),
            outcome,
            elapsed: Duration::from_millis(42),
            bedrock,
            local: false,
            note: None,
            proxy: None,
        }
    }

    // --- filtered IPs ------------------------------------------------------

    #[test]
    fn umbrella_block_range_is_flagged() {
        assert!(is_filtered_ip(ip("146.112.61.106")));
        assert!(is_filtered_ip(ip("146.112.0.0")));
        assert!(is_filtered_ip(ip("146.112.255.255")));
        // IPv4-mapped IPv6 form of a block-page address.
        assert!(is_filtered_ip(ip("::ffff:146.112.61.104")));
    }

    #[test]
    fn addresses_outside_the_block_range_are_not_flagged() {
        assert!(!is_filtered_ip(ip("146.111.255.255")));
        assert!(!is_filtered_ip(ip("146.113.0.1")));
        assert!(!is_filtered_ip(ip("160.79.104.10")));
        assert!(!is_filtered_ip(IpAddr::V4(Ipv4Addr::LOCALHOST)));
        assert!(!is_filtered_ip(ip("::1")));
        assert!(!is_filtered_ip(ip("2606:4700::6810:1")));
    }

    // --- HTTP status classification -------------------------------------------

    #[test]
    fn http_statuses_map_to_outcomes() {
        assert_eq!(
            classify_http_status(200),
            ProbeOutcome::Success { status: 200 }
        );
        assert_eq!(
            classify_http_status(204),
            ProbeOutcome::Success { status: 204 }
        );
        assert_eq!(
            classify_http_status(401),
            ProbeOutcome::AuthFailed { status: 401 }
        );
        assert_eq!(
            classify_http_status(403),
            ProbeOutcome::AuthFailed { status: 403 }
        );
        assert_eq!(classify_http_status(404), ProbeOutcome::NotFound);
        assert_eq!(classify_http_status(429), ProbeOutcome::RateLimited);
        assert_eq!(
            classify_http_status(503),
            ProbeOutcome::ServerError { status: 503 }
        );
        assert_eq!(
            classify_http_status(400),
            ProbeOutcome::UnexpectedStatus { status: 400 }
        );
        assert_eq!(
            classify_http_status(302),
            ProbeOutcome::UnexpectedStatus { status: 302 }
        );
    }

    #[test]
    fn reachability_probes_accept_any_status() {
        for status in [200, 403, 404, 500] {
            assert_eq!(
                classify_response(status, ProbeMode::Reachability),
                ProbeOutcome::Reachable { status }
            );
            assert_eq!(ProbeOutcome::Reachable { status }.status(), Status::Ok);
        }
    }

    #[test]
    fn outcome_severity() {
        assert_eq!(ProbeOutcome::Success { status: 200 }.status(), Status::Ok);
        assert_eq!(ProbeOutcome::RateLimited.status(), Status::Warn);
        assert_eq!(
            ProbeOutcome::ServerError { status: 502 }.status(),
            Status::Warn
        );
        assert_eq!(
            ProbeOutcome::AuthFailed { status: 401 }.status(),
            Status::Fail
        );
        assert_eq!(ProbeOutcome::NotFound.status(), Status::Fail);
        assert_eq!(ProbeOutcome::Timeout.status(), Status::Fail);
        assert!(ProbeOutcome::Timeout.is_network_failure());
        assert!(!ProbeOutcome::AuthFailed { status: 401 }.is_network_failure());
    }

    // --- transport error classification -------------------------------------

    fn facts(chain: &str) -> TransportFacts {
        TransportFacts {
            connect: true,
            chain: chain.to_string(),
            root: chain.rsplit(": ").next().unwrap_or_default().to_string(),
            ..TransportFacts::default()
        }
    }

    #[test]
    fn dns_failures_are_recognised() {
        let f = facts("error sending request for url (https://api.x.ai/v1/models): client error (Connect): dns error: failed to lookup address information: Name or service not known");
        assert!(matches!(classify_transport(&f), ProbeOutcome::Dns { .. }));
        let mac = facts("client error (Connect): dns error: failed to lookup address information: nodename nor servname provided, or not known");
        assert!(matches!(classify_transport(&mac), ProbeOutcome::Dns { .. }));
    }

    #[test]
    fn tls_failures_are_recognised() {
        // OpenSSL (native-tls on Linux), as seen behind Cisco Umbrella.
        let openssl = facts("client error (Connect): error:0A000086:SSL routines:tls_post_process_server_certificate:certificate verify failed:../ssl/statem/statem_clnt.c:1889: (self-signed certificate in certificate chain)");
        let outcome = classify_transport(&openssl);
        assert!(matches!(outcome, ProbeOutcome::Tls { .. }), "{outcome:?}");
        assert!(outcome.hint(false).is_some_and(|h| h.contains("proxy")));
        // rustls.
        let rustls = facts("client error (Connect): invalid peer certificate: UnknownIssuer");
        assert!(matches!(
            classify_transport(&rustls),
            ProbeOutcome::Tls { .. }
        ));
        // macOS Security.framework.
        let macos = facts("client error (Connect): The certificate was not trusted.");
        assert!(matches!(
            classify_transport(&macos),
            ProbeOutcome::Tls { .. }
        ));
    }

    #[test]
    fn refused_timeout_and_other_connect_failures() {
        let refused = TransportFacts {
            connect: true,
            io_kind: Some(std::io::ErrorKind::ConnectionRefused),
            chain: "client error (Connect): tcp connect error: Connection refused (os error 111)"
                .into(),
            ..TransportFacts::default()
        };
        assert_eq!(classify_transport(&refused), ProbeOutcome::ConnectRefused);
        assert!(ProbeOutcome::ConnectRefused
            .hint(true)
            .is_some_and(|h| h.contains("ollama serve")));

        let timeout = TransportFacts {
            timeout: true,
            chain: "operation timed out".into(),
            ..TransportFacts::default()
        };
        assert_eq!(classify_transport(&timeout), ProbeOutcome::Timeout);
        assert!(ProbeOutcome::Timeout
            .hint(true)
            .is_some_and(|h| h.contains("ollama serve")));
        assert!(ProbeOutcome::Timeout
            .hint(false)
            .is_some_and(|h| h.contains("firewall")));

        let reset = facts("client error (Connect): Connection reset by peer (os error 104)");
        assert!(matches!(
            classify_transport(&reset),
            ProbeOutcome::ConnectFailed { .. }
        ));
    }

    #[test]
    fn real_reqwest_builder_error_classifies_without_network() {
        let err = reqwest::Client::new()
            .get("http://")
            .build()
            .expect_err("an empty host is a builder error");
        let facts = transport_facts(&err);
        assert!(facts.builder);
        assert!(matches!(
            classify_reqwest_error(&err),
            ProbeOutcome::Other { .. }
        ));
    }

    // --- auth source ------------------------------------------------------------

    #[test]
    fn env_key_reports_name_and_length_only() {
        let env = FakeEnv::new(&[("OPENAI_API_KEY", "sk-test-abcdef")]);
        let (auth, secret) = resolve_auth("openai", &provider(Some("OPENAI_API_KEY"), None), &env);
        assert_eq!(
            auth,
            AuthSource::EnvVar {
                var: "OPENAI_API_KEY".into(),
                len: 14
            }
        );
        assert!(auth.usable());
        assert_eq!(auth.describe(), "env OPENAI_API_KEY (set, 14 chars)");
        assert!(secret.is_some());
    }

    #[test]
    fn whitespace_only_key_counts_as_missing() {
        let env = FakeEnv::new(&[("OPENAI_API_KEY", "   ")]);
        let (auth, secret) = resolve_auth("openai", &provider(Some("OPENAI_API_KEY"), None), &env);
        assert_eq!(
            auth,
            AuthSource::EnvBlank {
                var: "OPENAI_API_KEY".into()
            }
        );
        assert!(!auth.usable());
        assert!(secret.is_none());

        let (auth, _) = resolve_auth("openai", &provider(None, Some(" \t")), &env);
        assert_eq!(auth, AuthSource::ConfigKeyBlank);
        assert!(!auth.usable());
    }

    #[test]
    fn unset_env_key_is_reported() {
        let env = FakeEnv::new(&[]);
        let (auth, _) = resolve_auth("openai", &provider(Some("OPENAI_API_KEY"), None), &env);
        assert_eq!(
            auth,
            AuthSource::EnvUnset {
                vars: vec!["OPENAI_API_KEY".into()]
            }
        );
        assert_eq!(auth.describe(), "env OPENAI_API_KEY not set");
    }

    #[test]
    fn gemini_falls_back_to_google_api_key() {
        let env = FakeEnv::new(&[("GOOGLE_API_KEY", "AIza-test")]);
        let (auth, _) = resolve_auth("google", &provider(Some("GEMINI_API_KEY"), None), &env);
        assert_eq!(
            auth,
            AuthSource::EnvVar {
                var: "GOOGLE_API_KEY".into(),
                len: 9
            }
        );
        let none = FakeEnv::new(&[]);
        let (auth, _) = resolve_auth("google", &provider(Some("GEMINI_API_KEY"), None), &none);
        assert_eq!(
            auth.describe(),
            "env GEMINI_API_KEY / GOOGLE_API_KEY not set"
        );
    }

    #[test]
    fn config_api_key_takes_precedence_over_env() {
        let env = FakeEnv::new(&[("OPENAI_API_KEY", "from-env")]);
        let (auth, _) = resolve_auth(
            "openai",
            &provider(Some("OPENAI_API_KEY"), Some("from-config")),
            &env,
        );
        assert_eq!(auth, AuthSource::ConfigKey { len: 11 });
    }

    #[test]
    fn unreferenced_conventional_env_var_is_pointed_out() {
        let env = FakeEnv::new(&[("OPENAI_API_KEY", "sk-test")]);
        let (auth, _) = resolve_auth("openai", &provider(None, None), &env);
        assert_eq!(
            auth,
            AuthSource::NotConfigured {
                unreferenced: Some("OPENAI_API_KEY".into())
            }
        );
        assert!(!auth.usable());
        assert!(auth.describe().contains("api_key_env = \"OPENAI_API_KEY\""));
    }

    #[test]
    fn ollama_and_subscription_modes() {
        let mut env = FakeEnv::new(&[]);
        let (auth, _) = resolve_auth("ollama", &provider(None, None), &env);
        assert_eq!(auth, AuthSource::NotRequired);
        assert!(auth.usable());

        let mut sub = provider(None, None);
        sub.auth_mode = AuthMode::Subscription;
        let (auth, _) = resolve_auth("anthropic", &sub, &env);
        assert_eq!(auth, AuthSource::Subscription { available: false });
        assert!(!auth.usable());
        env.subscription = true;
        let (auth, secret) = resolve_auth("anthropic", &sub, &env);
        assert_eq!(auth, AuthSource::Subscription { available: true });
        assert!(auth.usable());
        assert!(
            secret.is_none(),
            "the OAuth token is never handed to a probe"
        );

        let (auth, _) = resolve_auth("openai", &sub, &env);
        assert_eq!(auth, AuthSource::SubscriptionUnsupported);
    }

    #[test]
    fn bedrock_reads_token_from_env_only() {
        let env = FakeEnv::new(&[("AWS_BEARER_TOKEN_BEDROCK", "bedrock-token")]);
        let mut cfg = provider(Some("AWS_BEARER_TOKEN_BEDROCK"), Some("ignored-config-key"));
        cfg.base_url = Some("https://bedrock-runtime.us-east-1.amazonaws.com/v1/".into());
        let (auth, _) = resolve_auth("anthropic", &cfg, &env);
        assert_eq!(
            auth,
            AuthSource::EnvVar {
                var: "AWS_BEARER_TOKEN_BEDROCK".into(),
                len: 13
            }
        );
    }

    #[test]
    fn secret_debug_is_redacted() {
        let secret = Secret("sk-live-do-not-print".into());
        assert_eq!(format!("{secret:?}"), "Secret(<redacted>)");
    }

    // --- probe planning -----------------------------------------------------------

    #[test]
    fn probe_requests_per_provider_kind() {
        let key = AuthSource::EnvVar {
            var: "K".into(),
            len: 3,
        };
        let anthropic = probe_request(
            ProviderKind::Anthropic,
            "https://api.anthropic.com/v1/",
            &key,
        )
        .unwrap();
        assert_eq!(anthropic.url, "https://api.anthropic.com/v1/models");
        assert_eq!(anthropic.auth, ProbeAuth::XApiKey);
        assert_eq!(anthropic.mode, ProbeMode::ExpectSuccess);

        let xai =
            probe_request(ProviderKind::OpenAiCompatible, "https://api.x.ai/v1/", &key).unwrap();
        assert_eq!(xai.url, "https://api.x.ai/v1/models");
        assert_eq!(xai.auth, ProbeAuth::Bearer);

        let gemini = probe_request(
            ProviderKind::Gemini,
            "https://generativelanguage.googleapis.com/v1beta/",
            &key,
        )
        .unwrap();
        assert_eq!(
            gemini.url,
            "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1"
        );
        assert_eq!(gemini.auth, ProbeAuth::GoogApiKey);
        assert!(!gemini.url.contains("key="), "Gemini key goes in a header");

        let openrouter = probe_request(
            ProviderKind::OpenRouter,
            "https://openrouter.ai/api/v1",
            &key,
        )
        .unwrap();
        assert_eq!(openrouter.url, "https://openrouter.ai/api/v1/key");
        assert_eq!(openrouter.auth, ProbeAuth::Bearer);

        assert!(probe_request(ProviderKind::Unknown, "https://x.example", &key).is_none());
    }

    #[test]
    fn ollama_probe_uses_native_tags_endpoint() {
        let auth = AuthSource::NotRequired;
        for base in [
            "http://localhost:11434",
            "http://localhost:11434/",
            "http://localhost:11434/v1/",
        ] {
            let req = probe_request(ProviderKind::Ollama, base, &auth).unwrap();
            assert_eq!(req.url, "http://localhost:11434/api/tags", "{base}");
            assert_eq!(req.auth, ProbeAuth::None);
        }
    }

    #[test]
    fn bedrock_and_subscription_probes_are_reachability_only() {
        let key = AuthSource::EnvVar {
            var: "AWS_BEARER_TOKEN_BEDROCK".into(),
            len: 3,
        };
        let bedrock = probe_request(
            ProviderKind::BedrockConverse,
            "https://bedrock-runtime.us-east-1.amazonaws.com/v1/",
            &key,
        )
        .unwrap();
        assert_eq!(
            bedrock.url,
            "https://bedrock-runtime.us-east-1.amazonaws.com/"
        );
        assert_eq!(bedrock.auth, ProbeAuth::None);
        assert_eq!(bedrock.mode, ProbeMode::Reachability);

        let sub = probe_request(
            ProviderKind::Anthropic,
            "https://api.anthropic.com/v1/",
            &AuthSource::Subscription { available: true },
        )
        .unwrap();
        assert_eq!(sub.auth, ProbeAuth::None);
        assert_eq!(sub.mode, ProbeMode::Reachability);
    }

    #[test]
    fn provider_kinds_and_bedrock_detection() {
        let mut cfg = provider(None, None);
        assert_eq!(provider_kind("anthropic", &cfg), ProviderKind::Anthropic);
        assert_eq!(provider_kind("xai", &cfg), ProviderKind::OpenAiCompatible);
        assert_eq!(provider_kind("custom", &cfg), ProviderKind::Unknown);
        cfg.base_url = Some("https://bedrock-mantle.us-east-1.api.aws/openai/v1".into());
        assert_eq!(
            provider_kind("custom", &cfg),
            ProviderKind::OpenAiCompatible
        );
        cfg.base_url = Some("https://bedrock-runtime.us-east-1.amazonaws.com/v1/".into());
        assert_eq!(
            provider_kind("anthropic", &cfg),
            ProviderKind::BedrockConverse
        );

        assert!(is_bedrock_endpoint(
            "https://bedrock-runtime.us-west-2.amazonaws.com"
        ));
        assert!(is_bedrock_endpoint(
            "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
        ));
        assert!(!is_bedrock_endpoint("https://api.openai.com/v1/"));
        assert!(!is_bedrock_endpoint("https://bedrock.example.com/v1"));
    }

    #[test]
    fn endpoint_prefers_base_url_over_default() {
        let mut cfg = provider(None, None);
        assert_eq!(
            endpoint_for("openai", &cfg),
            Some(("https://api.openai.com/v1/".to_string(), false))
        );
        cfg.base_url = Some("https://proxy.example/v1".into());
        assert_eq!(
            endpoint_for("openai", &cfg),
            Some(("https://proxy.example/v1".to_string(), true))
        );
        assert_eq!(endpoint_for("custom", &provider(None, None)), None);
    }

    #[test]
    fn providers_without_credentials_are_not_probed() {
        let env = FakeEnv::new(&[]);
        let mut config = Config::default();
        config.providers.retain(|name, _| name == "openai");
        let entries = provider_entries(&config, &env);
        assert_eq!(
            probe_target(&entries[0], &env).unwrap_err(),
            "no usable credentials"
        );
        assert_eq!(provider_check(&entries[0]).status, Status::Warn);
    }

    // --- summary --------------------------------------------------------------------

    #[test]
    fn umbrella_scenario_suggests_bedrock() {
        // Work laptop: public APIs resolve into Umbrella's block range and fail
        // TLS; the Bedrock provider is reachable.
        let results = vec![
            probe_result(
                "anthropic",
                ProbeOutcome::Tls {
                    detail: "certificate verify failed".into(),
                },
                &["146.112.61.106"],
                false,
            ),
            probe_result(
                "claude-bedrock",
                ProbeOutcome::Reachable { status: 404 },
                &["52.94.1.1"],
                true,
            ),
        ];
        let summary = summarize(&results);
        assert_eq!(summary.failed, 1);
        assert!(summary.public_blocked);
        assert_eq!(summary.bedrock_ok, vec!["claude-bedrock".to_string()]);
        assert!(should_fail(&ConfigState::Loaded, &summary));

        let out = render(&[
            connectivity_section(&results, &[], false),
            summary_section(&ConfigState::Loaded, false, &summary),
        ]);
        assert!(out.contains("DNS appears redirected by a network filter (Cisco Umbrella)"));
        assert!(out.contains(
            "public APIs look blocked on this network; Bedrock is reachable — consider using the Bedrock provider here"
        ));
        assert!(out.contains("`ai -p claude-bedrock`"));
    }

    #[test]
    fn all_ok_passes_and_offline_skips() {
        let results = vec![probe_result(
            "openai",
            ProbeOutcome::Success { status: 200 },
            &["162.159.140.245"],
            false,
        )];
        let summary = summarize(&results);
        assert!(!should_fail(&ConfigState::Loaded, &summary));
        assert!(!summary.public_blocked);
        let out = render(&[summary_section(&ConfigState::Loaded, false, &summary)]);
        assert!(out.contains("all 1 provider probes passed"));
        assert!(!out.contains("hint:"));

        let offline = summarize(&[]);
        assert!(!should_fail(&ConfigState::Loaded, &offline));
        let out = render(&[
            connectivity_section(&[], &[], true),
            summary_section(&ConfigState::Loaded, true, &offline),
        ]);
        assert!(out.contains("skipped (--offline)"));
    }

    #[test]
    fn rate_limit_warns_but_does_not_fail() {
        let results = vec![probe_result(
            "xai",
            ProbeOutcome::RateLimited,
            &["1.2.3.4"],
            false,
        )];
        let summary = summarize(&results);
        assert_eq!(summary.warned, 1);
        assert!(!should_fail(&ConfigState::Loaded, &summary));
    }

    #[test]
    fn invalid_config_fails_and_shows_the_error() {
        let state = ConfigState::Invalid("Failed to parse config file: expected `=`".into());
        assert!(should_fail(&state, &Summary::default()));
        let loaded = LoadedConfig {
            path: Ok(PathBuf::from("/home/u/.config/ai-cli/config.toml")),
            state,
            config: Config::default(),
            stray: None,
        };
        let check = config_check(&loaded);
        assert_eq!(check.status, Status::Fail);
        assert!(check.details.iter().any(|d| d.contains("expected `=`")));
    }

    #[test]
    fn stray_config_at_xdg_path_is_pointed_out() {
        let loaded = LoadedConfig {
            path: Ok(PathBuf::from(
                "/Users/u/Library/Application Support/ai-cli/config.toml",
            )),
            state: ConfigState::Missing,
            config: Config::default(),
            stray: Some(PathBuf::from("/Users/u/.config/ai-cli/config.toml")),
        };
        let check = config_check(&loaded);
        assert_eq!(check.status, Status::Warn);
        assert!(check.details[0].contains("/Users/u/.config/ai-cli/config.toml"));
    }

    // --- redaction ------------------------------------------------------------------

    #[test]
    fn config_errors_are_scrubbed_of_keys() {
        let toml_err = "TOML parse error at line 3, column 11\n  |\n3 | api_key = \"sk-ant-api03-AbCdEf123456\n  |           ^\ninvalid basic string";
        let out = redact_secrets(toml_err);
        assert!(!out.contains("AbCdEf123456"), "{out}");
        assert!(out.contains("3 | api_key = <redacted>"));
        assert!(out.contains("TOML parse error at line 3, column 11"));

        // A key pasted on a line of its own.
        let pasted = "5 | sk-proj-Q1w2E3r4T5y6U7i8O9p0\n  | ^\nexpected `=`";
        let out = redact_secrets(pasted);
        assert!(!out.contains("Q1w2E3r4T5y6U7i8O9p0"), "{out}");
        assert!(out.contains("expected `=`"));

        // A base_url with credentials on the quoted line.
        let url_err = "TOML parse error at line 4, column 51\n  |\n4 | base_url = \"https://user:hunter2@proxy.example/v1\" x\n  |                                                   ^";
        let out = redact_secrets(url_err);
        assert!(!out.contains("hunter2"), "{out}");
        assert!(
            out.contains("4 | base_url = \"https://<redacted>@proxy.example/v1\" x"),
            "{out}"
        );
    }

    #[test]
    fn urls_are_scrubbed_of_credentials() {
        let cases = [
            (
                "https://user:hunter2@proxy.example/v1",
                "https://<redacted>@proxy.example/v1",
            ),
            // A token used as the username, with no password.
            (
                "https://tok3n@proxy.example",
                "https://<redacted>@proxy.example",
            ),
            (
                "https://proxy.example/v1?api-version=2024-02-01&api_key=abc123",
                "https://proxy.example/v1?api-version=2024-02-01&api_key=<redacted>",
            ),
            (
                "https://acct.blob.example/c?sv=2022-11-02&sig=abc%2Bdef&X-Amz-Credential=AKIA%2F2024",
                "https://acct.blob.example/c?sv=2022-11-02&sig=<redacted>&X-Amz-Credential=<redacted>",
            ),
            (
                "https://proxy.example/v1#access_token=abc",
                "https://proxy.example/v1#access_token=<redacted>",
            ),
            // A key-shaped value under a harmless name.
            (
                "https://proxy.example/v1?x=sk-AbCdEf0123456789XyZ",
                "https://proxy.example/v1?x=<redacted>",
            ),
            // Doesn't parse (unencoded `/` in the password): still scrubbed.
            (
                "https://user:abc/def==@proxy.example/v1",
                "https://<redacted>@proxy.example/v1",
            ),
            ("user:pw@proxy.example/v1", "<redacted>@proxy.example/v1"),
        ];
        for (url, want) in cases {
            assert_eq!(redact_url(url), want, "{url}");
        }

        // Nothing to hide: shown exactly as configured.
        for url in [
            "https://api.openai.com/v1/",
            "http://localhost:11434",
            "https://proxy.example/v1?api-version=2024-02-01",
            "https://vertex.example/v1/models/claude-sonnet@20250514",
        ] {
            assert_eq!(redact_url(url), url);
        }
    }

    #[test]
    fn transport_errors_hide_url_credentials() {
        // reqwest names the URL in its own message. A URL without a host is
        // a builder error, so this needs no network.
        let err = reqwest::Client::new()
            .get("data:text/plain?api_key=SuperSecretValue")
            .build()
            .expect_err("a URL without a host is a builder error");
        let facts = transport_facts(&err);
        assert!(
            facts.chain.contains("api_key=<redacted>"),
            "{}",
            facts.chain
        );
        assert!(!facts.chain.contains("SuperSecretValue"), "{}", facts.chain);
        assert!(!facts.root.contains("SuperSecretValue"), "{}", facts.root);
    }

    #[test]
    fn redaction_leaves_ordinary_text_alone() {
        let text = "unknown variant `grok-4-1-fast-non-reasoning-extra`, expected one of `fast`, `smart` in /Users/Cole/Library/Application Support/ai-cli/config.toml";
        assert_eq!(redact_secrets(text), text);
    }

    // --- web search and telemetry ----------------------------------------------------

    #[test]
    fn search_keys_presence() {
        let search = SearchConfig::default(); // primary gemini, fallback tavily
        let env = FakeEnv::new(&[("GOOGLE_API_KEY", "g")]);
        let section = search_section(&search, &env);
        assert_eq!(section.checks[0].status, Status::Ok);
        assert!(section.checks[0]
            .text
            .contains("gemini (primary): GOOGLE_API_KEY set"));
        assert_eq!(section.checks[1].status, Status::Info);
        assert!(section.checks[1].text.contains("TAVILY_API_KEY not set"));

        let tavily_primary = SearchConfig {
            provider: "tavily".into(),
            ..SearchConfig::default()
        };
        let section = search_section(&tavily_primary, &FakeEnv::new(&[]));
        let tavily = section
            .checks
            .iter()
            .find(|c| c.text.starts_with("tavily"))
            .unwrap();
        assert_eq!(tavily.status, Status::Warn);
    }

    #[test]
    fn blank_search_key_shadows_later_vars_like_the_runtime() {
        // `GeminiSearchClient::from_env` takes the first non-empty var, so a
        // whitespace-only GEMINI_API_KEY is sent (and rejected) and
        // GOOGLE_API_KEY is never reached.
        let search = SearchConfig::default();
        let env = FakeEnv::new(&[("GEMINI_API_KEY", "  "), ("GOOGLE_API_KEY", "g")]);
        let gemini = &search_section(&search, &env).checks[0];
        assert_eq!(gemini.status, Status::Warn);
        assert!(
            gemini
                .text
                .contains("gemini (primary): GEMINI_API_KEY is set but blank"),
            "{}",
            gemini.text
        );
        assert_eq!(
            gemini.details,
            ["GOOGLE_API_KEY is ignored while GEMINI_API_KEY is set; unset GEMINI_API_KEY to use it"]
        );

        // An empty value counts as unset there, so the fallback var is used.
        let env = FakeEnv::new(&[("GEMINI_API_KEY", ""), ("GOOGLE_API_KEY", "g")]);
        let gemini = &search_section(&search, &env).checks[0];
        assert_eq!(gemini.status, Status::Ok);
        assert!(gemini.text.contains("gemini (primary): GOOGLE_API_KEY set"));

        // A blank fallback var with nothing set before it, and a blank Tavily key.
        let env = FakeEnv::new(&[("GOOGLE_API_KEY", " "), ("TAVILY_API_KEY", "\t")]);
        let section = search_section(&search, &env);
        assert_eq!(section.checks[0].status, Status::Warn);
        assert!(section.checks[0]
            .text
            .contains("GOOGLE_API_KEY is set but blank"));
        assert!(section.checks[0].details.is_empty());
        assert_eq!(section.checks[1].status, Status::Warn);
        assert!(section.checks[1]
            .text
            .contains("tavily (fallback): TAVILY_API_KEY is set but blank"));
    }

    #[test]
    fn telemetry_states() {
        let none = telemetry_section(&FakeEnv::new(&[]));
        assert_eq!(none.checks[0].status, Status::Info);

        let all = telemetry_section(&FakeEnv::new(&[
            ("LANGFUSE_HOST", "h"),
            ("LANGFUSE_PUBLIC_KEY", "p"),
            ("LANGFUSE_SECRET_KEY", "s"),
        ]));
        assert_eq!(all.checks[0].status, Status::Ok);

        let partial = telemetry_section(&FakeEnv::new(&[("LANGFUSE_HOST", "h")]));
        assert_eq!(partial.checks[0].status, Status::Warn);
        assert_eq!(
            partial.checks[0].text,
            "Langfuse telemetry off: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY not set (LANGFUSE_HOST set)"
        );

        let off = telemetry_section(&FakeEnv::new(&[
            ("LANGFUSE_HOST", "h"),
            ("LANGFUSE_PUBLIC_KEY", "p"),
            ("LANGFUSE_SECRET_KEY", "s"),
            ("LANGFUSE_ENABLED", "false"),
        ]));
        assert!(off.checks[0].text.contains("disabled"));
    }

    // --- end to end (no network) ----------------------------------------------------

    #[test]
    fn report_never_contains_key_values() {
        let secret = "sk-test-SuperSecretValue0123456789";
        let env = FakeEnv::new(&[
            ("ANTHROPIC_API_KEY", secret),
            ("OPENAI_API_KEY", secret),
            ("GEMINI_API_KEY", secret),
            ("TAVILY_API_KEY", secret),
            ("LANGFUSE_HOST", "https://langfuse.example"),
            ("LANGFUSE_PUBLIC_KEY", secret),
            ("LANGFUSE_SECRET_KEY", secret),
            ("HTTPS_PROXY", "http://user:hunter2@proxy.example:8080"),
        ]);
        let mut config = Config::default();
        if let Some(xai) = config.providers.get_mut("xai") {
            xai.api_key = Some(secret.to_string());
        }
        // Credentials embedded in a configured endpoint.
        if let Some(openai) = config.providers.get_mut("openai") {
            openai.base_url = Some(
                "https://user:hunter3@proxy.example/v1?api-version=2024-02-01&api_key=url-secret-0123"
                    .to_string(),
            );
        }
        let loaded = LoadedConfig {
            path: Ok(PathBuf::from("/home/u/.config/ai-cli/config.toml")),
            state: ConfigState::Loaded,
            config,
            stray: None,
        };
        let entries = provider_entries(&loaded.config, &env);
        let targets: Vec<ProbeTarget> = entries
            .iter()
            .filter(|e| e.enabled)
            .filter_map(|e| probe_target(e, &env).ok())
            .collect();
        assert!(!targets.is_empty());
        assert!(!format!("{targets:?}").contains(secret));
        assert!(!format!("{targets:?}").contains("hunter2"));

        let results = vec![probe_result(
            "anthropic",
            ProbeOutcome::AuthFailed { status: 401 },
            &["160.79.104.10"],
            false,
        )];
        let out = render(&[
            system_section(&loaded, &env),
            providers_section(&entries),
            connectivity_section(&results, &[], false),
            search_section(&loaded.config.search, &env),
            telemetry_section(&env),
            summary_section(&loaded.state, false, &summarize(&results)),
        ]);
        assert!(!out.contains(secret), "{out}");
        assert!(!out.contains("SuperSecret"), "{out}");
        assert!(!out.contains("hunter2"), "{out}");
        assert!(!out.contains("hunter3"), "{out}");
        assert!(!out.contains("url-secret"), "{out}");
        assert!(out.contains(
            "endpoint: https://<redacted>@proxy.example/v1?api-version=2024-02-01&api_key=<redacted> (base_url)"
        ), "{out}");
        assert!(out.contains(&format!(
            "env ANTHROPIC_API_KEY (set, {} chars)",
            secret.len()
        )));
        assert!(out.contains("config api_key (set,"));
        assert!(out.contains("proxy: HTTPS_PROXY set"));
    }
}
