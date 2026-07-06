use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// CLI mode settings (non-TUI)
    #[serde(default)]
    pub cli: ModeConfig,
    /// TUI mode settings
    #[serde(default)]
    pub tui_mode: ModeConfig,
    /// Legacy mode config - migrated to cli on load
    #[serde(default, skip_serializing)]
    mode: Option<ModeConfig>,
    #[serde(default)]
    pub default: DefaultConfig,
    #[serde(default)]
    pub footer: FooterConfig,
    #[serde(default)]
    pub search: SearchConfig,
    #[serde(default)]
    pub retry: RetryConfig,
    #[serde(default)]
    pub providers: HashMap<String, ProviderConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RetryConfig {
    /// Total attempts, including the initial try. 1 disables retry.
    #[serde(default = "default_retry_max_attempts")]
    pub max_attempts: u32,
    /// First backoff delay; doubles each subsequent retry up to `max_delay_ms`.
    #[serde(default = "default_retry_base_ms")]
    pub base_ms: u64,
    /// Cap on per-attempt backoff delay.
    #[serde(default = "default_retry_max_delay_ms")]
    pub max_delay_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: default_retry_max_attempts(),
            base_ms: default_retry_base_ms(),
            max_delay_ms: default_retry_max_delay_ms(),
        }
    }
}

fn default_retry_max_attempts() -> u32 {
    3
}
fn default_retry_base_ms() -> u64 {
    250
}
fn default_retry_max_delay_ms() -> u64 {
    5_000
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Active search provider: "gemini" or "tavily"
    #[serde(default = "default_search_provider")]
    pub provider: String,
    /// Maximum number of search results per query (Tavily only)
    #[serde(default = "default_max_results")]
    pub max_results: usize,
    /// Maximum rounds of tool calls per query (safety cap)
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
    /// Tavily-specific settings
    #[serde(default)]
    pub tavily: TavilyConfig,
    /// Gemini grounding-specific settings
    #[serde(default)]
    pub gemini: GeminiSearchConfig,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            provider: default_search_provider(),
            max_results: default_max_results(),
            max_iterations: default_max_iterations(),
            tavily: TavilyConfig::default(),
            gemini: GeminiSearchConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TavilyConfig {
    #[serde(default = "default_tavily_api_key_env")]
    pub api_key_env: String,
}

impl Default for TavilyConfig {
    fn default() -> Self {
        Self {
            api_key_env: default_tavily_api_key_env(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiSearchConfig {
    /// Primary env var to check (defaults to GEMINI_API_KEY)
    #[serde(default = "default_gemini_api_key_env")]
    pub api_key_env: String,
    /// Fallback env var (defaults to GOOGLE_API_KEY)
    #[serde(default = "default_gemini_fallback_env")]
    pub fallback_api_key_env: String,
    /// Which Gemini model to use for grounded search
    #[serde(default = "default_gemini_search_model")]
    pub model: String,
}

impl Default for GeminiSearchConfig {
    fn default() -> Self {
        Self {
            api_key_env: default_gemini_api_key_env(),
            fallback_api_key_env: default_gemini_fallback_env(),
            model: default_gemini_search_model(),
        }
    }
}

fn default_search_provider() -> String {
    "gemini".to_string()
}

fn default_tavily_api_key_env() -> String {
    "TAVILY_API_KEY".to_string()
}

fn default_gemini_api_key_env() -> String {
    "GEMINI_API_KEY".to_string()
}

fn default_gemini_fallback_env() -> String {
    "GOOGLE_API_KEY".to_string()
}

fn default_gemini_search_model() -> String {
    "gemini-3.1-flash-lite-preview".to_string()
}

fn default_max_results() -> usize {
    5
}

fn default_max_iterations() -> usize {
    5
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum QueryMode {
    #[default]
    Single,
    Multi,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ModelTier {
    #[default]
    Fast,
    Smart,
}

/// How a provider authenticates its requests.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum AuthMode {
    /// Standard metered API billing via `api_key` / `api_key_env`.
    #[default]
    ApiKey,
    /// Anthropic only. Rides a Claude Pro/Max subscription's included usage
    /// by replaying the OAuth token Claude Code stores after `claude login`,
    /// instead of billing against a separate API key. This is unofficial and
    /// reverse-engineered (not a documented Anthropic API): it swaps in a
    /// `Bearer` token and an `anthropic-beta: oauth-2025-04-20` header in
    /// place of `x-api-key`. It can stop working without notice, and using a
    /// subscription-scoped credential outside Anthropic's own clients is
    /// outside the letter of Anthropic's usage terms — there's real risk of
    /// the token or account getting flagged. See `anthropic_oauth.rs`.
    Subscription,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModeConfig {
    #[serde(default)]
    pub default_mode: QueryMode,
    #[serde(default = "default_single_provider")]
    pub single_provider: String,
    #[serde(default = "default_multi_providers")]
    pub multi_providers: Vec<String>,
    #[serde(default)]
    pub model_tier: ModelTier,
}

impl Default for ModeConfig {
    fn default() -> Self {
        Self {
            default_mode: QueryMode::Single,
            single_provider: default_single_provider(),
            multi_providers: default_multi_providers(),
            model_tier: ModelTier::Fast,
        }
    }
}

fn default_single_provider() -> String {
    "claude".to_string()
}

fn default_multi_providers() -> Vec<String> {
    vec![
        "claude".to_string(),
        "gpt".to_string(),
        "grok".to_string(),
        "gemini".to_string(),
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultConfig {
    #[serde(default = "default_providers")]
    pub providers: Vec<String>,
    #[serde(default = "default_true")]
    pub stream: bool,
    #[serde(default)]
    pub tui: bool,
    pub system_prompt: Option<String>,
}

impl Default for DefaultConfig {
    fn default() -> Self {
        Self {
            providers: default_providers(),
            stream: true,
            tui: false,
            system_prompt: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FooterConfig {
    #[serde(default = "default_true")]
    pub show_model: bool,
    #[serde(default = "default_true")]
    pub show_time: bool,
    #[serde(default)]
    pub show_tokens: bool,
    #[serde(default = "default_true")]
    pub show_reasoning: bool,
}

impl Default for FooterConfig {
    fn default() -> Self {
        Self {
            show_model: true,
            show_time: true,
            show_tokens: false,
            show_reasoning: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    pub api_key_env: Option<String>,
    pub api_key: Option<String>,
    /// Legacy field - use fast_model/smart_model instead
    #[serde(default)]
    pub default_model: String,
    /// Model for fast tier (cheaper, faster responses)
    pub fast_model: Option<String>,
    /// Model for smart tier (better quality, slower)
    pub smart_model: Option<String>,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default = "default_color")]
    pub color: String,
    pub base_url: Option<String>,
    /// How this provider authenticates. Defaults to `api_key`; `subscription`
    /// only has an effect for the `anthropic` provider.
    #[serde(default)]
    pub auth_mode: AuthMode,
}

impl ProviderConfig {
    /// Get model for the specified tier, falling back to default_model
    pub fn get_model(&self, tier: &ModelTier) -> &str {
        match tier {
            ModelTier::Fast => self.fast_model.as_deref().unwrap_or(&self.default_model),
            ModelTier::Smart => self.smart_model.as_deref().unwrap_or(&self.default_model),
        }
    }
}

fn default_providers() -> Vec<String> {
    vec![
        "claude".to_string(),
        "chatgpt".to_string(),
        "grok".to_string(),
        "gemini".to_string(),
    ]
}

fn default_true() -> bool {
    true
}

fn default_color() -> String {
    "white".to_string()
}

impl Default for Config {
    fn default() -> Self {
        let mut providers = HashMap::new();

        providers.insert(
            "anthropic".to_string(),
            ProviderConfig {
                enabled: true,
                api_key_env: Some("ANTHROPIC_API_KEY".to_string()),
                api_key: None,
                default_model: "claude-haiku-4-5".to_string(),
                fast_model: Some("claude-haiku-4-5".to_string()),
                smart_model: Some("claude-opus-4-8".to_string()),
                aliases: vec!["claude".to_string()],
                color: "magenta".to_string(),
                base_url: None,
                auth_mode: AuthMode::ApiKey,
            },
        );

        providers.insert(
            "openai".to_string(),
            ProviderConfig {
                enabled: true,
                api_key_env: Some("OPENAI_API_KEY".to_string()),
                api_key: None,
                default_model: "gpt-5-nano".to_string(),
                fast_model: Some("gpt-5-nano".to_string()),
                smart_model: Some("gpt-5.5".to_string()),
                aliases: vec!["chatgpt".to_string(), "gpt".to_string()],
                color: "green".to_string(),
                base_url: None,
                auth_mode: AuthMode::ApiKey,
            },
        );

        providers.insert(
            "xai".to_string(),
            ProviderConfig {
                enabled: true,
                api_key_env: Some("XAI_API_KEY".to_string()),
                api_key: None,
                default_model: "grok-4-1-fast-non-reasoning".to_string(),
                fast_model: Some("grok-4-1-fast-non-reasoning".to_string()),
                smart_model: Some("grok-4".to_string()),
                aliases: vec!["grok".to_string()],
                color: "blue".to_string(),
                base_url: None,
                auth_mode: AuthMode::ApiKey,
            },
        );

        providers.insert(
            "google".to_string(),
            ProviderConfig {
                enabled: true,
                api_key_env: Some("GEMINI_API_KEY".to_string()),
                api_key: None,
                default_model: "gemini-3.5-flash".to_string(),
                fast_model: Some("gemini-3.5-flash".to_string()),
                smart_model: Some("gemini-3.1-pro-preview".to_string()),
                aliases: vec!["gemini".to_string()],
                color: "cyan".to_string(),
                base_url: None,
                auth_mode: AuthMode::ApiKey,
            },
        );

        providers.insert(
            "openrouter".to_string(),
            ProviderConfig {
                enabled: true,
                api_key_env: Some("OPENROUTER_API_KEY".to_string()),
                api_key: None,
                default_model: "google/gemini-3.1-flash-lite-preview".to_string(),
                fast_model: Some("google/gemini-3.1-flash-lite-preview".to_string()),
                smart_model: Some("anthropic/claude-opus-4-8".to_string()),
                aliases: vec!["openrouter".to_string(), "or".to_string()],
                color: "yellow".to_string(),
                base_url: Some("https://openrouter.ai/api/v1".to_string()),
                auth_mode: AuthMode::ApiKey,
            },
        );

        providers.insert(
            "ollama".to_string(),
            ProviderConfig {
                enabled: false,
                api_key_env: None,
                api_key: None,
                default_model: "qwen3:8b".to_string(),
                fast_model: Some("qwen3:8b".to_string()),
                smart_model: Some("qwen3:14b".to_string()),
                aliases: vec!["ollama".to_string(), "local".to_string()],
                color: "red".to_string(),
                base_url: Some("http://localhost:11434".to_string()),
                auth_mode: AuthMode::ApiKey,
            },
        );

        Config {
            cli: ModeConfig::default(),
            tui_mode: ModeConfig {
                default_mode: QueryMode::Multi, // TUI defaults to multi
                ..ModeConfig::default()
            },
            mode: None,
            default: DefaultConfig::default(),
            footer: FooterConfig::default(),
            search: SearchConfig::default(),
            retry: RetryConfig::default(),
            providers,
        }
    }
}

impl Config {
    pub fn config_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("ai-cli")
            .join("config.toml")
    }

    pub fn load() -> Result<Self> {
        let config_path = Self::config_path();

        if config_path.exists() {
            let content = fs::read_to_string(&config_path)
                .with_context(|| format!("Failed to read config file: {:?}", config_path))?;
            let mut config: Config =
                toml::from_str(&content).with_context(|| "Failed to parse config file")?;

            // Migrate legacy [mode] section to [cli]
            if let Some(legacy_mode) = config.mode.take() {
                // Only migrate if cli section is default
                if config.cli == ModeConfig::default() {
                    config.cli = legacy_mode.clone();
                }
                // Also migrate to tui_mode if it's default (but default to Multi for TUI)
                if config.tui_mode == ModeConfig::default() {
                    config.tui_mode = ModeConfig {
                        default_mode: QueryMode::Multi,
                        ..legacy_mode
                    };
                }
            }

            Ok(config)
        } else {
            Ok(Config::default())
        }
    }

    /// Generate a richly-commented default config file.
    pub fn init() -> Result<()> {
        let config_path = Self::config_path();

        if config_path.exists() {
            eprintln!("Config file already exists: {:?}", config_path);
            eprintln!("Remove it first if you want to regenerate.");
            std::process::exit(1);
        }

        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create config directory: {:?}", parent))?;
        }

        let content = Self::generate_init_template();
        fs::write(&config_path, &content)
            .with_context(|| format!("Failed to write config file: {:?}", config_path))?;

        println!("Created config file: {:?}", config_path);
        println!("\nEdit it directly or run `ai config` for the interactive TUI editor.");

        Ok(())
    }

    fn env_status(var: &str) -> &'static str {
        if env::var(var).is_ok() {
            "✓ set"
        } else {
            "✗ missing"
        }
    }

    fn generate_init_template() -> String {
        format!(
            r##"# ─────────────────────────────────────────────────────────────────────
# ai-cli configuration
# ─────────────────────────────────────────────────────────────────────
# Edit this file or run `ai config` for the interactive TUI editor.
#
# API key status (detected at generation time):
#   ANTHROPIC_API_KEY   {anthropic_status}
#   OPENAI_API_KEY      {openai_status}
#   XAI_API_KEY         {xai_status}
#   GEMINI_API_KEY      {gemini_status}
#   OPENROUTER_API_KEY  {openrouter_status}
#
# Providers without a valid API key are silently skipped at query time.

# ── CLI mode (non-TUI) ──────────────────────────────────────────────
# Controls behavior when running `ai "question"` (no -t flag).

[cli]
# "single" = query one provider; "multi" = query several in parallel
default_mode = "single"
# Which provider to use in single mode (alias name, e.g. "claude")
single_provider = "claude"
# Providers queried in multi mode
multi_providers = ["claude", "gpt", "grok", "gemini"]
# "fast" = cheaper/faster models; "smart" = best quality models
model_tier = "fast"

# ── TUI mode ────────────────────────────────────────────────────────
# Controls behavior when running `ai -t "question"`.

[tui_mode]
default_mode = "multi"
single_provider = "claude"
multi_providers = ["claude", "gpt", "grok", "gemini"]
model_tier = "fast"

# ── General defaults ────────────────────────────────────────────────

[default]
# Providers queried when no -p flag is given (alias names)
providers = ["claude", "chatgpt", "grok", "gemini"]
# Stream tokens as they arrive (disable with --no-stream)
stream = true
# Start in TUI mode by default (override with -t / --no-tui)
tui = false
# Optional system prompt applied to every query
# system_prompt = "You are a helpful assistant."

# ── Response footer ─────────────────────────────────────────────────

[footer]
show_model = true
show_time = true
show_tokens = false
show_reasoning = true

# ── Web search (--search / -w) ──────────────────────────────────────
# Primary search backend; the other is used as automatic fallback.

[search]
# "gemini" (Google Search grounding) or "tavily"
provider = "gemini"
# Max results per search call (Tavily only)
max_results = 5
# Safety cap on tool-call rounds per query
max_iterations = 5

[search.gemini]
api_key_env = "GEMINI_API_KEY"
fallback_api_key_env = "GOOGLE_API_KEY"
model = "gemini-3.1-flash-lite-preview"

[search.tavily]
api_key_env = "TAVILY_API_KEY"

# ── Retry behavior ──────────────────────────────────────────────────
# Transient errors (429, 5xx, timeouts) are retried with exponential
# backoff. A Retry-After header from the server takes precedence.

[retry]
# Total attempts including the first try; 1 disables retry
max_attempts = 3
# Initial backoff delay in ms; doubles each subsequent retry
base_ms = 250
# Ceiling per-attempt delay in ms
max_delay_ms = 5000

# ── Providers ────────────────────────────────────────────────────────
# Each provider maps to a genai adapter. Use aliases to reference them
# in CLI flags and mode configs (e.g. -p claude,gpt).
#
# Fields:
#   enabled      – include in provider list (default: true)
#   api_key_env  – env var holding the API key
#   api_key      – hardcoded key (not recommended; prefer env vars)
#   fast_model   – model used when model_tier = "fast"
#   smart_model  – model used when model_tier = "smart"
#   aliases      – short names for -p flag (first alias used as display name)
#   color        – header color: red, green, blue, cyan, magenta, yellow, white
#   base_url     – override the default API endpoint (for proxies, Bedrock, etc.)
#   auth_mode    – "api_key" (default) or "subscription" (Anthropic only, see below)

[providers.anthropic]
enabled = true
api_key_env = "ANTHROPIC_API_KEY"
fast_model = "claude-haiku-4-5"
smart_model = "claude-opus-4-8"
aliases = ["claude"]
color = "magenta"
# For AWS Bedrock: set base_url and use a Bedrock API key.
# Model IDs like "us.anthropic.claude-opus-4-7" are automatically
# routed to the Anthropic adapter regardless of their prefix.
# base_url = "https://bedrock-runtime.us-east-1.amazonaws.com/v1/"
# api_key_env = "AWS_BEARER_TOKEN_BEDROCK"
#
# To bill against a Claude Pro/Max subscription instead of the metered API,
# replay the OAuth token Claude Code stores at ~/.claude/.credentials.json
# after `claude login`. This is unofficial/reverse-engineered — not a
# documented Anthropic API — and can break or get flagged without notice.
# auth_mode = "subscription"

[providers.openai]
enabled = true
api_key_env = "OPENAI_API_KEY"
fast_model = "gpt-5-nano"
smart_model = "gpt-5.4"
aliases = ["chatgpt", "gpt"]
color = "green"

[providers.xai]
enabled = true
api_key_env = "XAI_API_KEY"
fast_model = "grok-4-1-fast-non-reasoning"
smart_model = "grok-4"
aliases = ["grok"]
color = "blue"

[providers.google]
enabled = true
api_key_env = "GEMINI_API_KEY"
fast_model = "gemini-3.5-flash"
smart_model = "gemini-3.1-pro-preview"
aliases = ["gemini"]
color = "cyan"

[providers.openrouter]
enabled = true
api_key_env = "OPENROUTER_API_KEY"
fast_model = "google/gemini-3.1-flash-lite-preview"
smart_model = "anthropic/claude-opus-4-8"
aliases = ["openrouter", "or"]
color = "yellow"
base_url = "https://openrouter.ai/api/v1"

[providers.ollama]
enabled = false
fast_model = "qwen3:8b"
smart_model = "qwen3:14b"
aliases = ["ollama", "local"]
color = "red"
base_url = "http://localhost:11434"
"##,
            anthropic_status = Self::env_status("ANTHROPIC_API_KEY"),
            openai_status = Self::env_status("OPENAI_API_KEY"),
            xai_status = Self::env_status("XAI_API_KEY"),
            gemini_status = Self::env_status("GEMINI_API_KEY"),
            openrouter_status = Self::env_status("OPENROUTER_API_KEY"),
        )
    }

    /// Save config to disk
    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path();

        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create config directory: {:?}", parent))?;
        }

        let content = toml::to_string_pretty(self).with_context(|| "Failed to serialize config")?;
        fs::write(&config_path, &content)
            .with_context(|| format!("Failed to write config file: {:?}", config_path))?;

        Ok(())
    }

    pub fn get_api_key(&self, provider_name: &str) -> Option<String> {
        self.providers.get(provider_name).and_then(|p| {
            // First check direct api_key
            if let Some(ref key) = p.api_key {
                return Some(key.clone());
            }
            // Then check env var
            if let Some(ref env_var) = p.api_key_env {
                if let Ok(key) = env::var(env_var) {
                    return Some(key);
                }
                // Also check GOOGLE_API_KEY for gemini
                if env_var == "GEMINI_API_KEY" {
                    if let Ok(key) = env::var("GOOGLE_API_KEY") {
                        return Some(key);
                    }
                }
            }
            None
        })
    }

    pub fn resolve_provider_alias(&self, alias: &str) -> Option<String> {
        let alias_lower = alias.to_lowercase();
        for (name, config) in &self.providers {
            if name.to_lowercase() == alias_lower {
                return Some(name.clone());
            }
            for a in &config.aliases {
                if a.to_lowercase() == alias_lower {
                    return Some(name.clone());
                }
            }
        }
        None
    }

    pub fn get_enabled_providers(&self) -> Vec<String> {
        self.providers
            .iter()
            .filter(|(_, config)| config.enabled)
            .map(|(name, _)| name.clone())
            .collect()
    }

    pub fn get_default_providers(&self) -> Vec<String> {
        if self.default.providers.is_empty() {
            self.get_enabled_providers()
        } else {
            self.default
                .providers
                .iter()
                .filter_map(|alias| self.resolve_provider_alias(alias))
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.providers.contains_key("anthropic"));
        assert!(config.providers.contains_key("openai"));
        assert!(config.providers.contains_key("xai"));
        assert!(config.providers.contains_key("google"));
    }

    #[test]
    fn test_init_template_parses() {
        let template = Config::generate_init_template();
        let config: Config = toml::from_str(&template).expect("init template must be valid TOML");
        assert!(config.providers.contains_key("anthropic"));
        assert!(config.providers.contains_key("openai"));
        assert!(config.providers.contains_key("xai"));
        assert!(config.providers.contains_key("google"));
        assert!(config.providers.contains_key("openrouter"));
        assert!(config.providers.contains_key("ollama"));
        assert!(!config.providers["ollama"].enabled);
        assert_eq!(config.cli.single_provider, "claude");
        assert_eq!(config.search.provider, "gemini");
        assert_eq!(config.retry.max_attempts, 3);
    }

    #[test]
    fn test_resolve_alias() {
        let config = Config::default();
        assert_eq!(
            config.resolve_provider_alias("claude"),
            Some("anthropic".to_string())
        );
        assert_eq!(
            config.resolve_provider_alias("gpt"),
            Some("openai".to_string())
        );
        assert_eq!(
            config.resolve_provider_alias("grok"),
            Some("xai".to_string())
        );
        assert_eq!(
            config.resolve_provider_alias("gemini"),
            Some("google".to_string())
        );
    }
}
