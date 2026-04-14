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
    pub providers: HashMap<String, ProviderConfig>,
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
}

impl ProviderConfig {
    /// Get model for the specified tier, falling back to default_model
    pub fn get_model(&self, tier: &ModelTier) -> &str {
        match tier {
            ModelTier::Fast => self
                .fast_model
                .as_deref()
                .unwrap_or(&self.default_model),
            ModelTier::Smart => self
                .smart_model
                .as_deref()
                .unwrap_or(&self.default_model),
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
                smart_model: Some("claude-opus-4-6".to_string()),
                aliases: vec!["claude".to_string()],
                color: "magenta".to_string(),
                base_url: None,
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
                smart_model: Some("gpt-5.4".to_string()),
                aliases: vec!["chatgpt".to_string(), "gpt".to_string()],
                color: "green".to_string(),
                base_url: None,
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
            },
        );

        providers.insert(
            "google".to_string(),
            ProviderConfig {
                enabled: true,
                api_key_env: Some("GEMINI_API_KEY".to_string()),
                api_key: None,
                default_model: "gemini-3.1-flash-lite-preview".to_string(),
                fast_model: Some("gemini-3.1-flash-lite-preview".to_string()),
                smart_model: Some("gemini-3.1-pro-preview".to_string()),
                aliases: vec!["gemini".to_string()],
                color: "cyan".to_string(),
                base_url: None,
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
                smart_model: Some("anthropic/claude-opus-4-6".to_string()),
                aliases: vec!["openrouter".to_string(), "or".to_string()],
                color: "yellow".to_string(),
                base_url: Some("https://openrouter.ai/api/v1".to_string()),
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
            let mut config: Config = toml::from_str(&content)
                .with_context(|| "Failed to parse config file")?;

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

    /// Generate default config file
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

        let config = Config::default();
        let content = toml::to_string_pretty(&config)
            .with_context(|| "Failed to serialize config")?;
        fs::write(&config_path, &content)
            .with_context(|| format!("Failed to write config file: {:?}", config_path))?;

        println!("Created config file: {:?}", config_path);
        println!("\nYou can now customize your settings:");
        println!("  - Set default providers");
        println!("  - Change default models");
        println!("  - Add a default system prompt");
        println!("  - Configure API keys (or use environment variables)");

        Ok(())
    }

    /// Save config to disk
    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path();

        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create config directory: {:?}", parent))?;
        }

        let content = toml::to_string_pretty(self)
            .with_context(|| "Failed to serialize config")?;
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
    fn test_resolve_alias() {
        let config = Config::default();
        assert_eq!(config.resolve_provider_alias("claude"), Some("anthropic".to_string()));
        assert_eq!(config.resolve_provider_alias("gpt"), Some("openai".to_string()));
        assert_eq!(config.resolve_provider_alias("grok"), Some("xai".to_string()));
        assert_eq!(config.resolve_provider_alias("gemini"), Some("google".to_string()));
    }
}
