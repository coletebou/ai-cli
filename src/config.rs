use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub default: DefaultConfig,
    #[serde(default)]
    pub footer: FooterConfig,
    #[serde(default)]
    pub providers: HashMap<String, ProviderConfig>,
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
    pub default_model: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default = "default_color")]
    pub color: String,
    pub base_url: Option<String>,
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
                default_model: "gpt-5.2".to_string(),
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
                default_model: "gemini-3-flash-preview".to_string(),
                aliases: vec!["gemini".to_string()],
                color: "cyan".to_string(),
                base_url: None,
            },
        );

        providers.insert(
            "groq".to_string(),
            ProviderConfig {
                enabled: false,
                api_key_env: Some("GROQ_API_KEY".to_string()),
                api_key: None,
                default_model: "llama-3.3-70b-versatile".to_string(),
                aliases: vec!["groq".to_string()],
                color: "yellow".to_string(),
                base_url: None,
            },
        );

        providers.insert(
            "deepseek".to_string(),
            ProviderConfig {
                enabled: false,
                api_key_env: Some("DEEPSEEK_API_KEY".to_string()),
                api_key: None,
                default_model: "deepseek-chat".to_string(),
                aliases: vec!["deepseek".to_string(), "ds".to_string()],
                color: "white".to_string(),
                base_url: None,
            },
        );

        providers.insert(
            "ollama".to_string(),
            ProviderConfig {
                enabled: false,
                api_key_env: None,
                api_key: None,
                default_model: "llama3.2".to_string(),
                aliases: vec!["ollama".to_string(), "local".to_string()],
                color: "red".to_string(),
                base_url: Some("http://localhost:11434".to_string()),
            },
        );

        Config {
            default: DefaultConfig::default(),
            footer: FooterConfig::default(),
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
            let config: Config = toml::from_str(&content)
                .with_context(|| "Failed to parse config file")?;
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
