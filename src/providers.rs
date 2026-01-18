use crate::config::{Config, ProviderConfig};
use anyhow::{Context, Result};
use colored::Color;
use genai::chat::{ChatMessage, ChatOptions, ChatRequest, ChatResponse, ChatStreamResponse};
use genai::resolver::{AuthData, AuthResolver};
use genai::{Client, ClientConfig, ModelIden};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Provider {
    pub display_name: String,
    pub model: String,
    pub color: Color,
}

impl Provider {
    pub fn from_config(name: &str, config: &ProviderConfig) -> Self {
        let display_name = config
            .aliases
            .first()
            .cloned()
            .unwrap_or_else(|| name.to_string())
            .to_uppercase();

        let color = match config.color.to_lowercase().as_str() {
            "red" => Color::Red,
            "green" => Color::Green,
            "blue" => Color::Blue,
            "cyan" => Color::Cyan,
            "magenta" => Color::Magenta,
            "yellow" => Color::Yellow,
            "white" => Color::White,
            _ => Color::White,
        };

        Provider {
            display_name,
            model: config.default_model.clone(),
            color,
        }
    }
}

pub struct ProviderManager {
    client: Client,
    config: Arc<Config>,
    providers: HashMap<String, Provider>,
}

impl ProviderManager {
    pub fn new(config: Config) -> Result<Self> {
        let config = Arc::new(config);
        let config_clone = Arc::clone(&config);

        // Build custom auth resolver that uses our config
        let auth_resolver = AuthResolver::from_resolver_fn(
            move |model_iden: ModelIden| -> Result<Option<AuthData>, genai::resolver::Error> {
                let adapter_kind = model_iden.adapter_kind;

                // Map adapter kind to our provider name
                let provider_name = match adapter_kind.as_str() {
                    "Anthropic" => "anthropic",
                    "OpenAI" => "openai",
                    "Gemini" => "google",
                    "Groq" => "groq",
                    "Ollama" => "ollama",
                    "DeepSeek" => "deepseek",
                    "XAi" => "xai",
                    _ => return Ok(None),
                };

                if let Some(key) = config_clone.get_api_key(provider_name) {
                    Ok(Some(AuthData::from_single(key)))
                } else {
                    // Try standard env var names as fallback
                    let env_name = match adapter_kind.as_str() {
                        "Anthropic" => "ANTHROPIC_API_KEY",
                        "OpenAI" => "OPENAI_API_KEY",
                        "Gemini" => "GEMINI_API_KEY",
                        "Groq" => "GROQ_API_KEY",
                        "DeepSeek" => "DEEPSEEK_API_KEY",
                        "XAi" => "XAI_API_KEY",
                        _ => return Ok(None),
                    };

                    if let Ok(key) = std::env::var(env_name) {
                        Ok(Some(AuthData::from_single(key)))
                    } else {
                        Ok(None)
                    }
                }
            },
        );

        let client_config = ClientConfig::default();
        let client = Client::builder()
            .with_config(client_config)
            .with_auth_resolver(auth_resolver)
            .build();

        // Build provider map
        let mut providers = HashMap::new();
        for (name, provider_config) in &config.providers {
            let provider = Provider::from_config(name, provider_config);
            providers.insert(name.clone(), provider);
        }

        Ok(ProviderManager {
            client,
            config,
            providers,
        })
    }

    pub fn resolve_alias(&self, alias: &str) -> Option<String> {
        self.config.resolve_provider_alias(alias)
    }

    pub fn get_providers_for_query(&self, requested: Option<&[String]>) -> Vec<Provider> {
        let provider_names: Vec<String> = if let Some(requested) = requested {
            // Resolve aliases
            requested
                .iter()
                .filter_map(|alias| self.resolve_alias(alias))
                .collect()
        } else {
            // Use default providers from config
            self.config.get_default_providers()
        };

        // Filter to only providers that have API keys configured
        provider_names
            .into_iter()
            .filter_map(|name| {
                let provider = self.providers.get(&name)?;
                // Check if API key is available (skip for ollama)
                if name != "ollama" && self.config.get_api_key(&name).is_none() {
                    return None;
                }
                Some(provider.clone())
            })
            .collect()
    }

    pub async fn exec_chat(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        options: Option<&ChatOptions>,
    ) -> Result<ChatResponse> {
        let request = ChatRequest::new(messages);
        self.client
            .exec_chat(model, request, options)
            .await
            .with_context(|| format!("Failed to execute chat request for model: {}", model))
    }

    pub async fn exec_chat_stream(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        options: Option<&ChatOptions>,
    ) -> Result<ChatStreamResponse> {
        let request = ChatRequest::new(messages);
        match self.client.exec_chat_stream(model, request, options).await {
            Ok(response) => Ok(response),
            Err(e) => {
                // Include the full error chain
                anyhow::bail!("Model: {} - {}", model, e)
            }
        }
    }

}

#[derive(Debug, Clone)]
pub struct ProviderResponse {
    pub provider: Provider,
    pub model: String,
    pub text: String,
    pub input_tokens: Option<i32>,
    pub output_tokens: Option<i32>,
    pub duration_ms: u64,
    pub error: Option<String>,
}

impl ProviderResponse {
    pub fn success(
        provider: Provider,
        model: String,
        text: String,
        input_tokens: Option<i32>,
        output_tokens: Option<i32>,
        duration_ms: u64,
    ) -> Self {
        ProviderResponse {
            provider,
            model,
            text,
            input_tokens,
            output_tokens,
            duration_ms,
            error: None,
        }
    }

    pub fn error(provider: Provider, model: String, error: String, duration_ms: u64) -> Self {
        ProviderResponse {
            provider,
            model,
            text: String::new(),
            input_tokens: None,
            output_tokens: None,
            duration_ms,
            error: Some(error),
        }
    }

    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }
}
