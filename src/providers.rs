use crate::config::{Config, ModelTier, ProviderConfig};
use anyhow::{Context, Result};
use colored::Color;
use genai::adapter::AdapterKind;
use genai::chat::{ChatMessage, ChatOptions, ChatRequest, ChatResponse, ChatStreamResponse};
use genai::resolver::{AuthData, AuthResolver, Endpoint, ServiceTargetResolver};
use genai::{Client, ClientConfig, ModelIden, ServiceTarget};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Provider {
    pub display_name: String,
    pub model: String,
    pub color: Color,
}

impl Provider {
    pub fn from_config(name: &str, config: &ProviderConfig, tier: &ModelTier) -> Self {
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

        // Select model based on tier
        let model = config.get_model(tier).to_string();

        Provider {
            display_name,
            model,
            color,
        }
    }
}

pub struct ProviderManager {
    client: Client,
    config: Arc<Config>,
    providers: HashMap<String, Provider>,
    #[allow(dead_code)]
    model_tier: ModelTier,
}

impl ProviderManager {
    pub fn new(config: Config, model_tier: ModelTier) -> Result<Self> {
        let config = Arc::new(config);

        // Build custom auth resolver that uses our config
        let auth_config = Arc::clone(&config);
        let auth_resolver = AuthResolver::from_resolver_fn(
            move |model_iden: ModelIden| -> Result<Option<AuthData>, genai::resolver::Error> {
                let adapter_kind = model_iden.adapter_kind;

                // Map adapter kind to our provider name
                let provider_name = match adapter_kind.as_str() {
                    "Anthropic" => "anthropic",
                    "OpenAI" => "openai",
                    "Gemini" => "google",
                    "Ollama" => "ollama",
                    "XAi" => "xai",
                    _ => return Ok(None),
                };

                if let Some(key) = auth_config.get_api_key(provider_name) {
                    Ok(Some(AuthData::from_single(key)))
                } else {
                    // Try standard env var names as fallback
                    let env_name = match adapter_kind.as_str() {
                        "Anthropic" => "ANTHROPIC_API_KEY",
                        "OpenAI" => "OPENAI_API_KEY",
                        "Gemini" => "GEMINI_API_KEY",
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

        // ServiceTargetResolver: redirect OpenRouter models to OpenRouter's endpoint.
        // OpenRouter models use slash-namespaced IDs (e.g. "google/gemini-2.5-flash"),
        // so any model containing a '/' is routed through OpenRouter.
        let or_config = Arc::clone(&config);
        let or_enabled = config
            .providers
            .get("openrouter")
            .map(|p| p.enabled)
            .unwrap_or(false);

        let target_resolver = ServiceTargetResolver::from_resolver_fn(
            move |service_target: ServiceTarget| -> Result<ServiceTarget, genai::resolver::Error> {
                let ServiceTarget {
                    model,
                    endpoint,
                    auth,
                } = service_target;

                // Route slash-namespaced models through OpenRouter
                if or_enabled && model.model_name.contains('/') {
                    let or_endpoint =
                        Endpoint::from_static("https://openrouter.ai/api/v1/");
                    let or_auth = if let Some(key) = or_config.get_api_key("openrouter") {
                        AuthData::from_single(key)
                    } else {
                        AuthData::from_env("OPENROUTER_API_KEY")
                    };
                    let or_model = ModelIden::new(AdapterKind::OpenAI, model.model_name);
                    Ok(ServiceTarget {
                        endpoint: or_endpoint,
                        auth: or_auth,
                        model: or_model,
                    })
                } else {
                    Ok(ServiceTarget {
                        model,
                        endpoint,
                        auth,
                    })
                }
            },
        );

        let client_config = ClientConfig::default();
        let client = Client::builder()
            .with_config(client_config)
            .with_auth_resolver(auth_resolver)
            .with_service_target_resolver(target_resolver)
            .build();

        // Build provider map
        let mut providers = HashMap::new();
        for (name, provider_config) in &config.providers {
            let provider = Provider::from_config(name, provider_config, &model_tier);
            providers.insert(name.clone(), provider);
        }

        Ok(ProviderManager {
            client,
            config,
            providers,
            model_tier,
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
