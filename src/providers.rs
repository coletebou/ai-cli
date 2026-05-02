use crate::bedrock::{is_bedrock_base_url, BedrockClient, BedrockOutcome};
use crate::config::{Config, ModelTier, ProviderConfig, RetryConfig};
use crate::retry::{self, Retryable};
use anyhow::{Context, Result};
use colored::Color;
use genai::adapter::AdapterKind;
use genai::chat::{ChatMessage, ChatOptions, ChatRequest, ChatResponse, ChatStreamResponse, Tool};
use genai::resolver::{AuthData, AuthResolver, Endpoint, ServiceTargetResolver};
use genai::{Client, ClientConfig, ModelIden, ServiceTarget};
use std::collections::HashMap;
use std::sync::Arc;

fn adapter_kind_for_provider(name: &str) -> Option<AdapterKind> {
    match name {
        "anthropic" => Some(AdapterKind::Anthropic),
        "openai" => Some(AdapterKind::OpenAI),
        "google" => Some(AdapterKind::Gemini),
        "xai" => Some(AdapterKind::Xai),
        "ollama" => Some(AdapterKind::Ollama),
        _ => None,
    }
}

fn provider_name_for_adapter(kind: &str) -> Option<&'static str> {
    match kind {
        "Anthropic" => Some("anthropic"),
        "OpenAI" => Some("openai"),
        "Gemini" => Some("google"),
        "Ollama" => Some("ollama"),
        "xAi" => Some("xai"),
        _ => None,
    }
}

/// Build a map from every configured model name to its (provider_name, AdapterKind).
/// This lets both resolvers route models that genai can't detect from prefixes alone
/// (e.g. Bedrock IDs like "us.anthropic.claude-opus-4-7").
fn build_model_provider_map(config: &Config) -> HashMap<String, (String, AdapterKind)> {
    let mut map = HashMap::new();
    for (provider_name, provider_cfg) in &config.providers {
        let Some(adapter) = adapter_kind_for_provider(provider_name) else {
            continue;
        };
        let entry = (provider_name.clone(), adapter);
        for model in [
            &provider_cfg.default_model,
            provider_cfg.fast_model.as_deref().unwrap_or_default(),
            provider_cfg.smart_model.as_deref().unwrap_or_default(),
        ] {
            if !model.is_empty() {
                map.insert(model.to_string(), entry.clone());
            }
        }
    }
    map
}

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

fn ensure_trailing_slash(mut url: String) -> String {
    if !url.ends_with('/') {
        url.push('/');
    }
    url
}

pub struct ProviderManager {
    client: Client,
    config: Arc<Config>,
    providers: HashMap<String, Provider>,
    #[allow(dead_code)]
    model_tier: ModelTier,
    /// Map from model name -> Bedrock client. Populated for providers whose
    /// `base_url` points at AWS Bedrock (genai cannot speak Bedrock's wire
    /// protocol, so those requests bypass genai entirely).
    bedrock_by_model: HashMap<String, Arc<BedrockClient>>,
}

impl ProviderManager {
    pub fn new(config: Config, model_tier: ModelTier) -> Result<Self> {
        let config = Arc::new(config);
        let model_map = Arc::new(build_model_provider_map(&config));

        // Build custom auth resolver that uses our config.
        // First tries the model→provider lookup table (handles Bedrock-style IDs),
        // then falls back to adapter kind matching.
        let auth_config = Arc::clone(&config);
        let auth_model_map = Arc::clone(&model_map);
        let auth_resolver = AuthResolver::from_resolver_fn(
            move |model_iden: ModelIden| -> Result<Option<AuthData>, genai::resolver::Error> {
                let provider_name = auth_model_map
                    .get::<str>(&model_iden.model_name)
                    .map(|(name, _)| name.as_str())
                    .or_else(|| provider_name_for_adapter(model_iden.adapter_kind.as_str()));

                let Some(provider_name) = provider_name else {
                    return Ok(None);
                };

                if let Some(key) = auth_config.get_api_key(provider_name) {
                    return Ok(Some(AuthData::from_single(key)));
                }

                let fallback_env = match provider_name {
                    "anthropic" => Some("ANTHROPIC_API_KEY"),
                    "openai" => Some("OPENAI_API_KEY"),
                    "google" => Some("GEMINI_API_KEY"),
                    "xai" => Some("XAI_API_KEY"),
                    _ => None,
                };
                if let Some(env) = fallback_env {
                    if let Ok(key) = std::env::var(env) {
                        return Ok(Some(AuthData::from_single(key)));
                    }
                }
                Ok(None)
            },
        );

        let or_config = Arc::clone(&config);
        let or_enabled = config
            .providers
            .get("openrouter")
            .map(|p| p.enabled)
            .unwrap_or(false);
        let target_model_map = Arc::clone(&model_map);

        let target_resolver = ServiceTargetResolver::from_resolver_fn(
            move |service_target: ServiceTarget| -> Result<ServiceTarget, genai::resolver::Error> {
                let ServiceTarget {
                    model,
                    endpoint,
                    auth,
                } = service_target;

                // Route slash-namespaced models through OpenRouter.
                if or_enabled && model.model_name.contains('/') {
                    let or_url = or_config
                        .providers
                        .get("openrouter")
                        .and_then(|p| p.base_url.clone())
                        .unwrap_or_else(|| "https://openrouter.ai/api/v1".to_string());
                    let or_url = ensure_trailing_slash(or_url);
                    let or_endpoint = Endpoint::from_owned(or_url);
                    let or_auth = if let Some(key) = or_config.get_api_key("openrouter") {
                        AuthData::from_single(key)
                    } else {
                        AuthData::from_env("OPENROUTER_API_KEY")
                    };
                    let or_model = ModelIden::new(AdapterKind::OpenAI, model.model_name);
                    return Ok(ServiceTarget {
                        endpoint: or_endpoint,
                        auth: or_auth,
                        model: or_model,
                    });
                }

                // Resolve the provider for this model: first try our config's model
                // table (handles Bedrock-style IDs that genai misroutes), then fall
                // back to adapter-kind matching.
                let (provider_name, correct_kind) = target_model_map
                    .get::<str>(&model.model_name)
                    .map(|(name, kind)| (name.as_str(), Some(*kind)))
                    .unwrap_or_else(|| {
                        let name = provider_name_for_adapter(model.adapter_kind.as_str())
                            .unwrap_or("");
                        (name, None)
                    });

                // Fix the adapter kind if genai guessed wrong
                let kind_was_wrong = correct_kind.is_some_and(|k| k != model.adapter_kind);
                let model = if kind_was_wrong {
                    ModelIden::new(correct_kind.unwrap(), model.model_name)
                } else {
                    model
                };

                // Override endpoint: use [providers.<name>].base_url if set,
                // otherwise if the adapter kind was corrected we must also fix
                // the endpoint (genai resolved the default for the wrong adapter).
                let override_endpoint = or_config
                    .providers
                    .get(provider_name)
                    .and_then(|p| p.base_url.clone())
                    .or_else(|| {
                        if !kind_was_wrong {
                            return None;
                        }
                        Some(
                            match model.adapter_kind {
                                AdapterKind::Anthropic => "https://api.anthropic.com/v1/",
                                AdapterKind::OpenAI => "https://api.openai.com/v1/",
                                AdapterKind::Gemini => {
                                    "https://generativelanguage.googleapis.com/v1beta/"
                                }
                                AdapterKind::Xai => "https://api.x.ai/v1/",
                                _ => return None,
                            }
                            .to_string(),
                        )
                    })
                    .map(|u| Endpoint::from_owned(ensure_trailing_slash(u)));

                // Re-resolve auth when adapter kind was corrected, since the auth
                // resolver ran with the wrong kind earlier
                let auth = if correct_kind.is_some_and(|k| k != AdapterKind::Ollama) {
                    or_config
                        .get_api_key(provider_name)
                        .map(AuthData::from_single)
                        .unwrap_or(auth)
                } else {
                    auth
                };

                Ok(ServiceTarget {
                    model,
                    endpoint: override_endpoint.unwrap_or(endpoint),
                    auth,
                })
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

        // Build Bedrock clients for any provider whose base_url points at
        // AWS Bedrock. Every configured model for that provider gets mapped
        // to the same shared client so query_single() can dispatch by model.
        let mut bedrock_by_model: HashMap<String, Arc<BedrockClient>> = HashMap::new();
        for (name, p) in &config.providers {
            if !p.enabled {
                continue;
            }
            let Some(base_url) = p.base_url.as_deref() else {
                continue;
            };
            if !is_bedrock_base_url(base_url) {
                continue;
            }
            let Some(env_var) = p.api_key_env.as_deref() else {
                eprintln!(
                    "[warn] provider {} has a Bedrock base_url but no api_key_env; skipping",
                    name
                );
                continue;
            };
            let Some(client) = BedrockClient::from_env(base_url, env_var) else {
                eprintln!(
                    "[warn] Bedrock provider {} has no {} set; skipping",
                    name, env_var
                );
                continue;
            };
            let shared = Arc::new(client);
            for model in [
                &p.default_model,
                p.fast_model.as_deref().unwrap_or_default(),
                p.smart_model.as_deref().unwrap_or_default(),
            ] {
                if !model.is_empty() {
                    bedrock_by_model.insert(model.to_string(), Arc::clone(&shared));
                }
            }
        }

        Ok(ProviderManager {
            client,
            config,
            providers,
            model_tier,
            bedrock_by_model,
        })
    }

    /// If this model is served by Bedrock, return the client for it.
    pub fn bedrock_for(&self, model: &str) -> Option<Arc<BedrockClient>> {
        self.bedrock_by_model.get(model).cloned()
    }

    /// Non-streaming Bedrock Converse invocation with retry on transient errors.
    pub async fn exec_bedrock_converse(
        &self,
        client: &BedrockClient,
        model: &str,
        messages: &[ChatMessage],
        system_prompt: Option<&str>,
    ) -> Result<BedrockOutcome> {
        let cfg = self.retry_cfg();
        let mut attempt: u32 = 0;
        loop {
            attempt += 1;
            match client.converse(model, messages, system_prompt).await {
                Ok(r) => return Ok(r),
                Err(err) => {
                    if let Some(delay) = should_retry(&err, cfg, attempt) {
                        eprintln!(
                            "[retry] {} attempt {}/{} after {:?}",
                            model, attempt, cfg.max_attempts, delay
                        );
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                    return Err(err).with_context(|| {
                        format!("Failed to execute Bedrock Converse request for model: {}", model)
                    });
                }
            }
        }
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

    fn retry_cfg(&self) -> &RetryConfig {
        &self.config.retry
    }

    pub async fn exec_chat(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        options: Option<&ChatOptions>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse> {
        let cfg = self.retry_cfg();
        let mut attempt: u32 = 0;
        loop {
            attempt += 1;
            let mut request = ChatRequest::new(messages.clone());
            if let Some(t) = &tools {
                request = request.with_tools(t.clone());
            }
            match self.client.exec_chat(model, request, options).await {
                Ok(r) => return Ok(r),
                Err(e) => {
                    let err = anyhow::Error::from(e);
                    if let Some(delay) = should_retry(&err, cfg, attempt) {
                        eprintln!(
                            "[retry] {} attempt {}/{} after {:?}",
                            model, attempt, cfg.max_attempts, delay
                        );
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                    return Err(err).with_context(|| {
                        format!("Failed to execute chat request for model: {}", model)
                    });
                }
            }
        }
    }

    pub async fn exec_chat_stream(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        options: Option<&ChatOptions>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStreamResponse> {
        let cfg = self.retry_cfg();
        let mut attempt: u32 = 0;
        loop {
            attempt += 1;
            let mut request = ChatRequest::new(messages.clone());
            if let Some(t) = &tools {
                request = request.with_tools(t.clone());
            }
            match self.client.exec_chat_stream(model, request, options).await {
                Ok(r) => return Ok(r),
                Err(e) => {
                    let err = anyhow::Error::from(e);
                    if let Some(delay) = should_retry(&err, cfg, attempt) {
                        eprintln!(
                            "[retry] {} attempt {}/{} after {:?}",
                            model, attempt, cfg.max_attempts, delay
                        );
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                    return Err(err).with_context(|| format!("Model: {}", model));
                }
            }
        }
    }
}

/// Decide whether to retry and how long to wait. Returns `None` on give-up:
/// either the error is non-transient, or we've exhausted `max_attempts`.
fn should_retry(
    err: &anyhow::Error,
    cfg: &RetryConfig,
    attempt: u32,
) -> Option<std::time::Duration> {
    if attempt >= cfg.max_attempts {
        return None;
    }
    match retry::classify(err) {
        Retryable::No => None,
        Retryable::Yes { retry_after } => Some(retry::effective_delay(cfg, attempt, retry_after)),
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
