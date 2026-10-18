use crate::anthropic_oauth;
use crate::bedrock::{is_bedrock_base_url, BedrockClient, BedrockOutcome};
use crate::config::{AuthMode, Config, ModelTier, ProviderConfig, RetryConfig};
use crate::retry::{self, Retryable};
use anyhow::{Context, Result};
use colored::Color;
use futures::{Stream, StreamExt};
use genai::adapter::AdapterKind;
use genai::chat::{
    ChatMessage, ChatOptions, ChatRequest, ChatResponse, ChatStream, ChatStreamEvent, Tool,
};
use genai::resolver::{AuthData, AuthResolver, Endpoint, ServiceTargetResolver};
use genai::{Client, ClientConfig, ModelIden, ServiceTarget, WebConfig};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

/// A provider's streamed response, with any events consumed while checking
/// that the request actually succeeded replayed at the front.
pub type ChatEventStream = Pin<Box<dyn Stream<Item = genai::Result<ChatStreamEvent>> + Send>>;

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
        // genai sends GPT "pro"/"codex" model names to its Responses API
        // adapter; those are still served by the openai provider.
        "OpenAI" | "OpenAIResp" => Some("openai"),
        "Gemini" => Some("google"),
        "Ollama" => Some("ollama"),
        "xAi" => Some("xai"),
        _ => None,
    }
}

/// Provider name that gets the OpenAI-compatible OpenRouter treatment.
const OPENROUTER_PROVIDER: &str = "openrouter";
const OPENROUTER_DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1/";
const OPENROUTER_KEY_ENV: &str = "OPENROUTER_API_KEY";

/// Resolve auth for one provider, honoring `auth_mode`. This is the single
/// place credentials are picked (see `ModelRouter::auth`), so a
/// `subscription`-mode provider can't get silently clobbered back to plain
/// API-key auth by a second resolution pass. The service-target resolver
/// used to re-resolve auth after correcting a misdetected adapter kind; that
/// pass was redundant because the auth resolver already routes by the same
/// model name.
fn resolve_provider_auth(config: &Config, provider_name: &str) -> Option<AuthData> {
    let provider_cfg = config.providers.get(provider_name)?;

    if provider_cfg.auth_mode == AuthMode::Subscription {
        // Only Anthropic has a subscription-OAuth path; other providers
        // ignore this mode rather than misusing Anthropic's token format.
        if provider_name != "anthropic" {
            return None;
        }
        return match anthropic_oauth::subscription_auth_data(provider_cfg.base_url.as_deref()) {
            Ok(auth) => Some(auth),
            Err(err) => {
                crate::diag::warn(format!(
                    "[warn] {} subscription auth unavailable: {}",
                    provider_name, err
                ));
                None
            }
        };
    }

    if let Some(key) = config.get_api_key(provider_name) {
        return Some(AuthData::from_single(key));
    }

    let fallback_env = match provider_name {
        "anthropic" => Some("ANTHROPIC_API_KEY"),
        "openai" => Some("OPENAI_API_KEY"),
        "google" => Some("GEMINI_API_KEY"),
        "xai" => Some("XAI_API_KEY"),
        OPENROUTER_PROVIDER => Some(OPENROUTER_KEY_ENV),
        _ => None,
    };
    fallback_env
        .and_then(|env| std::env::var(env).ok())
        .map(AuthData::from_single)
}

/// The `[providers.<name>]` entry that lists a model, plus the adapter that
/// provider speaks. `adapter` is `None` for provider names we have no fixed
/// adapter for: those keep genai's name-based guess but still get their own
/// base_url and credentials.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelOwner {
    provider: String,
    adapter: Option<AdapterKind>,
}

/// Build a map from every configured model name to the provider that lists it.
/// This lets both resolvers route models that genai can't detect from prefixes
/// alone (e.g. Bedrock IDs like "us.anthropic.claude-opus-4-7"), and stops a
/// model configured under one provider from being claimed by another (e.g. a
/// local vLLM's "Qwen/Qwen2.5-7B-Instruct" under `openai` must not go to
/// OpenRouter just because it contains a '/').
///
/// If two providers list the same model name, the winner is deterministic:
/// enabled before disabled, direct providers before OpenRouter (the catch-all
/// aggregator), then by provider name. `config.providers` is a HashMap, so
/// without the sort the winner would change from run to run.
fn build_model_provider_map(config: &Config) -> HashMap<String, ModelOwner> {
    let mut providers: Vec<(&String, &ProviderConfig)> = config.providers.iter().collect();
    providers
        .sort_by_key(|&(name, cfg)| (!cfg.enabled, name == OPENROUTER_PROVIDER, name.as_str()));

    let mut map = HashMap::new();
    for (provider_name, provider_cfg) in providers {
        let adapter = if provider_name == OPENROUTER_PROVIDER {
            Some(AdapterKind::OpenAI)
        } else {
            adapter_kind_for_provider(provider_name)
        };
        for model in [
            &provider_cfg.default_model,
            provider_cfg.fast_model.as_deref().unwrap_or_default(),
            provider_cfg.smart_model.as_deref().unwrap_or_default(),
        ] {
            if !model.is_empty() {
                map.entry(model.to_string()).or_insert_with(|| ModelOwner {
                    provider: provider_name.clone(),
                    adapter,
                });
            }
        }
    }
    map
}

/// The conventional key variable for a built-in provider. A provider section
/// without `api_key_env` still authenticates with this at request time
/// (genai falls back to it), so availability checks must accept it too —
/// otherwise such a provider is silently dropped even though it would work.
fn standard_key_env(provider_name: &str) -> Option<&'static str> {
    match provider_name {
        "anthropic" => Some("ANTHROPIC_API_KEY"),
        "openai" => Some("OPENAI_API_KEY"),
        "google" => Some("GEMINI_API_KEY"),
        "xai" => Some("XAI_API_KEY"),
        "openrouter" => Some("OPENROUTER_API_KEY"),
        _ => None,
    }
}

fn env_is_set(var: &str) -> bool {
    std::env::var(var).is_ok_and(|v| !v.trim().is_empty())
}

/// Where the list of providers for a query came from. It changes how
/// skipped providers are reported, not which ones are skipped.
#[derive(Debug, Clone, Copy)]
pub enum ProviderRequest<'a> {
    /// Named explicitly with `-p`.
    Explicit(&'a [String]),
    /// From the active mode's config (`single_provider` / `multi_providers`).
    Configured(&'a [String]),
    /// `-a/--all`: every configured provider.
    All,
}

/// Why a requested provider was left out of a query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    /// The name matches no provider or alias in the config.
    Unknown,
    /// `enabled = false` in the provider's config section.
    Disabled,
    /// No API key is available.
    NoCredentials { env_var: Option<String> },
    /// `auth_mode = "subscription"` but no usable login token was found
    /// (and no API key to fall back on).
    NoSubscriptionToken,
}

impl std::fmt::Display for SkipReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SkipReason::Unknown => write!(f, "not a configured provider or alias"),
            SkipReason::Disabled => write!(
                f,
                "disabled in config (set `enabled = true`, or toggle it in `ai config`)"
            ),
            SkipReason::NoCredentials { env_var: Some(var) } => {
                write!(f, "no API key ({} is not set)", var)
            }
            SkipReason::NoCredentials { env_var: None } => {
                write!(f, "no API key configured (set api_key_env in the config)")
            }
            SkipReason::NoSubscriptionToken => write!(
                f,
                "auth_mode is \"subscription\" but no valid login token was found"
            ),
        }
    }
}

pub struct ProviderSelection {
    pub providers: Vec<Provider>,
    /// `(name as requested, reason)` for each provider that was left out.
    pub skipped: Vec<(String, SkipReason)>,
}

/// genai's default base URL for each adapter a `[providers.*]` entry can
/// resolve to. Needed when we override genai's adapter guess, because genai
/// has already picked the default endpoint for the adapter it guessed.
/// (genai's own table, `AdapterDispatcher::default_endpoint`, is
/// crate-private; `default_base_urls_match_genai` keeps these in sync.)
fn default_base_url(kind: AdapterKind) -> Option<&'static str> {
    Some(match kind {
        AdapterKind::Anthropic => "https://api.anthropic.com/v1/",
        AdapterKind::OpenAI | AdapterKind::OpenAIResp => "https://api.openai.com/v1/",
        AdapterKind::Gemini => "https://generativelanguage.googleapis.com/v1beta/",
        AdapterKind::Xai => "https://api.x.ai/v1/",
        AdapterKind::Ollama => "http://localhost:11434/v1/",
        _ => return None,
    })
}

/// Turn a configured `base_url` into the base genai expects. genai joins
/// request paths (`chat/completions`, `messages`, ...) onto it, so it must end
/// in '/'. The Ollama adapter speaks Ollama's OpenAI-compatible API, which
/// lives under `/v1/`: a bare host like the shipped default
/// `http://localhost:11434` would otherwise become
/// `http://localhost:11434/chat/completions`, which Ollama answers with 404.
fn normalize_base_url(url: &str, adapter_kind: AdapterKind) -> String {
    if adapter_kind == AdapterKind::Ollama {
        if let Ok(mut parsed) = reqwest::Url::parse(url) {
            if matches!(parsed.path(), "" | "/") {
                parsed.set_path("/v1/");
                return parsed.to_string();
            }
        }
    }
    ensure_trailing_slash(url.to_string())
}

/// Where one model's requests go. Plain data, so the routing rules can be
/// unit tested without a network or a genai client.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RouteDecision {
    /// `[providers.<name>]` entry whose credentials (and base_url) serve the
    /// model. `None` leaves auth to genai's per-adapter default.
    provider_name: Option<String>,
    /// genai adapter, i.e. the wire protocol to speak.
    adapter_kind: AdapterKind,
    /// Base URL to use instead of the default genai picked for the adapter it
    /// guessed. Always ends in '/'.
    endpoint_override: Option<String>,
    /// Sent to OpenRouter's OpenAI-compatible API.
    via_openrouter: bool,
}

/// Model name -> provider routing shared by the auth and service-target
/// resolvers. genai calls both with the same model name and the same
/// name-based adapter guess, so both reach the same decision, and the auth
/// resolver (which runs first) already picks the credentials of the provider
/// the request will actually be sent to.
struct ModelRouter {
    config: Arc<Config>,
    owners: HashMap<String, ModelOwner>,
    openrouter_enabled: bool,
}

impl ModelRouter {
    fn new(config: Arc<Config>) -> Self {
        let owners = build_model_provider_map(&config);
        let openrouter_enabled = config
            .providers
            .get(OPENROUTER_PROVIDER)
            .is_some_and(|p| p.enabled);
        Self {
            config,
            owners,
            openrouter_enabled,
        }
    }

    /// Decide where `model_name` goes. `genai_guess` is the adapter genai
    /// inferred from the name alone (`AdapterKind::from_model`).
    fn route(&self, model_name: &str, genai_guess: AdapterKind) -> RouteDecision {
        // 1. A configured model goes to the provider that lists it. This runs
        //    before the OpenRouter slash heuristic below so a local vLLM's
        //    "Qwen/Qwen2.5-7B-Instruct" or an Ollama "hf.co/..." tag is never
        //    sent to a third party the user didn't pick.
        if let Some(owner) = self.owners.get(model_name) {
            if owner.provider == OPENROUTER_PROVIDER {
                return self.openrouter_route();
            }
            let adapter_kind = match owner.adapter {
                // genai maps GPT "pro"/"codex" names to the Responses API,
                // and /chat/completions rejects those models. It's still the
                // openai provider, so keep genai's more specific adapter.
                Some(AdapterKind::OpenAI) if genai_guess == AdapterKind::OpenAIResp => {
                    AdapterKind::OpenAIResp
                }
                Some(kind) => kind,
                // No fixed adapter for this provider name: trust genai.
                None => genai_guess,
            };
            return self.provider_route(&owner.provider, adapter_kind, genai_guess);
        }

        // 2. An unconfigured "vendor/model" name goes to OpenRouter, if the
        //    user enabled it. genai's own "adapter::model" namespaces are an
        //    explicit adapter choice, so those are left alone.
        if self.openrouter_enabled && model_name.contains('/') && !model_name.contains("::") {
            return self.openrouter_route();
        }

        // 3. Otherwise trust genai's guess, with that adapter's provider settings.
        match provider_name_for_adapter(genai_guess.as_str()) {
            Some(provider) => self.provider_route(provider, genai_guess, genai_guess),
            None => RouteDecision {
                provider_name: None,
                adapter_kind: genai_guess,
                endpoint_override: None,
                via_openrouter: false,
            },
        }
    }

    fn provider_route(
        &self,
        provider: &str,
        adapter_kind: AdapterKind,
        genai_guess: AdapterKind,
    ) -> RouteDecision {
        let base_url = self
            .config
            .providers
            .get(provider)
            .and_then(|p| p.base_url.as_deref());
        let endpoint_override = match base_url {
            Some(url) => Some(normalize_base_url(url, adapter_kind)),
            // genai picked the default endpoint of the adapter it guessed; if
            // we changed the adapter, the endpoint has to follow.
            None if adapter_kind != genai_guess => default_base_url(adapter_kind).map(String::from),
            None => None,
        };
        RouteDecision {
            provider_name: Some(provider.to_string()),
            adapter_kind,
            endpoint_override,
            via_openrouter: false,
        }
    }

    fn openrouter_route(&self) -> RouteDecision {
        let base_url = self
            .config
            .providers
            .get(OPENROUTER_PROVIDER)
            .and_then(|p| p.base_url.as_deref())
            .unwrap_or(OPENROUTER_DEFAULT_BASE_URL);
        RouteDecision {
            provider_name: Some(OPENROUTER_PROVIDER.to_string()),
            adapter_kind: AdapterKind::OpenAI,
            endpoint_override: Some(ensure_trailing_slash(base_url.to_string())),
            via_openrouter: true,
        }
    }

    /// Credentials for a routed request. `None` (no owning provider) lets
    /// genai use its default for the adapter, which in that case is also the
    /// adapter we route to.
    fn auth(&self, route: &RouteDecision) -> Option<AuthData> {
        let provider = route.provider_name.as_deref()?;
        Some(resolve_provider_auth(&self.config, provider).unwrap_or_else(|| default_auth(route)))
    }
}

/// Fallback credentials when the owning provider has none configured: the
/// default of the adapter we route to, never of the adapter genai guessed
/// from the name. Returning `None` here would make genai use the guess's
/// default, so an Ollama model named like another vendor's ("glm4",
/// "command-r") would demand ZAI_API_KEY / COHERE_API_KEY, or send that
/// vendor's key to the local server.
fn default_auth(route: &RouteDecision) -> AuthData {
    if route.via_openrouter {
        return AuthData::from_env(OPENROUTER_KEY_ENV);
    }
    match route.adapter_kind.default_key_env_name() {
        Some(env) => AuthData::from_env(env),
        // Ollama needs no key; this placeholder is genai's own default.
        None => AuthData::from_single("ollama"),
    }
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

/// HTTP timeouts for every genai request (genai sets none by default, so a
/// provider that accepts the connection and then stalls hangs forever —
/// in non-streaming multi-provider mode that blocks every other answer too).
/// There's deliberately no overall request timeout: a long streamed answer
/// can legitimately run for minutes. The read timeout is per read (idle),
/// and generous because a non-streaming call to a slow reasoning model
/// sends nothing until it's done.
fn provider_web_config() -> WebConfig {
    WebConfig {
        connect_timeout: Some(Duration::from_secs(15)),
        read_timeout: Some(Duration::from_secs(600)),
        ..WebConfig::default()
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
        let router = Arc::new(ModelRouter::new(Arc::clone(&config)));

        // genai runs the auth resolver first, with the adapter it guessed from
        // the model name, then the service-target resolver. Both route through
        // the same ModelRouter (model→provider table first, so Bedrock-style
        // IDs and slash-named local models land on the provider that lists
        // them), so auth is resolved once per request, for the right provider.
        let auth_router = Arc::clone(&router);
        let auth_resolver = AuthResolver::from_resolver_fn(
            move |model_iden: ModelIden| -> Result<Option<AuthData>, genai::resolver::Error> {
                let route = auth_router.route(&model_iden.model_name, model_iden.adapter_kind);
                Ok(auth_router.auth(&route))
            },
        );

        let target_router = Arc::clone(&router);
        let target_resolver = ServiceTargetResolver::from_resolver_fn(
            move |service_target: ServiceTarget| -> Result<ServiceTarget, genai::resolver::Error> {
                let ServiceTarget {
                    model,
                    endpoint,
                    auth,
                } = service_target;

                let route = target_router.route(&model.model_name, model.adapter_kind);

                // Fix the adapter kind if genai guessed wrong.
                let model = if route.adapter_kind == model.adapter_kind {
                    model
                } else {
                    ModelIden::new(route.adapter_kind, model.model_name)
                };
                let endpoint = route
                    .endpoint_override
                    .map(Endpoint::from_owned)
                    .unwrap_or(endpoint);

                // `auth` already came from the auth resolver's identical route.
                Ok(ServiceTarget {
                    model,
                    endpoint,
                    auth,
                })
            },
        );

        let client_config = ClientConfig::default().with_web_config(provider_web_config());
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
        for p in config.providers.values() {
            if !p.enabled {
                continue;
            }
            let Some(base_url) = p.base_url.as_deref() else {
                continue;
            };
            if !is_bedrock_base_url(base_url) {
                continue;
            }
            // No warning here: this runs on every invocation, even when the
            // Bedrock provider isn't being queried. If it *is* requested,
            // select_providers() reports the missing token.
            let Some(env_var) = p.api_key_env.as_deref() else {
                continue;
            };
            let Some(client) = BedrockClient::from_env(base_url, env_var) else {
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
        tools: Option<&[Tool]>,
    ) -> Result<BedrockOutcome> {
        let cfg = self.retry_cfg();
        let mut attempt: u32 = 0;
        loop {
            attempt += 1;
            match client.converse(model, messages, system_prompt, tools).await {
                Ok(r) => return Ok(r),
                Err(err) => {
                    if let Some(delay) = should_retry(&err, cfg, attempt) {
                        crate::diag::warn(format!(
                            "[retry] {} attempt {}/{} after {:?}",
                            model, attempt, cfg.max_attempts, delay
                        ));
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                    return Err(err).with_context(|| {
                        format!(
                            "Failed to execute Bedrock Converse request for model: {}",
                            model
                        )
                    });
                }
            }
        }
    }

    pub fn resolve_alias(&self, alias: &str) -> Option<String> {
        self.config.resolve_provider_alias(alias)
    }

    /// Resolve which providers a query should go to, and why any requested
    /// ones were left out. Disabled providers and providers without
    /// credentials are skipped; the caller decides how loudly to report
    /// that (silently for the configured multi list, with a note for an
    /// explicit `-p`, and in full when nothing is left).
    pub fn select_providers(&self, request: ProviderRequest<'_>) -> ProviderSelection {
        let mut skipped: Vec<(String, SkipReason)> = Vec::new();

        // (what the user/config asked for, canonical provider name)
        let candidates: Vec<(String, String)> = match request {
            ProviderRequest::Explicit(names) | ProviderRequest::Configured(names) => names
                .iter()
                .filter_map(|alias| match self.resolve_alias(alias) {
                    Some(name) => Some((alias.clone(), name)),
                    None => {
                        skipped.push((alias.clone(), SkipReason::Unknown));
                        None
                    }
                })
                .collect(),
            ProviderRequest::All => self
                .all_providers_in_display_order()
                .into_iter()
                .map(|name| (name.clone(), name))
                .collect(),
        };

        // Dedupe by canonical name while preserving first-seen order. Users
        // commonly list multiple aliases for the same provider (e.g. both
        // "gpt" and "chatgpt"); without this, the TUI would render duplicate
        // panels and fan out the same query twice.
        let mut seen = std::collections::HashSet::new();
        let mut providers = Vec::new();
        for (requested_as, name) in candidates {
            if !seen.insert(name.clone()) {
                continue;
            }
            let (Some(provider), Some(cfg)) =
                (self.providers.get(&name), self.config.providers.get(&name))
            else {
                skipped.push((requested_as, SkipReason::Unknown));
                continue;
            };
            if !cfg.enabled {
                skipped.push((requested_as, SkipReason::Disabled));
                continue;
            }
            // Only providers that can actually authenticate: ollama needs
            // nothing, subscription-mode needs a live Claude Code OAuth
            // token, everything else needs an API key.
            let available = name == "ollama"
                || (cfg.auth_mode == AuthMode::Subscription && anthropic_oauth::is_available())
                || self.config.get_api_key(&name).is_some()
                // Only when no api_key_env is configured: a section pointing
                // at, say, AWS_BEARER_TOKEN_BEDROCK must not be "rescued" by
                // ANTHROPIC_API_KEY and sent to the Bedrock URL with it.
                || (cfg.api_key_env.is_none()
                    && standard_key_env(&name).is_some_and(env_is_set));
            if !available {
                let reason = if cfg.auth_mode == AuthMode::Subscription {
                    SkipReason::NoSubscriptionToken
                } else {
                    SkipReason::NoCredentials {
                        env_var: cfg
                            .api_key_env
                            .clone()
                            .or_else(|| standard_key_env(&name).map(String::from)),
                    }
                };
                skipped.push((requested_as, reason));
                continue;
            }
            providers.push(provider.clone());
        }

        ProviderSelection { providers, skipped }
    }

    /// Every configured provider name for `--all`: the `[default].providers`
    /// order first (so panel order stays familiar), then the rest sorted by
    /// name. HashMap iteration order would otherwise reshuffle panels on
    /// every run.
    fn all_providers_in_display_order(&self) -> Vec<String> {
        let mut ordered: Vec<String> = self
            .config
            .default
            .providers
            .iter()
            .filter_map(|alias| self.resolve_alias(alias))
            .collect();
        let mut rest: Vec<String> = self
            .config
            .providers
            .keys()
            .filter(|name| !ordered.contains(name))
            .cloned()
            .collect();
        rest.sort();
        ordered.extend(rest);
        ordered
    }

    /// Names users can pass to `-p`: each provider's aliases (or its name
    /// when it has none), sorted.
    pub fn known_provider_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .config
            .providers
            .iter()
            .flat_map(|(name, cfg)| {
                if cfg.aliases.is_empty() {
                    vec![name.clone()]
                } else {
                    cfg.aliases.clone()
                }
            })
            .collect();
        names.sort();
        names.dedup();
        names
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
                        crate::diag::warn(format!(
                            "[retry] {} attempt {}/{} after {:?}",
                            model, attempt, cfg.max_attempts, delay
                        ));
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

    /// Open a streaming chat request, retrying transient failures.
    ///
    /// genai sends the HTTP request lazily, on the first poll of the stream,
    /// so a 429/5xx or a connect error doesn't come back from
    /// `Client::exec_chat_stream` — it arrives as the first error *item* of
    /// the stream. We therefore poll until the provider has actually
    /// answered (see [`open_stream`]) and retry from here, before any text
    /// has reached the caller.
    pub async fn exec_chat_stream(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        options: Option<&ChatOptions>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatEventStream> {
        let cfg = self.retry_cfg();
        let mut attempt: u32 = 0;
        loop {
            attempt += 1;
            let mut request = ChatRequest::new(messages.clone());
            if let Some(t) = &tools {
                request = request.with_tools(t.clone());
            }
            let err = match self.client.exec_chat_stream(model, request, options).await {
                Ok(response) => match open_stream(response.stream).await {
                    Ok(stream) => return Ok(stream),
                    Err(e) => anyhow::Error::from(e),
                },
                Err(e) => anyhow::Error::from(e),
            };
            if let Some(delay) = should_retry(&err, cfg, attempt) {
                crate::diag::warn(format!(
                    "[retry] {} attempt {}/{} after {:?}",
                    model, attempt, cfg.max_attempts, delay
                ));
                tokio::time::sleep(delay).await;
                continue;
            }
            return Err(err).with_context(|| format!("Model: {}", model));
        }
    }

    /// Backoff before retry `attempt` of a failed request, or `None` to give
    /// up (non-transient error, or attempts exhausted).
    pub fn retry_delay(&self, err: &anyhow::Error, attempt: u32) -> Option<Duration> {
        should_retry(err, self.retry_cfg(), attempt)
    }

    pub fn max_attempts(&self) -> u32 {
        self.retry_cfg().max_attempts
    }
}

/// Poll a freshly-opened genai stream until the provider has really
/// responded. genai yields a synthetic `Start` before the request is even
/// sent, so we buffer leading `Start`s; if the next item is an error it's
/// returned (so the caller can retry), otherwise the buffered events are
/// replayed ahead of the rest of the stream.
async fn open_stream(mut stream: ChatStream) -> genai::Result<ChatEventStream> {
    let mut consumed = Vec::new();
    loop {
        match stream.next().await {
            // Events the query layer doesn't show (reasoning, tool-call and
            // thought-signature chunks) keep the retry window open: an error
            // after them still hasn't put anything on the user's screen.
            Some(Ok(
                event @ (ChatStreamEvent::Start
                | ChatStreamEvent::ReasoningChunk(_)
                | ChatStreamEvent::ToolCallChunk(_)
                | ChatStreamEvent::ThoughtSignatureChunk(_)),
            )) => consumed.push(Ok(event)),
            Some(Err(e)) => return Err(e),
            Some(Ok(event)) => {
                consumed.push(Ok(event));
                break;
            }
            None => break,
        }
    }
    Ok(Box::pin(futures::stream::iter(consumed).chain(stream)))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProviderConfig;

    fn fake_provider_config(aliases: Vec<&str>, color: &str) -> ProviderConfig {
        ProviderConfig {
            enabled: true,
            api_key_env: None,
            api_key: Some("test-key".to_string()),
            default_model: "test".to_string(),
            fast_model: Some("test".to_string()),
            smart_model: Some("test".to_string()),
            aliases: aliases.into_iter().map(String::from).collect(),
            color: color.to_string(),
            base_url: None,
            auth_mode: AuthMode::ApiKey,
        }
    }

    #[test]
    fn duplicate_aliases_for_same_provider_are_deduped() {
        // Real-world bug: a config with `multi_providers = ["gpt", "chatgpt"]`
        // would render two ChatGPT panels because both aliases resolve to
        // "openai" and there was no dedup pass.
        let mut config = Config::default();
        config.providers.clear();
        config.providers.insert(
            "openai".to_string(),
            fake_provider_config(vec!["chatgpt", "gpt"], "green"),
        );
        let manager = ProviderManager::new(config, ModelTier::Fast).unwrap();
        let requested = vec!["gpt".to_string(), "chatgpt".to_string()];
        let providers = manager
            .select_providers(ProviderRequest::Explicit(&requested))
            .providers;
        assert_eq!(providers.len(), 1);
    }

    #[test]
    fn dedup_preserves_first_seen_order() {
        // Stability matters: panel order in the TUI should follow the
        // user's config order, not be reshuffled by the dedup pass.
        let mut config = Config::default();
        config.providers.clear();
        config.providers.insert(
            "openai".to_string(),
            fake_provider_config(vec!["gpt"], "green"),
        );
        config.providers.insert(
            "anthropic".to_string(),
            fake_provider_config(vec!["claude"], "magenta"),
        );
        let manager = ProviderManager::new(config, ModelTier::Fast).unwrap();
        let requested = vec![
            "claude".to_string(),
            "gpt".to_string(),
            "claude".to_string(),
            "gpt".to_string(),
        ];
        let result = manager
            .select_providers(ProviderRequest::Explicit(&requested))
            .providers;
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].display_name, "CLAUDE");
        assert_eq!(result[1].display_name, "GPT");
    }

    fn two_provider_manager(openai_enabled: bool) -> ProviderManager {
        let mut config = Config::default();
        config.providers.clear();
        let mut openai = fake_provider_config(vec!["gpt"], "green");
        openai.enabled = openai_enabled;
        config.providers.insert("openai".to_string(), openai);
        config.providers.insert(
            "anthropic".to_string(),
            fake_provider_config(vec!["claude"], "magenta"),
        );
        ProviderManager::new(config, ModelTier::Fast).unwrap()
    }

    #[test]
    fn disabled_providers_are_skipped_with_a_reason() {
        // `enabled = false` used to be ignored: a disabled provider with a
        // key set was still queried.
        let manager = two_provider_manager(false);
        let requested = vec!["claude".to_string(), "gpt".to_string()];
        let selection = manager.select_providers(ProviderRequest::Configured(&requested));
        assert_eq!(selection.providers.len(), 1);
        assert_eq!(selection.providers[0].display_name, "CLAUDE");
        assert_eq!(
            selection.skipped,
            vec![("gpt".to_string(), SkipReason::Disabled)]
        );
    }

    #[test]
    fn unknown_aliases_are_reported_not_silently_dropped() {
        let manager = two_provider_manager(true);
        let requested = vec!["claud".to_string(), "gpt".to_string()];
        let selection = manager.select_providers(ProviderRequest::Explicit(&requested));
        assert_eq!(selection.providers.len(), 1);
        assert_eq!(
            selection.skipped,
            vec![("claud".to_string(), SkipReason::Unknown)]
        );
    }

    #[test]
    fn missing_credentials_name_the_env_var() {
        let mut config = Config::default();
        config.providers.clear();
        let mut cfg = fake_provider_config(vec!["grok"], "blue");
        cfg.api_key = None;
        cfg.api_key_env = Some("AI_CLI_TEST_DEFINITELY_UNSET_KEY".to_string());
        config.providers.insert("xai".to_string(), cfg);
        let manager = ProviderManager::new(config, ModelTier::Fast).unwrap();
        let requested = vec!["grok".to_string()];
        let selection = manager.select_providers(ProviderRequest::Explicit(&requested));
        assert!(selection.providers.is_empty());
        let (_, reason) = &selection.skipped[0];
        assert!(reason
            .to_string()
            .contains("AI_CLI_TEST_DEFINITELY_UNSET_KEY"));
    }

    #[test]
    fn all_means_every_enabled_provider_in_a_stable_order() {
        // `-a` used to query only `[default].providers`, leaving out
        // openrouter/ollama/custom providers.
        let mut config = Config::default();
        config.providers.clear();
        config.default.providers = vec!["claude".to_string()];
        for (name, alias) in [("zeta", "z"), ("anthropic", "claude"), ("alpha", "a")] {
            config
                .providers
                .insert(name.to_string(), fake_provider_config(vec![alias], "white"));
        }
        let mut off = fake_provider_config(vec!["off"], "white");
        off.enabled = false;
        config.providers.insert("disabled".to_string(), off);
        let manager = ProviderManager::new(config, ModelTier::Fast).unwrap();
        let names: Vec<String> = manager
            .select_providers(ProviderRequest::All)
            .providers
            .into_iter()
            .map(|p| p.display_name)
            .collect();
        // [default].providers first, then the rest alphabetically.
        assert_eq!(names, ["CLAUDE", "A", "Z"]);
    }

    // -- Model routing ------------------------------------------------------

    /// A provider entry with an inline key (so auth never depends on the
    /// environment) listing `fast` and optionally `smart`.
    fn routed_provider(fast: &str, smart: Option<&str>, base_url: Option<&str>) -> ProviderConfig {
        ProviderConfig {
            enabled: true,
            api_key_env: None,
            api_key: Some("test-key".to_string()),
            default_model: fast.to_string(),
            fast_model: Some(fast.to_string()),
            smart_model: smart.map(String::from),
            aliases: Vec::new(),
            color: "white".to_string(),
            base_url: base_url.map(String::from),
            auth_mode: AuthMode::ApiKey,
        }
    }

    fn config_with(providers: Vec<(&str, ProviderConfig)>) -> Config {
        let mut config = Config::default();
        config.providers.clear();
        for (name, provider) in providers {
            config.providers.insert(name.to_string(), provider);
        }
        config
    }

    fn router_for(providers: Vec<(&str, ProviderConfig)>) -> ModelRouter {
        ModelRouter::new(Arc::new(config_with(providers)))
    }

    /// Route `model` the way the resolvers do, with genai's real name-based guess.
    fn route(router: &ModelRouter, model: &str) -> RouteDecision {
        let guess = AdapterKind::from_model(model).unwrap();
        router.route(model, guess)
    }

    fn openrouter() -> ProviderConfig {
        routed_provider(
            "google/gemini-3.1-flash-lite-preview",
            Some("anthropic/claude-opus-4-8"),
            Some("https://openrouter.ai/api/v1"),
        )
    }

    fn assert_key(auth: Option<AuthData>, expected: &str) {
        match auth {
            Some(AuthData::Key(key)) => assert_eq!(key, expected),
            other => panic!("expected AuthData::Key, got {:?}", other),
        }
    }

    #[test]
    fn slash_model_configured_under_openai_stays_on_openai() {
        // Real-world bug: a local vLLM configured as the openai provider was
        // sent to openrouter.ai (a third party) because the model contains '/'.
        let router = router_for(vec![
            (
                "openai",
                routed_provider(
                    "Qwen/Qwen2.5-7B-Instruct",
                    None,
                    Some("http://localhost:8000/v1/"),
                ),
            ),
            ("openrouter", openrouter()),
        ]);
        let r = route(&router, "Qwen/Qwen2.5-7B-Instruct");
        assert_eq!(r.provider_name.as_deref(), Some("openai"));
        assert_eq!(r.adapter_kind, AdapterKind::OpenAI);
        assert_eq!(
            r.endpoint_override.as_deref(),
            Some("http://localhost:8000/v1/")
        );
        assert!(!r.via_openrouter);
    }

    #[test]
    fn slash_model_configured_under_ollama_stays_on_ollama() {
        let router = router_for(vec![
            (
                "ollama",
                routed_provider(
                    "huihui_ai/qwen3:8b",
                    Some("hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"),
                    Some("http://localhost:11434"),
                ),
            ),
            ("openrouter", openrouter()),
        ]);
        for model in [
            "huihui_ai/qwen3:8b",
            "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",
        ] {
            let r = route(&router, model);
            assert_eq!(r.provider_name.as_deref(), Some("ollama"), "{model}");
            assert_eq!(r.adapter_kind, AdapterKind::Ollama, "{model}");
            assert_eq!(
                r.endpoint_override.as_deref(),
                Some("http://localhost:11434/v1/"),
                "{model}"
            );
            assert!(!r.via_openrouter, "{model}");
        }
    }

    #[test]
    fn unknown_slash_model_goes_to_openrouter_only_when_enabled() {
        let model = "meta-llama/llama-3.3-70b-instruct";

        let router = router_for(vec![("openrouter", openrouter())]);
        let r = route(&router, model);
        assert!(r.via_openrouter);
        assert_eq!(r.provider_name.as_deref(), Some("openrouter"));
        assert_eq!(r.adapter_kind, AdapterKind::OpenAI);
        assert_eq!(
            r.endpoint_override.as_deref(),
            Some("https://openrouter.ai/api/v1/")
        );
        assert_key(router.auth(&r), "test-key");

        let mut disabled = openrouter();
        disabled.enabled = false;
        let router = router_for(vec![("openrouter", disabled)]);
        let r = route(&router, model);
        assert!(!r.via_openrouter);
        assert_ne!(r.provider_name.as_deref(), Some("openrouter"));
    }

    #[test]
    fn models_configured_under_openrouter_route_there_even_when_genai_guesses_otherwise() {
        // genai maps "qwen/qwen3-32b" to its Groq adapter (it's in Groq's
        // model list); configured under openrouter, it must go to OpenRouter.
        let mut or = openrouter();
        or.smart_model = Some("qwen/qwen3-32b".to_string());
        or.enabled = false; // enabled only gates the unknown-model fallback
        let router = router_for(vec![("openrouter", or)]);
        assert_eq!(
            AdapterKind::from_model("qwen/qwen3-32b").unwrap(),
            AdapterKind::Groq
        );
        let r = route(&router, "qwen/qwen3-32b");
        assert!(r.via_openrouter);
        assert_eq!(r.adapter_kind, AdapterKind::OpenAI);
    }

    #[test]
    fn genai_namespaced_models_are_not_sent_to_openrouter() {
        // "adapter::model" is genai's explicit adapter pin; the '/' in the
        // model part must not divert it to OpenRouter.
        let router = router_for(vec![("openrouter", openrouter())]);
        let r = route(&router, "together::meta-llama/Llama-3-8b-chat-hf");
        assert!(!r.via_openrouter);
        assert_eq!(r.adapter_kind, AdapterKind::Together);
    }

    #[test]
    fn bedrock_style_ids_route_to_their_provider() {
        let bedrock = "https://bedrock-runtime.us-east-1.amazonaws.com/v1/";
        let arn =
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-opus-4-7";
        let router = router_for(vec![
            (
                "anthropic",
                routed_provider("us.anthropic.claude-opus-4-7", Some(arn), Some(bedrock)),
            ),
            ("openrouter", openrouter()),
        ]);
        for model in ["us.anthropic.claude-opus-4-7", arn] {
            let r = route(&router, model);
            assert_eq!(r.provider_name.as_deref(), Some("anthropic"), "{model}");
            assert_eq!(r.adapter_kind, AdapterKind::Anthropic, "{model}");
            assert_eq!(r.endpoint_override.as_deref(), Some(bedrock), "{model}");
            assert!(!r.via_openrouter, "{model}");
        }
    }

    #[test]
    fn ollama_base_url_gets_openai_compat_path() {
        let ollama = AdapterKind::Ollama;
        assert_eq!(
            normalize_base_url("http://localhost:11434", ollama),
            "http://localhost:11434/v1/"
        );
        assert_eq!(
            normalize_base_url("http://localhost:11434/", ollama),
            "http://localhost:11434/v1/"
        );
        assert_eq!(
            normalize_base_url("http://localhost:11434/v1", ollama),
            "http://localhost:11434/v1/"
        );
        assert_eq!(
            normalize_base_url("http://gpu-box:11434/v1/", ollama),
            "http://gpu-box:11434/v1/"
        );
        // An explicit path is the user's choice; only the slash is added.
        assert_eq!(
            normalize_base_url("https://proxy.example/ollama", ollama),
            "https://proxy.example/ollama/"
        );
        // Other adapters never get a path appended.
        assert_eq!(
            normalize_base_url("http://localhost:8000", AdapterKind::OpenAI),
            "http://localhost:8000/"
        );
    }

    #[test]
    fn ollama_models_genai_misdetects_get_ollama_endpoint_and_auth() {
        // genai guesses Zai for "glm4:*" and Cohere for "command-r*". Without
        // a base_url, the endpoint used to stay on api.z.ai / api.cohere.com,
        // and auth fell back to ZAI_API_KEY / COHERE_API_KEY.
        let mut ollama = routed_provider("glm4:9b", Some("command-r"), None);
        ollama.api_key = None;
        let router = router_for(vec![("ollama", ollama)]);
        for model in ["glm4:9b", "command-r"] {
            assert_ne!(
                AdapterKind::from_model(model).unwrap(),
                AdapterKind::Ollama,
                "{model}"
            );
            let r = route(&router, model);
            assert_eq!(r.provider_name.as_deref(), Some("ollama"), "{model}");
            assert_eq!(r.adapter_kind, AdapterKind::Ollama, "{model}");
            assert_eq!(
                r.endpoint_override.as_deref(),
                Some("http://localhost:11434/v1/"),
                "{model}"
            );
            assert_key(router.auth(&r), "ollama");
        }
    }

    #[test]
    fn gpt_pro_and_codex_models_keep_the_responses_api() {
        let router = router_for(vec![(
            "openai",
            routed_provider("gpt-5-nano", Some("gpt-5-pro"), None),
        )]);
        let r = route(&router, "gpt-5-pro");
        assert_eq!(r.adapter_kind, AdapterKind::OpenAIResp);
        assert_eq!(r.provider_name.as_deref(), Some("openai"));
        // genai's default endpoint for OpenAIResp is already OpenAI's.
        assert_eq!(r.endpoint_override, None);

        let r = route(&router, "gpt-5-nano");
        assert_eq!(r.adapter_kind, AdapterKind::OpenAI);

        // Unconfigured Responses-API models still use the openai provider's settings.
        let r = route(&router, "gpt-5.1-codex");
        assert_eq!(r.adapter_kind, AdapterKind::OpenAIResp);
        assert_eq!(r.provider_name.as_deref(), Some("openai"));
        assert_eq!(provider_name_for_adapter("OpenAIResp"), Some("openai"));
    }

    #[test]
    fn duplicate_model_names_resolve_deterministically() {
        let shared = "acme/shared-model";

        // Direct provider beats OpenRouter when both are enabled.
        let router = router_for(vec![
            (
                "openai",
                routed_provider(shared, None, Some("http://vllm:8000/v1")),
            ),
            ("openrouter", routed_provider(shared, None, None)),
        ]);
        assert_eq!(
            route(&router, shared).provider_name.as_deref(),
            Some("openai")
        );

        // Enabled beats disabled.
        let mut disabled = routed_provider(shared, None, Some("http://vllm:8000/v1"));
        disabled.enabled = false;
        let router = router_for(vec![
            ("openai", disabled),
            ("openrouter", routed_provider(shared, None, None)),
        ]);
        assert!(route(&router, shared).via_openrouter);
    }

    #[test]
    fn configured_provider_auth_wins_over_adapter_default() {
        let router = router_for(vec![(
            "anthropic",
            routed_provider("us.anthropic.claude-opus-4-7", None, None),
        )]);
        let r = route(&router, "us.anthropic.claude-opus-4-7");
        // No base_url: the corrected adapter's default endpoint replaces the
        // one genai picked for its (Ollama) guess.
        assert_eq!(
            r.endpoint_override.as_deref(),
            Some("https://api.anthropic.com/v1/")
        );
        assert_key(router.auth(&r), "test-key");
    }

    #[tokio::test]
    async fn default_base_urls_match_genai() {
        // default_base_url() mirrors genai's crate-private endpoint table; a
        // resolver-less client exposes those defaults.
        let client = Client::default();
        for model in [
            "claude-haiku-4-5",
            "gpt-5-nano",
            "gpt-5-pro",
            "gemini-3.5-flash",
            "grok-4",
            "qwen3:8b",
        ] {
            let target = client.resolve_service_target(model).await.unwrap();
            assert_eq!(
                default_base_url(target.model.adapter_kind),
                Some(target.endpoint.base_url()),
                "{model}"
            );
        }
    }

    #[tokio::test]
    async fn genai_client_resolves_through_the_router() {
        // End to end through genai's resolver pipeline (no network).
        let config = config_with(vec![
            (
                "openai",
                routed_provider(
                    "Qwen/Qwen2.5-7B-Instruct",
                    Some("gpt-5-pro"),
                    Some("http://localhost:8000/v1"),
                ),
            ),
            (
                "ollama",
                routed_provider("qwen3:8b", None, Some("http://localhost:11434")),
            ),
            ("openrouter", openrouter()),
        ]);
        let manager = ProviderManager::new(config, ModelTier::Fast).unwrap();

        let target = manager
            .client
            .resolve_service_target("Qwen/Qwen2.5-7B-Instruct")
            .await
            .unwrap();
        assert_eq!(target.model.adapter_kind, AdapterKind::OpenAI);
        assert_eq!(target.endpoint.base_url(), "http://localhost:8000/v1/");
        assert_key(Some(target.auth), "test-key");

        let target = manager
            .client
            .resolve_service_target("gpt-5-pro")
            .await
            .unwrap();
        assert_eq!(target.model.adapter_kind, AdapterKind::OpenAIResp);
        assert_eq!(target.endpoint.base_url(), "http://localhost:8000/v1/");

        let target = manager
            .client
            .resolve_service_target("qwen3:8b")
            .await
            .unwrap();
        assert_eq!(target.model.adapter_kind, AdapterKind::Ollama);
        assert_eq!(target.endpoint.base_url(), "http://localhost:11434/v1/");

        let target = manager
            .client
            .resolve_service_target("anthropic/claude-opus-4-8")
            .await
            .unwrap();
        assert_eq!(target.model.adapter_kind, AdapterKind::OpenAI);
        assert_eq!(target.endpoint.base_url(), "https://openrouter.ai/api/v1/");
    }
}
