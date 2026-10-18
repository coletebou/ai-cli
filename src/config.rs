use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// CLI mode settings (non-TUI)
    #[serde(default)]
    pub cli: ModeConfig,
    /// TUI mode settings. Missing or partial sections default to multi mode,
    /// not `ModeConfig`'s single mode, so they match `Config::default()`.
    #[serde(
        default = "default_tui_mode",
        deserialize_with = "deserialize_tui_mode"
    )]
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
    /// A file with no `[providers.*]` sections at all gets the built-ins;
    /// otherwise only the providers it lists exist (see
    /// `fill_builtin_provider_fields` for how partial sections are handled).
    #[serde(default = "Config::builtin_providers")]
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

/// `[tui_mode]` defaults: like `[cli]`, but multi-provider, since the TUI's
/// point is side-by-side panels.
fn default_tui_mode() -> ModeConfig {
    ModeConfig {
        default_mode: QueryMode::Multi,
        ..ModeConfig::default()
    }
}

/// Deserialize `[tui_mode]` like any `ModeConfig`, except that a section
/// without `default_mode` means multi (the TUI default) rather than single.
/// A missing section is handled by `default_tui_mode` instead.
fn deserialize_tui_mode<'de, D>(deserializer: D) -> Result<ModeConfig, D::Error>
where
    D: serde::Deserializer<'de>,
{
    fn multi() -> QueryMode {
        QueryMode::Multi
    }

    #[derive(Deserialize)]
    struct TuiModeSection {
        #[serde(default = "multi")]
        default_mode: QueryMode,
        #[serde(default = "default_single_provider")]
        single_provider: String,
        #[serde(default = "default_multi_providers")]
        multi_providers: Vec<String>,
        #[serde(default)]
        model_tier: ModelTier,
    }

    let section = TuiModeSection::deserialize(deserializer)?;
    Ok(ModeConfig {
        default_mode: section.default_mode,
        single_provider: section.single_provider,
        multi_providers: section.multi_providers,
        model_tier: section.model_tier,
    })
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

impl Config {
    /// The providers ai-cli ships with, and the values a partial
    /// `[providers.<name>]` section for one of them falls back to.
    fn builtin_providers() -> HashMap<String, ProviderConfig> {
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
                // xAI's newest model backs both tiers. It always reasons, so
                // the fast tier trades ~0.6s of latency for recency versus
                // grok-4.20-0309-non-reasoning.
                default_model: "grok-4.7".to_string(),
                fast_model: Some("grok-4.7".to_string()),
                smart_model: Some("grok-4.7".to_string()),
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

        providers
    }
}

impl Default for Config {
    fn default() -> Self {
        Config {
            cli: ModeConfig::default(),
            tui_mode: default_tui_mode(),
            mode: None,
            default: DefaultConfig::default(),
            footer: FooterConfig::default(),
            search: SearchConfig::default(),
            retry: RetryConfig::default(),
            providers: Self::builtin_providers(),
        }
    }
}

/// Directory (under the config base dir) and file name of the config file.
const CONFIG_DIR_NAME: &str = "ai-cli";
const CONFIG_FILE_NAME: &str = "config.toml";

/// OS family, as far as config path resolution cares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Platform {
    /// Linux, the BSDs and other non-macOS Unixes.
    Unix,
    MacOs,
    Windows,
}

impl Platform {
    fn current() -> Self {
        if cfg!(target_os = "macos") {
            Platform::MacOs
        } else if cfg!(windows) {
            Platform::Windows
        } else {
            Platform::Unix
        }
    }
}

/// Everything config path resolution reads from the environment, gathered
/// up front so `resolve_config_path` is a pure function that tests can
/// drive without touching the real process environment.
#[derive(Debug, Clone)]
struct PathEnv {
    platform: Platform,
    /// `$AI_CLI_CONFIG`: full path to the config file.
    ai_cli_config: Option<PathBuf>,
    /// `$XDG_CONFIG_HOME`
    xdg_config_home: Option<PathBuf>,
    /// `dirs::home_dir()`
    home_dir: Option<PathBuf>,
    /// `dirs::config_dir()`: the XDG dir on Linux, `~/Library/Application
    /// Support` on macOS, `%APPDATA%` on Windows. Where the config lived
    /// before `resolve_config_path` existed.
    platform_config_dir: Option<PathBuf>,
}

impl PathEnv {
    fn from_process() -> Self {
        Self {
            platform: Platform::current(),
            ai_cli_config: env::var_os("AI_CLI_CONFIG").map(PathBuf::from),
            xdg_config_home: env::var_os("XDG_CONFIG_HOME").map(PathBuf::from),
            home_dir: dirs::home_dir(),
            platform_config_dir: dirs::config_dir(),
        }
    }
}

/// Where the config file lives, highest precedence first:
///
/// 1. `$AI_CLI_CONFIG`, used as-is.
/// 2. `$XDG_CONFIG_HOME/ai-cli/config.toml`, if that's an absolute path
///    (the XDG spec says to ignore relative ones).
/// 3. `~/.config/ai-cli/config.toml` on Linux and macOS;
///    `%APPDATA%\ai-cli\config.toml` on Windows.
///
/// Earlier versions used `dirs::config_dir()`, which on macOS is
/// `~/Library/Application Support` rather than the documented `~/.config`.
/// If there's no file at the path from 2/3 but there is one at that old
/// location, the old one is used, so existing installs keep their config.
/// On Linux the two locations are the same, so this never kicks in there.
///
/// `exists` is `Path::exists` outside of tests.
fn resolve_config_path(env: &PathEnv, exists: impl Fn(&Path) -> bool) -> Result<PathBuf> {
    // An empty env var means unset.
    let non_empty = |path: &Option<PathBuf>| path.clone().filter(|p| !p.as_os_str().is_empty());

    if let Some(explicit) = non_empty(&env.ai_cli_config) {
        return Ok(explicit);
    }

    let base_dir = match non_empty(&env.xdg_config_home).filter(|p| p.is_absolute()) {
        Some(xdg) => xdg,
        None => match env.platform {
            Platform::Windows => non_empty(&env.platform_config_dir).ok_or_else(|| {
                anyhow!(
                    "Could not find the config directory (%APPDATA%). \
                     Set AI_CLI_CONFIG to the full path of a config file."
                )
            })?,
            Platform::Unix | Platform::MacOs => non_empty(&env.home_dir)
                .map(|home| home.join(".config"))
                .ok_or_else(|| {
                    anyhow!(
                        "Could not find your home directory to locate the config file. \
                         Set AI_CLI_CONFIG to the full path of a config file."
                    )
                })?,
        },
    };
    let path = base_dir.join(CONFIG_DIR_NAME).join(CONFIG_FILE_NAME);

    if !exists(&path) {
        if let Some(legacy_dir) = non_empty(&env.platform_config_dir) {
            let legacy = legacy_dir.join(CONFIG_DIR_NAME).join(CONFIG_FILE_NAME);
            if legacy != path && exists(&legacy) {
                return Ok(legacy);
            }
        }
    }

    Ok(path)
}

/// Treat an empty or whitespace-only key as no key at all. `OPENAI_API_KEY=`
/// is how people "unset" a var in a lot of shell and CI setups, and a blank
/// key only gets the provider picked and then rejected with a 401.
fn non_blank(value: Option<String>) -> Option<String> {
    value.filter(|v| !v.trim().is_empty())
}

/// Overlay `overlay` onto `base`: keys in `overlay` win, and nested tables
/// are merged key by key rather than replaced wholesale.
fn deep_merge(base: &mut toml::Table, overlay: &toml::Table) {
    for (key, value) in overlay {
        match (base.get_mut(key), value) {
            (Some(toml::Value::Table(base_table)), toml::Value::Table(overlay_table)) => {
                deep_merge(base_table, overlay_table)
            }
            _ => {
                base.insert(key.clone(), value.clone());
            }
        }
    }
}

/// `OpenOptions` for creating a config file. The file can hold a plaintext
/// `api_key`, so on Unix it's created readable by its owner only (0600).
fn private_file_options() -> fs::OpenOptions {
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
}

fn create_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create config directory: {}", parent.display()))?;
    }
    Ok(())
}

/// Replace `path` with `contents` atomically: write a temp file in the same
/// directory, flush it to disk, then rename it over `path`. A crash or a
/// full disk mid-write leaves the previous config intact instead of a
/// truncated file, and the result is 0600 on Unix (see
/// `private_file_options`).
fn write_atomic(path: &Path, contents: &[u8]) -> Result<()> {
    // Resolve a symlinked config (e.g. into a dotfiles repo) so the rename
    // replaces the file it points to rather than the link itself — even
    // when the link is dangling, where `canonicalize` fails.
    let target = resolve_symlink_target(path)?;
    let file_name = target
        .file_name()
        .ok_or_else(|| anyhow!("Config path has no file name: {}", target.display()))?;
    let tmp = target.with_file_name(format!(
        ".{}.{}.tmp",
        file_name.to_string_lossy(),
        std::process::id()
    ));

    fn write_then_rename(tmp: &Path, target: &Path, contents: &[u8]) -> io::Result<()> {
        // Only a leftover from an earlier save that crashed with our PID
        // could be here; clear it so `create_new` can take the name.
        let _ = fs::remove_file(tmp);
        let mut file = private_file_options().open(tmp)?;
        file.write_all(contents)?;
        file.sync_all()?;
        fs::rename(tmp, target)
    }

    write_then_rename(&tmp, &target, contents).map_err(|err| {
        let _ = fs::remove_file(&tmp);
        anyhow::Error::new(err)
            .context(format!("Failed to write config file: {}", target.display()))
    })
}

/// Follow `path` through any symlinks to the file that should actually be
/// written. Unlike `fs::canonicalize` this works for a dangling link (the
/// target doesn't exist yet), so saving creates the link's target instead
/// of replacing the link with a regular file.
fn resolve_symlink_target(path: &Path) -> Result<PathBuf> {
    let mut current = path.to_path_buf();
    // Bounded, like the OS's own loop limit, so a symlink cycle can't hang.
    for _ in 0..40 {
        match fs::symlink_metadata(&current) {
            Ok(meta) if meta.file_type().is_symlink() => {
                let link = fs::read_link(&current).with_context(|| {
                    format!("Failed to read config symlink: {}", current.display())
                })?;
                current = match current.parent() {
                    Some(parent) if link.is_relative() => parent.join(link),
                    _ => link,
                };
            }
            // Not a symlink, or doesn't exist yet: write here.
            _ => return Ok(current),
        }
    }
    anyhow::bail!(
        "Config path has too many levels of symlinks: {}",
        path.display()
    )
}

impl Config {
    /// Path of the config file. See `resolve_config_path` for the rules.
    pub fn config_path() -> Result<PathBuf> {
        resolve_config_path(&PathEnv::from_process(), Path::exists)
    }

    pub fn load() -> Result<Self> {
        Self::load_from(&Self::config_path()?)
    }

    fn load_from(config_path: &Path) -> Result<Self> {
        if !config_path.exists() {
            return Ok(Config::default());
        }

        let content = fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;
        Self::from_toml_str(&content)
            .with_context(|| format!("Failed to parse config file: {}", config_path.display()))
    }

    fn from_toml_str(content: &str) -> Result<Self> {
        // Deserialize straight from the text first: toml's errors then name
        // the offending key and line, which deserializing the merged
        // `toml::Value` in `fill_builtin_provider_fields` would not.
        let mut config: Config = toml::from_str(content)?;
        let raw: toml::Table = toml::from_str(content)?;
        config.fill_builtin_provider_fields(&raw)?;

        // Migrate legacy [mode] section to [cli] / [tui_mode] — but only
        // into sections the file doesn't have. Comparing against defaults
        // instead would clobber an explicit section that happens to equal
        // them (e.g. `[tui_mode] default_mode = "multi"`).
        if let Some(legacy_mode) = config.mode.take() {
            if !raw.contains_key("cli") {
                config.cli = legacy_mode.clone();
            }
            // The TUI defaults to multi-provider regardless of [mode].
            if !raw.contains_key("tui_mode") {
                config.tui_mode = ModeConfig {
                    default_mode: QueryMode::Multi,
                    ..legacy_mode
                };
            }
        }

        Ok(config)
    }

    /// Fill in the fields a `[providers.<name>]` section for a built-in
    /// provider leaves out, from that provider's built-in values. Otherwise a
    /// section that only sets `fast_model` gets serde's field defaults for
    /// everything else: no aliases (so "claude" stops resolving), color
    /// "white", no `api_key_env`/`base_url`/`smart_model`.
    ///
    /// Only providers the file lists are touched. A built-in the file leaves
    /// out stays out (it may have been deleted on purpose), and custom
    /// providers have no built-in values to fall back to.
    fn fill_builtin_provider_fields(&mut self, raw: &toml::Table) -> Result<()> {
        let Some(toml::Value::Table(user_providers)) = raw.get("providers") else {
            return Ok(());
        };
        let builtins = Self::builtin_providers();

        for (name, user_section) in user_providers {
            let (Some(builtin), toml::Value::Table(user_table)) =
                (builtins.get(name), user_section)
            else {
                continue;
            };
            let toml::Value::Table(mut merged) = toml::Value::try_from(builtin)? else {
                continue;
            };
            // `enabled` keeps its documented default of true: listing a
            // provider means "use it" unless the section says otherwise.
            // (This only differs for ollama, which is built-in disabled.)
            merged.remove("enabled");
            // A legacy section that sets only `default_model` relies on the
            // tier models falling back to it (`ProviderConfig::get_model`),
            // so don't shadow it with the built-in tier models.
            if user_table.contains_key("default_model") {
                merged.remove("fast_model");
                merged.remove("smart_model");
            }
            deep_merge(&mut merged, user_table);

            let provider = toml::Value::Table(merged)
                .try_into::<ProviderConfig>()
                .with_context(|| format!("Invalid [providers.{name}] section"))?;
            self.providers.insert(name.clone(), provider);
        }

        Ok(())
    }

    /// Generate a richly-commented default config file.
    pub fn init() -> Result<()> {
        let config_path = Self::config_path()?;
        Self::write_init_template(&config_path)?;

        println!("Created config file: {}", config_path.display());
        println!("\nEdit it directly or run `ai config` for the interactive TUI editor.");

        Ok(())
    }

    /// Write the commented template to `path`, refusing to overwrite an
    /// existing file. A symlinked `path` (e.g. into a dotfiles repo) is
    /// followed the way `save` follows it, so a link whose target doesn't
    /// exist yet gets that target created, rather than failing because the
    /// link itself exists.
    fn write_init_template(path: &Path) -> Result<()> {
        let target = resolve_symlink_target(path)?;
        create_parent_dir(&target)?;

        // `create_new` makes the existence check and the create one step.
        // `target` is where symlink resolution stopped, so it's a regular
        // file (refused) or nothing; and `create_new` refuses a symlink too,
        // so one appearing there in the meantime isn't followed.
        let mut file = match private_file_options().open(&target) {
            Ok(file) => file,
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists => bail!(
                "Config file already exists: {}\n\
                 Remove it first to regenerate it, or run `ai config` to edit it.",
                path.display()
            ),
            Err(err) => {
                return Err(err)
                    .with_context(|| format!("Failed to create config file: {}", target.display()))
            }
        };

        if let Err(err) = file.write_all(Self::generate_init_template().as_bytes()) {
            // Don't leave a truncated file that the next --init refuses to
            // replace. Remove the file we created, never a link to it.
            let _ = fs::remove_file(&target);
            return Err(err)
                .with_context(|| format!("Failed to write config file: {}", target.display()));
        }

        Ok(())
    }

    fn env_status(var: &str) -> &'static str {
        if non_blank(env::var(var).ok()).is_some() {
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
# Providers queried by -a / --all (alias names); leave empty to query every
# enabled provider. Normal queries use [cli] / [tui_mode] above instead.
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
smart_model = "gpt-5.5"
aliases = ["chatgpt", "gpt"]
color = "green"

[providers.xai]
enabled = true
api_key_env = "XAI_API_KEY"
fast_model = "grok-4.7"
smart_model = "grok-4.7"
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
        self.save_to(&Self::config_path()?)
    }

    fn save_to(&self, config_path: &Path) -> Result<()> {
        let content = toml::to_string_pretty(self).with_context(|| "Failed to serialize config")?;
        create_parent_dir(config_path)?;
        write_atomic(config_path, content.as_bytes())
    }

    pub fn get_api_key(&self, provider_name: &str) -> Option<String> {
        self.get_api_key_with(provider_name, |var| env::var(var).ok())
    }

    /// `get_api_key` with the env lookup injected, so tests don't have to
    /// mutate the process environment. Blank values count as unset.
    fn get_api_key_with(
        &self,
        provider_name: &str,
        env_lookup: impl Fn(&str) -> Option<String>,
    ) -> Option<String> {
        let p = self.providers.get(provider_name)?;
        // First check direct api_key
        if let Some(key) = non_blank(p.api_key.clone()) {
            return Some(key);
        }
        // Then check env var
        let env_var = p.api_key_env.as_deref()?;
        if let Some(key) = non_blank(env_lookup(env_var)) {
            return Some(key);
        }
        // Also check GOOGLE_API_KEY for gemini
        if env_var == "GEMINI_API_KEY" {
            return non_blank(env_lookup("GOOGLE_API_KEY"));
        }
        None
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

    #[test]
    fn xai_defaults_use_current_grok_model() {
        // xAI silently serves retired ids as some other model (e.g.
        // grok-4-1-fast-non-reasoning comes back as grok-4.3), so a stale
        // default never errors; pin the current id explicitly instead.
        let xai = &Config::default().providers["xai"];
        assert_eq!(xai.default_model, "grok-4.7");
        assert_eq!(xai.get_model(&ModelTier::Fast), "grok-4.7");
        assert_eq!(xai.get_model(&ModelTier::Smart), "grok-4.7");
    }

    // ── Config path resolution ──────────────────────────────────────────

    fn path_env(platform: Platform) -> PathEnv {
        let platform_config_dir = match platform {
            Platform::Unix => "/home/me/.config",
            Platform::MacOs => "/Users/me/Library/Application Support",
            Platform::Windows => r"C:\Users\me\AppData\Roaming",
        };
        PathEnv {
            platform,
            ai_cli_config: None,
            xdg_config_home: None,
            home_dir: Some(PathBuf::from(if platform == Platform::MacOs {
                "/Users/me"
            } else {
                "/home/me"
            })),
            platform_config_dir: Some(PathBuf::from(platform_config_dir)),
        }
    }

    fn config_in(dir: &str) -> PathBuf {
        Path::new(dir).join("ai-cli").join("config.toml")
    }

    /// An `XDG_CONFIG_HOME` that is absolute on the host running the tests.
    /// `resolve_config_path` ignores a relative one, and on Windows `/xdg`
    /// has no drive prefix, so it counts as relative there.
    fn xdg_dir() -> &'static str {
        if cfg!(windows) {
            r"C:\xdg"
        } else {
            "/xdg"
        }
    }

    fn nothing_exists(_: &Path) -> bool {
        false
    }

    #[test]
    fn path_linux_defaults_to_dot_config() {
        let path = resolve_config_path(&path_env(Platform::Unix), nothing_exists).unwrap();
        assert_eq!(path, config_in("/home/me/.config"));
    }

    #[test]
    fn path_ai_cli_config_wins_over_everything() {
        let mut env = path_env(Platform::MacOs);
        env.ai_cli_config = Some("/tmp/custom.toml".into());
        env.xdg_config_home = Some(xdg_dir().into());
        let legacy = config_in("/Users/me/Library/Application Support");
        let path = resolve_config_path(&env, |p| p == legacy).unwrap();
        assert_eq!(path, PathBuf::from("/tmp/custom.toml"));
    }

    #[test]
    fn path_honours_xdg_config_home() {
        for platform in [Platform::Unix, Platform::MacOs, Platform::Windows] {
            let mut env = path_env(platform);
            env.xdg_config_home = Some(xdg_dir().into());
            let path = resolve_config_path(&env, nothing_exists).unwrap();
            assert_eq!(path, config_in(xdg_dir()), "{platform:?}");
        }
    }

    #[test]
    fn path_ignores_empty_and_relative_env_vars() {
        let mut env = path_env(Platform::Unix);
        env.ai_cli_config = Some(PathBuf::new());
        env.xdg_config_home = Some("relative/dir".into());
        let path = resolve_config_path(&env, nothing_exists).unwrap();
        assert_eq!(path, config_in("/home/me/.config"));

        env.xdg_config_home = Some(PathBuf::new());
        let path = resolve_config_path(&env, nothing_exists).unwrap();
        assert_eq!(path, config_in("/home/me/.config"));
    }

    #[test]
    fn path_macos_uses_dot_config_for_new_installs() {
        let path = resolve_config_path(&path_env(Platform::MacOs), nothing_exists).unwrap();
        assert_eq!(path, config_in("/Users/me/.config"));
    }

    #[test]
    fn path_macos_keeps_using_legacy_application_support_file() {
        let legacy = config_in("/Users/me/Library/Application Support");
        let path = resolve_config_path(&path_env(Platform::MacOs), |p| p == legacy).unwrap();
        assert_eq!(path, legacy);

        // Same when XDG_CONFIG_HOME points somewhere that has no config yet.
        let mut env = path_env(Platform::MacOs);
        env.xdg_config_home = Some(xdg_dir().into());
        let path = resolve_config_path(&env, |p| p == legacy).unwrap();
        assert_eq!(path, legacy);
    }

    #[test]
    fn path_macos_prefers_dot_config_when_both_exist() {
        let path = resolve_config_path(&path_env(Platform::MacOs), |_| true).unwrap();
        assert_eq!(path, config_in("/Users/me/.config"));
    }

    #[test]
    fn path_windows_uses_appdata() {
        let path = resolve_config_path(&path_env(Platform::Windows), nothing_exists).unwrap();
        assert_eq!(path, config_in(r"C:\Users\me\AppData\Roaming"));
    }

    #[test]
    fn path_errors_instead_of_falling_back_to_cwd() {
        let mut env = path_env(Platform::Unix);
        env.home_dir = None;
        let err = resolve_config_path(&env, nothing_exists).unwrap_err();
        assert!(err.to_string().contains("AI_CLI_CONFIG"), "{err}");

        let mut env = path_env(Platform::Windows);
        env.platform_config_dir = None;
        assert!(resolve_config_path(&env, nothing_exists).is_err());
    }

    // ── API keys ────────────────────────────────────────────────────────

    fn fake_env(vars: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let vars: HashMap<String, String> = vars
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |name| vars.get(name).cloned()
    }

    #[test]
    fn blank_env_api_keys_count_as_missing() {
        let config = Config::default();
        for blank in ["", "   ", "\t\n"] {
            let env = fake_env(&[("OPENAI_API_KEY", blank)]);
            assert_eq!(config.get_api_key_with("openai", env), None);
        }
        let env = fake_env(&[("OPENAI_API_KEY", "sk-test")]);
        assert_eq!(
            config.get_api_key_with("openai", env),
            Some("sk-test".to_string())
        );
    }

    #[test]
    fn blank_gemini_key_falls_back_to_google_api_key() {
        let config = Config::default();
        let env = fake_env(&[("GEMINI_API_KEY", ""), ("GOOGLE_API_KEY", "g-key")]);
        assert_eq!(
            config.get_api_key_with("google", env),
            Some("g-key".to_string())
        );
        let env = fake_env(&[("GEMINI_API_KEY", ""), ("GOOGLE_API_KEY", " ")]);
        assert_eq!(config.get_api_key_with("google", env), None);
    }

    #[test]
    fn blank_config_api_key_falls_through_to_env() {
        let mut config = Config::default();
        config.providers.get_mut("anthropic").unwrap().api_key = Some(" ".to_string());
        let env = fake_env(&[("ANTHROPIC_API_KEY", "env-key")]);
        assert_eq!(
            config.get_api_key_with("anthropic", env),
            Some("env-key".to_string())
        );
        assert_eq!(config.get_api_key_with("anthropic", fake_env(&[])), None);
    }

    // ── Reading and writing files ───────────────────────────────────────

    /// A scratch directory under the system temp dir, removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let dir = env::temp_dir().join(format!("ai-cli-config-{}-{}", tag, std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[cfg(unix)]
    fn mode_of(path: &Path) -> u32 {
        use std::os::unix::fs::PermissionsExt;
        fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    #[test]
    fn save_round_trips_and_is_private() {
        let dir = TempDir::new("save");
        let path = dir.0.join("nested").join("config.toml");

        let mut config = Config::default();
        config.cli.default_mode = QueryMode::Multi;
        config.cli.model_tier = ModelTier::Smart;
        config.tui_mode.multi_providers = vec!["claude".to_string(), "or".to_string()];
        config.footer.show_tokens = true;
        config.retry.max_attempts = 7;
        config.default.system_prompt = Some("be brief".to_string());
        let anthropic = config.providers.get_mut("anthropic").unwrap();
        anthropic.fast_model = Some("claude-custom".to_string());
        anthropic.auth_mode = AuthMode::Subscription;
        config.providers.get_mut("openai").unwrap().api_key = Some("sk-test".to_string());
        config.providers.remove("ollama");

        config.save_to(&path).unwrap();
        let loaded = Config::load_from(&path).unwrap();

        assert_eq!(loaded.cli, config.cli);
        assert_eq!(loaded.tui_mode, config.tui_mode);
        assert_eq!(loaded.retry, config.retry);
        assert!(loaded.footer.show_tokens);
        assert_eq!(loaded.default.system_prompt.as_deref(), Some("be brief"));
        let mut names: Vec<_> = loaded.providers.keys().cloned().collect();
        names.sort();
        assert_eq!(
            names,
            ["anthropic", "google", "openai", "openrouter", "xai"]
        );
        let anthropic = &loaded.providers["anthropic"];
        assert_eq!(anthropic.fast_model.as_deref(), Some("claude-custom"));
        assert_eq!(anthropic.auth_mode, AuthMode::Subscription);
        assert_eq!(
            loaded.providers["openai"].api_key.as_deref(),
            Some("sk-test")
        );

        // Only config.toml is left behind (no stray temp file).
        let entries: Vec<_> = fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(entries, ["config.toml"]);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(mode_of(&path), 0o600);
            // Saving over an existing world-readable file tightens it too.
            fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
            config.save_to(&path).unwrap();
            assert_eq!(mode_of(&path), 0o600);
        }
    }

    #[cfg(unix)]
    #[test]
    fn save_through_symlink_updates_the_target() {
        let dir = TempDir::new("symlink");
        let real = dir.0.join("dotfiles").join("config.toml");
        fs::create_dir_all(real.parent().unwrap()).unwrap();
        fs::write(&real, "").unwrap();
        let link = dir.0.join("config.toml");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let mut config = Config::default();
        config.retry.max_attempts = 9;
        config.save_to(&link).unwrap();

        assert!(fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(Config::load_from(&real).unwrap().retry.max_attempts, 9);
    }

    #[cfg(unix)]
    #[test]
    fn save_through_dangling_relative_symlink_creates_the_target() {
        // A dotfiles-managed link whose target doesn't exist yet must stay a
        // link; the save creates the file it points at.
        let dir = TempDir::new("dangling");
        fs::create_dir_all(dir.0.join("dotfiles")).unwrap();
        let link = dir.0.join("config.toml");
        std::os::unix::fs::symlink("dotfiles/config.toml", &link).unwrap();

        let mut config = Config::default();
        config.retry.max_attempts = 7;
        config.save_to(&link).unwrap();

        assert!(fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink());
        let real = dir.0.join("dotfiles").join("config.toml");
        assert_eq!(Config::load_from(&real).unwrap().retry.max_attempts, 7);
    }

    #[test]
    fn legacy_mode_does_not_clobber_an_explicit_tui_mode() {
        // Explicit [tui_mode] that happens to equal the defaults must win
        // over the legacy [mode] section.
        let config = Config::from_toml_str(
            "[mode]\nsingle_provider = \"gpt\"\nmodel_tier = \"smart\"\n\n\
             [tui_mode]\ndefault_mode = \"multi\"\n",
        )
        .unwrap();
        assert_eq!(config.cli.single_provider, "gpt");
        assert_eq!(config.tui_mode.single_provider, "claude");
        assert_eq!(config.tui_mode.model_tier, ModelTier::Fast);
    }

    #[test]
    fn init_creates_template_and_refuses_to_overwrite() {
        let dir = TempDir::new("init");
        let path = dir.0.join("ai-cli").join("config.toml");

        Config::write_init_template(&path).unwrap();
        let written = fs::read_to_string(&path).unwrap();
        assert!(written.contains("[providers.anthropic]"));
        #[cfg(unix)]
        assert_eq!(mode_of(&path), 0o600);

        fs::write(&path, "# my edits\n").unwrap();
        let err = Config::write_init_template(&path).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("already exists"), "{msg}");
        assert!(msg.contains(&path.display().to_string()), "{msg}");
        assert_eq!(fs::read_to_string(&path).unwrap(), "# my edits\n");
    }

    #[cfg(unix)]
    #[test]
    fn init_through_dangling_symlink_creates_the_target() {
        // `ai --init` with the config path pre-linked into a dotfiles repo
        // that doesn't have the file (or its directory) yet: the link stays
        // a link, its target is created, and a second init still refuses.
        let dir = TempDir::new("init-dangling");
        let link = dir.0.join("config.toml");
        std::os::unix::fs::symlink("dotfiles/ai-cli/config.toml", &link).unwrap();

        Config::write_init_template(&link).unwrap();

        assert!(fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink());
        let real = dir.0.join("dotfiles").join("ai-cli").join("config.toml");
        assert!(fs::read_to_string(&real)
            .unwrap()
            .contains("[providers.anthropic]"));
        assert_eq!(mode_of(&real), 0o600);

        fs::write(&real, "# my edits\n").unwrap();
        let err = Config::write_init_template(&link).unwrap_err();
        assert!(err.to_string().contains("already exists"), "{err}");
        assert_eq!(fs::read_to_string(&real).unwrap(), "# my edits\n");
        assert!(fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink());
    }

    #[test]
    fn parse_error_names_the_file() {
        let dir = TempDir::new("parse-error");
        let path = dir.0.join("config.toml");
        fs::write(&path, "[cli]\ndefault_mode = 42\n").unwrap();

        let err = Config::load_from(&path).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains(&path.display().to_string()), "{msg}");
        assert!(msg.contains("default_mode"), "{msg}");
    }

    #[test]
    fn missing_file_loads_defaults() {
        let dir = TempDir::new("missing");
        let config = Config::load_from(&dir.0.join("config.toml")).unwrap();
        assert_eq!(config.tui_mode, default_tui_mode());
        assert_eq!(config.providers.len(), Config::default().providers.len());
    }

    // ── Defaults for missing / partial sections ─────────────────────────

    #[test]
    fn init_template_matches_defaults() {
        // Plain toml parse, without `from_toml_str`'s built-in merge, so a
        // field missing from the template shows up as a mismatch.
        let template: Config = toml::from_str(&Config::generate_init_template()).unwrap();
        let defaults = Config::default();

        let mut template_names: Vec<_> = template.providers.keys().collect();
        let mut default_names: Vec<_> = defaults.providers.keys().collect();
        template_names.sort();
        default_names.sort();
        assert_eq!(template_names, default_names);

        for (name, want) in &defaults.providers {
            let got = &template.providers[name];
            assert_eq!(got.enabled, want.enabled, "{name}.enabled");
            assert_eq!(got.fast_model, want.fast_model, "{name}.fast_model");
            assert_eq!(got.smart_model, want.smart_model, "{name}.smart_model");
            assert_eq!(got.aliases, want.aliases, "{name}.aliases");
            assert_eq!(got.color, want.color, "{name}.color");
            assert_eq!(got.api_key_env, want.api_key_env, "{name}.api_key_env");
            assert_eq!(got.base_url, want.base_url, "{name}.base_url");
            assert_eq!(got.auth_mode, want.auth_mode, "{name}.auth_mode");
        }

        assert_eq!(template.cli, defaults.cli);
        assert_eq!(template.tui_mode, defaults.tui_mode);
        assert_eq!(template.retry, defaults.retry);
        assert_eq!(template.default.providers, defaults.default.providers);
        assert_eq!(template.search.provider, defaults.search.provider);
        assert_eq!(template.search.gemini.model, defaults.search.gemini.model);
    }

    #[test]
    fn missing_tui_mode_section_defaults_to_multi() {
        let config = Config::from_toml_str("[cli]\nmodel_tier = \"smart\"\n").unwrap();
        assert_eq!(config.tui_mode.default_mode, QueryMode::Multi);
        assert_eq!(config.tui_mode, Config::default().tui_mode);
        assert_eq!(config.cli.default_mode, QueryMode::Single);
    }

    #[test]
    fn partial_tui_mode_section_defaults_to_multi() {
        let config = Config::from_toml_str("[tui_mode]\nmodel_tier = \"smart\"\n").unwrap();
        assert_eq!(config.tui_mode.default_mode, QueryMode::Multi);
        assert_eq!(config.tui_mode.model_tier, ModelTier::Smart);

        let config = Config::from_toml_str("[tui_mode]\ndefault_mode = \"single\"\n").unwrap();
        assert_eq!(config.tui_mode.default_mode, QueryMode::Single);
    }

    #[test]
    fn legacy_mode_section_still_migrates() {
        let config = Config::from_toml_str(
            "[mode]\ndefault_mode = \"single\"\nsingle_provider = \"gpt\"\nmodel_tier = \"smart\"\n",
        )
        .unwrap();
        assert_eq!(config.cli.single_provider, "gpt");
        assert_eq!(config.cli.model_tier, ModelTier::Smart);
        assert_eq!(config.tui_mode.single_provider, "gpt");
        assert_eq!(config.tui_mode.default_mode, QueryMode::Multi);
    }

    #[test]
    fn missing_providers_section_gets_builtins() {
        let config = Config::from_toml_str("[footer]\nshow_tokens = true\n").unwrap();
        assert_eq!(config.providers.len(), Config::default().providers.len());
        assert_eq!(
            config.resolve_provider_alias("claude"),
            Some("anthropic".to_string())
        );
    }

    #[test]
    fn partial_builtin_provider_keeps_builtin_fields() {
        let config =
            Config::from_toml_str("[providers.anthropic]\nfast_model = \"my-haiku\"\n").unwrap();
        let anthropic = &config.providers["anthropic"];
        assert_eq!(anthropic.fast_model.as_deref(), Some("my-haiku"));
        assert_eq!(anthropic.smart_model.as_deref(), Some("claude-opus-4-8"));
        assert_eq!(anthropic.aliases, ["claude"]);
        assert_eq!(anthropic.color, "magenta");
        assert_eq!(anthropic.api_key_env.as_deref(), Some("ANTHROPIC_API_KEY"));
        assert!(anthropic.enabled);
        assert_eq!(
            config.resolve_provider_alias("claude"),
            Some("anthropic".to_string())
        );
        // Built-ins the file doesn't mention are not added back.
        assert_eq!(config.providers.len(), 1);
    }

    #[test]
    fn user_fields_win_over_builtin_fields() {
        let config = Config::from_toml_str(
            "[providers.openrouter]\naliases = []\ncolor = \"red\"\nbase_url = \"http://proxy\"\n\
             enabled = false\n",
        )
        .unwrap();
        let or = &config.providers["openrouter"];
        assert!(or.aliases.is_empty());
        assert_eq!(or.color, "red");
        assert_eq!(or.base_url.as_deref(), Some("http://proxy"));
        assert!(!or.enabled);
        assert_eq!(or.api_key_env.as_deref(), Some("OPENROUTER_API_KEY"));
    }

    #[test]
    fn builtin_section_without_enabled_stays_enabled() {
        // ollama is built-in disabled, but listing it has always enabled it.
        let config =
            Config::from_toml_str("[providers.ollama]\nbase_url = \"http://box:11434\"\n").unwrap();
        let ollama = &config.providers["ollama"];
        assert!(ollama.enabled);
        assert_eq!(ollama.base_url.as_deref(), Some("http://box:11434"));
        assert_eq!(ollama.aliases, ["ollama", "local"]);
    }

    #[test]
    fn legacy_default_model_is_not_shadowed_by_builtin_tiers() {
        let config =
            Config::from_toml_str("[providers.openai]\ndefault_model = \"gpt-legacy\"\n").unwrap();
        let openai = &config.providers["openai"];
        assert_eq!(openai.get_model(&ModelTier::Fast), "gpt-legacy");
        assert_eq!(openai.get_model(&ModelTier::Smart), "gpt-legacy");
        assert_eq!(openai.aliases, ["chatgpt", "gpt"]);
    }

    #[test]
    fn custom_provider_is_left_as_is() {
        let config = Config::from_toml_str(
            "[providers.mine]\nfast_model = \"m\"\n\n[providers.anthropic]\ncolor = \"blue\"\n",
        )
        .unwrap();
        let mine = &config.providers["mine"];
        assert_eq!(mine.fast_model.as_deref(), Some("m"));
        assert!(mine.aliases.is_empty());
        assert_eq!(mine.color, "white");
        assert_eq!(mine.api_key_env, None);
        assert_eq!(config.providers["anthropic"].color, "blue");
    }
}
