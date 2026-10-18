use crate::config::Config;
use crate::search::{http_client, HTTP_CONNECT_TIMEOUT};
use anyhow::{Context, Result};
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Overall budget for fetching the OpenRouter model list (a few hundred KB of
/// JSON). Without a timeout, `ai models` hangs for ~90s on a blocked host.
const MODELS_REQUEST_TIMEOUT: Duration = Duration::from_secs(20);

/// Width of the MODEL ID column in the `ai models` table.
const ID_COLUMN_WIDTH: usize = 45;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub context_length: Option<u64>,
    #[serde(default)]
    pub pricing: Option<ModelPricing>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPricing {
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub completion: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelsCache {
    fetched_at: u64,
    models: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterResponse {
    data: Vec<OpenRouterModel>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterModel {
    id: String,
    name: String,
    context_length: Option<u64>,
    pricing: Option<OpenRouterPricing>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterPricing {
    prompt: Option<String>,
    completion: Option<String>,
}

/// Where the model cache is written: the platform cache dir
/// (`~/.cache/ai-cli` on Linux), not the config dir, since it's disposable.
fn cache_path() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("ai-cli")
        .join("models_cache.json")
}

/// Where older versions wrote the cache (next to config.toml). Still read so
/// an upgrade doesn't force a refetch; removed after the next successful save.
fn legacy_cache_path() -> Option<PathBuf> {
    let legacy = dirs::config_dir()?.join("ai-cli").join("models_cache.json");
    (legacy != cache_path()).then_some(legacy)
}

/// The cache file to read: the current location, else the legacy one.
fn existing_cache_path() -> Option<PathBuf> {
    std::iter::once(cache_path())
        .chain(legacy_cache_path())
        .find(|p| p.exists())
}

fn cache_age_hours(path: &Path) -> Option<f64> {
    let metadata = fs::metadata(path).ok()?;
    let modified = metadata.modified().ok()?;
    let elapsed = SystemTime::now().duration_since(modified).ok()?;
    Some(elapsed.as_secs_f64() / 3600.0)
}

fn load_cache(path: &Path) -> Option<Vec<ModelInfo>> {
    let content = fs::read_to_string(path).ok()?;
    let cache: ModelsCache = serde_json::from_str(&content).ok()?;
    Some(cache.models)
}

fn save_cache(models: &[ModelInfo]) -> Result<PathBuf> {
    let path = cache_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let cache = ModelsCache {
        fetched_at: SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        models: models.to_vec(),
    };
    let content = serde_json::to_string_pretty(&cache)?;
    fs::write(&path, content)?;
    // Migrated: drop the stale copy in the config dir. Best-effort; a
    // leftover file is harmless since the new location is read first.
    if let Some(legacy) = legacy_cache_path() {
        let _ = fs::remove_file(legacy);
    }
    Ok(path)
}

async fn fetch_openrouter_models(api_key: Option<&str>) -> Result<Vec<ModelInfo>> {
    let client = http_client(HTTP_CONNECT_TIMEOUT, MODELS_REQUEST_TIMEOUT);
    let mut req = client.get("https://openrouter.ai/api/v1/models");

    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }

    let resp: OpenRouterResponse = req
        .send()
        .await
        .context("Failed to reach OpenRouter API")?
        .error_for_status()
        .context("OpenRouter API returned an error")?
        .json()
        .await
        .context("Failed to parse OpenRouter response")?;

    let models = resp
        .data
        .into_iter()
        .map(|m| ModelInfo {
            id: m.id,
            name: m.name,
            context_length: m.context_length,
            pricing: m.pricing.map(|p| ModelPricing {
                prompt: p.prompt,
                completion: p.completion,
            }),
        })
        .collect();

    Ok(models)
}

/// A model is free only when both prices are known to be zero. A missing or
/// unparseable price is unknown, not free, so `--free` doesn't list models we
/// know nothing about (and OpenRouter's `-1` "variable" price isn't zero).
fn is_free(model: &ModelInfo) -> bool {
    let is_zero = |price: Option<&str>| price.and_then(parse_price) == Some(0.0);
    match &model.pricing {
        Some(p) => is_zero(p.prompt.as_deref()) && is_zero(p.completion.as_deref()),
        None => false,
    }
}

/// Parse an OpenRouter per-token price string ("0.000003", "0", "-1").
fn parse_price(price: &str) -> Option<f64> {
    price.trim().parse::<f64>().ok().filter(|v| v.is_finite())
}

pub async fn list_models(search: Option<&str>, refresh: bool, free_only: bool) -> Result<()> {
    let config = Config::load()?;
    let api_key = config.get_api_key("openrouter");

    // Use cache if fresh enough (< 24 hours) and not forcing refresh
    let models = if !refresh {
        let cache = existing_cache_path().and_then(|p| Some((cache_age_hours(&p)?, p)));
        if let Some((age, path)) = cache {
            if age < 24.0 {
                if let Some(cached) = load_cache(&path) {
                    eprintln!(
                        "{} Using cached models ({}h old). Use {} to refresh.",
                        "[cache]".dimmed(),
                        age as u32,
                        "--refresh".bold()
                    );
                    cached
                } else {
                    fetch_and_cache(api_key.as_deref()).await?
                }
            } else {
                eprintln!(
                    "{} Cache expired, fetching fresh list...",
                    "[fetch]".dimmed()
                );
                fetch_and_cache(api_key.as_deref()).await?
            }
        } else {
            eprintln!("{} No cache found, fetching models...", "[fetch]".dimmed());
            fetch_and_cache(api_key.as_deref()).await?
        }
    } else {
        eprintln!("{} Refreshing model list...", "[fetch]".dimmed());
        fetch_and_cache(api_key.as_deref()).await?
    };

    // Filter by search term and free flag
    let filtered: Vec<&ModelInfo> = models
        .iter()
        .filter(|m| {
            if free_only && !is_free(m) {
                return false;
            }
            if let Some(query) = search {
                let q = query.to_lowercase();
                m.id.to_lowercase().contains(&q) || m.name.to_lowercase().contains(&q)
            } else {
                true
            }
        })
        .collect();

    if filtered.is_empty() {
        if let Some(query) = search {
            eprintln!("No models matching '{}'", query);
        } else {
            eprintln!("No models found.");
        }
        return Ok(());
    }

    // Print header
    let mut suffix = String::new();
    if free_only {
        suffix.push_str(" (free)");
    }
    if let Some(s) = search {
        suffix.push_str(&format!(" matching '{}'", s));
    }
    println!(
        "\n{} {} models{}",
        "OpenRouter".yellow().bold(),
        filtered.len(),
        suffix
    );
    // Prices are USD per token from OpenRouter, shown per million tokens.
    println!(
        "{:<width$} {:>8} {:>12} {:>12}",
        "MODEL ID".bold(),
        "CTX".bold(),
        "IN $/M".bold(),
        "OUT $/M".bold(),
        width = ID_COLUMN_WIDTH
    );
    println!("{}", "─".repeat(ID_COLUMN_WIDTH + 35));

    for model in &filtered {
        let ctx = model
            .context_length
            .map(format_context)
            .unwrap_or_else(|| "—".to_string());

        let (in_price, out_price) = match &model.pricing {
            Some(p) => (
                format_price(p.prompt.as_deref()),
                format_price(p.completion.as_deref()),
            ),
            None => ("—".to_string(), "—".to_string()),
        };

        println!(
            "{:<width$} {:>8} {} {}",
            truncate_id(&model.id, ID_COLUMN_WIDTH),
            ctx,
            price_cell(&in_price, 12),
            price_cell(&out_price, 12),
            width = ID_COLUMN_WIDTH
        );
    }

    println!(
        "\n{} Use model IDs with: {}",
        "Tip:".dimmed(),
        "ai config → Providers → openrouter".dimmed()
    );

    Ok(())
}

async fn fetch_and_cache(api_key: Option<&str>) -> Result<Vec<ModelInfo>> {
    let models = fetch_openrouter_models(api_key).await?;
    match save_cache(&models) {
        Ok(path) => eprintln!(
            "{} Fetched {} models, cached to {}",
            "[ok]".green(),
            models.len(),
            path.display()
        ),
        Err(e) => eprintln!(
            "{} Fetched {} models; failed to cache them: {:#}",
            "[warn]".yellow(),
            models.len(),
            e
        ),
    }
    Ok(models)
}

fn format_context(ctx: u64) -> String {
    if ctx >= 1_000_000 {
        format!("{}M", ctx / 1_000_000)
    } else if ctx >= 1000 {
        format!("{}k", ctx / 1000)
    } else {
        ctx.to_string()
    }
}

/// Render a per-token price as USD per million tokens. Plain text (no color)
/// so the table can pad it; see `price_cell`.
fn format_price(price: Option<&str>) -> String {
    let Some(val) = price.and_then(parse_price) else {
        // Missing or unparseable: unknown, not free.
        return "—".to_string();
    };
    if val == 0.0 {
        "free".to_string()
    } else if val < 0.0 {
        // OpenRouter reports -1 for routers whose price depends on the model
        // they pick (e.g. openrouter/auto).
        "varies".to_string()
    } else {
        // Price is per-token, convert to per-million
        let per_million = val * 1_000_000.0;
        if per_million < 0.01 {
            format!("${:.4}", per_million)
        } else if per_million < 1.0 {
            format!("${:.3}", per_million)
        } else {
            format!("${:.2}", per_million)
        }
    }
}

/// Right-align a price in a `width`-char column, coloring "free" green.
/// Padding happens before coloring: ANSI escape codes count toward `{:>N}`
/// widths, which pushed colored cells out of alignment.
fn price_cell(price: &str, width: usize) -> String {
    let cell = format!("{:>width$}", price);
    if price == "free" {
        cell.green().to_string()
    } else {
        cell
    }
}

/// Shorten a model ID to fit a `width`-char column, ending in "…" when cut.
fn truncate_id(id: &str, width: usize) -> String {
    if id.chars().count() <= width {
        id.to_string()
    } else {
        let prefix: String = id.chars().take(width.saturating_sub(1)).collect();
        format!("{}…", prefix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_with_pricing(prompt: Option<&str>, completion: Option<&str>) -> ModelInfo {
        ModelInfo {
            id: "vendor/model".to_string(),
            name: "Model".to_string(),
            context_length: None,
            pricing: Some(ModelPricing {
                prompt: prompt.map(str::to_string),
                completion: completion.map(str::to_string),
            }),
        }
    }

    #[test]
    fn format_price_free() {
        assert_eq!(format_price(Some("0")), "free");
        assert_eq!(format_price(Some("0.0")), "free");
    }

    #[test]
    fn format_price_variable_router() {
        assert_eq!(format_price(Some("-1")), "varies");
    }

    #[test]
    fn format_price_small_and_large() {
        // $0.000000001/token = $0.001/M
        assert_eq!(format_price(Some("0.000000001")), "$0.0010");
        // $0.0000005/token = $0.50/M
        assert_eq!(format_price(Some("0.0000005")), "$0.500");
        // $0.000003/token = $3/M
        assert_eq!(format_price(Some("0.000003")), "$3.00");
        // $0.00015/token = $150/M
        assert_eq!(format_price(Some("0.00015")), "$150.00");
    }

    #[test]
    fn format_price_unknown() {
        assert_eq!(format_price(None), "—");
        assert_eq!(format_price(Some("")), "—");
        assert_eq!(format_price(Some("n/a")), "—");
        assert_eq!(format_price(Some("NaN")), "—");
    }

    #[test]
    fn is_free_requires_both_prices_zero() {
        assert!(is_free(&model_with_pricing(Some("0"), Some("0"))));
        assert!(!is_free(&model_with_pricing(Some("0"), Some("0.000001"))));
        assert!(!is_free(&model_with_pricing(Some("-1"), Some("-1"))));
    }

    #[test]
    fn is_free_treats_missing_prices_as_not_free() {
        assert!(!is_free(&model_with_pricing(None, None)));
        assert!(!is_free(&model_with_pricing(Some("0"), None)));
        assert!(!is_free(&model_with_pricing(None, Some("0"))));
        assert!(!is_free(&model_with_pricing(Some("garbage"), Some("0"))));

        let no_pricing = ModelInfo {
            pricing: None,
            ..model_with_pricing(None, None)
        };
        assert!(!is_free(&no_pricing));
    }

    #[test]
    fn format_context_units() {
        assert_eq!(format_context(512), "512");
        assert_eq!(format_context(8_192), "8k");
        assert_eq!(format_context(131_072), "131k");
        assert_eq!(format_context(1_000_000), "1M");
        assert_eq!(format_context(1_048_576), "1M");
        assert_eq!(format_context(2_000_000), "2M");
    }

    #[test]
    fn truncate_id_fits_column() {
        assert_eq!(truncate_id("openai/gpt-4o", 45), "openai/gpt-4o");

        let long = "some-vendor/an-exceptionally-long-model-identifier:free";
        let cut = truncate_id(long, 45);
        assert_eq!(cut.chars().count(), 45);
        assert!(cut.ends_with('…'));
        assert!(long.starts_with(cut.trim_end_matches('…')));

        let exact = "x".repeat(45);
        assert_eq!(truncate_id(&exact, 45), exact);
    }

    #[test]
    fn price_cell_pads_before_coloring() {
        // Whether or not colors are on in this environment, the visible text
        // is right-aligned to 12 columns: escape codes (if any) wrap the
        // padded cell instead of eating into the padding.
        let cell = price_cell("free", 12);
        assert!(cell.contains("        free"), "{cell:?}");
        assert_eq!(price_cell("$3.00", 12), "       $3.00");
        assert_eq!(price_cell("varies", 12), "      varies");
    }
}
