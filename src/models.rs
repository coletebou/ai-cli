use crate::config::Config;
use anyhow::{Context, Result};
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;

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

fn cache_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("ai-cli")
        .join("models_cache.json")
}

fn cache_age_hours() -> Option<f64> {
    let path = cache_path();
    if !path.exists() {
        return None;
    }
    let metadata = fs::metadata(&path).ok()?;
    let modified = metadata.modified().ok()?;
    let elapsed = SystemTime::now().duration_since(modified).ok()?;
    Some(elapsed.as_secs_f64() / 3600.0)
}

fn load_cache() -> Option<Vec<ModelInfo>> {
    let content = fs::read_to_string(cache_path()).ok()?;
    let cache: ModelsCache = serde_json::from_str(&content).ok()?;
    Some(cache.models)
}

fn save_cache(models: &[ModelInfo]) -> Result<()> {
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
    Ok(())
}

async fn fetch_openrouter_models(api_key: Option<&str>) -> Result<Vec<ModelInfo>> {
    let client = reqwest::Client::new();
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

pub async fn list_models(search: Option<&str>, refresh: bool) -> Result<()> {
    let config = Config::load()?;
    let api_key = config.get_api_key("openrouter");

    // Use cache if fresh enough (< 24 hours) and not forcing refresh
    let models = if !refresh {
        if let Some(age) = cache_age_hours() {
            if age < 24.0 {
                if let Some(cached) = load_cache() {
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
                eprintln!("{} Cache expired, fetching fresh list...", "[fetch]".dimmed());
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

    // Filter by search term
    let filtered: Vec<&ModelInfo> = if let Some(query) = search {
        let q = query.to_lowercase();
        models
            .iter()
            .filter(|m| m.id.to_lowercase().contains(&q) || m.name.to_lowercase().contains(&q))
            .collect()
    } else {
        models.iter().collect()
    };

    if filtered.is_empty() {
        if let Some(query) = search {
            eprintln!("No models matching '{}'", query);
        } else {
            eprintln!("No models found.");
        }
        return Ok(());
    }

    // Print header
    println!(
        "\n{} {} models{}",
        "OpenRouter".yellow().bold(),
        filtered.len(),
        search
            .map(|s| format!(" matching '{}'", s))
            .unwrap_or_default()
    );
    println!(
        "{:<45} {:>8} {:>12} {:>12}",
        "MODEL ID".bold(),
        "CTX".bold(),
        "IN $/M".bold(),
        "OUT $/M".bold()
    );
    println!("{}", "─".repeat(80));

    for model in &filtered {
        let ctx = model
            .context_length
            .map(|c| format_context(c))
            .unwrap_or_else(|| "—".to_string());

        let (in_price, out_price) = match &model.pricing {
            Some(p) => (
                format_price(p.prompt.as_deref()),
                format_price(p.completion.as_deref()),
            ),
            None => ("—".to_string(), "—".to_string()),
        };

        println!("{:<45} {:>8} {:>12} {:>12}", model.id, ctx, in_price, out_price);
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
    if let Err(e) = save_cache(&models) {
        eprintln!("{} Failed to cache models: {}", "[warn]".yellow(), e);
    }
    eprintln!(
        "{} Fetched {} models, cached to {:?}",
        "[ok]".green(),
        models.len(),
        cache_path()
    );
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

fn format_price(price: Option<&str>) -> String {
    match price {
        Some(p) => {
            let val: f64 = p.parse().unwrap_or(0.0);
            if val == 0.0 {
                "free".green().to_string()
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
        None => "—".to_string(),
    }
}
