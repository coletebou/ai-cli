mod config;
mod config_ui;
mod display;
mod models;
mod providers;
mod query;
mod ui;

use anyhow::Result;
use clap::{Parser, Subcommand};
use config::Config;
use genai::chat::ReasoningEffort;
use providers::ProviderManager;
use std::process::Command as ProcessCommand;
use std::sync::Arc;

// Embedded at compile time by build.rs
const BUILD_COMMIT: &str = env!("AI_CLI_GIT_HASH");
const REPO_PATH: &str = env!("AI_CLI_REPO_PATH");

fn check_binary_staleness() {
    // Skip if repo path is empty or doesn't exist
    if REPO_PATH.is_empty() || !std::path::Path::new(REPO_PATH).exists() {
        return;
    }

    // Get current HEAD commit
    let output = ProcessCommand::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(REPO_PATH)
        .output()
        .ok();

    let current_hash = match output {
        Some(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        _ => return, // Can't determine current hash, skip check
    };

    // Compare with build commit
    if BUILD_COMMIT != "unknown" && BUILD_COMMIT != current_hash {
        eprintln!(
            "\x1b[33m[warn]\x1b[0m ai binary is stale (built: {}, current: {})",
            BUILD_COMMIT, current_hash
        );
        eprintln!(
            "\x1b[33m[warn]\x1b[0m Run 'ai-rebuild' or 'cargo install --path {}' to update\n",
            REPO_PATH
        );
    }
}

#[derive(Parser)]
#[command(name = "ai")]
#[command(version = "0.1.0")]
#[command(about = "Query multiple AI providers simultaneously")]
#[command(author = "Cole Tebou")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// The question to ask (multiple words joined automatically)
    question: Vec<String>,

    /// Generate default config file at ~/.config/ai-cli/config.toml
    #[arg(long)]
    init: bool,

    /// Providers to query (comma-separated, e.g., claude,gpt,grok,gemini)
    #[arg(short, long, value_delimiter = ',')]
    providers: Option<Vec<String>>,

    /// Query all configured providers
    #[arg(short, long)]
    all: bool,

    /// System prompt / persona for all providers
    #[arg(short, long)]
    system: Option<String>,

    /// Raw output (no formatting, pipeable)
    #[arg(long)]
    raw: bool,

    /// Disable streaming (wait for complete responses)
    #[arg(long)]
    no_stream: bool,

    /// Hide token usage statistics
    #[arg(long)]
    no_tokens: bool,

    /// Side-by-side TUI display
    #[arg(short = 't', long)]
    tui: bool,

    /// Disable TUI mode (use sequential display)
    #[arg(long)]
    no_tui: bool,

    /// Reasoning effort level (none, low, medium, high, or a number for budget)
    #[arg(short = 'r', long)]
    reasoning: Option<String>,
}

#[derive(Subcommand)]
enum Command {
    /// Open the TUI config editor
    Config,
    /// List available models from OpenRouter
    Models {
        /// Search/filter models by name or ID
        search: Option<String>,
        /// Force refresh the cached model list
        #[arg(long)]
        refresh: bool,
        /// Show only free models
        #[arg(long)]
        free: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Check if binary matches source (warns if stale)
    check_binary_staleness();

    let cli = Cli::parse();

    // Handle subcommands
    if let Some(command) = cli.command {
        return match command {
            Command::Config => config_ui::run_config_ui().await,
            Command::Models { search, refresh, free } => {
                models::list_models(search.as_deref(), refresh, free).await
            }
        };
    }

    // Handle --init flag
    if cli.init {
        Config::init()?;
        return Ok(());
    }

    // Require question if not --init
    if cli.question.is_empty() {
        anyhow::bail!("Missing required argument: <QUESTION>\n\nUsage: ai <QUESTION>\n       ai --init");
    }
    let question = cli.question.join(" ");

    // Load configuration
    let config = Config::load()?;

    // Create provider manager
    let manager = Arc::new(ProviderManager::new(config.clone(), config.cli.model_tier.clone())?);

    // Determine which providers to query based on CLI args and config
    let providers_to_query = if cli.all {
        // --all flag: query all enabled providers
        manager.get_providers_for_query(None)
    } else if let Some(ref requested) = cli.providers {
        // -p flag: query specific providers
        manager.get_providers_for_query(Some(requested))
    } else {
        // No flags: use cli mode settings (single vs multi)
        use config::QueryMode;
        match config.cli.default_mode {
            QueryMode::Single => {
                manager.get_providers_for_query(Some(&[config.cli.single_provider.clone()]))
            }
            QueryMode::Multi => {
                manager.get_providers_for_query(Some(&config.cli.multi_providers))
            }
        }
    };

    if providers_to_query.is_empty() {
        eprintln!("Error: No providers available. Please check your API keys and configuration.");
        eprintln!("Expected environment variables:");
        eprintln!("  - ANTHROPIC_API_KEY (for Claude)");
        eprintln!("  - OPENAI_API_KEY (for ChatGPT)");
        eprintln!("  - XAI_API_KEY (for Grok)");
        eprintln!("  - GEMINI_API_KEY or GOOGLE_API_KEY (for Gemini)");
        std::process::exit(1);
    }

    // Determine system prompt
    let system_prompt = cli.system.as_deref().or(config.default.system_prompt.as_deref());

    // Determine display options
    let _stream = !cli.no_stream && config.default.stream; // Reserved for future multi-provider streaming
    let use_tui = !cli.no_tui && (cli.tui || config.default.tui);
    let footer = config.footer.clone();

    // Parse reasoning effort and get display string
    let (reasoning_effort, reasoning_str) = match cli.reasoning.as_ref() {
        Some(r) => {
            let (effort, display) = match r.to_lowercase().as_str() {
                "none" | "off" | "0" => (ReasoningEffort::None, "none"),
                "low" | "l" | "1" => (ReasoningEffort::Low, "low"),
                "medium" | "med" | "m" | "2" => (ReasoningEffort::Medium, "medium"),
                "high" | "h" | "3" => (ReasoningEffort::High, "high"),
                s => {
                    // Try to parse as a budget number
                    if let Ok(budget) = s.parse::<u32>() {
                        (ReasoningEffort::Budget(budget), "budget")
                    } else {
                        eprintln!("Warning: Unknown reasoning level '{}', using Medium", s);
                        (ReasoningEffort::Medium, "medium")
                    }
                }
            };
            (Some(effort), Some(display.to_string()))
        }
        None => (None, None), // Default: no reasoning specified
    };

    if use_tui {
        // Phase 2: TUI mode
        crate::ui::run_tui(
            manager,
            providers_to_query,
            &question,
            system_prompt,
            footer.show_tokens,
            reasoning_effort,
        )
        .await?;
    } else if providers_to_query.len() == 1 && !cli.no_stream {
        // Single provider: always stream (no interleaving issue) unless explicitly disabled
        let rx = query::stream_all(
            Arc::clone(&manager),
            providers_to_query.clone(),
            &question,
            system_prompt,
            reasoning_effort,
        )
        .await;

        display::display_streaming(rx, &providers_to_query, &footer, cli.raw, reasoning_str.clone()).await;
    } else {
        // Non-streaming mode
        let responses = query::query_all(
            Arc::clone(&manager),
            providers_to_query,
            &question,
            system_prompt,
            reasoning_effort,
        )
        .await;

        if cli.raw {
            display::print_raw_responses(&responses);
        } else {
            display::print_responses(&responses, &footer, reasoning_str.as_deref());
        }
    }

    Ok(())
}
