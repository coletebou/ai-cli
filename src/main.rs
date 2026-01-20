mod config;
mod config_ui;
mod display;
mod providers;
mod query;
mod ui;

use anyhow::Result;
use clap::{Parser, Subcommand};
use config::Config;
use genai::chat::ReasoningEffort;
use providers::ProviderManager;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "ai")]
#[command(version = "0.1.0")]
#[command(about = "Query multiple AI providers simultaneously")]
#[command(author = "Cole Tebou")]
struct Cli {
    /// The question to ask
    question: Option<String>,

    #[command(subcommand)]
    command: Option<Command>,

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
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Handle subcommands
    if let Some(command) = cli.command {
        return match command {
            Command::Config => config_ui::run_config_ui().await,
        };
    }

    // Handle --init flag
    if cli.init {
        Config::init()?;
        return Ok(());
    }

    // Require question if not --init
    let question = cli.question.as_ref().ok_or_else(|| {
        anyhow::anyhow!("Missing required argument: <QUESTION>\n\nUsage: ai <QUESTION>\n       ai --init")
    })?;

    // Load configuration
    let config = Config::load()?;

    // Create provider manager
    let manager = Arc::new(ProviderManager::new(config.clone(), config.cli.model_tier.clone())?);

    // Determine which providers to query
    let providers_to_query = if cli.all {
        manager.get_providers_for_query(None)
    } else if let Some(ref requested) = cli.providers {
        manager.get_providers_for_query(Some(requested))
    } else {
        manager.get_providers_for_query(None)
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
    let stream = !cli.no_stream && config.default.stream;
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
            question,
            system_prompt,
            footer.show_tokens,
            reasoning_effort,
        )
        .await?;
    } else if stream && providers_to_query.len() == 1 {
        // Streaming mode - only for single provider to avoid interleaved output
        let rx = query::stream_all(
            Arc::clone(&manager),
            providers_to_query.clone(),
            question,
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
            question,
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
