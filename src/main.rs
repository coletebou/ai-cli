mod config;
mod display;
mod providers;
mod query;
mod ui;

use anyhow::Result;
use clap::Parser;
use config::Config;
use providers::ProviderManager;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "ai")]
#[command(version = "0.1.0")]
#[command(about = "Query multiple AI providers simultaneously")]
#[command(author = "Cole Tebou")]
struct Cli {
    /// The question to ask
    question: String,

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

    /// Side-by-side TUI display (Phase 2)
    #[arg(short = 't', long)]
    tui: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Load configuration
    let config = Config::load()?;

    // Create provider manager
    let manager = Arc::new(ProviderManager::new(config.clone())?);

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
    let show_tokens = !cli.no_tokens && config.default.show_tokens;
    let stream = !cli.no_stream && config.default.stream;

    if cli.tui {
        // Phase 2: TUI mode
        crate::ui::run_tui(
            manager,
            providers_to_query,
            &cli.question,
            system_prompt,
            show_tokens,
        )
        .await?;
    } else if stream {
        // Streaming mode (default)
        let rx = query::stream_all(
            Arc::clone(&manager),
            providers_to_query.clone(),
            &cli.question,
            system_prompt,
        )
        .await;

        display::display_streaming(rx, &providers_to_query, show_tokens, cli.raw).await;
    } else {
        // Non-streaming mode
        let responses = query::query_all(
            Arc::clone(&manager),
            providers_to_query,
            &cli.question,
            system_prompt,
        )
        .await;

        if cli.raw {
            display::print_raw_responses(&responses);
        } else {
            display::print_responses(&responses, show_tokens);
        }
    }

    Ok(())
}
