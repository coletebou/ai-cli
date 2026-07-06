mod anthropic_oauth;
mod bedrock;
mod bedrock_eventstream;
mod chat;
mod config;
mod config_ui;
mod context;
mod display;
mod markdown;
mod models;
mod providers;
mod query;
mod retry;
mod search;
mod telemetry;
mod ui;

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::{generate, Shell};
use config::Config;
use genai::chat::ReasoningEffort;
use providers::ProviderManager;
use std::process::Command as ProcessCommand;
use std::sync::Arc;

// Embedded at compile time by build.rs
const BUILD_COMMIT: &str = env!("AI_CLI_GIT_HASH");
const REPO_PATH: &str = env!("AI_CLI_REPO_PATH");

/// Shorten a free-form question into something suitable for a trace name in
/// Langfuse. The trace list view truncates anyway; we just want the leading
/// 60 chars so different invocations are visually distinguishable.
fn truncate_for_trace_name(question: &str) -> String {
    const MAX: usize = 60;
    let trimmed = question.trim();
    if trimmed.chars().count() <= MAX {
        trimmed.to_string()
    } else {
        let prefix: String = trimmed.chars().take(MAX).collect();
        format!("{}…", prefix)
    }
}

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

    /// Enable web search (uses provider from config; override with --search-provider)
    #[arg(short = 'w', long)]
    search: bool,

    /// Override search provider for this query ("gemini" or "tavily")
    #[arg(long, value_name = "PROVIDER")]
    search_provider: Option<String>,

    /// Interactive chat mode (follow-up questions with conversation history)
    #[arg(short = 'i', long)]
    interactive: bool,
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
    /// Generate shell completion script (bash, zsh, fish, elvish, powershell)
    Completions {
        /// Shell to generate completions for
        shell: Shell,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Check if binary matches source (warns if stale)
    check_binary_staleness();

    // Initialize OpenTelemetry → Langfuse if creds are present. No-op
    // otherwise. We capture whether init succeeded so we know whether to
    // bother flushing on exit.
    let telemetry_enabled = telemetry::init().is_some();

    let result = run(Cli::parse()).await;

    // Flush pending spans before the process exits. Without this, short-lived
    // CLI runs lose their last batch — Langfuse never sees the trace.
    if telemetry_enabled {
        telemetry::shutdown();
    }
    result
}

async fn run(cli: Cli) -> Result<()> {
    // Handle subcommands
    if let Some(command) = cli.command {
        return match command {
            Command::Config => config_ui::run_config_ui().await,
            Command::Models {
                search,
                refresh,
                free,
            } => models::list_models(search.as_deref(), refresh, free).await,
            Command::Completions { shell } => {
                let mut cmd = Cli::command();
                let bin_name = cmd.get_name().to_string();
                generate(shell, &mut cmd, bin_name, &mut std::io::stdout());
                Ok(())
            }
        };
    }

    // Handle --init flag
    if cli.init {
        Config::init()?;
        return Ok(());
    }

    // Gather context from piped stdin and inline @file references so the
    // user can do `cat notes.md | ai "summarize this"` or
    // `ai "explain @src/foo.rs"` without hand-rolling the prompt.
    let stdin = context::read_stdin_if_piped();
    let raw_question = cli.question.join(" ");
    let expanded = context::expand_file_refs(&raw_question)?;
    let question = context::merge_stdin(&expanded, stdin);

    // Load configuration
    let config = Config::load()?;

    // Determine TUI mode first — it affects provider and model tier selection
    let use_tui = !cli.no_tui && (cli.tui || config.default.tui);

    // Require a question only for one-shot modes. Interactive chat (-i) and
    // the TUI both now launch into a prompt/input box, so an empty question
    // is a valid "open the UI and wait" invocation.
    if question.trim().is_empty() && !cli.interactive && !use_tui {
        anyhow::bail!(
            "Missing required argument: <QUESTION>\n\nUsage: ai <QUESTION>\n       cat file | ai <QUESTION>\n       ai -i [QUESTION]\n       ai -t [QUESTION]\n       ai --init"
        );
    }
    let mode_config = if use_tui {
        &config.tui_mode
    } else {
        &config.cli
    };

    // Create provider manager with the correct model tier for the active mode
    let manager = Arc::new(ProviderManager::new(
        config.clone(),
        mode_config.model_tier,
    )?);

    // Determine which providers to query based on CLI args and active mode config
    let providers_to_query = if cli.all {
        manager.get_providers_for_query(None)
    } else if let Some(ref requested) = cli.providers {
        manager.get_providers_for_query(Some(requested))
    } else {
        use config::QueryMode;
        match mode_config.default_mode {
            QueryMode::Single => manager
                .get_providers_for_query(Some(std::slice::from_ref(&mode_config.single_provider))),
            QueryMode::Multi => manager.get_providers_for_query(Some(&mode_config.multi_providers)),
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
    let system_prompt = cli
        .system
        .as_deref()
        .or(config.default.system_prompt.as_deref());

    // Determine display options
    let _stream = !cli.no_stream && config.default.stream; // Reserved for future multi-provider streaming
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

    // Build search engine (primary + optional fallback) if --search flag set
    let search_client = if cli.search {
        // Determine primary: CLI override > config
        let primary_name = cli
            .search_provider
            .as_deref()
            .unwrap_or(&config.search.provider)
            .to_lowercase();

        // Build a named provider, returning None if its key isn't available
        let build = |name: &str| -> Option<search::SearchProvider> {
            match name {
                "gemini" => search::GeminiSearchClient::from_env(
                    &config.search.gemini.api_key_env,
                    &config.search.gemini.fallback_api_key_env,
                    &config.search.gemini.model,
                )
                .map(search::SearchProvider::Gemini),
                "tavily" => search::TavilyClient::from_env(
                    &config.search.tavily.api_key_env,
                    config.search.max_results,
                )
                .map(search::SearchProvider::Tavily),
                _ => None,
            }
        };

        // Primary must build successfully — user explicitly chose it
        let primary = match build(&primary_name) {
            Some(p) => p,
            None => {
                match primary_name.as_str() {
                    "gemini" => {
                        eprintln!(
                            "Error: search provider 'gemini' requires {} or {} to be set.",
                            config.search.gemini.api_key_env,
                            config.search.gemini.fallback_api_key_env,
                        );
                    }
                    "tavily" => {
                        eprintln!(
                            "Error: search provider 'tavily' requires {} to be set.",
                            config.search.tavily.api_key_env,
                        );
                        eprintln!("Get a free Tavily API key at https://tavily.com");
                    }
                    other => {
                        eprintln!(
                            "Error: unknown search provider '{}'. Valid options: gemini, tavily.",
                            other
                        );
                    }
                }
                std::process::exit(1);
            }
        };

        // Fallback: whichever provider the primary isn't, if its key is available
        let fallback_name = if primary_name == "gemini" {
            "tavily"
        } else {
            "gemini"
        };
        let fallback = build(fallback_name);

        Some(Arc::new(search::SearchEngine::new(primary, fallback)))
    } else {
        None
    };

    // Build query context
    let mut ctx = query::QueryContext::new(&question)
        .with_system_prompt(system_prompt)
        .with_reasoning(reasoning_effort);
    if let Some(client) = search_client {
        ctx = ctx.with_search(client, config.search.max_iterations);
    }

    // Root span for the whole CLI invocation. All per-provider spans created
    // during dispatch become children of this. Dropping `_root_guard` (when
    // run() returns) detaches the context; dropping `root_span` (when this
    // function's locals go out of scope) ends the span itself.
    use opentelemetry::trace::{TraceContextExt, Tracer};
    let tracer = opentelemetry::global::tracer(telemetry::TRACER_NAME);
    let root_span = tracer
        .span_builder("ai.query")
        .with_attributes(vec![
            opentelemetry::KeyValue::new("langfuse.trace.input", question.clone()),
            opentelemetry::KeyValue::new(
                "langfuse.trace.name",
                if question.is_empty() {
                    "interactive session".to_string()
                } else {
                    truncate_for_trace_name(&question)
                },
            ),
        ])
        .start(&tracer);
    let cx = opentelemetry::Context::current_with_span(root_span);
    let _root_guard = cx.attach();

    // Interactive chat mode takes precedence over other dispatch modes.
    // Multi-provider chat is confusing; pick the first viable provider.
    if cli.interactive {
        let provider = providers_to_query
            .first()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("no provider available for chat"))?;
        if providers_to_query.len() > 1 {
            eprintln!(
                "Note: chat mode is single-provider; using {} (ignoring others)",
                provider.display_name
            );
        }
        let initial = if question.is_empty() {
            None
        } else {
            Some(question.clone())
        };
        return crate::chat::run_interactive(manager, provider, ctx, footer, initial).await;
    }

    if use_tui {
        // Phase 2: TUI mode
        crate::ui::run_tui(
            manager,
            providers_to_query,
            ctx,
            &question,
            footer.show_tokens,
        )
        .await?;
    } else if providers_to_query.len() == 1 && !cli.no_stream {
        // Single provider: always stream (no interleaving issue) unless explicitly disabled
        let rx = query::stream_all(Arc::clone(&manager), providers_to_query.clone(), ctx).await;

        display::display_streaming(
            rx,
            &providers_to_query,
            &footer,
            cli.raw,
            reasoning_str.clone(),
        )
        .await;
    } else {
        // Non-streaming mode
        let responses = query::query_all(Arc::clone(&manager), providers_to_query, ctx).await;

        if cli.raw {
            display::print_raw_responses(&responses);
        } else {
            display::print_responses(&responses, &footer, reasoning_str.as_deref());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::truncate_for_trace_name;

    #[test]
    fn trace_name_passthrough_for_short_inputs() {
        assert_eq!(truncate_for_trace_name("hi"), "hi");
    }

    #[test]
    fn trace_name_trims_whitespace() {
        assert_eq!(truncate_for_trace_name("  hi  "), "hi");
    }

    #[test]
    fn trace_name_truncates_long_input_with_ellipsis() {
        let long = "a".repeat(120);
        let out = truncate_for_trace_name(&long);
        assert_eq!(out.chars().count(), 61); // 60 chars + ellipsis
        assert!(out.ends_with('…'));
    }

    #[test]
    fn trace_name_handles_unicode_correctly() {
        // Char-count truncation, not byte-count: a 60-emoji input should
        // pass through unchanged (still ≤60 chars). Byte-based truncation
        // would have split a multi-byte sequence and panicked.
        let emoji = "🎉".repeat(60);
        assert_eq!(truncate_for_trace_name(&emoji), emoji);
    }
}
