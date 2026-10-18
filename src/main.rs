mod anthropic_oauth;
mod bedrock;
mod bedrock_eventstream;
mod chat;
mod config;
mod config_ui;
mod context;
mod diag;
mod display;
mod doctor;
mod markdown;
mod models;
mod pricing;
mod providers;
mod query;
mod retry;
mod search;
mod sessions;
mod telemetry;
mod term;
mod ui;

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::{generate, Shell};
use config::Config;
use genai::chat::ReasoningEffort;
use providers::{ProviderManager, ProviderRequest, SkipReason};
use std::ffi::OsString;
use std::io::IsTerminal;
use std::process::Command as ProcessCommand;
use std::process::ExitCode;
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

    /// Generate default config file at ~/.config/ai-cli/config.toml (or $AI_CLI_CONFIG)
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

    /// Hide token usage statistics (overrides `show_tokens` in the config)
    #[arg(long)]
    no_tokens: bool,

    /// Show token usage statistics (overrides `show_tokens` in the config)
    #[arg(long, conflicts_with = "no_tokens")]
    tokens: bool,

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

    /// Hide the estimated cost in the footer (overrides `show_cost` in the config)
    #[arg(long)]
    no_cost: bool,

    /// Never read stdin as context (for scripts, loops, and subprocesses)
    #[arg(long)]
    no_stdin: bool,

    /// Send @path tokens as literal text instead of inlining files
    #[arg(long)]
    no_expand: bool,

    /// With -i: resume the most recently saved chat session
    #[arg(short = 'c', long = "continue", requires = "interactive")]
    continue_session: bool,
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
    /// Diagnose config, credentials and per-provider connectivity
    Doctor {
        /// Skip the network probes (config and credential checks only)
        #[arg(long)]
        offline: bool,
    },
}

/// Parse argv, treating a question that merely *starts* with a subcommand
/// name as a question. Unquoted `ai models are getting cheaper, why?` or
/// `ai help me write a regex` used to die with a subcommand usage error.
fn parse_cli() -> Cli {
    let args: Vec<OsString> = std::env::args_os().collect();
    match Cli::try_parse_from(&args) {
        Ok(cli) => cli,
        Err(err) => question_retry(&args, &err).unwrap_or_else(|| err.exit()),
    }
}

/// If parsing failed on a word that names a subcommand and only plain words
/// follow it, re-parse with `--` before that word so the rest is taken as
/// the question. A flag after the word (e.g. `ai models --fre`) suggests the
/// subcommand really was meant, so that keeps clap's error.
fn question_retry(args: &[OsString], err: &clap::Error) -> Option<Cli> {
    use clap::error::ErrorKind;
    // Only errors that mean "more words than this subcommand takes". A bad
    // value (e.g. `ai completions bas`) or a missing argument is a real
    // usage mistake: turning it into a prompt would silently send a billed
    // request instead of showing clap's diagnostic.
    if !matches!(
        err.kind(),
        ErrorKind::UnknownArgument | ErrorKind::InvalidSubcommand
    ) {
        return None;
    }
    // `completions` takes a fixed shell name, so extra words there are a
    // mistake too, never a question.
    let mut subcommands: Vec<String> = Cli::command()
        .get_subcommands()
        .map(|s| s.get_name().to_string())
        .filter(|name| name != "completions")
        .collect();
    subcommands.push("help".to_string());

    for i in 1..args.len() {
        let word = args[i].to_string_lossy();
        if !subcommands.iter().any(|s| *s == word) {
            continue;
        }
        let rest = &args[i + 1..];
        if rest.is_empty() || rest.iter().any(|a| a.to_string_lossy().starts_with('-')) {
            return None;
        }
        let mut retry = args[..i].to_vec();
        retry.push(OsString::from("--"));
        retry.extend_from_slice(&args[i..]);
        return Cli::try_parse_from(retry)
            .ok()
            .filter(|cli| cli.command.is_none());
    }
    None
}

#[tokio::main]
async fn main() -> Result<ExitCode> {
    // Check if binary matches source (warns if stale)
    check_binary_staleness();

    // Initialize OpenTelemetry → Langfuse if creds are present. No-op
    // otherwise. We capture whether init succeeded so we know whether to
    // bother flushing on exit.
    let telemetry_enabled = telemetry::init().is_some();

    let result = run(parse_cli()).await;

    // Flush pending spans before the process exits. Without this, short-lived
    // CLI runs lose their last batch — Langfuse never sees the trace.
    if telemetry_enabled {
        telemetry::shutdown();
    }
    result
}

/// Exit status for a finished run: non-zero when any provider failed, so
/// `ai "..." || fallback` and `set -e` scripts can tell something broke.
fn exit_status(any_failed: bool) -> ExitCode {
    if any_failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// Which settings and which display a one-shot run uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Mode {
    /// Take the providers and model tier from `[tui_mode]` instead of `[cli]`.
    tui_settings: bool,
    /// Draw the TUI; otherwise responses are printed as plain output.
    tui_display: bool,
}

/// Pick the mode for this run. `-i` is its own mode and never picks up the
/// TUI's providers or tier, and `--no-tui` asks for the CLI mode, `[cli]`
/// settings included.
///
/// A TUI that only comes from `[default] tui = true` falls back to plain
/// output when stdout isn't a terminal, so `ai "q" | less` works. Only the
/// display falls back: the run keeps `[tui_mode]`'s providers and tier, so
/// piping or redirecting a question gets the same answers as asking it on
/// screen. The user didn't ask for the CLI mode; the TUI just can't be drawn.
/// An explicit `-t` never falls back, so it still errors without a terminal
/// and the mistake is visible.
fn select_mode(cli: &Cli, config_tui: bool, stdout_is_terminal: bool) -> Mode {
    let tui_requested = !cli.interactive && !cli.no_tui && (cli.tui || config_tui);
    Mode {
        tui_settings: tui_requested,
        tui_display: tui_requested && (cli.tui || stdout_is_terminal),
    }
}

/// Parse `-r/--reasoning`. Returns the effort plus the label shown in the
/// footer. Unknown values are an error rather than a silent (and pricier)
/// fallback to medium.
fn parse_reasoning(value: &str) -> Result<(ReasoningEffort, String)> {
    let parsed = match value.to_lowercase().as_str() {
        "none" | "off" | "0" => (ReasoningEffort::None, "none".to_string()),
        "low" | "l" | "1" => (ReasoningEffort::Low, "low".to_string()),
        "medium" | "med" | "m" | "2" => (ReasoningEffort::Medium, "medium".to_string()),
        "high" | "h" | "3" => (ReasoningEffort::High, "high".to_string()),
        s => match s.parse::<u32>() {
            Ok(budget) => (
                ReasoningEffort::Budget(budget),
                format!("budget {}", budget),
            ),
            Err(_) => anyhow::bail!(
                "unknown reasoning level '{}': use none, low, medium, high, or a token budget \
                 (a number; 0-3 are shorthand for none..high)",
                value
            ),
        },
    };
    Ok(parsed)
}

/// Explain an empty provider selection: why each requested provider was
/// skipped, and which env vars would enable the configured ones.
fn no_providers_message(manager: &ProviderManager, skipped: &[(String, SkipReason)]) -> String {
    let mut msg = String::from("No providers available.\n");
    if !skipped.is_empty() {
        msg.push('\n');
        for (name, reason) in skipped {
            msg.push_str(&format!("  {}: {}\n", name, reason));
        }
    }
    if skipped
        .iter()
        .any(|(_, reason)| *reason == SkipReason::Unknown)
    {
        msg.push_str(&format!(
            "\nKnown providers: {}\n",
            manager.known_provider_names().join(", ")
        ));
    }
    msg.push_str(
        "\nSet an API key for at least one provider (e.g. ANTHROPIC_API_KEY, OPENAI_API_KEY, \
         XAI_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY), or run `ai --init` / `ai config`.",
    );
    msg
}

async fn run(cli: Cli) -> Result<ExitCode> {
    // Handle subcommands
    if let Some(command) = cli.command {
        match command {
            Command::Config => config_ui::run_config_ui().await?,
            Command::Models {
                search,
                refresh,
                free,
            } => models::list_models(search.as_deref(), refresh, free).await?,
            Command::Completions { shell } => {
                let mut cmd = Cli::command();
                let bin_name = cmd.get_name().to_string();
                generate(shell, &mut cmd, bin_name, &mut std::io::stdout());
            }
            Command::Doctor { offline } => doctor::run(offline).await?,
        }
        return Ok(ExitCode::SUCCESS);
    }

    // Handle --init flag
    if cli.init {
        Config::init()?;
        return Ok(ExitCode::SUCCESS);
    }

    // Gather context from piped stdin and inline @file references so the
    // user can do `cat notes.md | ai "summarize this"` or
    // `ai "explain @src/foo.rs"` without hand-rolling the prompt.
    let stdin = if cli.no_stdin {
        None
    } else {
        context::read_stdin_if_piped()?
    };
    let expanded = context::question_from_args(&cli.question, !cli.no_expand)?;
    let question = context::merge_stdin(&expanded, stdin);

    // Load configuration
    let config = Config::load()?;

    // Parse reasoning effort and get display string. Done before provider
    // selection so a bad flag is reported as such, not masked by a
    // missing-API-key error.
    let (reasoning_effort, reasoning_str) = match cli.reasoning.as_deref() {
        Some(r) => {
            let (effort, label) = parse_reasoning(r)?;
            (Some(effort), Some(label))
        }
        None => (None, None), // Default: no reasoning specified
    };

    // Determine the mode first — it affects provider and model tier selection
    // as well as the display (see `select_mode` for the non-terminal fallback).
    let mode = select_mode(&cli, config.default.tui, std::io::stdout().is_terminal());
    let use_tui = mode.tui_display;

    // Require a question only for one-shot modes. Interactive chat (-i) and
    // the TUI both now launch into a prompt/input box, so an empty question
    // is a valid "open the UI and wait" invocation.
    if question.trim().is_empty() && !cli.interactive && !use_tui {
        anyhow::bail!(
            "Missing required argument: <QUESTION>\n\nUsage: ai <QUESTION>\n       cat file | ai <QUESTION>\n       ai -i [QUESTION]\n       ai -t [QUESTION]\n       ai --init"
        );
    }
    let mode_config = if mode.tui_settings {
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
    let request = if cli.all {
        ProviderRequest::All
    } else if let Some(ref requested) = cli.providers {
        ProviderRequest::Explicit(requested)
    } else {
        use config::QueryMode;
        match mode_config.default_mode {
            QueryMode::Single => {
                ProviderRequest::Configured(std::slice::from_ref(&mode_config.single_provider))
            }
            QueryMode::Multi => ProviderRequest::Configured(&mode_config.multi_providers),
        }
    };
    let selection = manager.select_providers(request);

    if selection.providers.is_empty() {
        anyhow::bail!(no_providers_message(&manager, &selection.skipped));
    }

    // Say why something the user asked for by name was dropped. Providers
    // missing from the configured multi list are skipped quietly (a default
    // setup lists four providers and most people have keys for one or two),
    // but a typo'd name in the config is still worth a note.
    for (name, reason) in &selection.skipped {
        let explicit = matches!(request, ProviderRequest::Explicit(_));
        if explicit || *reason == SkipReason::Unknown {
            eprintln!("note: skipping {}: {}", name, reason);
        }
    }
    let providers_to_query = selection.providers;

    // Determine system prompt
    let system_prompt = cli
        .system
        .as_deref()
        .or(config.default.system_prompt.as_deref());

    // Determine display options. `[default] stream = false` and --no-stream
    // both switch to waiting for complete responses.
    let stream = !cli.no_stream && config.default.stream;
    let mut footer = config.footer.clone();
    if cli.no_tokens {
        footer.show_tokens = false;
    } else if cli.tokens {
        footer.show_tokens = true;
    }
    // --no-cost hides the estimate; `[pricing]` adds or overrides prices.
    footer.show_cost &= !cli.no_cost;
    footer.pricing = config.pricing.clone();

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
        // Returned as an error (not process::exit) so telemetry still flushes.
        let primary = match build(&primary_name) {
            Some(p) => p,
            None => match primary_name.as_str() {
                "gemini" => anyhow::bail!(
                    "search provider 'gemini' requires {} or {} to be set.",
                    config.search.gemini.api_key_env,
                    config.search.gemini.fallback_api_key_env,
                ),
                "tavily" => anyhow::bail!(
                    "search provider 'tavily' requires {} to be set.\n\
                     Get a free Tavily API key at https://tavily.com",
                    config.search.tavily.api_key_env,
                ),
                other => anyhow::bail!(
                    "unknown search provider '{}'. Valid options: gemini, tavily.",
                    other
                ),
            },
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
    // run() returns) detaches the context, and `_root_end` ends the span on
    // every return path. We can't rely on the span ending when its last
    // Context drops: detached tasks (stream_all, TUI panels) hold clones
    // that can outlive telemetry::shutdown(), which would lose the root.
    use opentelemetry::trace::{TraceContextExt, Tracer};
    let tracer = opentelemetry::global::tracer(telemetry::TRACER_NAME);
    let mut root_attrs = vec![opentelemetry::KeyValue::new(
        "langfuse.trace.name",
        if question.is_empty() {
            "interactive session".to_string()
        } else if telemetry::capture_content() {
            truncate_for_trace_name(&question)
        } else {
            // Content capture is off: the name must not leak the prompt.
            "query".to_string()
        },
    )];
    if let Some(input) = telemetry::content_attr(&question) {
        root_attrs.push(opentelemetry::KeyValue::new("langfuse.trace.input", input));
    }
    let root_span = tracer
        .span_builder("ai.query")
        .with_attributes(root_attrs)
        .start(&tracer);
    let cx = opentelemetry::Context::current_with_span(root_span);
    let _root_end = telemetry::EndSpanOnDrop(cx.clone());
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
        crate::chat::run_interactive(
            manager,
            provider,
            ctx,
            footer,
            initial,
            reasoning_str,
            cli.continue_session,
        )
        .await?;
        return Ok(ExitCode::SUCCESS);
    }

    let any_failed = if use_tui {
        // Phase 2: TUI mode. Per-panel errors are shown in the panels; the
        // exit status only reflects whether the TUI itself failed.
        crate::ui::run_tui(
            manager,
            providers_to_query,
            ctx,
            &question,
            footer.show_tokens,
        )
        .await?;
        false
    } else if stream && (providers_to_query.len() == 1 || !cli.raw) {
        // One provider renders progressively as it streams. Several
        // providers each print as one block the moment they finish, so the
        // fastest answer shows first instead of waiting for the slowest.
        // (Raw multi-provider output stays non-streaming below, so its
        // order is deterministic for scripts.)
        let rx = query::stream_all(Arc::clone(&manager), providers_to_query.clone(), ctx).await;

        display::display_streaming(
            rx,
            &providers_to_query,
            &footer,
            cli.raw,
            reasoning_str.clone(),
        )
        .await
    } else {
        // Non-streaming mode
        let responses = query::query_all(Arc::clone(&manager), providers_to_query, ctx).await;

        if cli.raw {
            display::print_raw_responses(&responses);
        } else {
            display::print_responses(&responses, &footer, reasoning_str.as_deref());
        }
        responses.iter().any(|r| r.is_error())
    };

    Ok(exit_status(any_failed))
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    fn parse(words: &[&str]) -> Option<Cli> {
        let args: Vec<OsString> = std::iter::once("ai")
            .chain(words.iter().copied())
            .map(OsString::from)
            .collect();
        match Cli::try_parse_from(&args) {
            Ok(cli) => Some(cli),
            Err(err) => question_retry(&args, &err),
        }
    }

    #[test]
    fn questions_starting_with_a_subcommand_word_are_questions() {
        let cli = parse(&["models", "are", "getting", "cheaper?"]).expect("parses");
        assert!(cli.command.is_none());
        assert_eq!(cli.question.join(" "), "models are getting cheaper?");

        let cli = parse(&["help", "me", "write", "a", "regex"]).expect("parses");
        assert!(cli.command.is_none());
        assert_eq!(cli.question[0], "help");

        let cli = parse(&["-p", "claude", "config", "files", "vs", "env", "vars"]).expect("parses");
        assert_eq!(cli.question.join(" "), "config files vs env vars");
        assert_eq!(cli.providers.as_deref(), Some(&["claude".to_string()][..]));
    }

    #[test]
    fn real_subcommands_still_parse() {
        assert!(matches!(
            parse(&["models", "llama"]).and_then(|c| c.command),
            Some(Command::Models { .. })
        ));
        assert!(matches!(
            parse(&["config"]).and_then(|c| c.command),
            Some(Command::Config)
        ));
        // A flag after the subcommand means the subcommand was intended, so
        // a typo there stays a usage error instead of becoming a prompt.
        assert!(parse(&["models", "--fre"]).is_none());
        // Bad or extra values for a structured subcommand stay errors too:
        // they must never turn into a (billed) question.
        assert!(parse(&["completions", "bas"]).is_none());
        assert!(parse(&["completions", "zsh", "extra"]).is_none());
    }

    #[test]
    fn reasoning_levels_parse_and_unknown_values_error() {
        assert_eq!(parse_reasoning("high").unwrap().1, "high");
        assert_eq!(parse_reasoning("2").unwrap().1, "medium");
        assert_eq!(parse_reasoning("4096").unwrap().1, "budget 4096");
        assert!(matches!(
            parse_reasoning("4096").unwrap().0,
            ReasoningEffort::Budget(4096)
        ));
        // Used to silently fall back to Medium.
        assert!(parse_reasoning("hihg").is_err());
    }

    #[test]
    fn tokens_flags_conflict() {
        assert!(Cli::try_parse_from(["ai", "--tokens", "--no-tokens", "q"]).is_err());
    }

    #[test]
    fn nonterminal_tui_fallback_changes_only_the_display() {
        const TUI: Mode = Mode {
            tui_settings: true,
            tui_display: true,
        };
        const TUI_SETTINGS_PLAIN_OUTPUT: Mode = Mode {
            tui_settings: true,
            tui_display: false,
        };
        const CLI: Mode = Mode {
            tui_settings: false,
            tui_display: false,
        };
        let q = parse(&["q"]).expect("parses");
        // `tui = true` with stdout piped: plain output, `[tui_mode]` settings.
        assert_eq!(select_mode(&q, true, false), TUI_SETTINGS_PLAIN_OUTPUT);
        assert_eq!(select_mode(&q, true, true), TUI);
        assert_eq!(select_mode(&q, false, true), CLI);
        // An explicit `-t` never falls back.
        let t = parse(&["-t", "q"]).expect("parses");
        assert_eq!(select_mode(&t, false, false), TUI);
        // `--no-tui` and `-i` pick the CLI settings even with `tui = true`.
        for flag in ["--no-tui", "-i"] {
            let cli = parse(&[flag, "q"]).expect("parses");
            assert_eq!(select_mode(&cli, true, true), CLI, "{flag}");
        }
    }
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
