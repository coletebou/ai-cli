use crate::config::FooterConfig;
use crate::display::StreamingDisplay;
use crate::providers::{Provider, ProviderManager};
use crate::query::{stream_with_messages, QueryContext, StreamEvent};
use anyhow::{Context, Result};
use colored::Colorize;
use genai::chat::{ChatMessage, ChatRole, ContentPart};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Slash command recognized in the interactive prompt.
#[derive(Debug)]
enum Command {
    Exit,
    Clear,
    History,
    Help,
    /// Write the conversation so far as markdown. `None` picks an
    /// auto-named file in the CWD (`chat-<timestamp>.md`).
    Export(Option<PathBuf>),
    Unknown(String),
}

fn parse_command(line: &str) -> Option<Command> {
    // strip_prefix removes exactly one leading '/', so `//history` becomes an
    // unknown command rather than silently aliasing to History.
    let body = line.trim().strip_prefix('/')?;
    let mut parts = body.splitn(2, char::is_whitespace);
    let cmd = parts.next().unwrap_or("").to_lowercase();
    let rest = parts.next().unwrap_or("").trim();
    Some(match cmd.as_str() {
        "exit" | "quit" | "q" => Command::Exit,
        "clear" | "reset" => Command::Clear,
        "history" | "log" => Command::History,
        "help" | "?" => Command::Help,
        "export" | "save" => {
            if rest.is_empty() {
                Command::Export(None)
            } else {
                Command::Export(Some(PathBuf::from(rest)))
            }
        }
        other => Command::Unknown(other.to_string()),
    })
}

fn print_banner(provider: &Provider, ctx: &QueryContext) {
    println!();
    println!(
        "{} {} {}",
        "Starting chat with".dimmed(),
        provider.display_name.color(provider.color).bold(),
        format!("({})", provider.model).dimmed(),
    );
    if ctx.search_client.is_some() {
        println!("  {} {}", "web search:".dimmed(), "enabled".green());
    }
    if ctx.system_prompt.is_some() {
        println!("  {} {}", "system prompt:".dimmed(), "set".green());
    }
    println!(
        "{}",
        "Type a message, /help for commands, /exit or Ctrl+D to quit.".dimmed()
    );
    println!();
}

fn print_help() {
    println!();
    println!("{}", "Commands:".bold());
    println!("  {}  — leave chat", "/exit, /quit".cyan());
    println!(
        "  {}  — clear conversation history (keeps system prompt)",
        "/clear, /reset".cyan()
    );
    println!("  {}  — print conversation so far", "/history".cyan());
    println!(
        "  {}  — save conversation as markdown (auto-named if omitted)",
        "/export [path.md]".cyan()
    );
    println!("  {}  — show this help", "/help".cyan());
    println!();
}

/// Current Unix timestamp in seconds, or 0 on the unfathomable-edge case
/// of a clock before the epoch.
fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Render the conversation as markdown with `##` headings per message. Tool
/// calls and responses round-trip as fenced blocks so a later reader can
/// reconstruct the full context, not just the user-visible text. The caller
/// supplies the timestamp so the title and any auto-named file can share a
/// single value (avoiding off-by-one drift across a second boundary).
fn render_markdown(messages: &[ChatMessage], unix_ts: u64) -> String {
    let mut out = String::new();
    out.push_str(&format!("# Chat export (unix {})\n\n", unix_ts));

    for msg in messages {
        let heading = match msg.role {
            ChatRole::System => "## System",
            ChatRole::User => "## User",
            ChatRole::Assistant => "## Assistant",
            ChatRole::Tool => "## Tool",
        };
        out.push_str(heading);
        out.push_str("\n\n");

        for part in msg.content.parts() {
            match part {
                ContentPart::Text(t) => {
                    out.push_str(t);
                    out.push_str("\n\n");
                }
                ContentPart::ToolCall(tc) => {
                    // Build the object via serde_json so any quote/backslash
                    // in the tool name or args is escaped correctly — direct
                    // string interpolation would emit malformed JSON.
                    let obj = serde_json::json!({
                        "tool": tc.fn_name,
                        "args": tc.fn_arguments,
                    });
                    out.push_str(&format!("```json\n{}\n```\n\n", obj));
                }
                ContentPart::ToolResponse(tr) => {
                    out.push_str("```\n");
                    out.push_str(&tr.content);
                    if !tr.content.ends_with('\n') {
                        out.push('\n');
                    }
                    out.push_str("```\n\n");
                }
                _ => {}
            }
        }
    }
    out
}

fn export_history(messages: &[ChatMessage], path: &Path, unix_ts: u64) -> Result<()> {
    let content = render_markdown(messages, unix_ts);
    std::fs::write(path, content)
        .with_context(|| format!("failed to write chat export to {}", path.display()))?;
    Ok(())
}

/// Maximum characters of each message body shown in `/history`. Longer
/// messages are truncated with a `… (+N chars)` suffix.
const HISTORY_MAX_CHARS: usize = 240;

/// One-line summary of a message's content for `/history` output.
/// Collapses whitespace, renders tool calls/responses as bracketed markers,
/// and truncates to `HISTORY_MAX_CHARS`.
fn summarize_content(msg: &ChatMessage) -> String {
    let mut pieces: Vec<String> = Vec::new();
    for part in msg.content.parts() {
        match part {
            ContentPart::Text(t) => pieces.push(t.clone()),
            ContentPart::ToolCall(tc) => {
                let arg_preview = tc
                    .fn_arguments
                    .get("query")
                    .and_then(|v| v.as_str())
                    .map(|s| format!("{:?}", s))
                    .unwrap_or_else(|| tc.fn_arguments.to_string());
                pieces.push(format!("[→ {}({})]", tc.fn_name, arg_preview));
            }
            ContentPart::ToolResponse(tr) => {
                pieces.push(format!("[← tool result: {} chars]", tr.content.len()));
            }
            _ => {}
        }
    }
    let joined = pieces
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");

    let total_chars = joined.chars().count();
    if total_chars > HISTORY_MAX_CHARS {
        let head: String = joined.chars().take(HISTORY_MAX_CHARS).collect();
        format!(
            "{}{}",
            head,
            format!("… (+{} chars)", total_chars - HISTORY_MAX_CHARS).dimmed()
        )
    } else {
        joined
    }
}

fn print_history(messages: &[ChatMessage]) {
    println!();
    if messages.is_empty() {
        println!("{}", "history: (empty)".dimmed());
        println!();
        return;
    }

    for (i, msg) in messages.iter().enumerate() {
        let (label, color) = match msg.role {
            ChatRole::System => ("system", colored::Color::Yellow),
            ChatRole::User => ("user", colored::Color::Cyan),
            ChatRole::Assistant => ("assistant", colored::Color::Green),
            ChatRole::Tool => ("tool", colored::Color::Magenta),
        };
        let body = summarize_content(msg);
        println!(
            "  {:>2}. [{}] {}",
            i + 1,
            format!("{:<9}", label).color(color).bold(),
            body
        );
    }

    let user_count = messages
        .iter()
        .filter(|m| matches!(m.role, ChatRole::User))
        .count();
    let assistant_count = messages
        .iter()
        .filter(|m| matches!(m.role, ChatRole::Assistant))
        .count();
    println!();
    println!(
        "  {} {} user, {} assistant",
        "total:".dimmed(),
        user_count,
        assistant_count,
    );
    println!();
}

/// Run one chat turn: stream the response, render it to the terminal, and
/// return the enriched message history (including any tool-call + tool-response
/// messages and the final assistant text). Returns `None` on a hard error so
/// the caller can revert the user's message and keep history clean.
async fn run_turn(
    manager: Arc<ProviderManager>,
    provider: Provider,
    messages: Vec<ChatMessage>,
    ctx: QueryContext,
    footer: FooterConfig,
) -> Option<Vec<ChatMessage>> {
    let (tx, mut rx) = mpsc::channel::<StreamEvent>(100);

    // Arm a cancellation token for this turn. A spawned listener flips it on
    // Ctrl+C so an in-flight stream bails promptly instead of forcing the
    // user to wait for the provider to finish producing tokens. Rustyline
    // handles Ctrl+C at the prompt itself, so this only covers mid-turn.
    let cancel = CancellationToken::new();
    let cancel_for_signal = cancel.clone();
    let sigint_task = tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            cancel_for_signal.cancel();
        }
    });

    // Consumer future: forward events to the display and flag hard errors.
    // Runs concurrently with the streamer via tokio::join! — no spawn needed,
    // which keeps borrows local to this task.
    let consume = async {
        let mut display = StreamingDisplay::new(
            std::slice::from_ref(&provider),
            footer,
            false, // raw mode off in chat
            None,  // reasoning label shown only if set; chat doesn't surface this per turn
        );
        let mut had_error = false;
        while let Some(event) = rx.recv().await {
            if matches!(&event, StreamEvent::Error { .. }) {
                had_error = true;
            }
            display.handle_event(event);
        }
        had_error
    };

    let streamer = stream_with_messages(&manager, &provider, messages, &ctx, tx, &cancel);
    let (enriched, had_error) = tokio::join!(streamer, consume);

    // Disarm the SIGINT listener so a later Ctrl+C at the readline prompt
    // goes to rustyline's handler instead of this task.
    sigint_task.abort();

    if had_error {
        None
    } else {
        Some(enriched)
    }
}

/// Entry point for interactive chat mode.
pub async fn run_interactive(
    manager: Arc<ProviderManager>,
    provider: Provider,
    ctx: QueryContext,
    footer: FooterConfig,
    initial_question: Option<String>,
) -> Result<()> {
    print_banner(&provider, &ctx);

    let mut rl =
        DefaultEditor::new().map_err(|e| anyhow::anyhow!("failed to init readline: {}", e))?;

    // Seed history: start with system prompt if any
    let mut messages: Vec<ChatMessage> = Vec::new();
    if let Some(system) = &ctx.system_prompt {
        messages.push(ChatMessage::system(system.clone()));
    }

    // If the user provided an initial question on the CLI, run it as turn 1
    if let Some(question) = initial_question {
        let question = question.trim().to_string();
        if !question.is_empty() {
            messages.push(ChatMessage::user(question));
            match run_turn(
                Arc::clone(&manager),
                provider.clone(),
                messages.clone(),
                ctx.clone(),
                footer.clone(),
            )
            .await
            {
                Some(enriched) => messages = enriched,
                None => {
                    // Hard error on turn 1 — drop the user message so the history stays clean
                    messages.pop();
                }
            }
        }
    }

    // Interactive loop
    loop {
        match rl.readline("> ") {
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                if let Some(cmd) = parse_command(trimmed) {
                    match cmd {
                        Command::Exit => {
                            println!("{}", "bye".dimmed());
                            break;
                        }
                        Command::Clear => {
                            let had_system = messages
                                .iter()
                                .any(|m| matches!(m.role, genai::chat::ChatRole::System));
                            messages.clear();
                            if had_system {
                                if let Some(system) = &ctx.system_prompt {
                                    messages.push(ChatMessage::system(system.clone()));
                                }
                            }
                            println!("{}", "conversation cleared".dimmed());
                            continue;
                        }
                        Command::History => {
                            print_history(&messages);
                            continue;
                        }
                        Command::Export(path) => {
                            // Capture the timestamp once so the auto-named
                            // file and the in-document title agree.
                            let ts = now_unix();
                            let target =
                                path.unwrap_or_else(|| PathBuf::from(format!("chat-{}.md", ts)));
                            match export_history(&messages, &target, ts) {
                                Ok(()) => println!(
                                    "  {} {}",
                                    "exported to".dimmed(),
                                    target.display().to_string().green()
                                ),
                                Err(e) => {
                                    println!("  {} {}", "export failed:".red(), e)
                                }
                            }
                            continue;
                        }
                        Command::Help => {
                            print_help();
                            continue;
                        }
                        Command::Unknown(name) => {
                            println!("{} unknown command: /{}", "!".yellow(), name);
                            continue;
                        }
                    }
                }

                // Regular user message
                let _ = rl.add_history_entry(trimmed);
                messages.push(ChatMessage::user(trimmed.to_string()));

                match run_turn(
                    Arc::clone(&manager),
                    provider.clone(),
                    messages.clone(),
                    ctx.clone(),
                    footer.clone(),
                )
                .await
                {
                    Some(enriched) => messages = enriched,
                    None => {
                        // Drop the user message we just added so the history
                        // doesn't accumulate dangling user turns after errors
                        messages.pop();
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                // Ctrl+C — treat as "cancel this line" on first press, exit hint otherwise
                println!("{}", "(press Ctrl+D or type /exit to quit)".dimmed());
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!("{}", "bye".dimmed());
                break;
            }
            Err(e) => {
                eprintln!("input error: {}", e);
                break;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_command_recognizes_aliases() {
        assert!(matches!(parse_command("/exit"), Some(Command::Exit)));
        assert!(matches!(parse_command("/quit"), Some(Command::Exit)));
        assert!(matches!(parse_command("/q"), Some(Command::Exit)));
        assert!(matches!(parse_command("/clear"), Some(Command::Clear)));
        assert!(matches!(parse_command("/reset"), Some(Command::Clear)));
        assert!(matches!(parse_command("/history"), Some(Command::History)));
        assert!(matches!(parse_command("/log"), Some(Command::History)));
        assert!(matches!(parse_command("/help"), Some(Command::Help)));
        assert!(matches!(parse_command("/?"), Some(Command::Help)));
    }

    #[test]
    fn parse_command_ignores_non_slash_input() {
        assert!(parse_command("hello world").is_none());
        assert!(parse_command("  hi").is_none());
    }

    #[test]
    fn parse_command_flags_unknown_slash_commands() {
        match parse_command("/wibble") {
            Some(Command::Unknown(name)) => assert_eq!(name, "wibble"),
            other => panic!("expected Unknown, got {:?}", other),
        }
    }

    #[test]
    fn parse_command_rejects_multiple_leading_slashes() {
        // strip_prefix only peels one '/', so `//history` is treated as the
        // unknown command "/history" rather than aliasing to History.
        match parse_command("//history") {
            Some(Command::Unknown(name)) => assert_eq!(name, "/history"),
            other => panic!("expected Unknown(\"/history\"), got {:?}", other),
        }
    }

    #[test]
    fn parse_command_captures_export_path() {
        match parse_command("/export my chat.md") {
            // The first whitespace splits command from argument; spaces in
            // the path are preserved so `/export my chat.md` works.
            Some(Command::Export(Some(p))) => assert_eq!(p, PathBuf::from("my chat.md")),
            other => panic!("expected Export(Some(...)), got {:?}", other),
        }
    }

    #[test]
    fn parse_command_bare_export_means_auto_path() {
        match parse_command("/export") {
            Some(Command::Export(None)) => {}
            other => panic!("expected Export(None), got {:?}", other),
        }
    }

    #[test]
    fn render_markdown_includes_each_message_with_heading() {
        let messages = vec![
            ChatMessage::system("you are helpful"),
            ChatMessage::user("hello"),
            ChatMessage::assistant("hi there"),
        ];
        let md = render_markdown(&messages, 1_700_000_000);
        assert!(
            md.contains("# Chat export (unix 1700000000)"),
            "missing title: {}",
            md
        );
        assert!(md.contains("## System"), "missing system heading");
        assert!(md.contains("## User"), "missing user heading");
        assert!(md.contains("## Assistant"), "missing assistant heading");
        assert!(md.contains("hello"));
        assert!(md.contains("hi there"));
    }

    #[test]
    fn summarize_content_truncates_long_text() {
        let long = "word ".repeat(200); // ~1000 chars
        let msg = ChatMessage::user(long.clone());
        let out = summarize_content(&msg);
        assert!(
            out.contains("… "),
            "expected truncation marker, got: {}",
            out
        );
        assert!(out.contains("chars)"), "expected char count suffix");
    }

    #[test]
    fn summarize_content_collapses_whitespace() {
        let msg = ChatMessage::user("hello\n\n   world\t\tagain");
        assert_eq!(summarize_content(&msg), "hello world again");
    }
}
