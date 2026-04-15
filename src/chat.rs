use crate::config::FooterConfig;
use crate::display::StreamingDisplay;
use crate::providers::{Provider, ProviderManager};
use crate::query::{stream_with_messages, QueryContext, StreamEvent};
use anyhow::Result;
use colored::Colorize;
use genai::chat::ChatMessage;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Slash command recognized in the interactive prompt.
enum Command {
    Exit,
    Clear,
    History,
    Help,
    Unknown(String),
}

fn parse_command(line: &str) -> Option<Command> {
    let trimmed = line.trim();
    if !trimmed.starts_with('/') {
        return None;
    }
    let cmd = trimmed.trim_start_matches('/').to_lowercase();
    let cmd = cmd.split_whitespace().next().unwrap_or("");
    Some(match cmd {
        "exit" | "quit" | "q" => Command::Exit,
        "clear" | "reset" => Command::Clear,
        "history" | "log" => Command::History,
        "help" | "?" => Command::Help,
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
    println!("  {}  — show message count", "/history".cyan());
    println!("  {}  — show this help", "/help".cyan());
    println!();
}

fn print_history_summary(messages: &[ChatMessage]) {
    let user_count = messages.iter().filter(|m| matches!(m.role, genai::chat::ChatRole::User)).count();
    let assistant_count = messages
        .iter()
        .filter(|m| matches!(m.role, genai::chat::ChatRole::Assistant))
        .count();
    let has_system = messages
        .iter()
        .any(|m| matches!(m.role, genai::chat::ChatRole::System));
    println!();
    println!(
        "{} {} user, {} assistant{}",
        "history:".dimmed(),
        user_count,
        assistant_count,
        if has_system { " (+ system prompt)" } else { "" },
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

    let streamer = stream_with_messages(&manager, &provider, messages, &ctx, tx);
    let (enriched, had_error) = tokio::join!(streamer, consume);

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
                            print_history_summary(&messages);
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
