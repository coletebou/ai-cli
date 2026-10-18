use crate::config::FooterConfig;
use crate::display::StreamingDisplay;
use crate::providers::{Provider, ProviderManager};
use crate::query::{stream_with_messages, QueryContext, StreamEvent};
use crate::sessions::{self, SessionStore, Tracker};
use anyhow::{Context, Result};
use colored::Colorize;
use genai::chat::{ChatMessage, ChatRole, ContentPart};
use rustyline::error::ReadlineError;
use rustyline::history::{FileHistory, History};
use rustyline::{
    Behavior, Cmd, ConditionalEventHandler, Config, Editor, Event, EventContext, EventHandler,
    KeyCode, KeyEvent, Modifiers, RepeatCount,
};
use std::io::{ErrorKind, IsTerminal, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Slash command recognized in the interactive prompt.
#[derive(Debug)]
enum Command {
    Exit,
    Clear,
    History,
    Help,
    /// Write the conversation so far as markdown. `path: None` picks an
    /// auto-named file in the CWD (`chat-<timestamp>.md`, suffixed `-1`,
    /// `-2`, … when taken). An explicit `path` that already exists is only
    /// replaced when `force` is set (`/export! <path>`).
    Export {
        path: Option<PathBuf>,
        force: bool,
    },
    /// Save the conversation as a session (see `sessions`). `name: None`
    /// keeps the current session's name, or picks a timestamp. `force`
    /// (`/save! <name>`) replaces a different session with that name.
    Save {
        name: Option<String>,
        force: bool,
    },
    /// Replace the conversation with a saved session. `None` prints usage.
    Load {
        name: Option<String>,
    },
    /// List saved sessions.
    Sessions,
    Unknown(String),
}

fn parse_command(line: &str) -> Option<Command> {
    // strip_prefix removes exactly one leading '/', so `//history` becomes an
    // unknown command rather than silently aliasing to History.
    let body = line.trim().strip_prefix('/')?;
    let mut parts = body.splitn(2, char::is_whitespace);
    let word = parts.next().unwrap_or("");
    // A pasted path such as `/etc/nginx/nginx.conf fails with …` starts with
    // '/' too. Command names never contain '/' or '.', so a first word that
    // does is a path and the whole line is an ordinary message. Extra leading
    // slashes are skipped for this check so `//history` still gets the
    // unknown-command hint instead of being sent as a (paid) turn.
    if word.trim_start_matches('/').contains(['/', '.']) {
        return None;
    }
    let cmd = word.to_lowercase();
    let rest = parts.next().unwrap_or("").trim();
    let home = dirs::home_dir();
    // Session names are validated when used, so a bad one gets a specific
    // error instead of "unknown command".
    let arg = (!rest.is_empty()).then(|| rest.to_string());
    Some(match cmd.as_str() {
        "exit" | "quit" | "q" => Command::Exit,
        "clear" | "reset" => Command::Clear,
        "history" | "log" => Command::History,
        "help" | "?" => Command::Help,
        "export" => Command::Export {
            path: parse_export_path(rest, home.as_deref()),
            force: false,
        },
        "export!" => Command::Export {
            path: parse_export_path(rest, home.as_deref()),
            force: true,
        },
        "save" => Command::Save {
            name: arg,
            force: false,
        },
        "save!" => Command::Save {
            name: arg,
            force: true,
        },
        "load" => Command::Load { name: arg },
        "sessions" => Command::Sessions,
        other => Command::Unknown(other.to_string()),
    })
}

/// Normalize the argument of `/export`: strip one pair of surrounding quotes
/// (users type `/export "my chat.md"` out of shell habit) and expand a
/// leading `~/` to `home`. Returns `None` when no path was given, which
/// means "auto-name".
fn parse_export_path(raw: &str, home: Option<&Path>) -> Option<PathBuf> {
    let mut s = raw.trim();
    for quote in ['"', '\''] {
        if s.len() >= 2 && s.starts_with(quote) && s.ends_with(quote) {
            // Both quote chars are one byte, so these indices are char
            // boundaries.
            s = &s[1..s.len() - 1];
            break;
        }
    }
    if s.is_empty() {
        return None;
    }
    match (s.strip_prefix("~/"), home) {
        (Some(rest), Some(home)) => Some(home.join(rest)),
        _ => Some(PathBuf::from(s)),
    }
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
        "  {}  — save conversation as markdown (auto-named if omitted; never overwrites)",
        "/export [path.md]".cyan()
    );
    println!(
        "  {}  — same, but replace the file if it already exists",
        "/export! <path.md>".cyan()
    );
    println!(
        "  {}  — save this chat as a session (default name: its current one, or a timestamp)",
        "/save [name]".cyan()
    );
    println!(
        "  {}  — same, but replace a different session that has this name",
        "/save! <name>".cyan()
    );
    println!(
        "  {}  — switch to a saved session (unsaved turns here are saved first)",
        "/load <name>".cyan()
    );
    println!(
        "  {}  — list saved sessions, newest first",
        "/sessions".cyan()
    );
    println!("  {}  — show this help", "/help".cyan());
    println!();
    println!(
        "{}",
        format!(
            "On exit, new turns are saved to the current session (or to `{}`). \
             `ai -i --continue` resumes the newest session.",
            sessions::AUTOSAVE_NAME
        )
        .dimmed()
    );
    println!();
    println!("{}", "Input:".bold());
    println!(
        "  {}  — end a line with a backslash to continue on the next line",
        "\\ + Enter".cyan()
    );
    println!(
        "  {}  — multi-line pastes stay one message; press Enter to send",
        "paste".cyan()
    );
    println!("  {}  — inline a file's contents", "@path".cyan());
    println!(
        "  {}  — recall earlier messages (kept across sessions)",
        "Up / Down".cyan()
    );
    println!(
        "  {}  — stop a streaming reply; press again to quit",
        "Ctrl+C".cyan()
    );
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

/// Write an export to an explicit `path`. Without `force`, an existing file
/// is left alone and the error tells the user how to overwrite it.
/// `create_new` makes the existence check and the create one atomic step,
/// so there's no window where another process's file could be clobbered.
fn write_export(path: &Path, content: &str, force: bool) -> Result<()> {
    if force {
        return std::fs::write(path, content)
            .with_context(|| format!("failed to write chat export to {}", path.display()));
    }
    let mut file = match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
    {
        Ok(f) => f,
        Err(e) if e.kind() == ErrorKind::AlreadyExists => anyhow::bail!(
            "{} already exists; use `/export! {}` to overwrite it",
            path.display(),
            path.display()
        ),
        Err(e) => {
            return Err(e)
                .with_context(|| format!("failed to write chat export to {}", path.display()))
        }
    };
    file.write_all(content.as_bytes())
        .with_context(|| format!("failed to write chat export to {}", path.display()))
}

/// Upper bound on `-N` suffixes tried for an auto-named export. Only reached
/// if something is creating `chat-<ts>*.md` files in a tight loop.
const MAX_AUTO_NAME_SUFFIX: u32 = 1000;

/// Write an auto-named export into `dir`: `chat-<ts>.md`, or `chat-<ts>-1.md`,
/// `-2`, … if that name is taken (two `/export`s within the same second share
/// a timestamp). Never overwrites anything. Returns the path written.
fn export_auto_named(dir: &Path, content: &str, unix_ts: u64) -> Result<PathBuf> {
    for n in 0..=MAX_AUTO_NAME_SUFFIX {
        let name = if n == 0 {
            format!("chat-{}.md", unix_ts)
        } else {
            format!("chat-{}-{}.md", unix_ts, n)
        };
        let path = dir.join(name);
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut file) => {
                file.write_all(content.as_bytes()).with_context(|| {
                    format!("failed to write chat export to {}", path.display())
                })?;
                return Ok(path);
            }
            Err(e) if e.kind() == ErrorKind::AlreadyExists => continue,
            Err(e) => {
                return Err(e)
                    .with_context(|| format!("failed to write chat export to {}", path.display()))
            }
        }
    }
    anyhow::bail!(
        "no free file name for chat-{}-N.md in {}",
        unix_ts,
        dir.display()
    )
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
                // `len()` would count UTF-8 bytes; the label says chars.
                pieces.push(format!(
                    "[← tool result: {} chars]",
                    tr.content.chars().count()
                ));
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

// ---------------------------------------------------------------------------
// Saved sessions: /save, /load, /sessions, --continue (storage in sessions.rs)
// ---------------------------------------------------------------------------

/// Messages a fresh conversation starts with: just the system prompt, if any.
fn initial_messages(ctx: &QueryContext) -> Vec<ChatMessage> {
    ctx.system_prompt
        .iter()
        .map(|system| ChatMessage::system(system.clone()))
        .collect()
}

/// The context every chat turn runs with: `ctx` minus the startup system
/// prompt. The conversation already carries its system prompt as a message
/// (the startup one from `initial_messages`, or whatever a loaded session
/// was saved with). The Bedrock path sends `ctx.system_prompt` on top of the
/// system messages, so passing it too would add the startup persona to a
/// loaded session's, or bring it back into one saved without a prompt.
fn turn_context(ctx: &QueryContext) -> QueryContext {
    let mut turn = ctx.clone();
    turn.system_prompt = None;
    turn
}

fn plural_turns(n: usize) -> String {
    format!("{} {}", n, if n == 1 { "turn" } else { "turns" })
}

/// Swap in saved session `name` (for `/load` and `--continue`) and say how it
/// differs from the current setup. The current provider always stays: the
/// user picked it for this run with `-p` or the config.
fn load_session(
    tracker: &mut Tracker,
    name: &str,
    messages: &mut Vec<ChatMessage>,
    provider: &Provider,
    ctx: &QueryContext,
    verb: &str,
) {
    let now = now_unix();
    let loaded = match tracker.load(name, provider, messages, now) {
        Ok(loaded) => loaded,
        Err(e) => {
            println!("  {} {:#}", "load failed:".red(), e);
            return;
        }
    };
    if let Some(stash) = &loaded.stashed {
        println!(
            "  {} {}",
            "saved the previous conversation as".dimmed(),
            stash.green()
        );
    }
    let session = loaded.session;
    println!(
        "  {} {} {}",
        verb.dimmed(),
        name.green(),
        format!(
            "({}, updated {}; /history shows it)",
            plural_turns(sessions::user_turns(&session.messages)),
            sessions::format_age(session.updated, now)
        )
        .dimmed()
    );
    if session.provider != provider.display_name || session.model != provider.model {
        println!(
            "  {} it was saved with {} ({}); continuing with {} ({})",
            "!".yellow(),
            session.provider,
            session.model,
            provider.display_name,
            provider.model
        );
    }
    // The conversation is restored as saved, system prompt included.
    match (
        sessions::system_prompt(&session.messages),
        &ctx.system_prompt,
    ) {
        (saved, current) if saved.as_ref() == current.as_ref() => {}
        (Some(_), _) => println!(
            "  {} it keeps the system prompt it was saved with",
            "!".yellow()
        ),
        (None, _) => println!(
            "  {} it was saved without a system prompt, so the current one isn't used",
            "!".yellow()
        ),
    }
    if ctx.search_client.is_none() && sessions::has_tool_calls(&session.messages) {
        println!(
            "  {} it contains web-search tool calls; if the provider rejects them, restart with -w",
            "!".yellow()
        );
    }
    *messages = session.messages;
}

/// `--continue`: load the most recently updated session, if there is one.
fn resume_latest(
    tracker: &mut Tracker,
    messages: &mut Vec<ChatMessage>,
    provider: &Provider,
    ctx: &QueryContext,
) {
    let listing = match tracker.store().map(SessionStore::list) {
        Some(Ok(listing)) => listing,
        Some(Err(e)) => {
            println!("  {} can't continue: {:#}", "!".yellow(), e);
            return;
        }
        None => {
            println!(
                "  {} can't continue: this system has no data directory",
                "!".yellow()
            );
            return;
        }
    };
    for (name, err) in &listing.unreadable {
        println!("  {} skipped session {}: {}", "!".yellow(), name, err);
    }
    match listing.sessions.first() {
        Some(latest) => {
            let name = latest.name.clone();
            load_session(tracker, &name, messages, provider, ctx, "resumed");
        }
        None => println!(
            "  {}",
            "no saved sessions to continue; starting a new chat".dimmed()
        ),
    }
    println!();
}

/// `/sessions`: saved sessions, newest first. `*` marks the current one.
fn print_sessions(tracker: &Tracker, now: u64) {
    println!();
    let Some(store) = tracker.store() else {
        println!(
            "  {} sessions are unavailable: this system has no data directory",
            "!".yellow()
        );
        println!();
        return;
    };
    let listing = match store.list() {
        Ok(listing) => listing,
        Err(e) => {
            println!("  {} {:#}", "could not list sessions:".red(), e);
            println!();
            return;
        }
    };
    if listing.sessions.is_empty() {
        println!(
            "  {}",
            "no saved sessions yet; /save [name] saves this one".dimmed()
        );
    }
    let name_width = listing.sessions.iter().map(|s| s.name.len()).max();
    let model_width = listing
        .sessions
        .iter()
        .map(|s| s.model.chars().count())
        .max();
    for s in &listing.sessions {
        let marker = if tracker.name() == Some(s.name.as_str()) {
            "*"
        } else {
            " "
        };
        println!(
            "  {} {}  {:>10}  {:>9}  {}  {}",
            marker,
            format!("{:<w$}", s.name, w = name_width.unwrap_or(0))
                .cyan()
                .bold(),
            sessions::format_age(s.updated, now),
            plural_turns(s.turns),
            format!("{:<w$}", s.model, w = model_width.unwrap_or(0)).dimmed(),
            s.preview.as_deref().unwrap_or(""),
        );
    }
    for (name, err) in &listing.unreadable {
        println!("  {} skipped {}: {}", "!".yellow(), name, err);
    }
    println!();
    println!("  {} {}", "stored in".dimmed(), store.dir().display());
    println!();
}

/// Exit autosave, so `--continue` can resume this conversation.
fn autosave_on_exit(tracker: &mut Tracker, messages: &[ChatMessage], provider: &Provider) {
    match tracker.autosave(provider, messages, now_unix()) {
        Ok(Some(name)) => println!(
            "  {} {} {}",
            "saved session".dimmed(),
            name.green(),
            "(`ai -i --continue` resumes it)".dimmed()
        ),
        Ok(None) => {}
        Err(e) => eprintln!("[warn] could not save the session: {:#}", e),
    }
}

// ---------------------------------------------------------------------------
// Line editor: terminal input, multi-line input, persistent history
// ---------------------------------------------------------------------------

/// Terminal types rustyline treats as unsupported. For these it reads plain
/// lines from stdin even with `Behavior::PreferTerm` (mirrors
/// `UNSUPPORTED_TERM` in rustyline 14's unix backend).
const UNSUPPORTED_TERMS: [&str; 3] = ["dumb", "cons25", "emacs"];

/// Decide whether chat has a terminal to read keystrokes from.
///
/// Chat input must come from the terminal, never from stdin: main.rs has
/// already drained piped stdin into the first question, and whatever is left
/// in the pipe (e.g. the tail of a log over the 20 MB inline cap) must not be
/// replayed line by line as separate paid turns. `Behavior::PreferTerm` makes
/// rustyline open the controlling terminal, but it silently falls back to
/// stdin when that fails, and always reads stdin for an unsupported `TERM`.
/// So check up front and refuse instead. Inputs are injected for testing.
fn check_terminal_input(
    stdin_is_tty: bool,
    tty_available: impl FnOnce() -> bool,
    term: Option<&str>,
) -> Result<()> {
    if stdin_is_tty {
        return Ok(());
    }
    const HINT: &str =
        "Drop -i to ask a one-shot question about piped input, e.g. `cat file | ai \"question\"`.";
    if !tty_available() {
        anyhow::bail!(
            "interactive chat needs a terminal, but stdin is piped and no controlling terminal is available.\n{}",
            HINT
        );
    }
    if let Some(term) = term {
        if UNSUPPORTED_TERMS
            .iter()
            .any(|t| t.eq_ignore_ascii_case(term))
        {
            anyhow::bail!(
                "interactive chat can't read from a TERM={} terminal while stdin is piped.\n{}",
                term,
                HINT
            );
        }
    }
    Ok(())
}

/// Whether the controlling terminal can be opened, the same way rustyline's
/// `Behavior::PreferTerm` opens it.
fn controlling_terminal_available() -> bool {
    #[cfg(unix)]
    let tty = "/dev/tty";
    #[cfg(windows)]
    let tty = "CONIN$";
    #[cfg(any(unix, windows))]
    {
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(tty)
            .is_ok()
    }
    #[cfg(not(any(unix, windows)))]
    {
        false
    }
}

/// Maximum entries kept in the persistent readline history.
const HISTORY_MAX_ENTRIES: usize = 1000;

fn editor_config() -> rustyline::Result<Config> {
    Ok(Config::builder()
        // Read keystrokes from the controlling terminal (/dev/tty, CONIN$)
        // even when stdin is a pipe. See `check_terminal_input`.
        .behavior(Behavior::PreferTerm)
        // Already rustyline 14's default on unix; set explicitly because
        // multi-line input relies on it: a paste lands in the buffer as one
        // message instead of each pasted newline submitting a turn.
        .bracketed_paste(true)
        .max_history_size(HISTORY_MAX_ENTRIES)?
        .auto_add_history(false)
        .build())
}

/// True when `input` ends in exactly one backslash, i.e. the user wants to
/// keep typing on the next line. A doubled `\\` is sent as typed.
fn ends_with_continuation(input: &str) -> bool {
    input.ends_with('\\') && !input.ends_with("\\\\")
}

/// Line breaks typed as `\` + Enter during the current `readline` call. The
/// Enter key handler below records them; once the message is submitted,
/// `join_continuations` removes their backslashes and keeps backslashes that
/// arrived in a paste (shell `\` continuations, C macros) intact.
///
/// Each break is stored as its index among the buffer's line breaks (0 = the
/// first), not as a byte offset: going back to fix a typo on an earlier line
/// shifts every later byte but leaves the line breaks and their order alone.
#[derive(Clone, Default)]
struct Continuations(Arc<Mutex<Vec<usize>>>);

impl Continuations {
    fn breaks(&self) -> MutexGuard<'_, Vec<usize>> {
        // A plain list can't be left half-updated, so a lock poisoned by a
        // panic elsewhere is still fine to use.
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn take(&self) -> Vec<usize> {
        std::mem::take(&mut *self.breaks())
    }

    /// Route Enter (and Ctrl+M / Ctrl+J, which rustyline treats the same)
    /// through `handle` below.
    fn bind(&self, rl: &mut Editor<(), FileHistory>) {
        let keys = [
            KeyEvent(KeyCode::Enter, Modifiers::NONE),
            KeyEvent::ctrl('M'),
            KeyEvent::ctrl('J'),
        ];
        for key in keys {
            rl.bind_sequence(key, EventHandler::Conditional(Box::new(self.clone())));
        }
    }
}

impl ConditionalEventHandler for Continuations {
    /// Enter right after a single `\` starts a new line instead of sending
    /// the message. The check looks at the text before the cursor, which is
    /// where the line break goes. Any other Enter keeps rustyline's default:
    /// send.
    fn handle(&self, _: &Event, _: RepeatCount, _: bool, ctx: &EventContext) -> Option<Cmd> {
        let before_cursor = ctx.line().get(..ctx.pos())?;
        if !ends_with_continuation(before_cursor) {
            return None;
        }
        record_continuation(&mut self.breaks(), before_cursor);
        Some(Cmd::Newline)
    }
}

/// Record a continuation break inserted right after `before_cursor` (the
/// buffer up to the cursor). Breaks recorded further down move one index on.
fn record_continuation(breaks: &mut Vec<usize>, before_cursor: &str) {
    let index = before_cursor.matches('\n').count();
    for later in breaks.iter_mut().filter(|b| **b >= index) {
        *later += 1;
    }
    breaks.push(index);
}

/// Remove the backslash in front of each line break recorded as a
/// continuation, so a message typed as `first \` Enter `second` is sent as
/// "first " and "second" joined by '\n'. A break only counts while a single
/// backslash still sits right before it, so one whose backslash the user
/// deleted afterwards doesn't eat an unrelated character.
fn join_continuations(input: &str, breaks: &[usize]) -> String {
    let mut out = String::with_capacity(input.len());
    let mut start = 0;
    for (index, (at, _)) in input.match_indices('\n').enumerate() {
        if breaks.contains(&index) && ends_with_continuation(&input[..at]) {
            // `at - 1` is an ASCII backslash, so both cuts are char boundaries.
            out.push_str(&input[start..at - 1]);
            start = at;
        }
    }
    out.push_str(&input[start..]);
    out
}

/// Default location of the persistent chat input history.
fn default_history_path() -> Option<PathBuf> {
    dirs::data_dir().map(|d| d.join("ai-cli").join("chat_history.txt"))
}

/// Readline history persisted across chat sessions. Every failure is
/// non-fatal (chat works fine without it) and only the first one is
/// reported, so a read-only data dir doesn't print a warning every turn.
struct HistoryFile {
    path: Option<PathBuf>,
    warned: bool,
}

impl HistoryFile {
    fn new(path: Option<PathBuf>) -> Self {
        Self {
            path,
            warned: false,
        }
    }

    /// Create the parent directory, trim the file if it has outgrown the cap
    /// (see `compact_history_file`), and load earlier entries. A missing
    /// file is the normal first-run case, not an error.
    fn load(&mut self, history: &mut FileHistory) {
        let Some(path) = self.path.clone() else {
            return;
        };
        if let Some(dir) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(dir) {
                self.warn(&format!("could not create {}: {}", dir.display(), e));
                return;
            }
        }
        if let Err(e) = compact_history_file(&path, HISTORY_MAX_ENTRIES, HISTORY_TRIM_SLACK) {
            self.warn(&format!("could not compact {}: {}", path.display(), e));
        }
        match history.load(&path) {
            Ok(()) => {}
            Err(ReadlineError::Io(e)) if e.kind() == ErrorKind::NotFound => {}
            Err(e) => self.warn(&format!("could not load {}: {}", path.display(), e)),
        }
    }

    /// Write new entries to disk. `append` rather than `save`: it merges with
    /// entries a concurrent chat session wrote meanwhile. (It only trims the
    /// file when it had to merge; `compact_history_file` handles the rest.)
    fn save(&mut self, history: &mut impl History) {
        let Some(path) = self.path.clone() else {
            return;
        };
        // Create the file first so `append` never falls back to rustyline's
        // `save` (see `create_private_file`).
        if let Err(e) = create_private_file(&path) {
            self.warn(&format!("could not save {}: {}", path.display(), e));
            return;
        }
        if let Err(e) = history.append(&path) {
            self.warn(&format!("could not save {}: {}", path.display(), e));
            return;
        }
        if let Err(e) = restrict_permissions(&path) {
            self.warn(&format!(
                "could not restrict permissions on {}: {}",
                path.display(),
                e
            ));
        }
    }

    fn warn(&mut self, msg: &str) {
        if !self.warned {
            self.warned = true;
            eprintln!("[warn] chat history: {}", msg);
        }
    }
}

/// Prompts can contain secrets or private text, so keep the history file
/// owner-only (0600). Rustyline's `FileHistory` currently does this too; doing
/// it here keeps the guarantee independent of that implementation detail.
#[cfg(unix)]
fn restrict_permissions(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    match std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)) {
        // Nothing written yet (e.g. every entry so far was a duplicate).
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(()),
        other => other,
    }
}

#[cfg(not(unix))]
fn restrict_permissions(_path: &Path) -> std::io::Result<()> {
    Ok(())
}

/// Open options for an owner-only (0600 on Unix) history file.
fn private_file_options() -> std::fs::OpenOptions {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
}

/// Create `path` owner-only if it doesn't exist yet; an existing file is left
/// untouched.
///
/// rustyline's `FileHistory::save`, which `append` falls back to when the
/// file is missing, creates it by temporarily changing the umask. The umask
/// is process-wide, not per-thread: a file or directory another thread
/// creates meanwhile gets that restrictive mask too (a directory ends up
/// 0600, so nothing can be created inside it), and two overlapping saves can
/// leave it set for good. Creating the file here, and compacting it in place
/// (`compact_history_file`), keeps rustyline off that path.
fn create_private_file(path: &Path) -> std::io::Result<()> {
    private_file_options().append(true).open(path).map(drop)
}

/// How far past `HISTORY_MAX_ENTRIES` the history file may grow before it is
/// trimmed, so a normal session doesn't rewrite the whole file on every
/// launch.
const HISTORY_TRIM_SLACK: usize = 100;

/// rustyline's `append` only trims the history file when another process
/// has changed it; a file written by one session at a time just keeps
/// growing past the cap, retaining prompts the user was told had dropped
/// out of history. So once `path` holds more than `keep + slack` entries,
/// cut it back to the newest `keep`.
///
/// This works from the file itself, not from a history loaded earlier, and
/// holds the lock rustyline's `load` and `append` take while it reads and
/// rewrites the file, so a prompt another chat session saves meanwhile is
/// kept rather than overwritten by a stale copy. That is also why the file
/// is rewritten in place instead of replaced by a rename: a session waiting
/// for the lock would then append to the old, unlinked file. A missing file
/// has nothing to trim; this never creates one.
fn compact_history_file(path: &Path, keep: usize, slack: usize) -> std::io::Result<()> {
    let file = match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
    {
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(()),
        other => other?,
    };
    let mut lock = fd_lock::RwLock::new(file);
    let mut file = lock.write()?;
    let mut contents = Vec::new();
    file.read_to_end(&mut contents)?;
    let Some(trimmed) = trim_history(&contents, keep, slack) else {
        return Ok(());
    };
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&trimmed)?;
    file.set_len(trimmed.len() as u64)?;
    file.sync_all()
}

/// The history file `contents` cut down to its newest `keep` entries, or
/// `None` when it holds at most `keep + slack`. In rustyline's "#V2" format
/// every entry is one line (newlines inside an entry are escaped), so the
/// kept lines are copied byte for byte. A file without that header is left
/// alone.
fn trim_history(contents: &[u8], keep: usize, slack: usize) -> Option<Vec<u8>> {
    const HEADER: &[u8] = b"#V2\n";
    let entries: Vec<&[u8]> = contents
        .strip_prefix(HEADER)?
        .split_inclusive(|&b| b == b'\n')
        .collect();
    if entries.len() <= keep.saturating_add(slack) {
        return None;
    }
    let mut out = HEADER.to_vec();
    for entry in &entries[entries.len() - keep..] {
        out.extend_from_slice(entry);
    }
    Some(out)
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
    reasoning: Option<String>,
) -> Option<Vec<ChatMessage>> {
    let (tx, mut rx) = mpsc::channel::<StreamEvent>(100);

    // Arm a Ctrl+C listener for this turn. Rustyline handles Ctrl+C at the
    // prompt itself (raw mode delivers it as a key), so this only covers
    // mid-turn. Once tokio has hooked SIGINT, the default "terminate" action
    // is gone for the rest of the process, so this listener is the only way
    // out of a turn that never finishes:
    //   1st press: cancel the token. A well-behaved stream bails promptly
    //              and the prompt comes back.
    //   2nd press: the provider is ignoring the cancel (hung connection, a
    //              non-cancellable step), so exit with the conventional
    //              SIGINT status 130. There is no terminal state to restore:
    //              rustyline only enables raw mode while reading a line.
    let cancel = CancellationToken::new();
    let cancel_for_signal = cancel.clone();
    let sigint_task = tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_err() {
            return;
        }
        cancel_for_signal.cancel();
        eprintln!("\n{}", "(cancelling; press Ctrl+C again to quit)".dimmed());
        if tokio::signal::ctrl_c().await.is_ok() {
            eprintln!();
            std::process::exit(130);
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
            reasoning,
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

/// Entry point for interactive chat mode. `reasoning` is the display label
/// for the `-r` level, shown in each turn's footer. `continue_session`
/// (`--continue`) starts from the most recently updated saved session.
pub async fn run_interactive(
    manager: Arc<ProviderManager>,
    provider: Provider,
    ctx: QueryContext,
    footer: FooterConfig,
    initial_question: Option<String>,
    reasoning: Option<String>,
    continue_session: bool,
) -> Result<()> {
    // Refuse before the first (paid) turn if there's no terminal to chat on.
    let term = if cfg!(unix) {
        std::env::var("TERM").ok()
    } else {
        None
    };
    check_terminal_input(
        std::io::stdin().is_terminal(),
        controlling_terminal_available,
        term.as_deref(),
    )?;

    print_banner(&provider, &ctx);

    let config =
        editor_config().map_err(|e| anyhow::anyhow!("failed to configure readline: {}", e))?;
    let mut rl: Editor<(), FileHistory> = Editor::with_config(config)
        .map_err(|e| anyhow::anyhow!("failed to init readline: {}", e))?;
    let continuations = Continuations::default();
    continuations.bind(&mut rl);
    let mut history_file = HistoryFile::new(default_history_path());
    history_file.load(rl.history_mut());

    // Seed history: start with system prompt if any, or with the saved
    // session for --continue.
    let mut messages = initial_messages(&ctx);
    let turn_ctx = turn_context(&ctx);
    let mut tracker = Tracker::new(SessionStore::default_location());
    if continue_session {
        resume_latest(&mut tracker, &mut messages, &provider, &ctx);
    }

    // If the user provided an initial question on the CLI, run it as turn 1
    // (the next turn, with --continue). main.rs has already expanded its
    // @file references.
    if let Some(question) = initial_question {
        let question = question.trim().to_string();
        if !question.is_empty() {
            messages.push(ChatMessage::user(question));
            match run_turn(
                Arc::clone(&manager),
                provider.clone(),
                messages.clone(),
                turn_ctx.clone(),
                footer.clone(),
                reasoning.clone(),
            )
            .await
            {
                Some(enriched) => {
                    messages = enriched;
                    tracker.turn_completed(now_unix());
                }
                None => {
                    // Hard error on turn 1 — drop the user message so the history stays clean
                    messages.pop();
                }
            }
        }
    }

    // Interactive loop. Every way out breaks with the result, so the exit
    // autosave below runs for Ctrl+D, /exit and a broken terminal alike.
    let outcome = loop {
        let read = rl.readline("> ");
        // Continuation breaks belong to this readline call only; take them
        // now so they can't leak into the next one.
        let breaks = continuations.take();
        match read {
            Ok(line) => {
                let text = join_continuations(&line, &breaks);
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    continue;
                }

                if let Some(cmd) = parse_command(trimmed) {
                    match cmd {
                        Command::Exit => break Ok(()),
                        Command::Clear => {
                            // A new conversation with the current settings.
                            // Detaching means it's saved as a new session
                            // rather than over the one it replaced.
                            messages = initial_messages(&ctx);
                            tracker.detach();
                            println!("{}", "conversation cleared".dimmed());
                            continue;
                        }
                        Command::History => {
                            print_history(&messages);
                            continue;
                        }
                        Command::Export { path, force } => {
                            // Capture the timestamp once so the auto-named
                            // file and the in-document title agree.
                            let ts = now_unix();
                            let content = render_markdown(&messages, ts);
                            let result = match path {
                                Some(path) => write_export(&path, &content, force).map(|()| path),
                                None => export_auto_named(Path::new(""), &content, ts),
                            };
                            match result {
                                Ok(target) => println!(
                                    "  {} {}",
                                    "exported to".dimmed(),
                                    target.display().to_string().green()
                                ),
                                Err(e) => {
                                    println!("  {} {:#}", "export failed:".red(), e)
                                }
                            }
                            continue;
                        }
                        Command::Save { name, force } => {
                            match tracker.save(
                                name.as_deref(),
                                force,
                                &provider,
                                &messages,
                                now_unix(),
                            ) {
                                Ok((name, path)) => println!(
                                    "  {} {} {}",
                                    "saved session".dimmed(),
                                    name.green(),
                                    format!("({})", path.display()).dimmed()
                                ),
                                Err(e) => println!("  {} {:#}", "save failed:".red(), e),
                            }
                            continue;
                        }
                        Command::Load { name: Some(name) } => {
                            load_session(
                                &mut tracker,
                                &name,
                                &mut messages,
                                &provider,
                                &ctx,
                                "loaded",
                            );
                            continue;
                        }
                        Command::Load { name: None } => {
                            println!(
                                "{} usage: /load <name> (/sessions lists saved sessions)",
                                "!".yellow()
                            );
                            continue;
                        }
                        Command::Sessions => {
                            print_sessions(&tracker, now_unix());
                            continue;
                        }
                        Command::Help => {
                            print_help();
                            continue;
                        }
                        Command::Unknown(name) => {
                            println!(
                                "{} unknown command: /{} (/help lists commands)",
                                "!".yellow(),
                                name
                            );
                            continue;
                        }
                    }
                }

                // Regular user message. Readline history keeps the text as
                // typed, before @file expansion, so Up recalls
                // `explain @src/foo.rs` rather than the inlined file, and a
                // failed expansion below can be recalled and fixed.
                if rl.add_history_entry(trimmed).unwrap_or(false) {
                    history_file.save(rl.history_mut());
                }
                let expanded = match crate::context::expand_file_refs(trimmed) {
                    Ok(expanded) => expanded,
                    Err(e) => {
                        println!("{} {:#} (message not sent)", "!".yellow(), e);
                        continue;
                    }
                };
                messages.push(ChatMessage::user(expanded));

                match run_turn(
                    Arc::clone(&manager),
                    provider.clone(),
                    messages.clone(),
                    turn_ctx.clone(),
                    footer.clone(),
                    reasoning.clone(),
                )
                .await
                {
                    Some(enriched) => {
                        messages = enriched;
                        tracker.turn_completed(now_unix());
                    }
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
            Err(ReadlineError::Eof) => break Ok(()),
            Err(e) => {
                // A broken terminal is a failure, not a normal exit.
                break Err(anyhow::anyhow!("chat input error: {}", e));
            }
        }
    };

    autosave_on_exit(&mut tracker, &messages, &provider);
    if outcome.is_ok() {
        println!("{}", "bye".dimmed());
    }
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fresh, empty per-test directory under the system temp dir.
    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ai-cli-chat-{}-{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Write an owner-only history file holding `entries` in the "#V2"
    /// format `FileHistory` writes (`\` and newlines escaped). Tests seed
    /// files this way rather than with rustyline's `save`, which changes the
    /// process-wide umask and so races with other tests creating files.
    fn write_history_file(path: &Path, entries: impl IntoIterator<Item = impl AsRef<str>>) {
        let mut contents = String::from("#V2\n");
        for entry in entries {
            contents.push_str(&entry.as_ref().replace('\\', "\\\\").replace('\n', "\\n"));
            contents.push('\n');
        }
        private_file_options()
            .truncate(true)
            .open(path)
            .unwrap()
            .write_all(contents.as_bytes())
            .unwrap();
    }

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
        // A bare top-level word is still treated as a (mistyped) command.
        assert!(matches!(parse_command("/tmp"), Some(Command::Unknown(_))));
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
    fn parse_command_treats_leading_paths_as_messages() {
        // A first word containing another '/' or a '.' is a path, so the line
        // goes to the model instead of being rejected as a command.
        assert!(parse_command("/etc/nginx/nginx.conf fails with a 502").is_none());
        assert!(parse_command("/usr/bin/env python3 not found").is_none());
        assert!(parse_command("/.bashrc is sourced twice").is_none());
        assert!(parse_command("/v1.2 changelog?").is_none());
        assert!(parse_command("//server/share/file.txt is locked").is_none());
        // ...but commands whose argument is a path are still commands.
        assert!(matches!(
            parse_command("/export ./notes/chat.md"),
            Some(Command::Export { .. })
        ));
    }

    #[test]
    fn parse_command_captures_export_path() {
        match parse_command("/export my chat.md") {
            // The first whitespace splits command from argument; spaces in
            // the path are preserved so `/export my chat.md` works.
            Some(Command::Export {
                path: Some(p),
                force: false,
            }) => assert_eq!(p, PathBuf::from("my chat.md")),
            other => panic!("expected Export(Some(...)), got {:?}", other),
        }
    }

    #[test]
    fn parse_command_bare_export_means_auto_path() {
        match parse_command("/export") {
            Some(Command::Export {
                path: None,
                force: false,
            }) => {}
            other => panic!("expected Export(None), got {:?}", other),
        }
    }

    #[test]
    fn parse_command_bang_export_sets_force() {
        match parse_command("/export! out.md") {
            Some(Command::Export {
                path: Some(p),
                force: true,
            }) => assert_eq!(p, PathBuf::from("out.md")),
            other => panic!("expected forced Export, got {:?}", other),
        }
    }

    #[test]
    fn parse_command_session_commands() {
        // `/save` used to be an alias for `/export`; it now saves a session.
        assert!(matches!(
            parse_command("/save"),
            Some(Command::Save {
                name: None,
                force: false
            })
        ));
        match parse_command("/save  proj-x ") {
            Some(Command::Save {
                name: Some(n),
                force: false,
            }) => assert_eq!(n, "proj-x"),
            other => panic!("expected Save(proj-x), got {:?}", other),
        }
        match parse_command("/SAVE! proj") {
            Some(Command::Save {
                name: Some(n),
                force: true,
            }) => assert_eq!(n, "proj"),
            other => panic!("expected forced Save, got {:?}", other),
        }
        // Names keep their case and are validated later, so a bad one gets a
        // specific error rather than being mangled here.
        match parse_command("/load My chat") {
            Some(Command::Load { name: Some(n) }) => assert_eq!(n, "My chat"),
            other => panic!("expected Load, got {:?}", other),
        }
        assert!(matches!(
            parse_command("/load"),
            Some(Command::Load { name: None })
        ));
        assert!(matches!(
            parse_command("/sessions"),
            Some(Command::Sessions)
        ));
        // A session name with a dot is still an argument, not a path.
        assert!(matches!(
            parse_command("/load notes.v2"),
            Some(Command::Load { name: Some(_) })
        ));
    }

    #[test]
    fn initial_messages_hold_only_the_system_prompt() {
        let ctx = QueryContext::new("");
        assert!(initial_messages(&ctx).is_empty());
        let ctx = ctx.with_system_prompt(Some("be brief"));
        let messages = initial_messages(&ctx);
        assert_eq!(messages.len(), 1);
        assert!(matches!(messages[0].role, ChatRole::System));
        assert_eq!(messages[0].content.first_text(), Some("be brief"));
    }

    #[test]
    fn turns_take_the_system_prompt_from_the_conversation_only() {
        // The startup prompt is in the conversation's first message; turns
        // don't pass it again, so a loaded session's own prompt (or its lack
        // of one) is what the provider gets.
        let ctx = QueryContext::new("q").with_system_prompt(Some("be brief"));
        let turn = turn_context(&ctx);
        assert_eq!(turn.system_prompt, None);
        assert_eq!(turn.question, "q");
        assert_eq!(turn.max_iterations, ctx.max_iterations);
        assert_eq!(
            sessions::system_prompt(&initial_messages(&ctx)).as_deref(),
            Some("be brief")
        );
    }

    #[test]
    fn parse_export_path_strips_quotes_and_expands_home() {
        let home = Path::new("/home/someone");
        assert_eq!(
            parse_export_path("\"my chat.md\"", Some(home)),
            Some(PathBuf::from("my chat.md"))
        );
        assert_eq!(
            parse_export_path("'my chat.md'", Some(home)),
            Some(PathBuf::from("my chat.md"))
        );
        assert_eq!(
            parse_export_path("~/notes/chat.md", Some(home)),
            Some(home.join("notes/chat.md"))
        );
        assert_eq!(
            parse_export_path("\"~/my notes.md\"", Some(home)),
            Some(home.join("my notes.md"))
        );
        // Mismatched quotes and `~user` forms are left alone.
        assert_eq!(
            parse_export_path("\"odd.md'", Some(home)),
            Some(PathBuf::from("\"odd.md'"))
        );
        assert_eq!(
            parse_export_path("~other/x.md", Some(home)),
            Some(PathBuf::from("~other/x.md"))
        );
        // No home dir known: keep the literal path rather than guessing.
        assert_eq!(
            parse_export_path("~/x.md", None),
            Some(PathBuf::from("~/x.md"))
        );
        assert_eq!(parse_export_path("", Some(home)), None);
        assert_eq!(parse_export_path("\"\"", Some(home)), None);
    }

    #[test]
    fn write_export_refuses_to_overwrite_without_force() {
        let dir = test_dir("export-noforce");
        let path = dir.join("out.md");
        std::fs::write(&path, "precious").unwrap();

        let err = write_export(&path, "new", false).unwrap_err();
        let msg = format!("{:#}", err);
        assert!(msg.contains("already exists"), "unexpected error: {msg}");
        assert!(msg.contains("/export!"), "missing overwrite hint: {msg}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "precious");
    }

    #[test]
    fn write_export_overwrites_with_force_and_creates_new_files() {
        let dir = test_dir("export-force");
        let path = dir.join("out.md");
        write_export(&path, "first", false).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "first");

        write_export(&path, "second", true).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "second");
    }

    #[test]
    fn export_auto_named_never_clobbers() {
        let dir = test_dir("export-auto");
        let ts = 1_700_000_000;
        let first = export_auto_named(&dir, "one", ts).unwrap();
        let second = export_auto_named(&dir, "two", ts).unwrap();
        let third = export_auto_named(&dir, "three", ts).unwrap();

        assert_eq!(first, dir.join("chat-1700000000.md"));
        assert_eq!(second, dir.join("chat-1700000000-1.md"));
        assert_eq!(third, dir.join("chat-1700000000-2.md"));
        assert_eq!(std::fs::read_to_string(&first).unwrap(), "one");
        assert_eq!(std::fs::read_to_string(&second).unwrap(), "two");
        assert_eq!(std::fs::read_to_string(&third).unwrap(), "three");
    }

    #[test]
    fn continuation_needs_exactly_one_trailing_backslash() {
        assert!(ends_with_continuation("first line \\"));
        assert!(ends_with_continuation("\\"));
        assert!(!ends_with_continuation("C:\\dir\\\\"));
        assert!(!ends_with_continuation("no backslash"));
        assert!(!ends_with_continuation(""));
    }

    #[test]
    fn join_continuations_removes_recorded_backslashes() {
        // Typed as: `one \` Enter `two\` Enter `three`.
        let mut breaks = Vec::new();
        record_continuation(&mut breaks, "one \\");
        record_continuation(&mut breaks, "one \\\ntwo\\");
        assert_eq!(breaks, [0, 1]);
        let input = "one \\\ntwo\\\nthree";
        assert_eq!(join_continuations(input, &breaks), "one \ntwo\nthree");
    }

    #[test]
    fn continuations_survive_edits_to_earlier_lines() {
        // Typed `one \` Enter `two`, then went back up and changed line one
        // before sending: the recorded break still finds its backslash.
        let mut breaks = Vec::new();
        record_continuation(&mut breaks, "one \\");
        assert_eq!(
            join_continuations("only one more \\\ntwo", &breaks),
            "only one more \ntwo"
        );
        assert_eq!(join_continuations("e \\\ntwo", &breaks), "e \ntwo");

        // A continuation added on an earlier line moves the later ones on:
        // `first second \` Enter `third`, then `\` Enter after "first ".
        let mut breaks = Vec::new();
        record_continuation(&mut breaks, "first second \\");
        record_continuation(&mut breaks, "first \\");
        assert_eq!(breaks, [1, 0]);
        assert_eq!(
            join_continuations("first \\\nsecond \\\nthird", &breaks),
            "first \nsecond \nthird"
        );
    }

    #[test]
    fn join_continuations_keeps_pasted_backslashes() {
        // A pasted shell command with its own `\` continuations, followed by a
        // typed continuation at the end.
        let pasted = "docker run \\\n  -p 80:80 \\\n  nginx fails \\";
        let mut breaks = Vec::new();
        record_continuation(&mut breaks, pasted);
        let out = join_continuations(&format!("{pasted}\nwhy?"), &breaks);
        assert_eq!(out, "docker run \\\n  -p 80:80 \\\n  nginx fails \nwhy?");
        // Still only the typed one after editing the first line.
        let out = join_continuations(&format!("sudo {pasted}\nwhy?"), &breaks);
        assert_eq!(
            out,
            "sudo docker run \\\n  -p 80:80 \\\n  nginx fails \nwhy?"
        );
    }

    #[test]
    fn join_continuations_ignores_stale_breaks() {
        // The continuation backslash was deleted or doubled afterwards, or the
        // break is past the end of the buffer.
        assert_eq!(join_continuations("one \ntwo", &[0]), "one \ntwo");
        assert_eq!(join_continuations("C:\\dir\\\\\nx", &[0]), "C:\\dir\\\\\nx");
        assert_eq!(join_continuations("abc\ndef", &[1, 99]), "abc\ndef");
        assert_eq!(join_continuations("", &[0]), "");
    }

    #[test]
    fn check_terminal_input_rules() {
        // Interactive stdin: nothing to check.
        assert!(check_terminal_input(true, || unreachable!(), Some("dumb")).is_ok());
        // Piped stdin with a usable controlling terminal.
        assert!(check_terminal_input(false, || true, Some("xterm-256color")).is_ok());
        assert!(check_terminal_input(false, || true, None).is_ok());
        // Piped stdin and no terminal: refuse rather than read turns from stdin.
        let err = check_terminal_input(false, || false, Some("xterm")).unwrap_err();
        assert!(err.to_string().contains("needs a terminal"), "{err}");
        // Rustyline reads stdin directly for unsupported terminals.
        let err = check_terminal_input(false, || true, Some("dumb")).unwrap_err();
        assert!(err.to_string().contains("TERM=dumb"), "{err}");
    }

    #[test]
    fn editor_config_reads_from_terminal_with_capped_history() {
        let config = editor_config().unwrap();
        assert_eq!(config.behavior(), Behavior::PreferTerm);
        assert!(config.enable_bracketed_paste());
        assert_eq!(config.max_history_size(), HISTORY_MAX_ENTRIES);
    }

    #[test]
    fn default_history_path_is_under_ai_cli_data_dir() {
        if let Some(path) = default_history_path() {
            assert!(path.ends_with("ai-cli/chat_history.txt"), "{:?}", path);
        }
    }

    #[test]
    fn history_file_creates_dir_and_round_trips_entries() {
        let dir = test_dir("history");
        let path = dir.join("nested").join("ai-cli").join("chat_history.txt");
        let config = editor_config().unwrap();

        let mut store = HistoryFile::new(Some(path.clone()));
        let mut history = FileHistory::with_config(config);
        store.load(&mut history);
        assert!(path.parent().unwrap().is_dir(), "parent dir not created");
        assert!(!store.warned, "a missing file is the normal first run");

        history.add("explain @src/main.rs").unwrap();
        history.add("line one\nline two").unwrap();
        store.save(&mut history);
        assert!(!store.warned);
        assert!(path.is_file());

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600, "history file mode {:o}", mode);
        }

        let mut reloaded = FileHistory::with_config(config);
        let mut store2 = HistoryFile::new(Some(path.clone()));
        store2.load(&mut reloaded);
        assert!(!store2.warned);
        let entries: Vec<&String> = reloaded.iter().collect();
        assert_eq!(entries, ["explain @src/main.rs", "line one\nline two"]);
    }

    #[test]
    fn oversized_history_file_is_compacted_on_load() {
        // A single session's `append` never trims, so the file can grow far
        // past the cap across sessions; loading must shrink it back.
        let dir = test_dir("history-compact");
        let path = dir.join("chat_history.txt");
        let seed = (0..(HISTORY_MAX_ENTRIES + 500)).map(|i| format!("prompt {}", i));
        write_history_file(&path, seed);

        let mut history = FileHistory::with_config(editor_config().unwrap());
        let mut store = HistoryFile::new(Some(path.clone()));
        store.load(&mut history);
        assert!(!store.warned);
        assert_eq!(history.len(), HISTORY_MAX_ENTRIES);
        let lines = std::fs::read_to_string(&path).unwrap().lines().count();
        assert!(
            lines <= HISTORY_MAX_ENTRIES + 1,
            "file still has {} lines",
            lines
        );
        // The newest prompts are the ones kept.
        let last: Vec<&String> = history.iter().rev().take(1).collect();
        assert_eq!(last[0], &format!("prompt {}", HISTORY_MAX_ENTRIES + 499));
        // Nothing left behind but the history file itself.
        let names: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names, ["chat_history.txt"]);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600, "compacted file mode {:o}", mode);
        }
    }

    #[test]
    fn trim_history_keeps_the_newest_entries_past_the_slack() {
        let file = b"#V2\na\nb\nc\nd\ne\n";
        // 5 entries: within keep 2 + slack 3, so left alone.
        assert_eq!(trim_history(file, 2, 3), None);
        assert_eq!(
            trim_history(file, 2, 2).as_deref(),
            Some(&b"#V2\nd\ne\n"[..])
        );
        // Not a "#V2" file: left alone rather than guessed at.
        assert_eq!(trim_history(b"a\nb\nc\n", 1, 0), None);
        assert_eq!(trim_history(b"", 1, 0), None);
    }

    #[test]
    fn trimmed_history_entries_load_back_unchanged() {
        // Kept entries are copied line for line, so escaped newlines and
        // backslashes survive compaction.
        let dir = test_dir("history-format");
        let path = dir.join("chat_history.txt");
        let entries = [
            "plain",
            "two\nlines",
            r"C:\path\to\file",
            r"trailing backslash \",
            "literal \\n is not a newline",
            "ünïcödé ✓",
        ];
        let older = (0..10).map(|i| format!("old {}", i));
        write_history_file(&path, older.chain(entries.map(String::from)));
        compact_history_file(&path, entries.len(), 2).unwrap();

        let mut loaded = FileHistory::with_config(editor_config().unwrap());
        loaded.load(&path).unwrap();
        let loaded: Vec<&String> = loaded.iter().collect();
        assert_eq!(loaded, entries);
    }

    #[test]
    fn compaction_keeps_prompts_saved_while_it_waits_for_the_lock() {
        // Another chat session is appending a prompt under the file lock
        // rustyline's `append` takes. Compaction must wait for it and keep
        // that prompt, not rewrite the file from an older copy.
        let dir = test_dir("history-compact-lock");
        let path = dir.join("chat_history.txt");
        write_history_file(&path, (0..30).map(|i| format!("prompt {}", i)));

        let other = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let mut other = fd_lock::RwLock::new(other);
        let mut guard = other.write().unwrap();
        let compactor = {
            let path = path.clone();
            std::thread::spawn(move || compact_history_file(&path, 10, 5))
        };
        // Let the compactor reach the lock. (It passes however long this
        // takes; without the lock it would lose the prompt below.)
        std::thread::sleep(std::time::Duration::from_millis(100));
        guard.seek(SeekFrom::End(0)).unwrap();
        guard.write_all(b"from another session\n").unwrap();
        drop(guard);
        compactor.join().unwrap().unwrap();

        let mut loaded = FileHistory::with_config(editor_config().unwrap());
        loaded.load(&path).unwrap();
        let loaded: Vec<&String> = loaded.iter().collect();
        assert_eq!(loaded.len(), 10);
        assert_eq!(loaded[0], "prompt 21");
        assert_eq!(loaded[9], "from another session");
    }

    #[test]
    fn history_file_failures_are_non_fatal() {
        let dir = test_dir("history-fail");
        // The "directory" is a regular file, so create_dir_all fails.
        let blocker = dir.join("not-a-dir");
        std::fs::write(&blocker, "x").unwrap();
        let mut store = HistoryFile::new(Some(blocker.join("chat_history.txt")));
        let mut history = FileHistory::with_config(editor_config().unwrap());
        store.load(&mut history);
        assert!(store.warned);
        // Saving afterwards fails too but must not panic.
        history.add("hello").unwrap();
        store.save(&mut history);

        // No data dir at all: silently a no-op.
        let mut none = HistoryFile::new(None);
        none.load(&mut history);
        none.save(&mut history);
        assert!(!none.warned);
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

    #[test]
    fn summarize_content_counts_tool_result_chars_not_bytes() {
        // "héllo wörld" is 11 chars but 13 UTF-8 bytes.
        let msg = ChatMessage::from(genai::chat::ToolResponse::new("call_1", "héllo wörld"));
        assert_eq!(summarize_content(&msg), "[← tool result: 11 chars]");
    }
}
