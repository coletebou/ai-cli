//! `ai sessions`: list saved chat sessions for scripts and other tools (for
//! example porchlight's login dashboard, which shells out to this to show
//! recent chats and offer to launch `ai -t`).
//!
//! A human table by default; `--json` for a small, stable shape other
//! programs can parse without guessing at column widths. Both are built
//! purely from [`SessionSummary`]-level metadata: never message text, API
//! keys, or config values, even when a session file holds them. `/sessions`
//! inside interactive chat (`src/chat.rs`) is the richer, colored version of
//! this for a human already in a conversation; this command is the
//! non-interactive, machine-readable one.

use crate::sessions::{self, SessionStore, SessionSummary};
use anyhow::Result;
use serde::Serialize;
use std::path::Path;

/// Default number of sessions shown without `--limit`.
pub const DEFAULT_LIMIT: usize = 20;

/// `{"version": 1, "dir": "...", "sessions": [...], "skipped": n}`.
#[derive(Debug, Serialize)]
struct SessionsJson {
    version: u32,
    dir: String,
    sessions: Vec<SessionJson>,
    skipped: usize,
}

#[derive(Debug, Serialize)]
struct SessionJson {
    name: String,
    /// RFC 3339, UTC.
    updated: String,
    turns: usize,
    /// Always one entry today (a session is saved with a single provider),
    /// kept as an array since a stable public shape should say "these are
    /// the providers involved" without the reader having to change if a
    /// session ever gets more than one.
    providers: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
}

impl From<&SessionSummary> for SessionJson {
    fn from(s: &SessionSummary) -> Self {
        Self {
            name: s.name.clone(),
            updated: sessions::rfc3339_utc(s.updated),
            turns: s.turns,
            providers: if s.provider.is_empty() {
                Vec::new()
            } else {
                vec![s.provider.clone()]
            },
            model: (!s.model.is_empty()).then(|| s.model.clone()),
        }
    }
}

/// `ai sessions [--json] [--limit N]`. `store` is `None` only when the
/// platform reports no data directory at all; a missing or empty sessions
/// directory is a normal, successful empty list (see [`SessionStore::list`]).
pub fn run(store: Option<&SessionStore>, json: bool, limit: usize) -> Result<()> {
    let Some(store) = store else {
        if json {
            print_json("", Vec::new(), 0);
        } else {
            println!("sessions are unavailable: this system has no data directory");
        }
        return Ok(());
    };

    let listing = store.list()?;
    let shown: Vec<&SessionSummary> = listing.sessions.iter().take(limit).collect();

    if json {
        let dir = display_dir(store.dir());
        let sessions: Vec<SessionJson> = shown.into_iter().map(SessionJson::from).collect();
        print_json(&dir, sessions, listing.unreadable.len());
    } else {
        print_table(&shown, listing.unreadable.len());
    }
    Ok(())
}

fn print_json(dir: &str, sessions: Vec<SessionJson>, skipped: usize) {
    let doc = SessionsJson {
        version: 1,
        dir: dir.to_string(),
        sessions,
        skipped,
    };
    // Built from a `Serialize` struct of plain strings/numbers, so this
    // can't fail; a fallback keeps the command from panicking regardless.
    println!(
        "{}",
        serde_json::to_string(&doc).unwrap_or_else(|_| "{}".to_string())
    );
}

/// Plain, uncolored table: name, provider/model, turn count, updated (RFC
/// 3339 UTC, same timestamp as `--json`, so the two never disagree). No
/// header row, matching `/sessions` inside chat — and so a porchlight
/// `[[custom]]` source can use `format = "lines"` and get one row per
/// session with nothing to skip.
fn print_table(sessions: &[&SessionSummary], skipped: usize) {
    for line in table_lines(sessions, skipped) {
        println!("{line}");
    }
}

/// The table's lines. The unreadable-file footer is printed even when no
/// session is shown (a directory of only corrupt files, or `--limit 0`), so
/// the table never hides what `--json` reports as `skipped`.
fn table_lines(sessions: &[&SessionSummary], skipped: usize) -> Vec<String> {
    let mut out = Vec::new();
    if sessions.is_empty() {
        out.push("no saved sessions".to_string());
    }
    let name_width = sessions.iter().map(|s| s.name.len()).max().unwrap_or(0);
    for s in sessions {
        let model = match (s.provider.is_empty(), s.model.is_empty()) {
            (false, false) => format!("{}/{}", s.provider, s.model),
            (false, true) => s.provider.clone(),
            (true, false) => s.model.clone(),
            (true, true) => "-".to_string(),
        };
        out.push(format!(
            "{:<name_width$}  {:<24}  {:>4}t  {}",
            s.name,
            model,
            s.turns,
            sessions::rfc3339_utc(s.updated),
        ));
    }
    if skipped > 0 {
        out.push(format!(
            "({} unreadable session file{} skipped)",
            skipped,
            if skipped == 1 { "" } else { "s" }
        ));
    }
    out
}

/// `dir`, with the user's home directory prefix replaced by `~` so the
/// output doesn't leak a full local username/path into logs or dashboards
/// unnecessarily.
fn display_dir(dir: &Path) -> String {
    match dirs::home_dir() {
        Some(home) => match dir.strip_prefix(&home) {
            Ok(rest) if !rest.as_os_str().is_empty() => {
                // Join components with '/' so Windows shows `~/foo/bar`, not `~/foo\bar`.
                let parts: Vec<String> = rest
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy().into_owned())
                    .collect();
                format!("~/{}", parts.join("/"))
            }
            Ok(_) => "~".to_string(),
            Err(_) => dir.display().to_string(),
        },
        None => dir.display().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sessions::{Session, FORMAT_VERSION};
    use genai::chat::ChatMessage;
    use std::path::PathBuf;

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ai-cli-sessions-cmd-{}-{}",
            std::process::id(),
            name
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn session(provider: &str, model: &str, updated: u64, user_text: &str) -> Session {
        Session {
            version: FORMAT_VERSION,
            provider: provider.to_string(),
            model: model.to_string(),
            created: updated,
            updated,
            messages: vec![ChatMessage::user(user_text), ChatMessage::assistant("ok")],
        }
    }

    #[test]
    fn display_dir_replaces_home_with_tilde() {
        if let Some(home) = dirs::home_dir() {
            let dir = home.join("foo").join("bar");
            assert_eq!(display_dir(&dir), "~/foo/bar");
            assert_eq!(display_dir(&home), "~");
        }
        assert_eq!(
            display_dir(Path::new("/elsewhere/chats")),
            "/elsewhere/chats"
        );
    }

    #[test]
    fn json_round_trips_three_sessions_one_corrupt_newest_first_with_limit() {
        let dir = test_dir("basic");
        let store = SessionStore::new(&dir);
        store
            .save("old", &session("CLAUDE", "claude-sonnet", 100, "hi"))
            .unwrap();
        store
            .save("new", &session("GPT", "gpt-5", 300, "hello"))
            .unwrap();
        store
            .save("mid", &session("GEMINI", "gemini-pro", 200, "hey"))
            .unwrap();
        std::fs::write(dir.join("broken.json"), "{ not json").unwrap();

        // Full listing, no limit bite.
        run(Some(&store), true, DEFAULT_LIMIT).unwrap();

        let listing = store.list().unwrap();
        let jsons: Vec<SessionJson> = listing.sessions.iter().map(SessionJson::from).collect();
        let names: Vec<&str> = jsons.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["new", "mid", "old"]);
        assert_eq!(jsons[0].updated, "1970-01-01T00:05:00Z");
        assert_eq!(jsons[0].turns, 1);
        assert_eq!(jsons[0].providers, vec!["GPT".to_string()]);
        assert_eq!(jsons[0].model.as_deref(), Some("gpt-5"));
        assert_eq!(listing.unreadable.len(), 1);

        // --limit 2 keeps only the two newest.
        let shown: Vec<&SessionSummary> = listing.sessions.iter().take(2).collect();
        assert_eq!(shown.len(), 2);
        assert_eq!(shown[0].name, "new");
        assert_eq!(shown[1].name, "mid");

        let doc = SessionsJson {
            version: 1,
            dir: display_dir(store.dir()),
            sessions: jsons,
            skipped: listing.unreadable.len(),
        };
        let value: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&doc).unwrap()).unwrap();
        assert_eq!(value["version"], 1);
        assert_eq!(value["skipped"], 1);
        assert_eq!(value["sessions"].as_array().unwrap().len(), 3);
        assert_eq!(value["sessions"][0]["name"], "new");
        assert!(value["dir"].as_str().unwrap().contains("basic"));
    }

    #[test]
    fn json_output_never_contains_message_text_even_a_fake_secret() {
        let dir = test_dir("secret");
        let store = SessionStore::new(&dir);
        let fake_key = format!("sk-test-{}", "x".repeat(40));
        store
            .save("leaky", &session("CLAUDE", "claude-sonnet", 50, &fake_key))
            .unwrap();

        let listing = store.list().unwrap();
        let jsons: Vec<SessionJson> = listing.sessions.iter().map(SessionJson::from).collect();
        let doc = SessionsJson {
            version: 1,
            dir: display_dir(store.dir()),
            sessions: jsons,
            skipped: 0,
        };
        let rendered = serde_json::to_string(&doc).unwrap();
        assert!(!rendered.contains("sk-test-"), "{rendered}");
        assert!(!rendered.contains(&fake_key), "{rendered}");
        assert!(!rendered.contains("hi"), "{rendered}"); // no stray message text either
    }

    #[test]
    fn table_reports_unreadable_files_even_when_nothing_is_shown() {
        let dir = test_dir("only-corrupt");
        let store = SessionStore::new(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("broken.json"), "{ not json").unwrap();
        let listing = store.list().unwrap();
        assert!(listing.sessions.is_empty());
        assert_eq!(
            table_lines(&[], listing.unreadable.len()),
            ["no saved sessions", "(1 unreadable session file skipped)"]
        );

        // --limit 0 with readable and unreadable files: the footer still shows.
        store
            .save("kept", &session("GPT", "gpt-5", 300, "hello"))
            .unwrap();
        let listing = store.list().unwrap();
        assert_eq!(listing.sessions.len(), 1);
        assert_eq!(
            table_lines(&[], listing.unreadable.len()),
            ["no saved sessions", "(1 unreadable session file skipped)"]
        );
        assert_eq!(table_lines(&[], 0), ["no saved sessions"]);
        let shown: Vec<&SessionSummary> = listing.sessions.iter().collect();
        let lines = table_lines(&shown, 2);
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("kept  GPT/gpt-5"), "{lines:?}");
        assert_eq!(lines[1], "(2 unreadable session files skipped)");
    }

    #[test]
    fn missing_dir_is_an_empty_list_not_an_error() {
        let dir = test_dir("missing-parent").join("never-created");
        let store = SessionStore::new(&dir);
        assert!(!dir.exists());
        // Should not error, and should list nothing.
        run(Some(&store), true, DEFAULT_LIMIT).unwrap();
        run(Some(&store), false, DEFAULT_LIMIT).unwrap();
        let listing = store.list().unwrap();
        assert!(listing.sessions.is_empty());
        assert!(listing.unreadable.is_empty());
    }

    #[test]
    fn no_data_dir_at_all_is_handled_without_panicking() {
        run(None, true, DEFAULT_LIMIT).unwrap();
        run(None, false, DEFAULT_LIMIT).unwrap();
    }

    #[test]
    fn human_table_runs_without_panicking_for_a_populated_and_an_empty_store() {
        let dir = test_dir("table");
        let store = SessionStore::new(&dir);
        store
            .save("proj", &session("CLAUDE", "claude-sonnet", 10, "hi"))
            .unwrap();
        run(Some(&store), false, 1).unwrap();

        let empty = SessionStore::new(test_dir("table-empty"));
        run(Some(&empty), false, DEFAULT_LIMIT).unwrap();
    }
}
