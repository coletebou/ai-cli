//! Saved sessions in the TUI: `/save`, `/load`, `/sessions`, the exit
//! autosave and `ai -t --continue`.
//!
//! A child module of `ui`, so it works on the `App` and `PanelState` fields
//! directly; the session file format and the naming rules are
//! `crate::sessions` (the same files and `Tracker` interactive chat uses).
//!
//! Each panel's genai history lives in its worker task. The worker shares
//! it through a [`History`]: it writes the history back after every turn
//! and bumps `committed`, and the UI counts the turns it asked for, so a
//! save can tell whether every finished turn has landed in the history yet.

use super::{App, PanelCommand, Turn, TurnKind};
use crate::sessions::{self, Content, PanelSession, Session};
use genai::chat::{ChatMessage, ChatRole};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// How many sessions `/sessions` names in the footer.
const SESSIONS_SHOWN: usize = 5;

/// How long quitting waits for in-flight turns to wind down (they were just
/// cancelled) before saving what's there.
const EXIT_SYNC_TIMEOUT: Duration = Duration::from_secs(3);

/// A panel's conversation, shared between its worker and the UI.
pub(super) type History = Arc<Mutex<PanelHistory>>;

#[derive(Debug, Default)]
pub(super) struct PanelHistory {
    /// The full genai history, system prompt and tool calls included.
    pub messages: Vec<ChatMessage>,
    /// Asks the worker has finished (answered, failed or cancelled) and
    /// written back.
    pub committed: u64,
}

pub(super) fn new_history() -> History {
    Arc::new(Mutex::new(PanelHistory::default()))
}

/// The history, even if a worker panicked while holding the lock: the data
/// is a plain message list, still usable.
pub(super) fn lock(history: &History) -> MutexGuard<'_, PanelHistory> {
    history.lock().unwrap_or_else(|e| e.into_inner())
}

/// What `ai -t --continue` asks the TUI to reopen.
#[derive(Debug, Clone)]
pub struct Resume {
    /// The session to load, or `None` when there was nothing saved.
    pub name: Option<String>,
    /// Put each panel back on the model it was saved with. Off when `-m`
    /// picked a model on the command line.
    pub apply_models: bool,
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn plural_turns(n: usize) -> String {
    if n == 1 {
        "1 turn".to_string()
    } else {
        format!("{n} turns")
    }
}

/// The scroll-back a panel shows for a saved history: each question and
/// each final answer. Tool calls, tool results and the system prompt aren't
/// shown live either, and a restored answer has no token or timing stats.
pub(super) fn rebuild_turns(messages: &[ChatMessage]) -> Vec<Turn> {
    messages
        .iter()
        .filter_map(|m| {
            let text = m.content.texts().join("\n");
            if text.trim().is_empty() {
                return None;
            }
            let kind = match m.role {
                ChatRole::User => TurnKind::User,
                ChatRole::Assistant => TurnKind::Assistant {
                    input_tokens: None,
                    output_tokens: None,
                    duration_ms: 0,
                },
                ChatRole::System | ChatRole::Tool => return None,
            };
            Some(Turn { kind, text })
        })
        .collect()
}

impl App {
    /// Every finished turn has been written back to its panel's history.
    pub(super) fn sessions_synced(&self) -> bool {
        self.panels
            .iter()
            .all(|p| lock(&p.history).committed >= p.asked)
    }

    /// The panels as they'd be saved now.
    pub(super) fn snapshot(&self) -> Vec<PanelSession> {
        self.panels
            .iter()
            .map(|p| PanelSession {
                provider: p.provider.display_name.clone(),
                name: p.provider.name.clone(),
                model: p.model.clone(),
                messages: lock(&p.history).messages.clone(),
            })
            .collect()
    }

    /// Put `session` into the panels. Each panel takes the first saved
    /// panel for the same provider (by config key, or by display name for a
    /// chat session); a panel with no match starts empty. Returns the
    /// providers of saved panels no open panel took.
    pub(super) fn apply_session(&mut self, session: &Session, apply_models: bool) -> Vec<String> {
        let saved = session.panel_sessions();
        let mut used = vec![false; saved.len()];
        for i in 0..self.panels.len() {
            let found = {
                let panel = &self.panels[i];
                (0..saved.len()).find(|&j| {
                    !used[j]
                        && if saved[j].name.is_empty() {
                            saved[j].provider == panel.provider.display_name
                        } else {
                            saved[j].name == panel.provider.name
                        }
                })
            };
            let panel = &mut self.panels[i];
            panel.current_text.clear();
            panel.scroll_offset = 0;
            panel.auto_scroll = true;
            match found {
                Some(j) => {
                    used[j] = true;
                    let restored = &saved[j];
                    lock(&panel.history).messages = restored.messages.clone();
                    panel.turns = rebuild_turns(&restored.messages);
                    if apply_models && !restored.model.is_empty() && restored.model != panel.model {
                        panel.model = restored.model.clone();
                        panel.provider.model = restored.model.clone();
                        if let Some(tx) = self.senders.get(i) {
                            let _ = tx.try_send(PanelCommand::SetModel {
                                provider_name: panel.provider.name.clone(),
                                model: restored.model.clone(),
                            });
                        }
                    }
                }
                None => {
                    panel.turns.clear();
                    if let Some(tx) = self.senders.get(i) {
                        let _ = tx.try_send(PanelCommand::Clear);
                    }
                }
            }
        }
        saved
            .iter()
            .zip(&used)
            .filter(|(_, used)| !**used)
            .map(|(p, _)| p.provider.clone())
            .collect()
    }

    /// Footer note for panels a loaded session had but this TUI doesn't.
    fn not_restored(missing: &[String]) -> String {
        if missing.is_empty() {
            String::new()
        } else {
            format!(
                "; not reopened: {} (start the TUI with them, e.g. `-p`)",
                missing.join(", ")
            )
        }
    }

    /// `/save [name]` and `/save! name`.
    pub(super) fn save_session(&mut self, name: Option<&str>, force: bool) {
        if self.any_busy() || !self.sessions_synced() {
            self.status_message = Some("busy — wait for the current turn to finish".into());
            return;
        }
        let snapshot = self.snapshot();
        self.status_message = Some(
            match self
                .tracker
                .save_content(name, force, Content::Panels(&snapshot), now_unix())
            {
                Ok((name, _)) => format!("saved session {name}"),
                Err(e) => format!("not saved: {e:#}"),
            },
        );
    }

    /// `/load name`: unsaved turns are saved first, under the current
    /// session's name or a new timestamp name.
    pub(super) fn load_session(&mut self, name: &str) {
        if self.any_busy() || !self.sessions_synced() {
            self.status_message = Some("busy — wait for the current turn to finish".into());
            return;
        }
        let snapshot = self.snapshot();
        match self
            .tracker
            .load_content(name, Content::Panels(&snapshot), now_unix())
        {
            Ok(loaded) => {
                let missing = self.apply_session(&loaded.session, true);
                let stashed = loaded
                    .stashed
                    .map(|s| format!(" (unsaved turns saved as {s})"))
                    .unwrap_or_default();
                self.status_message = Some(format!(
                    "loaded {name}: {}{stashed}{}",
                    plural_turns(sessions::user_turns(&loaded.session.messages)),
                    Self::not_restored(&missing)
                ));
            }
            Err(e) => self.status_message = Some(format!("not loaded: {e:#}")),
        }
    }

    /// `/sessions`: the newest few, in the footer.
    pub(super) fn list_sessions(&mut self) {
        let Some(store) = self.tracker.store() else {
            self.status_message =
                Some("sessions are unavailable: this system has no data directory".into());
            return;
        };
        let now = now_unix();
        self.status_message = Some(match store.list() {
            Ok(listing) if listing.sessions.is_empty() => {
                "no saved sessions yet; /save [name] saves this one".into()
            }
            Ok(listing) => {
                let shown: Vec<String> = listing
                    .sessions
                    .iter()
                    .take(SESSIONS_SHOWN)
                    .map(|s| {
                        let current = if self.tracker.name() == Some(s.name.as_str()) {
                            "*"
                        } else {
                            ""
                        };
                        format!(
                            "{current}{} ({}, {})",
                            s.name,
                            plural_turns(s.turns),
                            sessions::format_age(s.updated, now)
                        )
                    })
                    .collect();
                let more = listing.sessions.len().saturating_sub(SESSIONS_SHOWN);
                let more = if more > 0 {
                    format!(" … +{more} (ai sessions lists all)")
                } else {
                    String::new()
                };
                format!("sessions: {}{more} — /load <name>", shown.join(", "))
            }
            Err(e) => format!("could not list sessions: {e:#}"),
        });
    }

    /// `ai -t --continue`, before the first frame.
    pub(super) fn resume(&mut self, resume: &Resume) {
        let Some(name) = resume.name.as_deref() else {
            self.status_message = Some("no saved sessions to continue; starting fresh".into());
            return;
        };
        match self
            .tracker
            .load_content(name, Content::Panels(&[]), now_unix())
        {
            Ok(loaded) => {
                let missing = self.apply_session(&loaded.session, resume.apply_models);
                self.status_message = Some(format!(
                    "resumed {name} ({}, updated {}){}",
                    plural_turns(sessions::user_turns(&loaded.session.messages)),
                    sessions::format_age(loaded.session.updated, now_unix()),
                    Self::not_restored(&missing)
                ));
            }
            Err(e) => self.status_message = Some(format!("can't continue: {e:#}")),
        }
    }

    /// The exit autosave. The caller has cancelled any in-flight turn;
    /// this waits (briefly) for the workers to write their histories back,
    /// then saves. Returns a line to print once the terminal is restored.
    pub(super) async fn autosave_on_exit(&mut self) -> Option<String> {
        let deadline = tokio::time::Instant::now() + EXIT_SYNC_TIMEOUT;
        while !self.sessions_synced() && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let snapshot = self.snapshot();
        match self
            .tracker
            .autosave_content(Content::Panels(&snapshot), now_unix())
        {
            Ok(Some(name)) => Some(format!(
                "saved this TUI session as {name}; `ai -t -c` resumes it"
            )),
            Ok(None) => None,
            Err(e) => Some(format!("could not save this TUI session: {e:#}")),
        }
    }

    /// The session slash commands. Returns false for anything else.
    pub(super) fn session_command(&mut self, cmd_name: &str, arg: Option<&str>) -> bool {
        match cmd_name {
            "save" => self.save_session(arg, false),
            "save!" => match arg {
                Some(name) => self.save_session(Some(name), true),
                None => {
                    self.status_message = Some("usage: /save! <name> replaces that session".into())
                }
            },
            "load" => match arg {
                Some(name) => self.load_session(name),
                None => self.list_sessions(),
            },
            "sessions" => self.list_sessions(),
            _ => return false,
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::app_with;
    use super::*;
    use crate::sessions::{SessionStore, Tracker, FORMAT_VERSION, TUI_AUTOSAVE_NAME};
    use std::path::PathBuf;

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ai-cli-tui-sessions-{}-{}",
            std::process::id(),
            name
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// An app whose panels' workers aren't running: the test plays them by
    /// writing histories and `committed` itself.
    fn app_in(dir: &str, panels: usize) -> (App, Vec<tokio::sync::mpsc::Receiver<PanelCommand>>) {
        let (mut app, rxs) = app_with(panels);
        app.tracker = Tracker::new(Some(SessionStore::new(test_dir(dir))))
            .with_autosave_name(TUI_AUTOSAVE_NAME);
        (app, rxs)
    }

    /// Play one finished turn in panel `i`, as its worker would.
    fn finish_turn(app: &mut App, i: usize, question: &str, answer: &str) {
        let panel = &mut app.panels[i];
        panel.asked += 1;
        let mut h = lock(&panel.history);
        h.messages.push(ChatMessage::user(question));
        h.messages.push(ChatMessage::assistant(answer));
        h.committed += 1;
        drop(h);
        panel.turns = rebuild_turns(&lock(&panel.history).messages);
        app.tracker.turn_completed(1_700_000_000);
    }

    #[test]
    fn restored_turns_show_questions_and_answers_only() {
        let messages = vec![
            ChatMessage::system("be brief"),
            ChatMessage::user("what's new?"),
            ChatMessage::assistant(""),
            ChatMessage::assistant("Rust 1.100."),
        ];
        let turns = rebuild_turns(&messages);
        assert_eq!(turns.len(), 2);
        assert!(matches!(turns[0].kind, TurnKind::User));
        assert_eq!(turns[1].text, "Rust 1.100.");
        assert!(matches!(
            turns[1].kind,
            TurnKind::Assistant {
                duration_ms: 0,
                input_tokens: None,
                ..
            }
        ));
    }

    #[test]
    fn save_then_load_round_trips_every_panel() {
        let (mut app, _rxs) = app_in("roundtrip", 2);
        finish_turn(&mut app, 0, "q1", "from panel A");
        finish_turn(&mut app, 1, "q1", "from panel B");
        app.panels[1].model = "picked-at-runtime".into();
        app.save_session(Some("proj"), false);
        assert_eq!(app.status_message.as_deref(), Some("saved session proj"));

        // A fresh TUI over the same store loads it back into both panels.
        let (mut other, mut rxs) = app_with(2);
        other.tracker = Tracker::new(app.tracker.store().cloned());
        other.load_session("proj");
        assert!(
            other
                .status_message
                .as_deref()
                .unwrap()
                .starts_with("loaded proj: 1 turn"),
            "{:?}",
            other.status_message
        );
        assert_eq!(other.panels[0].turns[1].text, "from panel A");
        assert_eq!(other.panels[1].turns[1].text, "from panel B");
        assert_eq!(lock(&other.panels[1].history).messages.len(), 2);
        // The second panel goes back to the model it was saved with.
        assert_eq!(other.panels[1].model, "picked-at-runtime");
        assert!(matches!(
            rxs[1].try_recv(),
            Ok(PanelCommand::SetModel { model, .. }) if model == "picked-at-runtime"
        ));
    }

    #[test]
    fn saving_waits_for_the_workers_to_write_back() {
        let (mut app, _rxs) = app_in("unsynced", 1);
        finish_turn(&mut app, 0, "q", "a");
        app.panels[0].asked += 1; // asked, not yet written back
        app.save_session(Some("x"), false);
        assert!(app.status_message.as_deref().unwrap().starts_with("busy"));
        assert!(app
            .tracker
            .store()
            .unwrap()
            .list()
            .unwrap()
            .sessions
            .is_empty());
    }

    #[test]
    fn an_unmatched_panel_starts_empty_and_missing_ones_are_named() {
        let (mut app, mut rxs) = app_in("unmatched", 1);
        app.panels[0].turns.push(Turn {
            kind: TurnKind::User,
            text: "stale".into(),
        });
        let session = Session {
            version: FORMAT_VERSION,
            provider: "OTHER".into(),
            model: "m".into(),
            created: 0,
            updated: 0,
            messages: vec![ChatMessage::user("q")],
            panels: vec![PanelSession {
                provider: "OTHER".into(),
                name: "other".into(),
                model: "m".into(),
                messages: vec![ChatMessage::user("q")],
            }],
        };
        let missing = app.apply_session(&session, true);
        assert_eq!(missing, ["OTHER"]);
        assert!(app.panels[0].turns.is_empty());
        assert!(matches!(rxs[0].try_recv(), Ok(PanelCommand::Clear)));
    }

    #[test]
    fn a_chat_session_opens_in_the_panel_with_its_provider() {
        let (mut app, _rxs) = app_in("chat", 2);
        // fake_provider("P1") panels are named by display name in chats.
        let target = app.panels[1].provider.display_name.clone();
        let session = Session {
            version: FORMAT_VERSION,
            provider: target.clone(),
            model: "test-model".into(),
            created: 0,
            updated: 0,
            messages: vec![ChatMessage::user("hi"), ChatMessage::assistant("hello")],
            panels: Vec::new(),
        };
        assert!(app.apply_session(&session, true).is_empty());
        assert!(app.panels[0].turns.is_empty());
        assert_eq!(app.panels[1].turns.len(), 2);
    }

    #[tokio::test]
    async fn exit_autosaves_to_the_tui_slot_only_after_a_turn() {
        let (mut app, _rxs) = app_in("autosave", 2);
        assert_eq!(app.autosave_on_exit().await, None, "nothing happened");
        finish_turn(&mut app, 0, "q", "a");
        finish_turn(&mut app, 1, "q", "b");
        let msg = app.autosave_on_exit().await.unwrap();
        assert!(msg.contains("saved this TUI session as last-tui"), "{msg}");
        let listing = app.tracker.store().unwrap().list().unwrap();
        assert_eq!(listing.sessions[0].name, TUI_AUTOSAVE_NAME);
        assert_eq!(listing.sessions[0].providers.len(), 2);
    }

    #[test]
    fn resume_reopens_the_session_and_reports_it() {
        let (mut app, _rxs) = app_in("resume", 1);
        finish_turn(&mut app, 0, "q", "a");
        app.save_session(Some("work"), false);
        let (mut other, _rxs2) = app_with(1);
        other.tracker = Tracker::new(app.tracker.store().cloned());
        other.resume(&Resume {
            name: Some("work".into()),
            apply_models: true,
        });
        assert!(other
            .status_message
            .as_deref()
            .unwrap()
            .starts_with("resumed work (1 turn"));
        assert_eq!(other.tracker.name(), Some("work"));
        other.resume(&Resume {
            name: None,
            apply_models: true,
        });
        assert!(other
            .status_message
            .as_deref()
            .unwrap()
            .starts_with("no saved sessions"));
    }

    #[test]
    fn slash_commands_cover_save_load_and_listing() {
        let (mut app, _rxs) = app_in("slash", 1);
        assert!(app.session_command("sessions", None));
        assert!(app
            .status_message
            .as_deref()
            .unwrap()
            .starts_with("no saved sessions"));
        finish_turn(&mut app, 0, "q", "a");
        assert!(app.session_command("save", Some("one")));
        assert!(app.session_command("save", Some("one")));
        assert_eq!(app.status_message.as_deref(), Some("saved session one"));
        // A different session can't take the name without `/save!`.
        let (mut other, _rxs2) = app_with(1);
        other.tracker = Tracker::new(app.tracker.store().cloned());
        finish_turn(&mut other, 0, "q2", "a2");
        assert!(other.session_command("save", Some("one")));
        assert!(other
            .status_message
            .as_deref()
            .unwrap()
            .contains("/save! one"));
        assert!(other.session_command("save!", Some("one")));
        assert_eq!(other.status_message.as_deref(), Some("saved session one"));
        assert!(app.session_command("sessions", None));
        assert!(app
            .status_message
            .as_deref()
            .unwrap()
            .contains("one (1 turn"));
        assert!(!app.session_command("model", None));
    }
}
