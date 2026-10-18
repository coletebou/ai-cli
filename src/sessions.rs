//! Saved chat sessions for interactive mode: `/save`, `/load`, `/sessions`
//! and `ai -i --continue`.
//!
//! A session is one JSON file, `<data dir>/ai-cli/chats/<name>.json`. It
//! holds the full genai message history (system prompt, user turns,
//! assistant replies, web-search tool calls and results) plus the provider
//! and model it was last saved with. Conversations routinely contain pasted
//! secrets and private files, so the files are owner-only (0600 on Unix).
//! Every write goes to a temp file that is renamed over the target, so a
//! crash or a full disk never leaves a half-written session behind.

use crate::providers::Provider;
use anyhow::{Context, Result};
use genai::chat::{ChatMessage, ChatRole};
use serde::{Deserialize, Serialize};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};

/// Version written to every session file. Bump it on an incompatible layout
/// change: files from a newer version are refused rather than misread.
pub const FORMAT_VERSION: u32 = 1;

/// Where an unnamed conversation is saved on exit. It's a rolling slot: the
/// next unnamed conversation replaces it.
pub const AUTOSAVE_NAME: &str = "last";

/// Longest accepted session name. Names are ASCII, so bytes == chars.
const MAX_NAME_LEN: usize = 100;

/// Upper bound on `-N` suffixes tried by `SessionStore::unused_name`.
const MAX_NAME_SUFFIX: u32 = 1000;

/// Characters of the first user message shown by `/sessions`.
const PREVIEW_MAX_CHARS: usize = 60;

/// On-disk session file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub version: u32,
    /// Display name of the provider the session was last saved with, e.g.
    /// `CLAUDE`.
    pub provider: String,
    pub model: String,
    /// Unix seconds of the conversation's first turn.
    pub created: u64,
    /// Unix seconds of the last save.
    pub updated: u64,
    pub messages: Vec<ChatMessage>,
}

/// What `/sessions` shows for one saved session.
#[derive(Debug, Clone)]
pub struct SessionSummary {
    pub name: String,
    pub model: String,
    pub updated: u64,
    /// Number of user messages.
    pub turns: usize,
    /// Start of the first user message, whitespace collapsed and truncated.
    pub preview: Option<String>,
}

impl SessionSummary {
    fn of(name: &str, session: &Session) -> Self {
        Self {
            name: name.to_string(),
            model: session.model.clone(),
            updated: session.updated,
            turns: user_turns(&session.messages),
            preview: preview(&session.messages),
        }
    }
}

/// Result of scanning the sessions directory.
#[derive(Debug, Default)]
pub struct Listing {
    /// Newest first: by `updated`, then by name.
    pub sessions: Vec<SessionSummary>,
    /// `(name, error)` for session files that couldn't be read or parsed.
    /// Reported rather than silently hidden.
    pub unreadable: Vec<(String, String)>,
}

/// Windows device names. `nul.json` or `con.json` would open the device
/// instead of a file, so a "saved" session would be silently lost there.
/// They're refused on every platform so session names stay portable.
const RESERVED_NAMES: [&str; 22] = [
    "con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8",
    "com9", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
];

/// Session names become file names, so only `[A-Za-z0-9._-]` is allowed: no
/// path separators, no spaces, nothing that needs quoting. A leading `.` is
/// refused too, which rules out `.`, `..` and hidden files, and so are
/// Windows device names (see `RESERVED_NAMES`).
pub fn validate_name(name: &str) -> Result<()> {
    if name.is_empty() {
        anyhow::bail!("session name is empty");
    }
    if let Some(bad) = name
        .chars()
        .find(|c| !(c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')))
    {
        anyhow::bail!(
            "invalid session name {:?}: {:?} isn't allowed (use letters, digits, '.', '_' and '-')",
            name,
            bad
        );
    }
    if name.starts_with('.') {
        anyhow::bail!("invalid session name {:?}: it can't start with '.'", name);
    }
    if name.len() > MAX_NAME_LEN {
        anyhow::bail!(
            "invalid session name: longer than {} characters",
            MAX_NAME_LEN
        );
    }
    // Windows ignores everything from the first '.' when matching device
    // names (`nul.v2.json` is still NUL).
    let stem = name.split('.').next().unwrap_or(name);
    if RESERVED_NAMES.iter().any(|r| r.eq_ignore_ascii_case(stem)) {
        anyhow::bail!(
            "invalid session name {:?}: it's a reserved device name on Windows",
            name
        );
    }
    Ok(())
}

/// Number of user turns in `messages`.
pub fn user_turns(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .filter(|m| matches!(m.role, ChatRole::User))
        .count()
}

/// Text of the first system message, if there is one.
pub fn system_prompt(messages: &[ChatMessage]) -> Option<String> {
    messages
        .iter()
        .find(|m| matches!(m.role, ChatRole::System))
        .map(|m| m.content.texts().join("\n"))
}

/// True when the history contains tool calls (turns that used web search).
pub fn has_tool_calls(messages: &[ChatMessage]) -> bool {
    messages.iter().any(|m| !m.content.tool_calls().is_empty())
}

/// One-line start of the first user message for `/sessions`.
fn preview(messages: &[ChatMessage]) -> Option<String> {
    let first = messages.iter().find(|m| matches!(m.role, ChatRole::User))?;
    let text = first.content.texts().join(" ");
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        return None;
    }
    if collapsed.chars().count() <= PREVIEW_MAX_CHARS {
        return Some(collapsed);
    }
    let mut out: String = collapsed.chars().take(PREVIEW_MAX_CHARS).collect();
    out.push('…');
    Some(out)
}

/// UTC calendar date and seconds into the day for a Unix timestamp. This is
/// Howard Hinnant's `civil_from_days`; it saves a date-time dependency for
/// the two places sessions show a date.
fn civil_from_unix(ts: u64) -> (i64, u32, u32, u64) {
    let days = (ts / 86_400) as i64;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097); // day of era, [0, 146096]
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365], from March 1
    let mp = (5 * doy + 2) / 153; // month from March, [0, 11]
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day, ts % 86_400)
}

/// `YYYY-MM-DD` in UTC.
fn utc_date(ts: u64) -> String {
    let (y, m, d, _) = civil_from_unix(ts);
    format!("{:04}-{:02}-{:02}", y, m, d)
}

/// Default name for `/save` without a name: `chat-YYYYMMDD-HHMMSS` in UTC.
pub fn timestamp_name(now: u64) -> String {
    let (y, m, d, secs) = civil_from_unix(now);
    format!(
        "chat-{:04}{:02}{:02}-{:02}{:02}{:02}",
        y,
        m,
        d,
        secs / 3_600,
        secs / 60 % 60,
        secs % 60
    )
}

/// Short age for listings: `just now`, `5m ago`, `3h ago`, `2d ago`, or the
/// UTC date when older than 30 days (or in the future, after a clock change).
pub fn format_age(then: u64, now: u64) -> String {
    match now.checked_sub(then) {
        Some(0..=59) => "just now".to_string(),
        Some(age @ 60..=3_599) => format!("{}m ago", age / 60),
        Some(age @ 3_600..=86_399) => format!("{}h ago", age / 3_600),
        Some(age @ 86_400..=2_591_999) => format!("{}d ago", age / 86_400),
        _ => utc_date(then),
    }
}

/// Directory of saved sessions. The path is injectable so tests never touch
/// the real data dir.
#[derive(Debug, Clone)]
pub struct SessionStore {
    dir: PathBuf,
}

impl SessionStore {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// `<data dir>/ai-cli/chats`: `~/.local/share` on Linux,
    /// `~/Library/Application Support` on macOS, `%APPDATA%` on Windows.
    /// `None` when the platform reports no data dir.
    pub fn default_location() -> Option<Self> {
        dirs::data_dir().map(|d| Self::new(d.join("ai-cli").join("chats")))
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// File for session `name`, after validating the name.
    pub fn path_for(&self, name: &str) -> Result<PathBuf> {
        validate_name(name)?;
        Ok(self.dir.join(format!("{}.json", name)))
    }

    /// Claim `name` for a new session by creating its file, empty, with
    /// `create_new`. That is atomic, so of two chats claiming the same name
    /// at once exactly one succeeds. Checking that the name is free and
    /// writing afterwards would let both pass the check, and the second
    /// rename would silently replace the first chat's session. `Ok(false)`
    /// when a session, or another chat's claim, already has the name.
    ///
    /// The empty placeholder is replaced by the real save moments later.
    /// `list` ignores it if a crash leaves it behind, and `release` removes
    /// it if the save fails.
    fn reserve(&self, name: &str) -> Result<bool> {
        let path = self.path_for(name)?;
        create_private_dir(&self.dir)
            .with_context(|| format!("failed to create {}", self.dir.display()))?;
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&path) {
            Ok(_) => Ok(true),
            Err(e) if e.kind() == ErrorKind::AlreadyExists => Ok(false),
            Err(e) => Err(e).with_context(|| format!("failed to create {}", path.display())),
        }
    }

    /// Give up a name `reserve` claimed for a save that then failed, so it
    /// isn't blocked by a session nobody can see or load. Only the empty
    /// placeholder is removed, never a real session.
    fn release(&self, name: &str) {
        if let Ok(path) = self.path_for(name) {
            if std::fs::metadata(&path).is_ok_and(|m| m.len() == 0) {
                let _ = std::fs::remove_file(&path);
            }
        }
    }

    /// `base`, or `base-1`, `base-2`, … if a session with that name exists.
    /// The name returned is reserved (see `reserve`), so two chats
    /// auto-saving in the same second can't both pick the same timestamp
    /// name and then overwrite each other on rename.
    pub fn unused_name(&self, base: &str) -> Result<String> {
        for n in 0..=MAX_NAME_SUFFIX {
            let name = if n == 0 {
                base.to_string()
            } else {
                format!("{}-{}", base, n)
            };
            if self.reserve(&name)? {
                return Ok(name);
            }
        }
        anyhow::bail!("no free session name like {}-N", base)
    }

    /// Write `session` as `name`, replacing an existing file atomically: the
    /// JSON goes to a temp file in the same directory, is flushed to disk,
    /// then renamed over the target. Returns the file path.
    pub fn save(&self, name: &str, session: &Session) -> Result<PathBuf> {
        let path = self.path_for(name)?;
        create_private_dir(&self.dir)
            .with_context(|| format!("failed to create {}", self.dir.display()))?;
        let json = serde_json::to_vec_pretty(session).context("failed to serialize the session")?;
        // Same directory as the target, so the rename never crosses
        // filesystems. The pid keeps two chats saving the same name from
        // sharing a temp file; the leading '.' and the `.tmp` extension keep
        // it out of listings.
        let tmp = self
            .dir
            .join(format!(".{}.{}.tmp", name, std::process::id()));
        if let Err(e) = write_private(&tmp, &json).and_then(|()| std::fs::rename(&tmp, &path)) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e).with_context(|| format!("failed to save session to {}", path.display()));
        }
        Ok(path)
    }

    pub fn load(&self, name: &str) -> Result<Session> {
        let path = self.path_for(name)?;
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == ErrorKind::NotFound => {
                anyhow::bail!("no saved session named {} (/sessions lists them)", name)
            }
            Err(e) => return Err(e).with_context(|| format!("failed to read {}", path.display())),
        };
        parse_session(&bytes).with_context(|| format!("failed to read {}", path.display()))
    }

    /// All saved sessions, newest first. A missing directory is an empty
    /// list (nothing saved yet). Files whose stem isn't a valid session name
    /// weren't written by ai-cli and can't be loaded by name, so they're
    /// skipped.
    pub fn list(&self) -> Result<Listing> {
        let entries = match std::fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok(Listing::default()),
            Err(e) => {
                return Err(e).with_context(|| format!("failed to read {}", self.dir.display()))
            }
        };
        let mut listing = Listing::default();
        for entry in entries {
            let path = entry
                .with_context(|| format!("failed to read {}", self.dir.display()))?
                .path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") || !path.is_file() {
                continue;
            }
            // An empty file is a name reserved by `unused_name` whose save
            // never happened (or is happening right now); not a session.
            if std::fs::metadata(&path).is_ok_and(|m| m.len() == 0) {
                continue;
            }
            let Some(name) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if validate_name(name).is_err() {
                continue;
            }
            let parsed = std::fs::read(&path)
                .map_err(anyhow::Error::from)
                .and_then(|bytes| parse_session(&bytes));
            match parsed {
                Ok(session) => listing.sessions.push(SessionSummary::of(name, &session)),
                Err(e) => listing
                    .unreadable
                    .push((name.to_string(), format!("{:#}", e))),
            }
        }
        listing
            .sessions
            .sort_by(|a, b| b.updated.cmp(&a.updated).then_with(|| a.name.cmp(&b.name)));
        listing.unreadable.sort();
        Ok(listing)
    }
}

/// Parse a session file. The version is checked before the full parse so a
/// newer layout gets a clear message instead of a missing-field error.
fn parse_session(bytes: &[u8]) -> Result<Session> {
    let value: serde_json::Value = serde_json::from_slice(bytes).context("not valid JSON")?;
    let version = value
        .get("version")
        .and_then(serde_json::Value::as_u64)
        .context("not an ai-cli session file (no format version)")?;
    if version > u64::from(FORMAT_VERSION) {
        anyhow::bail!(
            "saved by a newer ai-cli (format version {}; this build reads up to {})",
            version,
            FORMAT_VERSION
        );
    }
    serde_json::from_value(value).context("unexpected session layout")
}

/// Create `dir` (and missing parents). On Unix, `dir` itself is created
/// owner-only (0700) so session names aren't visible to other users either;
/// parents get the default mode.
fn create_private_dir(dir: &Path) -> std::io::Result<()> {
    if dir.is_dir() {
        return Ok(());
    }
    if let Some(parent) = dir.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Only Unix sets anything on the builder.
    #[cfg_attr(not(unix), allow(unused_mut))]
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    match builder.create(dir) {
        // Another chat created it in the meantime.
        Err(e) if e.kind() == ErrorKind::AlreadyExists && dir.is_dir() => Ok(()),
        other => other,
    }
}

/// Create `path` owner-only (0600 on Unix) and write `bytes` to it, flushed
/// to disk. A temp file left behind by a crashed run with the same pid is
/// replaced.
fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    match std::fs::remove_file(path) {
        Err(e) if e.kind() != ErrorKind::NotFound => return Err(e),
        _ => {}
    }
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

/// What `Tracker::load` did.
#[derive(Debug)]
pub struct Loaded {
    pub session: Session,
    /// Name the previous conversation was saved under first, when it had
    /// unsaved turns.
    pub stashed: Option<String>,
}

/// Ties the conversation in memory to its saved session, if any. Owned by
/// the chat loop. Timestamps are passed in so tests can pin them.
#[derive(Debug)]
pub struct Tracker {
    /// `None` when the platform has no data dir; saving and loading then
    /// report an error, and the exit autosave is skipped.
    store: Option<SessionStore>,
    /// The session this conversation was loaded from or last saved as.
    /// `/save` without a name and the exit autosave reuse it.
    name: Option<String>,
    /// When the conversation's first turn happened, or the loaded session's
    /// `created`.
    created: Option<u64>,
    /// A turn completed since the last save or load.
    dirty: bool,
}

impl Tracker {
    pub fn new(store: Option<SessionStore>) -> Self {
        Self {
            store,
            name: None,
            created: None,
            dirty: false,
        }
    }

    pub fn store(&self) -> Option<&SessionStore> {
        self.store.as_ref()
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Record a completed turn.
    pub fn turn_completed(&mut self, now: u64) {
        self.created.get_or_insert(now);
        self.dirty = true;
    }

    /// Forget the saved session (after `/clear`), so the fresh conversation
    /// is saved as a new session instead of overwriting the old one.
    pub fn detach(&mut self) {
        self.name = None;
        self.created = None;
        self.dirty = false;
    }

    fn store_or_err(&self) -> Result<&SessionStore> {
        self.store
            .as_ref()
            .context("sessions are unavailable: this system has no data directory")
    }

    /// `/save [name]`. Without a name the conversation keeps its current
    /// name, or gets a fresh timestamp name. A different session that
    /// already has the requested name is only replaced with `force`
    /// (`/save! <name>`). Returns the name and file written.
    pub fn save(
        &mut self,
        requested: Option<&str>,
        force: bool,
        provider: &Provider,
        messages: &[ChatMessage],
        now: u64,
    ) -> Result<(String, PathBuf)> {
        let store = self.store_or_err()?;
        if user_turns(messages) == 0 {
            anyhow::bail!("nothing to save yet");
        }
        // `reserved`: the name was claimed just now for this save.
        let (name, reserved) = match requested {
            Some(name) => {
                let is_current = self.name.as_deref() == Some(name);
                // A name that's new to this conversation is claimed
                // atomically, so another chat running `/save <name>` at the
                // same moment can't slip in between the check and the write.
                let claim = !is_current && !force;
                if claim && !store.reserve(name)? {
                    anyhow::bail!(
                        "a session named {} already exists; use `/save! {}` to replace it",
                        name,
                        name
                    );
                }
                (name.to_string(), claim)
            }
            None => match &self.name {
                Some(name) => (name.clone(), false),
                None => (store.unused_name(&timestamp_name(now))?, true),
            },
        };
        let path = self.write(name.clone(), reserved, provider, messages, now)?;
        Ok((name, path))
    }

    /// Save on exit, so `--continue` can pick the conversation up again.
    /// Only when a turn completed since the last save or load. A named
    /// session is saved under its name, an unnamed one as `last`. Returns
    /// the name used, or `None` when there was nothing to save.
    pub fn autosave(
        &mut self,
        provider: &Provider,
        messages: &[ChatMessage],
        now: u64,
    ) -> Result<Option<String>> {
        if !self.dirty || user_turns(messages) == 0 || self.store.is_none() {
            return Ok(None);
        }
        let name = self
            .name
            .clone()
            .unwrap_or_else(|| AUTOSAVE_NAME.to_string());
        self.write(name.clone(), false, provider, messages, now)?;
        Ok(Some(name))
    }

    /// `/load <name>` and `--continue`: read session `name` and make it the
    /// current one. The caller swaps in `Loaded::session.messages`.
    ///
    /// Unsaved turns in the conversation being replaced are saved first, so
    /// loading never throws work away. They go under the conversation's own
    /// name, or under a fresh timestamp name when it has none (not `last`,
    /// which may be the very session being loaded) or when it *is* the
    /// session being loaded (so `/load <current>` still gets the saved copy
    /// back).
    pub fn load(
        &mut self,
        name: &str,
        provider: &Provider,
        messages: &[ChatMessage],
        now: u64,
    ) -> Result<Loaded> {
        let store = self.store_or_err()?;
        // Read first: a typo or a corrupt file must not stash anything.
        let session = store.load(name)?;
        let stash_name = if self.dirty && user_turns(messages) > 0 {
            // The flag says whether the name was claimed just now.
            Some(match self.name.as_deref() {
                Some(current) if current != name => (current.to_string(), false),
                _ => (store.unused_name(&timestamp_name(now))?, true),
            })
        } else {
            None
        };
        if let Some((stash, reserved)) = &stash_name {
            self.write(stash.clone(), *reserved, provider, messages, now)?;
        }
        let stash_name = stash_name.map(|(stash, _)| stash);
        self.name = Some(name.to_string());
        self.created = Some(session.created);
        self.dirty = false;
        Ok(Loaded {
            session,
            stashed: stash_name,
        })
    }

    /// Write the conversation as session `name` and make that the current
    /// session. `reserved` means the caller claimed `name` just now for this
    /// write (with `reserve` or `unused_name`): if the write fails, the claim
    /// is given up, so the name isn't left blocked by an empty placeholder
    /// nobody can list or load.
    fn write(
        &mut self,
        name: String,
        reserved: bool,
        provider: &Provider,
        messages: &[ChatMessage],
        now: u64,
    ) -> Result<PathBuf> {
        let store = self.store_or_err()?;
        let created = self.created.unwrap_or(now);
        let session = Session {
            version: FORMAT_VERSION,
            provider: provider.display_name.clone(),
            model: provider.model.clone(),
            created,
            updated: now,
            messages: messages.to_vec(),
        };
        let path = match store.save(&name, &session) {
            Ok(path) => path,
            Err(e) => {
                if reserved {
                    // Only removes the empty placeholder, never a session.
                    store.release(&name);
                }
                return Err(e);
            }
        };
        self.name = Some(name);
        self.created = Some(created);
        self.dirty = false;
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use genai::chat::{
        BinarySource, CacheControl, ContentPart, MessageContent, ToolCall, ToolResponse,
    };
    use serde_json::json;

    /// Fresh, empty per-test directory under the system temp dir.
    fn test_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ai-cli-sessions-{}-{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn provider(name: &str, model: &str) -> Provider {
        Provider {
            name: name.to_lowercase(),
            display_name: name.to_string(),
            model: model.to_string(),
            reasoning: None,
            service_tier: None,
            color: colored::Color::White,
        }
    }

    fn session_at(updated: u64, messages: Vec<ChatMessage>) -> Session {
        Session {
            version: FORMAT_VERSION,
            provider: "CLAUDE".to_string(),
            model: "claude-test".to_string(),
            created: updated,
            updated,
            messages,
        }
    }

    fn chat(user: &str) -> Vec<ChatMessage> {
        vec![ChatMessage::user(user), ChatMessage::assistant("ok")]
    }

    fn role(m: &ChatMessage) -> String {
        m.role.to_string()
    }

    /// Every kind of content chat mode can produce, plus the parts it can't
    /// produce yet (binary attachments, cache-control options) to show they
    /// survive too.
    fn every_kind_of_message() -> Vec<ChatMessage> {
        let call = ToolCall {
            call_id: "call_1".to_string(),
            fn_name: "web_search".to_string(),
            fn_arguments: json!({"query": "rust \"serde\" news", "n": 3}),
            thought_signatures: Some(vec!["sig-abc".to_string()]),
        };
        vec![
            ChatMessage::system("be brief"),
            ChatMessage::user("what's new?\n\n```text\npiped context\n```"),
            // With thought signatures on the first call, genai prepends
            // them as ThoughtSignature parts (Gemini needs them back).
            ChatMessage::from(vec![call]),
            ChatMessage::from(ToolResponse::new("call_1", "results: héllo wörld")),
            ChatMessage::assistant("Rust 1.90 shipped."),
            ChatMessage::user(MessageContent::from_parts(vec![
                ContentPart::from_text("and this image?"),
                ContentPart::from_binary_base64("image/png", "aGVsbG8=", Some("a.png".into())),
            ]))
            .with_options(CacheControl::Ephemeral),
        ]
    }

    #[test]
    fn auto_names_are_reserved_atomically_and_hidden_until_saved() {
        // Two chats auto-saving in the same second must get distinct names:
        // the first reservation claims the file, so the second can't reuse it.
        let store = SessionStore::new(test_dir("reserve"));
        let first = store.unused_name("chat-20260930-120000").unwrap();
        let second = store.unused_name("chat-20260930-120000").unwrap();
        assert_eq!(first, "chat-20260930-120000");
        assert_eq!(second, "chat-20260930-120000-1");
        // Reserved-but-unsaved placeholders are not listed as sessions.
        let listing = store.list().unwrap();
        assert!(listing.sessions.is_empty());
        assert!(listing.unreadable.is_empty());
    }

    #[test]
    fn round_trip_preserves_every_message_part() {
        let store = SessionStore::new(test_dir("roundtrip"));
        let original = every_kind_of_message();
        let saved = Session {
            version: FORMAT_VERSION,
            provider: "GEMINI".to_string(),
            model: "gemini-3.5-flash".to_string(),
            created: 1_700_000_000,
            updated: 1_700_000_500,
            messages: original.clone(),
        };
        store.save("roundtrip", &saved).unwrap();
        let loaded = store.load("roundtrip").unwrap();

        assert_eq!(loaded.version, FORMAT_VERSION);
        assert_eq!(loaded.provider, "GEMINI");
        assert_eq!(loaded.model, "gemini-3.5-flash");
        assert_eq!(loaded.created, 1_700_000_000);
        assert_eq!(loaded.updated, 1_700_000_500);

        let m = &loaded.messages;
        let roles: Vec<String> = m.iter().map(role).collect();
        assert_eq!(
            roles,
            ["System", "User", "Assistant", "Tool", "Assistant", "User"]
        );
        assert_eq!(m[0].content.first_text(), Some("be brief"));
        assert_eq!(
            m[1].content.first_text(),
            Some("what's new?\n\n```text\npiped context\n```")
        );

        // Assistant tool-call turn: the thought signature part, then the
        // call with its own copy of the signatures.
        let parts = m[2].content.parts();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].as_thought_signature(), Some("sig-abc"));
        let call = parts[1].as_tool_call().expect("tool call part");
        assert_eq!(call.call_id, "call_1");
        assert_eq!(call.fn_name, "web_search");
        assert_eq!(
            call.fn_arguments,
            json!({"query": "rust \"serde\" news", "n": 3})
        );
        assert_eq!(call.thought_signatures, Some(vec!["sig-abc".to_string()]));

        let response = m[3].content.parts()[0]
            .as_tool_response()
            .expect("tool response part");
        assert_eq!(response.call_id, "call_1");
        assert_eq!(response.content, "results: héllo wörld");

        assert_eq!(m[4].content.first_text(), Some("Rust 1.90 shipped."));

        let parts = m[5].content.parts();
        assert_eq!(parts[0].as_text(), Some("and this image?"));
        let binary = parts[1].as_binary().expect("binary part");
        assert_eq!(binary.content_type, "image/png");
        assert_eq!(binary.name.as_deref(), Some("a.png"));
        assert!(matches!(&binary.source, BinarySource::Base64(b) if &**b == "aGVsbG8="));
        assert!(matches!(
            m[5].options.as_ref().and_then(|o| o.cache_control.as_ref()),
            Some(CacheControl::Ephemeral)
        ));
        assert!(m[0].options.is_none());

        // Belt and braces: nothing else was lost or altered either.
        assert_eq!(
            serde_json::to_value(&original).unwrap(),
            serde_json::to_value(&loaded.messages).unwrap()
        );
    }

    #[test]
    fn save_writes_an_owner_only_file_atomically() {
        let dir = test_dir("private").join("nested").join("chats");
        let store = SessionStore::new(&dir);
        let path = store.save("proj", &session_at(10, chat("first"))).unwrap();
        assert_eq!(path, dir.join("proj.json"));

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600, "session file mode {:o}", mode);
            let mode = std::fs::metadata(&dir).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o700, "sessions dir mode {:o}", mode);
        }

        // Replacing an existing session leaves exactly one file behind: the
        // temp file was renamed, not left next to it.
        store.save("proj", &session_at(20, chat("second"))).unwrap();
        let files: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(files, ["proj.json"]);
        let loaded = store.load("proj").unwrap();
        assert_eq!(loaded.updated, 20);
        assert_eq!(loaded.messages[0].content.first_text(), Some("second"));
    }

    #[test]
    fn validate_name_allows_only_plain_file_names() {
        for ok in [
            "last",
            "proj-x",
            "my_chat.v2",
            "A1",
            "chat-20260930-142501",
            // Only exact device names are reserved.
            "console",
            "nul-check",
            "lpt10",
        ] {
            assert!(validate_name(ok).is_ok(), "{ok:?} should be valid");
        }
        let too_long = "x".repeat(MAX_NAME_LEN + 1);
        for bad in [
            "",
            ".",
            "..",
            ".hidden",
            "../escape",
            "a/b",
            "a\\b",
            "C:x",
            "has space",
            "tab\there",
            "ünïcode",
            "nul\0",
            "quote\"",
            "nul",
            "CON",
            "com1.v2",
            too_long.as_str(),
        ] {
            assert!(validate_name(bad).is_err(), "{bad:?} should be rejected");
        }
        assert!(validate_name(&"x".repeat(MAX_NAME_LEN)).is_ok());

        // Every entry point validates, so a bad name never reaches the
        // filesystem.
        let store = SessionStore::new(test_dir("names"));
        assert!(store.save("../evil", &session_at(1, chat("x"))).is_err());
        assert!(store.load("../evil").is_err());
        assert!(store.unused_name("a/b").is_err());
    }

    #[test]
    fn list_is_newest_first_and_reports_unreadable_files() {
        let dir = test_dir("list");
        let store = SessionStore::new(&dir);
        assert!(store.list().unwrap().sessions.is_empty());

        store.save("old", &session_at(100, chat("oldest"))).unwrap();
        store.save("new", &session_at(300, chat("newest"))).unwrap();
        store
            .save("mid-b", &session_at(200, chat("tie b")))
            .unwrap();
        store
            .save("mid-a", &session_at(200, chat("tie a")))
            .unwrap();
        std::fs::write(dir.join("broken.json"), "{ not json").unwrap();
        std::fs::write(
            dir.join("future.json"),
            json!({"version": FORMAT_VERSION + 1}).to_string(),
        )
        .unwrap();
        // Not sessions: wrong extension, invalid name, leftover temp file.
        std::fs::write(dir.join("notes.txt"), "x").unwrap();
        std::fs::write(dir.join("has space.json"), "{}").unwrap();
        std::fs::write(dir.join(".proj.123.tmp"), "{}").unwrap();

        let listing = store.list().unwrap();
        let names: Vec<&str> = listing.sessions.iter().map(|s| s.name.as_str()).collect();
        // Newest first; equal timestamps fall back to name order.
        assert_eq!(names, ["new", "mid-a", "mid-b", "old"]);
        let unreadable: Vec<&str> = listing.unreadable.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(unreadable, ["broken", "future"]);
        assert!(listing.unreadable[1].1.contains("newer ai-cli"));

        let newest = &listing.sessions[0];
        assert_eq!(newest.turns, 1);
        assert_eq!(newest.model, "claude-test");
        assert_eq!(newest.updated, 300);
        assert_eq!(newest.preview.as_deref(), Some("newest"));
    }

    #[test]
    fn summary_preview_is_first_user_message_collapsed_and_truncated() {
        let long = format!("  explain\n\nthis:  {}", "word ".repeat(40));
        let messages = vec![
            ChatMessage::system("sys"),
            ChatMessage::user(long),
            ChatMessage::assistant("a"),
            ChatMessage::user("second"),
        ];
        let summary = SessionSummary::of("s", &session_at(1, messages));
        assert_eq!(summary.turns, 2);
        let preview = summary.preview.unwrap();
        assert!(preview.starts_with("explain this: word word"), "{preview}");
        assert!(preview.ends_with('…'));
        assert_eq!(preview.chars().count(), PREVIEW_MAX_CHARS + 1);
    }

    #[test]
    fn load_reports_missing_newer_and_corrupt_sessions() {
        let dir = test_dir("load-errors");
        let store = SessionStore::new(&dir);
        let err = format!("{:#}", store.load("nope").unwrap_err());
        assert!(err.contains("no saved session named nope"), "{err}");

        std::fs::write(
            dir.join("future.json"),
            json!({"version": 99, "whatever": []}).to_string(),
        )
        .unwrap();
        let err = format!("{:#}", store.load("future").unwrap_err());
        assert!(err.contains("newer ai-cli"), "{err}");

        std::fs::write(dir.join("junk.json"), "[1, 2]").unwrap();
        let err = format!("{:#}", store.load("junk").unwrap_err());
        assert!(err.contains("no format version"), "{err}");
    }

    #[test]
    fn dates_and_ages() {
        assert_eq!(timestamp_name(0), "chat-19700101-000000");
        assert_eq!(timestamp_name(1_700_000_000), "chat-20231114-221320");
        // Leap day, and the day after.
        assert_eq!(utc_date(951_782_400), "2000-02-29");
        assert_eq!(utc_date(951_868_800), "2000-03-01");
        assert_eq!(utc_date(4_102_444_800), "2100-01-01");

        let now = 1_700_000_000;
        assert_eq!(format_age(now, now), "just now");
        assert_eq!(format_age(now - 59, now), "just now");
        assert_eq!(format_age(now - 60, now), "1m ago");
        assert_eq!(format_age(now - 3_599, now), "59m ago");
        assert_eq!(format_age(now - 7_200, now), "2h ago");
        assert_eq!(format_age(now - 86_400 * 3, now), "3d ago");
        assert_eq!(format_age(now - 86_400 * 31, now), "2023-10-14");
        // A timestamp from the future (clock moved back) shows its date.
        assert_eq!(format_age(now + 86_400, now), "2023-11-15");
    }

    #[test]
    fn default_location_is_under_ai_cli_data_dir() {
        if let Some(store) = SessionStore::default_location() {
            assert!(store.dir().ends_with("ai-cli/chats"), "{:?}", store.dir());
        }
    }

    #[test]
    fn autosave_uses_last_for_unnamed_chats_and_the_session_name_otherwise() {
        let store = SessionStore::new(test_dir("autosave"));
        let claude = provider("CLAUDE", "claude-test");
        let mut tracker = Tracker::new(Some(store.clone()));

        // Nothing happened yet: no file.
        let system_only = vec![ChatMessage::system("sys")];
        assert_eq!(tracker.autosave(&claude, &system_only, 50).unwrap(), None);

        let mut messages = chat("hello");
        tracker.turn_completed(100);
        assert_eq!(
            tracker
                .autosave(&claude, &messages, 200)
                .unwrap()
                .as_deref(),
            Some(AUTOSAVE_NAME)
        );
        let last = store.load(AUTOSAVE_NAME).unwrap();
        assert_eq!((last.created, last.updated), (100, 200));
        assert_eq!(last.provider, "CLAUDE");
        // No new turn since: nothing to write.
        assert_eq!(tracker.autosave(&claude, &messages, 300).unwrap(), None);

        // After `/save proj`, the autosave follows the name, and `last` is
        // left alone.
        tracker
            .save(Some("proj"), false, &claude, &messages, 400)
            .unwrap();
        messages.extend(chat("more"));
        tracker.turn_completed(500);
        assert_eq!(
            tracker
                .autosave(&claude, &messages, 600)
                .unwrap()
                .as_deref(),
            Some("proj")
        );
        assert_eq!(store.load("proj").unwrap().messages.len(), 4);
        assert_eq!(store.load(AUTOSAVE_NAME).unwrap().updated, 200);

        // `/clear` starts a new conversation: it's autosaved as `last`
        // again, never over `proj`.
        tracker.detach();
        tracker.turn_completed(700);
        assert_eq!(
            tracker
                .autosave(&claude, &chat("new"), 800)
                .unwrap()
                .as_deref(),
            Some(AUTOSAVE_NAME)
        );
        assert_eq!(store.load("proj").unwrap().updated, 600);

        // A dirty conversation with no user turns (e.g. only a system
        // prompt) isn't saved.
        let mut tracker = Tracker::new(Some(store.clone()));
        tracker.turn_completed(900);
        assert_eq!(tracker.autosave(&claude, &system_only, 900).unwrap(), None);

        // No data dir: autosave quietly does nothing.
        let mut tracker = Tracker::new(None);
        tracker.turn_completed(1);
        assert_eq!(tracker.autosave(&claude, &messages, 1).unwrap(), None);
    }

    #[test]
    fn save_names_default_to_current_name_then_timestamp() {
        let store = SessionStore::new(test_dir("save-names"));
        let claude = provider("CLAUDE", "claude-test");
        let messages = chat("hi");
        let now = 1_700_000_000;

        let mut tracker = Tracker::new(Some(store.clone()));
        let (name, _) = tracker.save(None, false, &claude, &messages, now).unwrap();
        assert_eq!(name, "chat-20231114-221320");
        assert_eq!(tracker.name(), Some("chat-20231114-221320"));
        // Saving again keeps the name and overwrites its own file.
        let (again, _) = tracker.save(None, false, &claude, &messages, now).unwrap();
        assert_eq!(again, name);

        // Another conversation saved in the same second gets a suffix.
        let mut other = Tracker::new(Some(store.clone()));
        let (name2, _) = other.save(None, false, &claude, &messages, now).unwrap();
        assert_eq!(name2, "chat-20231114-221320-1");

        // Nothing to save before the first turn.
        let err = Tracker::new(Some(store.clone()))
            .save(None, false, &claude, &[ChatMessage::system("s")], now)
            .unwrap_err();
        assert!(err.to_string().contains("nothing to save"), "{err}");

        // Bad names are rejected before anything is written.
        assert!(tracker
            .save(Some("a/b"), false, &claude, &messages, now)
            .is_err());
        assert_eq!(tracker.name(), Some("chat-20231114-221320"));

        // No data dir.
        let err = Tracker::new(None)
            .save(None, false, &claude, &messages, now)
            .unwrap_err();
        assert!(err.to_string().contains("no data directory"), "{err}");
    }

    #[test]
    fn save_refuses_to_replace_another_session_without_force() {
        let store = SessionStore::new(test_dir("save-force"));
        let claude = provider("CLAUDE", "claude-test");
        store
            .save("taken", &session_at(1, chat("precious")))
            .unwrap();

        let mut tracker = Tracker::new(Some(store.clone()));
        let err = tracker
            .save(Some("taken"), false, &claude, &chat("mine"), 10)
            .unwrap_err();
        assert!(err.to_string().contains("/save! taken"), "{err}");
        assert_eq!(
            store.load("taken").unwrap().messages[0]
                .content
                .first_text(),
            Some("precious")
        );
        assert_eq!(tracker.name(), None);

        tracker
            .save(Some("taken"), true, &claude, &chat("mine"), 10)
            .unwrap();
        assert_eq!(
            store.load("taken").unwrap().messages[0]
                .content
                .first_text(),
            Some("mine")
        );
        // Once it's the current session, plain `/save taken` is fine.
        tracker
            .save(Some("taken"), false, &claude, &chat("mine again"), 20)
            .unwrap();
    }

    #[test]
    fn concurrent_saves_under_a_new_name_never_replace_each_other() {
        // Several chats run `/save shared` at the same moment. Exactly one
        // gets the name; the others are told it's taken instead of silently
        // replacing that chat's session.
        let dir = test_dir("save-race");
        let chats = 8;
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(chats));
        let handles: Vec<_> = (0..chats)
            .map(|i| {
                let dir = dir.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let mut tracker = Tracker::new(Some(SessionStore::new(dir)));
                    let messages = chat(&format!("chat {}", i));
                    let claude = provider("CLAUDE", "claude-test");
                    barrier.wait();
                    tracker
                        .save(Some("shared"), false, &claude, &messages, 10)
                        .map(|_| i)
                })
            })
            .collect();
        let results: Vec<Result<usize>> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        let winners: Vec<usize> = results
            .iter()
            .filter_map(|r| r.as_ref().ok())
            .copied()
            .collect();
        assert_eq!(winners.len(), 1, "{results:?}");
        for err in results.iter().filter_map(|r| r.as_ref().err()) {
            assert!(err.to_string().contains("already exists"), "{err}");
        }
        let saved = SessionStore::new(&dir).load("shared").unwrap();
        let expected = format!("chat {}", winners[0]);
        assert_eq!(
            saved.messages[0].content.first_text(),
            Some(expected.as_str())
        );
    }

    #[test]
    fn a_failed_save_gives_its_new_name_back() {
        let dir = test_dir("save-release");
        let store = SessionStore::new(&dir);
        let claude = provider("CLAUDE", "claude-test");
        let mut tracker = Tracker::new(Some(store.clone()));
        // A directory where the save's temp file goes makes the write fail.
        let blocker = dir.join(format!(".proj.{}.tmp", std::process::id()));
        std::fs::create_dir_all(&blocker).unwrap();
        assert!(tracker
            .save(Some("proj"), false, &claude, &chat("x"), 10)
            .is_err());
        assert!(!store.path_for("proj").unwrap().exists());
        assert_eq!(tracker.name(), None);

        // So once the problem is gone, the same `/save proj` just works.
        std::fs::remove_dir(&blocker).unwrap();
        tracker
            .save(Some("proj"), false, &claude, &chat("x"), 20)
            .unwrap();
        assert_eq!(store.load("proj").unwrap().updated, 20);

        // A real session is never removed by a failed save: `/save!` over
        // it fails the same way and leaves it in place.
        std::fs::create_dir_all(&blocker).unwrap();
        let mut other = Tracker::new(Some(store.clone()));
        assert!(other
            .save(Some("proj"), true, &claude, &chat("y"), 30)
            .is_err());
        assert_eq!(store.load("proj").unwrap().updated, 20);

        // The same goes for a timestamp name picked by `/save` alone.
        let now = 1_700_000_000;
        let auto = timestamp_name(now);
        let blocker = dir.join(format!(".{}.{}.tmp", auto, std::process::id()));
        std::fs::create_dir_all(&blocker).unwrap();
        assert!(other.save(None, false, &claude, &chat("y"), now).is_err());
        assert!(!store.path_for(&auto).unwrap().exists());
        std::fs::remove_dir(&blocker).unwrap();
        let (name, _) = other.save(None, false, &claude, &chat("y"), now).unwrap();
        assert_eq!(name, auto);
    }

    #[test]
    fn a_failed_load_stash_gives_its_new_name_back() {
        let dir = test_dir("load-release");
        let store = SessionStore::new(&dir);
        let claude = provider("CLAUDE", "claude-test");
        store
            .save("other", &session_at(5, chat("other chat")))
            .unwrap();
        let now = 1_700_000_000;
        // A directory where the stash's temp file goes makes its write fail
        // after `unused_name` has claimed the name.
        let blocker_for = |name: &str| dir.join(format!(".{}.{}.tmp", name, std::process::id()));

        // Unnamed conversation with unsaved turns: the load fails, the
        // claimed timestamp name is given back and nothing changes.
        let mut tracker = Tracker::new(Some(store.clone()));
        let current = chat("work in progress");
        tracker.turn_completed(now - 10);
        let stash = timestamp_name(now);
        std::fs::create_dir_all(blocker_for(&stash)).unwrap();
        assert!(tracker.load("other", &claude, &current, now).is_err());
        assert!(!store.path_for(&stash).unwrap().exists());
        assert_eq!(tracker.name(), None);

        // So once the problem is gone, the retry stashes under that name.
        std::fs::remove_dir(blocker_for(&stash)).unwrap();
        let loaded = tracker.load("other", &claude, &current, now).unwrap();
        assert_eq!(loaded.stashed.as_deref(), Some(stash.as_str()));

        // Reloading the current session with unsaved edits fails the same
        // way; the session being loaded is a real file and stays intact.
        let later = now + 60;
        let stash = timestamp_name(later);
        std::fs::create_dir_all(blocker_for(&stash)).unwrap();
        tracker.turn_completed(later);
        assert!(tracker
            .load("other", &claude, &chat("edited"), later)
            .is_err());
        assert!(!store.path_for(&stash).unwrap().exists());
        assert_eq!(
            store.load("other").unwrap().messages[0]
                .content
                .first_text(),
            Some("other chat")
        );
        assert_eq!(tracker.name(), Some("other"));
        std::fs::remove_dir(blocker_for(&stash)).unwrap();
        let loaded = tracker
            .load("other", &claude, &chat("edited"), later)
            .unwrap();
        assert_eq!(loaded.stashed.as_deref(), Some(stash.as_str()));
    }

    #[test]
    fn load_saves_unsaved_turns_before_replacing_them() {
        let store = SessionStore::new(test_dir("load-stash"));
        let claude = provider("CLAUDE", "claude-test");
        store
            .save("other", &session_at(5, chat("other chat")))
            .unwrap();
        let now = 1_700_000_000;

        // Unnamed conversation with a turn: stashed under a timestamp name,
        // never `last`.
        let mut tracker = Tracker::new(Some(store.clone()));
        let current = chat("work in progress");
        tracker.turn_completed(now - 10);

        // A typo stashes nothing and changes nothing.
        assert!(tracker.load("nope", &claude, &current, now).is_err());
        assert!(!store.path_for("chat-20231114-221320").unwrap().exists());
        assert_eq!(tracker.name(), None);

        let loaded = tracker.load("other", &claude, &current, now).unwrap();
        assert_eq!(loaded.stashed.as_deref(), Some("chat-20231114-221320"));
        assert_eq!(
            loaded.session.messages[0].content.first_text(),
            Some("other chat")
        );
        assert_eq!(tracker.name(), Some("other"));
        let stash = store.load("chat-20231114-221320").unwrap();
        assert_eq!(
            stash.messages[0].content.first_text(),
            Some("work in progress")
        );
        assert_eq!(stash.created, now - 10);
        assert!(!store.path_for(AUTOSAVE_NAME).unwrap().exists());

        // Loading with nothing new to save stashes nothing.
        let loaded = tracker
            .load("other", &claude, &loaded.session.messages, now)
            .unwrap();
        assert_eq!(loaded.stashed, None);

        // Unsaved turns in a named session go to that session...
        tracker.turn_completed(now);
        let loaded = tracker
            .load("chat-20231114-221320", &claude, &chat("other, edited"), now)
            .unwrap();
        assert_eq!(loaded.stashed.as_deref(), Some("other"));
        assert_eq!(
            store.load("other").unwrap().messages[0]
                .content
                .first_text(),
            Some("other, edited")
        );

        // ...unless that's the session being loaded: then the saved copy
        // comes back and the edits are kept under a new name.
        tracker.turn_completed(now);
        let loaded = tracker
            .load("chat-20231114-221320", &claude, &chat("edited again"), now)
            .unwrap();
        assert_eq!(loaded.stashed.as_deref(), Some("chat-20231114-221320-1"));
        assert_eq!(
            loaded.session.messages[0].content.first_text(),
            Some("work in progress")
        );
        assert_eq!(tracker.name(), Some("chat-20231114-221320"));
    }

    #[test]
    fn history_helpers() {
        let messages = every_kind_of_message();
        assert_eq!(user_turns(&messages), 2);
        assert_eq!(system_prompt(&messages).as_deref(), Some("be brief"));
        assert!(has_tool_calls(&messages));
        assert!(!has_tool_calls(&chat("x")));
        assert_eq!(system_prompt(&chat("x")), None);
    }
}
