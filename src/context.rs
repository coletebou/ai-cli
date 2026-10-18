//! Expands context into user questions: inline `@file` references and
//! piped stdin content. Used to bridge between CLI args and what the
//! model actually sees.
//!
//! Everything inlined is announced on stderr (`[context] ...`), so a
//! prompt never silently grows a file or a pipe the user didn't mean to
//! send.

use anyhow::{Context, Result};
use std::io::{self, IsTerminal, Read};
use std::path::{Path, PathBuf};

/// Maximum bytes inlined from stdin or an `@file`. Kept in sync so the two
/// input modes share the same ceiling — otherwise a caller could bypass the
/// file-size guard by piping a giant log instead of referencing it as `@`.
const MAX_INLINE_BYTES: usize = 20 * 1024 * 1024;

/// Characters that may sit directly before `@` and still count as a word
/// boundary, so `(@notes.md)` and `"@notes.md"` expand like `@notes.md`.
const OPENING_PUNCT: &[char] = &['(', '[', '"', '\''];

/// Punctuation split off the end of an unquoted `@path` token and kept in
/// the sentence, so `see @notes.md.` resolves to `notes.md`.
const TRAILING_PUNCT: &[char] = &['.', ',', ';', ':', '?', '!', ')', ']', '"', '\''];

const UTF8_BOM: &[u8] = b"\xEF\xBB\xBF";

/// What fd 0 is connected to, reduced to what matters for auto-reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(unix), allow(dead_code))] // only Terminal/Pipe are detected off Unix
enum StdinKind {
    /// An interactive terminal: nothing was piped.
    Terminal,
    /// A pipe, e.g. `cmd | ai ...`.
    Pipe,
    /// A regular file, e.g. `ai ... < notes.md`.
    File,
    /// A socket. Node's `spawn()` gives children a socketpair by default,
    /// and it never reaches EOF while the parent holds it open.
    Socket,
    /// A non-terminal character device such as `/dev/null`.
    CharDevice,
    /// Anything else, or fd 0 couldn't be inspected (e.g. it's closed).
    Other,
}

/// Whether stdin of this kind is read as context without being asked.
///
/// `!is_terminal()` isn't enough: sockets and devices aren't TTYs either,
/// and a socket a parent process holds open (Node's `spawn()`, some `ssh`
/// sessions) blocks the read forever. Pipes and redirected files are what
/// `|` and `<` create, so only those are read. A pipe that is held open but
/// never written to still blocks; that is indistinguishable from a slow
/// `make | ai ...`, which is what `--no-stdin` is for.
fn should_auto_read(kind: StdinKind) -> bool {
    matches!(kind, StdinKind::Pipe | StdinKind::File)
}

#[cfg(unix)]
fn stdin_kind() -> StdinKind {
    use std::os::fd::AsFd;
    use std::os::unix::fs::FileTypeExt;

    let stdin = io::stdin();
    if stdin.is_terminal() {
        return StdinKind::Terminal;
    }
    // fstat a duplicate of fd 0: wrapping fd 0 itself in a `File` would
    // close stdin when the `File` drops.
    let Ok(fd) = stdin.as_fd().try_clone_to_owned() else {
        return StdinKind::Other;
    };
    let Ok(meta) = std::fs::File::from(fd).metadata() else {
        return StdinKind::Other;
    };
    let ft = meta.file_type();
    if ft.is_fifo() {
        StdinKind::Pipe
    } else if ft.is_file() {
        StdinKind::File
    } else if ft.is_socket() {
        StdinKind::Socket
    } else if ft.is_char_device() {
        StdinKind::CharDevice
    } else {
        StdinKind::Other
    }
}

#[cfg(not(unix))]
fn stdin_kind() -> StdinKind {
    // There's no dependency-free way to ask Windows what the handle is, so
    // keep the plain TTY check; `--no-stdin` covers the cases it gets wrong.
    if io::stdin().is_terminal() {
        StdinKind::Terminal
    } else {
        StdinKind::Pipe
    }
}

/// Read piped stdin as context. Returns `Ok(None)` when stdin isn't a pipe
/// or redirected file (see [`should_auto_read`]) or is blank.
///
/// Oversized input is an error, matching `@file`: answering without the
/// log the user piped in is worse than refusing.
pub fn read_stdin_if_piped() -> Result<Option<String>> {
    if !should_auto_read(stdin_kind()) {
        return Ok(None);
    }
    // Cap the read itself so a `cat huge.log | ai ...` doesn't exhaust
    // memory before we even get a chance to check the size.
    let mut bytes = Vec::new();
    io::stdin()
        .lock()
        .take(MAX_INLINE_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .context("failed to read piped stdin (pass --no-stdin to skip it)")?;
    stdin_text(bytes, &mut |line: String| eprintln!("{line}"))
}

/// Turn raw stdin bytes into context text. The size is checked on the raw
/// bytes before decoding, so a 20 MB cut that lands mid-character reports
/// the size rather than a bogus encoding error. Invalid UTF-8 is replaced
/// rather than dropped: a build log with one Latin-1 byte is still the log
/// the user asked about.
fn stdin_text(bytes: Vec<u8>, report: &mut dyn FnMut(String)) -> Result<Option<String>> {
    if bytes.len() > MAX_INLINE_BYTES {
        anyhow::bail!(
            "piped stdin exceeds 20 MB; refusing to inline it (pass --no-stdin to skip stdin)"
        );
    }
    let size = bytes.len() as u64;
    let text = match decode_text(bytes) {
        Ok(text) => text,
        Err(bytes) => {
            report(
                "[warn] piped stdin is not valid UTF-8; invalid bytes were replaced with U+FFFD"
                    .to_string(),
            );
            String::from_utf8_lossy(&bytes).into_owned()
        }
    };
    if text.trim().is_empty() {
        return Ok(None);
    }
    report(format!(
        "[context] appended piped stdin ({})",
        format_size(size)
    ));
    Ok(Some(text))
}

/// Compose the final question string from a user-provided question and
/// optional piped stdin. Stdin is appended as a fenced code block so it
/// reads consistently with `@file` expansion; the fence is sized so that
/// backticks in the piped content can't terminate the block.
pub fn merge_stdin(question: &str, stdin: Option<String>) -> String {
    match stdin {
        None => question.to_string(),
        Some(content) if question.trim().is_empty() => content,
        Some(content) => {
            let body = content.trim_end();
            let fence = fence_for(body);
            format!("{question}\n\n{fence}text\n{body}\n{fence}")
        }
    }
}

/// A backtick fence that `content` can't close early: one backtick longer
/// than the longest run inside it, and never shorter than the usual three.
/// A fixed ```` ``` ```` would end at the first code block of an inlined
/// README and spill the rest of the file out as prose.
fn fence_for(content: &str) -> String {
    let mut longest = 0;
    let mut run = 0;
    for b in content.bytes() {
        if b == b'`' {
            run += 1;
            longest = longest.max(run);
        } else {
            run = 0;
        }
    }
    "`".repeat((longest + 1).max(3))
}

/// Build the question from the CLI words, inlining `@file` references (see
/// [`expand_refs`]) unless `expand` is false (`--no-expand`).
pub fn question_from_args(args: &[String], expand: bool) -> Result<String> {
    question_from(
        args,
        expand,
        dirs::home_dir().as_deref(),
        &mut |line: String| eprintln!("{line}"),
    )
}

/// Inline `@file` references in text that didn't come from argv, such as a
/// message typed in interactive chat (see [`expand_refs`]). There's no shell
/// quoting to undo there, so this skips `question_from_args`' re-quoting.
// Only interactive chat calls this, and chat lands separately from this
// change, so it is unused in some builds.
#[allow(dead_code)]
pub fn expand_file_refs(text: &str) -> Result<String> {
    expand_refs(text, dirs::home_dir().as_deref(), &mut |line: String| {
        eprintln!("{line}")
    })
}

/// [`question_from_args`] with the home directory and the stderr sink
/// injected, so tests can check `~` expansion and the warnings. The one
/// `home` feeds both the argv re-quoting and the expander, so the two can't
/// disagree about what `~/` means.
fn question_from(
    args: &[String],
    expand: bool,
    home: Option<&Path>,
    report: &mut dyn FnMut(String),
) -> Result<String> {
    if !expand {
        // `--no-expand` sends the words exactly as the shell passed them.
        // The re-quoting in `join_args` only exists to feed the expander;
        // without it the inserted quotes would reach the model, and whether
        // they did would depend on a matching local file existing.
        return Ok(args.join(" "));
    }
    expand_refs(&join_args(args, home), home, report)
}

/// Join CLI words into the question, keeping `@path` arguments with spaces
/// intact. The shell strips the quotes from `ai read @"my notes.md"`, so the
/// program sees the single argument `@my notes.md`; joined naively the
/// expander would then look for a file named `my`. An argument that starts
/// with `@` and names an existing file as a whole is re-quoted as
/// `@"my notes.md"` so it expands as one path.
///
/// The file check resolves `~/` against `home` exactly as the expander
/// will, so `@"~/my notes.md"` (which arrives as `@~/my notes.md`) is kept
/// together too. The re-quoted token keeps the `~/` as written.
fn join_args(args: &[String], home: Option<&Path>) -> String {
    args.iter()
        .map(|arg| match arg.strip_prefix('@') {
            Some(path)
                if path.contains(char::is_whitespace)
                    && !path.contains('"')
                    && resolve_path(path, home).is_file() =>
            {
                format!("@\"{}\"", path)
            }
            _ => arg.clone(),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Expand `@path` tokens in `question` by inlining file contents as
/// fenced code blocks. A token only expands when `@` appears at the start
/// of the string, after whitespace, or after an opening `(`, `[`, `"` or
/// `'` — so email addresses and `foo@bar` literals pass through untouched.
/// Trailing punctuation (`.,;:?!)]"'`) is preserved in the output.
///
/// - `@"my notes.md"` quotes a path containing spaces.
/// - `@~/notes.md` resolves against `home`.
/// - `\@notes.md` is an escape: it sends a literal `@notes.md`.
///
/// Each inlined file is reported through `report` (stderr in the binary)
/// with its size. Paths that don't exist are left as literal text, with a
/// warning when the token looks like a path. Directories, unreadable files
/// and binary content are errors rather than something silently sent (or
/// not sent) to the model.
fn expand_refs(
    question: &str,
    home: Option<&Path>,
    report: &mut dyn FnMut(String),
) -> Result<String> {
    let mut out = String::with_capacity(question.len());
    let mut rest = question;
    let mut prev: Option<char> = None;

    while let Some(c) = rest.chars().next() {
        let at_boundary = prev.is_none_or(|p| p.is_whitespace() || OPENING_PUNCT.contains(&p));
        if at_boundary && c == '\\' && rest[1..].starts_with('@') {
            // `\@notes.md` → a literal `@notes.md`, never expanded.
            out.push('@');
            rest = &rest[2..];
            prev = Some('@');
            continue;
        }
        if at_boundary && c == '@' {
            if let Some(token) = parse_ref(&rest[1..]) {
                let raw = &rest[..1 + token.len];
                match inline_ref(&token, home, report)? {
                    Some(block) => {
                        out.push_str(&block);
                        out.push_str(token.trailing);
                    }
                    None => out.push_str(raw),
                }
                rest = &rest[raw.len()..];
                prev = raw.chars().next_back();
                continue;
            }
        }
        out.push(c);
        rest = &rest[c.len_utf8()..];
        prev = Some(c);
    }
    Ok(out)
}

/// An `@path` token as written, before touching the filesystem.
struct RefToken<'a> {
    /// The path as the user wrote it, without quotes or trailing punctuation.
    path: &'a str,
    /// Punctuation after an unquoted path that belongs to the sentence.
    trailing: &'a str,
    /// Written as `@"..."`, so it is a path even without `/` or `.`.
    quoted: bool,
    /// Bytes consumed after the `@`, including quotes and `trailing`.
    len: usize,
}

/// Parse the text right after an `@`. `None` means there's no usable path
/// (`@` followed by whitespace, only punctuation, or an unclosed quote), in
/// which case the `@` is just a character.
fn parse_ref(after_at: &str) -> Option<RefToken<'_>> {
    if let Some(inner) = after_at.strip_prefix('"') {
        let end = inner.find('"')?;
        let path = &inner[..end];
        if path.is_empty() {
            return None;
        }
        return Some(RefToken {
            path,
            trailing: "",
            quoted: true,
            len: end + 2,
        });
    }
    let len = after_at.find(char::is_whitespace).unwrap_or(after_at.len());
    let token = &after_at[..len];
    // `trim_end_matches` is UTF-8 safe, so multi-byte codepoints in the
    // path (e.g. `@/tmp/résumé.md.`) don't panic.
    let path = token.trim_end_matches(TRAILING_PUNCT);
    if path.is_empty() {
        return None;
    }
    Some(RefToken {
        path,
        trailing: &token[path.len()..],
        quoted: false,
        len,
    })
}

/// Resolve one reference and render it as a fenced block. `Ok(None)` means
/// "leave the token as literal text".
fn inline_ref(
    token: &RefToken,
    home: Option<&Path>,
    report: &mut dyn FnMut(String),
) -> Result<Option<String>> {
    let shown = token.path;
    let path = resolve_path(shown, home);
    let meta = match std::fs::metadata(&path) {
        Ok(meta) => meta,
        Err(e) => {
            // `@alice` is probably a mention; `@notes.md` is probably a typo.
            if token.quoted || looks_like_path(shown) {
                let why = if e.kind() == io::ErrorKind::NotFound {
                    "no such file".to_string()
                } else {
                    e.to_string()
                };
                report(format!(
                    "[warn] @{shown}: {why}; sent as literal text (write \\@ to silence this)"
                ));
            }
            return Ok(None);
        }
    };
    if meta.is_dir() {
        anyhow::bail!(
            "@{shown} is a directory; only files can be inlined (write \\@{shown} to send it as text)"
        );
    }
    if !meta.is_file() {
        anyhow::bail!("@{shown} is not a regular file; refusing to inline it");
    }
    if meta.len() > MAX_INLINE_BYTES as u64 {
        anyhow::bail!(
            "@{} is {:.1} MB; refusing to inline files larger than 20 MB",
            shown,
            meta.len() as f64 / 1_048_576.0
        );
    }
    let (content, size) = read_file_text(&path, shown)?;
    report(format!("[context] inlined {shown} ({})", format_size(size)));

    let lang = Path::new(shown)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    let body = content.trim_end_matches(['\r', '\n']);
    let fence = fence_for(body);
    Ok(Some(format!("\n{fence}{lang} {shown}\n{body}\n{fence}\n")))
}

/// Expand a leading `~/` against `home`. Everything else is used as
/// written, so relative paths resolve against the working directory.
fn resolve_path(path: &str, home: Option<&Path>) -> PathBuf {
    match (path.strip_prefix("~/"), home) {
        (Some(rest), Some(home)) => home.join(rest),
        _ => PathBuf::from(path),
    }
}

/// Whether a token that didn't resolve was likely meant as a path (and so
/// deserves a warning) rather than, say, an `@mention`.
fn looks_like_path(token: &str) -> bool {
    token.contains('.') || token.chars().any(std::path::is_separator)
}

/// Read an `@file` as text, returning it with its size in bytes. UTF-8
/// (with or without a BOM) and UTF-16 with a BOM — what Windows PowerShell
/// 5.1's `>` writes — are accepted. Anything else, including text with NUL
/// bytes (binaries, BOM-less UTF-16), is refused with a hint instead of a
/// bare "stream did not contain valid UTF-8".
fn read_file_text(path: &Path, shown: &str) -> Result<(String, u64)> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .and_then(|f| f.take(MAX_INLINE_BYTES as u64 + 1).read_to_end(&mut bytes))
        .with_context(|| format!("failed to read @{shown}"))?;
    // The metadata check already ran; this guards a file that grew since.
    if bytes.len() > MAX_INLINE_BYTES {
        anyhow::bail!("@{shown} exceeds 20 MB; refusing to inline it");
    }
    let size = bytes.len() as u64;
    match decode_text(bytes) {
        Ok(text) if !text.contains('\0') => Ok((text, size)),
        _ => anyhow::bail!(
            "@{shown} looks binary or non-UTF-8; only UTF-8 or UTF-16 (with a BOM) text can be inlined"
        ),
    }
}

/// Decode inlined bytes: a UTF-8 BOM is stripped, a UTF-16 LE/BE BOM
/// selects UTF-16, and everything else must be UTF-8. On failure the bytes
/// come back (minus any UTF-8 BOM) so stdin can fall back to lossy decoding.
fn decode_text(mut bytes: Vec<u8>) -> Result<String, Vec<u8>> {
    if bytes.starts_with(UTF8_BOM) {
        bytes.drain(..UTF8_BOM.len());
    } else if bytes.starts_with(&[0xFF, 0xFE]) {
        return decode_utf16(&bytes[2..], u16::from_le_bytes).ok_or(bytes);
    } else if bytes.starts_with(&[0xFE, 0xFF]) {
        return decode_utf16(&bytes[2..], u16::from_be_bytes).ok_or(bytes);
    }
    String::from_utf8(bytes).map_err(|e| e.into_bytes())
}

fn decode_utf16(body: &[u8], unit: fn([u8; 2]) -> u16) -> Option<String> {
    // Odd length can't be UTF-16. Written as `% 2 == 1` and indexed pairs
    // (rather than `chunks_exact(2)`) because newer clippy steers both
    // toward APIs that need a newer Rust than this crate supports.
    if body.len() % 2 == 1 {
        return None;
    }
    let units = (0..body.len())
        .step_by(2)
        .map(|i| unit([body[i], body[i + 1]]));
    char::decode_utf16(units)
        .collect::<Result<String, _>>()
        .ok()
}

/// Human-readable size for the `[context]` stderr lines.
fn format_size(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    let b = bytes as f64;
    if bytes < 1024 {
        format!("{bytes} B")
    } else if b < KB * KB {
        format!("{:.1} KB", b / KB)
    } else {
        format!("{:.1} MB", b / (KB * KB))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A per-test scratch directory, removed on drop so test runs don't
    /// leave `ai-cli-ctx-*` directories behind in the temp dir.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            // Tests run in parallel threads of one process, so the pid
            // alone isn't unique enough.
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!("ai-cli-ctx-{}-{}", std::process::id(), n));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }

        fn file(&self, name: &str, body: impl AsRef<[u8]>) -> PathBuf {
            let path = self.0.join(name);
            std::fs::write(&path, body).unwrap();
            path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Run expansion with an injected home, collecting stderr lines.
    fn expand(q: &str, home: Option<&Path>) -> (Result<String>, Vec<String>) {
        let mut lines = Vec::new();
        let out = expand_refs(q, home, &mut |line| lines.push(line));
        (out, lines)
    }

    fn expand_ok(q: &str) -> String {
        expand(q, None).0.unwrap()
    }

    fn expand_err(q: &str) -> String {
        format!("{:#}", expand(q, None).0.unwrap_err())
    }

    /// Build a question from argv words, collecting stderr lines.
    fn question(
        args: &[String],
        expand: bool,
        home: Option<&Path>,
    ) -> (Result<String>, Vec<String>) {
        let mut lines = Vec::new();
        let out = question_from(args, expand, home, &mut |line| lines.push(line));
        (out, lines)
    }

    fn utf16(s: &str, le: bool) -> Vec<u8> {
        let mut out = if le {
            vec![0xFF, 0xFE]
        } else {
            vec![0xFE, 0xFF]
        };
        for u in s.encode_utf16() {
            out.extend(if le { u.to_le_bytes() } else { u.to_be_bytes() });
        }
        out
    }

    #[test]
    fn temp_dir_is_removed_on_drop() {
        let dir = TempDir::new();
        dir.file("x.txt", "x");
        let path = dir.path().to_path_buf();
        drop(dir);
        assert!(!path.exists());
    }

    #[test]
    fn expand_inlines_file_contents() {
        let dir = TempDir::new();
        let path = dir.file("hello.txt", "world\n");
        let out = expand_ok(&format!("read @{}", path.display()));
        assert!(out.contains("```txt "), "missing fence: {}", out);
        assert!(out.contains("world"), "missing body: {}", out);
    }

    #[test]
    fn expand_reports_each_inlined_file_with_its_size() {
        let dir = TempDir::new();
        let path = dir.file("hello.txt", "world\n");
        let (out, lines) = expand(&format!("read @{}", path.display()), None);
        out.unwrap();
        assert_eq!(
            lines,
            vec![format!("[context] inlined {} (6 B)", path.display())]
        );
    }

    #[test]
    fn expand_preserves_trailing_punctuation() {
        let dir = TempDir::new();
        let path = dir.file("note.md", "hi");
        let out = expand_ok(&format!("see @{}.", path.display()));
        assert!(out.trim_end().ends_with('.'), "trailing dot lost: {}", out);
    }

    #[test]
    fn expand_strips_wrapping_punctuation() {
        let dir = TempDir::new();
        let path = dir.file("note.md", "wrapped body");
        for (q, tail) in [
            (format!("(see @{})", path.display()), ")"),
            (format!("(@{})", path.display()), ")"),
            (format!("\"@{}\"", path.display()), "\""),
            (format!("'@{}'", path.display()), "'"),
            (format!("[@{}],", path.display()), "],"),
        ] {
            let out = expand_ok(&q);
            assert!(out.contains("wrapped body"), "not expanded: {q} -> {out}");
            assert!(out.ends_with(tail), "lost {tail:?}: {q} -> {out}");
        }
    }

    #[test]
    fn expand_leaves_missing_paths_literal() {
        let out = expand_ok("look at @nonexistent-file-xyz.txt please");
        assert!(out.contains("@nonexistent-file-xyz.txt"));
    }

    #[test]
    fn expand_warns_about_missing_path_like_tokens_only() {
        let (out, lines) = expand("look at @nonexistent-file-xyz.txt please", None);
        assert_eq!(out.unwrap(), "look at @nonexistent-file-xyz.txt please");
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert!(lines[0].contains("@nonexistent-file-xyz.txt: no such file"));

        // `@alice` reads as a mention, not a mistyped path.
        let (out, lines) = expand("ask @alice about it", None);
        assert_eq!(out.unwrap(), "ask @alice about it");
        assert!(lines.is_empty(), "{lines:?}");
    }

    #[test]
    fn expand_ignores_at_mid_word() {
        // Email addresses, e.g. `foo@bar.com`, should never be treated as
        // a file reference because `@` is not at a word boundary.
        let (out, lines) = expand("mail me at foo@bar.com", None);
        assert_eq!(out.unwrap(), "mail me at foo@bar.com");
        assert!(lines.is_empty(), "{lines:?}");
    }

    #[test]
    fn expand_handles_non_ascii_paths_with_trailing_punctuation() {
        // Regression: `rfind` returns a byte index, so splitting after a
        // multi-byte codepoint used to panic on "not a char boundary".
        let dir = TempDir::new();
        let path = dir.file("résumé.md", "hi");
        let out = expand_ok(&format!("see @{}.", path.display()));
        assert!(out.contains("hi"), "missing body: {}", out);
        assert!(out.trim_end().ends_with('.'), "trailing dot lost: {}", out);
    }

    #[test]
    fn expand_fence_outgrows_backticks_in_the_file() {
        let dir = TempDir::new();
        let readme = "# Title\n\n```bash\nai \"hi\"\n```\n\nMore prose.\n";
        let path = dir.file("README.md", readme);
        let out = expand_ok(&format!("summarize @{}", path.display()));
        let open = format!("````md {}\n", path.display());
        assert!(out.contains(&open), "fence not widened: {out}");
        // The whole file sits between the widened fences.
        assert!(
            out.ends_with("More prose.\n````\n"),
            "content escaped: {out}"
        );
    }

    #[test]
    fn fence_is_longer_than_any_backtick_run() {
        assert_eq!(fence_for("plain"), "```");
        assert_eq!(fence_for("`inline` code"), "```");
        assert_eq!(fence_for("a\n```\nb"), "````");
        assert_eq!(fence_for("x ````` y ``` z"), "``````");
        assert_eq!(fence_for(""), "```");
    }

    #[test]
    fn backslash_escapes_an_at_reference() {
        let dir = TempDir::new();
        let path = dir.file("secret.txt", "do not send");
        let (out, lines) = expand(&format!("literally \\@{} please", path.display()), None);
        assert_eq!(
            out.unwrap(),
            format!("literally @{} please", path.display())
        );
        assert!(lines.is_empty(), "{lines:?}");
    }

    #[test]
    fn backslash_mid_word_is_left_alone() {
        // Only the boundary form is an escape; `\@` inside a word (LaTeX,
        // Windows-ish paths) is not rewritten.
        assert_eq!(expand_ok("etc.\\@ and foo\\@bar"), "etc.\\@ and foo\\@bar");
    }

    #[test]
    fn quoted_paths_may_contain_spaces() {
        let dir = TempDir::new();
        let path = dir.file("my notes.md", "spaced out");
        let out = expand_ok(&format!("read @\"{}\".", path.display()));
        assert!(out.contains("spaced out"), "not expanded: {out}");
        assert!(
            out.contains(&format!("```md {}\n", path.display())),
            "{out}"
        );
        assert!(out.ends_with('.'), "trailing dot lost: {out}");
    }

    #[test]
    fn shell_unquoted_spaced_path_args_are_requoted() {
        // `ai read @"<dir>/my notes.md"` reaches us as
        // ["read", "@<dir>/my notes.md"].
        let dir = TempDir::new();
        let path = dir.file("my notes.md", "spaced out");
        let args = vec!["read".to_string(), format!("@{}", path.display())];
        assert_eq!(
            join_args(&args, None),
            format!("read @\"{}\"", path.display())
        );
        let (out, _) = question(&args, true, None);
        assert!(out.unwrap().contains("spaced out"), "not expanded");
        // Not a file as a whole (e.g. a quoted sentence): left alone.
        assert_eq!(join_args(&["@a b".to_string()], None), "@a b");
        // No spaces: nothing to protect.
        assert_eq!(join_args(&["@x.md".to_string()], None), "@x.md");
    }

    #[test]
    fn shell_unquoted_spaced_tilde_path_resolves_against_home() {
        // `ai read @"~/my notes.md"` reaches us as ["read", "@~/my notes.md"].
        // The file check has to expand `~/` like the expander does, or the
        // argument is joined bare and expansion stops at `~/my`.
        let home = TempDir::new();
        home.file("my notes.md", "from home");
        let args = vec!["read".to_string(), "@~/my notes.md".to_string()];
        assert_eq!(
            join_args(&args, Some(home.path())),
            "read @\"~/my notes.md\""
        );

        let (out, lines) = question(&args, true, Some(home.path()));
        let out = out.unwrap();
        assert!(out.contains("from home"), "not expanded: {out}");
        assert!(out.contains("```md ~/my notes.md\n"), "{out}");
        assert_eq!(lines, vec!["[context] inlined ~/my notes.md (9 B)"]);

        // Without a home directory there's nothing to resolve against.
        assert_eq!(join_args(&args, None), "read @~/my notes.md");
    }

    #[test]
    fn no_expand_joins_args_verbatim() {
        // `--no-expand` sends the words exactly as the shell passed them:
        // no quotes inserted around an existing spaced path, nothing read.
        let home = TempDir::new();
        let path = home.file("my notes.md", "do not send");
        for arg in [format!("@{}", path.display()), "@~/my notes.md".into()] {
            let args = vec!["read".to_string(), arg.clone()];
            let (out, lines) = question(&args, false, Some(home.path()));
            assert_eq!(out.unwrap(), format!("read {arg}"));
            assert!(lines.is_empty(), "{lines:?}");
        }
    }

    #[test]
    fn unclosed_quote_is_literal() {
        let (out, lines) = expand("say @\"hello there", None);
        assert_eq!(out.unwrap(), "say @\"hello there");
        assert!(lines.is_empty(), "{lines:?}");
    }

    #[test]
    fn missing_quoted_path_warns() {
        let (out, lines) = expand("read @\"no such notes\"", None);
        assert_eq!(out.unwrap(), "read @\"no such notes\"");
        assert_eq!(lines.len(), 1, "{lines:?}");
    }

    #[test]
    fn tilde_expands_against_home() {
        let home = TempDir::new();
        home.file("note.md", "from home");
        let (out, _) = expand("see @~/note.md", Some(home.path()));
        let out = out.unwrap();
        assert!(out.contains("from home"), "not expanded: {out}");
        assert!(out.contains("```md ~/note.md\n"), "{out}");
    }

    #[test]
    fn tilde_without_home_is_literal() {
        let (out, lines) = expand("see @~/note.md", None);
        assert_eq!(out.unwrap(), "see @~/note.md");
        assert_eq!(lines.len(), 1, "{lines:?}");
    }

    #[test]
    fn directories_are_refused() {
        let dir = TempDir::new();
        let err = expand_err(&format!("look at @{}", dir.path().display()));
        assert!(err.contains("is a directory"), "{err}");
        let err = expand_err(&format!("look at @{}/", dir.path().display()));
        assert!(err.contains("is a directory"), "{err}");
    }

    #[test]
    fn oversized_files_are_refused() {
        let dir = TempDir::new();
        let path = dir.file("big.log", "");
        // Sparse, so this doesn't actually write 20 MB.
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(MAX_INLINE_BYTES as u64 + 1)
            .unwrap();
        let err = expand_err(&format!("@{}", path.display()));
        assert!(err.contains("larger than 20 MB"), "{err}");
    }

    #[test]
    fn utf8_bom_is_stripped() {
        let dir = TempDir::new();
        let path = dir.file("bom.txt", b"\xEF\xBB\xBFhello");
        let out = expand_ok(&format!("@{}", path.display()));
        assert!(out.contains("\nhello\n"), "{out:?}");
        assert!(!out.contains('\u{feff}'), "BOM leaked: {out:?}");
    }

    #[test]
    fn utf16_with_bom_is_decoded() {
        let dir = TempDir::new();
        for (name, le) in [("le.txt", true), ("be.txt", false)] {
            let path = dir.file(name, utf16("héllo wörld\r\n", le));
            let out = expand_ok(&format!("@{}", path.display()));
            assert!(out.contains("\nhéllo wörld\n"), "{name}: {out:?}");
        }
    }

    #[test]
    fn non_utf8_files_are_refused_with_a_hint() {
        let dir = TempDir::new();
        let latin1 = dir.file("latin1.txt", b"caf\xe9\n");
        let err = expand_err(&format!("@{}", latin1.display()));
        assert!(err.contains("looks binary or non-UTF-8"), "{err}");

        let binary = dir.file("blob.bin", b"\x7fELF\x00\x00\x01");
        let err = expand_err(&format!("@{}", binary.display()));
        assert!(err.contains("looks binary or non-UTF-8"), "{err}");
    }

    #[test]
    fn decode_text_handles_boms_and_rejects_bad_input() {
        assert_eq!(decode_text(b"plain".to_vec()).unwrap(), "plain");
        assert_eq!(decode_text(b"\xEF\xBB\xBFbom".to_vec()).unwrap(), "bom");
        assert_eq!(decode_text(utf16("ÿ16", true)).unwrap(), "ÿ16");
        assert_eq!(decode_text(utf16("ÿ16", false)).unwrap(), "ÿ16");
        // Odd-length UTF-16 and a lone surrogate are not text.
        assert!(decode_text(vec![0xFF, 0xFE, b'a']).is_err());
        assert!(decode_text(vec![0xFF, 0xFE, 0x00, 0xD8]).is_err());
        assert_eq!(decode_text(b"caf\xe9".to_vec()).unwrap_err(), b"caf\xe9");
    }

    #[test]
    fn only_pipes_and_files_are_auto_read() {
        use StdinKind::*;
        for (kind, read) in [
            (Terminal, false),
            (Pipe, true),
            (File, true),
            (Socket, false),
            (CharDevice, false),
            (Other, false),
        ] {
            assert_eq!(should_auto_read(kind), read, "{kind:?}");
        }
    }

    fn stdin(bytes: &[u8]) -> (Result<Option<String>>, Vec<String>) {
        let mut lines = Vec::new();
        let out = stdin_text(bytes.to_vec(), &mut |line| lines.push(line));
        (out, lines)
    }

    #[test]
    fn stdin_is_reported_when_appended() {
        let (out, lines) = stdin(b"log line\n");
        assert_eq!(out.unwrap().as_deref(), Some("log line\n"));
        assert_eq!(lines, vec!["[context] appended piped stdin (9 B)"]);
    }

    #[test]
    fn blank_stdin_is_ignored() {
        let (out, lines) = stdin(b"  \n\n");
        assert_eq!(out.unwrap(), None);
        assert!(lines.is_empty(), "{lines:?}");
    }

    #[test]
    fn non_utf8_stdin_is_kept_lossily_with_a_warning() {
        let (out, lines) = stdin(b"build failed: caf\xe9\n");
        assert_eq!(out.unwrap().as_deref(), Some("build failed: caf\u{FFFD}\n"));
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert!(lines[0].contains("not valid UTF-8"), "{lines:?}");
    }

    #[test]
    fn oversized_stdin_is_an_error() {
        let (out, _) = stdin(&vec![b'a'; MAX_INLINE_BYTES + 1]);
        assert!(format!("{:#}", out.unwrap_err()).contains("exceeds 20 MB"));
    }

    #[test]
    fn oversized_stdin_cut_mid_character_reports_size_not_encoding() {
        // The capped read keeps MAX + 1 bytes; with 2-byte characters that
        // cut lands mid-character.
        let mut bytes = "é".repeat(MAX_INLINE_BYTES / 2 + 1).into_bytes();
        bytes.truncate(MAX_INLINE_BYTES + 1);
        assert!(std::str::from_utf8(&bytes).is_err());
        let (out, _) = stdin(&bytes);
        assert!(format!("{:#}", out.unwrap_err()).contains("exceeds 20 MB"));
    }

    #[test]
    fn merge_stdin_wraps_piped_content_in_context_block() {
        let merged = merge_stdin("summarize this", Some("file contents".into()));
        assert!(merged.contains("summarize this"));
        assert!(merged.contains("```text"));
        assert!(merged.contains("file contents"));
    }

    #[test]
    fn merge_stdin_fence_outgrows_backticks_in_content() {
        let piped = "see:\n```\ncode\n```\nafter\n".to_string();
        let merged = merge_stdin("explain", Some(piped));
        assert_eq!(
            merged,
            "explain\n\n````text\nsee:\n```\ncode\n```\nafter\n````"
        );
    }

    #[test]
    fn merge_stdin_uses_stdin_when_question_is_empty() {
        let merged = merge_stdin("", Some("stuff".into()));
        assert_eq!(merged, "stuff");
    }

    #[test]
    fn sizes_are_human_readable() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(1023), "1023 B");
        assert_eq!(format_size(12_595), "12.3 KB");
        assert_eq!(format_size(3 * 1024 * 1024 / 2), "1.5 MB");
    }
}
