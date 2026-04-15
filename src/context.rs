//! Expands context into user questions: inline `@file` references and
//! piped stdin content. Used to bridge between CLI args and what the
//! model actually sees.

use anyhow::{Context, Result};
use std::io::{self, IsTerminal, Read};
use std::path::Path;

/// Maximum bytes inlined from stdin or an `@file`. Kept in sync so the two
/// input modes share the same ceiling — otherwise a caller could bypass the
/// file-size guard by piping a giant log instead of referencing it as `@`.
const MAX_INLINE_BYTES: usize = 20 * 1024 * 1024;

/// Read piped stdin into a string. Returns `None` if stdin is a terminal
/// (nothing piped), the pipe is empty, or the read fails (e.g. non-UTF-8
/// bytes from a binary file — we warn and skip rather than panic).
pub fn read_stdin_if_piped() -> Option<String> {
    if io::stdin().is_terminal() {
        return None;
    }
    // Cap the read itself so a `cat huge.log | ai ...` doesn't exhaust memory
    // before we even get a chance to check the size.
    let mut buf = String::new();
    let mut reader = io::stdin().take(MAX_INLINE_BYTES as u64 + 1);
    if let Err(e) = reader.read_to_string(&mut buf) {
        eprintln!("[warn] could not read stdin: {e}");
        return None;
    }
    if buf.len() > MAX_INLINE_BYTES {
        eprintln!("[warn] piped stdin exceeds 20 MB; refusing to inline it");
        return None;
    }
    if buf.trim().is_empty() {
        None
    } else {
        Some(buf)
    }
}

/// Compose the final question string from a user-provided question and
/// optional piped stdin. Stdin is appended as a fenced code block so it
/// reads consistently with `@file` expansion and so piped content that
/// contains `---` lines can't accidentally terminate the block.
pub fn merge_stdin(question: &str, stdin: Option<String>) -> String {
    match stdin {
        None => question.to_string(),
        Some(content) if question.trim().is_empty() => content,
        Some(content) => format!("{}\n\n```text\n{}\n```", question, content.trim_end()),
    }
}

/// Expand `@path` tokens in `question` by inlining file contents as
/// fenced code blocks. A token only expands when `@` appears at the
/// start of the string or directly after whitespace — so email
/// addresses and `foo@bar` literals pass through untouched. Trailing
/// punctuation (`.,;:?!)`) is preserved in the output.
///
/// Unreadable files surface an error; non-existent paths are left as
/// literal text so the user notices a typo instead of getting silent
/// misbehavior.
pub fn expand_file_refs(question: &str) -> Result<String> {
    let mut out = String::with_capacity(question.len());
    let mut chars = question.chars().peekable();
    let mut at_boundary = true;

    while let Some(c) = chars.next() {
        if c == '@' && at_boundary {
            let mut token = String::new();
            while let Some(&nc) = chars.peek() {
                if nc.is_whitespace() {
                    break;
                }
                token.push(nc);
                chars.next();
            }

            // Split off trailing punctuation so `see @notes.md.` resolves
            // to `notes.md` and the period stays in the sentence.
            // `trim_end_matches` is UTF-8 safe, so multi-byte codepoints
            // in the path (e.g. `@/tmp/résumé.md.`) don't panic.
            let path_str = token.trim_end_matches(&['.', ',', ';', ':', '?', '!', ')'][..]);
            let trailing = &token[path_str.len()..];

            if !path_str.is_empty() && Path::new(path_str).is_file() {
                let metadata = std::fs::metadata(path_str)
                    .with_context(|| format!("failed to stat @{}", path_str))?;
                if metadata.len() > MAX_INLINE_BYTES as u64 {
                    anyhow::bail!(
                        "@{} is {:.1} MB; refusing to inline files larger than 20 MB",
                        path_str,
                        metadata.len() as f64 / 1_048_576.0
                    );
                }
                let content = std::fs::read_to_string(path_str)
                    .with_context(|| format!("failed to read @{}", path_str))?;
                let lang = Path::new(path_str)
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("");
                out.push_str(&format!(
                    "\n```{} {}\n{}\n```\n",
                    lang,
                    path_str,
                    content.trim_end_matches('\n')
                ));
                out.push_str(trailing);
            } else {
                out.push('@');
                out.push_str(&token);
            }
            at_boundary = false;
            continue;
        }
        at_boundary = c.is_whitespace();
        out.push(c);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmpfile(name: &str, body: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ai-cli-ctx-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        path
    }

    #[test]
    fn expand_inlines_file_contents() {
        let path = tmpfile("hello.txt", "world\n");
        let q = format!("read @{}", path.display());
        let out = expand_file_refs(&q).unwrap();
        assert!(out.contains("```txt "), "missing fence: {}", out);
        assert!(out.contains("world"), "missing body: {}", out);
    }

    #[test]
    fn expand_preserves_trailing_punctuation() {
        let path = tmpfile("note.md", "hi");
        let q = format!("see @{}.", path.display());
        let out = expand_file_refs(&q).unwrap();
        assert!(out.trim_end().ends_with('.'), "trailing dot lost: {}", out);
    }

    #[test]
    fn expand_leaves_missing_paths_literal() {
        let out = expand_file_refs("look at @nonexistent-file-xyz.txt please").unwrap();
        assert!(out.contains("@nonexistent-file-xyz.txt"));
    }

    #[test]
    fn expand_ignores_at_mid_word() {
        // Email addresses, e.g. `foo@bar.com`, should never be treated as
        // a file reference because `@` is not at a word boundary.
        let out = expand_file_refs("mail me at foo@bar.com").unwrap();
        assert_eq!(out, "mail me at foo@bar.com");
    }

    #[test]
    fn expand_handles_non_ascii_paths_with_trailing_punctuation() {
        // Regression: `rfind` returns a byte index, so splitting after a
        // multi-byte codepoint used to panic on "not a char boundary".
        let path = tmpfile("résumé.md", "hi");
        let q = format!("see @{}.", path.display());
        let out = expand_file_refs(&q).unwrap();
        assert!(out.contains("hi"), "missing body: {}", out);
        assert!(out.trim_end().ends_with('.'), "trailing dot lost: {}", out);
    }

    #[test]
    fn merge_stdin_wraps_piped_content_in_context_block() {
        let merged = merge_stdin("summarize this", Some("file contents".into()));
        assert!(merged.contains("summarize this"));
        assert!(merged.contains("```text"));
        assert!(merged.contains("file contents"));
    }

    #[test]
    fn merge_stdin_uses_stdin_when_question_is_empty() {
        let merged = merge_stdin("", Some("stuff".into()));
        assert_eq!(merged, "stuff");
    }
}
