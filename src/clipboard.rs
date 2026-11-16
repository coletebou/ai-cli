//! Copy text to the system clipboard from inside the TUI.
//!
//! Two routes, both best effort:
//!
//! - **OSC 52**, an escape sequence the terminal turns into a clipboard
//!   write. It works over SSH and through tmux (with `set-clipboard on`; the
//!   sequence is also sent wrapped for tmux's passthrough in case only
//!   `allow-passthrough` is on). Ghostty, iTerm2, kitty, WezTerm, Alacritty,
//!   foot and Windows Terminal support it; macOS Terminal.app does not.
//! - **The local clipboard tool** (`pbcopy`, `wl-copy`, `xclip`, `xsel`),
//!   which covers terminals without OSC 52 (or with clipboard writes turned
//!   off). It runs even when `SSH_*` variables are set: inside tmux those are
//!   whatever the tmux server was started with, so they can't tell a remote
//!   session from a local one, and on a real remote session the tool only
//!   sets that machine's clipboard, which is harmless.
//!
//! Neither route can confirm the terminal accepted the text.

use base64::Engine;
use std::io::{self, Write};
use std::process::{Command, Stdio};

/// Copy `text` by every route available. Errors are ignored: a missing
/// clipboard tool or a terminal that drops OSC 52 just means that route did
/// nothing.
pub fn copy(text: &str) {
    let in_tmux = std::env::var_os("TMUX").is_some_and(|v| !v.is_empty());
    let mut out = io::stdout();
    let _ = out.write_all(osc52(text, in_tmux).as_bytes());
    let _ = out.flush();
    copy_native(text);
}

/// The OSC 52 "set clipboard" sequence for `text`, plus a tmux passthrough
/// copy of it when running inside tmux.
pub fn osc52(text: &str, in_tmux: bool) -> String {
    let encoded = base64::engine::general_purpose::STANDARD.encode(text);
    let plain = format!("\x1b]52;c;{encoded}\x07");
    if in_tmux {
        // DCS passthrough: tmux hands the inner sequence (with ESC doubled)
        // to the outer terminal when `allow-passthrough` is on.
        let inner = plain.replace('\x1b', "\x1b\x1b");
        format!("{plain}\x1bPtmux;{inner}\x1b\\")
    } else {
        plain
    }
}

/// Local clipboard commands to try, in order; the first that starts wins.
fn native_commands() -> Vec<(&'static str, &'static [&'static str])> {
    if cfg!(target_os = "macos") {
        return vec![("pbcopy", &[])];
    }
    if cfg!(windows) {
        // Windows Terminal handles OSC 52; clip.exe garbles non-ASCII text.
        return Vec::new();
    }
    let mut cmds: Vec<(&'static str, &'static [&'static str])> = Vec::new();
    if std::env::var_os("WAYLAND_DISPLAY").is_some() {
        cmds.push(("wl-copy", &[]));
    }
    if std::env::var_os("DISPLAY").is_some() {
        cmds.push(("xclip", &["-selection", "clipboard"]));
        cmds.push(("xsel", &["--clipboard", "--input"]));
    }
    cmds
}

fn copy_native(text: &str) {
    for (program, args) in native_commands() {
        // Never let the tool write over the TUI: stdout and stderr go nowhere.
        let child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
        let Ok(mut child) = child else {
            continue; // not installed: try the next one
        };
        let text = text.to_string();
        // Write and reap off the UI thread so a slow tool can't freeze input.
        std::thread::spawn(move || {
            if let Some(mut stdin) = child.stdin.take() {
                let _ = stdin.write_all(text.as_bytes());
            }
            let _ = child.wait();
        });
        return;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn osc52_encodes_the_text() {
        assert_eq!(osc52("hi", false), "\x1b]52;c;aGk=\x07");
    }

    #[test]
    fn osc52_in_tmux_adds_a_passthrough_copy() {
        let s = osc52("hi", true);
        assert!(s.starts_with("\x1b]52;c;aGk=\x07"));
        assert!(s.ends_with("\x1bPtmux;\x1b\x1b]52;c;aGk=\x07\x1b\\"));
    }

    #[test]
    fn osc52_handles_unicode() {
        let s = osc52("漢字 ✓", false);
        let b64 = s.trim_start_matches("\x1b]52;c;").trim_end_matches('\x07');
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .unwrap();
        assert_eq!(String::from_utf8(decoded).unwrap(), "漢字 ✓");
    }
}
