//! Shared terminal setup/teardown for the full-screen UIs (`ai -t` and
//! `ai config`).
//!
//! Raw mode, the alternate screen, mouse capture and bracketed paste must be
//! undone on *every* exit path — `?` early returns and panics included — or
//! the user's shell is left unusable (no echo, no line editing, mouse escape
//! codes on every click). `TerminalGuard` restores on drop, and a panic hook
//! restores before the panic message prints so it's actually readable.

use crossterm::{
    cursor::Show,
    event::{DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, Once};
use std::thread::{self, ThreadId};

/// True while a guard holds the terminal.
static ACTIVE: AtomicBool = AtomicBool::new(false);
/// The thread that owns the UI. Only a panic on *this* thread tears the
/// terminal down; see the panic hook for why.
static OWNER: Mutex<Option<ThreadId>> = Mutex::new(None);

pub struct TerminalGuard {
    mouse: bool,
}

impl TerminalGuard {
    /// Take over the terminal: raw mode, alternate screen, bracketed paste
    /// and (optionally) mouse capture.
    pub fn enter(mouse: bool) -> io::Result<Self> {
        install_panic_hook();
        enable_raw_mode()?;
        // From here on, Drop undoes whatever we managed to set up — even if
        // one of the steps below fails.
        let guard = TerminalGuard { mouse };
        *OWNER.lock().unwrap_or_else(|e| e.into_inner()) = Some(thread::current().id());
        ACTIVE.store(true, Ordering::SeqCst);
        let mut out = io::stdout();
        execute!(out, EnterAlternateScreen, EnableBracketedPaste)?;
        if mouse {
            execute!(out, EnableMouseCapture)?;
        }
        Ok(guard)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        ACTIVE.store(false, Ordering::SeqCst);
        restore(self.mouse);
    }
}

/// Best-effort restore. Every step runs even if an earlier one fails.
fn restore(mouse: bool) {
    let mut out = io::stdout();
    if mouse {
        let _ = execute!(out, DisableMouseCapture);
    }
    let _ = execute!(out, DisableBracketedPaste, LeaveAlternateScreen, Show);
    let _ = disable_raw_mode();
}

fn install_panic_hook() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !ACTIVE.load(Ordering::SeqCst) {
                previous(info);
                return;
            }
            let owner = *OWNER.lock().unwrap_or_else(|e| e.into_inner());
            if owner == Some(thread::current().id()) {
                // The UI itself is going down: hand the terminal back first
                // so the panic message lands in a sane shell.
                ACTIVE.store(false, Ordering::SeqCst);
                restore(true);
                previous(info);
            } else {
                // A background task (e.g. a panel's request) panicked while
                // the UI keeps running. Printing now would scribble over the
                // screen, so defer the report until the terminal is restored.
                crate::diag::warn(format!("[panic] {}", info));
            }
        }));
    });
}
