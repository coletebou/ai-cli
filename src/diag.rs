//! Diagnostics that must not scribble over a full-screen UI.
//!
//! Retry notices and auth warnings fire from deep inside a request. While
//! the TUI owns the terminal (raw mode + alternate screen), writing them to
//! stderr garbles the display, so they are held back and printed once the
//! TUI has handed the terminal back.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

static DEFERRED: AtomicBool = AtomicBool::new(false);
static PENDING: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Upper bound on held-back lines, so a pathological retry storm during a
/// long TUI session can't grow memory without limit.
const MAX_PENDING: usize = 500;

/// Print a diagnostic line to stderr, or hold it until [`flush`] while a
/// full-screen UI is active.
pub fn warn(msg: impl Into<String>) {
    let msg = msg.into();
    // Check the flag under the lock so a concurrent `flush` can't drain the
    // queue between our check and our push (which would strand the line).
    let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    if DEFERRED.load(Ordering::SeqCst) {
        if pending.len() < MAX_PENDING {
            pending.push(msg);
        }
        return;
    }
    drop(pending);
    eprintln!("{}", msg);
}

/// Start holding diagnostics back. Call before taking over the terminal.
pub fn defer() {
    let _pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    DEFERRED.store(true, Ordering::SeqCst);
}

/// Stop deferring and print everything that was held back. Call after the
/// terminal has been restored.
pub fn flush() {
    let held = {
        let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
        DEFERRED.store(false, Ordering::SeqCst);
        std::mem::take(&mut *pending)
    };
    for line in held {
        eprintln!("{}", line);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deferred_lines_are_held_until_flush() {
        defer();
        warn("held back");
        assert_eq!(
            PENDING.lock().unwrap().last().map(String::as_str),
            Some("held back")
        );
        flush();
        assert!(PENDING.lock().unwrap().is_empty());
        assert!(!DEFERRED.load(Ordering::SeqCst));
    }
}
