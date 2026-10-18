//! OpenTelemetry → Langfuse integration.
//!
//! Initialization is opt-in based on env vars: when LANGFUSE_HOST,
//! LANGFUSE_PUBLIC_KEY, and LANGFUSE_SECRET_KEY are all set (and
//! LANGFUSE_ENABLED is not "false"), spans created via
//! `opentelemetry::global::tracer(TRACER_NAME)` are batched and exported
//! over OTLP/HTTP to `<host>/api/public/otel/v1/traces` with HTTP basic
//! auth derived from the project keys.
//!
//! Spans follow the GenAI semantic conventions
//! (https://opentelemetry.io/docs/specs/semconv/gen-ai/) so Langfuse renders
//! them as native generations rather than opaque spans.
//!
//! Prompt and answer text is attached to spans only through [`content_attr`],
//! which caps each value at [`MAX_CONTENT_BYTES`] and returns `None` when the
//! user opted out with `LANGFUSE_CAPTURE_CONTENT=false`.

use base64::Engine;
use opentelemetry::trace::TraceContextExt;
use opentelemetry::{global, Context, KeyValue};
use opentelemetry_otlp::{Protocol, SpanExporter, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::{runtime, trace::TracerProvider, Resource};
use std::collections::HashMap;
use std::env;
use std::sync::OnceLock;
use std::time::Duration;

/// Name used for the global tracer and as `service.name` in Langfuse.
pub const TRACER_NAME: &str = "ai-cli";

/// Upper bound (in bytes) on any single content attribute (prompt, message
/// history, answer). Inputs can be huge — inlined `@file`s and up to 20 MB of
/// piped stdin — and one oversized attribute can make the collector reject
/// the whole OTLP batch, losing every span in it. 32 KiB keeps a useful
/// preview in Langfuse while staying far below typical request limits.
pub const MAX_CONTENT_BYTES: usize = 32 * 1024;

/// Cached result of reading `LANGFUSE_CAPTURE_CONTENT` (read once per process).
static CAPTURE_CONTENT: OnceLock<bool> = OnceLock::new();

/// Holds the provider so its background batch processor stays alive for the
/// duration of the process. Without this, dropping the provider would stop
/// flushing spans before the CLI exits.
static PROVIDER: OnceLock<TracerProvider> = OnceLock::new();

/// Snapshot of the resolved Langfuse config returned from `init()`. Currently
/// just a marker that telemetry was wired up; we keep it as a struct (rather
/// than `bool`) so we can add diagnostic fields (host URL, sampling rate, etc.)
/// later without changing the public API shape.
#[derive(Debug, Clone)]
pub struct InitInfo;

/// Initialize OpenTelemetry export to Langfuse.
///
/// Returns `Some(InitInfo)` if exporter was wired up; `None` if any required
/// env var is missing or `LANGFUSE_ENABLED=false`. Errors during exporter
/// construction return `None` rather than panicking — telemetry is a
/// nice-to-have, not load-bearing, and we don't want a misconfigured
/// endpoint to prevent the CLI from running. They are reported on stderr,
/// though: the user configured Langfuse, so silently exporting nothing
/// would be confusing.
pub fn init() -> Option<InitInfo> {
    let host = non_empty_env("LANGFUSE_HOST")?;
    let public = non_empty_env("LANGFUSE_PUBLIC_KEY")?;
    let secret = non_empty_env("LANGFUSE_SECRET_KEY")?;

    // Explicit kill switch wins over key presence.
    if is_switched_off(env::var("LANGFUSE_ENABLED").ok().as_deref()) {
        return None;
    }

    let endpoint = format!("{}/api/public/otel/v1/traces", host.trim_end_matches('/'));
    let credentials =
        base64::engine::general_purpose::STANDARD.encode(format!("{}:{}", public, secret));

    let mut headers = HashMap::new();
    headers.insert(
        "Authorization".to_string(),
        format!("Basic {}", credentials),
    );

    let exporter = match SpanExporter::builder()
        .with_http()
        .with_endpoint(&endpoint)
        .with_protocol(Protocol::HttpBinary)
        .with_headers(headers)
        .with_timeout(Duration::from_secs(5))
        .build()
    {
        Ok(exporter) => exporter,
        Err(e) => {
            eprintln!("[telemetry] disabled: {}", e);
            return None;
        }
    };

    let provider = TracerProvider::builder()
        .with_batch_exporter(exporter, runtime::Tokio)
        .with_resource(Resource::new(vec![
            KeyValue::new("service.name", TRACER_NAME),
            KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
        ]))
        .build();

    // Set the global default so `global::tracer(...)` everywhere returns one
    // backed by this provider; also stash it so the batch processor is kept
    // alive until shutdown().
    global::set_tracer_provider(provider.clone());
    let _ = PROVIDER.set(provider);

    let _ = host; // host is consumed via `endpoint` above
    Some(InitInfo)
}

/// Flush and shut down the tracer provider. Safe to call even if `init()`
/// returned `None`. Should be called before `main` returns so short-lived
/// CLI invocations don't lose the last batch of spans.
///
/// We call `force_flush` and `shutdown` on the stored `TracerProvider`
/// directly rather than `global::shutdown_tracer_provider()`. The global
/// version has a known interaction problem with `runtime::Tokio` when called
/// during runtime teardown — flush futures get dropped silently and short-
/// lived CLIs lose their final batch. Direct invocation on the provider
/// blocks synchronously until the exporter drains.
pub fn shutdown() {
    let Some(provider) = PROVIDER.get() else {
        return;
    };
    for r in provider.force_flush() {
        if let Err(e) = r {
            // Surface flush failures to stderr so a misconfigured endpoint or
            // dropped network is visible. Telemetry is opportunistic — we
            // don't propagate the error, but silent failure is worse than a
            // diagnostic line on the way out. One readable line: the Debug
            // form was a multi-line struct dump on every run while the
            // Langfuse host was down.
            eprintln!(
                "[telemetry] could not export traces to Langfuse: {}",
                error_chain(&e)
            );
        }
    }
    let _ = provider.shutdown();
}

/// `err` and its `source()` chain on one line, skipping causes whose text
/// the previous message already includes.
fn error_chain(err: &(dyn std::error::Error + 'static)) -> String {
    let mut parts = vec![err.to_string()];
    let mut cause = err.source();
    while let Some(e) = cause {
        let text = e.to_string();
        if !parts.last().is_some_and(|prev| prev.contains(&text)) {
            parts.push(text);
        }
        cause = e.source();
    }
    parts.join(": ")
}

/// Whether prompt/answer text may be attached to spans. On by default; set
/// `LANGFUSE_CAPTURE_CONTENT=false` (or `0`) to export only metadata (model,
/// token counts, timings, errors).
pub fn capture_content() -> bool {
    *CAPTURE_CONTENT
        .get_or_init(|| !is_switched_off(env::var("LANGFUSE_CAPTURE_CONTENT").ok().as_deref()))
}

/// Prepare prompt/answer text for use as a span attribute value. Returns
/// `None` when content capture is disabled, so callers skip the attribute
/// entirely; otherwise the text capped at [`MAX_CONTENT_BYTES`]. Every
/// attribute that carries user or model text must go through this.
pub fn content_attr(text: &str) -> Option<String> {
    content_attr_with(text, capture_content())
}

/// Pure core of [`content_attr`], with the capture decision passed in so it
/// can be unit tested without touching the process environment.
fn content_attr_with(text: &str, capture: bool) -> Option<String> {
    capture.then(|| truncate_content(text, MAX_CONTENT_BYTES))
}

/// Cut `text` to at most `max_bytes` bytes (backing off to the previous char
/// boundary so multi-byte characters are never split) and append a
/// `…[truncated N bytes]` marker saying how much was dropped. Text that
/// already fits is returned unchanged.
pub fn truncate_content(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let mut cut = max_bytes;
    while !text.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}…[truncated {} bytes]", &text[..cut], text.len() - cut)
}

/// Ends the span of the wrapped [`Context`] when dropped.
///
/// A span only ends on its own when the last `Context` holding it is
/// dropped, and the root `ai.query` context is cloned into detached tasks
/// (`query::stream_all`, the TUI panel workers). Those can outlive `run()` —
/// e.g. quitting the TUI mid-stream — so without an explicit end the root
/// span would close after `shutdown()` has flushed and shut the provider
/// down, and the SDK drops it: the trace loses its root. Holding this guard
/// in `run()` ends the root span on every return path, before `main` calls
/// `shutdown()`. Ending is idempotent, so the later drop of the clones is a
/// no-op.
pub struct EndSpanOnDrop(pub Context);

impl Drop for EndSpanOnDrop {
    fn drop(&mut self) {
        self.0.span().end();
    }
}

fn non_empty_env(name: &str) -> Option<String> {
    non_empty(env::var(name).ok())
}

/// Treat an empty value the same as an unset one.
fn non_empty(value: Option<String>) -> Option<String> {
    value.filter(|v| !v.is_empty())
}

/// Parse an on/off env switch where only an explicit `false` (any case) or
/// `0` turns the feature off; unset or anything else leaves it on.
fn is_switched_off(value: Option<&str>) -> bool {
    value
        .map(str::trim)
        .is_some_and(|v| v.eq_ignore_ascii_case("false") || v == "0")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_chain_is_one_line_without_repeats() {
        let io = std::io::Error::new(std::io::ErrorKind::ConnectionRefused, "Connection refused");
        #[derive(Debug)]
        struct Outer(std::io::Error);
        impl std::fmt::Display for Outer {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "error sending request")
            }
        }
        impl std::error::Error for Outer {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                Some(&self.0)
            }
        }
        let text = error_chain(&Outer(io));
        assert_eq!(text, "error sending request: Connection refused");
        assert!(!text.contains('\n'));
    }

    /// `non_empty` should treat unset and empty-string the same. It takes the
    /// value as a parameter so the test needs no `set_var`/`remove_var`:
    /// mutating the process environment while other test threads read it is
    /// a data race (UB on glibc).
    #[test]
    fn non_empty_treats_empty_as_missing() {
        assert!(non_empty(None).is_none());
        assert!(non_empty(Some(String::new())).is_none());
        assert_eq!(non_empty(Some("x".to_string())).as_deref(), Some("x"));
    }

    #[test]
    fn switch_is_off_only_for_false_or_zero() {
        for off in ["false", "FALSE", "False", "0", " false "] {
            assert!(is_switched_off(Some(off)), "{off:?} should switch off");
        }
        for on in [None, Some(""), Some("true"), Some("1"), Some("no")] {
            assert!(!is_switched_off(on), "{on:?} should leave it on");
        }
    }

    #[test]
    fn content_attr_is_none_when_capture_disabled() {
        assert_eq!(content_attr_with("secret prompt", false), None);
        assert_eq!(content_attr_with("", false), None);
    }

    #[test]
    fn content_attr_passes_short_text_through_when_enabled() {
        assert_eq!(content_attr_with("hello", true).as_deref(), Some("hello"));
    }

    #[test]
    fn content_attr_caps_large_text_when_enabled() {
        // Roughly the size of a large piped stdin: must come out capped.
        let big = "x".repeat(MAX_CONTENT_BYTES * 4);
        let out = content_attr_with(&big, true).expect("capture enabled");
        assert!(out.len() < MAX_CONTENT_BYTES + 64, "len {}", out.len());
        assert!(out.ends_with(&format!("…[truncated {} bytes]", MAX_CONTENT_BYTES * 3)));
    }

    #[test]
    fn truncate_leaves_text_at_or_under_limit_unchanged() {
        assert_eq!(truncate_content("", 4), "");
        assert_eq!(truncate_content("abcd", 4), "abcd");
        assert_eq!(truncate_content("abc", 4), "abc");
    }

    #[test]
    fn truncate_appends_marker_with_dropped_byte_count() {
        assert_eq!(truncate_content("abcdef", 4), "abcd…[truncated 2 bytes]");
    }

    #[test]
    fn truncate_backs_off_to_char_boundary() {
        // "é" is 2 bytes, "🎉" is 4. A byte cut inside either must back off
        // to the start of that character instead of panicking or splitting.
        assert_eq!(truncate_content("aé", 2), "a…[truncated 2 bytes]");
        assert_eq!(truncate_content("ab🎉cd", 4), "ab…[truncated 6 bytes]");
        // A limit that lands exactly on a boundary keeps the whole char.
        assert_eq!(truncate_content("ab🎉cd", 6), "ab🎉…[truncated 2 bytes]");
        // Degenerate limit smaller than the first char keeps nothing.
        assert_eq!(truncate_content("🎉", 1), "…[truncated 4 bytes]");
    }

    mod root_span {
        use super::*;
        use futures::future::BoxFuture;
        use opentelemetry::trace::{Tracer, TracerProvider as _};
        use opentelemetry_sdk::export::trace::{ExportResult, SpanData};
        use std::sync::{Arc, Mutex};

        /// Minimal in-memory exporter: records the names of exported spans.
        /// Paired with the simple (synchronous) span processor, a span shows
        /// up here the moment it ends.
        #[derive(Debug, Clone, Default)]
        struct Collect(Arc<Mutex<Vec<String>>>);

        impl opentelemetry_sdk::export::trace::SpanExporter for Collect {
            fn export(&mut self, batch: Vec<SpanData>) -> BoxFuture<'static, ExportResult> {
                let mut names = self.0.lock().unwrap();
                names.extend(batch.into_iter().map(|s| s.name.into_owned()));
                Box::pin(async { Ok(()) })
            }
        }

        impl Collect {
            fn names(&self) -> Vec<String> {
                self.0.lock().unwrap().clone()
            }
        }

        fn provider(exporter: &Collect) -> TracerProvider {
            TracerProvider::builder()
                .with_simple_exporter(exporter.clone())
                .build()
        }

        /// Documents the bug: while a detached task still holds a clone of
        /// the root Context, dropping `run()`'s copy does not end the span,
        /// so nothing is exported before `shutdown()` would run.
        #[test]
        fn root_span_outlives_run_without_guard() {
            let exporter = Collect::default();
            let provider = provider(&exporter);
            let tracer = provider.tracer("test");

            let cx = Context::current_with_span(tracer.start("ai.query"));
            let detached_task_cx = cx.clone();
            drop(cx); // run() returns
            assert!(exporter.names().is_empty());

            drop(detached_task_cx);
            assert_eq!(exporter.names(), vec!["ai.query"]);
        }

        #[test]
        fn guard_ends_root_span_even_while_clones_are_alive() {
            let exporter = Collect::default();
            let provider = provider(&exporter);
            let tracer = provider.tracer("test");

            let cx = Context::current_with_span(tracer.start("ai.query"));
            let detached_task_cx = cx.clone();
            {
                let _end = EndSpanOnDrop(cx);
            } // run() returns
            assert_eq!(exporter.names(), vec!["ai.query"]);

            // A later child span (e.g. a stream still in flight) remains
            // exportable, and dropping the last clone doesn't re-export the
            // root.
            let child = tracer.start_with_context("chat CLAUDE", &detached_task_cx);
            drop(child);
            drop(detached_task_cx);
            assert_eq!(exporter.names(), vec!["ai.query", "chat CLAUDE"]);
        }
    }
}
