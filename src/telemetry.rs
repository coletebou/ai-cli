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

use base64::Engine;
use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::{Protocol, SpanExporter, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::{runtime, trace::TracerProvider, Resource};
use std::collections::HashMap;
use std::env;
use std::sync::OnceLock;
use std::time::Duration;

/// Name used for the global tracer and as `service.name` in Langfuse.
pub const TRACER_NAME: &str = "ai-cli";

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
/// endpoint to prevent the CLI from running.
pub fn init() -> Option<InitInfo> {
    let host = non_empty_env("LANGFUSE_HOST")?;
    let public = non_empty_env("LANGFUSE_PUBLIC_KEY")?;
    let secret = non_empty_env("LANGFUSE_SECRET_KEY")?;

    // Explicit kill switch wins over key presence.
    if env::var("LANGFUSE_ENABLED")
        .map(|v| v.eq_ignore_ascii_case("false") || v == "0")
        .unwrap_or(false)
    {
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

    let exporter = SpanExporter::builder()
        .with_http()
        .with_endpoint(&endpoint)
        .with_protocol(Protocol::HttpBinary)
        .with_headers(headers)
        .with_timeout(Duration::from_secs(5))
        .build()
        .ok()?;

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
            // diagnostic line on the way out.
            eprintln!("[telemetry] flush error: {:?}", e);
        }
    }
    let _ = provider.shutdown();
}

fn non_empty_env(name: &str) -> Option<String> {
    env::var(name).ok().filter(|v| !v.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `non_empty_env` should treat unset and empty-string the same.
    /// We test it directly because env mutation in tests is racy.
    #[test]
    fn non_empty_env_treats_empty_as_missing() {
        // Use a name unlikely to be set in any reasonable environment.
        let name = "AI_CLI_TEST_NEVER_SET_VAR_X";
        env::remove_var(name);
        assert!(non_empty_env(name).is_none());
        env::set_var(name, "");
        assert!(non_empty_env(name).is_none());
        env::set_var(name, "x");
        assert_eq!(non_empty_env(name).as_deref(), Some("x"));
        env::remove_var(name);
    }
}
