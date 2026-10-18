//! OpenTelemetry → Langfuse integration.
//!
//! Export is opt-in and configured by env vars (see [`resolve`]):
//!
//! - `AI_CLI_LANGFUSE_HOST`, `AI_CLI_LANGFUSE_PUBLIC_KEY` and
//!   `AI_CLI_LANGFUSE_SECRET_KEY` name ai-cli's own Langfuse project. These
//!   are the recommended settings. If any one of them is set, ai-cli uses
//!   only these three and needs all of them: an incomplete set leaves
//!   telemetry off and prints one warning naming the missing variable(s).
//!   It never falls back to the generic keys in that case, because an
//!   ai-cli host paired with generic keys (or the other way round) would
//!   mix two projects.
//! - Otherwise the generic `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY` and
//!   `LANGFUSE_SECRET_KEY` are used when all three are set, for backward
//!   compatibility. Other Langfuse-instrumented apps read these same names,
//!   so in a shared shell they usually belong to another app's project, and
//!   ai-cli's prompts (inlined `@file`s included) would land there.
//! - `AI_CLI_LANGFUSE_ENABLED`, when set, overrides `LANGFUSE_ENABLED`; a
//!   `false` (or `0`) turns telemetry off whichever keys are set. That is
//!   the way to keep ai-cli out of another app's project without touching
//!   the generic variables that app relies on.
//!   `AI_CLI_LANGFUSE_CAPTURE_CONTENT` overrides `LANGFUSE_CAPTURE_CONTENT`
//!   the same way.
//!
//! When enabled, spans created via `opentelemetry::global::tracer(TRACER_NAME)`
//! are batched and exported over OTLP/HTTP to
//! `<host>/api/public/otel/v1/traces` with HTTP basic auth derived from the
//! project keys.
//!
//! Spans follow the GenAI semantic conventions
//! (https://opentelemetry.io/docs/specs/semconv/gen-ai/) so Langfuse renders
//! them as native generations rather than opaque spans.
//!
//! Prompt and answer text is attached to spans only through [`content_attr`],
//! which caps each value at [`MAX_CONTENT_BYTES`] and returns `None` when the
//! user opted out with `AI_CLI_LANGFUSE_CAPTURE_CONTENT=false` (or the
//! generic `LANGFUSE_CAPTURE_CONTENT=false`).

use base64::Engine;
use opentelemetry::trace::TraceContextExt;
use opentelemetry::{global, Context, KeyValue};
use opentelemetry_otlp::{Protocol, SpanExporter, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::{runtime, trace::TracerProvider, Resource};
use std::collections::HashMap;
use std::env;
use std::fmt;
use std::sync::OnceLock;
use std::time::Duration;

/// Name used for the global tracer and as `service.name` in Langfuse.
pub const TRACER_NAME: &str = "ai-cli";

/// ai-cli's own Langfuse credentials (host, public key, secret key). They
/// take precedence over [`GENERIC_KEYS`] and are never mixed with them.
const AI_CLI_KEYS: [&str; 3] = [
    "AI_CLI_LANGFUSE_HOST",
    "AI_CLI_LANGFUSE_PUBLIC_KEY",
    "AI_CLI_LANGFUSE_SECRET_KEY",
];

/// The generic Langfuse credentials, in the same order as [`AI_CLI_KEYS`].
/// Langfuse SDKs in other apps read these same names, so they are only a
/// backward-compatible fallback.
const GENERIC_KEYS: [&str; 3] = [
    "LANGFUSE_HOST",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
];

/// Kill switch: the ai-cli name wins over the generic one when set.
const ENABLED_SWITCH: (&str, &str) = ("AI_CLI_LANGFUSE_ENABLED", "LANGFUSE_ENABLED");

/// Metadata-only switch: the ai-cli name wins over the generic one when set.
const CAPTURE_CONTENT_SWITCH: (&str, &str) = (
    "AI_CLI_LANGFUSE_CAPTURE_CONTENT",
    "LANGFUSE_CAPTURE_CONTENT",
);

/// Upper bound (in bytes) on any single content attribute (prompt, message
/// history, answer). Inputs can be huge — inlined `@file`s and up to 20 MB of
/// piped stdin — and one oversized attribute can make the collector reject
/// the whole OTLP batch, losing every span in it. 32 KiB keeps a useful
/// preview in Langfuse while staying far below typical request limits.
pub const MAX_CONTENT_BYTES: usize = 32 * 1024;

/// Cached result of reading the content-capture switch (read once per process).
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
/// Returns `Some(InitInfo)` if exporter was wired up; `None` if no complete
/// set of credentials is configured or telemetry is switched off (see
/// [`resolve`] for the precedence rules). An incomplete `AI_CLI_LANGFUSE_*`
/// set is reported on stderr, naming the missing variables but never any
/// values. `main` calls this once per process, so that warning is printed
/// once. Errors during exporter construction return `None` rather than
/// panicking — telemetry is a nice-to-have, not load-bearing, and we don't
/// want a misconfigured endpoint to prevent the CLI from running. They are
/// reported on stderr, though: the user configured Langfuse, so silently
/// exporting nothing would be confusing.
pub fn init() -> Option<InitInfo> {
    let creds = match resolve(process_env) {
        Resolution::Export(creds) => creds,
        Resolution::Off => return None,
        Resolution::Incomplete(missing) => {
            eprintln!("{}", incomplete_warning(&missing));
            return None;
        }
    };

    let endpoint = format!(
        "{}/api/public/otel/v1/traces",
        creds.host.trim_end_matches('/')
    );
    let credentials = base64::engine::general_purpose::STANDARD
        .encode(format!("{}:{}", creds.public_key, creds.secret_key));

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

    Some(InitInfo)
}

/// Langfuse project credentials read from one consistent set of env vars.
#[derive(PartialEq, Eq)]
struct Credentials {
    host: String,
    public_key: String,
    secret_key: String,
}

impl fmt::Debug for Credentials {
    // The keys are secrets: never format them, not even in a failing test's
    // assertion message.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Credentials")
            .field("host", &self.host)
            .finish_non_exhaustive()
    }
}

/// What [`resolve`] decided from the environment.
#[derive(Debug, PartialEq, Eq)]
enum Resolution {
    /// Export traces with these credentials.
    Export(Credentials),
    /// Telemetry off, nothing worth reporting: no credentials configured, an
    /// incomplete generic set (quiet, as it always was), or switched off.
    Off,
    /// Telemetry off because only some of the `AI_CLI_LANGFUSE_*`
    /// credentials are set. Holds the names (never the values) of the
    /// missing ones, in [`AI_CLI_KEYS`] order.
    Incomplete(Vec<&'static str>),
}

/// Decide whether and where to export, reading env vars through `env` so the
/// precedence rules can be unit tested without mutating the process
/// environment (tests run in parallel). Rules, in order:
///
/// 1. `AI_CLI_LANGFUSE_ENABLED` if set, else `LANGFUSE_ENABLED`: `false` or
///    `0` turns telemetry off. This wins over any credentials and also
///    silences the incomplete-set warning, since the user asked for none.
/// 2. If any of [`AI_CLI_KEYS`] is set, use only those: all three set means
///    export, otherwise [`Resolution::Incomplete`]. Never fall back to the
///    generic keys here: mixing an ai-cli host with another project's keys
///    (or the reverse) would send traces to the wrong project.
/// 3. Otherwise use [`GENERIC_KEYS`] if all three are set.
///
/// An empty value counts as unset throughout.
fn resolve(env: impl Fn(&str) -> Option<String>) -> Resolution {
    if is_switched_off(switch_value(&env, ENABLED_SWITCH).as_deref()) {
        return Resolution::Off;
    }

    let get = |name: &str| non_empty(env(name));
    let ai_cli = AI_CLI_KEYS.iter().any(|&name| get(name).is_some());
    let names = if ai_cli { AI_CLI_KEYS } else { GENERIC_KEYS };
    match names.map(get) {
        [Some(host), Some(public_key), Some(secret_key)] => Resolution::Export(Credentials {
            host,
            public_key,
            secret_key,
        }),
        values if ai_cli => Resolution::Incomplete(
            names
                .into_iter()
                .zip(values)
                .filter(|(_, value)| value.is_none())
                .map(|(name, _)| name)
                .collect(),
        ),
        _ => Resolution::Off,
    }
}

/// How telemetry is configured in an environment, as `ai doctor` reports it.
/// Holds variable names and the host, never key values.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Setup {
    /// Switched off by this variable (`AI_CLI_LANGFUSE_ENABLED` or
    /// `LANGFUSE_ENABLED`).
    SwitchedOff(&'static str),
    /// Exporting to `host` with the credential set named by `vars`.
    Export {
        host: String,
        vars: [&'static str; 3],
    },
    /// Off: some of `vars` are set but `missing` aren't.
    Incomplete {
        vars: [&'static str; 3],
        missing: Vec<&'static str>,
    },
    /// No credentials configured at all.
    NotConfigured,
}

impl Setup {
    /// Whether the credentials come from the generic `LANGFUSE_*` names,
    /// which other apps on the same machine may share.
    pub(crate) fn uses_generic_vars(&self) -> bool {
        matches!(self, Setup::Export { vars, .. } | Setup::Incomplete { vars, .. } if *vars == GENERIC_KEYS)
    }
}

/// [`resolve`]'s decision, spelled out for `ai doctor` so the report can't
/// drift from what `init` actually does.
pub(crate) fn describe_setup(env: impl Fn(&str) -> Option<String>) -> Setup {
    let (ai_cli_switch, generic_switch) = ENABLED_SWITCH;
    if is_switched_off(switch_value(&env, ENABLED_SWITCH).as_deref()) {
        let var = if non_empty(env(ai_cli_switch)).is_some() {
            ai_cli_switch
        } else {
            generic_switch
        };
        return Setup::SwitchedOff(var);
    }
    let missing_of = |vars: [&'static str; 3]| {
        vars.into_iter()
            .filter(|&v| non_empty(env(v)).is_none())
            .collect::<Vec<_>>()
    };
    match resolve(&env) {
        Resolution::Export(credentials) => {
            let ai_cli = AI_CLI_KEYS.iter().any(|&v| non_empty(env(v)).is_some());
            Setup::Export {
                host: credentials.host,
                vars: if ai_cli { AI_CLI_KEYS } else { GENERIC_KEYS },
            }
        }
        Resolution::Incomplete(missing) => Setup::Incomplete {
            vars: AI_CLI_KEYS,
            missing,
        },
        Resolution::Off => {
            let missing = missing_of(GENERIC_KEYS);
            if missing.len() == GENERIC_KEYS.len() {
                Setup::NotConfigured
            } else {
                Setup::Incomplete {
                    vars: GENERIC_KEYS,
                    missing,
                }
            }
        }
    }
}

/// The one-line stderr warning for an incomplete `AI_CLI_LANGFUSE_*` set.
/// Takes only variable names, so no value can end up in it.
fn incomplete_warning(missing: &[&str]) -> String {
    let verb = if missing.len() == 1 { "is" } else { "are" };
    format!(
        "[telemetry] disabled: {} {} not set. ai-cli needs all of {}, and \
         doesn't fall back to LANGFUSE_* while any of them is set.",
        missing.join(" and "),
        verb,
        AI_CLI_KEYS.join(", "),
    )
}

/// Value of an on/off switch given as `(ai_cli_name, generic_name)`: the
/// ai-cli variable when set (non-empty), else the generic one.
fn switch_value(
    env: &impl Fn(&str) -> Option<String>,
    (ai_cli, generic): (&str, &str),
) -> Option<String> {
    non_empty(env(ai_cli)).or_else(|| env(generic))
}

/// The real process environment, in the shape [`resolve`] takes. A value
/// that isn't valid Unicode counts as unset.
fn process_env(name: &str) -> Option<String> {
    env::var(name).ok()
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
/// `AI_CLI_LANGFUSE_CAPTURE_CONTENT=false` (or `0`) to export only metadata
/// (model, token counts, timings, errors). The generic
/// `LANGFUSE_CAPTURE_CONTENT` applies when the ai-cli one is unset.
pub fn capture_content() -> bool {
    *CAPTURE_CONTENT.get_or_init(|| capture_content_from(process_env))
}

/// Pure core of [`capture_content`], reading env vars through `env`.
fn capture_content_from(env: impl Fn(&str) -> Option<String>) -> bool {
    !is_switched_off(switch_value(&env, CAPTURE_CONTENT_SWITCH).as_deref())
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

    mod env_precedence {
        use super::*;

        /// A fake environment for [`resolve`] / [`capture_content_from`]:
        /// no `set_var`, so these tests can run in parallel safely.
        fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
            let map: HashMap<String, String> = pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            move |name: &str| map.get(name).cloned()
        }

        static GENERIC: [(&str, &str); 3] = [
            ("LANGFUSE_HOST", "http://other-app.example"),
            ("LANGFUSE_PUBLIC_KEY", "pk-lf-other-app"),
            ("LANGFUSE_SECRET_KEY", "sk-lf-other-app"),
        ];

        static AI_CLI: [(&str, &str); 3] = [
            ("AI_CLI_LANGFUSE_HOST", "http://ai-cli.example"),
            ("AI_CLI_LANGFUSE_PUBLIC_KEY", "pk-lf-ai-cli"),
            ("AI_CLI_LANGFUSE_SECRET_KEY", "sk-lf-ai-cli"),
        ];

        /// Env var name/value pairs.
        type Vars = [(&'static str, &'static str)];

        fn with(base: &Vars, extra: &Vars) -> Vec<(&'static str, &'static str)> {
            base.iter().chain(extra).copied().collect()
        }

        fn creds(set: &[(&str, &str); 3]) -> Resolution {
            Resolution::Export(Credentials {
                host: set[0].1.to_string(),
                public_key: set[1].1.to_string(),
                secret_key: set[2].1.to_string(),
            })
        }

        #[test]
        fn nothing_set_is_quietly_off() {
            assert_eq!(resolve(env(&[])), Resolution::Off);
        }

        #[test]
        fn generic_trio_still_works_without_ai_cli_vars() {
            assert_eq!(resolve(env(&GENERIC)), creds(&GENERIC));
        }

        #[test]
        fn incomplete_generic_trio_is_quietly_off() {
            // Backward compatible: a partial generic set never warned.
            assert_eq!(resolve(env(&GENERIC[..2])), Resolution::Off);
            assert_eq!(resolve(env(&GENERIC[1..])), Resolution::Off);
        }

        #[test]
        fn ai_cli_trio_is_used_on_its_own() {
            assert_eq!(resolve(env(&AI_CLI)), creds(&AI_CLI));
        }

        #[test]
        fn ai_cli_trio_wins_over_generic_trio() {
            let all = with(&GENERIC, &AI_CLI);
            assert_eq!(resolve(env(&all)), creds(&AI_CLI));
        }

        #[test]
        fn incomplete_ai_cli_trio_is_off_and_never_falls_back() {
            // Each case has a complete generic trio that must NOT be used,
            // neither whole nor filling in the missing ai-cli values.
            let cases: [(&Vars, &[&str]); 4] = [
                // ai-cli host with generic keys would mix projects.
                (
                    &AI_CLI[..1],
                    &["AI_CLI_LANGFUSE_PUBLIC_KEY", "AI_CLI_LANGFUSE_SECRET_KEY"],
                ),
                // ai-cli keys with the generic host: the other way round.
                (&AI_CLI[1..], &["AI_CLI_LANGFUSE_HOST"]),
                (&AI_CLI[..2], &["AI_CLI_LANGFUSE_SECRET_KEY"]),
                (
                    &AI_CLI[2..],
                    &["AI_CLI_LANGFUSE_HOST", "AI_CLI_LANGFUSE_PUBLIC_KEY"],
                ),
            ];
            for (ai_cli, missing) in cases {
                let got = resolve(env(&with(&GENERIC, ai_cli)));
                assert_eq!(got, Resolution::Incomplete(missing.to_vec()), "{ai_cli:?}");
                // Same without the generic trio around.
                assert_eq!(
                    resolve(env(ai_cli)),
                    Resolution::Incomplete(missing.to_vec())
                );
            }
        }

        #[test]
        fn empty_ai_cli_var_counts_as_unset() {
            let vars = with(
                &GENERIC,
                &[
                    ("AI_CLI_LANGFUSE_HOST", ""),
                    ("AI_CLI_LANGFUSE_SECRET_KEY", ""),
                ],
            );
            assert_eq!(resolve(env(&vars)), creds(&GENERIC));

            // An empty value inside an otherwise-set ai-cli trio is missing.
            let vars = with(&AI_CLI[..2], &[("AI_CLI_LANGFUSE_SECRET_KEY", "")]);
            assert_eq!(
                resolve(env(&vars)),
                Resolution::Incomplete(vec!["AI_CLI_LANGFUSE_SECRET_KEY"])
            );
        }

        #[test]
        fn incomplete_warning_names_missing_vars_but_no_values() {
            // Distinctive values in every credential var, ai-cli and generic.
            let vars = [
                ("AI_CLI_LANGFUSE_HOST", "http://sentinel-host-1.example"),
                ("AI_CLI_LANGFUSE_PUBLIC_KEY", "pk-lf-sentinel-2"),
                ("LANGFUSE_HOST", "http://sentinel-host-3.example"),
                ("LANGFUSE_PUBLIC_KEY", "pk-lf-sentinel-4"),
                ("LANGFUSE_SECRET_KEY", "sk-lf-sentinel-5"),
            ];
            let Resolution::Incomplete(missing) = resolve(env(&vars)) else {
                panic!("expected an incomplete ai-cli trio");
            };
            let text = incomplete_warning(&missing);
            assert!(
                text.contains("AI_CLI_LANGFUSE_SECRET_KEY is not set"),
                "{text}"
            );
            assert!(!text.contains('\n'), "one line: {text}");
            for (_, value) in vars {
                assert!(!text.contains(value), "warning leaks a value: {text}");
            }
            assert!(!text.contains("sentinel"), "warning leaks a value: {text}");

            let two =
                incomplete_warning(&["AI_CLI_LANGFUSE_PUBLIC_KEY", "AI_CLI_LANGFUSE_SECRET_KEY"]);
            assert!(
                two.contains(
                    "AI_CLI_LANGFUSE_PUBLIC_KEY and AI_CLI_LANGFUSE_SECRET_KEY are not set"
                ),
                "{two}"
            );
        }

        #[test]
        fn credentials_debug_hides_keys() {
            let Resolution::Export(c) = creds(&AI_CLI) else {
                unreachable!()
            };
            let text = format!("{c:?}");
            assert!(text.contains("http://ai-cli.example"), "{text}");
            assert!(!text.contains("pk-lf-ai-cli"), "{text}");
            assert!(!text.contains("sk-lf-ai-cli"), "{text}");
        }

        #[test]
        fn ai_cli_enabled_false_switches_off_generic_trio() {
            // The shared-shell opt-out: another app's LANGFUSE_* stay as
            // they are, ai-cli exports nothing.
            for off in ["false", "0"] {
                let vars = with(&GENERIC, &[("AI_CLI_LANGFUSE_ENABLED", off)]);
                assert_eq!(resolve(env(&vars)), Resolution::Off, "{off}");
            }
        }

        #[test]
        fn ai_cli_enabled_false_switches_off_ai_cli_trio() {
            let vars = with(&AI_CLI, &[("AI_CLI_LANGFUSE_ENABLED", "false")]);
            assert_eq!(resolve(env(&vars)), Resolution::Off);
        }

        #[test]
        fn switched_off_incomplete_trio_does_not_warn() {
            let vars = with(&AI_CLI[..1], &[("AI_CLI_LANGFUSE_ENABLED", "false")]);
            assert_eq!(resolve(env(&vars)), Resolution::Off);
        }

        #[test]
        fn generic_enabled_still_applies_when_ai_cli_enabled_unset() {
            let off = with(&GENERIC, &[("LANGFUSE_ENABLED", "false")]);
            assert_eq!(resolve(env(&off)), Resolution::Off);
            // An empty ai-cli switch counts as unset.
            let off = with(&off, &[("AI_CLI_LANGFUSE_ENABLED", "")]);
            assert_eq!(resolve(env(&off)), Resolution::Off);
            let off = with(&AI_CLI, &[("LANGFUSE_ENABLED", "0")]);
            assert_eq!(resolve(env(&off)), Resolution::Off);
        }

        #[test]
        fn ai_cli_enabled_overrides_generic_enabled() {
            // Another app switched its own telemetry off; ai-cli's stays on.
            let vars = with(
                &AI_CLI,
                &[
                    ("LANGFUSE_ENABLED", "false"),
                    ("AI_CLI_LANGFUSE_ENABLED", "true"),
                ],
            );
            assert_eq!(resolve(env(&vars)), creds(&AI_CLI));
            // And the other way round.
            let vars = with(
                &GENERIC,
                &[
                    ("LANGFUSE_ENABLED", "true"),
                    ("AI_CLI_LANGFUSE_ENABLED", "false"),
                ],
            );
            assert_eq!(resolve(env(&vars)), Resolution::Off);
        }

        #[test]
        fn capture_content_is_on_by_default() {
            assert!(capture_content_from(env(&[])));
        }

        #[test]
        fn generic_capture_content_still_works() {
            assert!(!capture_content_from(env(&[(
                "LANGFUSE_CAPTURE_CONTENT",
                "false"
            )])));
            // An empty ai-cli switch counts as unset.
            assert!(!capture_content_from(env(&[
                ("LANGFUSE_CAPTURE_CONTENT", "0"),
                ("AI_CLI_LANGFUSE_CAPTURE_CONTENT", ""),
            ])));
        }

        #[test]
        fn ai_cli_capture_content_overrides_generic() {
            assert!(!capture_content_from(env(&[(
                "AI_CLI_LANGFUSE_CAPTURE_CONTENT",
                "false"
            )])));
            assert!(!capture_content_from(env(&[
                ("AI_CLI_LANGFUSE_CAPTURE_CONTENT", "false"),
                ("LANGFUSE_CAPTURE_CONTENT", "true"),
            ])));
            assert!(capture_content_from(env(&[
                ("AI_CLI_LANGFUSE_CAPTURE_CONTENT", "true"),
                ("LANGFUSE_CAPTURE_CONTENT", "false"),
            ])));
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
