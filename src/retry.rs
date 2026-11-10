//! Retry classification + backoff for transient provider errors.
//!
//! Wraps the genai error hierarchy to decide whether an error is worth
//! retrying (429, 5xx, network timeout/connect) and how long to wait.
//! A server-supplied `Retry-After` header wins over the default
//! exponential backoff; otherwise we double the base delay up to a
//! configured ceiling.

use crate::bedrock::{BedrockHttpError, BedrockStreamException, BedrockStreamTruncated};
use crate::config::RetryConfig;
use genai::Error as GenaiError;
use reqwest::StatusCode;
use std::time::{Duration, SystemTime};

/// Outcome of classifying an error. `None` means give up; `Some` means
/// retry — with an explicit `Retry-After` delay if the server set one.
pub enum Retryable {
    No,
    Yes { retry_after: Option<Duration> },
}

/// Walk the error chain and classify the first error type we recognize:
/// genai errors (both the request-time and stream-time shapes), Bedrock's
/// typed HTTP error, and raw reqwest transport errors.
pub fn classify(err: &anyhow::Error) -> Retryable {
    for cause in err.chain() {
        if let Some(g) = cause.downcast_ref::<GenaiError>() {
            return classify_genai(g);
        }
        if let Some(b) = cause.downcast_ref::<BedrockHttpError>() {
            return classify_status(b.status, b.retry_after);
        }
        // Bedrock failures that arrive after an HTTP 200, inside the stream.
        if let Some(e) = cause.downcast_ref::<BedrockStreamException>() {
            return if e.is_transient() {
                Retryable::Yes { retry_after: None }
            } else {
                Retryable::No
            };
        }
        if cause.downcast_ref::<BedrockStreamTruncated>().is_some() {
            return Retryable::Yes { retry_after: None };
        }
        if let Some(r) = cause.downcast_ref::<reqwest::Error>() {
            return classify_reqwest(r);
        }
    }
    Retryable::No
}

fn classify_genai(err: &GenaiError) -> Retryable {
    match err {
        GenaiError::WebAdapterCall { webc_error, .. }
        | GenaiError::WebModelCall { webc_error, .. } => classify_webc(webc_error),
        GenaiError::HttpError { status, .. } => classify_status(*status, None),
        // Streaming requests are sent lazily on the first poll of the stream,
        // so HTTP status and transport failures surface here, boxed, rather
        // than from `exec_chat_stream` itself.
        GenaiError::WebStream { error, .. } => classify_boxed(error.as_ref()),
        _ => Retryable::No,
    }
}

/// Classify the boxed error inside a `WebStream`. genai 0.5 boxes an
/// `HttpError` or a raw `reqwest::Error` here; `webc::Error` and the
/// `source()` chain are handled too so a genai upgrade that wraps these
/// differently doesn't silently turn retries off again.
fn classify_boxed(err: &(dyn std::error::Error + Send + Sync + 'static)) -> Retryable {
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(err);
    while let Some(e) = current {
        if let Some(g) = e.downcast_ref::<GenaiError>() {
            return classify_genai(g);
        }
        if let Some(w) = e.downcast_ref::<genai::webc::Error>() {
            return classify_webc(w);
        }
        if let Some(r) = e.downcast_ref::<reqwest::Error>() {
            return classify_reqwest(r);
        }
        current = e.source();
    }
    Retryable::No
}

fn classify_status(status: StatusCode, retry_after: Option<Duration>) -> Retryable {
    if is_retryable_status(status) {
        Retryable::Yes { retry_after }
    } else {
        Retryable::No
    }
}

/// Also used by the OpenAI Responses transport (#75).
pub fn classify_reqwest(err: &reqwest::Error) -> Retryable {
    if err.is_timeout() || err.is_connect() {
        return Retryable::Yes { retry_after: None };
    }
    match err.status() {
        Some(status) => classify_status(status, None),
        None => Retryable::No,
    }
}

fn classify_webc(err: &genai::webc::Error) -> Retryable {
    use genai::webc::Error as W;
    match err {
        W::ResponseFailedStatus {
            status, headers, ..
        } => {
            if is_retryable_status(*status) {
                let retry_after = headers
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(parse_retry_after);
                Retryable::Yes { retry_after }
            } else {
                Retryable::No
            }
        }
        W::Reqwest(e) if e.is_timeout() || e.is_connect() => Retryable::Yes { retry_after: None },
        _ => Retryable::No,
    }
}

/// 529 is Anthropic's "overloaded" status — transient, and one of the most
/// common errors a Claude user sees under load.
pub fn is_retryable_status(s: StatusCode) -> bool {
    matches!(s.as_u16(), 408 | 425 | 429 | 500 | 502 | 503 | 504 | 529)
}

/// Parse a `Retry-After` header value per RFC 7231 §7.1.3: either a
/// non-negative integer number of seconds, or an HTTP-date. Anything else
/// returns `None` so the caller falls back to the default backoff.
pub fn parse_retry_after(value: &str) -> Option<Duration> {
    let trimmed = value.trim();
    if let Ok(secs) = trimmed.parse::<u64>() {
        return Some(Duration::from_secs(secs));
    }
    let when = httpdate::parse_http_date(trimmed).ok()?;
    when.duration_since(SystemTime::now()).ok()
}

/// Cap a Retry-After value so a misconfigured (or hostile) server can't
/// stall the CLI beyond the user-configured ceiling. Falls back to the
/// exponential backoff for the current attempt if the header was absent.
pub fn effective_delay(cfg: &RetryConfig, attempt: u32, retry_after: Option<Duration>) -> Duration {
    let max = Duration::from_millis(cfg.max_delay_ms);
    match retry_after {
        Some(d) => d.min(max),
        None => backoff(cfg, attempt),
    }
}

/// Exponential backoff: base << (attempt-1), capped at `max`. Attempt starts at 1.
pub fn backoff(cfg: &RetryConfig, attempt: u32) -> Duration {
    let shift = (attempt.saturating_sub(1)).min(10);
    let ms = cfg.base_ms.saturating_mul(1u64 << shift);
    Duration::from_millis(ms.min(cfg.max_delay_ms))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(base: u64, max: u64, attempts: u32) -> RetryConfig {
        RetryConfig {
            max_attempts: attempts,
            base_ms: base,
            max_delay_ms: max,
        }
    }

    #[test]
    fn backoff_doubles_per_attempt_until_cap() {
        let c = cfg(250, 5_000, 5);
        assert_eq!(backoff(&c, 1), Duration::from_millis(250));
        assert_eq!(backoff(&c, 2), Duration::from_millis(500));
        assert_eq!(backoff(&c, 3), Duration::from_millis(1000));
        assert_eq!(backoff(&c, 4), Duration::from_millis(2000));
        assert_eq!(backoff(&c, 5), Duration::from_millis(4000));
        // capped
        assert_eq!(backoff(&c, 6), Duration::from_millis(5000));
        assert_eq!(backoff(&c, 100), Duration::from_millis(5000));
    }

    fn is_yes(r: Retryable) -> bool {
        matches!(r, Retryable::Yes { .. })
    }

    fn http_error(code: u16) -> GenaiError {
        GenaiError::HttpError {
            status: StatusCode::from_u16(code).unwrap(),
            canonical_reason: String::new(),
            body: String::new(),
        }
    }

    fn web_stream_error(inner: GenaiError) -> GenaiError {
        GenaiError::WebStream {
            model_iden: genai::ModelIden::new(genai::adapter::AdapterKind::Anthropic, "m"),
            cause: inner.to_string(),
            error: Box::new(inner),
        }
    }

    #[test]
    fn classifies_http_error_by_status() {
        assert!(is_yes(classify(&anyhow::Error::from(http_error(503)))));
        assert!(is_yes(classify(&anyhow::Error::from(http_error(529)))));
        assert!(!is_yes(classify(&anyhow::Error::from(http_error(401)))));
    }

    #[test]
    fn classifies_status_errors_surfaced_mid_stream() {
        // genai sends streaming requests lazily, so a 429 arrives as a
        // WebStream error wrapping a boxed HttpError. This used to fall
        // through the catch-all and never retry.
        let err = anyhow::Error::from(web_stream_error(http_error(429)));
        assert!(is_yes(classify(&err)));
        let err = anyhow::Error::from(web_stream_error(http_error(400)));
        assert!(!is_yes(classify(&err)));
    }

    #[test]
    fn classifies_stream_errors_wrapped_in_other_errors() {
        // A future genai could box a wrapper instead of the HttpError
        // itself; the classifier walks `source()` to find it.
        #[derive(Debug)]
        struct Wrapper(GenaiError);
        impl std::fmt::Display for Wrapper {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "wrapped")
            }
        }
        impl std::error::Error for Wrapper {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                Some(&self.0)
            }
        }
        let err = GenaiError::WebStream {
            model_iden: genai::ModelIden::new(genai::adapter::AdapterKind::Anthropic, "m"),
            cause: "wrapped".into(),
            error: Box::new(Wrapper(http_error(503))),
        };
        assert!(is_yes(classify(&anyhow::Error::from(err))));
    }

    #[test]
    fn classifies_bedrock_http_errors_and_keeps_retry_after() {
        let err = anyhow::Error::from(BedrockHttpError {
            status: StatusCode::TOO_MANY_REQUESTS,
            retry_after: Some(Duration::from_secs(2)),
            message: "throttled".into(),
        })
        .context("Failed to execute Bedrock Converse request");
        match classify(&err) {
            Retryable::Yes { retry_after } => {
                assert_eq!(retry_after, Some(Duration::from_secs(2)))
            }
            Retryable::No => panic!("429 from Bedrock should retry"),
        }
        let err = anyhow::Error::from(BedrockHttpError {
            status: StatusCode::FORBIDDEN,
            retry_after: None,
            message: "denied".into(),
        });
        assert!(!is_yes(classify(&err)));
    }

    #[test]
    fn unrelated_errors_do_not_retry() {
        assert!(!is_yes(classify(&anyhow::anyhow!("something else"))));
    }

    #[test]
    fn is_retryable_status_covers_transients() {
        for code in [408, 425, 429, 500, 502, 503, 504, 529] {
            assert!(
                is_retryable_status(StatusCode::from_u16(code).unwrap()),
                "{} should be retryable",
                code
            );
        }
        for code in [200, 400, 401, 403, 404, 422] {
            assert!(
                !is_retryable_status(StatusCode::from_u16(code).unwrap()),
                "{} should not be retryable",
                code
            );
        }
    }

    #[test]
    fn parse_retry_after_handles_seconds() {
        assert_eq!(parse_retry_after("0"), Some(Duration::from_secs(0)));
        assert_eq!(parse_retry_after("30"), Some(Duration::from_secs(30)));
        assert_eq!(parse_retry_after("  120  "), Some(Duration::from_secs(120)));
    }

    #[test]
    fn parse_retry_after_handles_http_date() {
        // A date well in the future relative to any sane test clock.
        let parsed = parse_retry_after("Fri, 01 Jan 2100 00:00:00 GMT")
            .expect("future HTTP-date should parse");
        // Don't assert exact seconds (depends on wall clock); just sanity-check
        // we got something nontrivial.
        assert!(parsed.as_secs() > 60);
    }

    #[test]
    fn parse_retry_after_rejects_garbage() {
        assert!(parse_retry_after("not a number or date").is_none());
        assert!(parse_retry_after("").is_none());
    }

    #[test]
    fn effective_delay_caps_retry_after_at_max() {
        let c = cfg(250, 5_000, 5);
        // An hour-long Retry-After must clamp to max_delay_ms.
        let capped = effective_delay(&c, 1, Some(Duration::from_secs(3600)));
        assert_eq!(capped, Duration::from_millis(5_000));
    }

    #[test]
    fn effective_delay_falls_back_to_backoff_when_header_absent() {
        let c = cfg(250, 5_000, 5);
        assert_eq!(effective_delay(&c, 3, None), Duration::from_millis(1000));
    }
}
