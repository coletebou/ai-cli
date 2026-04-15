//! Retry classification + backoff for transient provider errors.
//!
//! Wraps the genai error hierarchy to decide whether an error is worth
//! retrying (429, 5xx, network timeout/connect) and how long to wait.
//! A server-supplied `Retry-After` header wins over the default
//! exponential backoff; otherwise we double the base delay up to a
//! configured ceiling.

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

/// Walk the error chain for a genai::Error and classify it.
pub fn classify(err: &anyhow::Error) -> Retryable {
    for cause in err.chain() {
        if let Some(g) = cause.downcast_ref::<GenaiError>() {
            return classify_genai(g);
        }
    }
    Retryable::No
}

fn classify_genai(err: &GenaiError) -> Retryable {
    match err {
        GenaiError::WebAdapterCall { webc_error, .. }
        | GenaiError::WebModelCall { webc_error, .. } => classify_webc(webc_error),
        GenaiError::HttpError { status, .. } if is_retryable_status(*status) => {
            Retryable::Yes { retry_after: None }
        }
        _ => Retryable::No,
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

fn is_retryable_status(s: StatusCode) -> bool {
    matches!(s.as_u16(), 408 | 425 | 429 | 500 | 502 | 503 | 504)
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

    #[test]
    fn is_retryable_status_covers_transients() {
        for code in [408, 425, 429, 500, 502, 503, 504] {
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
