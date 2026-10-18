//! Per-model token prices and the footer's cost estimate.
//!
//! The estimate is `input tokens × input price + output tokens × output
//! price` at each model's base list price. It deliberately ignores what the
//! token counts can't tell us: prompt-cache discounts, batch pricing,
//! long-context tiers (Gemini Pro and Grok above 200k prompt tokens, GPT-5.4
//! and GPT-5.5 above 272k), regional/data-residency premiums (Bedrock and
//! Vertex regional endpoints, `inference_geo`), and per-search fees. Treat it
//! as a ballpark, not a bill.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A model's price in USD per million tokens. Also the shape of a
/// `[pricing."<model id>"]` entry in the config.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ModelPrice {
    pub input_per_mtok: f64,
    pub output_per_mtok: f64,
}

impl ModelPrice {
    /// Cost in USD for the given token counts, or `None` if the price isn't
    /// a usable number (e.g. a negative or `nan` override).
    pub fn cost(&self, input_tokens: i32, output_tokens: i32) -> Option<f64> {
        // Check each rate, not just the total: one negative rate can be
        // outweighed by the other and still yield a plausible-looking sum.
        let usable = |rate: f64| rate.is_finite() && rate >= 0.0;
        if !usable(self.input_per_mtok) || !usable(self.output_per_mtok) {
            return None;
        }
        let cost = (f64::from(input_tokens.max(0)) * self.input_per_mtok
            + f64::from(output_tokens.max(0)) * self.output_per_mtok)
            / 1_000_000.0;
        (cost.is_finite() && cost >= 0.0).then_some(cost)
    }
}

/// Built-in list prices as `(normalized model id, input $/Mtok, output
/// $/Mtok)`. Ids are in the form `normalize_model_id` produces, so dated
/// snapshots, Bedrock/Vertex spellings and OpenRouter routes (OpenRouter
/// passes through the provider's list price) of these models match too.
///
/// Prices as of 2026-09-30: standard tier, text input, base context. Verify at
///   Anthropic: https://platform.claude.com/docs/en/about-claude/pricing
///   OpenAI:    https://developers.openai.com/api/docs/pricing
///   Google:    https://ai.google.dev/gemini-api/docs/pricing
///   xAI:       https://docs.x.ai/docs/models
///
/// Leave a model out rather than guess: an unknown model simply shows no
/// cost, and users can add it under `[pricing]`.
const BUILTIN_PRICES: &[(&str, f64, f64)] = &[
    // Anthropic
    ("claude-fable-5-1", 10.0, 50.0),
    ("claude-fable-5", 10.0, 50.0),
    ("claude-opus-5-5", 4.0, 20.0),
    ("claude-opus-5", 5.0, 25.0),
    ("claude-opus-4-8", 5.0, 25.0),
    ("claude-opus-4-7", 5.0, 25.0),
    ("claude-opus-4-6", 5.0, 25.0),
    ("claude-opus-4-5", 5.0, 25.0),
    ("claude-opus-4-1", 15.0, 75.0),
    ("claude-opus-4", 15.0, 75.0),
    ("claude-sonnet-5-5", 2.0, 10.0),
    ("claude-sonnet-5", 2.0, 10.0),
    ("claude-sonnet-4-6", 3.0, 15.0),
    ("claude-sonnet-4-5", 3.0, 15.0),
    ("claude-sonnet-4", 3.0, 15.0),
    ("claude-haiku-4-5", 1.0, 5.0),
    ("claude-3-5-haiku", 0.80, 4.0),
    // OpenAI
    ("gpt-5.5-pro", 30.0, 180.0),
    ("gpt-5.5", 5.0, 30.0),
    ("gpt-5.4-pro", 30.0, 180.0),
    ("gpt-5.4", 2.50, 15.0),
    ("gpt-5.4-mini", 0.75, 4.50),
    ("gpt-5.4-nano", 0.20, 1.25),
    ("gpt-5.2", 1.75, 14.0),
    ("gpt-5.1", 1.25, 10.0),
    ("gpt-5", 1.25, 10.0),
    ("gpt-5-mini", 0.25, 2.0),
    ("gpt-5-nano", 0.05, 0.40),
    // Google
    ("gemini-3.5-flash", 1.50, 9.0),
    ("gemini-3.1-pro-preview", 2.0, 12.0),
    ("gemini-3.1-flash-lite", 0.25, 1.50),
    ("gemini-3.1-flash-lite-preview", 0.25, 1.50),
    ("gemini-3-flash-preview", 0.50, 3.0),
    ("gemini-2.5-pro", 1.25, 10.0),
    ("gemini-2.5-flash", 0.30, 2.50),
    ("gemini-2.5-flash-lite", 0.10, 0.40),
    // xAI
    ("grok-4.7", 2.0, 6.0),
    ("grok-4.6", 2.0, 6.0),
    ("grok-4.5", 2.0, 6.0),
    ("grok-4.3", 1.25, 2.50),
    ("grok-4.20-0309-reasoning", 1.25, 2.50),
    ("grok-4.20-0309-non-reasoning", 1.25, 2.50),
    ("grok-4.20-multi-agent-0309", 1.25, 2.50),
    // Retired on 2026-05-15; requests to these ids are redirected to grok-4.3
    // and billed at its price (docs.x.ai/developers/migration/may-15-retirement).
    // The bare `grok-4` alias isn't listed there, so it stays unknown.
    ("grok-4-1-fast-non-reasoning", 1.25, 2.50),
    ("grok-4-1-fast-reasoning", 1.25, 2.50),
    ("grok-4-fast-non-reasoning", 1.25, 2.50),
    ("grok-4-fast-reasoning", 1.25, 2.50),
    ("grok-4-0709", 1.25, 2.50),
];

/// Bedrock cross-region inference-profile prefixes (`us.anthropic.claude-…`).
const BEDROCK_GEOS: &[&str] = &["global", "us", "us-gov", "eu", "apac", "au", "jp", "ca"];

fn all_digits(s: &str, len: usize) -> bool {
    s.len() == len && s.bytes().all(|b| b.is_ascii_digit())
}

/// Drop a Bedrock version suffix: `-v1:0`, `-v2`.
fn strip_bedrock_version(id: &str) -> &str {
    if let Some(pos) = id.rfind("-v") {
        let version = &id[pos + 2..];
        let (major, minor) = version.split_once(':').unwrap_or((version, "0"));
        let numeric = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
        if numeric(major) && numeric(minor) {
            return &id[..pos];
        }
    }
    id
}

/// Drop a snapshot date: `-20251001` (Anthropic) or `-2025-08-07` (OpenAI).
/// Four-digit `-MMDD` suffixes are kept: for xAI they name distinct models
/// (`grok-4-0709` vs the `grok-4` alias).
fn strip_date_suffix(id: &str) -> &str {
    if let Some((base, date)) = id.rsplit_once('-') {
        if all_digits(date, 8) {
            return base;
        }
    }
    let mut parts = id.rsplitn(4, '-');
    if let (Some(day), Some(month), Some(year), Some(base)) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    {
        if all_digits(year, 4) && all_digits(month, 2) && all_digits(day, 2) {
            return base;
        }
    }
    id
}

/// Reduce the many spellings of a model id to the form used as a table key:
/// lowercase, without a route/namespace prefix (`anthropic/…`, `models/…`,
/// genai's `anthropic::…`), a Bedrock region/vendor prefix
/// (`us.anthropic.…`) or version suffix (`-v1:0`), a Vertex `@snapshot`, or
/// a date suffix. OpenRouter's dotted Claude versions (`claude-sonnet-4.5`)
/// become dashed like Anthropic's own ids.
pub fn normalize_model_id(model: &str) -> String {
    let lower = model.trim().to_ascii_lowercase();
    let mut id = lower.as_str();

    if let Some((_, rest)) = id.rsplit_once("::") {
        id = rest;
    }
    if let Some((_, rest)) = id.rsplit_once('/') {
        id = rest;
    }
    if let Some((geo, rest)) = id.split_once('.') {
        if BEDROCK_GEOS.contains(&geo) && rest.starts_with("anthropic.") {
            id = rest;
        }
    }
    if let Some(rest) = id.strip_prefix("anthropic.") {
        id = rest;
    }
    if let Some((base, _)) = id.split_once('@') {
        id = base;
    }
    id = strip_bedrock_version(id);
    id = strip_date_suffix(id);

    if id.starts_with("claude-") {
        id.replace('.', "-")
    } else {
        id.to_string()
    }
}

/// The price for `model`, most specific first:
/// 1. a `[pricing]` override keyed by exactly this id (so a key like
///    `"anthropic/claude-opus-4-8"` prices that OpenRouter route only);
/// 2. an override keyed by the normalized id (so `"claude-opus-4-8"` also
///    covers its dated, Bedrock and OpenRouter spellings);
/// 3. the built-in table.
///
/// Override keys are matched case-insensitively. `None` means unknown.
pub fn lookup(model: &str, overrides: &HashMap<String, ModelPrice>) -> Option<ModelPrice> {
    let raw = model.trim();
    let normalized = normalize_model_id(raw);
    let find_override = |id: &str| {
        overrides
            .iter()
            .find(|(key, _)| key.trim().eq_ignore_ascii_case(id))
            .map(|(_, price)| *price)
    };
    find_override(raw)
        .or_else(|| find_override(&normalized))
        .or_else(|| {
            BUILTIN_PRICES
                .iter()
                .find(|(id, _, _)| *id == normalized)
                .map(|&(_, input, output)| ModelPrice {
                    input_per_mtok: input,
                    output_per_mtok: output,
                })
        })
}

/// Estimated cost in USD of one answer, or `None` when the model's price is
/// unknown.
pub fn estimate_cost(
    model: &str,
    input_tokens: i32,
    output_tokens: i32,
    overrides: &HashMap<String, ModelPrice>,
) -> Option<f64> {
    lookup(model, overrides)?.cost(input_tokens, output_tokens)
}

/// Format a USD amount for the footer: two decimals from $1 up (`$12.35`),
/// four significant digits below that (`$0.004238`), with trailing zeros
/// trimmed down to two decimals (`$0.0042`, `$0.50`).
pub fn format_cost(usd: f64) -> String {
    if usd >= 1.0 {
        return format!("${:.2}", usd);
    }
    if usd <= 0.0 || !usd.is_finite() {
        return "$0.00".to_string();
    }
    // Decimals needed for four significant digits, capped so a
    // near-zero amount can't produce an absurdly long string.
    let decimals = (3 - usd.log10().floor() as i32).clamp(2, 10) as usize;
    let fixed = format!("{:.*}", decimals, usd);
    let trimmed = fixed.trim_end_matches('0');
    let (whole, fraction) = trimmed.split_once('.').unwrap_or((trimmed, ""));
    format!("${}.{:0<2}", whole, fraction)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn any_negative_or_non_finite_rate_makes_the_price_unusable() {
        // -1 input with 10 output and equal tokens sums positive, but the
        // override is still malformed.
        let bad = ModelPrice {
            input_per_mtok: -1.0,
            output_per_mtok: 10.0,
        };
        assert_eq!(bad.cost(1_000, 1_000), None);
        let nan = ModelPrice {
            input_per_mtok: 1.0,
            output_per_mtok: f64::NAN,
        };
        assert_eq!(nan.cost(1, 1), None);
        let ok = ModelPrice {
            input_per_mtok: 1.0,
            output_per_mtok: 2.0,
        };
        assert_eq!(ok.cost(1_000_000, 1_000_000), Some(3.0));
    }

    fn price(input: f64, output: f64) -> ModelPrice {
        ModelPrice {
            input_per_mtok: input,
            output_per_mtok: output,
        }
    }

    fn builtin(model: &str) -> Option<ModelPrice> {
        lookup(model, &HashMap::new())
    }

    #[test]
    fn exact_ids_match_the_table() {
        assert_eq!(builtin("claude-haiku-4-5"), Some(price(1.0, 5.0)));
        assert_eq!(builtin("claude-opus-4-8"), Some(price(5.0, 25.0)));
        assert_eq!(builtin("gpt-5-nano"), Some(price(0.05, 0.40)));
        assert_eq!(builtin("gpt-5.5"), Some(price(5.0, 30.0)));
        assert_eq!(builtin("gemini-3.5-flash"), Some(price(1.50, 9.0)));
        assert_eq!(builtin("grok-4.3"), Some(price(1.25, 2.50)));
        // Matching is exact, never by prefix: these are different models.
        assert_eq!(builtin("gpt-5"), Some(price(1.25, 10.0)));
        assert_eq!(builtin("gpt-5.4-mini"), Some(price(0.75, 4.50)));
        // Case and surrounding whitespace don't matter.
        assert_eq!(builtin("  Claude-Haiku-4-5 "), Some(price(1.0, 5.0)));
    }

    #[test]
    fn date_suffixed_ids_match() {
        assert_eq!(builtin("claude-haiku-4-5-20251001"), Some(price(1.0, 5.0)));
        assert_eq!(builtin("gpt-5-2025-08-07"), Some(price(1.25, 10.0)));
        assert_eq!(builtin("claude-haiku-4-5@20251001"), Some(price(1.0, 5.0)));
        // A four-digit xAI suffix names a distinct model, not a snapshot.
        assert_eq!(normalize_model_id("grok-4-0709"), "grok-4-0709");
    }

    #[test]
    fn bedrock_and_routed_ids_match() {
        assert_eq!(
            builtin("us.anthropic.claude-opus-4-7"),
            Some(price(5.0, 25.0))
        );
        assert_eq!(
            builtin("global.anthropic.claude-haiku-4-5-20251001-v1:0"),
            Some(price(1.0, 5.0))
        );
        assert_eq!(
            builtin("anthropic.claude-3-5-haiku-20241022-v1:0"),
            Some(price(0.80, 4.0))
        );
        assert_eq!(builtin("anthropic.claude-opus-5-5"), Some(price(4.0, 20.0)));
        // OpenRouter routes, including its dotted Claude versions.
        assert_eq!(builtin("anthropic/claude-opus-4-8"), Some(price(5.0, 25.0)));
        assert_eq!(
            builtin("anthropic/claude-sonnet-4.5"),
            Some(price(3.0, 15.0))
        );
        assert_eq!(
            builtin("google/gemini-3.1-flash-lite-preview"),
            Some(price(0.25, 1.50))
        );
        assert_eq!(builtin("models/gemini-2.5-flash"), Some(price(0.30, 2.50)));
        assert_eq!(builtin("openai::gpt-5-nano"), Some(price(0.05, 0.40)));
    }

    #[test]
    fn unknown_models_have_no_price() {
        for model in [
            "",
            "qwen3:8b",
            "grok-4",
            "some-new-model",
            "gpt-5.9",
            "claude-opus-9",
            "meta-llama/llama-4-maverick:free",
        ] {
            assert_eq!(builtin(model), None, "{model:?} should be unknown");
            assert_eq!(estimate_cost(model, 1000, 1000, &HashMap::new()), None);
        }
    }

    #[test]
    fn table_keys_are_normalized_unique_and_positive() {
        let mut seen = std::collections::HashSet::new();
        for &(id, input, output) in BUILTIN_PRICES {
            assert_eq!(
                normalize_model_id(id),
                id,
                "table key {id:?} isn't normalized"
            );
            assert!(seen.insert(id), "duplicate table key {id:?}");
            assert!(
                input > 0.0 && output > 0.0,
                "{id:?} has a non-positive price"
            );
        }
    }

    #[test]
    fn overrides_take_precedence_most_specific_first() {
        let mut overrides = HashMap::new();
        // A bare id overrides the built-in price for every spelling.
        overrides.insert("claude-opus-4-8".to_string(), price(4.0, 20.0));
        assert_eq!(
            lookup("claude-opus-4-8", &overrides),
            Some(price(4.0, 20.0))
        );
        assert_eq!(
            lookup("us.anthropic.claude-opus-4-8", &overrides),
            Some(price(4.0, 20.0))
        );

        // An exact routed id beats the bare one, but only for that route.
        overrides.insert("Anthropic/Claude-Opus-4-8".to_string(), price(15.0, 75.0));
        assert_eq!(
            lookup("anthropic/claude-opus-4-8", &overrides),
            Some(price(15.0, 75.0))
        );
        assert_eq!(
            lookup("claude-opus-4-8", &overrides),
            Some(price(4.0, 20.0))
        );

        // Overrides can add models the table doesn't know.
        overrides.insert("qwen3:8b".to_string(), price(0.0, 0.0));
        assert_eq!(estimate_cost("qwen3:8b", 5000, 5000, &overrides), Some(0.0));

        // Other models still use the table.
        assert_eq!(
            lookup("claude-haiku-4-5", &overrides),
            Some(price(1.0, 5.0))
        );
    }

    #[test]
    fn cost_is_tokens_times_price_per_million() {
        let haiku = price(1.0, 5.0);
        assert_eq!(haiku.cost(1_000_000, 0), Some(1.0));
        let cost = haiku.cost(1000, 500).unwrap();
        assert!((cost - 0.0035).abs() < 1e-12, "{cost}");
        // Nonsense from a bad override or a bad usage report never shows.
        assert_eq!(price(f64::NAN, 1.0).cost(10, 10), None);
        assert_eq!(price(-1.0, 1.0).cost(1000, 0), None);
        assert_eq!(haiku.cost(-5, 0), Some(0.0));
    }

    #[test]
    fn cost_formatting() {
        assert_eq!(format_cost(0.0042), "$0.0042");
        assert_eq!(format_cost(0.00423817), "$0.004238");
        assert_eq!(format_cost(0.123456), "$0.1235");
        assert_eq!(format_cost(0.5), "$0.50");
        assert_eq!(format_cost(0.001), "$0.001");
        assert_eq!(format_cost(0.99996), "$1.00");
        assert_eq!(format_cost(1.0), "$1.00");
        assert_eq!(format_cost(12.346), "$12.35");
        assert_eq!(format_cost(1234.5), "$1234.50");
        assert_eq!(format_cost(0.0), "$0.00");
        assert_eq!(format_cost(0.00000005), "$0.00000005");
    }

    #[test]
    fn pricing_config_parses_and_round_trips() {
        let toml_text = r#"
[footer]
show_tokens = true

[pricing."anthropic/claude-opus-4-8"]
input_per_mtok = 15.0
output_per_mtok = 75

[pricing."qwen3:8b"]
input_per_mtok = 0.0
output_per_mtok = 0.0
"#;
        let config: Config = toml::from_str(toml_text).expect("pricing config parses");
        // show_cost defaults to on when the key is absent.
        assert!(config.footer.show_cost);
        // Integers are accepted as prices.
        assert_eq!(
            config.pricing["anthropic/claude-opus-4-8"],
            price(15.0, 75.0)
        );

        let saved = toml::to_string_pretty(&config).expect("serializes");
        let reloaded: Config = toml::from_str(&saved).expect("saved config parses");
        assert_eq!(reloaded.pricing, config.pricing);
    }

    #[test]
    fn configs_without_pricing_still_parse() {
        let config: Config = toml::from_str("[footer]\nshow_model = false\n").unwrap();
        assert!(config.pricing.is_empty());
        assert!(config.footer.show_cost);
        // Nothing new is written for users who never set a price.
        let saved = toml::to_string_pretty(&Config::default()).unwrap();
        assert!(!saved.contains("[pricing"), "{saved}");
    }
}
