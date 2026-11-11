//! The list of chat models one Bedrock endpoint can serve through Converse,
//! for the TUI model picker and `ai models --bedrock`.
//!
//! Bedrock has two kinds of model ID a Converse call accepts: a foundation
//! model ID (`moonshotai.kimi-k2.5`), for models invocable on demand, and a
//! system-defined cross-region inference profile (`us.anthropic.claude-opus-5-5`,
//! `global.openai.gpt-6.1-sol`), which is the only way to call most newer
//! models. Neither list alone is the answer: `/foundation-models` says which
//! models produce text and which are on demand but not which profiles exist,
//! and `/inference-profiles` lists profiles (embedding and image models
//! included) without saying what they produce. So both are fetched and joined
//! on the profile's underlying foundation model.
use crate::bedrock::BedrockClient;
use anyhow::{Context, Result};
use serde::Deserialize;

/// Upper bound on `/inference-profiles` pages, so a misbehaving endpoint that
/// keeps returning a `nextToken` can't loop forever.
const MAX_PROFILE_PAGES: usize = 20;

/// One model ID the picker can offer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BedrockModel {
    /// What to send as Converse's `modelId`.
    pub id: String,
    /// Human name, e.g. "Claude Opus 5.5".
    pub name: String,
    /// Vendor, e.g. "Anthropic".
    pub vendor: String,
}

impl BedrockModel {
    /// True when every whitespace-separated term of `query` appears
    /// (case-insensitively) in the ID, name or vendor, so "opus 5" and
    /// "grok us" both narrow the list the way you'd expect.
    pub fn matches(&self, query: &str) -> bool {
        let haystack = format!("{} {} {}", self.id, self.name, self.vendor).to_lowercase();
        query
            .to_lowercase()
            .split_whitespace()
            .all(|term| haystack.contains(term))
    }
}

#[derive(Debug, Deserialize)]
struct FoundationModelsResponse {
    #[serde(default, rename = "modelSummaries")]
    model_summaries: Vec<FoundationModelSummary>,
}

#[derive(Debug, Deserialize)]
struct FoundationModelSummary {
    #[serde(rename = "modelId")]
    model_id: String,
    #[serde(default, rename = "modelName")]
    model_name: Option<String>,
    #[serde(default, rename = "providerName")]
    provider_name: Option<String>,
    #[serde(default, rename = "inputModalities")]
    input_modalities: Vec<String>,
    #[serde(default, rename = "outputModalities")]
    output_modalities: Vec<String>,
    #[serde(default, rename = "inferenceTypesSupported")]
    inference_types_supported: Vec<String>,
    #[serde(default, rename = "modelLifecycle")]
    model_lifecycle: Option<ModelLifecycle>,
}

#[derive(Debug, Deserialize)]
struct ModelLifecycle {
    #[serde(default)]
    status: Option<String>,
}

#[derive(Debug, Deserialize)]
struct InferenceProfilesResponse {
    #[serde(default, rename = "inferenceProfileSummaries")]
    inference_profile_summaries: Vec<InferenceProfileSummary>,
    #[serde(default, rename = "nextToken")]
    next_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct InferenceProfileSummary {
    #[serde(rename = "inferenceProfileId")]
    inference_profile_id: String,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    models: Vec<InferenceProfileModel>,
}

#[derive(Debug, Deserialize)]
struct InferenceProfileModel {
    #[serde(default, rename = "modelArn")]
    model_arn: Option<String>,
}

impl FoundationModelSummary {
    /// A text-in, text-out chat model that hasn't been retired. LEGACY
    /// models still answer until their end-of-life date but are hidden from
    /// the picker; a typed ID still reaches them. Also hidden: models
    /// Converse can't hold a text chat with even though they "output TEXT" —
    /// speech-to-speech (Nova Sonic, which also outputs SPEECH), rerankers,
    /// and video-only analysis (TwelveLabs Pegasus).
    fn is_active_text_model(&self) -> bool {
        let active = self
            .model_lifecycle
            .as_ref()
            .and_then(|l| l.status.as_deref())
            .is_none_or(|s| s == "ACTIVE");
        let has = |list: &[String], m: &str| list.iter().any(|x| x == m);
        let chat_shaped = has(&self.input_modalities, "TEXT")
            && has(&self.output_modalities, "TEXT")
            && !has(&self.output_modalities, "SPEECH")
            && !["rerank", "pegasus"]
                .iter()
                .any(|kind| self.model_id.contains(kind));
        active && chat_shaped
    }

    fn to_model(&self, id: &str) -> BedrockModel {
        BedrockModel {
            id: id.to_string(),
            name: self
                .model_name
                .clone()
                .unwrap_or_else(|| self.model_id.clone()),
            vendor: self.provider_name.clone().unwrap_or_default(),
        }
    }
}

/// `arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-opus-5-5`
/// → `anthropic.claude-opus-5-5`.
fn foundation_model_id(arn: &str) -> Option<&str> {
    arn.split_once("foundation-model/").map(|(_, id)| id)
}

/// Join the two listings into the picker's catalog: every active profile
/// whose model produces text, plus every on-demand text model, sorted by
/// vendor, then name, then ID (so `global.` and `us.` variants sit together).
fn build_catalog(
    foundation_models: &[FoundationModelSummary],
    profiles: &[InferenceProfileSummary],
) -> Vec<BedrockModel> {
    let by_id: std::collections::HashMap<&str, &FoundationModelSummary> = foundation_models
        .iter()
        .map(|m| (m.model_id.as_str(), m))
        .collect();

    let mut out: Vec<BedrockModel> = Vec::new();
    for profile in profiles {
        if profile.status.as_deref().is_some_and(|s| s != "ACTIVE") {
            continue;
        }
        let base = profile
            .models
            .iter()
            .filter_map(|m| m.model_arn.as_deref().and_then(foundation_model_id))
            .find_map(|id| by_id.get(id));
        if let Some(fm) = base.filter(|fm| fm.is_active_text_model()) {
            out.push(fm.to_model(&profile.inference_profile_id));
        }
    }
    for fm in foundation_models {
        if fm.is_active_text_model()
            && fm
                .inference_types_supported
                .iter()
                .any(|t| t == "ON_DEMAND")
        {
            out.push(fm.to_model(&fm.model_id));
        }
    }

    out.sort_by(|a, b| {
        (a.vendor.to_lowercase(), &a.name, &a.id).cmp(&(b.vendor.to_lowercase(), &b.name, &b.id))
    });
    out.dedup_by(|a, b| a.id == b.id);
    out
}

/// Fetch the catalog for `client`'s region.
pub async fn list_models(client: &BedrockClient) -> Result<Vec<BedrockModel>> {
    let fm_json = client
        .control_plane_get("/foundation-models?byOutputModality=TEXT")
        .await?;
    let foundation: FoundationModelsResponse =
        serde_json::from_value(fm_json).context("parsing /foundation-models")?;

    let mut profiles: Vec<InferenceProfileSummary> = Vec::new();
    let mut next_token: Option<String> = None;
    for _ in 0..MAX_PROFILE_PAGES {
        let mut path =
            String::from("/inference-profiles?typeEquals=SYSTEM_DEFINED&maxResults=1000");
        if let Some(token) = &next_token {
            path.push_str("&nextToken=");
            path.push_str(&urlencode_query_value(token));
        }
        let page: InferenceProfilesResponse =
            serde_json::from_value(client.control_plane_get(&path).await?)
                .context("parsing /inference-profiles")?;
        profiles.extend(page.inference_profile_summaries);
        next_token = page.next_token.filter(|t| !t.is_empty());
        if next_token.is_none() {
            break;
        }
    }

    Ok(build_catalog(&foundation.model_summaries, &profiles))
}

/// Explanation shown when a Bedrock listing is asked for but no provider
/// can reach Bedrock.
pub const NO_BEDROCK_PROVIDER: &str = "no Bedrock provider is available: give an enabled \
    provider base_url = \"https://bedrock-runtime.<region>.amazonaws.com/\" and api_key_env = \
    \"AWS_BEARER_TOKEN_BEDROCK\", and set that variable";

/// `ai models --bedrock [search]`: print the catalog of the first Bedrock
/// provider (in panel order).
pub async fn print_models(search: Option<&str>) -> Result<()> {
    use colored::Colorize;
    let config = crate::config::Config::load()?;
    let manager = crate::providers::ProviderManager::new(config, Default::default())?;
    let Some((provider, client)) = manager
        .bedrock_provider_names()
        .into_iter()
        .find_map(|name| manager.bedrock_client(&name).map(|c| (name, c)))
    else {
        anyhow::bail!(NO_BEDROCK_PROVIDER);
    };
    let models = list_models(&client).await?;
    let filtered: Vec<&BedrockModel> = models
        .iter()
        .filter(|m| search.is_none_or(|q| m.matches(q)))
        .collect();
    if filtered.is_empty() {
        match search {
            Some(q) => eprintln!("No Bedrock models matching '{}'", q),
            None => eprintln!("No Bedrock models found."),
        }
        return Ok(());
    }

    let id_width = filtered
        .iter()
        .map(|m| m.id.len())
        .max()
        .unwrap_or(8)
        .max(8);
    let name_width = filtered
        .iter()
        .map(|m| m.name.len())
        .max()
        .unwrap_or(4)
        .max(4);
    println!(
        "\n{} {} models (via provider `{}`){}",
        "Bedrock".yellow().bold(),
        filtered.len(),
        provider,
        search
            .map(|q| format!(" matching '{}'", q))
            .unwrap_or_default()
    );
    println!(
        "{:<iw$}  {:<nw$}  {}",
        "MODEL ID".bold(),
        "NAME".bold(),
        "VENDOR".bold(),
        iw = id_width,
        nw = name_width
    );
    println!("{}", "─".repeat(id_width + name_width + 16));
    for m in filtered {
        println!(
            "{:<iw$}  {:<nw$}  {}",
            m.id,
            m.name,
            m.vendor.dimmed(),
            iw = id_width,
            nw = name_width
        );
    }
    println!(
        "\n{} Use one with {} or pick it in the TUI ({} in normal mode, {} or {}).",
        "Tip:".dimmed(),
        "ai -p <provider> -m <MODEL ID>".bold(),
        "m".bold(),
        "Ctrl+O".bold(),
        "/model".bold()
    );
    Ok(())
}

/// Percent-encode a query-string value (a pagination token is opaque and
/// may contain `+`, `/` and `=`).
fn urlencode_query_value(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fm(
        id: &str,
        name: &str,
        vendor: &str,
        out: &[&str],
        types: &[&str],
    ) -> FoundationModelSummary {
        FoundationModelSummary {
            model_id: id.into(),
            model_name: Some(name.into()),
            provider_name: Some(vendor.into()),
            input_modalities: vec!["TEXT".into()],
            output_modalities: out.iter().map(|s| s.to_string()).collect(),
            inference_types_supported: types.iter().map(|s| s.to_string()).collect(),
            model_lifecycle: Some(ModelLifecycle {
                status: Some("ACTIVE".into()),
            }),
        }
    }

    fn profile(id: &str, base: &str) -> InferenceProfileSummary {
        InferenceProfileSummary {
            inference_profile_id: id.into(),
            status: Some("ACTIVE".into()),
            models: vec![InferenceProfileModel {
                model_arn: Some(format!(
                    "arn:aws:bedrock:us-east-1::foundation-model/{}",
                    base
                )),
            }],
        }
    }

    #[test]
    fn joins_profiles_with_text_models_and_keeps_on_demand_ids() {
        let fms = vec![
            fm(
                "anthropic.claude-opus-5-5",
                "Claude Opus 5.5",
                "Anthropic",
                &["TEXT"],
                &["INFERENCE_PROFILE"],
            ),
            fm(
                "moonshotai.kimi-k2.5",
                "Kimi K2.5",
                "Moonshot AI",
                &["TEXT"],
                &["ON_DEMAND"],
            ),
            fm(
                "xai.grok-4.7",
                "Grok 4.7",
                "xAI",
                &["TEXT"],
                &["INFERENCE_PROFILE"],
            ),
        ];
        let profiles = vec![
            profile("us.anthropic.claude-opus-5-5", "anthropic.claude-opus-5-5"),
            profile(
                "global.anthropic.claude-opus-5-5",
                "anthropic.claude-opus-5-5",
            ),
            profile("us.xai.grok-4.7", "xai.grok-4.7"),
            // An embedding model's profile: its base isn't a TEXT model.
            profile("us.cohere.embed-v4:0", "cohere.embed-v4:0"),
        ];
        let ids: Vec<String> = build_catalog(&fms, &profiles)
            .into_iter()
            .map(|m| m.id)
            .collect();
        assert_eq!(
            ids,
            [
                "global.anthropic.claude-opus-5-5",
                "us.anthropic.claude-opus-5-5",
                "moonshotai.kimi-k2.5",
                "us.xai.grok-4.7",
            ]
        );
    }

    #[test]
    fn profile_only_models_are_not_offered_by_bare_id() {
        // Calling `anthropic.claude-opus-5-5` directly fails with "on-demand
        // throughput isn't supported", so only its profiles are listed.
        let fms = vec![fm(
            "anthropic.claude-opus-5-5",
            "Opus",
            "Anthropic",
            &["TEXT"],
            &["INFERENCE_PROFILE"],
        )];
        assert!(build_catalog(&fms, &[]).is_empty());
    }

    #[test]
    fn retired_models_and_inactive_profiles_are_hidden() {
        let mut legacy = fm(
            "meta.llama3-8b",
            "Llama 3 8B",
            "Meta",
            &["TEXT"],
            &["ON_DEMAND"],
        );
        legacy.model_lifecycle = Some(ModelLifecycle {
            status: Some("LEGACY".into()),
        });
        let live = fm(
            "amazon.nova-pro-v1:0",
            "Nova Pro",
            "Amazon",
            &["TEXT"],
            &["ON_DEMAND"],
        );
        let mut paused = profile("us.amazon.nova-pro-v1:0", "amazon.nova-pro-v1:0");
        paused.status = Some("INACTIVE".into());
        let ids: Vec<String> = build_catalog(&[legacy, live], &[paused])
            .into_iter()
            .map(|m| m.id)
            .collect();
        assert_eq!(ids, ["amazon.nova-pro-v1:0"]);
    }

    #[test]
    fn non_chat_text_models_are_hidden() {
        let mut sonic = fm(
            "amazon.nova-2-sonic-v1:0",
            "Nova 2 Sonic",
            "Amazon",
            &["SPEECH", "TEXT"],
            &["ON_DEMAND"],
        );
        sonic.input_modalities = vec!["SPEECH".into()];
        let rerank = fm(
            "cohere.rerank-v3-5:0",
            "Rerank 3.5",
            "Cohere",
            &["TEXT"],
            &["ON_DEMAND"],
        );
        let pegasus = fm(
            "twelvelabs.pegasus-1-2-v1:0",
            "Pegasus",
            "TwelveLabs",
            &["TEXT"],
            &["ON_DEMAND"],
        );
        let mut voxtral = fm(
            "mistral.voxtral-small-24b-2507",
            "Voxtral Small",
            "Mistral AI",
            &["TEXT"],
            &["ON_DEMAND"],
        );
        voxtral.input_modalities = vec!["SPEECH".into(), "TEXT".into()];
        let ids: Vec<String> = build_catalog(&[sonic, rerank, pegasus, voxtral], &[])
            .into_iter()
            .map(|m| m.id)
            .collect();
        // Voxtral takes text as well as audio, so it can chat.
        assert_eq!(ids, ["mistral.voxtral-small-24b-2507"]);
    }

    #[test]
    fn query_terms_all_have_to_match() {
        let m = BedrockModel {
            id: "us.anthropic.claude-opus-5-5".into(),
            name: "Claude Opus 5.5".into(),
            vendor: "Anthropic".into(),
        };
        assert!(m.matches(""));
        assert!(m.matches("opus"));
        assert!(m.matches("OPUS 5.5 us."));
        assert!(m.matches("anthropic claude"));
        assert!(!m.matches("opus global"));
        assert!(!m.matches("grok"));
    }

    #[test]
    fn parses_the_real_response_shapes() {
        let fms: FoundationModelsResponse = serde_json::from_value(serde_json::json!({
            "modelSummaries": [{
                "modelId": "openai.gpt-6.1-sol",
                "modelName": "GPT-6.1 Sol",
                "providerName": "OpenAI",
                "inputModalities": ["TEXT", "IMAGE"],
                "outputModalities": ["TEXT"],
                "inferenceTypesSupported": ["INFERENCE_PROFILE"],
                "modelLifecycle": {"status": "ACTIVE"},
                "responseStreamingSupported": true
            }]
        }))
        .unwrap();
        let profiles: InferenceProfilesResponse = serde_json::from_value(serde_json::json!({
            "inferenceProfileSummaries": [{
                "inferenceProfileId": "us.openai.gpt-6.1-sol",
                "inferenceProfileName": "US GPT-6.1 Sol",
                "status": "ACTIVE",
                "type": "SYSTEM_DEFINED",
                "models": [
                    {"modelArn": "arn:aws:bedrock:us-east-1::foundation-model/openai.gpt-6.1-sol"},
                    {"modelArn": "arn:aws:bedrock:us-west-2::foundation-model/openai.gpt-6.1-sol"}
                ]
            }],
            "nextToken": null
        }))
        .unwrap();
        assert!(profiles.next_token.is_none());
        let catalog = build_catalog(&fms.model_summaries, &profiles.inference_profile_summaries);
        assert_eq!(
            catalog,
            [BedrockModel {
                id: "us.openai.gpt-6.1-sol".into(),
                name: "GPT-6.1 Sol".into(),
                vendor: "OpenAI".into(),
            }]
        );
    }

    #[test]
    fn pagination_tokens_are_percent_encoded() {
        assert_eq!(urlencode_query_value("a+b/c=="), "a%2Bb%2Fc%3D%3D");
    }
}
