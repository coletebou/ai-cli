use anyhow::{Context, Result};
use genai::chat::Tool;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// A single search result returned from the search provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub content: String,
}

// =============================================================================
// SearchProvider enum — unifies Tavily and Gemini backends
// =============================================================================

pub enum SearchProvider {
    Tavily(TavilyClient),
    Gemini(GeminiSearchClient),
}

impl SearchProvider {
    /// Human-readable label for display (e.g., "tavily" or "gemini").
    pub fn name(&self) -> &'static str {
        match self {
            SearchProvider::Tavily(_) => "tavily",
            SearchProvider::Gemini(_) => "gemini",
        }
    }

    pub async fn search(&self, query: &str) -> Result<Vec<SearchResult>> {
        match self {
            SearchProvider::Tavily(c) => c.search(query).await,
            SearchProvider::Gemini(c) => c.search(query).await,
        }
    }
}

// =============================================================================
// SearchEngine — primary + optional fallback
// =============================================================================

/// A search engine that tries a primary provider and falls back to a secondary
/// provider if the primary fails.
pub struct SearchEngine {
    primary: SearchProvider,
    fallback: Option<SearchProvider>,
}

impl SearchEngine {
    pub fn new(primary: SearchProvider, fallback: Option<SearchProvider>) -> Self {
        Self { primary, fallback }
    }

    /// Execute a search using the primary provider, falling back to the secondary
    /// provider on any error. Returns `(provider_name_used, results)`.
    pub async fn search(&self, query: &str) -> Result<(&'static str, Vec<SearchResult>)> {
        match self.primary.search(query).await {
            Ok(results) => Ok((self.primary.name(), results)),
            Err(primary_err) => {
                let Some(fallback) = self.fallback.as_ref() else {
                    return Err(primary_err);
                };
                match fallback.search(query).await {
                    Ok(results) => Ok((fallback.name(), results)),
                    Err(fallback_err) => anyhow::bail!(
                        "primary ({}) failed: {}; fallback ({}) also failed: {}",
                        self.primary.name(),
                        primary_err,
                        fallback.name(),
                        fallback_err
                    ),
                }
            }
        }
    }
}

// =============================================================================
// Tavily backend
// =============================================================================

#[derive(Debug, Serialize)]
struct TavilyRequest<'a> {
    api_key: &'a str,
    query: &'a str,
    search_depth: &'a str,
    max_results: usize,
    include_answer: bool,
}

#[derive(Debug, Deserialize)]
struct TavilyResponse {
    results: Vec<TavilyResult>,
}

#[derive(Debug, Deserialize)]
struct TavilyResult {
    title: String,
    url: String,
    content: String,
}

pub struct TavilyClient {
    api_key: String,
    max_results: usize,
    http: reqwest::Client,
}

impl TavilyClient {
    pub fn from_env(env_var: &str, max_results: usize) -> Option<Self> {
        std::env::var(env_var)
            .ok()
            .filter(|k| !k.is_empty())
            .map(|key| TavilyClient {
                api_key: key,
                max_results,
                http: reqwest::Client::new(),
            })
    }

    pub async fn search(&self, query: &str) -> Result<Vec<SearchResult>> {
        let request = TavilyRequest {
            api_key: &self.api_key,
            query,
            search_depth: "basic",
            max_results: self.max_results,
            include_answer: false,
        };

        let response = self
            .http
            .post("https://api.tavily.com/search")
            .json(&request)
            .send()
            .await
            .context("Tavily API request failed")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Tavily API returned {}: {}", status, body);
        }

        let tavily_response: TavilyResponse = response
            .json()
            .await
            .context("Failed to parse Tavily response")?;

        Ok(tavily_response
            .results
            .into_iter()
            .map(|r| SearchResult {
                title: r.title,
                url: r.url,
                content: r.content,
            })
            .collect())
    }
}

// =============================================================================
// Gemini backend — uses Google Search grounding via Gemini API
// =============================================================================

/// A client that uses Gemini with Google Search grounding as a web search backend.
///
/// Flow: we send the search query to Gemini with the `google_search` tool enabled.
/// Gemini performs the search internally, synthesizes an answer, and returns
/// `groundingMetadata` containing the URLs/titles it cited. We package both the
/// synthesized answer and the source list as a single SearchResult stream.
pub struct GeminiSearchClient {
    api_key: String,
    model: String,
    http: reqwest::Client,
}

impl GeminiSearchClient {
    pub fn from_env(env_var: &str, fallback_env: &str, model: &str) -> Option<Self> {
        let key = std::env::var(env_var)
            .ok()
            .filter(|k| !k.is_empty())
            .or_else(|| {
                std::env::var(fallback_env)
                    .ok()
                    .filter(|k| !k.is_empty())
            })?;

        Some(GeminiSearchClient {
            api_key: key,
            model: model.to_string(),
            http: reqwest::Client::new(),
        })
    }

    pub async fn search(&self, query: &str) -> Result<Vec<SearchResult>> {
        // Use x-goog-api-key header rather than ?key=... in the URL so the key
        // doesn't leak into HTTP-layer debug logs, proxy access logs, or
        // panic backtraces that capture local variable state.
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
            self.model
        );

        let body = json!({
            "contents": [{
                "role": "user",
                "parts": [{"text": query}]
            }],
            "tools": [{"google_search": {}}],
            "generationConfig": {
                "temperature": 0.2,
            }
        });

        let response = self
            .http
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .json(&body)
            .send()
            .await
            .context("Gemini API request failed")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Gemini API returned {}: {}", status, body);
        }

        let json: Value = response
            .json()
            .await
            .context("Failed to parse Gemini response")?;

        Ok(parse_gemini_grounding(&json, query))
    }
}

/// Extract grounded answer + sources from a Gemini response.
///
/// Produces a single "synthesized answer" SearchResult first (no URL),
/// followed by one SearchResult per grounding chunk (title + URL from
/// the chunks Gemini cited).
fn parse_gemini_grounding(json: &Value, query: &str) -> Vec<SearchResult> {
    let mut results = Vec::new();

    let candidate = json
        .get("candidates")
        .and_then(|c| c.get(0));

    // Extract synthesized answer text (concatenation of all text parts)
    let answer_text: String = candidate
        .and_then(|c| c.get("content"))
        .and_then(|c| c.get("parts"))
        .and_then(|p| p.as_array())
        .map(|parts| {
            parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default();

    if !answer_text.is_empty() {
        results.push(SearchResult {
            title: format!("Gemini grounded answer for: {}", query),
            url: String::new(),
            content: answer_text,
        });
    }

    // Extract source URLs from groundingMetadata.groundingChunks[].web
    if let Some(chunks) = candidate
        .and_then(|c| c.get("groundingMetadata"))
        .and_then(|g| g.get("groundingChunks"))
        .and_then(|c| c.as_array())
    {
        for chunk in chunks {
            let web = match chunk.get("web") {
                Some(w) => w,
                None => continue,
            };
            let title = web
                .get("title")
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string();
            let url = web
                .get("uri")
                .and_then(|u| u.as_str())
                .unwrap_or("")
                .to_string();
            if url.is_empty() {
                continue;
            }
            results.push(SearchResult {
                title,
                url,
                content: String::new(),
            });
        }
    }

    results
}

// =============================================================================
// Tool definition & formatting (shared across providers)
// =============================================================================

/// The JSON schema + description for the `web_search` tool we expose to models.
pub fn web_search_tool() -> Tool {
    Tool::new("web_search")
        .with_description(
            "Search the web for current information. Use this tool when you need up-to-date \
             information, current events, recent news, documentation, or information that might \
             have changed since your training data. Be specific and focused in your query. \
             After receiving results, cite sources by URL when relevant.",
        )
        .with_schema(json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A specific, focused search query"
                }
            },
            "required": ["query"]
        }))
}

/// Format search results into a string to feed back to the model as a tool response.
pub fn format_results_for_model(results: &[SearchResult]) -> String {
    if results.is_empty() {
        return "No search results found for this query.".to_string();
    }
    let mut out = String::from("Search results:\n\n");
    for (i, r) in results.iter().enumerate() {
        if r.url.is_empty() {
            // Synthesized answer (Gemini) — render without URL prefix
            out.push_str(&format!("{}\n\n", r.content));
        } else if r.content.is_empty() {
            // Source-only (Gemini citations) — render compact line
            out.push_str(&format!("[{}] {} — {}\n", i + 1, r.title, r.url));
        } else {
            // Full result (Tavily) — title, URL, content
            out.push_str(&format!(
                "[{}] {}\nURL: {}\n{}\n\n",
                i + 1,
                r.title,
                r.url,
                r.content
            ));
        }
    }
    out
}
