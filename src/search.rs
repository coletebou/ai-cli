use anyhow::{Context, Result};
use genai::chat::Tool;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;

/// A single search result returned from the search provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub content: String,
}

// =============================================================================
// HTTP plumbing shared by the search backends (and `ai models`)
// =============================================================================

/// How long to wait for DNS + TCP + TLS before giving up on a host.
///
/// Without this, a DNS-blocked or blackholed host (e.g. a corporate DNS
/// filter redirecting the API domain to a block page) hangs for the OS
/// default, which can be ~90s, while the model sits waiting on a tool call.
pub(crate) const HTTP_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Overall per-request budget for a web search. Gemini grounding runs a
/// Google search and synthesizes an answer server-side, so this is generous;
/// Tavily usually answers in 1-3s.
const SEARCH_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Error bodies are embedded in errors that reach both the user and the model
/// (as the tool result). HTML error pages from proxies can be tens of KB, so
/// cap what we keep.
const MAX_ERROR_BODY_CHARS: usize = 500;

/// Build a `reqwest::Client` with a connect timeout and an overall request
/// timeout. `reqwest::Client::new()` has neither, so a stalled or blocked
/// host hangs until the OS gives up.
pub(crate) fn http_client(connect_timeout: Duration, timeout: Duration) -> reqwest::Client {
    http_client_builder(connect_timeout, timeout)
        .build()
        // OK_BECAUSE: setting timeouts can't make the build fail; the only
        // realistic failure is the TLS backend failing to initialise, where
        // `reqwest::Client::new()` (used before this change) panics too.
        .expect("failed to initialise HTTP client (TLS backend)")
}

fn http_client_builder(connect_timeout: Duration, timeout: Duration) -> reqwest::ClientBuilder {
    reqwest::Client::builder()
        .connect_timeout(connect_timeout)
        .timeout(timeout)
}

/// Condense an HTTP error body for inclusion in an error message: collapse
/// whitespace (HTML and pretty-printed JSON are mostly indentation) and cap
/// the length on a char boundary.
fn summarize_error_body(body: &str) -> String {
    let collapsed = body.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        return "(empty body)".to_string();
    }
    match collapsed.char_indices().nth(MAX_ERROR_BODY_CHARS) {
        // `idx` is the byte offset of the first char past the limit, so it is
        // always a char boundary.
        Some((idx, _)) => format!("{}… (truncated)", &collapsed[..idx]),
        None => collapsed,
    }
}

// =============================================================================
// SearchProvider enum — unifies Tavily and Gemini backends
// =============================================================================

pub enum SearchProvider {
    Tavily(TavilyClient),
    Gemini(GeminiSearchClient),
    /// Canned backend so `SearchEngine`'s fallback logic can be tested
    /// without network access.
    #[cfg(test)]
    Mock(MockProvider),
}

impl SearchProvider {
    /// Human-readable label for display (e.g., "tavily" or "gemini").
    pub fn name(&self) -> &'static str {
        match self {
            SearchProvider::Tavily(_) => "tavily",
            SearchProvider::Gemini(_) => "gemini",
            #[cfg(test)]
            SearchProvider::Mock(m) => m.name,
        }
    }

    pub async fn search(&self, query: &str) -> Result<Vec<SearchResult>> {
        match self {
            SearchProvider::Tavily(c) => c.search(query).await,
            SearchProvider::Gemini(c) => c.search(query).await,
            #[cfg(test)]
            SearchProvider::Mock(m) => m.search(query),
        }
    }
}

/// Test double behind `SearchProvider::Mock`: returns a canned outcome and
/// counts how often it was called.
#[cfg(test)]
pub struct MockProvider {
    name: &'static str,
    outcome: MockOutcome,
    calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

#[cfg(test)]
enum MockOutcome {
    Results(Vec<SearchResult>),
    /// Built fresh on every call as `anyhow!(cause).context(context)`,
    /// mirroring how the real backends wrap reqwest errors.
    Error {
        context: &'static str,
        cause: &'static str,
    },
    /// A raw Gemini `generateContent` body, run through the real parser.
    GeminiBody(Value),
}

#[cfg(test)]
impl MockProvider {
    fn search(&self, query: &str) -> Result<Vec<SearchResult>> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        match &self.outcome {
            MockOutcome::Results(results) => Ok(results.clone()),
            MockOutcome::Error { context, cause } => Err(anyhow::anyhow!(*cause).context(*context)),
            MockOutcome::GeminiBody(body) => parse_gemini_grounding(body, query),
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
    ///
    /// Errors are flattened with `{:#}` so the whole context chain survives
    /// into the message: callers render it with `to_string()`, which would
    /// otherwise show only the outermost context ("Tavily API request
    /// failed") and drop the actual cause (timeout, DNS failure, ...).
    pub async fn search(&self, query: &str) -> Result<(&'static str, Vec<SearchResult>)> {
        match self.primary.search(query).await {
            Ok(results) => Ok((self.primary.name(), results)),
            Err(primary_err) => {
                let Some(fallback) = self.fallback.as_ref() else {
                    anyhow::bail!("{} search failed: {:#}", self.primary.name(), primary_err);
                };
                match fallback.search(query).await {
                    Ok(results) => Ok((fallback.name(), results)),
                    Err(fallback_err) => anyhow::bail!(
                        "primary ({}) failed: {:#}; fallback ({}) also failed: {:#}",
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
    #[serde(default)]
    results: Vec<TavilyResult>,
}

// Every field is optional (and `null`-tolerant, which `#[serde(default)]` on a
// plain `String` is not) so one odd result can't fail the whole search.
#[derive(Debug, Deserialize)]
struct TavilyResult {
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    content: Option<String>,
}

/// Convert Tavily's results, dropping any without a URL (nothing to cite, and
/// `format_results_for_model` treats a URL-less entry as a synthesized answer).
fn convert_tavily_results(results: Vec<TavilyResult>) -> Vec<SearchResult> {
    results
        .into_iter()
        .filter_map(|r| {
            let url = r.url.filter(|u| !u.is_empty())?;
            Some(SearchResult {
                title: r.title.unwrap_or_default(),
                url,
                content: r.content.unwrap_or_default(),
            })
        })
        .collect()
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
                http: http_client(HTTP_CONNECT_TIMEOUT, SEARCH_REQUEST_TIMEOUT),
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
            anyhow::bail!(
                "Tavily API returned {}: {}",
                status,
                summarize_error_body(&body)
            );
        }

        let tavily_response: TavilyResponse = response
            .json()
            .await
            .context("Failed to parse Tavily response")?;

        Ok(convert_tavily_results(tavily_response.results))
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
            .or_else(|| std::env::var(fallback_env).ok().filter(|k| !k.is_empty()))?;

        Some(GeminiSearchClient {
            api_key: key,
            model: model.to_string(),
            http: http_client(HTTP_CONNECT_TIMEOUT, SEARCH_REQUEST_TIMEOUT),
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
            anyhow::bail!(
                "Gemini API returned {}: {}",
                status,
                summarize_error_body(&body)
            );
        }

        let json: Value = response
            .json()
            .await
            .context("Failed to parse Gemini response")?;

        parse_gemini_grounding(&json, query)
    }
}

/// Extract grounded answer + sources from a Gemini response.
///
/// Produces a single "synthesized answer" SearchResult first (no URL),
/// followed by one SearchResult per grounding chunk (title + URL from
/// the chunks Gemini cited).
///
/// A 200 response can still carry nothing usable: the prompt was blocked
/// (`promptFeedback.blockReason`, no candidates), generation stopped early
/// (`finishReason` SAFETY / RECITATION / ... with no text), or Gemini simply
/// returned neither text nor sources. Those are errors rather than an empty
/// success, so `SearchEngine` falls back to the other provider instead of
/// telling the model "No search results found".
fn parse_gemini_grounding(json: &Value, query: &str) -> Result<Vec<SearchResult>> {
    if let Some(reason) = json
        .get("promptFeedback")
        .and_then(|f| f.get("blockReason"))
        .and_then(|r| r.as_str())
    {
        anyhow::bail!("Gemini blocked the search prompt (blockReason: {})", reason);
    }

    let Some(candidate) = json
        .get("candidates")
        .and_then(|c| c.as_array())
        .and_then(|c| c.first())
    else {
        anyhow::bail!("Gemini returned no candidates for the search");
    };

    let mut results = Vec::new();

    // Extract synthesized answer text (concatenation of all text parts)
    let answer_text: String = candidate
        .get("content")
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

    // Any finish reason other than STOP with no text means generation was
    // cut off (SAFETY, RECITATION, BLOCKLIST, PROHIBITED_CONTENT, ...). A
    // bare source list without the answer isn't worth passing on.
    let has_answer = !answer_text.trim().is_empty();
    let finish_reason = candidate.get("finishReason").and_then(|r| r.as_str());
    if !has_answer {
        if let Some(reason) = finish_reason.filter(|r| *r != "STOP") {
            anyhow::bail!(
                "Gemini search stopped without an answer (finishReason: {})",
                reason
            );
        }
    }

    if has_answer {
        results.push(SearchResult {
            title: format!("Gemini grounded answer for: {}", query),
            url: String::new(),
            content: answer_text,
        });
    }

    // Extract source URLs from groundingMetadata.groundingChunks[].web
    if let Some(chunks) = candidate
        .get("groundingMetadata")
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

    if results.is_empty() {
        anyhow::bail!("Gemini search returned neither an answer nor any sources");
    }

    Ok(results)
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
    // Only entries with a URL are citable sources, so only they get a number.
    // Counting the synthesized answer too would start Gemini's sources at [2].
    let mut source_num = 0;
    for r in results {
        if r.url.is_empty() {
            // Synthesized answer (Gemini) — render without URL prefix
            out.push_str(&format!("{}\n\n", r.content));
            continue;
        }
        source_num += 1;
        if r.content.is_empty() {
            // Source-only (Gemini citations) — render compact line
            out.push_str(&format!("[{}] {} — {}\n", source_num, r.title, r.url));
        } else {
            // Full result (Tavily) — title, URL, content
            out.push_str(&format!(
                "[{}] {}\nURL: {}\n{}\n\n",
                source_num, r.title, r.url, r.content
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn mock(name: &'static str, outcome: MockOutcome) -> (SearchProvider, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = SearchProvider::Mock(MockProvider {
            name,
            outcome,
            calls: calls.clone(),
        });
        (provider, calls)
    }

    fn ok_provider(name: &'static str, urls: &[&str]) -> (SearchProvider, Arc<AtomicUsize>) {
        let results = urls
            .iter()
            .map(|u| SearchResult {
                title: format!("title for {u}"),
                url: u.to_string(),
                content: "content".to_string(),
            })
            .collect();
        mock(name, MockOutcome::Results(results))
    }

    fn failing_provider(
        name: &'static str,
        context: &'static str,
        cause: &'static str,
    ) -> (SearchProvider, Arc<AtomicUsize>) {
        mock(name, MockOutcome::Error { context, cause })
    }

    // ---- SearchEngine fallback ordering ----

    #[tokio::test]
    async fn engine_uses_primary_when_it_succeeds() {
        let (primary, primary_calls) = ok_provider("primary", &["https://a.example"]);
        let (fallback, fallback_calls) = ok_provider("fallback", &["https://b.example"]);
        let engine = SearchEngine::new(primary, Some(fallback));

        let (name, results) = engine.search("q").await.unwrap();

        assert_eq!(name, "primary");
        assert_eq!(results[0].url, "https://a.example");
        assert_eq!(primary_calls.load(Ordering::SeqCst), 1);
        assert_eq!(fallback_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn engine_falls_back_when_primary_fails() {
        let (primary, primary_calls) =
            failing_provider("primary", "Gemini API request failed", "timed out");
        let (fallback, fallback_calls) = ok_provider("fallback", &["https://b.example"]);
        let engine = SearchEngine::new(primary, Some(fallback));

        let (name, results) = engine.search("q").await.unwrap();

        assert_eq!(name, "fallback");
        assert_eq!(results[0].url, "https://b.example");
        assert_eq!(primary_calls.load(Ordering::SeqCst), 1);
        assert_eq!(fallback_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn engine_error_without_fallback_keeps_root_cause() {
        let (primary, _) = failing_provider(
            "tavily",
            "Tavily API request failed",
            "dns error: failed to lookup address",
        );
        let engine = SearchEngine::new(primary, None);

        let err = engine.search("q").await.unwrap_err();
        // query.rs renders with to_string(), so the cause must be in the
        // top-level message, not just further down the chain.
        let msg = err.to_string();

        assert!(msg.contains("tavily"), "{msg}");
        assert!(msg.contains("Tavily API request failed"), "{msg}");
        assert!(msg.contains("dns error"), "{msg}");
    }

    #[tokio::test]
    async fn engine_error_when_both_fail_keeps_both_root_causes() {
        let (primary, _) =
            failing_provider("gemini", "Gemini API request failed", "operation timed out");
        let (fallback, _) =
            failing_provider("tavily", "Tavily API request failed", "connection refused");
        let engine = SearchEngine::new(primary, Some(fallback));

        let msg = engine.search("q").await.unwrap_err().to_string();

        assert!(msg.contains("primary (gemini) failed"), "{msg}");
        assert!(msg.contains("operation timed out"), "{msg}");
        assert!(msg.contains("fallback (tavily) also failed"), "{msg}");
        assert!(msg.contains("connection refused"), "{msg}");
    }

    #[tokio::test]
    async fn engine_falls_back_on_blocked_or_empty_gemini_response() {
        // Run real Gemini bodies through the real parser: a 200 with nothing
        // usable must reach the fallback instead of an empty success.
        let bodies = [
            json!({ "promptFeedback": { "blockReason": "SAFETY" } }),
            json!({ "candidates": [] }),
            json!({ "candidates": [{ "finishReason": "RECITATION" }] }),
            json!({ "candidates": [{ "content": { "parts": [] }, "finishReason": "STOP" }] }),
        ];
        for body in bodies {
            let (primary, _) = mock("gemini", MockOutcome::GeminiBody(body.clone()));
            let (fallback, fallback_calls) = ok_provider("tavily", &["https://b.example"]);
            let engine = SearchEngine::new(primary, Some(fallback));

            let (name, results) = engine.search("q").await.unwrap();

            assert_eq!(name, "tavily", "no fallback for {body}");
            assert_eq!(results.len(), 1);
            assert_eq!(fallback_calls.load(Ordering::SeqCst), 1);
        }
    }

    #[tokio::test]
    async fn engine_keeps_good_gemini_response() {
        let body = json!({
            "candidates": [{
                "content": { "parts": [{ "text": "An answer." }] },
                "finishReason": "STOP"
            }]
        });
        let (primary, _) = mock("gemini", MockOutcome::GeminiBody(body));
        let (fallback, fallback_calls) = ok_provider("tavily", &["https://b.example"]);
        let engine = SearchEngine::new(primary, Some(fallback));

        let (name, results) = engine.search("q").await.unwrap();

        assert_eq!(name, "gemini");
        assert_eq!(results[0].content, "An answer.");
        assert_eq!(fallback_calls.load(Ordering::SeqCst), 0);
    }

    // ---- parse_gemini_grounding ----

    #[test]
    fn gemini_answer_and_sources() {
        let body = json!({
            "candidates": [{
                "content": { "parts": [{ "text": "Rust 1.90 " }, { "text": "was released." }] },
                "finishReason": "STOP",
                "groundingMetadata": {
                    "groundingChunks": [
                        { "web": { "uri": "https://blog.rust-lang.org", "title": "Rust Blog" } },
                        { "web": { "uri": "", "title": "no url, skipped" } },
                        { "retrievedContext": {} },
                        { "web": { "uri": "https://example.com/news" } }
                    ]
                }
            }]
        });

        let results = parse_gemini_grounding(&body, "rust release").unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].url, "");
        assert_eq!(results[0].content, "Rust 1.90 was released.");
        assert!(results[0].title.contains("rust release"));
        assert_eq!(results[1].url, "https://blog.rust-lang.org");
        assert_eq!(results[1].title, "Rust Blog");
        assert_eq!(results[2].url, "https://example.com/news");
        assert_eq!(results[2].title, "");
    }

    #[test]
    fn gemini_answer_without_sources_is_ok() {
        let body = json!({
            "candidates": [{
                "content": { "parts": [{ "text": "An answer." }] },
                "finishReason": "STOP"
            }]
        });
        let results = parse_gemini_grounding(&body, "q").unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn gemini_blocked_prompt_is_error() {
        let body = json!({
            "promptFeedback": { "blockReason": "SAFETY" }
        });
        let err = parse_gemini_grounding(&body, "q").unwrap_err();
        assert!(err.to_string().contains("SAFETY"), "{err}");
    }

    #[test]
    fn gemini_empty_or_missing_candidates_is_error() {
        assert!(parse_gemini_grounding(&json!({ "candidates": [] }), "q").is_err());
        assert!(parse_gemini_grounding(&json!({}), "q").is_err());
    }

    #[test]
    fn gemini_safety_or_recitation_without_text_is_error() {
        for reason in ["SAFETY", "RECITATION"] {
            let body = json!({
                "candidates": [{
                    "finishReason": reason,
                    "groundingMetadata": {
                        "groundingChunks": [{ "web": { "uri": "https://x.example" } }]
                    }
                }]
            });
            let err = parse_gemini_grounding(&body, "q").unwrap_err();
            assert!(err.to_string().contains(reason), "{err}");
        }
    }

    #[test]
    fn gemini_no_text_and_no_sources_is_error() {
        let body = json!({
            "candidates": [{ "content": { "parts": [] }, "finishReason": "STOP" }]
        });
        assert!(parse_gemini_grounding(&body, "q").is_err());
    }

    #[test]
    fn gemini_sources_without_text_are_kept_on_normal_stop() {
        let body = json!({
            "candidates": [{
                "finishReason": "STOP",
                "groundingMetadata": {
                    "groundingChunks": [{ "web": { "uri": "https://x.example", "title": "X" } }]
                }
            }]
        });
        let results = parse_gemini_grounding(&body, "q").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].url, "https://x.example");
    }

    // ---- format_results_for_model ----

    fn result(title: &str, url: &str, content: &str) -> SearchResult {
        SearchResult {
            title: title.to_string(),
            url: url.to_string(),
            content: content.to_string(),
        }
    }

    #[test]
    fn format_numbers_gemini_sources_from_one() {
        let results = vec![
            result("answer", "", "The synthesized answer."),
            result("First", "https://one.example", ""),
            result("Second", "https://two.example", ""),
        ];
        let out = format_results_for_model(&results);

        assert!(out.contains("The synthesized answer."));
        assert!(out.contains("[1] First — https://one.example"), "{out}");
        assert!(out.contains("[2] Second — https://two.example"), "{out}");
        assert!(!out.contains("[3]"), "{out}");
    }

    #[test]
    fn format_numbers_tavily_results_from_one() {
        let results = vec![
            result("First", "https://one.example", "body one"),
            result("Second", "https://two.example", "body two"),
        ];
        let out = format_results_for_model(&results);

        assert!(
            out.contains("[1] First\nURL: https://one.example\nbody one"),
            "{out}"
        );
        assert!(
            out.contains("[2] Second\nURL: https://two.example\nbody two"),
            "{out}"
        );
    }

    #[test]
    fn format_empty_results() {
        assert_eq!(
            format_results_for_model(&[]),
            "No search results found for this query."
        );
    }

    // ---- Tavily result tolerance ----

    #[test]
    fn tavily_tolerates_missing_and_null_fields() {
        let body = json!({
            "results": [
                { "title": "Full", "url": "https://full.example", "content": "c" },
                { "title": null, "url": "https://null-title.example", "content": null },
                { "url": "https://missing.example" },
                { "title": "no url", "content": "dropped" }
            ]
        });
        let parsed: TavilyResponse = serde_json::from_value(body).unwrap();
        let results = convert_tavily_results(parsed.results);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].title, "Full");
        assert_eq!(results[1].title, "");
        assert_eq!(results[1].content, "");
        assert_eq!(results[2].url, "https://missing.example");
    }

    #[test]
    fn tavily_missing_results_array_is_empty() {
        let parsed: TavilyResponse = serde_json::from_value(json!({})).unwrap();
        assert!(parsed.results.is_empty());
    }

    // ---- error body summarizing ----

    #[test]
    fn error_body_short_is_kept_and_whitespace_collapsed() {
        assert_eq!(
            summarize_error_body("{\n  \"error\":   \"bad key\"\n}"),
            "{ \"error\": \"bad key\" }"
        );
        assert_eq!(summarize_error_body("  \n "), "(empty body)");
    }

    #[test]
    fn error_body_long_is_truncated_on_char_boundary() {
        // Multi-byte chars everywhere: a byte slice at 500 would panic.
        let body = "é".repeat(2000);
        let out = summarize_error_body(&body);

        assert!(out.ends_with("… (truncated)"));
        let kept = out.trim_end_matches("… (truncated)");
        assert_eq!(kept.chars().count(), MAX_ERROR_BODY_CHARS);
    }

    // ---- HTTP client timeouts ----

    #[tokio::test]
    async fn http_client_times_out_on_stalled_server() {
        // A local server that accepts the connection and never answers,
        // standing in for a blackholed/filtered API host.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (_socket, _) = listener.accept().await.unwrap();
            // Hold the socket open well past the client's timeout.
            tokio::time::sleep(Duration::from_secs(10)).await;
        });

        // `no_proxy()` so an HTTP(S)_PROXY in the environment can't turn the
        // stall into a quick proxy error; otherwise this is `http_client`.
        let client = http_client_builder(Duration::from_secs(2), Duration::from_millis(300))
            .no_proxy()
            .build()
            .unwrap();
        let start = std::time::Instant::now();
        let err = client
            .get(format!("http://{addr}/"))
            .send()
            .await
            .unwrap_err();

        assert!(err.is_timeout(), "{err:?}");
        assert!(start.elapsed() < Duration::from_secs(5));
        server.abort();
    }
}
