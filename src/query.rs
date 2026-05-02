use crate::providers::{Provider, ProviderManager, ProviderResponse};
use crate::search::{format_results_for_model, SearchEngine};
use futures::future::join_all;
use futures::StreamExt as FuturesStreamExt;
use genai::chat::{
    ChatMessage, ChatOptions, ChatStreamEvent, ReasoningEffort, Tool, ToolCall, ToolResponse,
};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Start {
        provider: String,
        model: String,
    },
    Chunk {
        provider: String,
        text: String,
    },
    /// Sent when a tool call is about to execute (e.g., a web search starts).
    SearchStart {
        provider: String,
        query: String,
    },
    /// Sent after a tool call completes, with outcome summary.
    SearchEnd {
        provider: String,
        /// Which search backend actually ran ("gemini", "tavily", or "" on error
        /// before either could respond).
        search_provider: String,
        #[allow(dead_code)] // kept for symmetry with SearchStart; available for custom consumers
        query: String,
        result_count: usize,
        error: Option<String>,
    },
    Done {
        provider: String,
        model: String,
        input_tokens: Option<i32>,
        output_tokens: Option<i32>,
        duration_ms: u64,
    },
    Error {
        provider: String,
        model: String,
        error: String,
        duration_ms: u64,
    },
}

/// Parameters for executing a query, bundled to keep function signatures manageable.
#[derive(Clone)]
pub struct QueryContext {
    pub question: String,
    pub system_prompt: Option<String>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub tools: Option<Vec<Tool>>,
    pub search_client: Option<Arc<SearchEngine>>,
    pub max_iterations: usize,
}

impl QueryContext {
    pub fn new(question: impl Into<String>) -> Self {
        Self {
            question: question.into(),
            system_prompt: None,
            reasoning_effort: None,
            tools: None,
            search_client: None,
            max_iterations: 5,
        }
    }

    pub fn with_system_prompt(mut self, system: Option<&str>) -> Self {
        self.system_prompt = system.map(|s| s.to_string());
        self
    }

    pub fn with_reasoning(mut self, reasoning: Option<ReasoningEffort>) -> Self {
        self.reasoning_effort = reasoning;
        self
    }

    pub fn with_search(mut self, search_client: Arc<SearchEngine>, max_iterations: usize) -> Self {
        self.tools = Some(vec![crate::search::web_search_tool()]);
        self.search_client = Some(search_client);
        self.max_iterations = max_iterations;
        self
    }
}

fn build_initial_messages(question: &str, system_prompt: Option<&str>) -> Vec<ChatMessage> {
    let mut messages = Vec::new();
    if let Some(system) = system_prompt {
        messages.push(ChatMessage::system(system));
    }
    messages.push(ChatMessage::user(question));
    messages
}

fn build_options(ctx: &QueryContext) -> Option<ChatOptions> {
    let needs_options = ctx.reasoning_effort.is_some() || ctx.tools.is_some();
    if !needs_options {
        return None;
    }
    let mut options = ChatOptions::default();
    if let Some(r) = ctx.reasoning_effort.clone() {
        options = options.with_reasoning_effort(r);
    }
    if ctx.tools.is_some() {
        options = options.with_capture_tool_calls(true);
    }
    Some(options)
}

/// Execute a tool call and return the text response for feeding back to the model.
async fn execute_tool_call(
    tool_call: &ToolCall,
    ctx: &QueryContext,
    provider_name: &str,
    tx: Option<&mpsc::Sender<StreamEvent>>,
) -> String {
    if tool_call.fn_name != "web_search" {
        return format!("Error: unknown tool '{}'", tool_call.fn_name);
    }

    let query = tool_call
        .fn_arguments
        .get("query")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    if query.is_empty() {
        return "Error: empty search query".to_string();
    }

    let search_client = match &ctx.search_client {
        Some(c) => c,
        None => return "Error: search is not configured".to_string(),
    };

    // Notify listeners that a search is starting
    if let Some(tx) = tx {
        let _ = tx
            .send(StreamEvent::SearchStart {
                provider: provider_name.to_string(),
                query: query.clone(),
            })
            .await;
    }

    match search_client.search(&query).await {
        Ok((search_provider_name, results)) => {
            if let Some(tx) = tx {
                let _ = tx
                    .send(StreamEvent::SearchEnd {
                        provider: provider_name.to_string(),
                        search_provider: search_provider_name.to_string(),
                        query: query.clone(),
                        result_count: results.len(),
                        error: None,
                    })
                    .await;
            }
            let formatted = format_results_for_model(&results);
            format!("(via {} search)\n\n{}", search_provider_name, formatted)
        }
        Err(e) => {
            let err_msg = e.to_string();
            if let Some(tx) = tx {
                let _ = tx
                    .send(StreamEvent::SearchEnd {
                        provider: provider_name.to_string(),
                        search_provider: String::new(),
                        query: query.clone(),
                        result_count: 0,
                        error: Some(err_msg.clone()),
                    })
                    .await;
            }
            format!("Search failed: {}", err_msg)
        }
    }
}

// =============================================================================
// Non-streaming queries
// =============================================================================

pub async fn query_single(
    manager: &ProviderManager,
    provider: &Provider,
    ctx: &QueryContext,
) -> ProviderResponse {
    let start = Instant::now();
    let model = provider.model.clone();

    // Bedrock-hosted models bypass genai (which cannot speak Bedrock's wire
    // protocol) and go through the direct Converse client.
    if let Some(bedrock) = manager.bedrock_for(&model) {
        return query_single_bedrock(manager, provider, ctx, &bedrock, start).await;
    }

    let mut messages = build_initial_messages(&ctx.question, ctx.system_prompt.as_deref());
    let options = build_options(ctx);

    let mut total_input: Option<i32> = None;
    let mut total_output: Option<i32> = None;
    let mut final_text = String::new();

    for iteration in 0..ctx.max_iterations.max(1) {
        let response = match manager
            .exec_chat(
                &provider.model,
                messages.clone(),
                options.as_ref(),
                ctx.tools.clone(),
            )
            .await
        {
            Ok(r) => r,
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                return ProviderResponse::error(
                    provider.clone(),
                    model,
                    format!("{:#}", e),
                    duration_ms,
                );
            }
        };

        // Accumulate usage
        if let Some(pt) = response.usage.prompt_tokens {
            total_input = Some(total_input.unwrap_or(0) + pt);
        }
        if let Some(ct) = response.usage.completion_tokens {
            total_output = Some(total_output.unwrap_or(0) + ct);
        }

        // If this response has a text answer, keep it as the candidate final text
        if let Some(text) = response.first_text() {
            if !text.is_empty() {
                final_text = text.to_string();
            }
        }

        // Check for tool calls; if none, we're done
        let tool_calls: Vec<ToolCall> = response.into_tool_calls();
        if tool_calls.is_empty() {
            break;
        }

        // No search client? Can't execute tools — stop the loop
        if ctx.search_client.is_none() {
            break;
        }

        // Don't execute tool calls on the final iteration: we'd spend search
        // quota but the results would never be fed back (loop exits next).
        if iteration + 1 >= ctx.max_iterations {
            break;
        }

        // Append assistant message with the tool calls
        messages.push(ChatMessage::from(tool_calls.clone()));

        // Execute each tool call and append responses
        for tool_call in &tool_calls {
            let result_text = execute_tool_call(tool_call, ctx, &provider.display_name, None).await;
            let tool_response = ToolResponse::new(tool_call.call_id.clone(), result_text);
            messages.push(ChatMessage::from(tool_response));
        }
    }

    let duration_ms = start.elapsed().as_millis() as u64;
    if final_text.is_empty() {
        return ProviderResponse::error(
            provider.clone(),
            model,
            "provider returned an empty response (likely a wire-protocol mismatch; check base_url)".to_string(),
            duration_ms,
        );
    }
    ProviderResponse::success(
        provider.clone(),
        model,
        final_text,
        total_input,
        total_output,
        duration_ms,
    )
}

/// Non-streaming Bedrock branch for `query_single`. Bedrock doesn't support
/// genai-style tool-use in this codepath yet, so any tools in `ctx` are
/// ignored (and a warning is not emitted — query_single keeps the same
/// silent-skip behavior for missing search capability).
async fn query_single_bedrock(
    manager: &ProviderManager,
    provider: &Provider,
    ctx: &QueryContext,
    bedrock: &crate::bedrock::BedrockClient,
    start: Instant,
) -> ProviderResponse {
    let messages = build_initial_messages(&ctx.question, ctx.system_prompt.as_deref());
    match manager
        .exec_bedrock_converse(bedrock, &provider.model, &messages, ctx.system_prompt.as_deref())
        .await
    {
        Ok(outcome) => {
            let duration_ms = start.elapsed().as_millis() as u64;
            if outcome.text.is_empty() {
                return ProviderResponse::error(
                    provider.clone(),
                    provider.model.clone(),
                    "Bedrock returned an empty response (no text content)".to_string(),
                    duration_ms,
                );
            }
            ProviderResponse::success(
                provider.clone(),
                provider.model.clone(),
                outcome.text,
                outcome.input_tokens,
                outcome.output_tokens,
                duration_ms,
            )
        }
        Err(e) => {
            let duration_ms = start.elapsed().as_millis() as u64;
            ProviderResponse::error(
                provider.clone(),
                provider.model.clone(),
                format!("{:#}", e),
                duration_ms,
            )
        }
    }
}

pub async fn query_all(
    manager: Arc<ProviderManager>,
    providers: Vec<Provider>,
    ctx: QueryContext,
) -> Vec<ProviderResponse> {
    let futures = providers.into_iter().map(|provider| {
        let manager = Arc::clone(&manager);
        let ctx = ctx.clone();
        async move { query_single(&manager, &provider, &ctx).await }
    });

    join_all(futures).await
}

// =============================================================================
// Streaming queries
// =============================================================================

pub async fn stream_single(
    manager: &ProviderManager,
    provider: &Provider,
    ctx: &QueryContext,
    tx: mpsc::Sender<StreamEvent>,
) {
    let messages = build_initial_messages(&ctx.question, ctx.system_prompt.as_deref());
    // Non-cancellable for single-shot callers; chat mode uses its own token.
    let cancel = CancellationToken::new();
    let _ = stream_with_messages(manager, provider, messages, ctx, tx, &cancel).await;
}

/// Stream a single response from `provider` using the given prebuilt message
/// history. This is the core streaming routine used by both single-shot and
/// multi-turn (chat) modes. The tool-use loop (for search) is executed here.
///
/// Returns the enriched message history including any tool-call assistant
/// messages, tool-response messages, and the final assistant text reply. For
/// single-shot callers the return value can be ignored; chat mode uses it to
/// preserve full tool-use context across turns.
pub async fn stream_with_messages(
    manager: &ProviderManager,
    provider: &Provider,
    mut messages: Vec<ChatMessage>,
    ctx: &QueryContext,
    tx: mpsc::Sender<StreamEvent>,
    cancel: &CancellationToken,
) -> Vec<ChatMessage> {
    let start = Instant::now();
    let provider_name = provider.display_name.clone();
    let model = provider.model.clone();

    // Bedrock-hosted models: use the direct Converse client. We simulate a
    // stream by emitting Start -> Chunk (full text) -> Done so downstream
    // display/TUI code can remain unchanged. Real token-level streaming is
    // a follow-up (requires AWS binary Event Stream framing).
    if let Some(bedrock) = manager.bedrock_for(&model) {
        let _ = tx
            .send(StreamEvent::Start {
                provider: provider_name.clone(),
                model: model.clone(),
            })
            .await;

        let bedrock_result = tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                let duration_ms = start.elapsed().as_millis() as u64;
                let _ = tx
                    .send(StreamEvent::Error {
                        provider: provider_name,
                        model,
                        error: "cancelled".to_string(),
                        duration_ms,
                    })
                    .await;
                return messages;
            }
            r = manager.exec_bedrock_converse(
                &bedrock,
                &model,
                &messages,
                ctx.system_prompt.as_deref(),
            ) => r,
        };

        let duration_ms = start.elapsed().as_millis() as u64;
        match bedrock_result {
            Ok(outcome) if outcome.text.is_empty() => {
                let _ = tx
                    .send(StreamEvent::Error {
                        provider: provider_name,
                        model,
                        error: "Bedrock returned an empty response (no text content)".to_string(),
                        duration_ms,
                    })
                    .await;
            }
            Ok(outcome) => {
                let _ = tx
                    .send(StreamEvent::Chunk {
                        provider: provider_name.clone(),
                        text: outcome.text.clone(),
                    })
                    .await;
                messages.push(ChatMessage::assistant(outcome.text));
                let _ = tx
                    .send(StreamEvent::Done {
                        provider: provider_name,
                        model,
                        input_tokens: outcome.input_tokens,
                        output_tokens: outcome.output_tokens,
                        duration_ms,
                    })
                    .await;
            }
            Err(e) => {
                let _ = tx
                    .send(StreamEvent::Error {
                        provider: provider_name,
                        model,
                        error: format!("{:#}", e),
                        duration_ms,
                    })
                    .await;
            }
        }
        return messages;
    }

    let options = build_options(ctx);

    let _ = tx
        .send(StreamEvent::Start {
            provider: provider_name.clone(),
            model: model.clone(),
        })
        .await;

    let mut total_input: Option<i32> = None;
    let mut total_output: Option<i32> = None;
    // Text emitted on the FINAL iteration (the one that exits the loop). Earlier
    // iterations may also emit narration before returning a tool call — we
    // discard that so it doesn't leak into the persisted assistant reply.
    let mut final_text = String::new();

    for iteration in 0..ctx.max_iterations.max(1) {
        let stream_response = match manager
            .exec_chat_stream(
                &provider.model,
                messages.clone(),
                options.as_ref(),
                ctx.tools.clone(),
            )
            .await
        {
            Ok(r) => r,
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                let _ = tx
                    .send(StreamEvent::Error {
                        provider: provider_name,
                        model,
                        error: format!("{:#}", e),
                        duration_ms,
                    })
                    .await;
                return messages;
            }
        };

        let mut stream = stream_response.stream;
        let mut captured_tool_calls: Vec<ToolCall> = Vec::new();
        let mut captured_thoughts: Vec<String> = Vec::new();
        // Reset per-iteration text. Only the text from the iteration that
        // actually exits the loop (the non-tool-call one) is kept as the
        // assistant's final reply.
        let mut iter_text = String::new();

        loop {
            // Race the next stream event against cancellation so a Ctrl+C
            // during mid-response breaks out promptly instead of waiting
            // for the provider to finish producing tokens.
            let event = tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    let duration_ms = start.elapsed().as_millis() as u64;
                    let _ = tx
                        .send(StreamEvent::Error {
                            provider: provider_name,
                            model,
                            error: "cancelled".to_string(),
                            duration_ms,
                        })
                        .await;
                    return messages;
                }
                ev = stream.next() => match ev {
                    Some(e) => e,
                    None => break,
                },
            };
            match event {
                Ok(ChatStreamEvent::Chunk(chunk)) => {
                    if !chunk.content.is_empty() {
                        iter_text.push_str(&chunk.content);
                        let _ = tx
                            .send(StreamEvent::Chunk {
                                provider: provider_name.clone(),
                                text: chunk.content,
                            })
                            .await;
                    }
                }
                Ok(ChatStreamEvent::End(end)) => {
                    if let Some(usage) = end.captured_usage {
                        if let Some(pt) = usage.prompt_tokens {
                            total_input = Some(total_input.unwrap_or(0) + pt);
                        }
                        if let Some(ct) = usage.completion_tokens {
                            total_output = Some(total_output.unwrap_or(0) + ct);
                        }
                    }

                    if let Some(content) = end.captured_content {
                        for part in content.into_parts() {
                            match part {
                                genai::chat::ContentPart::ToolCall(tc) => {
                                    captured_tool_calls.push(tc)
                                }
                                genai::chat::ContentPart::ThoughtSignature(t) => {
                                    captured_thoughts.push(t)
                                }
                                _ => {}
                            }
                        }
                    }
                }
                Ok(_) => {} // ignore Start, ReasoningChunk, ToolCallChunk, ThoughtSignatureChunk
                Err(e) => {
                    let duration_ms = start.elapsed().as_millis() as u64;
                    let _ = tx
                        .send(StreamEvent::Error {
                            provider: provider_name,
                            model,
                            error: format!("{:#}", e),
                            duration_ms,
                        })
                        .await;
                    return messages;
                }
            }
        }

        // Done if no tool calls — this iteration's text is the final answer
        if captured_tool_calls.is_empty() {
            final_text = iter_text;
            break;
        }

        // No search client? Can't execute — keep whatever text we got and stop
        if ctx.search_client.is_none() {
            final_text = iter_text;
            break;
        }

        // Safety: stop if we're at the last iteration (results would not be used)
        if iteration + 1 >= ctx.max_iterations {
            final_text = iter_text;
            break;
        }

        // Append assistant message including tool calls and any thought signatures
        let assistant_msg = if captured_thoughts.is_empty() {
            ChatMessage::from(captured_tool_calls.clone())
        } else {
            ChatMessage::assistant_tool_calls_with_thoughts(
                captured_tool_calls.clone(),
                captured_thoughts,
            )
        };
        messages.push(assistant_msg);

        // Execute each tool call
        for tool_call in &captured_tool_calls {
            let result_text = execute_tool_call(tool_call, ctx, &provider_name, Some(&tx)).await;
            let tool_response = ToolResponse::new(tool_call.call_id.clone(), result_text);
            messages.push(ChatMessage::from(tool_response));
        }
    }

    // Silent-failure guard: a stream that ended with no text AND no tool
    // calls is almost always a wire-protocol mismatch (e.g. an adapter
    // pointed at a server that doesn't speak its protocol — Bedrock's
    // 200 + UnknownOperationException is the canonical trigger). Report
    // it as an error so the user sees something instead of "[complete]"
    // with an empty body.
    let duration_ms = start.elapsed().as_millis() as u64;
    if final_text.is_empty() {
        let _ = tx
            .send(StreamEvent::Error {
                provider: provider_name,
                model,
                error: "provider returned an empty response (likely a wire-protocol mismatch; check base_url)".to_string(),
                duration_ms,
            })
            .await;
        return messages;
    }

    // Persist the final assistant reply in the message history so multi-turn
    // callers see it on subsequent turns.
    messages.push(ChatMessage::assistant(final_text));

    let _ = tx
        .send(StreamEvent::Done {
            provider: provider_name,
            model,
            input_tokens: total_input,
            output_tokens: total_output,
            duration_ms,
        })
        .await;

    messages
}

pub async fn stream_all(
    manager: Arc<ProviderManager>,
    providers: Vec<Provider>,
    ctx: QueryContext,
) -> mpsc::Receiver<StreamEvent> {
    let (tx, rx) = mpsc::channel(100);

    for provider in providers {
        let manager = Arc::clone(&manager);
        let ctx = ctx.clone();
        let tx = tx.clone();

        tokio::spawn(async move {
            stream_single(&manager, &provider, &ctx, tx).await;
        });
    }

    rx
}
