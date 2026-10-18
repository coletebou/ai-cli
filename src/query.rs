use crate::providers::{Provider, ProviderManager, ProviderResponse};
use crate::search::{format_results_for_model, SearchEngine};
use crate::telemetry::TRACER_NAME;
use futures::future::join_all;
use futures::StreamExt as FuturesStreamExt;
use genai::chat::{
    ChatMessage, ChatOptions, ChatStreamEvent, ContentPart, ReasoningEffort, Tool, ToolCall,
    ToolResponse,
};
use opentelemetry::trace::{FutureExt, Span, Tracer};
use opentelemetry::{global, Context, KeyValue};
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

/// Map our internal display name (uppercase aliases like "CLAUDE", "CHATGPT")
/// to the OpenTelemetry GenAI `gen_ai.system` semantic value. Falls back to
/// `"unknown"` for anything we haven't mapped; this is a valid semconv value
/// that Langfuse accepts, so new providers still produce usable telemetry.
fn gen_ai_system_for(display_name: &str) -> &'static str {
    match display_name.to_ascii_uppercase().as_str() {
        "CLAUDE" => "anthropic",
        "CHATGPT" | "GPT" => "openai",
        "GROK" => "xai",
        "GEMINI" => "google_vertex_ai",
        "OLLAMA" | "LOCAL" => "ollama",
        "OPENROUTER" | "OR" => "openrouter",
        _ => "unknown",
    }
}

/// Best-effort JSON-encode a message list for the `langfuse.observation.input`
/// attribute. We use `serde_json::Value` rather than the genai types directly
/// because `ChatMessage` doesn't implement Serialize. Failing to encode just
/// drops the attribute — telemetry is opportunistic, not load-bearing.
fn messages_to_json(messages: &[ChatMessage]) -> Option<String> {
    let parts: Vec<serde_json::Value> = messages
        .iter()
        .filter_map(|m| {
            let role = match m.role {
                genai::chat::ChatRole::System => "system",
                genai::chat::ChatRole::User => "user",
                genai::chat::ChatRole::Assistant => "assistant",
                genai::chat::ChatRole::Tool => "tool",
            };
            let text = m.content.texts().join("");
            if text.is_empty() {
                return None;
            }
            Some(serde_json::json!({ "role": role, "content": text }))
        })
        .collect();
    serde_json::to_string(&parts).ok()
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

    // OTel span — same shape as `stream_with_messages` so both code paths
    // produce comparable Langfuse generations.
    let tracer = global::tracer(TRACER_NAME);
    let mut span = tracer
        .span_builder(format!("chat {}", provider.display_name))
        .with_attributes(vec![
            KeyValue::new("gen_ai.system", gen_ai_system_for(&provider.display_name)),
            KeyValue::new("gen_ai.request.model", model.clone()),
            KeyValue::new("gen_ai.operation.name", "chat"),
            KeyValue::new("langfuse.observation.type", "generation"),
            KeyValue::new("langfuse.observation.model.name", model.clone()),
        ])
        .start(&tracer);
    if let Some(input_json) = messages_to_json(&messages) {
        span.set_attribute(KeyValue::new("langfuse.observation.input", input_json));
    }

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
                let err_msg = format!("{:#}", e);
                span.set_attribute(KeyValue::new("error", true));
                span.set_attribute(KeyValue::new("exception.message", err_msg.clone()));
                span.end();
                return ProviderResponse::error(provider.clone(), model, err_msg, duration_ms);
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
    if let Some(t) = total_input {
        span.set_attribute(KeyValue::new("gen_ai.usage.input_tokens", t as i64));
    }
    if let Some(t) = total_output {
        span.set_attribute(KeyValue::new("gen_ai.usage.output_tokens", t as i64));
    }
    span.set_attribute(KeyValue::new(
        "langfuse.observation.output",
        final_text.clone(),
    ));
    if final_text.is_empty() {
        span.set_attribute(KeyValue::new("error", true));
        span.set_attribute(KeyValue::new(
            "exception.message",
            "empty response (likely wire-protocol mismatch)",
        ));
        span.end();
        return ProviderResponse::error(
            provider.clone(),
            model,
            "provider returned an empty response (likely a wire-protocol mismatch; check base_url)"
                .to_string(),
            duration_ms,
        );
    }
    span.end();
    ProviderResponse::success(
        provider.clone(),
        model,
        final_text,
        total_input,
        total_output,
        duration_ms,
    )
}

/// Real streaming branch for Bedrock. Drives ConverseStream, forwards text
/// deltas to the main display channel, reconstructs any tool calls from
/// split input deltas, and loops through tool-use iterations in the same
/// way the genai streaming path does.
#[allow(clippy::too_many_arguments)]
async fn stream_bedrock(
    manager: &ProviderManager,
    bedrock: &crate::bedrock::BedrockClient,
    provider: &Provider,
    mut messages: Vec<ChatMessage>,
    ctx: &QueryContext,
    tx: mpsc::Sender<StreamEvent>,
    cancel: &CancellationToken,
    start: Instant,
) -> Vec<ChatMessage> {
    use crate::bedrock::BedrockStreamEvent as BE;
    use genai::chat::{MessageContent, ToolResponse};
    use std::collections::HashMap;

    let provider_name = provider.display_name.clone();
    let model = provider.model.clone();

    let _ = tx
        .send(StreamEvent::Start {
            provider: provider_name.clone(),
            model: model.clone(),
        })
        .await;

    let mut total_input: Option<i32> = None;
    let mut total_output: Option<i32> = None;
    let mut final_text = String::new();
    // Why the final answer is incomplete, if Bedrock said so (stopReason).
    let mut final_note: Option<&'static str> = None;

    // Per-call streaming state. Declared here so helpers below can borrow.
    for iteration in 0..ctx.max_iterations.max(1) {
        // State accumulated during this iteration.
        let mut iter_text = String::new();
        // block_index -> (tool_use_id, name, input_fragment_buffer)
        let mut tool_builders: HashMap<usize, (String, String, String)> = HashMap::new();
        let mut finalized_tool_calls: Vec<ToolCall> = Vec::new();
        let mut stop_reason: Option<String> = None;
        let mut stream_error: Option<String> = None;

        let mut attempt: u32 = 0;
        loop {
            attempt += 1;
            // Spin up a child channel that the Bedrock client drives.
            let (child_tx, mut child_rx) = mpsc::channel::<BE>(64);
            let msgs_clone = messages.clone();
            let tools = ctx.tools.clone();
            let model_for_call = model.clone();
            // The system prompt already rides in `messages` (every caller
            // seeds history with it), so don't pass it again separately —
            // that used to send it to Bedrock twice.
            let stream_task = async move {
                bedrock
                    .converse_stream(
                        &model_for_call,
                        &msgs_clone,
                        None,
                        tools.as_deref(),
                        child_tx,
                    )
                    .await
            };
            tokio::pin!(stream_task);

            // Pump both the stream task and events from the child channel.
            // The task running the HTTP request is kept polling so that
            // `child_rx.recv()` can observe events as they arrive.
            let task_result = loop {
                tokio::select! {
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
                    result = &mut stream_task => {
                        // Stream task has finished sending; drain any remaining
                        // events, then break the select loop.
                        while let Ok(ev) = child_rx.try_recv() {
                            handle_bedrock_event(
                                ev,
                                &provider_name,
                                &tx,
                                &mut iter_text,
                                &mut tool_builders,
                                &mut finalized_tool_calls,
                                &mut stop_reason,
                                &mut total_input,
                                &mut total_output,
                                &mut stream_error,
                            ).await;
                        }
                        break result;
                    }
                    ev = child_rx.recv() => match ev {
                        Some(e) => handle_bedrock_event(
                            e,
                            &provider_name,
                            &tx,
                            &mut iter_text,
                            &mut tool_builders,
                            &mut finalized_tool_calls,
                            &mut stop_reason,
                            &mut total_input,
                            &mut total_output,
                            &mut stream_error,
                        ).await,
                        // Channel closed: the task is finishing. Collect its
                        // result so a request-level error isn't dropped.
                        None => break (&mut stream_task).await,
                    },
                }
            };

            if let Err(e) = task_result {
                // Retry only while nothing has reached the user yet — once
                // text has streamed we can't take it back.
                let nothing_emitted = iter_text.is_empty()
                    && tool_builders.is_empty()
                    && finalized_tool_calls.is_empty();
                if nothing_emitted && stream_error.is_none() {
                    if let Some(delay) = manager.retry_delay(&e, attempt) {
                        crate::diag::warn(format!(
                            "[retry] {} attempt {}/{} after {:?}",
                            model,
                            attempt,
                            manager.max_attempts(),
                            delay
                        ));
                        tokio::select! {
                            _ = cancel.cancelled() => continue, // handled at the top of the next attempt
                            _ = tokio::time::sleep(delay) => continue,
                        }
                    }
                }
                stream_error.get_or_insert_with(|| format!("{:#}", e));
            }
            break;
        }

        // Surface any stream-level error and stop the iteration loop.
        if let Some(err) = stream_error {
            let duration_ms = start.elapsed().as_millis() as u64;
            let _ = tx
                .send(StreamEvent::Error {
                    provider: provider_name,
                    model,
                    error: err,
                    duration_ms,
                })
                .await;
            return messages;
        }

        // Terminal iteration: no tool calls, or we can't execute them.
        let no_tools = finalized_tool_calls.is_empty();
        let no_search = ctx.search_client.is_none();
        let last_iter = iteration + 1 >= ctx.max_iterations;

        if no_tools || no_search || last_iter {
            // Flag answers Bedrock cut short. Sent as a display-only chunk,
            // so the note isn't persisted into the conversation history.
            // With no text at all (e.g. a guardrail blocked everything) the
            // note becomes the error below instead.
            final_note = crate::bedrock::stop_reason_note(stop_reason.as_deref());
            if let (Some(note), false) = (final_note, iter_text.is_empty()) {
                let _ = tx
                    .send(StreamEvent::Chunk {
                        provider: provider_name.clone(),
                        text: format!("\n\n*({})*\n", note),
                    })
                    .await;
            }
            final_text = iter_text;
            break;
        }

        // Tool-use iteration: append assistant message with tool calls,
        // execute each, append tool responses, and loop. Note: if the
        // assistant produced narrative text alongside tool calls, include it
        // too so Bedrock sees a well-formed assistant turn.
        let mut parts: Vec<ContentPart> = Vec::new();
        if !iter_text.is_empty() {
            parts.push(ContentPart::Text(iter_text.clone()));
        }
        for tc in &finalized_tool_calls {
            parts.push(ContentPart::ToolCall(tc.clone()));
        }
        messages.push(ChatMessage::assistant(MessageContent::from_parts(parts)));

        for tool_call in &finalized_tool_calls {
            let result_text = execute_tool_call(tool_call, ctx, &provider_name, Some(&tx)).await;
            let tool_response = ToolResponse::new(tool_call.call_id.clone(), result_text);
            messages.push(ChatMessage::from(tool_response));
        }
    }

    // Silent-failure guard (mirrors the genai path): a terminal iteration
    // that produced neither text nor tool calls is reported as an error.
    let duration_ms = start.elapsed().as_millis() as u64;
    if final_text.is_empty() {
        let _ = tx
            .send(StreamEvent::Error {
                provider: provider_name,
                model,
                error: empty_bedrock_response_error(final_note),
                duration_ms,
            })
            .await;
        return messages;
    }

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

/// Per-event handler for the Bedrock streaming branch. Kept separate to
/// keep the main loop readable; mutates the accumulator state passed in.
#[allow(clippy::too_many_arguments)]
async fn handle_bedrock_event(
    ev: crate::bedrock::BedrockStreamEvent,
    provider_name: &str,
    tx: &mpsc::Sender<StreamEvent>,
    iter_text: &mut String,
    tool_builders: &mut std::collections::HashMap<usize, (String, String, String)>,
    finalized: &mut Vec<ToolCall>,
    stop_reason: &mut Option<String>,
    total_input: &mut Option<i32>,
    total_output: &mut Option<i32>,
    stream_error: &mut Option<String>,
) {
    use crate::bedrock::BedrockStreamEvent as BE;
    match ev {
        BE::TextDelta(text) => {
            iter_text.push_str(&text);
            let _ = tx
                .send(StreamEvent::Chunk {
                    provider: provider_name.to_string(),
                    text,
                })
                .await;
        }
        BE::ToolUseStart {
            block_index,
            tool_use_id,
            name,
        } => {
            tool_builders.insert(block_index, (tool_use_id, name, String::new()));
        }
        BE::ToolUseInputDelta {
            block_index,
            input_fragment,
        } => {
            if let Some(b) = tool_builders.get_mut(&block_index) {
                b.2.push_str(&input_fragment);
            }
        }
        BE::ContentBlockStop { block_index } => {
            if let Some((tool_use_id, name, input_buf)) = tool_builders.remove(&block_index) {
                // Tool-use blocks: parse the accumulated JSON input. If
                // parsing fails, the model emitted invalid JSON — surface it
                // as a stream error rather than fabricating a tool call.
                match serde_json::from_str::<serde_json::Value>(if input_buf.is_empty() {
                    "{}"
                } else {
                    &input_buf
                }) {
                    Ok(input) => finalized.push(ToolCall {
                        call_id: tool_use_id,
                        fn_name: name,
                        fn_arguments: input,
                        thought_signatures: None,
                    }),
                    Err(e) => {
                        *stream_error = Some(format!(
                            "Bedrock streamed invalid tool-use JSON for {}: {} (raw: {:?})",
                            name, e, input_buf
                        ));
                    }
                }
            }
            // Text block stops need no action (text is already streamed).
        }
        BE::MessageStop { stop_reason: sr } => {
            *stop_reason = sr;
        }
        BE::Metadata {
            input_tokens,
            output_tokens,
        } => {
            if let Some(v) = input_tokens {
                *total_input = Some(total_input.unwrap_or(0) + v);
            }
            if let Some(v) = output_tokens {
                *total_output = Some(total_output.unwrap_or(0) + v);
            }
        }
    }
}

/// The error for a Bedrock answer with no text. When Bedrock said why
/// (a guardrail or content filter removed everything, or the context window
/// ran out), say that rather than a generic "empty response".
fn empty_bedrock_response_error(stop_note: Option<&str>) -> String {
    match stop_note {
        Some(note) => format!("Bedrock returned no text: {}", note),
        None => "Bedrock returned an empty response (no text content)".to_string(),
    }
}

/// The answer from a terminal non-streaming Bedrock call and why it's
/// incomplete, if Bedrock said so. Only this call's text counts, as in
/// `stream_bedrock`: text from an earlier tool-use iteration is narration
/// ahead of a tool call ("let me search..."), and a guardrail or max_tokens
/// note appended to it would describe the wrong reply. With no text the
/// answer is empty and the caller reports the note as the error.
fn bedrock_terminal_answer(
    outcome: &crate::bedrock::BedrockOutcome,
) -> (String, Option<&'static str>) {
    let note = crate::bedrock::stop_reason_note(outcome.stop_reason.as_deref());
    let mut text = outcome.text.clone();
    if let (Some(note), false) = (note, text.is_empty()) {
        text.push_str(&format!("\n\n*({})*", note));
    }
    (text, note)
}

/// Non-streaming Bedrock branch for `query_single`. Runs the same tool-use
/// loop as the streaming path via the non-streaming Converse API.
async fn query_single_bedrock(
    manager: &ProviderManager,
    provider: &Provider,
    ctx: &QueryContext,
    bedrock: &crate::bedrock::BedrockClient,
    start: Instant,
) -> ProviderResponse {
    use genai::chat::MessageContent;

    let mut messages = build_initial_messages(&ctx.question, ctx.system_prompt.as_deref());
    let mut total_input: Option<i32> = None;
    let mut total_output: Option<i32> = None;
    let mut final_text = String::new();
    let mut final_note: Option<&'static str> = None;

    for iteration in 0..ctx.max_iterations.max(1) {
        let outcome = match manager
            .exec_bedrock_converse(
                bedrock,
                &provider.model,
                &messages,
                None, // already in `messages`; see stream_bedrock
                ctx.tools.as_deref(),
            )
            .await
        {
            Ok(o) => o,
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                return ProviderResponse::error(
                    provider.clone(),
                    provider.model.clone(),
                    format!("{:#}", e),
                    duration_ms,
                );
            }
        };

        if let Some(v) = outcome.input_tokens {
            total_input = Some(total_input.unwrap_or(0) + v);
        }
        if let Some(v) = outcome.output_tokens {
            total_output = Some(total_output.unwrap_or(0) + v);
        }

        // No tool calls -> we have our final answer.
        if outcome.tool_calls.is_empty()
            || ctx.search_client.is_none()
            || iteration + 1 >= ctx.max_iterations
        {
            (final_text, final_note) = bedrock_terminal_answer(&outcome);
            break;
        }

        // Assistant turn: text (if any) plus the tool-use blocks.
        let mut parts: Vec<ContentPart> = Vec::new();
        if !outcome.text.is_empty() {
            parts.push(ContentPart::Text(outcome.text.clone()));
        }
        for tc in &outcome.tool_calls {
            parts.push(ContentPart::ToolCall(tc.clone()));
        }
        messages.push(ChatMessage::assistant(MessageContent::from_parts(parts)));

        // Execute tool calls and append responses (user turns with toolResult).
        for tool_call in &outcome.tool_calls {
            let result_text = execute_tool_call(tool_call, ctx, &provider.display_name, None).await;
            let tool_response = ToolResponse::new(tool_call.call_id.clone(), result_text);
            messages.push(ChatMessage::from(tool_response));
        }
    }

    let duration_ms = start.elapsed().as_millis() as u64;
    if final_text.is_empty() {
        return ProviderResponse::error(
            provider.clone(),
            provider.model.clone(),
            empty_bedrock_response_error(final_note),
            duration_ms,
        );
    }
    ProviderResponse::success(
        provider.clone(),
        provider.model.clone(),
        final_text,
        total_input,
        total_output,
        duration_ms,
    )
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

    // Bedrock-hosted models bypass genai (which cannot speak Bedrock's wire
    // protocol) and drive ConverseStream directly. The dispatch happens before
    // telemetry span setup so Bedrock requests stay on a clean codepath; the
    // root `ai.query` span still wraps this call via the calling task's Context.
    if let Some(bedrock) = manager.bedrock_for(&model) {
        return stream_bedrock(
            manager, &bedrock, provider, messages, ctx, tx, cancel, start,
        )
        .await;
    }

    // Open a per-call OTel span. Inherits the current Context as parent —
    // when wrapped by a root "ai.query" span via `Context::attach`, this
    // becomes a child generation in Langfuse.
    let tracer = global::tracer(TRACER_NAME);
    let mut span = tracer
        .span_builder(format!("chat {}", provider_name))
        .with_attributes(vec![
            KeyValue::new("gen_ai.system", gen_ai_system_for(&provider_name)),
            KeyValue::new("gen_ai.request.model", model.clone()),
            KeyValue::new("gen_ai.operation.name", "chat"),
            // Tells Langfuse to render this span as a generation, not a
            // generic span — unlocks token/cost rollups in the UI.
            KeyValue::new("langfuse.observation.type", "generation"),
            KeyValue::new("langfuse.observation.model.name", model.clone()),
        ])
        .start(&tracer);
    if let Some(input_json) = messages_to_json(&messages) {
        span.set_attribute(KeyValue::new("langfuse.observation.input", input_json));
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
        // Opening the stream includes connecting, the first response bytes,
        // and any retry backoff — race all of it against cancellation so
        // Ctrl+C doesn't have to wait for a slow or stalled provider.
        let opened = tokio::select! {
            biased;
            _ = cancel.cancelled() => Err(anyhow::anyhow!("cancelled")),
            r = manager.exec_chat_stream(
                &provider.model,
                messages.clone(),
                options.as_ref(),
                ctx.tools.clone(),
            ) => r,
        };
        let mut stream = match opened {
            Ok(s) => s,
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                let err_msg = format!("{:#}", e);
                span.set_attribute(KeyValue::new("error", true));
                span.set_attribute(KeyValue::new("exception.message", err_msg.clone()));
                span.end();
                let _ = tx
                    .send(StreamEvent::Error {
                        provider: provider_name,
                        model,
                        error: err_msg,
                        duration_ms,
                    })
                    .await;
                return messages;
            }
        };

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
                    span.set_attribute(KeyValue::new("cancelled", true));
                    span.end();
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
                    let err_msg = format!("{:#}", e);
                    span.set_attribute(KeyValue::new("error", true));
                    span.set_attribute(KeyValue::new("exception.message", err_msg.clone()));
                    span.end();
                    let _ = tx
                        .send(StreamEvent::Error {
                            provider: provider_name,
                            model,
                            error: err_msg,
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

    // Finalize span attributes before moving `final_text` into the message
    // history below.
    let duration_ms = start.elapsed().as_millis() as u64;
    if let Some(t) = total_input {
        span.set_attribute(KeyValue::new("gen_ai.usage.input_tokens", t as i64));
    }
    if let Some(t) = total_output {
        span.set_attribute(KeyValue::new("gen_ai.usage.output_tokens", t as i64));
    }
    span.set_attribute(KeyValue::new(
        "langfuse.observation.output",
        final_text.clone(),
    ));

    // Silent-failure guard: a stream that ended with no text AND no tool
    // calls is almost always a wire-protocol mismatch (e.g. an adapter
    // pointed at a server that doesn't speak its protocol — Bedrock's
    // 200 + UnknownOperationException was the canonical trigger). Report
    // it as an error so the user sees something instead of "[complete]"
    // with an empty body.
    if final_text.is_empty() {
        span.set_attribute(KeyValue::new("error", true));
        span.set_attribute(KeyValue::new(
            "exception.message",
            "empty response (likely wire-protocol mismatch)",
        ));
        span.end();
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
    span.end();

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

    // tokio::spawn does not propagate the OTel Context. Capture it here and
    // wrap each spawned future with `.with_context(parent_cx)` so per-provider
    // spans inside `stream_single` become children of the caller's root
    // "ai.query" span rather than orphans. We use `with_context` (a
    // FutureExt method) instead of `Context::attach()` because the returned
    // ContextGuard is `!Send` and can't be held across await points.
    let parent_cx = Context::current();

    for provider in providers {
        let manager = Arc::clone(&manager);
        let ctx = ctx.clone();
        let tx = tx.clone();
        let parent_cx = parent_cx.clone();

        // DETACH: each task terminates naturally when stream_single returns;
        // the cloned tx drops with it, and the receiver sees a closed channel
        // once every task finishes. Cancellation propagates via the CLI's
        // CancellationToken (held inside ctx), not via JoinHandle.abort().
        tokio::spawn(
            async move { stream_single(&manager, &provider, &ctx, tx).await }
                .with_context(parent_cx),
        );
    }

    rx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_bedrock_answers_explain_why_when_bedrock_said() {
        let with_note =
            empty_bedrock_response_error(Some("response stopped by a Bedrock guardrail"));
        assert!(with_note.contains("guardrail"), "got: {}", with_note);
        assert!(empty_bedrock_response_error(None).contains("empty response"));
    }

    #[test]
    fn bedrock_stop_note_rides_only_on_the_terminal_calls_text() {
        let outcome = |text: &str, stop_reason: &str| crate::bedrock::BedrockOutcome {
            text: text.to_string(),
            tool_calls: Vec::new(),
            input_tokens: None,
            output_tokens: None,
            stop_reason: Some(stop_reason.to_string()),
        };

        let (text, note) = bedrock_terminal_answer(&outcome("partial answer", "max_tokens"));
        assert!(note.is_some());
        assert!(
            text.starts_with("partial answer\n\n*(response truncated"),
            "got: {}",
            text
        );

        // A guardrail blocked the whole final reply: nothing to annotate, so
        // the answer is empty and the note becomes the error, rather than
        // being attached to narration from an earlier tool-use call.
        let (text, note) = bedrock_terminal_answer(&outcome("", "guardrail_intervened"));
        assert!(text.is_empty());
        assert_eq!(
            empty_bedrock_response_error(note),
            "Bedrock returned no text: response stopped by a Bedrock guardrail"
        );

        let (text, note) = bedrock_terminal_answer(&outcome("done", "end_turn"));
        assert_eq!((text.as_str(), note), ("done", None));
    }

    #[test]
    fn gen_ai_system_maps_known_aliases() {
        // Maps each canonical UI display name to the OpenTelemetry GenAI
        // semconv value Langfuse uses for cost/usage rollups by provider.
        assert_eq!(gen_ai_system_for("CLAUDE"), "anthropic");
        assert_eq!(gen_ai_system_for("CHATGPT"), "openai");
        assert_eq!(gen_ai_system_for("GPT"), "openai");
        assert_eq!(gen_ai_system_for("GROK"), "xai");
        assert_eq!(gen_ai_system_for("GEMINI"), "google_vertex_ai");
        assert_eq!(gen_ai_system_for("OLLAMA"), "ollama");
    }

    #[test]
    fn gen_ai_system_falls_back_to_unknown_for_new_providers() {
        // Forward-compat: a provider added later that we forgot to map here
        // still gets a usable, non-panicking attribute value.
        assert_eq!(gen_ai_system_for("FANCY_NEW_PROVIDER"), "unknown");
    }

    #[test]
    fn messages_to_json_skips_empty_text_and_preserves_order() {
        let messages = vec![
            ChatMessage::system("you are helpful"),
            ChatMessage::user("hi"),
            ChatMessage::assistant("hello"),
        ];
        let json = messages_to_json(&messages).expect("encodes");
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let arr = parsed.as_array().unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0]["role"], "system");
        assert_eq!(arr[0]["content"], "you are helpful");
        assert_eq!(arr[1]["role"], "user");
        assert_eq!(arr[2]["role"], "assistant");
    }
}
