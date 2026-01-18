use crate::providers::{Provider, ProviderManager, ProviderResponse};
use futures::future::join_all;
use futures::StreamExt as FuturesStreamExt;
use genai::chat::{ChatMessage, ChatOptions, ChatStreamEvent, ReasoningEffort};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Start { provider: String, model: String },
    Chunk { provider: String, text: String },
    Done {
        provider: String,
        model: String,
        input_tokens: Option<i32>,
        output_tokens: Option<i32>,
        duration_ms: u64,
    },
    Error { provider: String, model: String, error: String, duration_ms: u64 },
}

pub async fn query_single(
    manager: &ProviderManager,
    provider: &Provider,
    question: &str,
    system_prompt: Option<&str>,
    reasoning_effort: Option<ReasoningEffort>,
) -> ProviderResponse {
    let start = Instant::now();
    let model = provider.model.clone();

    let mut messages = Vec::new();

    if let Some(system) = system_prompt {
        messages.push(ChatMessage::system(system));
    }
    messages.push(ChatMessage::user(question));

    // Build chat options with reasoning effort if specified
    let options = reasoning_effort.map(|r| ChatOptions::default().with_reasoning_effort(r));

    match manager.exec_chat(&provider.model, messages, options.as_ref()).await {
        Ok(response) => {
            let duration_ms = start.elapsed().as_millis() as u64;
            let text = response.first_text().unwrap_or_default().to_string();
            let input_tokens = response.usage.prompt_tokens;
            let output_tokens = response.usage.completion_tokens;
            ProviderResponse::success(provider.clone(), model, text, input_tokens, output_tokens, duration_ms)
        }
        Err(e) => {
            let duration_ms = start.elapsed().as_millis() as u64;
            ProviderResponse::error(provider.clone(), model, format!("{:#}", e), duration_ms)
        }
    }
}

pub async fn query_all(
    manager: Arc<ProviderManager>,
    providers: Vec<Provider>,
    question: &str,
    system_prompt: Option<&str>,
    reasoning_effort: Option<ReasoningEffort>,
) -> Vec<ProviderResponse> {
    let futures = providers.into_iter().map(|provider| {
        let manager = Arc::clone(&manager);
        let question = question.to_string();
        let system_prompt = system_prompt.map(|s| s.to_string());
        let reasoning = reasoning_effort.clone();

        async move {
            query_single(
                &manager,
                &provider,
                &question,
                system_prompt.as_deref(),
                reasoning,
            )
            .await
        }
    });

    join_all(futures).await
}

pub async fn stream_single(
    manager: &ProviderManager,
    provider: &Provider,
    question: &str,
    system_prompt: Option<&str>,
    reasoning_effort: Option<ReasoningEffort>,
    tx: mpsc::Sender<StreamEvent>,
) {
    let start = Instant::now();
    let provider_name = provider.display_name.clone();
    let model = provider.model.clone();

    let mut messages = Vec::new();
    if let Some(system) = system_prompt {
        messages.push(ChatMessage::system(system));
    }
    messages.push(ChatMessage::user(question));

    // Build chat options with reasoning effort if specified
    let options = reasoning_effort.map(|r| ChatOptions::default().with_reasoning_effort(r));

    // Send start event
    let _ = tx.send(StreamEvent::Start {
        provider: provider_name.clone(),
        model: model.clone(),
    }).await;

    match manager.exec_chat_stream(&provider.model, messages, options.as_ref()).await {
        Ok(response) => {
            let mut input_tokens: Option<i32> = None;
            let mut output_tokens: Option<i32> = None;

            // Access the inner ChatStream from ChatStreamResponse
            let mut stream = response.stream;

            while let Some(event) = stream.next().await {
                match event {
                    Ok(ChatStreamEvent::Chunk(chunk)) => {
                        let text = chunk.content;
                        if !text.is_empty() {
                            let _ = tx.send(StreamEvent::Chunk {
                                provider: provider_name.clone(),
                                text,
                            }).await;
                        }
                    }
                    Ok(ChatStreamEvent::End(end)) => {
                        if let Some(usage) = end.captured_usage {
                            input_tokens = usage.prompt_tokens;
                            output_tokens = usage.completion_tokens;
                        }
                    }
                    Ok(ChatStreamEvent::Start) => {}
                    Ok(ChatStreamEvent::ReasoningChunk(_)) => {}
                    Ok(ChatStreamEvent::ToolCallChunk(_)) => {}
                    Ok(ChatStreamEvent::ThoughtSignatureChunk(_)) => {}
                    Err(e) => {
                        let duration_ms = start.elapsed().as_millis() as u64;
                        let _ = tx.send(StreamEvent::Error {
                            provider: provider_name.clone(),
                            model: model.clone(),
                            error: format!("{:#}", e),
                            duration_ms,
                        }).await;
                        return;
                    }
                }
            }

            let duration_ms = start.elapsed().as_millis() as u64;
            let _ = tx.send(StreamEvent::Done {
                provider: provider_name,
                model,
                input_tokens,
                output_tokens,
                duration_ms,
            }).await;
        }
        Err(e) => {
            let duration_ms = start.elapsed().as_millis() as u64;
            let _ = tx.send(StreamEvent::Error {
                provider: provider_name,
                model,
                error: format!("{:#}", e),
                duration_ms,
            }).await;
        }
    }
}

pub async fn stream_all(
    manager: Arc<ProviderManager>,
    providers: Vec<Provider>,
    question: &str,
    system_prompt: Option<&str>,
    reasoning_effort: Option<ReasoningEffort>,
) -> mpsc::Receiver<StreamEvent> {
    let (tx, rx) = mpsc::channel(100);

    for provider in providers {
        let manager = Arc::clone(&manager);
        let question = question.to_string();
        let system_prompt = system_prompt.map(|s| s.to_string());
        let reasoning = reasoning_effort.clone();
        let tx = tx.clone();

        tokio::spawn(async move {
            stream_single(
                &manager,
                &provider,
                &question,
                system_prompt.as_deref(),
                reasoning,
                tx,
            )
            .await;
        });
    }

    rx
}

// StreamingResponse is used for TUI mode to track state per provider
#[allow(dead_code)]
pub struct StreamingResponse {
    pub provider: Provider,
    pub text: String,
    pub input_tokens: Option<i32>,
    pub output_tokens: Option<i32>,
    pub is_complete: bool,
    pub error: Option<String>,
}

#[allow(dead_code)]
impl StreamingResponse {
    pub fn new(provider: Provider) -> Self {
        StreamingResponse {
            provider,
            text: String::new(),
            input_tokens: None,
            output_tokens: None,
            is_complete: false,
            error: None,
        }
    }

    pub fn append(&mut self, text: &str) {
        self.text.push_str(text);
    }

    pub fn complete(&mut self, input_tokens: Option<i32>, output_tokens: Option<i32>) {
        self.input_tokens = input_tokens;
        self.output_tokens = output_tokens;
        self.is_complete = true;
    }

    pub fn set_error(&mut self, error: String) {
        self.error = Some(error);
        self.is_complete = true;
    }
}
