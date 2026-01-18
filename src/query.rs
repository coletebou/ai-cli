use crate::providers::{Provider, ProviderManager, ProviderResponse};
use futures::future::join_all;
use futures::StreamExt as FuturesStreamExt;
use genai::chat::{ChatMessage, ChatStreamEvent};
use std::sync::Arc;
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Start { provider: String },
    Chunk { provider: String, text: String },
    Done {
        provider: String,
        input_tokens: Option<i32>,
        output_tokens: Option<i32>,
    },
    Error { provider: String, error: String },
}

pub async fn query_single(
    manager: &ProviderManager,
    provider: &Provider,
    question: &str,
    system_prompt: Option<&str>,
) -> ProviderResponse {
    let mut messages = Vec::new();

    if let Some(system) = system_prompt {
        messages.push(ChatMessage::system(system));
    }
    messages.push(ChatMessage::user(question));

    match manager.exec_chat(&provider.model, messages, None).await {
        Ok(response) => {
            let text = response.first_text().unwrap_or_default().to_string();
            let input_tokens = response.usage.prompt_tokens;
            let output_tokens = response.usage.completion_tokens;
            ProviderResponse::success(provider.clone(), text, input_tokens, output_tokens)
        }
        Err(e) => ProviderResponse::error(provider.clone(), format!("{:#}", e)),
    }
}

pub async fn query_all(
    manager: Arc<ProviderManager>,
    providers: Vec<Provider>,
    question: &str,
    system_prompt: Option<&str>,
) -> Vec<ProviderResponse> {
    let futures = providers.into_iter().map(|provider| {
        let manager = Arc::clone(&manager);
        let question = question.to_string();
        let system_prompt = system_prompt.map(|s| s.to_string());

        async move {
            query_single(
                &manager,
                &provider,
                &question,
                system_prompt.as_deref(),
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
    tx: mpsc::Sender<StreamEvent>,
) {
    let provider_name = provider.display_name.clone();

    let mut messages = Vec::new();
    if let Some(system) = system_prompt {
        messages.push(ChatMessage::system(system));
    }
    messages.push(ChatMessage::user(question));

    // Send start event
    let _ = tx.send(StreamEvent::Start {
        provider: provider_name.clone()
    }).await;

    match manager.exec_chat_stream(&provider.model, messages, None).await {
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
                        let _ = tx.send(StreamEvent::Error {
                            provider: provider_name.clone(),
                            error: format!("{:#}", e),
                        }).await;
                        return;
                    }
                }
            }

            let _ = tx.send(StreamEvent::Done {
                provider: provider_name,
                input_tokens,
                output_tokens,
            }).await;
        }
        Err(e) => {
            let _ = tx.send(StreamEvent::Error {
                provider: provider_name,
                error: format!("{:#}", e),
            }).await;
        }
    }
}

pub async fn stream_all(
    manager: Arc<ProviderManager>,
    providers: Vec<Provider>,
    question: &str,
    system_prompt: Option<&str>,
) -> mpsc::Receiver<StreamEvent> {
    let (tx, rx) = mpsc::channel(100);

    for provider in providers {
        let manager = Arc::clone(&manager);
        let question = question.to_string();
        let system_prompt = system_prompt.map(|s| s.to_string());
        let tx = tx.clone();

        tokio::spawn(async move {
            stream_single(
                &manager,
                &provider,
                &question,
                system_prompt.as_deref(),
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
