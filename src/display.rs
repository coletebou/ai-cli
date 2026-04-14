use crate::config::FooterConfig;
use crate::providers::{Provider, ProviderResponse};
use crate::query::StreamEvent;
use colored::Colorize;
use crossterm::terminal;
use std::io::{self, Write};
use termimad::MadSkin;
use tokio::sync::mpsc;

fn get_markdown_skin() -> MadSkin {
    let mut skin = MadSkin::default();
    skin.set_headers_fg(termimad::crossterm::style::Color::Cyan);
    skin.bold.set_fg(termimad::crossterm::style::Color::White);
    skin.italic.set_fg(termimad::crossterm::style::Color::White);
    skin
}

fn render_markdown(text: &str) {
    let skin = get_markdown_skin();
    skin.print_text(text);
}

fn get_terminal_width() -> usize {
    terminal::size().map(|(w, _)| w as usize).unwrap_or(80)
}

fn make_separator(width: usize) -> String {
    "━".repeat(width.min(120)) // Cap at 120 to avoid excessive width
}

pub fn print_header(provider: &Provider, status: &str) {
    let width = get_terminal_width();
    let separator = make_separator(width);

    let header = format!(" {} ", provider.display_name);
    let status_str = format!("[{}]", status);

    // Calculate padding, ensuring we don't overflow on narrow terminals
    let content_len = header.len() + status_str.len();
    let padded_header = if content_len + 2 <= width {
        let padding = width.saturating_sub(content_len + 2);
        format!("{}{:>width$}", header, status_str, width = padding + status_str.len())
    } else {
        // Narrow terminal: just show provider name and status without padding
        format!("{} {}", header.trim(), status_str)
    };

    println!();
    println!("{}", separator.color(provider.color));
    println!("{}", padded_header.color(provider.color).bold());
    println!("{}", separator.color(provider.color));
}

pub fn format_reasoning(reasoning: Option<&str>) -> String {
    reasoning.unwrap_or("none").to_string()
}

pub fn print_metadata(
    model: &str,
    input_tokens: Option<i32>,
    output_tokens: Option<i32>,
    duration_ms: u64,
    footer: &FooterConfig,
    reasoning: Option<&str>,
) {
    println!();
    let mut parts = Vec::new();

    if footer.show_model {
        parts.push(format!("model: {}", model));
    }

    if footer.show_tokens {
        if let (Some(input), Some(output)) = (input_tokens, output_tokens) {
            parts.push(format!("tokens: {} in / {} out", input, output));
        }
    }

    if footer.show_time {
        let duration_str = if duration_ms >= 1000 {
            format!("{:.1}s", duration_ms as f64 / 1000.0)
        } else {
            format!("{}ms", duration_ms)
        };
        parts.push(format!("time: {}", duration_str));
    }

    if footer.show_reasoning {
        parts.push(format!("reasoning: {}", format_reasoning(reasoning)));
    }

    if !parts.is_empty() {
        let metadata = parts.join(" | ");
        println!("{}", metadata.dimmed());
    }
    println!();
}

pub fn print_error(error: &str) {
    println!("{} {}", "Error:".red().bold(), error.red());
}

pub fn print_response(response: &ProviderResponse, footer: &FooterConfig, reasoning: Option<&str>) {
    if response.is_error() {
        print_header(&response.provider, "error");
        print_error(response.error.as_ref().unwrap());
        // Minimal footer for errors
        let error_footer = FooterConfig {
            show_model: footer.show_model,
            show_time: footer.show_time,
            show_tokens: false,
            show_reasoning: false,
        };
        print_metadata(&response.model, None, None, response.duration_ms, &error_footer, reasoning);
    } else {
        print_header(&response.provider, "complete");
        render_markdown(&response.text);
        print_metadata(
            &response.model,
            response.input_tokens,
            response.output_tokens,
            response.duration_ms,
            footer,
            reasoning,
        );
    }
}

pub fn print_responses(responses: &[ProviderResponse], footer: &FooterConfig, reasoning: Option<&str>) {
    for response in responses {
        print_response(response, footer, reasoning);
    }
}

pub fn print_raw_responses(responses: &[ProviderResponse]) {
    for (i, response) in responses.iter().enumerate() {
        if i > 0 {
            println!("\n---\n");
        }
        if response.is_error() {
            eprintln!("[{}] Error: {}", response.provider.display_name, response.error.as_ref().unwrap());
        } else {
            println!("[{}]\n{}", response.provider.display_name, response.text);
        }
    }
}

/// Tracks buffered state for a single provider during streaming
struct ProviderBuffer {
    color: colored::Color,
    content: String,
}

pub struct StreamingDisplay {
    footer: FooterConfig,
    raw_mode: bool,
    reasoning: Option<String>,
    // Buffer content per provider until completion
    provider_buffers: std::collections::HashMap<String, ProviderBuffer>,
    // Provider colors lookup
    provider_colors: std::collections::HashMap<String, colored::Color>,
}

impl StreamingDisplay {
    pub fn new(providers: &[Provider], footer: FooterConfig, raw_mode: bool, reasoning: Option<String>) -> Self {
        // Build lookup map for provider colors
        let provider_colors: std::collections::HashMap<String, colored::Color> = providers
            .iter()
            .map(|p| (p.display_name.clone(), p.color))
            .collect();

        StreamingDisplay {
            footer,
            raw_mode,
            reasoning,
            provider_buffers: std::collections::HashMap::new(),
            provider_colors,
        }
    }

    pub fn handle_event(&mut self, event: StreamEvent) {
        if self.raw_mode {
            self.handle_event_raw(event);
        } else {
            self.handle_event_formatted(event);
        }
    }

    fn handle_event_raw(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::Start { provider, .. } => {
                println!("[{}]", provider);
            }
            StreamEvent::Chunk { text, .. } => {
                print!("{}", text);
                let _ = io::stdout().flush();
            }
            StreamEvent::Done { model, duration_ms, .. } => {
                let duration_str = if duration_ms >= 1000 {
                    format!("{:.1}s", duration_ms as f64 / 1000.0)
                } else {
                    format!("{}ms", duration_ms)
                };
                println!();
                println!("[model: {} | time: {} | reasoning: {}]", model, duration_str, format_reasoning(self.reasoning.as_deref()));
            }
            StreamEvent::Error { provider, error, .. } => {
                eprintln!("[{}] Error: {}", provider, error);
            }
            StreamEvent::SearchStart { provider, query } => {
                eprintln!("[{}/search] {}", provider, query);
            }
            StreamEvent::SearchEnd { provider, search_provider, result_count, error, .. } => {
                match error {
                    Some(e) => eprintln!("[{}/search] failed: {}", provider, e),
                    None => eprintln!(
                        "[{}/search via {}] {} result(s)",
                        provider, search_provider, result_count
                    ),
                }
            }
        }
    }

    fn handle_event_formatted(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::Start { provider, .. } => {
                // Buffer provider info - don't print header yet
                let color = self.provider_colors.get(&provider).copied().unwrap_or(colored::Color::White);
                self.provider_buffers.insert(provider, ProviderBuffer {
                    color,
                    content: String::new(),
                });
            }
            StreamEvent::Chunk { provider, text } => {
                // Buffer content for this provider
                if let Some(buffer) = self.provider_buffers.get_mut(&provider) {
                    buffer.content.push_str(&text);
                }
            }
            StreamEvent::Done {
                provider,
                model,
                input_tokens,
                output_tokens,
                duration_ms,
            } => {
                // Get buffered content and display complete response
                let (content, color) = if let Some(buffer) = self.provider_buffers.remove(&provider) {
                    (buffer.content, buffer.color)
                } else {
                    (String::new(), colored::Color::White)
                };

                // Print header
                let width = get_terminal_width();
                let separator = make_separator(width);
                let header = format!(" {} ", provider);
                let status_str = "[complete]";
                let content_len = header.len() + status_str.len();
                let padded_header = if content_len + 2 <= width {
                    let padding = width.saturating_sub(content_len + 2);
                    format!("{}{:>width$}", header, status_str, width = padding + status_str.len())
                } else {
                    format!("{} {}", header.trim(), status_str)
                };

                println!();
                println!("{}", separator.color(color));
                println!("{}", padded_header.color(color).bold());
                println!("{}", separator.color(color));

                // Print content
                if !content.is_empty() {
                    let skin = get_markdown_skin();
                    skin.print_text(&content);
                }

                // Print metadata footer
                let mut parts = Vec::new();

                if self.footer.show_model {
                    parts.push(format!("model: {}", model));
                }

                if self.footer.show_tokens {
                    if let (Some(input), Some(output)) = (input_tokens, output_tokens) {
                        parts.push(format!("tokens: {} in / {} out", input, output));
                    }
                }

                if self.footer.show_time {
                    let duration_str = if duration_ms >= 1000 {
                        format!("{:.1}s", duration_ms as f64 / 1000.0)
                    } else {
                        format!("{}ms", duration_ms)
                    };
                    parts.push(format!("time: {}", duration_str));
                }

                if self.footer.show_reasoning {
                    parts.push(format!("reasoning: {}", format_reasoning(self.reasoning.as_deref())));
                }

                if !parts.is_empty() {
                    let metadata = parts.join(" | ");
                    println!("{}", metadata.dimmed());
                }
            }
            StreamEvent::Error { provider, model, error, duration_ms } => {
                // Remove any buffered content for this provider
                let color = self.provider_buffers.remove(&provider)
                    .map(|b| b.color)
                    .unwrap_or_else(|| self.provider_colors.get(&provider).copied().unwrap_or(colored::Color::White));

                let width = get_terminal_width();
                let separator = make_separator(width);

                let header = format!(" {} ", provider);
                let status_str = "[error]";
                let content_len = header.len() + status_str.len();
                let padded_header = if content_len + 2 <= width {
                    let padding = width.saturating_sub(content_len + 2);
                    format!("{}{:>width$}", header, status_str, width = padding + status_str.len())
                } else {
                    format!("{} {}", header.trim(), status_str)
                };

                println!();
                println!("{}", separator.color(color));
                println!("{}", padded_header.color(color).bold());
                println!("{}", separator.color(color));

                println!("{} {}", "Error:".red().bold(), error.red());
                let duration_str = if duration_ms >= 1000 {
                    format!("{:.1}s", duration_ms as f64 / 1000.0)
                } else {
                    format!("{}ms", duration_ms)
                };
                println!("{}", format!("model: {} | time: {}", model, duration_str).dimmed());
            }
            StreamEvent::SearchStart { provider, query } => {
                // Live feedback to stderr so the user sees searches in progress,
                // and append an inline marker into the buffered content for the final render.
                eprintln!("  {} {}", "→ search:".dimmed(), query.italic());
                if let Some(buffer) = self.provider_buffers.get_mut(&provider) {
                    buffer.content.push_str(&format!("\n*→ searching: {}*\n\n", query));
                }
            }
            StreamEvent::SearchEnd { provider, search_provider, result_count, error, .. } => {
                match &error {
                    Some(e) => eprintln!("  {} {}", "← search failed:".dimmed(), e.red()),
                    None => eprintln!(
                        "  {} {}",
                        "←".dimmed(),
                        format!("via {}: {} result(s)", search_provider, result_count).dimmed()
                    ),
                }
                if let Some(buffer) = self.provider_buffers.get_mut(&provider) {
                    match error {
                        Some(e) => buffer.content.push_str(&format!("*← search failed: {}*\n\n", e)),
                        None => buffer.content.push_str(&format!(
                            "*← via {}: {} result(s)*\n\n",
                            search_provider, result_count
                        )),
                    }
                }
            }
        }
    }
}

pub async fn display_streaming(
    mut rx: mpsc::Receiver<StreamEvent>,
    providers: &[Provider],
    footer: &FooterConfig,
    raw_mode: bool,
    reasoning: Option<String>,
) {
    let mut display = StreamingDisplay::new(providers, footer.clone(), raw_mode, reasoning);

    while let Some(event) = rx.recv().await {
        display.handle_event(event);
    }
}
