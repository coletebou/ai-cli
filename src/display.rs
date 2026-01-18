use crate::config::FooterConfig;
use crate::providers::{Provider, ProviderResponse};
use crate::query::StreamEvent;
use colored::Colorize;
use crossterm::terminal;
use std::collections::HashMap;
use std::io::{self, Write};
use tokio::sync::mpsc;

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
        println!("{}", response.text);
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

pub struct StreamingDisplay {
    current_provider: Option<String>,
    provider_colors: HashMap<String, colored::Color>,
    footer: FooterConfig,
    raw_mode: bool,
    reasoning: Option<String>,
}

impl StreamingDisplay {
    pub fn new(providers: &[Provider], footer: FooterConfig, raw_mode: bool, reasoning: Option<String>) -> Self {
        let provider_colors: HashMap<String, colored::Color> = providers
            .iter()
            .map(|p| (p.display_name.clone(), p.color))
            .collect();

        StreamingDisplay {
            current_provider: None,
            provider_colors,
            footer,
            raw_mode,
            reasoning,
        }
    }

    fn get_color(&self, provider: &str) -> colored::Color {
        self.provider_colors
            .get(provider)
            .copied()
            .unwrap_or(colored::Color::White)
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
                if self.current_provider.is_some() {
                    println!("\n\n---\n");
                }
                println!("[{}]", provider);
                self.current_provider = Some(provider);
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
        }
    }

    fn handle_event_formatted(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::Start { provider, .. } => {
                let color = self.get_color(&provider);
                let width = get_terminal_width();
                let separator = make_separator(width);

                let header = format!(" {} ", provider);
                let status_str = "[streaming]";
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

                self.current_provider = Some(provider);
            }
            StreamEvent::Chunk { text, .. } => {
                print!("{}", text);
                let _ = io::stdout().flush();
            }
            StreamEvent::Done {
                model,
                input_tokens,
                output_tokens,
                duration_ms,
                ..
            } => {
                println!();
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
                println!();
            }
            StreamEvent::Error { provider, model, error, duration_ms } => {
                let color = self.get_color(&provider);

                // If we haven't printed a header for this provider yet
                if self.current_provider.as_ref() != Some(&provider) {
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
                }

                println!("{} {}", "Error:".red().bold(), error.red());
                let duration_str = if duration_ms >= 1000 {
                    format!("{:.1}s", duration_ms as f64 / 1000.0)
                } else {
                    format!("{}ms", duration_ms)
                };
                println!("{}", format!("model: {} | time: {}", model, duration_str).dimmed());
                println!();
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
