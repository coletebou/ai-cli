use crate::providers::{Provider, ProviderResponse};
use crate::query::StreamEvent;
use colored::Colorize;
use std::collections::HashMap;
use std::io::{self, Write};
use tokio::sync::mpsc;

const SEPARATOR: &str = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";

pub fn print_header(provider: &Provider, status: &str) {
    let header = format!(" {} ", provider.display_name);
    let status_str = format!("[{}]", status);
    let padding = SEPARATOR.len().saturating_sub(header.len() + status_str.len() + 2);
    let padded_header = format!("{}{:>width$}", header, status_str, width = padding + status_str.len());

    println!();
    println!("{}", SEPARATOR.color(provider.color));
    println!("{}", padded_header.color(provider.color).bold());
    println!("{}", SEPARATOR.color(provider.color));
}

pub fn print_footer(input_tokens: Option<i32>, output_tokens: Option<i32>, show_tokens: bool) {
    if show_tokens {
        if let (Some(input), Some(output)) = (input_tokens, output_tokens) {
            let token_info = format!("tokens: {} in / {} out", input, output);
            println!();
            println!("{:>64}", token_info.dimmed());
        }
    }
    println!();
}

pub fn print_error(error: &str) {
    println!("{} {}", "Error:".red().bold(), error.red());
}

pub fn print_response(response: &ProviderResponse, show_tokens: bool) {
    if response.is_error() {
        print_header(&response.provider, "error");
        print_error(response.error.as_ref().unwrap());
    } else {
        print_header(&response.provider, "complete");
        println!("{}", response.text);
        print_footer(response.input_tokens, response.output_tokens, show_tokens);
    }
}

pub fn print_responses(responses: &[ProviderResponse], show_tokens: bool) {
    for response in responses {
        print_response(response, show_tokens);
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
    show_tokens: bool,
    raw_mode: bool,
}

impl StreamingDisplay {
    pub fn new(providers: &[Provider], show_tokens: bool, raw_mode: bool) -> Self {
        let provider_colors: HashMap<String, colored::Color> = providers
            .iter()
            .map(|p| (p.display_name.clone(), p.color))
            .collect();

        StreamingDisplay {
            current_provider: None,
            provider_colors,
            show_tokens,
            raw_mode,
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
            StreamEvent::Start { provider } => {
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
            StreamEvent::Done { .. } => {
                println!();
            }
            StreamEvent::Error { provider, error } => {
                eprintln!("[{}] Error: {}", provider, error);
            }
        }
    }

    fn handle_event_formatted(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::Start { provider } => {
                let color = self.get_color(&provider);
                let header = format!(" {} ", provider);
                let status_str = "[streaming]";
                let padding = SEPARATOR.len().saturating_sub(header.len() + status_str.len() + 2);
                let padded_header = format!("{}{:>width$}", header, status_str, width = padding + status_str.len());

                println!();
                println!("{}", SEPARATOR.color(color));
                println!("{}", padded_header.color(color).bold());
                println!("{}", SEPARATOR.color(color));

                self.current_provider = Some(provider);
            }
            StreamEvent::Chunk { text, .. } => {
                print!("{}", text);
                let _ = io::stdout().flush();
            }
            StreamEvent::Done {
                input_tokens,
                output_tokens,
                ..
            } => {
                if self.show_tokens {
                    if let (Some(input), Some(output)) = (input_tokens, output_tokens) {
                        let token_info = format!("tokens: {} in / {} out", input, output);
                        println!();
                        println!();
                        println!("{:>64}", token_info.dimmed());
                    }
                }
                println!();
            }
            StreamEvent::Error { provider, error } => {
                let color = self.get_color(&provider);

                // If we haven't printed a header for this provider yet
                if self.current_provider.as_ref() != Some(&provider) {
                    let header = format!(" {} ", provider);
                    let status_str = "[error]";
                    let padding = SEPARATOR.len().saturating_sub(header.len() + status_str.len() + 2);
                    let padded_header = format!("{}{:>width$}", header, status_str, width = padding + status_str.len());

                    println!();
                    println!("{}", SEPARATOR.color(color));
                    println!("{}", padded_header.color(color).bold());
                    println!("{}", SEPARATOR.color(color));
                }

                println!("{} {}", "Error:".red().bold(), error.red());
                println!();
            }
        }
    }
}

pub async fn display_streaming(
    mut rx: mpsc::Receiver<StreamEvent>,
    providers: &[Provider],
    show_tokens: bool,
    raw_mode: bool,
) {
    let mut display = StreamingDisplay::new(providers, show_tokens, raw_mode);

    while let Some(event) = rx.recv().await {
        display.handle_event(event);
    }
}
