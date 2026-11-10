use crate::config::FooterConfig;
use crate::pricing;
use crate::providers::{Provider, ProviderResponse};
use crate::query::StreamEvent;
use colored::Colorize;
use crossterm::terminal;
use std::io::{self, Write};
use termimad::MadSkin;
use tokio::sync::mpsc;
use unicode_width::UnicodeWidthStr;

/// Separators and header padding stop at this width so very wide terminals
/// don't get a 300-column rule.
const MAX_RULE_WIDTH: usize = 120;

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

fn rule_width() -> usize {
    get_terminal_width().min(MAX_RULE_WIDTH)
}

fn format_duration(duration_ms: u64) -> String {
    if duration_ms >= 1000 {
        format!("{:.1}s", duration_ms as f64 / 1000.0)
    } else {
        format!("{}ms", duration_ms)
    }
}

/// `" NAME ...padding... [status]"`, sized by display width (not bytes) so
/// non-ASCII aliases line up, and to the same width as the separator.
fn header_line(name: &str, status: Option<&str>, width: usize) -> String {
    let header = format!(" {} ", name);
    let Some(status) = status else {
        return header;
    };
    let status = format!("[{}]", status);
    let used = header.width() + status.width();
    if used + 2 <= width {
        format!("{}{}{}", header, " ".repeat(width - used), status)
    } else {
        // Narrow terminal: just show provider name and status without padding
        format!("{} {}", header.trim(), status)
    }
}

fn print_block_header(name: &str, status: Option<&str>, color: colored::Color) {
    let width = rule_width();
    let separator = "━".repeat(width);
    println!();
    println!("{}", separator.color(color));
    println!("{}", header_line(name, status, width).color(color).bold());
    println!("{}", separator.color(color));
}

pub fn print_header(provider: &Provider, status: &str) {
    print_block_header(&provider.display_name, Some(status), provider.color);
}

pub fn format_reasoning(reasoning: Option<&str>) -> String {
    reasoning.unwrap_or("none").to_string()
}

/// Estimated cost of one answer, when cost display is on and both token
/// counts and a price for `model` are known.
fn estimated_cost(
    model: &str,
    input_tokens: Option<i32>,
    output_tokens: Option<i32>,
    footer: &FooterConfig,
) -> Option<f64> {
    if !footer.show_cost {
        return None;
    }
    pricing::estimate_cost(model, input_tokens?, output_tokens?, &footer.pricing)
}

/// The `model: … | tokens: … | cost: … | time: … | reasoning: …` footer, or
/// `None` when every part is switched off.
fn footer_text(
    model: &str,
    input_tokens: Option<i32>,
    output_tokens: Option<i32>,
    duration_ms: u64,
    footer: &FooterConfig,
    reasoning: Option<&str>,
) -> Option<String> {
    let mut parts = Vec::new();

    if footer.show_model {
        parts.push(format!("model: {}", model));
    }

    if footer.show_tokens {
        if let (Some(input), Some(output)) = (input_tokens, output_tokens) {
            parts.push(format!("tokens: {} in / {} out", input, output));
        }
    }

    if let Some(cost) = estimated_cost(model, input_tokens, output_tokens, footer) {
        parts.push(format!("cost: {}", pricing::format_cost(cost)));
    }

    if footer.show_time {
        parts.push(format!("time: {}", format_duration(duration_ms)));
    }

    if footer.show_reasoning {
        parts.push(format!("reasoning: {}", format_reasoning(reasoning)));
    }

    (!parts.is_empty()).then(|| parts.join(" | "))
}

/// Footer settings for an error block: no tokens or cost (there are none)
/// and no reasoning label.
fn error_footer(footer: &FooterConfig) -> FooterConfig {
    FooterConfig {
        show_model: footer.show_model,
        show_time: footer.show_time,
        show_tokens: false,
        show_reasoning: false,
        show_cost: false,
        pricing: Default::default(),
    }
}

/// Running total of the estimated cost of a multi-provider run.
#[derive(Debug, Default)]
struct CostTotal {
    usd: f64,
    /// Answers that had a cost estimate.
    priced: usize,
    /// All answers counted, priced or not (failed providers aren't counted).
    answers: usize,
}

impl CostTotal {
    fn add(&mut self, cost: Option<f64>) {
        self.answers += 1;
        if let Some(cost) = cost {
            self.usd += cost;
            self.priced += 1;
        }
    }

    /// `total: $…`, noting when some answers couldn't be priced (unknown
    /// model or no token counts) so a partial sum isn't mistaken for the
    /// whole. `None` when nothing was priced.
    fn line(&self) -> Option<String> {
        if self.priced == 0 {
            return None;
        }
        let mut line = format!("total: {}", pricing::format_cost(self.usd));
        if self.priced < self.answers {
            line.push_str(&format!(" ({} of {} priced)", self.priced, self.answers));
        }
        Some(line)
    }
}

/// The `total:` line for a completed multi-provider run, or `None` for a
/// single provider or when no answer could be priced.
fn responses_total(responses: &[ProviderResponse], footer: &FooterConfig) -> Option<String> {
    if responses.len() < 2 {
        return None;
    }
    let mut total = CostTotal::default();
    for response in responses.iter().filter(|r| !r.is_error()) {
        total.add(estimated_cost(
            &response.model,
            response.input_tokens,
            response.output_tokens,
            footer,
        ));
    }
    total.line()
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
    if let Some(text) = footer_text(
        model,
        input_tokens,
        output_tokens,
        duration_ms,
        footer,
        reasoning,
    ) {
        println!("{}", text.dimmed());
    }
    println!();
}

pub fn print_error(error: &str) {
    println!("{} {}", "Error:".red().bold(), error.red());
}

pub fn print_response(response: &ProviderResponse, footer: &FooterConfig, reasoning: Option<&str>) {
    if let Some(error) = &response.error {
        print_header(&response.provider, "error");
        print_error(error);
        print_metadata(
            &response.model,
            None,
            None,
            response.duration_ms,
            &error_footer(footer),
            reasoning,
        );
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

pub fn print_responses(
    responses: &[ProviderResponse],
    footer: &FooterConfig,
    reasoning: Option<&str>,
) {
    for response in responses {
        print_response(response, footer, reasoning);
    }
    if let Some(total) = responses_total(responses, footer) {
        println!("{}", total.dimmed());
    }
}

/// Raw, pipeable output. A single response prints exactly the model's text
/// (so `ai --raw "..." | pbcopy` copies just the answer); several responses
/// are labelled and separated so they can be told apart. Errors always go
/// to stderr.
pub fn print_raw_responses(responses: &[ProviderResponse]) {
    let labelled = responses.len() > 1;
    let mut printed_any = false;
    for response in responses {
        if let Some(error) = &response.error {
            eprintln!("[{}] Error: {}", response.provider.display_name, error);
            continue;
        }
        if labelled {
            if printed_any {
                println!("\n---\n");
            }
            println!("[{}]", response.provider.display_name);
        }
        // One trailing newline, matching the streaming raw path: don't add
        // a blank line when the answer already ends with one.
        if response.text.ends_with('\n') {
            print!("{}", response.text);
        } else {
            println!("{}", response.text);
        }
        printed_any = true;
    }
}

/// Escape markdown emphasis characters in text we splice into rendered
/// markdown (e.g. a search query containing `*` or `_` would otherwise
/// break the surrounding italics).
fn escape_markdown(text: &str) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut out = String::with_capacity(collapsed.len());
    for c in collapsed.chars() {
        if matches!(c, '*' | '_' | '`' | '[' | ']' | '\\') {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// The code fence a line opens, as (marker char, run length): per
/// CommonMark, 3+ backticks or tildes, indented at most three spaces.
fn fence_marker(line: &str) -> Option<(char, usize)> {
    let trimmed = line.trim_start_matches(' ');
    if line.len() - trimmed.len() > 3 {
        return None;
    }
    let c = trimmed.chars().next()?;
    if c != '`' && c != '~' {
        return None;
    }
    let run = trimmed.chars().take_while(|&x| x == c).count();
    (run >= 3).then_some((c, run))
}

/// Whether `line` closes a fence opened with `open`: the same character, a
/// run at least as long, and nothing else on the line. A shorter or
/// different fence inside the block (e.g. ``` inside ~~~ or ````) is code.
fn closes_fence(line: &str, open: (char, usize)) -> bool {
    match fence_marker(line) {
        Some((c, run)) => c == open.0 && run >= open.1 && line.trim().chars().all(|x| x == c),
        None => false,
    }
}

/// Find how much of a streaming markdown buffer can be rendered now without
/// changing how it will look once the rest arrives. Returns the byte length
/// of the longest prefix that ends right after a complete line which is
/// followed by a blank line, outside any code fence, where the next
/// non-blank line has also started and isn't indented (an indented line
/// could continue the previous list item or paragraph).
///
/// Everything from the blank line on stays buffered, so the next block
/// renders with its leading blank line exactly as a whole-document render
/// would.
pub fn markdown_flush_point(text: &str) -> Option<usize> {
    // The fence we're inside, if any: (marker char, opening run length).
    let mut fence: Option<(char, usize)> = None;
    let mut flush = None;
    // Byte offset just past the last non-blank line seen outside a fence,
    // if it has been followed by at least one blank line.
    let mut candidate: Option<usize> = None;
    let mut offset = 0;

    for line in text.split_inclusive('\n') {
        let complete = line.ends_with('\n');
        let start = offset;
        offset += line.len();

        if let Some(open) = fence {
            if complete && closes_fence(line, open) {
                fence = None;
            }
            continue;
        }
        if line.trim().is_empty() {
            if !complete {
                break;
            }
            // A blank line after content: the content before it may be a
            // finished block, pending a look at the next line.
            if candidate.is_none() && start > 0 {
                candidate = Some(start);
            }
            continue;
        }
        // A non-blank line: if it follows a blank gap and isn't indented, the
        // block before the gap is complete.
        if let Some(end) = candidate.take() {
            if !line.starts_with(' ') && !line.starts_with('\t') {
                flush = Some(end);
            }
        }
        if !complete {
            break;
        }
        fence = fence_marker(line);
    }
    flush
}

/// Tracks buffered state for a single provider during streaming
struct ProviderBuffer {
    color: colored::Color,
    /// Text received but not yet printed.
    content: String,
    /// Raw mode: whether any text has been written (so Done knows whether
    /// the output still needs a trailing newline).
    wrote_text: bool,
    ends_with_newline: bool,
}

pub struct StreamingDisplay {
    footer: FooterConfig,
    raw_mode: bool,
    /// Render markdown as it arrives (one provider) instead of buffering
    /// each provider's whole answer and printing it when complete (several
    /// providers, whose chunks would otherwise interleave).
    progressive: bool,
    reasoning: Option<String>,
    provider_buffers: std::collections::HashMap<String, ProviderBuffer>,
    // Provider colors lookup
    provider_colors: std::collections::HashMap<String, colored::Color>,
    /// Estimated cost across providers, for the `total:` line printed after
    /// the last one. Only kept for multi-provider (buffered) output.
    total: CostTotal,
}

impl StreamingDisplay {
    pub fn new(
        providers: &[Provider],
        footer: FooterConfig,
        raw_mode: bool,
        reasoning: Option<String>,
    ) -> Self {
        // Build lookup map for provider colors
        let provider_colors: std::collections::HashMap<String, colored::Color> = providers
            .iter()
            .map(|p| (p.display_name.clone(), p.color))
            .collect();

        StreamingDisplay {
            footer,
            raw_mode,
            progressive: providers.len() <= 1,
            reasoning,
            provider_buffers: std::collections::HashMap::new(),
            provider_colors,
            total: CostTotal::default(),
        }
    }

    /// The `total:` line to print once every provider has finished, if this
    /// was a multi-provider run and at least one answer could be priced.
    fn total_line(&self) -> Option<String> {
        self.total.line()
    }

    fn color_for(&self, provider: &str) -> colored::Color {
        self.provider_colors
            .get(provider)
            .copied()
            .unwrap_or(colored::Color::White)
    }

    pub fn handle_event(&mut self, event: StreamEvent) {
        if self.raw_mode {
            self.handle_event_raw(event);
        } else if self.progressive {
            self.handle_event_progressive(event);
        } else {
            self.handle_event_buffered(event);
        }
    }

    /// Raw mode: stdout gets only the model's text, streamed as it arrives;
    /// metadata, search progress and errors go to stderr.
    fn handle_event_raw(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::Start { provider, .. } => {
                let color = self.color_for(&provider);
                self.provider_buffers.insert(
                    provider,
                    ProviderBuffer {
                        color,
                        content: String::new(),
                        wrote_text: false,
                        ends_with_newline: false,
                    },
                );
            }
            StreamEvent::Chunk { provider, text } => {
                print!("{}", text);
                let _ = io::stdout().flush();
                if let Some(b) = self.provider_buffers.get_mut(&provider) {
                    b.wrote_text |= !text.is_empty();
                    if !text.is_empty() {
                        b.ends_with_newline = text.ends_with('\n');
                    }
                }
            }
            StreamEvent::Done {
                provider,
                model,
                duration_ms,
                ..
            } => {
                let needs_newline = self
                    .provider_buffers
                    .remove(&provider)
                    .is_some_and(|b| b.wrote_text && !b.ends_with_newline);
                if needs_newline {
                    println!();
                }
                eprintln!(
                    "[model: {} | time: {} | reasoning: {}]",
                    model,
                    format_duration(duration_ms),
                    format_reasoning(self.reasoning.as_deref())
                );
            }
            StreamEvent::Error {
                provider, error, ..
            } => {
                if self
                    .provider_buffers
                    .remove(&provider)
                    .is_some_and(|b| b.wrote_text && !b.ends_with_newline)
                {
                    println!();
                }
                eprintln!("[{}] Error: {}", provider, error);
            }
            StreamEvent::SearchStart { provider, query } => {
                eprintln!("[{}/search] {}", provider, query);
            }
            StreamEvent::SearchEnd {
                provider,
                search_provider,
                result_count,
                error,
                ..
            } => match error {
                Some(e) => eprintln!("[{}/search] failed: {}", provider, e),
                None => eprintln!(
                    "[{}/search via {}] {} result(s)",
                    provider, search_provider, result_count
                ),
            },
        }
    }

    /// Render and drop every complete markdown block buffered for
    /// `provider`. With `all`, render whatever is left too.
    fn flush_progressive(&mut self, provider: &str, all: bool) {
        let Some(buffer) = self.provider_buffers.get_mut(provider) else {
            return;
        };
        let end = if all {
            Some(buffer.content.len())
        } else {
            markdown_flush_point(&buffer.content)
        };
        if let Some(end) = end {
            let block: String = buffer.content.drain(..end).collect();
            if !block.trim().is_empty() {
                render_markdown(block.trim_end_matches('\n'));
                let _ = io::stdout().flush();
            }
        }
    }

    /// Formatted output for a single provider: the header prints as soon as
    /// the response starts and each finished markdown block (paragraph,
    /// list, table, code block) renders as soon as it's complete, so long
    /// answers appear as they stream instead of all at once at the end.
    fn handle_event_progressive(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::Start { provider, .. } => {
                let color = self.color_for(&provider);
                print_block_header(&provider, None, color);
                self.provider_buffers.insert(
                    provider,
                    ProviderBuffer {
                        color,
                        content: String::new(),
                        wrote_text: false,
                        ends_with_newline: false,
                    },
                );
            }
            StreamEvent::Chunk { provider, text } => {
                if let Some(buffer) = self.provider_buffers.get_mut(&provider) {
                    buffer.content.push_str(&text);
                }
                self.flush_progressive(&provider, false);
            }
            StreamEvent::Done {
                provider,
                model,
                input_tokens,
                output_tokens,
                duration_ms,
            } => {
                self.flush_progressive(&provider, true);
                self.provider_buffers.remove(&provider);
                println!();
                if let Some(text) = footer_text(
                    &model,
                    input_tokens,
                    output_tokens,
                    duration_ms,
                    &self.footer,
                    self.reasoning.as_deref(),
                ) {
                    println!("{}", text.dimmed());
                }
            }
            StreamEvent::Error {
                provider,
                model,
                error,
                duration_ms,
            } => {
                // Whatever already streamed stays on screen; show the rest
                // of what arrived, then the error beneath it.
                self.flush_progressive(&provider, true);
                self.provider_buffers.remove(&provider);
                println!();
                print_error(&error);
                if let Some(text) = footer_text(
                    &model,
                    None,
                    None,
                    duration_ms,
                    &error_footer(&self.footer),
                    None,
                ) {
                    println!("{}", text.dimmed());
                }
            }
            StreamEvent::SearchStart { provider, query } => {
                self.flush_progressive(&provider, true);
                println!("  {} {}", "→ search:".dimmed(), query.italic());
            }
            StreamEvent::SearchEnd {
                provider,
                search_provider,
                result_count,
                error,
                ..
            } => {
                self.flush_progressive(&provider, true);
                match error {
                    Some(e) => println!("  {} {}", "← search failed:".dimmed(), e.red()),
                    None => println!(
                        "  {} {}",
                        "←".dimmed(),
                        format!("via {}: {} result(s)", search_provider, result_count).dimmed()
                    ),
                }
                println!();
            }
        }
    }

    /// Formatted output for several providers at once: each provider's
    /// answer is buffered and printed as one block when it completes, in
    /// completion order, so the fastest provider shows up first.
    fn handle_event_buffered(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::Start { provider, .. } => {
                let color = self.color_for(&provider);
                self.provider_buffers.insert(
                    provider,
                    ProviderBuffer {
                        color,
                        content: String::new(),
                        wrote_text: false,
                        ends_with_newline: false,
                    },
                );
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
                let (content, color) = match self.provider_buffers.remove(&provider) {
                    Some(buffer) => (buffer.content, buffer.color),
                    None => (String::new(), self.color_for(&provider)),
                };

                print_block_header(&provider, Some("complete"), color);
                if !content.is_empty() {
                    render_markdown(&content);
                }
                self.total.add(estimated_cost(
                    &model,
                    input_tokens,
                    output_tokens,
                    &self.footer,
                ));
                if let Some(text) = footer_text(
                    &model,
                    input_tokens,
                    output_tokens,
                    duration_ms,
                    &self.footer,
                    self.reasoning.as_deref(),
                ) {
                    println!("{}", text.dimmed());
                }
            }
            StreamEvent::Error {
                provider,
                model,
                error,
                duration_ms,
            } => {
                // Drop any buffered content for this provider
                let color = self
                    .provider_buffers
                    .remove(&provider)
                    .map(|b| b.color)
                    .unwrap_or_else(|| self.color_for(&provider));

                print_block_header(&provider, Some("error"), color);
                print_error(&error);
                if let Some(text) = footer_text(
                    &model,
                    None,
                    None,
                    duration_ms,
                    &error_footer(&self.footer),
                    None,
                ) {
                    println!("{}", text.dimmed());
                }
            }
            StreamEvent::SearchStart { provider, query } => {
                // Live feedback to stderr so the user sees searches in progress,
                // and append an inline marker into the buffered content for the final render.
                eprintln!("  {} {}", "→ search:".dimmed(), query.italic());
                if let Some(buffer) = self.provider_buffers.get_mut(&provider) {
                    buffer.content.push_str(&format!(
                        "\n\n*→ searching: {}*\n\n",
                        escape_markdown(&query)
                    ));
                }
            }
            StreamEvent::SearchEnd {
                provider,
                search_provider,
                result_count,
                error,
                ..
            } => {
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
                        Some(e) => buffer
                            .content
                            .push_str(&format!("*← search failed: {}*\n\n", escape_markdown(&e))),
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

/// Render streamed events until every provider task has finished. Returns
/// true if any provider failed — reported an error, or ended without ever
/// sending `Done` (e.g. its task panicked) — so the caller can exit non-zero.
pub async fn display_streaming(
    mut rx: mpsc::Receiver<StreamEvent>,
    providers: &[Provider],
    footer: &FooterConfig,
    raw_mode: bool,
    reasoning: Option<String>,
) -> bool {
    let mut display = StreamingDisplay::new(providers, footer.clone(), raw_mode, reasoning);
    let mut any_error = false;
    let mut completed = 0usize;

    while let Some(event) = rx.recv().await {
        match &event {
            StreamEvent::Error { .. } => any_error = true,
            StreamEvent::Done { .. } => completed += 1,
            _ => {}
        }
        display.handle_event(event);
    }
    if let Some(total) = display.total_line() {
        println!();
        println!("{}", total.dimmed());
    }
    any_error || completed < providers.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed `doc` through the flush logic in `chunk`-byte pieces (as a
    /// stream would arrive) and return the blocks that would be rendered.
    fn stream_blocks(doc: &str, chunk: usize) -> Vec<String> {
        let mut pending = String::new();
        let mut blocks = Vec::new();
        let bytes = doc.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let mut j = (i + chunk).min(bytes.len());
            while !doc.is_char_boundary(j) {
                j += 1;
            }
            pending.push_str(&doc[i..j]);
            i = j;
            if let Some(end) = markdown_flush_point(&pending) {
                blocks.push(pending.drain(..end).collect());
            }
        }
        if !pending.is_empty() {
            blocks.push(pending);
        }
        blocks
    }

    fn render(skin: &MadSkin, md: &str) -> String {
        skin.text(md, Some(80)).to_string()
    }

    const DOC: &str = "# Title\n\nFirst paragraph with **bold** text\nthat wraps.\n\n\
        - item one\n- item two\n\n  continued item text\n\n\
        ```rust\nfn main() {\n\n    println!(\"hi\");\n}\n```\n\n\
        | a | b |\n|---|---|\n| 1 | 2 |\n\nLast line.";

    #[test]
    fn progressive_blocks_render_identically_to_the_whole_answer() {
        let skin = get_markdown_skin();
        let whole = render(&skin, DOC);
        for chunk in [1, 3, 7, 16, 1000] {
            let blocks = stream_blocks(DOC, chunk);
            assert!(
                blocks.len() > 1 || chunk == 1000,
                "chunk {}: no streaming",
                chunk
            );
            let joined: String = blocks
                .iter()
                .filter(|b| !b.trim().is_empty())
                .map(|b| render(&skin, b.trim_end_matches('\n')))
                .collect();
            assert_eq!(
                joined.lines().collect::<Vec<_>>(),
                whole.lines().collect::<Vec<_>>(),
                "chunk size {} rendered differently",
                chunk
            );
        }
    }

    #[test]
    fn never_flushes_inside_a_code_fence() {
        let text = "intro\n\n```\nline one\n\nline two\n";
        // Only the intro may flush; the blank line inside the fence must not
        // split the code block.
        assert_eq!(markdown_flush_point(text), Some("intro\n".len()));
        let open = "```\ncode\n\nmore\n";
        assert_eq!(markdown_flush_point(open), None);
    }

    #[test]
    fn inner_fences_of_another_kind_or_length_do_not_close_the_block() {
        // A ``` line inside a ~~~ block, or inside a ```` block, is code.
        let tilde = "~~~\n```\n\nstill code\n";
        assert_eq!(markdown_flush_point(tilde), None);
        let longer = "````md\n```\n\nstill code\n";
        assert_eq!(markdown_flush_point(longer), None);
        // The matching closer ends it, and the next block can flush.
        let closed = "````\n```\n````\n\nNext";
        assert_eq!(
            markdown_flush_point(closed),
            Some("````\n```\n````\n".len())
        );
    }

    #[test]
    fn waits_for_the_next_line_before_flushing() {
        // The line after the blank might be an indented continuation, so
        // nothing flushes until it starts arriving.
        assert_eq!(markdown_flush_point("para\n\n"), None);
        assert_eq!(markdown_flush_point("para\n\n  cont"), None);
        assert_eq!(markdown_flush_point("para\n\nNext"), Some(5));
    }

    /// termimad (minimad) has no ordered lists or reference links: `1.`
    /// markers and `[text]` / `[text]: url` lines render as literal text, so
    /// splitting between them can't renumber items or orphan a link. If the
    /// renderer is ever swapped for a CommonMark one, this fails and the
    /// flush logic needs to keep those blocks together.
    #[test]
    fn loose_ordered_lists_and_reference_links_stream_unchanged() {
        let skin = get_markdown_skin();
        for doc in [
            "1. first\n\n1. second\n\n1. third",
            "Read [docs].\n\n[docs]: https://example.com",
        ] {
            let blocks = stream_blocks(doc, 1);
            assert!(blocks.len() > 1, "{doc:?} didn't stream");
            let joined: String = blocks
                .iter()
                .filter(|b| !b.trim().is_empty())
                .map(|b| render(&skin, b.trim_end_matches('\n')))
                .collect();
            assert_eq!(joined, render(&skin, doc), "{doc:?}");
        }
    }

    #[test]
    fn header_pads_by_display_width_to_the_rule_width() {
        let line = header_line("CLAUDE", Some("complete"), 40);
        assert_eq!(line.width(), 40);
        // Double-width characters must not overflow the rule.
        let line = header_line("日本語", Some("complete"), 40);
        assert_eq!(line.width(), 40);
        assert_eq!(header_line("X", None, 40), " X ");
    }

    #[test]
    fn footer_respects_toggles() {
        let footer = FooterConfig {
            show_model: true,
            show_time: true,
            show_tokens: true,
            show_reasoning: false,
            ..FooterConfig::default()
        };
        let text = footer_text("m", Some(3), Some(5), 1500, &footer, None).unwrap();
        assert_eq!(text, "model: m | tokens: 3 in / 5 out | time: 1.5s");
        let none = FooterConfig {
            show_model: false,
            show_time: false,
            show_tokens: false,
            show_reasoning: false,
            show_cost: false,
            ..FooterConfig::default()
        };
        assert!(footer_text("claude-haiku-4-5", Some(3), Some(5), 1, &none, None).is_none());
    }

    fn cost_footer() -> FooterConfig {
        FooterConfig {
            show_model: true,
            show_time: true,
            show_tokens: true,
            show_reasoning: false,
            ..FooterConfig::default()
        }
    }

    #[test]
    fn footer_shows_cost_when_tokens_and_price_are_known() {
        let footer = cost_footer();
        // Haiku 4.5: 1000 × $1/M + 500 × $5/M = $0.0035.
        let text = footer_text(
            "claude-haiku-4-5",
            Some(1000),
            Some(500),
            1500,
            &footer,
            None,
        )
        .unwrap();
        assert_eq!(
            text,
            "model: claude-haiku-4-5 | tokens: 1000 in / 500 out | cost: $0.0035 | time: 1.5s"
        );

        // Cost doesn't depend on showing the token counts.
        let quiet = FooterConfig {
            show_tokens: false,
            ..cost_footer()
        };
        let text = footer_text(
            "claude-haiku-4-5",
            Some(1000),
            Some(500),
            1500,
            &quiet,
            None,
        )
        .unwrap();
        assert_eq!(text, "model: claude-haiku-4-5 | cost: $0.0035 | time: 1.5s");
    }

    #[test]
    fn footer_omits_cost_when_unknown_or_disabled() {
        let footer = cost_footer();
        // Unknown model: no cost, no crash.
        let text = footer_text("qwen3:8b", Some(1000), Some(500), 1500, &footer, None).unwrap();
        assert_eq!(
            text,
            "model: qwen3:8b | tokens: 1000 in / 500 out | time: 1.5s"
        );
        // Known model but no usage reported (e.g. a stream without usage).
        let text = footer_text("claude-haiku-4-5", None, None, 1500, &footer, None).unwrap();
        assert_eq!(text, "model: claude-haiku-4-5 | time: 1.5s");
        // Switched off (`show_cost = false` / `--no-cost`).
        let off = FooterConfig {
            show_cost: false,
            ..cost_footer()
        };
        let text =
            footer_text("claude-haiku-4-5", Some(1000), Some(500), 1500, &off, None).unwrap();
        assert!(!text.contains("cost"), "{text}");
        // Error footers never show a cost.
        assert!(!error_footer(&footer).show_cost);
    }

    #[test]
    fn footer_cost_uses_config_overrides() {
        let mut footer = cost_footer();
        footer.pricing.insert(
            "qwen3:8b".to_string(),
            pricing::ModelPrice {
                input_per_mtok: 1.0,
                output_per_mtok: 2.0,
            },
        );
        let text = footer_text("qwen3:8b", Some(1000), Some(1000), 1500, &footer, None).unwrap();
        assert!(text.contains("cost: $0.003 |"), "{text}");
    }

    fn provider(name: &str, model: &str) -> Provider {
        Provider {
            name: name.to_lowercase(),
            display_name: name.to_string(),
            model: model.to_string(),
            reasoning: None,
            service_tier: None,
            color: colored::Color::White,
        }
    }

    #[test]
    fn multi_provider_total_sums_priced_answers() {
        let footer = cost_footer();
        let ok = |name: &str, model: &str, input, output| {
            ProviderResponse::success(
                provider(name, model),
                model.to_string(),
                "answer".to_string(),
                input,
                output,
                10,
            )
        };
        // $0.0035 (Haiku) + 1000 × $0.05/M + 1000 × $0.40/M = $0.00045 (nano).
        let both = [
            ok("CLAUDE", "claude-haiku-4-5", Some(1000), Some(500)),
            ok("GPT", "gpt-5-nano", Some(1000), Some(1000)),
        ];
        assert_eq!(
            responses_total(&both, &footer).as_deref(),
            Some("total: $0.00395")
        );

        // An unpriced answer is called out; a failed provider isn't counted.
        let mixed = [
            ok("CLAUDE", "claude-haiku-4-5", Some(1000), Some(500)),
            ok("LOCAL", "qwen3:8b", Some(1000), Some(500)),
            ProviderResponse::error(
                provider("GROK", "grok-4.3"),
                "grok-4.3".to_string(),
                "boom".to_string(),
                10,
            ),
        ];
        assert_eq!(
            responses_total(&mixed, &footer).as_deref(),
            Some("total: $0.0035 (1 of 2 priced)")
        );

        // Nothing priced, a single provider, or cost switched off: no line.
        let unpriced = [
            ok("LOCAL", "qwen3:8b", Some(1), Some(1)),
            ok("OTHER", "mystery", Some(1), Some(1)),
        ];
        assert_eq!(responses_total(&unpriced, &footer), None);
        assert_eq!(responses_total(&both[..1], &footer), None);
        let off = FooterConfig {
            show_cost: false,
            ..cost_footer()
        };
        assert_eq!(responses_total(&both, &off), None);
    }

    #[test]
    fn streaming_total_covers_every_finished_provider() {
        let providers = [
            provider("CLAUDE", "claude-haiku-4-5"),
            provider("GPT", "gpt-5-nano"),
        ];
        let mut display = StreamingDisplay::new(&providers, cost_footer(), false, None);
        for p in &providers {
            display.handle_event(StreamEvent::Start {
                provider: p.display_name.clone(),
                model: p.model.clone(),
            });
        }
        display.handle_event(StreamEvent::Done {
            provider: "GPT".to_string(),
            model: "gpt-5-nano".to_string(),
            input_tokens: Some(1000),
            output_tokens: Some(1000),
            duration_ms: 10,
        });
        display.handle_event(StreamEvent::Done {
            provider: "CLAUDE".to_string(),
            model: "claude-haiku-4-5".to_string(),
            input_tokens: Some(1000),
            output_tokens: Some(500),
            duration_ms: 10,
        });
        assert_eq!(display.total_line().as_deref(), Some("total: $0.00395"));

        // A single provider streams progressively and never prints a total.
        let mut single = StreamingDisplay::new(&providers[..1], cost_footer(), false, None);
        single.handle_event(StreamEvent::Done {
            provider: "CLAUDE".to_string(),
            model: "claude-haiku-4-5".to_string(),
            input_tokens: Some(1000),
            output_tokens: Some(500),
            duration_ms: 10,
        });
        assert_eq!(single.total_line(), None);
    }

    #[test]
    fn escape_markdown_neutralizes_emphasis() {
        assert_eq!(escape_markdown("rust *vec* _x_"), "rust \\*vec\\* \\_x\\_");
        assert_eq!(escape_markdown("a  \n b"), "a b");
    }
}
