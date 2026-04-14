use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

fn heading_style(level: HeadingLevel) -> Style {
    match level {
        HeadingLevel::H1 => Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
        HeadingLevel::H2 => Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
        HeadingLevel::H3 => Style::default()
            .fg(Color::Magenta)
            .add_modifier(Modifier::BOLD),
        _ => Style::default().add_modifier(Modifier::BOLD),
    }
}

enum ListKind {
    Unordered,
    Ordered(u64),
}

struct Renderer {
    lines: Vec<Line<'static>>,
    spans: Vec<Span<'static>>,

    // Inline style state
    bold: bool,
    italic: bool,
    strikethrough: bool,

    // Block state
    heading: Option<HeadingLevel>,
    in_code_block: bool,
    in_blockquote: bool,
    in_link: bool,

    // List state
    list_stack: Vec<ListKind>,
    pending_bullet: bool,

    // Table state
    in_table: bool,
    in_table_head: bool,
    table_row: Vec<String>,
    table_cell: String,
}

impl Renderer {
    fn new() -> Self {
        Self {
            lines: Vec::new(),
            spans: Vec::new(),
            bold: false,
            italic: false,
            strikethrough: false,
            heading: None,
            in_code_block: false,
            in_blockquote: false,
            in_link: false,
            list_stack: Vec::new(),
            pending_bullet: false,
            in_table: false,
            in_table_head: false,
            table_row: Vec::new(),
            table_cell: String::new(),
        }
    }

    fn flush_line(&mut self) {
        self.lines
            .push(Line::from(std::mem::take(&mut self.spans)));
    }

    fn current_style(&self) -> Style {
        let mut style = if let Some(level) = self.heading {
            heading_style(level)
        } else if self.in_blockquote {
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC)
        } else if self.in_link {
            Style::default()
                .fg(Color::Blue)
                .add_modifier(Modifier::UNDERLINED)
        } else {
            Style::default()
        };

        if self.bold {
            style = style.add_modifier(Modifier::BOLD);
        }
        if self.italic {
            style = style.add_modifier(Modifier::ITALIC);
        }
        if self.strikethrough {
            style = style.add_modifier(Modifier::CROSSED_OUT);
        }
        style
    }

    fn push_list_prefix(&mut self) {
        let depth = self.list_stack.len().saturating_sub(1);
        let indent = "  ".repeat(depth);
        let prefix = match self.list_stack.last_mut() {
            Some(ListKind::Ordered(n)) => {
                let p = format!("{}{:>2}. ", indent, n);
                *n += 1;
                p
            }
            Some(ListKind::Unordered) => format!("{}  • ", indent),
            None => return,
        };
        self.spans.push(Span::styled(
            prefix,
            Style::default().fg(Color::Gray),
        ));
    }

    fn maybe_add_bullet(&mut self) {
        if self.pending_bullet && !self.list_stack.is_empty() {
            self.push_list_prefix();
            self.pending_bullet = false;
        }
    }

    fn process(&mut self, event: Event) {
        match event {
            // --- Start tags ---
            Event::Start(Tag::Heading { level, .. }) => {
                if !self.spans.is_empty() {
                    self.flush_line();
                }
                self.heading = Some(level);
            }
            Event::Start(Tag::Strong) => self.bold = true,
            Event::Start(Tag::Emphasis) => self.italic = true,
            Event::Start(Tag::Strikethrough) => self.strikethrough = true,
            Event::Start(Tag::Link { .. }) => self.in_link = true,
            Event::Start(Tag::BlockQuote(_)) => self.in_blockquote = true,

            Event::Start(Tag::CodeBlock(kind)) => {
                self.in_code_block = true;
                if let CodeBlockKind::Fenced(lang) = kind {
                    let lang = lang.to_string();
                    if !lang.is_empty() {
                        self.spans.push(Span::styled(
                            format!("  {}", lang),
                            Style::default().fg(Color::DarkGray),
                        ));
                        self.flush_line();
                    }
                }
            }

            Event::Start(Tag::List(start)) => {
                self.list_stack.push(match start {
                    Some(n) => ListKind::Ordered(n),
                    None => ListKind::Unordered,
                });
            }
            Event::Start(Tag::Item) => {
                self.pending_bullet = true;
            }

            Event::Start(Tag::Paragraph) => {
                self.maybe_add_bullet();
                if self.in_blockquote && self.list_stack.is_empty() {
                    self.spans.push(Span::styled(
                        "│ ",
                        Style::default().fg(Color::DarkGray),
                    ));
                }
            }

            // Table support
            Event::Start(Tag::Table(_)) => self.in_table = true,
            Event::Start(Tag::TableHead) => self.in_table_head = true,
            Event::Start(Tag::TableRow) => self.table_row.clear(),
            Event::Start(Tag::TableCell) => self.table_cell.clear(),

            // --- End tags ---
            Event::End(TagEnd::Heading(_)) => {
                self.flush_line();
                self.heading = None;
            }
            Event::End(TagEnd::Strong) => self.bold = false,
            Event::End(TagEnd::Emphasis) => self.italic = false,
            Event::End(TagEnd::Strikethrough) => self.strikethrough = false,
            Event::End(TagEnd::Link) => self.in_link = false,
            Event::End(TagEnd::BlockQuote) => self.in_blockquote = false,

            Event::End(TagEnd::CodeBlock) => {
                if !self.spans.is_empty() {
                    self.flush_line();
                }
                self.in_code_block = false;
            }

            Event::End(TagEnd::Paragraph) => {
                self.flush_line();
                if self.list_stack.is_empty() {
                    self.lines.push(Line::from(""));
                }
            }

            Event::End(TagEnd::List(_)) => {
                self.list_stack.pop();
                if self.list_stack.is_empty() {
                    self.lines.push(Line::from(""));
                }
            }
            Event::End(TagEnd::Item) => {
                if !self.spans.is_empty() {
                    self.flush_line();
                }
            }

            // Table end tags
            Event::End(TagEnd::TableCell) => {
                self.table_row.push(std::mem::take(&mut self.table_cell));
            }
            Event::End(TagEnd::TableRow) => {
                let row_text = self.table_row.join(" │ ");
                let style = if self.in_table_head {
                    Style::default().add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                self.lines.push(Line::styled(row_text, style));
            }
            Event::End(TagEnd::TableHead) => {
                // Separator line after header. Use .chars().count() rather than
                // .len(): the latter returns byte length, which overcounts for
                // multi-byte Unicode (CJK, emoji) and makes the separator wider
                // than the actual rendered cell.
                let sep = self
                    .table_row
                    .iter()
                    .map(|c| "─".repeat(c.chars().count().max(3)))
                    .collect::<Vec<_>>()
                    .join("─┼─");
                self.lines
                    .push(Line::styled(sep, Style::default().fg(Color::DarkGray)));
                self.in_table_head = false;
            }
            Event::End(TagEnd::Table) => {
                self.in_table = false;
                self.lines.push(Line::from(""));
            }

            // --- Content events ---
            Event::Code(code) => {
                self.maybe_add_bullet();
                self.spans.push(Span::styled(
                    code.to_string(),
                    Style::default().fg(Color::Yellow),
                ));
            }

            Event::Text(text) => {
                let text_str = text.to_string();

                // Table cell text
                if self.in_table {
                    self.table_cell.push_str(&text_str);
                    return;
                }

                self.maybe_add_bullet();

                if self.in_code_block {
                    let code_style = Style::default().fg(Color::Green);
                    let parts: Vec<&str> = text_str.split('\n').collect();
                    // Skip trailing empty part from trailing newline
                    let end = if parts.last() == Some(&"") && parts.len() > 1 {
                        parts.len() - 1
                    } else {
                        parts.len()
                    };
                    for (i, part) in parts[..end].iter().enumerate() {
                        if i > 0 {
                            self.flush_line();
                        }
                        self.spans
                            .push(Span::styled(format!("  {}", part), code_style));
                    }
                } else {
                    self.spans
                        .push(Span::styled(text_str, self.current_style()));
                }
            }

            Event::SoftBreak => {
                if self.in_code_block {
                    self.flush_line();
                } else {
                    self.spans.push(Span::raw(" "));
                }
            }

            Event::HardBreak => {
                self.flush_line();
            }

            Event::Rule => {
                self.flush_line();
                self.lines.push(Line::styled(
                    "─".repeat(40),
                    Style::default().fg(Color::DarkGray),
                ));
                self.lines.push(Line::from(""));
            }

            Event::TaskListMarker(checked) => {
                let marker = if checked { "☑ " } else { "☐ " };
                self.spans.push(Span::styled(
                    marker.to_string(),
                    Style::default().fg(Color::DarkGray),
                ));
            }

            _ => {}
        }
    }

    fn finish(mut self) -> Vec<Line<'static>> {
        if !self.spans.is_empty() {
            self.flush_line();
        }
        self.lines
    }
}

/// Convert a markdown string into styled ratatui lines.
pub fn markdown_to_lines(input: &str) -> Vec<Line<'static>> {
    let opts = Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TABLES | Options::ENABLE_TASKLISTS;
    let parser = Parser::new_ext(input, opts);
    let mut renderer = Renderer::new();
    for event in parser {
        renderer.process(event);
    }
    renderer.finish()
}
