use pulldown_cmark::{Alignment, CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

/// Columns per tab stop when expanding '\t'. ratatui measures a tab as one
/// column, but the terminal jumps to the next tab stop, so a raw tab makes
/// ratatui's idea of the cursor drift from the terminal's and garbles the
/// rest of the panel.
const TAB_WIDTH: usize = 4;

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

/// Style for chrome that isn't content: quote bars, table rules, code
/// block language labels, link URLs.
fn dim() -> Style {
    Style::default().fg(Color::DarkGray)
}

/// Display width of a run of spans (unicode-aware, so CJK and emoji count
/// as two columns).
fn width_of(spans: &[Span<'_>]) -> usize {
    spans.iter().map(Span::width).sum()
}

/// Replace each '\t' with spaces up to the next tab stop. `start_col` is the
/// column the text starts at; it's only computed when there is a tab.
fn expand_tabs(text: &str, start_col: impl FnOnce() -> usize) -> String {
    if !text.contains('\t') {
        return text.to_owned();
    }
    let start_col = start_col();
    let mut out = String::with_capacity(text.len() + TAB_WIDTH);
    for ch in text.chars() {
        if ch == '\t' {
            let col = start_col + Span::raw(out.as_str()).width();
            out.push_str(&" ".repeat(TAB_WIDTH - col % TAB_WIDTH));
        } else {
            out.push(ch);
        }
    }
    out
}

enum ListKind {
    Unordered,
    Ordered(u64),
}

/// A block that prefixes every line rendered inside it. Kept as a stack
/// (outermost first) so nesting composes: a list inside a quote renders as
/// "│   • item", and a nested quote as "│ │ text".
enum Container {
    BlockQuote,
    /// `marker` is the bullet or number shown on the item's first line. Later
    /// lines of the same item (nested lists, code blocks, hard breaks) are
    /// indented by the marker's width so they line up under the item text.
    Item {
        marker: String,
        marker_shown: bool,
    },
}

/// The link being rendered. Its URL is appended after the text on
/// `End(Link)`, unless the text already is the URL (autolinks).
struct LinkState {
    url: String,
    text: String,
}

type Cell = Vec<Span<'static>>;

/// A table is buffered whole so column widths are known before anything is
/// emitted. Cells are kept as styled spans so inline code, emphasis and
/// links inside a cell keep their styling.
#[derive(Default)]
struct TableState {
    alignments: Vec<Alignment>,
    header: Vec<Cell>,
    rows: Vec<Vec<Cell>>,
    row: Vec<Cell>,
    cell: Cell,
}

struct Renderer {
    lines: Vec<Line<'static>>,
    /// The line being built, container prefix included.
    spans: Vec<Span<'static>>,
    /// Whether a line is in progress (its prefix has already been pushed).
    line_open: bool,
    /// How many leading entries of `spans` are prefix rather than content.
    prefix_len: usize,
    /// A blank line is owed before the next line. It's emitted lazily so it
    /// never doubles up and carries the right quote bars; the value is the
    /// quote depth at the block that asked for it.
    pending_blank: Option<usize>,

    // Inline style state
    bold: bool,
    italic: bool,
    strikethrough: bool,

    // Block state
    heading: Option<HeadingLevel>,
    in_code_block: bool,
    link: Option<LinkState>,
    containers: Vec<Container>,
    list_stack: Vec<ListKind>,
    table: Option<TableState>,
}

impl Renderer {
    fn new() -> Self {
        Self {
            lines: Vec::new(),
            spans: Vec::new(),
            line_open: false,
            prefix_len: 0,
            pending_blank: None,
            bold: false,
            italic: false,
            strikethrough: false,
            heading: None,
            in_code_block: false,
            link: None,
            containers: Vec::new(),
            list_stack: Vec::new(),
            table: None,
        }
    }

    /// Blockquote nesting depth. A depth rather than a flag so the outer
    /// quote keeps its styling after a nested quote closes.
    fn quote_depth(&self) -> usize {
        self.containers
            .iter()
            .filter(|c| matches!(c, Container::BlockQuote))
            .count()
    }

    /// Start a line if none is in progress: emit any owed blank line, then
    /// the prefix each enclosing quote and list item contributes. Every line
    /// goes through here, which is what keeps "│ " on lines after a hard
    /// break and indents code blocks inside list items.
    fn open_line(&mut self) {
        if self.line_open {
            return;
        }
        let depth = self.quote_depth();
        if let Some(blank_depth) = self.pending_blank.take() {
            let bars = blank_depth.min(depth);
            self.lines.push(if bars == 0 {
                Line::from("")
            } else {
                Line::styled(vec!["│"; bars].join(" "), dim())
            });
        }
        for container in &mut self.containers {
            match container {
                Container::BlockQuote => self.spans.push(Span::styled("│ ", dim())),
                Container::Item {
                    marker,
                    marker_shown,
                } => {
                    if *marker_shown {
                        let indent = Span::raw(marker.as_str()).width();
                        self.spans.push(Span::raw(" ".repeat(indent)));
                    } else {
                        self.spans.push(Span::styled(
                            marker.clone(),
                            Style::default().fg(Color::Gray),
                        ));
                        *marker_shown = true;
                    }
                }
            }
        }
        if self.in_code_block {
            self.spans.push(Span::raw("  "));
        }
        self.prefix_len = self.spans.len();
        self.line_open = true;
    }

    /// Finish the current line (starting one first, so an empty line still
    /// gets its prefix).
    fn end_line(&mut self) {
        self.open_line();
        self.lines.push(Line::from(std::mem::take(&mut self.spans)));
        self.line_open = false;
        self.prefix_len = 0;
    }

    /// Finish the current line if one is in progress. Every block start
    /// calls this: tight lists emit no Paragraph events, so without it a
    /// nested list or code block would be appended to the parent item's
    /// text.
    fn close_line(&mut self) {
        if self.line_open {
            self.end_line();
        }
    }

    /// Owe a blank line after a block. Blocks inside list items stay tight,
    /// matching how the list itself is rendered.
    fn block_gap(&mut self) {
        if self.list_stack.is_empty() {
            let depth = self.quote_depth();
            self.pending_blank = Some(self.pending_blank.map_or(depth, |d| d.min(depth)));
        }
    }

    fn current_style(&self) -> Style {
        let mut style = if let Some(level) = self.heading {
            heading_style(level)
        } else if self.quote_depth() > 0 {
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC)
        } else if self.link.is_some() {
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

    /// Append inline content to the current table cell, or to the current
    /// line. All inline events go through here so none of them can escape a
    /// table cell.
    fn push_inline(&mut self, text: &str, style: Style) {
        if text.is_empty() {
            return;
        }
        if let Some(table) = &mut self.table {
            let content = expand_tabs(text, || width_of(&table.cell));
            table.cell.push(Span::styled(content, style));
            return;
        }
        self.open_line();
        let content = expand_tabs(text, || width_of(&self.spans[self.prefix_len..]));
        self.spans.push(Span::styled(content, style));
    }

    /// Render text whose line breaks are significant (code and HTML blocks).
    /// Every '\n' ends the line, including a trailing one: inside list items
    /// and quotes pulldown-cmark emits each source line as its own event
    /// ("cargo build\n", "cargo test\n"), and CRLF input splits lines across
    /// events ("a", "\nb", "\n"), so an event boundary isn't a line boundary.
    fn push_block_text(&mut self, text: &str, style: Style) {
        for chunk in text.split_inclusive('\n') {
            match chunk.strip_suffix('\n') {
                Some(line) => {
                    self.push_inline(line.strip_suffix('\r').unwrap_or(line), style);
                    self.end_line();
                }
                None => self.push_inline(chunk, style),
            }
        }
    }

    fn render_table(&mut self, table: TableState) {
        let TableState {
            alignments,
            header,
            rows,
            ..
        } = table;
        let columns = std::iter::once(&header)
            .chain(&rows)
            .map(Vec::len)
            .max()
            .unwrap_or(0);
        // Minimum of 3 so a column of one-character cells still gets a
        // visible rule.
        let mut widths = vec![3; columns];
        for row in std::iter::once(&header).chain(&rows) {
            for (width, cell) in widths.iter_mut().zip(row) {
                *width = (*width).max(width_of(cell));
            }
        }

        let header = header
            .into_iter()
            .map(|cell| {
                cell.into_iter()
                    .map(|mut span| {
                        span.style = span.style.add_modifier(Modifier::BOLD);
                        span
                    })
                    .collect()
            })
            .collect();
        self.push_table_row(header, &widths, &alignments);
        let rule = widths
            .iter()
            .map(|w| "─".repeat(*w))
            .collect::<Vec<_>>()
            .join("─┼─");
        self.push_inline(&rule, dim());
        self.end_line();
        for row in rows {
            self.push_table_row(row, &widths, &alignments);
        }
    }

    fn push_table_row(&mut self, cells: Vec<Cell>, widths: &[usize], alignments: &[Alignment]) {
        self.open_line();
        let mut cells = cells.into_iter();
        for (i, &width) in widths.iter().enumerate() {
            if i > 0 {
                self.spans.push(Span::styled(" │ ", dim()));
            }
            let cell = cells.next().unwrap_or_default();
            let pad = width.saturating_sub(width_of(&cell));
            let (left, mut right) = match alignments.get(i) {
                Some(Alignment::Right) => (pad, 0),
                Some(Alignment::Center) => (pad / 2, pad - pad / 2),
                _ => (0, pad),
            };
            // Trailing padding on the last column is invisible and only makes
            // the line wider than it looks, which can cause a spurious wrap.
            if i + 1 == widths.len() {
                right = 0;
            }
            if left > 0 {
                self.spans.push(Span::raw(" ".repeat(left)));
            }
            self.spans.extend(cell);
            if right > 0 {
                self.spans.push(Span::raw(" ".repeat(right)));
            }
        }
        self.end_line();
    }

    fn process(&mut self, event: Event) {
        match event {
            // --- Start tags ---
            Event::Start(Tag::Heading { level, .. }) => {
                self.close_line();
                self.heading = Some(level);
            }
            Event::Start(Tag::Strong) => self.bold = true,
            Event::Start(Tag::Emphasis) => self.italic = true,
            Event::Start(Tag::Strikethrough) => self.strikethrough = true,
            Event::Start(Tag::Link { dest_url, .. }) => {
                self.link = Some(LinkState {
                    url: dest_url.to_string(),
                    text: String::new(),
                });
            }
            Event::Start(Tag::BlockQuote(_)) => {
                self.close_line();
                self.containers.push(Container::BlockQuote);
            }

            Event::Start(Tag::CodeBlock(kind)) => {
                // Flush first, or the language label lands on the end of the
                // enclosing list item's text.
                self.close_line();
                self.in_code_block = true;
                if let CodeBlockKind::Fenced(lang) = kind {
                    if !lang.is_empty() {
                        self.push_inline(&lang, dim());
                        self.end_line();
                    }
                }
            }
            Event::Start(Tag::HtmlBlock) => self.close_line(),

            Event::Start(Tag::List(start)) => {
                self.close_line();
                self.list_stack.push(match start {
                    Some(n) => ListKind::Ordered(n),
                    None => ListKind::Unordered,
                });
            }
            Event::Start(Tag::Item) => {
                self.close_line();
                // Top-level markers keep a left margin; nested markers start
                // at the parent item's text column.
                let nested = self.list_stack.len() > 1;
                let marker = match self.list_stack.last_mut() {
                    Some(ListKind::Ordered(n)) => {
                        let marker = if nested {
                            format!("{n}. ")
                        } else {
                            format!("{n:>2}. ")
                        };
                        *n += 1;
                        marker
                    }
                    Some(ListKind::Unordered) | None => {
                        if nested { "• " } else { "  • " }.to_string()
                    }
                };
                self.containers.push(Container::Item {
                    marker,
                    marker_shown: false,
                });
            }

            Event::Start(Tag::Paragraph) => self.close_line(),

            Event::Start(Tag::Table(alignments)) => {
                self.close_line();
                self.table = Some(TableState {
                    alignments,
                    ..TableState::default()
                });
            }
            Event::Start(Tag::TableHead | Tag::TableRow) => {
                if let Some(table) = &mut self.table {
                    table.row.clear();
                }
            }
            Event::Start(Tag::TableCell) => {
                if let Some(table) = &mut self.table {
                    table.cell.clear();
                }
            }

            // --- End tags ---
            Event::End(TagEnd::Heading(_)) => {
                self.close_line();
                self.heading = None;
            }
            Event::End(TagEnd::Strong) => self.bold = false,
            Event::End(TagEnd::Emphasis) => self.italic = false,
            Event::End(TagEnd::Strikethrough) => self.strikethrough = false,
            Event::End(TagEnd::Link) => {
                if let Some(link) = self.link.take() {
                    // Autolinks (<https://…>, <a@b.c>) already show the URL.
                    if !link.url.is_empty() && link.text != link.url {
                        let suffix = if link.text.is_empty() {
                            link.url
                        } else {
                            format!(" ({})", link.url)
                        };
                        self.push_inline(&suffix, dim());
                    }
                }
            }
            Event::End(TagEnd::BlockQuote) => {
                self.close_line();
                self.containers.pop();
            }

            Event::End(TagEnd::CodeBlock) => {
                self.close_line();
                self.in_code_block = false;
                self.block_gap();
            }
            Event::End(TagEnd::HtmlBlock) | Event::End(TagEnd::Paragraph) => {
                self.close_line();
                self.block_gap();
            }

            Event::End(TagEnd::List(_)) => {
                self.close_line();
                self.list_stack.pop();
                self.block_gap();
            }
            Event::End(TagEnd::Item) => {
                // An empty item still shows its bullet.
                if let Some(Container::Item {
                    marker_shown: false,
                    ..
                }) = self.containers.last()
                {
                    self.open_line();
                }
                self.close_line();
                self.containers.pop();
            }

            Event::End(TagEnd::TableCell) => {
                if let Some(table) = &mut self.table {
                    let cell = std::mem::take(&mut table.cell);
                    table.row.push(cell);
                }
            }
            // pulldown-cmark doesn't wrap the header cells in a TableRow, so
            // the header has to be captured here.
            Event::End(TagEnd::TableHead) => {
                if let Some(table) = &mut self.table {
                    table.header = std::mem::take(&mut table.row);
                }
            }
            Event::End(TagEnd::TableRow) => {
                if let Some(table) = &mut self.table {
                    let row = std::mem::take(&mut table.row);
                    table.rows.push(row);
                }
            }
            Event::End(TagEnd::Table) => {
                if let Some(table) = self.table.take() {
                    self.render_table(table);
                }
                self.block_gap();
            }

            // --- Content events ---
            Event::Code(code) => {
                if let Some(link) = &mut self.link {
                    link.text.push_str(&code);
                }
                self.push_inline(&code, Style::default().fg(Color::Yellow));
            }

            Event::Text(text) => {
                if let Some(link) = &mut self.link {
                    link.text.push_str(&text);
                }
                if self.in_code_block {
                    self.push_block_text(&text, Style::default().fg(Color::Green));
                } else {
                    let style = self.current_style();
                    self.push_inline(&text, style);
                }
            }

            // Raw HTML is shown literally: in LLM output it's far more often
            // prose like `Vec<String>` than markup meant to be interpreted.
            Event::InlineHtml(html) => {
                let style = self.current_style();
                self.push_inline(&html, style);
            }
            Event::Html(html) => {
                let style = self.current_style();
                self.push_block_text(&html, style);
            }

            Event::SoftBreak => self.push_inline(" ", Style::default()),
            Event::HardBreak => self.end_line(),

            Event::Rule => {
                self.close_line();
                self.push_inline(&"─".repeat(40), dim());
                self.end_line();
                self.block_gap();
            }

            Event::TaskListMarker(checked) => {
                let marker = if checked { "☑ " } else { "☐ " };
                self.push_inline(marker, dim());
            }

            _ => {}
        }
    }

    fn finish(mut self) -> Vec<Line<'static>> {
        self.close_line();
        if self.pending_blank.is_some() {
            self.lines.push(Line::from(""));
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Render and flatten each line to its plain text, so layout can be
    /// asserted without spelling out every span.
    fn render(md: &str) -> Vec<String> {
        markdown_to_lines(md)
            .iter()
            .map(|line| line.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect()
    }

    /// Style of the first span whose text is exactly `text`.
    fn span_style(md: &str, text: &str) -> Style {
        markdown_to_lines(md)
            .iter()
            .flat_map(|line| line.spans.iter())
            .find(|s| s.content == text)
            .map(|s| s.style)
            .unwrap_or_else(|| panic!("no span {text:?} in {:?}", render(md)))
    }

    // --- Headings, paragraphs, inline styles ---

    #[test]
    fn heading_gets_its_own_line_and_heading_style() {
        assert_eq!(render("# Title\nBody text\n"), ["Title", "Body text", ""]);
        let style = span_style("# Title\n", "Title");
        assert_eq!(style.fg, Some(Color::Cyan));
        assert!(style
            .add_modifier
            .contains(Modifier::BOLD | Modifier::UNDERLINED));
    }

    #[test]
    fn paragraphs_join_soft_breaks_and_are_separated_by_blank_lines() {
        assert_eq!(render("one\ntwo\n\nthree\n"), ["one two", "", "three", ""]);
    }

    #[test]
    fn inline_emphasis_and_code_are_styled() {
        let md = "**bold** *italic* ~~gone~~ `code`\n";
        assert_eq!(render(md), ["bold italic gone code", ""]);
        assert!(span_style(md, "bold").add_modifier.contains(Modifier::BOLD));
        assert!(span_style(md, "italic")
            .add_modifier
            .contains(Modifier::ITALIC));
        assert!(span_style(md, "gone")
            .add_modifier
            .contains(Modifier::CROSSED_OUT));
        assert_eq!(span_style(md, "code").fg, Some(Color::Yellow));
    }

    #[test]
    fn tabs_expand_to_tab_stops() {
        assert_eq!(render("a\tb\n"), ["a   b", ""]);
        assert_eq!(render("```\nab\tc\n\td\n```\n"), ["  ab  c", "      d", ""]);
    }

    #[test]
    fn rule_renders_as_a_separator_line() {
        let rule = "─".repeat(40);
        assert_eq!(
            render("above\n\n---\n\nbelow\n"),
            ["above", "", rule.as_str(), "", "below", ""]
        );
    }

    // --- Lists ---

    #[test]
    fn nested_list_items_start_on_their_own_line() {
        assert_eq!(
            render("- a\n  - b\n- c\n"),
            ["  • a", "    • b", "  • c", ""]
        );
    }

    #[test]
    fn ordered_lists_count_from_their_start_number() {
        assert_eq!(
            render("3. three\n4. four\n   1. sub\n"),
            [" 3. three", " 4. four", "    1. sub", ""]
        );
    }

    #[test]
    fn loose_list_with_nested_list_stays_tight() {
        assert_eq!(
            render("1. first\n\n   - sub a\n   - sub b\n\n2. second\n"),
            [" 1. first", "    • sub a", "    • sub b", " 2. second", ""]
        );
    }

    #[test]
    fn list_is_separated_from_surrounding_paragraphs() {
        assert_eq!(
            render("Intro:\n\n- a\n\nOutro\n"),
            ["Intro:", "", "  • a", "", "Outro", ""]
        );
    }

    #[test]
    fn empty_list_item_keeps_its_bullet() {
        assert_eq!(render("- \n- b\n"), ["  • ", "  • b", ""]);
    }

    #[test]
    fn task_checkbox_follows_the_bullet() {
        assert_eq!(
            render("- [ ] todo\n- [x] done\n  - [ ] nested\n"),
            ["  • ☐ todo", "  • ☑ done", "    • ☐ nested", ""]
        );
    }

    // --- Code blocks ---

    #[test]
    fn top_level_code_block_has_label_and_indent() {
        let md = "```rust\nfn main() {\n\tx();\n}\n```\n";
        assert_eq!(
            render(md),
            ["  rust", "  fn main() {", "      x();", "  }", ""]
        );
        assert_eq!(span_style(md, "fn main() {").fg, Some(Color::Green));
    }

    #[test]
    fn code_block_keeps_blank_lines_and_handles_crlf() {
        assert_eq!(render("```\na\n\nb\n```\n"), ["  a", "  ", "  b", ""]);
        assert_eq!(render("```\r\na\r\nb\r\n```\r\n"), ["  a", "  b", ""]);
    }

    #[test]
    fn paragraph_after_code_block_is_separated() {
        assert_eq!(render("```\nx\n```\nafter\n"), ["  x", "", "after", ""]);
    }

    #[test]
    fn code_block_in_list_item_keeps_one_line_per_source_line() {
        let md = "1. Install:\n   ```bash\n   cargo build\n   cargo test\n   ```\n2. Done\n";
        assert_eq!(
            render(md),
            [
                " 1. Install:",
                "      bash",
                "      cargo build",
                "      cargo test",
                " 2. Done",
                ""
            ]
        );
    }

    #[test]
    fn code_block_in_blockquote_keeps_one_line_per_source_line() {
        assert_eq!(render("> ```\n> a\n> b\n> ```\n"), ["│   a", "│   b", ""]);
    }

    // --- Tables ---

    #[test]
    fn table_renders_bold_header_separator_and_rows() {
        let md = "| Name | Qty |\n|------|-----|\n| apple | 3 |\n";
        assert_eq!(render(md), ["Name  │ Qty", "──────┼────", "apple │ 3", ""]);
        assert!(span_style(md, "Name").add_modifier.contains(Modifier::BOLD));
        assert!(!span_style(md, "apple")
            .add_modifier
            .contains(Modifier::BOLD));
        // A header with no body rows still renders.
        assert_eq!(
            render("| a | b |\n|---|---|\n"),
            ["a   │ b", "────┼────", ""]
        );
    }

    #[test]
    fn inline_code_stays_inside_its_table_cell() {
        let md = "| Flag | Meaning |\n|---|---|\n| `-p` | providers |\n\nAfter table.\n";
        assert_eq!(
            render(md),
            [
                "Flag │ Meaning",
                "─────┼──────────",
                "-p   │ providers",
                "",
                "After table.",
                ""
            ]
        );
        assert_eq!(span_style(md, "-p").fg, Some(Color::Yellow));
    }

    #[test]
    fn table_columns_are_padded_and_aligned_by_display_width() {
        let md = "| L | C | R |\n|:--|:-:|--:|\n| 名前 | x | 1 |\n| a | yyyyy | 22 |\n";
        let lines = markdown_to_lines(md);
        assert_eq!(
            render(md),
            [
                "L    │   C   │   R",
                "─────┼───────┼────",
                "名前 │   x   │   1",
                "a    │ yyyyy │  22",
                ""
            ]
        );
        // "名前" is 2 chars but 4 columns wide; every row lines up.
        assert!(lines[..4].iter().all(|line| line.width() == 18));
    }

    #[test]
    fn consecutive_tables_do_not_share_rows() {
        assert_eq!(
            render("| a |\n|---|\n| 1 |\n\n| b |\n|---|\n| 2 |\n"),
            ["a", "───", "1", "", "b", "───", "2", ""]
        );
    }

    #[test]
    fn table_inside_list_item_is_indented() {
        assert_eq!(
            render("- item\n\n  | a | b |\n  |---|---|\n  | 1 | 2 |\n"),
            [
                "  • item",
                "    a   │ b",
                "    ────┼────",
                "    1   │ 2",
                ""
            ]
        );
    }

    // --- HTML ---

    #[test]
    fn inline_html_is_rendered_literally() {
        assert_eq!(
            render("Use Vec<String> or Box<dyn Error>\n"),
            ["Use Vec<String> or Box<dyn Error>", ""]
        );
    }

    #[test]
    fn block_html_is_rendered_as_lines() {
        assert_eq!(
            render("<details>\n<summary>More</summary>\n</details>\n\nafter\n"),
            [
                "<details>",
                "<summary>More</summary>",
                "</details>",
                "",
                "after",
                ""
            ]
        );
    }

    // --- Links ---

    #[test]
    fn link_url_is_shown_after_the_text() {
        let md = "See [the docs](https://example.com/docs) now.\n";
        assert_eq!(
            render(md),
            ["See the docs (https://example.com/docs) now.", ""]
        );
        assert!(span_style(md, "the docs")
            .add_modifier
            .contains(Modifier::UNDERLINED));
        assert_eq!(
            span_style(md, " (https://example.com/docs)").fg,
            Some(Color::DarkGray)
        );
    }

    #[test]
    fn autolink_url_is_not_repeated() {
        assert_eq!(
            render("Visit <https://example.com> or <me@example.com>.\n"),
            ["Visit https://example.com or me@example.com.", ""]
        );
    }

    // --- Blockquotes ---

    #[test]
    fn nested_blockquote_keeps_prefixes_and_outer_style() {
        let md = "> outer\n>\n> > inner\n>\n> outer again\n";
        assert_eq!(
            render(md),
            ["│ outer", "│", "│ │ inner", "│", "│ outer again", ""]
        );
        let style = span_style(md, "outer again");
        assert_eq!(style.fg, Some(Color::DarkGray));
        assert!(style.add_modifier.contains(Modifier::ITALIC));
    }

    #[test]
    fn hard_break_in_blockquote_keeps_the_prefix() {
        assert_eq!(
            render("> line one  \n> line two\n"),
            ["│ line one", "│ line two", ""]
        );
    }

    #[test]
    fn list_inside_blockquote_is_prefixed() {
        assert_eq!(render("> - a\n> - b\n"), ["│   • a", "│   • b", ""]);
    }

    #[test]
    fn every_streaming_prefix_renders_without_panicking() {
        // The TUI re-renders the partial answer on every frame, so any
        // prefix of a document has to be renderable.
        let doc = "# T\n\n1. a\n   ```sh\n   x\ty\n   ```\n   - [ ] b\n\n\
                   | h | `c` |\n|:-:|--:|\n| 名前 | [l](u) |\n\n\
                   > q  \n> > <b>n</b>\n\n---\n";
        for (i, _) in doc.char_indices() {
            let _ = markdown_to_lines(&doc[..i]);
        }
    }
}
