use crate::providers::{Provider, ProviderManager};
use crate::query::{stream_with_messages, QueryContext, StreamEvent};
use anyhow::Result;
use crossterm::event::{
    self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEventKind,
};
use genai::chat::{ChatMessage, ChatRole, ContentPart};
use opentelemetry::trace::FutureExt;
use opentelemetry::Context;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::Line,
    widgets::{Block, BorderType, Borders, Paragraph, Wrap},
    Frame, Terminal,
};
use std::io::{self, Stdout};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

/// Three lines per wheel tick — single-line feels sluggish, half-page feels too
/// jumpy. Matches most terminal pagers.
const MOUSE_SCROLL_STEP: u16 = 3;

/// Backpressure on the panel input queue. 8 is generous — a user typing
/// follow-ups can't plausibly outrun a task that processes each turn in
/// seconds. If it does saturate, we show a status message rather than blocking
/// the UI thread.
const PANEL_INPUT_CAPACITY: usize = 8;

/// Tabs are drawn as this many spaces. ratatui counts `\t` as one column but
/// terminals jump to the next tab stop, which garbles panel layout.
const TAB_WIDTH: usize = 4;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Mode {
    Normal,
    Insert,
}

/// What role a completed turn represents in the rendered history. Metadata on
/// `Assistant` turns lets the panel render a compact token/latency line
/// underneath without re-querying the provider.
#[derive(Clone)]
pub enum TurnKind {
    User,
    Assistant {
        input_tokens: Option<i32>,
        output_tokens: Option<i32>,
        duration_ms: u64,
    },
    Error,
}

#[derive(Clone)]
pub struct Turn {
    pub kind: TurnKind,
    pub text: String,
}

/// Command sent from the UI thread to a panel's background task. Each panel
/// owns a private `Vec<ChatMessage>` history; the UI never touches it directly
/// — it drives it by sending Ask/Clear and consuming the resulting stream
/// events.
enum PanelCommand {
    Ask {
        question: String,
        cancel: CancellationToken,
    },
    Clear,
}

pub struct PanelState {
    pub provider: Provider,
    pub model: String,
    /// Completed turns, rendered as scroll-back above the currently streaming
    /// assistant response (if any).
    pub turns: Vec<Turn>,
    /// Text being streamed into this panel right now. Finalized into a `Turn`
    /// on `Done`/`Error`.
    pub current_text: String,
    pub is_streaming: bool,
    /// A turn abandoned by `/clear` is still winding down in the panel task.
    /// Its remaining events are discarded until its closing `Done`/`Error`
    /// arrives, so they can't leak into the next turn.
    pub draining: bool,
    pub scroll_offset: u16,
    pub content_height: u16,
    pub visible_height: u16,
    /// Cached screen rectangle from the last draw, used to route mouse-scroll
    /// events to the panel under the cursor.
    pub area: Option<Rect>,
    /// While true, the panel snaps its scroll to the bottom each frame so the
    /// tail of the streaming response stays visible. Any manual scroll toggles
    /// this off until the user scrolls back to the bottom.
    pub auto_scroll: bool,
}

impl PanelState {
    pub fn new(provider: Provider) -> Self {
        let model = provider.model.clone();
        PanelState {
            provider,
            model,
            turns: Vec::new(),
            current_text: String::new(),
            is_streaming: false,
            draining: false,
            scroll_offset: 0,
            content_height: 0,
            visible_height: 0,
            area: None,
            auto_scroll: true,
        }
    }

    pub fn max_scroll(&self) -> u16 {
        self.content_height.saturating_sub(self.visible_height)
    }

    /// Busy means a request is still running in the panel task, whether or
    /// not the UI still cares about its output.
    fn is_busy(&self) -> bool {
        self.is_streaming || self.draining
    }

    pub fn get_color(&self) -> Color {
        match self.provider.color {
            colored::Color::Red => Color::Red,
            colored::Color::Green => Color::Green,
            colored::Color::Blue => Color::Blue,
            colored::Color::Cyan => Color::Cyan,
            colored::Color::Magenta => Color::Magenta,
            colored::Color::Yellow => Color::Yellow,
            colored::Color::White => Color::White,
            _ => Color::White,
        }
    }
}

pub struct App {
    panels: Vec<PanelState>,
    senders: Vec<mpsc::Sender<PanelCommand>>,
    /// Panel worker tasks, index-aligned with `panels`. Watched so a task
    /// that dies (e.g. panics) doesn't leave its panel "streaming" forever,
    /// and aborted on exit. Empty in unit tests.
    tasks: Vec<JoinHandle<()>>,
    selected_panel: usize,
    show_tokens: bool,
    mode: Mode,
    input_buffer: String,
    /// Byte offset of the cursor in `input_buffer`; always on a grapheme
    /// boundary.
    input_cursor: usize,
    should_quit: bool,
    /// Ephemeral one-line hint (e.g., "busy", "cleared") shown in the footer
    /// until the next key event or turn completion.
    status_message: Option<String>,
    /// Cancellation token for the currently in-flight turn. Cloned into every
    /// panel's Ask command so a single Ctrl+C cancels them all.
    current_turn_cancel: Option<CancellationToken>,
}

impl App {
    fn new(
        panels: Vec<PanelState>,
        senders: Vec<mpsc::Sender<PanelCommand>>,
        show_tokens: bool,
    ) -> Self {
        App {
            panels,
            senders,
            tasks: Vec::new(),
            selected_panel: 0,
            show_tokens,
            // Default to Insert so typing works immediately, like Claude Code.
            // Power users can press Esc once to drop into Normal for vim nav.
            mode: Mode::Insert,
            input_buffer: String::new(),
            input_cursor: 0,
            should_quit: false,
            status_message: None,
            current_turn_cancel: None,
        }
    }

    /// A turn is visibly in progress: Esc/Ctrl+C should cancel it rather
    /// than quit or clear the input line.
    fn any_streaming(&self) -> bool {
        self.panels.iter().any(|p| p.is_streaming)
    }

    /// Some panel's worker is still running a request — including one
    /// abandoned by `/clear` that is winding down. New turns must wait, but
    /// there's nothing left for Esc/Ctrl+C to cancel.
    fn any_busy(&self) -> bool {
        self.panels.iter().any(PanelState::is_busy)
    }

    fn find_panel_mut(&mut self, display_name: &str) -> Option<&mut PanelState> {
        self.panels
            .iter_mut()
            .find(|p| p.provider.display_name == display_name)
    }

    pub fn handle_event(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::Start { provider, model } => {
                if let Some(p) = self.find_panel_mut(&provider) {
                    if !p.draining {
                        p.model = model;
                    }
                }
            }
            StreamEvent::Chunk { provider, text } => {
                if let Some(p) = self.find_panel_mut(&provider) {
                    if p.is_streaming {
                        p.current_text.push_str(&text);
                    }
                }
            }
            StreamEvent::Done {
                provider,
                model,
                input_tokens,
                output_tokens,
                duration_ms,
            } => {
                if let Some(p) = self.find_panel_mut(&provider) {
                    if p.draining {
                        // The abandoned turn has finished; nothing to record.
                        p.draining = false;
                    } else if p.is_streaming {
                        p.model = model;
                        let text = std::mem::take(&mut p.current_text);
                        p.turns.push(Turn {
                            kind: TurnKind::Assistant {
                                input_tokens,
                                output_tokens,
                                duration_ms,
                            },
                            text,
                        });
                        p.is_streaming = false;
                    }
                }
                self.maybe_clear_turn_cancel();
            }
            StreamEvent::Error {
                provider,
                error,
                duration_ms,
                ..
            } => {
                if let Some(p) = self.find_panel_mut(&provider) {
                    if p.draining {
                        p.draining = false;
                    } else if p.is_streaming {
                        // Drop any partial streamed text — it's incomplete and
                        // mixing it into an "error" turn is misleading.
                        p.current_text.clear();
                        p.turns.push(Turn {
                            kind: TurnKind::Error,
                            text: format!("{} ({})", error, format_duration(duration_ms)),
                        });
                        p.is_streaming = false;
                    }
                }
                self.maybe_clear_turn_cancel();
            }
            StreamEvent::SearchStart { provider, query } => {
                if let Some(p) = self.find_panel_mut(&provider) {
                    if p.is_streaming {
                        // Appears inline in the streaming text; italic markdown
                        // so the markdown renderer styles it distinctly.
                        p.current_text
                            .push_str(&format!("\n\n*→ searching: {}*\n\n", query.trim()));
                    }
                }
            }
            StreamEvent::SearchEnd {
                provider,
                search_provider,
                result_count,
                error,
                ..
            } => {
                if let Some(p) = self.find_panel_mut(&provider) {
                    if !p.is_streaming {
                        return;
                    }
                    match error {
                        Some(err) => p
                            .current_text
                            .push_str(&format!("*← search failed: {}*\n\n", err.trim())),
                        None => p.current_text.push_str(&format!(
                            "*← via {}: {} result(s)*\n\n",
                            search_provider, result_count
                        )),
                    }
                }
            }
        }
    }

    /// Clear the shared cancel token once every panel has finished streaming,
    /// so the next Ctrl+C doesn't no-op against a stale token.
    fn maybe_clear_turn_cancel(&mut self) {
        if !self.any_busy() {
            self.current_turn_cancel = None;
        }
    }

    /// Send `question` to every panel. Returns false (and leaves the input
    /// alone for the caller) if a turn is still in flight or no panel could
    /// take it.
    fn submit_turn(&mut self, question: String) -> bool {
        if self.any_busy() {
            self.status_message = Some("busy — wait for current turn to finish".into());
            return false;
        }
        if self.senders.is_empty() {
            return false;
        }
        let cancel = CancellationToken::new();
        self.current_turn_cancel = Some(cancel.clone());
        let mut accepted = 0usize;
        for (i, tx) in self.senders.iter().enumerate() {
            let panel = &mut self.panels[i];
            panel.turns.push(Turn {
                kind: TurnKind::User,
                text: question.clone(),
            });
            panel.current_text.clear();
            panel.is_streaming = true;
            panel.auto_scroll = true;
            // try_send is non-blocking; if the queue is somehow full (capacity
            // hit) we surface it to the user rather than stalling the UI loop.
            if let Err(e) = tx.try_send(PanelCommand::Ask {
                question: question.clone(),
                cancel: cancel.clone(),
            }) {
                panel.is_streaming = false;
                // Roll back the user turn we just added so the history doesn't
                // show a question that will never be answered.
                panel.turns.pop();
                let reason = match e {
                    TrySendError::Full(_) => "input queue full",
                    TrySendError::Closed(_) => "its worker stopped",
                };
                self.status_message = Some(format!(
                    "{} is unavailable ({})",
                    panel.provider.display_name, reason
                ));
            } else {
                accepted += 1;
            }
        }
        if accepted == 0 {
            // Nothing will answer: keep the question in the input box.
            self.current_turn_cancel = None;
            return false;
        }
        true
    }

    fn cancel_current_turn(&mut self) {
        if let Some(cancel) = &self.current_turn_cancel {
            cancel.cancel();
        }
    }

    fn clear_all_panels(&mut self) {
        // Stop any in-flight turn first: otherwise it keeps streaming (and
        // billing) with no way to cancel it, and its late events would land
        // in the freshly cleared panels.
        self.cancel_current_turn();
        for (i, tx) in self.senders.iter().enumerate() {
            // Fire-and-forget: if the receiver is gone the task is ending
            // anyway, and the visible state is reset regardless.
            let _ = tx.try_send(PanelCommand::Clear);
            let p = &mut self.panels[i];
            if p.is_streaming {
                p.draining = true;
            }
            p.turns.clear();
            p.current_text.clear();
            p.is_streaming = false;
            p.scroll_offset = 0;
            p.auto_scroll = true;
        }
        self.current_turn_cancel = None;
        self.status_message = Some("cleared".into());
    }

    /// Notice panel tasks that ended on their own (a panic in the request
    /// path) so their panels don't sit at "streaming..." forever. Returns
    /// true if anything changed.
    fn reap_dead_tasks(&mut self) -> bool {
        let mut changed = false;
        for (i, task) in self.tasks.iter().enumerate() {
            let Some(panel) = self.panels.get_mut(i) else {
                continue;
            };
            if task.is_finished() && panel.is_busy() {
                let was_streaming = panel.is_streaming;
                panel.is_streaming = false;
                panel.draining = false;
                panel.current_text.clear();
                if was_streaming {
                    panel.turns.push(Turn {
                        kind: TurnKind::Error,
                        text: "this panel's worker stopped unexpectedly \
                               (details are printed after you quit)"
                            .into(),
                    });
                }
                changed = true;
            }
        }
        if changed {
            self.maybe_clear_turn_cancel();
        }
        changed
    }

    fn next_panel(&mut self) {
        if !self.panels.is_empty() {
            self.selected_panel = (self.selected_panel + 1) % self.panels.len();
        }
    }

    fn prev_panel(&mut self) {
        if !self.panels.is_empty() {
            self.selected_panel = (self.selected_panel + self.panels.len() - 1) % self.panels.len();
        }
    }

    fn scroll_by(&mut self, panel_idx: usize, delta: i32) {
        if let Some(panel) = self.panels.get_mut(panel_idx) {
            let max = panel.max_scroll();
            let new_offset = (panel.scroll_offset as i32 + delta).clamp(0, max as i32) as u16;
            // Any user scroll detaches auto-scroll; reaching the bottom
            // re-attaches it so hitting "scroll down" past the end sticks you
            // back to the streaming tail.
            panel.auto_scroll = new_offset == max;
            panel.scroll_offset = new_offset;
        }
    }

    fn page_scroll(&mut self, direction: i32) {
        let idx = self.selected_panel;
        let page = self
            .panels
            .get(idx)
            .map(|p| p.visible_height.saturating_sub(1) as i32)
            .unwrap_or(10)
            .max(1);
        self.scroll_by(idx, direction * page);
    }

    /// Locate which panel (if any) contains the given screen coordinate, based
    /// on the area each panel recorded on its last draw.
    fn panel_at(&self, col: u16, row: u16) -> Option<usize> {
        for (i, panel) in self.panels.iter().enumerate() {
            if let Some(area) = panel.area {
                if col >= area.x
                    && col < area.x + area.width
                    && row >= area.y
                    && row < area.y + area.height
                {
                    return Some(i);
                }
            }
        }
        None
    }

    // ── Input line editing ────────────────────────────────────────────────

    fn insert_text(&mut self, text: &str) {
        self.input_buffer.insert_str(self.input_cursor, text);
        self.input_cursor += text.len();
    }

    fn clear_input(&mut self) {
        self.input_buffer.clear();
        self.input_cursor = 0;
    }

    fn cursor_left(&mut self) {
        self.input_cursor = prev_grapheme_boundary(&self.input_buffer, self.input_cursor);
    }

    fn cursor_right(&mut self) {
        self.input_cursor = next_grapheme_boundary(&self.input_buffer, self.input_cursor);
    }

    /// Delete the grapheme before the cursor — a whole flag/ZWJ emoji or
    /// accented letter, not just its last code point.
    fn backspace(&mut self) {
        let start = prev_grapheme_boundary(&self.input_buffer, self.input_cursor);
        self.input_buffer.drain(start..self.input_cursor);
        self.input_cursor = start;
    }

    fn delete_forward(&mut self) {
        let end = next_grapheme_boundary(&self.input_buffer, self.input_cursor);
        self.input_buffer.drain(self.input_cursor..end);
    }

    /// Ctrl+W: delete the word (and any whitespace) before the cursor.
    fn delete_word_back(&mut self) {
        let before = &self.input_buffer[..self.input_cursor];
        let trimmed = before.trim_end();
        let start = trimmed
            .char_indices()
            .rev()
            .find(|(_, c)| c.is_whitespace())
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);
        self.input_buffer.drain(start..self.input_cursor);
        self.input_cursor = start;
    }
}

fn prev_grapheme_boundary(s: &str, idx: usize) -> usize {
    s[..idx]
        .grapheme_indices(true)
        .next_back()
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn next_grapheme_boundary(s: &str, idx: usize) -> usize {
    s[idx..]
        .graphemes(true)
        .next()
        .map(|g| idx + g.len())
        .unwrap_or(idx)
}

fn format_duration(duration_ms: u64) -> String {
    if duration_ms >= 1000 {
        format!("{:.1}s", duration_ms as f64 / 1000.0)
    } else {
        format!("{}ms", duration_ms)
    }
}

fn expand_tabs(s: &str) -> String {
    s.replace('\t', &" ".repeat(TAB_WIDTH))
}

/// Single-line rendering of the input buffer: newlines from a multi-line
/// paste show as `↵` (they're still sent as real newlines) and tabs as
/// spaces.
fn input_display(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\n' => "↵".to_string(),
            '\t' => " ".repeat(TAB_WIDTH),
            '\r' => String::new(),
            c => c.to_string(),
        })
        .collect()
}

/// Visible text and cursor column for a one-row input box `width` columns
/// wide: keeps the cursor visible with the text scrolling left under it once
/// the line is longer than the box.
///
/// The window is sliced here rather than handed to `Paragraph::scroll`,
/// whose offset is a `u16`: a paste wider than 65,535 columns would wrap the
/// offset and show an unrelated part of the buffer while the cursor kept
/// editing the end of it.
fn input_viewport(buffer: &str, cursor: usize, width: usize) -> (String, u16) {
    if width == 0 {
        return (String::new(), 0);
    }
    let cursor_col = input_display(&buffer[..cursor]).width();
    let offset = cursor_col.saturating_sub(width - 1);
    let mut visible = String::new();
    // Display column where the current grapheme starts.
    let mut col = 0;
    for g in input_display(buffer).graphemes(true) {
        let end = col + g.width();
        if end <= offset {
            col = end;
            continue;
        }
        if end - offset > width {
            break;
        }
        if col < offset {
            // A wide glyph straddling the left edge: blank its visible half
            // so everything after it stays aligned with the cursor.
            visible.push_str(&" ".repeat(end - offset));
        } else {
            visible.push_str(g);
        }
        col = end;
    }
    // `cursor_col - offset` is at most `width - 1`, and `width` came from a
    // u16 area, so this never truncates.
    (visible, (cursor_col - offset) as u16)
}

/// Render one completed turn or the in-flight streaming text as a block of
/// styled lines. Keeping this isolated makes testing the rendering hierarchy
/// tractable and avoids tangled borrow lifetimes during the draw.
fn lines_for_turn(turn: &Turn, panel_color: Color, show_tokens: bool) -> Vec<Line<'static>> {
    let mut out: Vec<Line<'static>> = Vec::new();
    match &turn.kind {
        TurnKind::User => {
            out.push(Line::styled(
                "you".to_string(),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ));
            for l in turn.text.lines() {
                out.push(Line::from(format!("  {}", expand_tabs(l))));
            }
        }
        TurnKind::Assistant {
            input_tokens,
            output_tokens,
            duration_ms,
        } => {
            out.push(Line::styled(
                "assistant".to_string(),
                Style::default()
                    .fg(panel_color)
                    .add_modifier(Modifier::BOLD),
            ));
            out.extend(crate::markdown::markdown_to_lines(&turn.text));
            let mut stats = Vec::new();
            if show_tokens {
                if let (Some(i), Some(o)) = (input_tokens, output_tokens) {
                    stats.push(format!("{} in / {} out", i, o));
                }
            }
            stats.push(format_duration(*duration_ms));
            out.push(Line::styled(
                stats.join(" | "),
                Style::default().fg(Color::DarkGray),
            ));
        }
        TurnKind::Error => {
            out.push(Line::styled(
                "error".to_string(),
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ));
            // One Line per text line: a Line renders its content on a
            // single row, so embedded newlines would otherwise run together.
            for l in turn.text.lines() {
                out.push(Line::styled(
                    expand_tabs(l),
                    Style::default().fg(Color::Red),
                ));
            }
        }
    }
    out.push(Line::from(""));
    out
}

fn draw_panel(
    frame: &mut Frame,
    panel: &mut PanelState,
    area: Rect,
    is_selected: bool,
    show_tokens: bool,
) {
    panel.area = Some(area);
    let color = panel.get_color();

    let status = if panel.is_streaming {
        "streaming..."
    } else if panel.turns.is_empty() {
        "ready"
    } else {
        "idle"
    };
    let title_text = format!(
        " {} · {} [{}] ",
        panel.provider.display_name, panel.model, status
    );

    // Bold alone is unreliable here: box-drawing glyphs don't always render
    // visibly heavier under the ANSI bold attribute, which is why the old
    // "just bold the border" selection cue was hard to spot. BorderType::Thick
    // swaps in different box-drawing characters entirely, so the selected
    // panel is unmistakable regardless of terminal/font. The reverse-video
    // title reinforces it as an "active tab".
    let (border_type, border_style, title_style) = if is_selected {
        (
            BorderType::Thick,
            Style::default().fg(color).add_modifier(Modifier::BOLD),
            Style::default()
                .fg(Color::Black)
                .bg(color)
                .add_modifier(Modifier::BOLD),
        )
    } else {
        (
            BorderType::Plain,
            Style::default().fg(color),
            Style::default().fg(color),
        )
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(border_type)
        .title(Line::styled(title_text, title_style))
        .border_style(border_style);

    // Compose lines from all completed turns plus the in-flight streaming text.
    let mut lines: Vec<Line<'static>> = Vec::new();
    for turn in &panel.turns {
        lines.extend(lines_for_turn(turn, color, show_tokens));
    }
    if panel.is_streaming {
        lines.push(Line::styled(
            "assistant".to_string(),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ));
        if panel.current_text.is_empty() {
            lines.push(Line::styled(
                "  ▋".to_string(),
                Style::default().fg(Color::DarkGray),
            ));
        } else {
            lines.extend(crate::markdown::markdown_to_lines(&panel.current_text));
        }
    }

    let inner_width = area.width.saturating_sub(2);
    let inner_height = area.height.saturating_sub(2);

    let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
    // Ask the paragraph itself how many rows it wraps to. Estimating with
    // `ceil(line_width / inner_width)` undercounts word-wrapped prose, which
    // left the tail of long answers unreachable (auto-scroll stopped short).
    let total_lines = paragraph.line_count(inner_width);
    panel.content_height = u16::try_from(total_lines).unwrap_or(u16::MAX);
    panel.visible_height = inner_height;

    let max_scroll = panel.max_scroll();
    // Re-pin to bottom under auto-scroll, or clamp an over-scrolled offset if
    // content shrank (e.g., after /clear). Both branches collapse to the same
    // assignment — we keep the structure so the intent is legible even if
    // clippy sees an identical block.
    if panel.auto_scroll || panel.scroll_offset > max_scroll {
        panel.scroll_offset = max_scroll;
    }

    let paragraph = paragraph.block(block).scroll((panel.scroll_offset, 0));

    frame.render_widget(paragraph, area);
}

fn draw_ui(frame: &mut Frame, app: &mut App) {
    let size = frame.area();

    let main_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(5),    // panels
            Constraint::Length(3), // input box
            Constraint::Length(1), // footer
        ])
        .split(size);

    let num_panels = app.panels.len();
    if num_panels > 0 {
        let constraints: Vec<Constraint> = (0..num_panels)
            .map(|_| Constraint::Ratio(1, num_panels as u32))
            .collect();
        let panel_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(constraints)
            .split(main_layout[0]);
        for i in 0..num_panels {
            let is_selected = i == app.selected_panel;
            let area = panel_layout[i];
            let show_tokens = app.show_tokens;
            draw_panel(frame, &mut app.panels[i], area, is_selected, show_tokens);
        }
    }

    // Input box: one row that scrolls horizontally to keep the cursor in
    // view. In Normal mode the buffer still displays but no cursor is drawn.
    let input_area = main_layout[1];
    let (input_title, input_style) = match app.mode {
        Mode::Insert => (
            " INSERT ",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ),
        Mode::Normal => (
            " NORMAL ",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        ),
    };
    let input_block = Block::default()
        .borders(Borders::ALL)
        .title(input_title)
        .border_style(input_style);
    let inner_width = input_area.width.saturating_sub(2) as usize;
    let (visible_input, cursor_col) =
        input_viewport(&app.input_buffer, app.input_cursor, inner_width);
    let input_paragraph = Paragraph::new(visible_input).block(input_block);
    frame.render_widget(input_paragraph, input_area);

    if app.mode == Mode::Insert {
        frame.set_cursor_position((input_area.x + 1 + cursor_col, input_area.y + 1));
    }

    // Footer: mode-appropriate keybind hint, or a transient status message.
    let help_text: String = if let Some(msg) = &app.status_message {
        msg.clone()
    } else {
        match app.mode {
            Mode::Insert => {
                "Esc: normal | Enter: send | Ctrl+C: cancel turn | PgUp/PgDn: scroll | /clear /exit"
                    .to_string()
            }
            Mode::Normal => {
                "i: insert | h/l or ←/→: select panel | j/k or ↑/↓: scroll | q: quit".to_string()
            }
        }
    };
    let footer = Paragraph::new(help_text).style(Style::default().fg(Color::Gray));
    frame.render_widget(footer, main_layout[2]);
}

/// Give panels distinct display names. Stream events are routed to panels
/// by name, so two providers sharing a first alias would otherwise both feed
/// the first panel and leave the second "streaming" forever.
///
/// The first provider with a name keeps it; later ones get the lowest free
/// ` (n)` suffix. Every configured name is reserved up front so a generated
/// name can't collide with one a later provider was configured with (aliases
/// `X`, `X`, `X (2)` become `X`, `X (3)`, `X (2)`).
fn dedupe_display_names(providers: &mut [Provider]) {
    use std::collections::HashSet;
    let configured: HashSet<String> = providers.iter().map(|p| p.display_name.clone()).collect();
    let mut taken: HashSet<String> = HashSet::new();
    for p in providers.iter_mut() {
        if taken.insert(p.display_name.clone()) {
            continue;
        }
        let mut n = 2;
        let unique = loop {
            let name = format!("{} ({})", p.display_name, n);
            if !configured.contains(&name) && !taken.contains(&name) {
                break name;
            }
            n += 1;
        };
        taken.insert(unique.clone());
        p.display_name = unique;
    }
}

pub async fn run_tui(
    manager: Arc<ProviderManager>,
    mut providers: Vec<Provider>,
    ctx: QueryContext,
    initial_question: &str,
    show_tokens: bool,
) -> Result<()> {
    use std::io::IsTerminal;
    if !io::stdout().is_terminal() {
        anyhow::bail!(
            "TUI mode requires an interactive terminal. Use default streaming mode instead."
        );
    }

    dedupe_display_names(&mut providers);

    // Spawn a task per panel; each owns its own history and listens on its
    // own mpsc<PanelCommand>. Stream events from all panels fan in to a
    // shared event channel the UI consumes.
    let (event_tx, event_rx) = mpsc::channel::<StreamEvent>(100);
    let mut senders: Vec<mpsc::Sender<PanelCommand>> = Vec::with_capacity(providers.len());
    let mut panels: Vec<PanelState> = Vec::with_capacity(providers.len());
    let mut tasks: Vec<JoinHandle<()>> = Vec::with_capacity(providers.len());

    // tokio::spawn does not propagate the OTel Context, so spans created inside
    // panel_task would otherwise become orphan roots in Langfuse instead of
    // children of the caller's `ai.query` span. Capture the current Context and
    // wrap each spawn via `FutureExt::with_context` (mirrors `stream_all`).
    let parent_cx = Context::current();

    for provider in &providers {
        let (cmd_tx, cmd_rx) = mpsc::channel::<PanelCommand>(PANEL_INPUT_CAPACITY);
        senders.push(cmd_tx);
        panels.push(PanelState::new(provider.clone()));

        let task_manager = Arc::clone(&manager);
        let task_provider = provider.clone();
        let task_ctx = ctx.clone();
        let task_event_tx = event_tx.clone();
        let task_cx = parent_cx.clone();
        // Handles are kept on the App: watched for unexpected exits and
        // aborted on quit.
        tasks.push(tokio::spawn(
            async move {
                panel_task(task_manager, task_provider, task_ctx, cmd_rx, task_event_tx).await;
            }
            .with_context(task_cx),
        ));
    }
    drop(event_tx); // only panel tasks should hold senders; drop our copy

    // Retry notices and auth warnings would print over the alternate screen;
    // hold them until the terminal is restored.
    crate::diag::defer();
    let result = {
        // The guard restores the terminal on every exit path from this
        // block, including `?` and panics.
        let guard = crate::term::TerminalGuard::enter(true);
        match guard {
            Ok(_guard) => {
                let mut app = App::new(panels, senders, show_tokens);
                app.tasks = tasks;

                // Seed the first turn with the CLI-provided question, if any.
                let initial = initial_question.trim();
                if !initial.is_empty() {
                    app.submit_turn(initial.to_string());
                }

                let result = match Terminal::new(CrosstermBackend::new(io::stdout())) {
                    Ok(mut terminal) => run_main_loop(&mut terminal, &mut app, event_rx).await,
                    Err(e) => Err(e.into()),
                };

                // Stop in-flight requests instead of abandoning them to run
                // (and bill) in the background while the process exits.
                app.cancel_current_turn();
                for task in &app.tasks {
                    task.abort();
                }
                result
            }
            Err(e) => Err(e.into()),
        }
    };
    crate::diag::flush();

    result
}

/// True when `messages` ends with a completed assistant reply (text, no
/// pending tool calls) — i.e. `stream_with_messages` succeeded.
fn ends_with_final_reply(messages: &[ChatMessage]) -> bool {
    let Some(last) = messages.last() else {
        return false;
    };
    matches!(last.role, ChatRole::Assistant)
        && !last
            .content
            .parts()
            .iter()
            .any(|p| matches!(p, ContentPart::ToolCall(_)))
        && !last.content.texts().join("").is_empty()
}

/// Long-lived per-panel worker. Owns a private `Vec<ChatMessage>` including
/// the system prompt (if any); executes one `stream_with_messages` per Ask
/// command. On Clear, resets history back to just the system prompt.
async fn panel_task(
    manager: Arc<ProviderManager>,
    provider: Provider,
    ctx: QueryContext,
    mut cmd_rx: mpsc::Receiver<PanelCommand>,
    event_tx: mpsc::Sender<StreamEvent>,
) {
    let base_messages: Vec<ChatMessage> = if let Some(s) = ctx.system_prompt.as_deref() {
        vec![ChatMessage::system(s.to_string())]
    } else {
        Vec::new()
    };
    let mut messages = base_messages.clone();

    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            PanelCommand::Ask { question, cancel } => {
                let pre_turn_len = messages.len();
                messages.push(ChatMessage::user(question));
                let enriched = stream_with_messages(
                    &manager,
                    &provider,
                    messages.clone(),
                    &ctx,
                    event_tx.clone(),
                    &cancel,
                )
                .await;
                if ends_with_final_reply(&enriched) {
                    messages = enriched;
                } else {
                    // Error or cancel. Roll back to where the turn started:
                    // besides the user message this drops any half-finished
                    // tool exchange (assistant tool-call + tool results),
                    // which would otherwise corrupt every later turn.
                    messages.truncate(pre_turn_len);
                }
            }
            PanelCommand::Clear => {
                messages = base_messages.clone();
            }
        }
    }
}

async fn run_main_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    app: &mut App,
    mut event_rx: mpsc::Receiver<StreamEvent>,
) -> Result<()> {
    // Only redraw when something changed. Redrawing re-parses every panel's
    // markdown, so doing it on a fixed 50ms tick burned CPU while idle.
    let mut dirty = true;
    loop {
        if dirty {
            terminal.draw(|frame| draw_ui(frame, app))?;
            dirty = false;
        }

        if event::poll(std::time::Duration::from_millis(50))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    // Any key clears the transient status hint so it doesn't
                    // linger past user acknowledgement.
                    app.status_message = None;
                    handle_key(app, key);
                    if app.should_quit {
                        return Ok(());
                    }
                }
                Event::Mouse(mouse) => handle_mouse(app, mouse),
                Event::Paste(text) if app.mode == Mode::Insert => {
                    app.insert_text(&normalize_newlines(&text));
                }
                _ => {}
            }
            dirty = true;
        }

        while let Ok(event) = event_rx.try_recv() {
            app.handle_event(event);
            dirty = true;
        }

        if app.reap_dead_tasks() {
            dirty = true;
        }
    }
}

/// Pasted text arrives with whatever line endings the source used; keep
/// plain LF so the provider sees normal newlines.
fn normalize_newlines(s: &str) -> String {
    s.replace("\r\n", "\n").replace('\r', "\n")
}

fn handle_key(app: &mut App, key: KeyEvent) {
    match app.mode {
        Mode::Normal => handle_normal_key(app, key),
        Mode::Insert => handle_insert_key(app, key),
    }
}

fn handle_normal_key(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Char('q') => {
            app.cancel_current_turn();
            app.should_quit = true;
        }
        KeyCode::Esc => {
            // Esc in Normal when a turn is streaming cancels it; otherwise
            // quits. This gives you a single ubiquitous "get out" key.
            if app.any_streaming() {
                app.cancel_current_turn();
                app.status_message = Some("cancelling...".into());
            } else {
                app.should_quit = true;
            }
        }
        KeyCode::Char('i') => app.mode = Mode::Insert,
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            if app.any_streaming() {
                app.cancel_current_turn();
                app.status_message = Some("cancelling...".into());
            } else {
                app.should_quit = true;
            }
        }
        KeyCode::Left | KeyCode::Char('h') => app.prev_panel(),
        KeyCode::Right | KeyCode::Char('l') | KeyCode::Tab => app.next_panel(),
        KeyCode::Up | KeyCode::Char('k') => {
            let idx = app.selected_panel;
            app.scroll_by(idx, -1);
        }
        KeyCode::Down | KeyCode::Char('j') => {
            let idx = app.selected_panel;
            app.scroll_by(idx, 1);
        }
        KeyCode::PageUp => app.page_scroll(-1),
        KeyCode::PageDown => app.page_scroll(1),
        _ => {}
    }
}

fn handle_insert_key(app: &mut App, key: KeyEvent) {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);
    match key.code {
        KeyCode::Esc => app.mode = Mode::Normal,
        KeyCode::Enter => {
            let line = app.input_buffer.trim().to_string();
            if line.is_empty() {
                return;
            }
            if let Some(cmd) = line.strip_prefix('/') {
                let cmd_lower = cmd.to_lowercase();
                let cmd_name = cmd_lower.split_whitespace().next().unwrap_or("");
                match cmd_name {
                    "exit" | "quit" | "q" => {
                        // Cancel any in-flight streams so we don't burn
                        // provider tokens / TCP connections after the App
                        // drops; no-op when nothing is streaming.
                        app.cancel_current_turn();
                        app.should_quit = true;
                    }
                    "clear" | "reset" => {
                        app.clear_all_panels();
                        app.clear_input();
                    }
                    _ => {
                        app.status_message = Some(format!("unknown command: /{}", cmd_name));
                        app.clear_input();
                    }
                }
                return;
            }
            // Keep the text if the turn was rejected (busy) so it isn't lost.
            if app.submit_turn(line) {
                app.clear_input();
            }
        }
        KeyCode::Backspace => app.backspace(),
        KeyCode::Delete => app.delete_forward(),
        KeyCode::Left => app.cursor_left(),
        KeyCode::Right => app.cursor_right(),
        KeyCode::Home => app.input_cursor = 0,
        KeyCode::End => app.input_cursor = app.input_buffer.len(),
        KeyCode::PageUp => app.page_scroll(-1),
        KeyCode::PageDown => app.page_scroll(1),
        KeyCode::Char('c') if ctrl => {
            if app.any_streaming() {
                app.cancel_current_turn();
                app.status_message = Some("cancelling...".into());
            } else {
                // Ctrl+C with no in-flight turn clears the input line — like
                // a shell prompt. Second press in Normal quits.
                app.clear_input();
                app.mode = Mode::Normal;
            }
        }
        // Readline-style editing chords.
        KeyCode::Char('a') if ctrl && !alt => app.input_cursor = 0,
        KeyCode::Char('e') if ctrl && !alt => app.input_cursor = app.input_buffer.len(),
        KeyCode::Char('u') if ctrl && !alt => {
            app.input_buffer.drain(..app.input_cursor);
            app.input_cursor = 0;
        }
        KeyCode::Char('k') if ctrl && !alt => {
            app.input_buffer.truncate(app.input_cursor);
        }
        KeyCode::Char('w') if ctrl && !alt => app.delete_word_back(),
        // Other Ctrl- or Alt- chords are commands, not text: don't type a
        // literal letter for them. Ctrl+Alt together is AltGr on Windows
        // and produces real characters (e.g. `@`, `€`), so let it through.
        KeyCode::Char(_) if ctrl != alt => {}
        KeyCode::Char(c) => {
            let mut buf = [0u8; 4];
            app.insert_text(c.encode_utf8(&mut buf));
        }
        _ => {}
    }
}

fn handle_mouse(app: &mut App, mouse: crossterm::event::MouseEvent) {
    match mouse.kind {
        MouseEventKind::ScrollUp => {
            let idx = app
                .panel_at(mouse.column, mouse.row)
                .unwrap_or(app.selected_panel);
            app.selected_panel = idx;
            app.scroll_by(idx, -(MOUSE_SCROLL_STEP as i32));
        }
        MouseEventKind::ScrollDown => {
            let idx = app
                .panel_at(mouse.column, mouse.row)
                .unwrap_or(app.selected_panel);
            app.selected_panel = idx;
            app.scroll_by(idx, MOUSE_SCROLL_STEP as i32);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::Provider;
    use ratatui::backend::TestBackend;

    fn fake_provider(name: &str) -> Provider {
        Provider {
            name: name.to_ascii_lowercase(),
            display_name: name.to_string(),
            model: "test-model".to_string(),
            reasoning: None,
            service_tier: None,
            color: colored::Color::Cyan,
        }
    }

    /// Build an `App` with `n` panels plus keep the paired receivers alive
    /// for the duration of the test. If the receivers drop, `try_send` sees
    /// `TrySendError::Closed` and submit_turn rolls back the user turn — so
    /// tests need to hold onto the receivers explicitly.
    fn app_with(n: usize) -> (App, Vec<mpsc::Receiver<PanelCommand>>) {
        let mut panels = Vec::with_capacity(n);
        let mut senders = Vec::with_capacity(n);
        let mut receivers = Vec::with_capacity(n);
        for i in 0..n {
            panels.push(PanelState::new(fake_provider(&format!("P{}", i))));
            let (tx, rx) = mpsc::channel::<PanelCommand>(4);
            senders.push(tx);
            receivers.push(rx);
        }
        (App::new(panels, senders, true), receivers)
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn ctrl(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL)
    }

    fn type_str(app: &mut App, s: &str) {
        for c in s.chars() {
            handle_key(app, key(KeyCode::Char(c)));
        }
    }

    fn done(provider: &str) -> StreamEvent {
        StreamEvent::Done {
            provider: provider.into(),
            model: "m".into(),
            input_tokens: None,
            output_tokens: None,
            duration_ms: 10,
        }
    }

    #[test]
    fn insert_mode_appends_chars_and_backspace_removes_them() {
        let (mut app, _rxs) = app_with(1);
        app.mode = Mode::Insert;
        type_str(&mut app, "hi");
        assert_eq!(app.input_buffer, "hi");
        handle_key(&mut app, key(KeyCode::Backspace));
        assert_eq!(app.input_buffer, "h");
    }

    #[test]
    fn app_starts_in_insert_mode() {
        // Default to Insert so users can type without first pressing 'i' —
        // matches Claude Code's no-modal UX.
        let (app, _rxs) = app_with(1);
        assert_eq!(app.mode, Mode::Insert);
    }

    #[test]
    fn normal_mode_i_enters_insert_and_esc_leaves() {
        let (mut app, _rxs) = app_with(1);
        app.mode = Mode::Normal;
        handle_key(&mut app, key(KeyCode::Char('i')));
        assert_eq!(app.mode, Mode::Insert);
        handle_key(&mut app, key(KeyCode::Esc));
        assert_eq!(app.mode, Mode::Normal);
    }

    #[test]
    fn normal_mode_q_quits() {
        let (mut app, _rxs) = app_with(1);
        app.mode = Mode::Normal;
        handle_key(&mut app, key(KeyCode::Char('q')));
        assert!(app.should_quit);
    }

    #[test]
    fn submit_turn_appends_user_turn_and_marks_streaming() {
        let (mut app, _rxs) = app_with(2);
        assert!(app.submit_turn("hello".into()));
        for panel in &app.panels {
            assert_eq!(panel.turns.len(), 1);
            assert!(matches!(panel.turns[0].kind, TurnKind::User));
            assert_eq!(panel.turns[0].text, "hello");
            assert!(panel.is_streaming);
        }
        assert!(app.current_turn_cancel.is_some());
    }

    #[test]
    fn submit_turn_blocked_while_streaming() {
        let (mut app, _rxs) = app_with(1);
        assert!(app.submit_turn("first".into()));
        assert!(!app.submit_turn("second".into()));
        assert_eq!(app.panels[0].turns.len(), 1);
        assert_eq!(app.panels[0].turns[0].text, "first");
        assert!(app.status_message.as_deref().unwrap_or("").contains("busy"));
    }

    #[test]
    fn enter_while_busy_keeps_the_typed_text() {
        let (mut app, _rxs) = app_with(1);
        app.submit_turn("first".into());
        type_str(&mut app, "follow-up");
        handle_key(&mut app, key(KeyCode::Enter));
        assert_eq!(app.input_buffer, "follow-up");
    }

    #[test]
    fn done_event_finalizes_assistant_turn() {
        let (mut app, _rxs) = app_with(1);
        app.submit_turn("hi".into());
        app.handle_event(StreamEvent::Chunk {
            provider: "P0".into(),
            text: "hello".into(),
        });
        app.handle_event(StreamEvent::Done {
            provider: "P0".into(),
            model: "test-model".into(),
            input_tokens: Some(3),
            output_tokens: Some(5),
            duration_ms: 120,
        });
        let panel = &app.panels[0];
        assert!(!panel.is_streaming);
        assert_eq!(panel.turns.len(), 2);
        match &panel.turns[1].kind {
            TurnKind::Assistant {
                input_tokens,
                output_tokens,
                duration_ms,
            } => {
                assert_eq!(*input_tokens, Some(3));
                assert_eq!(*output_tokens, Some(5));
                assert_eq!(*duration_ms, 120);
            }
            _ => panic!("expected Assistant turn"),
        }
        assert_eq!(panel.turns[1].text, "hello");
        // Once all panels settle, the shared cancel is released.
        assert!(app.current_turn_cancel.is_none());
    }

    #[test]
    fn error_event_drops_partial_text_and_records_error_turn() {
        let (mut app, _rxs) = app_with(1);
        app.submit_turn("hi".into());
        app.handle_event(StreamEvent::Chunk {
            provider: "P0".into(),
            text: "partial...".into(),
        });
        app.handle_event(StreamEvent::Error {
            provider: "P0".into(),
            model: "test-model".into(),
            error: "boom".into(),
            duration_ms: 50,
        });
        let panel = &app.panels[0];
        assert!(!panel.is_streaming);
        // User turn + error turn, no assistant turn with the partial text
        assert_eq!(panel.turns.len(), 2);
        assert!(matches!(panel.turns[1].kind, TurnKind::Error));
        assert!(panel.current_text.is_empty());
    }

    #[test]
    fn slash_exit_sets_quit() {
        let (mut app, _rxs) = app_with(1);
        app.mode = Mode::Insert;
        type_str(&mut app, "/exit");
        handle_key(&mut app, key(KeyCode::Enter));
        assert!(app.should_quit);
    }

    #[test]
    fn slash_clear_resets_visible_turns() {
        let (mut app, _rxs) = app_with(2);
        app.submit_turn("hi".into());
        // Simulate completion so the panel is no longer streaming, letting
        // /clear run against a stable state.
        app.handle_event(done("P0"));
        app.handle_event(done("P1"));
        app.mode = Mode::Insert;
        type_str(&mut app, "/clear");
        handle_key(&mut app, key(KeyCode::Enter));
        for panel in &app.panels {
            assert!(panel.turns.is_empty());
            assert!(panel.current_text.is_empty());
        }
    }

    #[test]
    fn submit_to_dead_workers_keeps_the_input() {
        let (mut app, rxs) = app_with(1);
        drop(rxs); // the panel's worker is gone
        type_str(&mut app, "question");
        handle_key(&mut app, key(KeyCode::Enter));
        assert_eq!(app.input_buffer, "question");
        assert!(app.status_message.unwrap().contains("worker stopped"));
        assert!(app.current_turn_cancel.is_none());
    }

    #[test]
    fn esc_quits_while_a_cleared_turn_is_still_draining() {
        let (mut app, _rxs) = app_with(1);
        app.submit_turn("hi".into());
        app.clear_all_panels();
        // Nothing visible is streaming, so Esc in Normal mode quits instead
        // of trying to cancel a turn that's already cancelled.
        app.mode = Mode::Normal;
        handle_key(&mut app, key(KeyCode::Esc));
        assert!(app.should_quit);
    }

    #[test]
    fn clear_mid_stream_cancels_and_discards_late_events() {
        let (mut app, _rxs) = app_with(1);
        app.submit_turn("hi".into());
        let cancel = app.current_turn_cancel.clone().unwrap();
        app.clear_all_panels();
        assert!(cancel.is_cancelled(), "/clear must stop the in-flight turn");
        // Late events from the abandoned turn must not show up...
        app.handle_event(StreamEvent::Chunk {
            provider: "P0".into(),
            text: "ghost".into(),
        });
        // ...and a new question can't start until the old one has wound down.
        assert!(!app.submit_turn("next".into()));
        app.handle_event(StreamEvent::Error {
            provider: "P0".into(),
            model: "m".into(),
            error: "cancelled".into(),
            duration_ms: 5,
        });
        assert!(app.panels[0].turns.is_empty(), "no ghost turns after clear");
        assert!(app.submit_turn("next".into()));
        app.handle_event(StreamEvent::Chunk {
            provider: "P0".into(),
            text: "real".into(),
        });
        app.handle_event(done("P0"));
        assert_eq!(app.panels[0].turns.len(), 2);
        assert_eq!(app.panels[0].turns[1].text, "real");
    }

    #[test]
    fn scroll_detaches_auto_scroll_until_bottom() {
        let (mut app, _rxs) = app_with(1);
        // Seed some scrollable content state.
        app.panels[0].content_height = 100;
        app.panels[0].visible_height = 20;
        app.panels[0].auto_scroll = true;
        app.scroll_by(0, -5);
        assert!(!app.panels[0].auto_scroll);
        // Scroll all the way back to the bottom.
        app.scroll_by(0, 1_000);
        assert!(app.panels[0].auto_scroll);
        assert_eq!(app.panels[0].scroll_offset, app.panels[0].max_scroll());
    }

    #[test]
    fn panel_at_hit_tests_against_cached_area() {
        let (mut app, _rxs) = app_with(2);
        app.panels[0].area = Some(Rect::new(0, 0, 50, 20));
        app.panels[1].area = Some(Rect::new(50, 0, 50, 20));
        assert_eq!(app.panel_at(10, 5), Some(0));
        assert_eq!(app.panel_at(60, 5), Some(1));
        assert_eq!(app.panel_at(5, 30), None);
    }

    #[test]
    fn content_height_counts_word_wrapped_rows() {
        // Word wrapping needs more rows than ceil(width / columns): these
        // six-letter words fit only one per 10-column row.
        let (mut app, _rxs) = app_with(1);
        app.panels[0].turns.push(Turn {
            kind: TurnKind::User,
            text: "aaaaaa bbbbbb cccccc dddddd".into(),
        });
        let mut terminal = Terminal::new(TestBackend::new(12, 30)).unwrap();
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 12, 30);
                draw_panel(f, &mut app.panels[0], area, true, false);
            })
            .unwrap();
        // "you" + 4 wrapped rows + trailing blank line.
        assert_eq!(app.panels[0].content_height, 6);
    }

    #[test]
    fn cursor_editing_moves_by_grapheme() {
        let (mut app, _rxs) = app_with(1);
        type_str(&mut app, "ab");
        app.insert_text("🇯🇵"); // a two-code-point flag
        type_str(&mut app, "c");
        handle_key(&mut app, key(KeyCode::Left));
        handle_key(&mut app, key(KeyCode::Backspace));
        assert_eq!(app.input_buffer, "abc", "whole flag deleted");
        handle_key(&mut app, key(KeyCode::Home));
        type_str(&mut app, ">");
        assert_eq!(app.input_buffer, ">abc");
        handle_key(&mut app, key(KeyCode::End));
        handle_key(&mut app, ctrl('w'));
        assert_eq!(app.input_buffer, "");
    }

    #[test]
    fn control_chords_do_not_insert_letters() {
        let (mut app, _rxs) = app_with(1);
        type_str(&mut app, "hello world");
        handle_key(&mut app, ctrl('w'));
        assert_eq!(app.input_buffer, "hello ");
        handle_key(&mut app, ctrl('x'));
        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('f'), KeyModifiers::ALT),
        );
        assert_eq!(app.input_buffer, "hello ");
        // AltGr (Ctrl+Alt) produces real characters on Windows layouts.
        handle_key(
            &mut app,
            KeyEvent::new(
                KeyCode::Char('@'),
                KeyModifiers::CONTROL | KeyModifiers::ALT,
            ),
        );
        assert_eq!(app.input_buffer, "hello @");
        handle_key(&mut app, ctrl('u'));
        assert_eq!(app.input_buffer, "");
    }

    #[test]
    fn long_input_scrolls_to_keep_cursor_visible() {
        let text: String = ('a'..='z').chain('A'..='D').collect();
        let (visible, col) = input_viewport(&text, text.len(), 20);
        // Scrolled 11 columns: the tail plus a free cell for the cursor.
        assert_eq!(visible, &text[11..]);
        assert_eq!(col, 19);
        let (visible, col) = input_viewport(&text, 0, 20);
        assert_eq!((visible.as_str(), col), (&text[..20], 0));
    }

    #[test]
    fn input_wider_than_u16_shows_the_text_under_the_cursor() {
        // A single-line paste wider than a u16 scroll offset can express.
        let mut text = "x".repeat(70_000);
        text.push_str("TAIL");
        let (visible, col) = input_viewport(&text, text.len(), 20);
        assert_eq!(visible, format!("{}TAIL", "x".repeat(15)));
        assert_eq!(col, 19);
    }

    #[test]
    fn wide_glyph_cut_by_the_left_edge_keeps_columns_aligned() {
        // "漢" is two columns wide; scrolling by one column cuts it in half.
        let text = "漢abcd";
        let (visible, col) = input_viewport(text, text.len(), 6);
        assert_eq!(visible, " abcd");
        assert_eq!(col, 5);
    }

    #[test]
    fn pasted_newlines_display_as_markers_but_are_kept() {
        let (mut app, _rxs) = app_with(1);
        app.insert_text("fn main() {\n}\n");
        assert_eq!(input_display(&app.input_buffer), "fn main() {↵}↵");
        assert!(app.input_buffer.contains('\n'));
    }

    #[test]
    fn duplicate_display_names_are_made_unique() {
        let mut providers = vec![fake_provider("X"), fake_provider("X"), fake_provider("Y")];
        dedupe_display_names(&mut providers);
        let names: Vec<&str> = providers.iter().map(|p| p.display_name.as_str()).collect();
        assert_eq!(names, ["X", "X (2)", "Y"]);
    }

    #[test]
    fn generated_names_skip_configured_ones() {
        let mut providers = vec![
            fake_provider("X"),
            fake_provider("X"),
            fake_provider("X (2)"),
            fake_provider("X"),
        ];
        dedupe_display_names(&mut providers);
        let names: Vec<&str> = providers.iter().map(|p| p.display_name.as_str()).collect();
        assert_eq!(names, ["X", "X (3)", "X (2)", "X (4)"]);
    }

    #[test]
    fn failed_turn_detection_ignores_tool_call_messages() {
        use genai::chat::ToolCall;
        let ok = vec![ChatMessage::user("q"), ChatMessage::assistant("a")];
        assert!(ends_with_final_reply(&ok));
        let mid_tool = vec![
            ChatMessage::user("q"),
            ChatMessage::from(vec![ToolCall {
                call_id: "c".into(),
                fn_name: "web_search".into(),
                fn_arguments: serde_json::json!({"query": "x"}),
                thought_signatures: None,
            }]),
        ];
        assert!(!ends_with_final_reply(&mid_tool));
        assert!(!ends_with_final_reply(&[ChatMessage::user("q")]));
    }

    #[test]
    fn multi_line_errors_render_one_row_per_line() {
        let turn = Turn {
            kind: TurnKind::Error,
            text: "first\nsecond".into(),
        };
        let lines = lines_for_turn(&turn, Color::Red, false);
        // header, two text lines, trailing blank
        assert_eq!(lines.len(), 4);
    }
}
