use crate::providers::{Provider, ProviderManager};
use crate::query::{stream_with_messages, QueryContext, StreamEvent};
use anyhow::Result;
use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers,
        MouseEventKind,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use genai::chat::{ChatMessage, ChatRole};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::Line,
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame, Terminal,
};
use std::io::{self, Stdout};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Three lines per wheel tick — single-line feels sluggish, half-page feels too
/// jumpy. Matches most terminal pagers.
const MOUSE_SCROLL_STEP: u16 = 3;

/// Backpressure on the panel input queue. 8 is generous — a user typing
/// follow-ups can't plausibly outrun a task that processes each turn in
/// seconds. If it does saturate, we show a status message rather than blocking
/// the UI thread.
const PANEL_INPUT_CAPACITY: usize = 8;

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
    selected_panel: usize,
    show_tokens: bool,
    mode: Mode,
    input_buffer: String,
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
            selected_panel: 0,
            show_tokens,
            mode: Mode::Normal,
            input_buffer: String::new(),
            should_quit: false,
            status_message: None,
            current_turn_cancel: None,
        }
    }

    fn any_streaming(&self) -> bool {
        self.panels.iter().any(|p| p.is_streaming)
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
                    p.model = model;
                }
            }
            StreamEvent::Chunk { provider, text } => {
                if let Some(p) = self.find_panel_mut(&provider) {
                    p.current_text.push_str(&text);
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
                self.maybe_clear_turn_cancel();
            }
            StreamEvent::Error {
                provider,
                error,
                duration_ms,
                ..
            } => {
                if let Some(p) = self.find_panel_mut(&provider) {
                    // Drop any partial streamed text — it's incomplete and
                    // mixing it into an "error" turn is misleading.
                    p.current_text.clear();
                    let dur = if duration_ms >= 1000 {
                        format!("{:.1}s", duration_ms as f64 / 1000.0)
                    } else {
                        format!("{}ms", duration_ms)
                    };
                    p.turns.push(Turn {
                        kind: TurnKind::Error,
                        text: format!("{} ({})", error, dur),
                    });
                    p.is_streaming = false;
                }
                self.maybe_clear_turn_cancel();
            }
            StreamEvent::SearchStart { provider, query } => {
                if let Some(p) = self.find_panel_mut(&provider) {
                    // Appears inline in the streaming text; italic markdown so
                    // the markdown renderer styles it distinctly.
                    p.current_text
                        .push_str(&format!("\n*→ searching: {}*\n\n", query));
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
                    match error {
                        Some(err) => p
                            .current_text
                            .push_str(&format!("*← search failed: {}*\n\n", err)),
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
        if !self.any_streaming() {
            self.current_turn_cancel = None;
        }
    }

    fn submit_turn(&mut self, question: String) {
        if self.any_streaming() {
            self.status_message = Some("busy — wait for current turn to finish".into());
            return;
        }
        if self.senders.is_empty() {
            return;
        }
        let cancel = CancellationToken::new();
        self.current_turn_cancel = Some(cancel.clone());
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
            if tx
                .try_send(PanelCommand::Ask {
                    question: question.clone(),
                    cancel: cancel.clone(),
                })
                .is_err()
            {
                panel.is_streaming = false;
                // Roll back the user turn we just added so the history doesn't
                // show a question that will never be answered.
                panel.turns.pop();
                self.status_message = Some(format!(
                    "{} is unavailable (input queue full)",
                    panel.provider.display_name
                ));
            }
        }
    }

    fn cancel_current_turn(&mut self) {
        if let Some(cancel) = &self.current_turn_cancel {
            cancel.cancel();
        }
    }

    fn clear_all_panels(&mut self) {
        for (i, tx) in self.senders.iter().enumerate() {
            // Fire-and-forget: if the receiver is gone the task is ending
            // anyway, and the visible state is reset regardless.
            let _ = tx.try_send(PanelCommand::Clear);
            let p = &mut self.panels[i];
            p.turns.clear();
            p.current_text.clear();
            p.is_streaming = false;
            p.scroll_offset = 0;
            p.auto_scroll = true;
        }
        self.current_turn_cancel = None;
        self.status_message = Some("cleared".into());
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
                out.push(Line::from(format!("  {}", l)));
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
            let dur = if *duration_ms >= 1000 {
                format!("{:.1}s", *duration_ms as f64 / 1000.0)
            } else {
                format!("{}ms", duration_ms)
            };
            stats.push(dur);
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
            out.push(Line::styled(
                turn.text.clone(),
                Style::default().fg(Color::Red),
            ));
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
    let title = format!(" {} [{}] ", panel.provider.display_name, status);

    let mut border_style = Style::default().fg(color);
    if is_selected {
        border_style = border_style.add_modifier(Modifier::BOLD);
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
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

    let inner_width = area.width.saturating_sub(2) as usize;
    let inner_height = area.height.saturating_sub(2);

    let mut total_lines: u16 = 0;
    for line in &lines {
        let line_len = line.width();
        if inner_width > 0 && line_len > 0 {
            total_lines = total_lines.saturating_add(line_len.div_ceil(inner_width) as u16);
        } else {
            total_lines = total_lines.saturating_add(1);
        }
    }

    panel.content_height = total_lines;
    panel.visible_height = inner_height;

    let max_scroll = panel.max_scroll();
    // Re-pin to bottom under auto-scroll, or clamp an over-scrolled offset if
    // content shrank (e.g., after /clear). Both branches collapse to the same
    // assignment — we keep the structure so the intent is legible even if
    // clippy sees an identical block.
    if panel.auto_scroll || panel.scroll_offset > max_scroll {
        panel.scroll_offset = max_scroll;
    }

    let paragraph = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((panel.scroll_offset, 0));

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

    // Input box. In Insert mode we show a real cursor position inside the
    // block; in Normal mode the buffer still displays but no cursor is drawn.
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
    let input_paragraph = Paragraph::new(app.input_buffer.as_str())
        .block(input_block)
        .wrap(Wrap { trim: false });
    frame.render_widget(input_paragraph, input_area);

    if app.mode == Mode::Insert {
        // Place the cursor one column past the end of the buffer, inside the
        // bordered area. Width measured in Unicode columns via `width()` so
        // multi-byte chars don't drift the cursor.
        let inner_width = input_area.width.saturating_sub(2) as usize;
        let buf_width = unicode_display_width(&app.input_buffer);
        let (col_in_inner, row_offset) = if inner_width == 0 {
            (0, 0)
        } else {
            let col = buf_width % inner_width;
            let row = buf_width / inner_width;
            (col as u16, row as u16)
        };
        let cx = input_area.x + 1 + col_in_inner;
        let cy = input_area.y + 1 + row_offset;
        frame.set_cursor_position((cx, cy));
    }

    // Footer: mode-appropriate keybind hint, or a transient status message.
    let help_text: String = if let Some(msg) = &app.status_message {
        msg.clone()
    } else {
        match app.mode {
            Mode::Insert => {
                "Esc: normal | Enter: send | Ctrl+C: cancel turn | /clear /exit".to_string()
            }
            Mode::Normal => {
                "i: insert | h/l or ←/→: select panel | j/k or ↑/↓: scroll | q: quit".to_string()
            }
        }
    };
    let footer = Paragraph::new(help_text).style(Style::default().fg(Color::Gray));
    frame.render_widget(footer, main_layout[2]);
}

/// Unicode display width — counts columns, not bytes, so multi-byte and
/// wide characters don't desync the cursor. `Line::width` from ratatui does
/// the same thing for styled lines; we use it here for the raw input buffer.
fn unicode_display_width(s: &str) -> usize {
    use ratatui::text::Text;
    Text::from(s.to_string()).width()
}

pub async fn run_tui(
    manager: Arc<ProviderManager>,
    providers: Vec<Provider>,
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

    // Spawn a task per panel; each owns its own history and listens on its
    // own mpsc<PanelCommand>. Stream events from all panels fan in to a
    // shared event channel the UI consumes.
    let (event_tx, event_rx) = mpsc::channel::<StreamEvent>(100);
    let mut senders: Vec<mpsc::Sender<PanelCommand>> = Vec::with_capacity(providers.len());
    let mut panels: Vec<PanelState> = Vec::with_capacity(providers.len());

    for provider in &providers {
        let (cmd_tx, cmd_rx) = mpsc::channel::<PanelCommand>(PANEL_INPUT_CAPACITY);
        senders.push(cmd_tx);
        panels.push(PanelState::new(provider.clone()));

        let task_manager = Arc::clone(&manager);
        let task_provider = provider.clone();
        let task_ctx = ctx.clone();
        let task_event_tx = event_tx.clone();
        tokio::spawn(async move {
            panel_task(task_manager, task_provider, task_ctx, cmd_rx, task_event_tx).await;
        });
    }
    drop(event_tx); // only panel tasks should hold senders; drop our copy

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new(panels, senders, show_tokens);

    // Seed the first turn with the CLI-provided question, if any.
    let initial = initial_question.trim();
    if !initial.is_empty() {
        app.submit_turn(initial.to_string());
    }

    let result = run_main_loop(&mut terminal, &mut app, event_rx).await;

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        DisableMouseCapture,
        LeaveAlternateScreen
    )?;
    terminal.show_cursor()?;

    result
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
                let pre_assistant_count = messages
                    .iter()
                    .filter(|m| matches!(m.role, ChatRole::Assistant))
                    .count();
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
                let post_assistant_count = enriched
                    .iter()
                    .filter(|m| matches!(m.role, ChatRole::Assistant))
                    .count();
                if post_assistant_count > pre_assistant_count {
                    messages = enriched;
                } else {
                    // No new assistant message was persisted — error or
                    // cancel. Roll back the user turn so a retry doesn't
                    // double-stack duplicated user messages.
                    messages.pop();
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
    loop {
        terminal.draw(|frame| draw_ui(frame, app))?;

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
                _ => {}
            }
        }

        while let Ok(event) = event_rx.try_recv() {
            app.handle_event(event);
        }
    }
}

fn handle_key(app: &mut App, key: crossterm::event::KeyEvent) {
    match app.mode {
        Mode::Normal => handle_normal_key(app, key),
        Mode::Insert => handle_insert_key(app, key),
    }
}

fn handle_normal_key(app: &mut App, key: crossterm::event::KeyEvent) {
    match key.code {
        KeyCode::Char('q') => app.should_quit = true,
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
        KeyCode::PageUp => {
            let idx = app.selected_panel;
            let page = app
                .panels
                .get(idx)
                .map(|p| p.visible_height.saturating_sub(1) as i32)
                .unwrap_or(10)
                .max(1);
            app.scroll_by(idx, -page);
        }
        KeyCode::PageDown => {
            let idx = app.selected_panel;
            let page = app
                .panels
                .get(idx)
                .map(|p| p.visible_height.saturating_sub(1) as i32)
                .unwrap_or(10)
                .max(1);
            app.scroll_by(idx, page);
        }
        _ => {}
    }
}

fn handle_insert_key(app: &mut App, key: crossterm::event::KeyEvent) {
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
                        app.should_quit = true;
                    }
                    "clear" | "reset" => {
                        app.clear_all_panels();
                        app.input_buffer.clear();
                    }
                    _ => {
                        app.status_message = Some(format!("unknown command: /{}", cmd_name));
                        app.input_buffer.clear();
                    }
                }
                return;
            }
            app.submit_turn(line);
            app.input_buffer.clear();
        }
        KeyCode::Backspace => {
            // Pop a full grapheme's worth of bytes by draining the trailing
            // char; `String::pop` is char-aware and therefore UTF-8 safe.
            app.input_buffer.pop();
        }
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            if app.any_streaming() {
                app.cancel_current_turn();
                app.status_message = Some("cancelling...".into());
            } else {
                // Ctrl+C with no in-flight turn clears the input line — like
                // a shell prompt. Second press in Normal quits.
                app.input_buffer.clear();
                app.mode = Mode::Normal;
            }
        }
        KeyCode::Char(c) => {
            app.input_buffer.push(c);
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

    fn fake_provider(name: &str) -> Provider {
        Provider {
            display_name: name.to_string(),
            model: "test-model".to_string(),
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

    #[test]
    fn insert_mode_appends_chars_and_backspace_removes_them() {
        let (mut app, _rxs) = app_with(1);
        app.mode = Mode::Insert;
        handle_key(
            &mut app,
            crossterm::event::KeyEvent::new(KeyCode::Char('h'), KeyModifiers::NONE),
        );
        handle_key(
            &mut app,
            crossterm::event::KeyEvent::new(KeyCode::Char('i'), KeyModifiers::NONE),
        );
        assert_eq!(app.input_buffer, "hi");
        handle_key(
            &mut app,
            crossterm::event::KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE),
        );
        assert_eq!(app.input_buffer, "h");
    }

    #[test]
    fn normal_mode_i_enters_insert_and_esc_leaves() {
        let (mut app, _rxs) = app_with(1);
        assert_eq!(app.mode, Mode::Normal);
        handle_key(
            &mut app,
            crossterm::event::KeyEvent::new(KeyCode::Char('i'), KeyModifiers::NONE),
        );
        assert_eq!(app.mode, Mode::Insert);
        handle_key(
            &mut app,
            crossterm::event::KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE),
        );
        assert_eq!(app.mode, Mode::Normal);
    }

    #[test]
    fn normal_mode_q_quits() {
        let (mut app, _rxs) = app_with(1);
        handle_key(
            &mut app,
            crossterm::event::KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE),
        );
        assert!(app.should_quit);
    }

    #[test]
    fn submit_turn_appends_user_turn_and_marks_streaming() {
        let (mut app, _rxs) = app_with(2);
        app.submit_turn("hello".into());
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
        app.submit_turn("first".into());
        app.submit_turn("second".into());
        assert_eq!(app.panels[0].turns.len(), 1);
        assert_eq!(app.panels[0].turns[0].text, "first");
        assert!(app.status_message.as_deref().unwrap_or("").contains("busy"));
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
        for c in "/exit".chars() {
            handle_key(
                &mut app,
                crossterm::event::KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE),
            );
        }
        handle_key(
            &mut app,
            crossterm::event::KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
        );
        assert!(app.should_quit);
    }

    #[test]
    fn slash_clear_resets_visible_turns() {
        let (mut app, _rxs) = app_with(2);
        app.submit_turn("hi".into());
        // Simulate completion so the panel is no longer streaming, letting
        // /clear run against a stable state.
        app.handle_event(StreamEvent::Done {
            provider: "P0".into(),
            model: "m".into(),
            input_tokens: None,
            output_tokens: None,
            duration_ms: 10,
        });
        app.handle_event(StreamEvent::Done {
            provider: "P1".into(),
            model: "m".into(),
            input_tokens: None,
            output_tokens: None,
            duration_ms: 10,
        });
        app.mode = Mode::Insert;
        app.input_buffer = "/clear".into();
        handle_key(
            &mut app,
            crossterm::event::KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
        );
        for panel in &app.panels {
            assert!(panel.turns.is_empty());
            assert!(panel.current_text.is_empty());
        }
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
}
