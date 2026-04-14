use crate::providers::{Provider, ProviderManager};
use crate::query::{QueryContext, StreamEvent};
use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame, Terminal,
};
use std::io::{self, Stdout};
use std::sync::Arc;
use tokio::sync::mpsc;

pub struct PanelState {
    pub provider: Provider,
    pub model: String,
    pub text: String,
    pub input_tokens: Option<i32>,
    pub output_tokens: Option<i32>,
    pub duration_ms: u64,
    pub is_complete: bool,
    pub error: Option<String>,
    pub scroll_offset: u16,
    pub content_height: u16,  // Total lines of content (with wrapping)
    pub visible_height: u16,  // Visible area height
}

impl PanelState {
    pub fn new(provider: Provider) -> Self {
        let model = provider.model.clone();
        PanelState {
            provider,
            model,
            text: String::new(),
            input_tokens: None,
            output_tokens: None,
            duration_ms: 0,
            is_complete: false,
            error: None,
            scroll_offset: 0,
            content_height: 0,
            visible_height: 0,
        }
    }

    pub fn max_scroll(&self) -> u16 {
        self.content_height.saturating_sub(self.visible_height)
    }

    pub fn append(&mut self, text: &str) {
        self.text.push_str(text);
    }

    pub fn complete(&mut self, input_tokens: Option<i32>, output_tokens: Option<i32>, duration_ms: u64) {
        self.input_tokens = input_tokens;
        self.output_tokens = output_tokens;
        self.duration_ms = duration_ms;
        self.is_complete = true;
    }

    pub fn set_error(&mut self, error: String, duration_ms: u64) {
        self.error = Some(error);
        self.duration_ms = duration_ms;
        self.is_complete = true;
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
    selected_panel: usize,
    show_tokens: bool,
    question: String,
}

impl App {
    pub fn new(providers: Vec<Provider>, question: &str, show_tokens: bool) -> Self {
        let panels = providers.into_iter().map(PanelState::new).collect();
        App {
            panels,
            selected_panel: 0,
            show_tokens,
            question: question.to_string(),
        }
    }

    pub fn handle_event(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::Start { provider, model } => {
                if let Some(panel) = self.panels.iter_mut().find(|p| p.provider.display_name == provider) {
                    panel.text.clear();
                    panel.model = model;
                }
            }
            StreamEvent::Chunk { provider, text } => {
                if let Some(panel) = self.panels.iter_mut().find(|p| p.provider.display_name == provider) {
                    panel.append(&text);
                }
            }
            StreamEvent::Done {
                provider,
                input_tokens,
                output_tokens,
                duration_ms,
                ..
            } => {
                if let Some(panel) = self.panels.iter_mut().find(|p| p.provider.display_name == provider) {
                    panel.complete(input_tokens, output_tokens, duration_ms);
                }
            }
            StreamEvent::Error { provider, error, duration_ms, .. } => {
                if let Some(panel) = self.panels.iter_mut().find(|p| p.provider.display_name == provider) {
                    panel.set_error(error, duration_ms);
                }
            }
            StreamEvent::SearchStart { provider, query } => {
                if let Some(panel) = self.panels.iter_mut().find(|p| p.provider.display_name == provider) {
                    // Insert a styled marker that the markdown renderer will italicize
                    panel.append(&format!("\n*→ searching: {}*\n\n", query));
                }
            }
            StreamEvent::SearchEnd { provider, search_provider, result_count, error, .. } => {
                if let Some(panel) = self.panels.iter_mut().find(|p| p.provider.display_name == provider) {
                    match error {
                        Some(err) => panel.append(&format!("*← search failed: {}*\n\n", err)),
                        None => panel.append(&format!(
                            "*← via {}: {} result(s)*\n\n",
                            search_provider, result_count
                        )),
                    }
                }
            }
        }
    }

    pub fn all_complete(&self) -> bool {
        self.panels.iter().all(|p| p.is_complete)
    }

    pub fn next_panel(&mut self) {
        if !self.panels.is_empty() {
            self.selected_panel = (self.selected_panel + 1) % self.panels.len();
        }
    }

    pub fn prev_panel(&mut self) {
        if !self.panels.is_empty() {
            self.selected_panel = (self.selected_panel + self.panels.len() - 1) % self.panels.len();
        }
    }

    pub fn scroll_up(&mut self) {
        if let Some(panel) = self.panels.get_mut(self.selected_panel) {
            panel.scroll_offset = panel.scroll_offset.saturating_sub(1);
        }
    }

    pub fn scroll_down(&mut self) {
        if let Some(panel) = self.panels.get_mut(self.selected_panel) {
            let max = panel.max_scroll();
            if panel.scroll_offset < max {
                panel.scroll_offset = panel.scroll_offset.saturating_add(1);
            }
        }
    }
}

fn draw_ui(frame: &mut Frame, app: &mut App) {
    let size = frame.area();

    // Main layout: header, content, footer
    let main_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header with question
            Constraint::Min(5),    // Content area for panels
            Constraint::Length(1), // Footer with help
        ])
        .split(size);

    // Draw header with question
    let header_block = Block::default()
        .borders(Borders::ALL)
        .title(" Question ")
        .style(Style::default().fg(Color::Cyan));
    let header_text = Paragraph::new(app.question.as_str())
        .block(header_block)
        .wrap(Wrap { trim: true });
    frame.render_widget(header_text, main_layout[0]);

    // Draw panels side by side
    let num_panels = app.panels.len();
    if num_panels > 0 {
        let constraints: Vec<Constraint> = (0..num_panels)
            .map(|_| Constraint::Ratio(1, num_panels as u32))
            .collect();

        let panel_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(constraints)
            .split(main_layout[1]);

        for i in 0..app.panels.len() {
            let is_selected = i == app.selected_panel;
            let area = panel_layout[i];
            let show_tokens = app.show_tokens;
            draw_panel(frame, &mut app.panels[i], area, is_selected, show_tokens);
        }
    }

    // Draw footer with help
    let help_text = if app.all_complete() {
        "←/→: select panel | ↑/↓: scroll | q: quit"
    } else {
        "Loading... | q: quit"
    };
    let footer = Paragraph::new(help_text)
        .style(Style::default().fg(Color::Gray));
    frame.render_widget(footer, main_layout[2]);
}

fn draw_panel(frame: &mut Frame, panel: &mut PanelState, area: Rect, is_selected: bool, show_tokens: bool) {
    let color = panel.get_color();
    let status = if panel.error.is_some() {
        "error"
    } else if panel.is_complete {
        "complete"
    } else {
        "streaming..."
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

    // Prepare content: parse markdown into styled lines
    let mut lines: Vec<Line> = if let Some(ref error) = panel.error {
        vec![Line::styled(
            format!("Error: {}", error),
            Style::default().fg(Color::Red),
        )]
    } else {
        crate::markdown::markdown_to_lines(&panel.text)
    };
    if panel.is_complete {
        lines.push(Line::from(""));

        let mut parts = vec![format!("model: {}", panel.model)];

        if show_tokens {
            if let (Some(input), Some(output)) = (panel.input_tokens, panel.output_tokens) {
                parts.push(format!("tokens: {} in / {} out", input, output));
            }
        }

        let duration_str = if panel.duration_ms >= 1000 {
            format!("{:.1}s", panel.duration_ms as f64 / 1000.0)
        } else {
            format!("{}ms", panel.duration_ms)
        };
        parts.push(format!("time: {}", duration_str));

        lines.push(Line::styled(
            parts.join(" | "),
            Style::default().fg(Color::DarkGray),
        ));
    }

    let text = Text::from(lines.clone());

    // Calculate content height with wrapping
    // Inner area is area minus borders (2 for top/bottom, 2 for left/right)
    let inner_width = area.width.saturating_sub(2) as usize;
    let inner_height = area.height.saturating_sub(2);

    // Count wrapped lines
    let mut total_lines: u16 = 0;
    for line in &lines {
        let line_len = line.width();
        if inner_width > 0 && line_len > 0 {
            total_lines += ((line_len + inner_width - 1) / inner_width) as u16;
        } else {
            total_lines += 1;
        }
    }

    // Update panel dimensions for scroll bounds
    panel.content_height = total_lines;
    panel.visible_height = inner_height;

    // Clamp scroll offset if content shrank
    let max_scroll = panel.max_scroll();
    if panel.scroll_offset > max_scroll {
        panel.scroll_offset = max_scroll;
    }

    let paragraph = Paragraph::new(text)
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((panel.scroll_offset, 0));

    frame.render_widget(paragraph, area);
}

pub async fn run_tui(
    manager: Arc<ProviderManager>,
    providers: Vec<Provider>,
    ctx: QueryContext,
    question: &str,
    show_tokens: bool,
) -> Result<()> {
    // Check if we have a proper terminal
    use std::io::IsTerminal;
    if !io::stdout().is_terminal() {
        anyhow::bail!("TUI mode requires an interactive terminal. Use default streaming mode instead.");
    }

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app state
    let mut app = App::new(providers.clone(), question, show_tokens);

    // Start streaming
    let rx = crate::query::stream_all(manager, providers, ctx).await;

    // Run the main loop
    let result = run_main_loop(&mut terminal, &mut app, rx).await;

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

async fn run_main_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    app: &mut App,
    mut rx: mpsc::Receiver<StreamEvent>,
) -> Result<()> {
    loop {
        // Draw UI
        terminal.draw(|frame| draw_ui(frame, app))?;

        // Poll for events with a short timeout
        if event::poll(std::time::Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                        KeyCode::Left | KeyCode::Char('h') => app.prev_panel(),
                        KeyCode::Right | KeyCode::Char('l') | KeyCode::Tab => app.next_panel(),
                        KeyCode::Up | KeyCode::Char('k') => app.scroll_up(),
                        KeyCode::Down | KeyCode::Char('j') => app.scroll_down(),
                        _ => {}
                    }
                }
            }
        }

        // Try to receive stream events
        while let Ok(event) = rx.try_recv() {
            app.handle_event(event);
        }
    }
}
