use crate::config::{Config, ModelTier, QueryMode};
use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Tabs},
};
use std::io::stdout;

#[derive(Clone, Copy, PartialEq)]
enum Section {
    General,
    CliMode,
    TuiMode,
    Display,
    Providers,
}

impl Section {
    fn title(&self) -> &str {
        match self {
            Section::General => "General",
            Section::CliMode => "CLI Mode",
            Section::TuiMode => "TUI Mode",
            Section::Display => "Display",
            Section::Providers => "Providers",
        }
    }

    fn all() -> Vec<Section> {
        vec![Section::General, Section::CliMode, Section::TuiMode, Section::Display, Section::Providers]
    }

    fn index(&self) -> usize {
        match self {
            Section::General => 0,
            Section::CliMode => 1,
            Section::TuiMode => 2,
            Section::Display => 3,
            Section::Providers => 4,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum ModeField {
    DefaultMode,
    SingleProvider,
    MultiProviders,
    ModelTier,
}

#[derive(Clone, Copy, PartialEq)]
enum DisplayField {
    ShowModel,
    ShowTime,
    ShowTokens,
    ShowReasoning,
}

#[derive(Clone, Copy, PartialEq)]
enum GeneralField {
    DefaultInterface,
}

struct App {
    config: Config,
    original_config: Config,
    section: Section,
    general_field: GeneralField,
    mode_field: ModeField,
    display_field: DisplayField,
    provider_index: usize,
    multi_provider_index: usize,
    available_providers: Vec<String>,
    should_quit: bool,
    message: Option<String>,
}

impl App {
    fn new(config: Config) -> Self {
        let original = config.clone();
        let mut available_providers: Vec<String> = config
            .providers
            .keys()
            .cloned()
            .collect();
        available_providers.sort();

        App {
            config,
            original_config: original,
            section: Section::General,
            general_field: GeneralField::DefaultInterface,
            mode_field: ModeField::DefaultMode,
            display_field: DisplayField::ShowModel,
            provider_index: 0,
            multi_provider_index: 0,
            available_providers,
            should_quit: false,
            message: None,
        }
    }

    fn has_changes(&self) -> bool {
        self.config.cli != self.original_config.cli
            || self.config.tui_mode != self.original_config.tui_mode
            || self.config.default.tui != self.original_config.default.tui
            || self.config.footer.show_model != self.original_config.footer.show_model
            || self.config.footer.show_time != self.original_config.footer.show_time
            || self.config.footer.show_tokens != self.original_config.footer.show_tokens
            || self.config.footer.show_reasoning != self.original_config.footer.show_reasoning
    }

    fn save(&mut self) -> Result<()> {
        self.config.save()?;
        self.original_config = self.config.clone();
        self.message = Some("Config saved!".to_string());
        Ok(())
    }

    fn next_section(&mut self) {
        self.section = match self.section {
            Section::General => Section::CliMode,
            Section::CliMode => Section::TuiMode,
            Section::TuiMode => Section::Display,
            Section::Display => Section::Providers,
            Section::Providers => Section::General,
        };
        // Reset field selection when changing sections
        self.general_field = GeneralField::DefaultInterface;
        self.mode_field = ModeField::DefaultMode;
        self.display_field = DisplayField::ShowModel;
    }

    fn prev_section(&mut self) {
        self.section = match self.section {
            Section::General => Section::Providers,
            Section::CliMode => Section::General,
            Section::TuiMode => Section::CliMode,
            Section::Display => Section::TuiMode,
            Section::Providers => Section::Display,
        };
        self.general_field = GeneralField::DefaultInterface;
        self.mode_field = ModeField::DefaultMode;
        self.display_field = DisplayField::ShowModel;
    }

    fn next_field(&mut self) {
        match self.section {
            Section::General => {
                // Only one field for now, no-op
            }
            Section::CliMode | Section::TuiMode => {
                self.mode_field = match self.mode_field {
                    ModeField::DefaultMode => ModeField::SingleProvider,
                    ModeField::SingleProvider => ModeField::MultiProviders,
                    ModeField::MultiProviders => ModeField::ModelTier,
                    ModeField::ModelTier => ModeField::DefaultMode,
                };
            }
            Section::Display => {
                self.display_field = match self.display_field {
                    DisplayField::ShowModel => DisplayField::ShowTime,
                    DisplayField::ShowTime => DisplayField::ShowTokens,
                    DisplayField::ShowTokens => DisplayField::ShowReasoning,
                    DisplayField::ShowReasoning => DisplayField::ShowModel,
                };
            }
            Section::Providers => {
                let count = self.available_providers.len();
                if count > 0 {
                    self.provider_index = (self.provider_index + 1) % count;
                }
            }
        }
    }

    fn prev_field(&mut self) {
        match self.section {
            Section::General => {
                // Only one field for now, no-op
            }
            Section::CliMode | Section::TuiMode => {
                self.mode_field = match self.mode_field {
                    ModeField::DefaultMode => ModeField::ModelTier,
                    ModeField::SingleProvider => ModeField::DefaultMode,
                    ModeField::MultiProviders => ModeField::SingleProvider,
                    ModeField::ModelTier => ModeField::MultiProviders,
                };
            }
            Section::Display => {
                self.display_field = match self.display_field {
                    DisplayField::ShowModel => DisplayField::ShowReasoning,
                    DisplayField::ShowTime => DisplayField::ShowModel,
                    DisplayField::ShowTokens => DisplayField::ShowTime,
                    DisplayField::ShowReasoning => DisplayField::ShowTokens,
                };
            }
            Section::Providers => {
                let count = self.available_providers.len();
                if count > 0 {
                    if self.provider_index == 0 {
                        self.provider_index = count - 1;
                    } else {
                        self.provider_index -= 1;
                    }
                }
            }
        }
    }

    fn get_current_mode_config(&mut self) -> &mut crate::config::ModeConfig {
        match self.section {
            Section::CliMode => &mut self.config.cli,
            Section::TuiMode => &mut self.config.tui_mode,
            _ => &mut self.config.cli, // fallback
        }
    }

    fn get_current_mode_config_ref(&self) -> &crate::config::ModeConfig {
        match self.section {
            Section::CliMode => &self.config.cli,
            Section::TuiMode => &self.config.tui_mode,
            _ => &self.config.cli,
        }
    }

    fn toggle_or_cycle(&mut self) {
        match self.section {
            Section::General => {
                match self.general_field {
                    GeneralField::DefaultInterface => {
                        self.config.default.tui = !self.config.default.tui;
                    }
                }
            }
            Section::CliMode | Section::TuiMode => {
                match self.mode_field {
                    ModeField::DefaultMode => {
                        let mode = self.get_current_mode_config();
                        mode.default_mode = match mode.default_mode {
                            QueryMode::Single => QueryMode::Multi,
                            QueryMode::Multi => QueryMode::Single,
                        };
                    }
                    ModeField::SingleProvider => {
                        // Cycle through available provider aliases
                        let aliases: Vec<String> = self
                            .config
                            .providers
                            .values()
                            .filter_map(|p| p.aliases.first().cloned())
                            .collect();
                        let current = self.get_current_mode_config_ref().single_provider.clone();
                        if let Some(pos) = aliases.iter().position(|a| *a == current) {
                            let next = (pos + 1) % aliases.len();
                            self.get_current_mode_config().single_provider = aliases[next].clone();
                        } else if !aliases.is_empty() {
                            self.get_current_mode_config().single_provider = aliases[0].clone();
                        }
                    }
                    ModeField::MultiProviders => {
                        // Toggle the currently selected provider
                        let aliases: Vec<String> = self
                            .config
                            .providers
                            .values()
                            .filter_map(|p| p.aliases.first().cloned())
                            .collect();
                        if self.multi_provider_index < aliases.len() {
                            let alias = aliases[self.multi_provider_index].clone();
                            let mode = self.get_current_mode_config();
                            if mode.multi_providers.contains(&alias) {
                                mode.multi_providers.retain(|p| p != &alias);
                            } else {
                                mode.multi_providers.push(alias);
                            }
                        }
                    }
                    ModeField::ModelTier => {
                        let mode = self.get_current_mode_config();
                        mode.model_tier = match mode.model_tier {
                            ModelTier::Fast => ModelTier::Smart,
                            ModelTier::Smart => ModelTier::Fast,
                        };
                    }
                }
            }
            Section::Display => match self.display_field {
                DisplayField::ShowModel => {
                    self.config.footer.show_model = !self.config.footer.show_model;
                }
                DisplayField::ShowTime => {
                    self.config.footer.show_time = !self.config.footer.show_time;
                }
                DisplayField::ShowTokens => {
                    self.config.footer.show_tokens = !self.config.footer.show_tokens;
                }
                DisplayField::ShowReasoning => {
                    self.config.footer.show_reasoning = !self.config.footer.show_reasoning;
                }
            },
            Section::Providers => {
                if self.provider_index < self.available_providers.len() {
                    let provider_name = &self.available_providers[self.provider_index];
                    if let Some(p) = self.config.providers.get_mut(provider_name) {
                        p.enabled = !p.enabled;
                    }
                }
            }
        }
    }

    fn handle_multi_provider_nav(&mut self, left: bool) {
        if matches!(self.section, Section::CliMode | Section::TuiMode)
            && self.mode_field == ModeField::MultiProviders
        {
            let count = self.config.providers.len();
            if count > 0 {
                if left {
                    if self.multi_provider_index == 0 {
                        self.multi_provider_index = count - 1;
                    } else {
                        self.multi_provider_index -= 1;
                    }
                } else {
                    self.multi_provider_index = (self.multi_provider_index + 1) % count;
                }
            }
        }
    }
}

pub async fn run_config_ui() -> Result<()> {
    let config = Config::load()?;
    let mut app = App::new(config);

    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    loop {
        terminal.draw(|frame| draw_ui(frame, &app))?;

        let event = event::read()?;
        let key = match event {
            Event::Key(key) if key.kind == KeyEventKind::Press => key,
            _ => continue,
        };

        app.message = None;

        match (key.code, key.modifiers) {
            (KeyCode::Char('q'), _) | (KeyCode::Esc, _) => {
                if app.has_changes() {
                    app.message = Some("Unsaved changes! Press Ctrl+S to save, or q again to discard".to_string());
                    if app.should_quit {
                        break;
                    }
                    app.should_quit = true;
                } else {
                    break;
                }
            }
            (KeyCode::Char('s'), KeyModifiers::CONTROL) => {
                if let Err(e) = app.save() {
                    app.message = Some(format!("Save failed: {}", e));
                }
                app.should_quit = false;
            }
            (KeyCode::Tab, _) => {
                app.next_section();
                app.should_quit = false;
            }
            (KeyCode::BackTab, _) => {
                app.prev_section();
                app.should_quit = false;
            }
            (KeyCode::Down, _) | (KeyCode::Char('j'), _) => {
                app.next_field();
                app.should_quit = false;
            }
            (KeyCode::Up, _) | (KeyCode::Char('k'), _) => {
                app.prev_field();
                app.should_quit = false;
            }
            (KeyCode::Left, _) | (KeyCode::Char('h'), _) => {
                app.handle_multi_provider_nav(true);
                app.should_quit = false;
            }
            (KeyCode::Right, _) | (KeyCode::Char('l'), _) => {
                app.handle_multi_provider_nav(false);
                app.should_quit = false;
            }
            (KeyCode::Enter, _) | (KeyCode::Char(' '), _) => {
                app.toggle_or_cycle();
                app.should_quit = false;
            }
            _ => {
                app.should_quit = false;
            }
        }
    }

    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;

    Ok(())
}

fn draw_ui(frame: &mut Frame, app: &App) {
    let area = frame.area();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(3),
        ])
        .split(area);

    // Header with section tabs
    let sections = Section::all();
    let titles: Vec<&str> = sections.iter().map(|s| s.title()).collect();
    let tabs = Tabs::new(titles)
        .block(Block::default().borders(Borders::ALL).title(" AI CLI Configuration "))
        .select(app.section.index())
        .style(Style::default().fg(Color::White))
        .highlight_style(Style::default().fg(Color::Yellow).bold());
    frame.render_widget(tabs, chunks[0]);

    // Content area
    match app.section {
        Section::General => draw_general_section(frame, chunks[1], app),
        Section::CliMode => draw_mode_section(frame, chunks[1], app, "CLI Mode (non-TUI)", &app.config.cli),
        Section::TuiMode => draw_mode_section(frame, chunks[1], app, "TUI Mode", &app.config.tui_mode),
        Section::Display => draw_display_section(frame, chunks[1], app),
        Section::Providers => draw_providers_section(frame, chunks[1], app),
    }

    // Footer
    let modified = if app.has_changes() { " [MODIFIED]" } else { "" };
    let msg = app.message.as_deref().unwrap_or("");
    let help = format!(
        " Tab: sections | ↑↓: navigate | ←→: multi select | Enter/Space: toggle | Ctrl+S: save | q: quit{}  {}",
        modified, msg
    );
    let footer = Paragraph::new(help)
        .style(Style::default().fg(Color::DarkGray))
        .block(Block::default().borders(Borders::ALL));
    frame.render_widget(footer, chunks[2]);
}

fn draw_general_section(frame: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" General Settings ");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let items = vec![
        format_mode_item(
            "Default Interface",
            if app.config.default.tui {
                "CLI / [TUI]"
            } else {
                "[CLI] / TUI"
            },
            app.general_field == GeneralField::DefaultInterface,
        ),
    ];

    let list = List::new(items);
    frame.render_widget(list, inner);
}

fn draw_mode_section(frame: &mut Frame, area: Rect, app: &App, title: &str, mode: &crate::config::ModeConfig) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!(" {} ", title));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let items = vec![
        format_mode_item(
            "Default Mode",
            match mode.default_mode {
                QueryMode::Single => "[Single] / Multi",
                QueryMode::Multi => "Single / [Multi]",
            },
            app.mode_field == ModeField::DefaultMode,
        ),
        format_mode_item(
            "Single Provider",
            &mode.single_provider,
            app.mode_field == ModeField::SingleProvider,
        ),
        format_multi_providers_item(app, mode),
        format_mode_item(
            "Model Tier",
            match mode.model_tier {
                ModelTier::Fast => "[Fast] / Smart",
                ModelTier::Smart => "Fast / [Smart]",
            },
            app.mode_field == ModeField::ModelTier,
        ),
    ];

    let list = List::new(items);
    frame.render_widget(list, inner);
}

fn format_mode_item<'a>(label: &str, value: &str, selected: bool) -> ListItem<'a> {
    let style = if selected {
        Style::default().fg(Color::Yellow).bold()
    } else {
        Style::default()
    };
    let prefix = if selected { "▶ " } else { "  " };
    ListItem::new(format!("{}{}: {}", prefix, label, value)).style(style)
}

fn format_multi_providers_item(app: &App, mode: &crate::config::ModeConfig) -> ListItem<'static> {
    let selected = app.mode_field == ModeField::MultiProviders;
    let style = if selected {
        Style::default().fg(Color::Yellow).bold()
    } else {
        Style::default()
    };
    let prefix = if selected { "▶ " } else { "  " };

    let aliases: Vec<String> = app
        .config
        .providers
        .values()
        .filter_map(|p| p.aliases.first().cloned())
        .collect();

    let checkboxes: Vec<String> = aliases
        .iter()
        .enumerate()
        .map(|(i, alias)| {
            let checked = if mode.multi_providers.contains(alias) { "x" } else { " " };
            if selected && i == app.multi_provider_index {
                format!(">[{}] {}<", checked, alias)
            } else {
                format!("[{}] {}", checked, alias)
            }
        })
        .collect();

    ListItem::new(format!("{}Multi Providers: {}", prefix, checkboxes.join(" "))).style(style)
}

fn draw_display_section(frame: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Display Settings ");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let items = vec![
        format_bool_item("Show Model", app.config.footer.show_model, app.display_field == DisplayField::ShowModel),
        format_bool_item("Show Time", app.config.footer.show_time, app.display_field == DisplayField::ShowTime),
        format_bool_item("Show Tokens", app.config.footer.show_tokens, app.display_field == DisplayField::ShowTokens),
        format_bool_item("Show Reasoning", app.config.footer.show_reasoning, app.display_field == DisplayField::ShowReasoning),
    ];

    let list = List::new(items);
    frame.render_widget(list, inner);
}

fn format_bool_item<'a>(label: &str, value: bool, selected: bool) -> ListItem<'a> {
    let style = if selected {
        Style::default().fg(Color::Yellow).bold()
    } else {
        Style::default()
    };
    let prefix = if selected { "▶ " } else { "  " };
    let checkbox = if value { "[x]" } else { "[ ]" };
    ListItem::new(format!("{}{} {}", prefix, checkbox, label)).style(style)
}

fn draw_providers_section(frame: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Providers ");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(30), Constraint::Percentage(70)])
        .split(inner);

    let provider_items: Vec<ListItem> = app
        .available_providers
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let enabled = app
                .config
                .providers
                .get(name)
                .map(|p| p.enabled)
                .unwrap_or(false);
            let checkbox = if enabled { "[x]" } else { "[ ]" };
            let alias = app
                .config
                .providers
                .get(name)
                .and_then(|p| p.aliases.first())
                .map(|s| s.as_str())
                .unwrap_or(name);
            let style = if i == app.provider_index {
                Style::default().fg(Color::Yellow).bold()
            } else {
                Style::default()
            };
            ListItem::new(format!("{} {}", checkbox, alias)).style(style)
        })
        .collect();

    let mut list_state = ListState::default();
    list_state.select(Some(app.provider_index));

    let list = List::new(provider_items)
        .block(Block::default().borders(Borders::RIGHT))
        .highlight_style(Style::default().bg(Color::DarkGray));
    frame.render_stateful_widget(list, chunks[0], &mut list_state);

    if app.provider_index < app.available_providers.len() {
        let provider_name = &app.available_providers[app.provider_index];
        if let Some(provider) = app.config.providers.get(provider_name) {
            let fast = provider.fast_model.as_deref().unwrap_or(&provider.default_model);
            let smart = provider.smart_model.as_deref().unwrap_or(&provider.default_model);
            let details = vec![
                format!("Provider: {}", provider_name),
                format!("Enabled: {}", if provider.enabled { "Yes" } else { "No" }),
                format!("Fast Model: {}", fast),
                format!("Smart Model: {}", smart),
                format!("Color: {}", provider.color),
            ];
            let details_text = details.join("\n");
            let details_para = Paragraph::new(details_text)
                .style(Style::default().fg(Color::White))
                .block(Block::default().borders(Borders::NONE));
            frame.render_widget(details_para, chunks[1]);
        }
    }
}
