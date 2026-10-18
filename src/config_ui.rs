use crate::config::{Config, ModelTier, QueryMode};
use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Tabs, Wrap},
};
use std::io::stdout;

#[derive(Clone, Copy, PartialEq)]
enum Section {
    General,
    CliMode,
    TuiMode,
    Display,
    Search,
    Providers,
}

impl Section {
    fn title(&self) -> &str {
        match self {
            Section::General => "General",
            Section::CliMode => "CLI Mode",
            Section::TuiMode => "TUI Mode",
            Section::Display => "Display",
            Section::Search => "Search",
            Section::Providers => "Providers",
        }
    }

    fn all() -> Vec<Section> {
        vec![
            Section::General,
            Section::CliMode,
            Section::TuiMode,
            Section::Display,
            Section::Search,
            Section::Providers,
        ]
    }

    fn index(&self) -> usize {
        match self {
            Section::General => 0,
            Section::CliMode => 1,
            Section::TuiMode => 2,
            Section::Display => 3,
            Section::Search => 4,
            Section::Providers => 5,
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
#[allow(clippy::enum_variant_names)]
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

#[derive(Clone, Copy, PartialEq)]
enum SearchField {
    Provider,
    GeminiModel,
    MaxResults,
    MaxIterations,
}

/// Gemini models offered for cycling in the Search config UI.
/// Users can still hand-edit config.toml for any other model.
const GEMINI_MODEL_CHOICES: &[&str] = &[
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-2.5-flash",
];
/// Preset values users can cycle through for numeric search fields.
const MAX_RESULTS_CHOICES: &[usize] = &[3, 5, 10];
const MAX_ITERATIONS_CHOICES: &[usize] = &[3, 5, 10];

const HELP_TEXT: &str =
    " Tab: sections | ↑↓: navigate | ←→: multi select | Enter/Space: toggle | Ctrl+S: save | q: quit";

/// One row in the provider pickers (Providers tab, Single Provider, Multi
/// Providers). `name` is the canonical key from `[providers.<name>]`;
/// `label` is what the UI shows and what gets written into
/// `single_provider` / `multi_providers`: the first alias, or the canonical
/// name for a provider with no aliases (`Config::resolve_provider_alias`
/// accepts either).
#[derive(Clone, Debug, PartialEq)]
struct ProviderEntry {
    name: String,
    label: String,
}

/// Every configured provider in a stable order: sorted by canonical name.
/// `config.providers` is a HashMap, so iterating it directly reshuffled the
/// pickers on every launch. All three pickers share this one list, so a
/// cursor index always maps to a rendered row.
fn provider_entries(config: &Config) -> Vec<ProviderEntry> {
    let mut entries: Vec<ProviderEntry> = config
        .providers
        .iter()
        .map(|(name, p)| ProviderEntry {
            name: name.clone(),
            label: p.aliases.first().cloned().unwrap_or_else(|| name.clone()),
        })
        .collect();
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    entries
}

/// True if `alias` (any alias or the canonical name, any case) refers to the
/// provider whose canonical name is `name`.
fn refers_to(config: &Config, alias: &str, name: &str) -> bool {
    config.resolve_provider_alias(alias).as_deref() == Some(name)
}

/// Whether the provider `name` is in a `multi_providers` list under any of
/// its aliases. The list may hold "gpt" while the UI labels OpenAI as
/// "chatgpt", so compare canonical names rather than strings.
fn is_multi_selected(config: &Config, selected: &[String], name: &str) -> bool {
    selected.iter().any(|alias| refers_to(config, alias, name))
}

/// Return `selected` with `entry` toggled. Turning a provider off drops every
/// entry that resolves to it (e.g. both "gpt" and "chatgpt"); turning it on
/// appends its label. Entries that don't resolve to `entry` are kept as-is,
/// in their original order.
fn toggle_multi_provider(
    config: &Config,
    selected: &[String],
    entry: &ProviderEntry,
) -> Vec<String> {
    if is_multi_selected(config, selected, &entry.name) {
        selected
            .iter()
            .filter(|alias| !refers_to(config, alias, &entry.name))
            .cloned()
            .collect()
    } else {
        let mut updated = selected.to_vec();
        updated.push(entry.label.clone());
        updated
    }
}

/// The label that follows `current` when cycling `single_provider`. `current`
/// may be any alias of its provider. An unknown value restarts at the first
/// entry. Returns None only when there are no providers at all.
fn next_single_provider(
    config: &Config,
    entries: &[ProviderEntry],
    current: &str,
) -> Option<String> {
    let first = entries.first()?;
    let next = config
        .resolve_provider_alias(current)
        .and_then(|name| entries.iter().position(|e| e.name == name))
        .map_or(first, |pos| &entries[(pos + 1) % entries.len()]);
    Some(next.label.clone())
}

/// True if any provider's `enabled` flag differs between the two configs.
fn provider_enabled_changed(a: &Config, b: &Config) -> bool {
    a.providers.len() != b.providers.len()
        || a.providers
            .iter()
            .any(|(name, p)| b.providers.get(name).map(|q| q.enabled) != Some(p.enabled))
}

/// Move a cursor one step through `len` items, wrapping at both ends. An
/// out-of-range `index` is clamped first; an empty list always yields 0.
fn step_index(index: usize, len: usize, back: bool) -> usize {
    if len == 0 {
        return 0;
    }
    let index = index.min(len - 1);
    if back {
        (index + len - 1) % len
    } else {
        (index + 1) % len
    }
}

struct App {
    config: Config,
    original_config: Config,
    section: Section,
    general_field: GeneralField,
    mode_field: ModeField,
    display_field: DisplayField,
    search_field: SearchField,
    /// Cursor in the Providers tab; indexes `provider_entries`.
    provider_index: usize,
    /// Cursor in the Multi Providers checkboxes; indexes `provider_entries`.
    multi_provider_index: usize,
    /// Providers in display order, shared by every picker. The set of
    /// providers can't change while the editor is open, so this is built once.
    provider_entries: Vec<ProviderEntry>,
    /// Set by the first `q` with unsaved edits; a second consecutive `q`
    /// discards them. Any other key clears it.
    quit_pending: bool,
    message: Option<String>,
}

impl App {
    fn new(config: Config) -> Self {
        let original = config.clone();
        let provider_entries = provider_entries(&config);

        App {
            config,
            original_config: original,
            section: Section::General,
            general_field: GeneralField::DefaultInterface,
            mode_field: ModeField::DefaultMode,
            display_field: DisplayField::ShowModel,
            search_field: SearchField::Provider,
            provider_index: 0,
            multi_provider_index: 0,
            provider_entries,
            quit_pending: false,
            message: None,
        }
    }

    /// Whether anything the editor can change differs from what was loaded
    /// (or last saved). This must cover every field `toggle_or_cycle` edits:
    /// anything missing here is thrown away on `q` without the
    /// unsaved-changes prompt.
    fn has_changes(&self) -> bool {
        let (now, orig) = (&self.config, &self.original_config);
        now.cli != orig.cli
            || now.tui_mode != orig.tui_mode
            || now.default.tui != orig.default.tui
            || now.footer.show_model != orig.footer.show_model
            || now.footer.show_time != orig.footer.show_time
            || now.footer.show_tokens != orig.footer.show_tokens
            || now.footer.show_reasoning != orig.footer.show_reasoning
            || now.search.provider != orig.search.provider
            || now.search.gemini.model != orig.search.gemini.model
            || now.search.max_results != orig.search.max_results
            || now.search.max_iterations != orig.search.max_iterations
            || provider_enabled_changed(now, orig)
    }

    /// Apply one key press. Returns true when the editor should exit.
    /// Kept apart from the terminal loop so the quit and unsaved-changes
    /// flow can be tested without a TTY.
    fn handle_key(&mut self, key: KeyEvent) -> bool {
        self.message = None;

        // Raw mode delivers Ctrl+C as a key rather than SIGINT, so treat it
        // like `q` instead of silently ignoring it.
        let quit_key = matches!(key.code, KeyCode::Char('q') | KeyCode::Esc)
            || (key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL));
        if quit_key {
            return self.request_quit();
        }
        // Any other key cancels a pending "q again to discard".
        self.quit_pending = false;

        match (key.code, key.modifiers) {
            (KeyCode::Char('s'), KeyModifiers::CONTROL) => {
                if let Err(e) = self.save() {
                    // `{:#}` includes the underlying cause (e.g. permission
                    // denied), not just the top-level context.
                    self.message = Some(format!("Save failed: {:#}", e));
                }
            }
            (KeyCode::Tab, _) => self.next_section(),
            (KeyCode::BackTab, _) => self.prev_section(),
            (KeyCode::Down, _) | (KeyCode::Char('j'), _) => self.next_field(),
            (KeyCode::Up, _) | (KeyCode::Char('k'), _) => self.prev_field(),
            (KeyCode::Left, _) | (KeyCode::Char('h'), _) => self.handle_multi_provider_nav(true),
            (KeyCode::Right, _) | (KeyCode::Char('l'), _) => self.handle_multi_provider_nav(false),
            (KeyCode::Enter, _) | (KeyCode::Char(' '), _) => self.toggle_or_cycle(),
            _ => {}
        }
        false
    }

    /// Exit straight away when there's nothing to lose. With unsaved edits,
    /// the first request only warns and the second one discards.
    fn request_quit(&mut self) -> bool {
        if !self.has_changes() || self.quit_pending {
            return true;
        }
        self.message =
            Some("Unsaved changes! Press Ctrl+S to save, or q again to discard".to_string());
        self.quit_pending = true;
        false
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
            Section::Display => Section::Search,
            Section::Search => Section::Providers,
            Section::Providers => Section::General,
        };
        self.reset_fields();
    }

    fn prev_section(&mut self) {
        self.section = match self.section {
            Section::General => Section::Providers,
            Section::CliMode => Section::General,
            Section::TuiMode => Section::CliMode,
            Section::Display => Section::TuiMode,
            Section::Search => Section::Display,
            Section::Providers => Section::Search,
        };
        self.reset_fields();
    }

    fn reset_fields(&mut self) {
        self.general_field = GeneralField::DefaultInterface;
        self.mode_field = ModeField::DefaultMode;
        self.display_field = DisplayField::ShowModel;
        self.search_field = SearchField::Provider;
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
            Section::Search => {
                self.search_field = match self.search_field {
                    SearchField::Provider => SearchField::GeminiModel,
                    SearchField::GeminiModel => SearchField::MaxResults,
                    SearchField::MaxResults => SearchField::MaxIterations,
                    SearchField::MaxIterations => SearchField::Provider,
                };
            }
            Section::Providers => {
                self.provider_index =
                    step_index(self.provider_index, self.provider_entries.len(), false);
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
            Section::Search => {
                self.search_field = match self.search_field {
                    SearchField::Provider => SearchField::MaxIterations,
                    SearchField::GeminiModel => SearchField::Provider,
                    SearchField::MaxResults => SearchField::GeminiModel,
                    SearchField::MaxIterations => SearchField::MaxResults,
                };
            }
            Section::Providers => {
                self.provider_index =
                    step_index(self.provider_index, self.provider_entries.len(), true);
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
            Section::General => match self.general_field {
                GeneralField::DefaultInterface => {
                    self.config.default.tui = !self.config.default.tui;
                }
            },
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
                        // Cycle through providers in display order, starting
                        // from whichever provider the current alias names.
                        let current = &self.get_current_mode_config_ref().single_provider;
                        if let Some(next) =
                            next_single_provider(&self.config, &self.provider_entries, current)
                        {
                            self.get_current_mode_config().single_provider = next;
                        }
                    }
                    ModeField::MultiProviders => {
                        // Toggle the provider under the cursor
                        if let Some(entry) = self
                            .provider_entries
                            .get(self.multi_provider_index)
                            .cloned()
                        {
                            let updated = toggle_multi_provider(
                                &self.config,
                                &self.get_current_mode_config_ref().multi_providers,
                                &entry,
                            );
                            self.get_current_mode_config().multi_providers = updated;
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
            Section::Search => match self.search_field {
                SearchField::Provider => {
                    // Cycle between gemini and tavily
                    self.config.search.provider = if self.config.search.provider == "gemini" {
                        "tavily".to_string()
                    } else {
                        "gemini".to_string()
                    };
                }
                SearchField::GeminiModel => {
                    // Cycle through known Gemini models
                    let current = self.config.search.gemini.model.as_str();
                    let next_index = GEMINI_MODEL_CHOICES
                        .iter()
                        .position(|m| *m == current)
                        .map(|i| (i + 1) % GEMINI_MODEL_CHOICES.len())
                        .unwrap_or(0);
                    self.config.search.gemini.model = GEMINI_MODEL_CHOICES[next_index].to_string();
                }
                SearchField::MaxResults => {
                    let current = self.config.search.max_results;
                    let next = cycle_usize(MAX_RESULTS_CHOICES, current);
                    self.config.search.max_results = next;
                }
                SearchField::MaxIterations => {
                    let current = self.config.search.max_iterations;
                    let next = cycle_usize(MAX_ITERATIONS_CHOICES, current);
                    self.config.search.max_iterations = next;
                }
            },
            Section::Providers => {
                if let Some(entry) = self.provider_entries.get(self.provider_index) {
                    if let Some(p) = self.config.providers.get_mut(&entry.name) {
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
            // Same list the checkboxes are rendered from, so there are no
            // dead cursor positions.
            self.multi_provider_index =
                step_index(self.multi_provider_index, self.provider_entries.len(), left);
        }
    }
}

pub async fn run_config_ui() -> Result<()> {
    let config = Config::load()?;
    let mut app = App::new(config);

    // Restores the terminal on every exit path, including the `?`s below.
    let _guard = crate::term::TerminalGuard::enter(false)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    loop {
        terminal.draw(|frame| draw_ui(frame, &app))?;

        let event = event::read()?;
        let key = match event {
            Event::Key(key) if key.kind == KeyEventKind::Press => key,
            _ => continue,
        };

        if app.handle_key(key) {
            break;
        }
    }

    Ok(())
}

fn draw_ui(frame: &mut Frame, app: &App) {
    let area = frame.area();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            // Two inner lines: status on top, key help below it
            Constraint::Length(4),
        ])
        .split(area);

    // Header with section tabs
    let sections = Section::all();
    let titles: Vec<&str> = sections.iter().map(|s| s.title()).collect();
    let tabs = Tabs::new(titles)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" AI CLI Configuration "),
        )
        .select(app.section.index())
        .style(Style::default().fg(Color::White))
        .highlight_style(Style::default().fg(Color::Yellow).bold());
    frame.render_widget(tabs, chunks[0]);

    // Content area
    match app.section {
        Section::General => draw_general_section(frame, chunks[1], app),
        Section::CliMode => {
            draw_mode_section(frame, chunks[1], app, "CLI Mode (non-TUI)", &app.config.cli)
        }
        Section::TuiMode => {
            draw_mode_section(frame, chunks[1], app, "TUI Mode", &app.config.tui_mode)
        }
        Section::Display => draw_display_section(frame, chunks[1], app),
        Section::Search => draw_search_section(frame, chunks[1], app),
        Section::Providers => draw_providers_section(frame, chunks[1], app),
    }

    // Footer
    let footer = Paragraph::new(footer_lines(app))
        .wrap(Wrap { trim: false })
        .block(Block::default().borders(Borders::ALL));
    frame.render_widget(footer, chunks[2]);
}

/// Footer content: the status ([MODIFIED] plus any message) on its own line
/// ahead of the key help. It used to be appended after the ~95-column help
/// text in a non-wrapping paragraph, so on terminals narrower than ~110
/// columns "q again to discard", "Save failed" and "Config saved!" were
/// clipped off-screen. The paragraph wraps, so a long error pushes the help
/// down rather than getting cut off itself.
fn footer_lines(app: &App) -> Vec<Line<'static>> {
    let mut status: Vec<Span<'static>> = Vec::new();
    if app.has_changes() {
        status.push(Span::styled(
            "[MODIFIED]",
            Style::default().fg(Color::Yellow).bold(),
        ));
    }
    if let Some(msg) = &app.message {
        if !status.is_empty() {
            status.push(Span::raw(" "));
        }
        status.push(Span::styled(
            msg.clone(),
            Style::default().fg(Color::Yellow),
        ));
    }

    let help = Line::styled(HELP_TEXT, Style::default().fg(Color::DarkGray));
    if status.is_empty() {
        vec![help]
    } else {
        // Leading space lines the status up with the help text
        status.insert(0, Span::raw(" "));
        vec![Line::from(status), help]
    }
}

fn draw_general_section(frame: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" General Settings ");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let items = vec![format_mode_item(
        "Default Interface",
        if app.config.default.tui {
            "CLI / [TUI]"
        } else {
            "[CLI] / TUI"
        },
        app.general_field == GeneralField::DefaultInterface,
    )];

    let list = List::new(items);
    frame.render_widget(list, inner);
}

fn draw_mode_section(
    frame: &mut Frame,
    area: Rect,
    app: &App,
    title: &str,
    mode: &crate::config::ModeConfig,
) {
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
        format_multi_providers_item(app, mode, inner.width),
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

fn format_multi_providers_item(
    app: &App,
    mode: &crate::config::ModeConfig,
    width: u16,
) -> ListItem<'static> {
    let selected = app.mode_field == ModeField::MultiProviders;
    let style = if selected {
        Style::default().fg(Color::Yellow).bold()
    } else {
        Style::default()
    };
    let prefix = if selected { "▶ " } else { "  " };

    let checkboxes: Vec<String> = app
        .provider_entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let checked = if is_multi_selected(&app.config, &mode.multi_providers, &entry.name) {
                "x"
            } else {
                " "
            };
            if selected && i == app.multi_provider_index {
                format!(">[{}] {}<", checked, entry.label)
            } else {
                format!("[{}] {}", checked, entry.label)
            }
        })
        .collect();

    // With the six default providers this row is ~90 columns; wrap it so the
    // last checkboxes (and the cursor on them) stay visible at 80 columns.
    let head = format!("{}Multi Providers: ", prefix);
    let lines: Vec<Line<'static>> = wrap_tokens(&head, &checkboxes, usize::from(width))
        .into_iter()
        .map(Line::from)
        .collect();
    ListItem::new(lines).style(style)
}

/// Lay `tokens` out after `head`, space-separated, starting a new line
/// (indented to line up under the first token) whenever the next token would
/// overflow `width`. A token wider than a whole line is placed on its own.
fn wrap_tokens(head: &str, tokens: &[String], width: usize) -> Vec<String> {
    let text_width = |s: &str| Span::raw(s).width();
    let indent = " ".repeat(text_width(head));
    let mut lines = Vec::new();
    let mut line = head.to_string();
    let mut line_has_token = false;
    for token in tokens {
        if line_has_token && text_width(&line) + 1 + text_width(token) > width {
            lines.push(std::mem::replace(&mut line, indent.clone()));
            line_has_token = false;
        }
        if line_has_token {
            line.push(' ');
        }
        line.push_str(token);
        line_has_token = true;
    }
    lines.push(line);
    lines
}

fn draw_display_section(frame: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Display Settings ");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let items = vec![
        format_bool_item(
            "Show Model",
            app.config.footer.show_model,
            app.display_field == DisplayField::ShowModel,
        ),
        format_bool_item(
            "Show Time",
            app.config.footer.show_time,
            app.display_field == DisplayField::ShowTime,
        ),
        format_bool_item(
            "Show Tokens",
            app.config.footer.show_tokens,
            app.display_field == DisplayField::ShowTokens,
        ),
        format_bool_item(
            "Show Reasoning",
            app.config.footer.show_reasoning,
            app.display_field == DisplayField::ShowReasoning,
        ),
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

fn draw_search_section(frame: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Search Settings ");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let search = &app.config.search;

    let provider_display = if search.provider == "gemini" {
        "[gemini] / tavily"
    } else {
        "gemini / [tavily]"
    };

    let items = vec![
        format_mode_item(
            "Provider",
            provider_display,
            app.search_field == SearchField::Provider,
        ),
        format_mode_item(
            "Gemini Model",
            &search.gemini.model,
            app.search_field == SearchField::GeminiModel,
        ),
        format_mode_item(
            "Max Results",
            &search.max_results.to_string(),
            app.search_field == SearchField::MaxResults,
        ),
        format_mode_item(
            "Max Iterations",
            &search.max_iterations.to_string(),
            app.search_field == SearchField::MaxIterations,
        ),
    ];

    let list = List::new(items);
    frame.render_widget(list, inner);
}

/// Cycle through a slice of preset values, returning the next one. If `current`
/// isn't in the list, returns the first value.
fn cycle_usize(choices: &[usize], current: usize) -> usize {
    let idx = choices
        .iter()
        .position(|&v| v == current)
        .map(|i| (i + 1) % choices.len())
        .unwrap_or(0);
    choices[idx]
}

fn draw_providers_section(frame: &mut Frame, area: Rect, app: &App) {
    let block = Block::default().borders(Borders::ALL).title(" Providers ");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(30), Constraint::Percentage(70)])
        .split(inner);

    let provider_items: Vec<ListItem> = app
        .provider_entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let enabled = app
                .config
                .providers
                .get(&entry.name)
                .map(|p| p.enabled)
                .unwrap_or(false);
            let checkbox = if enabled { "[x]" } else { "[ ]" };
            let style = if i == app.provider_index {
                Style::default().fg(Color::Yellow).bold()
            } else {
                Style::default()
            };
            ListItem::new(format!("{} {}", checkbox, entry.label)).style(style)
        })
        .collect();

    let mut list_state = ListState::default();
    list_state.select(Some(app.provider_index));

    let list = List::new(provider_items)
        .block(Block::default().borders(Borders::RIGHT))
        .highlight_style(Style::default().bg(Color::DarkGray));
    frame.render_stateful_widget(list, chunks[0], &mut list_state);

    if let Some(entry) = app.provider_entries.get(app.provider_index) {
        let provider_name = &entry.name;
        if let Some(provider) = app.config.providers.get(provider_name) {
            let fast = provider
                .fast_model
                .as_deref()
                .unwrap_or(&provider.default_model);
            let smart = provider
                .smart_model
                .as_deref()
                .unwrap_or(&provider.default_model);
            let details_text = [
                format!("Provider: {}", provider_name),
                format!("Enabled: {}", if provider.enabled { "Yes" } else { "No" }),
                format!("Fast Model: {}", fast),
                format!("Smart Model: {}", smart),
                format!("Color: {}", provider.color),
            ]
            .join("\n");
            let details_para = Paragraph::new(details_text)
                .style(Style::default().fg(Color::White))
                .block(Block::default().borders(Borders::NONE));
            frame.render_widget(details_para, chunks[1]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProviderConfig;
    use ratatui::backend::TestBackend;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    /// A provider config with the given aliases, cloned from a default so the
    /// tests don't depend on every ProviderConfig field.
    fn provider_with_aliases(aliases: &[&str]) -> ProviderConfig {
        let mut p = Config::default().providers["openai"].clone();
        p.aliases = aliases.iter().map(|a| a.to_string()).collect();
        p
    }

    fn index_of(app: &App, name: &str) -> usize {
        app.provider_entries
            .iter()
            .position(|e| e.name == name)
            .unwrap_or_else(|| panic!("provider {name} not listed"))
    }

    fn app_on_multi_providers() -> App {
        let mut app = App::new(Config::default());
        app.section = Section::CliMode;
        app.mode_field = ModeField::MultiProviders;
        app
    }

    fn render(app: &App, width: u16, height: u16) -> Vec<String> {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| draw_ui(frame, app)).unwrap();
        let buffer = terminal.backend().buffer();
        buffer
            .content()
            .chunks(usize::from(width))
            .map(|row| row.iter().map(|cell| cell.symbol()).collect())
            .collect()
    }

    #[test]
    fn every_editable_field_is_detected_as_a_change() {
        // Drive each field through the UI itself, so a field added to
        // toggle_or_cycle without a matching has_changes check fails here.
        let providers = Config::default().providers.len();
        let fields = [
            (Section::General, 1),
            (Section::CliMode, 4),
            (Section::TuiMode, 4),
            (Section::Display, 4),
            (Section::Search, 4),
            (Section::Providers, providers),
        ];
        for (section, count) in fields {
            for field in 0..count {
                let mut app = App::new(Config::default());
                app.section = section;
                for _ in 0..field {
                    app.next_field();
                }
                assert!(!app.has_changes());
                app.toggle_or_cycle();
                assert!(
                    app.has_changes(),
                    "{} field #{field} changed without being detected",
                    section.title()
                );
            }
        }
    }

    #[test]
    fn provider_enable_toggle_is_detected_and_guards_quit() {
        let mut app = App::new(Config::default());
        app.section = Section::Providers;
        app.provider_index = index_of(&app, "ollama");
        app.toggle_or_cycle();
        assert!(app.config.providers["ollama"].enabled);
        assert!(app.has_changes());

        // First q only warns; the second discards.
        assert!(!app.handle_key(key(KeyCode::Char('q'))));
        assert!(app.message.as_deref().unwrap().contains("Unsaved changes"));
        assert!(app.handle_key(key(KeyCode::Char('q'))));

        // Toggling back restores a clean state.
        let mut app = App::new(Config::default());
        app.section = Section::Providers;
        app.toggle_or_cycle();
        app.toggle_or_cycle();
        assert!(!app.has_changes());
    }

    #[test]
    fn quit_flow() {
        // Nothing to lose: q exits straight away.
        let mut app = App::new(Config::default());
        assert!(app.handle_key(key(KeyCode::Char('q'))));

        // Ctrl+C is guarded the same way as q.
        let mut app = App::new(Config::default());
        app.toggle_or_cycle(); // General > Default Interface
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(!app.handle_key(ctrl_c));
        // Any other key cancels the pending discard.
        assert!(!app.handle_key(key(KeyCode::Down)));
        assert!(!app.handle_key(key(KeyCode::Esc)));
        assert!(app.handle_key(key(KeyCode::Esc)));
    }

    #[test]
    fn provider_entries_have_a_stable_order() {
        let entries = provider_entries(&Config::default());
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "anthropic",
                "google",
                "ollama",
                "openai",
                "openrouter",
                "xai"
            ]
        );
        let labels: Vec<&str> = entries.iter().map(|e| e.label.as_str()).collect();
        assert_eq!(
            labels,
            [
                "claude",
                "gemini",
                "ollama",
                "chatgpt",
                "openrouter",
                "grok"
            ]
        );

        // Insertion order (and so HashMap layout) doesn't matter.
        let mut forward = Config::default();
        let mut backward = Config::default();
        forward.providers.clear();
        backward.providers.clear();
        let names = ["zeta", "alpha", "mid", "beta"];
        for name in names {
            forward
                .providers
                .insert(name.to_string(), provider_with_aliases(&[name]));
        }
        for &name in names.iter().rev() {
            backward
                .providers
                .insert(name.to_string(), provider_with_aliases(&[name]));
        }
        assert_eq!(provider_entries(&forward), provider_entries(&backward));
        let sorted: Vec<String> = provider_entries(&forward)
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert_eq!(sorted, ["alpha", "beta", "mid", "zeta"]);
    }

    #[test]
    fn multi_selection_matches_any_alias() {
        // The default config stores "gpt", but OpenAI's first alias (its
        // label) is "chatgpt". It must still show as selected.
        let config = Config::default();
        assert!(config.cli.multi_providers.contains(&"gpt".to_string()));
        assert!(is_multi_selected(
            &config,
            &config.cli.multi_providers,
            "openai"
        ));
        assert!(!is_multi_selected(
            &config,
            &config.cli.multi_providers,
            "ollama"
        ));

        let app = app_on_multi_providers();
        let rows = render(&app, 120, 30);
        assert!(
            rows.iter().any(|r| r.contains("[x] chatgpt")),
            "chatgpt should render as checked: {rows:#?}"
        );
    }

    #[test]
    fn toggling_via_non_first_alias_removes_every_alias() {
        let mut app = app_on_multi_providers();
        app.config.cli.multi_providers = ["claude", "gpt", "grok", "CHATGPT"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        app.multi_provider_index = index_of(&app, "openai");

        // Off: both "gpt" and "CHATGPT" go, the rest keep their order.
        app.toggle_or_cycle();
        assert_eq!(app.config.cli.multi_providers, ["claude", "grok"]);

        // On again: added once, under its label.
        app.toggle_or_cycle();
        assert_eq!(
            app.config.cli.multi_providers,
            ["claude", "grok", "chatgpt"]
        );

        // Only the section being edited is touched.
        assert_eq!(
            app.config.tui_mode.multi_providers,
            Config::default().tui_mode.multi_providers
        );
    }

    #[test]
    fn toggling_leaves_unknown_entries_alone() {
        let config = Config::default();
        let entries = provider_entries(&config);
        let claude = entries.iter().find(|e| e.name == "anthropic").unwrap();
        let selected = vec!["typo".to_string(), "claude".to_string()];
        assert_eq!(toggle_multi_provider(&config, &selected, claude), ["typo"]);
    }

    #[test]
    fn single_provider_cycles_from_any_alias() {
        let config = Config::default();
        let entries = provider_entries(&config);
        let next = |current: &str| next_single_provider(&config, &entries, current).unwrap();

        // "gpt" is OpenAI's second alias; cycling must continue from OpenAI.
        assert_eq!(next("gpt"), "openrouter");
        assert_eq!(next("chatgpt"), "openrouter");
        assert_eq!(next("claude"), "gemini");
        // Wraps from the last provider back to the first.
        assert_eq!(next("grok"), "claude");
        // Unknown values restart at the first provider.
        assert_eq!(next("nope"), "claude");

        // A full cycle visits every provider once, in display order.
        let mut current = "claude".to_string();
        let mut seen = Vec::new();
        for _ in 0..entries.len() {
            current = next(current.as_str());
            seen.push(current.clone());
        }
        assert_eq!(
            seen,
            [
                "gemini",
                "ollama",
                "chatgpt",
                "openrouter",
                "grok",
                "claude"
            ]
        );

        assert_eq!(next_single_provider(&config, &[], "claude"), None);
    }

    #[test]
    fn navigation_stays_on_rendered_rows() {
        // A provider without aliases used to be counted by the cursor but
        // left out of the checkbox row, leaving a dead position.
        let mut config = Config::default();
        config
            .providers
            .insert("custom".to_string(), provider_with_aliases(&[]));
        let mut app = App::new(config);
        app.section = Section::CliMode;
        app.mode_field = ModeField::MultiProviders;
        let len = app.provider_entries.len();
        assert_eq!(len, app.config.providers.len());
        let custom = index_of(&app, "custom");
        assert_eq!(app.provider_entries[custom].label, "custom");

        for step in 1..=len {
            app.handle_multi_provider_nav(false);
            assert_eq!(app.multi_provider_index, step % len);
        }
        app.handle_multi_provider_nav(true);
        assert_eq!(app.multi_provider_index, len - 1);

        // The alias-less provider can be toggled via its canonical name.
        app.multi_provider_index = custom;
        app.toggle_or_cycle();
        assert!(app
            .config
            .cli
            .multi_providers
            .contains(&"custom".to_string()));
        assert!(is_multi_selected(
            &app.config,
            &app.config.cli.multi_providers,
            "custom"
        ));

        // The Providers tab uses the same list and bounds.
        app.section = Section::Providers;
        app.provider_index = 0;
        app.prev_field();
        assert_eq!(app.provider_index, len - 1);
        app.next_field();
        assert_eq!(app.provider_index, 0);
    }

    #[test]
    fn step_index_bounds() {
        assert_eq!(step_index(0, 0, false), 0);
        assert_eq!(step_index(0, 0, true), 0);
        assert_eq!(step_index(0, 3, true), 2);
        assert_eq!(step_index(2, 3, false), 0);
        // Out-of-range input is clamped before stepping.
        assert_eq!(step_index(10, 3, false), 0);
        assert_eq!(step_index(10, 3, true), 1);
    }

    #[test]
    fn status_stays_visible_on_80_column_terminal() {
        let mut app = App::new(Config::default());
        app.section = Section::Providers;
        app.toggle_or_cycle();
        assert!(!app.handle_key(key(KeyCode::Char('q'))));

        let rows = render(&app, 80, 24);
        let footer = rows[rows.len() - 3..].join("\n");
        assert!(footer.contains("[MODIFIED]"), "{footer}");
        assert!(footer.contains("q again to discard"), "{footer}");

        // Without a status the help text gets the whole footer.
        let rows = render(&App::new(Config::default()), 80, 24);
        let footer = rows[rows.len() - 3..].join("\n");
        assert!(footer.contains("Tab: sections"), "{footer}");
    }

    #[test]
    fn multi_provider_row_wraps_instead_of_clipping() {
        let mut app = app_on_multi_providers();
        app.multi_provider_index = app.provider_entries.len() - 1;
        let rows = render(&app, 80, 24);
        let screen = rows.join("\n");
        for entry in &app.provider_entries {
            assert!(
                screen.contains(&format!("] {}", entry.label)),
                "{} missing from:\n{screen}",
                entry.label
            );
        }
        // The cursor on the last provider is visible too.
        assert!(screen.contains(">[x] grok<"), "{screen}");
    }

    #[test]
    fn wrap_tokens_respects_width() {
        let tokens: Vec<String> = ["aaaa", "bbbb", "cccc", "dddd"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let lines = wrap_tokens("head: ", &tokens, 16);
        assert_eq!(lines, ["head: aaaa bbbb", "      cccc dddd"]);
        assert!(lines.iter().all(|l| l.chars().count() <= 16));

        // Everything on one line when it fits.
        assert_eq!(wrap_tokens("h: ", &tokens, 80), ["h: aaaa bbbb cccc dddd"]);
        // A token wider than the line still gets placed.
        assert_eq!(wrap_tokens("h: ", &tokens[..1], 2), ["h: aaaa"]);
    }
}
