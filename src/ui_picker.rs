//! The TUI's model picker: a popup over the panels that lists every chat
//! model the Bedrock endpoint serves, narrows the list as you type, and
//! switches the selected panel to the model you pick. A typed ID that isn't
//! in the list (a model the catalog hides, a non-Bedrock provider's model)
//! is offered as the last row, so any ID can be used.
//!
//! Plain state plus a key handler and a draw function; `ui.rs` owns the
//! `App` wiring (where the catalog comes from, what picking does).
use crate::bedrock_catalog::BedrockModel;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, Paragraph},
    Frame,
};

/// Loading state of one Bedrock provider's model list.
#[derive(Debug, Clone)]
pub enum CatalogState {
    Loading,
    Ready(Vec<BedrockModel>),
    Failed(String),
}

/// One row of the picker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PickerItem {
    /// A model from the Bedrock catalog.
    Model(BedrockModel),
    /// The query itself, used verbatim as a model ID.
    Typed(String),
}

impl PickerItem {
    pub fn model_id(&self) -> &str {
        match self {
            PickerItem::Model(m) => &m.id,
            PickerItem::Typed(id) => id,
        }
    }
}

/// What a key press in the picker asks the app to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PickerAction {
    None,
    Close,
    Pick(PickerItem),
}

pub struct ModelPicker {
    /// Index of the panel whose model is being changed.
    pub panel: usize,
    /// That panel's display name, for the title.
    pub panel_label: String,
    /// The model the panel uses now, marked in the list.
    pub current_model: String,
    /// The Bedrock provider whose catalog is listed (and which serves a
    /// picked catalog model). `None` when no provider reaches Bedrock: the
    /// picker then only accepts a typed ID.
    pub catalog_provider: Option<String>,
    pub query: String,
    pub selected: usize,
    /// First visible row; kept so the selection stays on screen.
    pub scroll: usize,
}

impl ModelPicker {
    pub fn new(
        panel: usize,
        panel_label: String,
        current_model: String,
        catalog_provider: Option<String>,
    ) -> Self {
        Self {
            panel,
            panel_label,
            current_model,
            catalog_provider,
            query: String::new(),
            selected: 0,
            scroll: 0,
        }
    }

    /// Rows for the current query: matching catalog models, then the query
    /// as a literal ID unless it already names a listed model exactly.
    pub fn items(&self, catalog: Option<&CatalogState>) -> Vec<PickerItem> {
        let query = self.query.trim();
        let mut items: Vec<PickerItem> = match catalog {
            Some(CatalogState::Ready(models)) => models
                .iter()
                .filter(|m| m.matches(query))
                .cloned()
                .map(PickerItem::Model)
                .collect(),
            _ => Vec::new(),
        };
        let exact = items.iter().any(|item| item.model_id() == query);
        if !query.is_empty() && !query.contains(char::is_whitespace) && !exact {
            items.push(PickerItem::Typed(query.to_string()));
        }
        items
    }

    /// Typed or pasted text goes into the query (as one line).
    pub fn insert_text(&mut self, text: &str) {
        self.query.extend(text.chars().filter(|c| !c.is_control()));
        self.selected = 0;
        self.scroll = 0;
    }

    pub fn handle_key(&mut self, key: KeyEvent, items: &[PickerItem]) -> PickerAction {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        let last = items.len().saturating_sub(1);
        match key.code {
            KeyCode::Esc => return PickerAction::Close,
            KeyCode::Char('c') if ctrl => return PickerAction::Close,
            KeyCode::Enter => {
                return match items.get(self.selected) {
                    Some(item) => PickerAction::Pick(item.clone()),
                    None => PickerAction::None,
                };
            }
            KeyCode::Up => self.selected = self.selected.saturating_sub(1),
            KeyCode::Char('p') if ctrl => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down => self.selected = (self.selected + 1).min(last),
            KeyCode::Char('n') if ctrl => self.selected = (self.selected + 1).min(last),
            KeyCode::PageUp => self.selected = self.selected.saturating_sub(10),
            KeyCode::PageDown => self.selected = (self.selected + 10).min(last),
            KeyCode::Home => self.selected = 0,
            KeyCode::End => self.selected = last,
            KeyCode::Backspace => {
                self.query.pop();
                self.selected = 0;
                self.scroll = 0;
            }
            KeyCode::Char('u') if ctrl => {
                self.query.clear();
                self.selected = 0;
                self.scroll = 0;
            }
            // Same rule as the input line: Ctrl/Alt chords aren't text, but
            // Ctrl+Alt (AltGr on Windows) produces real characters.
            KeyCode::Char(_) if ctrl != alt => {}
            KeyCode::Char(c) => {
                let mut buf = [0u8; 4];
                self.insert_text(c.encode_utf8(&mut buf));
            }
            _ => {}
        }
        PickerAction::None
    }
}

/// A centered rectangle `pct_w`% of `area` wide and `rows` tall, but no
/// taller than `max_pct_h`% of it (and at least 8 rows, screen permitting).
fn centered(area: Rect, pct_w: u16, rows: u16, max_pct_h: u16) -> Rect {
    let w = (area.width as u32 * pct_w as u32 / 100) as u16;
    let max_h = (area.height as u32 * max_pct_h as u32 / 100) as u16;
    let w = w.max(40).min(area.width);
    let h = rows.min(max_h).max(8).min(area.height);
    Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    }
}

pub fn draw(frame: &mut Frame, picker: &mut ModelPicker, catalog: Option<&CatalogState>) {
    let items = picker.items(catalog);
    // Borders + query + status/footer + rows: shrink to fit a short list.
    let wanted = u16::try_from(items.len())
        .unwrap_or(u16::MAX)
        .saturating_add(5);
    let area = centered(frame.area(), 80, wanted, 75);
    picker.selected = picker.selected.min(items.len().saturating_sub(1));

    let source = match &picker.catalog_provider {
        Some(name) => format!("Bedrock via `{}`", name),
        None => "no Bedrock provider: type a model ID".to_string(),
    };
    let title = format!(" Model for {} · {} ", picker.panel_label, source);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .title(Line::styled(
            title,
            Style::default().add_modifier(Modifier::BOLD),
        ))
        .border_style(Style::default().fg(Color::Yellow));
    let inner = block.inner(area);
    frame.render_widget(Clear, area);
    frame.render_widget(block, area);
    if inner.height < 3 || inner.width < 10 {
        return;
    }

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(vec![
        Span::styled("> ", Style::default().fg(Color::Yellow)),
        Span::raw(picker.query.clone()),
        Span::styled("▏", Style::default().fg(Color::Yellow)),
    ]));

    let status: Option<Line> = match catalog {
        Some(CatalogState::Loading) => Some(Line::styled(
            "  loading the Bedrock model list…",
            Style::default().fg(Color::DarkGray),
        )),
        Some(CatalogState::Failed(err)) => Some(Line::styled(
            format!("  couldn't load the Bedrock model list: {}", err),
            Style::default().fg(Color::Red),
        )),
        _ if items.is_empty() && picker.query.trim().is_empty() => Some(Line::styled(
            "  type a model ID",
            Style::default().fg(Color::DarkGray),
        )),
        _ if items.is_empty() => Some(Line::styled(
            "  no matching models",
            Style::default().fg(Color::DarkGray),
        )),
        _ => None,
    };
    // Query line + optional status line + footer line.
    let reserved = 2 + usize::from(status.is_some());
    let rows = (inner.height as usize).saturating_sub(reserved).max(1);
    if let Some(line) = status {
        lines.push(line);
    }

    // Keep the selection inside the visible window.
    if picker.selected < picker.scroll {
        picker.scroll = picker.selected;
    } else if picker.selected >= picker.scroll + rows {
        picker.scroll = picker.selected + 1 - rows;
    }

    let id_width = items
        .iter()
        .map(|i| i.model_id().chars().count())
        .max()
        .unwrap_or(0)
        .min(inner.width as usize / 2);
    for (i, item) in items.iter().enumerate().skip(picker.scroll).take(rows) {
        let selected = i == picker.selected;
        let marker = if item.model_id() == picker.current_model {
            "● "
        } else {
            "  "
        };
        let text = match item {
            PickerItem::Model(m) => format!(
                "{}{:<w$}  {} · {}",
                marker,
                m.id,
                m.name,
                m.vendor,
                w = id_width
            ),
            PickerItem::Typed(id) => format!("{}use \"{}\" as the model ID", marker, id),
        };
        let style = match (selected, item) {
            (true, _) => Style::default()
                .fg(Color::Black)
                .bg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
            (false, PickerItem::Typed(_)) => Style::default().fg(Color::Cyan),
            (false, _) => Style::default(),
        };
        lines.push(Line::styled(text, style));
    }

    let body = Rect {
        height: inner.height - 1,
        ..inner
    };
    frame.render_widget(Paragraph::new(lines), body);

    let count = items
        .iter()
        .filter(|i| matches!(i, PickerItem::Model(_)))
        .count();
    let footer = format!(
        " ↑/↓ select · Enter use · Esc cancel · {} model{}",
        count,
        if count == 1 { "" } else { "s" }
    );
    frame.render_widget(
        Paragraph::new(footer).style(Style::default().fg(Color::DarkGray)),
        Rect {
            y: inner.y + inner.height - 1,
            height: 1,
            ..inner
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyEventKind;

    fn model(id: &str, name: &str) -> BedrockModel {
        BedrockModel {
            id: id.into(),
            name: name.into(),
            vendor: "Vendor".into(),
        }
    }

    fn catalog() -> CatalogState {
        CatalogState::Ready(vec![
            model("us.anthropic.claude-opus-5-5", "Claude Opus 5.5"),
            model("us.xai.grok-4.7", "Grok 4.7"),
            model("global.xai.grok-4.7", "Grok 4.7"),
        ])
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent {
            code,
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: crossterm::event::KeyEventState::NONE,
        }
    }

    fn picker() -> ModelPicker {
        ModelPicker::new(
            0,
            "CLAUDE".into(),
            "us.xai.grok-4.7".into(),
            Some("anthropic".into()),
        )
    }

    #[test]
    fn empty_query_lists_everything_without_a_typed_row() {
        let p = picker();
        assert_eq!(p.items(Some(&catalog())).len(), 3);
    }

    #[test]
    fn typing_narrows_and_offers_the_literal_id_last() {
        let mut p = picker();
        let cat = catalog();
        for c in "grok".chars() {
            let items = p.items(Some(&cat));
            p.handle_key(key(KeyCode::Char(c)), &items);
        }
        let items = p.items(Some(&cat));
        let ids: Vec<&str> = items.iter().map(PickerItem::model_id).collect();
        assert_eq!(ids, ["us.xai.grok-4.7", "global.xai.grok-4.7", "grok"]);
        assert_eq!(items[2], PickerItem::Typed("grok".into()));
    }

    #[test]
    fn an_exact_listed_id_isnt_offered_twice() {
        let mut p = picker();
        p.insert_text("us.xai.grok-4.7");
        let items = p.items(Some(&catalog()));
        assert_eq!(
            items,
            [PickerItem::Model(model("us.xai.grok-4.7", "Grok 4.7"))]
        );
    }

    #[test]
    fn unlisted_ids_are_accepted_even_without_a_catalog() {
        let mut p = picker();
        p.insert_text("us.meta.llama3-1-8b-instruct-v1:0");
        for catalog in [
            None,
            Some(&CatalogState::Loading),
            Some(&CatalogState::Failed("x".into())),
        ] {
            let items = p.items(catalog);
            assert_eq!(
                p.handle_key(key(KeyCode::Enter), &items),
                PickerAction::Pick(PickerItem::Typed(
                    "us.meta.llama3-1-8b-instruct-v1:0".into()
                ))
            );
        }
    }

    #[test]
    fn a_multi_word_query_is_a_search_not_an_id() {
        let mut p = picker();
        p.insert_text("opus 5.5");
        let items = p.items(Some(&catalog()));
        assert_eq!(items.len(), 1);
        assert!(matches!(items[0], PickerItem::Model(_)));
    }

    #[test]
    fn arrows_move_within_bounds_and_enter_picks_the_selection() {
        let mut p = picker();
        let cat = catalog();
        let items = p.items(Some(&cat));
        p.handle_key(key(KeyCode::Up), &items);
        assert_eq!(p.selected, 0);
        for _ in 0..10 {
            p.handle_key(key(KeyCode::Down), &items);
        }
        assert_eq!(p.selected, 2);
        assert_eq!(
            p.handle_key(key(KeyCode::Enter), &items),
            PickerAction::Pick(items[2].clone())
        );
        assert_eq!(p.handle_key(key(KeyCode::Esc), &items), PickerAction::Close);
    }

    #[test]
    fn enter_on_an_empty_list_does_nothing() {
        let mut p = picker();
        assert_eq!(p.handle_key(key(KeyCode::Enter), &[]), PickerAction::None);
    }

    #[test]
    fn pasted_newlines_dont_end_up_in_the_query() {
        let mut p = picker();
        p.insert_text("us.xai.grok-4.7\r\n");
        assert_eq!(p.query, "us.xai.grok-4.7");
    }
}
