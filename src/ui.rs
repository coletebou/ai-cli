use crate::bedrock_catalog::BedrockModel;
use crate::composer::{self, Layout as ComposerLayout};
use crate::providers::{Provider, ProviderManager};
use crate::query::{stream_with_messages, QueryContext, StreamEvent};
use crate::selection::{self, Selection, Target};
use crate::sessions::{SessionStore, Tracker, TUI_AUTOSAVE_NAME};
use crate::ui_picker::{CatalogState, ModelPicker, PickerAction, PickerItem};
use anyhow::Result;
use crossterm::event::{
    self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEventKind,
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
use std::collections::HashMap;
use std::io::{self, Stdout};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use unicode_segmentation::UnicodeSegmentation;

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

/// Rows the panels keep however tall the composer grows.
const PANEL_MIN_HEIGHT: usize = 5;

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
    /// Use a different model (served by `provider_name`'s config entry)
    /// from the next turn on. The conversation so far is kept.
    SetModel {
        provider_name: String,
        model: String,
    },
}

#[path = "ui_sessions.rs"]
mod ui_sessions;
use ui_sessions::History;
pub use ui_sessions::Resume;

/// A finished catalog fetch: (Bedrock provider name, models or error text).
type CatalogResult = (String, Result<Vec<BedrockModel>, String>);

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
    /// The conversation, shared with this panel's worker (see `ui_sessions`).
    history: History,
    /// Questions sent to the worker; with the history's `committed`, tells
    /// whether every finished turn has been written back yet.
    asked: u64,
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
            history: ui_sessions::new_history(),
            asked: 0,
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
    /// First composer row shown once the text outgrows the box.
    input_scroll: usize,
    /// Keep the cursor's row in view. Cleared while the user scrolls or
    /// drag-selects the composer by mouse, set again by any key.
    input_follow_cursor: bool,
    /// Composer text width from the last draw, for Up/Down between rows.
    input_width: usize,
    /// Column Up/Down aim for, kept across a run of vertical moves so a short
    /// row in between doesn't drag the cursor left for good.
    input_goal_col: Option<usize>,
    /// Composer text area from the last draw, for mouse hit-testing.
    input_inner: Option<Rect>,
    /// The mouse selection being dragged or last copied.
    selection: Option<Selection>,
    /// Where copied text goes. Tests swap in a recorder so they never touch
    /// the real clipboard.
    copy_to_clipboard: fn(&str),
    should_quit: bool,
    /// Ephemeral one-line hint (e.g., "busy", "cleared") shown in the footer
    /// until the next key event or turn completion.
    status_message: Option<String>,
    /// Cancellation token for the currently in-flight turn. Cloned into every
    /// panel's Ask command so a single Ctrl+C cancels them all.
    current_turn_cancel: Option<CancellationToken>,
    /// The model picker, while it's open.
    picker: Option<ModelPicker>,
    /// Bedrock model lists, by the provider name they were fetched through.
    catalogs: HashMap<String, CatalogState>,
    /// Providers that route to Bedrock, in display order (empty in tests
    /// unless set).
    bedrock_providers: Vec<String>,
    /// Catalogs to fetch; the main loop (which holds the ProviderManager)
    /// drains this and reports back through `catalog_loaded`.
    catalog_requests: Vec<String>,
    /// The saved session these panels belong to, for `/save`, `/load` and
    /// the exit autosave.
    tracker: Tracker,
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
            input_scroll: 0,
            input_follow_cursor: true,
            input_width: 0,
            input_goal_col: None,
            input_inner: None,
            selection: None,
            copy_to_clipboard: crate::clipboard::copy,
            should_quit: false,
            status_message: None,
            current_turn_cancel: None,
            picker: None,
            catalogs: HashMap::new(),
            bedrock_providers: Vec::new(),
            catalog_requests: Vec::new(),
            // Tests never touch the real sessions directory.
            tracker: Tracker::new(if cfg!(test) {
                None
            } else {
                SessionStore::default_location()
            })
            .with_autosave_name(TUI_AUTOSAVE_NAME),
        }
    }

    // ── Model picker ──────────────────────────────────────────────────────

    /// The Bedrock provider whose catalog panel `idx` picks from: its own
    /// provider when that routes to Bedrock, else the first one that does.
    fn catalog_provider_for(&self, idx: usize) -> Option<String> {
        let own = &self.panels.get(idx)?.provider.name;
        if self.bedrock_providers.contains(own) {
            return Some(own.clone());
        }
        self.bedrock_providers.first().cloned()
    }

    /// Queue a catalog fetch unless one is loaded or already loading. A
    /// failed fetch is retried the next time the picker opens.
    fn request_catalog(&mut self, provider: &str) {
        if matches!(
            self.catalogs.get(provider),
            Some(CatalogState::Loading | CatalogState::Ready(_))
        ) {
            return;
        }
        self.catalogs
            .insert(provider.to_string(), CatalogState::Loading);
        self.catalog_requests.push(provider.to_string());
    }

    fn catalog_loaded(&mut self, provider: String, result: Result<Vec<BedrockModel>, String>) {
        let state = match result {
            Ok(models) => CatalogState::Ready(models),
            Err(err) => CatalogState::Failed(err),
        };
        self.catalogs.insert(provider, state);
    }

    fn open_model_picker(&mut self) {
        let idx = self.selected_panel;
        let Some(panel) = self.panels.get(idx) else {
            return;
        };
        if panel.is_busy() {
            self.status_message = Some(format!(
                "{} is answering; change its model when it's done",
                panel.provider.display_name
            ));
            return;
        }
        let catalog_provider = self.catalog_provider_for(idx);
        self.picker = Some(ModelPicker::new(
            idx,
            panel.provider.display_name.clone(),
            panel.model.clone(),
            catalog_provider.clone(),
        ));
        if let Some(name) = catalog_provider {
            self.request_catalog(&name);
        }
    }

    fn picker_items(&self) -> Vec<PickerItem> {
        let Some(picker) = &self.picker else {
            return Vec::new();
        };
        let catalog = picker
            .catalog_provider
            .as_ref()
            .and_then(|name| self.catalogs.get(name));
        picker.items(catalog)
    }

    fn handle_picker_key(&mut self, key: KeyEvent) {
        let items = self.picker_items();
        let Some(picker) = self.picker.as_mut() else {
            return;
        };
        match picker.handle_key(key, &items) {
            PickerAction::None => {}
            PickerAction::Close => self.picker = None,
            PickerAction::Pick(item) => {
                let picker = self.picker.take().expect("picker is open");
                let provider = match &item {
                    // Catalog models are served by the provider the catalog
                    // came from.
                    PickerItem::Model(_) => picker
                        .catalog_provider
                        .unwrap_or_else(|| self.panels[picker.panel].provider.name.clone()),
                    PickerItem::Typed(id) => self.route_for_typed(picker.panel, id),
                };
                self.set_panel_model(picker.panel, provider, item.model_id().to_string());
            }
        }
    }

    /// Which provider serves a model ID typed by hand on panel `idx`: the
    /// panel's own provider, unless that doesn't reach Bedrock and the ID is
    /// a model some Bedrock catalog lists (so `/model us.xai.grok-4.7` on a
    /// Gemini panel goes to Bedrock, while `/model gemini-3.1-pro-preview`
    /// stays with Google).
    fn route_for_typed(&self, idx: usize, model: &str) -> String {
        let own = self.panels[idx].provider.name.clone();
        if self.bedrock_providers.contains(&own) {
            return own;
        }
        self.bedrock_providers
            .iter()
            .find(|name| match self.catalogs.get(*name) {
                Some(CatalogState::Ready(models)) => models.iter().any(|m| m.id == model),
                _ => false,
            })
            .cloned()
            .unwrap_or(own)
    }

    /// Switch panel `idx` to `model`, served by `provider_name`, from its
    /// next turn on. The panel keeps its conversation.
    fn set_panel_model(&mut self, idx: usize, provider_name: String, model: String) {
        let Some(panel) = self.panels.get_mut(idx) else {
            return;
        };
        if panel.is_busy() {
            self.status_message = Some(format!(
                "{} is answering; change its model when it's done",
                panel.provider.display_name
            ));
            return;
        }
        let Some(tx) = self.senders.get(idx) else {
            return;
        };
        if tx
            .try_send(PanelCommand::SetModel {
                provider_name: provider_name.clone(),
                model: model.clone(),
            })
            .is_err()
        {
            self.status_message = Some(format!(
                "couldn't switch {}: its worker stopped",
                panel.provider.display_name
            ));
            return;
        }
        panel.provider.name = provider_name;
        panel.provider.model = model.clone();
        panel.model = model.clone();
        self.status_message = Some(format!(
            "{} now uses {}",
            panel.provider.display_name, model
        ));
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
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map_or(0, |d| d.as_secs());
                        self.tracker.turn_completed(now);
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
                panel.asked += 1;
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
        // A fresh conversation: saved as a new session, not over this one.
        self.tracker.detach();
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
        self.selection = None;
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
        self.input_scroll = 0;
    }

    /// The composer as last drawn (or unwrapped before the first draw).
    fn composer_layout(&self) -> ComposerLayout {
        let width = if self.input_width == 0 {
            usize::MAX / 2
        } else {
            self.input_width
        };
        ComposerLayout::new(&self.input_buffer, width, TAB_WIDTH)
    }

    /// Up/Down in the composer: move one wrapped row, keeping the column.
    fn cursor_vertical(&mut self, delta: isize) {
        let layout = self.composer_layout();
        let goal = *self
            .input_goal_col
            .get_or_insert_with(|| layout.position(self.input_cursor).1);
        self.input_cursor = layout.vertical(self.input_cursor, delta, goal);
    }

    // ── Mouse selection ───────────────────────────────────────────────────

    /// The region under screen cell (`x`, `y`) and the content point there.
    fn hit(&self, x: u16, y: u16) -> Option<(Target, selection::Point)> {
        let inside = |r: Rect| x >= r.x && x < r.right() && y >= r.y && y < r.bottom();
        if let Some(inner) = self.input_inner.filter(|r| inside(panel_inner_outer(*r))) {
            return Some((
                Target::Composer,
                self.content_point(Target::Composer, inner, x, y),
            ));
        }
        let i = self.panel_at(x, y)?;
        let inner = panel_inner(self.panels[i].area?);
        Some((
            Target::Panel(i),
            self.content_point(Target::Panel(i), inner, x, y),
        ))
    }

    /// Content point for screen cell (`x`, `y`), clamped into `inner` so a
    /// drag past the edge of a region selects up to that edge.
    fn content_point(&self, target: Target, inner: Rect, x: u16, y: u16) -> selection::Point {
        let col = x.clamp(inner.x, inner.right().saturating_sub(1).max(inner.x)) - inner.x;
        let row = y.clamp(inner.y, inner.bottom().saturating_sub(1).max(inner.y)) - inner.y;
        let scroll = match target {
            Target::Composer => self.input_scroll,
            Target::Panel(i) => self.panels.get(i).map_or(0, |p| p.scroll_offset as usize),
        };
        (scroll + row as usize, col as usize)
    }

    fn target_inner(&self, target: Target) -> Option<Rect> {
        match target {
            Target::Composer => self.input_inner,
            Target::Panel(i) => self.panels.get(i)?.area.map(panel_inner),
        }
    }

    /// Scroll a region one row because a drag went past its top or bottom.
    fn drag_scroll(&mut self, target: Target, delta: i32) {
        match target {
            Target::Panel(i) => self.scroll_by(i, delta),
            Target::Composer => {
                self.input_follow_cursor = false;
                let rows = self.composer_layout().row_count();
                let visible = self.input_inner.map_or(1, |r| r.height.max(1) as usize);
                let max = rows.saturating_sub(visible);
                self.input_scroll =
                    (self.input_scroll as i64 + delta as i64).clamp(0, max as i64) as usize;
            }
        }
    }

    /// The text under the current selection, if it covers any.
    fn selected_text(&self) -> Option<String> {
        let sel = self.selection.filter(|s| s.dragged)?;
        let (start, end) = sel.bounds();
        let text = match sel.target {
            Target::Composer => {
                let (a, b) = self.composer_layout().selected_range(start, end)?;
                self.input_buffer[a..b].to_string()
            }
            Target::Panel(i) => {
                let panel = self.panels.get(i)?;
                let width = panel.area.map(panel_inner)?.width;
                selection::lines_text(&panel_lines(panel, self.show_tokens), width, start, end)
            }
        };
        (!text.trim().is_empty()).then_some(text)
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
            // A turn restored from a saved session has no stats to show.
            let restored = *duration_ms == 0 && input_tokens.is_none() && output_tokens.is_none();
            let mut stats = Vec::new();
            if show_tokens {
                if let (Some(i), Some(o)) = (input_tokens, output_tokens) {
                    stats.push(format!("{} in / {} out", i, o));
                }
            }
            if !restored {
                stats.push(format_duration(*duration_ms));
                out.push(Line::styled(
                    stats.join(" | "),
                    Style::default().fg(Color::DarkGray),
                ));
            }
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
    selection: Option<&Selection>,
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

    let lines = panel_lines(panel, show_tokens);

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
    if let Some(sel) = selection {
        selection::highlight(
            frame.buffer_mut(),
            panel_inner(area),
            panel.scroll_offset as usize,
            sel,
        );
    }
}

/// Every line a panel shows: its completed turns plus the in-flight
/// streaming text. Selection copies from the same lines the panel draws.
fn panel_lines(panel: &PanelState, show_tokens: bool) -> Vec<Line<'static>> {
    let color = panel.get_color();
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
    lines
}

/// The bordered box around an inner text area (the inverse of
/// `panel_inner`).
fn panel_inner_outer(inner: Rect) -> Rect {
    Rect::new(
        inner.x.saturating_sub(1),
        inner.y.saturating_sub(1),
        inner.width.saturating_add(2),
        inner.height.saturating_add(2),
    )
}

/// The text area inside a bordered box.
fn panel_inner(area: Rect) -> Rect {
    Rect::new(
        area.x.saturating_add(1),
        area.y.saturating_add(1),
        area.width.saturating_sub(2),
        area.height.saturating_sub(2),
    )
}

fn draw_ui(frame: &mut Frame, app: &mut App) {
    let size = frame.area();

    // The composer wraps its text and grows a row at a time, up to
    // MAX_ROWS (fewer on a short terminal, so the panels keep 5 rows), then
    // scrolls. It spans the full width, so its text width is known up front.
    let text_width = size.width.saturating_sub(2) as usize;
    let layout = ComposerLayout::new(&app.input_buffer, text_width, TAB_WIDTH);
    let room = (size.height as usize).saturating_sub(PANEL_MIN_HEIGHT + 1 + 2);
    let visible_rows = layout.row_count().min(composer::MAX_ROWS).min(room).max(1);

    let main_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(PANEL_MIN_HEIGHT as u16),    // panels
            Constraint::Length(visible_rows as u16 + 2), // input box
            Constraint::Length(1),                       // footer
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
            let sel = app
                .selection
                .as_ref()
                .filter(|s| s.target == Target::Panel(i));
            draw_panel(
                frame,
                &mut app.panels[i],
                area,
                is_selected,
                show_tokens,
                sel,
            );
        }
    }

    let input_area = main_layout[1];
    draw_composer(frame, app, &layout, input_area);

    // Footer: mode-appropriate keybind hint, or a transient status message.
    let help_text: String = if let Some(msg) = &app.status_message {
        msg.clone()
    } else {
        match app.mode {
            Mode::Insert => {
                "Esc: normal | Enter: send | Alt+Enter: newline | Ctrl+C: cancel turn | Ctrl+O: model | PgUp/PgDn: scroll | drag: copy | /save /load /sessions /model /clear /exit"
                    .to_string()
            }
            Mode::Normal => {
                "i: insert | h/l or ←/→: select panel | j/k or ↑/↓: scroll | m: model | drag: copy | q: quit"
                    .to_string()
            }
        }
    };
    let footer = Paragraph::new(help_text).style(Style::default().fg(Color::Gray));
    frame.render_widget(footer, main_layout[2]);

    if let Some(picker) = app.picker.as_mut() {
        let catalog = picker
            .catalog_provider
            .as_ref()
            .and_then(|name| app.catalogs.get(name));
        crate::ui_picker::draw(frame, picker, catalog);
    }
}

/// Draw the input box: the rows of `layout` in view, a "rows shown / total"
/// marker once it scrolls, the cursor in Insert mode and any selection.
fn draw_composer(frame: &mut Frame, app: &mut App, layout: &ComposerLayout, area: Rect) {
    let (title, border_style) = match app.mode {
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
    let inner = panel_inner(area);
    app.input_inner = Some(inner);
    app.input_width = inner.width as usize;

    let total = layout.row_count();
    let visible = (inner.height as usize).clamp(1, total.max(1));
    let (cursor_row, cursor_col) = layout.position(app.input_cursor);
    if app.input_follow_cursor {
        if cursor_row < app.input_scroll {
            app.input_scroll = cursor_row;
        } else if cursor_row >= app.input_scroll + visible {
            app.input_scroll = cursor_row + 1 - visible;
        }
    }
    app.input_scroll = app.input_scroll.min(total - visible);
    let scroll = app.input_scroll;

    let mut block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(border_style);
    if total > visible {
        block = block.title(
            Line::styled(
                format!(" {}-{}/{} ", scroll + 1, scroll + visible, total),
                Style::default().fg(Color::DarkGray),
            )
            .right_aligned(),
        );
    }
    // Rows are sliced here rather than handed to `Paragraph::scroll`, whose
    // u16 offset would wrap for a paste taller than 65,535 rows.
    let rows: Vec<Line> = layout.rows[scroll..scroll + visible]
        .iter()
        .map(|r| Line::from(r.as_str()))
        .collect();
    frame.render_widget(Paragraph::new(rows).block(block), area);

    if let Some(sel) = app
        .selection
        .as_ref()
        .filter(|s| s.target == Target::Composer)
    {
        selection::highlight(frame.buffer_mut(), inner, scroll, sel);
    }

    let cursor_in_view = (scroll..scroll + visible).contains(&cursor_row);
    if app.mode == Mode::Insert && cursor_in_view && inner.width > 0 {
        let col = cursor_col.min(inner.width as usize - 1) as u16;
        let row = (cursor_row - scroll) as u16;
        frame.set_cursor_position((inner.x + col, inner.y + row));
    }
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
    resume: Option<Resume>,
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
        let panel = PanelState::new(provider.clone());
        let task_history = Arc::clone(&panel.history);
        panels.push(panel);

        let task_manager = Arc::clone(&manager);
        let task_provider = provider.clone();
        let task_ctx = ctx.clone();
        let task_event_tx = event_tx.clone();
        let task_cx = parent_cx.clone();
        // Handles are kept on the App: watched for unexpected exits and
        // aborted on quit.
        tasks.push(tokio::spawn(
            async move {
                panel_task(
                    task_manager,
                    task_provider,
                    task_ctx,
                    cmd_rx,
                    task_event_tx,
                    task_history,
                )
                .await;
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
                app.bedrock_providers = manager.bedrock_provider_names();
                // Fetch the catalog the picker will most likely show now, so
                // it opens ready.
                if let Some(first) = app.catalog_provider_for(0) {
                    app.request_catalog(&first);
                }

                // `--continue`: reopen the saved session first, so a
                // question given on the command line is its next turn.
                if let Some(resume) = &resume {
                    app.resume(resume);
                }

                // Seed the first turn with the CLI-provided question, if any.
                let initial = initial_question.trim();
                if !initial.is_empty() {
                    app.submit_turn(initial.to_string());
                }

                let result = match Terminal::new(CrosstermBackend::new(io::stdout())) {
                    Ok(mut terminal) => {
                        run_main_loop(&mut terminal, &mut app, &manager, event_rx).await
                    }
                    Err(e) => Err(e.into()),
                };

                // Stop in-flight requests instead of abandoning them to run
                // (and bill) in the background while the process exits.
                app.cancel_current_turn();
                // Save the conversation for `ai -t --continue` once the
                // cancelled turns have been written back.
                let saved = app.autosave_on_exit().await;
                for task in &app.tasks {
                    task.abort();
                }
                (result, saved)
            }
            Err(e) => (Err(e.into()), None),
        }
    };
    let (result, saved) = result;
    crate::diag::flush();
    // The terminal is back to normal, so this prints in the shell.
    if let Some(note) = saved {
        eprintln!("ai: {note}");
    }

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

/// Long-lived per-panel worker. Keeps the panel's `Vec<ChatMessage>`
/// (system prompt included, if any) in the shared `history`; executes one
/// `stream_with_messages` per Ask command. On Clear, resets history back to
/// just the system prompt. The UI only replaces the history (loading a
/// session) while no turn is in flight, so each Ask reads it fresh and
/// writes it back when done, counting the Ask as committed either way.
async fn panel_task(
    manager: Arc<ProviderManager>,
    mut provider: Provider,
    ctx: QueryContext,
    mut cmd_rx: mpsc::Receiver<PanelCommand>,
    event_tx: mpsc::Sender<StreamEvent>,
    history: History,
) {
    let base_messages: Vec<ChatMessage> = if let Some(s) = ctx.system_prompt.as_deref() {
        vec![ChatMessage::system(s.to_string())]
    } else {
        Vec::new()
    };
    {
        // A resumed session fills the history before the worker starts.
        let mut h = ui_sessions::lock(&history);
        if h.messages.is_empty() {
            h.messages = base_messages.clone();
        }
    }

    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            PanelCommand::Ask { question, cancel } => {
                let mut messages = ui_sessions::lock(&history).messages.clone();
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
                let mut h = ui_sessions::lock(&history);
                h.messages = messages;
                h.committed += 1;
            }
            PanelCommand::Clear => {
                ui_sessions::lock(&history).messages = base_messages.clone();
            }
            PanelCommand::SetModel {
                provider_name,
                model,
            } => {
                provider.name = provider_name;
                provider.model = model;
            }
        }
    }
}

async fn run_main_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    app: &mut App,
    manager: &Arc<ProviderManager>,
    mut event_rx: mpsc::Receiver<StreamEvent>,
) -> Result<()> {
    // Only redraw when something changed. Redrawing re-parses every panel's
    // markdown, so doing it on a fixed 50ms tick burned CPU while idle.
    let mut dirty = true;
    let (catalog_tx, mut catalog_rx) = mpsc::unbounded_channel::<CatalogResult>();
    let mut catalog_tasks: Vec<JoinHandle<()>> = Vec::new();
    let result = loop {
        for provider in std::mem::take(&mut app.catalog_requests) {
            let Some(client) = manager.bedrock_client(&provider) else {
                app.catalog_loaded(provider, Err("not a Bedrock provider".into()));
                continue;
            };
            let tx = catalog_tx.clone();
            catalog_tasks.push(tokio::spawn(async move {
                let result = crate::bedrock_catalog::list_models(&client)
                    .await
                    .map_err(|e| format!("{:#}", e));
                let _ = tx.send((provider, result));
            }));
        }
        while let Ok((provider, result)) = catalog_rx.try_recv() {
            app.catalog_loaded(provider, result);
            dirty = true;
        }
        match main_loop_step(terminal, app, &mut event_rx, &mut dirty).await {
            Ok(true) => continue,
            Ok(false) => break Ok(()),
            Err(e) => break Err(e),
        }
    };
    for task in catalog_tasks {
        task.abort();
    }
    result
}

/// One pass of the UI loop: redraw if needed, handle input and stream
/// events. `Ok(false)` means quit.
async fn main_loop_step(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    app: &mut App,
    event_rx: &mut mpsc::Receiver<StreamEvent>,
    dirty: &mut bool,
) -> Result<bool> {
    if *dirty {
        terminal.draw(|frame| draw_ui(frame, app))?;
        *dirty = false;
    }

    if event::poll(std::time::Duration::from_millis(50))? {
        match event::read()? {
            Event::Key(key) if key.kind == KeyEventKind::Press => {
                // Any key clears the transient status hint so it doesn't
                // linger past user acknowledgement.
                app.status_message = None;
                handle_key(app, key);
                if app.should_quit {
                    return Ok(false);
                }
            }
            Event::Mouse(mouse) if app.picker.is_none() => handle_mouse(app, mouse),
            Event::Paste(text) => {
                if let Some(picker) = app.picker.as_mut() {
                    picker.insert_text(&text);
                } else if app.mode == Mode::Insert {
                    app.selection = None;
                    app.input_follow_cursor = true;
                    app.insert_text(&normalize_newlines(&text));
                }
            }
            // Content coordinates shift with the width: drop the selection
            // rather than highlight the wrong text.
            Event::Resize(..) => app.selection = None,
            _ => {}
        }
        *dirty = true;
    }

    while let Ok(event) = event_rx.try_recv() {
        app.handle_event(event);
        *dirty = true;
    }

    if app.reap_dead_tasks() {
        *dirty = true;
    }
    Ok(true)
}

/// Pasted text arrives with whatever line endings the source used; keep
/// plain LF so the provider sees normal newlines.
fn normalize_newlines(s: &str) -> String {
    s.replace("\r\n", "\n").replace('\r', "\n")
}

fn handle_key(app: &mut App, key: KeyEvent) {
    if app.picker.is_some() {
        app.handle_picker_key(key);
        return;
    }
    // Typing ends a selection and brings the cursor back into view.
    app.selection = None;
    app.input_follow_cursor = true;
    let vertical = app.mode == Mode::Insert && matches!(key.code, KeyCode::Up | KeyCode::Down);
    if !vertical {
        app.input_goal_col = None;
    }
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
        KeyCode::Char('m') => app.open_model_picker(),
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
        // A newline without sending. Shift+Enter only arrives as such on
        // terminals that report modifiers on Enter; Alt+Enter and Ctrl+J
        // work everywhere.
        KeyCode::Enter
            if key
                .modifiers
                .intersects(KeyModifiers::ALT | KeyModifiers::SHIFT) =>
        {
            app.insert_text("\n")
        }
        KeyCode::Char('j') if ctrl && !alt => app.insert_text("\n"),
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
                    "model" | "models" => {
                        // The ID keeps its case: only the command name is
                        // matched case-insensitively.
                        let id = cmd.split_whitespace().nth(1).unwrap_or("");
                        app.clear_input();
                        if id.is_empty() {
                            app.open_model_picker();
                        } else {
                            let idx = app.selected_panel;
                            if idx < app.panels.len() {
                                let provider = app.route_for_typed(idx, id);
                                app.set_panel_model(idx, provider, id.to_string());
                            }
                        }
                    }
                    _ => {
                        // Session names keep their case.
                        let arg = cmd.split_whitespace().nth(1);
                        if app.session_command(cmd_name, arg) {
                            app.clear_input();
                        } else {
                            app.status_message = Some(format!("unknown command: /{}", cmd_name));
                            app.clear_input();
                        }
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
        KeyCode::Up => app.cursor_vertical(-1),
        KeyCode::Down => app.cursor_vertical(1),
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
        KeyCode::Char('o') if ctrl && !alt => app.open_model_picker(),
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
    let (x, y) = (mouse.column, mouse.row);
    match mouse.kind {
        MouseEventKind::Down(MouseButton::Left) => {
            app.selection = None;
            let Some((target, at)) = app.hit(x, y) else {
                return;
            };
            match target {
                Target::Panel(i) => app.selected_panel = i,
                Target::Composer => {
                    // Click to place the cursor, like any text field.
                    app.mode = Mode::Insert;
                    app.input_goal_col = None;
                    app.input_cursor = app.composer_layout().offset_at(at.0, at.1);
                }
            }
            app.selection = Some(Selection::new(target, at));
        }
        MouseEventKind::Drag(MouseButton::Left) => {
            let Some(target) = app.selection.map(|s| s.target) else {
                return;
            };
            let Some(inner) = app.target_inner(target) else {
                return;
            };
            // Dragging past the top or bottom edge scrolls to select more.
            if y < inner.y {
                app.drag_scroll(target, -1);
            } else if y >= inner.bottom() {
                app.drag_scroll(target, 1);
            } else if target == Target::Composer {
                app.input_follow_cursor = false;
            }
            let head = app.content_point(target, inner, x, y);
            if let Some(sel) = app.selection.as_mut() {
                sel.dragged |= head != sel.anchor;
                sel.head = head;
            }
        }
        MouseEventKind::Up(MouseButton::Left) => {
            // Take the release point too, in case no drag event reported it.
            if let Some(sel) = app.selection {
                if let Some(inner) = app.target_inner(sel.target) {
                    let head = app.content_point(sel.target, inner, x, y);
                    if let Some(sel) = app.selection.as_mut() {
                        sel.dragged |= head != sel.anchor;
                        sel.head = head;
                    }
                }
            }
            match app.selected_text() {
                Some(text) => {
                    (app.copy_to_clipboard)(&text);
                    let n = text.chars().count();
                    app.status_message = Some(format!(
                        "copied {} character{} to the clipboard",
                        n,
                        if n == 1 { "" } else { "s" }
                    ));
                }
                // A click, or a drag over nothing: no selection to keep.
                None => app.selection = None,
            }
        }
        MouseEventKind::ScrollUp | MouseEventKind::ScrollDown
            if app.hit(x, y).is_some_and(|(t, _)| t == Target::Composer) =>
        {
            let delta = if mouse.kind == MouseEventKind::ScrollUp {
                -1
            } else {
                1
            };
            app.drag_scroll(Target::Composer, delta);
        }
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
pub(super) mod tests {
    use super::*;
    use crate::providers::Provider;
    use ratatui::backend::TestBackend;

    pub(super) fn fake_provider(name: &str) -> Provider {
        Provider {
            name: name.to_lowercase(),
            display_name: name.to_string(),
            model: "test-model".to_string(),
            color: colored::Color::Cyan,
        }
    }

    /// Build an `App` with `n` panels plus keep the paired receivers alive
    /// for the duration of the test. If the receivers drop, `try_send` sees
    /// `TrySendError::Closed` and submit_turn rolls back the user turn — so
    /// tests need to hold onto the receivers explicitly.
    pub(super) fn app_with(n: usize) -> (App, Vec<mpsc::Receiver<PanelCommand>>) {
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
                draw_panel(f, &mut app.panels[0], area, true, false, None);
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

    thread_local! {
        static COPIED: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    fn record_copy(text: &str) {
        COPIED.with(|c| c.borrow_mut().push(text.to_string()));
    }

    fn take_copied() -> Vec<String> {
        COPIED.with(|c| std::mem::take(&mut *c.borrow_mut()))
    }

    /// Draw the whole UI on a `width` x `height` test terminal.
    fn draw(app: &mut App, width: u16, height: u16) -> Terminal<TestBackend> {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|f| draw_ui(f, app)).unwrap();
        terminal
    }

    /// The composer's text rows as drawn, and its total height with borders.
    fn composer_rows(terminal: &Terminal<TestBackend>, app: &App) -> (Vec<String>, u16) {
        let inner = app.input_inner.expect("composer drawn");
        let buf = terminal.backend().buffer();
        let rows = (inner.y..inner.bottom())
            .map(|y| {
                (inner.x..inner.right())
                    .map(|x| buf[(x, y)].symbol())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect();
        (rows, inner.height + 2)
    }

    fn mouse(kind: MouseEventKind, column: u16, row: u16) -> crossterm::event::MouseEvent {
        crossterm::event::MouseEvent {
            kind,
            column,
            row,
            modifiers: KeyModifiers::NONE,
        }
    }

    #[test]
    fn composer_grows_a_row_when_the_text_fills_it() {
        // The reported bug: a full row must make the box taller and put the
        // cursor at the start of the new row.
        let (mut app, _rxs) = app_with(1);
        let mut t = draw(&mut app, 22, 20); // 20 text columns
        assert_eq!(composer_rows(&t, &app).1, 3);
        type_str(&mut app, &"a".repeat(20));
        t.draw(|f| draw_ui(f, &mut app)).unwrap();
        let (rows, height) = composer_rows(&t, &app);
        assert_eq!(height, 4);
        assert_eq!(rows, ["a".repeat(20), String::new()]);
        let inner = app.input_inner.unwrap();
        t.backend_mut()
            .assert_cursor_position((inner.x, inner.y + 1));
        type_str(&mut app, "bc");
        t.draw(|f| draw_ui(f, &mut app)).unwrap();
        assert_eq!(composer_rows(&t, &app).0[1], "bc");
    }

    #[test]
    fn composer_stops_growing_at_five_rows_and_scrolls() {
        let (mut app, _rxs) = app_with(1);
        draw(&mut app, 22, 30);
        let text: String = (0..9).map(|i| format!("row{i}").repeat(5)).collect();
        type_str(&mut app, &text); // nine full 20-column rows + the cursor row
        let mut t = draw(&mut app, 22, 30);
        let (rows, height) = composer_rows(&t, &app);
        assert_eq!(height, 7, "5 text rows plus borders");
        assert_eq!(rows[0], "row5".repeat(5));
        assert_eq!(rows[4], "", "the cursor's empty row is in view");
        // The marker says which rows are shown.
        let buf = t.backend().buffer();
        let top = app.input_inner.unwrap().y - 1;
        let title: String = (0..22).map(|x| buf[(x, top)].symbol()).collect();
        assert!(title.contains(" 6-10/10 "), "title: {title:?}");
        // Up walks the cursor back and the view follows it.
        for _ in 0..9 {
            handle_key(&mut app, key(KeyCode::Up));
        }
        t.draw(|f| draw_ui(f, &mut app)).unwrap();
        assert_eq!(app.input_cursor, 0);
        assert_eq!(composer_rows(&t, &app).0[0], "row0".repeat(5));
    }

    #[test]
    fn composer_leaves_the_panels_room_on_a_short_terminal() {
        let (mut app, _rxs) = app_with(1);
        draw(&mut app, 22, 10);
        type_str(&mut app, &"x".repeat(200));
        let t = draw(&mut app, 22, 10);
        // 10 rows: 5 for panels, 1 footer, so the composer gets 4 (2 text).
        assert_eq!(composer_rows(&t, &app).1, 4);
        assert!(app.panels[0].area.unwrap().height >= 5);
    }

    #[test]
    fn newlines_show_as_line_breaks_and_alt_enter_adds_one() {
        let (mut app, _rxs) = app_with(1);
        app.insert_text("fn main() {\n}");
        handle_key(&mut app, KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT));
        handle_key(&mut app, ctrl('j'));
        type_str(&mut app, "x");
        assert_eq!(app.input_buffer, "fn main() {\n}\n\nx");
        let t = draw(&mut app, 30, 20);
        assert_eq!(composer_rows(&t, &app).0, ["fn main() {", "}", "", "x"]);
        assert!(app.panels[0].turns.is_empty(), "nothing was sent");
    }

    #[test]
    fn up_and_down_keep_the_goal_column_across_short_rows() {
        let (mut app, _rxs) = app_with(1);
        app.insert_text("abcdefgh\nab\nabcdefgh");
        draw(&mut app, 22, 20);
        app.input_cursor = "abcdefgh\nab\nabcdef".len(); // row 2, col 6
        handle_key(&mut app, key(KeyCode::Up));
        assert_eq!(
            app.input_cursor,
            "abcdefgh\nab".len(),
            "clamped to the short row"
        );
        handle_key(&mut app, key(KeyCode::Up));
        assert_eq!(app.input_cursor, 6, "back at column 6");
        handle_key(&mut app, key(KeyCode::Down));
        handle_key(&mut app, key(KeyCode::Down));
        assert_eq!(app.input_cursor, "abcdefgh\nab\nabcdef".len());
    }

    #[test]
    fn clicking_the_composer_places_the_cursor() {
        let (mut app, _rxs) = app_with(1);
        app.copy_to_clipboard = record_copy;
        app.insert_text("hello world");
        app.mode = Mode::Normal;
        draw(&mut app, 30, 20);
        let inner = app.input_inner.unwrap();
        handle_mouse(
            &mut app,
            mouse(
                MouseEventKind::Down(MouseButton::Left),
                inner.x + 3,
                inner.y,
            ),
        );
        handle_mouse(
            &mut app,
            mouse(MouseEventKind::Up(MouseButton::Left), inner.x + 3, inner.y),
        );
        assert_eq!(app.input_cursor, 3);
        assert_eq!(app.mode, Mode::Insert);
        assert!(app.selection.is_none(), "a click is not a selection");
        assert!(take_copied().is_empty());
    }

    #[test]
    fn dragging_in_a_panel_copies_the_text_and_highlights_it() {
        let (mut app, _rxs) = app_with(2);
        app.copy_to_clipboard = record_copy;
        app.panels[1].turns.push(Turn {
            kind: TurnKind::User,
            text: "the quick brown fox jumps over the lazy dog".into(),
        });
        let mut t = draw(&mut app, 60, 20);
        let inner = panel_inner(app.panels[1].area.unwrap());
        // Row 0 is "you"; the question (indented two columns) wraps from
        // row 1 at 28 columns, so column 6 is the "q" of "quick".
        handle_mouse(
            &mut app,
            mouse(
                MouseEventKind::Down(MouseButton::Left),
                inner.x + 6,
                inner.y + 1,
            ),
        );
        assert_eq!(app.selected_panel, 1, "clicking a panel selects it");
        handle_mouse(
            &mut app,
            mouse(
                MouseEventKind::Drag(MouseButton::Left),
                inner.x + 10,
                inner.y + 2,
            ),
        );
        handle_mouse(
            &mut app,
            mouse(
                MouseEventKind::Up(MouseButton::Left),
                inner.x + 10,
                inner.y + 2,
            ),
        );
        let copied = take_copied();
        assert_eq!(copied.len(), 1);
        assert!(
            copied[0].starts_with("quick brown fox jumps over"),
            "soft-wrapped rows join into one line: {copied:?}"
        );
        assert!(!copied[0].contains('\n'));
        assert!(app
            .status_message
            .as_deref()
            .unwrap()
            .starts_with("copied "));
        t.draw(|f| draw_ui(f, &mut app)).unwrap();
        let buf = t.backend().buffer();
        assert!(buf[(inner.x + 6, inner.y + 1)]
            .modifier
            .contains(Modifier::REVERSED));
        assert!(!buf[(inner.x + 5, inner.y + 1)]
            .modifier
            .contains(Modifier::REVERSED));
        // The other panel is untouched.
        let other = panel_inner(app.panels[0].area.unwrap());
        assert!(!buf[(other.x + 6, other.y + 1)]
            .modifier
            .contains(Modifier::REVERSED));
        // Typing clears the highlight.
        type_str(&mut app, "x");
        assert!(app.selection.is_none());
    }

    #[test]
    fn dragging_in_the_composer_copies_the_exact_buffer_text() {
        let (mut app, _rxs) = app_with(1);
        app.copy_to_clipboard = record_copy;
        app.insert_text("line one\nline two");
        draw(&mut app, 30, 20);
        let inner = app.input_inner.unwrap();
        handle_mouse(
            &mut app,
            mouse(
                MouseEventKind::Down(MouseButton::Left),
                inner.x + 5,
                inner.y,
            ),
        );
        handle_mouse(
            &mut app,
            mouse(
                MouseEventKind::Drag(MouseButton::Left),
                inner.x + 3,
                inner.y + 1,
            ),
        );
        handle_mouse(
            &mut app,
            mouse(
                MouseEventKind::Up(MouseButton::Left),
                inner.x + 3,
                inner.y + 1,
            ),
        );
        assert_eq!(take_copied(), ["one\nline"]);
        // A release with no drag before it still selects up to that point.
        handle_mouse(
            &mut app,
            mouse(MouseEventKind::Down(MouseButton::Left), inner.x, inner.y),
        );
        handle_mouse(
            &mut app,
            mouse(MouseEventKind::Up(MouseButton::Left), inner.x + 3, inner.y),
        );
        assert_eq!(take_copied(), ["line"]);
    }

    #[test]
    fn dragging_past_a_panel_edge_scrolls_it() {
        let (mut app, _rxs) = app_with(1);
        app.copy_to_clipboard = record_copy;
        for i in 0..30 {
            app.panels[0].turns.push(Turn {
                kind: TurnKind::User,
                text: format!("message {i}"),
            });
        }
        draw(&mut app, 40, 20);
        let inner = panel_inner(app.panels[0].area.unwrap());
        let before = app.panels[0].scroll_offset;
        assert!(before > 0, "pinned to the bottom");
        handle_mouse(
            &mut app,
            mouse(
                MouseEventKind::Down(MouseButton::Left),
                inner.x,
                inner.bottom() - 1,
            ),
        );
        for _ in 0..3 {
            handle_mouse(
                &mut app,
                mouse(MouseEventKind::Drag(MouseButton::Left), inner.x + 5, 0),
            );
        }
        assert_eq!(app.panels[0].scroll_offset, before - 3);
        handle_mouse(
            &mut app,
            mouse(MouseEventKind::Up(MouseButton::Left), inner.x + 5, 0),
        );
        let copied = take_copied();
        assert_eq!(copied.len(), 1);
        assert!(copied[0].lines().count() > inner.height as usize);
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

    // ── Model picker ──────────────────────────────────────────────────────

    fn bedrock_model(id: &str) -> BedrockModel {
        BedrockModel {
            id: id.into(),
            name: id.into(),
            vendor: "V".into(),
        }
    }

    /// Two panels; "p0" routes to Bedrock (its catalog is loaded), "p1"
    /// doesn't.
    fn picker_app() -> (App, Vec<mpsc::Receiver<PanelCommand>>) {
        let (mut app, rxs) = app_with(2);
        app.bedrock_providers = vec!["p0".into()];
        app.catalog_loaded(
            "p0".into(),
            Ok(vec![
                bedrock_model("us.anthropic.claude-opus-5-5"),
                bedrock_model("us.xai.grok-4.7"),
            ]),
        );
        (app, rxs)
    }

    fn expect_set_model(rx: &mut mpsc::Receiver<PanelCommand>) -> (String, String) {
        match rx.try_recv() {
            Ok(PanelCommand::SetModel {
                provider_name,
                model,
            }) => (provider_name, model),
            Ok(_) => panic!("expected SetModel, got another command"),
            Err(e) => panic!("expected SetModel, got {:?}", e),
        }
    }

    #[test]
    fn m_opens_the_picker_and_enter_switches_the_panel() {
        let (mut app, mut rxs) = picker_app();
        app.mode = Mode::Normal;
        handle_key(&mut app, key(KeyCode::Char('m')));
        assert!(app.picker.is_some());
        // Keys now go to the picker, not to normal-mode navigation.
        type_str(&mut app, "grok");
        assert_eq!(app.selected_panel, 0);
        handle_key(&mut app, key(KeyCode::Enter));
        assert!(app.picker.is_none());
        assert_eq!(
            expect_set_model(&mut rxs[0]),
            ("p0".into(), "us.xai.grok-4.7".into())
        );
        assert_eq!(app.panels[0].model, "us.xai.grok-4.7");
        assert_eq!(app.panels[0].provider.model, "us.xai.grok-4.7");
    }

    #[test]
    fn ctrl_o_opens_the_picker_and_esc_closes_it_unchanged() {
        let (mut app, mut rxs) = picker_app();
        app.mode = Mode::Insert;
        type_str(&mut app, "draft");
        handle_key(&mut app, ctrl('o'));
        assert!(app.picker.is_some());
        type_str(&mut app, "opus");
        handle_key(&mut app, key(KeyCode::Esc));
        assert!(app.picker.is_none());
        // Esc closed the picker only: still in insert mode, draft intact.
        assert_eq!(app.mode, Mode::Insert);
        assert_eq!(app.input_buffer, "draft");
        assert!(rxs[0].try_recv().is_err());
        assert_eq!(app.panels[0].model, "test-model");
    }

    #[test]
    fn a_non_bedrock_panel_picks_from_the_bedrock_catalog_and_is_rerouted() {
        let (mut app, mut rxs) = picker_app();
        app.mode = Mode::Normal;
        app.selected_panel = 1;
        handle_key(&mut app, key(KeyCode::Char('m')));
        assert_eq!(
            app.picker.as_ref().unwrap().catalog_provider.as_deref(),
            Some("p0")
        );
        type_str(&mut app, "opus");
        handle_key(&mut app, key(KeyCode::Enter));
        assert_eq!(
            expect_set_model(&mut rxs[1]),
            ("p0".into(), "us.anthropic.claude-opus-5-5".into())
        );
        assert_eq!(app.panels[1].provider.name, "p0");
    }

    #[test]
    fn model_command_sets_any_id_and_keeps_its_case() {
        let (mut app, mut rxs) = picker_app();
        app.mode = Mode::Insert;
        type_str(&mut app, "/model us.meta.Llama-Custom:0");
        handle_key(&mut app, key(KeyCode::Enter));
        assert_eq!(
            expect_set_model(&mut rxs[0]),
            ("p0".into(), "us.meta.Llama-Custom:0".into())
        );
        assert!(app.input_buffer.is_empty());
    }

    #[test]
    fn typed_ids_on_a_non_bedrock_panel_only_go_to_bedrock_when_listed() {
        let (mut app, mut rxs) = picker_app();
        app.selected_panel = 1;
        app.mode = Mode::Insert;
        // Not in the catalog: stays with the panel's own provider.
        type_str(&mut app, "/model gemini-3.1-pro-preview");
        handle_key(&mut app, key(KeyCode::Enter));
        assert_eq!(
            expect_set_model(&mut rxs[1]),
            ("p1".into(), "gemini-3.1-pro-preview".into())
        );
        // A listed Bedrock model goes to Bedrock.
        type_str(&mut app, "/model us.xai.grok-4.7");
        handle_key(&mut app, key(KeyCode::Enter));
        assert_eq!(
            expect_set_model(&mut rxs[1]),
            ("p0".into(), "us.xai.grok-4.7".into())
        );
    }

    #[test]
    fn model_command_without_an_id_opens_the_picker() {
        let (mut app, _rxs) = picker_app();
        app.mode = Mode::Insert;
        type_str(&mut app, "/model");
        handle_key(&mut app, key(KeyCode::Enter));
        assert!(app.picker.is_some());
    }

    #[test]
    fn the_model_cant_change_mid_answer() {
        let (mut app, mut rxs) = picker_app();
        app.panels[0].is_streaming = true;
        app.mode = Mode::Normal;
        handle_key(&mut app, key(KeyCode::Char('m')));
        assert!(app.picker.is_none());
        assert!(app.status_message.as_deref().unwrap().contains("answering"));
        app.set_panel_model(0, "p0".into(), "us.xai.grok-4.7".into());
        assert!(rxs[0].try_recv().is_err());
        assert_eq!(app.panels[0].model, "test-model");
    }

    #[test]
    fn opening_the_picker_requests_its_catalog_once() {
        let (mut app, _rxs) = app_with(1);
        app.bedrock_providers = vec!["p0".into()];
        app.mode = Mode::Normal;
        handle_key(&mut app, key(KeyCode::Char('m')));
        handle_key(&mut app, key(KeyCode::Esc));
        handle_key(&mut app, key(KeyCode::Char('m')));
        assert_eq!(app.catalog_requests, ["p0"]);
        // A failed fetch is retried on the next open.
        app.catalog_requests.clear();
        app.catalog_loaded("p0".into(), Err("boom".into()));
        handle_key(&mut app, key(KeyCode::Esc));
        handle_key(&mut app, key(KeyCode::Char('m')));
        assert_eq!(app.catalog_requests, ["p0"]);
    }

    #[test]
    fn without_bedrock_the_picker_still_accepts_a_typed_id() {
        let (mut app, mut rxs) = app_with(1);
        app.mode = Mode::Normal;
        handle_key(&mut app, key(KeyCode::Char('m')));
        assert!(app.picker.as_ref().unwrap().catalog_provider.is_none());
        type_str(&mut app, "gpt-5.6-sol");
        handle_key(&mut app, key(KeyCode::Enter));
        assert_eq!(
            expect_set_model(&mut rxs[0]),
            ("p0".into(), "gpt-5.6-sol".into())
        );
    }

    #[test]
    fn the_picker_draws_over_the_panels() {
        let (mut app, _rxs) = picker_app();
        app.mode = Mode::Normal;
        handle_key(&mut app, key(KeyCode::Char('m')));
        type_str(&mut app, "grok");
        let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
        terminal.draw(|f| draw_ui(f, &mut app)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains("Model for P0"), "{text}");
        assert!(text.contains("us.xai.grok-4.7"), "{text}");
        assert!(text.contains("use \"grok\" as the model ID"), "{text}");
        assert!(!text.contains("claude-opus"), "{text}");
    }
}
