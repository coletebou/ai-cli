//! Mouse text selection for `ai -t`.
//!
//! The TUI captures the mouse (wheel scrolling needs it), so the terminal's
//! own click-and-drag selection never happens, and with side-by-side panels
//! it would grab both panels and their borders anyway. Instead the TUI
//! selects inside one panel or the composer itself: press, drag, release, and
//! the text is copied to the clipboard.
//!
//! Points are kept in *content* coordinates, (row of the wrapped content,
//! column), not screen cells, so a selection stays on the same text while
//! the panel scrolls or keeps streaming.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::Modifier,
    text::Line,
    widgets::{Paragraph, Widget, Wrap},
};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

/// Where a selection was started. A selection never spans two regions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Panel(usize),
    Composer,
}

/// A (row, column) point in a region's wrapped content.
pub type Point = (usize, usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selection {
    pub target: Target,
    /// Where the button went down.
    pub anchor: Point,
    /// Where the pointer is now.
    pub head: Point,
    /// The pointer moved while the button was down. A plain click selects
    /// nothing.
    pub dragged: bool,
}

impl Selection {
    pub fn new(target: Target, at: Point) -> Self {
        Selection {
            target,
            anchor: at,
            head: at,
            dragged: false,
        }
    }

    /// Start and end in reading order, both inclusive.
    pub fn bounds(&self) -> (Point, Point) {
        if self.anchor <= self.head {
            (self.anchor, self.head)
        } else {
            (self.head, self.anchor)
        }
    }

    /// Whether the content cell (`row`, `col`) is selected. Selections run
    /// like text, not as a rectangle: from the start to the end of every
    /// row in between.
    pub fn contains(&self, row: usize, col: usize) -> bool {
        if !self.dragged {
            return false;
        }
        let (start, end) = self.bounds();
        (start.0 < row || (start.0 == row && start.1 <= col))
            && (row < end.0 || (row == end.0 && col <= end.1))
    }
}

/// Reverse-video the selected cells of a region drawn in `inner`, whose
/// first visible content row is `scroll`.
pub fn highlight(buf: &mut Buffer, inner: Rect, scroll: usize, sel: &Selection) {
    for dy in 0..inner.height {
        let row = scroll + dy as usize;
        for dx in 0..inner.width {
            if sel.contains(row, dx as usize) {
                let cell = &mut buf[(inner.x + dx, inner.y + dy)];
                cell.modifier.insert(Modifier::REVERSED);
            }
        }
    }
}

/// The rows `line` wraps to at `width` columns, as rendered text with the
/// trailing padding trimmed. Rendering through `Paragraph` (the same widget
/// the panel uses) guarantees the rows match what is on screen.
fn wrapped_rows(line: &Line<'_>, width: u16) -> Vec<String> {
    let paragraph = Paragraph::new(line.clone()).wrap(Wrap { trim: false });
    let height = paragraph.line_count(width).max(1);
    let height = u16::try_from(height).unwrap_or(u16::MAX);
    let area = Rect::new(0, 0, width, height);
    let mut buf = Buffer::empty(area);
    paragraph.render(area, &mut buf);
    (0..height)
        .map(|y| {
            let mut row = String::new();
            let mut x = 0;
            while x < width {
                let symbol = buf[(x, y)].symbol();
                row.push_str(symbol);
                // A wide glyph's second cell holds filler, not text.
                x += symbol.width().max(1) as u16;
            }
            row.trim_end().to_string()
        })
        .collect()
}

/// Byte offset in `row` of the grapheme covering display column `col`, or
/// None when `col` is past the row's text.
fn byte_at_col(row: &str, col: usize) -> Option<(usize, usize)> {
    let mut at = 0;
    for (i, g) in row.grapheme_indices(true) {
        let w = g.width().max(1);
        if col < at + w {
            return Some((i, i + g.len()));
        }
        at += w;
    }
    None
}

/// The text of `lines` between content points `start` and `end` (both
/// inclusive) when the lines are wrapped at `width` columns.
///
/// Rows that are soft wraps of one logical line are joined back into that
/// line, with the spaces the wrap swallowed, so copying a long paragraph
/// gives one line of prose rather than one line per screen row. Logical
/// lines are joined with newlines and lose their trailing whitespace.
pub fn lines_text(lines: &[Line<'_>], width: u16, start: Point, end: Point) -> String {
    let (start, end) = if start <= end {
        (start, end)
    } else {
        (end, start)
    };
    let width = width.max(1);
    let mut out: Vec<String> = Vec::new();
    let mut first_row = 0usize;
    for line in lines {
        if first_row > end.0 {
            break;
        }
        let rows = wrapped_rows(line, width);
        let last_row = first_row + rows.len() - 1;
        if last_row < start.0 {
            first_row = last_row + 1;
            continue;
        }
        let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        let piece = match align_rows(&text, &rows) {
            Some(bases) => {
                // Byte where each row's text ends in `text`: up to the next
                // row's start, so the whitespace eaten by the wrap is kept.
                let row_end = |j: usize| bases.get(j + 1).copied().unwrap_or(text.len());
                let from = if start.0 >= first_row {
                    let j = start.0 - first_row;
                    byte_at_col(&rows[j], start.1)
                        .map(|(b, _)| bases[j] + b)
                        .unwrap_or_else(|| row_end(j))
                } else {
                    0
                };
                let to = if end.0 <= last_row {
                    let j = end.0 - first_row;
                    byte_at_col(&rows[j], end.1)
                        .map(|(_, e)| bases[j] + e)
                        .unwrap_or_else(|| row_end(j))
                } else {
                    text.len()
                };
                text.get(from..to.max(from)).unwrap_or("").to_string()
            }
            // The rows didn't line up with the source text (a widget quirk
            // we don't model): fall back to the screen rows as shown.
            None => rows_as_shown(&rows, first_row, start, end),
        };
        out.push(piece.trim_end().to_string());
        first_row = last_row + 1;
    }
    out.join("\n")
}

/// Where each wrapped row starts in the logical line's `text`, or None if a
/// row can't be found in order.
fn align_rows(text: &str, rows: &[String]) -> Option<Vec<usize>> {
    let mut bases = Vec::with_capacity(rows.len());
    let mut pos = 0;
    for row in rows {
        if row.is_empty() {
            bases.push(pos);
            continue;
        }
        let found = text.get(pos..)?.find(row.as_str())?;
        bases.push(pos + found);
        pos += found + row.len();
    }
    Some(bases)
}

fn rows_as_shown(rows: &[String], first_row: usize, start: Point, end: Point) -> String {
    let mut parts = Vec::new();
    for (j, row) in rows.iter().enumerate() {
        let r = first_row + j;
        if r < start.0 || r > end.0 {
            continue;
        }
        let from = if r == start.0 {
            byte_at_col(row, start.1)
                .map(|(b, _)| b)
                .unwrap_or(row.len())
        } else {
            0
        };
        let to = if r == end.0 {
            byte_at_col(row, end.1).map(|(_, e)| e).unwrap_or(row.len())
        } else {
            row.len()
        };
        parts.push(row.get(from..to.max(from)).unwrap_or("").to_string());
    }
    parts.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::text::Span;

    fn lines(texts: &[&str]) -> Vec<Line<'static>> {
        texts.iter().map(|t| Line::from(t.to_string())).collect()
    }

    #[test]
    fn copies_a_range_inside_one_row() {
        let l = lines(&["hello world"]);
        assert_eq!(lines_text(&l, 40, (0, 6), (0, 10)), "world");
        // Dragged right to left gives the same text.
        assert_eq!(lines_text(&l, 40, (0, 10), (0, 6)), "world");
    }

    #[test]
    fn soft_wrapped_rows_are_joined_back_into_one_line() {
        let l = lines(&["the quick brown fox jumps over the lazy dog"]);
        let rows = wrapped_rows(&l[0], 12);
        assert!(rows.len() > 1, "test needs a wrapped line: {rows:?}");
        let last = rows.len() - 1;
        assert_eq!(
            lines_text(&l, 12, (0, 0), (last, 40)),
            "the quick brown fox jumps over the lazy dog"
        );
    }

    #[test]
    fn logical_lines_are_joined_with_newlines() {
        let l = lines(&["you", "  first question", "", "assistant", "an answer"]);
        assert_eq!(
            lines_text(&l, 40, (1, 2), (4, 1)),
            "first question\n\nassistant\nan"
        );
    }

    #[test]
    fn styled_spans_copy_as_plain_text() {
        let l = vec![Line::from(vec![
            Span::raw("call "),
            Span::styled("`foo()`", ratatui::style::Style::default()),
            Span::raw(" now"),
        ])];
        assert_eq!(lines_text(&l, 40, (0, 0), (0, 40)), "call `foo()` now");
    }

    #[test]
    fn selecting_past_the_end_of_a_row_takes_the_rest_of_it() {
        let l = lines(&["short", "next"]);
        assert_eq!(lines_text(&l, 40, (0, 2), (0, 30)), "ort");
        assert_eq!(lines_text(&l, 40, (0, 30), (1, 1)), "\nne");
    }

    #[test]
    fn wide_glyphs_copy_whole() {
        let l = lines(&["漢字 text"]);
        // Column 1 is the second half of "漢".
        assert_eq!(lines_text(&l, 40, (0, 1), (0, 2)), "漢字");
    }

    #[test]
    fn contains_runs_like_text_not_a_rectangle() {
        let mut s = Selection::new(Target::Composer, (2, 5));
        assert!(!s.contains(2, 5), "a click alone selects nothing");
        s.head = (0, 3);
        s.dragged = true;
        assert!(s.contains(0, 3));
        assert!(!s.contains(0, 2));
        assert!(s.contains(1, 0) && s.contains(1, 999));
        assert!(s.contains(2, 5));
        assert!(!s.contains(2, 6));
    }
}
