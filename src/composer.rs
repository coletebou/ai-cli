//! Layout for the `ai -t` composer (the input box): soft-wraps the buffer
//! into rows that fit the box, finds the cursor's row and column, and maps
//! screen cells back to byte offsets for Up/Down, click-to-place and
//! selection.
//!
//! The box grows one row at a time as the text wraps, up to [`MAX_ROWS`],
//! and scrolls vertically after that. Wrapping is per grapheme rather than
//! per word so every cell maps to exactly one place in the buffer and the
//! cursor never lands somewhere the text isn't.

use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

/// Rows the composer grows to before it starts scrolling.
pub const MAX_ROWS: usize = 5;

/// One grapheme of the buffer and where it was drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Cell {
    /// Byte offset of the grapheme in the buffer.
    byte: usize,
    /// Byte length of the grapheme.
    len: usize,
    row: usize,
    col: usize,
    /// Display columns it covers; a newline counts as one so it can be
    /// selected at the end of its row.
    width: usize,
}

/// The wrapped composer text for one box width.
#[derive(Debug)]
pub struct Layout {
    /// Display text of each row (tabs expanded, newlines dropped). Always at
    /// least one row, even for an empty buffer.
    pub rows: Vec<String>,
    /// Every grapheme in buffer order.
    cells: Vec<Cell>,
    /// Where the cursor sits when it is past the last grapheme.
    end: (usize, usize),
    /// Buffer length in bytes.
    len: usize,
}

impl Layout {
    pub fn new(buffer: &str, width: usize, tab_width: usize) -> Self {
        let width = width.max(1);
        let mut rows = vec![String::new()];
        let mut cells = Vec::new();
        let (mut row, mut col) = (0usize, 0usize);
        for (byte, g) in buffer.grapheme_indices(true) {
            if g.contains('\n') {
                // A hard line break (also `\r\n`, a single grapheme). It
                // occupies the cell after the row's text so the cursor can sit
                // in front of it, at the end of the line.
                cells.push(Cell {
                    byte,
                    len: g.len(),
                    row,
                    col,
                    width: 1,
                });
                row += 1;
                col = 0;
                rows.push(String::new());
                continue;
            }
            let (shown, w) = match g {
                "\t" => (" ".repeat(tab_width), tab_width),
                "\r" => (String::new(), 0),
                g => (g.to_string(), g.width()),
            };
            // Wrap before a grapheme that would cross the right edge. A
            // zero-width one still needs a free cell for the cursor, so it
            // wraps too once the row is full.
            if col > 0 && col + w.max(1) > width {
                row += 1;
                col = 0;
                rows.push(String::new());
            }
            cells.push(Cell {
                byte,
                len: g.len(),
                row,
                col,
                width: w,
            });
            rows[row].push_str(&shown);
            col += w;
        }
        // Text that exactly fills its last row puts the cursor on a fresh row
        // below, which is what makes the box grow as soon as the line is full.
        if col >= width {
            row += 1;
            col = 0;
            rows.push(String::new());
        }
        Layout {
            rows,
            cells,
            end: (row, col),
            len: buffer.len(),
        }
    }

    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Row and column of the cursor at byte offset `byte`.
    pub fn position(&self, byte: usize) -> (usize, usize) {
        if byte >= self.len {
            return self.end;
        }
        match self.cells.binary_search_by_key(&byte, |c| c.byte) {
            Ok(i) => (self.cells[i].row, self.cells[i].col),
            // Not on a grapheme boundary: report the grapheme it falls in.
            Err(i) => {
                let c = self.cells[i.saturating_sub(1)];
                (c.row, c.col)
            }
        }
    }

    /// The cursor offset nearest to screen cell (`row`, `col`): the last
    /// place on that row at or left of `col`. Past the end of a soft-wrapped
    /// row that is in front of its last grapheme, so the cursor stays on the
    /// row that was aimed at instead of jumping to the start of the next one.
    pub fn offset_at(&self, row: usize, col: usize) -> usize {
        if row > self.end.0 {
            return self.len;
        }
        let mut best: Option<(usize, usize)> = None; // (col, byte)
        let mut first: Option<(usize, usize)> = None;
        let candidates = self
            .cells
            .iter()
            .filter(|c| c.row == row)
            .map(|c| (c.col, c.byte))
            .chain((self.end.0 == row).then_some((self.end.1, self.len)));
        for (c, b) in candidates {
            first.get_or_insert((c, b));
            if c <= col {
                best = Some((c, b));
            }
        }
        best.or(first).map(|(_, b)| b).unwrap_or(self.len)
    }

    /// The offset one row up (`delta` < 0) or down from `byte`, aiming for
    /// display column `goal_col`. Moving up from the first row goes to the
    /// start of the buffer and down from the last row to the end, like most
    /// editors.
    pub fn vertical(&self, byte: usize, delta: isize, goal_col: usize) -> usize {
        let (row, _) = self.position(byte);
        if delta < 0 && row == 0 {
            return 0;
        }
        let target = row as isize + delta;
        if target < 0 {
            return 0;
        }
        if target as usize > self.end.0 {
            return self.len;
        }
        self.offset_at(target as usize, goal_col)
    }

    /// The buffer text between screen cells `start` and `end` (row, col),
    /// both inclusive, in reading order. Every grapheme whose cell overlaps
    /// the selection is taken whole, newlines included.
    pub fn selected_range(
        &self,
        start: (usize, usize),
        end: (usize, usize),
    ) -> Option<(usize, usize)> {
        let (start, end) = if start <= end {
            (start, end)
        } else {
            (end, start)
        };
        let mut range: Option<(usize, usize)> = None;
        for c in &self.cells {
            if c.row < start.0 || c.row > end.0 {
                continue;
            }
            let lo = if c.row == start.0 { start.1 } else { 0 };
            let hi = if c.row == end.0 { end.1 } else { usize::MAX };
            let last_col = c.col + c.width.max(1) - 1;
            if c.col <= hi && last_col >= lo {
                let r = range.get_or_insert((c.byte, c.byte + c.len));
                r.1 = c.byte + c.len;
            }
        }
        range
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(s: &str, width: usize) -> Layout {
        Layout::new(s, width, 4)
    }

    #[test]
    fn empty_buffer_is_one_row() {
        let l = layout("", 10);
        assert_eq!(l.rows, [""]);
        assert_eq!(l.position(0), (0, 0));
    }

    #[test]
    fn short_text_stays_on_one_row() {
        let l = layout("hello", 10);
        assert_eq!(l.rows, ["hello"]);
        assert_eq!(l.position(5), (0, 5));
    }

    #[test]
    fn filling_the_row_moves_the_cursor_to_a_new_row() {
        // The reported bug: a full row must grow the box, not send the cursor
        // back to the start of the same row.
        let l = layout("0123456789", 10);
        assert_eq!(l.rows, ["0123456789", ""]);
        assert_eq!(l.position(10), (1, 0));
        // The cursor in the middle of the row stays where it is.
        assert_eq!(l.position(4), (0, 4));
    }

    #[test]
    fn long_text_wraps_into_rows_of_the_box_width() {
        let text: String = "abcdefghij".repeat(3) + "xyz";
        let l = layout(&text, 10);
        assert_eq!(l.rows, ["abcdefghij", "abcdefghij", "abcdefghij", "xyz"]);
        assert_eq!(l.position(text.len()), (3, 3));
        assert_eq!(l.position(15), (1, 5));
    }

    #[test]
    fn newlines_break_rows_and_hold_a_cursor_cell() {
        let l = layout("ab\ncd\n", 10);
        assert_eq!(l.rows, ["ab", "cd", ""]);
        // In front of the first newline = end of line one.
        assert_eq!(l.position(2), (0, 2));
        assert_eq!(l.position(3), (1, 0));
        assert_eq!(l.position(6), (2, 0));
        let crlf = layout("ab\r\ncd", 10);
        assert_eq!(crlf.rows, ["ab", "cd"]);
    }

    #[test]
    fn wide_glyph_that_does_not_fit_wraps_whole() {
        // 9 columns of ASCII leave one free cell; "漢" needs two.
        let l = layout("123456789漢", 10);
        assert_eq!(l.rows, ["123456789", "漢"]);
        assert_eq!(l.position("123456789漢".len()), (1, 2));
    }

    #[test]
    fn tabs_expand_and_wrap_like_spaces() {
        let l = layout("abcdefgh\tz", 10);
        assert_eq!(l.rows, ["abcdefgh", "    z"]);
    }

    #[test]
    fn a_huge_paste_lays_out_without_overflow() {
        let text = "x".repeat(70_000);
        let l = layout(&text, 20);
        assert_eq!(l.row_count(), 3_501);
        assert_eq!(l.position(text.len()), (3_500, 0));
    }

    #[test]
    fn offset_at_clamps_to_the_row_that_was_aimed_at() {
        let l = layout("hello world foo", 6);
        // Rows: "hello ", "world ", "foo".
        assert_eq!(l.rows, ["hello ", "world ", "foo"]);
        assert_eq!(l.offset_at(1, 2), 8);
        // Past the end of a soft-wrapped row: before its last grapheme.
        assert_eq!(l.offset_at(0, 50), 5);
        // Past the end of the last row: the end of the buffer.
        assert_eq!(l.offset_at(2, 50), 15);
        assert_eq!(l.offset_at(9, 0), 15);
        // Past the end of a line ended by a newline: in front of the newline.
        let l = layout("ab\ncdef", 10);
        assert_eq!(l.offset_at(0, 9), 2);
    }

    #[test]
    fn vertical_moves_between_rows_and_stops_at_the_ends() {
        let text = "abcdefghij".repeat(2) + "xyz";
        let l = layout(&text, 10);
        // From row 2 col 3 up to row 1 col 3.
        assert_eq!(l.vertical(23, -1, 3), 13);
        assert_eq!(l.vertical(13, -1, 3), 3);
        assert_eq!(l.vertical(3, -1, 3), 0);
        // Down onto the short last row clamps to its end.
        assert_eq!(l.vertical(18, 1, 8), 23);
        assert_eq!(l.vertical(23, 1, 3), 23);
    }

    #[test]
    fn selected_range_covers_cells_in_reading_order() {
        let l = layout("hello world foo", 6);
        // From "w" on row 1 back to "l" on row 0: dragging up works too.
        assert_eq!(l.selected_range((1, 0), (0, 2)), Some((2, 7)));
        // A selection past the end of a row takes the rest of the row.
        let l = layout("ab\ncd", 10);
        assert_eq!(l.selected_range((0, 1), (1, 0)), Some((1, 4)));
        // Only empty cells past the text: nothing to copy.
        assert_eq!(l.selected_range((0, 5), (0, 9)), None);
        assert_eq!(layout("", 10).selected_range((0, 0), (0, 5)), None);
    }

    #[test]
    fn selecting_half_a_wide_glyph_takes_all_of_it() {
        let text = "a漢b";
        let l = layout(text, 10);
        assert_eq!(l.selected_range((0, 2), (0, 2)), Some((1, 4)));
    }
}
