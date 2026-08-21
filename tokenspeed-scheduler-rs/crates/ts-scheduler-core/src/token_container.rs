// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//! Single-buffer token storage with offset windows.
//!
//! Design principle (ported from `tokenspeed-scheduler/csrc/core/token_container.h`):
//! only one `tokens` member variable; no extra vectors storing tokens; use
//! offsets to determine meaning.

/// A half-open token window `[begin, begin + size)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    pub begin: i32,
    pub size: i32,
}

/// Stores prompt + generated tokens in one buffer and tracks how many of them
/// belong to the prefill window.
#[derive(Debug)]
pub struct TokenContainer {
    tokens: Vec<i32>,
    num_prefill_tokens: i32,
}

impl TokenContainer {
    /// Construct from the initial prompt tokens; the whole input is prefill.
    pub fn new(new_tokens: Vec<i32>) -> Self {
        let num_prefill_tokens = new_tokens.len() as i32;
        Self {
            tokens: new_tokens,
            num_prefill_tokens,
        }
    }

    /// Append generated tokens.
    pub fn extend(&mut self, new_tokens: &[i32]) {
        self.tokens.extend_from_slice(new_tokens);
    }

    /// Retraction folds generated tokens into the prefill window so the
    /// requeued request prefills prompt + generated as one fresh extend.
    pub fn rebase_prefill(&mut self) {
        self.num_prefill_tokens = self.tokens.len() as i32;
    }

    /// Splits the complete prefix into full pages of `prefix_granularity`
    /// tokens. When `except_last` is set, the final token is excluded from the
    /// countable prefix (used for shifted-input alignment).
    pub fn full_prefix_pages(&self, prefix_granularity: i32, except_last: bool) -> Vec<&[i32]> {
        let mut result = Vec::new();
        if self.tokens.is_empty() {
            return result;
        }
        let token_size = if except_last {
            self.tokens.len() - 1
        } else {
            self.tokens.len()
        };
        let grain = prefix_granularity as usize;
        let num_full_pages = token_size / grain;
        result.reserve(num_full_pages);
        for i in 0..num_full_pages {
            let start = i * grain;
            result.push(&self.tokens[start..start + grain]);
        }
        result
    }

    /// Total number of tokens (prompt + generated).
    pub fn size(&self) -> i32 {
        self.tokens.len() as i32
    }

    /// Number of prompt tokens (prefill window).
    pub fn prefill_size(&self) -> i32 {
        self.num_prefill_tokens
    }

    /// Borrow the token slice for `window`. Panics on out-of-range windows
    /// (the C++ original indexed the buffer without bounds checks).
    pub fn token_slice(&self, window: Window) -> &[i32] {
        let begin = window.begin as usize;
        let size = window.size as usize;
        let end = begin
            .checked_add(size)
            .expect("TokenSlice window overflows usize");
        self.tokens.get(begin..end).unwrap_or_else(|| {
            panic!(
                "TokenSlice window out of range: begin={}, size={}",
                window.begin, window.size
            )
        })
    }

    /// Last token in the buffer. Panics on an empty container.
    pub fn last_token(&self) -> i32 {
        *self
            .tokens
            .last()
            .expect("TokenContainer::last_token called on an empty container")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_marks_all_tokens_as_prefill() {
        let tc = TokenContainer::new(vec![1, 2, 3]);
        assert_eq!(tc.size(), 3);
        assert_eq!(tc.prefill_size(), 3);
        assert_eq!(tc.last_token(), 3);
    }

    #[test]
    fn extend_appends_and_preserves_prefill_size() {
        let mut tc = TokenContainer::new(vec![1, 2, 3]);
        tc.extend(&[4, 5]);
        assert_eq!(tc.size(), 5);
        assert_eq!(tc.prefill_size(), 3);
        assert_eq!(tc.last_token(), 5);
    }

    #[test]
    fn rebase_prefill_folds_generated_tokens_in() {
        let mut tc = TokenContainer::new(vec![1, 2]);
        tc.extend(&[3, 4]);
        tc.rebase_prefill();
        assert_eq!(tc.prefill_size(), 4);
    }

    #[test]
    fn full_prefix_pages_splits_by_granularity() {
        let tc = TokenContainer::new(vec![1, 2, 3, 4, 5, 6, 7]);
        let pages = tc.full_prefix_pages(3, false);
        assert_eq!(pages.len(), 2);
        assert_eq!(pages[0], &[1, 2, 3]);
        assert_eq!(pages[1], &[4, 5, 6]);
    }

    #[test]
    fn full_prefix_pages_except_last_drops_final_token() {
        let tc = TokenContainer::new(vec![1, 2, 3, 4, 5, 6]);
        // Countable prefix is [1..=5] => one full page of 3, not two.
        let pages = tc.full_prefix_pages(3, true);
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0], &[1, 2, 3]);
        // Without except_last the same buffer yields two pages.
        let all = tc.full_prefix_pages(3, false);
        assert_eq!(all.len(), 2);
        assert_eq!(all[1], &[4, 5, 6]);
    }

    #[test]
    fn full_prefix_pages_empty_returns_nothing() {
        let tc = TokenContainer::new(vec![]);
        assert!(tc.full_prefix_pages(3, false).is_empty());
        assert!(tc.full_prefix_pages(3, true).is_empty());
    }

    #[test]
    fn token_slice_returns_window() {
        let tc = TokenContainer::new(vec![10, 20, 30, 40]);
        assert_eq!(tc.token_slice(Window { begin: 1, size: 2 }), &[20, 30]);
    }

    #[test]
    #[should_panic(expected = "TokenSlice window out of range")]
    fn token_slice_panics_out_of_range() {
        let tc = TokenContainer::new(vec![1, 2, 3]);
        tc.token_slice(Window { begin: 1, size: 10 });
    }
}
