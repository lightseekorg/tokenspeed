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

//! Protocol tokens kept out of a framed JSON answer's strings.
//!
//! A model whose messages are framed by protocol tokens (Inkling2's
//! `<|content_text|>` ... `<|end_message|>`) may spell a protocol name as
//! ordinary text inside a JSON string, but must not emit the protocol token
//! itself there: the frontend would read it as framing. xgrammar matches
//! bytes, and a protocol token's bytes are valid string text, so the guard
//! tracks whether the answer is inside a JSON string and, there, rejects the
//! protocol tokens and clears them from every bitmask.

use std::pin::Pin;

use crate::{ffi, Error, Matcher};

/// The vocabulary's facts for the guard, from the data plane at startup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonStringGuard {
    /// The token that opens an answer's payload.
    text_token: i32,
    /// The token that closes a message.
    end_token: i32,
    /// Protocol tokens that cannot stand for JSON string text, ascending.
    reserved: Vec<i32>,
    /// `(word, mask)`: ANDing bitmask word `word` with `mask` clears the
    /// reserved tokens in it.
    clear_masks: Vec<(usize, i32)>,
    /// Per token, the string state after its bytes from each start state:
    /// bits `2 * s` and `2 * s + 1` hold the state from state `s`
    /// (`StringState::index`).
    string_steps: Vec<u8>,
}

/// Where the answer is: in its payload, in a JSON string there, and right
/// after a backslash in that string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StringState {
    inside: bool,
    quoted: bool,
    escaped: bool,
}

impl StringState {
    /// The index of `(quoted, escaped)` in a token's string steps.
    fn index(self) -> usize {
        (usize::from(self.quoted) << 1) | usize::from(self.escaped)
    }
}

/// The string state of `bytes` after starting in `(quoted, escaped)`, packed
/// as `StringState::index` does.
fn step(bytes: &[u8], mut quoted: bool, mut escaped: bool) -> u8 {
    for &byte in bytes {
        if escaped {
            escaped = false;
        } else if quoted && byte == b'\\' {
            escaped = true;
        } else if byte == b'"' {
            quoted = !quoted;
        }
    }
    (u8::from(quoted) << 1) | u8::from(escaped)
}

/// A token's string steps from the four start states.
pub fn string_steps_of(bytes: &[u8]) -> u8 {
    let mut packed = 0;
    for start in 0..4u8 {
        packed |= step(bytes, start & 2 != 0, start & 1 != 0) << (2 * start);
    }
    packed
}

impl JsonStringGuard {
    /// A guard for a vocabulary of `string_steps.len()` tokens.
    ///
    /// `text_token` opens a payload, `end_token` closes a message,
    /// `reserved_tokens` may not stand inside a JSON string, and
    /// `string_steps` holds each token's [`string_steps_of`] its bytes.
    pub fn new(
        text_token: i32,
        end_token: i32,
        reserved_tokens: &[i32],
        string_steps: Vec<u8>,
    ) -> Result<Self, Error> {
        let vocab = string_steps.len();
        let in_vocab = |token: i32| usize::try_from(token).is_ok_and(|token| token < vocab);
        if !in_vocab(text_token) || !in_vocab(end_token) {
            return Err(Error(
                "the framing tokens are outside the vocabulary".into(),
            ));
        }
        let mut reserved = reserved_tokens.to_vec();
        if !reserved.iter().all(|&token| in_vocab(token)) {
            return Err(Error("a reserved token is outside the vocabulary".into()));
        }
        reserved.sort_unstable();
        reserved.dedup();
        let mut clear_masks: Vec<(usize, i32)> = Vec::new();
        for &token in &reserved {
            let word = token as usize / 32;
            let bit = 1i32 << (token % 32);
            match clear_masks.last_mut() {
                Some((last, mask)) if *last == word => *mask &= !bit,
                _ => clear_masks.push((word, !bit)),
            }
        }
        Ok(Self {
            text_token,
            end_token,
            reserved,
            clear_masks,
            string_steps,
        })
    }

    /// The number of tokens the guard covers.
    pub fn vocab_size(&self) -> usize {
        self.string_steps.len()
    }

    fn is_reserved(&self, token: i32) -> bool {
        self.reserved.binary_search(&token).is_ok()
    }

    fn after(&self, state: StringState, token: i32) -> StringState {
        if token == self.text_token {
            return StringState {
                inside: true,
                quoted: false,
                escaped: false,
            };
        }
        if token == self.end_token {
            return StringState {
                inside: false,
                quoted: false,
                escaped: false,
            };
        }
        if !state.inside {
            return state;
        }
        let next = (self.string_steps[token as usize] >> (2 * state.index())) & 3;
        StringState {
            inside: true,
            quoted: next & 2 != 0,
            escaped: next & 1 != 0,
        }
    }
}

/// One matcher's guard: the string state after each accepted token, the
/// first the state the answer starts in.
#[derive(Debug, Clone)]
pub(crate) struct Guard {
    spec: std::sync::Arc<JsonStringGuard>,
    states: Vec<StringState>,
}

impl Guard {
    /// A guard at the grammar's start; `starts_in_json` when the prompt
    /// already opened the payload.
    pub(crate) fn new(spec: std::sync::Arc<JsonStringGuard>, starts_in_json: bool) -> Self {
        Self {
            spec,
            states: vec![StringState {
                inside: starts_in_json,
                quoted: false,
                escaped: false,
            }],
        }
    }

    fn current(&self) -> StringState {
        *self.states.last().expect("the start state stays")
    }

    /// `token` is a protocol token inside a JSON string.
    pub(crate) fn rejects(&self, token: i32) -> bool {
        self.current().quoted && self.spec.is_reserved(token)
    }

    /// Record an accepted token.
    pub(crate) fn accepted(&mut self, token: i32) {
        let next = self.spec.after(self.current(), token);
        self.states.push(next);
    }

    /// Whether the last `num_tokens` accepted tokens can be undone.
    pub(crate) fn can_roll_back(&self, num_tokens: usize) -> bool {
        num_tokens < self.states.len()
    }

    /// Undo the last `num_tokens` accepted tokens (`can_roll_back`).
    pub(crate) fn roll_back(&mut self, num_tokens: usize) {
        self.states.truncate(self.states.len() - num_tokens);
    }

    /// Clear the reserved tokens from row `row` of a `[rows, width]` int32
    /// bitmask inside a JSON string.
    ///
    /// # Safety
    ///
    /// `data` must point to `rows * width` writable int32 values and `row`
    /// must be below `rows`.
    pub(crate) unsafe fn clear_reserved(&self, data: usize, width: i64, row: i64) {
        if !self.current().quoted {
            return;
        }
        let row_start = data as *mut i32;
        for &(word, mask) in &self.spec.clear_masks {
            if (word as i64) < width {
                // SAFETY: the word is inside row `row` of the caller's bitmask.
                unsafe { *row_start.add((row * width) as usize + word) &= mask };
            }
        }
    }

    /// [`Matcher::accept_token`] behind the guard.
    pub(crate) fn accept_token(&mut self, matcher: Pin<&mut ffi::Matcher>, token: i32) -> bool {
        if self.rejects(token) {
            return false;
        }
        if !ffi::matcher_accept_token(matcher, token) {
            return false;
        }
        self.accepted(token);
        true
    }

    /// [`Matcher::fill_bitmask`] behind the guard.
    ///
    /// # Safety
    ///
    /// `data` must point to `rows * width` writable int32 values.
    pub(crate) unsafe fn fill_bitmask(
        &self,
        matcher: Pin<&mut ffi::Matcher>,
        data: usize,
        rows: i64,
        width: i64,
        row: i32,
    ) -> Result<(), Error> {
        ffi::matcher_fill_bitmask(matcher, data, rows, width, row)?;
        // SAFETY: xgrammar just filled row `row` of this bitmask.
        unsafe { self.clear_reserved(data, width, i64::from(row)) };
        Ok(())
    }

    /// [`Matcher::rollback`] behind the guard.
    pub(crate) fn rollback(
        &mut self,
        matcher: Pin<&mut ffi::Matcher>,
        num_tokens: i32,
    ) -> Result<(), Error> {
        let count = usize::try_from(num_tokens).map_err(|_| Error("a negative rollback".into()))?;
        if count == 0 {
            return Ok(());
        }
        if !self.can_roll_back(count) {
            return Err(Error("cannot roll back beyond the accepted tokens".into()));
        }
        ffi::matcher_rollback(matcher, num_tokens)?;
        self.roll_back(num_tokens as usize);
        Ok(())
    }
}

/// [`Matcher::traverse_draft_tree`] for a guarded matcher.
///
/// # Safety
///
/// `bitmask` must point to `draft_tokens.len() * width` writable int32
/// values.
pub(crate) unsafe fn traverse_draft_tree(
    matcher: &mut Matcher,
    next_node: &[i64],
    next_sibling: &[i64],
    draft_tokens: &[i64],
    bitmask: usize,
    width: i64,
) -> Result<(), Error> {
    let nodes = draft_tokens.len();
    if next_node.len() != nodes || next_sibling.len() != nodes {
        return Err(Error("the draft tree's columns differ in length".into()));
    }
    if nodes == 0 {
        return Ok(());
    }
    // xgrammar's walk cannot consult the guard, so walk the tree through the guarded methods.
    let rows = nodes as i64;
    let zero = |node: usize| {
        // SAFETY: row `node` of the caller's `[nodes, width]` bitmask.
        unsafe {
            std::ptr::write_bytes(
                (bitmask as *mut i32).add(node * width as usize),
                0,
                width as usize,
            )
        };
    };
    if matcher.is_terminated() {
        zero(0);
        return Ok(());
    }
    // SAFETY: as above.
    unsafe { matcher.fill_bitmask(bitmask, rows, width, 0)? };
    let mut child = next_node[0];
    while child >= 0 {
        visit_draft(
            matcher,
            child as usize,
            next_node,
            next_sibling,
            draft_tokens,
            bitmask,
            width,
            &zero,
        )?;
        child = next_sibling[child as usize];
    }
    Ok(())
}

/// Mask the draft subtree at `node` behind the matcher's state, which
/// is as it was once the walk succeeds.
#[allow(clippy::too_many_arguments)]
fn visit_draft(
    matcher: &mut Matcher,
    node: usize,
    next_node: &[i64],
    next_sibling: &[i64],
    draft_tokens: &[i64],
    bitmask: usize,
    width: i64,
    zero: &dyn Fn(usize),
) -> Result<(), Error> {
    let Ok(token) = i32::try_from(draft_tokens[node]) else {
        zero(node);
        return Ok(());
    };
    if !matcher.accept_token(token) {
        zero(node);
        return Ok(());
    }
    if matcher.is_terminated() {
        zero(node);
    } else {
        // SAFETY: row `node` of the caller's bitmask.
        unsafe { matcher.fill_bitmask(bitmask, draft_tokens.len() as i64, width, node as i32)? };
        let mut child = next_node[node];
        while child >= 0 {
            visit_draft(
                matcher,
                child as usize,
                next_node,
                next_sibling,
                draft_tokens,
                bitmask,
                width,
                zero,
            )?;
            child = next_sibling[child as usize];
        }
    }
    matcher.rollback(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_steps_follow_quotes_and_escapes() {
        // (quoted, escaped) packed as 2 * quoted + escaped.
        let from = |bytes: &[u8], start: u8| (string_steps_of(bytes) >> (2 * start)) & 3;
        assert_eq!(from(b"ab", 0), 0);
        assert_eq!(from(b"a\"", 0), 2);
        assert_eq!(from(b"\"a\"", 0), 0);
        // A backslash escapes only inside a string, and only the next byte.
        assert_eq!(from(b"\\", 0), 0);
        assert_eq!(from(b"\\", 2), 3);
        assert_eq!(from(b"\\\"", 2), 2);
        assert_eq!(from(b"\"", 3), 2);
        assert_eq!(from(b"x\"", 3), 0);
        // An empty token keeps the state.
        assert_eq!(from(b"", 3), 3);
    }

    #[test]
    fn a_guard_rejects_protocol_tokens_only_inside_strings() {
        // Tokens: 0 `"`, 1 `\`, 2 a, 3 text, 4 end, 5 other protocol.
        let steps = [&b"\""[..], b"\\", b"a", b"", b"", b""]
            .iter()
            .map(|bytes| string_steps_of(bytes))
            .collect();
        let spec = std::sync::Arc::new(JsonStringGuard::new(3, 4, &[5, 3, 5], steps).unwrap());
        let mut guard = Guard::new(spec, false);
        // Outside the payload a quote opens no string.
        guard.accepted(0);
        assert!(!guard.rejects(5));
        guard.accepted(3);
        guard.accepted(0);
        assert!(guard.rejects(5));
        assert!(guard.rejects(3));
        assert!(!guard.rejects(4));
        // An escaped quote keeps the string open; a plain one closes it.
        guard.accepted(1);
        guard.accepted(0);
        assert!(guard.rejects(5));
        guard.accepted(0);
        assert!(!guard.rejects(5));
        assert!(guard.can_roll_back(6));
        assert!(!guard.can_roll_back(7));
        guard.roll_back(2);
        assert!(guard.rejects(5));
    }

    #[test]
    fn clearing_masks_only_the_reserved_bits_inside_a_string() {
        let spec = std::sync::Arc::new(
            JsonStringGuard::new(0, 1, &[2, 33, 63], vec![string_steps_of(b"\""); 64]).unwrap(),
        );
        let mut guard = Guard::new(spec, true);
        let mut bitmask = vec![-1i32; 4];
        // SAFETY: two rows of two words.
        unsafe { guard.clear_reserved(bitmask.as_mut_ptr() as usize, 2, 1) };
        assert_eq!(bitmask, [-1; 4]);
        guard.accepted(5);
        // SAFETY: as above.
        unsafe { guard.clear_reserved(bitmask.as_mut_ptr() as usize, 2, 1) };
        assert_eq!(bitmask[..2], [-1, -1]);
        assert_eq!(bitmask[2], !(1 << 2));
        assert_eq!(bitmask[3], !(1 << 1) & !(1 << 31));
    }

    #[test]
    fn tokens_outside_the_vocabulary_are_refused() {
        assert!(JsonStringGuard::new(0, 9, &[], vec![0; 4]).is_err());
        assert!(JsonStringGuard::new(0, 1, &[-1], vec![0; 4]).is_err());
    }
}
