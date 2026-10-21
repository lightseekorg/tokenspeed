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

//! The process's matchers, by handle, and the C entries the data plane
//! drives them through (RFC 0002, "Grammar").
//!
//! The control plane creates a request's matcher at admission and reads its
//! state at commit; the data plane fills token bitmasks from it and, on the
//! capturable path, advances it inside the forward. Both reach the same
//! matcher, serialized by its lock. The data plane gets the entries'
//! addresses (`entries`) and calls them from Python; it frees a handle once
//! no queued forward can still use it.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    },
};

use crate::Matcher;

/// A matcher both planes share.
pub type SharedMatcher = Arc<Mutex<Matcher>>;

fn table() -> &'static Mutex<HashMap<u64, SharedMatcher>> {
    static TABLE: OnceLock<Mutex<HashMap<u64, SharedMatcher>>> = OnceLock::new();
    TABLE.get_or_init(Mutex::default)
}

/// Register `matcher`; its handle and the shared matcher.
pub fn register(matcher: Matcher) -> (u64, SharedMatcher) {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let handle = NEXT.fetch_add(1, Ordering::Relaxed);
    let shared = Arc::new(Mutex::new(matcher));
    table()
        .lock()
        .expect("grammar table lock")
        .insert(handle, shared.clone());
    (handle, shared)
}

fn matcher(handle: u64) -> Option<SharedMatcher> {
    table()
        .lock()
        .expect("grammar table lock")
        .get(&handle)
        .cloned()
}

fn with<R>(handle: u64, failed: R, run: impl FnOnce(&mut Matcher) -> R) -> R {
    match matcher(handle) {
        Some(shared) => run(&mut shared.lock().expect("matcher lock")),
        None => failed,
    }
}

/// Whether the grammar accepted its stop token: 1 yes, 0 no, -1 an unknown
/// handle.
extern "C" fn ts_grammar_is_terminated(handle: u64) -> i32 {
    with(handle, -1, |m| i32::from(m.is_terminated()))
}

/// Advance by `token`: 1 accepted, 0 rejected (the matcher unchanged), -1 an
/// unknown handle.
extern "C" fn ts_grammar_accept_token(handle: u64, token: i32) -> i32 {
    with(handle, -1, |m| i32::from(m.accept_token(token)))
}

/// Undo the last `num_tokens` tokens: 0 done, -1 failed.
extern "C" fn ts_grammar_rollback(handle: u64, num_tokens: i32) -> i32 {
    with(handle, -1, |m| {
        if m.rollback(num_tokens).is_ok() {
            0
        } else {
            -1
        }
    })
}

/// Fill row `row` of the `[rows, width]` int32 bitmask at `data`: 0 done,
/// -1 failed.
extern "C" fn ts_grammar_fill_bitmask(
    handle: u64,
    data: usize,
    rows: i64,
    width: i64,
    row: i32,
) -> i32 {
    // SAFETY: the data plane passes a live tensor of that shape.
    with(handle, -1, |m| {
        match unsafe { m.fill_bitmask(data, rows, width, row) } {
            Ok(()) => 0,
            Err(_) => -1,
        }
    })
}

/// Fill one bitmask row per draft-tree node (`traverse_draft_tree`), the
/// tree's columns `num_nodes` int64 each at the given addresses: 0 done, -1
/// failed.
extern "C" fn ts_grammar_traverse_draft_tree(
    handle: u64,
    next_node: usize,
    next_sibling: usize,
    draft_tokens: usize,
    num_nodes: i64,
    bitmask: usize,
    width: i64,
) -> i32 {
    let Ok(nodes) = usize::try_from(num_nodes) else {
        return -1;
    };
    // SAFETY: the data plane passes live int64 tensors of `num_nodes`
    // values and a bitmask of `num_nodes * width` int32 values.
    let column =
        |address: usize| unsafe { std::slice::from_raw_parts(address as *const i64, nodes) };
    with(handle, -1, |m| {
        match unsafe {
            m.traverse_draft_tree(
                column(next_node),
                column(next_sibling),
                column(draft_tokens),
                bitmask,
                width,
            )
        } {
            Ok(()) => 0,
            Err(_) => -1,
        }
    })
}

/// Forget a handle; its matcher lives on while the control plane holds it.
extern "C" fn ts_grammar_free(handle: u64) {
    table().lock().expect("grammar table lock").remove(&handle);
}

/// The C entries, by name, for the data plane to call (`HostGrammar`).
pub fn entries() -> Vec<(&'static str, usize)> {
    vec![
        (
            "is_terminated",
            ts_grammar_is_terminated as *const () as usize,
        ),
        (
            "accept_token",
            ts_grammar_accept_token as *const () as usize,
        ),
        ("rollback", ts_grammar_rollback as *const () as usize),
        (
            "fill_bitmask",
            ts_grammar_fill_bitmask as *const () as usize,
        ),
        (
            "traverse_draft_tree",
            ts_grammar_traverse_draft_tree as *const () as usize,
        ),
        ("free", ts_grammar_free as *const () as usize),
    ]
}
