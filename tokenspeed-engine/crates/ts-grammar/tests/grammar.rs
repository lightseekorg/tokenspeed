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

//! xgrammar through the facade, on a nine-token vocabulary
//! (`fixtures/generate.py`): a, b, {, }, ", x, :, ",", and the stop token.

// The bitmask fills write through raw pointers to local buffers.
#![allow(unsafe_code)]

use ts_grammar::{Compiler, GrammarKind, TokenizerInfo};

const A: i32 = 0;
const B: i32 = 1;
const X: i32 = 5;
const STOP: i32 = 8;

fn compiler() -> Compiler {
    let json = include_str!("fixtures/toy_tokenizer_info.json");
    let info = TokenizerInfo::from_json(json).unwrap();
    assert_eq!(info.vocab_size(), 9);
    Compiler::new(&info, 2).unwrap()
}

/// The token ids row `row` of a bitmask allows.
fn allowed(bitmask: &[i32], width: usize, row: usize) -> Vec<i32> {
    (0..9)
        .filter(|&token| bitmask[row * width + token as usize / 32] >> (token % 32) & 1 == 1)
        .collect()
}

#[test]
fn a_matcher_masks_follows_and_ends_its_grammar() {
    let compiled = compiler()
        .compile(GrammarKind::Regex, "a(b|x)", true)
        .unwrap();
    assert!(compiled.memory_size_bytes() > 0);
    let mut matcher = compiled.matcher(200).unwrap();
    assert_eq!(matcher.stop_token_ids(), [STOP]);
    let mut bitmask = vec![0i32; 2];
    // SAFETY: two rows of one word each.
    unsafe {
        matcher
            .fill_bitmask(bitmask.as_mut_ptr() as usize, 2, 1, 1)
            .unwrap()
    };
    assert_eq!(allowed(&bitmask, 1, 1), [A]);
    assert!(!matcher.accept_token(B));
    assert!(matcher.accept_token(A));
    // SAFETY: as above.
    unsafe {
        matcher
            .fill_bitmask(bitmask.as_mut_ptr() as usize, 2, 1, 0)
            .unwrap()
    };
    assert_eq!(allowed(&bitmask, 1, 0), [B, X]);
    assert!(matcher.accept_token(X));
    assert!(!matcher.is_terminated());
    assert!(matcher.accept_token(STOP));
    assert!(matcher.is_terminated());
    // Rolling back undoes the stop and the x.
    matcher.rollback(2).unwrap();
    assert!(!matcher.is_terminated());
    assert!(matcher.accept_token(B));
}

#[test]
fn a_draft_walk_masks_each_position_and_leaves_the_matcher() {
    let compiled = compiler().compile(GrammarKind::Regex, "abx", true).unwrap();
    let mut matcher = compiled.matcher(200).unwrap();
    assert!(matcher.accept_token(A));
    // A chain of three nodes: the accepted a, then drafts b and a.
    let next = [1i64, 2, -1];
    let sibling = [-1i64, -1, -1];
    let drafts = [i64::from(A), i64::from(B), i64::from(A)];
    let mut bitmask = vec![0i32; 3];
    // SAFETY: one row of one word per node.
    unsafe {
        matcher
            .traverse_draft_tree(&next, &sibling, &drafts, bitmask.as_mut_ptr() as usize, 1)
            .unwrap();
    }
    assert_eq!(allowed(&bitmask, 1, 0), [B]);
    assert_eq!(allowed(&bitmask, 1, 1), [X]);
    // The grammar rejects the second draft: its row is zeroed.
    assert!(allowed(&bitmask, 1, 2).is_empty());
    // The walk did not advance the matcher.
    assert!(matcher.accept_token(B));
}

#[test]
fn compile_failures_carry_xgrammar_s_reason() {
    let compiler = compiler();
    let error = compiler
        .compile(GrammarKind::Regex, "a(b", true)
        .err()
        .unwrap();
    assert!(!error.0.is_empty());
    let schema = compiler
        .compile(GrammarKind::JsonSchema, r#"{"type": "object"}"#, false)
        .unwrap();
    assert!(schema.memory_size_bytes() > 0);
    assert!(compiler
        .compile(GrammarKind::StructuralTag, "not json", true)
        .is_err());
    assert!(TokenizerInfo::from_json("{}").is_err());
}

#[test]
fn the_c_entries_drive_a_registered_matcher() {
    let compiled = compiler().compile(GrammarKind::Regex, "ab", true).unwrap();
    let (handle, shared) = ts_grammar::table::register(compiled.matcher(200).unwrap());
    let entries: std::collections::HashMap<&str, usize> =
        ts_grammar::table::entries().into_iter().collect();
    // SAFETY: the addresses are the entries' own, with these signatures.
    let (is_terminated, accept, retain, free) = unsafe {
        (
            std::mem::transmute::<usize, extern "C" fn(u64) -> i32>(entries["is_terminated"]),
            std::mem::transmute::<usize, extern "C" fn(u64, i32) -> i32>(entries["accept_token"]),
            std::mem::transmute::<usize, extern "C" fn(u64) -> i32>(entries["retain"]),
            std::mem::transmute::<usize, extern "C" fn(u64)>(entries["free"]),
        )
    };
    assert_eq!(is_terminated(handle), 0);
    assert_eq!(accept(handle, B), 0, "b cannot come first");
    assert_eq!(accept(handle, A), 1);
    assert_eq!(accept(handle, B), 1);
    assert_eq!(accept(handle, STOP), 1);
    assert_eq!(is_terminated(handle), 1);
    // The control plane's view is the same matcher.
    assert!(shared.lock().unwrap().is_terminated());
    // A queued forward's hold keeps the handle past the request's release;
    // the last hold forgets it.
    assert_eq!(retain(handle), 0);
    free(handle);
    assert_eq!(is_terminated(handle), 1);
    free(handle);
    assert_eq!(is_terminated(handle), -1);
    assert_eq!(retain(handle), -1);
}
