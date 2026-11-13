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

//! A framed JSON answer's matchers on the framed vocabulary
//! (`fixtures/generate.py`): protocol tokens stay out of its JSON strings.

// The bitmask fills write through raw pointers to local buffers.
#![allow(unsafe_code)]

use std::sync::Arc;

use ts_grammar::{
    string_steps_of, CompiledGrammar, Compiler, GrammarKind, JsonStringGuard, Matcher,
    TokenizerInfo,
};

const A: i32 = 0;
const B: i32 = 1;
const CLOSE: i32 = 3;
const QUOTE: i32 = 4;
const COLON: i32 = 6;
const BACKSLASH: i32 = 8;
const MODEL: i32 = 9;
const TEXT: i32 = 10;
const THINKING: i32 = 11;
const END: i32 = 12;
const TOOL: i32 = 13;
const STOP: i32 = 14;
const VOCAB: usize = 15;
const OPEN: i32 = 2;

const DECODED: [&[u8]; VOCAB] = [
    b"a",
    b"b",
    b"{",
    b"}",
    b"\"",
    b"x",
    b":",
    b",",
    b"\\",
    b"<|message_model|>",
    b"<|content_text|>",
    b"<|content_thinking|>",
    b"<|end_message|>",
    b"<|message_tool|>",
    b"</s>",
];

/// `{"a": <string>}`, closed by its message terminator.
const SCHEMA: &str = r#"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}"#;

fn compiler() -> Compiler {
    let json = include_str!("fixtures/framed_tokenizer_info.json");
    let info = TokenizerInfo::from_json(json).unwrap();
    assert_eq!(info.vocab_size(), VOCAB as i32);
    Compiler::new(&info, 2).unwrap()
}

fn guard() -> Arc<JsonStringGuard> {
    let steps = DECODED.iter().map(|bytes| string_steps_of(bytes)).collect();
    Arc::new(JsonStringGuard::new(TEXT, END, &[MODEL, TEXT, THINKING, END, TOOL], steps).unwrap())
}

/// The answer as a structural tag whose header (none when the prompt opened
/// it) and terminator are protocol tokens.
fn answer(with_header: bool) -> String {
    let header = if with_header {
        format!(r#"{{"type":"token","token":{MODEL}}},{{"type":"token","token":{TEXT}}},"#)
    } else {
        String::new()
    };
    format!(
        r#"{{"type":"structural_tag","format":{{"type":"sequence","elements":[{header}{{"type":"json_schema","json_schema":{SCHEMA},"style":"json","any_order":false}},{{"type":"token","token":{END}}}]}}}}"#
    )
}

/// The answer with its matchers keeping `guard()`.
fn guarded(with_header: bool, starts_in_json: bool) -> CompiledGrammar {
    compiler()
        .compile_framed_answer(&answer(with_header), starts_in_json, guard())
        .unwrap()
}

fn accept_all(matcher: &mut Matcher, tokens: &[i32]) {
    for &token in tokens {
        assert!(matcher.accept_token(token), "token {token}");
    }
}

/// The token ids row `row` of a one-word-wide bitmask allows.
fn allowed(bitmask: &[i32], row: usize) -> Vec<i32> {
    (0..VOCAB as i32)
        .filter(|&token| bitmask[row] >> token & 1 == 1)
        .collect()
}

fn mask(matcher: &mut Matcher) -> Vec<i32> {
    let mut bitmask = vec![0i32; 1];
    // SAFETY: one row of one word.
    unsafe {
        matcher
            .fill_bitmask(bitmask.as_mut_ptr() as usize, 1, 1, 0)
            .unwrap()
    };
    allowed(&bitmask, 0)
}

/// Into the answer's string value: `{"a":"`.
const INTO_VALUE: [i32; 8] = [MODEL, TEXT, OPEN, QUOTE, A, QUOTE, COLON, QUOTE];

#[test]
fn protocol_tokens_are_text_to_xgrammar_but_not_to_the_guard() {
    // Without the guard a protocol token is ordinary string text.
    let mut plain = compiler()
        .compile(GrammarKind::StructuralTag, &answer(true), false)
        .unwrap()
        .matcher(200)
        .unwrap();
    accept_all(&mut plain, &INTO_VALUE);
    assert!(mask(&mut plain).contains(&TOOL));
    assert!(plain.accept_token(TOOL));

    let mut guarded = guarded(true, false).matcher(200).unwrap();
    accept_all(&mut guarded, &INTO_VALUE);
    let inside = mask(&mut guarded);
    assert!(inside.contains(&A) && inside.contains(&BACKSLASH));
    for reserved in [MODEL, TEXT, THINKING, END, TOOL] {
        assert!(!inside.contains(&reserved), "{reserved}");
        assert!(!guarded.accept_token(reserved));
    }
    // An escaped quote stays in the string; a plain one closes it.
    accept_all(&mut guarded, &[B, BACKSLASH, QUOTE]);
    assert!(!guarded.accept_token(TOOL));
    accept_all(&mut guarded, &[QUOTE, CLOSE, END, STOP]);
    assert!(guarded.is_terminated());
}

#[test]
fn rolling_back_restores_the_string_state() {
    let mut matcher = guarded(true, false).matcher(200).unwrap();
    accept_all(&mut matcher, &INTO_VALUE);
    accept_all(&mut matcher, &[BACKSLASH]);
    // Undo the backslash: the next quote closes the string.
    matcher.rollback(1).unwrap();
    accept_all(&mut matcher, &[QUOTE, CLOSE]);
    // Undo the close and the quote: back inside the string.
    matcher.rollback(2).unwrap();
    assert!(!matcher.accept_token(TOOL));
    assert!(matcher.rollback(INTO_VALUE.len() as i32 + 1).is_err());
    assert!(matcher.accept_token(A));
}

#[test]
fn a_draft_walk_stops_at_a_guarded_token_and_leaves_the_matcher() {
    let mut matcher = guarded(true, false).matcher(200).unwrap();
    accept_all(&mut matcher, &INTO_VALUE);
    // Root -> a -> (protocol token -> b, closing quote).
    let next = [1i64, 2, 3, -1, -1];
    let sibling = [-1i64, -1, 4, -1, -1];
    let drafts = [0i64, A as i64, TOOL as i64, B as i64, QUOTE as i64];
    let mut bitmask = vec![-1i32; 5];
    // SAFETY: one row of one word per node.
    unsafe {
        matcher
            .traverse_draft_tree(&next, &sibling, &drafts, bitmask.as_mut_ptr() as usize, 1)
            .unwrap();
    }
    for row in [0, 1] {
        assert!(!allowed(&bitmask, row).contains(&TOOL), "row {row}");
        assert!(allowed(&bitmask, row).contains(&A), "row {row}");
    }
    // The guard rejects the protocol token: its row is empty, its subtree's untouched.
    assert!(allowed(&bitmask, 2).is_empty());
    assert_eq!(bitmask[3], -1);
    // Beside it, the closing quote ends the string: the object may close.
    assert!(allowed(&bitmask, 4).contains(&CLOSE));
    // The walk left the matcher inside the string.
    assert!(!matcher.accept_token(TOOL));
    assert!(matcher.accept_token(A));
}

#[test]
fn an_answer_the_prompt_opened_starts_in_its_payload() {
    let mut matcher = guarded(false, true).matcher(200).unwrap();
    accept_all(&mut matcher, &[OPEN, QUOTE]);
    assert!(!matcher.accept_token(TOOL));
    assert!(!mask(&mut matcher).contains(&TOOL));
    accept_all(
        &mut matcher,
        &[A, QUOTE, COLON, QUOTE, B, QUOTE, CLOSE, END, STOP],
    );
    assert!(matcher.is_terminated());
}

#[test]
fn a_guard_over_another_vocabulary_is_refused() {
    for vocab in [VOCAB - 1, VOCAB + 1] {
        let steps = vec![string_steps_of(b"a"); vocab];
        let other = Arc::new(JsonStringGuard::new(TEXT, END, &[TEXT, END], steps).unwrap());
        let Err(error) = compiler().compile_framed_answer(&answer(true), false, other) else {
            panic!("a guard over {vocab} tokens pairs with a {VOCAB}-token grammar");
        };
        assert!(error.0.contains("tokenizer info"), "{error}");
    }
}
