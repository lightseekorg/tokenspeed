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

//! Constrained decoding on xgrammar's C++ core (RFC 0002, "Grammar"): the
//! tokenizer facts the data plane derives from the HF tokenizer, grammar
//! compilation, and the per-request matchers the control plane owns.
//!
//! xgrammar is built from the source release of the version the Python
//! package pins (`CMakeLists.txt`); its wheels ship no C++ library.

#![allow(unsafe_code)]

pub mod table;

use std::fmt;

use cxx::UniquePtr;

#[cxx::bridge(namespace = "ts_grammar")]
mod ffi {
    unsafe extern "C++" {
        include!("ts-grammar/cxx/facade.h");

        type TokenizerInfo;
        type Compiler;
        type Compiled;
        type Matcher;

        fn tokenizer_info_from_json(json: &str) -> Result<UniquePtr<TokenizerInfo>>;
        fn tokenizer_vocab_size(info: &TokenizerInfo) -> i32;

        fn new_compiler(info: &TokenizerInfo, max_threads: i32) -> Result<UniquePtr<Compiler>>;
        fn compile_json_schema(
            compiler: &Compiler,
            schema: &str,
            any_whitespace: bool,
        ) -> Result<UniquePtr<Compiled>>;
        fn compile_regex(compiler: &Compiler, regex: &str) -> Result<UniquePtr<Compiled>>;
        fn compile_ebnf(compiler: &Compiler, ebnf: &str) -> Result<UniquePtr<Compiled>>;
        fn compile_structural_tag(compiler: &Compiler, json: &str) -> Result<UniquePtr<Compiled>>;
        fn compiled_memory_size(compiled: &Compiled) -> usize;

        fn new_matcher(compiled: &Compiled, max_rollback_tokens: i32)
            -> Result<UniquePtr<Matcher>>;
        fn matcher_accept_token(matcher: Pin<&mut Matcher>, token: i32) -> bool;
        fn matcher_fill_bitmask(
            matcher: Pin<&mut Matcher>,
            data: usize,
            rows: i64,
            width: i64,
            row: i32,
        ) -> Result<()>;
        fn matcher_traverse_draft_tree(
            matcher: Pin<&mut Matcher>,
            next_node: &[i64],
            next_sibling: &[i64],
            draft_tokens: &[i64],
            bitmask: usize,
            width: i64,
        ) -> Result<()>;
        fn matcher_rollback(matcher: Pin<&mut Matcher>, num_tokens: i32) -> Result<()>;
        fn matcher_is_terminated(matcher: &Matcher) -> bool;
        fn matcher_stop_token_ids(matcher: &Matcher) -> Vec<i32>;
    }
}

/// An xgrammar failure: a grammar that does not compile, or a bad call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<cxx::Exception> for Error {
    fn from(error: cxx::Exception) -> Self {
        Self(error.what().to_owned())
    }
}

/// The vocabulary as xgrammar sees it, from the JSON the data plane
/// serializes (`TokenizerInfo.serialize_json`).
pub struct TokenizerInfo(UniquePtr<ffi::TokenizerInfo>);

// SAFETY: a tokenizer info is immutable once built.
unsafe impl Send for TokenizerInfo {}
// SAFETY: as above.
unsafe impl Sync for TokenizerInfo {}

impl TokenizerInfo {
    /// Parse serialized tokenizer info.
    pub fn from_json(json: &str) -> Result<Self, Error> {
        Ok(Self(ffi::tokenizer_info_from_json(json)?))
    }

    /// Token ids in the vocabulary, the width a bitmask covers.
    pub fn vocab_size(&self) -> i32 {
        ffi::tokenizer_vocab_size(&self.0)
    }
}

/// What a request constrains its output to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GrammarKind {
    /// A JSON schema (`json_schema`).
    JsonSchema,
    /// A regular expression (`regex`).
    Regex,
    /// An EBNF grammar (`ebnf`).
    Ebnf,
    /// A structural tag in xgrammar's JSON form (`structural_tag`).
    StructuralTag,
}

/// Compiles grammars against one vocabulary. Compiles may run on several
/// threads at once: xgrammar's compiler guards its own state.
pub struct Compiler(UniquePtr<ffi::Compiler>);

// SAFETY: see the type's documentation.
unsafe impl Send for Compiler {}
// SAFETY: as above.
unsafe impl Sync for Compiler {}

impl Compiler {
    /// A compiler using up to `max_threads` threads per compile, with
    /// xgrammar's own cache off.
    pub fn new(info: &TokenizerInfo, max_threads: i32) -> Result<Self, Error> {
        Ok(Self(ffi::new_compiler(&info.0, max_threads)?))
    }

    /// Compile one grammar. `any_whitespace` applies to JSON schemas.
    pub fn compile(
        &self,
        kind: GrammarKind,
        source: &str,
        any_whitespace: bool,
    ) -> Result<CompiledGrammar, Error> {
        let compiled = match kind {
            GrammarKind::JsonSchema => ffi::compile_json_schema(&self.0, source, any_whitespace),
            GrammarKind::Regex => ffi::compile_regex(&self.0, source),
            GrammarKind::Ebnf => ffi::compile_ebnf(&self.0, source),
            GrammarKind::StructuralTag => ffi::compile_structural_tag(&self.0, source),
        }?;
        Ok(CompiledGrammar(compiled))
    }
}

/// A compiled grammar; matchers share it.
pub struct CompiledGrammar(UniquePtr<ffi::Compiled>);

// SAFETY: a compiled grammar is immutable; matchers only read it.
unsafe impl Send for CompiledGrammar {}
// SAFETY: as above.
unsafe impl Sync for CompiledGrammar {}

impl CompiledGrammar {
    /// xgrammar's estimate of the memory it holds.
    pub fn memory_size_bytes(&self) -> usize {
        ffi::compiled_memory_size(&self.0)
    }

    /// A fresh matcher at the grammar's start.
    pub fn matcher(&self, max_rollback_tokens: i32) -> Result<Matcher, Error> {
        Ok(Matcher(ffi::new_matcher(&self.0, max_rollback_tokens)?))
    }
}

/// One request's progress through its grammar. Not thread-safe: callers
/// serialize access.
pub struct Matcher(UniquePtr<ffi::Matcher>);

// SAFETY: a matcher has no thread affinity; access is serialized by its
// owner.
unsafe impl Send for Matcher {}

impl Matcher {
    /// Advance by one token; `false` if the grammar rejects it, leaving
    /// the matcher as it was.
    pub fn accept_token(&mut self, token: i32) -> bool {
        ffi::matcher_accept_token(self.0.pin_mut(), token)
    }

    /// Write the tokens allowed next into row `row` of a `[rows, width]`
    /// int32 bitmask at `data`.
    ///
    /// # Safety
    ///
    /// `data` must point to `rows * width` writable int32 values.
    pub unsafe fn fill_bitmask(
        &mut self,
        data: usize,
        rows: i64,
        width: i64,
        row: i32,
    ) -> Result<(), Error> {
        Ok(ffi::matcher_fill_bitmask(
            self.0.pin_mut(),
            data,
            rows,
            width,
            row,
        )?)
    }

    /// Fill one bitmask row per draft-tree node: row `k` holds the tokens
    /// allowed after the drafts on the path to node `k`; the walk stops at
    /// a rejected or ending draft. The matcher is left as it was.
    ///
    /// # Safety
    ///
    /// `bitmask` must point to `draft_tokens.len() * width` writable
    /// int32 values.
    pub unsafe fn traverse_draft_tree(
        &mut self,
        next_node: &[i64],
        next_sibling: &[i64],
        draft_tokens: &[i64],
        bitmask: usize,
        width: i64,
    ) -> Result<(), Error> {
        Ok(ffi::matcher_traverse_draft_tree(
            self.0.pin_mut(),
            next_node,
            next_sibling,
            draft_tokens,
            bitmask,
            width,
        )?)
    }

    /// Undo the last `num_tokens` accepted tokens.
    pub fn rollback(&mut self, num_tokens: i32) -> Result<(), Error> {
        Ok(ffi::matcher_rollback(self.0.pin_mut(), num_tokens)?)
    }

    /// The grammar accepted its stop token.
    pub fn is_terminated(&self) -> bool {
        ffi::matcher_is_terminated(&self.0)
    }

    /// The tokens that end the grammar.
    pub fn stop_token_ids(&self) -> Vec<i32> {
        ffi::matcher_stop_token_ids(&self.0)
    }
}
