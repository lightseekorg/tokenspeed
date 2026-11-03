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

// The control plane's face of xgrammar's C++ core: tokenizer info,
// compilation and matchers, with errors as exceptions cxx turns into
// Results.

#pragma once

#include <cstdint>
#include <memory>

#include "rust/cxx.h"
#include "xgrammar/xgrammar.h"

namespace ts_grammar {

using TokenizerInfo = xgrammar::TokenizerInfo;
using Compiler = xgrammar::GrammarCompiler;
using Matcher = xgrammar::GrammarMatcher;

// xgrammar's CompiledGrammar is a shared handle; this owns one.
struct Compiled {
    xgrammar::CompiledGrammar grammar;
};

std::unique_ptr<TokenizerInfo> tokenizer_info_from_json(rust::Str json);
int32_t tokenizer_vocab_size(const TokenizerInfo& info);

std::unique_ptr<Compiler> new_compiler(const TokenizerInfo& info, int32_t max_threads);
std::unique_ptr<Compiled> compile_json_schema(const Compiler& compiler, rust::Str schema,
                                              bool any_whitespace);
std::unique_ptr<Compiled> compile_regex(const Compiler& compiler, rust::Str regex);
std::unique_ptr<Compiled> compile_ebnf(const Compiler& compiler, rust::Str ebnf);
std::unique_ptr<Compiled> compile_structural_tag(const Compiler& compiler, rust::Str json);
size_t compiled_memory_size(const Compiled& compiled);

std::unique_ptr<Matcher> new_matcher(const Compiled& compiled, int32_t max_rollback_tokens);
bool matcher_accept_token(Matcher& matcher, int32_t token);
void matcher_fill_bitmask(Matcher& matcher, size_t data, int64_t rows, int64_t width,
                          int32_t row);
void matcher_traverse_draft_tree(Matcher& matcher, rust::Slice<const int64_t> next_node,
                                 rust::Slice<const int64_t> next_sibling,
                                 rust::Slice<const int64_t> draft_tokens, size_t bitmask,
                                 int64_t width);
void matcher_rollback(Matcher& matcher, int32_t num_tokens);
bool matcher_is_terminated(const Matcher& matcher);
rust::Vec<int32_t> matcher_stop_token_ids(const Matcher& matcher);

}  // namespace ts_grammar
