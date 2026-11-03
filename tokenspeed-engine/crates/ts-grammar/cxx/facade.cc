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

#include "ts-grammar/cxx/facade.h"

#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "dlpack/dlpack.h"

namespace ts_grammar {

namespace {

std::string to_string(rust::Str value) { return std::string(value.data(), value.size()); }

// A CPU tensor over memory the caller owns.
DLTensor cpu_tensor(void* data, DLDataType dtype, int64_t* shape, int32_t ndim) {
    DLTensor tensor{};
    tensor.data = data;
    tensor.device = DLDevice{kDLCPU, 0};
    tensor.ndim = ndim;
    tensor.dtype = dtype;
    tensor.shape = shape;
    tensor.strides = nullptr;
    tensor.byte_offset = 0;
    return tensor;
}

constexpr DLDataType kInt32{kDLInt, 32, 1};
constexpr DLDataType kInt64{kDLInt, 64, 1};

std::unique_ptr<Compiled> own(xgrammar::CompiledGrammar grammar) {
    return std::make_unique<Compiled>(Compiled{std::move(grammar)});
}

}  // namespace

std::unique_ptr<TokenizerInfo> tokenizer_info_from_json(rust::Str json) {
    auto parsed = xgrammar::TokenizerInfo::DeserializeJSON(to_string(json));
    if (auto* error = std::get_if<xgrammar::SerializationError>(&parsed)) {
        std::visit([](const auto& failure) { throw std::runtime_error(failure.what()); }, *error);
    }
    return std::make_unique<TokenizerInfo>(std::get<xgrammar::TokenizerInfo>(std::move(parsed)));
}

int32_t tokenizer_vocab_size(const TokenizerInfo& info) { return info.GetVocabSize(); }

std::unique_ptr<Compiler> new_compiler(const TokenizerInfo& info, int32_t max_threads) {
    // xgrammar's own cache is off: the control plane keeps the grammars
    // requests reuse, under its own bound.
    return std::make_unique<Compiler>(info, max_threads, /*cache_enabled=*/false);
}

std::unique_ptr<Compiled> compile_json_schema(const Compiler& compiler, rust::Str schema,
                                              bool any_whitespace) {
    return own(const_cast<Compiler&>(compiler).CompileJSONSchema(to_string(schema), any_whitespace));
}

std::unique_ptr<Compiled> compile_regex(const Compiler& compiler, rust::Str regex) {
    return own(const_cast<Compiler&>(compiler).CompileRegex(to_string(regex)));
}

std::unique_ptr<Compiled> compile_ebnf(const Compiler& compiler, rust::Str ebnf) {
    return own(const_cast<Compiler&>(compiler).CompileGrammar(to_string(ebnf)));
}

std::unique_ptr<Compiled> compile_structural_tag(const Compiler& compiler, rust::Str json) {
    return own(const_cast<Compiler&>(compiler).CompileStructuralTag(to_string(json)));
}

size_t compiled_memory_size(const Compiled& compiled) { return compiled.grammar.MemorySizeBytes(); }

std::unique_ptr<Matcher> new_matcher(const Compiled& compiled, int32_t max_rollback_tokens) {
    return std::make_unique<Matcher>(compiled.grammar, std::nullopt,
                                     /*terminate_without_stop_token=*/false, max_rollback_tokens);
}

bool matcher_accept_token(Matcher& matcher, int32_t token) { return matcher.AcceptToken(token); }

void matcher_fill_bitmask(Matcher& matcher, size_t data, int64_t rows, int64_t width,
                          int32_t row) {
    int64_t shape[2] = {rows, width};
    DLTensor bitmask = cpu_tensor(reinterpret_cast<void*>(data), kInt32, shape, 2);
    matcher.FillNextTokenBitmask(&bitmask, row);
}

void matcher_traverse_draft_tree(Matcher& matcher, rust::Slice<const int64_t> next_node,
                                 rust::Slice<const int64_t> next_sibling,
                                 rust::Slice<const int64_t> draft_tokens, size_t bitmask,
                                 int64_t width) {
    int64_t nodes = static_cast<int64_t>(draft_tokens.size());
    if (static_cast<int64_t>(next_node.size()) != nodes ||
        static_cast<int64_t>(next_sibling.size()) != nodes) {
        throw std::invalid_argument("the draft tree's columns differ in length");
    }
    int64_t node_shape[1] = {nodes};
    int64_t sibling_shape[1] = {nodes};
    int64_t token_shape[1] = {nodes};
    int64_t mask_shape[2] = {nodes, width};
    DLTensor next = cpu_tensor(const_cast<int64_t*>(next_node.data()), kInt64, node_shape, 1);
    DLTensor sibling =
        cpu_tensor(const_cast<int64_t*>(next_sibling.data()), kInt64, sibling_shape, 1);
    DLTensor tokens = cpu_tensor(const_cast<int64_t*>(draft_tokens.data()), kInt64, token_shape, 1);
    DLTensor masks = cpu_tensor(reinterpret_cast<void*>(bitmask), kInt32, mask_shape, 2);
    matcher.TraverseDraftTree(&next, &sibling, &tokens, &masks);
}

void matcher_rollback(Matcher& matcher, int32_t num_tokens) { matcher.Rollback(num_tokens); }

bool matcher_is_terminated(const Matcher& matcher) { return matcher.IsTerminated(); }

rust::Vec<int32_t> matcher_stop_token_ids(const Matcher& matcher) {
    rust::Vec<int32_t> out;
    for (int id : matcher.GetStopTokenIds()) {
        out.push_back(id);
    }
    return out;
}

}  // namespace ts_grammar
