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

#include "scheduler/request.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include "cache/prefix/prefix_hasher.h"
#include "fsm/forward_events.h"
#include "scheduler/kv_cache_events.h"

namespace tokenspeed {

Request::Request(const RequestSpec& spec, std::int32_t prefix_granularity, Role role)
    : id_{spec.request_id},
      session_id_{spec.session_id},
      cache_group_id_{spec.cache_group_id},
      shared_prefix_tokens_{spec.shared_prefix_tokens},
      token_container_{spec.tokens},
      submitted_prompt_size_{static_cast<std::int32_t>(spec.tokens.size())},
      max_new_tokens_{spec.max_new_tokens},
      max_cached_prefix_tokens_{spec.max_cached_prefix_tokens},
      prefix_granularity_{prefix_granularity},
      state_{role == Role::kFused ? fsm::State{fsm::Submitted{&token_container_, prefix_granularity}}
                                  : fsm::State{fsm::Bootstrapping{&token_container_, prefix_granularity}}} {}

std::span<const std::string> Request::PromptPrefixHashes() {
    return PromptPrefixHashes(submitted_prompt_size_ / prefix_granularity_);
}

std::span<const std::string> Request::PromptPrefixHashes(std::int32_t num_pages) {
    const std::int32_t prompt_pages = submitted_prompt_size_ / prefix_granularity_;
    _assert(num_pages >= 0 && num_pages <= prompt_pages, "prompt hashes stay within complete submitted pages");
    const bool first_lookup = !prompt_prefix_hashes_;
    if (first_lookup) {
        // Room for every complete prompt page up front: the chain never grows
        // past that, so it never reallocates and earlier spans stay valid.
        prompt_prefix_hashes_.emplace().reserve(static_cast<std::size_t>(prompt_pages));
    }
    auto& hashes = *prompt_prefix_hashes_;
    const auto first = static_cast<std::int32_t>(hashes.size());
    // The first lookup is a hashing pass even with no complete page, so a
    // profile shows every admitted prompt's hashing phase.
    if (first_lookup || num_pages > first) {
        ExtendPrefixHashes(hashes, PrefixPages(first, num_pages));
    }
    return std::span<const std::string>(hashes).first(num_pages);
}

void Request::AppendPrefixHashes(std::vector<std::string>& hashes, std::int32_t num_pages) const {
    _assert(num_pages >= static_cast<std::int32_t>(hashes.size()) && num_pages <= TokenSize() / prefix_granularity_,
            "completed hashes append only full token pages");
    // Grow geometrically: decode appends one completed page at a time.
    if (static_cast<std::size_t>(num_pages) > hashes.capacity()) {
        hashes.reserve(std::max<std::size_t>(num_pages, 2 * hashes.size()));
    }
    if (prompt_prefix_hashes_) {
        const auto& cached = *prompt_prefix_hashes_;
        const auto end = std::min<std::size_t>(num_pages, cached.size());
        if (hashes.size() < end) {
            hashes.insert(hashes.end(), cached.begin() + hashes.size(), cached.begin() + end);
        }
    }
    const auto first = static_cast<std::int32_t>(hashes.size());
    if (first < num_pages) {
        ExtendPrefixHashes(hashes, PrefixPages(first, num_pages));
    }
}

std::span<const std::uint64_t> Request::KvEventBlockHashes(std::size_t num_pages) {
    if (kv_event_block_hashes_.size() < num_pages) {
        const auto first_page = static_cast<std::int32_t>(kv_event_block_hashes_.size());
        for (const std::span<const std::int32_t> page : PrefixPages(first_page, static_cast<std::int32_t>(num_pages))) {
            const std::optional<std::uint64_t> parent_hash =
                kv_event_block_hashes_.empty() ? std::nullopt
                                               : std::optional<std::uint64_t>{kv_event_block_hashes_.back()};
            kv_event_block_hashes_.push_back(HashKvBlock(page, parent_hash));
        }
    }
    return std::span<const std::uint64_t>{kv_event_block_hashes_}.first(num_pages);
}

PrefillInfo Request::CurrentPrefillInfo() const {
    return std::visit(
        Overloaded{
            [](const fsm::Prefilling& state) { return state.CurrentPrefillInfo(); },
            [](const fsm::RemotePrefilling& state) { return state.CurrentPrefillInfo(); },
            [](const fsm::PrefillDone& state) { return state.CurrentPrefillInfo(); },
            [](const fsm::PrefillAwaitingResult& state) { return state.CurrentPrefillInfo(); },
            [](const fsm::PrefillAwaitingDestination& state) { return state.CurrentPrefillInfo(); },
            [this](const auto&) -> PrefillInfo {
                throw std::logic_error(
                    "Request::CurrentPrefillInfo: expected Prefilling, RemotePrefilling or PrefillDone; got " +
                    StateName());
            },
        },
        state_);
}

std::optional<TailPublication> Request::TailPublicationAt(std::int32_t num_computed_tokens,
                                                          const std::string& prior) const {
    _assert(num_computed_tokens >= 0 && num_computed_tokens <= TokenSize(),
            "the computed frontier lies inside the request's tokens");
    const std::int32_t tail_tokens = num_computed_tokens % prefix_granularity_;
    if (tail_tokens == 0) {
        return std::nullopt;
    }
    return TailPublication{
        .tail_tokens = tail_tokens,
        .hash = HashPrefixPage(TokenSlice(num_computed_tokens - tail_tokens, tail_tokens), ContentHash{prior}),
    };
}

std::int32_t Request::NumComputedTokens() const {
    return std::visit(
        Overloaded{
            [](const fsm::Prefilling& state) { return state.window.begin + state.window.size; },
            [](const fsm::PrefillDone& state) { return state.window.begin + state.window.size; },
            [](const fsm::PrefillAwaitingDestination& state) { return state.window.begin + state.window.size; },
            // Every token but the last is computed; the last is the sampled
            // token no landed forward consumed.
            [this](const fsm::Decoding&) { return TokenSize() - 1; },
            [this](const fsm::Finishing&) { return TokenSize() - 1; },
            [this](const auto&) -> std::int32_t {
                throw std::logic_error(
                    "Request::NumComputedTokens: expected Prefilling, PrefillDone, Decoding or Finishing; got " +
                    StateName());
            },
        },
        state_);
}

fsm::ForwardResources& Request::forwardResources(const char* operation) {
    fsm::ForwardResources* result = std::visit(
        []<typename State>(State& state) -> fsm::ForwardResources* {
            if constexpr (fsm::HoldsForwardResources<State>) {
                return &state.resources;
            }
            return nullptr;
        },
        state_);
    if (result == nullptr) {
        throw std::logic_error(std::string{"Request::"} + operation + ": expected a forward state; got " + StateName());
    }
    return *result;
}

const fsm::ForwardResources& Request::forwardResources(const char* operation) const {
    const fsm::ForwardResources* result = std::visit(
        []<typename State>(const State& state) -> const fsm::ForwardResources* {
            if constexpr (fsm::HoldsForwardResources<State>) {
                return &state.resources;
            }
            return nullptr;
        },
        state_);
    if (result == nullptr) {
        throw std::logic_error(std::string{"Request::"} + operation + ": expected a forward state; got " + StateName());
    }
    return *result;
}

}  // namespace tokenspeed
