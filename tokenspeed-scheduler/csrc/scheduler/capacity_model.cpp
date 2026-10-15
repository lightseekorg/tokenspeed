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

#include "scheduler/capacity_model.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/prefix/prefix_matcher.h"
#include "scheduler/operations/cache.h"
#include "utils.h"

namespace tokenspeed {

namespace {

std::int64_t ceilDiv(std::int64_t value, std::int64_t divisor) {
    _assert(value >= 0 && divisor > 0, "ceilDiv requires non-negative value and positive divisor");
    return (value + divisor - 1) / divisor;
}

// A State group's table: the checkpoint the next forward reads and the
// block it writes.
constexpr std::int64_t kStateSlots = 2;

}  // namespace

CapacityModel::CapacityModel(const SchedulerConfig& config) : config_{config} {
    config_.ValidateCapacityInputs();
    const std::vector<CacheGroupSpec> specs = MakeSpecsFromConfig(config_);
    groups_.reserve(specs.size());
    for (std::size_t i = 0; i < specs.size(); ++i) {
        const std::unique_ptr<PrefixMatcher> matcher = MakePrefixMatcher(specs[i]);
        // The most blocks a sliding window [T - W + 1, T) can intersect: one
        // more than behind a page-aligned T when its first row falls inside a
        // block.
        const std::int64_t window_blocks =
            specs[i].attn_kind == AttnKind::kSlidingWindow && specs[i].sliding_window > 1
                ? (static_cast<std::int64_t>(specs[i].sliding_window) - 1 + 2 * (specs[i].block_granularity - 1)) /
                      specs[i].block_granularity
                : 0;
        groups_.push_back(Group{
            .block_granularity = specs[i].block_granularity,
            .cache_blocks_per_lcm_block = specs[i].cache_blocks_per_lcm_block,
            .prefix_closed = matcher->IsPrefixClosed(),
            .lookback_pages = window_blocks,
        });
    }
}

std::int64_t CapacityModel::decodeWidth() const {
    // The slot the chunk completing a prompt reserves on every role: the
    // first decode/verify window, which the P role's drafter fills before
    // the remote decode ships it.
    return config_.decode_input_tokens;
}

std::int64_t CapacityModel::protectedTokens() const {
    // An overlapped forward protects one additional decode reservation that
    // cannot yet be reclaimed from the request table. The prefill role never
    // decodes locally, so it has no in-flight decode step to protect.
    if (config_.role == Role::kP) {
        return 0;
    }
    return static_cast<std::int64_t>(config_.overlap_schedule_depth) * decodeWidth();
}

std::int64_t CapacityModel::DraftWindowTokens(const SchedulerConfig& config) {
    // P hands off before its drafter writes past the first window.
    if (config.role == Role::kP || config.decode_input_tokens <= 1) {
        return 0;
    }
    return config.decode_input_tokens;
}

std::int64_t CapacityModel::tailHandoverBlocks(std::size_t group_index) const {
    // Blocks a request's table holds beyond its history while a published
    // tail changes hands, one admission long: a hitter copies the tail's
    // full blocks out of the shared entries, which stay pinned as sources
    // until the copies land; an owner that keeps writing past its prompt end
    // (the fused and P roles' local prefill) moves off every published tail
    // block, the partial one included, into fresh copies. A tail holds at
    // most P - 1 tokens; the D role lands the partial block as a fresh page
    // instead and never continues past a publication. A sliding group
    // publishes only its lookback behind the boundary.
    if (config_.disable_prefix_cache) {
        return 0;
    }
    const Group& group = groups_[group_index];
    const std::int64_t tail_tokens = static_cast<std::int64_t>(config_.prefix_granularity) - 1;
    const std::int64_t blocks = config_.role == Role::kD ? tail_tokens / group.block_granularity
                                                         : ceilDiv(tail_tokens, group.block_granularity);
    return group.prefix_closed ? blocks : std::min(blocks, group.lookback_pages);
}

std::vector<std::int64_t> CapacityModel::SingleRequestGroupPages(std::int32_t token_limit) const {
    _assert(token_limit >= 0, "single-request token limit must be non-negative");
    const std::int64_t decode_width = decodeWidth();
    const std::int64_t protected_tokens = protectedTokens();
    // The largest accepted prompt must still leave the first decode/MTP
    // reservation inside token_limit.
    const std::int64_t max_prompt_tokens =
        std::max<std::int64_t>(static_cast<std::int64_t>(token_limit) - decode_width, 0);
    const std::int64_t chunk_tokens = config_.max_scheduled_tokens;

    std::vector<std::int64_t> group_pages(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const CacheGroupConfig& group = config_.cache_groups[i];
        if (group.attn_kind == AttnKind::kState) {
            // Two slots on every role and in every phase: the checkpoint the
            // next forward reads and the block it writes.
            group_pages[i] = token_limit == 0 ? 0 : kStateSlots;
            continue;
        }
        const std::int64_t block_granularity = groups_[i].block_granularity;
        const std::int64_t lookback = groups_[i].lookback_pages;
        const auto local_prefill_peak = [&] {
            // Across every prompt up to max_prompt_tokens, retain the largest
            // resident window seen by either the first chunk or a later chunk.
            const std::int64_t first_prompt = std::min(max_prompt_tokens, chunk_tokens);
            std::int64_t pages = ceilDiv(first_prompt + decode_width + protected_tokens, block_granularity);
            if (max_prompt_tokens > chunk_tokens) {
                const std::int64_t later_prompt = std::min(max_prompt_tokens - chunk_tokens, chunk_tokens);
                pages = std::max(pages, lookback + ceilDiv(chunk_tokens, block_granularity));
                pages = std::max(pages,
                                 lookback + ceilDiv(later_prompt + decode_width + protected_tokens, block_granularity));
            }
            return pages;
        };
        std::int64_t child_pages = token_limit == 0 ? 0 : tailHandoverBlocks(i);
        if (groups_[i].prefix_closed) {
            child_pages += ceilDiv(static_cast<std::int64_t>(token_limit) + protected_tokens, block_granularity);
        } else if (config_.role == Role::kD) {
            const std::int64_t dense_pages =
                ceilDiv(static_cast<std::int64_t>(token_limit) + protected_tokens, block_granularity);
            const std::int64_t window_pages = ceilDiv(static_cast<std::int64_t>(*group.sliding_window_tokens - 1) +
                                                          decode_width + protected_tokens + block_granularity - 1,
                                                      block_granularity);
            // A sliding prefix probe can retain one older lookback island
            // across null holes while the remote prompt tail is restored at
            // absolute slots. Bound both intervals, capped by a dense table.
            child_pages += std::min<std::int64_t>(dense_pages, lookback + window_pages);
        } else {
            child_pages += local_prefill_peak();
        }
        group_pages[i] = child_pages;
    }
    return group_pages;
}

std::vector<std::int64_t> CapacityModel::ConcurrentGroupPages(std::int64_t max_total_tokens,
                                                              std::int32_t max_context_len) const {
    if (max_total_tokens < 0) {
        throw std::invalid_argument("CapacityModel: max_total_tokens must be >= 0");
    }
    if (max_context_len < 0) {
        throw std::invalid_argument("CapacityModel: max_context_len must be >= 0");
    }
    const std::int64_t live_requests = config_.max_batch_size;
    const std::int64_t decode_width = decodeWidth();
    const std::int64_t protected_tokens = protectedTokens() + DraftWindowTokens(config_);
    const std::int64_t chunk_tokens = config_.max_scheduled_tokens;

    std::vector<std::int64_t> group_pages(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const CacheGroupConfig& group = config_.cache_groups[i];
        if (group.attn_kind == AttnKind::kState) {
            // A State group's working set is two slots per live request,
            // whatever history the request carries.
            group_pages[i] = live_requests * kStateSlots;
            continue;
        }
        const std::int64_t block_granularity = groups_[i].block_granularity;
        const std::int64_t lookback = groups_[i].lookback_pages;
        // Dense history for every token, plus per request the page its
        // unaligned tail occupies and the protected tokens that may spill
        // past it: ceil((h + protected) / g) <= floor(h / g) + ceil((g - 1 +
        // protected) / g) for any history h, summed over the live requests;
        // plus per request the blocks a tail handover holds for one
        // admission.
        const std::int64_t dense_pages =
            ceilDiv(max_total_tokens, block_granularity) +
            live_requests * ceilDiv(block_granularity - 1 + protected_tokens, block_granularity);
        const std::int64_t handover_pages = live_requests * tailHandoverBlocks(i);
        if (groups_[i].prefix_closed) {
            group_pages[i] = dense_pages + handover_pages;
        } else {
            // A sliding request retains at most its window (or its whole
            // context when shorter), the next decode input and the protected
            // step, at any page alignment.
            const std::int64_t resident_tokens =
                std::min<std::int64_t>(static_cast<std::int64_t>(*group.sliding_window_tokens) - 1, max_context_len);
            const std::int64_t window_pages =
                ceilDiv(resident_tokens + decode_width + protected_tokens + block_granularity - 1, block_granularity);
            if (config_.role == Role::kD) {
                // The single-request landing bound, once per live request.
                group_pages[i] = std::min(dense_pages, live_requests * (lookback + window_pages)) + handover_pages;
            } else {
                // Every live request's resident window, plus one in-flight
                // prefill chunk behind its lookback before those rows slide.
                group_pages[i] = live_requests * window_pages + lookback +
                                 ceilDiv(std::min(chunk_tokens, max_total_tokens), block_granularity) + handover_pages;
            }
        }
    }
    return group_pages;
}

std::int64_t CapacityModel::LcmBlocksNeededFor(std::span<const std::int64_t> group_pages) const {
    _assert(group_pages.size() == groups_.size(), "page demand requires one entry per cache group");
    std::int64_t lcm_blocks = 0;
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        _assert(group_pages[i] >= 0, "group page demand must be non-negative");
        lcm_blocks += ceilDiv(group_pages[i], groups_[i].cache_blocks_per_lcm_block);
    }
    return lcm_blocks;
}

std::int32_t CapacityModel::MaxSingleRequestTokens(std::int64_t usable_lcm_blocks) const {
    std::int64_t low = 0;
    std::int64_t high = std::numeric_limits<std::int32_t>::max();
    while (low < high) {
        const std::int64_t candidate = low + (high - low + 1) / 2;
        if (LcmBlocksNeededFor(SingleRequestGroupPages(static_cast<std::int32_t>(candidate))) <= usable_lcm_blocks) {
            low = candidate;
        } else {
            high = candidate - 1;
        }
    }
    return static_cast<std::int32_t>(low);
}

}  // namespace tokenspeed
