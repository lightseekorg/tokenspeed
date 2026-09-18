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

#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "scheduler/types.h"

namespace tokenspeed {

// The per-request cache working-set model: the decode reservation, the
// overlap-protected step, a snapshot-state group's checkpoints and banked
// growth, a sliding group's lookback and resident window. Two callers must
// agree on it -- the Python recipes size a pool from it before any pool
// exists, and the Scheduler bounds a single request against the pool they
// sized -- so it lives here once and reads only SchedulerConfig fields that
// are known before sizing. No total_pages is ever consulted.
class CapacityModel {
public:
    // Validates the sizing inputs (SchedulerConfig::ValidateCapacityInputs)
    // and copies the config, so a model outlives the config it was built from.
    explicit CapacityModel(const SchedulerConfig& config);

    std::int32_t NumGroups() const { return static_cast<std::int32_t>(groups_.size()); }

    // Child pages (null page excluded) each group holds for one request
    // whose prompt plus first decode reservation spans token_limit tokens,
    // at the moment of its largest working set.
    std::vector<std::int64_t> SingleRequestGroupPages(std::int32_t token_limit) const;

    // Child pages (null page excluded) each group holds for max_batch_size
    // live requests whose histories -- first decode reservation included --
    // total max_total_tokens, no request spanning more than max_context_len.
    // For every group, one request of L tokens needs no more than this
    // reports for max_batch_size = 1 and max_total_tokens = max_context_len
    // = L: a pool sized from it admits every request the bound accepts.
    std::vector<std::int64_t> ConcurrentGroupPages(std::int64_t max_total_tokens, std::int32_t max_context_len) const;

    // LCM blocks that place group_pages[g] child pages in every group g.
    std::int64_t LcmBlocksNeededFor(std::span<const std::int64_t> group_pages) const;

    // Largest token_limit whose SingleRequestGroupPages fit usable_lcm_blocks.
    std::int32_t MaxSingleRequestTokens(std::int64_t usable_lcm_blocks) const;

private:
    struct Group {
        std::int64_t block_granularity{};
        std::int64_t cache_blocks_per_lcm_block{};
        // Prefix-match policy, the same one the coordinator's matcher answers.
        bool prefix_closed{};
        std::int64_t lookback_pages{};
    };

    std::int64_t decodeWidth() const;
    std::int64_t protectedTokens() const;

    SchedulerConfig config_;
    std::vector<Group> groups_;
};

}  // namespace tokenspeed
