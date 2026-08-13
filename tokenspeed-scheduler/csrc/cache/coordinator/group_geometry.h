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
#include <limits>

#include "cache/core/acquire_plan.h"
#include "cache/core/block_table.h"
#include "cache/core/cache_types.h"
#include "utils.h"

namespace tokenspeed {

// One cache group's token -> page arithmetic. This is the logical side of the
// placement seam: it turns token demands into the token-free AcquirePlan /
// block counts that GroupAllocator executes, so the allocator never perceives
// block_granularity or any other token quantity.
class GroupGeometry {
public:
    explicit GroupGeometry(std::int32_t block_granularity) : block_granularity_{block_granularity} {
        _assert(block_granularity > 0, "block_granularity must be > 0");
    }

    std::int32_t BlockGranularity() const noexcept { return block_granularity_; }

    std::int32_t BlocksNeededFor(const BlockTable& table, std::int32_t num_tokens) const {
        if (num_tokens <= table.AvailableTokens()) {
            return 0;
        }
        const std::int32_t over = num_tokens - table.AvailableTokens();
        return (over + block_granularity_ - 1) / block_granularity_;
    }

    std::int32_t BlocksNeededFor(const BlockTable& table, const GroupDemand& demand) const {
        if (demand.materialized_suffix_start < 0) {
            return BlocksNeededFor(table, demand.num_tokens + demand.reserve_tokens);
        }
        return sparseSuffixBlocks(table, demand);
    }

    AcquirePlan PlanAcquire(const BlockTable& table, const GroupDemand& demand) const {
        if (demand.materialized_suffix_start < 0) {
            return PlanAcquire(table, demand.num_tokens, demand.reserve_tokens);
        }
        const std::int32_t num_blocks = sparseSuffixBlocks(table, demand);
        const std::int64_t extent = static_cast<std::int64_t>(demand.num_tokens) + demand.reserve_tokens;
        const std::int32_t logical_blocks = static_cast<std::int32_t>((extent + block_granularity_ - 1) / block_granularity_);
        return AcquirePlan{
            .num_blocks = num_blocks,
            .suffix_start = demand.materialized_suffix_start,
            .table_blocks_after = logical_blocks,
            .available_tokens_after = logical_blocks * block_granularity_ - demand.num_tokens,
        };
    }

    AcquirePlan PlanAcquire(const BlockTable& table, std::int32_t num_tokens, std::int32_t reserve_tokens = 0) const {
        _assert(num_tokens >= 0 && reserve_tokens >= 0, "token demand and reserve must be non-negative");
        const std::int32_t num_blocks = BlocksNeededFor(table, num_tokens + reserve_tokens);
        return AcquirePlan{
            .num_blocks = num_blocks,
            .suffix_start = -1,
            .table_blocks_after = table.NumBlocks() + num_blocks,
            .available_tokens_after = table.AvailableTokens() + num_blocks * block_granularity_ - num_tokens,
        };
    }

    // Pages [0, result) of the table have fully expired under the group's
    // retention policy at this progress; kFull never expires. This is where
    // the sliding-window/state token policy meets page arithmetic, so it
    // lives here and not in the (token-free) allocator.
    std::int32_t ExpiredBlocksAt(const CacheGroupSpec& spec, std::int32_t num_computed_tokens) const {
        std::int32_t window = 0;
        switch (spec.kind) {
            case AttnKind::kFull:
                return 0;
            case AttnKind::kSlidingWindow:
                window = spec.sliding_window;
                break;
            case AttnKind::kMambaState:
                // Keep exactly the live state page plus its snapshot.
                window = kMambaStateWindow;
                break;
            default:
                FatalCheck(false, "unknown AttnKind in retention policy");
        }
        _assert(window > 0, "retention window must be > 0");
        const std::int32_t skipped = num_computed_tokens - window + 1;
        // Only fully-slid-out pages expire.
        return skipped <= 0 ? 0 : skipped / block_granularity_;
    }

    static constexpr std::int32_t kMambaStateWindow = 2;

private:
    std::int32_t sparseSuffixBlocks(const BlockTable& table, const GroupDemand& demand) const {
        // Decode-side prefix acquisition may have already installed aligned
        // null holes for state. They carry no ownership and remain safe to
        // extend sparsely up to the remote endpoint snapshot.
        _assert(table.AvailableTokens() == 0, "sparse suffix materialization requires a page boundary");
        _assert(table.NumBlocks() <= demand.materialized_suffix_start,
                "sparse suffix overlaps the existing block table");
        _assert(demand.num_tokens > 0 && demand.reserve_tokens >= 0,
                "sparse suffix materialization requires a positive extent");
        const std::int64_t extent = static_cast<std::int64_t>(demand.num_tokens) + demand.reserve_tokens;
        _assert(extent <= std::numeric_limits<std::int32_t>::max(), "sparse suffix extent exceeds int32 range");
        const std::int32_t last_block = static_cast<std::int32_t>((extent - 1) / block_granularity_);
        _assert(demand.materialized_suffix_start <= last_block,
                "materialized suffix starts beyond the requested extent");
        return last_block - demand.materialized_suffix_start + 1;
    }

    std::int32_t block_granularity_;
};

}  // namespace tokenspeed
