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
#include <optional>
#include <string>

namespace tokenspeed {

enum class CacheGroupFamily { History, State };

enum class CacheTransferPolicy {
    Unspecified,
    FullSuffix,
    LatestSnapshot,
};

struct CacheGroupConfig {
    enum class Retention {
        FullHistory,
        SlidingWindow,
    };

    std::string group_id;
    std::int32_t rows_per_page{};
    std::int32_t entry_stride_tokens{};
    std::int32_t total_pages{};
    // Number of this group's CacheBlocks packed into one physical LCM block.
    std::int32_t cache_blocks_per_lcm_block{1};
    Retention retention{Retention::FullHistory};
    std::optional<std::int32_t> sliding_window_tokens{};
    CacheGroupFamily family{CacheGroupFamily::History};
    CacheTransferPolicy transfer_policy{CacheTransferPolicy::Unspecified};

    std::int32_t BlockGranularity() const { return rows_per_page * entry_stride_tokens; }

    // A State group WITHOUT SlidingWindow retention keeps one recurrent-state
    // checkpoint per block instead of a token history: the mamba-style group
    // whose PD destination layout is a LatestSnapshot. family=State alone is
    // not enough -- it also covers linear-attention sliding groups.
    bool IsSnapshotStateGroup() const {
        return family == CacheGroupFamily::State && retention != Retention::SlidingWindow;
    }

    void Validate() const;
};

}  // namespace tokenspeed
