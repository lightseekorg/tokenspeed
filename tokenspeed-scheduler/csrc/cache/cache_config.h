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

enum class PagedCacheGroupFamily { History, State };

enum class PagedCacheTransferPolicy {
    Unspecified,
    FullSuffix,
    LatestSnapshot,
};

struct PagedCacheGroupConfig {
    enum class Retention {
        FullHistory,
        SlidingWindow,
    };

    std::string group_id;
    std::int32_t rows_per_page{};
    std::int32_t entry_stride_tokens{};
    std::int32_t total_pages{};
    // Serialization compatibility only. Logical cache granularity is the
    // scheduler-wide block_size and cannot vary by group.
    std::int32_t block_size{0};
    // Number of this group's CacheBlocks packed into one physical LCM block.
    std::int32_t cache_blocks_per_lcm_block{1};
    Retention retention{Retention::FullHistory};
    std::optional<std::int32_t> sliding_window_tokens{};
    PagedCacheGroupFamily family{PagedCacheGroupFamily::History};
    PagedCacheTransferPolicy transfer_policy{PagedCacheTransferPolicy::Unspecified};

    std::int32_t RawTokensPerPage() const { return rows_per_page * entry_stride_tokens; }
    void Validate() const;
};

}  // namespace tokenspeed
