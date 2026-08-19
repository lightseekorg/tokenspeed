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

#include "cache/core/cache_config.h"

#include <stdexcept>
#include <string>

namespace tokenspeed {

void CacheGroupConfig::Validate() const {
    if (group_id.empty()) {
        throw std::invalid_argument("CacheGroupConfig: group_id must be non-empty");
    }
    // Every remaining message names the group: a model mixes several of them,
    // and the offending one is the only actionable part of the diagnostic.
    const std::string where = "Cache group '" + group_id + "': ";
    if (rows_per_page <= 0) {
        throw std::invalid_argument(where + "rows_per_page must be > 0");
    }
    if (entry_stride_tokens <= 0) {
        throw std::invalid_argument(where + "entry_stride_tokens must be > 0");
    }
    if (total_pages < 1) {
        throw std::invalid_argument(where + "total_pages must include the null page");
    }
    if (cache_blocks_per_lcm_block <= 0) {
        throw std::invalid_argument(where + "cache_blocks_per_lcm_block must be > 0");
    }
    if (retention == Retention::SlidingWindow && (!sliding_window_tokens || *sliding_window_tokens <= 0)) {
        throw std::invalid_argument(where + "sliding_window_tokens must be > 0 for sliding groups");
    }
}

}  // namespace tokenspeed
