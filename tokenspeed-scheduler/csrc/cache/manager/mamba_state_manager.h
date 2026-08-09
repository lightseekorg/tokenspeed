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

#include "cache/manager/swa_manager.h"

namespace tokenspeed {

// GDN/mamba semantics: hit the nearest P-boundary snapshot and retain only the
// request's live state page. Completed pages remain cache-owned after the
// request table releases them.
class MambaStateManager : public SwaManager {
public:
    explicit MambaStateManager(std::int32_t cache_block_tokens, std::int32_t cache_blocks_per_lcm_block = 1,
                               std::uint32_t group_id = 0)
        : SwaManager(cache_block_tokens, cache_blocks_per_lcm_block, /*sliding_window=*/2, group_id) {}
};

}  // namespace tokenspeed
