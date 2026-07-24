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

#include "cache/swa_manager.h"

namespace tokenspeed {

// GDN/mamba align semantics: hit = the nearest aligned snapshot, retention = last page only
// (= SwaManager(P, 2) matching), plus final-page-only registration -- interior chunk
// boundaries never received a state write, so only the aligned chunk end is publishable.
class MambaStateManager : public SwaManager {
public:
    explicit MambaStateManager(std::int32_t block_size, KvTableLayout table_layout = KvTableLayout::kAbsolute)
        : SwaManager(block_size, /*sliding_window=*/2, table_layout) {}

    bool RegistersAlignedFinalPageOnly() const override { return true; }
};

}  // namespace tokenspeed
