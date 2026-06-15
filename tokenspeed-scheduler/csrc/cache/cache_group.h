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
#include <memory>

#include "cache/cache_types.h"
#include "cache/kv_cache_manager.h"

namespace tokenspeed {

// One attention group: its spec, its index-derived group_id, and the manager
// that runs it. Plain aggregate -- all per-attention-type behavior lives inside
// manager_ (the virtual KvCacheManager). Move-only (owns a unique_ptr).
class CacheGroup {
public:
    CacheGroup(KvCacheSpec spec, std::uint32_t group_id, std::unique_ptr<KvCacheManager> manager)
        : spec_{spec}, group_id_{group_id}, manager_{std::move(manager)} {}

    KvCacheManager& Manager() { return *manager_; }
    const KvCacheManager& Manager() const { return *manager_; }
    const KvCacheSpec& Spec() const { return spec_; }
    std::uint32_t GroupId() const { return group_id_; }

private:
    KvCacheSpec spec_;
    std::uint32_t group_id_;
    std::unique_ptr<KvCacheManager> manager_;
};

}  // namespace tokenspeed
