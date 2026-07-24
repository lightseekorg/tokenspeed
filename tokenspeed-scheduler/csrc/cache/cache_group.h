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

// Runtime state for one coordinator schema entry. The coordinator owns the
// immutable schema and BlockPoolSet owns the pool; CacheGroup observes both by
// index/pointer and owns only its policy manager.
class CacheGroup {
public:
    CacheGroup(std::uint32_t schema_index, std::unique_ptr<KvCacheManager> manager, BlockPool& pool)
        : schema_index_{schema_index}, pool_{&pool}, manager_{std::move(manager)} {}

    CacheGroup(const CacheGroup&) = delete;
    CacheGroup& operator=(const CacheGroup&) = delete;
    CacheGroup(CacheGroup&&) = default;
    CacheGroup& operator=(CacheGroup&&) = default;

    KvCacheManager& Manager() { return *manager_; }
    const KvCacheManager& Manager() const { return *manager_; }
    std::uint32_t SchemaIndex() const { return schema_index_; }
    BlockPool& Pool() { return *pool_; }
    const BlockPool& Pool() const { return *pool_; }

private:
    std::uint32_t schema_index_;
    BlockPool* pool_;
    std::unique_ptr<KvCacheManager> manager_;
};

}  // namespace tokenspeed
