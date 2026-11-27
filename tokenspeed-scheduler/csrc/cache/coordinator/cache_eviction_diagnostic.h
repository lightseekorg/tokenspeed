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

#include <span>
#include <string>
#include <string_view>
#include <unordered_set>

#include "cache/coordinator/cache_evictor.h"

namespace tokenspeed {

// Binds a configured request tag and token boundary when the request is submitted,
// before any planning can evict its target while the request waits.
// All selection state stays in the diagnostic helper, never in the scheduler.
void BindCacheEvictionDiagnostic(std::string_view request_id, std::span<const std::string> hashes,
                                 std::int32_t prefix_granularity) noexcept;

// Private diagnostic helpers for successful removals. Presence is the value
// already used by eviction accounting; pending keys are read only after hash
// selection. No references escape the call, and diagnostic errors never escape
// into a cache operation whose removal has already committed.
void LogCacheEviction(const CacheKey& key,
                      CacheEvictor::DeviceEvictionCounts CacheEvictor::DeviceEvictionStats::*reason,
                      const PrefixCacheIndex::CachedBlockMetadata& metadata, bool host_present,
                      const std::unordered_set<CacheKey, CacheKeyHash>& pending_keys) noexcept;
void LogCacheEviction(const CacheKey& key, CacheEvictor::HostEvictionCounts CacheEvictor::HostEvictionStats::*reason,
                      const PrefixCacheIndex::CachedBlockMetadata& metadata, bool device_present,
                      const std::unordered_set<CacheKey, CacheKeyHash>& pending_keys) noexcept;

}  // namespace tokenspeed
