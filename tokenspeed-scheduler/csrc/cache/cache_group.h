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
#include <utility>

#include "cache/core/cache_types.h"
#include "cache/allocator/group_allocator.h"
#include "cache/prefix/prefix_index.h"
#include "cache/prefix/prefix_matcher.h"

namespace tokenspeed {

// One attention group: its spec, the allocator (where blocks live), the
// prefix-match policy (matcher), and the prefix-reuse index. Allocation and
// prefix matching stay separate; the group is the only place they are paired.
class CacheGroup {
public:
    CacheGroup(CacheGroupSpec spec, std::unique_ptr<GroupAllocator> allocator, std::unique_ptr<PrefixMatcher> matcher)
        : spec_{spec}, allocator_{std::move(allocator)}, matcher_{std::move(matcher)}, index_{allocator_->Id()} {}

    CacheGroup(const CacheGroup&) = delete;
    CacheGroup& operator=(const CacheGroup&) = delete;
    CacheGroup(CacheGroup&&) = default;
    CacheGroup& operator=(CacheGroup&&) = default;

    GroupAllocator& Allocator() { return *allocator_; }
    const GroupAllocator& Allocator() const { return *allocator_; }
    const PrefixMatcher& Matcher() const { return *matcher_; }
    PrefixCacheIndex& Index() { return index_; }
    const PrefixCacheIndex& Index() const { return index_; }
    const CacheGroupSpec& Spec() const { return spec_; }
    std::uint32_t Id() const { return allocator_->Id(); }

private:
    CacheGroupSpec spec_;
    std::unique_ptr<GroupAllocator> allocator_;
    std::unique_ptr<PrefixMatcher> matcher_;
    PrefixCacheIndex index_;
};

}  // namespace tokenspeed
