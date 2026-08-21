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

//! One attention group: spec, allocator, match policy, and prefix-reuse index.
//!
//! Ported from `tokenspeed-scheduler/csrc/cache/cache_group.h`. Allocation and
//! prefix matching stay separate; the group is the only place they are paired.

use crate::cache_types::CacheGroupSpec;
use crate::group_allocator::GroupAllocator;
use crate::prefix_index::PrefixCacheIndex;
use crate::prefix_matcher::PrefixMatcher;

/// One attention group.
pub struct CacheGroup {
    spec: CacheGroupSpec,
    allocator: GroupAllocator,
    matcher: PrefixMatcher,
    index: PrefixCacheIndex,
}

impl CacheGroup {
    pub fn new(spec: CacheGroupSpec, allocator: GroupAllocator, matcher: PrefixMatcher) -> Self {
        let group_id = allocator.id();
        Self {
            spec,
            allocator,
            matcher,
            index: PrefixCacheIndex::new(group_id),
        }
    }

    pub fn allocator(&self) -> &GroupAllocator {
        &self.allocator
    }

    pub fn matcher(&self) -> &PrefixMatcher {
        &self.matcher
    }

    pub fn index(&self) -> &PrefixCacheIndex {
        &self.index
    }

    pub fn index_mut(&mut self) -> &mut PrefixCacheIndex {
        &mut self.index
    }

    pub fn spec(&self) -> &CacheGroupSpec {
        &self.spec
    }

    pub fn id(&self) -> u32 {
        self.allocator.id()
    }
}
