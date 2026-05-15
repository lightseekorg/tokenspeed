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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <string>

#include "resource/allocator/owned_pages.h"
#include "resource/allocator/paged_cache_group.h"

namespace tokenspeed {

// Per-group snapshot held by a TreeNode. RAII returns pages to the allocator.
struct PagedCacheGroupSnapshot {
    OwnedPages pages;
    std::int32_t base_logical_page{0};
    std::int32_t raw_token_cursor{0};
    bool sliding{false};
};

// Snapshot for a TreeNode at a history-alignment-aligned raw-token boundary;
// completeness is tracked per family.
struct PagedCacheSnapshot {
    std::int32_t prefix_len_tokens{0};
    std::map<std::string, PagedCacheGroupSnapshot> groups;
    // Filled by HybridPrefixCache::AttachPagedCacheSnapshotToNode based on
    // which group ids landed in `groups` vs required-per-family lists.
    std::set<PagedCacheGroupFamily> complete_families;

    bool IsCompleteFor(PagedCacheGroupFamily f) const {
        return complete_families.find(f) != complete_families.end();
    }
};

}  // namespace tokenspeed
