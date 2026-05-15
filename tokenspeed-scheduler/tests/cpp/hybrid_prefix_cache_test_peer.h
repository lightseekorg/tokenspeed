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

// Test-only friend of HybridPrefixCache; exposes hooks needed to drive prune
// paths whose direct public surface is non-trivial to set up via AdmitChunk.

#include "resource/hybrid_prefix_cache/hybrid_prefix_cache.h"
#include "resource/radix_tree/tree_node.h"

namespace tokenspeed {

class HybridPrefixCacheTestPeer {
public:
    static bool DetachStateSnapshotFromNode(HybridPrefixCache& cache, TreeNode* node) {
        return cache.DetachStateSnapshotFromNode(node);
    }
};

}  // namespace tokenspeed
