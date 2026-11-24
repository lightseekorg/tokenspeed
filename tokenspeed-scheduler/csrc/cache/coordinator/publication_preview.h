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

#include <unordered_map>
#include <vector>

#include "cache/coordinator/cache_store.h"

namespace tokenspeed {

// Completed producer-owned components visible only inside an admission plan.
// They can repair dependencies and change fallbacks, but cannot fund capacity.
// No registration, access stamp, reference count or watermark changes here.
class PublicationPreview {
public:
    PublicationPreview(const CacheStore& store, std::span<const GroupDemand> demands,
                       const CheckpointPublication& publication);
    std::vector<RecoveryIndex::Leaf> Leaves(const CacheStore& store) const;
    std::vector<CheckpointTree::RecoveryPoint> Points(const CacheStore& store) const;
    void Apply(RecoveryIndex::View& view) const { view.Apply(publications_); }
    const RecoveryIndex::Entry& At(const ContentHash& hash, const RecoveryIndex::View& view) const;
    RecoveryIndex::Entry Resolve(const ContentHash& hash, const RecoveryIndex::View& view,
                                 bool allow_device_copy) const;
    bool Holds(const ContentHash& hash) const { return routes_.contains(hash); }
    const auto& Protected() const { return protected_; }

private:
    struct Point {
        ContentHash hash;
        std::int32_t tokens;
    };
    std::vector<Point> points_;
    std::vector<RecoveryIndex::Leaf> leaves_;
    std::vector<RecoveryIndex::View::Change> publications_;
    std::vector<std::pair<CacheTier, CacheBlockLocation>> protected_;
    std::unordered_map<ContentHash, RecoveryIndex::Entry> routes_;
    std::vector<std::int32_t> tokens_;
    std::vector<ContentHash> hashes_;
};
}  // namespace tokenspeed
