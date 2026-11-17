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
#include <vector>

#include "cache/core/cache_block_ref.h"

namespace tokenspeed {

class SharedL2Proxy;

// A waiting prompt's claim on its pool lookup, owned by its pre-admission
// state: pending until the pool's answer is applied, then the proxy Host
// entries the answer mirrored, pinned. Whatever ends the wait -- admission
// consuming the state, an abort -- drops it, and the proxy entries are let
// go (the load-back an admission started holds its sources itself). The
// runtime answers a lookup in the call that sent it, so a hold is pending
// only between SharedL2Lookups and ApplySharedL2Lookups. Move-only; an empty
// hold claims nothing. The proxy must outlive every hold it issued.
class SharedL2LookupHold {
public:
    SharedL2LookupHold() noexcept = default;
    SharedL2LookupHold(const SharedL2LookupHold&) = delete;
    SharedL2LookupHold& operator=(const SharedL2LookupHold&) = delete;
    SharedL2LookupHold(SharedL2LookupHold&& other) noexcept;
    SharedL2LookupHold& operator=(SharedL2LookupHold&& other) noexcept;
    ~SharedL2LookupHold();

    explicit operator bool() const noexcept { return proxy_ != nullptr; }
    // The id the runtime answers the lookup under.
    std::uint64_t LookupId() const noexcept { return lookup_id_; }
    // Ends the claim now and leaves the hold empty.
    void reset() noexcept;

private:
    friend class SharedL2Proxy;

    SharedL2LookupHold(SharedL2Proxy& proxy, std::uint64_t lookup_id) noexcept
        : proxy_{&proxy}, lookup_id_{lookup_id} {}

    SharedL2Proxy* proxy_{nullptr};
    std::uint64_t lookup_id_{0};
    std::vector<CacheBlockRef> blocks_;
};

}  // namespace tokenspeed
