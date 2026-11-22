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

#include "cache/tier/transfer.h"
#include "resource/allocator/req_pool_allocator.h"

namespace tokenspeed {

// Queried by the native host before model startup. Version 1 supplies native
// restore/prefix attribution and slot ownership; the runtime owns copy completion.
inline constexpr std::uint32_t kCacheEventProtocolVersion{1};

// Optional observation of native ownership. The caller supplies engine/rank
// context and bounded delivery. No GPU operations or blocking I/O belong here.
class CacheEventSink : public SlotEventSink {
public:
    virtual void OnPrefixLoadScheduled(const LoadBackOperation& operation) noexcept = 0;
    virtual void OnRestoreScheduled(const SnapshotRestoreOperation& operation) noexcept = 0;
};

}  // namespace tokenspeed
