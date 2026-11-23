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

#include <cstddef>
#include <memory>

namespace tokenspeed {

// A software prefetch of every cache line `object` spans: a hint that loads
// from it are coming. It never faults and changes no program state, so a
// stale address costs bandwidth, never correctness; computing the address
// must still read only live objects. How many prefetches it issues depends
// on T alone, never on the address, so it costs no branch: one per 64 bytes
// from the object's first byte, and one at its last byte unless T's
// alignment keeps that byte on the line before. Where the object spans one
// line fewer, a prefetch repeats a line.
template <typename T>
inline void PrefetchObject([[maybe_unused]] const T& object) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    // The line size of the x86-64 and most Arm cores; with longer lines some
    // of these hints are redundant.
    constexpr std::size_t kCacheLine = 64;
    static_assert(alignof(T) <= kCacheLine);
    const char* const first = reinterpret_cast<const char*>(std::addressof(object));
    for (std::size_t offset = 0; offset < sizeof(T); offset += kCacheLine) __builtin_prefetch(first + offset);
    // The object starts at most kCacheLine - alignof(T) bytes into its line.
    if constexpr ((sizeof(T) - 1) % kCacheLine >= alignof(T)) __builtin_prefetch(first + sizeof(T) - 1);
#endif
}

}  // namespace tokenspeed
