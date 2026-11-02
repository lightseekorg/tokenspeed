/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Adapted from Folly StableRadixSort.h at 83f525dc2b583925b857ac153e3befd51eb4879e.
// Specializes integer projection for nonpositive cache position ranks and
// replaces Folly utilities with standard C++ headers. See third_party/folly.
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>
namespace tokenspeed::folly_radix {
// Radix configuration
inline constexpr size_t kRadixBits = 8;
static_assert(kRadixBits > 0 && kRadixBits <= 16);
inline constexpr size_t kRadixBuckets = 1 << kRadixBits;  // 256
inline constexpr size_t kRadixMask = kRadixBuckets - 1;

template <typename Key>
inline constexpr size_t kRadixPasses = (sizeof(Key) * 8 + kRadixBits - 1) / kRadixBits;

/**
 * Extract the radix digit at the given pass index from a key.
 * Pass 0 extracts the least significant digit.
 */
template <typename Key>
inline uint32_t extractRadixDigit(Key key, size_t pass) noexcept {
    static_assert(std::is_unsigned_v<Key>, "Key must be unsigned");
    // kRadixPasses ensures pass is always < sizeof(Key), so shift is safe
    return static_cast<uint32_t>((key >> (pass * kRadixBits)) & kRadixMask);
}

/**
 * Core LSD radix sort implementation with alternating buffers.
 *
 * Uses counting sort for each pass, processing from least to most
 * significant byte. The alternating buffer pattern avoids copying
 * data between passes - we simply swap source and destination pointers.
 */
template <typename T, typename Projection>
void stableRadixSortImpl(T* data, T* buffer, size_t n, Projection proj) {
    using Key = std::invoke_result_t<Projection, const T&>;
    static_assert(std::is_unsigned_v<Key>, "Key type must be unsigned");

    constexpr size_t kPasses = kRadixPasses<Key>;

    T* src = data;
    T* dst = buffer;

    // Phase 1: Count occurrences of each byte value for every pass in a single
    // sweep. The per-digit histogram is permutation-invariant, so counting once
    // on the original input yields the same counts each pass needs.
    alignas(64) std::array<std::array<size_t, kRadixBuckets>, kPasses> counts{};
    for (size_t i = 0; i < n; ++i) {
        auto key = proj(src[i]);
        for (size_t pass = 0; pass < kPasses; ++pass) {
            ++counts[pass][extractRadixDigit(key, pass)];
        }
    }

    for (size_t pass = 0; pass < kPasses; ++pass) {
        const auto& passCounts = counts[pass];

        // Check if this pass can be skipped (all elements have the same byte value)
        size_t nonzeroCount = 0;
        for (size_t c : passCounts) {
            if (c != 0 && ++nonzeroCount > 1) {
                break;
            }
        }
        if (nonzeroCount <= 1) {
            continue;
        }

        // Phase 2: Convert counts to starting offsets (prefix sum)
        alignas(64) std::array<size_t, kRadixBuckets> offsets{};
        size_t total = 0;
        for (size_t i = 0; i < kRadixBuckets; ++i) {
            offsets[i] = total;
            total += passCounts[i];
        }

        // Phase 3: Scatter elements to output (preserves stability)
        for (size_t i = 0; i < n; ++i) {
            auto key = proj(src[i]);
            size_t& offset = offsets[extractRadixDigit(key, pass)];
            dst[offset] = src[i];
            ++offset;
        }

        // Swap source and destination for next pass
        std::swap(src, dst);
    }

    // If the final result is in the buffer, copy it back to data
    if (src != data) {
        std::copy(src, src + n, data);
    }
}

template <typename T>
void sortPosition(std::vector<T>& candidates) {
    // Cache ranks are zero or the negation of a nonnegative int32 block index.
    const auto key = [](const T& candidate) {
        // Known domain: rank is in [-INT32_MAX, 0]. Map rank zero
        // to UINT32_MAX so nearby negative ranks share their upper digits.
        return 0xFFFFFFFFU - static_cast<std::uint32_t>(-candidate.position_rank);
    };
    auto buffer = std::make_unique_for_overwrite<T[]>(candidates.size());
    stableRadixSortImpl(candidates.data(), buffer.get(), candidates.size(), key);
}
}  // namespace tokenspeed::folly_radix
