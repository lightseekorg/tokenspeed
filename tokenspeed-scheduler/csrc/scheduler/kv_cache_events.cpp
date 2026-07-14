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

#include "scheduler/kv_cache_events.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <vector>

#include "scheduler/xxhash3.h"

namespace tokenspeed {
namespace {

constexpr std::uint64_t kFnvOffsetBasis = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

std::atomic<bool> g_use_xxh3_block_hash{false};

void MixByte(std::uint64_t& hash, std::uint8_t byte) {
    hash ^= byte;
    hash *= kFnvPrime;
}

void MixUint64(std::uint64_t& hash, std::uint64_t value) {
    for (std::size_t i = 0; i < sizeof(value); ++i) {
        MixByte(hash, static_cast<std::uint8_t>((value >> (i * 8)) & 0xffu));
    }
}

void MixInt32(std::uint64_t& hash, std::int32_t value) {
    const auto raw = static_cast<std::uint32_t>(value);
    for (std::size_t i = 0; i < sizeof(raw); ++i) {
        MixByte(hash, static_cast<std::uint8_t>((raw >> (i * 8)) & 0xffu));
    }
}

std::vector<std::uint8_t> TokenIdsToLittleEndianBytes(std::span<const std::int32_t> token_ids) {
    std::vector<std::uint8_t> bytes(token_ids.size() * sizeof(std::uint32_t));
    for (std::size_t i = 0; i < token_ids.size(); ++i) {
        const auto raw = static_cast<std::uint32_t>(token_ids[i]);
        std::memcpy(bytes.data() + i * sizeof(raw), &raw, sizeof(raw));
    }
    return bytes;
}

void AppendUint64LittleEndian(std::vector<std::uint8_t>& bytes, std::uint64_t value) {
    for (std::size_t i = 0; i < sizeof(value); ++i) {
        bytes.push_back(static_cast<std::uint8_t>((value >> (i * 8)) & 0xffu));
    }
}

std::uint64_t LocalBlockHashXxh3(std::span<const std::int32_t> token_ids) {
    const std::vector<std::uint8_t> token_bytes = TokenIdsToLittleEndianBytes(token_ids);
    return Xxh3Hash64WithSeed(token_bytes.data(), token_bytes.size(), kKvBlockXxh3Seed);
}

}  // namespace

std::uint64_t HashKvBlock(std::span<const std::int32_t> token_ids, std::optional<std::uint64_t> parent_hash) {
    std::uint64_t hash = kFnvOffsetBasis;
    MixByte(hash, parent_hash.has_value() ? 1 : 0);
    if (parent_hash.has_value()) {
        MixUint64(hash, *parent_hash);
    }
    MixUint64(hash, static_cast<std::uint64_t>(token_ids.size()));
    for (std::int32_t token_id : token_ids) {
        MixInt32(hash, token_id);
    }
    return hash;
}

std::uint64_t HashKvBlockXxh3(std::span<const std::int32_t> token_ids, std::optional<std::uint64_t> parent_hash) {
    const std::uint64_t local_hash = LocalBlockHashXxh3(token_ids);
    if (!parent_hash.has_value()) {
        return local_hash;
    }

    std::vector<std::uint8_t> combined;
    combined.reserve(2 * sizeof(std::uint64_t));
    AppendUint64LittleEndian(combined, *parent_hash);
    AppendUint64LittleEndian(combined, local_hash);
    return Xxh3Hash64WithSeed(combined.data(), combined.size(), kKvBlockXxh3Seed);
}

bool UseXxh3BlockHash() {
    return g_use_xxh3_block_hash.load();
}

void SetUseXxh3BlockHash(const bool enabled) {
    g_use_xxh3_block_hash.store(enabled);
}

std::uint64_t HashKvBlockForEvents(std::span<const std::int32_t> token_ids, std::optional<std::uint64_t> parent_hash) {
    if (UseXxh3BlockHash()) {
        return HashKvBlockXxh3(token_ids, parent_hash);
    }
    return HashKvBlock(token_ids, parent_hash);
}

}  // namespace tokenspeed
