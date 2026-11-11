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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <utility>
#include <vector>

#include "cache/core/cache_block_ref.h"
#include "utils.h"

namespace tokenspeed {

// The keys a planner's lookup tables take, each with a value no real key
// has: a location in the reserved null parent, or a null pointer.
template <typename Key>
struct FlatHashKey;

template <>
struct FlatHashKey<CacheBlockLocation> {
    static bool IsEmpty(CacheBlockLocation location) noexcept { return location.lcm_block_id == 0; }
    static std::uint64_t Bits(CacheBlockLocation location) noexcept {
        return static_cast<std::uint32_t>(location.lcm_block_id) |
               static_cast<std::uint64_t>(static_cast<std::uint32_t>(location.slot_index)) << 32U;
    }
};

template <typename T>
struct FlatHashKey<const T*> {
    static bool IsEmpty(const T* pointer) noexcept { return pointer == nullptr; }
    static std::uint64_t Bits(const T* pointer) noexcept { return reinterpret_cast<std::uintptr_t>(pointer); }
};

// An open-addressed map for one plan's lookups, its slots in one array from
// the given memory resource (a planning scratch lease). Linear probing and
// backward-shift erase keep no tombstones. Every key bit is mixed before
// masking: page IDs and pooled node addresses come in regular strides, which
// would pile into one cluster under a locality-preserving hash. Slot order is
// deliberately not exposed as a traversal order. Growing and erasing move
// values, so a returned pointer lasts until the next insertion, erase or
// clear. FlatPointerIndex (cache/prefix/flat_pointer_index.h) probes and
// erases the same way for the prefix index's long-lived handles, caching
// their hashes; this table keeps small keys and values inline for one plan.
template <typename Key, typename Mapped>
class FlatHashMap {
    using Traits = FlatHashKey<Key>;

public:
    explicit FlatHashMap(std::pmr::memory_resource* resource) : slots_{resource} {}

    // Makes room for `count` keys without growing.
    void reserve(std::size_t count) {
        if (count * 4 <= slots_.size() * 3) {
            return;
        }
        std::size_t capacity = slots_.empty() ? kMinimumCapacity : slots_.size();
        while (count * 4 > capacity * 3) {
            capacity *= 2;
        }
        rehash(capacity);
    }

    Mapped* Find(const Key& key) {
        if (size_ == 0) {
            return nullptr;
        }
        Slot& slot = slots_[position(key)];
        return Traits::IsEmpty(slot.key) ? nullptr : &slot.mapped;
    }
    const Mapped* Find(const Key& key) const {
        if (size_ == 0) {
            return nullptr;
        }
        const Slot& slot = slots_[position(key)];
        return Traits::IsEmpty(slot.key) ? nullptr : &slot.mapped;
    }
    bool contains(const Key& key) const { return Find(key) != nullptr; }

    // The value at `key`, inserted as `value` when absent, and whether it was.
    std::pair<Mapped*, bool> TryEmplace(const Key& key, const Mapped& value) {
        _assert(!Traits::IsEmpty(key), "a planning lookup key cannot be the empty key");
        if (slots_.empty()) {
            rehash(kMinimumCapacity);
        }
        std::size_t at = position(key);
        if (!Traits::IsEmpty(slots_[at].key)) {
            return {&slots_[at].mapped, false};
        }
        if ((size_ + 1) * 4 > slots_.size() * 3) {
            rehash(2 * slots_.size());
            at = position(key);
        }
        slots_[at] = Slot{key, value};
        ++size_;
        return {&slots_[at].mapped, true};
    }

    // The value at `key`, value-initialized when absent.
    Mapped& operator[](const Key& key) { return *TryEmplace(key, Mapped{}).first; }

    bool Erase(const Key& key) {
        if (size_ == 0) {
            return false;
        }
        std::size_t hole = position(key);
        if (Traits::IsEmpty(slots_[hole].key)) {
            return false;
        }
        const std::size_t mask = slots_.size() - 1;
        for (std::size_t next = (hole + 1) & mask; !Traits::IsEmpty(slots_[next].key); next = (next + 1) & mask) {
            const std::size_t home = hash(slots_[next].key) & mask;
            // Move an entry only when the hole is on its probe path, also
            // across the end of the table.
            if (((next - home) & mask) >= ((next - hole) & mask)) {
                slots_[hole] = slots_[next];
                hole = next;
            }
        }
        slots_[hole] = Slot{};
        --size_;
        return true;
    }

    // Keeps the capacity.
    void clear() {
        std::fill(slots_.begin(), slots_.end(), Slot{});
        size_ = 0;
    }

    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

private:
    static constexpr std::size_t kMinimumCapacity = 16;

    struct Slot {
        Key key{};
        [[no_unique_address]] Mapped mapped{};
    };

    // The splitmix64 finalizer: every input bit reaches the low bits.
    static std::size_t hash(const Key& key) noexcept {
        std::uint64_t bits = Traits::Bits(key);
        bits ^= bits >> 30U;
        bits *= 0xbf58476d1ce4e5b9ULL;
        bits ^= bits >> 27U;
        bits *= 0x94d049bb133111ebULL;
        bits ^= bits >> 31U;
        return static_cast<std::size_t>(bits);
    }

    // The slot holding `key`, else the empty slot that ends its probe.
    std::size_t position(const Key& key) const {
        const std::size_t mask = slots_.size() - 1;
        std::size_t at = hash(key) & mask;
        while (!Traits::IsEmpty(slots_[at].key) && !(slots_[at].key == key)) {
            at = (at + 1) & mask;
        }
        return at;
    }

    void rehash(std::size_t capacity) {
        std::pmr::vector<Slot> next(capacity, slots_.get_allocator());
        const std::size_t mask = capacity - 1;
        for (const Slot& slot : slots_) {
            if (Traits::IsEmpty(slot.key)) {
                continue;
            }
            std::size_t at = hash(slot.key) & mask;
            while (!Traits::IsEmpty(next[at].key)) {
                at = (at + 1) & mask;
            }
            next[at] = slot;
        }
        slots_.swap(next);
    }

    std::pmr::vector<Slot> slots_;
    std::size_t size_{0};
};

// The keys of a FlatHashMap with no values: as many bytes per slot as a key.
template <typename Key>
class FlatHashSet {
public:
    explicit FlatHashSet(std::pmr::memory_resource* resource) : keys_{resource} {}

    void reserve(std::size_t count) { keys_.reserve(count); }
    // Whether `key` was absent.
    bool insert(const Key& key) { return keys_.TryEmplace(key, Present{}).second; }
    bool contains(const Key& key) const { return keys_.contains(key); }
    bool Erase(const Key& key) { return keys_.Erase(key); }
    void clear() { keys_.clear(); }
    std::size_t size() const noexcept { return keys_.size(); }
    bool empty() const noexcept { return keys_.empty(); }

private:
    struct Present {};
    FlatHashMap<Key, Present> keys_;
};

}  // namespace tokenspeed
