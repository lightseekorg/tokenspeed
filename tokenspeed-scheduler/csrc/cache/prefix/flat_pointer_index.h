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
#include <utility>
#include <vector>

#include "cache/core/prefetch.h"

namespace tokenspeed {

// An open-addressed index over stable, externally owned objects. `empty`, given
// at construction, is the handle of no object: for list iterators, the owning
// list's end(), which stays valid for the list's lifetime and compares
// defined against its other iterators (a value-initialized iterator does
// not). Growing or backward-shifting slots never moves an object or its key.
// Callers erase its handle before destroying or changing its key. Slot order
// is deliberately not exposed as a traversal order. A slot caches its handle's
// hash: probing and erasing never hash a stored key again. The table doubles
// at 75% occupancy and never shrinks; with 16-byte slots (an 8-byte handle) it
// holds roughly 21 to 43 bytes per live key just before and after growing,
// and a rehash briefly holds both slot arrays.
template <typename T, typename Key, typename KeyOf, typename Hash>
class FlatPointerIndex {
public:
    explicit FlatPointerIndex(T empty) : empty_{empty} {}

    // The object's handle, or the empty handle.
    T Find(const Key& key) const {
        if (slots_.empty()) return empty_;
        return slots_[probe(key, Hash{}(key))].value;
    }

    // The handle of key's object and false, or, when none is indexed, the
    // handle make() returns, now indexed, and true. make creates the object,
    // whose key equals `key`, and must not use this index. Either way the key
    // is hashed and probed once. The table grows before make runs, so a make
    // that throws leaves every handle as it was.
    template <typename Make>
    std::pair<T, bool> FindOrInsert(const Key& key, Make&& make) {
        const auto hash = Hash{}(key);
        std::size_t position = 0;
        if (!slots_.empty()) {
            position = probe(key, hash);
            if (slots_[position].value != empty_) return {slots_[position].value, false};
        }
        if ((size_ + 1) * 4 > slots_.size() * 3) {
            grow();
            position = emptySlot(slots_, hash);
        }
        const T value = std::forward<Make>(make)();
        slots_[position] = Slot{hash, value};
        ++size_;
        return {value, true};
    }

    // Removes and returns the handle of `object`, which the caller holds and
    // still owns: `key` is that object's own key. Slots match by cached hash
    // and object identity, so no stored key is read or compared. Returns the
    // empty handle when the object is not indexed under `key`.
    template <typename Object>
    T ExtractObject(const Key& key, const Object& object) {
        if (slots_.empty()) return empty_;
        const T empty = empty_;
        const auto hash = Hash{}(key);
        const auto mask = slots_.size() - 1;
        for (auto position = hash & mask; slots_[position].value != empty; position = (position + 1) & mask) {
            if (slots_[position].hash == hash && std::addressof(*slots_[position].value) == std::addressof(object)) {
                return removeAt(position);
            }
        }
        return empty;
    }

    // A hint for a later Find or ExtractObject of `key`: prefetches the slot
    // its probe starts at and the next cache line, which a backward-shift
    // erase usually reads too. Reads no slot.
    void Prefetch(const Key& key) const noexcept {
        if (slots_.empty()) return;
        const auto hash = Hash{}(key);
        const auto mask = slots_.size() - 1;
        PrefetchObject(slots_[hash & mask]);
        PrefetchObject(slots_[(hash + kSlotsPerCacheLine) & mask]);
    }

    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    bool contains(const Key& key) const { return Find(key) != empty_; }

private:
    struct Slot {
        std::size_t hash;
        T value;
    };
    static constexpr std::size_t kSlotsPerCacheLine = sizeof(Slot) < 64 ? 64 / sizeof(Slot) : 1;
    // The slot holding key's handle, or else the empty slot key's probe ends
    // at. The table must not be empty; it always keeps an empty slot.
    std::size_t probe(const Key& key, std::size_t hash) const {
        const T empty = empty_;
        const auto mask = slots_.size() - 1;
        auto position = hash & mask;
        for (; slots_[position].value != empty; position = (position + 1) & mask) {
            const auto& slot = slots_[position];
            if (slot.hash == hash && KeyOf{}(slot.value) == key) break;
        }
        return position;
    }
    T removeAt(std::size_t hole) {
        // A local copy: stores into the slots may alias the member.
        const T empty = empty_;
        const T value = slots_[hole].value;
        const auto mask = slots_.size() - 1;
        for (auto next = (hole + 1) & mask; slots_[next].value != empty; next = (next + 1) & mask) {
            const auto home = slots_[next].hash & mask;
            // Move an entry only when the hole is on its probe path. This
            // also handles clusters crossing the end of the table.
            if (((next - home) & mask) >= ((next - hole) & mask)) {
                slots_[hole] = slots_[next];
                hole = next;
            }
        }
        slots_[hole] = Slot{0, empty};
        --size_;
        return value;
    }
    // The first empty slot from hash's home: where an absent key goes.
    std::size_t emptySlot(const std::vector<Slot>& slots, std::size_t hash) const {
        const T empty = empty_;
        const auto mask = slots.size() - 1;
        auto position = hash & mask;
        while (slots[position].value != empty) position = (position + 1) & mask;
        return position;
    }
    void grow() {
        const T empty = empty_;
        std::vector<Slot> next(slots_.empty() ? 16 : 2 * slots_.size(), Slot{0, empty});
        for (const auto& slot : slots_)
            if (slot.value != empty) next[emptySlot(next, slot.hash)] = slot;
        slots_.swap(next);
    }

    T empty_;
    std::vector<Slot> slots_;
    std::size_t size_{0};
};

}  // namespace tokenspeed
