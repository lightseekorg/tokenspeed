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
#include <functional>
#include <utility>
#include <vector>

namespace tokenspeed {

// An open-addressed index over stable, externally owned objects. T{} is the
// empty handle. Growing or backward-shifting slots never moves an object or
// its key. Callers erase its handle before destroying or changing its key.
// Slot order is deliberately not exposed as a traversal order.
template <typename T, typename Key, typename KeyOf, typename Hash>
class FlatPointerIndex {
public:
    T Find(const Key& key) const {
        if (slots_.empty()) return {};
        const auto hash = Hash{}(key);
        const auto position = find(key, hash);
        return position == slots_.size() ? T{} : slots_[position].value;
    }

    bool Insert(T value) {
        const Key& key = KeyOf{}(value);
        const auto hash = Hash{}(key);
        if (!slots_.empty() && find(key, hash) != slots_.size()) return false;
        if ((size_ + 1) * 4 > slots_.size() * 3) grow();
        put(slots_, Slot{hash, value});
        ++size_;
        return true;
    }

    bool Erase(const Key& key) {
        if (slots_.empty()) return false;
        std::size_t hole = find(key, Hash{}(key));
        if (hole == slots_.size()) return false;
        const auto mask = slots_.size() - 1;
        for (auto next = (hole + 1) & mask; slots_[next].value != T{}; next = (next + 1) & mask) {
            const auto home = slots_[next].hash & mask;
            // Move an entry only when the hole is on its probe path. This
            // also handles clusters crossing the end of the table.
            if (((next - home) & mask) >= ((next - hole) & mask)) {
                slots_[hole] = slots_[next];
                hole = next;
            }
        }
        slots_[hole] = {};
        --size_;
        return true;
    }

    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    bool contains(const Key& key) const { return Find(key) != T{}; }

private:
    struct Slot {
        std::size_t hash{0};
        T value{};
    };
    std::size_t find(const Key& key, std::size_t hash) const {
        const auto mask = slots_.size() - 1;
        for (auto position = hash & mask; slots_[position].value != T{}; position = (position + 1) & mask) {
            const auto& slot = slots_[position];
            if (slot.hash == hash && KeyOf{}(slot.value) == key) return position;
        }
        return slots_.size();
    }
    static void put(std::vector<Slot>& slots, Slot value) {
        const auto mask = slots.size() - 1;
        auto position = value.hash & mask;
        while (slots[position].value != T{}) position = (position + 1) & mask;
        slots[position] = value;
    }
    void grow() {
        std::vector<Slot> next(slots_.empty() ? 16 : 2 * slots_.size());
        for (const auto& slot : slots_)
            if (slot.value != T{}) put(next, slot);
        slots_.swap(next);
    }

    std::vector<Slot> slots_;
    std::size_t size_{0};
};

}  // namespace tokenspeed
