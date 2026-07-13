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
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "cache/block_pool_set.h"

namespace tokenspeed {

// Owns per-request decode reservations and their component-wise aggregate.
// Request admission prepares one zero-demand node for the active lifetime;
// scheduling then updates it allocation-free, so admission can charge every
// other request with one hash lookup plus O(pool_count) arithmetic instead of
// rescanning the whole batch.
class FlatReservationLedger {
public:
    explicit FlatReservationLedger(std::size_t pool_count) : total_(pool_count, 0) {}

    // Prepare zero-demand nodes before admitting a request.
    // Nodes remain for the active request lifetime, so later updates only
    // update existing fixed-shape storage; terminal lifecycle paths Erase().
    void ReservePrepared(std::size_t additional) {
        if (additional > reservations_.max_size() - reservations_.size()) {
            throw std::length_error("flat reservation prepared-node count exceeds max_size");
        }
        reservations_.reserve(reservations_.size() + additional);
    }

    bool Prepare(const std::string& request_id) {
        auto [it, inserted] = reservations_.try_emplace(request_id, total_.Size(), 0);
        return inserted;
    }

    // Scheduling-time update for a key prepared at request admission. It does
    // no lookup-node or value allocation and is fail-stop on an invariant
    // violation, so it cannot unwind after another request in the batch has
    // already changed FSM/cache state.
    void SetPrepared(const std::string& request_id, PoolDemand reservation) noexcept {
        auto existing = reservations_.find(request_id);
        if (existing == reservations_.end() || reservation.Size() != total_.Size() ||
            existing->second.Size() != total_.Size()) {
            std::terminate();
        }
        for (PoolIndex i = 0; i < total_.Size(); ++i) {
            const std::int64_t next = static_cast<std::int64_t>(total_[i]) - existing->second[i] + reservation[i];
            if (next < 0 || next > std::numeric_limits<std::int32_t>::max()) {
                std::terminate();
            }
        }
        for (PoolIndex i = 0; i < total_.Size(); ++i) {
            total_[i] = total_[i] - existing->second[i] + reservation[i];
        }
        existing->second = std::move(reservation);
    }

    void ClearPrepared(const std::string& request_id) noexcept {
        auto existing = reservations_.find(request_id);
        if (existing == reservations_.end() || existing->second.Size() != total_.Size()) {
            std::terminate();
        }
        for (PoolIndex i = 0; i < total_.Size(); ++i) {
            if (total_[i] < existing->second[i]) {
                std::terminate();
            }
        }
        for (PoolIndex i = 0; i < total_.Size(); ++i) {
            total_[i] -= existing->second[i];
            existing->second[i] = 0;
        }
    }

    bool Erase(const std::string& request_id) noexcept {
        auto existing = reservations_.find(request_id);
        if (existing == reservations_.end()) {
            return false;
        }
        total_.SubtractInPlace(existing->second);
        reservations_.erase(existing);
        return true;
    }

    const PoolDemand* Find(const std::string& request_id) const {
        const auto reservation = reservations_.find(request_id);
        return reservation == reservations_.end() ? nullptr : &reservation->second;
    }

    const PoolDemand& Total() const noexcept { return total_; }

    bool Empty() const noexcept { return !total_.AnyPositive(); }

private:
    static_assert(std::is_nothrow_move_assignable_v<PoolDemand>);

    std::unordered_map<std::string, PoolDemand> reservations_;
    PoolDemand total_;
};

}  // namespace tokenspeed
