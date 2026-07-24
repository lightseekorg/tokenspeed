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
#include <cstdint>
#include <exception>
#include <limits>
#include <utility>

#include "cache/block_pool_set.h"

namespace tokenspeed {

// Tracks only the component-wise aggregate. Each request owns one fixed-shape
// Account, so admission reads its own reservation directly without a
// request-id hash lookup. Accounts must die before their tracker.
class FlatReservationTracker {
public:
    class Account {
    public:
        Account(const Account&) = delete;
        Account& operator=(const Account&) = delete;

        Account(Account&& other) noexcept
            : tracker_{std::exchange(other.tracker_, nullptr)}, demand_{std::move(other.demand_)} {}

        Account& operator=(Account&& other) noexcept {
            if (this != &other) {
                release();
                tracker_ = std::exchange(other.tracker_, nullptr);
                demand_ = std::move(other.demand_);
            }
            return *this;
        }

        ~Account() noexcept { release(); }

        const PoolDemand& Demand() const noexcept {
            if (tracker_ == nullptr) {
                std::terminate();
            }
            return demand_;
        }

        bool Empty() const noexcept { return !Demand().AnyPositive(); }

        // Fixed-shape PoolDemand move assignment is noexcept. All aggregate
        // arithmetic is preflighted before either side is mutated.
        void Set(PoolDemand reservation) noexcept {
            if (tracker_ == nullptr) {
                std::terminate();
            }
            tracker_->set(*this, std::move(reservation));
        }

        void Clear() noexcept {
            if (tracker_ != nullptr) {
                tracker_->clear(*this);
            }
        }

    private:
        friend class FlatReservationTracker;

        explicit Account(FlatReservationTracker& tracker) : tracker_{&tracker}, demand_(tracker.total_.Size(), 0) {
            ++tracker_->account_count_;
        }

        void release() noexcept {
            if (tracker_ == nullptr) {
                return;
            }
            tracker_->clear(*this);
            if (tracker_->account_count_ == 0) {
                std::terminate();
            }
            --tracker_->account_count_;
            tracker_ = nullptr;
        }

        FlatReservationTracker* tracker_{};
        PoolDemand demand_;
    };

    explicit FlatReservationTracker(std::size_t pool_count) : total_(pool_count, 0) {}

    FlatReservationTracker(const FlatReservationTracker&) = delete;
    FlatReservationTracker& operator=(const FlatReservationTracker&) = delete;
    FlatReservationTracker(FlatReservationTracker&&) = delete;
    FlatReservationTracker& operator=(FlatReservationTracker&&) = delete;

    ~FlatReservationTracker() noexcept {
        if (account_count_ != 0 || total_.AnyPositive()) {
            std::terminate();
        }
    }

    Account MakeAccount() { return Account{*this}; }

    const PoolDemand& Total() const noexcept { return total_; }
    bool Empty() const noexcept { return !total_.AnyPositive(); }

private:
    void set(Account& account, PoolDemand reservation) noexcept {
        if (account.tracker_ != this || account.demand_.Size() != total_.Size() ||
            reservation.Size() != total_.Size()) {
            std::terminate();
        }
        for (PoolIndex i = 0; i < total_.Size(); ++i) {
            const std::int64_t next = static_cast<std::int64_t>(total_[i]) - account.demand_[i] + reservation[i];
            if (next < 0 || next > std::numeric_limits<std::int32_t>::max()) {
                std::terminate();
            }
        }
        for (PoolIndex i = 0; i < total_.Size(); ++i) {
            total_[i] = total_[i] - account.demand_[i] + reservation[i];
        }
        account.demand_ = std::move(reservation);
    }

    void clear(Account& account) noexcept {
        if (account.tracker_ != this || account.demand_.Size() != total_.Size()) {
            std::terminate();
        }
        for (PoolIndex i = 0; i < total_.Size(); ++i) {
            if (total_[i] < account.demand_[i]) {
                std::terminate();
            }
        }
        for (PoolIndex i = 0; i < total_.Size(); ++i) {
            total_[i] -= account.demand_[i];
            account.demand_[i] = 0;
        }
    }

    PoolDemand total_;
    std::size_t account_count_{0};
};

}  // namespace tokenspeed
