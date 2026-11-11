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

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <memory_resource>

#include "utils.h"

namespace tokenspeed {

// Reusable allocation storage, never reusable occupancy state. Each planning
// lease gets independent bytes; its containers must die before the lease.
// Three lazily allocated banks retain at most 768 KiB per pool: an admission
// planner's lookup tables, its shadow and its pre-scan's shadow live at once.
// Larger or more deeply nested plans use the normal allocator and return that
// memory at the end of the plan. Only the scheduler control thread uses this
// owner.
class PlanningScratch {
    static constexpr std::size_t kBankBytes = 256 * 1024;
    struct Bank {
        std::unique_ptr<std::byte[]> bytes;
        bool leased{false};
    };

public:
    class Lease final : public std::pmr::memory_resource {
    public:
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        ~Lease() override {
            if (bank_) bank_->leased = false;
        }

    private:
        friend class PlanningScratch;
        explicit Lease(const PlanningScratch& owner) {
            for (auto& bank : owner.banks_) {
                if (!bank.leased) {
                    bank.leased = true;
                    bank_ = &bank;
                    break;
                }
            }
        }
        void* do_allocate(std::size_t bytes, std::size_t alignment) override {
            if (bank_ && bytes <= kBankBytes) {
                if (!bank_->bytes) bank_->bytes.reset(new std::byte[kBankBytes]);
                void* next = bank_->bytes.get() + used_;
                std::size_t remaining = kBankBytes - used_;
                if (std::align(alignment, bytes, next, remaining)) {
                    used_ = static_cast<std::byte*>(next) - bank_->bytes.get() + bytes;
                    return next;
                }
            }
            return std::pmr::new_delete_resource()->allocate(bytes, alignment);
        }
        void do_deallocate(void* pointer, std::size_t bytes, std::size_t alignment) override {
            if (bank_ && bank_->bytes) {
                const auto base = reinterpret_cast<std::uintptr_t>(bank_->bytes.get());
                const auto address = reinterpret_cast<std::uintptr_t>(pointer);
                if (address >= base && address - base < kBankBytes) return;
            }
            std::pmr::new_delete_resource()->deallocate(pointer, bytes, alignment);
        }
        bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override { return this == &other; }

        Bank* bank_{nullptr};
        std::size_t used_{0};
    };

    Lease Acquire() const { return Lease{*this}; }
    ~PlanningScratch() {
        for (const auto& bank : banks_) FatalCheck(!bank.leased, "planning scratch outlives all its leases");
    }

private:
    mutable std::array<Bank, 3> banks_;
};

}  // namespace tokenspeed
