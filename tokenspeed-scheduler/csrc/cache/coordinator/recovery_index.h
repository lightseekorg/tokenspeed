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
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "cache/core/cache_types.h"

namespace tokenspeed {
class CacheStore;

struct RecoveryResidencyEvent {
    CacheKey key;
    CacheTier tier;
    bool present;
};
struct RecoveryPromiseEvent {
    CacheKey key;
    bool reserved;
};
struct CheckpointUseEvent {
    enum class Cause { kAcquire, kPeerUse, kLastRelease };
    ContentHash hash;
    Cause cause;
};

// Persistent, non-owning recovery dependencies. Shared aligned ranges retain
// residency summaries; they never own cache blocks or expand per-point history.
class RecoveryIndex {
public:
    struct Path {
        CacheKey first;
        CacheKey last;
    };
    struct Leaf {
        CacheKey key;
        std::optional<CacheKey> previous;
    };
    struct Entry {
        bool ready{false};
        bool device_ready{false};
        bool pending{false};
        bool disk_recoverable{false};
        double disk_read_cost{0};
        std::vector<Path> device_paths;
        std::vector<Path> host_paths;
        std::vector<Path> disk_paths;
        bool Available() const { return ready || pending || disk_recoverable; }
    };
    struct Stats {
        std::size_t points;
        std::size_t ranges;
        std::size_t leaves;
        std::uint64_t route_updates;
        std::uint64_t range_visits;
        std::size_t ready;
        std::size_t device_ready;
        std::size_t pending;
        std::size_t broken;
        std::size_t disk_recoverable;
        std::uint64_t disk_route_losses;
    };
    // A transaction's residency overlay. Queries use the shared range summaries;
    // only ancestors of changed leaves acquire shadow counts. No blocks are held.
    class View {
    public:
        struct Change {
            CacheKey key;
            CacheTier tier;
            bool present;
        };
        explicit View(const RecoveryIndex& index);
        ~View();
        View(const View& other);
        View& operator=(const View& other);
        View(View&&) noexcept;
        View& operator=(View&&) noexcept;
        // Update only routes watching changed ranges. Unchanged routes are read
        // directly from the resident index; queries never mutate the view.
        void Apply(std::span<const Change> changes);
        const Entry& At(const ContentHash& hash) const;
        Entry Resolve(const ContentHash& hash, bool allow_device_copy) const;
        bool Present(const CacheKey& key, CacheTier tier) const;

    private:
        struct State;
        std::unique_ptr<State> state_;
    };
    class Registration {
    public:
        Registration() = default;
        Registration(RecoveryIndex& index, const ContentHash& hash);
        Registration(const Registration&) = delete;
        Registration& operator=(const Registration&) = delete;
        Registration(Registration&& other) noexcept;
        Registration& operator=(Registration&& other) noexcept;
        ~Registration();
        void Reset() noexcept;

    private:
        RecoveryIndex* index_{nullptr};
        ContentHash hash_{};
    };
    // Owned by the canonical index entry, including clear and destruction.
    class Residency {
    public:
        Residency() = default;
        Residency(RecoveryIndex& index, const CacheKey& key, CacheTier tier);
        Residency(const Residency&) = delete;
        Residency& operator=(const Residency&) = delete;
        Residency(Residency&& other) noexcept;
        Residency& operator=(Residency&& other) noexcept;
        ~Residency();
        void Reset() noexcept;

    private:
        RecoveryIndex* index_{nullptr};
        CacheKey key_{};
        CacheTier tier_{CacheTier::kDevice};
    };

    explicit RecoveryIndex(const CacheStore& store);
    ~RecoveryIndex();
    RecoveryIndex(const RecoveryIndex&) = delete;
    RecoveryIndex& operator=(const RecoveryIndex&) = delete;
    Registration Register(const ContentHash& hash, std::int32_t tokens);
    void Apply(const RecoveryResidencyEvent& event);
    void Apply(const RecoveryPromiseEvent& event);
    const Entry& At(const ContentHash& hash) const;
    std::vector<Leaf> Leaves() const;
    Stats GetStats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    void Remove(const ContentHash& hash) noexcept;
};
}  // namespace tokenspeed
