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
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cache/block_pool.h"

namespace tokenspeed {

using PoolIndex = std::size_t;

class Scheduler;

// Canonical scheduler-side description of one homogeneous flat block pool.
// Device tensor ownership remains in Python; this config only sizes metadata
// and provides byte weights for admission/observability.
struct FlatBlockPoolConfig {
    std::string pool_id;
    std::int32_t total_blocks{0};  // Includes reserved null page 0.
    std::int64_t bytes_per_block{0};
};

// Compact fixed-shape non-negative counts. Tags keep unrelated index domains
// type-safe while sharing the inline-storage and checked-arithmetic machinery.
template <typename Tag>
class CompactCounts {
public:
    // Eight int32 entries fit in one 32-byte inline array. Larger heterogeneous
    // layouts retain the same contract through the heap fallback.
    static constexpr std::size_t kInlineCapacity = 8;

    CompactCounts() = default;
    explicit CompactCounts(std::size_t size, std::int32_t value = 0) {
        if (value < 0) {
            throw std::invalid_argument("component counts must be non-negative");
        }
        Initialize(size, value);
    }
    CompactCounts(std::initializer_list<std::int32_t> values) {
        const std::span<const std::int32_t> view{values.begin(), values.size()};
        ValidateNonNegative(view);
        Initialize(view);
    }
    explicit CompactCounts(std::vector<std::int32_t> values) {
        const std::span<const std::int32_t> view{values};
        ValidateNonNegative(view);
        Initialize(view);
    }

    CompactCounts(const CompactCounts& other) { Initialize(other.Values()); }
    CompactCounts& operator=(const CompactCounts& other) {
        if (this != &other) {
            CompactCounts copy{other};
            Swap(copy);
        }
        return *this;
    }

    CompactCounts(CompactCounts&& other) noexcept
        : inline_values_{other.inline_values_},
          heap_values_{std::move(other.heap_values_)},
          size_{std::exchange(other.size_, 0)} {}

    CompactCounts& operator=(CompactCounts&& other) noexcept {
        if (this != &other) {
            inline_values_ = other.inline_values_;
            heap_values_ = std::move(other.heap_values_);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    std::size_t Size() const noexcept { return size_; }
    bool Empty() const noexcept { return size_ == 0; }
    std::int32_t operator[](PoolIndex index) const noexcept { return Data()[index]; }
    std::int32_t& operator[](PoolIndex index) noexcept { return Data()[index]; }
    std::span<const std::int32_t> Values() const noexcept { return {Data(), size_}; }

    void AddInPlace(const CompactCounts& other) {
        RequireSameSize(other);
        for (PoolIndex i = 0; i < Size(); ++i) {
            if ((*this)[i] > std::numeric_limits<std::int32_t>::max() - other[i]) {
                throw std::overflow_error("component-count addition overflow");
            }
        }
        for (PoolIndex i = 0; i < Size(); ++i) {
            (*this)[i] += other[i];
        }
    }

    void SubtractInPlace(const CompactCounts& other) {
        RequireSameSize(other);
        for (PoolIndex i = 0; i < Size(); ++i) {
            if ((*this)[i] < other[i]) {
                throw std::underflow_error("component-count subtraction would become negative");
            }
        }
        for (PoolIndex i = 0; i < Size(); ++i) {
            (*this)[i] -= other[i];
        }
    }

    bool FitsWithin(const CompactCounts& capacity) const {
        RequireSameSize(capacity);
        for (PoolIndex i = 0; i < Size(); ++i) {
            if ((*this)[i] > capacity[i]) {
                return false;
            }
        }
        return true;
    }

    bool AnyPositive() const noexcept {
        return std::ranges::any_of(Values(), [](std::int32_t value) { return value > 0; });
    }

    friend bool operator==(const CompactCounts& lhs, const CompactCounts& rhs) noexcept {
        return std::ranges::equal(lhs.Values(), rhs.Values());
    }

private:
    void Initialize(std::size_t size, std::int32_t value) {
        if (size > kInlineCapacity) {
            auto values = std::make_unique<std::int32_t[]>(size);
            std::fill_n(values.get(), size, value);
            heap_values_ = std::move(values);
        } else {
            std::fill_n(inline_values_.begin(), size, value);
        }
        size_ = size;
    }

    void Initialize(std::span<const std::int32_t> values) {
        if (values.size() > kInlineCapacity) {
            auto copy = std::make_unique<std::int32_t[]>(values.size());
            std::ranges::copy(values, copy.get());
            heap_values_ = std::move(copy);
        } else {
            std::ranges::copy(values, inline_values_.begin());
        }
        size_ = values.size();
    }

    const std::int32_t* Data() const noexcept {
        return size_ > kInlineCapacity ? heap_values_.get() : inline_values_.data();
    }

    std::int32_t* Data() noexcept { return size_ > kInlineCapacity ? heap_values_.get() : inline_values_.data(); }

    void Swap(CompactCounts& other) noexcept {
        inline_values_.swap(other.inline_values_);
        heap_values_.swap(other.heap_values_);
        std::swap(size_, other.size_);
    }

    void RequireSameSize(const CompactCounts& other) const {
        if (Size() != other.Size()) {
            throw std::invalid_argument("component-count shape mismatch");
        }
    }

    static void ValidateNonNegative(std::span<const std::int32_t> values) {
        if (std::ranges::any_of(values, [](std::int32_t value) { return value < 0; })) {
            throw std::invalid_argument("component counts must be non-negative");
        }
    }

    std::array<std::int32_t, kInlineCapacity> inline_values_{};
    std::unique_ptr<std::int32_t[]> heap_values_;
    std::size_t size_{0};
};

struct PoolDemandTag {};
using PoolDemand = CompactCounts<PoolDemandTag>;

struct BlockPoolSnapshot {
    std::string pool_id;
    std::int32_t total_blocks{0};
    std::int32_t usable_blocks{0};
    std::int32_t free_blocks{0};
    std::int32_t active_blocks{0};
    std::int32_t cached_evictable_blocks{0};
    std::int32_t pinned_cached_blocks{0};
    std::int32_t reserved_blocks{0};
    std::int64_t bytes_per_block{0};
};

// Immutable canonical registry of homogeneous BlockPools. Pools are heap-owned
// so BlockControl's owner link remains stable for the full scheduler lifetime
// even though the registry itself uses a contiguous canonical index.
class BlockPoolSet {
public:
    explicit BlockPoolSet(std::vector<FlatBlockPoolConfig> configs) { Initialize(std::move(configs)); }

    BlockPoolSet(const BlockPoolSet&) = delete;
    BlockPoolSet& operator=(const BlockPoolSet&) = delete;
    BlockPoolSet(BlockPoolSet&&) = delete;
    BlockPoolSet& operator=(BlockPoolSet&&) = delete;

    // Construction-time schema is immutable and remains cross-thread readable.
    std::size_t Size() const noexcept { return pools_.size(); }
    BlockPool& Pool(PoolIndex index) { return *pools_.at(index); }
    const BlockPool& Pool(PoolIndex index) const { return *pools_.at(index); }

    std::uint64_t Generation() const noexcept { return generation_; }

    PoolIndex IndexOf(std::string_view pool_id) const {
        auto it = pool_index_.find(std::string{pool_id});
        if (it == pool_index_.end()) {
            throw std::out_of_range("unknown flat block pool id: " + std::string{pool_id});
        }
        return it->second;
    }

    const std::string& PoolId(PoolIndex index) const { return configs_.at(index).pool_id; }
    const FlatBlockPoolConfig& Config(PoolIndex index) const { return configs_.at(index); }

    PoolDemand FreeBlocks() const {
        PoolDemand out(Size(), 0);
        for (PoolIndex i = 0; i < Size(); ++i) {
            out[i] = Pool(i).NumFreeBlocks();
        }
        return out;
    }

    std::int64_t BytesFor(const PoolDemand& demand) const {
        if (demand.Size() != Size()) {
            throw std::invalid_argument("PoolDemand shape does not match BlockPoolSet");
        }
        std::int64_t total = 0;
        for (PoolIndex i = 0; i < Size(); ++i) {
            const std::int64_t bytes = Config(i).bytes_per_block;
            if (bytes != 0 && demand[i] > (std::numeric_limits<std::int64_t>::max() - total) / bytes) {
                throw std::overflow_error("flat block-pool byte demand overflow");
            }
            total += static_cast<std::int64_t>(demand[i]) * bytes;
        }
        return total;
    }

    std::vector<BlockPoolSnapshot> Snapshot() const {
        std::vector<BlockPoolSnapshot> out;
        out.reserve(Size());
        for (PoolIndex i = 0; i < Size(); ++i) {
            const BlockPool& pool = Pool(i);
            const std::int32_t usable = Config(i).total_blocks - 1;
            const std::int32_t free = pool.NumFreeBlocks();
            out.push_back(BlockPoolSnapshot{
                .pool_id = Config(i).pool_id,
                .total_blocks = Config(i).total_blocks,
                .usable_blocks = usable,
                .free_blocks = free,
                .active_blocks = usable - free,
                .cached_evictable_blocks = pool.NumCachedFreeBlocks(),
                .pinned_cached_blocks = pool.NumPinnedCachedBlocks(),
                .bytes_per_block = Config(i).bytes_per_block,
            });
        }
        return out;
    }

    bool IsQuiescent() const {
        for (PoolIndex i = 0; i < Size(); ++i) {
            if (!Pool(i).IsQuiescent()) {
                return false;
            }
        }
        return true;
    }

    std::uint64_t ResetQuiescent() {
        // Validate every independent pool before mutating any of them.
        if (!IsQuiescent()) {
            throw std::logic_error("cannot reset flat block pool set with live refs");
        }
        if (generation_ == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("flat block pool generation exhausted");
        }
        for (PoolIndex i = 0; i < Size(); ++i) {
            Pool(i).ResetQuiescent();
        }
        return ++generation_;
    }

private:
    void Initialize(std::vector<FlatBlockPoolConfig> configs) {
        if (configs.empty()) {
            throw std::invalid_argument("BlockPoolSet requires at least one pool");
        }
        std::ranges::sort(configs, {}, &FlatBlockPoolConfig::pool_id);

        pools_.reserve(configs.size());
        configs_.reserve(configs.size());
        for (FlatBlockPoolConfig& config : configs) {
            ValidateConfig(config);
            if (!configs_.empty() && configs_.back().pool_id == config.pool_id) {
                throw std::invalid_argument("duplicate flat block pool id: " + config.pool_id);
            }
            const PoolIndex index = configs_.size();
            pool_index_.emplace(config.pool_id, index);
            pools_.push_back(std::make_unique<BlockPool>(config.total_blocks));
            configs_.push_back(std::move(config));
        }
    }
    static void ValidateConfig(const FlatBlockPoolConfig& config) {
        if (config.pool_id.empty()) {
            throw std::invalid_argument("flat block pool id must not be empty");
        }
        if (config.total_blocks < 2) {
            throw std::invalid_argument("flat block pool must contain null page 0 and at least one usable block");
        }
        if (config.bytes_per_block < 0) {
            throw std::invalid_argument("flat block pool bytes_per_block must be non-negative");
        }
    }

    std::vector<FlatBlockPoolConfig> configs_;
    std::vector<std::unique_ptr<BlockPool>> pools_;
    std::unordered_map<std::string, PoolIndex> pool_index_;
    std::uint64_t generation_{0};

    friend class Scheduler;
};

}  // namespace tokenspeed
