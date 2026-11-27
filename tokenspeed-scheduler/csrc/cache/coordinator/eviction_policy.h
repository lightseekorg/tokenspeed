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
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace tokenspeed {

// Coefficients are deployment inputs, not measured latency. Widen token
// positions before evaluating the quadratic term.
struct PrefillCost {
    double linear;
    double quadratic;

    void Validate() const {
        if (!std::isfinite(linear) || linear <= 0 || !std::isfinite(quadratic) || quadratic < 0) {
            throw std::invalid_argument("GDS prefill cost requires finite D > 0 and E >= 0");
        }
    }
    double Between(double from, double to) const { return (to - from) * (linear + quadratic * (to + from) / 200000.0); }
};

struct GdsEvictionStats {
    std::uint64_t evicted_units{0};
    std::uint64_t retention_rejections{0};
    std::uint64_t demoted_units{0};
    std::uint64_t mixed_tier_units{0};
    std::uint64_t orphan_blocks{0};
    std::uint64_t fallback_blocks{0};
    std::uint64_t fallback_units{0};
    double watermark{0};
    double host_watermark{0};
    std::uint64_t logical_evictions{0};
    std::uint64_t logical_rejections{0};
    std::uint64_t device_copies_released{0};
    std::uint64_t host_copies_reserved{0};
    std::uint64_t host_copy_rejections{0};
    std::uint64_t host_clean_copies_released{0};
    std::uint64_t disk_recoverable_units{0};
    std::uint64_t disk_route_losses{0};
    std::uint64_t committed_plans{0};
    std::uint64_t checkpoint_planned{0};
    std::uint64_t checkpoint_allocated{0};
    std::uint64_t checkpoint_materialized{0};
    std::uint64_t checkpoint_published{0};
    std::uint64_t checkpoint_cancelled{0};
    std::uint64_t checkpoint_extra_blocks{0};
    std::uint64_t checkpoint_chunk_end_fallbacks{0};
    std::uint64_t checkpoint_fresh_rejected{0};
    std::uint64_t checkpoint_rejected_extra_blocks{0};
    std::uint64_t checkpoint_rejected_unmeasured{0};
    std::uint64_t planning_nanoseconds{0};
    std::uint64_t ready_units{0};
    std::uint64_t device_ready_units{0};
    std::uint64_t pending_units{0};
    std::uint64_t broken_units{0};
    std::uint64_t dependency_points{0};
    std::uint64_t dependency_ranges{0};
    std::uint64_t dependency_leaves{0};

    void Add(const GdsEvictionStats& other) {
        evicted_units += other.evicted_units;
        retention_rejections += other.retention_rejections;
        demoted_units += other.demoted_units;
        mixed_tier_units += other.mixed_tier_units;
        orphan_blocks += other.orphan_blocks;
        fallback_blocks += other.fallback_blocks;
        fallback_units += other.fallback_units;
        watermark = std::max(watermark, other.watermark);
        host_watermark = std::max(host_watermark, other.host_watermark);
        logical_evictions += other.logical_evictions;
        logical_rejections += other.logical_rejections;
        device_copies_released += other.device_copies_released;
        host_copies_reserved += other.host_copies_reserved;
        host_copy_rejections += other.host_copy_rejections;
        host_clean_copies_released += other.host_clean_copies_released;
        committed_plans += other.committed_plans;
        planning_nanoseconds += other.planning_nanoseconds;
    }
};

struct PrefixCacheEvictionConfig {
    enum class Policy { kLru, kGds };
    Policy policy;
    std::optional<PrefillCost> prefill_cost;
    // Per physical Host/Disk block, in the same units as prefill_cost.
    // Empty means there is no Disk read-cost model.
    std::vector<double> disk_block_costs;

    void Validate() const {
        if (policy == Policy::kGds && prefill_cost) {
            prefill_cost->Validate();
            for (double cost : disk_block_costs)
                if (!std::isfinite(cost) || cost <= 0)
                    throw std::invalid_argument("GDS Disk block costs must be finite and positive");
            return;
        }
        if (policy == Policy::kLru && !prefill_cost && disk_block_costs.empty()) return;
        throw std::invalid_argument("eviction policy and explicit GDS cost must agree");
    }
    static PrefixCacheEvictionConfig Lru() { return {Policy::kLru, std::nullopt}; }
    static PrefixCacheEvictionConfig Gds(PrefillCost cost) {
        cost.Validate();
        return {Policy::kGds, cost};
    }
    static PrefixCacheEvictionConfig Gds(PrefillCost cost, std::vector<double> disk_costs) {
        PrefixCacheEvictionConfig result{Policy::kGds, cost, std::move(disk_costs)};
        result.Validate();
        return result;
    }
    static PrefixCacheEvictionConfig Parse(const std::string& name, std::optional<PrefillCost> cost) {
        if (name == "lru" && !cost) return Lru();
        if (name == "gds" && cost) return Gds(*cost);
        throw std::invalid_argument("prefix-cache-evict-policy must be lru, or gds with explicit prefill cost D E");
    }
};

}  // namespace tokenspeed
