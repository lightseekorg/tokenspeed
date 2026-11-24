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

#include "cache/coordinator/gds_planner.h"
#include "core/profile_span.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "cache/coordinator/gds_coverage.h"
#include "cache/coordinator/gds_tiered.h"

namespace tokenspeed {
namespace {
struct Component {
    std::uint32_t group;
    CacheBlockLocation location;
    std::optional<CacheKey> key;
    bool demoted;
    bool reclaim;
    std::optional<std::size_t> position;
};
struct Unit {
    CheckpointTree::RecoveryPoint point;
    std::vector<RecoveryInterval> ranges;
    bool device_ready;
    bool ready;
    bool pending;
    bool held;
    bool removed{false};
    std::int32_t fallback_tokens{0};
};
}  // namespace

std::optional<std::vector<VictimSet>> PlanGdsEviction(const CacheStore& store,
                                                      const AdmissionCandidateConstraints& constraints,
                                                      std::span<const BlockPlacementDemand> demands,
                                                      CacheEvictionPlan& capacity,
                                                      std::pmr::memory_resource* resource) {
    ProfileSpan profile_span{"Gds::PlanAttempt"};
    if (store.HostPool()) {
        auto result = PlanTieredDeviceEviction(store, constraints, demands, resource);
        if (result)
            for (const auto& set : *result)
                for (const auto& member : set.members) capacity.Remove(member.group_id, member.location);
        return result;
    }
    const auto& pool = store.DevicePool();
    const auto started = std::chrono::steady_clock::now();
    const auto* preview = constraints.publication.get();
    const RecoveryLayout layout{preview ? preview->Leaves(store) : store.RecoveryDependencies().Leaves()};
    RecoveryIndex::View view{store.RecoveryDependencies()};
    if (preview) preview->Apply(view);
    std::vector<Component> components;
    std::unordered_map<CacheBlockLocation, std::size_t, CacheBlockLocationHash> by_location;
    CacheEvictionCandidates candidates{store.Groups(), pool, CheckpointAging{}};
    while (auto candidate =
               candidates.Next(constraints.request_reclaim_candidates, constraints.protected_locations,
                               constraints.request_reclaim_locations, [](CacheBlockLocation) { return false; })) {
        const auto& index = store.Groups()[candidate->group_id].Index();
        const auto metadata = index.MetadataFor(pool, candidate->location);
        if (!by_location.emplace(candidate->location, components.size()).second) continue;
        components.push_back({candidate->group_id, candidate->location, index.KeyOf(pool, candidate->location),
                              metadata && (metadata->demoted || (metadata->session && metadata->session->demoted)),
                              constraints.request_reclaim_locations.contains(candidate->location),
                              index.KeyOf(pool, candidate->location)
                                  ? layout.Position(*index.KeyOf(pool, candidate->location))
                                  : std::nullopt});
    }
    std::vector<Unit> units;
    std::unordered_map<ContentHash, std::size_t> by_hash;
    for (auto& point : preview ? preview->Points(store) : store.Checkpoints().RecoveryPoints()) {
        const auto entry =
            preview ? preview->Resolve(point.hash, view, false) : store.RecoveryDependencies().At(point.hash);
        const bool held = entry.pending || (preview && preview->Holds(point.hash));
        Unit unit{std::move(point),   layout.Intervals(entry.device_paths),
                  entry.device_ready, entry.ready,
                  entry.pending,      held};
        // Live State owners exclude the unit. Other owned pages are absent
        // from the candidate census and cannot contribute reclaimable size.
        for (std::size_t i = 0; i < store.Groups().size(); ++i) {
            if (!store.GroupIsState(i)) continue;
            const CacheKey key{.group_id = static_cast<std::uint32_t>(i), .content_hash = unit.point.hash};
            if (auto location = store.LocationOf(pool, i, key); location && !by_location.contains(*location))
                unit.held = true;
        }
        by_hash.emplace(unit.point.hash, units.size());
        units.push_back(std::move(unit));
    }
    std::vector<std::int64_t> count_delta(layout.Size() + 1, 0);
    std::vector<std::size_t> owner_delta(layout.Size() + 1, 0);
    std::vector<std::vector<std::size_t>> children(units.size());
    std::vector<std::size_t> roots;
    for (std::size_t u = 0; u < units.size(); ++u) {
        for (auto range : units[u].ranges) {
            ++count_delta[range.first];
            --count_delta[range.past_end];
            owner_delta[range.first] ^= u + 1;
            owner_delta[range.past_end] ^= u + 1;
        }
        auto parent = units[u].point.parent ? by_hash.find(*units[u].point.parent) : by_hash.end();
        if (parent == by_hash.end())
            roots.push_back(u);
        else
            children[parent->second].push_back(u);
    }
    // Compute nearest Device fallback once in lineage order, including long
    // chains of Host-only checkpoints, instead of walking every parent chain.
    for (std::size_t at = 0; at < roots.size(); ++at) {
        const auto u = roots[at];
        for (auto child : children[u]) {
            units[child].fallback_tokens = units[u].device_ready ? units[u].point.tokens : units[u].fallback_tokens;
            roots.push_back(child);
        }
    }
    std::vector<RecoveryCoverage::Leaf> leaves(layout.Size());
    std::vector<std::size_t> component_at(layout.Size(), components.size());
    for (std::size_t c = 0; c < components.size(); ++c) {
        if (auto position = components[c].position) {
            leaves[*position].eligible = true;
            component_at[*position] = c;
        }
    }
    std::int64_t count = 0;
    std::size_t owners = 0;
    for (std::size_t i = 0; i < leaves.size(); ++i) {
        count += count_delta[i];
        owners ^= owner_delta[i];
        leaves[i].users = count;
        leaves[i].owners = owners;
    }
    RecoveryCoverage coverage{leaves};
    std::vector<bool> removed(components.size(), false);
    std::vector<std::size_t> selected;
    double watermark = store.Checkpoints().Watermark();
    GdsEvictionStats stats;
    std::vector<ContentHash> fresh_rejections;
    GdsTrace trace;
    using Cause = GdsScoreObservation::Cause;
    const auto take = [&](std::size_t c) {
        if (removed[c]) return;
        removed[c] = true;
        selected.push_back(c);
        if (components[c].position) coverage.Disable(*components[c].position);
        capacity.Remove(components[c].group, components[c].location);
    };

    std::vector<double> sizes(units.size(), 0);
    std::vector<std::size_t> undemoted(units.size(), 0);
    std::queue<std::size_t> orphans;
    const auto component_size = [&](std::size_t c) { return 1.0 / pool.Packing(components[c].group); };
    for (std::size_t c = 0; c < components.size(); ++c) {
        const auto position = components[c].position;
        if (!position || leaves[*position].users == 0)
            orphans.push(c);
        else if (leaves[*position].users == 1) {
            const auto u = leaves[*position].owners - 1;
            sizes[u] += component_size(c);
            undemoted[u] += !components[c].demoted;
        }
    }
    struct Priority {
        bool ordinary;
        double score;
        bool fresh;
        std::uint64_t sequence;
        std::size_t unit;
        std::uint64_t version;
        auto Key() const { return std::tie(ordinary, score, fresh, sequence); }
        bool operator>(const Priority& other) const { return Key() > other.Key(); }
    };
    std::priority_queue<Priority, std::vector<Priority>, std::greater<Priority>> heap;
    std::vector<std::uint64_t> versions(units.size(), 0);
    std::vector<std::uint8_t> dirty(units.size(), 0);
    const auto refresh = [&](std::size_t u) {
        const auto version = ++versions[u];
        const auto& unit = units[u];
        if (unit.removed || unit.held || sizes[u] <= 0) return;
        const double score = unit.ready ? unit.point.stamp + store.EvictionConfig().prefill_cost->Between(
                                                                 unit.fallback_tokens, unit.point.tokens) /
                                                                 sizes[u]
                                        : 0;
        if (version == 1 || dirty[u]) {
            const auto cause = version == 1    ? (preview ? Cause::kPublication : Cause::kInitial)
                               : dirty[u] == 1 ? Cause::kExclusiveCapacity
                               : dirty[u] == 2 ? Cause::kFallback
                                               : Cause::kDependencies;
            ObserveGdsScore(trace, {unit.point.hash, CacheTier::kDevice, cause, unit.point.tokens, unit.fallback_tokens,
                                    unit.point.stamp, sizes[u], score, watermark});
        }
        heap.push(Priority{undemoted[u] != 0, score, unit.point.fresh, unit.point.sequence, u, version});
    };
    for (std::size_t u = 0; u < units.size(); ++u) refresh(u);
    const auto peek = [&]() -> std::optional<Priority> {
        while (!heap.empty() && (units[heap.top().unit].removed || heap.top().version != versions[heap.top().unit])) {
            heap.pop();
        }
        return heap.empty() ? std::nullopt : std::optional{heap.top()};
    };
    std::vector<std::size_t> changed;
    const auto mark = [&](std::size_t u, std::uint8_t cause) {
        if (!dirty[u]) {
            changed.push_back(u);
        }
        dirty[u] |= cause;
    };
    const auto remove_unit = [&](std::size_t u) {
        auto& unit = units[u];
        if (unit.removed) return;
        unit.removed = true;
        for (auto range : unit.ranges) {
            coverage.Remove(range, u + 1, [&](std::size_t position, std::int64_t users, std::size_t owners) {
                const auto c = component_at[position];
                if (users == 0)
                    orphans.push(c);
                else if (users == 1) {
                    const auto owner = owners - 1;
                    sizes[owner] += component_size(c);
                    undemoted[owner] += !components[c].demoted;
                    mark(owner, 1);
                }
            });
        }
        if (unit.device_ready) {
            std::vector<std::size_t> descendants = children[u];
            while (!descendants.empty()) {
                const auto v = descendants.back();
                descendants.pop_back();
                if (!units[v].removed) {
                    units[v].fallback_tokens = unit.fallback_tokens;
                    mark(v, 2);
                    // This surviving Device checkpoint remains its own
                    // descendants' fallback, so changes stop at this boundary.
                    if (units[v].device_ready) continue;
                }
                descendants.insert(descendants.end(), children[v].begin(), children[v].end());
            }
        }
    };
    const auto refresh_changed = [&] {
        // Expose decreases before peeking at the next minimum. Periodic
        // rebuilding bounds obsolete heap records even on a long shared chain.
        if (heap.size() > 4 * units.size()) {
            heap = {};
            for (std::size_t u = 0; u < units.size(); ++u) refresh(u);
        } else
            for (auto u : changed) refresh(u);
        for (auto u : changed) dirty[u] = false;
        changed.clear();
    };
    std::size_t next_fallback = 0;
    while (!capacity.Fits(demands)) {
        while (!orphans.empty() && !capacity.Fits(demands)) {
            const auto c = orphans.front();
            orphans.pop();
            if (!removed[c]) {
                take(c);
                ++stats.orphan_blocks;
            }
        }
        if (capacity.Fits(demands)) break;
        const auto chosen = peek();
        if (!chosen) {
            // Capacity fallback explicitly invalidates every affected recovery
            // point. It never reports a one-point loss for shared KV removal.
            while (next_fallback < components.size() && removed[next_fallback]) ++next_fallback;
            if (next_fallback == components.size()) return std::nullopt;
            if (auto position = components[next_fallback].position) {
                // The rare block fallback reports every lost route. Its scan
                // is separate from ordinary range-based unit selection.
                for (std::size_t u = 0; u < units.size(); ++u) {
                    if (units[u].removed || !std::ranges::any_of(units[u].ranges, [&](auto range) {
                            return range.first <= *position && *position < range.past_end;
                        }))
                        continue;
                    ++stats.fallback_units;
                    ObserveGdsScore(trace, {units[u].point.hash, CacheTier::kDevice, Cause::kCapacityFallback,
                                            units[u].point.tokens, units[u].fallback_tokens, units[u].point.stamp,
                                            sizes[u], 0, watermark});
                    remove_unit(u);
                }
            }
            take(next_fallback);
            ++stats.fallback_blocks;
            refresh_changed();
            continue;
        }
        heap.pop();
        const auto next = peek();
        const bool reject = chosen->fresh && chosen->ordinary && next && next->ordinary && chosen->score < next->score;
        const auto& unit = units[chosen->unit];
        std::vector<std::size_t> released;
        for (auto range : unit.ranges)
            coverage.ForEachExclusive(range, chosen->unit + 1,
                                      [&](std::size_t position) { released.push_back(component_at[position]); });
        if (!reject && chosen->ordinary && unit.ready) watermark = std::max(watermark, chosen->score);
        if (reject) {
            ++stats.retention_rejections;
            fresh_rejections.push_back(unit.point.hash);
        } else if (!chosen->ordinary)
            ++stats.demoted_units;
        else
            ++stats.evicted_units;
        if (unit.ready && !unit.device_ready) ++stats.mixed_tier_units;
        ObserveGdsScore(trace, {unit.point.hash, CacheTier::kDevice,
                                reject             ? Cause::kRejection
                                : chosen->ordinary ? Cause::kEviction
                                                   : Cause::kDemotion,
                                unit.point.tokens, unit.fallback_tokens, unit.point.stamp, sizes[chosen->unit],
                                chosen->score, watermark});
        remove_unit(chosen->unit);
        // Commit the entire chosen unit in the shadow; no block minimization.
        for (auto c : released) take(c);
        refresh_changed();
    }
    // Record single-boundary segments in physical dependency order. The
    // existing write-back merger can then combine shared ancestor KV without
    // assuming the score order has the shape of a tree traversal.
    std::ranges::sort(selected, [&](std::size_t a, std::size_t b) {
        const auto depth = [&](std::size_t c) {
            const auto* node = components[c].key ? store.Boundaries().Find(components[c].key->content_hash) : nullptr;
            return node ? node->boundary_tokens : 0;
        };
        if (depth(a) != depth(b)) return depth(a) > depth(b);
        if (store.GroupIsState(components[a].group) != store.GroupIsState(components[b].group))
            return store.GroupIsState(components[a].group);
        return a < b;
    });
    std::vector<VictimSet> result;
    for (auto c : selected) {
        auto& set = result.emplace_back(resource);
        set.exact_removal = true;
        set.gds_watermark = watermark;
        set.members.push_back({components[c].group, components[c].location, false, components[c].reclaim});
    }
    if (!result.empty()) {
        stats.watermark = watermark;
        stats.committed_plans = 1;
        stats.planning_nanoseconds =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count();
        result.front().fresh_rejections = std::move(fresh_rejections);
        result.front().gds_stats = stats;
        result.front().gds_trace = std::move(trace);
    }
    return result;
}

}  // namespace tokenspeed
