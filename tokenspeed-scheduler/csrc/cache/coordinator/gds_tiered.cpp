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

#include "cache/coordinator/gds_tiered.h"

#include "core/profile_span.h"

#include <array>
#include <chrono>
#include <numeric>
#include <limits>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "cache/coordinator/eviction_candidates.h"
#include "cache/coordinator/eviction_plan.h"
#include "cache/coordinator/gds_coverage.h"
#include "cache/coordinator/tier_eviction_planner.h"

namespace tokenspeed {
namespace {
using KeySet = std::unordered_set<CacheKey, CacheKeyHash>;
struct Component {
    TieredGdsPlan::Member member;
    std::optional<std::size_t> position;
    bool demoted;
};
struct Coverage;
struct Work {
    RecoveryIndex::View view;
    std::shared_ptr<const Coverage> coverage;
    std::array<std::unordered_set<std::size_t>, 2> removed;
    KeySet copies;
    std::unordered_set<std::size_t> dropped;
    GdsEvictionStats stats;
    GdsTrace trace;
    std::vector<ContentHash> fresh_rejections;
    explicit Work(const RecoveryIndex& index) : view{index} {}
};
struct Coverage {
    std::vector<RecoveryIndex::Entry> routes;
    std::array<std::vector<std::int64_t>, 2> users;
    std::array<std::vector<std::size_t>, 2> owners;
};

// Physical ordering proposes placements. Only Drop() authorizes logical loss.
// Both tiers use the same route overlay, ownership census and GDS decisions.
class Planner {
public:
    Planner(const CacheStore& store, const AdmissionCandidateConstraints* constraints)
        : store_{store},
          preview_{constraints ? constraints->publication.get() : nullptr},
          layout_{preview_ ? preview_->Leaves(store) : store.RecoveryDependencies().Leaves()},
          points_{preview_ ? preview_->Points(store) : store.Checkpoints().RecoveryPoints()},
          work_{store.RecoveryDependencies()},
          demand_{EmptyBlockDemands(store), EmptyBlockDemands(store)} {
        if (preview_) preview_->Apply(work_.view);
        work_.stats.watermark = store.Checkpoints().Watermark();
        work_.stats.host_watermark = store.Checkpoints().HostWatermark();
        for (std::size_t u = 0; u < points_.size(); ++u) {
            by_hash_.emplace(points_[u].hash, u);
            const auto& route = Route(points_[u].hash);
            baseline_.push_back(route.Available());
            held_.push_back(route.pending || (preview_ && preview_->Holds(points_[u].hash)));
        }
        for (std::size_t tier = 0; tier < NumTiers(); ++tier) {
            const auto& pool = Pool(tier);
            std::pmr::monotonic_buffer_resource scratch;
            const CacheEvictionCandidates::LocationSet empty{&scratch};
            CacheEvictionCandidates candidates{store.Groups(), pool, CheckpointAging{}};
            while (auto c = candidates.Next(
                       tier == 0 && constraints ? std::span{constraints->request_reclaim_candidates}
                                                : std::span<const CacheEvictionCandidates::Candidate>{},
                       constraints
                           ? (tier == 0 ? constraints->protected_locations : constraints->host_protected_locations)
                           : empty,
                       tier == 0 && constraints ? constraints->request_reclaim_locations : empty,
                       [](CacheBlockLocation) { return false; })) {
                const auto& index = store.Groups()[c->group_id].Index();
                auto key = index.KeyOf(pool, c->location);
                const auto metadata = index.MetadataFor(pool, c->location);
                if (!locations_[tier].insert(c->location).second) continue;
                const bool reclaim =
                    tier == 0 && constraints && constraints->request_reclaim_locations.contains(c->location);
                if (key) by_key_[tier].emplace(*key, components_[tier].size());
                components_[tier].push_back(
                    {{c->group_id, c->location, key, reclaim},
                     key ? layout_.Position(*key) : std::nullopt,
                     metadata && (metadata->demoted || (metadata->session && metadata->session->demoted))});
            }
            for (std::size_t u = 0; u < points_.size(); ++u) {
                for (std::size_t g = 0; g < store.Groups().size(); ++g) {
                    if (!store.GroupIsState(g)) continue;
                    const CacheKey key{.group_id = static_cast<std::uint32_t>(g), .content_hash = points_[u].hash};
                    if (auto loc = store.LocationOf(pool, g, key); loc && !locations_[tier].contains(*loc))
                        held_[u] = true;
                }
            }
        }
    }

    bool ReleaseDeviceCopy(const CacheKey& key) {
        const auto it = by_key_[0].find(key);
        if (it == by_key_[0].end()) return false;
        Remove(0, it->second);
        return Preserve(false);
    }

    bool FitDevice(std::span<const BlockPlacementDemand> demand) {
        ProfileSpan profile_span{"Planner::FitDevice"};
        demand_[0].assign(demand.begin(), demand.end());
        if (Fits(0)) return true;
        // Deep suffixes avoid turning a cheap Device release into a copy of
        // the whole prefix. Existing candidate order resolves equal depths.
        std::vector<std::size_t> order(components_[0].size());
        std::iota(order.begin(), order.end(), 0);
        std::vector<std::tuple<bool, std::int32_t, bool>> ranks;
        ranks.reserve(order.size());
        for (const auto& c : components_[0])
            ranks.emplace_back(!c.demoted, -Depth(c.member.key), !store_.GroupIsState(c.member.group));
        std::ranges::stable_sort(order, [&](auto a, auto b) { return ranks[a] < ranks[b]; });
        if (ReleaseSafeDeviceCopies(order)) return true;
        // A later failed migration must not discard the safe releases already
        // found. Logical loss, including demotion, starts only after this pass.
        const Work initial = work_;
        while (Drop(0, true))
            if (Fits(0)) return true;
        for (auto c : order) {
            if (work_.removed[0].contains(c)) continue;
            Work before = work_;
            Remove(0, c);
            if (Preserve(store_.WritesBackEvictions()) && (!store_.HostPool() || FundHost(true))) {
                if (Fits(0)) return true;
                continue;
            }
            work_ = std::move(before);
            if (!Drop(0, false)) continue;
            if (Fits(0)) return true;
        }
        while (!Fits(0) && Drop(0, false)) {
        }
        if (Fits(0)) return true;
        // Preservation is optional. Do not send an otherwise feasible
        // admission to retraction because Host placement could not preserve it.
        work_ = initial;
        return FitDeviceWithoutMigration();
    }

    bool AddHostRows(std::span<const std::uint32_t> groups, std::span<const std::int32_t> buckets, bool allow_loss) {
        const Work before = work_;
        const auto old_demand = demand_[1];
        for (std::size_t i = 0; i < groups.size(); ++i) ++demand_[1][groups[i]].blocks_by_bucket[buckets[i]];
        if (FundHost(allow_loss)) return true;
        work_ = before;
        demand_[1] = old_demand;
        return false;
    }

    std::shared_ptr<const TieredGdsPlan> Finish() {
        ProfileSpan profile_span{"Planner::Finish"};
        auto result = std::make_shared<TieredGdsPlan>();
        for (std::size_t tier = 0; tier < 2; ++tier) {
            auto& out = tier == 0 ? result->device_removals : result->host_removals;
            for (auto c : work_.removed[tier]) out.push_back(components_[tier][c].member);
            std::ranges::sort(out, [&](const auto& a, const auto& b) {
                return std::tuple{-Depth(a.key), !store_.GroupIsState(a.group), a.group, a.location.lcm_block_id,
                                  a.location.slot_index} < std::tuple{-Depth(b.key), !store_.GroupIsState(b.group),
                                                                      b.group, b.location.lcm_block_id,
                                                                      b.location.slot_index};
            });
        }
        result->host_copies.assign(work_.copies.begin(), work_.copies.end());
        std::ranges::sort(result->host_copies, [&](const auto& a, const auto& b) {
            return std::tuple{-Depth(a), !store_.GroupIsState(a.group_id), a.group_id, a.content_hash.ToString(),
                              a.page_offset} < std::tuple{-Depth(b), !store_.GroupIsState(b.group_id), b.group_id,
                                                          b.content_hash.ToString(), b.page_offset};
        });
        KeySet kept;
        std::vector<std::size_t> copy_positions;
        for (const auto& key : work_.copies) copy_positions.push_back(*layout_.Position(key));
        std::ranges::sort(copy_positions);
        std::vector<RecoveryIndex::Path> kept_paths;
        for (std::size_t u = 0; u < points_.size(); ++u) {
            if (!baseline_[u] || work_.dropped.contains(u)) continue;
            const auto& route = Route(points_[u].hash);
            FatalCheck(route.Available(), "a joint GDS plan lost an unauthorized recovery point");
            const bool migrating = std::ranges::any_of(layout_.Intervals(route.host_paths), [&](auto range) {
                const auto at = std::ranges::lower_bound(copy_positions, range.first);
                return at != copy_positions.end() && *at < range.past_end;
            });
            if (migrating) kept_paths.insert(kept_paths.end(), route.host_paths.begin(), route.host_paths.end());
        }
        Visit(kept_paths, [&](const CacheKey& key) {
            if (!work_.copies.contains(key) && store_.Groups()[key.group_id].Index().Contains(Pool(1), key))
                kept.insert(key);
        });
        result->host_kept.assign(kept.begin(), kept.end());
        for (auto u : work_.dropped) result->losses.push_back(points_[u].hash);
        result->fresh_rejections = work_.fresh_rejections;
        result->stats = work_.stats;
        result->trace = std::move(work_.trace);
        result->stats.committed_plans = 1;
        result->stats.planning_nanoseconds =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started_).count();
        result->stats.device_copies_released = result->device_removals.size();
        result->stats.host_copies_reserved = result->host_copies.size();
        return result;
    }

    void VisitCleaningCandidates(const std::function<bool(HostCleaningCandidate)>& visit) {
        ProfileSpan profile_span{"Planner::Cleaning"};
        const auto coverage = Cover();
        struct Ranked {
            std::size_t unit;
            double score;
        };
        std::vector<Ranked> result;
        std::vector<double> sizes(points_.size(), 0);
        for (const auto& component : components_[1]) {
            if (!component.position || coverage->users[1][*component.position] != 1) continue;
            sizes[coverage->owners[1][*component.position] - 1] += 1.0 / Pool(1).Packing(component.member.group);
        }
        for (std::size_t u = 0; u < points_.size(); ++u) {
            if (!baseline_[u] || held_[u] || sizes[u] <= 0) continue;
            const auto [benefit, fallback] = Benefit(u, *coverage);
            if (benefit <= 0) continue;
            double extra_read_cost = 0;
            // Intervals may merge adjacent coordinates from different groups.
            // Union paths within each group before applying its block cost.
            std::vector<RecoveryIndex::Path> group_paths;
            for (std::size_t group = 0; group < store_.Groups().size(); ++group) {
                group_paths.clear();
                for (const auto& path : coverage->routes[u].host_paths)
                    if (path.first.group_id == group) group_paths.push_back(path);
                std::size_t blocks = 0;
                for (const auto range : layout_.Intervals(group_paths)) blocks += range.past_end - range.first;
                if (blocks > 0)
                    extra_read_cost += static_cast<double>(blocks) * store_.EvictionConfig().disk_block_costs.at(group);
            }
            if (extra_read_cost >= benefit || coverage->routes[u].host_paths.empty()) continue;
            result.push_back({u, points_[u].host_stamp + benefit / sizes[u]});
        }
        std::ranges::sort(result, [&](const auto& a, const auto& b) {
            return std::tuple{a.score, points_[a.unit].sequence} < std::tuple{b.score, points_[b.unit].sequence};
        });
        // Keep dependencies as ranges until this candidate reaches the write
        // budget. A dense checkpoint chain must not expand every prefix first.
        for (const auto& ranked : result) {
            const auto u = ranked.unit;
            HostCleaningCandidate candidate{points_[u].hash, ranked.score, points_[u].sequence, {}};
            Visit(coverage->routes[u].host_paths, [&](const CacheKey& key) { candidate.keys.push_back(key); });
            Visit(coverage->routes[u].disk_paths,
                  [&](const CacheKey& key) { candidate.disk_dependencies.push_back(key); });
            if (!visit(std::move(candidate))) break;
        }
    }

private:
    bool ReleaseSafeDeviceCopies(std::span<const std::size_t> order) {
        ProfileSpan profile_span{"Planner::ReleaseSafeDeviceCopies"};
        // With no lower-tier residency/promises and no broken point, every
        // indexed key is required on Device. Only unindexed orphans can leave.
        const bool only_device_routes =
            !preview_ && store_.RecoveryDependencies().GetStats().broken == 0 && store_.PendingHostKeys().empty() &&
            std::ranges::all_of(store_.Groups(), [&](const auto& group) {
                return (!store_.HostPool() || group.Index().NumEntries(*store_.HostPool()) == 0) &&
                       (!store_.DiskPool() || group.Index().NumEntries(*store_.DiskPool()) == 0);
            });
        CacheEvictionPlan capacity{Pool(0), store_.Groups()};
        std::size_t next = 0;
        return capacity.SelectUntilFits(demand_[0], [&] {
            while (next < order.size()) {
                const auto c = order[next++];
                const auto& component = components_[0][c];
                if (only_device_routes && component.position) continue;
                if (component.member.key && !work_.view.TryRemoveDeviceCopy(*component.member.key)) continue;
                work_.removed[0].insert(c);
                work_.coverage.reset();
                capacity.Remove(component.member.group, component.member.location);
                return true;
            }
            return false;
        });
    }

    bool FitDeviceWithoutMigration() {
        CacheEvictionPlan capacity{Pool(0), store_.Groups()};
        for (const auto& component : components_[0]) capacity.Remove(component.member.group, component.member.location);
        if (!capacity.Fits(demand_[0])) return false;

        // Ordinary losses still use Device scores and advance only Lambda_D.
        while (!Fits(0) && Drop(0, false)) {
        }
        if (Fits(0)) return true;

        // A held State excludes its unit from ordinary ranking, but cannot
        // reserve unowned KV on its behalf. The candidate census has already
        // excluded all live owners and request-protected locations. Reclaim
        // only those candidates, accounting for every route actually lost.
        // As in the no-Host planner, this exceptional block fallback has no
        // finite exclusive-unit score and does not advance either watermark.
        for (std::size_t c = 0; c < components_[0].size(); ++c) {
            if (work_.removed[0].contains(c)) continue;
            Remove(0, c);
            ++work_.stats.fallback_blocks;
            for (std::size_t u = 0; u < points_.size(); ++u) {
                if (!baseline_[u] || work_.dropped.contains(u) || Resolve(points_[u].hash, false).Available()) continue;
                work_.coverage.reset();
                work_.dropped.insert(u);
                ++work_.stats.fallback_units;
                ObserveGdsScore(work_.trace,
                                {points_[u].hash, CacheTier::kDevice, GdsScoreObservation::Cause::kCapacityFallback,
                                 points_[u].tokens, 0, points_[u].stamp, 0, 0, work_.stats.watermark});
            }
            if (Fits(0)) return true;
        }
        FatalCheck(false, "the complete eligible Device census must satisfy its proven demand");
        return false;
    }

    const RecoveryIndex::Entry& Route(const ContentHash& hash) const {
        return preview_ ? preview_->At(hash, work_.view) : work_.view.At(hash);
    }
    RecoveryIndex::Entry Resolve(const ContentHash& hash, bool allow_copy) const {
        return preview_ ? preview_->Resolve(hash, work_.view, allow_copy) : work_.view.Resolve(hash, allow_copy);
    }
    std::size_t NumTiers() const { return store_.HostPool() ? 2 : 1; }
    const BlockPool& Pool(std::size_t tier) const { return tier == 0 ? store_.DevicePool() : *store_.HostPool(); }
    std::int32_t Depth(const std::optional<CacheKey>& key) const {
        const auto* node = key ? store_.Boundaries().Find(key->content_hash) : nullptr;
        return node ? node->boundary_tokens : 0;
    }
    template <typename F>
    void Visit(std::span<const RecoveryIndex::Path> paths, F&& f) const {
        for (auto interval : layout_.Intervals(paths))
            for (auto p = interval.first; p < interval.past_end; ++p) f(layout_.KeyAt(p));
    }
    void Change(const CacheKey& key, std::size_t tier, bool present) {
        const RecoveryIndex::View::Change change{key, tier == 0 ? CacheTier::kDevice : CacheTier::kHost, present};
        work_.coverage.reset();
        work_.view.Apply(std::span{&change, 1});
    }
    void Remove(std::size_t tier, std::size_t c) {
        if (!work_.removed[tier].insert(c).second) return;
        if (const auto& key = components_[tier][c].member.key) Change(*key, tier, false);
    }
    void RemoveKey(std::size_t tier, const CacheKey& key) {
        if (tier == 1 && work_.copies.erase(key)) Change(key, 1, false);
        if (auto c = by_key_[tier].find(key); c != by_key_[tier].end()) Remove(tier, c->second);
    }
    bool Fits(std::size_t tier) const {
        CacheEvictionPlan capacity{Pool(tier), store_.Groups()};
        for (auto c : work_.removed[tier])
            capacity.Remove(components_[tier][c].member.group, components_[tier][c].member.location);
        auto demands = demand_[tier];
        if (tier == 1) {
            for (const auto& key : work_.copies) {
                const auto loc = store_.LocationOf(Pool(0), key.group_id, key);
                FatalCheck(loc.has_value(), "a reserved copy has a Device source before commit");
                const auto bucket = store_.Groups()[key.group_id].Allocator().BucketOf(*loc);
                ++demands[key.group_id].blocks_by_bucket[bucket];
            }
        }
        return capacity.Fits(demands);
    }
    std::shared_ptr<const Coverage> Cover() {
        ProfileSpan profile_span{"Planner::Cover"};
        if (work_.coverage) return work_.coverage;
        auto coverage = std::make_shared<Coverage>();
        auto& result = *coverage;
        result.routes.reserve(points_.size());
        std::array<std::vector<std::int64_t>, 2> delta{std::vector<std::int64_t>(layout_.Size() + 1),
                                                       std::vector<std::int64_t>(layout_.Size() + 1)};
        std::array<std::vector<std::size_t>, 2> owner_delta{std::vector<std::size_t>(layout_.Size() + 1),
                                                            std::vector<std::size_t>(layout_.Size() + 1)};
        for (std::size_t u = 0; u < points_.size(); ++u) {
            auto route = Route(points_[u].hash);
            if (baseline_[u] && !work_.dropped.contains(u)) {
                for (std::size_t tier = 0; tier < 2; ++tier) {
                    for (auto r : layout_.Intervals(tier == 0 ? route.device_paths : route.host_paths)) {
                        ++delta[tier][r.first];
                        --delta[tier][r.past_end];
                        owner_delta[tier][r.first] ^= u + 1;
                        owner_delta[tier][r.past_end] ^= u + 1;
                    }
                }
            }
            result.routes.push_back(std::move(route));
        }
        for (std::size_t tier = 0; tier < 2; ++tier) {
            result.users[tier].resize(layout_.Size());
            result.owners[tier].resize(layout_.Size());
            std::int64_t count = 0;
            std::size_t owner = 0;
            for (std::size_t p = 0; p < layout_.Size(); ++p) {
                count += delta[tier][p];
                owner ^= owner_delta[tier][p];
                result.users[tier][p] = count;
                result.owners[tier][p] = owner;
            }
        }
        work_.coverage = coverage;
        return coverage;
    }
    bool Preserve(bool allow_copy) {
        ProfileSpan profile_span{"Planner::Preserve"};
        KeySet needed;
        std::vector<RecoveryIndex::Path> required;
        for (std::size_t u = 0; u < points_.size(); ++u) {
            if (!baseline_[u] || work_.dropped.contains(u)) continue;
            if (Route(points_[u].hash).Available()) continue;
            if (!allow_copy) return false;
            const auto route = Resolve(points_[u].hash, true);
            if (!route.Available()) return false;
            required.insert(required.end(), route.host_paths.begin(), route.host_paths.end());
        }
        bool copyable = true;
        Visit(required, [&](const CacheKey& key) {
            if (work_.view.Present(key, CacheTier::kHost) || store_.HasReservedHostKey(key)) return;
            const auto metadata = store_.Groups()[key.group_id].Index().Inspect(Pool(0), key);
            if (!metadata || metadata->metadata.demoted ||
                (metadata->metadata.session && metadata->metadata.session->demoted) ||
                metadata->metadata.eviction.last_access_epoch == CacheStore::kSupersededAccessEpoch) {
                copyable = false;
                return;
            }
            needed.insert(key);
        });
        if (!copyable) return false;
        std::vector<RecoveryIndex::View::Change> changes;
        for (const auto& key : needed) {
            if (work_.copies.insert(key).second) changes.push_back({key, CacheTier::kHost, true});
        }
        if (!changes.empty()) work_.coverage.reset();
        work_.view.Apply(changes);
        return true;
    }
    bool FundHost(bool allow_loss) {
        if (Fits(1)) return true;
        auto coverage = Cover();
        // A copy unused by the selected legal routes is a placement victim,
        // irrespective of its old aged rank. Recheck after every logical drop.
        for (;;) {
            for (std::size_t c = 0; c < components_[1].size(); ++c) {
                if (work_.removed[1].contains(c)) continue;
                const auto p = components_[1][c].position;
                if (p && coverage->users[1][*p] != 0) continue;
                Remove(1, c);
                if (Fits(1)) return true;
            }
            // Replacing an eligible Host key with the same published Disk
            // key leaves Host-or-Disk coverage unchanged. Price the added
            // reads on selected routes from their shared consumer counts;
            // do not clone and resolve a full plan for every clean key.
            std::optional<std::size_t> clean;
            double least_cost = std::numeric_limits<double>::infinity();
            for (std::size_t c = 0; c < components_[1].size(); ++c) {
                const auto& component = components_[1][c];
                if (work_.removed[1].contains(c) || !component.member.key ||
                    !work_.view.Present(*component.member.key, CacheTier::kDisk))
                    continue;
                const double rank = store_.UsesGds()
                                        ? store_.EvictionConfig().disk_block_costs[component.member.group] *
                                              coverage->users[1][*component.position] *
                                              Pool(1).Packing(component.member.group)
                                        : 0;
                if (!clean || rank < least_cost) {
                    least_cost = rank;
                    clean = c;
                }
            }
            if (clean) {
                Remove(1, *clean);
                FatalCheck(Preserve(false), "a same-key Disk copy must preserve the Host route");
                ++work_.stats.host_clean_copies_released;
                if (Fits(1)) return true;
                coverage = Cover();
                continue;
            }
            if (!allow_loss || !Drop(1, false)) return false;
            if (Fits(1)) return true;
            coverage = Cover();
        }
    }
    std::pair<double, std::int32_t> Benefit(std::size_t u, const Coverage& coverage) const {
        const auto& cost = *store_.EvictionConfig().prefill_cost;
        double without = cost.Between(0, points_[u].tokens);
        std::int32_t fallback = 0;
        auto parent = points_[u].parent;
        while (parent) {
            const auto p = by_hash_.find(*parent);
            if (p == by_hash_.end()) break;
            if (!work_.dropped.contains(p->second) && coverage.routes[p->second].Available()) {
                const double candidate = coverage.routes[p->second].disk_read_cost +
                                         cost.Between(points_[p->second].tokens, points_[u].tokens);
                if (candidate < without) {
                    without = candidate;
                    fallback = points_[p->second].tokens;
                }
                // A shallower ancestor has nonnegative read cost and more
                // prefill work, so it cannot beat this memory-ready one.
                if (coverage.routes[p->second].disk_read_cost == 0) break;
            }
            parent = points_[p->second].parent;
        }
        return {std::max(0.0, without - coverage.routes[u].disk_read_cost), fallback};
    }
    bool Drop(std::size_t tier, bool demoted_only) {
        ProfileSpan profile_span{"Planner::Drop"};
        if (demoted_only && std::ranges::none_of(components_[tier], [](const auto& c) { return c.demoted; }))
            return false;
        const auto coverage = Cover();
        std::vector<double> sizes(points_.size(), 0);
        std::vector<bool> demoted(points_.size(), true);
        for (std::size_t c = 0; c < components_[tier].size(); ++c) {
            if (work_.removed[tier].contains(c)) continue;
            const auto p = components_[tier][c].position;
            if (!p || coverage->users[tier][*p] != 1) continue;
            const auto u = coverage->owners[tier][*p] - 1;
            sizes[u] += 1.0 / Pool(tier).Packing(components_[tier][c].member.group);
            demoted[u] = demoted[u] && components_[tier][c].demoted;
        }
        if (tier == 1)
            for (const auto& key : work_.copies) {
                const auto p = layout_.Position(key);
                if (p && coverage->users[1][*p] == 1) {
                    const auto u = coverage->owners[1][*p] - 1;
                    sizes[u] += 1.0 / Pool(1).Packing(key.group_id);
                    demoted[u] = false;
                }
            }
        struct Choice {
            std::size_t unit;
            double score;
            bool ordinary;
            std::int32_t fallback;
        };
        std::vector<Choice> choices;
        for (std::size_t u = 0; u < points_.size(); ++u) {
            if (!baseline_[u] || held_[u] || work_.dropped.contains(u) || sizes[u] <= 0) continue;
            if (demoted_only && !demoted[u]) continue;
            const auto [gain, fallback] = Benefit(u, *coverage);
            const double stamp = tier == 0 ? points_[u].stamp : points_[u].host_stamp;
            choices.push_back({u, stamp + gain / sizes[u], !demoted[u], fallback});
            ObserveGdsScore(work_.trace,
                            {points_[u].hash, tier == 0 ? CacheTier::kDevice : CacheTier::kHost,
                             GdsScoreObservation::Cause::kJointRoutes, points_[u].tokens, fallback, stamp, sizes[u],
                             choices.back().score, tier == 0 ? work_.stats.watermark : work_.stats.host_watermark});
        }
        if (choices.empty()) return false;
        const auto rank = [&](const Choice& choice) {
            return std::tuple{choice.ordinary, choice.score, points_[choice.unit].fresh, points_[choice.unit].sequence};
        };
        std::ranges::sort(choices, [&](const auto& a, const auto& b) { return rank(a) < rank(b); });
        const auto chosen = choices.front();
        const auto u = chosen.unit;
        const bool rejected = points_[u].fresh && chosen.ordinary && choices.size() > 1 && choices[1].ordinary &&
                              chosen.score < choices[1].score;
        if (rejected) {
            work_.fresh_rejections.push_back(points_[u].hash);
            ++work_.stats.logical_rejections;
            ++work_.stats.retention_rejections;
        } else {
            ++work_.stats.logical_evictions;
            if (chosen.ordinary) {
                ++work_.stats.evicted_units;
                auto& watermark = tier == 0 ? work_.stats.watermark : work_.stats.host_watermark;
                watermark = std::max(watermark, chosen.score);
            } else
                ++work_.stats.demoted_units;
        }
        if (!coverage->routes[u].device_ready) ++work_.stats.mixed_tier_units;
        using Cause = GdsScoreObservation::Cause;
        ObserveGdsScore(work_.trace,
                        {points_[u].hash, tier == 0 ? CacheTier::kDevice : CacheTier::kHost,
                         rejected          ? Cause::kRejection
                         : chosen.ordinary ? Cause::kEviction
                                           : Cause::kDemotion,
                         points_[u].tokens, chosen.fallback, tier == 0 ? points_[u].stamp : points_[u].host_stamp,
                         sizes[u], chosen.score, tier == 0 ? work_.stats.watermark : work_.stats.host_watermark});
        work_.coverage.reset();
        work_.dropped.insert(u);
        // State copies identify the point. Other points can still use its
        // history, which leaves only when the range has one remaining owner.
        for (std::size_t t = 0; t < 2; ++t) {
            for (std::size_t c = 0; c < components_[t].size(); ++c) {
                if (work_.removed[t].contains(c)) continue;
                const auto& component = components_[t][c];
                const bool own_state = component.member.key && store_.GroupIsState(component.member.group) &&
                                       component.member.key->content_hash == points_[u].hash;
                const bool exclusive = component.position && coverage->users[t][*component.position] == 1 &&
                                       coverage->owners[t][*component.position] == u + 1;
                if (own_state || exclusive) Remove(t, c);
            }
        }
        std::vector<CacheKey> cancelled;
        for (const auto& key : work_.copies) {
            const auto p = layout_.Position(key);
            if ((store_.GroupIsState(key.group_id) && key.content_hash == points_[u].hash) ||
                (p && coverage->users[1][*p] == 1 && coverage->owners[1][*p] == u + 1))
                cancelled.push_back(key);
        }
        for (const auto& key : cancelled) RemoveKey(1, key);
        return true;
    }

    const CacheStore& store_;
    const std::chrono::steady_clock::time_point started_{std::chrono::steady_clock::now()};
    const PublicationPreview* const preview_;
    RecoveryLayout layout_;
    std::vector<CheckpointTree::RecoveryPoint> points_;
    std::unordered_map<ContentHash, std::size_t> by_hash_;
    std::vector<bool> baseline_, held_;
    std::array<std::vector<Component>, 2> components_;
    std::array<std::unordered_map<CacheKey, std::size_t, CacheKeyHash>, 2> by_key_;
    std::array<std::unordered_set<CacheBlockLocation, CacheBlockLocationHash>, 2> locations_;
    Work work_;
    std::array<std::vector<BlockPlacementDemand>, 2> demand_;
};
}  // namespace

PlannedEviction TieredVictims(std::shared_ptr<const TieredGdsPlan> plan, std::pmr::memory_resource* resource) {
    PlannedEviction result;
    auto& set = result.victims.emplace_back(resource);
    set.exact_removal = true;
    for (const auto& member : plan->device_removals)
        set.members.push_back({member.group, member.location, false, member.request_reclaim});
    result.gds = std::make_unique<GdsEvictionCommit>();
    result.gds->tiered = std::move(plan);
    return result;
}

std::optional<PlannedEviction> PlanTieredDeviceEviction(const CacheStore& store,
                                                        const AdmissionCandidateConstraints& constraints,
                                                        std::span<const BlockPlacementDemand> demands,
                                                        std::pmr::memory_resource* resource) {
    ProfileSpan profile_span{"Gds::TieredPlanAttempt"};
    Planner planner{store, &constraints};
    if (!planner.FitDevice(demands)) return std::nullopt;
    auto result = TieredVictims(planner.Finish(), resource);
    return result;
}

std::optional<PlannedEviction> PlanTieredCompaction(const CacheStore& store,
                                                    const AdmissionCandidateConstraints& constraints,
                                                    const CacheKey& key, std::pmr::memory_resource* resource) {
    ProfileSpan profile_span{"Gds::TieredPlanAttempt"};
    Planner planner{store, &constraints};
    if (!planner.ReleaseDeviceCopy(key)) return std::nullopt;
    return TieredVictims(planner.Finish(), resource);
}

TieredHostAllocation PlanTieredHostAllocation(const CacheStore& store, std::span<const std::uint32_t> groups,
                                              std::span<const std::int32_t> buckets,
                                              std::span<const std::size_t> segment_ends, bool allow_logical_loss,
                                              bool prefix) {
    ProfileSpan profile_span{"Gds::HostPlanAttempt"};
    auto demand = EmptyBlockDemands(store);
    for (std::size_t i = 0; i < groups.size(); ++i) ++demand[groups[i]].blocks_by_bucket[buckets[i]];
    CacheEvictionPlan empty{*store.HostPool(), store.Groups()};
    if (empty.Fits(demand)) {
        auto plan = std::make_shared<TieredGdsPlan>();
        plan->stats.watermark = store.Checkpoints().Watermark();
        plan->stats.host_watermark = store.Checkpoints().HostWatermark();
        return {std::move(plan), std::vector<bool>(groups.size(), true), segment_ends.size()};
    }
    Planner planner{store, nullptr};
    std::vector<bool> accepted(groups.size(), false);
    std::size_t first = 0, count = 0;
    for (auto end : segment_ends) {
        _assert(end >= first && end <= groups.size(), "Host segments cover rows in order");
        if (planner.AddHostRows(groups.subspan(first, end - first), buckets.subspan(first, end - first),
                                allow_logical_loss)) {
            std::fill(accepted.begin() + first, accepted.begin() + end, true);
            ++count;
        } else if (prefix)
            break;
        first = end;
    }
    auto plan = planner.Finish();
    if (!allow_logical_loss) {
        auto counted = std::make_shared<TieredGdsPlan>(*plan);
        counted->stats.host_copy_rejections = std::ranges::count(accepted, false);
        plan = std::move(counted);
    }
    return {std::move(plan), std::move(accepted), count};
}
void VisitGdsHostCleaningCandidates(const CacheStore& store, const std::function<bool(HostCleaningCandidate)>& visit) {
    ProfileSpan profile_span{"Gds::CleaningAttempt"};
    Planner{store, nullptr}.VisitCleaningCandidates(visit);
}

}  // namespace tokenspeed
