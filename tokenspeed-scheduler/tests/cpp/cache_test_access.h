// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
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
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "utils.h"

namespace tokenspeed {

struct CacheCoordinatorTestAccess {
    static std::size_t CacheGroupRegistrySize(const CacheCoordinator& coordinator) {
        return coordinator.cache_group_tags_.size();
    }

    static auto MatchPrefix(CacheCoordinator& coordinator, std::span<const std::string> content_hashes) {
        return coordinator.acquirePrefix(coordinator.ProbePrefix(content_hashes), ++coordinator.next_access_epoch_);
    }
    // A probe over the prompt's tokens too (tails included), acquired as an
    // admission would.
    static auto MatchPromptPrefix(CacheCoordinator& coordinator, const CacheCoordinator::PromptPrefix& prompt) {
        return coordinator.acquirePrefix(coordinator.ProbePromptPrefix(prompt), ++coordinator.next_access_epoch_);
    }

    static std::uint64_t NextAccessEpoch(CacheCoordinator& coordinator) { return ++coordinator.next_access_epoch_; }

    // The planner's Device eviction of one entry, with its tail cascade,
    // counted as an admission's.
    static bool EvictCachedBlock(CacheCoordinator& coordinator, std::uint32_t group_id, CacheBlockLocation location) {
        return coordinator.evictCachedBlock(group_id, location, &CacheCoordinator::DeviceEvictionStats::admission,
                                            /*write_back_segment=*/nullptr);
    }

    // One Device entry's eviction without its cascade, counted under `reason`
    // and recorded into `segment` as an eviction write-back would (nullptr
    // drops it); false when the entry is pinned or gone.
    static bool EvictAloneInto(CacheCoordinator& coordinator, std::uint32_t group_id, CacheBlockLocation location,
                               CacheCoordinator::DeviceEvictionCounts CacheCoordinator::DeviceEvictionStats::*reason,
                               CacheCoordinator::EvictedSegment* segment) {
        return coordinator.evictCachedBlockAlone(group_id, location, reason, segment).has_value();
    }

    static bool RecordDeviceEviction(CacheCoordinator& coordinator, const CacheKey& key,
                                     CacheCoordinator::EvictedSegment& segment) {
        const auto location = coordinator.locationOf(coordinator.pool_, key.group_id, key);
        return location && coordinator.evictCachedBlock(key.group_id, *location,
                                                        &CacheCoordinator::DeviceEvictionStats::admission, &segment);
    }

    // A Device-cached block, located rather than referenced: a held reference
    // would pin it against eviction.
    static std::optional<CacheBlockLocation> DeviceLocationOf(const CacheCoordinator& coordinator,
                                                              std::size_t group_index, const CacheKey& key) {
        return coordinator.locationOf(coordinator.pool_, group_index, key);
    }

    // A Host LRU eviction of the entry under `key` (counted under lru), with
    // its State cascade; false when it is pinned or absent.
    static bool EvictHostCachedBlock(CacheCoordinator& coordinator, const CacheKey& key) {
        const std::optional<CacheBlockLocation> location =
            coordinator.locationOf(*coordinator.host_pool_, key.group_id, key);
        if (!location) {
            return false;
        }
        return coordinator.evictHostCachedBlock(key.group_id, *location, &CacheCoordinator::HostEvictionStats::lru);
    }

    static const BoundaryTree& BoundaryTreeOf(const CacheCoordinator& coordinator) {
        return coordinator.boundary_tree_;
    }

    static auto NodeEntries(const CacheCoordinator& coordinator, const BlockPool& pool, const ContentHash& hash) {
        return coordinator.NodeEntries(pool, *coordinator.boundary_tree_.Find(hash));
    }

    // Where the boundary tree disagrees with the indexes, recounted from
    // scratch: every indexed hash has a node whose entry, State, prefix-closed
    // and pair counts match its entries on both tiers, and a node without
    // entries stays only for a child or a write-back on its way to the Host.
    // The CheckpointTree tracks only checkpoints a node holds, and ranks each
    // by the residency the node's counts give. Empty when they agree. Call it
    // between coordinator operations.
    static std::vector<std::string> BoundaryTreeMismatches(const CacheCoordinator& coordinator) {
        struct Counts {
            std::array<std::int32_t, 2> entries{};
            std::array<std::int32_t, 2> state{};
            std::array<std::int32_t, 2> closed{};
            std::int32_t pairs{0};
        };
        std::unordered_map<ContentHash, Counts> expected;
        const std::array<const BlockPool*, 2> pools{&coordinator.pool_, coordinator.host_pool_};
        for (std::size_t i = 0; i < coordinator.groups_.size(); ++i) {
            const PrefixCacheIndex& index = coordinator.groups_[i].Index();
            const bool state = coordinator.GroupIsState(i);
            const bool closed = coordinator.groups_[i].Matcher().IsPrefixClosed();
            for (std::size_t tier = 0; tier < pools.size(); ++tier) {
                if (pools[tier] == nullptr) {
                    continue;
                }
                index.ForEachEntry(
                    *pools[tier], [&](const CacheKey& key, const PrefixCacheIndex::CachedBlockSnapshot&) {
                        Counts& counts = expected[key.content_hash];
                        ++counts.entries[tier];
                        counts.state[tier] += state ? 1 : 0;
                        counts.closed[tier] += closed ? 1 : 0;
                        if (tier == BoundaryTree::kDevice && closed && pools[BoundaryTree::kHost] != nullptr &&
                            index.Contains(*pools[BoundaryTree::kHost], key)) {
                            ++counts.pairs;
                        }
                    });
            }
        }
        std::vector<std::string> mismatches;
        const BoundaryTree& tree = coordinator.boundary_tree_;
        for (const auto& [hash, counts] : expected) {
            const BoundaryTree::Node* node = tree.Find(hash);
            if (node == nullptr) {
                mismatches.push_back("no node for the indexed hash " + hash.ToString());
                continue;
            }
            for (const std::size_t tier : {BoundaryTree::kDevice, BoundaryTree::kHost}) {
                if (node->Entries(tier) != counts.entries[tier] || node->StateEntries(tier) != counts.state[tier] ||
                    node->ClosedEntries(tier) != counts.closed[tier]) {
                    mismatches.push_back("entry counts of " + hash.ToString() + " on tier " + std::to_string(tier));
                }
            }
            if (node->ClosedPairs() != counts.pairs) {
                mismatches.push_back("pair count of " + hash.ToString());
            }
        }
        // Every node, by the tree's own links: one without entries hangs on
        // for a child or a store; and none is out of reach.
        const CheckpointTree& checkpoints = coordinator.Checkpoints();
        std::size_t reached = 0;
        std::size_t tracked_checkpoints = 0;
        std::vector<const BoundaryTree::Node*> pending;
        for (const BoundaryTree::Node* root = tree.FirstRoot(); root != nullptr; root = root->next_sibling) {
            pending.push_back(root);
        }
        while (!pending.empty()) {
            const BoundaryTree::Node* node = pending.back();
            pending.pop_back();
            ++reached;
            if (!expected.contains(node->hash) && node->first_child == nullptr && node->PendingHostStores() == 0) {
                mismatches.push_back("nothing holds the node " + node->hash.ToString());
            }
            if (checkpoints.Contains(node->hash)) {
                ++tracked_checkpoints;
                const CheckpointTree::Residency residency = tree.CheckpointResidency(*node);
                if (!residency.held) {
                    mismatches.push_back("the checkpoint tree tracks the unheld checkpoint " + node->hash.ToString());
                }
                // Distance is reported for a checkpoint present on either tier.
                const bool present = residency.complete[0] || residency.complete[1] || residency.pending_host_store;
                if (checkpoints.Distance(node->hash).has_value() != present) {
                    mismatches.push_back("checkpoint presence of " + node->hash.ToString());
                }
                for (const CacheTier tier : {CacheTier::kDevice, CacheTier::kHost}) {
                    if (checkpoints.AgedDistance(node->hash, tier) != 0 &&
                        !residency.complete[static_cast<std::size_t>(tier)]) {
                        mismatches.push_back("aged checkpoint " + node->hash.ToString() + " incomplete on tier " +
                                             std::to_string(static_cast<int>(tier)));
                    }
                }
            }
            for (const BoundaryTree::Node* child = node->first_child; child != nullptr; child = child->next_sibling) {
                pending.push_back(child);
            }
        }
        if (reached != tree.Size()) {
            mismatches.push_back("nodes out of reach of the roots");
        }
        if (tracked_checkpoints != checkpoints.Size()) {
            mismatches.push_back("the checkpoint tree tracks a checkpoint without a boundary node");
        }
        return mismatches;
    }
};

inline auto MatchPrefixForTest(CacheCoordinator& coordinator, std::span<const std::string> content_hashes) {
    return CacheCoordinatorTestAccess::MatchPrefix(coordinator, content_hashes);
}

inline void CacheFullBlocksForTest(CacheCoordinator& coordinator, std::span<BlockTable> tables,
                                   std::span<const std::string> content_hashes, std::int32_t first_slot = 0) {
    coordinator.CacheFullBlocks(tables, content_hashes, CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                first_slot, CacheBoundaryKind::kChunk);
}

// Publishes the newly hashed pages [first_new_prefix_page, prefix_hashes.size())
// at a page-aligned point; publish_state marks it a natural checkpoint of the
// State groups (their slot 1 is registered under the last page's hash).
inline void CacheCompletedBlocksForTest(CacheCoordinator& coordinator, std::span<BlockTable> tables,
                                        std::span<const std::string> prefix_hashes, std::uint64_t access_epoch,
                                        std::int32_t first_new_prefix_page, std::int32_t num_computed_tokens,
                                        CacheBoundaryKind boundary_kind, bool stream_completed_to_host,
                                        bool publish_state) {
    _assert(tables.size() == static_cast<std::size_t>(coordinator.NumGroups()), "tables/groups size mismatch");
    _assert(first_new_prefix_page >= 0 && static_cast<std::size_t>(first_new_prefix_page) < prefix_hashes.size(),
            "completed page range must be non-empty");
    const RequestProgress progress{
        .completed_pages =
            CompletedPages{
                .prefix_hashes = prefix_hashes,
                .first_new_prefix_page = first_new_prefix_page,
                .publish_state = publish_state,
                .boundary_kind = boundary_kind,
                .stream_completed_to_host = stream_completed_to_host,
            },
        .num_computed_tokens = num_computed_tokens,
    };
    coordinator.CacheCompletedBlocks(tables, progress, access_epoch);
}

// Admits every History group with the same demand prototype as a new request;
// a State group takes `state_step` (kRotate: a prefill step, the shape of a
// first admission).
// An admission that spends `copy_in_blocks`, the copy-in blocks the request
// holds for the point its progress publishes (OwnerTail::kCopiedIn), and
// reserves those of `copy_in_point` for the slots whose blocks it allocates
// (AdmissionResult::copy_in_blocks).
inline std::optional<CacheCoordinator::AdmissionResult> AdmitForTest(
    CacheCoordinator& coordinator, std::vector<BlockTable>& tables, CacheCoordinator::PrefixProbe&& prefix,
    GroupDemand prototype, const RequestProgress& progress, StateStep state_step,
    std::span<std::vector<CacheBlockRef>> copy_in_blocks, std::optional<CacheCoordinator::CopyInPoint> copy_in_point) {
    std::vector<GroupDemand> demands;
    demands.reserve(tables.size());
    for (std::size_t i = 0; i < tables.size(); ++i) {
        GroupDemand demand = prototype;
        demand.table = &tables[i];
        if (coordinator.GroupIsState(i)) {
            demand.extent = state_step;
            demand.reserve_tokens = 0;
        }
        demands.push_back(demand);
    }
    return coordinator.Admit(std::move(prefix), demands, progress, copy_in_blocks, copy_in_point,
                             /*request_access_epoch=*/std::nullopt, /*evict_blocks_past_segments=*/false,
                             /*session_id=*/{}, /*cache_group_id=*/{});
}

inline std::optional<CacheCoordinator::AdmissionResult> AdmitForTest(
    CacheCoordinator& coordinator, std::vector<BlockTable>& tables, CacheCoordinator::PrefixProbe&& prefix,
    GroupDemand prototype, const RequestProgress& progress,
    StateStep state_step = StateStep{StateStep::Kind::kRotate}) {
    return AdmitForTest(coordinator, tables, std::move(prefix), prototype, progress, state_step,
                        /*copy_in_blocks=*/{}, /*copy_in_point=*/std::nullopt);
}

// The overloads without progress admit a request that has computed nothing
// yet: nothing to publish, nothing for retention to reclaim.
inline std::optional<CacheCoordinator::AdmissionResult> AdmitForTest(CacheCoordinator& coordinator,
                                                                     std::vector<BlockTable>& tables,
                                                                     CacheCoordinator::PrefixProbe&& prefix,
                                                                     GroupDemand prototype) {
    return AdmitForTest(coordinator, tables, std::move(prefix), prototype, RequestProgress{});
}

inline std::optional<CacheCoordinator::AdmissionResult> AdmitForTest(CacheCoordinator& coordinator,
                                                                     std::vector<BlockTable>& tables,
                                                                     GroupDemand prototype,
                                                                     const RequestProgress& progress) {
    return AdmitForTest(coordinator, tables, coordinator.ProbePrefix({}), prototype, progress);
}

inline std::optional<CacheCoordinator::AdmissionResult> AdmitForTest(CacheCoordinator& coordinator,
                                                                     std::vector<BlockTable>& tables,
                                                                     GroupDemand prototype) {
    return AdmitForTest(coordinator, tables, prototype, RequestProgress{});
}

inline std::optional<CacheCoordinator::AdmissionResult> AdmitForTest(CacheCoordinator& coordinator,
                                                                     std::vector<BlockTable>& tables,
                                                                     std::int32_t num_tokens) {
    return AdmitForTest(coordinator, tables, GroupDemand{.extent = DenseGrowth{num_tokens}});
}

}  // namespace tokenspeed
