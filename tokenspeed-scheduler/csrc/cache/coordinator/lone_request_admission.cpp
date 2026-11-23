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

#include "cache/coordinator/lone_request_admission.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <memory_resource>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cache/coordinator/admission_planner.h"
#include "cache/core/block_placement_demand.h"
#include "core/profile_span.h"
#include "utils.h"

namespace tokenspeed {
namespace {

// The admission planner of a lone request. Once the segment rules have
// nothing left, it takes what they passed over again, block by block: a State
// checkpoint leaves alone with every State group's at its hash, as a nested
// one does; a page leaves with the blocks of its boundary that can leave and
// everything the Device holds under it, climbing nowhere, so the Device's
// coverage stays closed. It also lends a compaction (LoneRequestAdmission::Compact) its
// capacity checks and victim expansion.
class LoneRequestPlanner final : public AdmissionPlanner {
public:
    using AdmissionPlanner::AdmissionPlanner;

    // The parents a dense layout of the request's blocks and this demand
    // would take, every parent empty: per group ParentsNeededForBucketsAndPairs
    // over the blocks that keep their bucket (bucket_pinned, by bucket) plus
    // the bucket-exact demand, the blocks that may take any bucket
    // (free_movers, a request's unwritten reserve) plus the dense demand, and
    // the demand's same-bucket pairs (copyInReservations). The gate of
    // LoneRequestAdmission::Compact, and what prepares the planner's
    // capacity bound used by the compaction.
    std::int64_t DenseLayoutParents(std::span<const std::vector<std::int64_t>> bucket_pinned,
                                    std::span<const std::int64_t> free_movers) const {
        _assert(bucket_pinned.size() == groups_.size() && free_movers.size() == groups_.size(),
                "one bucket layout and one free-mover count per group");
        const AdmissionCapacity capacity = buildCapacity();
        std::int64_t parents = 0;
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            BlockPlacementDemand block_demand = capacity.block_demands[i];
            _assert(bucket_pinned[i].size() == block_demand.blocks_by_bucket.size(),
                    "bucket layout must match the shard count");
            for (std::size_t b = 0; b < bucket_pinned[i].size(); ++b) {
                block_demand.blocks_by_bucket[b] += bucket_pinned[i][b];
            }
            block_demand.any_bucket_blocks += free_movers[i];
            const std::vector<std::int64_t> holes(bucket_pinned[i].size(), 0);
            std::vector<std::int64_t> scratch(bucket_pinned[i].size(), 0);
            parents += ParentsNeededForBucketsAndPairs(block_demand, holes, pool_.Packing(groups_[i].Id()), scratch);
        }
        return parents;
    }

    // The cache entry at `location`, expanded as the block-by-block pass
    // takes it (expandCandidatePastSegments): the entry alone, or under
    // the write-back policy with everything the Device holds under its node;
    // nullopt when no entry the index alone holds is there (a free slot, a
    // block a table or a ticket holds), when a block under it is in use or
    // kept, or when the entry itself is one this admission keeps -- a page,
    // tail block or State checkpoint its probe hit, a restore's claim -- or
    // a slot the retry reclaims as the request's own: the candidate pass
    // excludes those through AdmissionCandidateConstraints, so a compaction never clears
    // them either.
    std::optional<VictimSet> PlanBlockAlone(std::uint32_t group_id, CacheBlockLocation location,
                                            const AdmissionCandidateConstraints& constraints,
                                            AdmissionSelectionState& selection) const {
        if (!groups_[group_id].Index().UnpinnedKeyAt(pool_, location) ||
            constraints.protected_locations.contains(location) ||
            constraints.request_reclaim_locations.contains(location)) {
            return std::nullopt;
        }
        std::optional<ExpandedVictimSet> expanded = expandCandidatePastSegments(
            CacheEvictionCandidates::MakeCandidate(groups_[group_id], location,
                                                   groups_[group_id].Index().MetadataFor(pool_, location)),
            constraints, selection);
        if (!expanded) {
            return std::nullopt;
        }
        _assert(std::ranges::find(expanded->set.members, location, &VictimSet::Member::location) !=
                    expanded->set.members.end(),
                "a victim set holds the candidate it was chosen for");
        return std::move(expanded->set);
    }

protected:
    std::optional<VictimCandidate> nextCandidatePastSegments(AdmissionSelectionState& selection) const override {
        if (selection.next_passed_over == selection.passed_over.size()) {
            return std::nullopt;
        }
        return selection.passed_over[selection.next_passed_over++];
    }

    std::optional<ExpandedVictimSet> expandCandidatePastSegments(const VictimCandidate& candidate,
                                                                 const AdmissionCandidateConstraints& constraints,
                                                                 AdmissionSelectionState& selection) const override {
        if (selection.set_member_by_location.contains(candidate.location)) {
            return std::nullopt;
        }
        const BoundaryTree::Node* candidate_node = candidateNode(candidate);
        if (auto set = expandAloneOrReclaim(candidate, candidate_node, constraints, selection)) {
            return ExpandedVictimSet{.set = std::move(*set)};
        }
        if (store_.GroupIsState(candidate.group_id)) {
            auto set = expandCheckpointsAlone(*candidate_node, constraints, selection);
            if (!set) return std::nullopt;
            return ExpandedVictimSet{.set = std::move(*set)};
        }
        // A page leaving block by block may leave blocks in use at its
        // boundary, so its set starts no segment: no later walk takes it for
        // that node's subtree.
        const std::vector<NodeEntry> candidate_entries = store_.NodeEntries(pool_, *candidate_node);
        return walkSubtree(*candidate_node, *candidate_node, candidate_entries, /*climbed=*/{},
                           /*top_keeps_unavailable=*/true, constraints, selection);
    }
};

}  // namespace

std::optional<AdmissionResult> LoneRequestAdmission::Admit(PrefixProbe&& prefix, std::span<const GroupDemand> demands,
                                                           const RequestProgress& progress,
                                                           std::span<std::vector<CacheBlockRef>> copy_in_blocks,
                                                           std::optional<CopyInPoint> copy_in_point,
                                                           std::optional<std::uint64_t> request_access_epoch,
                                                           const std::string& session_id,
                                                           const std::string& cache_group_id) {
    ProfileSpan profile_span{"LoneRequestAdmission::Admit"};
    return coordinator_.AdmitWith<LoneRequestPlanner>(std::move(prefix), demands, progress, copy_in_blocks,
                                                      copy_in_point, request_access_epoch, session_id, cache_group_id);
}

std::optional<AdmissionResult> LoneRequestAdmission::Restore(const RetractionImage& image,
                                                             std::span<const GroupDemand> demands,
                                                             std::optional<CopyInPoint> copy_in_point,
                                                             std::uint64_t request_access_epoch) {
    // Claimed blocks one per parent beside cached pages can bind more parents
    // than a dense layout of the image needs, and no release will come.
    return coordinator_.RestoreWithoutClaimsIfUnfit(image, demands, copy_in_point, request_access_epoch);
}

std::optional<AdmissionResult> LoneRequestAdmission::Compact(PrefixProbe&& prefix, std::span<const GroupDemand> demands,
                                                             const RequestProgress& progress,
                                                             std::span<std::vector<CacheBlockRef>> copy_in_blocks,
                                                             std::optional<CopyInPoint> copy_in_point) {
    CacheStore& store = coordinator_.Store();
    CacheEvictor& evictor = coordinator_.Evictor();
    const bool write_back = coordinator_.WritesBackEvictions();
    _assert(demands.size() == store.Groups().size(), "demands/groups size mismatch");
    store.ValidateProgress(progress);
    _assert(copy_in_blocks.empty() || copy_in_blocks.size() == store.Groups().size(),
            "copy-in blocks are held per cache group");
    _assert(copy_in_blocks.empty() || copy_in_point || progress.num_computed_tokens,
            "held copy-in blocks serve the point an admission reserves for or the step publishes");
    const std::vector<GroupDemand> replayed = store.ReplayedDemands(prefix, demands);
    if (!replayed.empty()) {
        demands = replayed;
    }
    const std::optional<std::int32_t> reserve_point =
        copy_in_point ? std::optional<std::int32_t>{copy_in_point->tokens} : std::nullopt;
    // Whether the retry admits now: LoneRequestAdmission::Admit's plan.
    const auto admission_fits = [&] {
        // The aging read for each plan, as an admission reads it: the
        // compaction's evictions can empty the checkpoint tree.
        return planAdmissionWith<LoneRequestPlanner>(
                   store, coordinator_.History().CheckpointAgingFor(CacheTier::kDevice), PrefixProbe{prefix}, demands,
                   progress.num_computed_tokens, reserve_point, /*shapes=*/{},
                   /*planned_claim_locations=*/{}, write_back)
            .has_value();
    };

    // The blocks of the request in each packed, token-spanned group -- the
    // table's, and the copy-in blocks it holds beside them -- and at a first
    // admission the Device pages and tail blocks its probe hits, by what may
    // happen to them. An unwritten block moves anywhere and is zeroed, unless
    // a copy-in block is held for its slot: the two share a bucket (the
    // copy-in's contract), so each re-places within it. A written block moves
    // by a same-bucket copy, a published one with its entry; a hit page or
    // tail block moves like a published block. Fixed: a page this admission
    // publishes (the runtime stores it before the plan's copies land), a
    // block another holder pins (a store ticket, a retained checkpoint,
    // another table), a hit another request pins. Slots the retry reclaims
    // first (a released State input, expired window blocks) are not the
    // request's footprint and are left out.
    struct Member {
        enum class Kind : std::uint8_t { kUnwritten, kCopyIn, kWritten, kPublished, kHit, kTail, kFixed };
        Kind kind;
        // Whether the block keeps its bucket: every one but a free unwritten
        // block.
        bool in_bucket;
        CacheBlockLocation location;
        std::int32_t slot;  // the table slot (a copy-in block: the slot it serves), -1 for a probe block
        std::size_t hit;    // the probe block's pin in hit_pins (kHit, kTail; 0 otherwise)
    };
    // Every Device block the probe hits is held from here to the return --
    // the pages and tail blocks of every group, compacted or not, and on the
    // D role each State group's checkpoint at the prompt end: clearing a
    // destination slot evicts what its entry takes along (a tail's
    // companions, a page's checkpoints), and an unheld hit taken so would be
    // gone when the retry acquires it. The slot itself may be a block the
    // probe leaves a hole (the D role's partial tail block, the prompt's last
    // page past its bound).
    std::vector<CacheBlockRef> hit_pins;
    std::vector<std::vector<Member>> members(store.Groups().size());
    std::vector<std::vector<std::int64_t>> bucket_pinned(store.Groups().size());
    std::vector<std::int64_t> free_movers(store.Groups().size(), 0);
    // Per group, the slots the held copy-in blocks serve.
    std::vector<CacheStore::BlockRange> copy_in_slots(store.Groups().size());
    for (std::size_t i = 0; i < store.Groups().size(); ++i) {
        const GroupAllocator& allocator = store.Groups()[i].Allocator();
        const PrefixCacheIndex& index = store.Groups()[i].Index();
        const BlockTable& table = *demands[i].table;
        bucket_pinned[i].assign(static_cast<std::size_t>(allocator.ShardCount()), 0);
        const bool compacts =
            store.Geometry()[i].HasTokenSpan() && store.DevicePool().Packing(store.Groups()[i].Id()) > 1;
        // The copy-in blocks the request holds, each with the block it serves
        // (Admit reserves them so): that block keeps its bucket as well.
        std::vector<bool> copy_in_beside(static_cast<std::size_t>(table.NumBlocks()), false);
        if (!copy_in_blocks.empty()) {
            copy_in_slots[i] =
                store.CopyInSlots(i, copy_in_point ? copy_in_point->tokens : *progress.num_computed_tokens);
            _assert(copy_in_blocks[i].size() ==
                        static_cast<std::size_t>(copy_in_slots[i].past_end - copy_in_slots[i].first),
                    "the request holds one copy-in block per copy-in slot");
            for (std::size_t j = 0; j < copy_in_blocks[i].size(); ++j) {
                const CacheBlockRef& block = copy_in_blocks[i][j];
                if (!block) {
                    continue;
                }
                const std::int32_t slot = copy_in_slots[i].first + static_cast<std::int32_t>(j);
                _assert(slot < table.NumBlocks(), "a copy-in block is held with the block it serves");
                copy_in_beside[static_cast<std::size_t>(slot)] = true;
                ++bucket_pinned[i][static_cast<std::size_t>(allocator.BucketOf(block->Location()))];
                if (compacts) {
                    members[i].push_back(Member{.kind = Member::Kind::kCopyIn,
                                                .in_bucket = true,
                                                .location = block->Location(),
                                                .slot = slot,
                                                .hit = 0});
                }
            }
        }
        std::int32_t first_slot = table.ReclaimedPrefixBlocks();
        if (store.GroupIsState(i)) {
            first_slot = std::get<StateStep>(demands[i].extent).kind == StateStep::Kind::kKeep ? 0 : 1;
        } else if (progress.num_computed_tokens) {
            first_slot = std::max(first_slot, store.Geometry()[i].ExpiredBlockEndAt(store.Groups()[i].Spec(),
                                                                                    *progress.num_computed_tokens));
        }
        std::vector<bool> publishing(static_cast<std::size_t>(table.NumBlocks()), false);
        if (compacts && progress.completed_pages) {
            for (const CacheStore::PublishedRange& range : store.CompletedBlockRanges(i, *progress.completed_pages)) {
                for (std::size_t j = 0; j < range.keys.size(); ++j) {
                    const std::int32_t slot = range.first_cache_block + static_cast<std::int32_t>(j);
                    if (slot < table.NumBlocks()) {
                        publishing[static_cast<std::size_t>(slot)] = true;
                    }
                }
            }
        }
        for (std::int32_t slot = allocator.KeptHeadBlocks() > 0 ? 0 : first_slot; slot < table.NumBlocks(); ++slot) {
            if (slot >= allocator.KeptHeadBlocks() && slot < first_slot) {
                continue;
            }
            const CacheBlockRef& block = table.Blocks()[static_cast<std::size_t>(slot)];
            if (!block) {
                continue;
            }
            const bool cached = index.Contains(block);
            Member::Kind kind = Member::Kind::kFixed;
            if (compacts && !publishing[static_cast<std::size_t>(slot)] &&
                (cached ? block.use_count() == 2 : block.unique())) {
                // Rows past the computed frontier are unwritten (a draft row a
                // verify step rewrites counts as unwritten too).
                kind = cached ? Member::Kind::kPublished
                       : progress.num_computed_tokens &&
                               static_cast<std::int64_t>(slot) * store.Geometry()[i].BlockGranularity() >=
                                   *progress.num_computed_tokens
                           ? Member::Kind::kUnwritten
                           : Member::Kind::kWritten;
            }
            const bool in_bucket = kind != Member::Kind::kUnwritten || copy_in_beside[static_cast<std::size_t>(slot)];
            if (in_bucket) {
                ++bucket_pinned[i][static_cast<std::size_t>(allocator.BucketOf(block->Location()))];
            } else {
                ++free_movers[i];
            }
            if (compacts) {
                members[i].push_back(Member{
                    .kind = kind, .in_bucket = in_bucket, .location = block->Location(), .slot = slot, .hit = 0});
            }
        }
        if (prefix.group_keys == nullptr || i >= prefix.device.per_group.size()) {
            continue;
        }
        // The probe's hit pages, then its Device tail's blocks: the retry
        // claims the pages, and copies a tail block into the hitter's own
        // fresh block from where it was before this plan moved it
        // (RecordAdmittedPages).
        std::vector<CacheBlockLocation> probe_blocks = index.MatchedLocations(
            store.DevicePool(), (*prefix.group_keys)[i], /*begin_block=*/0, prefix.device.per_group[i]);
        const std::size_t num_full_page_blocks = probe_blocks.size();
        if (const auto& tail = prefix.device.tail; tail && !tail->group_keys[i].empty()) {
            const std::vector<CacheBlockLocation> tail_blocks =
                index.MatchedLocations(store.DevicePool(), tail->group_keys[i], /*begin_block=*/0, tail->per_group[i]);
            probe_blocks.insert(probe_blocks.end(), tail_blocks.begin(), tail_blocks.end());
        }
        if (!compacts) {
            // A group the compaction does not gather (a State group among
            // them, with the prompt-end checkpoint the D role claims): its
            // hits are held where they are.
            if (const auto& state = prefix.state_at_prompt_end; state && !state->group_keys[i].empty()) {
                const std::vector<CacheBlockLocation> state_blocks = index.MatchedLocations(
                    store.DevicePool(), state->group_keys[i], /*begin_block=*/0, state->per_group[i]);
                probe_blocks.insert(probe_blocks.end(), state_blocks.begin(), state_blocks.end());
            }
            for (const CacheBlockLocation location : probe_blocks) {
                hit_pins.push_back(index.Find(store.DevicePool(), *index.KeyOf(store.DevicePool(), location)));
            }
            continue;
        }
        for (std::size_t b = 0; b < probe_blocks.size(); ++b) {
            const CacheBlockLocation location = probe_blocks[b];
            const CacheKey key = *index.KeyOf(store.DevicePool(), location);
            Member::Kind kind = Member::Kind::kFixed;
            std::size_t hit = 0;
            if (!index.Inspect(store.DevicePool(), key)->pinned) {
                hit = hit_pins.size();
                hit_pins.push_back(index.Find(store.DevicePool(), key));
                kind = b < num_full_page_blocks ? Member::Kind::kHit : Member::Kind::kTail;
            }
            ++bucket_pinned[i][static_cast<std::size_t>(allocator.BucketOf(location))];
            members[i].push_back(Member{.kind = kind, .in_bucket = true, .location = location, .slot = -1, .hit = hit});
        }
    }

    auto member_resource = std::make_unique<std::pmr::monotonic_buffer_resource>();
    const CheckpointAging aging = coordinator_.History().CheckpointAgingFor(CacheTier::kDevice);
    const LoneRequestPlanner planner{store,         aging,
                                     demands,       progress.num_computed_tokens,
                                     reserve_point, prefix,
                                     /*shapes=*/{}, /*planned_claim_locations=*/{},
                                     write_back};
    // Not fragmentation: the bound's own caveats, which compacting would
    // only churn at. The request waits as before. The count covers the
    // groups compaction gathers and the demand, not the parents the probe
    // holds in the others (a hit page or tail block of a group of one block
    // per parent, a State checkpoint), so passing it does not promise that
    // the retry fits (docs/design/scheduler.md §1.4).
    if (planner.DenseLayoutParents(bucket_pinned, free_movers) > store.DevicePool().NumLcmBlocks()) {
        return std::nullopt;
    }

    AdmissionSelectionState selection{store, member_resource.get()};
    const AdmissionCandidateConstraints constraints = BuildAdmissionCandidateConstraints(
        store, demands, progress.num_computed_tokens, prefix, /*planned_claim_locations=*/{}, &selection.scratch);
    AdmissionResult result{
        .new_page_ids = std::vector<std::vector<std::int32_t>>(store.Groups().size()),
        .pages_to_copy = std::vector<std::vector<std::pair<std::int32_t, std::int32_t>>>(store.Groups().size()),
    };
    const std::uint64_t moves_before = compaction_moves_;
    bool fits = false;
    // At most one of the probe's tail blocks moves per compaction, in any
    // group: the retry copies each from where it was (RecordAdmittedPages),
    // and two such copies landing on each other's old places would swap their
    // bytes, which no order of the plan's copies can do.
    std::int64_t tails_moved = 0;
    for (std::size_t i = 0; i < store.Groups().size() && !fits; ++i) {
        if (members[i].empty()) {
            continue;
        }
        const std::uint32_t group_id = store.Groups()[i].Id();
        GroupAllocator& allocator = store.Groups()[i].Allocator();
        PrefixCacheIndex& index = store.Groups()[i].Index();
        BlockTable& table = *demands[i].table;
        const std::int32_t packing = store.DevicePool().Packing(group_id);
        const std::int32_t buckets = allocator.ShardCount();
        // The parents holding the request's blocks: one with a block that
        // cannot move is kept whatever it holds, then the densest, and of
        // equally dense ones the one with more bucket-bound blocks. So a
        // parent of free blocks alone is vacated before an equally dense one
        // holding a bound block: its blocks re-place in any bucket, with no
        // copy and no displacement. The order also keeps every kept parent
        // bound through the moves. A free block gives its slot up (below)
        // only to a bound block of a vacated parent, which ranks after every
        // kept one. So a kept parent holding one block of the request, a
        // free one (never a fixed parent), loses none: every parent ranked
        // after it holds one free block alone. Any other kept parent keeps a
        // block of the request (held, so no clearing evicts it) until the
        // bound block takes the slot a free one gave up.
        struct Parent {
            std::int32_t id;
            std::int32_t count{0};
            std::int32_t bound{0};
            bool fixed{false};
            std::vector<std::size_t> members;
        };
        std::vector<Parent> parents;
        std::unordered_map<std::int32_t, std::size_t> parent_index;
        for (std::size_t m = 0; m < members[i].size(); ++m) {
            const Member& member = members[i][m];
            const auto [it, inserted] = parent_index.try_emplace(member.location.lcm_block_id, parents.size());
            if (inserted) {
                parents.push_back(Parent{.id = member.location.lcm_block_id});
            }
            Parent& parent = parents[it->second];
            ++parent.count;
            parent.bound += member.in_bucket ? 1 : 0;
            parent.fixed = parent.fixed || member.kind == Member::Kind::kFixed;
            parent.members.push_back(m);
        }
        std::ranges::sort(parents, [](const Parent& a, const Parent& b) {
            return std::tuple(!a.fixed, -a.count, -a.bound, a.id) < std::tuple(!b.fixed, -b.count, -b.bound, b.id);
        });
        const std::vector<std::int64_t> holes(static_cast<std::size_t>(buckets), 0);
        const std::size_t kept = static_cast<std::size_t>(
            std::max<std::int64_t>(ParentsNeededForBuckets(bucket_pinned[i], holes, free_movers[i], packing),
                                   std::ranges::count_if(parents, &Parent::fixed)));
        // Vacated in the reverse order -- the sparsest first, of those the
        // one with the fewest bucket-bound blocks -- a parent whole or not
        // at all: a block moved out of a parent that keeps another of the
        // request's blocks frees nothing.
        for (std::size_t p = parents.size(); p-- > kept && !fits;) {
            // Nor does one moved out of a parent the retry cannot empty:
            // beside the request's blocks there, every occupant must be a
            // slot the retry reclaims as the request's own or a cache entry
            // its block-by-block pass takes. A held page keeps the parent
            // bound, so it is skipped like a parent with a block that finds
            // no slot. So is one whose moves would make a second tail block
            // of the probe move: it waits for a later compaction when
            // another moved first (tails_moved), and is never vacated when it
            // holds two of them.
            const auto tails_here = std::ranges::count_if(
                parents[p].members, [&](std::size_t m) { return members[i][m].kind == Member::Kind::kTail; });
            bool vacatable = tails_moved + tails_here <= 1;
            for (std::int32_t child = 0; child < packing && vacatable; ++child) {
                const CacheBlockLocation slot{.lcm_block_id = parents[p].id, .slot_index = child};
                vacatable = !store.DevicePool().IsOccupied(slot) ||
                            std::ranges::any_of(parents[p].members,
                                                [&](std::size_t m) { return members[i][m].location == slot; }) ||
                            constraints.request_reclaim_locations.contains(slot) ||
                            planner.PlanBlockAlone(group_id, slot, constraints, selection).has_value();
            }
            if (!vacatable) {
                continue;
            }
            struct Move {
                std::size_t member{0};
                CacheBlockLocation destination;
                // The cache entry at the destination and what it takes along.
                std::optional<VictimSet> clearing;
            };
            std::vector<Move> moves;
            std::vector<CacheBlockLocation> reserved;
            // A slot of a kept parent, in `bucket` or in any, `first_parent`'s
            // before the others when given: free, or held by a cache entry
            // nothing holds that the block-by-block pass may take. A block of
            // the request's own, or one in use, never.
            const auto find_destination = [&](std::optional<std::int32_t> bucket,
                                              std::optional<std::int32_t> first_parent) -> std::optional<Move> {
                std::vector<std::int32_t> searched;
                searched.reserve(kept + 1);
                if (first_parent) {
                    searched.push_back(*first_parent);
                }
                for (std::size_t k = 0; k < kept; ++k) {
                    if (parents[k].id != first_parent) {
                        searched.push_back(parents[k].id);
                    }
                }
                for (const std::int32_t parent : searched) {
                    for (std::int32_t child = bucket.value_or(0); child < packing; child += bucket ? buckets : 1) {
                        const CacheBlockLocation candidate{.lcm_block_id = parent, .slot_index = child};
                        if (std::ranges::find(reserved, candidate) != reserved.end()) {
                            continue;
                        }
                        if (!store.DevicePool().IsOccupied(candidate)) {
                            return Move{.destination = candidate};
                        }
                        if (std::optional<VictimSet> clearing =
                                planner.PlanBlockAlone(group_id, candidate, constraints, selection)) {
                            return Move{.destination = candidate, .clearing = std::move(clearing)};
                        }
                    }
                }
                return std::nullopt;
            };
            // Bucket-bound blocks first; a free block takes what is left, in
            // any bucket.
            std::vector<std::size_t> order = parents[p].members;
            std::ranges::stable_partition(order, [&](std::size_t m) { return members[i][m].in_bucket; });
            bool whole = true;
            for (const std::size_t m : order) {
                const Member& member = members[i][m];
                const std::int32_t bucket = allocator.BucketOf(member.location);
                std::optional<Move> move = find_destination(
                    member.in_bucket ? std::optional<std::int32_t>{bucket} : std::nullopt, std::nullopt);
                // No slot of its bucket left in the kept parents: one of the
                // request's free blocks there gives its slot up and re-places
                // anywhere (page-cyclic sharding keeps a written block's bytes
                // with their owner) -- in its own parent first. It leaves
                // before the member arrives, and a parent it left empty would
                // be released (unbound from the group) in between. The parent
                // order above leaves another of the request's blocks there in
                // any case; re-placing in its own parent first keeps the block
                // itself there whenever a free or clearable slot is left in it
                // (in another bucket: none of this bucket is, in any kept
                // parent), whatever the order.
                for (std::size_t k = 0; k < kept && !move && member.in_bucket; ++k) {
                    for (const std::size_t u : parents[k].members) {
                        const Member& other = members[i][u];
                        if (other.in_bucket || allocator.BucketOf(other.location) != bucket ||
                            std::ranges::find(reserved, other.location) != reserved.end()) {
                            continue;
                        }
                        std::optional<Move> displaced = find_destination(std::nullopt, other.location.lcm_block_id);
                        if (!displaced) {
                            break;
                        }
                        displaced->member = u;
                        reserved.push_back(displaced->destination);
                        moves.push_back(std::move(*displaced));
                        move = Move{.destination = other.location};
                        break;
                    }
                }
                if (!move) {
                    whole = false;
                    break;
                }
                move->member = m;
                reserved.push_back(move->destination);
                moves.push_back(std::move(*move));
            }
            if (!whole) {
                continue;
            }
            for (Move& move : moves) {
                Member& member = members[i][move.member];
                if (move.clearing && store.DevicePool().IsOccupied(move.destination)) {
                    evictor.EvictPlanned(
                        std::span{&*move.clearing, 1}, write_back,
                        CacheEvictor::VictimReasons{.victim = &CacheEvictor::DeviceEvictionStats::compaction,
                                                    .stranded_suffix = &CacheEvictor::DeviceEvictionStats::compaction,
                                                    .request_reclaim = &CacheEvictor::DeviceEvictionStats::compaction});
                }
                FatalCheck(!store.DevicePool().IsOccupied(move.destination),
                           "compaction destination changed before placement");
                // A kept parent never empties before the moves into it
                // (above): one released would read unoccupied and unbound.
                _assert(store.DevicePool().BoundGroup(move.destination.lcm_block_id) == group_id,
                        "a compaction destination lies in a parent bound to its group");
                if (member.kind == Member::Kind::kHit || member.kind == Member::Kind::kTail) {
                    // Pinned through the move: the entry follows its copy,
                    // and the pin with it.
                    CacheBlockRef& pin = hit_pins[member.hit];
                    CacheBlockRef fresh = store.DevicePool().AcquireBlockAt(group_id, move.destination);
                    _assert(allocator.BucketOf(fresh->Location()) == allocator.BucketOf(member.location),
                            "a copied block keeps its bucket");
                    result.pages_to_copy[i].emplace_back(allocator.ResolveCacheBlockId(*pin),
                                                         allocator.ResolveCacheBlockId(*fresh));
                    index.Relocate(store.DevicePool(), member.location, fresh);
                    pin = std::move(fresh);
                } else if (member.kind == Member::Kind::kCopyIn) {
                    // Held outside the table for the slot it serves: re-placed
                    // in its bucket, never zeroed. The copy-in fills it whole,
                    // in this very plan when the retry spends it, and the
                    // runtime copies before it zeroes.
                    CacheBlockRef fresh = store.DevicePool().AcquireBlockAt(group_id, move.destination);
                    _assert(allocator.BucketOf(fresh->Location()) == allocator.BucketOf(member.location),
                            "a copy-in block keeps its bucket");
                    copy_in_blocks[i][static_cast<std::size_t>(member.slot - copy_in_slots[i].first)] =
                        std::move(fresh);
                } else {
                    // The block the slot gives up is the entry's until the
                    // entry has followed the copy; it leaves with `replaced`.
                    const CacheBlockRef replaced = allocator.SwapInFreshBlock(
                        store.DevicePool(), table, member.slot, table.AvailableTokens(), move.destination);
                    const CacheBlockRef& moved = table.Blocks()[static_cast<std::size_t>(member.slot)];
                    _assert(!member.in_bucket ||
                                allocator.BucketOf(moved->Location()) == allocator.BucketOf(member.location),
                            "a block bound to its bucket keeps it");
                    if (member.kind == Member::Kind::kUnwritten) {
                        // A free block an earlier vacate round displaced
                        // can be displaced again, and the page it leaves
                        // can take a copy: it is no longer zeroed.
                        std::erase(result.new_page_ids[i], allocator.ResolveCacheBlockId(*replaced));
                        result.new_page_ids[i].push_back(allocator.ResolveCacheBlockId(*moved));
                    } else {
                        result.pages_to_copy[i].emplace_back(allocator.ResolveCacheBlockId(*replaced),
                                                             allocator.ResolveCacheBlockId(*moved));
                    }
                    if (member.kind == Member::Kind::kPublished) {
                        index.Relocate(store.DevicePool(), member.location, moved);
                    }
                }
                member.location = move.destination;
                ++compaction_moves_;
            }
            tails_moved += tails_here;
            fits = admission_fits();
        }
    }
    // A set whose every block was dropped records nothing.
    evictor.DropEmptySegments();
    if (compaction_moves_ == moves_before) {
        return std::nullopt;
    }
    ++compactions_;
    return result;
}

}  // namespace tokenspeed
