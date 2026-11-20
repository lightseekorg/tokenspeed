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

#include "cache/coordinator/cache_coordinator.h"

#include "cache/coordinator/admission_planner.h"
#include "core/profile_span.h"
#include "cache/coordinator/victim_set.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "utils.h"

namespace tokenspeed {
namespace {

// Acquires the copy-in blocks an admission reserves in group `group_id`
// (copy_ins, slot order) into `reserved`, aligned to the copy-in point's slots
// [first_slot, ...): each in its slot's bucket. A slot `table` holds already
// (a prefix's tail block, a rebuilt slot) has its block's bucket; a slot the
// dense growth fills takes its own block in its copy-in block's bucket,
// acquired in the same call ahead of the copy-in blocks, and returned for
// GroupAllocator::Acquire to place. So every bucket-exact block of the group
// is acquired before the growth's other blocks, the order the planner's fit
// assumes.
struct PairedBlocks {
    std::int32_t first_slot{0};
    std::vector<CacheBlockRef> blocks;
};

PairedBlocks acquireCopyIns(BlockPool& pool, std::uint32_t group_id, const GroupAllocator& allocator,
                            const BlockTable& table, std::span<const CopyInReservation> copy_ins,
                            std::int32_t first_slot, std::vector<CacheBlockRef>& reserved) {
    PairedBlocks paired{.first_slot = table.NumBlocks()};
    std::vector<std::int32_t> buckets;
    for (const CopyInReservation& copy_in : copy_ins) {
        _assert(copy_in.bucket >= 0, "the plan buckets every copy-in block");
        if (copy_in.slot < table.NumBlocks()) {
            const CacheBlockRef& block = table.Blocks()[static_cast<std::size_t>(copy_in.slot)];
            _assert(block && allocator.BucketOf(block->Location()) == copy_in.bucket,
                    "a copy-in block shares its slot's bucket");
            continue;
        }
        if (buckets.empty()) {
            paired.first_slot = copy_in.slot;
        }
        buckets.push_back(copy_in.bucket);
    }
    const std::size_t num_paired = buckets.size();
    for (const CopyInReservation& copy_in : copy_ins) {
        buckets.push_back(copy_in.bucket);
    }
    std::vector<CacheBlockRef> acquired = pool.AcquireBlocksInBuckets(group_id, buckets);
    FatalCheck(acquired.size() == buckets.size(), "admission plan no longer fits the block pool");
    auto next = std::make_move_iterator(acquired.begin());
    paired.blocks.assign(next, next + static_cast<std::ptrdiff_t>(num_paired));
    next += static_cast<std::ptrdiff_t>(num_paired);
    for (const CopyInReservation& copy_in : copy_ins) {
        reserved[static_cast<std::size_t>(copy_in.slot - first_slot)] = *next++;
    }
    return paired;
}

}  // namespace

bool CacheCoordinator::writableImageSlot(std::size_t group_index, const ImageTable& shape,
                                         std::int32_t slot_index) const {
    if (GroupIsState(group_index)) {
        return slot_index == 1;  // the live slot; slot 0 is only ever read
    }
    // The tail's blocks: the slots of the page the computed frontier ends
    // inside. The request writes the last of them and later publishes all of
    // them under the page's hash, so none may stay the tail entry's block.
    const std::int32_t block_granularity = store_.Geometry()[group_index].BlockGranularity();
    // The image's blocks can end past INT32_MAX; the tokens they hold fit.
    const std::int32_t num_tokens =
        CheckedInt32(std::int64_t{shape.num_blocks} * block_granularity - shape.available_tokens);
    const std::int32_t tail_start = (num_tokens - num_tokens % store_.PrefixGranularity()) / block_granularity;
    return slot_index >= tail_start;
}

void CacheCoordinator::ReleaseStateInputs(std::span<BlockTable> tables) {
    _assert(tables.size() == store_.Groups().size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
        if (GroupIsState(i) && tables[i].NumBlocks() == 2) {
            store_.Groups()[i].Allocator().ReleaseStateInput(tables[i]);
        }
    }
}

std::optional<AdmissionResult> CacheCoordinator::Admit(PrefixProbe&& prefix, std::span<const GroupDemand> demands,
                                                       const RequestProgress& progress,
                                                       std::span<std::vector<CacheBlockRef>> copy_in_blocks,
                                                       std::optional<CopyInPoint> copy_in_point,
                                                       std::optional<std::uint64_t> request_access_epoch,
                                                       const std::string& session_id,
                                                       const std::string& cache_group_id) {
    ProfileSpan profile_span{"CacheCoordinator::Admit"};
    return AdmitWith<AdmissionPlanner>(std::move(prefix), demands, progress, copy_in_blocks, copy_in_point,
                                       request_access_epoch, session_id, cache_group_id);
}

std::vector<GroupDemand> CacheCoordinator::prepareAdmission(const PrefixProbe& prefix,
                                                            std::span<const GroupDemand> demands,
                                                            const RequestProgress& progress,
                                                            std::span<std::vector<CacheBlockRef>> copy_in_blocks,
                                                            std::optional<CopyInPoint> copy_in_point) {
    _assert(demands.size() == store_.Groups().size(), "demands/groups size mismatch");
    for (std::size_t i = 0; i < demands.size(); ++i) {
        _assert(demands[i].table != nullptr, "group demand requires a block table");
        _assert(GroupIsState(i) == std::holds_alternative<StateStep>(demands[i].extent),
                "a State group takes a StateStep; a token-spanned group a dense or sparse extent");
        demands[i].table->CountActiveParentsIn(&store_.ActiveParentGauge());
    }
    store_.ValidateProgress(progress);
    validateCopyInBlocks(progress, copy_in_blocks);
    _assert(!copy_in_point || copy_in_point->tokens > progress.num_computed_tokens.value_or(0),
            "a copy-in point lies past the computed frontier");
    return store_.ReplayedDemands(prefix, demands);
}

AdmissionResult CacheCoordinator::commitAdmission(AdmissionPlan&& plan, std::span<const GroupDemand> demands,
                                                  const RequestProgress& progress,
                                                  std::span<std::vector<CacheBlockRef>> copy_in_blocks,
                                                  std::optional<CopyInPoint> copy_in_point,
                                                  std::optional<std::uint64_t> request_access_epoch,
                                                  const std::string& session_id, const std::string& cache_group_id) {
    if (request_access_epoch.has_value()) {
        _assert(*request_access_epoch > 0 && *request_access_epoch <= history_.LatestAccessEpoch(),
                "request access epoch was not issued by this coordinator");
    }
    const std::uint64_t access_epoch =
        request_access_epoch.has_value() ? *request_access_epoch : history_.IssueAccessEpoch();
    // Registered before anything is acquired or published under the epoch.
    SessionAccess session_access;
    if (!request_access_epoch) {
        if (!cache_group_id.empty()) {
            session_access = SessionAccess{sessions_, access_epoch, sessions_.AcquireCacheGroupOwner(cache_group_id)};
        } else if (!session_id.empty()) {
            session_access = SessionAccess{sessions_, access_epoch, sessions_.AcquireSessionOwner(session_id)};
        }
    }
    // The boundaries inside a block, per tier: the hitter's table ends in a
    // private copy of the Device tail's partial block, or in a fresh page the
    // Host tail is loaded into.
    const std::int32_t device_tail_tokens = plan.prefix.device.tail ? plan.prefix.device.tail->tail_tokens : 0;
    const std::int32_t host_tail_tokens = plan.prefix.host.tail ? plan.prefix.host.tail->tail_tokens : 0;
    const std::int32_t admitted_tokens =
        std::max(plan.prefix.device.num_common_tokens, plan.prefix.host.num_common_tokens);
    const std::int32_t kv_tokens = std::max(plan.prefix.device.kv_tokens, plan.prefix.host.kv_tokens);
    const bool state_prefix_cached = plan.prefix.state_at_prompt_end.has_value();
    AcquiredPrefix acquired_prefix = acquirePrefix(std::move(plan.prefix), access_epoch);
    if (!request_access_epoch.has_value()) {
        history_.RecordAdmission(access_epoch);
    }
    AdmissionResult result{
        .device_prefix_tokens = acquired_prefix.device.num_common_tokens,
        .host_prefix_tokens = acquired_prefix.host.num_common_tokens,
        .junction_tokens = kv_tokens > admitted_tokens ? kv_tokens : 0,
        .state_prefix_cached = state_prefix_cached,
        .access_epoch = access_epoch,
        .new_page_ids = std::vector<std::vector<std::int32_t>>(store_.Groups().size()),
        .pages_to_copy = std::vector<std::vector<std::pair<std::int32_t, std::int32_t>>>(store_.Groups().size()),
        .state_checkpoint_blocks = std::vector<CacheBlockRef>(store_.Groups().size()),
        .copy_in_blocks = std::vector<std::vector<CacheBlockRef>>(copy_in_point ? store_.Groups().size() : 0),
        .session_access = std::move(session_access),
    };
    for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
        if (GroupIsState(i)) {
            continue;  // a State hit is installed below, as slot 0
        }
        if (acquired_prefix.device.num_common_tokens > 0) {
            store_.Groups()[i].Allocator().ClaimHitBlocks(*demands[i].table,
                                                          std::move(acquired_prefix.device.per_group[i]));
        }
    }
    // A State group's first admission shapes its two slots: slot 0 the one
    // checkpoint the probe matched on the Device (a Host checkpoint lands by
    // load below; the D's prompt-end checkpoint is claimed here), else null
    // until the step below installs the zero state; slot 1 is acquired with
    // the step below. The shaping is the first rotation: a shaped-now table
    // rotates nothing further.
    std::vector<bool> shaped_now(store_.Groups().size(), false);
    for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
        if (!GroupIsState(i) || demands[i].table->NumBlocks() != 0) {
            continue;
        }
        const StateStep::Kind kind = std::get<StateStep>(demands[i].extent).kind;
        if (kind != StateStep::Kind::kRotate && kind != StateStep::Kind::kLand) {
            continue;
        }
        shaped_now[i] = true;
        CacheBlockRef input;
        for (CacheBlockRef& block : acquired_prefix.device.per_group[i].blocks) {
            if (block) {
                _assert(!input, "a State group matches one checkpoint");
                input = std::move(block);
            }
        }
        if (!acquired_prefix.state_at_prompt_end.empty() && acquired_prefix.state_at_prompt_end[i]) {
            _assert(!input, "a State group matches one checkpoint");
            input = std::move(acquired_prefix.state_at_prompt_end[i]);
        }
        if (const std::optional<CacheKey> key = store_.Groups()[i].Index().KeyOf(store_.DevicePool(), input);
            key && store_.Boundaries().Tails().Contains(key->content_hash)) {
            result.resumed_tail_hash = key->content_hash;
        }
        store_.Groups()[i].Allocator().ShapeStateTable(*demands[i].table, std::move(input));
    }
    // A remote landing fills whole every block that ends at or below the
    // prompt end (slot b holds tokens [b*g, (b+1)*g)): a copy destination,
    // not a page to zero.
    const auto landed_whole = [&](std::size_t i, std::int32_t slot) {
        const std::int32_t landed_tokens = demands[i].landed_tokens;
        return landed_tokens > 0 &&
               static_cast<std::int64_t>(slot + 1) * store_.Geometry()[i].BlockGranularity() <= landed_tokens;
    };
    // Gives a hitter's table a private block in `slot`: a copy of the shared
    // block there (copy-on-write), or a fresh one where the slot was a hole,
    // zeroed unless the peer's landing fills it.
    const auto privatize_slot = [&](std::size_t i, BlockTable& table, std::int32_t slot,
                                    std::int32_t available_tokens) {
        GroupAllocator& allocator = store_.Groups()[i].Allocator();
        const CacheBlockRef replaced = allocator.SwapInFreshBlock(store_.DevicePool(), table, slot, available_tokens,
                                                                  /*destination=*/std::nullopt);
        const std::int32_t destination = allocator.ResolveCacheBlockId(*table.Blocks()[static_cast<std::size_t>(slot)]);
        if (replaced) {
            result.pages_to_copy[i].emplace_back(allocator.ResolveCacheBlockId(*replaced), destination);
        } else if (!landed_whole(i, slot)) {
            result.new_page_ids[i].push_back(destination);
        }
    };
    // Record cached victims for write-back even when this request releases
    // them: a published input can still resume another request. Uncached
    // slots and explicitly superseded inputs contribute no store records.
    // A reclaimable table block may still be pinned by both the request and
    // cache here. Evict what is already free, then slide the request tables and
    // retry the blocks whose request reference has just been released.
    const std::vector<CacheEvictor::UnevictedVictim> prospective_victims = [&] {
        ProfileSpan evicting{"CacheCoordinator::commitAdmission (evict)"};
        return evictor_.EvictPlanned(
            plan.victims, write_back_evictions_,
            CacheEvictor::VictimReasons{.victim = &CacheEvictor::DeviceEvictionStats::admission,
                                        .stranded_suffix = &CacheEvictor::DeviceEvictionStats::stranded_suffix,
                                        .request_reclaim = &CacheEvictor::DeviceEvictionStats::request_reclaim});
    }();
    // Held while the groups register under the publication's nodes.
    BoundaryTree::Hold publication;
    if (progress.completed_pages) {
        publication = store_.Boundaries().LinkPublication(*progress.completed_pages);
    }
    for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
        BlockTable& table = *demands[i].table;
        if (progress.completed_pages) {
            cacheDeviceCompletedBlocksForGroup(i, table, *progress.completed_pages, access_epoch);
        }
        if (GroupIsState(i)) {
            // The step's release: slot 0 leaves the table (the index keeps it
            // when published, else it returns to the pool and may fund this
            // very admission); a rotation moves the block the last forward
            // wrote -- published just above when the point was a checkpoint
            // -- into slot 0 and leaves slot 1 for the acquisition below.
            if (table.NumBlocks() == 0) {
                continue;
            }
            switch (std::get<StateStep>(demands[i].extent).kind) {
                case StateStep::Kind::kRotate:
                    if (!shaped_now[i]) {
                        store_.Groups()[i].Allocator().RotateState(table);
                    }
                    break;
                case StateStep::Kind::kRelease:
                    store_.Groups()[i].Allocator().ReleaseStateInput(table);
                    break;
                case StateStep::Kind::kKeep:
                case StateStep::Kind::kLand:
                    break;
            }
            continue;
        }
        if (progress.num_computed_tokens) {
            store_.Groups()[i].Allocator().ReclaimExpired(
                store_.DevicePool(), table,
                store_.Geometry()[i].ExpiredBlockEndAt(store_.Groups()[i].Spec(), *progress.num_computed_tokens));
        }
    }
    if (progress.completed_pages) {
        const CompletedPages& completed = *progress.completed_pages;
        // A model without a prefix-closed group may register nothing under
        // the deepest boundary linked (a chunk end of State groups alone): it
        // leaves now.
        publication.reset();
        recordCheckpoints(completed, access_epoch);
        recordPublishedTail(completed);
    }
    for (const auto& [group_id, location, reason, segment] : prospective_victims) {
        if (!evictor_.EvictCachedBlock(group_id, location, reason, segment)) {
            FatalCheck(!store_.DevicePool().IsOccupied(location), "admission victim changed before acquisition");
        }
    }
    // A set whose every block was dropped records nothing: this admission's
    // empty records go, and an earlier call left none.
    evictor_.DropEmptySegments();

    for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
        // A group's bucket-exact blocks (a hitter's tail copies, the Host
        // extension, the copy-in blocks reserved below) are acquired before
        // its dense growth, which is what the planner's fit assumes.
        const GroupDemand& demand = demands[i];
        BlockTable& table = *demand.table;
        // An owner that keeps writing past its published prompt end moves off
        // every block it just registered into the copy-in block it holds for
        // the slot, in that block's bucket: a published block is never
        // written again and never carries a second key. A slot whose key
        // another request published first was not registered and stays the
        // owner's to write; its copy-in block goes back unused. (A State
        // group's published slot 1 moves to slot 0 by rotation instead.)
        if (!copy_in_blocks.empty()) {
            const CacheStore::BlockRange slots = store_.CopyInSlots(i, *progress.num_computed_tokens);
            _assert(slots.past_end <= table.NumBlocks(), "the table holds every copy-in slot of its frontier");
            std::vector<CacheBlockRef>& held = copy_in_blocks[i];
            GroupAllocator& allocator = store_.Groups()[i].Allocator();
            for (std::int32_t slot = slots.first; slot < slots.past_end; ++slot) {
                if (!store_.Groups()[i].Index().Contains(table.Blocks()[static_cast<std::size_t>(slot)])) {
                    continue;
                }
                const CacheBlockRef published =
                    allocator.SwapInBlock(table, slot, std::move(held[static_cast<std::size_t>(slot - slots.first)]));
                result.pages_to_copy[i].emplace_back(
                    allocator.ResolveCacheBlockId(*published),
                    allocator.ResolveCacheBlockId(*table.Blocks()[static_cast<std::size_t>(slot)]));
            }
            held.clear();  // the unused ones go back
        }
        // A first admission's table ends in the tail's blocks: shared full
        // pages, then this request's own copies.
        if (!GroupIsState(i) && device_tail_tokens > 0 && table.NumBlocks() > 0) {
            const TailSlots slots =
                tailSlots(store_.Geometry()[i], store_.PrefixGranularity(), acquired_prefix.device.num_common_tokens);
            _assert(slots.past_end == table.NumBlocks(), "the Device tail's blocks end the claimed table");
            // A sliding group's tail blocks below its window stay holes.
            const std::int32_t first =
                slots.first + store_.FirstTailBlockNeeded(static_cast<std::int32_t>(i), device_tail_tokens);
            for (std::int32_t slot = first; slot < slots.past_end; ++slot) {
                privatize_slot(i, table, slot, slots.available_tokens);
            }
        }
        if (GroupIsState(i)) {
            const StateStep& step = std::get<StateStep>(demand.extent);
            GroupAllocator& allocator = store_.Groups()[i].Allocator();
            if (table.NumBlocks() == 0) {
                continue;  // never shaped: a step other than a rotation leaves it so
            }
            // A Host checkpoint is loaded into a fresh slot 0.
            if (!acquired_prefix.host.per_group.empty()) {
                for (CacheBlockRef& host_block : acquired_prefix.host.per_group[i].blocks) {
                    if (!host_block) {
                        continue;
                    }
                    _assert(!table.Blocks()[0], "a State group matches one checkpoint");
                    std::vector<CacheBlockRef> destination = store_.DevicePool().AcquireBlocksInBuckets(
                        store_.Groups()[i].Id(),
                        std::array<std::int32_t, 1>{allocator.BucketOf(host_block->Location())});
                    FatalCheck(destination.size() == 1, "admission plan no longer fits the block pool");
                    allocator.InstallStateInput(table, std::move(destination.front()));
                    result.load_pairs.push_back(BlockTransfer{.group_id = store_.Groups()[i].Id(),
                                                              .source = std::move(host_block),
                                                              .destination = table.Blocks()[0]});
                }
            }
            // A prompt with no checkpoint to read starts from the zero state:
            // a fresh zeroed block in slot 0, so the request holds its two
            // blocks from its first admission on.
            if (step.kind == StateStep::Kind::kRotate && shaped_now[i] && !table.Blocks()[0]) {
                FatalCheck(allocator.AcquireStateInput(store_.DevicePool(), table),
                           "admission plan no longer fits the block pool");
                result.new_page_ids[i].push_back(allocator.ResolveCacheBlockId(*table.Blocks()[0]));
            }
            if (step.kind == StateStep::Kind::kRotate || step.kind == StateStep::Kind::kLand) {
                FatalCheck(allocator.AcquireStateLive(store_.DevicePool(), table),
                           "admission plan no longer fits the block pool");
                const std::int32_t live = allocator.ResolveCacheBlockId(*table.Blocks()[1]);
                if (step.kind == StateStep::Kind::kLand && state_prefix_cached) {
                    // The peer omits this state transfer. A bootstrap token
                    // can finish the request before any decode writes live.
                    _assert(static_cast<bool>(table.Blocks()[0]), "an exact state hit must have an input checkpoint");
                    result.pages_to_copy[i].emplace_back(allocator.ResolveCacheBlockId(*table.Blocks()[0]), live);
                } else if (step.kind != StateStep::Kind::kLand) {
                    // A landing's live slot is filled whole by the peer's
                    // transfer, like a copy destination: nothing to zero.
                    result.new_page_ids[i].push_back(live);
                }
            }
            if (step.checkpoint) {
                _assert(step.kind == StateStep::Kind::kRotate, "an in-forward checkpoint rides a prefill chunk");
                CacheBlockRef checkpoint = allocator.AcquireStateCheckpoint(store_.DevicePool(), table);
                FatalCheck(static_cast<bool>(checkpoint), "admission plan no longer fits the block pool");
                result.new_page_ids[i].push_back(allocator.ResolveCacheBlockId(*checkpoint));
                result.state_checkpoint_blocks[i] = std::move(checkpoint);
            }
            continue;
        }
        if (!acquired_prefix.host.per_group.empty() && !acquired_prefix.host.per_group[i].blocks.empty()) {
            const std::int32_t available = host_tail_tokens > 0
                                               ? tailSlots(store_.Geometry()[i], store_.PrefixGranularity(),
                                                           acquired_prefix.host.num_common_tokens)
                                                     .available_tokens
                                               : 0;
            // A Host tail's blocks land on fresh destinations (one load row
            // each); a sliding group's tail blocks below its window stay holes.
            store_.Groups()[i].Allocator().AppendHostExtension(store_.DevicePool(), table,
                                                               std::move(acquired_prefix.host.per_group[i].blocks),
                                                               result.load_pairs, available);
        }
        // The copy-in blocks this admission reserves, each with the block it
        // serves (acquireCopyIns), the paired ones placed by the growth.
        PairedBlocks paired{.first_slot = demand.table->NumBlocks()};
        if (copy_in_point) {
            const CacheStore::BlockRange slots = store_.CopyInSlots(i, copy_in_point->tokens);
            result.copy_in_blocks[i].resize(static_cast<std::size_t>(slots.past_end - slots.first));
            paired = acquireCopyIns(store_.DevicePool(), store_.Groups()[i].Id(), store_.Groups()[i].Allocator(),
                                    *demand.table, plan.copy_ins[i], slots.first, result.copy_in_blocks[i]);
        }
        const std::int32_t first_new_block = demand.table->NumBlocks();
        const bool acquired = store_.Groups()[i].Allocator().Acquire(
            store_.DevicePool(), *demand.table, store_.Geometry()[i].PlanAcquire(*demand.table, demand),
            paired.first_slot, std::move(paired.blocks));
        FatalCheck(acquired, "admission plan no longer fits the block pool");
        // Only a dense table's slots are block-aligned to the prompt the way
        // landed_whole assumes.
        _assert(demand.landed_tokens == 0 || std::holds_alternative<DenseGrowth>(demand.extent),
                "a remote landing's extent is a dense growth");
        const std::span<const CacheBlockRef> blocks = demand.table->Blocks();
        for (std::int32_t block = first_new_block; block < demand.table->NumBlocks(); ++block) {
            if (!blocks[static_cast<std::size_t>(block)] || landed_whole(i, block)) {
                continue;
            }
            result.new_page_ids[i].push_back(
                store_.Groups()[i].Allocator().ResolveCacheBlockId(*blocks[static_cast<std::size_t>(block)]));
        }
    }
    return result;
}

std::optional<AdmissionResult> CacheCoordinator::Restore(const RetractionImage& image,
                                                         std::span<const GroupDemand> demands,
                                                         std::optional<CopyInPoint> copy_in_point,
                                                         std::uint64_t request_access_epoch) {
    // Under segment eviction the claims can pin the only evictable cache.
    return restore(image, demands, copy_in_point, request_access_epoch,
                   /*plan_without_claims_if_unfit=*/write_back_evictions_);
}

std::optional<AdmissionResult> CacheCoordinator::RestoreWithoutClaimsIfUnfit(const RetractionImage& image,
                                                                             std::span<const GroupDemand> demands,
                                                                             std::optional<CopyInPoint> copy_in_point,
                                                                             std::uint64_t request_access_epoch) {
    return restore(image, demands, copy_in_point, request_access_epoch, /*plan_without_claims_if_unfit=*/true);
}

std::optional<AdmissionResult> CacheCoordinator::restore(const RetractionImage& image,
                                                         std::span<const GroupDemand> demands,
                                                         std::optional<CopyInPoint> copy_in_point,
                                                         std::uint64_t request_access_epoch,
                                                         bool plan_without_claims_if_unfit) {
    _assert(demands.size() == store_.Groups().size(), "demands/groups size mismatch");
    _assert(image.tables.size() == store_.Groups().size(), "image/groups size mismatch");
    _assert(request_access_epoch > 0 && request_access_epoch <= history_.LatestAccessEpoch(),
            "request access epoch was not issued by this coordinator");
    for (const GroupDemand& demand : demands) {
        _assert(demand.table != nullptr, "group demand requires a block table");
        demand.table->CountActiveParentsIn(&store_.ActiveParentGauge());
    }
    // The rebuilt slots among the copy-in slots, each in its imaged block's
    // bucket: the partial page at the image's frontier, which the restore
    // always copies (never claims).
    const auto rebuilt_copy_ins = [&](std::size_t i) {
        std::vector<CopyInReservation> rebuilt;
        if (!copy_in_point) {
            return rebuilt;
        }
        const CacheStore::BlockRange range = store_.CopyInSlots(i, copy_in_point->tokens);
        const ImageTable& shape = image.tables[i];
        for (std::int32_t slot = range.first; slot < std::min(range.past_end, shape.num_blocks); ++slot) {
            const auto it = std::ranges::find(shape.slots, slot, &ImageSlot::slot_index);
            _assert(it != shape.slots.end() && writableImageSlot(i, shape, slot),
                    "a rebuilt copy-in slot is a slot of the partial page at the image's frontier");
            rebuilt.push_back(CopyInReservation{
                .slot = slot, .bucket = store_.Groups()[i].Allocator().BucketOf(it->block->Location())});
        }
        return rebuilt;
    };
    // A Host L2 slot whose key still has a Device-cached canonical block is
    // claimed instead of copied: the same bytes by construction (the victim's
    // own block, cache-only since it was freed, the canonical block its table
    // already pointed at, or any block published under the key since, wherever
    // it sits). Those blocks are protected from the planner's
    // eviction like prefix hits; every other slot is a bucket-constrained
    // fresh block.
    // A published slot the resumed request will never write again is claimed
    // from the Device index when its key is still cached there; the slot it
    // keeps writing -- the partial page at its computed frontier -- always
    // gets a fresh copy, so the published block stays immutable.
    const auto claimable = [&](std::size_t i, const ImageTable& shape, const ImageSlot& slot) {
        return slot.InHostCache() && !writableImageSlot(i, shape, slot.slot_index);
    };
    // When a plan that claims fails and plan_without_claims_if_unfit, plan
    // once more claiming nothing; every slot is then a fresh copy, an L2
    // slot republished at its ACK (docs/design/cache-concepts.md,
    // AdmissionPlanner).
    bool claim_cached_slots = true;
    std::optional<AdmissionPlan> candidate;
    for (const bool claim : {true, false}) {
        claim_cached_slots = claim;
        std::vector<PlannedShape> shapes(store_.Groups().size());
        std::vector<std::pair<std::uint32_t, CacheBlockLocation>> planned_claim_locations;
        for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
            _assert(demands[i].table != nullptr && demands[i].table->NumBlocks() == 0,
                    "a restore rebuilds into an empty table");
            const GroupAllocator& allocator = store_.Groups()[i].Allocator();
            const ImageTable& shape = image.tables[i];
            shapes[i].blocks_by_bucket.assign(static_cast<std::size_t>(allocator.ShardCount()), 0);
            shapes[i].num_blocks = shape.num_blocks;
            shapes[i].available_tokens = shape.available_tokens;
            shapes[i].rebuilt_copy_ins = rebuilt_copy_ins(i);
            for (const ImageSlot& slot : shape.slots) {
                _assert(static_cast<bool>(slot.block), "image slot lost its block");
                const CacheBlockRef canonical = claim && claimable(i, shape, slot)
                                                    ? store_.Groups()[i].Index().Find(store_.DevicePool(), slot.key)
                                                    : CacheBlockRef{};
                if (canonical) {
                    planned_claim_locations.emplace_back(store_.Groups()[i].Id(), canonical->Location());
                    continue;
                }
                ++shapes[i].blocks_by_bucket[static_cast<std::size_t>(allocator.BucketOf(slot.block->Location()))];
            }
        }
        candidate =
            planAdmission(store_, history_.CheckpointAgingFor(CacheTier::kDevice), prober_.ProbePrefix({}), demands,
                          /*num_computed_tokens=*/std::nullopt,
                          copy_in_point ? std::optional<std::int32_t>{copy_in_point->tokens} : std::nullopt, shapes,
                          planned_claim_locations, write_back_evictions_);
        if (candidate || planned_claim_locations.empty() || !plan_without_claims_if_unfit) {
            break;
        }
    }
    if (!candidate) {
        return std::nullopt;
    }
    // Claim before evicting, as Admit does: a claimed entry is pinned, so a
    // victim's eviction cascade (a tail's companions, a page's state
    // checkpoints) cannot take a block the image is about to resume on.
    std::vector<std::vector<CacheBlockRef>> claimed(store_.Groups().size());
    for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
        const ImageTable& shape = image.tables[i];
        claimed[i].resize(shape.slots.size());
        for (std::size_t j = 0; j < shape.slots.size(); ++j) {
            const ImageSlot& slot = shape.slots[j];
            if (!claim_cached_slots || !claimable(i, shape, slot) ||
                !store_.Groups()[i].Index().Find(store_.DevicePool(), slot.key)) {
                continue;
            }
            // The planner protected this entry; claim it like a prefix hit.
            const std::array<CacheKey, 1> keys{slot.key};
            PrefixMatch hit = store_.Groups()[i].Index().AcquireMatched(
                store_.DevicePool(), keys, /*begin_block=*/0, GroupPrefixProbe{.hits = {1}}, request_access_epoch,
                SessionForAccess(request_access_epoch));
            claimed[i][j] = std::move(hit.blocks.front());
        }
    }
    // Nothing is published here and the tables start empty, so no set is the
    // request's own reclaim: every planned victim is a cache-only block the
    // planner saw as evictable (or a tail companion or same-hash State
    // checkpoint an earlier victim's cascade already took). Under the
    // write-back policy each set is recorded whole as one segment for this
    // plan's drain, as Admit does.
    for (const VictimSet& set : candidate->victims) {
        _assert(std::ranges::none_of(set.members, &VictimSet::Member::is_request_reclaim),
                "a restore rebuilds empty tables and reclaims nothing of its own");
    }
    const CacheEvictor::VictimReasons restore_reasons{.victim = &CacheEvictor::DeviceEvictionStats::restore,
                                                      .stranded_suffix = &CacheEvictor::DeviceEvictionStats::restore,
                                                      .request_reclaim = &CacheEvictor::DeviceEvictionStats::restore};
    for (const CacheEvictor::UnevictedVictim& victim :
         evictor_.EvictPlanned(candidate->victims, write_back_evictions_, restore_reasons)) {
        FatalCheck(!store_.DevicePool().IsOccupied(victim.location), "restore victim changed before acquisition");
    }
    // A set whose every block was dropped records nothing: this restore's
    // empty records go, and an earlier call left none.
    evictor_.DropEmptySegments();

    AdmissionResult result{
        .access_epoch = request_access_epoch,
        .new_page_ids = std::vector<std::vector<std::int32_t>>(store_.Groups().size()),
        .pages_to_copy = std::vector<std::vector<std::pair<std::int32_t, std::int32_t>>>(store_.Groups().size()),
        .copy_in_blocks = std::vector<std::vector<CacheBlockRef>>(copy_in_point ? store_.Groups().size() : 0),
    };
    for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
        GroupAllocator& allocator = store_.Groups()[i].Allocator();
        BlockTable& table = *demands[i].table;
        const ImageTable& shape = image.tables[i];
        FatalCheck(allocator.AcquireShape(store_.DevicePool(), table, shape, std::move(claimed[i])),
                   "restore plan no longer fits the block pool");
        for (const ImageSlot& slot : shape.slots) {
            const CacheBlockRef& destination = table.Blocks()[static_cast<std::size_t>(slot.slot_index)];
            if (slot.InHostCache() && store_.Groups()[i].Index().Contains(destination)) {
                continue;  // claimed: already the published Device block, nothing to copy
            }
            // The writable slot's row is unkeyed: its destination is the
            // request's private copy and must not be republished at the ACK.
            const bool republish = claimable(i, shape, slot);
            BlockTransfer transfer{
                .group_id = store_.Groups()[i].Id(),
                .source = slot.block,
                .destination = destination,
                .key = republish ? slot.key : CacheKey{},
                .logical_block_index = slot.logical_block_index,
                .boundary_kind = slot.boundary_kind,
            };
            // A published slot comes back from Host L2 (and is re-published at
            // the ACK); a private slot from the snapshot pool. Neither is
            // listed in new_page_ids: the copy fills the whole block, as a
            // Host hit's load-back does.
            if (slot.InHostCache()) {
                result.load_pairs.push_back(std::move(transfer));
            } else {
                result.snapshot_pairs.push_back(std::move(transfer));
            }
        }
        if (GroupIsState(i)) {
            continue;  // the imaged two slots are the whole table
        }
        // The copy-in blocks, each with the rebuilt or grown block it serves
        // (acquireCopyIns), before the reserve's other blocks.
        PairedBlocks paired{.first_slot = table.NumBlocks()};
        if (copy_in_point) {
            const CacheStore::BlockRange slots = store_.CopyInSlots(i, copy_in_point->tokens);
            result.copy_in_blocks[i].resize(static_cast<std::size_t>(slots.past_end - slots.first));
            paired = acquireCopyIns(store_.DevicePool(), store_.Groups()[i].Id(), allocator, table,
                                    candidate->copy_ins[i], slots.first, result.copy_in_blocks[i]);
        }
        // Only the reserve appended beyond the imaged shape is fresh, unfilled
        // Device memory for the plan to zero.
        const std::int32_t first_new_block = table.NumBlocks();
        FatalCheck(allocator.Acquire(store_.DevicePool(), table, store_.Geometry()[i].PlanAcquire(table, demands[i]),
                                     paired.first_slot, std::move(paired.blocks)),
                   "restore plan no longer fits the block pool");
        const std::span<const CacheBlockRef> blocks = table.Blocks();
        for (std::int32_t block = first_new_block; block < table.NumBlocks(); ++block) {
            result.new_page_ids[i].push_back(allocator.ResolveCacheBlockId(*blocks[static_cast<std::size_t>(block)]));
        }
    }
    return result;
}

}  // namespace tokenspeed
