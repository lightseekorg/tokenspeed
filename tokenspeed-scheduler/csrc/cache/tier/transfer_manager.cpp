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

#include "cache/tier/transfer_manager.h"

#include <algorithm>
#include <iterator>
#include <numeric>
#include <unordered_set>
#include <utility>

#include <spdlog/spdlog.h>

#include "utils.h"

namespace tokenspeed {

TierTransferManager::TierTransferManager(CacheCoordinator& coordinator) : coordinator_{coordinator} {
    const auto num_groups = static_cast<std::size_t>(coordinator_.NumGroups());
    for (WriteBackStats& stats : write_back_stats_) {
        stats.blocks.resize(num_groups);
    }
    prefix_load_stats_.blocks.resize(num_groups);
    restore_load_stats_.blocks.resize(num_groups);
}

std::optional<WriteBackOperation> TierTransferManager::StartPendingStores(StoreSourceGuard guard,
                                                                          HostStoreCause cause) {
    _assert(cause == HostStoreCause::kPublication || cause == HostStoreCause::kFinish,
            "only publication and finish drain the pending-store queue");
    WriteBackStats& stats = write_back_stats_[static_cast<std::size_t>(cause)];
    std::vector<CacheKey> keys;
    std::vector<CacheBlockRef> device_block_refs;
    std::vector<std::uint32_t> group_ids;
    std::vector<std::int32_t> buckets;
    // Keys already travelling: every ticket of every in-flight write-back
    // whose ACK publishes it -- an eviction write-back's key may never be
    // published (its segment can be cancelled), so it is stored again.
    // Derived from write_backs_ on demand rather than mirrored in a second
    // container that would have to be kept in step with it. Candidates join
    // the same set so a key queued twice in one round is stored once.
    std::unordered_set<CacheKey, CacheKeyHash> storing_keys;
    for (const auto& [op_id, write_back] : write_backs_) {
        if (!write_back.publishes_at_ack) {
            continue;
        }
        for (const StoreTicket& ticket : write_back.tickets) {
            storing_keys.insert(ticket.key);
        }
    }
    for (auto& candidate : coordinator_.TakePendingStores()) {
        if (coordinator_.ContainsHostCachedBlock(candidate.key)) {
            ++stats.host_cached;
            continue;
        }
        if (!storing_keys.insert(candidate.key).second) {
            ++stats.in_flight;
            continue;
        }

        CacheBlockRef device_block_ref = coordinator_.AcquireDeviceCachedBlock(candidate.key);
        if (!device_block_ref) {
            ++stats.evicted_before_drain;
            continue;
        }

        group_ids.push_back(candidate.key.group_id);
        // The Host copy lives in its Device source's bucket (same owner).
        buckets.push_back(coordinator_.Allocator(static_cast<std::int32_t>(candidate.key.group_id))
                              .BucketOf(device_block_ref->Location()));
        keys.push_back(std::move(candidate.key));
        device_block_refs.push_back(std::move(device_block_ref));
    }

    if (keys.empty()) {
        return std::nullopt;
    }

    CacheCoordinator::HostAllocationBatch host_allocation = coordinator_.AcquireHostBlocks(group_ids, buckets);
    _assert(host_allocation.blocks.size() == keys.size(), "Host allocation result must stay aligned");
    stats.no_placement += host_allocation.stats.unallocated;

    const bool pin_source = guard == StoreSourceGuard::kPinnedUntilAck;
    std::vector<CacheTransfer> transfers;
    std::vector<StoreTicket> tickets;
    transfers.reserve(host_allocation.stats.allocated);
    tickets.reserve(host_allocation.stats.allocated);
    for (std::size_t i = 0; i < keys.size(); ++i) {
        CacheBlockRef& host_block_ref = host_allocation.blocks[i];
        if (!host_block_ref) {
            continue;
        }
        const GroupAllocator& manager = coordinator_.Allocator(static_cast<std::int32_t>(group_ids[i]));
        _assert(manager.BucketOf(device_block_refs[i]->Location()) == manager.BucketOf(host_block_ref->Location()),
                "Host store pairs blocks of different owners");
        transfers.push_back(CacheTransfer{
            .group_id = group_ids[i],
            .source_page = manager.ResolveCacheBlockId(*device_block_refs[i]),
            .destination_page = manager.ResolveCacheBlockId(*host_block_ref),
            .content_hash = keys[i].content_hash,
            .page_offset = keys[i].page_offset,
        });
        // A stream-ordered store resolves the source page id and lets the
        // reference go: the forward thread's stream orders the copy ahead of
        // any later reuse. A pinned store keeps it until the ACK.
        tickets.push_back(StoreTicket{
            std::move(keys[i]),
            pin_source ? std::move(device_block_refs[i]) : CacheBlockRef{},
            std::move(host_block_ref),
        });
        ++stats.blocks[group_ids[i]];
    }

    if (transfers.empty()) {
        return std::nullopt;
    }
    const std::uint32_t op_id = nextOpId();
    const bool inserted =
        write_backs_
            .emplace(op_id, InFlightWriteBack{.guard = guard, .tickets = std::move(tickets), .publishes_at_ack = true})
            .second;
    _assert(inserted, "duplicate store op id");
    ++stats.ops;
    return WriteBackOperation{
        .op_id = op_id,
        .transfers = std::move(transfers),
        .source_pinned = pin_source,
    };
}

bool TierTransferManager::HasPinnedStoresInFlight() const {
    return std::ranges::any_of(
        write_backs_, [](const auto& entry) { return entry.second.guard == StoreSourceGuard::kPinnedUntilAck; });
}

LoadBackOperation TierTransferManager::StartPrefixLoad(std::vector<BlockTransfer> block_transfers) {
    _assert(!block_transfers.empty(), "prefix load requires at least one block transfer");
    for (const BlockTransfer& pair : block_transfers) {
        _assert(coordinator_.IsHostCachedBlock(pair.source->Location()),
                "pinned Host block lost its cache entry before load emission");
    }
    LoadBackOperation op = startLoadBack(std::move(block_transfers));
    ++prefix_load_stats_.ops;
    for (const CacheTransfer& transfer : op.transfers) {
        ++prefix_load_stats_.blocks[transfer.group_id];
    }
    return op;
}

LoadBackOperation TierTransferManager::startLoadBack(std::vector<BlockTransfer> block_transfers) {
    std::vector<CacheTransfer> transfers = resolveTransfers(block_transfers);
    const std::uint32_t op_id = nextOpId();
    const bool inserted = load_backs_.emplace(op_id, std::move(block_transfers)).second;
    _assert(inserted, "duplicate loadback op id");
    return LoadBackOperation{op_id, std::move(transfers)};
}

std::vector<HostPublication> TierTransferManager::CompleteWriteBack(std::uint32_t op_id) {
    // The runtime emits this ACK only after the asynchronous copy completes.
    // Transfer errors terminate the runtime and must never publish cache state.
    auto it = write_backs_.find(op_id);
    if (it == write_backs_.end()) {
        return {};
    }
    // An eviction write-back's tickets publish through their segments: they
    // go with the op, and the segments hold the same Host blocks.
    std::vector<StoreTicket> stores =
        it->second.publishes_at_ack ? std::move(it->second.tickets) : std::vector<StoreTicket>{};
    write_backs_.erase(it);
    // Publishing the Host entry also drops the tickets' Device pins (if any)
    // when `stores` goes out of scope: the source is evictable again. When
    // the key already has a canonical Host entry (an L3 prefetch of it landed
    // first), Register re-points the ticket's reference to that entry and the
    // ticket's own block goes unindexed; the returned publication names the
    // canonical block so the images pinned on the ticket's block follow.
    std::vector<HostPublication> published;
    published.reserve(stores.size());
    for (StoreTicket& ticket : stores) {
        coordinator_.CacheHostBlock(ticket.host_block_ref, ticket.key, /*inherited_metadata=*/std::nullopt);
        published.push_back(HostPublication{.key = ticket.key, .block = ticket.host_block_ref});
    }
    // A segment publishes at the last ACK it waits for, in drain order,
    // less the members that became dead meanwhile.
    WriteBackStats& eviction_stats = write_back_stats_[static_cast<std::size_t>(HostStoreCause::kEviction)];
    for (auto segment = in_flight_segments_.begin(); segment != in_flight_segments_.end();) {
        std::erase(segment->pending_store_ops, op_id);
        if (!segment->pending_store_ops.empty()) {
            ++segment;
            continue;
        }
        const std::size_t num_members = segment->segment.members.size();
        std::vector<HostPublication> settled =
            coordinator_.SettleEvictedSegment(std::move(segment->segment), /*placed_on_host=*/true);
        eviction_stats.cancelled += num_members - settled.size();
        published.insert(published.end(), std::make_move_iterator(settled.begin()),
                         std::make_move_iterator(settled.end()));
        segment = in_flight_segments_.erase(segment);
    }
    // Every segment left still waits for an op that is out.
    for (const InFlightSegment& segment : in_flight_segments_) {
        _assert(!segment.pending_store_ops.empty() &&
                    std::ranges::all_of(segment.pending_store_ops,
                                        [&](std::uint32_t waited) { return write_backs_.contains(waited); }),
                "an unsettled write-back segment waits for an op still in flight");
    }
    return published;
}

std::optional<WriteBackOperation> TierTransferManager::StartEvictionStores() {
    std::vector<CacheCoordinator::EvictedSegment> segments = coordinator_.TakeEvictedSegments();
    if (segments.empty()) {
        return std::nullopt;
    }
    WriteBackStats& stats = write_back_stats_[static_cast<std::size_t>(HostStoreCause::kEviction)];
    // The newest Device access first, recording order among equals: the
    // placement drops whole segments from the back, the oldest first.
    std::ranges::stable_sort(segments, std::ranges::greater{}, &CacheCoordinator::EvictedSegment::newest_access_epoch);
    const InFlightHostBlocks in_flight = inFlightHostBlocks();
    std::vector<std::vector<std::uint32_t>> pending_store_ops(segments.size());
    // Per row to place, its member's index in its segment, with the group and
    // bucket of each: the Host copy lives in its Device source's bucket.
    std::vector<std::size_t> member_of_row;
    std::vector<std::uint32_t> group_ids;
    std::vector<std::int32_t> buckets;
    std::vector<CacheCoordinator::HostSegmentRows> segment_rows;
    // The row copying each key. A key recorded twice is copied once: the
    // segments recording it were merged (TakeEvictedSegments), and its later
    // members share that row's block.
    std::unordered_map<CacheKey, std::size_t, CacheKeyHash> row_of_key;
    for (std::size_t s = 0; s < segments.size(); ++s) {
        const std::size_t first_row = member_of_row.size();
        for (std::size_t m = 0; m < segments[s].members.size(); ++m) {
            CacheCoordinator::EvictedBlock& member = segments[s].members[m];
            if (member.host_block) {
                ++stats.host_cached;  // held since the eviction
                continue;
            }
            if (const auto it = in_flight.find(member.key); it != in_flight.end()) {
                ++stats.in_flight;
                member.host_block = it->second.block;
                if (std::ranges::find(pending_store_ops[s], it->second.op_id) == pending_store_ops[s].end()) {
                    pending_store_ops[s].push_back(it->second.op_id);
                }
                continue;
            }
            if (!row_of_key.try_emplace(member.key, member_of_row.size()).second) {
                continue;  // a key's later member, counted with its segment below
            }
            member_of_row.push_back(m);
            group_ids.push_back(member.key.group_id);
            buckets.push_back(coordinator_.Allocator(static_cast<std::int32_t>(member.key.group_id))
                                  .BucketOf(member.device_location));
        }
        segment_rows.push_back(
            CacheCoordinator::HostSegmentRows{.first_row = first_row, .past_end_row = member_of_row.size()});
    }
    CacheCoordinator::HostAllocationBatch host_allocation =
        coordinator_.AcquireHostSegments(group_ids, buckets, segment_rows);
    _assert(host_allocation.blocks.size() == member_of_row.size(), "Host allocation result must stay aligned");

    // A stream-ordered store: the Device sources left the index at the
    // eviction and may be granted again in this plan, so no ticket holds
    // them; the runtime orders the copy ahead of the plan's page reuse.
    std::optional<std::uint32_t> op_id;
    std::vector<CacheTransfer> transfers;
    std::vector<StoreTicket> tickets;
    for (std::size_t s = 0; s < segments.size(); ++s) {
        const CacheCoordinator::HostSegmentRows& row_range = segment_rows[s];
        // A segment is placed whole or not at all.
        if (row_range.first_row < row_range.past_end_row && !host_allocation.blocks[row_range.first_row]) {
            // Its rows and the later members that would have shared them.
            stats.no_placement += static_cast<std::uint64_t>(std::ranges::count_if(
                segments[s].members, [](const CacheCoordinator::EvictedBlock& member) { return !member.host_block; }));
            coordinator_.SettleEvictedSegment(std::move(segments[s]), /*placed_on_host=*/false);
            continue;
        }
        for (std::size_t row = row_range.first_row; row < row_range.past_end_row; ++row) {
            CacheCoordinator::EvictedBlock& member = segments[s].members[member_of_row[row]];
            CacheBlockRef& host_block = host_allocation.blocks[row];
            _assert(static_cast<bool>(host_block), "a segment is placed whole or not at all");
            const GroupAllocator& allocator = coordinator_.Allocator(static_cast<std::int32_t>(member.key.group_id));
            _assert(allocator.BucketOf(member.device_location) == allocator.BucketOf(host_block->Location()),
                    "Host store pairs blocks of different owners");
            if (!op_id) {
                op_id = nextOpId();
            }
            transfers.push_back(CacheTransfer{
                .group_id = member.key.group_id,
                .source_page = allocator.ResolveDeviceCacheBlockId(member.device_location),
                .destination_page = allocator.ResolveCacheBlockId(*host_block),
                .content_hash = member.key.content_hash,
                .page_offset = member.key.page_offset,
            });
            tickets.push_back(StoreTicket{member.key, CacheBlockRef{}, host_block});
            member.host_block = std::move(host_block);
            ++stats.blocks[member.key.group_id];
            if (member.written_back_before) {
                ++stats.recopied;
            }
        }
        // A key's later members share the block its row was placed in.
        for (CacheCoordinator::EvictedBlock& member : segments[s].members) {
            if (!member.host_block) {
                const std::size_t row = row_of_key.at(member.key);
                _assert(row >= row_range.first_row && row < row_range.past_end_row,
                        "the segments recording a key are merged into one");
                member.host_block = segments[s].members[member_of_row[row]].host_block;
                ++stats.in_flight;  // queued twice in this drain
            }
        }
        if (row_range.first_row < row_range.past_end_row) {
            pending_store_ops[s].push_back(*op_id);
        }
        if (pending_store_ops[s].empty()) {
            // Every member is on the Host already: nothing to wait for.
            const std::size_t num_members = segments[s].members.size();
            stats.cancelled +=
                num_members - coordinator_.SettleEvictedSegment(std::move(segments[s]), /*placed_on_host=*/true).size();
            continue;
        }
        in_flight_segments_.push_back(
            InFlightSegment{.segment = std::move(segments[s]), .pending_store_ops = std::move(pending_store_ops[s])});
    }
    if (!op_id) {
        return std::nullopt;
    }
    const bool inserted = write_backs_
                              .emplace(*op_id, InFlightWriteBack{.guard = StoreSourceGuard::kStreamOrdered,
                                                                 .tickets = std::move(tickets),
                                                                 .publishes_at_ack = false})
                              .second;
    _assert(inserted, "duplicate store op id");
    ++stats.ops;
    return WriteBackOperation{
        .op_id = *op_id,
        .transfers = std::move(transfers),
        .source_pinned = false,
    };
}

void TierTransferManager::CompleteLoadBack(std::uint32_t op_id) {
    // Both ends were pinned for the copy; the destinations were published
    // when the admission claimed the Host hit, so the ACK only lets go.
    load_backs_.erase(op_id);
}

TierTransferManager::InFlightHostBlocks TierTransferManager::inFlightHostBlocks() const {
    InFlightHostBlocks by_key;
    for (const auto& [op_id, write_back] : write_backs_) {
        for (const StoreTicket& ticket : write_back.tickets) {
            // Every drain dedupes against in-flight keys (StartPendingStores
            // skips them, the others borrow their ticket's block), so a key
            // travels on at most one ticket an ACK publishes, beside those of
            // eviction write-backs (which only an eviction drain borrows).
            const InFlightHostBlock carried{
                .block = ticket.host_block_ref, .op_id = op_id, .publishes_at_ack = write_back.publishes_at_ack};
            if (const auto [it, inserted] = by_key.emplace(ticket.key, carried);
                !inserted && carried.publishes_at_ack) {
                it->second = carried;
            }
        }
    }
    return by_key;
}

TierTransferManager::RetractionImageSplit TierTransferManager::SplitRetractionImage(
    std::span<const BlockTable> tables, std::int32_t num_tokens, std::vector<std::vector<ImageSlot>> published) const {
    const std::int32_t num_groups = coordinator_.NumGroups();
    RetractionImageSplit split{.host_slots = std::vector<std::vector<ImageSlot>>(static_cast<std::size_t>(num_groups))};
    // A published slot rides Host L2 as its prefix entry: the existing entry
    // when the key is already Host-cached, the in-flight ticket's block when
    // an earlier store carries it (the image then waits for that store too),
    // else a Host block acquired in the Device block's bucket and copied by
    // the retraction's own write-back -- one row to place.
    if (coordinator_.HasHostPool()) {
        const InFlightHostBlocks in_flight = inFlightHostBlocks();
        // The rows to place by boundary, each content hash numbered in the
        // order the image's slots first name it, the prefix-closed groups'
        // slots first so the shallowest boundary comes first whatever order
        // the groups are declared in; within one, in that group order and slot
        // order.
        struct Row {
            std::uint32_t group_id;
            std::int32_t bucket;
            ImageSlot slot;
        };
        std::unordered_map<ContentHash, std::size_t> boundary_of_hash;
        std::vector<std::vector<Row>> rows_by_boundary;
        std::vector<std::int32_t> group_order(static_cast<std::size_t>(num_groups));
        std::iota(group_order.begin(), group_order.end(), 0);
        std::ranges::stable_partition(group_order,
                                      [&](std::int32_t g) { return coordinator_.GroupMatcher(g).IsPrefixClosed(); });
        for (const std::int32_t g : group_order) {
            const GroupAllocator& allocator = coordinator_.Allocator(g);
            for (ImageSlot& slot : published[static_cast<std::size_t>(g)]) {
                const auto [boundary, numbered] =
                    boundary_of_hash.try_emplace(slot.key.content_hash, rows_by_boundary.size());
                if (numbered) {
                    rows_by_boundary.emplace_back();
                }
                if (CacheBlockRef cached = coordinator_.FindHostCachedBlock(slot.key)) {
                    slot.block = std::move(cached);
                    split.host_slots[static_cast<std::size_t>(g)].push_back(std::move(slot));
                    continue;
                }
                // Only a store whose ACK publishes the key: an eviction
                // write-back's segment may publish it later or never, after
                // the image has counted its L2 leg as landed.
                if (const auto it = in_flight.find(slot.key); it != in_flight.end() && it->second.publishes_at_ack) {
                    if (std::ranges::find(split.pending_store_ops, it->second.op_id) == split.pending_store_ops.end()) {
                        split.pending_store_ops.push_back(it->second.op_id);
                    }
                    slot.block = it->second.block;
                    split.host_slots[static_cast<std::size_t>(g)].push_back(std::move(slot));
                    continue;
                }
                const std::int32_t bucket = allocator.BucketOf(slot.block->Location());
                rows_by_boundary[boundary->second].push_back(
                    Row{.group_id = static_cast<std::uint32_t>(g), .bucket = bucket, .slot = std::move(slot)});
            }
        }
        for (std::vector<Row>& rows : rows_by_boundary) {
            if (rows.empty()) {
                continue;  // the Host serves every slot at this hash already
            }
            split.boundaries.push_back(CacheCoordinator::HostSegmentRows{
                .first_row = split.slots.size(), .past_end_row = split.slots.size() + rows.size()});
            for (Row& row : rows) {
                split.group_ids.push_back(row.group_id);
                split.buckets.push_back(row.bucket);
                split.slots.push_back(std::move(row.slot));
            }
        }
    }

    // What the Host serves with the rows of the first `num_boundaries`
    // boundaries placed, per group: the rest is the snapshot pool's.
    const auto host_served = [&](std::size_t num_boundaries) {
        std::vector<std::vector<ImageSlot>> served = split.host_slots;
        const std::size_t num_rows = num_boundaries == 0 ? 0 : split.boundaries[num_boundaries - 1].past_end_row;
        for (std::size_t row = 0; row < num_rows; ++row) {
            served[split.group_ids[row]].push_back(split.slots[row]);
        }
        return served;
    };
    if (!split.boundaries.empty()) {
        split.num_host_boundaries =
            coordinator_.NumHostBoundariesThatFit(split.group_ids, split.buckets, split.boundaries);
    }
    if (!coordinator_.SnapshotPoolHolds(tables, num_tokens, host_served(split.boundaries.size()))) {
        split.shortfall = ImageShortfall::kSnapshotPool;
    } else if (!coordinator_.SnapshotPoolHolds(tables, num_tokens, host_served(split.num_host_boundaries))) {
        split.shortfall = ImageShortfall::kHostL2;
    }
    return split;
}

std::optional<TierTransferManager::RetractionStores> TierTransferManager::StartRetractionStores(
    const std::string& request_id, std::int32_t request_pool_index, std::shared_ptr<SnapshotSlotIndex> blob_slot,
    std::span<const BlockTable> tables, std::int32_t num_tokens, RetractionImageSplit split) {
    _assert(blob_slot != nullptr && blob_slot->valid(), "a retraction image needs its blob slot");
    _assert(!split.shortfall, "a retraction takes only an image its split holds");
    WriteBackStats& stats = write_back_stats_[static_cast<std::size_t>(HostStoreCause::kRetraction)];
    // The split's Host slots: Host entries, or the blocks of stores in flight.
    for (const std::vector<ImageSlot>& slots : split.host_slots) {
        for (const ImageSlot& slot : slots) {
            if (coordinator_.ContainsHostCachedBlock(slot.key)) {
                ++stats.host_cached;
            } else {
                ++stats.in_flight;
            }
        }
    }
    std::vector<std::vector<ImageSlot>> host_slots = std::move(split.host_slots);
    std::vector<std::uint32_t> pending_store_ops = std::move(split.pending_store_ops);

    // The L2 leg: the rows of the boundaries the split counted get Host
    // blocks by its plan, copied by this retraction's own write-back.
    std::vector<StoreTicket> tickets;
    std::vector<BlockTransfer> host_store_pairs;
    if (!split.slots.empty()) {
        CacheCoordinator::HostAllocationBatch host_blocks =
            coordinator_.AcquireHostBoundaries(split.group_ids, split.buckets, split.boundaries);
        const std::size_t num_host_rows =
            split.num_host_boundaries == 0 ? 0 : split.boundaries[split.num_host_boundaries - 1].past_end_row;
        _assert(host_blocks.stats.allocated == num_host_rows,
                "the Host places the rows of the boundaries the split counted");
        for (std::size_t i = 0; i < split.slots.size(); ++i) {
            CacheBlockRef& host_block = host_blocks.blocks[i];
            if (!host_block) {
                ++stats.snapshot_fallback;
                continue;  // a boundary past the ones the Host holds: this slot rides the snapshot pool instead
            }
            ImageSlot& slot = split.slots[i];
            host_store_pairs.push_back(BlockTransfer{
                .group_id = split.group_ids[i],
                .source = slot.block,
                .destination = host_block,
                .key = slot.key,
            });
            // The ticket's Host pin is dropped at the ACK, which publishes the
            // entry; the image's own pin on the same block outlives it.
            tickets.push_back(StoreTicket{slot.key, CacheBlockRef{}, host_block});
            slot.block = std::move(host_block);
            host_slots[split.group_ids[i]].push_back(std::move(slot));
        }
        for (std::vector<ImageSlot>& slots : host_slots) {
            std::ranges::sort(slots, {}, &ImageSlot::slot_index);
        }
    }

    // The tail leg: every other data slot, in the snapshot pool.
    std::optional<CacheCoordinator::ImageTaken> taken = coordinator_.TakeImage(tables, num_tokens, host_slots);
    if (!taken) {
        ++stats.image_refused;
        return std::nullopt;  // the acquired Host blocks and tickets die here, unused (no plan reaches this)
    }

    RetractionStores stores{.image = std::move(taken->image)};
    if (!tickets.empty()) {
        const std::uint32_t op_id = nextOpId();
        std::vector<CacheTransfer> transfers = resolveTransfers(host_store_pairs);
        for (std::size_t i = 0; i < transfers.size(); ++i) {
            transfers[i].content_hash = host_store_pairs[i].key.content_hash;
            transfers[i].page_offset = host_store_pairs[i].key.page_offset;
            ++stats.blocks[transfers[i].group_id];
        }
        const bool inserted = write_backs_
                                  .emplace(op_id, InFlightWriteBack{.guard = StoreSourceGuard::kStreamOrdered,
                                                                    .tickets = std::move(tickets),
                                                                    .publishes_at_ack = true})
                                  .second;
        _assert(inserted, "duplicate store op id");
        ++stats.ops;
        stores.host_store = WriteBackOperation{
            .op_id = op_id,
            .transfers = std::move(transfers),
            .source_pinned = false,
        };
        pending_store_ops.push_back(op_id);
    }
    {
        const std::uint32_t op_id = nextOpId();
        std::vector<CacheBlockRef> destinations;
        destinations.reserve(taken->store_pairs.size());
        for (const BlockTransfer& pair : taken->store_pairs) {
            destinations.push_back(pair.destination);
        }
        stores.snapshot_store = SnapshotStoreOperation{
            .op_id = op_id,
            .request_id = request_id,
            .request_pool_index = request_pool_index,
            .snapshot_slot = blob_slot->slot_,
            .transfers = resolveTransfers(taken->store_pairs),
        };
        const bool inserted = snapshot_stores_
                                  .emplace(op_id, InFlightSnapshotStore{.request_id = request_id,
                                                                        .destinations = std::move(destinations),
                                                                        .blob_slot = std::move(blob_slot)})
                                  .second;
        _assert(inserted, "duplicate snapshot store op id");
        pending_store_ops.push_back(op_id);
    }
    stores.pending_store_ops = std::move(pending_store_ops);
    return stores;
}

SnapshotRestoreOperation TierTransferManager::StartSnapshotRestore(
    const std::string& request_id, ReqPoolIndex request_pool_index, std::shared_ptr<SnapshotSlotIndex> blob_slot,
    std::vector<BlockTransfer> load_pairs, std::vector<BlockTransfer> snapshot_pairs, std::uint64_t access_epoch) {
    _assert(request_pool_index.valid(), "a restore imports the slot-state blob into a request-pool row");
    _assert(blob_slot != nullptr && blob_slot->valid(), "a restore imports from the image's blob slot");
    for (const BlockTransfer& pair : load_pairs) {
        _assert(coordinator_.IsHostCachedBlock(pair.source->Location()),
                "a restore's Host L2 row must come from a published Host entry");
    }
    SnapshotRestoreOperation op{
        .op_id = nextOpId(),
        .request_id = request_id,
        .request_pool_index = request_pool_index.slot_,
        .snapshot_slot = blob_slot->slot_,
        .transfers = resolveTransfers(load_pairs),
        .source_tier = std::vector<HostTier>(load_pairs.size(), HostTier::kL2),
    };
    // Only the L2 rows are Host L2 traffic; the snapshot-pool rows follow.
    if (!op.transfers.empty()) {
        ++restore_load_stats_.ops;
        for (const CacheTransfer& transfer : op.transfers) {
            ++restore_load_stats_.blocks[transfer.group_id];
        }
    }
    std::vector<CacheTransfer> pool_rows = resolveTransfers(snapshot_pairs);
    op.transfers.insert(op.transfers.end(), std::make_move_iterator(pool_rows.begin()),
                        std::make_move_iterator(pool_rows.end()));
    op.source_tier.insert(op.source_tier.end(), snapshot_pairs.size(), HostTier::kSnapshotPool);
    std::vector<BlockTransfer> transfers = std::move(load_pairs);
    transfers.insert(transfers.end(), std::make_move_iterator(snapshot_pairs.begin()),
                     std::make_move_iterator(snapshot_pairs.end()));
    const bool inserted =
        snapshot_restores_
            .emplace(op.op_id, InFlightSnapshotRestore{.request_id = request_id,
                                                       .transfers = std::move(transfers),
                                                       .blob_slot = std::move(blob_slot),
                                                       .request_pool_index = std::move(request_pool_index),
                                                       .access_epoch = access_epoch})
            .second;
    _assert(inserted, "duplicate snapshot restore op id");
    return op;
}

std::optional<std::string> TierTransferManager::CompleteSnapshotStore(std::uint32_t op_id) {
    auto it = snapshot_stores_.find(op_id);
    if (it == snapshot_stores_.end()) {
        return std::nullopt;
    }
    std::string request_id = std::move(it->second.request_id);
    snapshot_stores_.erase(it);
    return request_id;
}

std::optional<std::string> TierTransferManager::SnapshotRestoreRequest(std::uint32_t op_id) const {
    const auto it = snapshot_restores_.find(op_id);
    return it == snapshot_restores_.end() ? std::nullopt : std::optional<std::string>{it->second.request_id};
}

ReqPoolIndex TierTransferManager::CompleteSnapshotRestore(std::uint32_t op_id, bool publish) {
    auto it = snapshot_restores_.find(op_id);
    if (it == snapshot_restores_.end()) {
        return ReqPoolIndex{};
    }
    // The L2-tier rows are the request's own prefix pages, copied back whole:
    // publish them as they were published before the retraction.
    if (publish) {
        for (BlockTransfer& transfer : it->second.transfers) {
            if (transfer.destination && !transfer.key.content_hash.empty()) {
                coordinator_.CacheDeviceBlock(transfer.destination, transfer.key, transfer.logical_block_index,
                                              transfer.boundary_kind, it->second.access_epoch);
            }
        }
    }
    // The row the blob was imported into goes back to the request (or, for a
    // request that is gone, to the allocator); the blob slot's last owner
    // may be this ticket.
    ReqPoolIndex request_pool_index = std::move(it->second.request_pool_index);
    snapshot_restores_.erase(it);
    return request_pool_index;
}

std::vector<CacheTransfer> TierTransferManager::resolveTransfers(std::span<const BlockTransfer> block_transfers) const {
    std::vector<CacheTransfer> transfers;
    transfers.reserve(block_transfers.size());
    for (const BlockTransfer& block_transfer : block_transfers) {
        _assert(block_transfer.source && block_transfer.destination,
                "cache transfer requires pinned source and destination blocks");
        const GroupAllocator& manager = coordinator_.Allocator(static_cast<std::int32_t>(block_transfer.group_id));
        // Every tier transfer pairs blocks of equal residue: under page-cyclic
        // sharding the rank that owns one end owns the other.
        _assert(manager.BucketOf(block_transfer.source->Location()) ==
                    manager.BucketOf(block_transfer.destination->Location()),
                "cache transfer pairs blocks of different owners");
        transfers.push_back(CacheTransfer{
            .group_id = block_transfer.group_id,
            .source_page = manager.ResolveCacheBlockId(*block_transfer.source),
            .destination_page = manager.ResolveCacheBlockId(*block_transfer.destination),
            .content_hash = block_transfer.key.content_hash,
            .page_offset = block_transfer.key.page_offset,
        });
    }
    return transfers;
}

}  // namespace tokenspeed
