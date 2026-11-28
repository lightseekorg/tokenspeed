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

#include "cache/tier/disk_manager.h"
#include "cache/coordinator/gds_tiered.h"
#include "cache/coordinator/clean_host_copies.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

#include <spdlog/spdlog.h>

#include "utils.h"

namespace tokenspeed {

void DiskConfig::Validate() const {
    if (max_operations <= 0 || max_blocks_per_operation <= 0 || max_host_blocks <= 0 || max_write_host_blocks <= 0 ||
        max_queued_cleanings <= 0 || min_prefetch_tokens <= 0 || wait_timeout_us <= 0 || ready_timeout_us <= 0) {
        throw std::invalid_argument("Disk tier limits and deadlines must be positive");
    }
    if (max_write_host_blocks > max_host_blocks) {
        throw std::invalid_argument("Disk tier write credits must not exceed its Host credits");
    }
    if (min_clean_tokens < 0) {
        throw std::invalid_argument("Disk tier min_clean_tokens must be non-negative");
    }
}

DiskTransferManager::DiskTransferManager(CacheCoordinator& coordinator, DiskConfig config)
    : coordinator_{coordinator}, config_{config} {
    config_.Validate();
    FatalCheck(coordinator_.Disk().HasPool(), "the Disk tier's transfers need a Disk pool");
}

std::uint64_t DiskTransferManager::nextId() {
    if (next_id_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("Disk operation ids exhausted");
    }
    return next_id_++;
}

bool DiskTransferManager::HasProgressPending() const {
    return std::ranges::any_of(operations_, [](const auto& entry) { return !entry.second.completed; });
}

std::size_t DiskTransferManager::NumHeldHostBlocks() const {
    return heldHostBlocks(DiskOpKind::kRead) + heldHostBlocks(DiskOpKind::kWrite);
}

std::size_t DiskTransferManager::numOperations(DiskOpKind kind) const {
    std::size_t count = 0;
    for (const auto& [id, ticket] : operations_) {
        if (ticket.kind == kind) ++count;
    }
    return count;
}

bool DiskTransferManager::HasReadCredits() const {
    return operations_.size() < static_cast<std::size_t>(config_.max_operations) &&
           NumHeldHostBlocks() < static_cast<std::size_t>(config_.max_host_blocks);
}

std::size_t DiskTransferManager::heldHostBlocks(DiskOpKind kind) const {
    std::size_t count = 0;
    for (const auto& [id, ticket] : operations_) {
        if (ticket.kind == kind) count += ticket.host_blocks.size() + ticket.cleaning_sources.size();
    }
    return count;
}

std::vector<const CacheBlockRef*> DiskTransferManager::HeldHostBlockRefs() const {
    std::vector<const CacheBlockRef*> refs;
    for (const auto& [id, ticket] : operations_) {
        for (const auto& source : ticket.cleaning_sources) {
            if (source) refs.push_back(&source.Block());
        }
        for (const CacheBlockRef& block : ticket.host_blocks) {
            if (block) refs.push_back(&block);
        }
    }
    return refs;
}

bool DiskTransferManager::Completed(std::uint64_t id) const {
    const auto it = operations_.find(id);
    return it == operations_.end() || it->second.completed;
}

DiskOperation DiskTransferManager::command(std::uint64_t id, const Ticket& ticket) const {
    DiskOperation op{.kind = ticket.kind, .op_id = id};
    for (std::size_t i = 0; i < ticket.keys.size(); ++i) {
        const auto group = static_cast<std::int32_t>(ticket.keys[i].group_id);
        const GroupAllocator& allocator = coordinator_.Allocator(group);
        const auto& host =
            ticket.kind == DiskOpKind::kWrite ? ticket.cleaning_sources[i].Block() : ticket.host_blocks[i];
        _assert(allocator.BucketOf(host->Location()) == allocator.BucketOf(ticket.disk_blocks[i]->Location()),
                "a Disk operation pairs blocks of different owners");
        op.group_ids.push_back(ticket.keys[i].group_id);
        op.host_blocks.push_back(allocator.ResolveCacheBlockId(*host));
        op.disk_blocks.push_back(allocator.ResolveCacheBlockId(*ticket.disk_blocks[i]));
    }
    return op;
}

DiskTransferManager::ReadStartResult DiskTransferManager::TryStartRead(const PromptPrefix& prompt,
                                                                       std::int32_t local_tokens) {
    const std::vector<CacheKey> hits = coordinator_.Disk().PrefixHits(prompt);
    if (hits.empty()) return {.retryable = false};
    // Only report `retryable = true` when Disk actually extends the resumable
    // prefix beyond `local_tokens`; when the Disk keys are already covered on
    // Device/Host or lack a required State checkpoint, no retry can gain tokens.
    if (coordinator_.Storage().ResumableStorageTokens(prompt, hits) <= local_tokens) {
        return {.retryable = false};
    }
    if (!HasReadCredits()) return {.retryable = true};
    const std::size_t held = NumHeldHostBlocks();
    const auto limit = static_cast<std::size_t>(config_.max_host_blocks);
    // The Disk's keys stand in as Host hits for the Host-suffix probe; every
    // key the boundary needs and the Host lacks is then one the Disk holds.
    const std::size_t available_blocks = limit - held;
    const auto max_op_blocks = static_cast<std::size_t>(config_.max_blocks_per_operation);
    StorageReservation reservation =
        coordinator_.Storage().ReserveDiskPrefix(prompt, hits, std::min(available_blocks, max_op_blocks));
    if (reservation.keys.empty() || reservation.num_tokens <= local_tokens) {
        // If Host-block credits were partially consumed by other in-flight
        // operations, let the caller retry once credits return.
        return {.retryable = available_blocks < max_op_blocks && held > 0};
    }
    Ticket ticket{.kind = DiskOpKind::kRead,
                  .keys = std::move(reservation.keys),
                  .host_blocks = std::move(reservation.blocks),
                  .disk_blocks = std::move(reservation.sources),
                  .tail_prior = std::move(reservation.tail_prior),
                  .tail = std::move(reservation.tail),
                  .recovery_metadata = coordinator_.TrackRecoveryMetadata(prompt, reservation.num_tokens)};
    const std::uint64_t id = nextId();
    commands_.push_back(command(id, ticket));
    operations_.emplace(id, std::move(ticket));
    return {.lease = StorageLease{*this, id}, .retryable = false};
}

StorageLease DiskTransferManager::StartRead(const PromptPrefix& prompt) {
    const PrefixProbe local_probe = coordinator_.Prober().ProbePromptPrefix(prompt);
    const std::int32_t local_tokens =
        std::max(local_probe.device.num_common_tokens, local_probe.host.num_common_tokens);
    return TryStartRead(prompt, local_tokens).lease;
}

void DiskTransferManager::QueueCleaning(std::span<const CacheKey> keys) {
    // GDS discovers dirty entries from residency, including Disk removals;
    // ACK notifications neither schedule its writes nor consume queue capacity.
    if (coordinator_.Store().UsesGds()) return;
    for (const CacheKey& key : keys) {
        if (cleaning_.size() == static_cast<std::size_t>(config_.max_queued_cleanings)) {
            cleaning_.pop_front();
            ++dropped_cleanings_;
        }
        cleaning_.push_back(key);
    }
}

void DiskTransferManager::DropCleaning() {
    dropped_cleanings_ += cleaning_.size() + awaiting_reuse_.size();
    cleaning_.clear();
    awaiting_reuse_.clear();
}

void DiskTransferManager::requeueReused() {
    // Each plan looks at a bounded slice, oldest first, and rotates what still waits.
    constexpr std::size_t kMaxChecked = 64;
    CacheStore& store = coordinator_.Store();
    const BlockPool& host_pool = *store.HostPool();
    for (std::size_t checked = std::min(awaiting_reuse_.size(), kMaxChecked); checked > 0; --checked) {
        CacheKey key = std::move(awaiting_reuse_.front());
        awaiting_reuse_.pop_front();
        const auto loc = store.LocationOf(host_pool, key.group_id, key);
        std::optional<PrefixCacheIndex::CachedBlockMetadata> meta;
        if (loc) meta = store.Groups()[key.group_id].Index().MetadataFor(host_pool, *loc);
        if (!meta) {
            // The Host dropped it while it waited.
            ++dropped_cleanings_;
        } else if (coordinator_.Disk().ContainsDiskCachedBlock(key)) {
            // A descendant's cleaning already wrote it to Disk.
        } else if (meta->eviction.was_acquired) {
            QueueCleaning(std::span{&key, 1});
        } else {
            awaiting_reuse_.push_back(std::move(key));
        }
    }
}

void DiskTransferManager::StartCleaning() {
    CacheStore& store = coordinator_.Store();
    const auto max_ops = static_cast<std::size_t>(config_.max_operations);
    const std::size_t max_write_ops = std::max<std::size_t>(1, max_ops / 4);
    const std::size_t read_reserve = max_ops > 1 ? 1 : 0;
    requeueReused();
    if ((!store.UsesGds() && cleaning_.empty()) || operations_.size() + read_reserve >= max_ops ||
        numOperations(DiskOpKind::kWrite) >= max_write_ops) {
        return;
    }
    const std::size_t held = NumHeldHostBlocks();
    const std::size_t write_held = heldHostBlocks(DiskOpKind::kWrite);
    const auto limit = static_cast<std::size_t>(config_.max_host_blocks);
    const auto write_limit = static_cast<std::size_t>(config_.max_write_host_blocks);
    if (held >= limit || write_held >= write_limit) return;
    const std::size_t credits =
        std::min({limit - held, write_limit - write_held, static_cast<std::size_t>(config_.max_blocks_per_operation)});

    const BlockPool& host_pool = *store.HostPool();
    std::unordered_set<CacheKey, CacheKeyHash> writing;
    for (const auto& [id, ticket] : operations_) {
        if (ticket.kind == DiskOpKind::kWrite) writing.insert(ticket.keys.begin(), ticket.keys.end());
    }
    const auto dirty = [&](const CacheKey& key) {
        return coordinator_.Host().ContainsHostCachedBlock(key) && !coordinator_.Disk().ContainsDiskCachedBlock(key) &&
               !writing.contains(key) && mayClean(key);
    };
    const auto skip_cleaning = [&](const CacheKey& key) {
        const auto loc = store.LocationOf(host_pool, key.group_id, key);
        if (!loc) return true;
        const auto meta = store.Groups()[key.group_id].Index().MetadataFor(host_pool, *loc);
        return !meta || coordinator_.Evictor().IsDemotedEntry(*meta) ||
               meta->eviction.last_access_epoch == CacheStore::kSupersededAccessEpoch;
    };
    // The dirty Host entries of a boundary node, every group's. Skip demoted or
    // superseded State checkpoints, while keeping ancestor KV entries needed by
    // non-demoted descendants so no prefix gap opens.
    const auto dirty_keys_of = [&](const BoundaryTree::Node& node) {
        std::vector<CacheKey> keys;
        for (const CacheStore::NodeEntry& entry : store.NodeEntries(host_pool, node)) {
            std::optional<CacheKey> key = store.Groups()[entry.group_id].Index().KeyOf(host_pool, entry.location);
            if (!key || !dirty(*key)) continue;
            if (store.GroupIsState(entry.group_id) && skip_cleaning(*key)) continue;
            keys.push_back(std::move(*key));
        }
        return keys;
    };

    // Admission filter applied to the branch tip (`cleaning_.front()`): skips
    // short branches and single-hit cold tails without breaking ancestor chains
    // of qualifying long/reused branches.
    enum class Admission { kClean, kDrop, kAwaitReuse };
    const auto admission_of = [&](const CacheKey& key, const BoundaryTree::Node& node, const BlockPool& pool) {
        if (config_.min_clean_tokens == 0 && !config_.require_reuse_for_cleaning) return Admission::kClean;
        const auto loc = store.LocationOf(pool, key.group_id, key);
        if (!loc) return Admission::kDrop;
        const auto meta = store.Groups()[key.group_id].Index().MetadataFor(pool, *loc);
        if (!meta) return Admission::kDrop;
        if (config_.min_clean_tokens > 0) {
            std::int64_t branch_tokens = node.boundary_tokens;
            if (branch_tokens <= 0 && meta->eviction.logical_block_index >= 0) {
                const std::int32_t group_granularity = store.Groups()[key.group_id].Spec().block_granularity;
                const std::int64_t span = group_granularity > 0 ? group_granularity : store.PrefixGranularity();
                branch_tokens = (static_cast<std::int64_t>(meta->eviction.logical_block_index) + 1) * span;
            }
            if (branch_tokens > 0 && branch_tokens < config_.min_clean_tokens) return Admission::kDrop;
        }
        // A branch not reused yet may be reused later: it waits instead of being lost.
        if (config_.require_reuse_for_cleaning && !meta->eviction.was_acquired) return Admission::kAwaitReuse;
        return Admission::kClean;
    };

    Ticket ticket{.kind = DiskOpKind::kWrite};
    std::unordered_set<CacheKey, CacheKeyHash> chosen;
    if (store.UsesGds()) {
        // Maintain a clean reserve bounded by the configured write credits.
        // Count parent-equivalent capacity, not raw bytes or Disk capacity.
        double safe_capacity = host_pool.NumEmptyLcmBlocks();
        for (const auto& group : store.Groups())
            safe_capacity += static_cast<double>(host_pool.NumFreeSlots(group.Id())) / host_pool.Packing(group.Id());
        for (const auto& copy : DiskCleanHostCopies(store))
            for (const auto& [group, location] : copy.members) safe_capacity += 1.0 / host_pool.Packing(group);
        const auto target = std::min(host_pool.NumLcmBlocks(), config_.max_write_host_blocks);
        if (safe_capacity >= target) return;
        std::unordered_set<CacheKey, CacheKeyHash> dependencies;
        VisitGdsHostCleaningCandidates(store, [&](HostCleaningCandidate candidate) {
            // Gate only the recovery point, not its cold ancestor dependencies.
            // A skipped GDS candidate is rediscovered on the next cleaning pass.
            if (config_.min_clean_tokens > 0 || config_.require_reuse_for_cleaning) {
                const auto* node = store.Boundaries().Find(candidate.hash);
                if (node == nullptr) return true;
                const auto qualifies = [&](const CacheKey& key, const BlockPool& pool) {
                    return key.content_hash == candidate.hash && admission_of(key, *node, pool) == Admission::kClean;
                };
                const bool host_tip =
                    std::ranges::any_of(candidate.keys, [&](const CacheKey& key) { return qualifies(key, host_pool); });
                // A published Disk tip already passed cleaning admission.
                // Its new tier-local entry does not inherit was_acquired; it
                // must still allow lost ancestor copies to be cleaned again.
                const bool disk_tip = node->boundary_tokens >= config_.min_clean_tokens &&
                                      std::ranges::any_of(candidate.disk_dependencies, [&](const CacheKey& key) {
                                          return key.content_hash == candidate.hash;
                                      });
                if (!host_tip && !disk_tip) return true;
            }
            if (std::ranges::any_of(candidate.keys, [&](const CacheKey& key) { return !mayClean(key); })) return true;
            // A queued write is not a published prerequisite: its ACK may
            // fail. Retry this closure after the existing write retires.
            if (std::ranges::any_of(candidate.keys, [&](const CacheKey& key) { return writing.contains(key); }))
                return true;
            std::ranges::sort(candidate.keys, [&](const CacheKey& a, const CacheKey& b) {
                const auto* an = store.Boundaries().Find(a.content_hash);
                const auto* bn = store.Boundaries().Find(b.content_hash);
                return std::tuple{an->boundary_tokens, store.GroupIsState(a.group_id), a.group_id, a.page_offset} <
                       std::tuple{bn->boundary_tokens, store.GroupIsState(b.group_id), b.group_id, b.page_offset};
            });
            bool added = false;
            for (const auto& key : candidate.keys) {
                if (!dirty(key) || !chosen.insert(key).second) continue;
                ticket.keys.push_back(key);
                added = true;
                if (ticket.keys.size() == credits) break;
            }
            if (added) {
                // A later Disk allocation must not evict a prerequisite of
                // the very closure being cleaned.
                for (const auto& key : candidate.disk_dependencies) dependencies.insert(key);
                for (const auto& key : candidate.keys)
                    if (coordinator_.Disk().ContainsDiskCachedBlock(key)) dependencies.insert(key);
            }
            return ticket.keys.size() < credits;
        });
        for (const auto& key : dependencies)
            ticket.disk_dependencies.push_back(coordinator_.Disk().AcquireDiskCachedBlock(key));
    } else {
        while (!cleaning_.empty() && ticket.keys.size() < credits) {
            const CacheKey front = cleaning_.front();
            const BoundaryTree::Node* node = store.Boundaries().Find(front.content_hash);
            if (node == nullptr || !dirty(front) || (store.GroupIsState(front.group_id) && skip_cleaning(front)) ||
                chosen.contains(front)) {
                cleaning_.pop_front();
                continue;
            }
            if (const Admission admission = admission_of(front, *node, host_pool); admission != Admission::kClean) {
                cleaning_.pop_front();
                if (admission == Admission::kAwaitReuse) {
                    if (awaiting_reuse_.size() == static_cast<std::size_t>(config_.max_queued_cleanings)) {
                        awaiting_reuse_.pop_front();
                        ++dropped_cleanings_;
                    }
                    awaiting_reuse_.push_back(front);
                } else {
                    ++dropped_cleanings_;
                }
                continue;
            }
            // The pages above it first, nearest the root first, so a cleaned page
            // never waits on the Disk for a parent only the Host holds: a Host
            // placement dropping that parent would leave it unresumable. The
            // climb stops at a page on the Disk that has no remaining dirty Host
            // keys (e.g., partial-OnDisk boundaries where History was cleaned
            // earlier while SWA/State keys are still dirty on Host), or a page
            // the Host does not hold.
            std::vector<const BoundaryTree::Node*> chain{node};
            for (const BoundaryTree::Node* above = node->parent; above != nullptr && above->OnHost();
                 above = above->parent) {
                if (above->OnDisk() && dirty_keys_of(*above).empty()) {
                    break;
                }
                chain.push_back(above);
            }
            std::vector<CacheKey> batch;
            for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
                for (CacheKey& key : dirty_keys_of(**it)) {
                    if (!chosen.contains(key)) batch.push_back(std::move(key));
                }
            }
            if (!ticket.keys.empty() && ticket.keys.size() + batch.size() > credits) break;
            cleaning_.pop_front();
            if (batch.size() > credits) {
                // A chain longer than one operation: its root-most part now, the
                // rest when the key comes up again.
                batch.resize(credits);
                cleaning_.push_front(front);
            }
            for (CacheKey& key : batch) {
                chosen.insert(key);
                ticket.keys.push_back(std::move(key));
            }
        }
    }
    if (ticket.keys.empty()) return;

    std::vector<std::uint32_t> groups;
    std::vector<std::int32_t> buckets;
    for (const CacheKey& key : ticket.keys) {
        auto source = store.Groups()[key.group_id].Index().AcquireCleaningSource(host_pool, key);
        FatalCheck(static_cast<bool>(source), "a dirty Host entry has a block");
        const auto group = static_cast<std::int32_t>(key.group_id);
        groups.push_back(key.group_id);
        // The Disk copy lives in its Host source's bucket (same owner).
        buckets.push_back(coordinator_.Allocator(group).BucketOf(source.Block()->Location()));
        ticket.metadata.push_back(
            *store.Groups()[key.group_id].Index().MetadataFor(host_pool, source.Block()->Location()));
        ticket.cleaning_sources.push_back(std::move(source));
    }
    DiskCache::DiskAllocationBatch allocation = coordinator_.Disk().AcquireDiskBlocks(groups, buckets);
    // Keep only the dependency-ordered prefix. A later row may fit another
    // group even when an earlier history row failed to place.
    std::size_t kept = 0;
    for (std::size_t row = 0; row < ticket.keys.size(); ++row) {
        if (!allocation.blocks[row]) {
            dropped_cleanings_ += ticket.keys.size() - row;
            break;
        }
        if (kept != row) {
            ticket.keys[kept] = std::move(ticket.keys[row]);
            ticket.cleaning_sources[kept] = std::move(ticket.cleaning_sources[row]);
            ticket.metadata[kept] = std::move(ticket.metadata[row]);
        }
        ticket.disk_blocks.push_back(std::move(allocation.blocks[row]));
        ++kept;
    }
    ticket.keys.resize(kept);
    ticket.cleaning_sources.resize(kept);
    ticket.metadata.resize(kept);
    if (kept == 0) return;
    const std::uint64_t id = nextId();
    commands_.push_back(command(id, ticket));
    operations_.emplace(id, std::move(ticket));
}

void DiskTransferManager::Complete(const cache::DiskDone& event) {
    const auto it = operations_.find(event.op_id);
    if (it == operations_.end() || it->second.completed) {
        throw std::invalid_argument("a Disk completion names no pending operation");
    }
    Ticket& ticket = it->second;
    ticket.completed = true;
    if (ticket.kind == DiskOpKind::kRead) {
        if (event.success) {
            read_blocks_ += ticket.keys.size();
            if (!ticket.detached) {
                for (std::size_t i = 0; i < ticket.keys.size(); ++i) {
                    coordinator_.Host().CacheHostBlock(ticket.host_blocks[i], ticket.keys[i],
                                                       /*inherited_metadata=*/std::nullopt);
                }
                if (ticket.tail) {
                    coordinator_.Host().CacheHostTail(ticket.tail_prior, *ticket.tail);
                }
            }
        } else {
            ++failed_reads_;
            spdlog::warn("[Disk] read op_id={} of {} blocks failed; dropping their Disk entries", event.op_id,
                         ticket.keys.size());
            for (std::size_t i = 0; i < ticket.keys.size(); ++i)
                coordinator_.Disk().DropUnreadableEntry(ticket.keys[i], ticket.disk_blocks[i]);
            ticket.host_blocks.clear();
        }
        ticket.disk_blocks.clear();
        // A request still holds the published Host entries until its
        // admission acquires them (the Ready hold); one that gave up let go.
        if (ticket.detached) operations_.erase(it);
        return;
    }
    if (event.success) {
        written_blocks_ += ticket.keys.size();
        const bool valid = std::ranges::all_of(ticket.keys, [&](const CacheKey& key) { return mayClean(key); });
        if (valid) {
            for (std::size_t i = 0; i < ticket.keys.size(); ++i) {
                const auto& key = ticket.keys[i];
                const auto& index = coordinator_.Store().Groups()[key.group_id].Index();
                const auto* host = coordinator_.Store().HostPool();
                if (index.Find(*host, key) != ticket.cleaning_sources[i].Block()) continue;
                // Content is immutable while owned. Inherit a newer use,
                // rather than resurrecting an older demotion at a late ACK.
                const auto metadata = index.MetadataFor(*host, ticket.cleaning_sources[i].Block()->Location());
                coordinator_.Disk().CacheDiskBlock(ticket.disk_blocks[i], key, *metadata);
            }
        } else {
            dropped_cleanings_ += ticket.keys.size();
        }
    } else {
        ++failed_writes_;
        spdlog::warn("[Disk] write op_id={} of {} blocks failed", event.op_id, ticket.keys.size());
    }
    operations_.erase(it);
}

void DiskTransferManager::detach(std::uint64_t id) noexcept {
    const auto it = operations_.find(id);
    FatalCheck(it != operations_.end() && !it->second.detached, "a Disk lease names an attached operation");
    Ticket& ticket = it->second;
    ticket.detached = true;
    if (ticket.completed) {
        operations_.erase(it);
        return;
    }
    // Nothing has seen a command still in this scheduler-owned outbox. One
    // already issued keeps its blocks until it retires.
    const auto queued = std::ranges::find_if(commands_, [id](const DiskOperation& op) { return op.op_id == id; });
    if (queued != commands_.end()) {
        commands_.erase(queued);
        operations_.erase(it);
    }
}

std::vector<DiskOperation> DiskTransferManager::TakeCommands() {
    std::vector<DiskOperation> result;
    for (auto& op : commands_) {
        auto& ticket = operations_.at(op.op_id);
        if (ticket.kind == DiskOpKind::kWrite) {
            filterCleaning(ticket);
            if (ticket.keys.empty()) {
                operations_.erase(op.op_id);
                continue;
            }
        }
        result.push_back(std::move(op));
    }
    commands_.clear();
    return result;
}

bool DiskTransferManager::mayClean(const CacheKey& key) const {
    const auto& store = coordinator_.Store();
    const auto entry = store.Groups()[key.group_id].Index().Inspect(*store.HostPool(), key);
    return entry && !entry->metadata.demoted && !(entry->metadata.session && entry->metadata.session->demoted);
}

void DiskTransferManager::filterCleaning(Ticket& ticket) {
    if (std::ranges::all_of(ticket.keys, [&](const CacheKey& key) { return mayClean(key); })) return;
    // An unissued operation has no external writers. Once issued, Complete
    // suppresses publication but the owning ticket stays until retirement.
    dropped_cleanings_ += ticket.keys.size();
    ticket.keys.clear();
    ticket.cleaning_sources.clear();
    ticket.disk_blocks.clear();
    ticket.disk_dependencies.clear();
    ticket.metadata.clear();
}

}  // namespace tokenspeed
