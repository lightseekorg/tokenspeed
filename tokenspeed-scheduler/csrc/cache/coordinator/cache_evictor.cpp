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

#include "cache/coordinator/cache_evictor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <numeric>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "utils.h"

namespace tokenspeed {

namespace {

// Software prefetch for evicting a plan's victims in plan order
// (CacheEvictor::EvictPlanned). Before each victim leaves,
// Advance() issues each stage of PrefixCacheIndex::PrefetchEvictions for the
// victim that many places ahead: the location index slot, with the LCM
// parent record, 2 * kDistance ahead; the entry kDistance ahead; and what
// removing the entry touches beyond it kDistance / 2 ahead, so each stage
// finds the lines it reads already on their way. It keeps positions in the
// plan, never an entry: a victim a cascade took first is skipped by the
// index. A cascade's own victims are not prefetched.
class VictimPrefetch {
public:
    VictimPrefetch(std::span<const VictimSet> victims, const std::vector<CacheGroup>& groups, const BlockPool& pool)
        : groups_{groups},
          pool_{pool},
          slot_{victims, 2 * kDistance},
          entry_{victims, kDistance},
          removal_{victims, kDistance / 2} {}

    // Once before each victim leaves, in plan order. The stages' victims of
    // one group share one call, which finds the group's Device entries once.
    void Advance() noexcept {
        const VictimSet::Member* slot = slot_.Next();
        const VictimSet::Member* entry = entry_.Next();
        const VictimSet::Member* removal = removal_.Next();
        if (slot != nullptr) pool_.PrefetchRelease(slot->location);
        while (const VictimSet::Member* first = slot != nullptr ? slot : entry != nullptr ? entry : removal) {
            const std::uint32_t group_id = first->group_id;
            // A stage's victim when it is of this group, which then leaves the
            // stage.
            const auto take = [group_id](const VictimSet::Member*& member) -> const CacheBlockLocation* {
                if (member == nullptr || member->group_id != group_id) return nullptr;
                return &std::exchange(member, nullptr)->location;
            };
            groups_[group_id].Index().PrefetchEvictions(pool_, take(slot), take(entry), take(removal));
        }
    }

private:
    // In victims. A full-cache benchmark ran equally fast from 2 to 16: a
    // victim or two ahead already covers a memory miss.
    static constexpr std::size_t kDistance = 4;

    // A position in the victims of all sets, in plan order.
    class Cursor {
    public:
        Cursor(std::span<const VictimSet> victims, std::size_t ahead) : victims_{victims} {
            settle();
            for (std::size_t i = 0; i < ahead && set_ < victims_.size(); ++i) step();
        }
        // The victim here, then moves past it; null past the last.
        const VictimSet::Member* Next() noexcept {
            if (set_ >= victims_.size()) return nullptr;
            const VictimSet::Member* member = &victims_[set_].members[member_];
            step();
            return member;
        }

    private:
        void step() noexcept {
            ++member_;
            settle();
        }
        void settle() noexcept {
            while (set_ < victims_.size() && member_ >= victims_[set_].members.size()) {
                ++set_;
                member_ = 0;
            }
        }

        std::span<const VictimSet> victims_;
        std::size_t set_{0};
        std::size_t member_{0};
    };

    const std::vector<CacheGroup>& groups_;
    const BlockPool& pool_;
    Cursor slot_;
    Cursor entry_;
    Cursor removal_;
};

}  // namespace

CacheEvictor::PendingHostStore::PendingHostStore(CacheEvictor& evictor, BoundaryTree::Node& node, const CacheKey& key,
                                                 bool checkpoint)
    : evictor_{&evictor} {
    const auto [key_it, inserted] = evictor.pending_eviction_keys_.insert(key);
    if (!inserted) {
        ++evictor.pending_eviction_extra_refs_[key];
    }
    node_ = evictor.store_.Boundaries().HoldPendingStore(node, checkpoint);
    key_ = &*key_it;
}

void CacheEvictor::releasePendingKey(const CacheKey& key) noexcept {
    // `key` may be the set's own element: find before erasing.
    const auto extra = pending_eviction_extra_refs_.find(key);
    if (extra == pending_eviction_extra_refs_.end()) {
        pending_eviction_keys_.erase(pending_eviction_keys_.find(key));
    } else if (--extra->second == 0) {
        pending_eviction_extra_refs_.erase(extra);
    }
}

CacheEvictor::PendingHostStore::PendingHostStore(PendingHostStore&& other) noexcept
    : evictor_{std::exchange(other.evictor_, nullptr)},
      node_{std::move(other.node_)},
      key_{std::exchange(other.key_, nullptr)} {}

CacheEvictor::PendingHostStore& CacheEvictor::PendingHostStore::operator=(PendingHostStore&& other) noexcept {
    if (this != &other) {
        reset();
        evictor_ = std::exchange(other.evictor_, nullptr);
        node_ = std::move(other.node_);
        key_ = std::exchange(other.key_, nullptr);
    }
    return *this;
}

void CacheEvictor::PendingHostStore::reset() noexcept {
    if (evictor_ != nullptr) {
        evictor_->releasePendingKey(*key_);
        evictor_ = nullptr;
        key_ = nullptr;
    }
    node_.reset();
}

std::optional<CacheEvictor::EvictedEntry> CacheEvictor::EvictCachedBlockAlone(
    std::uint32_t group_id, CacheBlockLocation location, DeviceEvictionCounts DeviceEvictionStats::*reason,
    EvictedSegment* write_back_segment) {
    PrefixCacheIndex& index = store_.Groups()[group_id].Index();
    PrefixCacheIndex::EvictableEntry entry = index.FindEvictable(store_.DevicePool(), location);
    if (!entry) {
        return std::nullopt;
    }
    // The entry's own key and tree node: read them only before Evict below.
    const CacheKey& key = entry.key();
    BoundaryTree::Node* const node = entry.boundary_node();
    _assert(node != nullptr, "an indexed entry has a tree node");
    const PrefixCacheIndex::CachedBlockMetadata metadata = entry.metadata();
    // The node still counts this entry among its Device entries.
    const bool device_companions_left = node->Entries(BoundaryTree::kDevice) > 1;
    // A demoted entry never rides an eviction write-back, whatever evicts it
    // -- an admission's or a restore's victim, a request's own released slot
    // or a stranded suffix: it is dropped, counted under what evicted it. A
    // demoted session's entry or a superseded input shares its epoch without
    // being demoted: the rule below takes it.
    const bool demoted = IsDemotedEntry(metadata);
    // A superseded input an admission, rotation, restore or compaction evicts leaves as the
    // reclaim of the request that superseded it, under either write policy,
    // and is never written back. It is never a stranded suffix: a tail at
    // epoch 0 is a candidate of its own before any page above it.
    const bool superseded =
        !demoted && metadata.eviction.last_access_epoch == CacheStore::kSupersededAccessEpoch &&
        (reason == &DeviceEvictionStats::admission || reason == &DeviceEvictionStats::restore ||
         reason == &DeviceEvictionStats::compaction || reason == &DeviceEvictionStats::request_reclaim);
    DeviceEvictionCounts& evicted = device_evictions_.*(superseded ? &DeviceEvictionStats::request_reclaim : reason);
    if (demoted && write_back_segment != nullptr) {
        ++demoted_entries_not_written_back_;
        // A request's own released slot the drain would have considered is
        // dropped like a dead one, and counted with them, so the drain's
        // outcomes still add up to the evictions it considered.
        if (reason == &DeviceEvictionStats::request_reclaim) {
            ++device_evictions_.request_reclaim_for_write_back;
        }
    }
    if (write_back_segment != nullptr && !demoted && !superseded) {
        if (reason == &DeviceEvictionStats::request_reclaim) {
            ++device_evictions_.request_reclaim_for_write_back;
        }
        // Judged and recorded while the entry still holds the node, so the
        // checkpoints of a checkpoint boundary are recorded and hold it on the
        // Host before the pages above it are judged: no probe could resume
        // from a tail whose hint the cap dropped, nor, with State groups, from
        // a boundary with no checkpoint boundary at or under it on either tier
        // and none on its way there.
        const bool dead =
            (node->tail_tokens > 0 && !store_.Boundaries().Tails().Contains(key.content_hash)) ||
            (store_.NumStateGroups() > 0 && !node->HasCheckpointBoundaryInSubtree() && node->PendingHostStores() == 0);
        if (dead) {
            ++dead_entries_not_written_back_;
        } else {
            write_back_segment->members.push_back(
                EvictedBlock{.key = key,
                             .device_location = location,
                             .metadata = metadata,
                             .host_block = store_.AcquireHostCachedBlock(key),
                             .written_back_before = (node->written_back_group_bits >> key.group_id & 1U) != 0,
                             .pending = PendingHostStore{*this, *node, key, store_.GroupIsState(group_id)}});
            write_back_segment->newest_access_epoch =
                std::max(write_back_segment->newest_access_epoch, metadata.eviction.last_access_epoch);
        }
    }
    CacheKey removed = index.Evict(std::move(entry));
    if (store_.MutationSink()) {
        store_.MutationSink()(removed, CacheMutation::kRemoved);
    }
    ++evicted.blocks;
    // Contains takes no reference, so it pins nothing on the Host tier.
    if (store_.HostPool() != nullptr && index.Contains(*store_.HostPool(), removed)) {
        ++evicted.host_resident;
    }
    return EvictedEntry{.key = std::move(removed), .device_companions_left = device_companions_left};
}

bool CacheEvictor::EvictCachedBlock(std::uint32_t group_id, CacheBlockLocation location,
                                    DeviceEvictionCounts DeviceEvictionStats::*reason,
                                    EvictedSegment* write_back_segment) {
    // Its own entry first: when that was the last holder of the hash's node,
    // the node leaves with its tail hint. Either cascade evicts only Device
    // entries under the hash: with none left, it would find nothing.
    const std::optional<EvictedEntry> evicted = EvictCachedBlockAlone(group_id, location, reason, write_back_segment);
    if (!evicted) {
        return false;
    }
    if (!evicted->device_companions_left) {
        return true;
    }
    const CacheKey& removed = evicted->key;
    if (const std::optional<std::int32_t> tail_tokens =
            store_.Boundaries().Tails().TailTokensOf(removed.content_hash)) {
        EvictTailEntries(removed.content_hash, *tail_tokens, reason, write_back_segment);
    } else {
        // A state checkpoint is matched only with the KV below it and with
        // the checkpoints every other State group holds at its hash: a KV
        // block of its page gone, or one of those checkpoints, it is dead
        // weight and leaves too. A checkpoint takes no KV along: the page
        // still serves the boundaries past it.
        store_.ForEachBoundaryKey(removed, /*tail_tokens=*/0, [&](std::size_t i, const CacheKey& key) {
            if (!store_.GroupIsState(i)) {
                return;
            }
            if (const std::optional<CacheBlockLocation> location = store_.LocationOf(store_.DevicePool(), i, key)) {
                EvictCachedBlockAlone(store_.Groups()[i].Id(), *location, reason, write_back_segment);
            }
        });
    }
    return true;
}

std::vector<CacheEvictor::UnevictedVictim> CacheEvictor::EvictPlanned(std::span<const VictimSet> victims,
                                                                      bool write_back, const VictimReasons& reasons) {
    std::vector<UnevictedVictim> unevicted;
    VictimPrefetch prefetch{victims, store_.Groups(), store_.DevicePool()};
    for (const VictimSet& set : victims) {
        EvictedSegment* segment = write_back ? &evicted_segments_.emplace_back() : nullptr;
        for (const auto& [group_id, location, in_stranded_suffix, is_request_reclaim] : set.members) {
            prefetch.Advance();
            DeviceEvictionCounts DeviceEvictionStats::*reason = is_request_reclaim   ? reasons.request_reclaim
                                                                : in_stranded_suffix ? reasons.stranded_suffix
                                                                                     : reasons.victim;
            if (!EvictCachedBlock(group_id, location, reason, segment)) {
                unevicted.push_back(UnevictedVictim{
                    .group_id = group_id, .location = location, .reason = reason, .write_back_segment = segment});
            }
        }
    }
    return unevicted;
}

void CacheEvictor::EvictTailEntries(const ContentHash& hash, std::int32_t tail_tokens,
                                    DeviceEvictionCounts DeviceEvictionStats::*reason,
                                    EvictedSegment* write_back_segment) {
    store_.ForEachBoundaryKey(CacheKey{.content_hash = hash}, tail_tokens, [&](std::size_t i, const CacheKey& key) {
        // A companion still pinned by a request table stays indexed; it
        // ages out through the ordinary eviction order.
        if (const std::optional<CacheBlockLocation> location = store_.LocationOf(store_.DevicePool(), i, key)) {
            EvictCachedBlockAlone(store_.Groups()[i].Id(), *location, reason, write_back_segment);
        }
    });
}

std::vector<CacheEvictor::EvictedSegment> CacheEvictor::TakeSegments() {
    std::deque<EvictedSegment> recorded = std::exchange(evicted_segments_, {});
    // Union-find over the recording order: merged_into[s] leads to the
    // segment s was merged with.
    std::vector<std::size_t> merged_into(recorded.size());
    std::iota(merged_into.begin(), merged_into.end(), std::size_t{0});
    const auto root_of = [&](std::size_t s) {
        while (merged_into[s] != s) {
            s = merged_into[s] = merged_into[merged_into[s]];
        }
        return s;
    };
    // The segment that carries each tree node's KV page. A deeper checkpoint
    // boundary depends on the KV above it, never on a checkpoint above it.
    std::unordered_map<const BoundaryTree::Node*, std::size_t> segment_with_kv_page;
    // The first segment recording each key. A key evicted again after it was
    // published again is copied once, for both records, so their segments are
    // merged.
    std::unordered_map<CacheKey, std::size_t, CacheKeyHash> segment_with_key;
    std::vector<const BoundaryTree::Node*> segment_top_of(recorded.size(), nullptr);
    for (std::size_t s = 0; s < recorded.size(); ++s) {
        for (const EvictedBlock& member : recorded[s].members) {
            // The record holds the node.
            const BoundaryTree::Node* node = store_.Boundaries().Find(member.key.content_hash);
            if (!store_.GroupIsState(member.key.group_id)) {
                segment_with_kv_page.emplace(node, s);
            }
            if (const auto [it, inserted] = segment_with_key.try_emplace(member.key, s); !inserted) {
                merged_into[root_of(it->second)] = root_of(s);
            }
            if (segment_top_of[s] == nullptr || node->boundary_tokens < segment_top_of[s]->boundary_tokens) {
                segment_top_of[s] = node;
            }
        }
    }
    // A segment depends on another whose KV page lies at or above its
    // shallowest boundary; every boundary it holds is under that one.
    for (std::size_t s = 0; s < recorded.size(); ++s) {
        for (const BoundaryTree::Node* ancestor = segment_top_of[s]; ancestor != nullptr; ancestor = ancestor->parent) {
            if (const auto it = segment_with_kv_page.find(ancestor); it != segment_with_kv_page.end()) {
                merged_into[root_of(it->second)] = root_of(s);
            }
        }
    }
    // Members concatenated in recording order: a segment another one hangs
    // under was recorded after it (its climb took a page the earlier segment's
    // checkpoint boundary still depended on), so deeper checkpoint boundaries
    // still come first.
    std::vector<EvictedSegment> segments;
    std::vector<std::size_t> output_index_of_root(recorded.size(), recorded.size());
    for (std::size_t s = 0; s < recorded.size(); ++s) {
        std::size_t& output_index = output_index_of_root[root_of(s)];
        if (output_index == recorded.size()) {
            output_index = segments.size();
            segments.emplace_back();
        }
        EvictedSegment& segment = segments[output_index];
        segment.newest_access_epoch = std::max(segment.newest_access_epoch, recorded[s].newest_access_epoch);
        segment.members.insert(segment.members.end(), std::make_move_iterator(recorded[s].members.begin()),
                               std::make_move_iterator(recorded[s].members.end()));
    }
    return segments;
}

bool CacheEvictor::EvictHostCachedBlock(std::uint32_t group_id, CacheBlockLocation location,
                                        HostEvictionCounts HostEvictionStats::*reason) {
    const auto evict_entry = [&](std::uint32_t entry_group, CacheBlockLocation entry_location) {
        std::optional<CacheKey> removed =
            store_.Groups()[entry_group].Index().Evict(*store_.HostPool(), entry_location);
        if (!removed) {
            return removed;
        }
        HostEvictionCounts& evicted = host_evictions_.*reason;
        ++evicted.blocks;
        // Contains takes no reference, so it pins nothing on the Device tier.
        if (store_.Groups()[entry_group].Index().Contains(store_.DevicePool(), *removed)) {
            ++evicted.device_resident;
        }
        return removed;
    };
    const std::optional<CacheKey> removed = evict_entry(group_id, location);
    if (!removed) {
        return false;
    }
    if (store_.GroupIsState(group_id)) {
        // A checkpoint resumes only with every State group's block at its
        // hash: the others leave with it unless pinned. A Host checkpoint
        // can resume on the Device's KV of its page, so no Host KV block
        // takes a checkpoint along.
        store_.ForEachBoundaryKey(*removed, /*tail_tokens=*/0, [&](std::size_t i, const CacheKey& key) {
            if (!store_.GroupIsState(i)) {
                return;
            }
            if (const std::optional<CacheBlockLocation> sibling = store_.LocationOf(*store_.HostPool(), i, key)) {
                evict_entry(store_.Groups()[i].Id(), *sibling);
            }
        });
    }
    return true;
}

void CacheEvictor::FlushHostEntry(std::uint32_t group_id, CacheBlockLocation location) {
    _assert(store_.Groups()[group_id].Index().Evict(*store_.HostPool(), location).has_value(),
            "clearable Host cache entry disappeared");
    ++host_evictions_.flush.blocks;
}

void CacheEvictor::RecordTail(const ContentHash& prior, const TailPublication& tail) {
    const std::optional<TailTable::Tail> dropped = store_.Boundaries().RecordTail(prior, tail.tail_tokens, tail.hash);
    if (!dropped) {
        return;
    }
    EvictTailEntries(dropped->hash, dropped->tail_tokens, &DeviceEvictionStats::tail_cap,
                     /*write_back_segment=*/nullptr);
    if (store_.HostPool() == nullptr) {
        return;
    }
    // No probe enumerates a dropped tail, so its Host entries could never be
    // matched again either. One still in use (by a retracted request's image,
    // or as a copy's source) stays, and so does what a write-through store
    // still in flight publishes at its ACK; both age out through ordinary
    // Host eviction. An eviction write-back still in flight publishes nothing
    // of the tail: its settlement finds the hint gone and cancels it. Under
    // write-back what stays is collected as a dead entry once released.
    store_.ForEachBoundaryKey(
        CacheKey{.content_hash = dropped->hash}, dropped->tail_tokens, [&](std::size_t i, const CacheKey& key) {
            if (const std::optional<CacheBlockLocation> location = store_.LocationOf(*store_.HostPool(), i, key)) {
                EvictHostCachedBlock(store_.Groups()[i].Id(), *location, &HostEvictionStats::tail_cap);
            }
        });
    // What stayed in use on either tier goes once released.
    store_.Boundaries().QueueDeadCandidate(dropped->hash);
}

void CacheEvictor::DropEmptySegments() {
    std::erase_if(evicted_segments_, [](const EvictedSegment& segment) { return segment.members.empty(); });
}

}  // namespace tokenspeed
