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

#include "resource/hybrid_prefix_cache/hybrid_prefix_cache.h"
#include "resource/allocator/mamba_chunk_allocator.h"
#include "resource/allocator/mamba_host_allocator.h"
#include "resource/allocator/paged_cache_group.h"
#include "resource/radix_tree/paged_cache_snapshot.h"
#include "resource/radix_tree/node_range.h"
#include "resource/radix_tree/radix_tree.h"
#include "resource/radix_tree/tree_node.h"
#include "scheduler/operations/forward.h"
#include "utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace tokenspeed {
namespace {

const TreeNode* FindExactRadixNode(const KVPrefixCache& cache,
                                   std::span<const std::span<const std::int32_t>> token_pages,
                                   std::int32_t target_raw_tokens) {
    const RadixTree& tree = cache.GetRadixTree();
    const std::int32_t page_size = tree.PageSize();
    if (target_raw_tokens <= 0 || page_size <= 0 || target_raw_tokens % page_size != 0) {
        return nullptr;
    }

    const std::int32_t target_pages = target_raw_tokens / page_size;
    if (target_pages > static_cast<std::int32_t>(token_pages.size())) {
        return nullptr;
    }

    const TreeNode* current = tree.Root();
    std::int32_t matched_pages = 0;
    while (matched_pages < target_pages) {
        const auto& first_page = token_pages[matched_pages];
        if (first_page.size() != static_cast<std::size_t>(page_size)) {
            return nullptr;
        }

        const token_vec_t key(first_page.begin(), first_page.end());
        const TreeNode* child = FindChild(current, key);
        if (child == nullptr || child->Tokens().size() % static_cast<std::size_t>(page_size) != 0) {
            return nullptr;
        }

        const std::int32_t child_pages = static_cast<std::int32_t>(child->Tokens().size()) / page_size;
        if (child_pages <= 0 || matched_pages + child_pages > target_pages) {
            return nullptr;
        }
        for (std::int32_t page = 0; page < child_pages; ++page) {
            const auto& expected = token_pages[matched_pages + page];
            if (expected.size() != static_cast<std::size_t>(page_size)) {
                return nullptr;
            }
            const auto actual_begin = child->Tokens().begin() + page * page_size;
            if (!std::equal(actual_begin, actual_begin + page_size, expected.begin(), expected.end())) {
                return nullptr;
            }
        }

        matched_pages += child_pages;
        current = child;
    }
    return current;
}

}  // namespace

HybridPrefixCache::HybridPrefixCache(KVPrefixCache& kv_prefix_cache, MambaChunkAllocator* mamba_allocator,
                                     std::int32_t mamba_cache_chunk_size, MambaHostAllocator* mamba_host_allocator)
    : kv_prefix_cache_{kv_prefix_cache},
      mamba_allocator_{mamba_allocator},
      mamba_host_allocator_{mamba_host_allocator},
      mamba_eviction_manager_{mamba_allocator},
      mamba_cache_chunk_size_{mamba_cache_chunk_size} {}

HybridPrefixCache::~HybridPrefixCache() {
    // Paged snapshots live on KVPrefixCache TreeNodes but own pages from the
    // group allocators below. Scheduler destroys HybridPrefixCache before the
    // KV tree, so detach every external owner while those allocators are still
    // alive. Request tables are cleared first to drop their borrowed ids and
    // return any request-owned tail pages.
    request_paged_cache_tables_.clear();
    while (!paged_cache_snapshot_nodes_.empty()) {
        TreeNode* node = *paged_cache_snapshot_nodes_.begin();
        DetachPagedCacheSnapshotFromNode(node);
    }
    while (!paged_cache_host_snapshot_nodes_.empty()) {
        TreeNode* node = *paged_cache_host_snapshot_nodes_.begin();
        DetachPagedCacheHostSnapshotFromNode(node);
    }
    while (!paged_cache_pending_host_snapshot_nodes_.empty()) {
        TreeNode* node = *paged_cache_pending_host_snapshot_nodes_.begin();
        DetachPagedCachePendingHostSnapshotFromNode(node);
    }
}

MatchResult HybridPrefixCache::Match(const token_vec_t& token_ids, MatchIntent intent) {
    auto match = kv_prefix_cache_.Match(token_ids, intent);
    augmentMatch(match);
    augmentMatchPagedCache(match, intent);
    return match;
}

MatchResult HybridPrefixCache::Match(const std::vector<std::span<const std::int32_t>>& token_pages,
                                     MatchIntent intent) {
    auto match = kv_prefix_cache_.Match(token_pages, intent);
    augmentMatch(match);
    augmentMatchPagedCache(match, intent);
    return match;
}

void HybridPrefixCache::augmentMatch(MatchResult& match) const {
    if (mamba_allocator_ == nullptr) return;
    TreeNode* root = match.device.last_node;
    while (root != nullptr && !root->IsRoot()) root = root->Parent();
    if (root == nullptr) return;

    // Backward-compatible path: before Mamba L2 is enabled, only device Mamba is
    // a valid hybrid prefix source and both match tiers are truncated together.
    if (mamba_host_allocator_ == nullptr) {
        TreeNode* device_terminal = match.device.last_node;
        TreeNode* host_terminal = match.host.last_node;
        const std::int32_t page_size = match.device.page_size;
        const std::int32_t device_depth = device_terminal != nullptr ? device_terminal->DepthInPage(page_size) : 0;
        const std::int32_t host_depth = host_terminal != nullptr ? host_terminal->DepthInPage(page_size) : 0;
        const std::int32_t kv_depth = device_depth;
        TreeNode* kv_terminal = device_terminal;
        if (kv_terminal == nullptr || kv_terminal->IsRoot()) {
            if (host_terminal != nullptr && !host_terminal->IsRoot()) {
                TreeNode* mamba_node = FindLastMambaNode(host_terminal);
                if (mamba_node != nullptr) {
                    match.mamba_cow_src_index = mamba_node->MambaSlotIndex();
                    std::int32_t mamba_depth = mamba_node->DepthInPage(page_size);
                    if (host_depth > mamba_depth) {
                        const std::int32_t aligned_seqlen = AlignMambaCacheSeqlen(host_depth * page_size);
                        if (aligned_seqlen > mamba_depth * page_size) {
                            match.mamba_branching_seqlen = aligned_seqlen;
                        }
                    }
                    return;
                }
            }
            match.device.last_node = root;
            match.host.last_node = root;
            return;
        }

        TreeNode* mamba_node = FindLastMambaNode(kv_terminal);
        if (mamba_node == nullptr) {
            const std::int32_t aligned_seqlen = AlignMambaCacheSeqlen(kv_depth * match.device.page_size);
            if (aligned_seqlen > 0) {
                match.mamba_branching_seqlen = aligned_seqlen;
            }
            match.device.last_node = root;
            match.host.last_node = root;
            return;
        }

        std::int32_t mamba_depth = mamba_node->DepthInPage(page_size);
        match.mamba_cow_src_index = mamba_node->MambaSlotIndex();
        if (kv_depth > mamba_depth) {
            const std::int32_t aligned_seqlen = AlignMambaCacheSeqlen(kv_depth * page_size);
            if (aligned_seqlen > mamba_depth * page_size) {
                match.mamba_branching_seqlen = aligned_seqlen;
            }
        }
        match.device.last_node = device_depth >= mamba_depth ? mamba_node : root;
        match.host.last_node = host_depth >= mamba_depth ? mamba_node : root;
        return;
    }

    const std::int32_t page_size = match.device.page_size;
    const std::int32_t kv_depth = std::max(match.device.DepthInPage(), match.host.DepthInPage());

    TreeNode* device_mamba_node = FindLastMambaNode(match.device.last_node);
    TreeNode* host_mamba_node = FindLastMambaHostNode(match.host.last_node);
    const std::int32_t device_mamba_depth =
        device_mamba_node == nullptr ? 0 : device_mamba_node->DepthInPage(page_size);
    const std::int32_t host_mamba_depth = host_mamba_node == nullptr ? 0 : host_mamba_node->DepthInPage(page_size);
    const bool prefer_host_mamba = host_mamba_depth > device_mamba_depth;
    std::int32_t mamba_depth = 0;

    if (device_mamba_node != nullptr) {
        match.device.last_node = device_mamba_node;
        if (!prefer_host_mamba) {
            match.mamba_cow_src_index = device_mamba_node->MambaSlotIndex();
        }
        mamba_depth = std::max(mamba_depth, device_mamba_depth);
    } else {
        match.device.last_node = root;
    }

    if (host_mamba_node != nullptr) {
        match.host.last_node = host_mamba_node;
        match.mamba_host_src_index = host_mamba_node->MambaHostSlotIndex();
        if (prefer_host_mamba) {
            match.mamba_cow_src_index = -1;
        }
        mamba_depth = std::max(mamba_depth, host_mamba_depth);
    } else {
        match.host.last_node = root;
    }

    if (kv_depth > mamba_depth) {
        const std::int32_t aligned_seqlen = AlignMambaCacheSeqlen(kv_depth * page_size);
        if (aligned_seqlen > mamba_depth * page_size) {
            match.mamba_branching_seqlen = aligned_seqlen;
        }
    }
}

std::int32_t HybridPrefixCache::AlignMambaCacheSeqlen(std::int32_t seqlen) const {
    if (mamba_cache_chunk_size_ <= 0) return seqlen;
    return (seqlen / mamba_cache_chunk_size_) * mamba_cache_chunk_size_;
}

TreeNode* HybridPrefixCache::FindLastMambaNode(TreeNode* from) const {
    for (TreeNode* cur = from; cur != nullptr && !cur->IsRoot(); cur = cur->Parent()) {
        if (cur->HasMamba()) return cur;
    }
    return nullptr;
}

TreeNode* HybridPrefixCache::FindLastMambaHostNode(TreeNode* from) const {
    for (TreeNode* cur = from; cur != nullptr && !cur->IsRoot(); cur = cur->Parent()) {
        if (cur->HasMambaOnHost()) return cur;
    }
    return nullptr;
}

bool HybridPrefixCache::EnsureMambaHostCapacityByEvict(std::int32_t num_slots, TreeNode* protected_node) {
    if (mamba_host_allocator_ == nullptr) return num_slots <= 0;
    if (mamba_host_allocator_->AvailableSlots() >= num_slots) return true;

    std::vector<TreeNode*> candidates;
    candidates.reserve(mamba_host_nodes_.size());
    for (TreeNode* node : mamba_host_nodes_) {
        if (node == nullptr || node == protected_node || !node->HasMambaOnHost()) continue;
        if (node->OnHost() && GetResource<ResourceType::Host>(node).RefCount() > 0) continue;
        candidates.push_back(node);
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const TreeNode* lhs, const TreeNode* rhs) { return lhs->Time() < rhs->Time(); });

    for (TreeNode* node : candidates) {
        if (mamba_host_allocator_->AvailableSlots() >= num_slots) break;
        node->DetachMambaHost();
        mamba_host_nodes_.erase(node);
    }
    if (mamba_host_allocator_->AvailableSlots() < num_slots) {
        spdlog::warn("[HybridPrefixCache] mamba host capacity exhausted required={} after_evict_available={}",
                     num_slots, mamba_host_allocator_->AvailableSlots());
    }
    return mamba_host_allocator_->AvailableSlots() >= num_slots;
}

std::vector<TransferPair> HybridPrefixCache::PrepareMambaHostWriteBack(const std::vector<TreeNode*>& nodes) {
    std::vector<TransferPair> transfers;
    if (mamba_allocator_ == nullptr || mamba_host_allocator_ == nullptr) return transfers;

    std::int32_t needed = 0;
    for (TreeNode* node : nodes) {
        if (node != nullptr && node->HasMamba() && !node->HasMambaOnHost() &&
            pending_mamba_host_writebacks_.find(node) == pending_mamba_host_writebacks_.end()) {
            needed++;
        }
    }
    if (!EnsureMambaHostCapacityByEvict(needed)) return transfers;

    for (TreeNode* node : nodes) {
        if (node == nullptr || !node->HasMamba() || node->HasMambaOnHost()) continue;
        if (pending_mamba_host_writebacks_.find(node) != pending_mamba_host_writebacks_.end()) continue;
        auto slot = mamba_host_allocator_->Allocate();
        if (!slot.has_value()) break;
        const std::int32_t device_idx = node->MambaSlotIndex();
        const std::int32_t host_idx = slot->Index();
        pending_mamba_host_writebacks_.emplace(node, std::make_unique<MambaSlot>(std::move(*slot)));
        transfers.push_back(TransferPair{CacheKind::kMamba, device_idx, host_idx});
    }
    return transfers;
}

std::vector<TransferPair> HybridPrefixCache::PrepareMambaDeviceLoadBack(const std::vector<TreeNode*>& nodes) {
    std::vector<TransferPair> transfers;
    if (mamba_allocator_ == nullptr || mamba_host_allocator_ == nullptr) return transfers;

    for (TreeNode* node : nodes) {
        if (node == nullptr || !node->HasMambaOnHost() || node->HasMamba()) continue;
        auto slot = mamba_allocator_->Allocate();
        if (!slot.has_value()) break;
        const std::int32_t host_idx = node->MambaHostSlotIndex();
        const std::int32_t device_idx = slot->Index();
        node->AttachMamba(std::make_unique<MambaSlot>(std::move(*slot)));
        mamba_eviction_manager_.TrackNode(node);
        transfers.push_back(TransferPair{CacheKind::kMamba, host_idx, device_idx});
    }
    return transfers;
}

bool HybridPrefixCache::EnsureMambaCapacityByEvict(std::int32_t num_slots, TreeNode* protected_node) {
    if (mamba_allocator_ == nullptr) return num_slots <= 0;
    return mamba_eviction_manager_.EnsureCapacity(num_slots, protected_node);
}

void HybridPrefixCache::InsertMamba(TreeNode* terminal_node, std::unique_ptr<MambaSlot> slot) {
    if (terminal_node == nullptr || slot == nullptr) return;
    if (mamba_allocator_ == nullptr) {
        throw std::logic_error("HybridPrefixCache::InsertMamba: mamba adjunct not enabled");
    }
    const std::int32_t page_size = kv_prefix_cache_.PageSize();
    if (page_size <= 0 || terminal_node->DepthInTokens() % static_cast<std::size_t>(page_size) != 0) {
        throw std::logic_error("HybridPrefixCache::InsertMamba: terminal node is not block-aligned");
    }
    terminal_node->AttachMamba(std::move(slot));
    mamba_eviction_manager_.TrackNode(terminal_node);
}

void HybridPrefixCache::RefreshPagedCacheSnapshotCompleteness(PagedCacheSnapshot& snapshot) const {
    snapshot.complete_families.clear();
    snapshot.continuation_state_complete = false;
    bool history_complete = !paged_cache_history_groups_.empty();
    for (const auto& gid : paged_cache_history_groups_) {
        if (snapshot.groups.find(gid) == snapshot.groups.end()) {
            history_complete = false;
            break;
        }
    }
    if (history_complete) {
        snapshot.complete_families.insert(PagedCacheGroupFamily::History);
    }
    bool state_complete = !paged_cache_state_groups_.empty();
    for (const auto& gid : paged_cache_state_groups_) {
        if (snapshot.groups.find(gid) == snapshot.groups.end()) {
            state_complete = false;
            break;
        }
    }
    if (state_complete) {
        snapshot.complete_families.insert(PagedCacheGroupFamily::State);
    }
    bool continuation_state_complete = !paged_cache_continuation_state_groups_.empty();
    for (const auto& gid : paged_cache_continuation_state_groups_) {
        if (snapshot.groups.find(gid) == snapshot.groups.end()) {
            continuation_state_complete = false;
            break;
        }
    }
    snapshot.continuation_state_complete = continuation_state_complete;
}

bool HybridPrefixCache::adoptExistingPagedCacheSnapshot(PagedCacheSnapshot& existing,
                                                        std::map<std::string, PagedCacheGroupTable>& tables,
                                                        std::int32_t target) {
    struct Adoption {
        std::string gid;
        PagedCacheGroupTable* table{nullptr};
        const PagedCacheGroupSnapshot* segment{nullptr};
        PagedCacheGroupFamily family{PagedCacheGroupFamily::History};
    };

    std::vector<Adoption> actions;
    actions.reserve(paged_cache_required_groups_.size());
    for (const auto& gid : paged_cache_required_groups_) {
        auto table_it = tables.find(gid);
        auto alloc_it = paged_cache_allocators_.find(gid);
        if (table_it == tables.end() || alloc_it == paged_cache_allocators_.end()) {
            return false;
        }

        const PagedCacheGroupFamily family = alloc_it->second->Config().family;
        auto segment_it = existing.groups.find(gid);
        if (segment_it == existing.groups.end()) {
            return false;
        }

        actions.push_back(Adoption{gid, &table_it->second, &segment_it->second, family});
    }

    for (const auto& action : actions) {
        if (action.family == PagedCacheGroupFamily::History) {
            action.table->AdoptSnapshotSegment(action.segment->pages.Ids(), target);
            continue;
        }

        action.table->AdoptStateSnapshotSegment(action.segment->pages.Ids(), action.segment->base_logical_page, target);
    }
    return true;
}

bool HybridPrefixCache::commitTerminalContinuationSnapshot(std::map<std::string, PagedCacheGroupTable>& tables,
                                                           TreeNode* terminal, std::int32_t target) {
    if (paged_cache_continuation_state_groups_.empty()) return false;
    if (terminal == nullptr || target <= 0) return false;

    PagedCacheSnapshot* snapshot = terminal->GetPagedCacheSnapshotMut();
    if (snapshot == nullptr || snapshot->prefix_len_tokens != target ||
        !snapshot->IsCompleteFor(PagedCacheGroupFamily::History)) {
        return false;
    }

    struct Action {
        std::string gid;
        PagedCacheGroupTable* table{nullptr};
    };

    std::vector<Action> actions;
    actions.reserve(paged_cache_continuation_state_groups_.size());
    for (const auto& gid : paged_cache_continuation_state_groups_) {
        if (snapshot->groups.find(gid) != snapshot->groups.end()) continue;

        auto table_it = tables.find(gid);
        auto alloc_it = paged_cache_allocators_.find(gid);
        if (table_it == tables.end() || alloc_it == paged_cache_allocators_.end()) return false;

        const auto& cfg = alloc_it->second->Config();
        const std::int32_t raw_per_page = cfg.RawTokensPerPage();
        if (raw_per_page <= 0 || target % raw_per_page != 0 || target > table_it->second.RawTokenCursor()) {
            return false;
        }

        const std::int32_t retained_tokens = cfg.sliding_window_tokens.value_or(0);
        if (retained_tokens <= 0 || retained_tokens % raw_per_page != 0) return false;

        const std::int32_t lower_raw = std::max(0, target - retained_tokens);
        const std::int32_t lower_page = lower_raw / raw_per_page;
        const std::int32_t target_page = target / raw_per_page;
        const std::int32_t table_base = table_it->second.BaseLogicalPage();
        const std::int32_t table_end = table_base + table_it->second.Size();
        if (table_base > lower_page || table_end < target_page) return false;

        actions.push_back(Action{gid, &table_it->second});
    }

    bool committed_all = true;
    for (const auto& action : actions) {
        auto result = action.table->CheckpointStateToSnapshot(target);
        if (result.pages.Empty()) {
            committed_all = false;
            break;
        }

        PagedCacheGroupSnapshot group_snap{};
        group_snap.pages = std::move(result.pages);
        group_snap.base_logical_page = result.segment_base_logical_page;
        group_snap.raw_token_cursor = target;
        group_snap.sliding = action.table->IsSliding();
        snapshot->groups.emplace(action.gid, std::move(group_snap));
    }

    RefreshPagedCacheSnapshotCompleteness(*snapshot);
    return committed_all && snapshot->continuation_state_complete;
}

bool HybridPrefixCache::AttachPagedCacheSnapshotToNode(TreeNode* node, std::unique_ptr<PagedCacheSnapshot> snapshot) {
    if (node == nullptr || snapshot == nullptr) return false;
    // Compute completeness from what is present. The policy-driven "snapshot
    // must be full" invariant is enforced upstream by CommitChunk, which only
    // attaches full snapshots; direct callers (tests, future restore paths)
    // may attach history-only or state-only snapshots without policy gating.
    RefreshPagedCacheSnapshotCompleteness(*snapshot);
    node->AttachPagedCacheSnapshot(std::move(snapshot));
    paged_cache_snapshot_nodes_.insert(node);
    return true;
}

bool HybridPrefixCache::AttachPagedCacheHostSnapshotToNode(TreeNode* node,
                                                           std::unique_ptr<PagedCacheSnapshot> snapshot) {
    if (node == nullptr || snapshot == nullptr) return false;
    if (!node->OnHost()) return false;
    RefreshPagedCacheSnapshotCompleteness(*snapshot);
    node->AttachPagedCacheHostSnapshot(std::move(snapshot));
    paged_cache_host_snapshot_nodes_.insert(node);
    return true;
}

std::unique_ptr<PagedCacheSnapshot> HybridPrefixCache::DetachPagedCacheSnapshotFromNode(TreeNode* node) {
    if (node == nullptr) return nullptr;
    paged_cache_snapshot_nodes_.erase(node);
    return node->DetachPagedCacheSnapshot();
}

bool HybridPrefixCache::isPagedCacheSnapshotBorrowed(const TreeNode* node,
                                                     std::optional<PagedCacheGroupFamily> family) const {
    if (node == nullptr) return false;
    const PagedCacheSnapshot* snapshot = node->GetPagedCacheSnapshot();
    if (snapshot == nullptr) return false;

    for (const auto& [_, tables] : request_paged_cache_tables_) {
        for (const auto& [group_id, group_snapshot] : snapshot->groups) {
            if (family.has_value()) {
                auto allocator_it = paged_cache_allocators_.find(group_id);
                if (allocator_it == paged_cache_allocators_.end() || allocator_it->second == nullptr ||
                    allocator_it->second->Config().family != *family) {
                    continue;
                }
            }
            auto table_it = tables.find(group_id);
            if (table_it == tables.end()) continue;
            const auto& borrowed = table_it->second.BorrowedPageIds();
            const auto& snapshot_pages = group_snapshot.pages.Ids();
            if (std::any_of(snapshot_pages.begin(), snapshot_pages.end(), [&borrowed](std::int32_t page_id) {
                    return std::find(borrowed.begin(), borrowed.end(), page_id) != borrowed.end();
                })) {
                return true;
            }
        }
    }
    return false;
}

std::unique_ptr<PagedCacheSnapshot> HybridPrefixCache::DetachPagedCacheHostSnapshotFromNode(TreeNode* node) {
    if (node == nullptr) return nullptr;
    paged_cache_host_snapshot_nodes_.erase(node);
    return node->DetachPagedCacheHostSnapshot();
}

std::unique_ptr<PagedCacheSnapshot> HybridPrefixCache::DetachPagedCachePendingHostSnapshotFromNode(TreeNode* node) {
    if (node == nullptr) return nullptr;
    paged_cache_pending_host_snapshot_nodes_.erase(node);
    return node->DetachPagedCachePendingHostSnapshot();
}

void HybridPrefixCache::OnKVEvict(TreeNode* node) {
    if (node == nullptr) return;
    if (mamba_allocator_ != nullptr && node->HasMamba()) {
        mamba_eviction_manager_.UntrackNode(node);
        node->DetachMamba();
        if (node->Parent() != nullptr) {
            mamba_eviction_manager_.UpdateLeaf(node->Parent());
        }
    }
    // Passive paged-cache detach on KV LRU drop normally returns OwnedPages
    // via RAII. A retracted/loadback request's side table can outlive the
    // device KV resource, so retain snapshots whose page ids it still borrows.
    if (node->HasPagedCacheSnapshot() && !isPagedCacheSnapshotBorrowed(node)) {
        DetachPagedCacheSnapshotFromNode(node);
    }
}

void HybridPrefixCache::OnNodeDestroyed(TreeNode* node) {
    if (node == nullptr) return;
    // The node is about to be freed by RadixTree::PruneEmptyByNode. Drop it from
    // every adjunct set that stores a raw TreeNode* so none is left dangling.
    // Mirror the per-resource eviction callbacks (OnKVEvict / OnKVHostEvict):
    // detach the node's slots first, then refresh the parent's mamba-leaf
    // membership. Detaching here (rather than relying on ~TreeNode) is required
    // so UpdateLeaf(parent) re-derives the parent's status with this node
    // already mamba-less -- otherwise the still-attached child would mask the
    // parent and it would never be promoted to a mamba leaf.
    if (mamba_allocator_ != nullptr && node->HasMamba()) {
        mamba_eviction_manager_.UntrackNode(node);
        node->DetachMamba();
        if (node->Parent() != nullptr) {
            mamba_eviction_manager_.UpdateLeaf(node->Parent());
        }
    }
    if (node->HasPagedCacheSnapshot()) {
        _assert(!isPagedCacheSnapshotBorrowed(node),
                "HybridPrefixCache::OnNodeDestroyed: paged snapshot still has a request-table borrower");
        DetachPagedCacheSnapshotFromNode(node);
    }
    if (node->HasPagedCacheHostSnapshot()) {
        DetachPagedCacheHostSnapshotFromNode(node);
    }
    if (node->HasPagedCachePendingHostSnapshot()) {
        DetachPagedCachePendingHostSnapshotFromNode(node);
    }
    // Host-side Mamba L2 bookkeeping (no-op when the L2 pool is disabled, since
    // these sets are only populated when mamba_host_allocator_ is present).
    pending_mamba_host_writebacks_.erase(node);
    if (node->HasMambaOnHost()) {
        node->DetachMambaHost();
        mamba_host_nodes_.erase(node);
        mamba_host_writeback_done_nodes_.erase(node);
    }
}

void HybridPrefixCache::OnKVHostEvict(TreeNode* node) {
    if (node == nullptr) return;
    if (node->HasPagedCacheHostSnapshot()) {
        DetachPagedCacheHostSnapshotFromNode(node);
    }
    if (node->HasPagedCachePendingHostSnapshot()) {
        DetachPagedCachePendingHostSnapshotFromNode(node);
    }
    if (mamba_host_allocator_ == nullptr) return;
    pending_mamba_host_writebacks_.erase(node);
    if (node->HasMambaOnHost()) {
        node->DetachMambaHost();
        mamba_host_nodes_.erase(node);
        mamba_host_writeback_done_nodes_.erase(node);
    }
}

void HybridPrefixCache::DemoteIdleMambaDeviceCopiesPresentOnHost() {
    if (mamba_allocator_ == nullptr || mamba_host_allocator_ == nullptr) return;

    std::int32_t demoted = 0;
    std::vector<TreeNode*> nodes(mamba_host_writeback_done_nodes_.begin(), mamba_host_writeback_done_nodes_.end());
    for (TreeNode* node : nodes) {
        if (node == nullptr || !node->HasMambaOnHost()) {
            mamba_host_writeback_done_nodes_.erase(node);
            continue;
        }
        if (!node->HasMamba()) {
            mamba_host_writeback_done_nodes_.erase(node);
            continue;
        }
        if (node->OnDevice() && node->Device().RefCount() != 0) {
            continue;
        }
        OnKVDeviceDemote(node);
        mamba_host_writeback_done_nodes_.erase(node);
        ++demoted;
    }
    if (demoted > 0) {
        spdlog::debug("[HybridPrefixCache][mamba_l2] demoted device copies after host writeback count={}", demoted);
    }
}

void HybridPrefixCache::OnMambaHostWriteBackDone(TreeNode* last_node, bool success) {
    if (last_node == nullptr) return;
    std::vector<TreeNode*> nodes;
    for (TreeNode* node : LeafToRoot(last_node)) {
        if (node == nullptr || !node->OnHost()) break;
        nodes.push_back(node);
    }
    OnMambaHostWriteBackDone(nodes, success);
}

void HybridPrefixCache::OnMambaHostWriteBackDone(const std::vector<TreeNode*>& nodes, bool success) {
    if (mamba_allocator_ == nullptr || mamba_host_allocator_ == nullptr) return;

    std::int32_t attached = 0;
    std::int32_t completed = 0;
    for (TreeNode* node : nodes) {
        if (node == nullptr || !node->OnHost()) continue;
        auto pending = pending_mamba_host_writebacks_.find(node);
        if (pending != pending_mamba_host_writebacks_.end()) {
            if (!success) {
                pending_mamba_host_writebacks_.erase(pending);
                continue;
            }
            node->AttachMambaHost(std::move(pending->second));
            pending_mamba_host_writebacks_.erase(pending);
            mamba_host_nodes_.insert(node);
            ++attached;
        }
        if (node->HasMambaOnHost()) {
            mamba_host_writeback_done_nodes_.insert(node);
            ++completed;
        }
    }
    if (attached > 0 || completed > 0) {
        spdlog::debug("[HybridPrefixCache][mamba_l2] host writeback done attach_count={} completed_nodes={}", attached,
                      completed);
    }
    if (success) {
        DemoteIdleMambaDeviceCopiesPresentOnHost();
    }
}

void HybridPrefixCache::OnKVDeviceDemote(TreeNode* node) {
    if (node == nullptr) return;
    if (node->HasPagedCacheSnapshot() && node->HasPagedCacheHostSnapshot() && !isPagedCacheSnapshotBorrowed(node)) {
        DetachPagedCacheSnapshotFromNode(node);
    }
    if (mamba_allocator_ == nullptr) return;
    if (node->HasMamba() && node->HasMambaOnHost()) {
        mamba_eviction_manager_.UntrackNode(node);
        node->DetachMamba();
        if (node->Parent() != nullptr) {
            mamba_eviction_manager_.UpdateLeaf(node->Parent());
        }
    }
}

std::int32_t HybridPrefixCache::AvailableSlots() const {
    if (mamba_allocator_ == nullptr) return 0;
    return mamba_allocator_->AvailableSlots();
}

void HybridPrefixCache::RegisterPagedCacheGroup(std::unique_ptr<PagedCacheGroupAllocator> allocator) {
    if (allocator == nullptr) {
        throw std::invalid_argument("HybridPrefixCache::RegisterPagedCacheGroup: null allocator");
    }
    std::string gid = allocator->Config().group_id;
    if (paged_cache_allocators_.find(gid) != paged_cache_allocators_.end()) {
        throw std::invalid_argument("HybridPrefixCache::RegisterPagedCacheGroup: duplicate group_id: " + gid);
    }
    paged_cache_allocators_.emplace(std::move(gid), std::move(allocator));
}

void HybridPrefixCache::RegisterPagedCacheHostGroup(std::unique_ptr<PagedCacheGroupAllocator> allocator) {
    if (allocator == nullptr) {
        throw std::invalid_argument("HybridPrefixCache::RegisterPagedCacheHostGroup: null allocator");
    }
    const std::string gid = allocator->Config().group_id;
    if (paged_cache_host_allocators_.find(gid) != paged_cache_host_allocators_.end()) {
        throw std::invalid_argument("HybridPrefixCache::RegisterPagedCacheHostGroup: duplicate group_id: " + gid);
    }
    auto device_it = paged_cache_allocators_.find(gid);
    if (device_it == paged_cache_allocators_.end() || device_it->second == nullptr) {
        throw std::invalid_argument("HybridPrefixCache::RegisterPagedCacheHostGroup: device group missing: " + gid);
    }
    const auto& host_cfg = allocator->Config();
    const auto& device_cfg = device_it->second->Config();
    if (host_cfg.rows_per_page != device_cfg.rows_per_page ||
        host_cfg.entry_stride_tokens != device_cfg.entry_stride_tokens || host_cfg.retention != device_cfg.retention ||
        host_cfg.sliding_window_tokens != device_cfg.sliding_window_tokens || host_cfg.family != device_cfg.family) {
        throw std::invalid_argument("HybridPrefixCache::RegisterPagedCacheHostGroup: host/device config mismatch for " +
                                    gid);
    }
    paged_cache_host_allocators_.emplace(gid, std::move(allocator));
}

void HybridPrefixCache::EnablePagedCacheAdjunct(
    std::vector<std::string> required_groups, std::unordered_map<std::string, std::int32_t> sliding_window_per_group) {
    if (required_groups.empty()) {
        throw std::invalid_argument("HybridPrefixCache::EnablePagedCacheAdjunct: required_groups must be non-empty");
    }
    std::vector<std::string> history_gids;
    std::vector<std::string> state_gids;
    std::vector<std::string> required_sliding_gids;
    history_gids.reserve(required_groups.size());
    state_gids.reserve(required_groups.size());
    required_sliding_gids.reserve(required_groups.size());

    // Partition required groups by family; collect sliding-group entries for
    // post-validation against `sliding_window_per_group`.
    for (const auto& gid : required_groups) {
        auto it = paged_cache_allocators_.find(gid);
        if (it == paged_cache_allocators_.end() || it->second == nullptr) {
            throw std::invalid_argument("HybridPrefixCache::EnablePagedCacheAdjunct: required group '" + gid +
                                        "' missing from registered allocators");
        }
        const auto& cfg = it->second->Config();
        const std::int32_t raw_per_page = cfg.RawTokensPerPage();
        if (raw_per_page <= 0) {
            throw std::invalid_argument("HybridPrefixCache::EnablePagedCacheAdjunct: required group '" + gid +
                                        "' has non-positive RawTokensPerPage");
        }
        if (cfg.family == PagedCacheGroupFamily::History) {
            history_gids.push_back(gid);
        } else {
            state_gids.push_back(gid);
        }
        if (cfg.retention == PagedCacheGroupConfig::Retention::SlidingWindow) {
            auto win_it = sliding_window_per_group.find(gid);
            if (win_it == sliding_window_per_group.end() || win_it->second <= 0) {
                throw std::invalid_argument("HybridPrefixCache::EnablePagedCacheAdjunct: sliding group '" + gid +
                                            "' missing positive sliding_window entry");
            }
            required_sliding_gids.push_back(gid);
        }
    }
    if (history_gids.empty()) {
        throw std::invalid_argument(
            "HybridPrefixCache::EnablePagedCacheAdjunct: at least one History-family group required");
    }
    if (sliding_window_per_group.size() != required_sliding_gids.size()) {
        throw std::invalid_argument(
            "HybridPrefixCache::EnablePagedCacheAdjunct: sliding_window_per_group keys must exactly "
            "match the set of required groups whose retention is SlidingWindow");
    }
    // History alignment = LCM(raw_per_page) across History-family groups.
    std::int32_t history_alignment = 1;
    for (const auto& gid : history_gids) {
        const auto& cfg = paged_cache_allocators_.find(gid)->second->Config();
        history_alignment = std::lcm(history_alignment, cfg.RawTokensPerPage());
    }
    std::vector<std::string> continuation_state_gids;
    continuation_state_gids.reserve(paged_cache_allocators_.size());
    for (const auto& [gid, allocator] : paged_cache_allocators_) {
        const auto& cfg = allocator->Config();
        if (cfg.family != PagedCacheGroupFamily::State) continue;
        if (cfg.retention != PagedCacheGroupConfig::Retention::SlidingWindow ||
            !cfg.sliding_window_tokens.has_value() || *cfg.sliding_window_tokens <= 0) {
            throw std::invalid_argument("HybridPrefixCache::EnablePagedCacheAdjunct: state group '" + gid +
                                        "' must declare positive sliding_window_tokens");
        }
        continuation_state_gids.push_back(gid);
    }
    // Required State restore needs page-aligned State segments at every history
    // snapshot boundary. Transport-only State groups are only restored from the
    // exact terminal continuation snapshot.
    for (const auto& gid : state_gids) {
        const auto& cfg = paged_cache_allocators_.find(gid)->second->Config();
        const std::int32_t raw_per_page = cfg.RawTokensPerPage();
        if (history_alignment % raw_per_page != 0) {
            throw std::invalid_argument("HybridPrefixCache::EnablePagedCacheAdjunct: state group '" + gid +
                                        "' RawTokensPerPage=" + std::to_string(raw_per_page) +
                                        " does not divide history_alignment=" + std::to_string(history_alignment));
        }
    }

    paged_cache_history_alignment_tokens_ = history_alignment;
    paged_cache_required_groups_ = std::move(required_groups);
    paged_cache_sliding_window_per_group_ = std::move(sliding_window_per_group);
    paged_cache_history_groups_ = std::move(history_gids);
    paged_cache_state_groups_ = std::move(state_gids);
    paged_cache_continuation_state_groups_ = std::move(continuation_state_gids);
    paged_cache_history_group_set_ =
        std::unordered_set<std::string>(paged_cache_history_groups_.begin(), paged_cache_history_groups_.end());
    paged_cache_state_group_set_ =
        std::unordered_set<std::string>(paged_cache_state_groups_.begin(), paged_cache_state_groups_.end());
    paged_cache_continuation_state_group_set_ = std::unordered_set<std::string>(
        paged_cache_continuation_state_groups_.begin(), paged_cache_continuation_state_groups_.end());
}

namespace {

TreeNode* RootOf(TreeNode* from) {
    TreeNode* root = from;
    while (root != nullptr && !root->IsRoot()) root = root->Parent();
    return root;
}

TreeNode* CapNodeToDepth(TreeNode* from, std::int32_t depth) {
    TreeNode* node = from;
    while (node != nullptr && !node->IsRoot() && static_cast<std::int32_t>(node->DepthInTokens()) > depth) {
        node = node->Parent();
    }
    return node;
}

enum class PagedCacheResidency { kDevice, kHost };

const PagedCacheSnapshot* GetPagedCacheSnapshotForResidency(TreeNode* node, PagedCacheResidency residency) {
    if (node == nullptr) return nullptr;
    switch (residency) {
        case PagedCacheResidency::kDevice:
            if (!node->OnDevice()) return nullptr;
            return node->GetPagedCacheSnapshot();
        case PagedCacheResidency::kHost:
            if (!node->OnHost()) return nullptr;
            return node->GetPagedCacheHostSnapshot();
    }
    return nullptr;
}

// Ancestor path (excluding root), reversed so element 0 is closest to root.
std::vector<TreeNode*> CollectAncestorPathRootToLeaf(TreeNode* from) {
    std::vector<TreeNode*> path;
    for (TreeNode* n = from; n != nullptr && !n->IsRoot(); n = n->Parent()) {
        path.push_back(n);
    }
    std::reverse(path.begin(), path.end());
    return path;
}

void AssemblePagedCacheGroupPages(MatchResult::PagedCache& out, const std::string& gid,
                                  std::span<TreeNode* const> chain, bool is_sliding, PagedCacheResidency residency) {
    std::vector<std::int32_t> page_ids;
    std::int32_t base_logical_page = 0;
    if (!chain.empty()) {
        const PagedCacheSnapshot* earliest_snap = GetPagedCacheSnapshotForResidency(chain.front(), residency);
        if (earliest_snap != nullptr && is_sliding) {
            auto git = earliest_snap->groups.find(gid);
            if (git != earliest_snap->groups.end()) {
                base_logical_page = git->second.base_logical_page;
            }
        }
        for (TreeNode* anc : chain) {
            const PagedCacheSnapshot* snap = GetPagedCacheSnapshotForResidency(anc, residency);
            if (snap == nullptr) continue;
            auto git = snap->groups.find(gid);
            if (git == snap->groups.end()) continue;
            const auto& seg_ids = git->second.pages.Ids();
            page_ids.insert(page_ids.end(), seg_ids.begin(), seg_ids.end());
        }
    }
    out.per_group_page_ids[gid] = std::move(page_ids);
    out.per_group_base_logical_page[gid] = base_logical_page;
}

bool ImportPagedCacheGroupSnapshot(MatchResult::PagedCache& out, const std::string& gid,
                                   const PagedCacheGroupSnapshot& group_snap, const PagedCacheGroupAllocator& allocator,
                                   std::int32_t target_raw_tokens) {
    const std::int32_t raw_per_page = allocator.Config().RawTokensPerPage();
    const auto& ids = group_snap.pages.Ids();
    if (raw_per_page <= 0 || ids.empty()) return false;
    const std::int32_t end_raw = (group_snap.base_logical_page + static_cast<std::int32_t>(ids.size())) * raw_per_page;
    if (end_raw != target_raw_tokens) return false;
    out.per_group_page_ids[gid] = ids;
    out.per_group_base_logical_page[gid] = group_snap.base_logical_page;
    return true;
}

bool AssemblePagedCacheStateGroupPagesToTarget(MatchResult::PagedCache& out, const std::string& gid,
                                               std::span<TreeNode* const> chain,
                                               const PagedCacheGroupAllocator& allocator,
                                               std::int32_t target_raw_tokens, std::int32_t retained_tokens,
                                               PagedCacheResidency residency) {
    const std::int32_t raw_per_page = allocator.Config().RawTokensPerPage();
    if (raw_per_page <= 0 || target_raw_tokens <= 0 || target_raw_tokens % raw_per_page != 0) return false;
    if (retained_tokens <= 0 || retained_tokens % raw_per_page != 0) return false;

    const std::int32_t lower_raw = std::max(0, target_raw_tokens - retained_tokens);
    const std::int32_t lower_page = lower_raw / raw_per_page;
    const std::int32_t target_page = target_raw_tokens / raw_per_page;

    std::vector<std::int32_t> page_ids;
    std::int32_t base_logical_page = lower_page;
    std::int32_t next_logical_page = lower_page;
    bool started = false;

    for (TreeNode* anc : chain) {
        const PagedCacheSnapshot* snap = GetPagedCacheSnapshotForResidency(anc, residency);
        if (snap == nullptr) continue;
        auto git = snap->groups.find(gid);
        if (git == snap->groups.end()) continue;

        const auto& seg_ids = git->second.pages.Ids();
        if (seg_ids.empty()) continue;

        const std::int32_t seg_base = git->second.base_logical_page;
        const std::int32_t seg_end = seg_base + static_cast<std::int32_t>(seg_ids.size());
        if (seg_end <= lower_page) continue;
        if (seg_base >= target_page) break;

        std::int32_t copy_begin = std::max(seg_base, lower_page);
        if (!started) {
            if (copy_begin > lower_page) return false;
            started = true;
            base_logical_page = copy_begin;
            next_logical_page = copy_begin;
        }
        if (copy_begin > next_logical_page) return false;
        copy_begin = std::max(copy_begin, next_logical_page);
        const std::int32_t copy_end = std::min(seg_end, target_page);
        if (copy_end <= copy_begin) continue;

        const auto begin = seg_ids.begin() + (copy_begin - seg_base);
        const auto end = seg_ids.begin() + (copy_end - seg_base);
        page_ids.insert(page_ids.end(), begin, end);
        next_logical_page = copy_end;
    }

    if (!started || page_ids.empty() || next_logical_page != target_page) return false;
    out.per_group_page_ids[gid] = std::move(page_ids);
    out.per_group_base_logical_page[gid] = base_logical_page;
    return true;
}

}  // namespace

void HybridPrefixCache::augmentMatchPagedCache(MatchResult& match, MatchIntent intent) const {
    if (!HasPagedCacheAdjunct()) return;
    TreeNode* token_terminal = match.token_terminal != nullptr ? match.token_terminal : match.device.last_node;
    if (token_terminal == nullptr) return;

    const std::int32_t align = paged_cache_history_alignment_tokens_;

    auto cap_to_root = [&]() {
        TreeNode* root = RootOf(token_terminal);
        match.device.last_node = root;
        match.host.last_node = RootOf(match.host.last_node);
        match.paged_cache = MatchResult::PagedCache{};
        match.paged_cache_host = MatchResult::PagedCache{};
    };

    auto build_hit = [&](PagedCacheResidency residency) -> MatchResult::PagedCache {
        MatchResult::PagedCache hit{};
        std::vector<TreeNode*> path = CollectAncestorPathRootToLeaf(token_terminal);

        TreeNode* deepest_history = nullptr;
        std::vector<TreeNode*> history_chain;
        std::int32_t expected_depth = align;
        for (TreeNode* n : path) {
            const std::int32_t d = static_cast<std::int32_t>(n->DepthInTokens());
            if (d < expected_depth) continue;
            if (d > expected_depth) break;
            const auto* snap = GetPagedCacheSnapshotForResidency(n, residency);
            if (snap == nullptr) break;
            if (!snap->IsCompleteFor(PagedCacheGroupFamily::History)) break;
            deepest_history = n;
            history_chain.push_back(n);
            expected_depth += align;
        }
        if (deepest_history == nullptr) {
            return hit;
        }

        hit.history_hit_tokens = static_cast<std::int32_t>(deepest_history->DepthInTokens());

        const std::int32_t history_hit = hit.history_hit_tokens;
        auto build_exact_continuation_hit = [&](TreeNode* terminal,
                                                std::span<TreeNode* const> history_span) -> MatchResult::PagedCache {
            const std::int32_t terminal_depth = static_cast<std::int32_t>(terminal->DepthInTokens());
            const auto* terminal_snap = GetPagedCacheSnapshotForResidency(terminal, residency);
            const bool terminal_state_complete =
                paged_cache_continuation_state_groups_.empty() ||
                (terminal_snap != nullptr && terminal_snap->continuation_state_complete);
            if (terminal_snap == nullptr || terminal_snap->prefix_len_tokens != terminal_depth ||
                !terminal_state_complete) {
                return {};
            }

            MatchResult::PagedCache terminal_hit{};
            terminal_hit.last_node = terminal;
            terminal_hit.prefix_len_tokens = terminal_depth;
            terminal_hit.history_hit_tokens = terminal_depth;

            for (const auto& gid : paged_cache_history_groups_) {
                const bool is_sliding =
                    paged_cache_sliding_window_per_group_.find(gid) != paged_cache_sliding_window_per_group_.end();
                AssemblePagedCacheGroupPages(terminal_hit, gid, history_span, is_sliding, residency);
            }

            for (const auto& gid : paged_cache_continuation_state_groups_) {
                auto alloc_it = paged_cache_allocators_.find(gid);
                if (alloc_it == paged_cache_allocators_.end()) return {};
                const std::int32_t retained_tokens = alloc_it->second->Config().sliding_window_tokens.value_or(0);
                if (!AssemblePagedCacheStateGroupPagesToTarget(terminal_hit, gid, history_span, *alloc_it->second,
                                                               terminal_depth, retained_tokens, residency)) {
                    return {};
                }
            }
            return terminal_hit;
        };

        if (intent == MatchIntent::StateRecovery) {
            if (deepest_history != token_terminal) return {};
            const std::span<TreeNode* const> history_span{history_chain};
            return build_exact_continuation_hit(token_terminal, history_span);
        }

        if (!paged_cache_continuation_state_groups_.empty()) {
            const std::span<TreeNode* const> history_span{history_chain};
            MatchResult::PagedCache terminal_hit = build_exact_continuation_hit(deepest_history, history_span);
            if (terminal_hit.last_node != nullptr) return terminal_hit;

            const bool has_transport_only_state = std::any_of(
                paged_cache_continuation_state_groups_.begin(), paged_cache_continuation_state_groups_.end(),
                [&](const std::string& gid) { return !paged_cache_state_group_set_.contains(gid); });
            if (has_transport_only_state) return {};
        }

        // Configurations whose State groups are all required can rebuild them from
        // checkpoint segments. Pure History configurations also use this path.
        std::int32_t worst_window = 0;
        for (const auto& gid : paged_cache_state_groups_) {
            auto it = paged_cache_sliding_window_per_group_.find(gid);
            if (it != paged_cache_sliding_window_per_group_.end()) {
                worst_window = std::max(worst_window, it->second);
            }
        }
        const std::int32_t segments_needed = worst_window > 0 ? (worst_window + align - 1) / align : 1;

        TreeNode* usable_node = nullptr;
        if (paged_cache_state_groups_.empty()) {
            usable_node = deepest_history;
        } else {
            for (std::int32_t end_idx = static_cast<std::int32_t>(history_chain.size()) - 1; end_idx >= 0; --end_idx) {
                const std::int32_t start_idx = std::max(0, end_idx - segments_needed + 1);
                bool ok = true;
                for (std::int32_t i = start_idx; i <= end_idx; ++i) {
                    const auto* snap = GetPagedCacheSnapshotForResidency(history_chain[i], residency);
                    if (snap == nullptr || !snap->IsCompleteFor(PagedCacheGroupFamily::State)) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    usable_node = history_chain[end_idx];
                    break;
                }
            }
        }
        if (usable_node == nullptr) {
            return MatchResult::PagedCache{};
        }

        const std::int32_t usable = static_cast<std::int32_t>(usable_node->DepthInTokens());
        while (!history_chain.empty() && static_cast<std::int32_t>(history_chain.back()->DepthInTokens()) > usable) {
            history_chain.pop_back();
        }

        hit.last_node = usable_node;
        hit.prefix_len_tokens = usable;

        const std::span<TreeNode* const> history_span{history_chain};
        for (const auto& gid : paged_cache_history_groups_) {
            const bool is_sliding =
                paged_cache_sliding_window_per_group_.find(gid) != paged_cache_sliding_window_per_group_.end();
            AssemblePagedCacheGroupPages(hit, gid, history_span, is_sliding, residency);
        }
        if (!paged_cache_state_groups_.empty()) {
            const std::size_t take =
                std::min<std::size_t>(history_chain.size(), static_cast<std::size_t>(segments_needed));
            const std::span<TreeNode* const> state_span = history_span.last(take);
            for (const auto& gid : paged_cache_state_groups_) {
                const bool is_sliding =
                    paged_cache_sliding_window_per_group_.find(gid) != paged_cache_sliding_window_per_group_.end();
                AssemblePagedCacheGroupPages(hit, gid, state_span, is_sliding, residency);
            }
        }
        return hit;
    };

    match.paged_cache = build_hit(PagedCacheResidency::kDevice);
    match.paged_cache_host = build_hit(PagedCacheResidency::kHost);

    const bool has_device_hit = match.paged_cache.last_node != nullptr && match.paged_cache.prefix_len_tokens > 0;
    const bool has_host_hit =
        match.paged_cache_host.last_node != nullptr && match.paged_cache_host.prefix_len_tokens > 0;
    if (!has_device_hit && !has_host_hit) {
        cap_to_root();
        return;
    }

    TreeNode* root = RootOf(token_terminal);
    if (has_device_hit) {
        match.device.last_node = match.paged_cache.last_node;
    } else {
        match.device.last_node = root;
    }
    const std::int32_t usable = std::max(match.paged_cache.prefix_len_tokens, match.paged_cache_host.prefix_len_tokens);
    match.host.last_node = CapNodeToDepth(match.host.last_node, usable);
    if (match.host.last_node == nullptr) {
        match.host.last_node = RootOf(token_terminal);
    }
}

std::vector<std::string> HybridPrefixCache::PagedCacheGroupIds() const {
    std::vector<std::string> ids;
    ids.reserve(paged_cache_allocators_.size());
    for (const auto& [gid, _] : paged_cache_allocators_) {
        ids.push_back(gid);
    }
    return ids;
}

std::int32_t HybridPrefixCache::PagedCacheGroupTotalPages(const std::string& group_id) const {
    auto it = paged_cache_allocators_.find(group_id);
    if (it == paged_cache_allocators_.end()) {
        throw std::out_of_range("HybridPrefixCache::PagedCacheGroupTotalPages: group_id not configured");
    }
    return it->second->TotalPages();
}

std::int32_t HybridPrefixCache::PagedCacheGroupAvailablePages(const std::string& group_id) const {
    auto it = paged_cache_allocators_.find(group_id);
    if (it == paged_cache_allocators_.end()) {
        throw std::out_of_range("HybridPrefixCache::PagedCacheGroupAvailablePages: group_id not configured");
    }
    return it->second->AvailablePages();
}

std::int64_t HybridPrefixCache::PagedCacheGroupFailedAllocCount(const std::string& group_id) const {
    auto it = paged_cache_allocators_.find(group_id);
    if (it == paged_cache_allocators_.end()) {
        throw std::out_of_range("HybridPrefixCache::PagedCacheGroupFailedAllocCount: group_id not configured");
    }
    return it->second->FailedAllocCount();
}

std::int32_t HybridPrefixCache::PagedCacheHostGroupTotalPages(const std::string& group_id) const {
    auto it = paged_cache_host_allocators_.find(group_id);
    if (it == paged_cache_host_allocators_.end()) {
        throw std::out_of_range("HybridPrefixCache::PagedCacheHostGroupTotalPages: group_id not configured");
    }
    return it->second->TotalPages();
}

std::int32_t HybridPrefixCache::PagedCacheHostGroupAvailablePages(const std::string& group_id) const {
    auto it = paged_cache_host_allocators_.find(group_id);
    if (it == paged_cache_host_allocators_.end()) {
        throw std::out_of_range("HybridPrefixCache::PagedCacheHostGroupAvailablePages: group_id not configured");
    }
    return it->second->AvailablePages();
}

std::int64_t HybridPrefixCache::PagedCacheHostGroupFailedAllocCount(const std::string& group_id) const {
    auto it = paged_cache_host_allocators_.find(group_id);
    if (it == paged_cache_host_allocators_.end()) {
        throw std::out_of_range("HybridPrefixCache::PagedCacheHostGroupFailedAllocCount: group_id not configured");
    }
    return it->second->FailedAllocCount();
}

std::vector<std::int32_t> HybridPrefixCache::GetRequestPagedCachePageIds(const std::string& request_id,
                                                                         const std::string& group_id) const {
    if (paged_cache_allocators_.find(group_id) == paged_cache_allocators_.end()) {
        throw std::out_of_range("HybridPrefixCache::GetRequestPagedCachePageIds: group_id not configured");
    }
    auto req_it = request_paged_cache_tables_.find(request_id);
    if (req_it == request_paged_cache_tables_.end()) {
        return {};
    }
    auto group_it = req_it->second.find(group_id);
    if (group_it == req_it->second.end()) {
        return {};
    }
    return group_it->second.PageIds();
}

std::int32_t HybridPrefixCache::GetRequestPagedCacheBaseLogicalPage(const std::string& request_id,
                                                                    const std::string& group_id) const {
    if (paged_cache_allocators_.find(group_id) == paged_cache_allocators_.end()) {
        throw std::out_of_range("HybridPrefixCache::GetRequestPagedCacheBaseLogicalPage: group_id not configured");
    }
    auto req_it = request_paged_cache_tables_.find(request_id);
    if (req_it == request_paged_cache_tables_.end()) {
        return 0;
    }
    auto group_it = req_it->second.find(group_id);
    if (group_it == req_it->second.end()) {
        return 0;
    }
    return group_it->second.BaseLogicalPage();
}

bool HybridPrefixCache::HasRequestPagedCacheTables(const std::string& request_id) const {
    auto it = request_paged_cache_tables_.find(request_id);
    return it != request_paged_cache_tables_.end() && !it->second.empty();
}

std::map<std::string, std::int32_t> HybridPrefixCache::InitialSimulatedFree() const {
    std::map<std::string, std::int32_t> out;
    for (const auto& [gid, allocator] : paged_cache_allocators_) {
        out[gid] = allocator->AvailablePages();
    }
    return out;
}

void HybridPrefixCache::AcquireForRequest(const std::string& request_id, std::int32_t first_raw_position_of_op,
                                          std::int32_t target_raw_tokens_exclusive,
                                          const MatchResult::PagedCache& paged_cache_hit) {
    if (paged_cache_allocators_.empty()) return;
    auto& tables = request_paged_cache_tables_[request_id];
    const bool has_hit = (paged_cache_hit.last_node != nullptr) && (paged_cache_hit.prefix_len_tokens > 0);
    for (const auto& [group_id, allocator] : paged_cache_allocators_) {
        auto it = tables.find(group_id);
        const bool fresh_table = (it == tables.end());
        if (fresh_table) {
            it = tables.emplace(group_id, PagedCacheGroupTable(allocator.get())).first;
            // Import borrowed-prefix BEFORE ReleaseSkipped/Acquire on a fresh table.
            if (has_hit) {
                auto pid_it = paged_cache_hit.per_group_page_ids.find(group_id);
                if (pid_it != paged_cache_hit.per_group_page_ids.end() && !pid_it->second.empty()) {
                    std::int32_t base_logical_page = 0;
                    auto base_it = paged_cache_hit.per_group_base_logical_page.find(group_id);
                    if (base_it != paged_cache_hit.per_group_base_logical_page.end()) {
                        base_logical_page = base_it->second;
                    }
                    std::vector<std::int32_t> page_ids_copy = pid_it->second;
                    it->second.ImportPrefixBorrowed(std::move(page_ids_copy), base_logical_page,
                                                    paged_cache_hit.prefix_len_tokens);
                }
            }
        }
        const auto& cfg = allocator->Config();
        if (cfg.retention == PagedCacheGroupConfig::Retention::SlidingWindow && cfg.sliding_window_tokens.has_value()) {
            const std::int32_t lower = std::max(0, first_raw_position_of_op - *cfg.sliding_window_tokens + 1);
            it->second.ReleaseSkipped(lower);
        }
        it->second.Acquire(target_raw_tokens_exclusive);
    }
}

void HybridPrefixCache::ReleaseRequest(const std::string& request_id) {
    auto it = request_paged_cache_tables_.find(request_id);
    if (it != request_paged_cache_tables_.end()) {
        for (auto& [_, table] : it->second) {
            table.ReleaseAll();
        }
        request_paged_cache_tables_.erase(it);
    }
    DemoteIdleMambaDeviceCopiesPresentOnHost();
}

void HybridPrefixCache::RewindRequest(const std::string& request_id, std::int32_t accepted_raw_tokens,
                                      std::int32_t protected_tail_tokens) {
    if (accepted_raw_tokens < 0) {
        throw std::invalid_argument("HybridPrefixCache::RewindRequest: accepted_raw_tokens must be >= 0");
    }
    if (protected_tail_tokens < 0) {
        throw std::invalid_argument("HybridPrefixCache::RewindRequest: protected_tail_tokens must be >= 0");
    }
    const std::int64_t target =
        static_cast<std::int64_t>(accepted_raw_tokens) + static_cast<std::int64_t>(protected_tail_tokens);
    if (target > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("HybridPrefixCache::RewindRequest: retained target exceeds int32 range");
    }
    auto it = request_paged_cache_tables_.find(request_id);
    if (it == request_paged_cache_tables_.end()) {
        return;
    }
    for (auto& [_, table] : it->second) {
        table.RewindTail(static_cast<std::int32_t>(target));
    }
}

std::vector<PagedCacheTransferPair> HybridPrefixCache::PreparePagedCacheDeviceLoadBack(
    const std::string& request_id, const MatchResult::PagedCache& host_hit) {
    std::vector<PagedCacheTransferPair> transfers;
    if (!HasPagedCacheHostAdjunct() || host_hit.last_node == nullptr || host_hit.prefix_len_tokens <= 0) {
        return transfers;
    }
    auto& tables = request_paged_cache_tables_[request_id];
    if (!tables.empty()) {
        throw std::logic_error("HybridPrefixCache::PreparePagedCacheDeviceLoadBack: request already has tables");
    }

    for (const auto& [gid, host_pages] : host_hit.per_group_page_ids) {
        if (host_pages.empty()) continue;
        auto device_alloc_it = paged_cache_allocators_.find(gid);
        if (device_alloc_it == paged_cache_allocators_.end() || device_alloc_it->second == nullptr) {
            ReleaseRequest(request_id);
            return {};
        }
        OwnedPages device_pages = device_alloc_it->second->AcquireOwned(static_cast<std::int32_t>(host_pages.size()));
        if (device_pages.Size() != static_cast<std::int32_t>(host_pages.size())) {
            ReleaseRequest(request_id);
            return {};
        }
        std::int32_t base_logical_page = 0;
        auto base_it = host_hit.per_group_base_logical_page.find(gid);
        if (base_it != host_hit.per_group_base_logical_page.end()) {
            base_logical_page = base_it->second;
        }

        PagedCacheTransferPair transfer{gid, host_pages, device_pages.Ids()};
        auto [table_it, inserted] = tables.emplace(gid, PagedCacheGroupTable(device_alloc_it->second.get()));
        if (!inserted) {
            ReleaseRequest(request_id);
            throw std::logic_error("HybridPrefixCache::PreparePagedCacheDeviceLoadBack: duplicate table for group " +
                                   gid);
        }
        table_it->second.ImportPrefixOwned(std::move(device_pages), base_logical_page, host_hit.prefix_len_tokens);
        transfers.push_back(std::move(transfer));
    }
    if (transfers.empty()) {
        ReleaseRequest(request_id);
    }
    return transfers;
}

std::vector<PagedCacheTransferPair> HybridPrefixCache::PreparePagedCacheHostWriteBack(
    const std::vector<TreeNode*>& nodes) {
    std::vector<PagedCacheTransferPair> transfers;
    if (!HasPagedCacheHostAdjunct()) return transfers;

    auto is_pinned = [](TreeNode* node) {
        for (TreeNode* cur = node; cur != nullptr && !cur->IsRoot(); cur = cur->Parent()) {
            if (!cur->OnHost()) return true;
            if (cur->Host().RefCount() > 0) return true;
        }
        return false;
    };
    std::vector<TreeNode*> prune_candidates;
    std::size_t prune_cursor = 0;
    bool prune_candidates_initialized = false;
    auto prune_one = [&]() {
        if (!prune_candidates_initialized) {
            prune_candidates.reserve(paged_cache_host_snapshot_nodes_.size());
            for (TreeNode* candidate : paged_cache_host_snapshot_nodes_) {
                if (candidate != nullptr && candidate->HasPagedCacheHostSnapshot()) {
                    prune_candidates.push_back(candidate);
                }
            }
            std::sort(prune_candidates.begin(), prune_candidates.end(), [](TreeNode* a, TreeNode* b) {
                if (a->Time() != b->Time()) return a->Time() < b->Time();
                if (a->DepthInTokens() != b->DepthInTokens()) return a->DepthInTokens() > b->DepthInTokens();
                return a->SeqId() < b->SeqId();
            });
            prune_candidates_initialized = true;
        }

        while (prune_cursor < prune_candidates.size()) {
            TreeNode* victim = prune_candidates[prune_cursor++];
            if (victim == nullptr || !victim->HasPagedCacheHostSnapshot() || is_pinned(victim)) continue;

            const std::size_t victim_depth = victim->DepthInTokens();
            auto primary = DetachPagedCacheHostSnapshotFromNode(victim);
            (void)primary;

            std::vector<TreeNode*> descendants;
            for (TreeNode* candidate : paged_cache_host_snapshot_nodes_) {
                if (candidate == nullptr || candidate == victim || !candidate->HasPagedCacheHostSnapshot()) continue;
                if (candidate->DepthInTokens() <= victim_depth) continue;
                for (TreeNode* cur = candidate->Parent(); cur != nullptr && !cur->IsRoot(); cur = cur->Parent()) {
                    if (cur == victim) {
                        descendants.push_back(candidate);
                        break;
                    }
                }
            }
            for (TreeNode* descendant : descendants) {
                if (is_pinned(descendant)) continue;
                auto cascaded = DetachPagedCacheHostSnapshotFromNode(descendant);
                (void)cascaded;
            }
            return true;
        }
        return false;
    };

    for (TreeNode* node : nodes) {
        if (node == nullptr || !node->HasPagedCacheSnapshot() || node->HasPagedCacheHostSnapshot() ||
            node->HasPagedCachePendingHostSnapshot() || !node->OnHost()) {
            continue;
        }
        const PagedCacheSnapshot* device_snapshot = node->GetPagedCacheSnapshot();
        if (device_snapshot == nullptr || device_snapshot->groups.empty()) continue;

        auto pending = std::make_unique<PagedCacheSnapshot>();
        pending->prefix_len_tokens = device_snapshot->prefix_len_tokens;
        std::vector<PagedCacheTransferPair> node_transfers;
        bool ok = true;
        for (const auto& [gid, device_group] : device_snapshot->groups) {
            auto host_alloc_it = paged_cache_host_allocators_.find(gid);
            if (host_alloc_it == paged_cache_host_allocators_.end() || host_alloc_it->second == nullptr) {
                ok = false;
                break;
            }
            const auto& src_ids = device_group.pages.Ids();
            if (src_ids.empty()) continue;
            OwnedPages host_pages = host_alloc_it->second->AcquireOwned(static_cast<std::int32_t>(src_ids.size()));
            while (host_pages.Size() != static_cast<std::int32_t>(src_ids.size()) && prune_one()) {
                host_pages = host_alloc_it->second->AcquireOwned(static_cast<std::int32_t>(src_ids.size()));
            }
            if (host_pages.Size() != static_cast<std::int32_t>(src_ids.size())) {
                ok = false;
                break;
            }

            PagedCacheGroupSnapshot host_group{};
            host_group.pages = std::move(host_pages);
            host_group.base_logical_page = device_group.base_logical_page;
            host_group.raw_token_cursor = device_group.raw_token_cursor;
            host_group.sliding = device_group.sliding;

            node_transfers.push_back(PagedCacheTransferPair{gid, src_ids, host_group.pages.Ids()});
            pending->groups.emplace(gid, std::move(host_group));
        }
        if (!ok || pending->groups.empty()) {
            continue;
        }
        RefreshPagedCacheSnapshotCompleteness(*pending);
        node->AttachPagedCachePendingHostSnapshot(std::move(pending));
        paged_cache_pending_host_snapshot_nodes_.insert(node);
        transfers.insert(transfers.end(), std::make_move_iterator(node_transfers.begin()),
                         std::make_move_iterator(node_transfers.end()));
    }

    return transfers;
}

void HybridPrefixCache::OnPagedCacheHostWriteBackDone(const std::vector<TreeNode*>& nodes, bool success) {
    if (!HasPagedCacheHostAdjunct()) return;

    for (TreeNode* node : nodes) {
        if (node == nullptr || !node->HasPagedCachePendingHostSnapshot()) continue;
        if (!success) {
            DetachPagedCachePendingHostSnapshotFromNode(node);
            continue;
        }
        paged_cache_pending_host_snapshot_nodes_.erase(node);
        node->PromotePagedCachePendingHostSnapshot();
        if (node->HasPagedCacheHostSnapshot()) {
            paged_cache_host_snapshot_nodes_.insert(node);
        }
    }
}

void HybridPrefixCache::PopulateOp(ForwardOperationBase& op_base) const {
    if (paged_cache_allocators_.empty()) return;
    auto req_it = request_paged_cache_tables_.find(op_base.request_id);
    for (const auto& [gid, allocator] : paged_cache_allocators_) {
        std::vector<std::int32_t> pages;
        std::int32_t base_offset = 0;
        if (req_it != request_paged_cache_tables_.end()) {
            auto table_it = req_it->second.find(gid);
            if (table_it != req_it->second.end()) {
                pages = table_it->second.PageIds();
                base_offset = table_it->second.BaseLogicalPage();
            }
        }
        op_base.paged_cache_pages[gid] = std::move(pages);
        if (allocator->Config().retention == PagedCacheGroupConfig::Retention::SlidingWindow) {
            op_base.paged_cache_page_base_offsets[gid] = base_offset;
        }
    }
}

HybridPrefixCache::PagedCacheGroupAdmission HybridPrefixCache::checkPagedCacheGroupAdmission(
    const std::string& request_id, std::int32_t first_raw_position_of_op, std::int32_t target_raw_tokens_exclusive,
    const std::map<std::string, std::int32_t>& simulated_free, const MatchResult::PagedCache& paged_cache_hit,
    const PagedCacheAdmissionContext& context) const {
    PagedCacheGroupAdmission result;
    if (paged_cache_allocators_.empty() || target_raw_tokens_exclusive < 0) {
        return result;
    }

    std::unordered_set<std::string> existing_commit_snapshot_groups;
    if (context.commit_target_raw_tokens.has_value()) {
        const TreeNode* terminal =
            FindExactRadixNode(kv_prefix_cache_, context.commit_token_pages, *context.commit_target_raw_tokens);
        const PagedCacheSnapshot* snapshot = terminal != nullptr ? terminal->GetPagedCacheSnapshot() : nullptr;
        if (snapshot != nullptr) {
            for (const auto& [gid, _] : snapshot->groups) {
                existing_commit_snapshot_groups.insert(gid);
            }
        }
    }

    auto req_it = request_paged_cache_tables_.find(request_id);
    const bool has_hit = (paged_cache_hit.last_node != nullptr) && (paged_cache_hit.prefix_len_tokens > 0);
    for (const auto& [gid, allocator] : paged_cache_allocators_) {
        const auto& cfg = allocator->Config();
        const std::int32_t raw_per_page = cfg.RawTokensPerPage();
        if (cfg.entry_stride_tokens <= 0 || cfg.rows_per_page <= 0 || raw_per_page <= 0) {
            continue;
        }

        const std::int32_t entries = CeilDivPositive(target_raw_tokens_exclusive, cfg.entry_stride_tokens);
        const std::int32_t required = (entries + cfg.rows_per_page - 1) / cfg.rows_per_page;

        std::int32_t current_size = 0;
        std::int32_t current_active = 0;
        std::int32_t borrowed_in_table = 0;
        std::int32_t owned_in_table = 0;
        std::int32_t already_released = 0;
        std::int32_t committed_prefix = 0;
        std::int32_t raw_cursor = 0;
        bool table_exists = false;
        if (req_it != request_paged_cache_tables_.end()) {
            auto t_it = req_it->second.find(gid);
            if (t_it != req_it->second.end()) {
                table_exists = true;
                current_size = t_it->second.Size();
                current_active = t_it->second.ActivePagesCount();
                borrowed_in_table = t_it->second.BorrowedPagesCount();
                owned_in_table = t_it->second.OwnedPagesCount();
                already_released = t_it->second.ReleasedPagesCount();
                committed_prefix = t_it->second.CommittedPrefixLenTokens();
                raw_cursor = t_it->second.RawTokenCursor();
            }
        }

        std::int32_t borrowed_count = 0;
        std::int32_t borrowed_base = 0;
        if (has_hit && !table_exists) {
            auto pid_it = paged_cache_hit.per_group_page_ids.find(gid);
            if (pid_it != paged_cache_hit.per_group_page_ids.end()) {
                borrowed_count = static_cast<std::int32_t>(pid_it->second.size());
            }
            auto base_it = paged_cache_hit.per_group_base_logical_page.find(gid);
            if (base_it != paged_cache_hit.per_group_base_logical_page.end()) {
                borrowed_base = base_it->second;
            }
        }

        std::int32_t releasable_total = 0;
        std::int32_t releasable_owned = 0;
        std::int32_t empty_sliding_base_after_release = -1;
        if (cfg.retention == PagedCacheGroupConfig::Retention::SlidingWindow && cfg.sliding_window_tokens.has_value()) {
            const std::int32_t lower = std::max(0, first_raw_position_of_op - *cfg.sliding_window_tokens + 1);
            const std::int32_t target_releases = lower / raw_per_page;
            const std::int32_t logical_released_base = table_exists ? already_released : borrowed_base;
            const std::int32_t logical_size = table_exists ? current_size : borrowed_count;
            if (logical_size == 0 && target_releases > logical_released_base) {
                empty_sliding_base_after_release = target_releases;
            }
            releasable_total = std::max(0, target_releases - logical_released_base);
            releasable_total = std::min(releasable_total, current_active + borrowed_count);

            // Borrowed pages drop the index only (no pool credit); only the
            // owned-prefix slice contributes to releasable_owned.
            const std::int32_t borrowed_present_total = table_exists ? borrowed_in_table : borrowed_count;
            releasable_owned = releasable_total - std::min(releasable_total, borrowed_present_total);
            if (table_exists) {
                releasable_owned = std::min(releasable_owned, owned_in_table);
            }

            // Required State-family groups: CommitChunk converts the retained
            // snapshot tail from owned to borrowed before ReleaseSkipped runs,
            // so the only immediate pool credit is from stale-owned pages
            // dropped at the first commit step. Transport-only State groups do
            // not participate in intermediate snapshots; their terminal-only
            // checkpoint is accounted separately below when one is pending.
            const std::int32_t lcm = paged_cache_history_alignment_tokens_;
            const bool required_state_group =
                paged_cache_state_group_set_.find(gid) != paged_cache_state_group_set_.end();
            if (required_state_group && table_exists && lcm > 0 && committed_prefix + lcm <= raw_cursor) {
                const std::int32_t commit_target = committed_prefix + lcm;
                const std::int32_t retained_tokens = *cfg.sliding_window_tokens;
                const std::int32_t live_lower_raw = std::max(0, commit_target - retained_tokens);
                const std::int32_t live_lower_page = live_lower_raw / raw_per_page;
                std::int32_t base = already_released;
                if (live_lower_page > base) {
                    base += std::min(live_lower_page - base, borrowed_in_table);
                }
                releasable_owned = (live_lower_page > base) ? std::min(live_lower_page - base, owned_in_table) : 0;
            }

            // A terminal continuation checkpoint runs before ReleaseSkipped.
            // For transport-only State groups, owned pages in the trailing
            // checkpoint window become borrowed snapshot pages. Dropping
            // those borrowed ids later does not return pages to this pool,
            // so remove their overlap from the immediate release credit.
            const bool continuation_state_group =
                paged_cache_continuation_state_group_set_.find(gid) != paged_cache_continuation_state_group_set_.end();
            const bool terminal_group_already_exists =
                existing_commit_snapshot_groups.find(gid) != existing_commit_snapshot_groups.end();
            if (!required_state_group && continuation_state_group && table_exists && lcm > 0 &&
                context.commit_target_raw_tokens.has_value() && !terminal_group_already_exists) {
                const std::int32_t commit_target = *context.commit_target_raw_tokens;
                const std::int32_t retained_tokens = *cfg.sliding_window_tokens;
                if (commit_target > committed_prefix && commit_target <= raw_cursor && commit_target % lcm == 0 &&
                    commit_target % raw_per_page == 0 && retained_tokens % raw_per_page == 0) {
                    const std::int32_t owned_begin = already_released + borrowed_in_table;
                    const std::int32_t owned_end = owned_begin + owned_in_table;
                    const std::int32_t checkpoint_begin = std::max(0, commit_target - retained_tokens) / raw_per_page;
                    const std::int32_t checkpoint_end = commit_target / raw_per_page;
                    const std::int32_t retained_begin = std::max(owned_begin, checkpoint_begin);
                    const std::int32_t retained_end = std::min({owned_end, checkpoint_end, target_releases});
                    const std::int32_t retained_release_credit = std::max(0, retained_end - retained_begin);
                    releasable_owned = std::max(0, releasable_owned - retained_release_credit);
                }
            }
        }

        std::int32_t absolute_have =
            table_exists ? (already_released + current_size) : (borrowed_base + borrowed_count);
        if (empty_sliding_base_after_release > absolute_have) {
            absolute_have = empty_sliding_base_after_release;
        }
        const std::int32_t new_pages = std::max(0, required - absolute_have);
        std::int32_t free = allocator->AvailablePages();
        auto sf_it = simulated_free.find(gid);
        if (sf_it != simulated_free.end()) {
            free = sf_it->second;
        }
        result.releasable_owned_pages[gid] = releasable_owned;
        result.new_pages_needed[gid] = new_pages;
        if (free + releasable_owned < new_pages) {
            result.ok = false;
            result.failed_groups.insert(gid);
        }
    }
    return result;
}

void HybridPrefixCache::applyPagedCacheGroupAdmissionDebit(std::map<std::string, std::int32_t>& simulated_free,
                                                           const PagedCacheGroupAdmission& admission) {
    for (const auto& [gid, releasable_owned] : admission.releasable_owned_pages) {
        simulated_free[gid] += releasable_owned;
    }
    for (const auto& [gid, new_pages] : admission.new_pages_needed) {
        simulated_free[gid] -= new_pages;
    }
}

HybridPrefixCache::AdmissionFailureKind HybridPrefixCache::ClassifyAdmissionFailure(
    const PagedCacheGroupAdmission& admission) const {
    if (admission.ok) return AdmissionFailureKind::kNone;
    bool history_starved = false;
    bool state_starved = false;
    for (const auto& gid : admission.failed_groups) {
        if (paged_cache_history_group_set_.find(gid) != paged_cache_history_group_set_.end()) {
            history_starved = true;
        }
        if (paged_cache_continuation_state_group_set_.find(gid) != paged_cache_continuation_state_group_set_.end()) {
            state_starved = true;
        }
    }
    if (history_starved && state_starved) return AdmissionFailureKind::kBothStarved;
    if (history_starved) return AdmissionFailureKind::kHistoryStarved;
    if (state_starved) return AdmissionFailureKind::kStateStarved;
    return AdmissionFailureKind::kNone;
}

void HybridPrefixCache::refreshPagedCacheSimulatedFree(std::map<std::string, std::int32_t>& simulated_free) const {
    for (const auto& [gid, allocator] : paged_cache_allocators_) {
        simulated_free[gid] = allocator->AvailablePages();
    }
}

bool HybridPrefixCache::admitPagedCacheChunk(const std::string& request_id, std::int32_t first_raw_position_of_op,
                                             std::int32_t target_raw_tokens_exclusive,
                                             std::map<std::string, std::int32_t>& simulated_free,
                                             const MatchResult::PagedCache& paged_cache_hit,
                                             const PagedCacheAdmissionContext& context) {
    PagedCacheGroupAdmission admission = checkPagedCacheGroupAdmission(
        request_id, first_raw_position_of_op, target_raw_tokens_exclusive, simulated_free, paged_cache_hit, context);
    if (admission.ok) {
        applyPagedCacheGroupAdmissionDebit(simulated_free, admission);
        return true;
    }

    std::vector<TreeNode*> candidates;
    candidates.reserve(paged_cache_snapshot_nodes_.size());
    for (TreeNode* node : paged_cache_snapshot_nodes_) {
        if (node != nullptr && node->HasPagedCacheSnapshot()) {
            candidates.push_back(node);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](TreeNode* a, TreeNode* b) {
        if (a->Time() != b->Time()) return a->Time() < b->Time();
        if (a->DepthInTokens() != b->DepthInTokens()) return a->DepthInTokens() > b->DepthInTokens();
        return a->SeqId() < b->SeqId();
    });

    std::unordered_map<std::string, std::unordered_set<std::int32_t>> borrowed_pages_by_group;
    for (const auto& [_, tables] : request_paged_cache_tables_) {
        for (const auto& [group_id, table] : tables) {
            auto& borrowed = borrowed_pages_by_group[group_id];
            borrowed.insert(table.BorrowedPageIds().begin(), table.BorrowedPageIds().end());
        }
    }
    auto is_borrowed = [&](const TreeNode* node, std::optional<PagedCacheGroupFamily> family = std::nullopt) {
        if (node == nullptr || node->GetPagedCacheSnapshot() == nullptr) return false;
        for (const auto& [group_id, group_snapshot] : node->GetPagedCacheSnapshot()->groups) {
            if (family.has_value()) {
                auto allocator_it = paged_cache_allocators_.find(group_id);
                if (allocator_it == paged_cache_allocators_.end() || allocator_it->second == nullptr ||
                    allocator_it->second->Config().family != *family) {
                    continue;
                }
            }
            auto borrowed_it = borrowed_pages_by_group.find(group_id);
            if (borrowed_it == borrowed_pages_by_group.end()) continue;
            for (std::int32_t page_id : group_snapshot.pages.Ids()) {
                if (borrowed_it->second.contains(page_id)) return true;
            }
        }
        return false;
    };
    auto is_pinned = [&](TreeNode* node, std::optional<PagedCacheGroupFamily> family = std::nullopt) {
        if (is_borrowed(node, family)) return true;
        for (TreeNode* cur = node; cur != nullptr && !cur->IsRoot(); cur = cur->Parent()) {
            if (cur->OnDevice() && cur->Device().RefCount() > 0) return true;
        }
        return false;
    };

    std::size_t state_cursor = 0;
    std::size_t full_cursor = 0;
    auto prune_state = [&]() {
        while (state_cursor < candidates.size()) {
            TreeNode* node = candidates[state_cursor++];
            if (node == nullptr || !node->HasPagedCacheSnapshot() || is_pinned(node, PagedCacheGroupFamily::State)) {
                continue;
            }
            const auto* snapshot = node->GetPagedCacheSnapshot();
            if (!snapshot->IsCompleteFor(PagedCacheGroupFamily::State) && !snapshot->continuation_state_complete) {
                continue;
            }
            if (DetachStateSnapshotFromNode(node)) return true;
        }
        return false;
    };
    auto prune_full = [&]() {
        while (full_cursor < candidates.size()) {
            TreeNode* victim = candidates[full_cursor++];
            if (victim == nullptr || !victim->HasPagedCacheSnapshot() || is_pinned(victim)) continue;

            const std::size_t victim_depth = victim->DepthInTokens();
            auto primary = DetachPagedCacheSnapshotFromNode(victim);
            (void)primary;
            std::vector<TreeNode*> descendants;
            for (TreeNode* node : paged_cache_snapshot_nodes_) {
                if (node == nullptr || node == victim || !node->HasPagedCacheSnapshot()) continue;
                if (node->DepthInTokens() <= victim_depth) continue;
                for (TreeNode* cur = node->Parent(); cur != nullptr && !cur->IsRoot(); cur = cur->Parent()) {
                    if (cur == victim) {
                        descendants.push_back(node);
                        break;
                    }
                }
            }
            for (TreeNode* descendant : descendants) {
                if (is_pinned(descendant)) continue;
                auto cascaded = DetachPagedCacheSnapshotFromNode(descendant);
                (void)cascaded;
            }
            return true;
        }
        return false;
    };

    while (!admission.ok) {
        AdmissionFailureKind kind = ClassifyAdmissionFailure(admission);
        if (kind == AdmissionFailureKind::kNone) break;
        const bool pruned = kind == AdmissionFailureKind::kStateStarved ? prune_state() : prune_full();
        if (!pruned) break;
        refreshPagedCacheSimulatedFree(simulated_free);
        admission = checkPagedCacheGroupAdmission(request_id, first_raw_position_of_op, target_raw_tokens_exclusive,
                                                  simulated_free, paged_cache_hit, context);
    }
    if (!admission.ok) return false;
    applyPagedCacheGroupAdmissionDebit(simulated_free, admission);
    return true;
}

bool HybridPrefixCache::DetachStateSnapshotFromNode(TreeNode* node) {
    if (node == nullptr) return false;
    PagedCacheSnapshot* snap = node->GetPagedCacheSnapshotMut();
    if (snap == nullptr) return false;
    bool removed_any = false;
    for (const auto& gid : paged_cache_continuation_state_groups_) {
        auto it = snap->groups.find(gid);
        if (it != snap->groups.end()) {
            snap->groups.erase(it);
            removed_any = true;
        }
    }
    if (!removed_any) return false;
    snap->complete_families.erase(PagedCacheGroupFamily::State);
    snap->continuation_state_complete = false;
    // If nothing remains, fall through to full detach to keep invariants tidy.
    if (snap->groups.empty()) {
        DetachPagedCacheSnapshotFromNode(node);
    }
    return true;
}

bool HybridPrefixCache::AdmitChunk(const std::string& request_id, std::int32_t first_raw_position_of_op,
                                   std::int32_t target_raw_tokens_exclusive,
                                   std::map<std::string, std::int32_t>& simulated_free,
                                   const MatchResult::PagedCache& paged_cache_hit,
                                   std::optional<std::int32_t> commit_target_raw_tokens,
                                   std::span<const std::span<const std::int32_t>> commit_token_pages) {
    PagedCacheAdmissionContext context{
        .commit_target_raw_tokens = commit_target_raw_tokens,
        .commit_token_pages = commit_token_pages,
    };
    return admitPagedCacheChunk(request_id, first_raw_position_of_op, target_raw_tokens_exclusive, simulated_free,
                                paged_cache_hit, context);
}

void HybridPrefixCache::CommitChunk(const std::string& request_id, TreeNode* terminal) {
    if (!HasPagedCacheAdjunct()) return;
    if (terminal == nullptr) return;

    auto tables_it = request_paged_cache_tables_.find(request_id);
    if (tables_it == request_paged_cache_tables_.end()) return;
    auto& tables = tables_it->second;

    const std::int32_t lcm = paged_cache_history_alignment_tokens_;
    if (lcm <= 0) return;
    const auto& required_groups = paged_cache_required_groups_;
    if (required_groups.empty()) return;
    const auto& canonical_groups = paged_cache_history_groups_.empty() ? required_groups : paged_cache_history_groups_;

    auto canonical_it = tables.find(canonical_groups.front());
    if (canonical_it == tables.end()) return;
    std::int32_t last_committed = canonical_it->second.CommittedPrefixLenTokens();

    const std::int32_t chunk_depth = static_cast<std::int32_t>(terminal->DepthInTokens());
    if (chunk_depth <= 0) return;

    while (last_committed + lcm <= chunk_depth) {
        const std::int32_t target = last_committed + lcm;

        TreeNode* attach_node = kv_prefix_cache_.GetRadixTree().SplitAt(terminal, target);
        if (attach_node == nullptr) break;

        bool preflight_ok = true;
        const char* preflight_fail_reason = "unknown";
        std::string preflight_fail_group;
        std::int32_t fail_raw_per_page = -1;
        std::int32_t fail_committed = -1;
        std::int32_t fail_raw_cursor = -1;
        std::int32_t fail_base = -1;
        std::int32_t fail_borrowed = -1;
        std::int32_t fail_owned = -1;
        std::int32_t fail_committed_page = -1;
        std::int32_t fail_owned_base_page = -1;
        std::int32_t fail_pages_to_commit = -1;

        auto mark_preflight_failure = [&](const char* reason, const std::string& gid,
                                          const PagedCacheGroupTable* table = nullptr, std::int32_t raw_per_page = -1) {
            preflight_ok = false;
            preflight_fail_reason = reason;
            preflight_fail_group = gid;
            fail_raw_per_page = raw_per_page;
            if (table == nullptr) return;
            fail_committed = table->CommittedPrefixLenTokens();
            fail_raw_cursor = table->RawTokenCursor();
            fail_base = table->BaseLogicalPage();
            fail_borrowed = table->BorrowedPagesCount();
            fail_owned = table->OwnedPagesCount();
        };

        for (const auto& gid : required_groups) {
            auto t_it = tables.find(gid);
            if (t_it == tables.end()) {
                mark_preflight_failure("missing_table", gid);
                break;
            }
            const auto& table = t_it->second;
            const std::int32_t raw_per_page = table.RawTokensPerPage();
            if (raw_per_page <= 0) {
                mark_preflight_failure("invalid_raw_per_page", gid, &table, raw_per_page);
                break;
            }
            if (table.CommittedPrefixLenTokens() % raw_per_page != 0) {
                mark_preflight_failure("committed_unaligned", gid, &table, raw_per_page);
                break;
            }
            if (target % raw_per_page != 0) {
                mark_preflight_failure("target_unaligned", gid, &table, raw_per_page);
                break;
            }
            if (target <= table.CommittedPrefixLenTokens()) {
                mark_preflight_failure("target_not_advanced", gid, &table, raw_per_page);
                break;
            }
            if (target > table.RawTokenCursor()) {
                mark_preflight_failure("target_exceeds_raw_cursor", gid, &table, raw_per_page);
                break;
            }
            auto group_alloc_it = paged_cache_allocators_.find(gid);
            if (group_alloc_it == paged_cache_allocators_.end() || group_alloc_it->second == nullptr) {
                mark_preflight_failure("missing_allocator", gid, &table, raw_per_page);
                break;
            }
            const auto& cfg = group_alloc_it->second->Config();
            if (cfg.family == PagedCacheGroupFamily::History) {
                const std::int32_t committed_page = table.CommittedPrefixLenTokens() / raw_per_page;
                const std::int32_t owned_base_page = table.BaseLogicalPage() + table.BorrowedPagesCount();
                const std::int32_t pages_to_commit = (target - table.CommittedPrefixLenTokens()) / raw_per_page;
                if (owned_base_page != committed_page) {
                    mark_preflight_failure("history_owned_base_mismatch", gid, &table, raw_per_page);
                    fail_committed_page = committed_page;
                    fail_owned_base_page = owned_base_page;
                    fail_pages_to_commit = pages_to_commit;
                    break;
                }
                if (pages_to_commit > table.OwnedPagesCount()) {
                    mark_preflight_failure("history_owned_pages_short", gid, &table, raw_per_page);
                    fail_committed_page = committed_page;
                    fail_owned_base_page = owned_base_page;
                    fail_pages_to_commit = pages_to_commit;
                    break;
                }
            }
        }
        if (!preflight_ok) {
            spdlog::warn(
                "[HybridPrefixCache] CommitChunk: preflight failed request={} target={} chunk_depth={} lcm={} "
                "reason={} group={} committed={} raw_cursor={} raw_per_page={} base={} borrowed={} owned={} "
                "committed_page={} owned_base_page={} pages_to_commit={}; leaving prior commits intact",
                request_id, target, chunk_depth, lcm, preflight_fail_reason, preflight_fail_group, fail_committed,
                fail_raw_cursor, fail_raw_per_page, fail_base, fail_borrowed, fail_owned, fail_committed_page,
                fail_owned_base_page, fail_pages_to_commit);
            break;
        }

        if (attach_node->HasPagedCacheSnapshot()) {
            PagedCacheSnapshot* existing = attach_node->GetPagedCacheSnapshotMut();
            const bool adopted = existing != nullptr && adoptExistingPagedCacheSnapshot(*existing, tables, target);
            if (!adopted) {
                spdlog::warn(
                    "[HybridPrefixCache] CommitChunk: existing snapshot adoption failed for request {} at target "
                    "depth {}; leaving existing snapshot intact",
                    request_id, target);
                break;
            }
            RefreshPagedCacheSnapshotCompleteness(*existing);
            last_committed = target;
            continue;
        }

        auto snapshot = std::make_unique<PagedCacheSnapshot>();
        snapshot->prefix_len_tokens = target;
        for (const auto& gid : required_groups) {
            auto& table = tables.find(gid)->second;
            auto group_alloc_it = paged_cache_allocators_.find(gid);
            const auto& cfg = group_alloc_it->second->Config();
            auto result = cfg.family == PagedCacheGroupFamily::History ? table.CommitHistoryToSnapshot(target)
                                                                       : table.CheckpointStateToSnapshot(target);
            PagedCacheGroupSnapshot group_snap{};
            group_snap.pages = std::move(result.pages);
            group_snap.base_logical_page = result.segment_base_logical_page;
            group_snap.raw_token_cursor = target;
            group_snap.sliding = table.IsSliding();
            snapshot->groups.emplace(gid, std::move(group_snap));
        }

        bool snapshot_complete = true;
        for (const auto& gid : required_groups) {
            if (snapshot->groups.find(gid) == snapshot->groups.end()) {
                snapshot_complete = false;
                break;
            }
        }
        _assert(snapshot_complete,
                "HybridPrefixCache::CommitChunk: built snapshot missing a required group after "
                "preflight+commit; invariant violated");
        const bool attached = AttachPagedCacheSnapshotToNode(attach_node, std::move(snapshot));
        _assert(attached,
                "HybridPrefixCache::CommitChunk: attach rejected a non-null snapshot on a non-null "
                "node; invariant violated");

        last_committed = target;
    }

    const bool terminal_state_committed = commitTerminalContinuationSnapshot(tables, terminal, chunk_depth);

    // Release superseded interior continuation-state snapshots.
    //
    // A continuation-state restore resumes from the deepest matching terminal
    // (Match). Each turn's terminal becomes an interior ancestor on the next
    // turn, but nothing released its now-superseded trailing-window state
    // snapshot, so these pinned pages accumulate one window per turn and
    // exhaust the small State pools (e.g. v4.c128a.compressor_state). Release an
    // ancestor's State portion (keeping its History chain) only when it is
    // provably unreferenced, which requires BOTH:
    //   (1) the owning request's sliding window has advanced past the ancestor
    //       (node_depth + window <= chunk_depth), so ReleaseSkipped has already
    //       dropped those pages from this request's own borrowed set; and
    //   (2) no request table still borrows the ancestor's State pages.
    //       Device().RefCount() == 1 means this committing request is the sole
    //       active device referencer, but its own table must still be checked:
    //       an already-existing terminal snapshot can skip State adoption and
    //       leave an older borrow alive.
    // Gate on a complete terminal snapshot so a resume anchor always remains.
    if (terminal_state_committed) {
        std::int32_t max_state_window = 0;
        for (const auto& gid : paged_cache_continuation_state_groups_) {
            auto alloc_it = paged_cache_allocators_.find(gid);
            if (alloc_it != paged_cache_allocators_.end() && alloc_it->second != nullptr) {
                max_state_window =
                    std::max(max_state_window, alloc_it->second->Config().sliding_window_tokens.value_or(0));
            }
        }
        for (TreeNode* cur = terminal->Parent(); cur != nullptr && !cur->IsRoot(); cur = cur->Parent()) {
            if (!cur->HasPagedCacheSnapshot()) continue;
            if (static_cast<std::int32_t>(cur->DepthInTokens()) + max_state_window > chunk_depth) {
                continue;
            }
            if (!cur->OnDevice() || cur->Device().RefCount() != 1) continue;
            if (isPagedCacheSnapshotBorrowed(cur, PagedCacheGroupFamily::State)) continue;
            DetachStateSnapshotFromNode(cur);
        }
    }
}

}  // namespace tokenspeed
