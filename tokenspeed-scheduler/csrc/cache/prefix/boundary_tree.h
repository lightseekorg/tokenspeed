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

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <list>
#include <memory_resource>
#include <optional>
#include <span>
#include <string>
#include <utility>

#include "cache/core/cache_types.h"
#include "cache/core/fixed_slot_resource.h"
#include "cache/core/prefetch.h"
#include "cache/prefix/checkpoint_tree.h"
#include "cache/prefix/flat_pointer_index.h"
#include "cache/prefix/tail_table.h"
#include "utils.h"

namespace tokenspeed {

// The tree of published prefix boundaries, shared by both tiers: one node per
// content hash some tier indexes or an eviction write-back is bringing to the
// Host, linked to the chain hash of the page before it (a tail to its prior;
// page 0 has none). It answers what the flat per-group indexes cannot: which
// pages a checkpoint boundary (a boundary with a State checkpoint) depends on,
// how many checkpoint boundaries depend on a page, and which Host pages end a
// chain. Walks follow parent pointers and the insertion-ordered child and leaf
// lists, never the map, so mirrored ranks agree. No match reads it.
//
// What a node knows about the indexes comes from what holds it (Hold): every
// entry of a tracked index tier, every eviction write-back on its way to the
// Host and every publication in progress holds the node of its hash. The node
// counts its holders, its presence and checkpoint marks follow from the
// counts, and it leaves the tree with its last holder once no child hangs
// under it, taking its tail hint along. Each change of a State checkpoint's
// residency reaches the checkpoint tree the same way.
class BoundaryTree {
public:
    // Indexes into an array by CacheTier, such as a Node's per-tier fields.
    static constexpr std::size_t kDevice = static_cast<std::size_t>(CacheTier::kDevice);
    static constexpr std::size_t kHost = static_cast<std::size_t>(CacheTier::kHost);

    // The coordinator's geometry and policy, fixed for its lifetime.
    struct Config {
        // P: the tokens of one prefix page.
        std::int32_t prefix_granularity;
        // The blocks one page holds across the prefix-closed groups: the
        // Device holds a page whole once it indexes that many of its
        // prefix-closed blocks.
        std::int32_t closed_blocks_per_page;
        // With any State group, a boundary serves a resume only with a
        // checkpoint boundary at or under it; a checkpoint is complete on a
        // tier when every State group holds it there.
        std::int32_t num_state_groups;
        // A prefix-closed group exists: a page no such group holds on either
        // tier is then a gap, at and under which nothing resumes.
        bool has_prefix_closed_groups;
        // A Host tier exists.
        bool has_host_tier;
        // Queue the nodes that may have died for the coordinator's collection
        // (write_back_evictions).
        bool queue_dead_candidates;
    };

    // What an index entry stands for in its node: its tier and the kind of its
    // group.
    struct EntryKind {
        CacheTier tier;
        // A State group's checkpoint.
        bool state;
        // A prefix-closed group's block.
        bool closed;
    };

    class Hold;

    struct Node {
        Node(const ContentHash& hash, BoundaryTree& tree) : hash{hash}, tree_{&tree} {}

        // Some group indexes this hash on the Device tier.
        bool OnDevice() const noexcept { return entries_[kDevice] > 0; }
        // Some group indexes this hash on the Host tier, or a store is
        // bringing one there.
        bool OnHost() const noexcept { return entries_[kHost] > 0 || pending_stores_ > 0; }
        // Some State group holds this boundary's checkpoint on `tier` (on the
        // Host, also one on its way there).
        bool HasCheckpoint(std::size_t tier) const noexcept {
            return state_entries_[tier] > 0 || (tier == kHost && pending_checkpoint_stores_ > 0);
        }
        // Whether a checkpoint boundary is at or under this node on either
        // tier.
        bool HasCheckpointBoundaryInSubtree() const noexcept {
            return checkpoint_boundaries_in_subtree[kDevice] > 0 || checkpoint_boundaries_in_subtree[kHost] > 0;
        }
        // Some prefix-closed group indexes a block of this hash on both tiers.
        bool HasHostKvDuplicate() const noexcept { return closed_pairs_ > 0; }
        // The entries every group indexes under this hash on `tier`, and those
        // of the State and of the prefix-closed groups among them.
        std::int32_t Entries(std::size_t tier) const noexcept { return entries_[tier]; }
        std::int32_t StateEntries(std::size_t tier) const noexcept { return state_entries_[tier]; }
        std::int32_t ClosedEntries(std::size_t tier) const noexcept { return closed_entries_[tier]; }
        // The prefix-closed blocks of this hash indexed on both tiers.
        std::int32_t ClosedPairs() const noexcept { return closed_pairs_; }
        // The blocks of this hash an eviction write-back has recorded and not
        // settled yet.
        std::int32_t PendingHostStores() const noexcept { return pending_stores_; }

        // The node's key in the tree's index, fixed for its lifetime.
        const ContentHash hash;
        // Null for page 0, and for a node started without a known parent until
        // a publication through its hash links it (LinkPublication).
        Node* parent{nullptr};
        // The children in insertion order, linked through the siblings:
        // appending and unlinking take constant time. Roots use the same
        // sibling links in the tree's root list.
        Node* first_child{nullptr};
        Node* last_child{nullptr};
        Node* prev_sibling{nullptr};
        Node* next_sibling{nullptr};
        // The boundary T in tokens and T mod P (0 for a chain hash); both 0
        // for a node started without a known parent until a publication
        // through its hash gives it its boundary (LinkPublication).
        std::int32_t boundary_tokens{0};
        std::int32_t tail_tokens{0};
        // One request owns each access epoch. Checkpoints acquired or
        // published by that request follow the same token history. Keep the
        // latest State access while either tier still holds this boundary.
        std::uint64_t latest_state_access_epoch{0};
        // By CacheTier: how many boundaries in the subtree, this node
        // included, have a checkpoint on that tier (a boundary counts once,
        // however many State groups hold one there).
        std::array<std::int32_t, 2> checkpoint_boundaries_in_subtree{};
        // How many children are on the Host (OnHost). A Host leaf -- on the
        // Host, no child on the Host -- is in the Host leaf list, in the order
        // it became one.
        std::int32_t children_on_host{0};
        Node* prev_host_leaf{nullptr};
        Node* next_host_leaf{nullptr};
        // Accounting only, bit i for cache group i: an eviction write-back
        // published this hash's blocks of that group on the Host before, so
        // copying one again is a recopy.
        std::uint64_t written_back_group_bits{0};

    private:
        friend class BoundaryTree;
        friend class Hold;

        // What holds the node, written only through its holds: index entries
        // by tier, and those of State and of prefix-closed groups among them;
        // prefix-closed blocks indexed on both tiers; eviction write-backs, and
        // the State checkpoints among them; publications in progress.
        std::array<std::int32_t, 2> entries_{};
        std::array<std::int32_t, 2> state_entries_{};
        std::array<std::int32_t, 2> closed_entries_{};
        std::int32_t closed_pairs_{0};
        std::int32_t pending_stores_{0};
        std::int32_t pending_checkpoint_stores_{0};
        std::int32_t publications_{0};
        BoundaryTree* tree_;
    };

    // One holder's claim on one node: while it lives the node stays, and what
    // it stands for -- an index entry, an eviction write-back on its way to the
    // Host, a publication in progress -- counts in the node. Move-only; an
    // empty hold claims nothing. The tree must outlive it. Dropping it may take
    // the node out of the tree, and with it each parent left with nothing.
    class Hold {
    public:
        Hold() noexcept = default;
        Hold(const Hold&) = delete;
        Hold& operator=(const Hold&) = delete;
        Hold(Hold&& other) noexcept;
        Hold& operator=(Hold&& other) noexcept;
        ~Hold() { reset(); }

        explicit operator bool() const noexcept { return node_ != nullptr; }
        // The held node; the hold must not be empty.
        Node& node() const noexcept { return *node_; }
        // For a prefix-closed group's entry: whether the other tier indexes
        // the same key too (PrefixCacheIndex's common entries). Ignored by an
        // empty hold.
        void SetPaired(bool paired) noexcept;
        // Drops the claim now and leaves the hold empty.
        void reset() noexcept;

    private:
        friend class BoundaryTree;
        enum class Kind : std::uint8_t { kEntry, kPendingStore, kPublication };

        Hold(Node& node, Kind kind, std::size_t tier, bool state, bool closed) noexcept
            : node_{&node}, kind_{kind}, tier_{static_cast<std::uint8_t>(tier)}, state_{state}, closed_{closed} {}

        Node* node_{nullptr};
        Kind kind_{Kind::kEntry};
        std::uint8_t tier_{0};
        // An entry of a State group, or the write-back of a State checkpoint.
        bool state_{false};
        // An entry of a prefix-closed group.
        bool closed_{false};
        // Counted among the node's prefix-closed blocks on both tiers.
        bool paired_{false};
    };

    // `checkpoints` (null: none) receives every change of a State
    // checkpoint's residency; it must outlive every hold.
    BoundaryTree(Config config, CheckpointTree* checkpoints) : config_{config}, checkpoints_{checkpoints} {}
    // Nodes and holds point at the tree, and nodes and the index at each
    // other's storage: a copy or a move would leave them pointing at the
    // original.
    BoundaryTree(const BoundaryTree&) = delete;
    BoundaryTree& operator=(const BoundaryTree&) = delete;
    BoundaryTree(BoundaryTree&&) = delete;
    BoundaryTree& operator=(BoundaryTree&&) = delete;

    // Holds the node of an index entry's hash for as long as the entry lives.
    // A hash no node has yet starts one without a parent, until a publication
    // through it gives it its boundary (and, past page 0, links it).
    Hold HoldEntry(const ContentHash& hash, EntryKind kind) {
        return acquire(findOrAdd(hash, /*parent=*/nullptr, /*boundary_tokens=*/0, /*tail_tokens=*/0),
                       Hold::Kind::kEntry, static_cast<std::size_t>(kind.tier), kind.state, kind.closed);
    }

    // Holds `node` for an eviction write-back of one of its blocks (a State
    // checkpoint when `checkpoint`) until the write-back settles: the node, its
    // tail hint and its Host presence stay meanwhile.
    Hold HoldPendingStore(Node& node, bool checkpoint) {
        return acquire(node, Hold::Kind::kPendingStore, kHost, checkpoint, /*closed=*/false);
    }

    // Adds the nodes of a publication before any group registers under them,
    // so a checkpoint counts in every ancestor as it is published: one per
    // newly hashed page under the page before it, and the tail under the last
    // page. A page below them whose node is gone (a model without a
    // prefix-closed group keeps no entry for some pages) gets one again, so
    // the tree stays whole; a node that exists keeps its links, but one
    // started without a known parent gets its boundary and, past page 0, is
    // linked under the page before it. Returns a hold on the deepest node it
    // links, kept while the groups register: dropping it takes that node away
    // when none registered under it (a chunk end of State groups alone).
    Hold LinkPublication(const CompletedPages& completed) {
        const std::span<const std::string> page_hashes = completed.prefix_hashes;
        // The publisher's table holds the pages below the new ones, so a
        // prefix-closed group keeps their nodes and the first lookup finds the
        // parent. A node an index entry started without a parent (any page
        // but page 0's) is not taken as the anchor: the search goes on upward,
        // and the loop below links it under the page before it.
        Node* parent = nullptr;
        auto first_page_to_link = static_cast<std::size_t>(completed.first_new_prefix_page);
        for (; first_page_to_link > 0; --first_page_to_link) {
            parent = Find(ContentHash{page_hashes[first_page_to_link - 1]});
            if (parent != nullptr && (parent->parent != nullptr || first_page_to_link == 1)) {
                break;
            }
        }
        for (std::size_t page = first_page_to_link; page < page_hashes.size(); ++page) {
            parent = &findOrAdd(ContentHash{page_hashes[page]}, parent,
                                static_cast<std::int32_t>(page + 1) * config_.prefix_granularity, /*tail_tokens=*/0);
        }
        Node* deepest = parent;
        if (completed.tail) {
            deepest = &findOrAdd(completed.tail->hash, parent,
                                 static_cast<std::int32_t>(page_hashes.size()) * config_.prefix_granularity +
                                     completed.tail->tail_tokens,
                                 completed.tail->tail_tokens);
        }
        _assert(deepest != nullptr, "a publication links at least one boundary");
        return acquire(*deepest, Hold::Kind::kPublication, kDevice, /*state=*/false, /*closed=*/false);
    }

    // A hint for dropping `hold`, a hold on this tree's node of `hash`: the
    // holder passes the hash, so neither the node nor an index slot is read
    // here. Prefetches the node and the index slot of its hash, which taking
    // the node out of the tree reads.
    void PrefetchRelease(const Hold& hold, const ContentHash& hash) const noexcept {
        if (hold.node_ == nullptr) return;
        PrefetchObject(*hold.node_);
        nodes_.Prefetch(hash);
    }

    Node* Find(const ContentHash& hash) {
        const auto it = nodes_.Find(hash);
        return it == node_storage_.end() ? nullptr : &*it;
    }
    const Node* Find(const ContentHash& hash) const {
        const auto it = nodes_.Find(hash);
        return it == node_storage_.end() ? nullptr : &*it;
    }

    // The oldest root; next_sibling walks the rest.
    const Node* FirstRoot() const { return first_root_; }
    // The oldest Host leaf; next_host_leaf walks the rest.
    const Node* FirstHostLeaf() const { return first_host_leaf_; }

    // Whether the Device holds `node`'s page whole -- every prefix-closed block
    // of a chain hash -- and the page continues its parent's boundary (page 0:
    // no parent). The Device prefix up to a page is complete when this holds
    // for the page and every page above it.
    bool DeviceKvPageExtendsPrefix(const Node& node) const noexcept {
        return config_.has_prefix_closed_groups && node.tail_tokens == 0 &&
               node.closed_entries_[kDevice] == config_.closed_blocks_per_page &&
               node.boundary_tokens ==
                   (node.parent == nullptr ? 0 : node.parent->boundary_tokens) + config_.prefix_granularity;
    }

    // A State checkpoint's residency as its node's counts give it: complete on
    // a tier when every State group holds it there, held while some State
    // entry or a checkpoint's store on its way to the Host holds it.
    CheckpointTree::Residency CheckpointResidency(const Node& node) const noexcept {
        const bool has_state = config_.num_state_groups > 0;
        return CheckpointTree::Residency{
            .complete = {has_state && node.state_entries_[kDevice] == config_.num_state_groups,
                         has_state && config_.has_host_tier && node.state_entries_[kHost] == config_.num_state_groups},
            .pending_host_store = node.pending_checkpoint_stores_ > 0,
            .held = node.state_entries_[kDevice] > 0 || node.state_entries_[kHost] > 0 ||
                    node.pending_checkpoint_stores_ > 0,
        };
    }

    // Only a complete Device prefix without a pending store at or above it can
    // replace a Host copy. Prune whole blocked subtrees and stop at the first
    // possible duplicate; pins and checkpoint windows are checked by the
    // coordinator. The walk enters a node only under a parent it found
    // covered, so the node's own page decides there.
    bool HasHostKvDuplicateCandidate() const {
        if (host_kv_duplicates_ == 0) {
            return false;
        }
        const Node* node = first_root_;
        while (node != nullptr) {
            if (DeviceKvPageExtendsPrefix(*node) && node->pending_stores_ == 0) {
                if (node->HasHostKvDuplicate()) {
                    return true;
                }
                if (node->first_child != nullptr) {
                    node = node->first_child;
                    continue;
                }
            }
            while (node != nullptr && node->next_sibling == nullptr) {
                node = node->parent;
            }
            if (node != nullptr) {
                node = node->next_sibling;
            }
        }
        return false;
    }

    std::size_t Size() const { return nodes_.size(); }

    // The published tail boundaries. A tail's hint leaves with its node.
    const TailTable& Tails() const noexcept { return tails_; }
    // TailTable::Add: the tail the per-prior cap dropped, if any.
    std::optional<TailTable::Tail> RecordTail(const ContentHash& prior, std::int32_t tail_tokens,
                                              const ContentHash& hash) {
        return tails_.Add(prior, tail_tokens, hash);
    }
    bool TouchTail(const ContentHash& hash) { return tails_.Touch(hash); }

    // Hashes of possibly dead nodes, in the order they were queued, for the
    // coordinator's collection (queue_dead_candidates only). Dropping a holder
    // queues the node its removal stops at once no checkpoint boundary is left
    // under it (the deepest such node left; the collection climbs from it),
    // and a node no prefix-closed group holds on either tier and no store is
    // bringing to the Host (a gap); the coordinator queues others. A queued
    // node may be gone or alive again by the time it is collected. A hash
    // already at the back is not queued twice in a row.
    void QueueDeadCandidate(const ContentHash& hash) {
        if (config_.queue_dead_candidates && nodes_.contains(hash)) {
            queue(hash);
        }
    }
    std::optional<ContentHash> PopDeadCandidate() {
        if (dead_candidates_.empty()) {
            return std::nullopt;
        }
        ContentHash hash = std::move(dead_candidates_.front());
        dead_candidates_.pop_front();
        return hash;
    }
    void ClearDeadCandidates() { dead_candidates_.clear(); }

    // After a cache clear dropped every holder: no node is left, and the tail
    // hints no node backs (a tail published with nothing registered under it)
    // and the queued candidates go too.
    void ClearAfterCacheClear() {
        _assert(nodes_.empty(), "a cleared cache leaves no boundary node behind");
        tails_.Clear();
        dead_candidates_.clear();
    }

private:
    static bool isHostLeaf(const Node& node) noexcept { return node.OnHost() && node.children_on_host == 0; }
    static bool unheld(const Node& node) noexcept {
        return node.entries_[kDevice] == 0 && node.entries_[kHost] == 0 && node.pending_stores_ == 0 &&
               node.publications_ == 0;
    }

    Hold acquire(Node& node, Hold::Kind kind, std::size_t tier, bool state, bool closed) {
        Hold hold{node, kind, tier, state, closed};
        apply(hold, 1);
        return hold;
    }

    // Counts `delta` (+1 or -1) of what `hold` stands for in its node and
    // carries every mark that flips on: a checkpoint into each ancestor's
    // count, Host presence into the parent's count and the Host leaf list,
    // duplicate membership into the tree's count. Returns whether a
    // checkpoint mark flipped.
    bool apply(const Hold& hold, std::int32_t delta) noexcept {
        Node& node = *hold.node_;
        if (hold.kind_ == Hold::Kind::kEntry && hold.tier_ == kDevice && !hold.state_ && !hold.paired_) {
            // A Device entry that is no State checkpoint and pairs with no
            // Host copy counts in no checkpoint, Host or duplicate mark: none
            // flips, and only its own counts change.
            node.entries_[kDevice] += delta;
            node.closed_entries_[kDevice] += hold.closed_ ? delta : 0;
            FatalCheck(node.entries_[kDevice] >= 0 && node.closed_entries_[kDevice] >= 0,
                       "a boundary node lost a holder it did not have");
            return false;
        }
        // Only a State group's entry or a checkpoint's store changes a
        // checkpoint's residency.
        const bool moves_checkpoint = hold.state_ && checkpoints_ != nullptr;
        const CheckpointTree::Residency had_residency =
            moves_checkpoint ? CheckpointResidency(node) : CheckpointTree::Residency{};
        const std::array<bool, 2> had_checkpoint{node.HasCheckpoint(kDevice), node.HasCheckpoint(kHost)};
        const bool was_on_host = node.OnHost();
        const bool was_host_leaf = isHostLeaf(node);
        const bool had_duplicate = node.HasHostKvDuplicate();
        switch (hold.kind_) {
            case Hold::Kind::kEntry:
                node.entries_[hold.tier_] += delta;
                node.state_entries_[hold.tier_] += hold.state_ ? delta : 0;
                node.closed_entries_[hold.tier_] += hold.closed_ ? delta : 0;
                node.closed_pairs_ += hold.paired_ ? delta : 0;
                break;
            case Hold::Kind::kPendingStore:
                node.pending_stores_ += delta;
                node.pending_checkpoint_stores_ += hold.state_ ? delta : 0;
                break;
            case Hold::Kind::kPublication:
                node.publications_ += delta;
                break;
        }
        FatalCheck(node.entries_[hold.tier_] >= 0 && node.state_entries_[hold.tier_] >= 0 &&
                       node.closed_entries_[hold.tier_] >= 0 && node.closed_pairs_ >= 0 && node.pending_stores_ >= 0 &&
                       node.pending_checkpoint_stores_ >= 0 && node.publications_ >= 0,
                   "a boundary node lost a holder it did not have");
        bool checkpoint_flipped = false;
        for (const std::size_t tier : {kDevice, kHost}) {
            if (node.HasCheckpoint(tier) != had_checkpoint[tier]) {
                countCheckpointBoundary(node, tier, had_checkpoint[tier] ? -1 : 1);
                checkpoint_flipped = true;
            }
        }
        if (node.OnHost() != was_on_host) {
            carryHostPresence(node, was_host_leaf);
        }
        if (node.HasHostKvDuplicate() != had_duplicate) {
            host_kv_duplicates_ += had_duplicate ? -1 : 1;
        }
        // A hold's release runs in destructors, so the CheckpointTree's checks
        // on this path are fatal checks, like the ones above.
        if (moves_checkpoint) {
            if (const CheckpointTree::Residency residency = CheckpointResidency(node); residency != had_residency) {
                checkpoints_->Update(node.hash, residency);
            }
        }
        return checkpoint_flipped;
    }

    // Drops what `hold` stands for. A node nothing holds and nothing hangs
    // under leaves, then each parent it leaves so. Under queue_dead_candidates
    // it queues the node the walk stops at when a checkpoint mark flipped or a
    // node left and no checkpoint boundary is under it any more, and the
    // released node when it is a gap.
    void release(Hold& hold) noexcept {
        Node& node = *hold.node_;
        bool checkpoint_flipped_or_node_removed = apply(hold, -1);
        hold.node_ = nullptr;
        hold.paired_ = false;
        bool is_gap = config_.has_prefix_closed_groups && node.closed_entries_[kDevice] == 0 &&
                      node.closed_entries_[kHost] == 0 && node.pending_stores_ == 0;
        Node* current = &node;
        while (unheld(*current) && current->first_child == nullptr) {
            Node* parent = current->parent;
            erase(*current);
            if (parent == nullptr) {
                return;
            }
            current = parent;
            checkpoint_flipped_or_node_removed = true;
            is_gap = false;
        }
        if (config_.queue_dead_candidates && ((checkpoint_flipped_or_node_removed && config_.num_state_groups > 0 &&
                                               !current->HasCheckpointBoundaryInSubtree()) ||
                                              is_gap)) {
            queue(current->hash);
        }
    }

    void changePairs(Node& node, std::int32_t delta) noexcept {
        const bool had_duplicate = node.HasHostKvDuplicate();
        node.closed_pairs_ += delta;
        FatalCheck(node.closed_pairs_ >= 0, "a boundary node lost a duplicate it did not have");
        if (node.HasHostKvDuplicate() != had_duplicate) {
            host_kv_duplicates_ += had_duplicate ? -1 : 1;
        }
    }

    // The node of `hash`. A new one is the last child of `parent` (null:
    // none); an existing one keeps its links and boundary, unless it was
    // started without a known parent: a publication through its hash gives
    // it its boundary and, past page 0, a parent, whose last child it becomes,
    // its subtree's checkpoint boundaries counted in every new ancestor.
    Node& findOrAdd(const ContentHash& hash, Node* parent, std::int32_t boundary_tokens, std::int32_t tail_tokens) {
        // A node is created with its hash before the index stores it; a
        // throw on the way leaves neither.
        const auto [it, inserted] =
            nodes_.FindOrInsert(hash, [&] { return node_storage_.emplace(node_storage_.end(), hash, *this); });
        Node& node = *it;
        if (!inserted && (node.parent != nullptr || parent == nullptr)) {
            // Every real boundary is at least 1: only a node started without
            // a known parent has 0, and a registration passes 0.
            if (node.boundary_tokens == 0 && boundary_tokens > 0) {
                node.boundary_tokens = boundary_tokens;
                node.tail_tokens = tail_tokens;
            }
            return node;
        }
        if (!inserted) {
            // An existing root is about to acquire its known parent.
            (node.prev_sibling != nullptr ? node.prev_sibling->next_sibling : first_root_) = node.next_sibling;
            (node.next_sibling != nullptr ? node.next_sibling->prev_sibling : last_root_) = node.prev_sibling;
            node.prev_sibling = nullptr;
            node.next_sibling = nullptr;
        }
        node.parent = parent;
        node.boundary_tokens = boundary_tokens;
        node.tail_tokens = tail_tokens;
        if (parent != nullptr) {
            node.prev_sibling = parent->last_child;
            (parent->last_child != nullptr ? parent->last_child->next_sibling : parent->first_child) = &node;
            parent->last_child = &node;
            // A new node has no checkpoint boundary under it and is not on the
            // Host, so linking it touches its parent alone. A node started
            // without a known parent may have both: its subtree's checkpoint
            // boundaries count in every new ancestor, and its Host presence in
            // the parent.
            if (!inserted) {
                for (Node* ancestor = parent; ancestor != nullptr; ancestor = ancestor->parent) {
                    ancestor->checkpoint_boundaries_in_subtree[kDevice] +=
                        node.checkpoint_boundaries_in_subtree[kDevice];
                    ancestor->checkpoint_boundaries_in_subtree[kHost] += node.checkpoint_boundaries_in_subtree[kHost];
                }
                if (node.OnHost()) {
                    const bool parent_was_host_leaf = isHostLeaf(*parent);
                    ++parent->children_on_host;
                    updateHostLeafList(*parent, parent_was_host_leaf);
                }
            }
        } else {
            node.prev_sibling = last_root_;
            (last_root_ != nullptr ? last_root_->next_sibling : first_root_) = &node;
            last_root_ = &node;
        }
        return node;
    }

    // A checkpoint boundary appeared (+1) or went (-1) at `node` on `tier`.
    void countCheckpointBoundary(Node& node, std::size_t tier, std::int32_t delta) noexcept {
        for (Node* ancestor = &node; ancestor != nullptr; ancestor = ancestor->parent) {
            ancestor->checkpoint_boundaries_in_subtree[tier] += delta;
            FatalCheck(ancestor->checkpoint_boundaries_in_subtree[tier] >= 0,
                       "a subtree holds a negative number of checkpoint boundaries");
        }
    }

    // `node`'s Host presence just flipped: it joins or leaves the Host leaf
    // list, and its parent counts it among its children on the Host.
    void carryHostPresence(Node& node, bool was_host_leaf) noexcept {
        updateHostLeafList(node, was_host_leaf);
        if (node.parent != nullptr) {
            Node& parent = *node.parent;
            const bool parent_was_host_leaf = isHostLeaf(parent);
            parent.children_on_host += node.OnHost() ? 1 : -1;
            FatalCheck(parent.children_on_host >= 0, "a node has a negative number of children on the Host");
            updateHostLeafList(parent, parent_was_host_leaf);
        }
    }

    // Appends `node` to the Host leaf list when it just became a Host leaf,
    // unlinks it when it just stopped being one.
    void updateHostLeafList(Node& node, bool was_host_leaf) noexcept {
        const bool is_host_leaf = isHostLeaf(node);
        if (is_host_leaf == was_host_leaf) {
            return;
        }
        if (is_host_leaf) {
            node.prev_host_leaf = last_host_leaf_;
            (last_host_leaf_ != nullptr ? last_host_leaf_->next_host_leaf : first_host_leaf_) = &node;
            last_host_leaf_ = &node;
            return;
        }
        (node.prev_host_leaf != nullptr ? node.prev_host_leaf->next_host_leaf : first_host_leaf_) = node.next_host_leaf;
        (node.next_host_leaf != nullptr ? node.next_host_leaf->prev_host_leaf : last_host_leaf_) = node.prev_host_leaf;
        node.prev_host_leaf = nullptr;
        node.next_host_leaf = nullptr;
    }

    // Takes a childless node nothing holds out of the tree, with its tail hint.
    void erase(Node& node) noexcept {
        FatalCheck(unheld(node) && node.first_child == nullptr && node.closed_pairs_ == 0 &&
                       !node.HasCheckpointBoundaryInSubtree(),
                   "only a childless node nothing holds leaves the tree");
        // Nothing hangs under a tail, so a tail hash leaves the tail table
        // exactly when nothing holds its node.
        tails_.Remove(node.hash);
        // A candidate the last release queued for this node leaves with it
        // (the walk queues what is left above), so taking a chain from below
        // queues one node, not one per page.
        if (!dead_candidates_.empty() && dead_candidates_.back() == node.hash) {
            dead_candidates_.pop_back();
        }
        if (node.parent != nullptr) {
            Node& parent = *node.parent;
            (node.prev_sibling != nullptr ? node.prev_sibling->next_sibling : parent.first_child) = node.next_sibling;
            (node.next_sibling != nullptr ? node.next_sibling->prev_sibling : parent.last_child) = node.prev_sibling;
        } else {
            (node.prev_sibling != nullptr ? node.prev_sibling->next_sibling : first_root_) = node.next_sibling;
            (node.next_sibling != nullptr ? node.next_sibling->prev_sibling : last_root_) = node.prev_sibling;
        }
        // Erase by iterator: the node's own key must not be read after the
        // node is gone.
        const auto entry = nodes_.ExtractObject(node.hash, node);
        FatalCheck(entry != node_storage_.end(), "a removed boundary keeps its index entry");
        node_storage_.erase(entry);
    }

    void queue(const ContentHash& hash) {
        if (dead_candidates_.empty() || dead_candidates_.back() != hash) {
            dead_candidates_.push_back(hash);
        }
    }

    Config config_;
    CheckpointTree* checkpoints_;
    // Node-based, so nodes, with the keys they own, keep their addresses
    // across rehashes.
    using NodeList = std::pmr::list<Node>;
    using NodeIterator = NodeList::iterator;
    struct NodeKey {
        const ContentHash& operator()(NodeIterator node) const noexcept { return node->hash; }
    };
    // Each node and its key live in stable pooled list storage. Index rehash and
    // backward-shift erase move handles only; Holds and traversal links stay
    // valid until their own node's ordinary last-holder removal.
    FixedSlotResource node_memory_{SlotShape::ListNodeOf<Node>(), 256, std::pmr::new_delete_resource()};
    NodeList node_storage_{&node_memory_};
    // Answers node_storage_.end() for an absent hash.
    FlatPointerIndex<NodeIterator, ContentHash, NodeKey, std::hash<ContentHash>> nodes_{node_storage_.end()};
    Node* first_root_{nullptr};
    Node* last_root_{nullptr};
    Node* first_host_leaf_{nullptr};
    Node* last_host_leaf_{nullptr};
    // How many nodes have HasHostKvDuplicate.
    std::int32_t host_kv_duplicates_{0};
    TailTable tails_;
    std::deque<ContentHash> dead_candidates_;
};

inline BoundaryTree::Hold::Hold(Hold&& other) noexcept
    : node_{std::exchange(other.node_, nullptr)},
      kind_{other.kind_},
      tier_{other.tier_},
      state_{other.state_},
      closed_{other.closed_},
      paired_{std::exchange(other.paired_, false)} {}

inline BoundaryTree::Hold& BoundaryTree::Hold::operator=(Hold&& other) noexcept {
    if (this != &other) {
        reset();
        node_ = std::exchange(other.node_, nullptr);
        kind_ = other.kind_;
        tier_ = other.tier_;
        state_ = other.state_;
        closed_ = other.closed_;
        paired_ = std::exchange(other.paired_, false);
    }
    return *this;
}

inline void BoundaryTree::Hold::SetPaired(bool paired) noexcept {
    if (node_ == nullptr || paired == paired_) {
        return;
    }
    FatalCheck(kind_ == Kind::kEntry && closed_, "only a prefix-closed group's entry pairs across tiers");
    paired_ = paired;
    node_->tree_->changePairs(*node_, paired ? 1 : -1);
}

inline void BoundaryTree::Hold::reset() noexcept {
    if (node_ != nullptr) {
        node_->tree_->release(*this);
    }
}

}  // namespace tokenspeed
