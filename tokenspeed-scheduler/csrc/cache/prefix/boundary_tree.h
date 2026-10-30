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
#include <unordered_map>
#include <vector>

#include "cache/core/cache_types.h"
#include "utils.h"

namespace tokenspeed {

// The tree of published prefix boundaries, shared by both tiers: one node per
// content hash some tier indexes or an eviction write-back is bringing to the
// Host, linked to the chain hash of the page before it (a tail to its prior;
// page 0 has none). It answers what the flat per-group indexes cannot: which
// pages a checkpoint boundary (a boundary with a State checkpoint) depends on,
// how many checkpoint boundaries depend on a page, and which Host pages end a
// chain. Walks follow parent pointers and the insertion-ordered child and leaf
// lists, never the map, so mirrored ranks agree. The coordinator keeps it in
// step with its indexes; no match reads it.
class BoundaryTree {
public:
    // Indexes into an array by CacheTier, such as a Node's per-tier fields.
    static constexpr std::size_t kDevice = static_cast<std::size_t>(CacheTier::kDevice);
    static constexpr std::size_t kHost = static_cast<std::size_t>(CacheTier::kHost);

    struct Node {
        // Whether a checkpoint boundary is at or under this node on either
        // tier.
        bool HasCheckpointBoundaryInSubtree() const {
            return checkpoint_boundaries_in_subtree[kDevice] > 0 || checkpoint_boundaries_in_subtree[kHost] > 0;
        }

        const ContentHash* hash{nullptr};  // the map key
        // Null for page 0, and for a node started without a known parent until
        // a publication through its hash links it (FindOrAdd).
        Node* parent{nullptr};
        // The children in insertion order, linked through the siblings:
        // appending and unlinking take constant time.
        Node* first_child{nullptr};
        Node* last_child{nullptr};
        Node* prev_sibling{nullptr};
        Node* next_sibling{nullptr};
        // The boundary T in tokens and T mod P (0 for a chain hash); both 0
        // for a node started without a known parent until a publication
        // through its hash gives it its boundary (FindOrAdd).
        std::int32_t boundary_tokens{0};
        std::int32_t tail_tokens{0};
        // One request owns each access epoch. Checkpoints acquired or
        // published by that request follow the same token history. Keep the
        // latest State access while either tier still holds this boundary.
        std::uint64_t latest_state_access_epoch{0};
        // By CacheTier: whether this boundary has a checkpoint on that tier
        // (some State group holds its checkpoint there; on the Host also one on
        // its way there), and how many boundaries in the subtree, this node
        // included, have a checkpoint on that tier (a boundary counts once,
        // however many State groups hold one there).
        std::array<bool, 2> has_checkpoint{};
        std::array<std::int32_t, 2> checkpoint_boundaries_in_subtree{};
        // Whether some group holds this hash on the Device tier.
        bool on_device{false};
        // Whether some group holds this hash on the Host tier or a store is
        // bringing one there, and how many children are on the Host in that
        // sense (children_on_host). A Host leaf -- on the Host, no child on the
        // Host -- is in the Host leaf list, in the order it became one.
        bool on_host{false};
        std::int32_t children_on_host{0};
        Node* prev_host_leaf{nullptr};
        Node* next_host_leaf{nullptr};
        // The blocks of this hash an eviction write-back has recorded and not
        // settled yet, and how many of them are State checkpoints: while any
        // is out the node, its tail hint and its Host presence stay.
        std::int32_t pending_host_stores{0};
        std::int32_t pending_host_checkpoint_stores{0};
        // Scratch for one duplicate-candidate scan. A new scan must recheck
        // stores and links that may have changed since the previous one.
        mutable std::uint64_t host_store_scan_generation{0};
        mutable bool host_store_ancestor_pending{false};
        // Accounting only, bit i for cache group i: an eviction write-back
        // published this hash's blocks of that group on the Host before, so
        // copying one again is a recopy.
        std::uint64_t written_back_group_bits{0};
        // The coordinator checks the closed groups at this page. The tree
        // carries that coverage through parents so duplicate reclaim does not
        // repeatedly prove the same Device prefix from its root.
        bool device_kv_page_complete{false};
        bool device_kv_prefix_complete{false};
        bool has_host_kv_duplicate{false};
        Node* prev_host_kv_duplicate{nullptr};
        Node* next_host_kv_duplicate{nullptr};
    };

    BoundaryTree() = default;
    // The nodes point at each other and at their map keys: a copy would point
    // into the original.
    BoundaryTree(const BoundaryTree&) = delete;
    BoundaryTree& operator=(const BoundaryTree&) = delete;
    BoundaryTree(BoundaryTree&&) = default;
    BoundaryTree& operator=(BoundaryTree&&) = default;

    // The node of `hash`. A new one is the last child of `parent` (null:
    // none); an existing one keeps its links and boundary, unless it was
    // started without a known parent: a publication through its hash gives
    // it its boundary and, past page 0, a parent, whose last child it becomes,
    // its subtree's checkpoint boundaries counted in every new ancestor.
    Node& FindOrAdd(const ContentHash& hash, Node* parent, std::int32_t boundary_tokens, std::int32_t tail_tokens) {
        auto [it, inserted] = nodes_.try_emplace(hash);
        Node& node = it->second;
        if (!inserted && (node.parent != nullptr || parent == nullptr)) {
            // Every real boundary is at least 1: only a node started without
            // a known parent has 0, and a registration passes 0.
            if (node.boundary_tokens == 0 && boundary_tokens > 0) {
                node.boundary_tokens = boundary_tokens;
                node.tail_tokens = tail_tokens;
            }
            return node;
        }
        node.hash = &it->first;
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
                if (node.on_host) {
                    const bool parent_was_host_leaf = isHostLeaf(*parent);
                    ++parent->children_on_host;
                    updateHostLeafList(*parent, parent_was_host_leaf);
                }
            }
        }
        return node;
    }

    Node* Find(const ContentHash& hash) {
        const auto it = nodes_.find(hash);
        return it == nodes_.end() ? nullptr : &it->second;
    }
    const Node* Find(const ContentHash& hash) const {
        const auto it = nodes_.find(hash);
        return it == nodes_.end() ? nullptr : &it->second;
    }

    // Marks whether `node` has a checkpoint on `tier` and counts the change in
    // every ancestor. A checkpoint boundary counts at most once, so setting
    // what is already set changes nothing. Returns whether the mark changed.
    bool SetCheckpoint(Node& node, CacheTier tier, bool has_checkpoint) {
        const auto tier_index = static_cast<std::size_t>(tier);
        if (node.has_checkpoint[tier_index] == has_checkpoint) {
            return false;
        }
        node.has_checkpoint[tier_index] = has_checkpoint;
        for (Node* ancestor = &node; ancestor != nullptr; ancestor = ancestor->parent) {
            ancestor->checkpoint_boundaries_in_subtree[tier_index] += has_checkpoint ? 1 : -1;
            _assert(ancestor->checkpoint_boundaries_in_subtree[tier_index] >= 0,
                    "a subtree holds a negative number of checkpoint boundaries");
        }
        return true;
    }

    // Marks whether some group holds `node`'s hash on the Host tier, keeping
    // the parent's children_on_host and the Host leaf list.
    void SetOnHost(Node& node, bool on_host) {
        if (node.on_host == on_host) {
            return;
        }
        const bool was_host_leaf = isHostLeaf(node);
        node.on_host = on_host;
        updateHostLeafList(node, was_host_leaf);
        if (node.parent != nullptr) {
            Node& parent = *node.parent;
            const bool parent_was_host_leaf = isHostLeaf(parent);
            parent.children_on_host += on_host ? 1 : -1;
            _assert(parent.children_on_host >= 0, "a node has a negative number of children on the Host");
            updateHostLeafList(parent, parent_was_host_leaf);
        }
    }

    // The oldest Host leaf; next_host_leaf walks the rest.
    const Node* FirstHostLeaf() const { return first_host_leaf_; }

    void SetDeviceKvCoverage(Node& node, bool page_complete, std::int32_t prefix_granularity) {
        node.device_kv_page_complete = page_complete;
        const auto prefix_complete = [&](const Node& page) {
            return page.device_kv_page_complete && page.tail_tokens == 0 &&
                   page.boundary_tokens ==
                       (page.parent == nullptr ? 0 : page.parent->boundary_tokens) + prefix_granularity &&
                   (page.parent == nullptr || page.parent->device_kv_prefix_complete);
        };
        if (node.device_kv_prefix_complete == prefix_complete(node)) {
            return;
        }
        std::vector<Node*> pending{&node};
        while (!pending.empty()) {
            Node& page = *pending.back();
            pending.pop_back();
            const bool complete = prefix_complete(page);
            if (page.device_kv_prefix_complete == complete) {
                continue;
            }
            page.device_kv_prefix_complete = complete;
            for (Node* child = page.first_child; child != nullptr; child = child->next_sibling) {
                if (child->device_kv_page_complete) {
                    pending.push_back(child);
                }
            }
        }
    }

    // Nodes with at least one settled closed-group key on both tiers, in
    // insertion order. Membership does not pin a block or imply a complete
    // Device prefix; the coordinator checks both when planning reclamation.
    void SetHostKvDuplicate(Node& node, bool duplicate) {
        if (node.has_host_kv_duplicate == duplicate) {
            return;
        }
        node.has_host_kv_duplicate = duplicate;
        if (duplicate) {
            node.prev_host_kv_duplicate = last_host_kv_duplicate_;
            (last_host_kv_duplicate_ != nullptr ? last_host_kv_duplicate_->next_host_kv_duplicate
                                                : first_host_kv_duplicate_) = &node;
            last_host_kv_duplicate_ = &node;
            return;
        }
        (node.prev_host_kv_duplicate != nullptr ? node.prev_host_kv_duplicate->next_host_kv_duplicate
                                                : first_host_kv_duplicate_) = node.next_host_kv_duplicate;
        (node.next_host_kv_duplicate != nullptr ? node.next_host_kv_duplicate->prev_host_kv_duplicate
                                                : last_host_kv_duplicate_) = node.prev_host_kv_duplicate;
        node.prev_host_kv_duplicate = nullptr;
        node.next_host_kv_duplicate = nullptr;
    }

    const Node* FirstHostKvDuplicate() const { return first_host_kv_duplicate_; }

    // Drops a childless node that holds nothing: no checkpoint on either tier,
    // nothing on either tier and no store on its way to the Host.
    void Remove(Node& node) {
        _assert(node.first_child == nullptr && !node.on_device && !node.on_host &&
                    !node.HasCheckpointBoundaryInSubtree() && node.pending_host_stores == 0,
                "only a childless node that holds nothing leaves the tree");
        _assert(!node.has_host_kv_duplicate, "a removed boundary cannot retain a Host KV duplicate");
        if (node.parent != nullptr) {
            Node& parent = *node.parent;
            (node.prev_sibling != nullptr ? node.prev_sibling->next_sibling : parent.first_child) = node.next_sibling;
            (node.next_sibling != nullptr ? node.next_sibling->prev_sibling : parent.last_child) = node.prev_sibling;
        }
        // Erase by iterator: the node's own key must not be read after the
        // node is gone.
        nodes_.erase(nodes_.find(*node.hash));
    }

    void Clear() {
        nodes_.clear();
        first_host_leaf_ = nullptr;
        last_host_leaf_ = nullptr;
        first_host_kv_duplicate_ = nullptr;
        last_host_kv_duplicate_ = nullptr;
    }

    std::size_t Size() const { return nodes_.size(); }

private:
    static bool isHostLeaf(const Node& node) { return node.on_host && node.children_on_host == 0; }

    // Appends `node` to the Host leaf list when it just became a Host leaf,
    // unlinks it when it just stopped being one.
    void updateHostLeafList(Node& node, bool was_host_leaf) {
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

    // Node-based, so nodes and keys keep their addresses across rehashes.
    std::unordered_map<ContentHash, Node> nodes_;
    Node* first_host_leaf_{nullptr};
    Node* last_host_leaf_{nullptr};
    Node* first_host_kv_duplicate_{nullptr};
    Node* last_host_kv_duplicate_{nullptr};
};

}  // namespace tokenspeed
