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

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cache/core/cache_types.h"
#include "cache/prefix/boundary_tree.h"

namespace tokenspeed::test {
namespace {

using Node = BoundaryTree::Node;
using CheckpointBoundaryCounts = std::array<std::int32_t, 2>;  // checkpoint_boundaries_in_subtree by CacheTier

constexpr std::int32_t kP = 4;
constexpr BoundaryTree::EntryKind kDeviceKv{.tier = CacheTier::kDevice, .state = false, .closed = true};
constexpr BoundaryTree::EntryKind kHostKv{.tier = CacheTier::kHost, .state = false, .closed = true};
constexpr BoundaryTree::EntryKind kDeviceCheckpoint{.tier = CacheTier::kDevice, .state = true, .closed = false};
constexpr BoundaryTree::EntryKind kHostCheckpoint{.tier = CacheTier::kHost, .state = true, .closed = false};
// A sliding window's block: neither a checkpoint nor prefix-closed.
constexpr BoundaryTree::EntryKind kDeviceWindow{.tier = CacheTier::kDevice, .state = false, .closed = false};

// A model with one State group, on a Device and a Host tier, and
// prefix-closed groups that keep closed_blocks_per_page blocks of each page.
BoundaryTree::Config TreeConfig(std::int32_t closed_blocks_per_page, bool queue_dead_candidates) {
    return BoundaryTree::Config{
        .prefix_granularity = kP,
        .closed_blocks_per_page = closed_blocks_per_page,
        .num_state_groups = 1,
        .has_prefix_closed_groups = true,
        .has_host_tier = true,
        .queue_dead_candidates = queue_dead_candidates,
    };
}

// Links what one publication of `pages` from `first_new_page` on, and `tail`,
// links; the hold keeps the deepest node.
BoundaryTree::Hold Publish(BoundaryTree& tree, const std::vector<std::string>& pages, std::int32_t first_new_page,
                           std::optional<TailPublication> tail) {
    return tree.LinkPublication(
        CompletedPages{.prefix_hashes = pages, .first_new_prefix_page = first_new_page, .tail = std::move(tail)});
}

const Node& NodeOf(const BoundaryTree& tree, const std::string& hash) {
    const Node* node = tree.Find(hash);
    EXPECT_NE(node, nullptr) << hash;
    return *node;
}

std::vector<std::string> Children(const Node& node) {
    std::vector<std::string> hashes;
    for (const Node* child = node.first_child; child != nullptr; child = child->next_sibling) {
        hashes.push_back(*child->hash);
    }
    return hashes;
}

std::vector<std::string> HostLeaves(const BoundaryTree& tree) {
    std::vector<std::string> hashes;
    for (const Node* leaf = tree.FirstHostLeaf(); leaf != nullptr; leaf = leaf->next_host_leaf) {
        hashes.push_back(*leaf->hash);
    }
    return hashes;
}

TEST(BoundaryTreeTest, APublicationLinksPagesUnderTheirParentsAndTheTailUnderItsPrior) {
    BoundaryTree tree{TreeConfig(/*closed_blocks_per_page=*/1, /*queue_dead_candidates=*/false),
                      /*checkpoints=*/nullptr};
    const std::vector<std::string> pages{"page0", "page1"};
    BoundaryTree::Hold publication = Publish(tree, pages, 0, TailPublication{.tail_tokens = 2, .hash = "tail"});
    const Node& page0 = NodeOf(tree, "page0");
    const Node& page1 = NodeOf(tree, "page1");
    const Node& tail = NodeOf(tree, "tail");

    EXPECT_EQ(&publication.node(), &tail) << "the publication holds its deepest node";
    EXPECT_EQ(tree.Size(), 3u);
    EXPECT_EQ(page0.parent, nullptr);
    EXPECT_EQ(page1.parent, &page0);
    EXPECT_EQ(tail.parent, &page1);
    EXPECT_EQ(page0.boundary_tokens, 4);
    EXPECT_EQ(page1.boundary_tokens, 8);
    EXPECT_EQ(page1.tail_tokens, 0);
    EXPECT_EQ(tail.boundary_tokens, 10);
    EXPECT_EQ(tail.tail_tokens, 2);
    EXPECT_EQ(Children(page0), std::vector<std::string>{"page1"});
    EXPECT_EQ(Children(page1), std::vector<std::string>{"tail"});
    EXPECT_TRUE(Children(tail).empty());
    EXPECT_EQ(tree.Find("absent"), nullptr);
    for (const Node* node : {&page0, &page1, &tail}) {
        EXPECT_EQ(node->checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 0}));
        EXPECT_FALSE(node->OnDevice());
        EXPECT_FALSE(node->OnHost());
    }

    // A hash published again keeps its node, links and boundary.
    BoundaryTree::Hold again = Publish(tree, {"page0"}, 0, TailPublication{.tail_tokens = 3, .hash = "tail"});
    EXPECT_EQ(&again.node(), &tail);
    EXPECT_EQ(tail.parent, &page1);
    EXPECT_EQ(tail.boundary_tokens, 10);
    EXPECT_EQ(tail.tail_tokens, 2);
    EXPECT_EQ(tree.Size(), 3u);

    // Nothing registered under the tail: once no publication holds it, the
    // chain leaves with it.
    again.reset();
    EXPECT_EQ(tree.Size(), 3u) << "the first publication still holds the tail";
    publication.reset();
    EXPECT_EQ(tree.Size(), 0u);
}

TEST(BoundaryTreeTest, NodesAndKeysStayStableAcrossIndexGrowthDeletionAndSlotReuse) {
    BoundaryTree tree{TreeConfig(1, false), nullptr};
    auto root_publication = Publish(tree, {"root"}, 0, std::nullopt);
    auto root_hold = tree.HoldEntry("root", kDeviceKv);
    root_publication.reset();
    const Node* root = tree.Find("root");
    const ContentHash* root_key = root->hash;
    std::vector<BoundaryTree::Hold> holds;
    for (int i = 0; i < 2048; ++i) {
        const auto hash = "child-" + std::to_string(i);
        auto publication = Publish(tree, {"root", hash}, 1, std::nullopt);
        holds.push_back(tree.HoldEntry(hash, kDeviceKv));
    }
    const Node* survivor = tree.Find("child-1");
    EXPECT_EQ(tree.Find("root"), root);
    EXPECT_EQ(root->hash, root_key);
    EXPECT_EQ(*root_key, ContentHash("root"));
    std::vector<std::string> expected;
    for (int i = 0; i < 2048; ++i) {
        if (i % 2 == 0)
            holds[i].reset();
        else
            expected.push_back("child-" + std::to_string(i));
    }
    EXPECT_EQ(Children(*root), expected);
    for (int i = 0; i < 2048; i += 2) {
        const auto hash = "child-" + std::to_string(i);
        EXPECT_EQ(tree.Find(hash), nullptr);
        auto publication = Publish(tree, {"root", hash}, 1, std::nullopt);
        holds[i] = tree.HoldEntry(hash, kDeviceKv);
        expected.push_back(hash);
    }
    EXPECT_EQ(tree.Find("child-1"), survivor);
    EXPECT_EQ(survivor->parent, root);
    EXPECT_EQ(Children(*root), expected);
    holds.clear();
    EXPECT_EQ(tree.Size(), 1);
    root_hold.reset();
    EXPECT_EQ(tree.Size(), 0);
}

// Every hash shares the eight leading bytes the node index hashes, all ones,
// so every node probes one cluster that wraps around the end of the table at
// each capacity. A leaving node takes out only its own slot: the rest keep
// their nodes and keys, down a chain and in any release order.
TEST(BoundaryTreeTest, NodesWithCollidingIndexHashesLeaveOneByOne) {
    BoundaryTree tree{TreeConfig(1, false), nullptr};
    const auto colliding = [](int i) {
        std::string hex = std::string(16, 'f') + std::string(48, '0');
        hex[62] = "0123456789abcdef"[i / 16];
        hex[63] = "0123456789abcdef"[i % 16];
        return hex;
    };
    std::vector<std::string> chain;
    std::vector<BoundaryTree::Hold> chain_holds;
    for (int i = 0; i < 40; ++i) chain.push_back(colliding(i));
    {
        BoundaryTree::Hold publication = Publish(tree, chain, 0, std::nullopt);
        for (const auto& hash : chain) chain_holds.push_back(tree.HoldEntry(hash, kDeviceKv));
    }
    std::map<std::string, const Node*> roots;
    std::vector<BoundaryTree::Hold> root_holds;
    for (int i = 40; i < 120; ++i) {
        const auto hash = colliding(i);
        BoundaryTree::Hold publication = Publish(tree, {hash}, 0, std::nullopt);
        root_holds.push_back(tree.HoldEntry(hash, kDeviceKv));
        roots.emplace(hash, tree.Find(hash));
    }
    ASSERT_EQ(tree.Size(), 120u);
    const auto expect_roots = [&] {
        for (const auto& [hash, node] : roots) {
            EXPECT_EQ(tree.Find(hash), node);
            EXPECT_EQ(*node->hash, ContentHash(hash));
        }
    };

    // Each page stays under its child until the leaf leaves and takes the
    // whole chain with it.
    for (std::size_t i = 0; i + 1 < chain.size(); ++i) {
        chain_holds[i].reset();
        EXPECT_EQ(tree.Size(), 120u);
    }
    chain_holds.back().reset();
    EXPECT_EQ(tree.Size(), 80u);
    for (const auto& hash : chain) EXPECT_EQ(tree.Find(hash), nullptr);
    expect_roots();

    std::vector<int> order(root_holds.size());
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), std::mt19937(77));
    for (const int k : order) {
        root_holds[static_cast<std::size_t>(k)].reset();
        roots.erase(colliding(40 + k));
        EXPECT_EQ(tree.Find(colliding(40 + k)), nullptr);
        expect_roots();
    }
    EXPECT_EQ(tree.Size(), 0u);
}

// A node started without a known parent (a Host entry no publication linked)
// is linked once a publication through its hash names its parent: it becomes
// the parent's child with its boundary, and the parent counts the checkpoint
// boundaries of its subtree and its Host presence. A later registration keeps
// the link.
TEST(BoundaryTreeTest, APublicationLinksANodeStartedWithoutAParent) {
    BoundaryTree tree{TreeConfig(1, false), /*checkpoints=*/nullptr};
    BoundaryTree::Hold page1_checkpoint = tree.HoldEntry("page1", kHostCheckpoint);
    const Node& page1 = page1_checkpoint.node();
    BoundaryTree::Hold page0_kv = tree.HoldEntry("page0", kHostKv);
    const Node& page0 = page0_kv.node();
    ASSERT_EQ(page1.parent, nullptr);
    ASSERT_EQ(page1.boundary_tokens, 0);
    ASSERT_EQ(HostLeaves(tree), (std::vector<std::string>{"page1", "page0"}));

    const std::vector<std::string> pages{"page0", "page1", "page2"};
    BoundaryTree::Hold publication = Publish(tree, pages, /*first_new_page=*/0, std::nullopt);
    BoundaryTree::Hold page2_checkpoint = tree.HoldEntry("page2", kDeviceCheckpoint);
    publication.reset();
    EXPECT_EQ(page0.parent, nullptr);
    EXPECT_EQ(page0.boundary_tokens, 4) << "page 0 gets its boundary";
    EXPECT_EQ(page1.parent, &page0);
    EXPECT_EQ(page1.boundary_tokens, 8);
    EXPECT_EQ(Children(page0), std::vector<std::string>{"page1"});
    EXPECT_EQ(Children(page1), std::vector<std::string>{"page2"});
    EXPECT_EQ(page0.checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 1}));
    EXPECT_EQ(page1.checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 1}));
    EXPECT_EQ(page0.children_on_host, 1);
    EXPECT_EQ(HostLeaves(tree), std::vector<std::string>{"page1"}) << "page0 has a child on the Host now";

    BoundaryTree::Hold page1_kv = tree.HoldEntry("page1", kDeviceKv);
    EXPECT_EQ(page1.parent, &page0) << "a registration keeps the link";
    EXPECT_EQ(page1.boundary_tokens, 8);

    // A tail under the empty prior has no parent to link it to: its
    // publication gives it its boundary alone.
    BoundaryTree::Hold tail_kv = tree.HoldEntry("tail", kDeviceKv);
    Publish(tree, {}, 0, TailPublication{.tail_tokens = 3, .hash = "tail"});
    EXPECT_EQ(tail_kv.node().parent, nullptr);
    EXPECT_EQ(tail_kv.node().boundary_tokens, 3);
    EXPECT_EQ(tail_kv.node().tail_tokens, 3);
    tail_kv.reset();
    EXPECT_EQ(tree.Find("tail"), nullptr);

    // The subtree leaves again, its counts with it.
    page2_checkpoint.reset();
    EXPECT_EQ(tree.Find("page2"), nullptr);
    page1_kv.reset();
    page1_checkpoint.reset();
    EXPECT_EQ(tree.Find("page1"), nullptr);
    EXPECT_EQ(page0.checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 0}));
    EXPECT_EQ(page0.children_on_host, 0);
    EXPECT_TRUE(Children(page0).empty());
    EXPECT_EQ(HostLeaves(tree), std::vector<std::string>{"page0"});
    EXPECT_EQ(tree.Size(), 1u);
}

TEST(BoundaryTreeTest, ACheckpointCountsInEveryAncestor) {
    BoundaryTree tree{TreeConfig(1, false), /*checkpoints=*/nullptr};
    const std::vector<std::string> pages{"page0", "page1", "page2", "page3", "page4"};
    BoundaryTree::Hold publication = Publish(tree, pages, 0, std::nullopt);
    BoundaryTree::Hold first = tree.HoldEntry("page4", kDeviceCheckpoint);
    for (const std::string& hash : pages) {
        const Node& node = NodeOf(tree, hash);
        EXPECT_EQ(node.checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 0})) << hash;
        EXPECT_EQ(node.HasCheckpoint(BoundaryTree::kDevice), hash == "page4") << hash;
    }
    // Another State group's checkpoint at the same boundary: a checkpoint
    // boundary counts once.
    BoundaryTree::Hold second = tree.HoldEntry("page4", kDeviceCheckpoint);
    EXPECT_EQ(NodeOf(tree, "page0").checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 0}));

    // A branch off page2 with a Device checkpoint boundary of its own: the
    // shared pages count both; a Host checkpoint boundary counts on the Host
    // alone.
    const std::vector<std::string> branch_pages{"page0", "page1", "page2", "branch"};
    BoundaryTree::Hold branch_publication = Publish(tree, branch_pages, 3, std::nullopt);
    BoundaryTree::Hold branch = tree.HoldEntry("branch", kDeviceCheckpoint);
    BoundaryTree::Hold host = tree.HoldEntry("page3", kHostCheckpoint);
    EXPECT_EQ(NodeOf(tree, "page0").checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{2, 1}));
    EXPECT_EQ(NodeOf(tree, "page1").checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{2, 1}));
    EXPECT_EQ(NodeOf(tree, "page2").checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{2, 1}));
    EXPECT_EQ(NodeOf(tree, "page3").checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 1}));
    EXPECT_EQ(NodeOf(tree, "page4").checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 0}));
    EXPECT_EQ(NodeOf(tree, "branch").checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 0}));

    // Dropping one of a boundary's two checkpoints leaves it counted.
    first.reset();
    EXPECT_TRUE(NodeOf(tree, "page4").HasCheckpoint(BoundaryTree::kDevice));
    EXPECT_EQ(NodeOf(tree, "page0").checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{2, 1}));
}

TEST(BoundaryTreeTest, DroppingTheLastCheckpointUncountsItInEveryAncestor) {
    // a - b - c - d, and b - e: d and e are checkpoint boundaries, b is the
    // fork.
    BoundaryTree tree{TreeConfig(1, false), /*checkpoints=*/nullptr};
    const std::vector<std::string> chain{"a", "b", "c", "d"};
    const std::vector<std::string> fork{"a", "b", "e"};
    BoundaryTree::Hold chain_publication = Publish(tree, chain, 0, std::nullopt);
    BoundaryTree::Hold fork_publication = Publish(tree, fork, 2, std::nullopt);
    BoundaryTree::Hold d = tree.HoldEntry("d", kDeviceCheckpoint);
    BoundaryTree::Hold e = tree.HoldEntry("e", kDeviceCheckpoint);
    const Node& a = NodeOf(tree, "a");
    const Node& b = NodeOf(tree, "b");
    const Node& c = NodeOf(tree, "c");

    d.reset();
    EXPECT_EQ(c.checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 0}));
    EXPECT_FALSE(c.HasCheckpointBoundaryInSubtree()) << "nothing is left under c";
    EXPECT_EQ(b.checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 0}));
    EXPECT_TRUE(b.HasCheckpointBoundaryInSubtree());
    EXPECT_EQ(a.checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 0}));

    // A checkpoint boundary with a deeper one under it: dropping it leaves
    // that one.
    BoundaryTree::Hold at_b = tree.HoldEntry("b", kDeviceCheckpoint);
    at_b.reset();
    EXPECT_EQ(b.checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 0})) << "e is still under b";

    // A checkpoint boundary on the other tier keeps the node's subtree
    // counted.
    BoundaryTree::Hold e_on_host = tree.HoldEntry("e", kHostCheckpoint);
    e.reset();
    EXPECT_EQ(a.checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 1}));
    EXPECT_TRUE(a.HasCheckpointBoundaryInSubtree());
    e_on_host.reset();
    EXPECT_EQ(a.checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 0}));
    EXPECT_FALSE(a.HasCheckpointBoundaryInSubtree());
}

TEST(BoundaryTreeTest, TheLastHolderTakesItsNodeAndTheParentsLeftEmpty) {
    BoundaryTree tree{TreeConfig(1, false), /*checkpoints=*/nullptr};
    std::vector<BoundaryTree::Hold> leaves;
    for (const char* leaf : {"x", "y", "z"}) {
        const std::vector<std::string> pages{"parent", leaf};
        BoundaryTree::Hold publication = Publish(tree, pages, 1, std::nullopt);
        leaves.push_back(tree.HoldEntry(leaf, kDeviceKv));
    }
    const Node& parent = NodeOf(tree, "parent");
    const Node& x = NodeOf(tree, "x");
    const Node& z = NodeOf(tree, "z");
    EXPECT_EQ(Children(parent), (std::vector<std::string>{"x", "y", "z"})) << "siblings follow insertion";

    leaves[1].reset();
    EXPECT_EQ(tree.Find("y"), nullptr);
    EXPECT_EQ(Children(parent), (std::vector<std::string>{"x", "z"}));
    EXPECT_EQ(x.next_sibling, &z);
    EXPECT_EQ(z.prev_sibling, &x);

    // A moved hold carries the claim; the one moved from releases nothing.
    BoundaryTree::Hold moved = std::move(leaves[0]);
    EXPECT_FALSE(leaves[0]);
    leaves[0].reset();
    EXPECT_EQ(tree.Find("x"), &x);
    moved.reset();
    EXPECT_EQ(parent.first_child, &z);
    EXPECT_EQ(z.prev_sibling, nullptr);

    // The parent holds nothing of its own: it leaves with its last child.
    leaves[2].reset();
    EXPECT_EQ(tree.Size(), 0u);
}

TEST(BoundaryTreeTest, HostLeavesFollowPresence) {
    // page0 - page1 - page2 - page3, and page1 - branch.
    BoundaryTree tree{TreeConfig(1, false), /*checkpoints=*/nullptr};
    const std::vector<std::string> chain{"page0", "page1", "page2", "page3"};
    const std::vector<std::string> fork{"page0", "page1", "branch"};
    BoundaryTree::Hold chain_publication = Publish(tree, chain, 0, std::nullopt);
    BoundaryTree::Hold fork_publication = Publish(tree, fork, 2, std::nullopt);

    // Page by page down the chain, only the deepest page is a leaf.
    std::vector<BoundaryTree::Hold> on_host;
    for (const std::string& page : chain) {
        on_host.push_back(tree.HoldEntry(page, kHostKv));
        EXPECT_EQ(HostLeaves(tree), std::vector<std::string>{page});
    }
    // A write-back on its way to the Host puts the branch there too.
    BoundaryTree::Hold store = tree.HoldPendingStore(*tree.Find("branch"), /*checkpoint=*/false);
    EXPECT_EQ(HostLeaves(tree), (std::vector<std::string>{"page3", "branch"}));
    EXPECT_EQ(NodeOf(tree, "page1").children_on_host, 2);

    // Taking the deepest page off the Host makes its parent a leaf again, at
    // the back of the list.
    on_host[3].reset();
    EXPECT_EQ(HostLeaves(tree), (std::vector<std::string>{"branch", "page2"}));
    store.reset();
    EXPECT_EQ(HostLeaves(tree), std::vector<std::string>{"page2"});
    EXPECT_EQ(NodeOf(tree, "page1").children_on_host, 1) << "page2 is still on the Host";
    on_host[2].reset();
    EXPECT_EQ(HostLeaves(tree), std::vector<std::string>{"page1"});

    // A page off the Host is no leaf, whatever its children.
    on_host[1].reset();
    on_host[0].reset();
    EXPECT_TRUE(HostLeaves(tree).empty());
    EXPECT_EQ(tree.FirstHostLeaf(), nullptr);
}

TEST(BoundaryTreeTest, ANodeLeavesWithItsTailHint) {
    BoundaryTree tree{TreeConfig(1, false), /*checkpoints=*/nullptr};
    const std::vector<std::string> pages{"page0"};
    BoundaryTree::Hold publication = Publish(tree, pages, 0, TailPublication{.tail_tokens = 2, .hash = "tail"});
    BoundaryTree::Hold tail_kv = tree.HoldEntry("tail", kDeviceKv);
    publication.reset();
    EXPECT_FALSE(tree.RecordTail("page0", 2, "tail").has_value());
    ASSERT_TRUE(tree.Tails().Contains("tail"));

    tail_kv.reset();
    EXPECT_FALSE(tree.Tails().Contains("tail"));
    EXPECT_EQ(tree.Size(), 0u) << "the page above held nothing either";
}

TEST(BoundaryTreeTest, DuplicatePreflightPrunesPendingAndIncompletePrefixesAcrossRoots) {
    // Two prefix-closed blocks make a page whole on the Device.
    BoundaryTree tree{TreeConfig(/*closed_blocks_per_page=*/2, false), /*checkpoints=*/nullptr};
    const std::vector<std::string> chain{"root", "left", "leaf"};
    const std::vector<std::string> other_page{"other"};
    BoundaryTree::Hold chain_publication = Publish(tree, chain, 0, std::nullopt);
    BoundaryTree::Hold other_publication = Publish(tree, other_page, 0, std::nullopt);
    std::map<std::string, std::array<BoundaryTree::Hold, 2>> kv;
    for (const char* hash : {"root", "left", "leaf", "other"}) {
        kv[hash][0] = tree.HoldEntry(hash, kDeviceKv);
        kv[hash][1] = tree.HoldEntry(hash, kDeviceKv);
    }
    kv["leaf"][0].SetPaired(true);
    ASSERT_TRUE(tree.HasHostKvDuplicateCandidate());
    BoundaryTree::Hold root_store = tree.HoldPendingStore(*tree.Find("root"), /*checkpoint=*/false);
    EXPECT_FALSE(tree.HasHostKvDuplicateCandidate());
    kv["other"][0].SetPaired(true);
    EXPECT_TRUE(tree.HasHostKvDuplicateCandidate()) << "another root is not covered";
    kv["other"][1].reset();
    EXPECT_FALSE(tree.HasHostKvDuplicateCandidate()) << "a page the Device no longer holds whole backs nothing";
    root_store.reset();
    EXPECT_TRUE(tree.HasHostKvDuplicateCandidate());
    kv["root"][1].reset();
    EXPECT_FALSE(tree.HasHostKvDuplicateCandidate()) << "the Device gap also covers the descendant";
    kv["root"][1] = tree.HoldEntry("root", kDeviceKv);
    EXPECT_TRUE(tree.HasHostKvDuplicateCandidate());

    // The first root leaves; the next one still counts.
    chain_publication.reset();
    for (const char* hash : {"leaf", "left", "root"}) {
        kv.erase(hash);
    }
    ASSERT_EQ(tree.Find("root"), nullptr);
    kv["other"][1] = tree.HoldEntry("other", kDeviceKv);
    EXPECT_TRUE(tree.HasHostKvDuplicateCandidate()) << "removing the first root keeps the next";
}

TEST(BoundaryTreeTest, DuplicatePreflightCountsEachPairOnce) {
    BoundaryTree tree{TreeConfig(1, false), /*checkpoints=*/nullptr};
    const std::vector<std::string> first_page{"first"};
    const std::vector<std::string> second_page{"second"};
    BoundaryTree::Hold first_publication = Publish(tree, first_page, 0, std::nullopt);
    BoundaryTree::Hold second_publication = Publish(tree, second_page, 0, std::nullopt);
    BoundaryTree::Hold first = tree.HoldEntry("first", kDeviceKv);
    BoundaryTree::Hold second = tree.HoldEntry("second", kDeviceKv);
    first_publication.reset();
    second_publication.reset();

    first.SetPaired(true);
    first.SetPaired(true);
    second.SetPaired(true);
    first.SetPaired(false);
    EXPECT_TRUE(tree.HasHostKvDuplicateCandidate()) << "unpairing one keeps the other";
    second.SetPaired(false);
    EXPECT_FALSE(tree.HasHostKvDuplicateCandidate()) << "a repeated pairing counts once";
    second.SetPaired(false);
    first.SetPaired(true);
    EXPECT_TRUE(tree.HasHostKvDuplicateCandidate()) << "a repeated unpairing counts once";
    first.reset();
    EXPECT_FALSE(tree.HasHostKvDuplicateCandidate()) << "an entry takes its pair along";
}

TEST(BoundaryTreeTest, LinkingAPendingRootAppliesBothStoreHolds) {
    for (const bool ancestor_first : {true, false}) {
        SCOPED_TRACE(ancestor_first);
        BoundaryTree tree{TreeConfig(1, false), /*checkpoints=*/nullptr};
        // A node started without a parent, whole on the Device with a Host
        // copy, and a store on its way to the Host.
        BoundaryTree::Hold detached = tree.HoldEntry("detached", kDeviceKv);
        detached.SetPaired(true);
        BoundaryTree::Hold detached_store = tree.HoldPendingStore(detached.node(), /*checkpoint=*/false);
        EXPECT_FALSE(tree.HasHostKvDuplicateCandidate()) << "a node without its boundary backs nothing";

        const std::vector<std::string> pages{"root", "detached"};
        BoundaryTree::Hold publication = Publish(tree, pages, 0, std::nullopt);
        BoundaryTree::Hold root = tree.HoldEntry("root", kDeviceKv);
        BoundaryTree::Hold root_store = tree.HoldPendingStore(root.node(), /*checkpoint=*/false);
        publication.reset();
        ASSERT_EQ(detached.node().parent, &root.node());
        EXPECT_FALSE(tree.HasHostKvDuplicateCandidate());
        (ancestor_first ? root_store : detached_store).reset();
        EXPECT_FALSE(tree.HasHostKvDuplicateCandidate());
        (ancestor_first ? detached_store : root_store).reset();
        EXPECT_TRUE(tree.HasHostKvDuplicateCandidate());

        detached.reset();
        root.reset();
        EXPECT_FALSE(tree.HasHostKvDuplicateCandidate());
        EXPECT_EQ(tree.Size(), 0u);
    }
}

TEST(BoundaryTreeTest, DroppingAHolderQueuesTheNodesThatMayHaveDied) {
    BoundaryTree tree{TreeConfig(1, /*queue_dead_candidates=*/true), /*checkpoints=*/nullptr};
    const std::vector<std::string> pages{"page0", "page1", "page2"};
    BoundaryTree::Hold publication = Publish(tree, pages, 0, std::nullopt);
    std::vector<BoundaryTree::Hold> kv;
    for (const std::string& page : pages) {
        kv.push_back(tree.HoldEntry(page, kDeviceKv));
    }
    BoundaryTree::Hold checkpoint = tree.HoldEntry("page2", kDeviceCheckpoint);
    publication.reset();
    EXPECT_EQ(tree.PopDeadCandidate(), std::nullopt) << "page2 still holds a checkpoint boundary";

    // No checkpoint boundary is left under page2, then page2 leaves: the
    // candidate it queued leaves with it, and the walk queues the node it
    // stops at, so taking a chain from below queues one node.
    checkpoint.reset();
    kv[2].reset();
    EXPECT_EQ(tree.PopDeadCandidate(), std::optional<ContentHash>{"page1"});
    EXPECT_EQ(tree.PopDeadCandidate(), std::nullopt);

    // A page no prefix-closed group holds on either tier is a gap, whatever
    // else holds it.
    BoundaryTree::Hold window = tree.HoldEntry("page1", kDeviceWindow);
    kv[1].reset();
    EXPECT_EQ(tree.PopDeadCandidate(), std::optional<ContentHash>{"page1"});

    // A hash at the back is not queued again, and a node that is gone not at
    // all.
    tree.QueueDeadCandidate("page0");
    tree.QueueDeadCandidate("page0");
    tree.QueueDeadCandidate("absent");
    EXPECT_EQ(tree.PopDeadCandidate(), std::optional<ContentHash>{"page0"});
    EXPECT_EQ(tree.PopDeadCandidate(), std::nullopt);
}

TEST(BoundaryTreeTest, NothingIsQueuedWithoutCollection) {
    BoundaryTree tree{TreeConfig(1, /*queue_dead_candidates=*/false), /*checkpoints=*/nullptr};
    const std::vector<std::string> pages{"page0", "page1"};
    BoundaryTree::Hold publication = Publish(tree, pages, 0, std::nullopt);
    BoundaryTree::Hold kv = tree.HoldEntry("page1", kDeviceKv);
    BoundaryTree::Hold checkpoint = tree.HoldEntry("page1", kDeviceCheckpoint);
    publication.reset();
    checkpoint.reset();
    kv.reset();
    tree.QueueDeadCandidate("page0");
    EXPECT_EQ(tree.PopDeadCandidate(), std::nullopt);
}

TEST(BoundaryTreeTest, AStateCheckpointsResidencyFollowsItsHolds) {
    CheckpointTree checkpoints{kP, /*leaf_distance=*/0};
    BoundaryTree::Config config = TreeConfig(1, false);
    config.num_state_groups = 2;
    BoundaryTree tree{config, &checkpoints};
    // One of the two State groups holds the checkpoint at the end of page0
    // when a publication observes it, as the coordinator does.
    const std::vector<std::int32_t> tokens{1, 2, 3, 4};
    const std::vector<std::string> pages{"page0"};
    BoundaryTree::Hold first = tree.HoldEntry("page0", kDeviceCheckpoint);
    ASSERT_TRUE(checkpoints.Observe(tokens, pages, std::nullopt, /*access_epoch=*/1));
    checkpoints.Update("page0", tree.CheckpointResidency(first.node()));
    EXPECT_EQ(checkpoints.Distance("page0"), std::nullopt) << "complete on no tier yet";

    BoundaryTree::Hold second = tree.HoldEntry("page0", kDeviceCheckpoint);
    EXPECT_EQ(checkpoints.Distance("page0"), std::optional<std::int32_t>{0}) << "complete on the Device";
    // A store on its way to the Host keeps it once the Device lets it go.
    BoundaryTree::Hold store = tree.HoldPendingStore(first.node(), /*checkpoint=*/true);
    first.reset();
    second.reset();
    EXPECT_EQ(checkpoints.Distance("page0"), std::optional<std::int32_t>{0});
    store.reset();
    EXPECT_FALSE(checkpoints.Contains("page0")) << "nothing holds it any more";
}

TEST(BoundaryTreeTest, AClearedCacheLeavesNoNodeAndDropsTheHintsLeft) {
    BoundaryTree tree{TreeConfig(1, true), /*checkpoints=*/nullptr};
    {
        BoundaryTree::Hold kv = tree.HoldEntry("page0", kDeviceKv);
        EXPECT_THROW(tree.ClearAfterCacheClear(), std::runtime_error);
    }
    // A tail published with nothing registered under it leaves a hint no
    // node backs.
    EXPECT_FALSE(tree.RecordTail("", 2, "tail").has_value());
    ASSERT_EQ(tree.Tails().Size(), 1u);
    tree.ClearAfterCacheClear();
    EXPECT_EQ(tree.Tails().Size(), 0u);
    EXPECT_EQ(tree.Size(), 0u);
}

}  // namespace
}  // namespace tokenspeed::test
