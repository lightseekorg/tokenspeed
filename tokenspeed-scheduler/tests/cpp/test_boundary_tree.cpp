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

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "cache/prefix/boundary_tree.h"

namespace tokenspeed::test {
namespace {

using Node = BoundaryTree::Node;
using CheckpointBoundaryCounts = std::array<std::int32_t, 2>;  // checkpoint_boundaries_in_subtree by CacheTier

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

TEST(BoundaryTreeTest, FindOrAddLinksAPageUnderItsParentAndATailUnderItsPrior) {
    BoundaryTree tree;
    Node& page0 = tree.FindOrAdd("page0", /*parent=*/nullptr, /*boundary_tokens=*/4, /*tail_tokens=*/0);
    Node& page1 = tree.FindOrAdd("page1", &page0, 8, 0);
    Node& tail = tree.FindOrAdd("tail", &page1, 10, 2);

    EXPECT_EQ(tree.Size(), 3u);
    EXPECT_EQ(page0.parent, nullptr);
    EXPECT_EQ(page1.parent, &page0);
    EXPECT_EQ(tail.parent, &page1);
    EXPECT_EQ(*tail.hash, "tail");
    EXPECT_EQ(page1.boundary_tokens, 8);
    EXPECT_EQ(page1.tail_tokens, 0);
    EXPECT_EQ(tail.boundary_tokens, 10);
    EXPECT_EQ(tail.tail_tokens, 2);
    EXPECT_EQ(Children(page0), std::vector<std::string>{"page1"});
    EXPECT_EQ(Children(page1), std::vector<std::string>{"tail"});
    EXPECT_TRUE(Children(tail).empty());
    EXPECT_EQ(tree.Find("page1"), &page1);
    EXPECT_EQ(tree.Find("absent"), nullptr);
    for (const Node* node : {&page0, &page1, &tail}) {
        EXPECT_EQ(node->checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 0}));
        EXPECT_FALSE(node->on_host);
    }

    // A hash added again is the node it already has, links and boundary kept.
    EXPECT_EQ(&tree.FindOrAdd("tail", &page0, 99, 3), &tail);
    EXPECT_EQ(tail.parent, &page1);
    EXPECT_EQ(tail.boundary_tokens, 10);
    EXPECT_EQ(tail.tail_tokens, 2);
    EXPECT_EQ(Children(page0), std::vector<std::string>{"page1"});
    EXPECT_EQ(tree.Size(), 3u);
}

// A node started without a known parent (a Host entry no publication linked)
// is linked once a publication through its hash names its parent: it becomes
// the parent's child with its boundary, and the parent counts the checkpoint
// boundaries of its subtree and its Host presence. A later FindOrAdd without a
// parent keeps the link.
TEST(BoundaryTreeTest, FindOrAddLinksANodeStartedWithoutAParent) {
    BoundaryTree tree;
    Node& page1 = tree.FindOrAdd("page1", /*parent=*/nullptr, /*boundary_tokens=*/0, /*tail_tokens=*/0);
    tree.SetOnHost(page1, true);
    tree.SetCheckpoint(page1, CacheTier::kHost, true);
    Node& page2 = tree.FindOrAdd("page2", &page1, 12, 0);
    tree.SetCheckpoint(page2, CacheTier::kDevice, true);
    Node& page0 = tree.FindOrAdd("page0", nullptr, 4, 0);
    tree.SetOnHost(page0, true);
    ASSERT_EQ(HostLeaves(tree), (std::vector<std::string>{"page1", "page0"}));
    ASSERT_EQ(page0.checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 0}));

    EXPECT_EQ(&tree.FindOrAdd("page1", &page0, 8, 0), &page1);
    EXPECT_EQ(page1.parent, &page0);
    EXPECT_EQ(page1.boundary_tokens, 8);
    EXPECT_EQ(page1.tail_tokens, 0);
    EXPECT_EQ(Children(page0), std::vector<std::string>{"page1"});
    EXPECT_EQ(Children(page1), std::vector<std::string>{"page2"});
    EXPECT_EQ(page0.checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 1}));
    EXPECT_EQ(page1.checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 1}));
    EXPECT_EQ(page0.children_on_host, 1);
    EXPECT_EQ(HostLeaves(tree), std::vector<std::string>{"page1"}) << "page0 has a child on the Host now";

    EXPECT_EQ(&tree.FindOrAdd("page1", nullptr, 0, 0), &page1);
    EXPECT_EQ(page1.parent, &page0);
    EXPECT_EQ(page1.boundary_tokens, 8);

    // A tail under the empty prior has no parent to link it to: its
    // publication gives it its boundary alone.
    Node& tail = tree.FindOrAdd("tail", nullptr, 0, 0);
    EXPECT_EQ(&tree.FindOrAdd("tail", nullptr, 3, 3), &tail);
    EXPECT_EQ(tail.parent, nullptr);
    EXPECT_EQ(tail.boundary_tokens, 3);
    EXPECT_EQ(tail.tail_tokens, 3);
    tree.Remove(tail);

    // The subtree leaves again, its counts with it.
    tree.SetCheckpoint(page2, CacheTier::kDevice, false);
    tree.Remove(page2);
    tree.SetCheckpoint(page1, CacheTier::kHost, false);
    tree.SetOnHost(page1, false);
    tree.Remove(page1);
    EXPECT_EQ(page0.checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 0}));
    EXPECT_EQ(page0.children_on_host, 0);
    EXPECT_TRUE(Children(page0).empty());
    EXPECT_EQ(HostLeaves(tree), std::vector<std::string>{"page0"});
    EXPECT_EQ(tree.Size(), 1u);
}

TEST(BoundaryTreeTest, ACheckpointFlipCountsInEveryAncestor) {
    BoundaryTree tree;
    std::vector<Node*> chain;
    for (std::int32_t page = 0; page < 5; ++page) {
        chain.push_back(&tree.FindOrAdd("page" + std::to_string(page), page == 0 ? nullptr : chain.back(),
                                        4 * (page + 1),
                                        /*tail_tokens=*/0));
    }
    EXPECT_TRUE(tree.SetCheckpoint(*chain[4], CacheTier::kDevice, true));
    for (const Node* node : chain) {
        EXPECT_EQ(node->checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 0})) << *node->hash;
        EXPECT_EQ(node->has_checkpoint[0], node == chain[4]) << *node->hash;
    }
    EXPECT_FALSE(tree.SetCheckpoint(*chain[4], CacheTier::kDevice, true)) << "the mark is set already";
    EXPECT_EQ(chain[0]->checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 0}))
        << "a checkpoint boundary counts once";

    // A branch off page2 with a Device checkpoint boundary of its own: the
    // shared pages count both; a Host checkpoint boundary counts on the Host
    // alone.
    Node& branch = tree.FindOrAdd("branch", chain[2], 12, 0);
    tree.SetCheckpoint(branch, CacheTier::kDevice, true);
    tree.SetCheckpoint(*chain[3], CacheTier::kHost, true);
    EXPECT_EQ(chain[0]->checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{2, 1}));
    EXPECT_EQ(chain[1]->checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{2, 1}));
    EXPECT_EQ(chain[2]->checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{2, 1}));
    EXPECT_EQ(chain[3]->checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 1}));
    EXPECT_EQ(chain[4]->checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 0}));
    EXPECT_EQ(branch.checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 0}));
}

TEST(BoundaryTreeTest, ClearingACheckpointUncountsItInEveryAncestor) {
    // a - b - c - d, and b - e: d and e are checkpoint boundaries, b is the
    // fork.
    BoundaryTree tree;
    Node& a = tree.FindOrAdd("a", nullptr, 4, 0);
    Node& b = tree.FindOrAdd("b", &a, 8, 0);
    Node& c = tree.FindOrAdd("c", &b, 12, 0);
    Node& d = tree.FindOrAdd("d", &c, 16, 0);
    Node& e = tree.FindOrAdd("e", &b, 12, 0);
    tree.SetCheckpoint(d, CacheTier::kDevice, true);
    tree.SetCheckpoint(e, CacheTier::kDevice, true);

    EXPECT_TRUE(tree.SetCheckpoint(d, CacheTier::kDevice, false));
    EXPECT_EQ(c.checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 0}));
    EXPECT_FALSE(c.HasCheckpointBoundaryInSubtree()) << "nothing is left under c";
    EXPECT_EQ(b.checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 0}));
    EXPECT_TRUE(b.HasCheckpointBoundaryInSubtree());
    EXPECT_EQ(a.checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 0}));

    // A checkpoint boundary with a deeper one under it: clearing it leaves that
    // one.
    tree.SetCheckpoint(b, CacheTier::kDevice, true);
    EXPECT_TRUE(tree.SetCheckpoint(b, CacheTier::kDevice, false));
    EXPECT_EQ(b.checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 0})) << "e is still under b";

    // A checkpoint boundary on the other tier keeps the node's subtree alive.
    tree.SetCheckpoint(e, CacheTier::kHost, true);
    EXPECT_TRUE(tree.SetCheckpoint(e, CacheTier::kDevice, false));
    EXPECT_EQ(a.checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 1}));
    EXPECT_TRUE(a.HasCheckpointBoundaryInSubtree());
    EXPECT_TRUE(tree.SetCheckpoint(e, CacheTier::kHost, false));
    EXPECT_EQ(a.checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 0}));
    EXPECT_FALSE(a.HasCheckpointBoundaryInSubtree());
    EXPECT_FALSE(tree.SetCheckpoint(e, CacheTier::kHost, false)) << "clearing a cleared checkpoint changes nothing";
}

TEST(BoundaryTreeTest, RemovingALeafUnlinksItInConstantTime) {
    BoundaryTree tree;
    Node& parent = tree.FindOrAdd("parent", nullptr, 4, 0);
    Node& x = tree.FindOrAdd("x", &parent, 8, 0);
    tree.FindOrAdd("y", &parent, 8, 0);
    Node& z = tree.FindOrAdd("z", &parent, 8, 0);

    tree.Remove(*tree.Find("y"));
    EXPECT_EQ(tree.Find("y"), nullptr);
    EXPECT_EQ(Children(parent), (std::vector<std::string>{"x", "z"}));
    EXPECT_EQ(x.next_sibling, &z);
    EXPECT_EQ(z.prev_sibling, &x);

    tree.Remove(x);
    EXPECT_EQ(parent.first_child, &z);
    EXPECT_EQ(z.prev_sibling, nullptr);
    tree.Remove(z);
    EXPECT_EQ(parent.first_child, nullptr);
    EXPECT_EQ(parent.last_child, nullptr);
    tree.Remove(parent);
    EXPECT_EQ(tree.Size(), 0u);
}

TEST(BoundaryTreeTest, SiblingOrderFollowsInsertion) {
    BoundaryTree tree;
    Node& parent = tree.FindOrAdd("parent", nullptr, 4, 0);
    for (const char* hash : {"c", "a", "b"}) {
        tree.FindOrAdd(hash, &parent, 8, 0);
    }
    EXPECT_EQ(Children(parent), (std::vector<std::string>{"c", "a", "b"}));
    tree.Remove(*tree.Find("a"));
    tree.FindOrAdd("d", &parent, 8, 0);
    EXPECT_EQ(Children(parent), (std::vector<std::string>{"c", "b", "d"}));
}

TEST(BoundaryTreeTest, HostLeavesFollowPresence) {
    // page0 - page1 - page2 - page3, and page1 - branch.
    BoundaryTree tree;
    std::vector<Node*> chain;
    for (std::int32_t page = 0; page < 4; ++page) {
        chain.push_back(&tree.FindOrAdd("page" + std::to_string(page), page == 0 ? nullptr : chain.back(),
                                        4 * (page + 1),
                                        /*tail_tokens=*/0));
    }
    Node& branch = tree.FindOrAdd("branch", chain[1], 8, 0);

    // Page by page down the chain, only the deepest page is a leaf.
    for (std::int32_t page = 0; page < 4; ++page) {
        tree.SetOnHost(*chain[static_cast<std::size_t>(page)], true);
        EXPECT_EQ(HostLeaves(tree), std::vector<std::string>{"page" + std::to_string(page)});
    }
    tree.SetOnHost(branch, true);
    EXPECT_EQ(HostLeaves(tree), (std::vector<std::string>{"page3", "branch"}));
    EXPECT_EQ(chain[1]->children_on_host, 2);

    // Taking the deepest page off the Host makes its parent a leaf again, at
    // the back of the list.
    tree.SetOnHost(*chain[3], false);
    EXPECT_EQ(HostLeaves(tree), (std::vector<std::string>{"branch", "page2"}));
    tree.SetOnHost(branch, false);
    EXPECT_EQ(HostLeaves(tree), std::vector<std::string>{"page2"});
    EXPECT_EQ(chain[1]->children_on_host, 1) << "page2 is still on the Host";
    tree.SetOnHost(*chain[2], false);
    EXPECT_EQ(HostLeaves(tree), std::vector<std::string>{"page1"});

    // A page off the Host is no leaf, whatever its children.
    tree.SetOnHost(*chain[1], false);
    tree.SetOnHost(*chain[0], false);
    EXPECT_TRUE(HostLeaves(tree).empty());
    EXPECT_EQ(tree.FirstHostLeaf(), nullptr);
}

}  // namespace
}  // namespace tokenspeed::test
