# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""EAGLE-Harness: EAGLE-4's cost-driven dynamic draft trees.

``EagleHarness`` grows each round's tree one depth at a time and decides after
each depth, from measured cost data, whether to stop and how many nodes to
verify. The decisions run on the device (``DevicePolicy``), bit for bit
EAGLE-4's, into slots the draft kernels read, so one captured graph
per padded batch and depth serves every round and the host never reads a
round's values. The captured graphs chain on the GPU (``DeviceChain``): each
depth's graph, after its decision, launches the next depth while its go flag
is set, else the finalize, so the host launches depth 1 only, behind the
decode graph (``replay_decode_graph``); an eager round has the host poll a
pinned flag to launch each depth (the event loop's one named host-polling
exception). The round drafts on the drafter's round stream, which owns the
kernel workspace until the next forward's settle (``complete_round``) joins
it.
Exec mode 2 drains the GPU around each round's depths to time them; it
measures, it does not serve.

A chain's width table (``--speculative-chain-width-table``) fixes the width
``W`` of the rounds whose live batch size it covers: such a round drafts
depths 1 to ``W - 2`` (depth 1 always runs) and writes a verify tree of ``W``
nodes, the window's other slots never accepted, as in any round that stops
early. Its start writes every slot, go flag and count of the round, so it
reads no work deltas or cost data, the decisions launched in its depths
change nothing, and an eager round's host launches its depths at once
instead of polling their go flags. The confidence rule decides the other rounds, of at most 256
requests (its cost data's batch sizes); chains serve up to 512, the table
deciding above 256. A padded batch captures the depth graphs its rounds can
reach, over the live batch sizes that pad to it (``decode_graph_floors``, from
the decode graph ladder): all of them where the confidence rule decides one
of those sizes, else up to the table's deepest ``max(W - 2, 1)``. A round that
can reach a depth its padded batch did not capture raises before it launches.

Where it departs from ``docs/design/tree-speculation.md``:

* Its chains (``chain_intensity``) are one-child trees on the tree path.
* Its lane windows exceed ``(S - 1) * K <= N``: up to 128 slots, so a lane
  row's ancestor mask takes two 64-bit words (bit ``j % 64`` of word
  ``j // 64``), and its deepest lanes can write past the reserved next verify
  window, costing draft acceptance only. Its depth update writes the lane
  masks in place of ``draft_tree_expand``.
* Sampled trees and chains (``--enable-speculative-sampling``) draw each
  node's children from the draft q (``RuntimeStates.future_draft_q``), which
  the verifier accepts by recursive rejection sampling
  (``TreeRejectionSampler``). A sampled round stops at a depth planned before
  its draws, so where it stops never depends on them.
"""
