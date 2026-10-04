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

"""Greedy verification of a speculative draft tree.

Node 0 of every request is the root (the last verified token); ``parent``
links each other node to an earlier one. Walking from the root, a node is
accepted when its token equals the target's pick at its parent, preferring
the lowest node index among matching siblings. The accepted path is emitted
packed, so downstream code reads it exactly like a chain.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["verify_tree"]


@triton.jit
def _verify_tree_kernel(
    predicts_ptr,
    accept_length_ptr,
    path_ptr,
    candidates_ptr,
    parent_ptr,
    target_ptr,
    NUM_NODES: tl.constexpr,
    NODES_PAD: tl.constexpr,
):
    req = tl.program_id(0)
    base = req * NUM_NODES
    nodes = tl.arange(0, NODES_PAD)
    node_ok = nodes < NUM_NODES

    tokens = tl.load(candidates_ptr + base + nodes, mask=node_ok, other=-1)
    parents = tl.load(parent_ptr + base + nodes, mask=node_ok, other=-2)

    # Built in registers and stored once: no cross-thread store ordering on the path.
    path = tl.where(nodes == 0, 0, -1)
    current = tl.full((), 0, tl.int32)
    depth = tl.full((), 0, tl.int32)
    found = tl.full((), True, tl.int1)
    while found:
        pick = tl.load(target_ptr + base + current)
        match = (parents == current) & (tokens == pick) & node_ok
        child = tl.min(tl.where(match, nodes, NODES_PAD), axis=0)
        found = child < NODES_PAD
        if found:
            depth += 1
            tl.store(predicts_ptr + base + depth - 1, pick)
            path = tl.where(nodes == depth, child, path)
            current = child

    tl.store(path_ptr + base + nodes, path, mask=node_ok)
    tl.store(predicts_ptr + base + depth, tl.load(target_ptr + base + current))
    tl.store(accept_length_ptr + req, depth + 1)


def verify_tree(
    predicts: torch.Tensor,
    accept_length: torch.Tensor,
    path: torch.Tensor,
    candidates: torch.Tensor,
    parent: torch.Tensor,
    target: torch.Tensor,
) -> None:
    """Accept the longest root path whose tokens match the target's picks.

    Args:
        predicts: ``[bs * N]`` int32 output; row ``b`` holds the target's picks
            along the accepted path at ``[b * N, b * N + accept_length[b])``.
        accept_length: ``[bs]`` int32 output; accepted drafts plus the bonus
            token.
        path: ``[bs, N]`` int32 output; accepted node indices, root first,
            ``-1`` past the path.
        candidates: ``[bs, N]`` int32 node tokens, node 0 the root.
        parent: ``[bs, N]`` int32 parent node index, ``-1`` for the root.
        target: ``[bs * N]`` int32 target pick at every node (greedy argmax
            or a sampled pick).
    """
    bs, num_nodes = candidates.shape
    if bs == 0:
        return
    _verify_tree_kernel[(bs,)](
        predicts,
        accept_length,
        path,
        candidates,
        parent,
        target,
        NUM_NODES=num_nodes,
        NODES_PAD=triton.next_power_of_2(num_nodes),
    )
