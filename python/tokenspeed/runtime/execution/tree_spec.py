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

"""Per-step draft-tree state shared by the executor, attention and drafter.

A decode request verifies ``N`` nodes: node 0 the last verified token, the
rest drafted, linked by ``parent``. Positions follow depth, KV slots follow
node index (the verify write window), and after verification the accepted
path is compacted to the front of the request's window -- target KV, target
hidden rows and the packed predictions -- so everything downstream sees a
chain. A chain is the tree ``parent[i] = i - 1``; with ``topk == 1`` none of
this state exists and the chain path runs unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.sampling.triton import tree_ancestry

from tokenspeed.runtime.utils.triton import tl, triton

__all__ = ["TreeSpec", "TreeSpecConfig"]


@triton.jit
def _compact_rows_kernel(
    path_ptr,  # [bs, N] int32 accepted path, -1 past it
    hidden_ptr,  # [bs * N, hidden] verify hidden rows
    stride_hidden,
    positions_ptr,  # [bs * N] int64 window positions
    depth_ptr,  # [bs, N] int32 node depth
    HIDDEN: tl.constexpr,
    N: tl.constexpr,
    N_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Program (0, req) packs request req's accepted hidden rows to the front of
    its window; program (1, req) shifts its positions back to vc + i."""
    req = tl.program_id(1)
    if tl.program_id(0) == 1:
        # Row i now holds the path node at depth i: back from vc + depth to vc + i.
        nodes = tl.arange(0, N_PAD)
        ok = nodes < N
        depth = tl.load(depth_ptr + req * N + nodes, mask=ok, other=0)
        pos = tl.load(positions_ptr + req * N + nodes, mask=ok, other=0)
        tl.store(
            positions_ptr + req * N + nodes, pos + (nodes - depth).to(tl.int64), mask=ok
        )
    else:
        offs = tl.arange(0, BLOCK)
        # The path is increasing, so row d never overwrites a later row's source.
        for d in range(N):
            src = tl.load(path_ptr + req * N + d)
            if (src >= 0) & (src != d):
                src_row = hidden_ptr + (req * N + src).to(tl.int64) * stride_hidden
                dst_row = hidden_ptr + (req * N + d).to(tl.int64) * stride_hidden
                for start in range(0, HIDDEN, BLOCK):
                    cols = start + offs
                    row = tl.load(src_row + cols, mask=cols < HIDDEN)
                    tl.store(dst_row + cols, row, mask=cols < HIDDEN)


@dataclass(frozen=True)
class TreeSpecConfig:
    topk: int
    num_steps: int
    num_nodes: int


class TreeSpec:
    """Device buffers for the tree being verified this step."""

    def __init__(
        self, config: TreeSpecConfig, max_bs: int, device: torch.device
    ) -> None:
        self.config = config
        n = config.num_nodes
        self.chain_parent = torch.arange(-1, n - 1, dtype=torch.int32, device=device)
        self.parent_buf = self.chain_parent.repeat(max_bs, 1)
        # This round's drafted tree; the executor publishes it to RuntimeStates.future_parent_map.
        self.draft_parent_buf = self.chain_parent.repeat(max_bs, 1)
        # The chain the parents describe, so graph warmup's compact (no load_step) is a no-op.
        self.depth_buf = torch.arange(n, dtype=torch.int32, device=device).repeat(
            max_bs, 1
        )
        chain_mask = [(1 << (i + 1)) - 1 for i in range(n)]
        chain_mask = [
            m - (1 << 64) if m >> 63 else m for m in chain_mask
        ]  # bit 63 is the sign
        self.mask_buf = torch.tensor(
            chain_mask, dtype=torch.int64, device=device
        ).repeat(max_bs)
        self._node_offsets = torch.arange(n, dtype=torch.int64, device=device)

    @property
    def num_nodes(self) -> int:
        return self.config.num_nodes

    def load_step(
        self, bs: int, req_pool_indices: torch.Tensor, parent_map: torch.Tensor
    ) -> None:
        """Read this step's trees from ``parent_map`` (``[pool, N]``, by pool
        slot) and derive depth and ancestor masks."""
        parent = self.parent_buf[:bs]
        torch.index_select(parent_map, 0, req_pool_indices, out=parent)
        tree_ancestry(
            parent,
            self.depth_buf[:bs],
            self.mask_buf[: bs * self.num_nodes].view(bs, self.num_nodes),
        )

    def depth_positions(self, bs: int, positions: torch.Tensor) -> None:
        """Turn the verify window's ``vc + i`` positions into ``vc + depth``."""
        view = positions.view(bs, self.num_nodes)
        view.add_(self.depth_buf[:bs] - self._node_offsets.to(view.dtype))

    def compact_rows(
        self,
        path: torch.Tensor,
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Pack each request's accepted path to the front of its verify window:
        target hidden rows, and the positions back to ``vc + i`` (row ``i``
        then holds the path node at depth ``i``). The attention backend moves
        the KV rows (``compact_verify_window``).

        Args:
            path: ``[bs, N]`` int32 accepted path, root first, ``-1`` past it.
            hidden: ``[bs * N, hidden]`` verify hidden rows, compacted in place.
            positions: ``[bs * N]`` int64 window positions, shifted in place.
        """
        bs, n = path.shape
        if bs == 0:
            return
        _compact_rows_kernel[(2, bs)](
            path,
            hidden,
            hidden.stride(0),
            positions,
            self.depth_buf,
            HIDDEN=hidden.shape[1],
            N=n,
            N_PAD=max(16, triton.next_power_of_2(n)),
            BLOCK=1024,
        )
