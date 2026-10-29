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

"""The draft trees a sampling backend verifies (--speculative-eagle-topk > 1)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["DraftTreeQ", "TreeVerifyBatch", "accepted_path_rows"]


@dataclass(frozen=True)
class DraftTreeQ:
    """Each pool slot's draft distributions behind its next-round tree (sampled
    trees) or chain (DFlash2 under --enable-speculative-sampling): the q a
    sampled node's siblings were drawn from, for the verifier's rejection
    sampling. Indexed by ``req_pool_idx``.

    Attributes:
        q: ``[rows, S, NC]`` fp32 q of each expanded row (slot) over its
            candidates.
        q_ids: ``[rows, S, NC]`` int32 the candidates' token ids.
        slot: ``[rows, N]`` int32 per tree node, the slot its parent's q is in;
            ``-1`` for the root and deterministic nodes.
        pos: ``[rows, N]`` int32 per tree node, its position in that q.
    """

    q: torch.Tensor
    q_ids: torch.Tensor
    slot: torch.Tensor
    pos: torch.Tensor

    @classmethod
    def zeros(
        cls,
        rows: int,
        num_slots: int,
        num_candidates: int,
        num_nodes: int,
        device: torch.device | str,
    ) -> DraftTreeQ:
        """Empty tables: every node deterministic (slot ``-1``)."""
        shape = (rows, num_slots, num_candidates)
        return cls(
            q=torch.zeros(shape, dtype=torch.float32, device=device),
            q_ids=torch.zeros(shape, dtype=torch.int32, device=device),
            slot=torch.full((rows, num_nodes), -1, dtype=torch.int32, device=device),
            pos=torch.zeros((rows, num_nodes), dtype=torch.int32, device=device),
        )

    def publish_chain(
        self,
        pool_indices: torch.Tensor,
        q: torch.Tensor,
        q_ids: torch.Tensor,
        pos: torch.Tensor,
        slot: torch.Tensor,
    ) -> None:
        """Record a sampled chain draft for the next round's verify.

        The chain layout: node 0 is the anchor (slot ``-1``); node ``i`` (draft
        token ``i``) reads slot ``i - 1``, which holds the q of draft step
        ``i - 1`` over its candidates, at position ``pos[:, i]``. A request
        without a proposal has ``-1`` on every node, which verify reads as
        "check the draft against the target's own draw" (the target-only
        rule). The DFlash2 walk (``dflash2_sample_path``) writes all four
        tables in this layout, so recording them is four ``index_copy_``
        calls: no temporaries, and it records into a captured graph.

        Args:
            pool_indices: ``[bs]`` int64 state-write pool slot per request;
                graph padding rows point at the reserved slot verify never
                reads.
            q: ``[bs, S, NC]`` fp32 each draft step's q over its candidates.
            q_ids: ``[bs, S, NC]`` int32 the candidates' token ids.
            pos: ``[bs, S + 1]`` int32 each node's position in its q.
            slot: ``[bs, S + 1]`` int32 each node's slot, or ``-1``.
        """
        if q.shape[1:] != self.q.shape[1:] or slot.shape[1:] != self.slot.shape[1:]:
            raise ValueError(
                f"a chain of q {tuple(q.shape[1:])} over {slot.shape[1]} nodes "
                f"does not fit tables of q {tuple(self.q.shape[1:])} and "
                f"{self.slot.shape[1]} nodes"
            )
        self.q.index_copy_(0, pool_indices, q)
        self.q_ids.index_copy_(0, pool_indices, q_ids)
        self.pos.index_copy_(0, pool_indices, pos)
        self.slot.index_copy_(0, pool_indices, slot)


@dataclass(frozen=True)
class TreeVerifyBatch:
    """This step's draft trees, one per request of the verify window.

    Attributes:
        parents: ``[bs, N]`` int32 parent node, ``-1`` for the root; a parent
            precedes its children.
        depths: ``[bs, N]`` int32 node depth, 0 for the root.
        draft_q: The draft distributions of sampled trees (None: every node
            deterministic), read at the requests' pool slots.
    """

    parents: torch.Tensor
    depths: torch.Tensor
    draft_q: DraftTreeQ | None


def accepted_path_rows(path: torch.Tensor) -> torch.Tensor:
    """``[bs * N]`` window row of each packed prediction: the accepted path
    first, then identity.

    Args:
        path: ``[bs, N]`` accepted path, root first, ``-1`` past it.
    """
    bs, n = path.shape
    path = path.long()
    nodes = torch.arange(n, device=path.device)
    local = torch.where(path >= 0, path, nodes)
    return (local + torch.arange(bs, device=path.device)[:, None] * n).view(-1)
