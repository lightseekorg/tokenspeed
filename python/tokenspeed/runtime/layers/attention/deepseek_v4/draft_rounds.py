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

"""The V4 draft's per-step decode metadata, as pointer-stable per-bs views.

A V4 MTP draft alternates two row shapes each round: the packed bs*N verify
step (step 0, served by the GraphBuffers views) and the plain bs-row steps
1+ that this object serves. The plain-step metadata is one pointer-stable
view per bs, exactly like the unified decode path's per-bs views. ``prepare``
and ``advance`` are copy-only plus the sanctioned ``cache``-slot swap — no
arm ever rebinds a tensor field, so there is nothing a captured graph could
have recorded that a later round can invalidate, on the graph path and the
eager path alike (they are the same object).

Storage-sharing rules with :class:`DeepseekV4GraphBuffers` (request-major
row state is safe to share, token-major packed state is NOT):

* ``query_start_loc`` / ``token_to_req_indices`` — shared width-1 constant
  arange tables (``graph.query_start_by_width[1]`` etc.), never mutated.
* ``seq_lens`` — OWN buffer: ``advance`` records a ``copy_`` of the
  drafter's lengths into it inside the graph; ``graph.seq_lens`` carries the
  packed step's lengths and must not move underneath it.
* ``is_valid_token`` — OWN buffer: the per-request gather of the packed
  token-major mask; sharing ``graph.is_valid_token`` would alias rows with
  wrong values.
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.deepseek_v4.graph_buffers import (
    DeepseekV4GraphBuffers,
)
from tokenspeed.runtime.layers.attention.deepseek_v4.metadata import (
    DeepseekV4ForwardMetadata,
)


class DeepseekV4DraftRounds:
    """Owner of the draft's plain-step (bs-row) decode metadata views."""

    def __init__(self, graph: DeepseekV4GraphBuffers) -> None:
        self._graph = graph
        device = graph.seq_lens.device
        max_bs = graph.max_bs
        # Own row state (see the module docstring's sharing rules).
        self.step_seq_lens = torch.zeros(max_bs, dtype=torch.int32, device=device)
        self.step_valid = torch.ones(max_bs, dtype=torch.bool, device=device)
        # Width-1 query lens are a constant the graph never mutates; own a
        # dedicated ones table so packed refreshes (which fill
        # graph.query_lens with N) cannot disturb the recorded step views.
        self._step_query_lens = torch.ones(max_bs, dtype=torch.int32, device=device)
        self._views: dict[int, DeepseekV4ForwardMetadata] = {}
        # The round's live step metadata; None before the first prepare.
        self.current: DeepseekV4ForwardMetadata | None = None

    def step_views(self, bs: int) -> DeepseekV4ForwardMetadata:
        """The pointer-stable plain-step views for ``bs`` (built once)."""
        metadata = self._views.get(bs)
        if metadata is not None:
            return metadata
        graph = self._graph
        metadata = DeepseekV4ForwardMetadata(
            seq_lens=self.step_seq_lens[:bs],
            query_lens=self._step_query_lens[:bs],
            query_start_loc=graph.query_start_by_width[1][: bs + 1],
            token_to_req_indices=graph.token_to_req_by_width[1][:bs],
            cache=None,
            is_valid_token=self.step_valid[:bs],
            forward_mode=ForwardMode.DECODE,
        )
        self._views[bs] = metadata
        return metadata

    def prepare(
        self,
        prefill_metadata: DeepseekV4ForwardMetadata,
        base_seq_lens: torch.Tensor,
    ) -> DeepseekV4ForwardMetadata:
        """Bind this round's step metadata from the packed/prefill state.

        Copy-only over the per-bs views plus the sanctioned ``cache`` swap
        (the guard-exempt slot); returns the views for the backend to run
        the indexer/slot-mapping refresh hooks over and publish.
        """
        bs = prefill_metadata.seq_lens.numel()
        metadata = self.step_views(bs)
        metadata.cache = prefill_metadata.cache
        self.step_seq_lens[:bs].copy_(base_seq_lens[:bs])
        packed_mask = prefill_metadata.is_valid_token
        if packed_mask is None:
            self.step_valid[:bs].fill_(True)
        else:
            self.step_valid[:bs].copy_(
                packed_mask[prefill_metadata.query_start_loc[:bs].to(torch.int64)]
            )
        metadata.num_prefill_reqs = 0
        metadata.num_prefill_tokens = 0
        metadata.forward_mode = ForwardMode.DECODE
        self.current = metadata
        return metadata

    def advance(self, seq_lens: torch.Tensor) -> DeepseekV4ForwardMetadata:
        """Advance the live step metadata to the next draft step in place."""
        metadata = self.current
        if metadata is None:
            raise RuntimeError("DeepSeek V4 draft metadata was not initialized")
        metadata.seq_lens.copy_(seq_lens[: metadata.seq_lens.numel()])
        metadata.forward_mode = ForwardMode.DECODE
        return metadata
