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

"""Model-free deterministic candidates for target multi-token verification."""

from __future__ import annotations

import torch
from typing_extensions import override

from tokenspeed.runtime.execution.drafter.base import BaseDrafter


class DummyDrafter(BaseDrafter):
    """Generate a repeatable candidate window without draft weights or cache."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.spec_num_tokens <= 1:
            raise ValueError("DUMMY drafting requires at least two verify tokens")
        if self.runtime_states is None or self.input_buffers is None:
            raise ValueError("DUMMY drafting requires runtime and input buffers")
        if self.vocab_size is None or self.vocab_size <= 0:
            raise ValueError("DUMMY drafting requires a positive vocabulary size")
        if self.draft_model_runner is not None:
            raise ValueError("DUMMY drafting must not load a draft model")

    @override
    def idle_forward_global_num_tokens(
        self, global_num_tokens: list[int], global_bs: list[int]
    ) -> list[list[int]]:
        del global_num_tokens, global_bs
        return []

    def _selected_tokens(
        self,
        base_ctx,
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
    ) -> torch.Tensor:
        bs = base_ctx.bs
        num_extends = base_ctx.num_extends
        selected = torch.empty(bs, dtype=torch.int32, device=output_tokens.device)
        if num_extends:
            selected[:num_extends].copy_(output_tokens[:num_extends])
        num_decodes = bs - num_extends
        if num_decodes:
            decoded = output_tokens[num_extends:].view(
                num_decodes, self.spec_num_tokens
            )
            accepted = (
                accept_lengths[num_extends:]
                .to(torch.int64)
                .sub(1)
                .clamp(0, self.spec_num_tokens - 1)
            )
            selected[num_extends:].copy_(
                decoded.gather(1, accepted[:, None]).squeeze(1)
            )
        return selected

    def _candidate_window(
        self,
        selected: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ) -> torch.Tensor:
        bs = selected.shape[0]
        window = torch.empty(
            (bs, self.spec_num_tokens), dtype=torch.int32, device=selected.device
        )
        window[:, 0].copy_(selected)
        slots = req_pool_indices[:bs].to(torch.int64)
        cache_lengths = self.runtime_states.valid_cache_lengths.index_select(0, slots)
        columns = torch.arange(
            1, self.spec_num_tokens, dtype=torch.int64, device=selected.device
        )
        # No rank enters this formula: every TP rank proposes the same window.
        # Live inputs also avoid freezing random candidates into a captured graph.
        base = (
            selected.to(torch.int64) * 1_000_003
            + slots * 97_409
            + cache_lengths.to(torch.int64) * 65_537
            + 17
        )
        candidates = torch.remainder(
            base[:, None] + columns[None, :] * 104_729,
            int(self.vocab_size),
        )
        window[:, 1:].copy_(candidates.to(torch.int32))
        return window

    @override
    def run(
        self,
        base_ctx,
        logits_output,
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
    ) -> torch.Tensor:
        del logits_output
        selected = self._selected_tokens(base_ctx, output_tokens, accept_lengths)
        return self._candidate_window(
            selected,
            self.input_buffers.req_pool_indices_buf[: base_ctx.bs],
        )

    @override
    def draft(self, *args, **kwargs) -> torch.Tensor | None:
        del args, kwargs
        raise RuntimeError("DummyDrafter generates candidates through run()")
