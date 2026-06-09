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

"""Declarative logits layout plans used by sampling."""

from __future__ import annotations

import dataclasses

import torch

from tokenspeed.runtime.distributed.comm_backend import Group
from tokenspeed.runtime.distributed.dp_sampling_comm import DpSamplingComm


@dataclasses.dataclass(frozen=True)
class LogitsLayoutPlan:
    effective_bs: int
    bucket_bs: int
    tp_size: int
    num_tokens_per_req: int


class LogitsLayoutExecutor:
    """Executes sampling-provided logits layout plans."""

    def __init__(
        self,
        *,
        tp_rank: int,
        tp_size: int,
        tp_group: Group,
        max_bucket_bs: int,
        num_tokens_per_req: int,
        vocab_size: int,
        device: torch.device | str,
    ) -> None:
        self._tp_rank = tp_rank
        self._tp_size = tp_size
        self._num_tokens_per_req = num_tokens_per_req
        self._comm = DpSamplingComm(
            tp_size=tp_size,
            rank=tp_rank,
            group=tp_group,
            max_pad_bs=max_bucket_bs,
            num_tokens_per_req=num_tokens_per_req,
            vocab_size=vocab_size,
            logits_dtype=None,
            device=device,
        )

    def slice_hidden_states(
        self,
        hidden_states: torch.Tensor,
        plan: LogitsLayoutPlan,
    ) -> torch.Tensor:
        n = self._tokens_per_req(plan)
        rows = hidden_states.shape[0]
        assert rows % n == 0, f"hidden_states have {rows} rows, not divisible by N={n}"
        bs = rows // n
        assert bs == plan.effective_bs, (
            f"hidden_states imply effective_bs={bs}, but logits layout plan has "
            f"effective_bs={plan.effective_bs}"
        )
        pad_rows = (plan.bucket_bs - plan.effective_bs) * n
        if pad_rows > 0:
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, pad_rows))
        reqs_per_rank = plan.bucket_bs // self._tp_size
        start = self._tp_rank * reqs_per_rank * n
        return hidden_states[start : start + reqs_per_rank * n]

    def swap_batch_vocab(
        self,
        local_logits: torch.Tensor,
        plan: LogitsLayoutPlan,
    ) -> torch.Tensor:
        n = self._tokens_per_req(plan)
        rows = local_logits.shape[0]
        assert rows % n == 0, f"local logits have {rows} rows, not divisible by N={n}"
        bs = rows // n
        assert bs == plan.effective_bs, (
            f"local logits imply effective_bs={bs}, but logits layout plan has "
            f"effective_bs={plan.effective_bs}"
        )
        pad_rows = (plan.bucket_bs - plan.effective_bs) * n
        if pad_rows > 0:
            local_logits = torch.nn.functional.pad(local_logits, (0, 0, 0, pad_rows))
        return self._comm.swap_batch_vocab(local_logits, pad_bs=plan.bucket_bs)

    def _tokens_per_req(self, plan: LogitsLayoutPlan) -> int:
        if (
            plan.tp_size != self._tp_size
            or plan.num_tokens_per_req != self._num_tokens_per_req
            or plan.bucket_bs < plan.effective_bs
            or plan.bucket_bs % self._tp_size != 0
        ):
            raise RuntimeError("invalid DP logits layout plan")
        return plan.num_tokens_per_req
