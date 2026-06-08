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
from typing import Literal

import torch

from tokenspeed.runtime.distributed.comm_backend import Group
from tokenspeed.runtime.distributed.dp_sampling_comm import DpSamplingComm


@dataclasses.dataclass(frozen=True)
class LogitsLayoutPlan:
    mode: Literal["normal", "dp_all_to_all"]
    real_bs: int
    effective_bs: int
    bucket_bs: int
    tp_size: int
    num_tokens_per_req: int

    @property
    def is_dp_all_to_all(self) -> bool:
        return self.mode == "dp_all_to_all"

    @classmethod
    def normal(
        cls,
        *,
        real_bs: int,
        effective_bs: int,
        bucket_bs: int,
        tp_size: int,
        num_tokens_per_req: int,
    ) -> "LogitsLayoutPlan":
        return cls(
            mode="normal",
            real_bs=real_bs,
            effective_bs=effective_bs,
            bucket_bs=bucket_bs,
            tp_size=tp_size,
            num_tokens_per_req=num_tokens_per_req,
        )

    @classmethod
    def dp_all_to_all(
        cls,
        *,
        real_bs: int,
        effective_bs: int,
        bucket_bs: int,
        tp_size: int,
        num_tokens_per_req: int,
    ) -> "LogitsLayoutPlan":
        return cls(
            mode="dp_all_to_all",
            real_bs=real_bs,
            effective_bs=effective_bs,
            bucket_bs=bucket_bs,
            tp_size=tp_size,
            num_tokens_per_req=num_tokens_per_req,
        )


def resolve_dp_sampling_min_bs(
    tp_size: int,
    configured_min_bs: int | None,
    *,
    num_tokens_per_req: int | None = None,
    pdl_enabled: bool | None = None,
) -> int:
    if configured_min_bs is not None:
        min_bs = int(configured_min_bs)
    else:
        min_bs = 2 * tp_size
    if min_bs < 1:
        raise ValueError("dp_sampling_min_bs must be >= 1")
    return min_bs


def should_use_dp_sampling_for_bucket(
    *,
    dp_sampling_enabled: bool,
    forward_mode,
    effective_bs: int,
    min_bs: int,
) -> bool:
    return (
        dp_sampling_enabled
        and forward_mode is not None
        and (forward_mode.is_decode() or forward_mode.is_target_verify())
        and effective_bs >= min_bs
    )


@dataclasses.dataclass(frozen=True)
class LogitsLayoutPlanner:
    dp_sampling_enabled: bool
    dp_sampling_min_bs: int
    tp_size: int
    num_tokens_per_req: int

    @classmethod
    def from_settings(
        cls,
        *,
        dp_sampling_enabled: bool,
        configured_min_bs: int | None,
        tp_size: int,
        num_tokens_per_req: int,
    ) -> "LogitsLayoutPlanner":
        return cls(
            dp_sampling_enabled=dp_sampling_enabled,
            dp_sampling_min_bs=resolve_dp_sampling_min_bs(
                tp_size=tp_size,
                configured_min_bs=configured_min_bs,
            ),
            tp_size=tp_size,
            num_tokens_per_req=num_tokens_per_req,
        )

    def build_plan(
        self,
        *,
        forward_mode,
        real_bs: int,
        effective_bs: int,
    ) -> LogitsLayoutPlan:
        if should_use_dp_sampling_for_bucket(
            dp_sampling_enabled=self.dp_sampling_enabled,
            forward_mode=forward_mode,
            effective_bs=effective_bs,
            min_bs=self.dp_sampling_min_bs,
        ):
            bucket_bs = (
                (effective_bs + self.tp_size - 1) // self.tp_size
            ) * self.tp_size
            return LogitsLayoutPlan.dp_all_to_all(
                real_bs=real_bs,
                effective_bs=effective_bs,
                bucket_bs=bucket_bs,
                tp_size=self.tp_size,
                num_tokens_per_req=self.num_tokens_per_req,
            )

        return LogitsLayoutPlan.normal(
            real_bs=real_bs,
            effective_bs=effective_bs,
            bucket_bs=effective_bs,
            tp_size=self.tp_size,
            num_tokens_per_req=self.num_tokens_per_req,
        )


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
        self._tp_group = tp_group
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
        n = self._validate_plan(plan)
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
        n = self._validate_plan(plan)
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

    def _validate_plan(self, plan: LogitsLayoutPlan) -> int:
        assert plan.is_dp_all_to_all
        assert (
            plan.tp_size == self._tp_size
        ), f"plan tp_size={plan.tp_size} != executor tp_size={self._tp_size}"
        assert plan.num_tokens_per_req == self._num_tokens_per_req, (
            f"plan N={plan.num_tokens_per_req} != executor "
            f"N={self._num_tokens_per_req}"
        )
        assert (
            plan.effective_bs >= plan.real_bs
        ), f"effective_bs={plan.effective_bs} must be >= real_bs={plan.real_bs}"
        assert (
            plan.bucket_bs >= plan.effective_bs
        ), f"bucket_bs={plan.bucket_bs} must be >= effective_bs={plan.effective_bs}"
        assert (
            plan.bucket_bs % self._tp_size == 0
        ), f"bucket_bs={plan.bucket_bs} must be divisible by tp_size={self._tp_size}"
        return plan.num_tokens_per_req
