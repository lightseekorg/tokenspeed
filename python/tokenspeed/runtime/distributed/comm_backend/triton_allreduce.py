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

"""Triton all-reduce backend for latency-sensitive small AMD tensors."""

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.triton import (
    all_reduce,
    all_reduce_can_run,
    create_state,
)

from tokenspeed.runtime.distributed.comm_backend.base import CommBackend, Group
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)


class TritonAllReduceBackend(CommBackend):
    def __init__(self, fallback: CommBackend):
        self._fallback = fallback
        self._instances = {}
        self._max_bytes = 512 * 1024
        self._max_numel = (
            self._max_bytes // torch.empty((), dtype=torch.bfloat16).element_size()
        )

    def _get_or_create(self, group: Group):
        if group in self._instances:
            return self._instances[group]

        state = create_state(
            group=pg_manager.get_process_group("nccl", group),
            rank_in_group=group.index(dist.get_rank()),
            max_numel=self._max_numel,
            device=torch.device(f"cuda:{torch.cuda.current_device()}"),
        )
        self._instances[group] = state
        return state

    def can_run(self, tensor: torch.Tensor, group: Group, op=None) -> bool:
        if len(group) <= 1:
            return False
        if op is None:
            op = torch.distributed.ReduceOp.SUM
        if not (
            op == torch.distributed.ReduceOp.SUM
            and tensor.is_cuda
            and tensor.is_contiguous()
            and tensor.dtype == torch.bfloat16
            and 0 < tensor.numel() <= self._max_numel
        ):
            return False
        try:
            return all_reduce_can_run(self._get_or_create(group), tensor, op=op)
        except Exception:
            return False

    def all_reduce(self, tensor: torch.Tensor, group: Group, op=None) -> torch.Tensor:
        state = self._get_or_create(group)
        if all_reduce_can_run(state, tensor, op=op):
            return all_reduce(state, tensor, op=op)
        return self._fallback.all_reduce(tensor, group, op=op)

    def all_gather(
        self, tensor: torch.Tensor, group: Group, dim: int = 0
    ) -> torch.Tensor:
        return self._fallback.all_gather(tensor, group, dim)

    def all_gather_into_tensor(
        self, output: torch.Tensor, input: torch.Tensor, group: Group
    ) -> None:
        return self._fallback.all_gather_into_tensor(output, input, group)

    def reduce_scatter(self, tensor: torch.Tensor, group: Group) -> torch.Tensor:
        return self._fallback.reduce_scatter(tensor, group)

    def token_all_gather(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor:
        raise NotImplementedError("Use AutoBackend for token-aware ops")

    def token_reduce_scatter(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor:
        raise NotImplementedError("Use AutoBackend for token-aware ops")
