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
    acquire_symm_outputs,
    all_reduce,
    all_reduce_can_run,
    all_reduce_symm_can_run,
    all_reduce_symmetric,
    create_state,
    symm_outputs_can_run,
)
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.distributed.comm_backend.base import CommBackend, Group
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)

# Preserve the measured ordinary-Iris window while allowing a larger
# producer-direct backing allocation.
_DEFAULT_PRODUCER_DIRECT_MAX_BYTES = 1024 * 1024
_DEFAULT_ALL_REDUCE_MAX_BYTES = 512 * 1024


class TritonAllReduceBackend(CommBackend):
    def __init__(
        self,
        fallback: CommBackend,
        producer_direct_max_bytes: int = _DEFAULT_PRODUCER_DIRECT_MAX_BYTES,
    ):
        self._fallback = fallback
        self._instances = {}
        self._producer_direct_max_bytes = producer_direct_max_bytes
        self._max_numel = (
            min(producer_direct_max_bytes, _DEFAULT_ALL_REDUCE_MAX_BYTES)
            // torch.empty((), dtype=torch.bfloat16).element_size()
        )

    @property
    def producer_direct_max_bytes(self) -> int:
        return self._producer_direct_max_bytes

    def _get_or_create(self, group: Group):
        if group in self._instances:
            return self._instances[group]

        state = create_state(
            group=pg_manager.get_process_group("nccl", group),
            rank_in_group=group.index(dist.get_rank()),
            max_numel=self._max_numel,
            max_bytes=self._producer_direct_max_bytes,
            device=torch.device(f"cuda:{torch.cuda.current_device()}"),
        )
        self._instances[group] = state
        return state

    def can_run(self, tensor: torch.Tensor, group: Group, op=None) -> bool:
        if len(group) <= 1 or not current_platform().is_amd:
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

    def all_reduce(
        self,
        tensor: torch.Tensor | tuple[torch.Tensor, ...],
        group: Group,
        op=None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if not isinstance(tensor, torch.Tensor):
            if self.can_reduce_outputs(tensor, group, op=op):
                return all_reduce_symmetric(self._instances[group], tensor)
            return super().all_reduce(tensor, group, op=op)

        state = self._get_or_create(group)
        if all_reduce_can_run(state, tensor, op=op):
            return all_reduce(state, tensor, op=op)
        return self._fallback.all_reduce(tensor, group, op=op)

    def acquire_all_reduce_outputs(
        self,
        shapes: tuple[tuple[int, ...], ...],
        like: torch.Tensor,
        group: Group,
        op=None,
    ) -> tuple[torch.Tensor, ...]:
        """Acquire symmetric outputs when Iris supports the request."""
        if not self.can_acquire_outputs(shapes, like, group, op=op):
            return super().acquire_all_reduce_outputs(shapes, like, group, op=op)

        # Do not let one rank silently select a different collective protocol.
        state = self._get_or_create(group)
        return acquire_symm_outputs(state, shapes, like.dtype)

    def can_acquire_all_reduce_outputs(
        self,
        shapes: tuple[tuple[int, ...], ...],
        like: torch.Tensor,
        group: Group,
        op=None,
    ) -> bool:
        """Iris returns symmetric outputs the reduction consumes in place."""
        return self.can_acquire_outputs(shapes, like, group, op=op)

    def can_acquire_outputs(
        self,
        shapes: tuple[tuple[int, ...], ...],
        like: torch.Tensor,
        group: Group,
        op=None,
    ) -> bool:
        """Check producer-direct eligibility without initializing Iris."""
        if not current_platform().is_cdna4 or not like.is_cuda:
            return False
        state = self._get_or_create(group)
        return symm_outputs_can_run(state, shapes, like.dtype, op=op)

    def can_reduce_outputs(
        self,
        tensors: tuple[torch.Tensor, ...],
        group: Group,
        op=None,
    ) -> bool:
        """Check whether tensors are this group's symmetric outputs."""
        state = self._instances.get(group)
        return state is not None and all_reduce_symm_can_run(state, tensors, op=op)

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

    def all_to_all_single(
        self, output: torch.Tensor, input: torch.Tensor, group: Group
    ) -> None:
        return self._fallback.all_to_all_single(output, input, group)

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
