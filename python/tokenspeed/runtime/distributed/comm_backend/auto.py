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

"""Auto backend: per-call strategy selection.

Wraps NcclBackend and optionally CustomAllReduceBackend and
the fused all-reduce backend. For all_reduce, selects the lowest-latency
backend based on tensor size and hardware.  For other ops, always uses
NCCL.
"""

import torch

from tokenspeed.runtime.distributed.comm_backend.base import CommBackend, Group
from tokenspeed.runtime.distributed.comm_backend.custom_allreduce import (
    CustomAllReduceBackend,
)
from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
from tokenspeed.runtime.distributed.comm_backend.triton_allreduce import (
    TritonAllReduceBackend,
)
from tokenspeed.runtime.distributed.comm_backend.triton_rsag import TritonRSAGBackend
from tokenspeed.runtime.distributed.comm_backend.trtllm_allreduce import (
    TrtllmAllReduceBackend,
)


class AutoBackend(CommBackend):
    """Composite backend that selects the best strategy per call."""

    def __init__(self):
        self._nccl = NcclBackend()
        self._custom_ar = CustomAllReduceBackend(fallback=self._nccl)
        self._trtllm_ar = TrtllmAllReduceBackend(fallback=self._nccl)
        self._triton_ar = TritonAllReduceBackend(fallback=self._nccl)
        self._rsag = TritonRSAGBackend(fallback=self._nccl)

    @property
    def nccl(self) -> NcclBackend:
        return self._nccl

    @property
    def custom_ar(self) -> CustomAllReduceBackend:
        return self._custom_ar

    @property
    def trtllm_ar(self) -> TrtllmAllReduceBackend:
        return self._trtllm_ar

    def configure(
        self, use_pynccl: bool = False, use_custom_allreduce: bool = False
    ) -> None:
        self._nccl.configure(use_pynccl=use_pynccl)
        self._custom_ar.configure(use_custom_allreduce=use_custom_allreduce)

    # ---- Token-aware ops ----

    def token_all_gather(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor:
        return self._rsag.token_all_gather(tensor, group, scattered_num_tokens)

    def token_reduce_scatter(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor:
        return self._rsag.token_reduce_scatter(tensor, group, scattered_num_tokens)

    # ---- Public CommBackend interface ----

    def all_reduce(self, tensor: torch.Tensor, group: Group, op=None) -> torch.Tensor:
        if self._custom_ar.has_custom_ar(group):
            return self._custom_ar.all_reduce(tensor, group, op=op)
        if self._trtllm_ar.has_trtllm_ar(group):
            return self._trtllm_ar.all_reduce(tensor, group, op=op)
        if self._triton_ar.can_run(tensor, group, op=op):
            return self._triton_ar.all_reduce(tensor, group, op=op)
        return self._nccl.all_reduce(tensor, group, op=op)

    def all_gather(
        self, tensor: torch.Tensor, group: Group, dim: int = 0
    ) -> torch.Tensor:
        if tensor.dim() == 2 and dim in (-1, tensor.dim() - 1):
            return self._rsag.all_gather(tensor, group, dim)

        return self._nccl.all_gather(tensor, group, dim)

    def all_gather_into_tensor(
        self, output: torch.Tensor, input: torch.Tensor, group: Group
    ) -> None:
        return self._nccl.all_gather_into_tensor(output, input, group)

    def reduce_scatter(self, tensor: torch.Tensor, group: Group) -> torch.Tensor:
        return self._nccl.reduce_scatter(tensor, group)

    def send(self, tensor: torch.Tensor, dst: int, group: Group) -> None:
        return self._nccl.send(tensor, dst, group)

    def recv(
        self,
        size: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        src: int,
        group: Group,
    ) -> torch.Tensor:
        return self._nccl.recv(size, dtype, device, src, group)
