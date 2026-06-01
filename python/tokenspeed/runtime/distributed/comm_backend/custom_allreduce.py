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

"""Custom all-reduce backend using P2P GPU shared memory.

Only supports all_reduce. Other ops delegate to a fallback backend.
"""

from contextlib import nullcontext

import torch

from tokenspeed.runtime.distributed.comm_backend.base import CommBackend, Group


class CustomAllReduceBackend(CommBackend):
    """Backend using custom P2P all-reduce (NVLink shared memory).

    Maintains per-group ca_comm in an internal registry, keyed by group tuple.
    Falls back to the provided fallback backend for ops other than
    all_reduce, or when the tensor is not eligible for custom AR.
    """

    def __init__(self, fallback: CommBackend):
        self._fallback = fallback
        self._resources = {}  # group_tuple → {ca_comm}
        self._use_custom_allreduce = False

    def configure(self, use_custom_allreduce: bool = False) -> None:
        self._use_custom_allreduce = use_custom_allreduce

    def _get_or_create_resources(self, group: Group):
        if group in self._resources:
            return self._resources[group]

        ca_comm = None
        if self._use_custom_allreduce and len(group) > 1:
            try:
                from tokenspeed.runtime.distributed.device_communicators.custom_all_reduce import (
                    CustomAllreduce,
                )
                from tokenspeed.runtime.distributed.process_group_manager import (
                    process_group_manager as pg_manager,
                )

                gloo_group = pg_manager.get_process_group("gloo", group)
                ca_comm = CustomAllreduce(
                    group=gloo_group,
                    device=torch.device(f"cuda:{torch.cuda.current_device()}"),
                )
            except Exception:
                ca_comm = None

        self._resources[group] = {"ca_comm": ca_comm}
        return self._resources[group]

    def has_custom_ar(self, group: Group) -> bool:
        if group not in self._resources:
            return False
        res = self._resources[group]
        ca_comm = res["ca_comm"]
        return ca_comm is not None and not ca_comm.disabled

    def capture(self, group: Group):
        res = self._get_or_create_resources(group)
        ca_comm = res["ca_comm"]
        if ca_comm is None or ca_comm.disabled:
            return nullcontext()
        return ca_comm.capture()

    # ---- Public CommBackend interface ----

    def all_reduce(self, tensor: torch.Tensor, group: Group, op=None) -> torch.Tensor:
        if op is None:
            op = torch.distributed.ReduceOp.SUM
        res = self._get_or_create_resources(group)
        ca_comm = res["ca_comm"]
        if (
            op == torch.distributed.ReduceOp.SUM
            and ca_comm is not None
            and not ca_comm.disabled
            and ca_comm.should_custom_ar(tensor)
        ):
            out = ca_comm.custom_all_reduce(tensor)
            assert out is not None
            return out
        else:
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
