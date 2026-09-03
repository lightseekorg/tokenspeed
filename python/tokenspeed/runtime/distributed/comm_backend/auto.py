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

Wraps NCCL and optional low-latency GPU backends. CUDA IPC and symmetric-memory
backends are only selected for node-local groups; groups spanning nodes fall
back to NCCL.
"""

import torch
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.distributed.comm_backend.base import (
    CommBackend,
    Group,
)
from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
from tokenspeed.runtime.distributed.comm_backend.triton_allreduce import (
    TritonAllReduceBackend,
)
from tokenspeed.runtime.distributed.comm_backend.triton_rsag import TritonRSAGBackend
from tokenspeed.runtime.distributed.comm_backend.trtllm_allreduce import (
    MAX_ONESHOT_BYTES,
    TrtllmAllReduceBackend,
)
from tokenspeed.runtime.utils.env import global_server_args_dict


class AutoBackend(CommBackend):
    """Composite backend that selects the best strategy per call."""

    def __init__(self):
        self._nccl = NcclBackend()
        self._trtllm_ar = TrtllmAllReduceBackend(fallback=self._nccl)
        self._triton_ar = TritonAllReduceBackend(fallback=self._nccl)
        self._rsag = TritonRSAGBackend(fallback=self._nccl)

    @property
    def nccl(self) -> NcclBackend:
        return self._nccl

    @property
    def trtllm_ar(self) -> TrtllmAllReduceBackend:
        return self._trtllm_ar

    def configure(self, use_pynccl: bool = False) -> None:
        self._nccl.configure(use_pynccl=use_pynccl)

    @staticmethod
    def _force_deterministic_rsag() -> bool:
        return bool(global_server_args_dict.get("force_deterministic_rsag", False))

    @staticmethod
    def _group_spans_nodes(group: Group) -> bool:
        mapping = global_server_args_dict.get("mapping")
        nprocs_per_node = getattr(mapping, "nprocs_per_node", None)
        if not nprocs_per_node:
            return False
        return len({rank // nprocs_per_node for rank in group}) > 1

    @staticmethod
    def _multicast_reachable(group: Group) -> bool:
        """Whether symmetric-memory multicast can map across ``group``.

        The rsag paths rendezvous a symmetric buffer and store through its
        multicast pointer, so a group the fabric cannot map hangs inside the
        rendezvous instead of falling back. Topology alone was too strict --
        a rack's NVLink domain can span hosts, and vetoing on spread gives
        those groups NCCL forever -- so it now only admits, and anything
        crossing a host is probed rather than refused.

        The rank count is not a substitute for the topology test. ``Mapping``
        builds strided groups: an attention DP group is ``(0, 8)`` at
        ``attn_tp_size=8``, which is smaller than one host's device count while
        living on two hosts, so counting would admit it with no probe at all.

        The world fabric map is gathered during distributed initialization, so
        the group verdict is a local lookup with no dispatch-time collective.
        """
        from tokenspeed_kernel.ops.communication.fabric import group_has_fabric

        if not AutoBackend._group_spans_nodes(group):
            return True
        return group_has_fabric(group)

    # ---- Token-aware ops ----

    def token_all_gather(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor:
        if self._force_deterministic_rsag() or not self._multicast_reachable(group):
            return self._nccl.token_all_gather(tensor, group, scattered_num_tokens)
        return self._rsag.token_all_gather(tensor, group, scattered_num_tokens)

    def token_reduce_scatter(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor:
        if self._force_deterministic_rsag() or not self._multicast_reachable(group):
            return self._nccl.token_reduce_scatter(tensor, group, scattered_num_tokens)
        return self._rsag.token_reduce_scatter(tensor, group, scattered_num_tokens)

    # ---- Public CommBackend interface ----

    def all_reduce(
        self,
        tensor: torch.Tensor | tuple[torch.Tensor, ...],
        group: Group,
        op=None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if not isinstance(tensor, torch.Tensor):
            tensors = tensor
            if len(tensors) == 0:
                raise ValueError("all-reduce requires at least one tensor")
            use_nccl = self._force_deterministic_rsag() or self._group_spans_nodes(
                group
            )
            # Collections past the one-shot window are headed for NCCL;
            # grouping avoids the copy required to concatenate them first.
            use_nccl = use_nccl or all(
                value.numel() * value.element_size() > MAX_ONESHOT_BYTES
                for value in tensors
            )
            use_nccl = use_nccl or (
                current_platform().is_amd
                and sum(value.numel() * value.element_size() for value in tensors)
                > self._triton_ar.producer_direct_max_bytes
            )
            if (
                not use_nccl
                and current_platform().is_amd
                and self._triton_ar.can_reduce_outputs(tensors, group, op=op)
            ):
                return self._triton_ar.all_reduce(tensors, group, op=op)
            if use_nccl and len(tensors) == 2:
                return self._nccl.all_reduce_two(*tensors, group, op=op)
            return super().all_reduce(tensors, group, op=op)

        # AR backend dispatch -- first match wins. This is Tier 1 (which
        # backend); the trtllm backend then runs Tier 2 (mnnvl vs IPC, by
        # payload bytes) inside _ar_fusion_workspace.
        #   1. force_deterministic_rsag ............ NCCL
        #   2. trtllm_ar armed for this group ...... trtllm_ar   (mnnvl / IPC fusion)
        #   3. group spans nodes ................... NCCL
        #   4. triton_ar can run ................... triton_ar
        #   5. otherwise ........................... NCCL
        if self._force_deterministic_rsag():
            return self._nccl.all_reduce(tensor, group, op=op)
        spans_nodes = self._group_spans_nodes(group)
        # trtllm_ar carries an mnnvl workspace that spans nodes; it is only
        # armed for a group when that succeeded, so has_trtllm_ar() is itself
        # the "usable here" test. Checking it before the cross-node NCCL
        # fallback is what lets a cross-node group use mnnvl at all -- otherwise
        # the workspace is armed and never called.
        if self._trtllm_ar.has_trtllm_ar(group):
            return self._trtllm_ar.all_reduce(tensor, group, op=op)
        if spans_nodes:
            return self._nccl.all_reduce(tensor, group, op=op)
        if self._triton_ar.can_run(tensor, group, op=op):
            return self._triton_ar.all_reduce(tensor, group, op=op)
        return self._nccl.all_reduce(tensor, group, op=op)

    def prepare_all_reduce_lane(self, group: Group, hidden_dim: int) -> bool:
        return self._trtllm_ar.ensure_group_lane(group, hidden_dim)

    def acquire_all_reduce_outputs(
        self,
        shapes: tuple[tuple[int, ...], ...],
        like: torch.Tensor,
        group: Group,
        op=None,
    ) -> tuple[torch.Tensor, ...]:
        """Acquire ordinary or producer-direct all-reduce outputs."""
        if (
            self._force_deterministic_rsag()
            or self._group_spans_nodes(group)
            or self._trtllm_ar.has_trtllm_ar(group)
        ):
            return super().acquire_all_reduce_outputs(shapes, like, group, op=op)
        if current_platform().is_amd and not self._triton_ar.can_acquire_outputs(
            shapes,
            like,
            group,
            op=op,
        ):
            return super().acquire_all_reduce_outputs(shapes, like, group, op=op)
        return self._triton_ar.acquire_all_reduce_outputs(
            shapes,
            like,
            group,
            op=op,
        )

    def all_gather(
        self, tensor: torch.Tensor, group: Group, dim: int = 0
    ) -> torch.Tensor:
        if self._force_deterministic_rsag() or not self._multicast_reachable(group):
            return self._nccl.all_gather(tensor, group, dim)
        if tensor.dim() == 2 and dim in (-1, tensor.dim() - 1):
            return self._rsag.all_gather(tensor, group, dim)

        return self._nccl.all_gather(tensor, group, dim)

    def all_gather_into_tensor(
        self, output: torch.Tensor, input: torch.Tensor, group: Group
    ) -> None:
        return self._nccl.all_gather_into_tensor(output, input, group)

    def reduce_scatter(self, tensor: torch.Tensor, group: Group) -> torch.Tensor:
        return self._nccl.reduce_scatter(tensor, group)

    def all_to_all_single(
        self, output: torch.Tensor, input: torch.Tensor, group: Group
    ) -> None:
        return self._nccl.all_to_all_single(output, input, group)

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
