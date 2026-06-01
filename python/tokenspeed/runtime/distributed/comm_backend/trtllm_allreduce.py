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

"""Lamport 1-shot all-reduce backend.

Uses an IPC workspace with Lamport barriers and shared memory for low-latency
all-reduce on small tensors. Falls back to a provided fallback backend for
large tensors or unsupported ops.

The workspace is created once per group via ``configure_group`` and
reused for every subsequent ``all_reduce`` on that group.
"""

import torch
from tokenspeed_kernel.ops.communication.trtllm import (
    AllReduceFusionPattern,
    trtllm_allreduce_fusion,
    trtllm_create_ipc_workspace_for_all_reduce_fusion,
)
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.distributed.comm_backend.base import CommBackend, Group

_MAX_ONESHOT_BYTES = 2 * 1024 * 1024


class TrtllmAllReduceBackend(CommBackend):
    """Backend using Lamport 1-shot all-reduce.

    Keyed per-group: each group gets its own IPC workspace so handles
    are never reused across groups.  Only ``all_reduce`` (SUM) is
    accelerated; every other op delegates to *fallback*.
    """

    def __init__(self, fallback: CommBackend):
        self._fallback = fallback
        self._resources = {}  # group_tuple → {workspace, rank, world_size}

    def _load_comm(self):
        return current_platform().is_nvidia

    # ------------------------------------------------------------------
    # Group configuration
    # ------------------------------------------------------------------

    def configure_group(
        self,
        rank: int,
        group: Group,
        max_token_num: int,
        hidden_dim: int,
        use_fp32_lamport: bool = False,
    ) -> bool:
        """Create IPC workspace for *group*.  Returns True on success."""
        if group in self._resources:
            return True

        if not self._load_comm():
            return False

        try:

            from tokenspeed.runtime.distributed.process_group_manager import (
                process_group_manager as pg_manager,
            )

            device_group = pg_manager.get_process_group("nccl", group)

            ipc_handles, workspace_tensor = (
                trtllm_create_ipc_workspace_for_all_reduce_fusion(
                    rank,
                    len(group),
                    max_token_num,
                    hidden_dim,
                    group=device_group,
                    use_fp32_lamport=use_fp32_lamport,
                )
            )

            self._resources[group] = {
                "ipc_handles": ipc_handles,
                "workspace": workspace_tensor,
                "rank": rank,
                "world_size": len(group),
                "max_token_num": max_token_num,
                "hidden_dim": hidden_dim,
                "device_group": device_group,
            }

            return True

        except Exception:

            return False

    def has_trtllm_ar(self, group: Group) -> bool:
        return group in self._resources

    # ------------------------------------------------------------------
    # CommBackend interface
    # ------------------------------------------------------------------

    def all_reduce(self, tensor: torch.Tensor, group: Group, op=None) -> torch.Tensor:

        if op is None:
            op = torch.distributed.ReduceOp.SUM

        res = self._resources.get(group)

        if (
            res is not None
            and op == torch.distributed.ReduceOp.SUM
            and tensor.numel() * tensor.element_size() <= _MAX_ONESHOT_BYTES
        ):

            result = self._lamport_allreduce(tensor, res)

            if result is not None:
                return result

        return self._fallback.all_reduce(tensor, group, op=op)

    def _lamport_allreduce(
        self, tensor: torch.Tensor, res: dict
    ) -> torch.Tensor | None:
        """Run the Lamport 1-shot kernel, return None on failure."""
        orig_shape = tensor.shape

        # The fused kernel expects 2D [token_num, hidden_dim].
        if tensor.dim() == 1:
            tensor_2d = tensor.unsqueeze(0)
        elif tensor.dim() > 2:
            tensor_2d = tensor.reshape(-1, tensor.shape[-1])
        else:
            tensor_2d = tensor

        token_num, hidden_dim = tensor_2d.shape
        if hidden_dim > res["hidden_dim"] or token_num > res["max_token_num"]:
            return None

        from tokenspeed.runtime.utils.pdl import pdl_enabled

        allreduce_out = torch.empty_like(tensor_2d)

        trtllm_allreduce_fusion(
            allreduce_in=tensor_2d,
            world_size=res["world_size"],
            world_rank=res["rank"],
            token_num=token_num,
            hidden_dim=hidden_dim,
            workspace_ptrs=res["workspace"],
            launch_with_pdl=pdl_enabled(),
            use_oneshot=True,
            trigger_completion_at_end=True,
            fp32_acc=False,
            pattern_code=AllReduceFusionPattern.kAllReduce,
            allreduce_out=allreduce_out,
        )

        return allreduce_out.view(orig_shape)

    # ---- Delegate everything else to fallback ----

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
