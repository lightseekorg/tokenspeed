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

"""TRT-LLM workspace all-reduce backend.

Routes plain SUM all-reduces through the group's fusion workspace (IPC
lamport single-node, mnnvl fabric cross-node) with the strategy resolved by
size: one-shot inside its traffic window, two-shot up to the workspace's
token capacity. Falls back to the provided backend for unsupported shapes
and every other op.

The workspace is owned by the kernel package's per-group registry -- the
same workspace the fused-pattern wrappers use. ``configure_group`` arms it
once at init; model-level preparation may later grow it in place.
"""

import torch
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.distributed.comm_backend.base import CommBackend, Group

# Public: dispatch layers route tensor COLLECTIONS by this one-shot window.
MAX_ONESHOT_BYTES = 2 * 1024 * 1024


class TrtllmAllReduceBackend(CommBackend):
    """Backend routing plain SUM all-reduces through the fusion workspace.

    Keyed per-group: ``configure_group`` arms the kernel package's shared
    per-group workspace and records the device group here. Only
    ``all_reduce`` (SUM) is accelerated; every other op delegates to
    *fallback*.
    """

    def __init__(self, fallback: CommBackend):
        self._fallback = fallback
        self._resources = {}  # group_tuple -> {device_group, rank, ...}

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
        """Arm the group's shared fusion workspace.  Returns True on success."""
        if group in self._resources:
            return True

        if not self._load_comm():
            return False

        try:
            from tokenspeed.runtime.distributed.process_group_manager import (
                process_group_manager as pg_manager,
            )

            device_group = pg_manager.get_process_group("nccl", group)

            from tokenspeed_kernel.ops.communication.trtllm import (
                MNNVL_TWOSHOT_MAX_TOKEN,
                ensure_workspace_initialized,
                group_spans_nodes,
            )

            if group_spans_nodes(device_group):
                # Cross-node arms mnnvl only; size it for the two-shot range.
                max_token_num = max(max_token_num, MNNVL_TWOSHOT_MAX_TOKEN)

            # Nothing armed reports False; the group keeps routing through NCCL.
            if not ensure_workspace_initialized(
                rank=rank,
                group=device_group,
                max_token_num=max_token_num,
                hidden_dim=hidden_dim,
                use_fp32_lamport=use_fp32_lamport,
            ):
                return False

            self._resources[group] = {
                "rank": rank,
                "max_token_num": max_token_num,
                "hidden_dim": hidden_dim,
                "device_group": device_group,
            }

            return True

        except Exception:
            return False

    def has_trtllm_ar(self, group: Group) -> bool:
        return group in self._resources

    def oneshot_hidden_dim(self, group: Group) -> int:
        """Hidden lane width the group's workspace is armed for (0 if unarmed)."""
        res = self._resources.get(group)
        if res is None:
            return 0
        from tokenspeed_kernel.ops.communication.trtllm import (
            armed_workspace_hidden_dim,
        )

        return armed_workspace_hidden_dim(res["device_group"])

    def ensure_group_lane(self, group: Group, hidden_dim: int) -> bool:
        """Widen *group*'s one-shot lane to at least *hidden_dim*.

        Collective (all ranks of *group* must call in the same order); only
        widens an already-configured group -- initial arming stays with
        ``configure_group``. Returns True when the lane covers *hidden_dim*.
        """
        res = self._resources.get(group)
        if res is None:
            return False
        if res["hidden_dim"] >= hidden_dim:
            return True
        from tokenspeed_kernel.ops.communication.trtllm import (
            armed_workspace_hidden_dim,
            ensure_workspace_initialized,
        )

        try:
            ok = ensure_workspace_initialized(
                rank=res["rank"],
                group=res["device_group"],
                max_token_num=res["max_token_num"],
                hidden_dim=hidden_dim,
            )
        except Exception:
            self._resources.pop(group, None)
            return False
        if not ok:
            # A graph-freeze refusal keeps the old lane; a failed grow unarms it.
            if armed_workspace_hidden_dim(res["device_group"]) == 0:
                self._resources.pop(group, None)
            return False
        res["hidden_dim"] = hidden_dim
        return True

    # ------------------------------------------------------------------
    # CommBackend interface
    # ------------------------------------------------------------------

    def all_reduce(
        self,
        tensor: torch.Tensor | tuple[torch.Tensor, ...],
        group: Group,
        op=None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if not isinstance(tensor, torch.Tensor):
            return super().all_reduce(tensor, group, op=op)

        if op is None:
            op = torch.distributed.ReduceOp.SUM

        res = self._resources.get(group)

        if res is not None and op == torch.distributed.ReduceOp.SUM:
            from tokenspeed_kernel.ops.communication.trtllm import (
                trtllm_workspace_allreduce,
            )

            result = trtllm_workspace_allreduce(tensor, res["device_group"])
            if result is not None:
                return result

        return self._fallback.all_reduce(tensor, group, op=op)

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
