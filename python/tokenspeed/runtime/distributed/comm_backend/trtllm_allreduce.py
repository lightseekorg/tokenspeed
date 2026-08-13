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

# Public: dispatch layers (AutoBackend, comm_ops) key size-based routing on
# the one-shot admission window; tensors past it always take an NCCL path.
MAX_ONESHOT_BYTES = 2 * 1024 * 1024
_MAX_ONESHOT_BYTES = MAX_ONESHOT_BYTES


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

            # No same-node gate here: only the IPC workspace is node-local
            # (opening a remote rank's IPC handle poisons the CUDA context),
            # and _skip_ipc_workspace below detects exactly that and skips it.
            # The mnnvl fabric workspace is the cross-node path -- an early
            # same-node return would disable it on precisely the groups it
            # exists to serve.
            device_group = pg_manager.get_process_group("nccl", group)

            from tokenspeed_kernel.ops.communication.trtllm import (
                _skip_ipc_workspace,
            )

            if _skip_ipc_workspace(device_group):
                ipc_handles, workspace_tensor = None, None
            else:
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

            # NVLS variants for the plain one-shot path (capability-gated,
            # collective, symmetric fallback): the fused-pattern wrappers in
            # the kernel package arm their own copy; these serve the
            # backend-level all_reduce (post-restructure attention ARs).
            # Two workspaces, split by token count like _ar_fusion_workspace:
            # the private kernel keeps decode-sized calls, upstream flashinfer
            # takes M >= MNNVL_FLASHINFER_MIN_TOKENS and cross-node. The
            # flashinfer workspace is a process-lifetime block shared with the
            # kernel-package manager (group-keyed cache), so arming it here
            # costs no extra memory. Both arming attempts are collectively
            # voted, so every rank creates the same set.
            from tokenspeed_kernel.ops.communication.trtllm import (
                _try_create_mnnvl_fi_workspace,
                _try_create_mnnvl_workspace,
            )

            mnnvl_workspace = _try_create_mnnvl_workspace(
                rank,
                len(group),
                max_token_num,
                hidden_dim,
                device_group,
            )
            mnnvl_fi_workspace = _try_create_mnnvl_fi_workspace(
                rank,
                len(group),
                max_token_num,
                hidden_dim,
                device_group,
            )

            # Nothing usable -> report failure so the backend keeps routing
            # this group through NCCL.
            if (
                workspace_tensor is None
                and mnnvl_workspace is None
                and mnnvl_fi_workspace is None
            ):
                return False

            self._resources[group] = {
                "ipc_handles": ipc_handles,
                "workspace": workspace_tensor,
                "mnnvl": mnnvl_workspace,
                "mnnvl_fi": mnnvl_fi_workspace,
                "rank": rank,
                "world_size": len(group),
                "max_token_num": max_token_num,
                "hidden_dim": hidden_dim,
                "device_group": device_group,
                "use_fp32_lamport": use_fp32_lamport,
            }

            return True

        except Exception:

            return False

    def has_trtllm_ar(self, group: Group) -> bool:
        return group in self._resources

    def oneshot_hidden_dim(self, group: Group) -> int:
        """Armed one-shot lane width for *group* (0 when unavailable)."""
        res = self._resources.get(group)
        return int(res["hidden_dim"]) if res else 0

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
            trtllm_destroy_ipc_workspace_for_all_reduce_fusion,
        )

        try:
            # Cross-node groups have no IPC workspace (ipc_handles is None);
            # destroying it raised, and the except below then popped the group
            # and permanently disabled the fused path.
            if res["ipc_handles"] is not None:
                trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
                    res["ipc_handles"], group=res["device_group"]
                )
            del self._resources[group]
            return (
                self.configure_group(
                    rank=res["rank"],
                    group=group,
                    max_token_num=res["max_token_num"],
                    hidden_dim=hidden_dim,
                )
                and self.oneshot_hidden_dim(group) >= hidden_dim
            )
        except Exception:
            self._resources.pop(group, None)
            return False

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
        if token_num == 0:
            return tensor
        # The Lamport workspace is monomorphic in element type: it was
        # initialized (and is re-armed after each call) with either fp16/bf16
        # or fp32 negative-zero sentinels. A payload whose dtype width differs
        # reads the sentinel pattern as ordinary data and spins forever waiting
        # for completion flags that never arrive -- e.g. the DSpark Markov
        # head's fp32 embedding all-reduce during CUDA-graph capture.
        expected_fp32 = bool(res.get("use_fp32_lamport", False))
        if (tensor_2d.dtype == torch.float32) != expected_fp32:
            return None

        from tokenspeed_kernel.ops.communication.trtllm import (
            MNNVL_FLASHINFER_MIN_TOKENS,
            MNNVL_PREFER_IPC_BYTES,
        )

        from tokenspeed.runtime.utils.pdl import pdl_enabled

        allreduce_out = torch.empty_like(tensor_2d)

        workspace = res["workspace"]
        mnnvl = res.get("mnnvl")
        mnnvl_fi = res.get("mnnvl_fi")
        # Same Tier-2 split as _ar_fusion_workspace: prefer the IPC lamport
        # workspace (single-node only; cross-node it is None) once the payload
        # reaches MNNVL_PREFER_IPC_BYTES; below it the private mnnvl kernel
        # keeps decode-sized calls and upstream flashinfer takes M >=
        # MNNVL_FLASHINFER_MIN_TOKENS (and cross-node), each falling back to
        # the other when its supports() rejects the shape. Unreachable while
        # _MAX_ONESHOT_BYTES stays under the IPC threshold, but keeps this
        # path consistent with the kernel-side dispatch if either bound moves.
        payload_bytes = token_num * hidden_dim * tensor_2d.dtype.itemsize
        prefer_ipc = workspace is not None and payload_bytes >= MNNVL_PREFER_IPC_BYTES
        prefer_fi = token_num >= MNNVL_FLASHINFER_MIN_TOKENS or res["workspace"] is None
        if not prefer_ipc:
            candidates = (mnnvl_fi, mnnvl) if prefer_fi else (mnnvl, mnnvl_fi)
            for candidate in candidates:
                if candidate is not None and candidate.supports(
                    token_num,
                    hidden_dim,
                    tensor_2d.dtype,
                    res["world_size"],
                    AllReduceFusionPattern.kAllReduce,
                    use_oneshot=True,
                ):
                    workspace = candidate
                    break

        # Shape not covered by mnnvl and no IPC fallback -> let the caller
        # fall back to NCCL (this function's None contract).
        if workspace is None:
            return None

        trtllm_allreduce_fusion(
            allreduce_in=tensor_2d,
            world_size=res["world_size"],
            world_rank=res["rank"],
            token_num=token_num,
            hidden_dim=hidden_dim,
            workspace_ptrs=workspace,
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
