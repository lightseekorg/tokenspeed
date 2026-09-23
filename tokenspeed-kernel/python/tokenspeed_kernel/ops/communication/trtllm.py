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

import logging
import os
from ctypes import c_void_p

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.gemm.fp8_utils import (
    create_per_token_group_quant_fp8_output_scale,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import ErrorClass, error_fn, register_kernel
from tokenspeed_kernel.signature import format_signatures

logger = logging.getLogger(__name__)


__all__ = [
    "TrtllmAllGatherState",
    "TrtllmReduceScatterState",
    "trtllm_allgather",
    "trtllm_reduce_scatter",
    "AllReduceFusionPattern",
    "allgather_dual_rmsnorm",
    "allreduce_residual_rmsnorm",
    "reducescatter_residual_rmsnorm",
    "trtllm_allreduce_fusion",
    "trtllm_create_ipc_workspace_for_all_reduce_fusion",
    "trtllm_workspace_allreduce",
    "group_spans_nodes",
    "armed_workspace_hidden_dim",
    "ensure_workspace_initialized",
    "MNNVL_TWOSHOT_MAX_TOKEN",
]

platform = current_platform()

AllReduceFusionPattern = ErrorClass
TrtllmAllGatherState = ErrorClass
TrtllmReduceScatterState = ErrorClass
trtllm_allgather = error_fn
trtllm_reduce_scatter = error_fn
# Two-shot token capacity of the mnnvl workspace; 0 where the path is absent.
MNNVL_TWOSHOT_MAX_TOKEN = 0
allgather_dual_rmsnorm = error_fn
allreduce_residual_rmsnorm = error_fn
trtllm_workspace_allreduce = error_fn
armed_workspace_hidden_dim = error_fn
ensure_workspace_initialized = error_fn
group_spans_nodes = error_fn
allreduce_residual_attnres_combine = error_fn
allreduce_lane_latent_norm = error_fn
reducescatter_residual_rmsnorm = error_fn
trtllm_allreduce_fusion = error_fn
trtllm_create_ipc_workspace_for_all_reduce_fusion = error_fn

if current_platform().is_nvidia:
    from tokenspeed_kernel.ops.communication.fabric import fabric_allocation_supported
    from tokenspeed_kernel.thirdparty.cuda.trtllm import (
        _MNNVL_SUPPORTED_WORLD_SIZES,
        MNNVL_PREFER_IPC_BYTES,
        MNNVL_TWOSHOT_MAX_TOKEN,
        AllGatherFusionPattern,
        AllReduceFusionPattern,
        ReduceScatterFusionPattern,
        _ar_should_use_oneshot,
        _load_trtllm_comm_module,
        trtllm_allgather_fusion,
        trtllm_allreduce_fusion,
        trtllm_create_ipc_workspace_for_all_reduce_fusion,
        trtllm_create_mnnvl_workspace_for_all_reduce_fusion,
        trtllm_destroy_ipc_workspace_for_all_reduce_fusion,
        trtllm_reducescatter_fusion,
    )

    def _mnnvl_locally_available(world_size: int) -> bool:
        """Non-collective capability probe for the MNNVL one-shot AR path.

        Checks the compiled kernel symbol, torch symmetric-memory support, NVLS
        multicast availability, and -- for groups wider than this host -- that
        fabric-handle memory really works. Purely local: safe to call before any
        collective.
        """
        # Single source of truth: the kernel's own list. A duplicated literal
        # here silently gated out world 16 even after the kernel gained it --
        # the correctness suite passed (it calls the creator directly) while
        # end-to-end serving found no workspace at all.
        if world_size not in _MNNVL_SUPPORTED_WORLD_SIZES:
            return False
        try:
            if torch.cuda.get_device_capability()[0] < 9:
                return False
            from torch._C._autograd import DeviceType
            from torch._C._distributed_c10d import _SymmetricMemory
            from torch.distributed import _symmetric_memory  # noqa: F401

            if not _SymmetricMemory.has_multicast_support(
                DeviceType.CUDA, torch.cuda.current_device()
            ):
                return False
            # One rank per GPU, so a wider group necessarily spans hosts, and
            # its symmetric buffer needs multi-node NVLink rather than plain
            # NVLS multicast. Multicast support is still advertised on hosts
            # without the IMEX stack, where symm_mem.rendezvous() then hangs
            # instead of failing, so the allocation has to be probed.
            if (
                world_size > torch.cuda.device_count()
                and not fabric_allocation_supported(torch.cuda.current_device())
            ):
                return False
            return hasattr(_load_trtllm_comm_module(), "trtllm_mnnvl_allreduce_fusion")
        except Exception as exc:  # noqa: BLE001 - capability probe must not raise
            logger.debug(f"mnnvl capability probe failed: {exc!s}")
            return False

    def _try_create_mnnvl_workspace(
        rank: int,
        world_size: int,
        max_token_num: int,
        hidden_dim: int,
        group,
    ):
        """Collectively arm the mnnvl workspace; returns None on fallback.

        Two-phase agreement so every rank takes the same path: (1) all-reduce
        the local capability probe before the symm_mem rendezvous, (2)
        all-reduce the creation result. No environment knobs: capability
        auto-detection only.
        """
        device = torch.device("cuda", torch.cuda.current_device())
        ok = torch.tensor(
            [1 if _mnnvl_locally_available(world_size) else 0],
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
        if ok.item() == 0:
            return None

        workspace = None
        try:
            workspace = trtllm_create_mnnvl_workspace_for_all_reduce_fusion(
                rank, world_size, max_token_num, hidden_dim, group=group
            )
        except Exception as exc:  # noqa: BLE001 - fall back to the IPC path
            logger.warning(f"mnnvl workspace creation failed, using IPC: {exc!s}")

        ok = torch.tensor(
            [1 if workspace is not None else 0], dtype=torch.int32, device=device
        )
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
        if ok.item() == 0:
            return None
        logger.info(
            f"MNNVL one-shot AR workspace armed: rank={rank!s} world_size="
            f"{world_size!s} "
            f"max_token_num={workspace.max_token_num!s} hidden_dim={hidden_dim!s} "
            f"buffer={workspace.buffer_size_bytes!s} bytes",
        )
        return workspace

    def _group_spans_nodes(group) -> bool:
        """True when the process group spans hosts.

        CUDA-IPC handles cannot cross a node boundary, and a failed creation
        attempt does not merely fail -- it leaves a sticky CUDA context error
        that kills the next allocation. So this must be decided before trying,
        not caught afterwards.
        """
        import socket

        try:
            world = (
                dist.get_world_size(group)
                if group is not None
                else dist.get_world_size()
            )
            names = [None] * world
            dist.all_gather_object(names, socket.gethostname(), group=group)
            return len(set(names)) > 1
        except Exception:  # noqa: BLE001 -- no distributed context: single node
            return False

    def group_spans_nodes(group: dist.ProcessGroup) -> bool:
        """Whether the group's ranks live on more than one host.

        Args:
            group: process group to inspect; collective (hostname gather).

        Returns:
            True when at least two distinct hosts are present.
        """
        return _group_spans_nodes(group)

    def _skip_ipc_workspace(group) -> bool:
        """Whether to skip arming the CUDA-IPC workspace for *group*.

        Auto-detected so a cross-node run is safe by default;
        TOKENSPEED_TRTLLM_AR_SKIP_IPC=0/1 forces the decision.
        """
        override = os.getenv("TOKENSPEED_TRTLLM_AR_SKIP_IPC")
        if override is not None:
            return override == "1"
        return _group_spans_nodes(group)

    class TrtllmFusionWorkspaceManager:
        def __init__(self):
            self.workspace_tensor = None
            self.ipc_handles = None
            self.mnnvl_workspace = None
            self.world_size = None
            self.rank = None
            self.max_token_num = None
            self.hidden_dim = None
            self.use_fp32_lamport = None
            self.initialized = False
            self.graph_consumed = False
            self.group = None

        def initialize(
            self,
            world_size: int,
            rank: int,
            max_token_num: int,
            hidden_dim: int,
            group: dist.ProcessGroup,
            use_fp32_lamport: bool = False,
        ):
            """(Re)create the workspace at exactly the requested geometry.

            No-op when already armed at it; otherwise the old workspace is
            destroyed first, so a creation failure (raised from inside
            ``_create``) leaves the manager unarmed (``initialized`` False);
            IPC handles already created stay recorded for the next
            ``cleanup()`` to destroy on the recorded group.
            """
            if (
                self.initialized
                and self.world_size == world_size
                and self.max_token_num == max_token_num
                and self.hidden_dim == hidden_dim
                and self.use_fp32_lamport == use_fp32_lamport
            ):
                return

            # Fail fast: collective recovery here opens rank-desync windows.
            self.cleanup()
            self._create(
                world_size, rank, max_token_num, hidden_dim, group, use_fp32_lamport
            )

        def _create(
            self,
            world_size: int,
            rank: int,
            max_token_num: int,
            hidden_dim: int,
            group: dist.ProcessGroup,
            use_fp32_lamport: bool,
        ) -> None:
            # CUDA-IPC handles cannot span nodes -- attempting creation on a
            # cross-node group fails AND leaves a sticky CUDA context error
            # ('invalid resource handle' on the next allocation). Gate it off
            # for cross-node runs; the MNNVL fabric workspace below is the
            # multi-node path.
            # Recorded first: a failed arm must still destroy on the right group.
            self.group = group
            _skip_ipc = _skip_ipc_workspace(group)
            if _skip_ipc:
                self.ipc_handles, self.workspace_tensor = None, None
            else:
                # allreduce_fusion, allgather_fusion, reducescatter_fusion all use the same workspace to create entry
                self.ipc_handles, self.workspace_tensor = (
                    trtllm_create_ipc_workspace_for_all_reduce_fusion(
                        rank,
                        world_size,
                        max_token_num,
                        hidden_dim,
                        group=group,
                        use_fp32_lamport=use_fp32_lamport,
                    )
                )
            # Additionally arm the MNNVL one-shot AR workspace (NVLS multicast
            # + Lamport rotation). Capability auto-detected; the IPC workspace
            # above stays as the always-available fallback and continues to
            # serve allgather/reducescatter and unsupported AR shapes.
            self.mnnvl_workspace = _try_create_mnnvl_workspace(
                rank, world_size, max_token_num, hidden_dim, group
            )

            # With IPC skipped, mnnvl is the only workspace; if it failed to
            # arm there is nothing to fuse with -- stay uninitialized so
            # prepare_allreduce_fusion() returns False and the model layer
            # keeps the plain NCCL path.
            if self.workspace_tensor is None and self.mnnvl_workspace is None:
                logger.warning(
                    "trtllm AR: no workspace available (ipc skipped, mnnvl "
                    "failed); fusion disabled for this group"
                )
                return

            self.world_size = world_size
            self.rank = rank
            self.max_token_num = max_token_num
            self.hidden_dim = hidden_dim
            self.use_fp32_lamport = use_fp32_lamport
            self.initialized = True

            logger.info(
                f"TRT-LLM fusion workspace initialized for rank {rank}, "
                f"world_size {world_size}, "
                f"max_token_num {max_token_num}, "
                f"hidden_dim {hidden_dim} "
            )

        def cleanup(self):
            """Clean up workspace"""
            # Keyed on resource presence, not just the initialized flag: a
            # failed re-arm can leave ipc_handles set with initialized=False,
            # and skipping the destroy then would orphan them.
            if self.initialized or self.ipc_handles is not None:
                try:
                    # Cross-node groups arm mnnvl only; there is no IPC
                    # workspace to destroy, but the state reset below must
                    # still run or a re-init leaks the symm_mem allocation and
                    # a failed re-arm leaves initialized=True with no
                    # workspace behind it.
                    if self.ipc_handles is not None:
                        trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
                            self.ipc_handles, group=self.group
                        )
                except Exception as e:
                    logger.warning(f"Failed to cleanup TRT-LLM fusion workspace: {e}")
                finally:
                    self.workspace_tensor = None
                    self.ipc_handles = None
                    # symm_mem allocations are process-lifetime; dropping the
                    # reference is all we can (and need to) do here.
                    self.mnnvl_workspace = None
                    self.initialized = False
                    self.world_size = None
                    self.rank = None
                    self.max_token_num = None
                    self.hidden_dim = None
                    self.use_fp32_lamport = None
                    self.graph_consumed = False
                    self.group = None

    _workspace_managers: dict[tuple[int, ...], TrtllmFusionWorkspaceManager] = {}

    def _manager_for_group(group: dist.ProcessGroup) -> TrtllmFusionWorkspaceManager:
        """One fusion-workspace manager per distinct rank set.

        Keyed by global ranks rather than ProcessGroup identity: the runtime
        AR backend and the fused-pattern wrappers reach the same group through
        different ProcessGroup objects, and both must land on the same
        workspace.
        """
        key = tuple(dist.get_process_group_ranks(group))
        manager = _workspace_managers.get(key)
        if manager is None:
            manager = _workspace_managers[key] = TrtllmFusionWorkspaceManager()
        return manager

    # Reduce-scatter reuses the group's fusion workspace (IPC lamport).

    def ensure_workspace_initialized(
        rank: int,
        group: dist.ProcessGroup,
        max_token_num: int = 2048,
        hidden_dim: int = 4096,
        use_fp32_lamport: bool = False,
    ) -> bool:
        """Arm (or grow) the group's shared fusion workspace, collectively.

        Args:
            rank: this rank within ``group``.
            group: process group the workspace serves; all ranks must call in
                lockstep with identical arguments.
            max_token_num: minimum token capacity to arm; grow-only.
            hidden_dim: minimum hidden lane width to arm; grow-only.
            use_fp32_lamport: arm the fp32 lamport sentinel; sticky once set.

        Returns:
            True when a workspace is armed for the group at (at least) the
            requested capacity. False for single-rank groups, when nothing
            could be armed, or when growth was refused because a captured
            CUDA graph references the current workspace — arm the full
            geometry every captured graph will use BEFORE capture (capture
            state is per-rank; keep capture lockstep across ranks). Creation
            failures propagate (fail fast; a failed grow leaves the group
            unarmed). Arming fp32 lamport is sticky; a 16-bit plain AR
            then declines to NCCL regardless of mnnvl, the rmsnorm wrapper
            degrades to the unfused path and attnres-combine/latent-norm
            raise where no mnnvl workspace serves the shape, and the
            reducescatter/allgather wrappers -- IPC-only, no mnnvl
            implementation -- always raise on the mismatch.
        """
        world_size = group.size()
        if world_size <= 1:
            return False

        manager = _manager_for_group(group)
        target_max_token_num = max_token_num
        target_hidden_dim = hidden_dim
        target_use_fp32_lamport = use_fp32_lamport
        if manager.initialized:
            if manager.max_token_num is not None:
                target_max_token_num = max(manager.max_token_num, max_token_num)
            if manager.hidden_dim is not None:
                target_hidden_dim = max(manager.hidden_dim, hidden_dim)
            if manager.use_fp32_lamport:
                target_use_fp32_lamport = True

        if (
            (not manager.initialized)
            or (manager.max_token_num != target_max_token_num)
            or (manager.hidden_dim != target_hidden_dim)
            or (manager.use_fp32_lamport != target_use_fp32_lamport)
        ):
            if manager.initialized and manager.graph_consumed:
                # Recreating would leave captured graphs on freed peer pointers.
                logger.warning(
                    "trtllm AR: refusing to grow the fusion workspace "
                    f"(tokens {manager.max_token_num!s}->{target_max_token_num!s} "
                    f"hidden {manager.hidden_dim!s}->{target_hidden_dim!s} fp32 "
                    f"{manager.use_fp32_lamport!s}->{target_use_fp32_lamport!s}): a "
                    "captured "
                    "CUDA graph references it. Arm the full geometry before "
                    "graph capture.",
                )
                return False
            if (
                manager.initialized
                and target_use_fp32_lamport
                and not (manager.use_fp32_lamport)
            ):
                # Sticky flip: bf16/fp16 payloads are rejected from here on.
                logger.warning(
                    "trtllm AR: workspace sentinel flips to fp32 lamport; "
                    "16-bit plain all-reduces on this group fall back to NCCL"
                )
            logger.info(
                "Re/initializing TRT-LLM fusion IPC workspace: "
                f"world_size={world_size!s} rank={rank!s} max_token_num="
                f"{target_max_token_num!s} hidden_dim={target_hidden_dim!s} "
                f"use_fp32_lamport={target_use_fp32_lamport!s} "
                f"(prev max_token_num={manager.max_token_num!s} hidden_dim="
                f"{manager.hidden_dim!s} use_fp32_lamport={manager.use_fp32_lamport!s})",
            )
            manager.initialize(
                world_size=world_size,
                rank=rank,
                max_token_num=target_max_token_num,
                hidden_dim=target_hidden_dim,
                use_fp32_lamport=target_use_fp32_lamport,
                group=group,
            )

        return (
            manager.initialized
            and manager.max_token_num == target_max_token_num
            and manager.hidden_dim == target_hidden_dim
            and manager.use_fp32_lamport == target_use_fp32_lamport
        )

    def _mark_captured(
        manager: TrtllmFusionWorkspaceManager, workspace: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Freeze the workspace against growth once a graph captures it."""
        if torch.cuda.is_current_stream_capturing():
            manager.graph_consumed = True
        return workspace

    def _ar_fusion_workspace(
        manager: TrtllmFusionWorkspaceManager,
        token_num: int,
        hidden_dim: int,
        dtype: torch.dtype,
        pattern_code: int,
        use_oneshot: bool,
        residual_reduce_scattered: bool = False,
    ):
        """Pick the AR workspace for one call: mnnvl when eligible, else IPC.

        Tier 2 of AR dispatch (Tier 1 chose the trtllm backend in auto.py).
        Decision, first match wins:
          1. compatible IPC workspace exists (single-node, lamport sentinel
             width matches) AND (payload >= MNNVL_PREFER_IPC_BYTES
             OR pattern == kAllReduceLatentNorm) ....... IPC lamport/twoshot
          2. mnnvl supports this shape ................. mnnvl (the caller's
             ``use_oneshot`` parameter selects its strategy)
          3. no usable IPC (cross-node, or lamport sentinel width mismatch)
             ............................................ None (caller degrades:
             the rmsnorm family runs unfused NCCL + torch epilogue, the rest
             raise loudly -- never a null workspace into the kernel)
          4. otherwise ................................ IPC
        Cross-node, workspace_tensor is None so only 2/3 apply -- mnnvl serves
        the whole range (it beats NCCL everywhere there).
        """
        mnnvl = manager.mnnvl_workspace
        # Byte-based split between the two fused workspaces: multicast (mnnvl)
        # for small payloads, IPC lamport once bandwidth dominates. Only bites
        # single-node -- cross-node workspace_tensor is None and mnnvl is the
        # only option. See MNNVL_PREFER_IPC_BYTES for the measurement.
        from tokenspeed_kernel.thirdparty.cuda.trtllm import AllReduceFusionPattern

        payload_bytes = token_num * hidden_dim * dtype.itemsize
        # Prefer IPC when it exists (single node) for large payloads, and for
        # latent-norm whose wide lane the mnnvl geometry handles slightly worse.
        # Cross-node workspace_tensor is None, so mnnvl serves both.
        prefer_ipc = payload_bytes >= MNNVL_PREFER_IPC_BYTES or (
            pattern_code == AllReduceFusionPattern.kAllReduceLatentNorm
        )
        # A sentinel-width mismatch corrupts the lamport neg-zero wait/clear protocol.
        ipc_ok = manager.workspace_tensor is not None and (
            (dtype == torch.float32) == manager.use_fp32_lamport
        )
        if ipc_ok and prefer_ipc:
            return _mark_captured(manager, manager.workspace_tensor)
        if mnnvl is not None and mnnvl.supports(
            token_num,
            hidden_dim,
            dtype,
            manager.world_size,
            pattern_code,
            use_oneshot=use_oneshot,
            residual_reduce_scattered=residual_reduce_scattered,
        ):
            return _mark_captured(manager, mnnvl)
        # Cross-node there is no IPC workspace, so a shape/pattern mnnvl rejects
        # has no fused home. Return None and let the caller decide: the rmsnorm
        # family degrades to the unfused NCCL path, the rest raise loudly. Never
        # hand the kernel a null workspace.
        if not ipc_ok:
            logger.debug(
                f"trtllm AR fusion: shape (tokens={token_num!s}, hidden={hidden_dim!s},"
                f" dtype={dtype!s}, "
                f"pattern={pattern_code!s}, oneshot={use_oneshot!s}) not supported by "
                "mnnvl and no "
                "compatible IPC workspace (missing or sentinel-width "
                "mismatch); caller falls back unfused",
            )
            return None
        return _mark_captured(manager, manager.workspace_tensor)

    def get_num_tokens_per_rank(world_size: int, total_tokens_in_group: int) -> list:
        token_list_in_group = []
        for rank in range(0, world_size):
            num_tokens_per_rank = total_tokens_in_group // world_size + (
                1 if (rank < total_tokens_in_group % world_size) else 0
            )
            token_list_in_group.append(num_tokens_per_rank)
        return token_list_in_group

    def _unfused_allreduce_residual_rmsnorm(
        input_tensor: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        rank: int,
        group: dist.ProcessGroup,
        eps: float,
        block_quant_fp8: bool,
        has_partial_norm_out: bool,
    ):
        """Unfused NCCL + torch epilogue for the rmsnorm fusion family.

        Cross-node groups have no IPC workspace, and the mnnvl kernel does not
        implement the block-quant / partial-out epilogues, so those calls land
        here instead of aborting. Mirrors the fused contract exactly:
        (quant_out, residual_out, scale_out, partial_norm_out) when
        block_quant_fp8 else (norm_out, residual_out, None, partial_norm_out).
        The all-reduce runs on NCCL (different reduction order than the fused
        kernels -- numerically equivalent, not bitwise).
        """
        reduced = input_tensor.contiguous().clone()
        dist.all_reduce(reduced, group=group)
        res32 = reduced.float() + residual.float()
        residual_out = res32.to(input_tensor.dtype)
        # fp32 epilogue matching the fused kernels: variance over the hidden
        # lane, gamma multiply in fp32, single rounding back to the payload
        # dtype.
        norm32 = res32 * torch.rsqrt(res32.pow(2).mean(-1, keepdim=True) + eps)
        norm_out = (norm32 * weight.float()).to(input_tensor.dtype)

        partial_norm_out = None
        if has_partial_norm_out:
            world_size = dist.get_world_size(group)
            counts = get_num_tokens_per_rank(world_size, input_tensor.shape[0])
            start = sum(counts[:rank])
            partial_norm_out = norm_out[start : start + counts[rank]].contiguous()

        if block_quant_fp8:
            from tokenspeed_kernel.ops.gemm.fp8_utils import per_token_group_quant_fp8

            quant_out, scale_out = per_token_group_quant_fp8(
                norm_out,
                group_size=128,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=False,
            )
            return quant_out, residual_out, scale_out, partial_norm_out
        return norm_out, residual_out, None, partial_norm_out

    def armed_workspace_hidden_dim(group: dist.ProcessGroup) -> int:
        """Hidden lane width the group's workspace is armed for.

        Args:
            group: process group whose shared fusion workspace to inspect.

        Returns:
            The armed ``hidden_dim`` in elements, or 0 when no workspace is
            armed for *group*.
        """
        manager = _manager_for_group(group)
        return int(manager.hidden_dim) if manager.initialized else 0

    def trtllm_workspace_allreduce(
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
    ) -> torch.Tensor | None:
        """Plain SUM all-reduce through the group's armed fusion workspace.

        Tier 2 of the plain-AR dispatch (Tier 1 chose the trtllm backend in
        auto.py). The strategy is resolved by size -- one-shot inside its
        traffic window, two-shot up to the workspace's token capacity -- over
        the same IPC-vs-mnnvl split the fused-pattern wrappers use.

        Args:
            input_tensor: CUDA tensor to sum-reduce; the trailing dimension
                is treated as the hidden lane. A non-contiguous tensor is
                served through a contiguous copy and is never mutated.
            group: process group to reduce over.

        Returns:
            The reduced tensor (new allocation, same shape), or ``None`` when
            no workspace is armed for *group* or the shape is not served --
            the caller keeps its own (NCCL) fallback.
        """
        if input_tensor.ndim == 0 or input_tensor.numel() == 0:
            return None
        if not input_tensor.is_cuda:
            return None
        if input_tensor.dtype not in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ):
            return None
        manager = _manager_for_group(group)
        if not manager.initialized:
            return None

        tensor_2d = input_tensor.reshape(-1, input_tensor.shape[-1])
        token_num, hidden_dim = tensor_2d.shape
        # A sentinel-width mismatch corrupts the lamport neg-zero wait/clear protocol.
        if (tensor_2d.dtype == torch.float32) != manager.use_fp32_lamport:
            return None
        # The device kernels read 16-byte vectors along the hidden lane.
        if hidden_dim % (16 // tensor_2d.dtype.itemsize) != 0:
            return None
        if hidden_dim > manager.hidden_dim or token_num > manager.max_token_num:
            return None

        world_size = manager.world_size
        # The device kernels support exactly these fan-ins at every size.
        if world_size not in _MNNVL_SUPPORTED_WORLD_SIZES:
            return None
        requested_oneshot = _ar_should_use_oneshot(
            token_num, hidden_dim, tensor_2d.dtype, world_size
        )
        resolved_oneshot = requested_oneshot
        if manager.mnnvl_workspace is not None:
            resolved_oneshot = manager.mnnvl_workspace.resolve_use_oneshot(
                token_num, None, hidden_dim
            )
        workspace = _ar_fusion_workspace(
            manager,
            token_num,
            hidden_dim,
            tensor_2d.dtype,
            AllReduceFusionPattern.kAllReduce,
            resolved_oneshot,
        )
        if workspace is None:
            return None
        # IPC has its own heuristic; only MNNVL uses the workspace's frozen cap.
        if workspace is not manager.mnnvl_workspace:
            resolved_oneshot = requested_oneshot
            # The IPC two-shot kernel asserts token_num > world_size.
            if not resolved_oneshot and token_num <= world_size:
                return None
        if not tensor_2d.is_contiguous():
            tensor_2d = tensor_2d.contiguous()

        allreduce_out = torch.empty_like(tensor_2d)
        trtllm_allreduce_fusion(
            allreduce_in=tensor_2d,
            world_size=world_size,
            world_rank=manager.rank,
            token_num=token_num,
            hidden_dim=hidden_dim,
            workspace_ptrs=workspace,
            use_oneshot=resolved_oneshot,
            trigger_completion_at_end=True,
            fp32_acc=False,
            pattern_code=AllReduceFusionPattern.kAllReduce,
            allreduce_out=allreduce_out,
        )
        return allreduce_out.view(input_tensor.shape)

    def allreduce_residual_rmsnorm(
        input_tensor: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        rank: int,
        group: dist.ProcessGroup,
        eps: float = 1e-6,
        max_token_num: int = 2048,
        use_oneshot: bool | None = None,
        trigger_completion_at_end: bool = False,
        fp32_acc: bool = False,
        block_quant_fp8: bool = False,
        residual_reduce_scattered: bool = False,
        has_partial_norm_out: bool = False,
        max_sm_to_use: int | None = None,
        launch_with_pdl: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Use TRT-LLM fused allreduce + residual + RMS norm operation.
        """
        world_size = group.size()
        assert world_size > 1, "Single GPU, no need for allreduce fusion"
        assert input_tensor.shape[0] <= max_token_num

        if not ensure_workspace_initialized(
            rank=rank,
            group=group,
            max_token_num=max_token_num,
            hidden_dim=input_tensor.shape[-1],
            use_fp32_lamport=(input_tensor.dtype == torch.float32),
        ):
            raise RuntimeError("TRT-LLM fusion workspace not available")

        token_num, hidden_dim = input_tensor.shape

        residual_out = torch.empty_like(residual)
        norm_out = torch.empty_like(input_tensor)

        partial_norm_out = None
        pattern_code = None
        if has_partial_norm_out:
            num_tokens_list = get_num_tokens_per_rank(world_size, input_tensor.shape[0])
            partial_num_tokens = num_tokens_list[rank]
            partial_norm_out = torch.empty(
                (partial_num_tokens, hidden_dim),
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )
            pattern_code = (
                AllReduceFusionPattern.kARResidualRMSNormPartialOutFP8BlockWiseQuant
                if block_quant_fp8
                else AllReduceFusionPattern.kARResidualRMSNormPartialOut
            )
        else:
            pattern_code = (
                AllReduceFusionPattern.kARResidualRMSNormFP8BlockWiseQuant
                if block_quant_fp8
                else AllReduceFusionPattern.kARResidualRMSNorm
            )

        if block_quant_fp8:
            quant_out = torch.empty(
                input_tensor.size(),
                dtype=torch.float8_e4m3fn,
                device=input_tensor.device,
            )
            out_shape = (*quant_out.shape[:-1], quant_out.shape[-1])
            scale_out = create_per_token_group_quant_fp8_output_scale(
                x_shape=out_shape,
                device=quant_out.device,
                group_size=128,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=False,
            )
        else:
            quant_out = None
            scale_out = None

        if residual_reduce_scattered or has_partial_norm_out:
            use_oneshot = True

        requested_oneshot = (
            use_oneshot
            if use_oneshot is not None
            else _ar_should_use_oneshot(
                token_num, hidden_dim, input_tensor.dtype, world_size
            )
        )
        manager = _manager_for_group(group)
        resolved_oneshot = requested_oneshot
        if manager.mnnvl_workspace is not None:
            resolved_oneshot = manager.mnnvl_workspace.resolve_use_oneshot(
                token_num, use_oneshot, hidden_dim
            )
        workspace = _ar_fusion_workspace(
            manager,
            token_num,
            hidden_dim,
            input_tensor.dtype,
            pattern_code,
            resolved_oneshot,
            residual_reduce_scattered,
        )
        # IPC has its own heuristic; only MNNVL uses the workspace's frozen cap.
        if workspace is not manager.mnnvl_workspace:
            resolved_oneshot = requested_oneshot
        if workspace is None:
            # Cross-node group, pattern/shape the mnnvl kernel cannot serve
            # (block-quant or partial-out epilogue, oversized call). Degrade to
            # the unfused NCCL path instead of aborting the forward pass.
            if residual_reduce_scattered:
                # The input arrives reduce-scattered; a blind all-reduce would
                # double-count. No unfused equivalent exists here, so fail
                # loudly rather than corrupt.
                raise RuntimeError(
                    "trtllm AR fusion: residual_reduce_scattered has no fused "
                    f"workspace for this call (tokens={token_num}, "
                    f"hidden={hidden_dim}, dtype={input_tensor.dtype}, "
                    f"pattern={pattern_code}) and no unfused fallback"
                )
            return _unfused_allreduce_residual_rmsnorm(
                input_tensor,
                residual,
                weight,
                rank,
                group,
                eps,
                block_quant_fp8,
                has_partial_norm_out,
            )

        trtllm_allreduce_fusion(
            allreduce_in=input_tensor,
            world_size=world_size,
            world_rank=rank,
            token_num=token_num,
            hidden_dim=hidden_dim,
            workspace_ptrs=workspace,
            launch_with_pdl=launch_with_pdl,
            use_oneshot=resolved_oneshot,
            trigger_completion_at_end=trigger_completion_at_end,
            fp32_acc=fp32_acc,
            pattern_code=(pattern_code),
            allreduce_out=None,
            residual_in=residual,
            residual_out=residual_out,
            norm_out=norm_out,
            quant_out=quant_out,
            scale_out=scale_out,
            rms_gamma=weight,
            rms_eps=eps,
            scale_factor=None,
            layout_code=None,
            residual_reduce_scattered=residual_reduce_scattered,
            max_sm_to_use=max_sm_to_use,
            partial_norm_out=partial_norm_out,
        )
        if block_quant_fp8:
            return quant_out, residual_out, scale_out, partial_norm_out
        else:
            return norm_out, residual_out, None, partial_norm_out

    def allreduce_residual_attnres_combine(
        input_tensor: torch.Tensor,
        residual: torch.Tensor,
        res_w: torch.Tensor,
        rms_w: torch.Tensor,
        out_norm_w: torch.Tensor | None,
        scratch: tuple,
        rank: int,
        group: dist.ProcessGroup,
        eps: float = 1e-6,
        max_token_num: int = 2048,
        trigger_completion_at_end: bool = False,
        launch_with_pdl: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """AR + residual + AttnRes prefix combine in one kernel (Kimi-K3).

        Args:
            input_tensor: per-rank partial to all-reduce, ``[T, H]``.
            residual: running prefix stream before this accumulate.
            res_w/rms_w: the mix projection and RMS weights (``[H]``).
            out_norm_w: optional fused out-norm gamma.
            scratch: (m [T], s [T], acc [T, H] fp32) blocks partial.

        Returns:
            (norm_out, residual_out): the combined hidden and the new prefix.
        """
        world_size = group.size()
        assert world_size > 1, "Single GPU, no need for allreduce fusion"
        assert input_tensor.shape[0] <= max_token_num
        if not ensure_workspace_initialized(
            rank=rank,
            group=group,
            max_token_num=max_token_num,
            hidden_dim=input_tensor.shape[-1],
            use_fp32_lamport=(input_tensor.dtype == torch.float32),
        ):
            raise RuntimeError("TRT-LLM fusion workspace not available")

        token_num, hidden_dim = input_tensor.shape
        residual_out = torch.empty_like(residual)
        norm_out = torch.empty_like(input_tensor)
        m, s_, acc = scratch
        workspace = _ar_fusion_workspace(
            _manager_for_group(group),
            token_num,
            hidden_dim,
            input_tensor.dtype,
            AllReduceFusionPattern.kARResidualAttnResCombine,
            use_oneshot=True,
        )
        if workspace is None:
            # Lands here cross-node out-of-mnnvl-range, or single-node on a
            # sentinel-width mismatch. No unfused equivalent of the combine
            # epilogue -- fail loudly rather than skip the reduce.
            raise RuntimeError(
                "trtllm AR fusion: kARResidualAttnResCombine has no fused "
                f"workspace for this call (tokens={token_num}, "
                f"hidden={hidden_dim}, dtype={input_tensor.dtype})"
            )
        trtllm_allreduce_fusion(
            allreduce_in=input_tensor,
            world_size=world_size,
            world_rank=rank,
            token_num=token_num,
            hidden_dim=hidden_dim,
            workspace_ptrs=workspace,
            launch_with_pdl=launch_with_pdl,
            use_oneshot=True,
            trigger_completion_at_end=trigger_completion_at_end,
            fp32_acc=False,
            pattern_code=AllReduceFusionPattern.kARResidualAttnResCombine,
            allreduce_out=None,
            residual_in=residual,
            residual_out=residual_out,
            norm_out=norm_out,
            quant_out=None,
            scale_out=None,
            rms_gamma=rms_w,
            rms_eps=eps,
            scale_factor=None,
            layout_code=None,
            residual_reduce_scattered=False,
            max_sm_to_use=None,
            attnres_m=m,
            attnres_s=s_,
            attnres_acc=acc,
            attnres_res_w=res_w,
            attnres_out_norm_w=out_norm_w,
        )
        return norm_out, residual_out

    def allreduce_lane_latent_norm(
        lane: torch.Tensor,
        gamma: torch.Tensor,
        latent_width: int,
        rank: int,
        group: dist.ProcessGroup,
        eps: float = 1e-6,
        max_token_num: int = 2048,
        trigger_completion_at_end: bool = False,
        launch_with_pdl: bool | None = None,
    ) -> torch.Tensor:
        """All-reduce the [latent | hidden] lane and RMS-norm the latent slice.

        Args:
            lane: ``[T, latent_width + hidden]`` concatenated partials;
                reduced and written back in place.
            gamma: latent RMS weight (``[latent_width]``).

        Returns:
            ``lane`` (reduced, latent slice normed).
        """
        world_size = group.size()
        assert world_size > 1, "Single GPU, no need for allreduce fusion"
        token_num, lane_dim = lane.shape
        assert token_num <= max_token_num
        if not ensure_workspace_initialized(
            rank=rank,
            group=group,
            max_token_num=max_token_num,
            hidden_dim=lane_dim,
            use_fp32_lamport=(lane.dtype == torch.float32),
        ):
            raise RuntimeError("TRT-LLM fusion workspace not available")

        workspace = _ar_fusion_workspace(
            _manager_for_group(group),
            token_num,
            lane_dim,
            lane.dtype,
            AllReduceFusionPattern.kAllReduceLatentNorm,
            use_oneshot=True,
        )
        if workspace is None:
            # Lands here cross-node out-of-mnnvl-range, or single-node on a
            # sentinel-width mismatch. Fail loudly rather than skip the reduce.
            raise RuntimeError(
                "trtllm AR fusion: kAllReduceLatentNorm has no fused workspace "
                f"for this call (tokens={token_num}, lane={lane_dim}, "
                f"dtype={lane.dtype})"
            )
        trtllm_allreduce_fusion(
            allreduce_in=lane,
            world_size=world_size,
            world_rank=rank,
            token_num=token_num,
            hidden_dim=lane_dim,
            workspace_ptrs=workspace,
            launch_with_pdl=launch_with_pdl,
            use_oneshot=True,
            trigger_completion_at_end=trigger_completion_at_end,
            fp32_acc=False,
            pattern_code=AllReduceFusionPattern.kAllReduceLatentNorm,
            allreduce_out=lane,
            residual_in=None,
            residual_out=None,
            norm_out=None,
            quant_out=None,
            scale_out=None,
            rms_gamma=gamma,
            rms_eps=eps,
            scale_factor=None,
            layout_code=None,
            residual_reduce_scattered=False,
            max_sm_to_use=None,
            latent_width=latent_width,
        )
        return lane

    def reducescatter_residual_rmsnorm(
        input_tensor: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        rank: int,
        group: dist.ProcessGroup,
        eps: float = 1e-6,
        max_token_num: int = 2048,
        use_oneshot: bool | None = None,
        trigger_completion_at_end: bool = False,
        fp32_acc: bool = False,
        block_quant_fp8: bool = False,
        add_in: torch.Tensor | None = None,
        launch_with_pdl: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Use TRT-LLM fused reducescatter + residual + RMS norm operation.
        """
        world_size = group.size()
        assert world_size > 1, "Single GPU, no need for reducescatter fusion"
        assert input_tensor.shape[0] <= max_token_num

        if not ensure_workspace_initialized(
            rank=rank,
            group=group,
            max_token_num=max_token_num,
            hidden_dim=input_tensor.shape[-1],
            use_fp32_lamport=(input_tensor.dtype == torch.float32),
        ):
            raise RuntimeError("TRT-LLM reduce scatter fusion workspace not available")

        token_num, hidden_dim = input_tensor.shape

        tokens_per_rank = token_num // world_size
        remaining = token_num % world_size
        token_count = tokens_per_rank + (1 if rank < remaining else 0)

        residual_out = torch.empty(
            (token_count, hidden_dim), dtype=residual.dtype, device=residual.device
        )
        norm_out = torch.empty(
            (token_count, hidden_dim),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        if block_quant_fp8:
            if add_in is not None:
                pattern_code = (
                    ReduceScatterFusionPattern.kRSAddResidualRMSNormFP8BlockWiseQuant
                )
            else:
                pattern_code = (
                    ReduceScatterFusionPattern.kRSResidualRMSNormFP8BlockWiseQuant
                )
        else:
            if add_in is not None:
                pattern_code = ReduceScatterFusionPattern.kRSAddResidualRMSNorm
            else:
                pattern_code = ReduceScatterFusionPattern.kRSResidualRMSNorm

        if block_quant_fp8:
            quant_out = torch.empty(
                (token_count, hidden_dim),
                dtype=torch.float8_e4m3fn,
                device=input_tensor.device,
            )
            out_shape = (*quant_out.shape[:-1], quant_out.shape[-1])
            scale_out = create_per_token_group_quant_fp8_output_scale(
                x_shape=out_shape,
                device=quant_out.device,
                group_size=128,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=False,
            )
        else:
            quant_out = None
            scale_out = None
        # allgather/reducescatter have no mnnvl implementation -- they run on
        # the IPC lamport workspace only. Since IPC is skipped on cross-node
        # groups, `initialized` can be True (mnnvl armed) while this workspace
        # is None; without this check a null pointer reaches the FFI.
        manager = _manager_for_group(group)
        if manager.workspace_tensor is None:
            raise RuntimeError(
                "trtllm reducescatter fusion requires the IPC lamport workspace, which is "
                "unavailable on this group (cross-node, or IPC explicitly "
                "skipped). Use the unfused path for this collective."
            )
        # A sentinel-width mismatch corrupts the lamport neg-zero wait/clear protocol.
        if (input_tensor.dtype == torch.float32) != manager.use_fp32_lamport:
            raise RuntimeError(
                "trtllm reducescatter fusion: payload width does not match "
                "the armed lamport sentinel"
            )

        trtllm_reducescatter_fusion(
            reducescatter_in=input_tensor,
            world_size=world_size,
            world_rank=rank,
            token_num=token_num,
            hidden_dim=hidden_dim,
            workspace_ptrs=_mark_captured(manager, manager.workspace_tensor),
            launch_with_pdl=launch_with_pdl,
            trigger_completion_at_end=trigger_completion_at_end,
            num_token_current_rank=token_count,
            fp32_acc=fp32_acc,
            pattern_code=pattern_code,
            use_oneshot=use_oneshot,
            reducescatter_out=None,
            add_in=add_in,
            residual_in=residual,
            residual_out=residual_out,
            norm_out=norm_out,
            quant_out=quant_out,
            scale_out=scale_out,
            rms_gamma=weight,
            rms_eps=eps,
            scale_factor=None,
            layout_code=None,
        )
        if block_quant_fp8:
            return quant_out, residual_out, scale_out
        else:
            return norm_out, residual_out, None

    def allgather_dual_rmsnorm(
        qkv: torch.Tensor,
        total_num_tokens: int,
        weight_q_a: torch.nn.Parameter,
        weight_kv_a: torch.nn.Parameter,
        rank: int,
        group: dist.ProcessGroup,
        eps_q: float,
        eps_kv: float,
        max_token_num: int,
        block_quant_fp8: bool = False,
        trigger_completion_at_end: bool = False,
        fp32_acc: bool = False,
        launch_with_pdl: bool | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """
        Use TRT-LLM fused allgather + dual RMS norm + optional FP8 quantization.
        """
        world_size = group.size()
        assert world_size > 1, "Single GPU, no need for allgather fusion"

        num_token_current_rank = qkv.shape[0]
        hidden_dim = qkv.shape[1]

        if num_token_current_rank > max_token_num:
            raise RuntimeError(
                f"Token count {num_token_current_rank} exceeds max {max_token_num}"
            )

        if not ensure_workspace_initialized(
            rank=rank,
            group=group,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            use_fp32_lamport=(qkv.dtype == torch.float32),
        ):
            raise RuntimeError("TRT-LLM fusion workspace not available")

        q_lora_rank = weight_q_a.shape[0]
        kv_lora_rank = weight_kv_a.shape[0]
        qk_rope_head_dim = hidden_dim - q_lora_rank - kv_lora_rank

        num_token_all_group = total_num_tokens

        allgather_out = torch.empty(
            (num_token_all_group, hidden_dim), dtype=qkv.dtype, device=qkv.device
        )

        x_norm_out = torch.empty(
            (num_token_all_group, q_lora_rank), dtype=qkv.dtype, device=qkv.device
        )

        # y_norm_out output is on the slice of allgather_out
        y_norm_out = allgather_out[..., q_lora_rank : q_lora_rank + kv_lora_rank]

        if block_quant_fp8:
            block_size = 128
            quant_out = torch.empty(
                (num_token_all_group, q_lora_rank),
                dtype=torch.float8_e4m3fn,
                device=qkv.device,
            )
            out_shape = (*quant_out.shape[:-1], quant_out.shape[-1])
            scale_out = create_per_token_group_quant_fp8_output_scale(
                x_shape=out_shape,
                device=quant_out.device,
                group_size=block_size,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=False,
            )
        else:
            quant_out = None
            scale_out = None

        pattern_code = (
            AllGatherFusionPattern.kAllGatherfusedRMSFP8BlockWiseQuant
            if block_quant_fp8
            else AllGatherFusionPattern.kAllGatherfusedRMS
        )

        # allgather/reducescatter have no mnnvl implementation -- they run on
        # the IPC lamport workspace only. Since IPC is skipped on cross-node
        # groups, `initialized` can be True (mnnvl armed) while this workspace
        # is None; without this check a null pointer reaches the FFI.
        manager = _manager_for_group(group)
        if manager.workspace_tensor is None:
            raise RuntimeError(
                "trtllm allgather fusion requires the IPC lamport workspace, which is "
                "unavailable on this group (cross-node, or IPC explicitly "
                "skipped). Use the unfused path for this collective."
            )
        # A sentinel-width mismatch corrupts the lamport neg-zero wait/clear protocol.
        if (qkv.dtype == torch.float32) != manager.use_fp32_lamport:
            raise RuntimeError(
                "trtllm allgather fusion: payload width does not match "
                "the armed lamport sentinel"
            )

        trtllm_allgather_fusion(
            allgather_in=qkv,
            world_size=world_size,
            world_rank=rank,
            hidden_dim=hidden_dim,
            workspace_ptrs=_mark_captured(manager, manager.workspace_tensor),
            launch_with_pdl=launch_with_pdl,
            trigger_completion_at_end=trigger_completion_at_end,
            num_token_current_rank=num_token_current_rank,
            allgather_out=allgather_out,
            num_token_all_group=num_token_all_group,
            pattern_code=pattern_code,
            use_oneshot=True,
            fp32_acc=fp32_acc,
            x_norm_out=x_norm_out,
            y_norm_out=y_norm_out,
            quant_out=quant_out,
            scale_out=scale_out,
            x_rms_gamma=weight_q_a,
            y_rms_gamma=weight_kv_a,
            x_rms_eps=eps_q,
            y_rms_eps=eps_kv,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )

        return (
            allgather_out,
            quant_out if block_quant_fp8 else x_norm_out,
            y_norm_out,
            scale_out,
        )

    # Explicit-state plain collectives use private IPC allocations, not the
    # per-group fusion workspace above. Keep scratch isolated across streams.
    # Their registry modes are separate from stateless/auto-dispatched ops.

    class TrtllmAllGatherState:
        """Own BF16 AllGather IPC scratch for serialized auxiliary-stream calls."""

        def __init__(self, group, max_rows, hidden, device, oneshot):
            from tokenspeed_kernel.thirdparty.cuda.trtllm import (
                trtllm_create_ipc_workspace_for_allgather_fusion,
            )

            self.tp_size = group.size()
            if self.tp_size not in (2, 4, 8, 16) or not 0 < max_rows <= 128:
                raise ValueError(
                    "TRT-LLM one-shot gather requires TP2/4/8/16 and 1..128 rows"
                )
            if hidden <= 0 or hidden % 128:
                raise ValueError(
                    "TRT-LLM gather width must be a positive multiple of 128"
                )
            if self.tp_size * max_rows * hidden * 2 >= 2**31 - 2**21:
                raise ValueError("TRT-LLM gather exceeds one-shot Lamport capacity")
            self.max_rows = max_rows
            self.group, self.hidden, self.oneshot = group, hidden, oneshot
            # Plain gather has no normalization semantics: subdividing rows is an
            # exact view. Choose the largest divisor satisfying the fused wrapper's
            # <=2112 hidden limit and 128-element q_lora_rank alignment.
            self.kernel_hidden = next(
                width
                for width in range(min(hidden, 2112) // 128 * 128, 0, -128)
                if hidden % width == 0
            )
            self.handles, self.workspace = (
                trtllm_create_ipc_workspace_for_allgather_fusion(
                    tp_rank=group.rank(),
                    tp_size=self.tp_size,
                    max_token_num=self.tp_size
                    * max_rows
                    * hidden
                    // self.kernel_hidden,
                    hidden_dim=self.kernel_hidden,
                    use_fp32_lamport=False,
                    group=group,
                    create_metadata=False,
                )
            )
            self.control_ptr = self.workspace[-1].item()
            self.out = torch.empty(
                (self.tp_size * max_rows, hidden), dtype=torch.bfloat16, device=device
            )

        def gather(self, inputs):
            """Return borrowed [TP*M,H] BF16 rows, valid until the next gather."""
            if (
                inputs.dtype != torch.bfloat16
                or not inputs.is_contiguous()
                or inputs.shape[1] != self.hidden
                or not 0 < inputs.shape[0] <= self.max_rows
            ):
                raise ValueError("Invalid TRT-LLM AllGather input")
            from tokenspeed_kernel.thirdparty.cuda.trtllm import trtllm_allgather_fusion

            local = inputs.view(-1, self.kernel_hidden)
            out = self.out[: self.tp_size * inputs.shape[0]]
            trtllm_allgather_fusion(
                allgather_in=local,
                world_size=self.tp_size,
                world_rank=self.group.rank(),
                hidden_dim=self.kernel_hidden,
                workspace_ptrs=self.workspace,
                trigger_completion_at_end=False,
                num_token_current_rank=local.shape[0],
                allgather_out=out.view(-1, self.kernel_hidden),
                num_token_all_group=self.tp_size * local.shape[0],
                launch_with_pdl=False,
                pattern_code=0,
                use_oneshot=self.oneshot,
                fp32_acc=False,
                x_norm_out=None,
                y_norm_out=None,
                quant_out=None,
                scale_out=None,
                x_rms_gamma=None,
                y_rms_gamma=None,
                x_rms_eps=1e-6,
                y_rms_eps=1e-6,
                q_lora_rank=self.kernel_hidden,
                kv_lora_rank=0,
                qk_rope_head_dim=0,
            )
            return out

        def close(self):
            from tokenspeed_kernel.thirdparty.cuda.cuda_ipc import cudart
            from tokenspeed_kernel.thirdparty.cuda.trtllm import (
                trtllm_destroy_ipc_workspace_for_allgather_fusion,
            )

            torch.cuda.synchronize()
            dist.barrier(group=self.group)
            trtllm_destroy_ipc_workspace_for_allgather_fusion(
                self.handles, group=self.group
            )
            cudart.cudaFree(c_void_p(self.control_ptr))

    @register_kernel(
        "communication",
        "stateful_allgather",
        name="trtllm_allgather",
        solution="trtllm",
        signatures=format_signatures(("inputs",), "dense", {torch.bfloat16}),
    )
    def trtllm_allgather(state, inputs):
        """Gather BF16 [M,H] rows into borrowed [TP*M,H] subgroup-rank order.

        state is a preallocated TrtllmAllGatherState. All peers call on one
        serialized stream; finish reading the returned view before the next call.
        """
        return state.gather(inputs)

    class TrtllmReduceScatterState:
        """Own IPC scratch shared by sequential layers, not by concurrent streams.

        Args:
            group: 2/4/8/16-rank process group on a CUDA-IPC-accessible topology.
            max_rows: Prepared physical rows per rank, at most 128.
            hidden: Output width, a positive multiple of eight BF16 elements.
            device: Current rank's CUDA device, already selected by the caller.
        """

        def __init__(self, group, max_rows: int, hidden: int, device: torch.device):
            from tokenspeed_kernel.thirdparty.cuda.trtllm import (
                trtllm_create_ipc_workspace_for_reduce_scatter_fusion,
            )

            self.tp_size = group.size()
            if (
                self.tp_size not in (2, 4, 8, 16)
                or not 0 < max_rows <= 128
                or hidden <= 0
            ):
                raise ValueError(
                    "Lamport TRT-LLM reduction requires TP2/4/8/16, positive width and 1..128 rows"
                )
            if hidden % 8:
                raise ValueError(
                    "Lamport BF16 TRT-LLM reduction width must be divisible by 8"
                )
            # Bound allocation and prevent the native wrapper silently selecting
            # two-shot on inputs larger than its signed-int32 Lamport address space.
            if self.tp_size * self.tp_size * max_rows * hidden * 2 >= 2**31 - 2**21:
                raise ValueError("TRT-LLM reduction exceeds one-shot Lamport capacity")
            self.group = group
            self.max_rows = max_rows
            self.hidden = hidden
            self.buffer = torch.empty(
                (self.tp_size * max_rows, hidden), dtype=torch.bfloat16, device=device
            )
            self.handles, self.workspace = (
                trtllm_create_ipc_workspace_for_reduce_scatter_fusion(
                    tp_rank=group.rank(),
                    tp_size=self.tp_size,
                    max_token_num=self.tp_size * max_rows,
                    hidden_dim=hidden,
                    use_fp32_lamport=False,
                    group=group,
                    create_metadata=False,
                )
            )
            # The native helper's destroy routine frees shared IPC allocations,
            # but not the separately allocated local ring-control flags.
            self.control_ptr = self.workspace[-1].item()

        def input_buffer(self, rows: int) -> torch.Tensor:
            """Borrow a local GEMM destination; peers never read this buffer."""
            if self.handles is None or not 0 < rows <= self.max_rows:
                raise ValueError(
                    "Lamport TRT-LLM reduction is closed or exceeds capacity"
                )
            return self.buffer[: self.tp_size * rows]

        def close(self) -> None:
            """Collectively release IPC storage after all referencing graphs die."""
            from tokenspeed_kernel.thirdparty.cuda.cuda_ipc import cudart
            from tokenspeed_kernel.thirdparty.cuda.trtllm import (
                trtllm_destroy_ipc_workspace_for_reduce_scatter_fusion,
            )

            if self.handles is not None:
                torch.cuda.synchronize(self.buffer.device)
                dist.barrier(group=self.group)
                trtllm_destroy_ipc_workspace_for_reduce_scatter_fusion(
                    self.handles, group=self.group
                )
                cudart.cudaFree(c_void_p(self.control_ptr))
                self.handles = None

    @register_kernel(
        "communication",
        "stateful_reduce_scatter",
        name="trtllm_reduce_scatter",
        solution="trtllm",
        signatures=format_signatures(("partial",), "dense", {torch.bfloat16}),
    )
    def trtllm_reduce_scatter(state, partial, rows):
        """Reduce BF16 [TP*rows,H] partials into owned [rows,H] output.

        Prepare state collectively before capture. The native one-shot protocol
        publishes payloads into its own IPC ring and synchronizes their reuse;
        no symmetric-memory publication barrier is needed around local partials.
        Calls must be serialized on one stream. Output survives later state reuse.

        Args:
            state: Collectively prepared TrtllmReduceScatterState.
            partial: Contiguous BF16 [TP*rows,H] local GEMM partials.
            rows: Equal physical row count per rank, including padding.

        Returns:
            Owned BF16 [rows,H] tensor for the calling rank's token segment.
        """
        from tokenspeed_kernel.thirdparty.cuda.trtllm import trtllm_reducescatter_fusion

        destination = state.input_buffer(rows)
        if (
            partial.shape != destination.shape
            or partial.dtype != destination.dtype
            or partial.device != destination.device
            or not partial.is_contiguous()
        ):
            raise ValueError(
                "Lamport TRT-LLM reduction partials have incompatible layout"
            )
        out = torch.empty(
            (rows, state.hidden), dtype=partial.dtype, device=partial.device
        )
        trtllm_reducescatter_fusion(
            reducescatter_in=partial,
            world_size=state.tp_size,
            world_rank=state.group.rank(),
            token_num=state.tp_size * rows,
            hidden_dim=state.hidden,
            workspace_ptrs=state.workspace,
            trigger_completion_at_end=False,
            fp32_acc=True,
            num_token_current_rank=rows,
            pattern_code=0,
            launch_with_pdl=False,
            use_oneshot=True,
            reducescatter_out=out,
            add_in=None,
            residual_in=None,
            residual_out=None,
            norm_out=None,
            quant_out=None,
            scale_out=None,
            rms_gamma=None,
            rms_eps=None,
            scale_factor=None,
            layout_code=None,
            metadata=None,
        )
        return out
