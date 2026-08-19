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

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.gemm.fp8_utils import (
    create_per_token_group_quant_fp8_output_scale,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import ErrorClass, error_fn

logger = logging.getLogger(__name__)


__all__ = [
    "AllReduceFusionPattern",
    "allgather_dual_rmsnorm",
    "allreduce_residual_rmsnorm",
    "minimax_allreduce_rms_qk",
    "reducescatter_residual_rmsnorm",
    "trtllm_allreduce_fusion",
    "trtllm_create_ipc_workspace_for_all_reduce_fusion",
    "trtllm_create_ipc_workspace_for_minimax",
]

platform = current_platform()

AllReduceFusionPattern = ErrorClass
allgather_dual_rmsnorm = error_fn
allreduce_residual_rmsnorm = error_fn
allreduce_residual_attnres_combine = error_fn
allreduce_lane_latent_norm = error_fn
minimax_allreduce_rms_qk = error_fn
reducescatter_residual_rmsnorm = error_fn
trtllm_allreduce_fusion = error_fn
trtllm_create_ipc_workspace_for_all_reduce_fusion = error_fn
trtllm_create_ipc_workspace_for_minimax = error_fn

if current_platform().is_nvidia:
    from tokenspeed_kernel.ops.communication.fabric import fabric_allocation_supported
    from tokenspeed_kernel.thirdparty.cuda.trtllm import (
        _MNNVL_SUPPORTED_WORLD_SIZES,
        MNNVL_FLASHINFER_MIN_TOKENS,
        MNNVL_ONESHOT_MAX_TOKEN,
        MNNVL_PREFER_IPC_BYTES,
        AllGatherFusionPattern,
        AllReduceFusionPattern,
        ReduceScatterFusionPattern,
        _ar_should_use_oneshot,
        _load_trtllm_comm_module,
        flashinfer_mnnvl_module_available,
        minimax_allreduce_rms_qk,
        trtllm_allgather_fusion,
        trtllm_allreduce_fusion,
        trtllm_create_flashinfer_mnnvl_workspace_for_all_reduce_fusion,
        trtllm_create_ipc_workspace_for_all_reduce_fusion,
        trtllm_create_ipc_workspace_for_minimax,
        trtllm_create_mnnvl_workspace_for_all_reduce_fusion,
        trtllm_destroy_ipc_workspace_for_all_reduce_fusion,
        trtllm_reducescatter_fusion,
    )

    _workspace_manager = None

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
            logger.debug("mnnvl capability probe failed: %s", exc)
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
            logger.warning("mnnvl workspace creation failed, using IPC: %s", exc)

        ok = torch.tensor(
            [1 if workspace is not None else 0], dtype=torch.int32, device=device
        )
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
        if ok.item() == 0:
            return None
        logger.info(
            "MNNVL one-shot AR workspace armed: rank=%s world_size=%s "
            "max_token_num=%s hidden_dim=%s buffer=%s bytes",
            rank,
            world_size,
            workspace.max_token_num,
            hidden_dim,
            workspace.buffer_size_bytes,
        )
        return workspace

    def _mnnvl_fi_locally_available(world_size: int) -> bool:
        """Non-collective capability probe for the upstream flashinfer MNNVL AR.

        Same hardware requirements as the private kernel (NVLS multicast,
        fabric-handle memory for cross-host groups) minus the world-size
        whitelist: the upstream kernels serve arbitrary group sizes. Purely
        local: safe to call before any collective.
        """
        if world_size <= 1 or not flashinfer_mnnvl_module_available():
            return False
        try:
            if torch.cuda.get_device_capability()[0] < 9:
                return False
            from torch._C._autograd import DeviceType
            from torch._C._distributed_c10d import _SymmetricMemory

            if not _SymmetricMemory.has_multicast_support(
                DeviceType.CUDA, torch.cuda.current_device()
            ):
                return False
            # Cross-host groups need fabric-handle memory; the multicast
            # capability alone is advertised even where the IMEX stack is
            # absent and the handle exchange would then hang.
            if world_size > torch.cuda.device_count() and not (
                fabric_allocation_supported(torch.cuda.current_device())
            ):
                return False
            return True
        except Exception as exc:  # noqa: BLE001 - capability probe must not raise
            logger.debug("flashinfer mnnvl capability probe failed: %s", exc)
            return False

    # One flashinfer workspace per process group, shared by every arming site
    # (comm_backend per-group managers and the fused-pattern manager here).
    # The multicast allocation granularity (hundreds of MB) dwarfs any
    # requested size, so the first allocation has capacity for all callers;
    # per-call sufficiency is still checked in supports(). Keyed by the
    # group's global ranks -- distinct ProcessGroup objects over the same
    # ranks share one workspace. Process-lifetime, like symm_mem allocations.
    # Sharing requires all users of a group to issue their AR calls in a
    # single total order (one compute stream), which the engine guarantees.
    _mnnvl_fi_workspace_cache: dict = {}

    def _try_create_mnnvl_fi_workspace(
        rank: int,
        world_size: int,
        max_token_num: int,
        hidden_dim: int,
        group,
    ):
        """Collectively arm (or reuse) the upstream flashinfer MNNVL workspace.

        Same two-phase agreement as :func:`_try_create_mnnvl_workspace`:
        (1) all-reduce the local capability probe before the multicast handle
        exchange, (2) all-reduce the creation result. Returns None on fallback
        so every rank takes the same path. The cache lookup key is local
        metadata computed identically on every rank, so hits and misses are
        rank-uniform too.
        """
        cache_key = tuple(dist.get_process_group_ranks(group))
        cached = _mnnvl_fi_workspace_cache.get(cache_key)
        if cached is not None:
            return cached
        device = torch.device("cuda", torch.cuda.current_device())
        ok = torch.tensor(
            [1 if _mnnvl_fi_locally_available(world_size) else 0],
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
        if ok.item() == 0:
            return None

        workspace = None
        try:
            workspace = trtllm_create_flashinfer_mnnvl_workspace_for_all_reduce_fusion(
                rank, world_size, max_token_num, hidden_dim, group=group
            )
        except Exception as exc:  # noqa: BLE001 - fall back to private/IPC
            logger.warning("flashinfer mnnvl workspace creation failed: %s", exc)

        ok = torch.tensor(
            [1 if workspace is not None else 0], dtype=torch.int32, device=device
        )
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
        if ok.item() == 0:
            return None
        logger.info(
            "flashinfer MNNVL AR workspace armed: rank=%s world_size=%s "
            "max_token_num=%s hidden_dim=%s buffer=%s bytes",
            rank,
            world_size,
            workspace.max_token_num,
            hidden_dim,
            workspace.buffer_size_bytes,
        )
        _mnnvl_fi_workspace_cache[cache_key] = workspace
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
            self.mnnvl_fi_workspace = None
            self.world_size = None
            self.rank = None
            self.max_token_num = None
            self.hidden_dim = None
            self.use_fp32_lamport = None
            self.initialized = False
            self.group_ranks = (
                None  # tuple of global ranks this workspace was created for
            )

        def initialize(
            self,
            world_size: int,
            rank: int,
            max_token_num: int,
            hidden_dim: int,
            group,
            use_fp32_lamport: bool = False,
        ):
            """Initialize workspace"""
            if (
                self.initialized
                and self.world_size == world_size
                and self.max_token_num == max_token_num
                and self.hidden_dim == hidden_dim
                and self.use_fp32_lamport == use_fp32_lamport
            ):
                return

            self.cleanup()
            # CUDA-IPC handles cannot span nodes -- attempting creation on a
            # cross-node group fails AND leaves a sticky CUDA context error
            # ('invalid resource handle' on the next allocation). Gate it off
            # for cross-node runs; the MNNVL fabric workspace below is the
            # multi-node path.
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
            # Upstream flashinfer MNNVL workspace: preferred provider for the
            # generic kAllReduce / kARResidualRMSNorm patterns (faster kernels,
            # maintained upstream). The private mnnvl workspace above remains
            # for the K3-specific epilogue patterns upstream does not have.
            self.mnnvl_fi_workspace = _try_create_mnnvl_fi_workspace(
                rank, world_size, max_token_num, hidden_dim, group
            )

            # With IPC skipped, mnnvl is the only workspace; if it failed to
            # arm there is nothing to fuse with -- stay uninitialized so
            # prepare_allreduce_fusion() returns False and the model layer
            # keeps the plain NCCL path.
            if (
                self.workspace_tensor is None
                and self.mnnvl_workspace is None
                and self.mnnvl_fi_workspace is None
            ):
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
            self.group = group

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
                    # The flashinfer workspace is shared across arming sites
                    # via the process-lifetime cache; drop the reference only,
                    # never destroy() it out from under other users.
                    self.mnnvl_fi_workspace = None
                    self.initialized = False
                    self.world_size = None
                    self.rank = None
                    self.max_token_num = None
                    self.hidden_dim = None
                    self.use_fp32_lamport = None
                    self.group_ranks = None

    _workspace_manager = TrtllmFusionWorkspaceManager()

    #
    #  # Reduce-scatter now reuses `_workspace_manager` (allreduce-style IPC workspace).
    # This avoids keeping a second, similarly-sized workspace alive.

    def ensure_workspace_initialized(
        rank: int,
        group: dist.ProcessGroup,
        max_token_num: int = 2048,
        hidden_dim: int = 4096,
        use_fp32_lamport: bool = False,
    ):
        world_size = group.size()
        if world_size <= 1:
            return False

        target_max_token_num = max_token_num
        target_hidden_dim = hidden_dim
        target_use_fp32_lamport = use_fp32_lamport
        if (
            _workspace_manager.initialized
            and _workspace_manager.world_size == world_size
        ):
            if _workspace_manager.max_token_num is not None:
                target_max_token_num = max(
                    _workspace_manager.max_token_num, max_token_num
                )
            if _workspace_manager.hidden_dim is not None:
                target_hidden_dim = max(_workspace_manager.hidden_dim, hidden_dim)
            if _workspace_manager.use_fp32_lamport:
                target_use_fp32_lamport = True

        if (
            (not _workspace_manager.initialized)
            or (_workspace_manager.world_size != world_size)
            or (_workspace_manager.max_token_num != target_max_token_num)
            or (_workspace_manager.hidden_dim != target_hidden_dim)
            or (_workspace_manager.use_fp32_lamport != target_use_fp32_lamport)
        ):
            logger.info(
                "Re/initializing TRT-LLM fusion IPC workspace: "
                "world_size=%s rank=%s max_token_num=%s hidden_dim=%s use_fp32_lamport=%s "
                "(prev max_token_num=%s hidden_dim=%s use_fp32_lamport=%s)",
                world_size,
                rank,
                target_max_token_num,
                target_hidden_dim,
                target_use_fp32_lamport,
                _workspace_manager.max_token_num,
                _workspace_manager.hidden_dim,
                _workspace_manager.use_fp32_lamport,
            )
            _workspace_manager.initialize(
                world_size=world_size,
                rank=rank,
                max_token_num=target_max_token_num,
                hidden_dim=target_hidden_dim,
                use_fp32_lamport=target_use_fp32_lamport,
                group=group,
            )

        return _workspace_manager.initialized

    def _ar_fusion_workspace(
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
          1. IPC exists (single-node) AND (payload >= MNNVL_PREFER_IPC_BYTES
             OR pattern == kAllReduceLatentNorm) ....... IPC lamport/twoshot
          2. between the mnnvl workspaces: upstream flashinfer when
             token_num >= MNNVL_FLASHINFER_MIN_TOKENS or cross-node (generic
             patterns only), the private kernel otherwise (and always for the
             K3-specific epilogue patterns); whichever was not preferred is
             the fallback when the preferred one rejects the shape
          3. no IPC fallback (cross-node) ............. None (caller degrades:
             the rmsnorm family runs unfused NCCL + torch epilogue, the rest
             raise loudly -- never a null workspace into the kernel)
          4. otherwise ................................ IPC
        Cross-node, workspace_tensor is None so only 2-3 apply -- the two
        mnnvl workspaces together serve the whole range (they beat NCCL
        everywhere there).
        """
        mnnvl = _workspace_manager.mnnvl_workspace
        mnnvl_fi = _workspace_manager.mnnvl_fi_workspace
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
        if _workspace_manager.workspace_tensor is not None and prefer_ipc:
            return _workspace_manager.workspace_tensor
        # Between the two mnnvl workspaces, prefer upstream flashinfer from
        # MNNVL_FLASHINFER_MIN_TOKENS up and cross-node (no IPC): that is
        # where it decisively wins (1.8-3.2x by M=64-128). Decode-sized calls
        # stay on the private kernel -- in-situ the two are equals there and
        # flashinfer's multicast traffic slows overlapped neighbors (see the
        # constant's note). Either serves as the other's fallback when its
        # supports() rejects a shape.
        prefer_fi = (
            token_num >= MNNVL_FLASHINFER_MIN_TOKENS
            or _workspace_manager.workspace_tensor is None
        )
        if (
            prefer_fi
            and mnnvl_fi is not None
            and mnnvl_fi.supports(
                token_num,
                hidden_dim,
                dtype,
                _workspace_manager.world_size,
                pattern_code,
                use_oneshot=use_oneshot,
                residual_reduce_scattered=residual_reduce_scattered,
            )
        ):
            return mnnvl_fi
        if mnnvl is not None and mnnvl.supports(
            token_num,
            hidden_dim,
            dtype,
            _workspace_manager.world_size,
            pattern_code,
            use_oneshot=use_oneshot,
            residual_reduce_scattered=residual_reduce_scattered,
        ):
            return mnnvl
        if (
            not prefer_fi
            and mnnvl_fi is not None
            and mnnvl_fi.supports(
                token_num,
                hidden_dim,
                dtype,
                _workspace_manager.world_size,
                pattern_code,
                use_oneshot=use_oneshot,
                residual_reduce_scattered=residual_reduce_scattered,
            )
        ):
            return mnnvl_fi
        # Cross-node there is no IPC workspace, so a shape/pattern mnnvl rejects
        # has no fused home. Return None and let the caller decide: the rmsnorm
        # family degrades to the unfused NCCL path, the rest raise loudly. Never
        # hand the kernel a null workspace.
        if _workspace_manager.workspace_tensor is None:
            logger.debug(
                "trtllm AR fusion: shape (tokens=%s, hidden=%s, dtype=%s, "
                "pattern=%s, oneshot=%s) not supported by mnnvl and no IPC "
                "workspace; caller falls back unfused",
                token_num,
                hidden_dim,
                dtype,
                pattern_code,
                use_oneshot,
            )
            return None
        return _workspace_manager.workspace_tensor

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
        launch_with_pdl: bool = False,
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

        resolved_oneshot = (
            use_oneshot
            if use_oneshot is not None
            else _ar_should_use_oneshot(
                token_num, hidden_dim, input_tensor.dtype, world_size
            )
        )
        # The size heuristic above encodes the IPC-lamport crossover in BYTES,
        # while the mnnvl one-shot kernel is capped in TOKENS
        # (MNNVL_ONESHOT_MAX_TOKEN). They disagree: at world=8 the 42 MB
        # threshold lands on 192 tokens, which one-shot mnnvl cannot serve, and
        # cross-node there is no IPC workspace to fall back to. Respect the
        # kernel's own cap so such shapes take the two-shot path instead.
        if (
            resolved_oneshot
            and token_num > MNNVL_ONESHOT_MAX_TOKEN
            and _workspace_manager.mnnvl_workspace is not None
            and _workspace_manager.workspace_tensor is None
        ):
            resolved_oneshot = False

        workspace = _ar_fusion_workspace(
            token_num,
            hidden_dim,
            input_tensor.dtype,
            pattern_code,
            resolved_oneshot,
            residual_reduce_scattered,
        )
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
        launch_with_pdl: bool = False,
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
            token_num,
            hidden_dim,
            input_tensor.dtype,
            AllReduceFusionPattern.kARResidualAttnResCombine,
            use_oneshot=True,
        )
        if workspace is None:
            # mnnvl serves this pattern in-range; only out-of-range shapes land
            # here cross-node. No unfused equivalent of the combine epilogue --
            # fail loudly rather than skip the reduce.
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
        launch_with_pdl: bool = False,
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
            token_num,
            lane_dim,
            lane.dtype,
            AllReduceFusionPattern.kAllReduceLatentNorm,
            use_oneshot=True,
        )
        if workspace is None:
            # mnnvl serves this pattern in-range; only out-of-range shapes land
            # here cross-node. Fail loudly rather than skip the reduce.
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
        launch_with_pdl: bool = False,
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
        if _workspace_manager.workspace_tensor is None:
            raise RuntimeError(
                "trtllm reducescatter fusion requires the IPC lamport workspace, which is "
                "unavailable on this group (cross-node, or IPC explicitly "
                "skipped). Use the unfused path for this collective."
            )

        trtllm_reducescatter_fusion(
            reducescatter_in=input_tensor,
            world_size=world_size,
            world_rank=rank,
            token_num=token_num,
            hidden_dim=hidden_dim,
            workspace_ptrs=_workspace_manager.workspace_tensor,
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
        launch_with_pdl: bool = False,
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
        if _workspace_manager.workspace_tensor is None:
            raise RuntimeError(
                "trtllm allgather fusion requires the IPC lamport workspace, which is "
                "unavailable on this group (cross-node, or IPC explicitly "
                "skipped). Use the unfused path for this collective."
            )

        trtllm_allgather_fusion(
            allgather_in=qkv,
            world_size=world_size,
            world_rank=rank,
            hidden_dim=hidden_dim,
            workspace_ptrs=_workspace_manager.workspace_tensor,
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
