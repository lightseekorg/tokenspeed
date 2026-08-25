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

"""Communication fusion kernels (AOT-compiled).
Drop-in replacement for `flashinfer.comm` used by TokenSpeed.
Loads the pre-compiled trtllm_comm.so via tvm_ffi instead of JIT.
Usage:
    import tokenspeed_kernel.comm as comm
    # Then use comm.trtllm_allreduce_fusion(...), comm.AllReduceFusionPattern, etc.
"""

import functools
import logging
import os
from ctypes import c_void_p, cast
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from tokenspeed_kernel.thirdparty.cuda.cuda_ipc import (
    create_shared_buffer,
    cudart,
    free_shared_buffer,
)
from torch.distributed import ProcessGroup

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


BarrierFlagCount = 256
MAX_COMM_SIZE = 2147483647 & ~((1 << 21) - 1)  # MAX_INT32 rounded down to 2MB


# ---------------------------------------------------------------------------
# AOT module loader (replaces JIT gen_trtllm_comm_module().build_and_load())
# ---------------------------------------------------------------------------


@functools.cache
def _load_trtllm_comm_module():
    import tvm_ffi

    so_path = (
        Path(__file__).resolve().parent / "objs" / "trtllm_comm" / "trtllm_comm.so"
    )
    if not so_path.exists():
        raise RuntimeError(
            f"trtllm_comm.so not found at {so_path}. "
            "Run `python tokenspeed_kernel/setup.py build_ext` to compile."
        )
    return tvm_ffi.load_module(str(so_path))


# ---------------------------------------------------------------------------
# Pattern enums (pure Python, identical to flashinfer)
# ---------------------------------------------------------------------------


class AllReduceStrategyType:
    NCCL = 0
    MIN_LATENCY = 1
    UB = 2
    AUTO = 3
    ONESHOT = 4
    TWOSHOT = 5
    LOWPRECISION = 6


class AllReduceStrategyConfig:
    USE_MEMCPY = 1 << 0
    PUSH_MODE = 1 << 1


class AllReduceFusionOp:
    NONE = 0
    RESIDUAL_RMS_NORM = 1
    LAST_PROCESS_FOR_UB = 2
    RESIDUAL_RMS_PREPOST_NORM = 3
    RESIDUAL_RMS_NORM_QUANT_FP8 = 4
    RESIDUAL_RMS_NORM_QUANT_NVFP4 = 5
    RESIDUAL_RMS_NORM_OUT_QUANT_FP8 = 6
    RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4 = 7
    MOE_ALLREDUCE_RESIDUAL_RMS_NORM = 8
    MOE_FINALIZE_ALLREDUCE_RESIDUAL_RMS_NORM = 9


class AllReduceFusionPattern:
    kAllReduce = 0
    kARResidualRMSNorm = 1
    kARResidualRMSNormFP8Quant = 2
    kARResidualRMSNormFP4Quant = 3
    kARResidualRMSNormOutFP8Quant = 4
    kARResidualRMSNormOutFP4Quant = 5
    kARResidualRMSNormFP8BlockWiseQuant = 6
    kARResidualRMSNormPartialOutFP8BlockWiseQuant = 7
    kARResidualRMSNormPartialOut = 8
    kARResidualAttnResCombine = 9
    kAllReduceLatentNorm = 10


class AllGatherFusionPattern:
    kAllGather = 0
    kAllGatherfusedRMS = 1
    kAllGatherfusedRMSFP8BlockWiseQuant = 2


class ReduceScatterFusionPattern:
    kReduceScatter = 0
    kRSResidualRMSNorm = 1
    kRSResidualRMSNormFP8Quant = 2
    kRSResidualRMSNormFP4Quant = 3
    kRSResidualRMSNormOutFP8Quant = 4
    kRSResidualRMSNormOutFP4Quant = 5
    kRSResidualRMSNormFP8BlockWiseQuant = 6
    kRSAddResidualRMSNormFP8BlockWiseQuant = 7
    kRSAddResidualRMSNorm = 8


class QuantizationSFLayout:
    SWIZZLED_128x4 = 0
    SWIZZLED_8x4 = 1
    LINEAR = 2


# ---------------------------------------------------------------------------
# Lamport initialization
# ---------------------------------------------------------------------------


def trtllm_lamport_initialize(buffer_ptr: int, size: int, dtype: torch.dtype) -> None:
    _load_trtllm_comm_module().trtllm_lamport_initialize(buffer_ptr, size, dtype)


def trtllm_lamport_initialize_all(
    buffer_0_ptr: int,
    buffer_1_ptr: int,
    buffer_2_ptr: int,
    size: int,
    dtype: torch.dtype,
) -> None:
    _load_trtllm_comm_module().trtllm_lamport_initialize_all(
        buffer_0_ptr, buffer_1_ptr, buffer_2_ptr, size, dtype
    )


# ---------------------------------------------------------------------------
# IPC workspace helpers (shared pattern for allreduce/allgather/reducescatter)
# ---------------------------------------------------------------------------


def _create_ipc_workspace(
    tp_rank: int,
    tp_size: int,
    buffer_size: int,
    flag_size: int,
    lamport_comm_size: int,
    use_fp32_lamport: bool,
    group: Optional[ProcessGroup],
) -> Tuple[List[List[int]], torch.Tensor]:
    """Common IPC workspace creation logic."""
    if lamport_comm_size > MAX_COMM_SIZE:
        logging.warning(
            f"lamport_comm_size {lamport_comm_size} > MAX_COMM_SIZE {MAX_COMM_SIZE}, clamping"
        )
        lamport_comm_size = MAX_COMM_SIZE

    lamport_buffer_size = lamport_comm_size * 3

    ipc_handles: List[List[int]] = []
    for size in [buffer_size, flag_size, lamport_buffer_size]:
        aligned_size = _round_up(size, 1 << 21)
        ipc_handles.append(create_shared_buffer(aligned_size, group))

    # Initialize lamport buffer
    aligned_lamport_buffer_size = _round_up(lamport_buffer_size, 1 << 21)
    if use_fp32_lamport:
        trtllm_lamport_initialize(
            ipc_handles[2][tp_rank], aligned_lamport_buffer_size // 4, torch.float32
        )
    else:
        trtllm_lamport_initialize(
            ipc_handles[2][tp_rank], aligned_lamport_buffer_size // 2, torch.float16
        )

    # Build workspace pointer list
    workspace = []
    for ipc_handle in ipc_handles:
        for rank in range(tp_size):
            workspace.append(ipc_handle[rank])

    # Allocate and initialize flags: [0, 0, 0, lamport_comm_size, 0]
    flag_ptr = cudart.cudaMalloc(5 * 4)
    cudart.cudaMemset(flag_ptr, 0, 5 * 4)
    lamport_comm_size_bytes = lamport_comm_size.to_bytes(4, byteorder="little")
    cudart.cudaMemcpy(
        c_void_p(flag_ptr.value + 3 * 4), cast(lamport_comm_size_bytes, c_void_p), 4
    )
    workspace.append(flag_ptr.value)

    workspace_tensor = torch.tensor(
        workspace, dtype=torch.int64, device=torch.device("cuda")
    )

    dist.barrier(group=group)
    return ipc_handles, workspace_tensor


def _destroy_ipc_workspace(
    workspace: List[List[int]], group: Optional[ProcessGroup] = None
) -> None:
    for ipc_handle in workspace:
        free_shared_buffer(ipc_handle, group)


# ---------------------------------------------------------------------------
# MNNVL-structured one-shot workspace (NVLS multicast + Lamport rotation)
# ---------------------------------------------------------------------------

# Match flashinfer's one-shot traffic threshold. Traffic includes the payload
# sent by every rank, so the token boundary varies with hidden width, dtype,
# and TP size.
MNNVL_ONESHOT_BYTES = int(
    os.environ.get("TOKENSPEED_MNNVL_ONESHOT_BYTES", 64 * 1024 * 8 * 2)
)


def _mnnvl_oneshot_token_cap(
    hidden_dim: int,
    elem_size: int,
    world_size: int,
    threshold_bytes: int = MNNVL_ONESHOT_BYTES,
) -> int:
    """Maximum token count admitted by the aggregate-traffic rule."""
    return threshold_bytes // (hidden_dim * world_size * elem_size)


def _mnnvl_should_use_oneshot(
    token_num: int,
    hidden_dim: int,
    elem_size: int,
    world_size: int,
    threshold_bytes: int = MNNVL_ONESHOT_BYTES,
) -> bool:
    """Select one-shot when aggregate one-shot traffic is within its limit."""
    return token_num <= _mnnvl_oneshot_token_cap(
        hidden_dim, elem_size, world_size, threshold_bytes
    )


# Two-shot serves calls through 2048 tokens; mirrors kMnnvlTwoShotMaxToken in the
# kernel header. Workspace slot layout: [A: token][rank][hidden] + [B: token][hidden].
MNNVL_TWOSHOT_MAX_TOKEN = 2048

# Above this payload the IPC lamport workspace beats the mnnvl one on a single
# node: multicast/scatter wins while cost tracks the number of stores, but past
# ~10 MiB bandwidth dominates and IPC's direct peer writes come out ahead.
# Measured at world 4, hidden 7168, bf16, with the scatter phase A: mnnvl leads
# through 768 tokens (10.5 MiB, 60.0 vs 63.4 us) and trails from 1024 (14 MiB,
# 86.1 vs 75.4). Expressed in input-tensor bytes so it holds across hidden sizes
# and dtypes. Only reachable single-node -- cross-node there is no IPC
# workspace, and there mnnvl beats NCCL across the whole supported range.
MNNVL_PREFER_IPC_BYTES = int(
    os.environ.get("TOKENSPEED_MNNVL_PREFER_IPC_BYTES", 12 * 1024 * 1024)
)

_MNNVL_SUPPORTED_PATTERNS = frozenset(
    {
        AllReduceFusionPattern.kAllReduce,
        AllReduceFusionPattern.kARResidualRMSNorm,
        AllReduceFusionPattern.kARResidualAttnResCombine,
        # kAllReduceLatentNorm: the mnnvl kernel handles it, but the wide
        # [latent|hidden] lane makes it a hair slower than IPC lamport on a
        # single node (8.60us vs 6.64 on 8x B300), so _ar_fusion_workspace
        # prefers IPC for it when an IPC workspace exists. Cross-node there is
        # no IPC and the alternative is NCCL, which mnnvl beats ~3.3x --
        # excluding it there crashed K3's latent-MoE tail with no fallback.
        AllReduceFusionPattern.kAllReduceLatentNorm,
    }
)

# 16 covers 4-node TP16 (Kimi-K3); the NVLink domain spans the rack, so the
# limit was template instantiation, not fabric reach.
_MNNVL_SUPPORTED_WORLD_SIZES = (2, 4, 8, 16)


def _mnnvl_grid_config_ok(hidden_dim: int, elem_size: int) -> bool:
    """Mirror of mnnvl_oneshot_grid_config's feasibility check (exact,
    warp-aligned partition of the hidden dim across one cluster)."""
    vec = 16 // elem_size
    if hidden_dim % vec != 0:
        return False
    threads = hidden_dim // vec
    cluster = 1
    while threads // cluster > 1024:
        cluster *= 2
    return cluster <= 8 and threads % cluster == 0 and (threads // cluster) % 32 == 0


class MnnvlAllReduceFusionWorkspace:
    """Symmetric-memory workspace for the MNNVL-structured one-shot AR.

    Layout: one symm_mem allocation = 3 lamport buffers of
    ``buffer_size_bytes`` each, pre-filled with the fp32 -0.0 sentinel, plus a
    9-word uint32 ``buffer_flags`` rotation-state array (see the protocol notes
    in trtllm_mnnvl_allreduce_fusion.cuh). Passing an instance of this class as
    ``workspace_ptrs`` to :func:`trtllm_allreduce_fusion` routes the call onto
    the mnnvl kernel; a plain int64 tensor keeps the IPC lamport path.
    """

    backend = "mnnvl"

    def __init__(
        self,
        tp_rank: int,
        tp_size: int,
        max_token_num: int,
        hidden_dim: int,
        buffer_size_bytes: int,
        multicast_ptr: int,
        peer_ptrs: torch.Tensor,
        local_ptr: int,
        buffer_flags: torch.Tensor,
        oneshot_token_cap: int,
        refs: tuple,
    ):
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.max_token_num = max_token_num
        self.hidden_dim = hidden_dim
        self.buffer_size_bytes = buffer_size_bytes
        self.multicast_ptr = multicast_ptr
        self.peer_ptrs = peer_ptrs
        self.local_ptr = local_ptr
        self.buffer_flags = buffer_flags
        # Frozen when the allocation is built: dispatch must never re-read an
        # environment override that could select a layout this buffer cannot hold.
        self.oneshot_token_cap = oneshot_token_cap
        self._refs = refs  # keep the symm_mem tensor + handle alive

    def resolve_use_oneshot(self, token_num: int, requested: Optional[bool]) -> bool:
        """Resolve dispatch against the one-shot lane this workspace owns."""
        if requested is False:
            return False
        # This cap is both the creation-time traffic rule and the allocation
        # bound. A forced request beyond it must use the always-sized two-shot lane.
        return token_num <= self.oneshot_token_cap

    def supports(
        self,
        token_num: int,
        hidden_dim: int,
        dtype: torch.dtype,
        world_size: int,
        pattern_code: int,
        use_oneshot: Optional[bool] = None,
        residual_reduce_scattered: bool = False,
    ) -> bool:
        """Whether this call shape can run on the mnnvl kernel.

        Args:
            token_num: number of tokens in this call.
            hidden_dim: hidden (lane) width of this call.
            dtype: payload dtype (fp16/bf16 only).
            world_size: TP world size of this call.
            pattern_code: AllReduceFusionPattern value.
            use_oneshot: strategy the device launcher will execute.
            residual_reduce_scattered: RS-residual mode (unsupported).

        Returns:
            True when the mnnvl path can serve the call; callers must fall
            back to the IPC lamport workspace otherwise.
        """
        if residual_reduce_scattered:
            return False
        # Both strategies are served through MNNVL_TWOSHOT_MAX_TOKEN when the
        # workspace was sized for the requested shape.
        if world_size != self.tp_size or world_size not in _MNNVL_SUPPORTED_WORLD_SIZES:
            return False
        if dtype not in (torch.bfloat16, torch.float16):
            return False
        if pattern_code not in _MNNVL_SUPPORTED_PATTERNS:
            return False
        # token_num <= 0 would reach cudaLaunchKernelEx with gridDim.x = 0:
        # the launch fails with a STICKY invalid-argument error that poisons
        # the CUDA context, unlike every other rejected shape which raises a
        # recoverable RuntimeError before any launch.
        if token_num <= 0 or token_num > MNNVL_TWOSHOT_MAX_TOKEN:
            return False
        use_oneshot = self.resolve_use_oneshot(token_num, use_oneshot)
        # Validate the layout selected by the real device parameter. Resolution
        # above downgrades an unsafe forced one-shot request to the always-sized
        # two-shot layout.
        if not use_oneshot:
            stage_tokens = -(-token_num // world_size) * world_size
            lane_tokens = 2 * stage_tokens
        else:
            lane_tokens = token_num * world_size
        payload = lane_tokens * hidden_dim * dtype.itemsize
        if payload > self.buffer_size_bytes:
            return False
        return _mnnvl_grid_config_ok(hidden_dim, dtype.itemsize)


def trtllm_create_mnnvl_workspace_for_all_reduce_fusion(
    tp_rank: int,
    tp_size: int,
    max_token_num: int,
    hidden_dim: int,
    group: Optional[ProcessGroup] = None,
    dtype_elem_size: int = 2,
) -> MnnvlAllReduceFusionWorkspace:
    """Create the symmetric-memory workspace for the mnnvl one-shot AR.

    Collective: every rank of ``group`` must call this in lockstep (symm_mem
    rendezvous). Raises when NVLS multicast is unavailable; callers should
    probe capabilities first (see ops/communication/trtllm.py).

    Args:
        tp_rank: this rank within the group.
        tp_size: group world size (2, 4, 8 or 16).
        max_token_num: maximum tokens per call; clamped to
            ``MNNVL_TWOSHOT_MAX_TOKEN``.
        hidden_dim: maximum hidden (lane) width the workspace must hold.
        group: the process group to rendezvous over.
        dtype_elem_size: payload element size in bytes (2 for bf16/fp16).

    Returns:
        MnnvlAllReduceFusionWorkspace bound to this group.
    """
    import torch.distributed._symmetric_memory as symm_mem

    if tp_size not in _MNNVL_SUPPORTED_WORLD_SIZES:
        raise RuntimeError(f"mnnvl workspace: unsupported tp_size {tp_size}")

    device = torch.device("cuda", torch.cuda.current_device())
    ws_max_token = min(max_token_num, MNNVL_TWOSHOT_MAX_TOKEN)
    oneshot_threshold_bytes = MNNVL_ONESHOT_BYTES
    oneshot_token_cap = _mnnvl_oneshot_token_cap(
        hidden_dim, dtype_elem_size, tp_size, oneshot_threshold_bytes
    )
    # ONE buffer serves BOTH kernels, so it must fit whichever layout is larger:
    #   two-shot: 2 stages of ceil(T/tp)*tp vectors (SCATTER is shard-local);
    #   one-shot: T*tp vectors, but only through its creation-time traffic cap.
    # Do not "fix" this back to ws_max_token: the byte rule can never select
    # one-shot above this cap, and dispatch is pinned to the same stored value.
    stage_tokens = -(-ws_max_token // tp_size) * tp_size
    lane_tokens = max(
        2 * stage_tokens,
        min(ws_max_token, oneshot_token_cap) * tp_size,
    )
    bytes_per_buffer = _round_up(lane_tokens * hidden_dim * dtype_elem_size, 32)
    total_bytes = 3 * bytes_per_buffer

    buf = symm_mem.empty(total_bytes // 4, dtype=torch.float32, device=device)
    handle = symm_mem.rendezvous(buf, group)
    multicast_ptr = handle.multicast_ptr
    if multicast_ptr is None or multicast_ptr == 0:
        raise RuntimeError("mnnvl workspace: NVLS multicast mapping unavailable")

    # The allocation may be larger than requested; partition the actual size
    # and initialize the WHOLE local allocation (not just the requested
    # prefix) with the fp32 -0.0 Lamport sentinel.
    # Round to 32, not 16: the kernel's two-stage layout puts stage 1 at
    # bytes_per_buffer // 2, and that base must stay 16-byte aligned for the
    # 128-bit lamport accesses. 16-aligned bpb only guarantees 8-aligned bpb/2.
    buffer_size_bytes = handle.buffer_size // 3 // 32 * 32
    if buffer_size_bytes < bytes_per_buffer:
        raise RuntimeError("mnnvl workspace: symmetric allocation too small")
    handle.get_buffer(tp_rank, (handle.buffer_size // 4,), torch.float32).fill_(-0.0)

    # Rotation state, same layout as flashinfer's buffer_flags:
    # [cur idx, dirty idx, bytes per buffer, dirty stages, bytes_to_clear[4],
    #  arrival counter]
    buffer_flags = torch.tensor(
        [0, 2, buffer_size_bytes, 0, 0, 0, 0, 0, 0],
        dtype=torch.uint32,
        device=device,
    )
    local_ptr = int(handle.buffer_ptrs[tp_rank])
    # Two-shot phase A scatters each rank's shard to its OWNER only, so the
    # kernel needs every peer's unicast base, not just the local one and the
    # multicast VA. Broadcasting there (as one-shot does) would make two-shot
    # move MORE bytes than one-shot, defeating its purpose.
    peer_ptrs = torch.tensor(
        [int(pt) for pt in handle.buffer_ptrs], dtype=torch.int64, device=device
    )

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier(group=group)

    return MnnvlAllReduceFusionWorkspace(
        tp_rank=tp_rank,
        tp_size=tp_size,
        max_token_num=ws_max_token,
        hidden_dim=hidden_dim,
        buffer_size_bytes=buffer_size_bytes,
        multicast_ptr=int(multicast_ptr),
        peer_ptrs=peer_ptrs,
        local_ptr=local_ptr,
        buffer_flags=buffer_flags,
        oneshot_token_cap=oneshot_token_cap,
        refs=(buf, handle),
    )


# ---------------------------------------------------------------------------
# AllReduce fusion
# ---------------------------------------------------------------------------

# One-shot payload ceiling in MiB per world size, for the IPC lamport path.
# 16 extrapolates the 4->8 halving (64 -> 42 -> ~21): without an entry
# `.get(world, 0)` returns 0, which silently forced every TP16 call onto the
# two-shot path regardless of size.
_ar_oneshot_heuristics: dict = {2: 512, 4: 64, 8: 42, 16: 21}


def _ar_should_use_oneshot(
    token_num: int, hidden_dim: int, dtype: torch.dtype, world_size: int
) -> bool:
    comm_size_mb = (
        token_num * hidden_dim * 2 * world_size * dtype.itemsize / 1024 / 1024
    )
    return comm_size_mb <= _ar_oneshot_heuristics.get(world_size, 0)


def trtllm_create_ipc_workspace_for_all_reduce_fusion(
    tp_rank: int,
    tp_size: int,
    max_token_num: int,
    hidden_dim,
    use_fp32_lamport: bool = False,
    group: Optional[ProcessGroup] = None,
    create_metadata: bool = False,
) -> Union[
    Tuple[List[List[int]], torch.Tensor],
    Tuple[List[List[int]], torch.Tensor, dict],
]:
    buffer_size = tp_size * max_token_num * hidden_dim * 2
    flag_size = tp_size * BarrierFlagCount * 4
    lamport_comm_size = (
        tp_size * max_token_num * hidden_dim * 2
        if not use_fp32_lamport
        else tp_size * max_token_num * hidden_dim * 4
    )

    ipc_handles, workspace_tensor = _create_ipc_workspace(
        tp_rank,
        tp_size,
        buffer_size,
        flag_size,
        lamport_comm_size,
        use_fp32_lamport,
        group,
    )

    if create_metadata:
        metadata = {
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "max_token_num": max_token_num,
            "hidden_dim": hidden_dim,
            "use_fp32_lamport": use_fp32_lamport,
            "buffer_size": buffer_size,
            "flag_size": flag_size,
            "lamport_comm_size": min(lamport_comm_size, MAX_COMM_SIZE),
        }
        return ipc_handles, workspace_tensor, metadata
    return ipc_handles, workspace_tensor


def trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
    workspace: List[List[int]], group: Optional[ProcessGroup] = None
) -> None:
    _destroy_ipc_workspace(workspace, group)


def trtllm_allreduce_fusion(
    allreduce_in: torch.Tensor,
    world_size: int,
    world_rank: int,
    token_num: int,
    hidden_dim: int,
    workspace_ptrs: torch.Tensor,
    launch_with_pdl: bool,
    trigger_completion_at_end: bool,
    fp32_acc: bool,
    pattern_code: int,
    use_oneshot: Optional[bool] = None,
    allreduce_out: Optional[torch.Tensor] = None,
    residual_in: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    norm_out: Optional[torch.Tensor] = None,
    partial_norm_out: Optional[torch.Tensor] = None,
    quant_out: Optional[torch.Tensor] = None,
    scale_out: Optional[torch.Tensor] = None,
    rms_gamma: Optional[torch.Tensor] = None,
    rms_eps: Optional[float] = None,
    scale_factor: Optional[Union[torch.Tensor, float]] = None,
    layout_code: Optional[int] = None,
    metadata: Optional[dict] = None,
    residual_reduce_scattered: bool = False,
    max_sm_to_use: Optional[int] = None,
    attnres_m: Optional[torch.Tensor] = None,
    attnres_s: Optional[torch.Tensor] = None,
    attnres_acc: Optional[torch.Tensor] = None,
    attnres_res_w: Optional[torch.Tensor] = None,
    attnres_out_norm_w: Optional[torch.Tensor] = None,
    latent_width: Optional[int] = None,
) -> None:
    if use_oneshot is None:
        if isinstance(workspace_ptrs, MnnvlAllReduceFusionWorkspace):
            use_oneshot = workspace_ptrs.resolve_use_oneshot(token_num, None)
        else:
            use_oneshot = _ar_should_use_oneshot(
                token_num, hidden_dim, allreduce_in.dtype, world_size
            )

    if isinstance(workspace_ptrs, MnnvlAllReduceFusionWorkspace):
        use_oneshot = workspace_ptrs.resolve_use_oneshot(token_num, use_oneshot)
        # MNNVL-structured one-shot path: single NVLS multicast payload store,
        # local-buffer Lamport polling, same FusedOp epilogues.
        # RuntimeError, not assert: callers such as PrefillGraph catch
        # RuntimeError to degrade to eager prefill, while an AssertionError
        # escapes and kills every scheduler at startup. An assert would also
        # vanish under `python -O`, letting an unsupported call reach the
        # kernel.
        if not workspace_ptrs.supports(
            token_num,
            hidden_dim,
            allreduce_in.dtype,
            world_size,
            pattern_code,
            use_oneshot=use_oneshot,
            residual_reduce_scattered=residual_reduce_scattered,
        ):
            raise RuntimeError(
                "mnnvl workspace does not support this call "
                f"(token_num={token_num}, hidden_dim={hidden_dim}, "
                f"dtype={allreduce_in.dtype}, pattern={pattern_code}, "
                f"use_oneshot={use_oneshot})"
            )
        _load_trtllm_comm_module().trtllm_mnnvl_allreduce_fusion(
            allreduce_in,
            world_size,
            world_rank,
            token_num,
            hidden_dim,
            workspace_ptrs.multicast_ptr,
            workspace_ptrs.local_ptr,
            workspace_ptrs.peer_ptrs,
            workspace_ptrs.buffer_flags,
            launch_with_pdl,
            use_oneshot,
            trigger_completion_at_end,
            fp32_acc,
            pattern_code,
            allreduce_out,
            residual_in,
            residual_out,
            norm_out,
            rms_gamma,
            rms_eps,
            attnres_m,
            attnres_s,
            attnres_acc,
            attnres_res_w,
            attnres_out_norm_w,
            latent_width,
        )
        return

    if not use_oneshot:
        assert not residual_reduce_scattered, "Currently not supported!"
        assert token_num > world_size, "sequence length should be larger than tp_size"

    required_lamport_comm_size = (
        token_num * hidden_dim * 2 * world_size
        if allreduce_in.dtype != torch.float32
        else token_num * hidden_dim * 4 * world_size
    )
    if required_lamport_comm_size > MAX_COMM_SIZE and use_oneshot:
        logging.warning(
            f"required_lamport_comm_size {required_lamport_comm_size} > MAX_COMM_SIZE. Falling back to twoshot."
        )
        use_oneshot = False

    if scale_factor is not None:
        if isinstance(scale_factor, torch.Tensor):
            scale_factor = scale_factor.to(torch.float32)
        else:
            scale_factor = torch.tensor(
                [scale_factor], dtype=torch.float32, device=allreduce_in.device
            )

    _load_trtllm_comm_module().trtllm_allreduce_fusion(
        allreduce_in,
        world_size,
        world_rank,
        token_num,
        hidden_dim,
        workspace_ptrs,
        launch_with_pdl,
        use_oneshot,
        trigger_completion_at_end,
        fp32_acc,
        residual_reduce_scattered,
        pattern_code,
        allreduce_out,
        residual_in,
        residual_out,
        norm_out,
        partial_norm_out,
        quant_out,
        scale_out,
        rms_gamma,
        rms_eps,
        scale_factor,
        layout_code,
        max_sm_to_use,
        attnres_m,
        attnres_s,
        attnres_acc,
        attnres_res_w,
        attnres_out_norm_w,
        latent_width,
    )


# ---------------------------------------------------------------------------
# AllGather fusion
# ---------------------------------------------------------------------------

_ag_oneshot_heuristics: dict = {2: 256, 4: 128, 8: 64, 16: 32}


def _ag_should_use_oneshot(
    token_num: int, hidden_dim: int, dtype: torch.dtype, world_size: int
) -> bool:
    comm_size_mb = (
        token_num * hidden_dim * 2 * world_size * dtype.itemsize / 1024 / 1024
    )
    return comm_size_mb <= _ag_oneshot_heuristics.get(world_size, 0)


def trtllm_create_ipc_workspace_for_allgather_fusion(
    tp_rank: int,
    tp_size: int,
    max_token_num: int,
    hidden_dim,
    use_fp32_lamport: bool = False,
    group: Optional[ProcessGroup] = None,
    create_metadata: bool = False,
) -> Union[
    Tuple[List[List[int]], torch.Tensor],
    Tuple[List[List[int]], torch.Tensor, dict],
]:
    # AllGather: buffer_size is NOT multiplied by tp_size
    buffer_size = max_token_num * hidden_dim * 2
    flag_size = tp_size * BarrierFlagCount * 4
    lamport_comm_size = (
        max_token_num * hidden_dim * 2
        if not use_fp32_lamport
        else max_token_num * hidden_dim * 4
    )

    ipc_handles, workspace_tensor = _create_ipc_workspace(
        tp_rank,
        tp_size,
        buffer_size,
        flag_size,
        lamport_comm_size,
        use_fp32_lamport,
        group,
    )

    if create_metadata:
        metadata = {
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "max_token_num": max_token_num,
            "hidden_dim": hidden_dim,
            "use_fp32_lamport": use_fp32_lamport,
            "buffer_size": buffer_size,
            "flag_size": flag_size,
            "lamport_comm_size": min(lamport_comm_size, MAX_COMM_SIZE),
        }
        return ipc_handles, workspace_tensor, metadata
    return ipc_handles, workspace_tensor


def trtllm_destroy_ipc_workspace_for_allgather_fusion(
    workspace: List[List[int]], group: Optional[ProcessGroup] = None
) -> None:
    _destroy_ipc_workspace(workspace, group)


def trtllm_allgather_fusion(
    allgather_in: torch.Tensor,
    world_size: int,
    world_rank: int,
    hidden_dim: int,
    workspace_ptrs: torch.Tensor,
    launch_with_pdl: bool,
    trigger_completion_at_end: bool,
    num_token_current_rank: int,
    allgather_out: torch.Tensor,
    num_token_all_group: int,
    pattern_code: int = AllGatherFusionPattern.kAllGather,
    use_oneshot: Optional[bool] = None,
    fp32_acc: bool = False,
    x_norm_out: Optional[torch.Tensor] = None,
    y_norm_out: Optional[torch.Tensor] = None,
    quant_out: Optional[torch.Tensor] = None,
    scale_out: Optional[torch.Tensor] = None,
    x_rms_gamma: Optional[torch.Tensor] = None,
    y_rms_gamma: Optional[torch.Tensor] = None,
    x_rms_eps: Optional[float] = 1e-6,
    y_rms_eps: Optional[float] = 1e-6,
    q_lora_rank: int = 0,
    kv_lora_rank: int = 0,
    qk_rope_head_dim: int = 0,
) -> None:
    assert (
        q_lora_rank % 128 == 0
    ), f"q_lora_rank ({q_lora_rank}) must be divisible by block_size (128)"
    assert hidden_dim <= 2112, f"hidden_dim ({hidden_dim}) must be <= 2112"
    total_rank = q_lora_rank + kv_lora_rank + qk_rope_head_dim
    assert total_rank == hidden_dim, (
        f"q_lora_rank + kv_lora_rank + qk_rope_head_dim must equal hidden_dim, "
        f"got {total_rank} != {hidden_dim}"
    )

    if use_oneshot is None:
        use_oneshot = _ag_should_use_oneshot(
            num_token_all_group, hidden_dim, allgather_in.dtype, world_size
        )

    required_lamport_comm_size = (
        num_token_all_group * hidden_dim * 2
        if allgather_in.dtype != torch.float32
        else num_token_all_group * hidden_dim * 4
    )
    if required_lamport_comm_size > MAX_COMM_SIZE and use_oneshot:
        logging.warning(
            f"required_lamport_comm_size {required_lamport_comm_size} > MAX_COMM_SIZE. Falling back."
        )
        use_oneshot = False

    _load_trtllm_comm_module().trtllm_allgather_fusion(
        allgather_in,
        world_size,
        world_rank,
        hidden_dim,
        workspace_ptrs,
        launch_with_pdl,
        use_oneshot,
        trigger_completion_at_end,
        fp32_acc,
        pattern_code,
        num_token_current_rank,
        num_token_all_group,
        allgather_out,
        x_norm_out,
        y_norm_out,
        quant_out,
        scale_out,
        x_rms_gamma,
        y_rms_gamma,
        x_rms_eps,
        y_rms_eps,
        q_lora_rank,
        kv_lora_rank,
        qk_rope_head_dim,
    )


# ---------------------------------------------------------------------------
# ReduceScatter fusion
# ---------------------------------------------------------------------------

_rs_oneshot_heuristics: dict = {2: 256, 4: 128, 8: 64, 16: 32}


def _rs_should_use_oneshot(
    token_num: int, hidden_dim: int, dtype: torch.dtype, world_size: int
) -> bool:
    comm_size_mb = (
        token_num * hidden_dim * 2 * world_size * dtype.itemsize / 1024 / 1024
    )
    return comm_size_mb <= _rs_oneshot_heuristics.get(world_size, 0)


def trtllm_create_ipc_workspace_for_reduce_scatter_fusion(
    tp_rank: int,
    tp_size: int,
    max_token_num: int,
    hidden_dim,
    use_fp32_lamport: bool = False,
    group: Optional[ProcessGroup] = None,
    create_metadata: bool = False,
) -> Union[
    Tuple[List[List[int]], torch.Tensor],
    Tuple[List[List[int]], torch.Tensor, dict],
]:
    buffer_size = tp_size * max_token_num * hidden_dim * 2
    flag_size = tp_size * BarrierFlagCount * 4
    lamport_comm_size = (
        tp_size * max_token_num * hidden_dim * 2
        if not use_fp32_lamport
        else tp_size * max_token_num * hidden_dim * 4
    )

    ipc_handles, workspace_tensor = _create_ipc_workspace(
        tp_rank,
        tp_size,
        buffer_size,
        flag_size,
        lamport_comm_size,
        use_fp32_lamport,
        group,
    )

    if create_metadata:
        metadata = {
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "max_token_num": max_token_num,
            "hidden_dim": hidden_dim,
            "use_fp32_lamport": use_fp32_lamport,
            "buffer_size": buffer_size,
            "flag_size": flag_size,
            "lamport_comm_size": min(lamport_comm_size, MAX_COMM_SIZE),
        }
        return ipc_handles, workspace_tensor, metadata
    return ipc_handles, workspace_tensor


def trtllm_destroy_ipc_workspace_for_reduce_scatter_fusion(
    workspace: List[List[int]], group: Optional[ProcessGroup] = None
) -> None:
    _destroy_ipc_workspace(workspace, group)


def trtllm_reducescatter_fusion(
    reducescatter_in: torch.Tensor,
    world_size: int,
    world_rank: int,
    token_num: int,
    hidden_dim: int,
    workspace_ptrs: torch.Tensor,
    launch_with_pdl: bool,
    trigger_completion_at_end: bool,
    fp32_acc: bool,
    num_token_current_rank: int,
    pattern_code: int,
    use_oneshot: Optional[bool] = None,
    reducescatter_out: Optional[torch.Tensor] = None,
    add_in: Optional[torch.Tensor] = None,
    residual_in: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    norm_out: Optional[torch.Tensor] = None,
    quant_out: Optional[torch.Tensor] = None,
    scale_out: Optional[torch.Tensor] = None,
    rms_gamma: Optional[torch.Tensor] = None,
    rms_eps: Optional[float] = None,
    scale_factor: Optional[Union[torch.Tensor, float]] = None,
    layout_code: Optional[int] = None,
    metadata: Optional[dict] = None,
) -> None:
    if use_oneshot is None:
        use_oneshot = _rs_should_use_oneshot(
            token_num, hidden_dim, reducescatter_in.dtype, world_size
        )

    if not use_oneshot:
        assert token_num > world_size, "sequence length should be larger than tp_size"

    if pattern_code == ReduceScatterFusionPattern.kRSResidualRMSNormFP8BlockWiseQuant:
        assert use_oneshot, "FP8 blockwise quant requires oneshot!"

    required_lamport_comm_size = (
        token_num * hidden_dim * 2 * world_size
        if reducescatter_in.dtype != torch.float32
        else token_num * hidden_dim * 4 * world_size
    )
    if required_lamport_comm_size > MAX_COMM_SIZE and use_oneshot:
        logging.warning(
            f"required_lamport_comm_size {required_lamport_comm_size} > MAX_COMM_SIZE. Falling back."
        )
        use_oneshot = False

    if scale_factor is not None:
        if isinstance(scale_factor, torch.Tensor):
            scale_factor = scale_factor.to(torch.float32)
        else:
            scale_factor = torch.tensor(
                [scale_factor], dtype=torch.float32, device=reducescatter_in.device
            )

    _load_trtllm_comm_module().trtllm_reducescatter_fusion(
        reducescatter_in,
        world_size,
        world_rank,
        token_num,
        hidden_dim,
        workspace_ptrs,
        launch_with_pdl,
        use_oneshot,
        trigger_completion_at_end,
        fp32_acc,
        pattern_code,
        num_token_current_rank,
        reducescatter_out,
        add_in,
        residual_in,
        residual_out,
        norm_out,
        quant_out,
        scale_out,
        rms_gamma,
        rms_eps,
        scale_factor,
        layout_code,
    )


# ---------------------------------------------------------------------------
# MiniMax QK fused AR + RMSNorm
# ---------------------------------------------------------------------------


def _minimax_lamport_comm_size_bytes(tp_size: int, max_token_num: int) -> int:
    """Conservative upper bound (in bytes) of a single rotation of the MiniMax
    lamport comm buffer.

    QK-fused path (TokenPerBlock=4) writes `2*tot_groups*sizeof(float4) = 32*tot_groups`
    bytes per rank; the next-iter clear writes the same amount. Worst case:
    `32 * ceil(max_token/4) * NRanks` bytes = `8 * max_token * NRanks`, with
    2x headroom and rounded up to 2MB for the shared-memory allocator.
    """
    raw = max(8 * max_token_num * tp_size, 1 << 16)
    return _round_up(raw * 2, 1 << 21)


def trtllm_create_ipc_workspace_for_minimax(
    tp_rank: int,
    tp_size: int,
    max_token_num: int,
    group: Optional[ProcessGroup] = None,
    dtype_elem_size: int = 2,
) -> Tuple[List[List[int]], torch.Tensor]:
    """Create an IPC workspace dedicated to the MiniMax QK fused AR+RMSNorm kernel.

    Layout of the returned `workspace_tensor` (each slot is an int64 device-ptr):
      [0, 2*tp_size)       : unused placeholders (kept to match the indexing the
                             kernel uses: `workspace[2*NRanks + r]` for lamport)
      [2*tp_size, 3*tp_size): per-rank lamport buffer pointers
      [3*tp_size]          : pointer to a 20-byte int32 scratch with
                               [0]=counter, [2]=flag (rotation in 0/1/2)
      [3*tp_size + 1]      : pointer to a 16-byte int64 scratch with
                               [0]=clear_size, [1]=comm_size_bytes

    This layout is NOT interchangeable with the regular trtllm_allreduce_fusion
    workspace; MiniMax must have its own because the two kernels read/write
    different sizes and increment the rotation flag independently.
    """
    # `dtype_elem_size` is accepted for API continuity but the lamport buffer
    # always stores fp32 variance sums regardless of input dtype, so sizing
    # and init are dtype-independent.
    del dtype_elem_size
    lamport_comm_size = _minimax_lamport_comm_size_bytes(tp_size, max_token_num)
    if lamport_comm_size > MAX_COMM_SIZE:
        lamport_comm_size = MAX_COMM_SIZE
    lamport_buffer_size = lamport_comm_size * 3

    # 3 × per-rank lamport buffers. We use the IPC allocator so each rank sees
    # peer pointers.
    lamport_handles = create_shared_buffer(
        _round_up(lamport_buffer_size, 1 << 21), group
    )
    # Placeholder IPC allocation for the two unused slot groups. Using zero-sized
    # allocations is not portable, so we allocate small (2MB) dummy buffers that
    # the kernel never touches.
    dummy_a = create_shared_buffer(1 << 21, group)
    dummy_b = create_shared_buffer(1 << 21, group)

    # Lamport sentinel: ALWAYS fp32 -0 (0x80000000). The MiniMax kernel stores
    # per-token variance sums (fp32) in the lamport buffer regardless of the
    # input/gamma dtype, so we must init with the fp32 sentinel pattern.
    # Initialising with fp16 -0 (0x8000) would set the bytes to 0x80008000
    # repeating, which an fp32 read would see as non-negative-zero and
    # immediately consume as "already written", producing garbage.
    trtllm_lamport_initialize(
        lamport_handles[tp_rank],
        lamport_buffer_size // 4,
        torch.float32,
    )

    # Scratch #0: 5 × int32 at workspace[3*tp_size]
    flag_ptr = cudart.cudaMalloc(5 * 4)
    cudart.cudaMemset(flag_ptr, 0, 5 * 4)
    # Scratch #1: 2 × int64 at workspace[3*tp_size + 1]: {clear_size=0, comm_size}
    clear_scalar = cudart.cudaMalloc(2 * 8)
    cudart.cudaMemset(clear_scalar, 0, 2 * 8)
    comm_size_bytes = int(lamport_comm_size).to_bytes(8, byteorder="little")
    cudart.cudaMemcpy(
        c_void_p(clear_scalar.value + 8), cast(comm_size_bytes, c_void_p), 8
    )

    workspace: List[int] = []
    # Slots [0, 2*tp_size): dummies. The kernel indexes [2*tp_size + r] for lamport.
    for r in range(tp_size):
        workspace.append(dummy_a[r])
    for r in range(tp_size):
        workspace.append(dummy_b[r])
    for r in range(tp_size):
        workspace.append(lamport_handles[r])
    workspace.append(flag_ptr.value)
    workspace.append(clear_scalar.value)

    workspace_tensor = torch.tensor(
        workspace, dtype=torch.int64, device=torch.device("cuda")
    )

    if dist.is_initialized() and group is not None:
        dist.barrier(group=group)

    ipc_handles = [dummy_a, dummy_b, lamport_handles]
    return ipc_handles, workspace_tensor


def trtllm_destroy_ipc_workspace_for_minimax(
    ipc_handles: List[List[int]], group: Optional[ProcessGroup] = None
) -> None:
    for handle in ipc_handles:
        free_shared_buffer(handle, group)


def minimax_allreduce_rms(
    input: torch.Tensor,
    norm_weight: torch.Tensor,
    workspace_ptrs: torch.Tensor,
    rank: int,
    nranks: int,
    eps: float,
    trigger_completion_at_end: bool = True,
    launch_with_pdl: bool = False,
    rms_norm_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Single-matrix Lamport AR + RMSNorm over sharded hidden dim.

    `input` is [token_num, local_hidden_dim] (= global / nranks). `norm_weight`
    must be bf16 of shape [local_hidden_dim]. Reuses the same workspace layout
    as `trtllm_create_ipc_workspace_for_all_reduce_fusion`.
    """
    if rms_norm_out is None:
        rms_norm_out = torch.empty_like(input)
    _load_trtllm_comm_module().minimax_allreduce_rms(
        input,
        norm_weight,
        rms_norm_out,
        workspace_ptrs,
        rank,
        nranks,
        eps,
        trigger_completion_at_end,
        launch_with_pdl,
    )
    return rms_norm_out


def minimax_allreduce_rms_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    norm_weight_q: torch.Tensor,
    norm_weight_k: torch.Tensor,
    workspace_ptrs: torch.Tensor,
    rank: int,
    nranks: int,
    eps: float,
    trigger_completion_at_end: bool = True,
    launch_with_pdl: bool = False,
    rms_norm_out_q: Optional[torch.Tensor] = None,
    rms_norm_out_k: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused Q+K Lamport AR + RMSNorm. Requires global head_dim_q==6144 and
    global head_dim_k==1024 (i.e. MiniMax M2 attention)."""
    # Outputs must be tightly packed (kernel writes them at head_dim stride);
    # `q`/`k` may be strided slices, so don't preserve their layout via
    # empty_like default (preserve_format) — force contiguous.
    if rms_norm_out_q is None:
        rms_norm_out_q = torch.empty_like(q, memory_format=torch.contiguous_format)
    if rms_norm_out_k is None:
        rms_norm_out_k = torch.empty_like(k, memory_format=torch.contiguous_format)
    _load_trtllm_comm_module().minimax_allreduce_rms_qk(
        q,
        k,
        norm_weight_q,
        norm_weight_k,
        rms_norm_out_q,
        rms_norm_out_k,
        workspace_ptrs,
        rank,
        nranks,
        eps,
        trigger_completion_at_end,
        launch_with_pdl,
    )
    return rms_norm_out_q, rms_norm_out_k
