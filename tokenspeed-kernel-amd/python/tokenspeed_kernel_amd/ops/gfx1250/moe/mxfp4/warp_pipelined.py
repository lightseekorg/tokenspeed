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

"""Launchers for the gfx1250 warp-pipelined FP8 x MXFP4 MoE kernels.

``gluon_mxfp_precomputed_mxfp4_warp_pipelined_moe`` runs a whole MoE layer
with precomputed top-k on the same weights, routing and FP8 activation
numerics as ``gluon_mxfp_precomputed_mxfp4_fused_moe(activation_dtype="fp8")``:
route, quantize the input to E4M3, the gate/up GEMM with the fused activation
and FP8 output, the down GEMM scattered to the route rows, and the weighted
top-k reduction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from tokenspeed_kernel_amd._triton import gl, triton
from tokenspeed_kernel_amd.ops.gfx1250.moe._common import RaggedTensorMetadata
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4._common import (
    ragged_metadata_fields,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4._warp_pipeline_common import (
    num_compute_units,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.decode_warp_pipelined import (
    gluon_moe_decode_warp_pipelined_gfx1250,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.down_warp_pipelined import (
    gluon_moe_down_warp_pipelined_gfx1250,
    gluon_moe_down_warp_pipelined_persistent_gfx1250,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.fused import (
    _precomputed_topk_route,
    _quantize_fp8_activation,
    _weighted_topk_reduce_gfx1250,
    should_upcast_indices,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.gate_up_warp_pipelined import (
    gluon_moe_gate_up_warp_pipelined_gfx1250,
)

_PREFILL_BLOCK_N = 512
_PREFILL_BLOCK_K = 256
_UINT16_MAX = (1 << 16) - 1


@dataclass(frozen=True)
class PrefillTiles:
    """Launch configuration of a prefill (warp-pipelined) projection.

    Attributes:
        block_m: Rows per workgroup; a cluster owns ``block_m * cta_m`` rows of
            one expert.
        cta_m: Workgroups per cluster sharing each W tile by multicast.
        num_buffers: LDS ring depth: 3 or 4, or 2 for the down GEMM; K must
            span at least this many 256-wide K tiles.
        resolve_partition_conflicts: Split X and W over the two LDS partitions.
        xcd_swizzle: Give each XCD a contiguous run of tiles.
        group_m: M tiles per swizzle group.
        waves_per_eu: Occupancy hint, 0 for none.
        persistent: Down projection only: keep the ring across tiles.
        programs_per_cu: Persistent programs (clusters count each workgroup).
        w_cache_modifier: TDM cache modifier of the W and W-scale loads.
    """

    block_m: int
    cta_m: int
    num_buffers: int
    resolve_partition_conflicts: bool
    xcd_swizzle: int
    group_m: int
    waves_per_eu: int
    persistent: bool
    programs_per_cu: int
    w_cache_modifier: str


@dataclass(frozen=True)
class DecodeTiles:
    """Launch configuration of the persistent decode kernel.

    Attributes:
        block_k: K tile; split into ``num_k_subtiles`` WMMA steps.
        num_k_subtiles: K subtiles per K tile (each at least 128 wide).
        num_buffers: LDS ring depth; K must span at least this many K tiles.
        warp_pipeline: Run warps 4-7 one stage behind warps 0-3.
        programs_per_cu: Persistent programs per compute unit.
        async_store: Down projection only: stage each output tile and copy it
            out asynchronously.
        w_partition: Spread W over the LDS partitions (0, 1 or 2).
        x_partition: Spread X over the LDS partitions (0, 1 or 2; down only).
        w_cache_modifier: TDM cache modifier of the W and W-scale loads.
    """

    block_k: int
    num_k_subtiles: int
    num_buffers: int
    warp_pipeline: bool
    programs_per_cu: int
    async_store: bool
    w_partition: int
    x_partition: int
    w_cache_modifier: str


# 128-row clusters of one workgroup, three-slot rings and one program per
# CU: the ring of a 128 x 512 x 256 tile fills the WGP's LDS. Splitting the
# gate/up X and W over the two LDS partitions measured 3% slower to 1% faster on
# the benchmarked shapes, so only the down GEMM splits them.
PREFILL_GATE_UP_TILES = PrefillTiles(
    block_m=128,
    cta_m=1,
    num_buffers=3,
    resolve_partition_conflicts=False,
    xcd_swizzle=1,
    group_m=1,
    waves_per_eu=2,
    persistent=True,
    programs_per_cu=1,
    w_cache_modifier="",
)
PREFILL_DOWN_TILES = replace(
    PREFILL_GATE_UP_TILES, resolve_partition_conflicts=True, persistent=False
)
# Two 16-row programs per CU stream each touched expert's W once, so the W
# loads bypass the cache (".cs"). Staging the down output for asynchronous
# stores measured no faster than storing from registers.
DECODE_TILES = DecodeTiles(
    block_k=512,
    num_k_subtiles=2,
    num_buffers=3,
    warp_pipeline=False,
    programs_per_cu=2,
    async_store=False,
    w_partition=0,
    x_partition=0,
    w_cache_modifier=".cs",
)


# Token ranges where a warp-pipelined GEMM ran more than 10% faster than the
# A8W4 GEMM it replaces in ``gluon_mxfp_precomputed_mxfp4_fused_moe``: median
# of three graph-replayed runs on MI450 at 1, 2, 4, ..., 16384 tokens, with
# decode kernels up to 16 routes per expert. A range spans at least two
# adjacent measured points that all cleared 10%, so a lone point that did is
# taken as noise. Shapes and token counts not listed keep the A8W4 kernels.
# Key: (experts, hidden, intermediate per partition, top-k, SiTU activation).
_MEASURED_GAINS: dict[tuple[int, int, int, int, bool], dict[str, tuple]] = {
    # GPT-OSS-120B
    (128, 2880, 2880, 4, False): {
        "gate_up": ((1, 8), (2048, 16384)),
        "down": ((1024, 16384),),
    },
    # Kimi K3, TP8
    (896, 3584, 384, 16, True): {"gate_up": ((8, 32),), "down": ()},
    # GLM-5.3-Flash expert shape
    (288, 4096, 512, 8, False): {
        "gate_up": ((32, 64), (4096, 16384)),
        "down": ((4096, 16384),),
    },
    # DeepSeek V4.1-Flash expert shape, intermediate padded to 2560
    (384, 5120, 2560, 6, False): {
        "gate_up": ((1, 4), (4096, 16384)),
        "down": ((8, 16384),),
    },
}


def select_warp_pipelined_gemms(
    num_experts: int,
    hidden: int,
    intermediate: int,
    top_k: int,
    situ: bool,
    num_tokens: int,
) -> tuple[bool, bool]:
    """Whether the gate/up and the down GEMM of a layer measured more than 10%
    faster on the warp-pipelined kernels than on the A8W4 ones.

    Args:
        num_experts: Local experts.
        hidden: Hidden size, the gate/up K and the down N.
        intermediate: Intermediate size per partition, the down K.
        top_k: Routes per token.
        situ: The gate activation is SiTU rather than SwiGLU/SiLU.
        num_tokens: Tokens in the batch.

    Returns:
        ``(gate_up, down)``.
    """
    gains = _MEASURED_GAINS.get((num_experts, hidden, intermediate, top_k, situ))
    if gains is None:
        return False, False
    return tuple(
        any(lo <= num_tokens <= hi for lo, hi in gains[gemm])
        for gemm in ("gate_up", "down")
    )


def select_decode_tiles(k: int) -> DecodeTiles:
    """Decode tiles measured on the same shapes at 8 and 128 tokens.

    512-wide K tiles win wherever K spans three of them. Shorter reductions
    (down K of 384 or 512) take 128-wide K tiles, which leave LDS for four
    programs per CU to hide each one's latency.
    """
    if triton.cdiv(k, DECODE_TILES.block_k) >= DECODE_TILES.num_buffers:
        return DECODE_TILES
    return replace(DECODE_TILES, block_k=128, num_k_subtiles=1, programs_per_cu=4)


def select_prefill_tiles(
    gate_up: bool, routed_rows: int, num_experts: int, k: int
) -> PrefillTiles:
    """Prefill tiles measured on MI450 for GPT-OSS-120B, Kimi K3, GLM-5.3-Flash
    and DeepSeek V4.1-Flash expert shapes, 1024 to 8192 tokens.

    Giving each XCD a contiguous run of tiles lets the M tiles of one expert
    share W in L2 (5% to 10%). 64-row tiles win while experts get fewer than
    48 rows on average, and for gate/up when 128-row tiles would leave more
    than 96 of each expert's last tile empty. Gate/up multicasts W across a
    two-workgroup cluster when experts get 64 to 128 rows; elsewhere clusters
    gained at most 2%.
    """
    rows_per_expert = routed_rows / num_experts
    padding_128 = -rows_per_expert % 128
    small = rows_per_expert < 48 or (gate_up and padding_128 > 96)
    base = PREFILL_GATE_UP_TILES if gate_up else PREFILL_DOWN_TILES
    tiles = replace(base, block_m=64 if small else 128, xcd_swizzle=8)
    if gate_up and 64 < rows_per_expert < 128:
        # One 128-row cluster tile per expert, its W loaded once for both
        # workgroups (Kimi K3 at 4096 tokens: 23% faster).
        tiles = replace(tiles, block_m=64, cta_m=2)
    if not gate_up and triton.cdiv(k, _PREFILL_BLOCK_K) < tiles.num_buffers:
        tiles = replace(tiles, num_buffers=2)
    return tiles


@dataclass(frozen=True)
class _Activation:
    do_swiglu: bool
    swiglu_alpha: float
    swiglu_limit: float
    swiglu_beta: float
    do_situ: bool
    situ_beta: float
    situ_linear_beta: float


def make_activation(
    activation: str,
    *,
    swiglu_alpha: float,
    swiglu_limit: float,
    swiglu_beta: float,
    situ_beta: float,
    situ_linear_beta: float,
) -> _Activation:
    """Fused gate activation of the gate/up GEMM.

    ``silu`` is SwiGLU with ``alpha = 1``, no clamp and ``beta = 0``.
    """
    if activation == "situ":
        if situ_beta <= 0.0 or situ_linear_beta <= 0.0:
            raise ValueError("SiTU beta and linear_beta must be positive")
        return _Activation(False, 0.0, 0.0, 0.0, True, situ_beta, situ_linear_beta)
    if activation == "silu":
        return _Activation(True, 1.0, 0.0, 0.0, False, 0.0, 0.0)
    if activation == "swiglu":
        return _Activation(
            True, swiglu_alpha, swiglu_limit, swiglu_beta, False, 0.0, 0.0
        )
    raise ValueError(
        f"activation must be 'silu', 'swiglu' or 'situ', got {activation!r}"
    )


def _tdm_index_type(max_index: int) -> gl.dtype:
    # TDM zero-extends 16-bit indices, so gl.int16 covers the unsigned range.
    return gl.int16 if max_index <= _UINT16_MAX else gl.int32


def _check_weight(w: torch.Tensor, w_mx_scale: torch.Tensor, k: int) -> int:
    """Validate a ``(E, K // 2, N)`` K-contiguous MXFP4 weight; return N."""
    if w.dtype != torch.uint8 or w.ndim != 3 or w.stride(-2) != 1:
        raise ValueError(
            "W must be packed MXFP4 uint8 (E, K // 2, N) with contiguous K, got "
            f"{w.dtype} {tuple(w.shape)} strides {w.stride()}"
        )
    if int(w.shape[1]) * 2 != k:
        raise ValueError(f"K mismatch: activation K={k}, weight K={w.shape[1] * 2}")
    if w_mx_scale.dtype != torch.uint8 or w_mx_scale.ndim != 3:
        raise ValueError("W scales must be the gfx1250 preshuffled uint8 layout")
    return int(w.shape[2])


def _ragged_args(ragged_metadata: RaggedTensorMetadata, block_m: int):
    slice_sizes, slice_offs, block_offs, block_schedule, _, divisibility = (
        ragged_metadata_fields(ragged_metadata, block_m)
    )
    return (slice_sizes, slice_offs, block_offs, block_schedule), divisibility


def _check_prefill(tiles: PrefillTiles, k: int, min_buffers: int) -> int:
    cluster_m = tiles.block_m * tiles.cta_m
    if cluster_m not in RaggedTensorMetadata.block_sizes():
        raise ValueError(
            f"block_m * cta_m must be a routing block size, got {cluster_m}"
        )
    if not min_buffers <= tiles.num_buffers <= 4:
        raise ValueError(
            f"num_buffers must be {min_buffers} to 4, got {tiles.num_buffers}"
        )
    if triton.cdiv(k, _PREFILL_BLOCK_K) < tiles.num_buffers:
        raise ValueError(
            f"K={k} spans fewer than num_buffers={tiles.num_buffers} "
            f"{_PREFILL_BLOCK_K}-wide K tiles"
        )
    return cluster_m


def _persistent_grid(device, programs_per_cu: int, cta_m: int, max_tiles: int):
    # The launch grid counts clusters of cta_m workgroups.
    clusters = num_compute_units(device) * programs_per_cu // cta_m
    return max(1, min(clusters, max_tiles))


def launch_gluon_moe_gate_up_warp_pipelined_gfx1250(
    x: torch.Tensor,
    w: torch.Tensor,
    w_mx_scale: torch.Tensor,
    bias: torch.Tensor | None,
    ragged_metadata: RaggedTensorMetadata,
    gather_indx: torch.Tensor,
    *,
    x_global_scale: torch.Tensor | None,
    y_global_scale: torch.Tensor | None,
    activation: _Activation,
    tiles: PrefillTiles,
) -> torch.Tensor:
    """Gate/up GEMM of expert-routed rows with the fused gate activation.

    Args:
        x: ``(tokens, K)`` FP8 E4M3 activations.
        w: ``(E, K // 2, N)`` packed MXFP4 gate/up weights (gate and up rows
            interleaved), K contiguous.
        w_mx_scale: gfx1250-preshuffled UE8M0 scales of ``w``.
        bias: Optional ``(E, N)`` FP32 bias.
        ragged_metadata: Expert slices of the routed rows.
        gather_indx: ``(rows,)`` token of each expert-sorted routed row.
        x_global_scale: Optional one-element FP32 scale ``x`` is divided by.
        y_global_scale: Optional one-element FP32 scale the output is divided
            by before its E4M3 cast.
        activation: The fused SwiGLU or SiTU.
        tiles: Launch configuration.

    Returns:
        ``(rows, N // 2)`` E4M3 activation output in expert-sorted row order.
    """
    if x.dtype != torch.float8_e4m3fn or x.ndim != 2 or x.stride(1) != 1:
        raise ValueError("x must be a K-contiguous (tokens, K) E4M3 tensor")
    m, k = map(int, x.shape)
    n = _check_weight(w, w_mx_scale, k)
    if n % 2:
        raise ValueError(f"gate/up N must be even, got {n}")
    cluster_m = _check_prefill(tiles, k, min_buffers=3)
    rows = int(gather_indx.shape[0])
    y = torch.empty((rows, n // 2), device=x.device, dtype=torch.float8_e4m3fn)
    if rows == 0:
        return y
    (slice_sizes, slice_offs, block_offs, block_schedule), divisibility = _ragged_args(
        ragged_metadata, cluster_m
    )
    grid_n = triton.cdiv(n, _PREFILL_BLOCK_N)
    max_tiles = (
        RaggedTensorMetadata.n_blocks(ragged_metadata.n_slices, rows, cluster_m)
        * grid_n
    )
    # Gather indices point one row past X for inactive lanes. Past 16 bits
    # the kernel indexes relative to each tile's lowest row, which mostly fits
    # 16 bits again.
    index_type = _tdm_index_type(m + 1)
    gluon_moe_gate_up_warp_pipelined_gfx1250[
        (_persistent_grid(x.device, tiles.programs_per_cu, tiles.cta_m, max_tiles),)
    ](
        y,
        y.stride(0),
        y.stride(1),
        y_global_scale,
        x,
        x.stride(0),
        x.stride(1),
        x_global_scale,
        w,
        *w.stride(),
        w_mx_scale,
        *w_mx_scale.stride(),
        bias,
        None if bias is None else bias.stride(0),
        m,
        n,
        k,
        rows,
        gather_indx,
        slice_sizes,
        slice_offs,
        block_offs,
        block_schedule,
        divisibility,
        grid_n,
        activation.do_swiglu,
        activation.swiglu_alpha,
        activation.swiglu_limit,
        activation.swiglu_beta,
        activation.do_situ,
        activation.situ_beta,
        activation.situ_linear_beta,
        2,
        ragged_metadata.n_slices,
        cluster_m,
        _PREFILL_BLOCK_N,
        _PREFILL_BLOCK_K,
        tiles.group_m,
        tiles.xcd_swizzle,
        tiles.num_buffers,
        tiles.w_cache_modifier,
        index_type,
        should_upcast_indices(x, w, y),
        tiles.resolve_partition_conflicts,
        index_type == gl.int32,
        num_warps=8,
        num_ctas=tiles.cta_m,
        waves_per_eu=tiles.waves_per_eu,
        # Triton defaults to the co-execution scheduler only up to four warps;
        # the eight-warp ping-pong otherwise gets max-occupancy scheduling,
        # which spills VGPRs in the epilogue.
        llvm_fn_attrs="amdgpu-sched-strategy=coexec",
    )
    return y


def launch_gluon_moe_down_warp_pipelined_gfx1250(
    x: torch.Tensor,
    w: torch.Tensor,
    w_mx_scale: torch.Tensor,
    bias: torch.Tensor | None,
    ragged_metadata: RaggedTensorMetadata,
    scatter_indx: torch.Tensor | None,
    *,
    x_global_scale: torch.Tensor | None,
    out_dtype: torch.dtype,
    tiles: PrefillTiles,
) -> torch.Tensor:
    """Down GEMM of expert-sorted rows, scattered to the route rows.

    Args:
        x: ``(rows, K)`` E4M3 gate/up output in expert-sorted row order.
        w: ``(E, K // 2, N)`` packed MXFP4 down weights, K contiguous.
        w_mx_scale: gfx1250-preshuffled UE8M0 scales of ``w``.
        bias: Optional ``(E, N)`` FP32 bias.
        ragged_metadata: Expert slices of the rows.
        scatter_indx: Optional ``(rows,)`` output row of each input row; rows
            map one to one without it.
        x_global_scale: Optional one-element FP32 scale ``x`` is divided by.
        out_dtype: Output dtype.
        tiles: Launch configuration.

    Returns:
        ``(rows, N)`` output.
    """
    if x.dtype != torch.float8_e4m3fn or x.ndim != 2 or x.stride(1) != 1:
        raise ValueError("x must be a K-contiguous (rows, K) E4M3 tensor")
    rows, k = map(int, x.shape)
    n = _check_weight(w, w_mx_scale, k)
    cluster_m = _check_prefill(tiles, k, min_buffers=2)
    writeback_size = rows if scatter_indx is None else int(scatter_indx.shape[0])
    y = torch.empty((writeback_size, n), device=x.device, dtype=out_dtype)
    if rows == 0:
        return y
    (slice_sizes, slice_offs, block_offs, block_schedule), divisibility = _ragged_args(
        ragged_metadata, cluster_m
    )
    grid_n = triton.cdiv(n, _PREFILL_BLOCK_N)
    max_tiles = (
        RaggedTensorMetadata.n_blocks(ragged_metadata.n_slices, rows, cluster_m)
        * grid_n
    )
    if tiles.persistent:
        kernel = gluon_moe_down_warp_pipelined_persistent_gfx1250
        grid = _persistent_grid(x.device, tiles.programs_per_cu, tiles.cta_m, max_tiles)
    else:
        kernel = gluon_moe_down_warp_pipelined_gfx1250
        grid = max_tiles
    kernel[(grid,)](
        y,
        y.stride(0),
        y.stride(1),
        x,
        x.stride(0),
        x.stride(1),
        x_global_scale,
        w,
        *w.stride(),
        w_mx_scale,
        *w_mx_scale.stride(),
        bias,
        None if bias is None else bias.stride(0),
        n,
        k,
        rows,
        scatter_indx,
        writeback_size,
        slice_sizes,
        slice_offs,
        block_offs,
        block_schedule,
        divisibility,
        grid_n,
        1,
        ragged_metadata.n_slices,
        cluster_m,
        _PREFILL_BLOCK_N,
        _PREFILL_BLOCK_K,
        tiles.group_m,
        tiles.xcd_swizzle,
        tiles.num_buffers,
        tiles.w_cache_modifier,
        # Rows without a destination point at writeback_size.
        _tdm_index_type(writeback_size),
        should_upcast_indices(x, w, y),
        tiles.resolve_partition_conflicts,
        num_warps=8,
        num_ctas=tiles.cta_m,
        waves_per_eu=tiles.waves_per_eu,
        llvm_fn_attrs="amdgpu-sched-strategy=coexec",
    )
    return y


def launch_gluon_moe_decode_warp_pipelined_gfx1250(
    x: torch.Tensor,
    w: torch.Tensor,
    w_mx_scale: torch.Tensor,
    bias: torch.Tensor | None,
    ragged_metadata: RaggedTensorMetadata,
    gather_indx: torch.Tensor | None,
    scatter_indx: torch.Tensor | None,
    *,
    x_global_scale: torch.Tensor | None,
    y_global_scale: torch.Tensor | None,
    activation: _Activation | None,
    out_dtype: torch.dtype,
    tiles: DecodeTiles,
) -> torch.Tensor:
    """Small-M gate/up (``gather_indx``) or down (``scatter_indx``) GEMM.

    Args:
        x: E4M3 activations, ``(tokens, K)`` for gate/up and ``(rows, K)``
            expert-sorted for down.
        w: ``(E, K // 2, N)`` packed MXFP4 weights, K contiguous.
        w_mx_scale: gfx1250-preshuffled UE8M0 scales of ``w``.
        bias: Optional ``(E, N)`` FP32 bias.
        ragged_metadata: Expert slices of the routed rows.
        gather_indx: Gate/up only: token of each expert-sorted routed row.
        scatter_indx: Down only: output row of each expert-sorted row.
        x_global_scale: Optional one-element FP32 scale ``x`` is divided by.
        y_global_scale: Optional one-element FP32 scale the output is divided
            by before its cast.
        activation: Gate/up only: the fused SwiGLU or SiTU.
        out_dtype: Output dtype.
        tiles: Launch configuration.

    Returns:
        ``(rows, N // 2)`` gate/up output in expert-sorted order, or the
        ``(len(scatter_indx), N)`` down output.
    """
    if (gather_indx is None) == (scatter_indx is None):
        raise ValueError("pass exactly one of gather_indx (gate/up) and scatter_indx")
    if (activation is None) != (gather_indx is None):
        raise ValueError("the gate/up GEMM, and only it, takes an activation")
    if x.dtype != torch.float8_e4m3fn or x.ndim != 2 or x.stride(1) != 1:
        raise ValueError("x must be a K-contiguous 2-D E4M3 tensor")
    m, k = map(int, x.shape)
    n = _check_weight(w, w_mx_scale, k)
    block_m, block_n = 16, 128
    sub_k = tiles.block_k // tiles.num_k_subtiles
    if sub_k < 128 or tiles.block_k % tiles.num_k_subtiles:
        raise ValueError("decode K subtiles must be at least 128 wide")
    if triton.cdiv(k, tiles.block_k) < tiles.num_buffers:
        raise ValueError(
            f"K={k} spans fewer than num_buffers={tiles.num_buffers} "
            f"{tiles.block_k}-wide K tiles"
        )
    gate_up = gather_indx is not None
    if tiles.async_store and (gate_up or n % block_n):
        raise ValueError("async_store needs the down GEMM and whole N tiles")
    reduction_n = 2 if gate_up else 1
    rows = int(gather_indx.shape[0]) if gate_up else m
    writeback_size = rows if gate_up else int(scatter_indx.shape[0])
    y = torch.empty(
        (writeback_size, n // reduction_n), device=x.device, dtype=out_dtype
    )
    if rows == 0:
        return y
    (slice_sizes, slice_offs, block_offs, block_schedule), divisibility = _ragged_args(
        ragged_metadata, block_m
    )
    grid_n = triton.cdiv(n, block_n)
    max_tiles = (
        RaggedTensorMetadata.n_blocks(ragged_metadata.n_slices, rows, block_m) * grid_n
    )
    act = activation or _Activation(False, 0.0, 0.0, 0.0, False, 0.0, 0.0)
    gluon_moe_decode_warp_pipelined_gfx1250[
        (_persistent_grid(x.device, tiles.programs_per_cu, 1, max_tiles),)
    ](
        y,
        y.stride(0),
        y.stride(1),
        y_global_scale,
        x,
        x.stride(0),
        x.stride(1),
        x_global_scale,
        w,
        *w.stride(),
        w_mx_scale,
        *w_mx_scale.stride(),
        bias,
        None if bias is None else bias.stride(0),
        m,
        n,
        k,
        rows,
        gather_indx,
        scatter_indx,
        writeback_size,
        slice_sizes,
        slice_offs,
        block_offs,
        block_schedule,
        divisibility,
        grid_n,
        act.do_swiglu,
        act.swiglu_alpha,
        act.swiglu_limit,
        act.swiglu_beta,
        act.do_situ,
        act.situ_beta,
        act.situ_linear_beta,
        reduction_n,
        ragged_metadata.n_slices,
        block_m,
        block_n,
        tiles.block_k,
        1,
        1,
        tiles.num_buffers,
        tiles.num_k_subtiles,
        tiles.warp_pipeline,
        None,
        tiles.w_partition,
        tiles.x_partition,
        tiles.async_store,
        tiles.w_cache_modifier,
        _tdm_index_type(m + 1 if gate_up else writeback_size),
        should_upcast_indices(x, w, y),
        num_warps=8,
    )
    return y


def launch_warp_pipelined_gate_up(
    x: torch.Tensor,
    w: torch.Tensor,
    w_mx_scale: torch.Tensor,
    bias: torch.Tensor | None,
    ragged_metadata: RaggedTensorMetadata,
    gather_indx: torch.Tensor,
    *,
    x_global_scale: torch.Tensor | None,
    y_global_scale: torch.Tensor | None,
    activation: _Activation,
    decode: bool,
) -> torch.Tensor:
    """Gate/up GEMM on the decode or prefill kernel with the selected tiles;
    arguments as ``launch_gluon_moe_gate_up_warp_pipelined_gfx1250``."""
    if decode:
        return launch_gluon_moe_decode_warp_pipelined_gfx1250(
            x,
            w,
            w_mx_scale,
            bias,
            ragged_metadata,
            gather_indx,
            None,
            x_global_scale=x_global_scale,
            y_global_scale=y_global_scale,
            activation=activation,
            out_dtype=torch.float8_e4m3fn,
            tiles=select_decode_tiles(int(x.shape[-1])),
        )
    return launch_gluon_moe_gate_up_warp_pipelined_gfx1250(
        x,
        w,
        w_mx_scale,
        bias,
        ragged_metadata,
        gather_indx,
        x_global_scale=x_global_scale,
        y_global_scale=y_global_scale,
        activation=activation,
        tiles=select_prefill_tiles(
            True, int(gather_indx.shape[0]), int(w.shape[0]), int(x.shape[-1])
        ),
    )


def launch_warp_pipelined_down(
    x: torch.Tensor,
    w: torch.Tensor,
    w_mx_scale: torch.Tensor,
    bias: torch.Tensor | None,
    ragged_metadata: RaggedTensorMetadata,
    scatter_indx: torch.Tensor,
    *,
    x_global_scale: torch.Tensor | None,
    out_dtype: torch.dtype,
    decode: bool,
) -> torch.Tensor:
    """Down GEMM on the decode or prefill kernel with the selected tiles;
    arguments as ``launch_gluon_moe_down_warp_pipelined_gfx1250``."""
    if decode:
        return launch_gluon_moe_decode_warp_pipelined_gfx1250(
            x,
            w,
            w_mx_scale,
            bias,
            ragged_metadata,
            None,
            scatter_indx,
            x_global_scale=x_global_scale,
            y_global_scale=None,
            activation=None,
            out_dtype=out_dtype,
            tiles=select_decode_tiles(int(x.shape[-1])),
        )
    return launch_gluon_moe_down_warp_pipelined_gfx1250(
        x,
        w,
        w_mx_scale,
        bias,
        ragged_metadata,
        scatter_indx,
        x_global_scale=x_global_scale,
        out_dtype=out_dtype,
        tiles=select_prefill_tiles(
            False, int(x.shape[0]), int(w.shape[0]), int(x.shape[-1])
        ),
    )


def gluon_mxfp_precomputed_mxfp4_warp_pipelined_moe(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    *,
    w13_mx_scale: torch.Tensor,
    w2_mx_scale: torch.Tensor,
    w13_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    activation: _Activation,
    out_dtype: torch.dtype,
    decode: bool,
    gate_up_tiles: PrefillTiles | DecodeTiles | None = None,
    down_tiles: PrefillTiles | DecodeTiles | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """FP8-activation MXFP4 MoE layer with precomputed top-k.

    Args:
        hidden_states: ``(tokens, hidden)`` BF16/FP16 (or E4M3) activations.
        topk_weights: ``(tokens, top_k)`` route weights.
        topk_ids: ``(tokens, top_k)`` expert ids.
        w13_weight: gfx1250-preprocessed interleaved gate/up weights with an
            ``act_scale`` attribute (one-element FP32 input scale or None).
        w2_weight: gfx1250-preprocessed down weights with ``act_scale``.
        w13_mx_scale: gfx1250-preshuffled scales of ``w13_weight``.
        w2_mx_scale: gfx1250-preshuffled scales of ``w2_weight``.
        w13_bias: Optional ``(E, 2 * intermediate)`` FP32 gate/up bias.
        w2_bias: Optional ``(E, hidden)`` FP32 down bias.
        activation: The fused gate activation (see ``make_activation``).
        out_dtype: Output dtype.
        decode: Run both projections on the small-M decode kernel, with
            ``DecodeTiles``; otherwise on the prefill kernels with
            ``PrefillTiles``.
        gate_up_tiles: Gate/up launch configuration; by default
            ``select_decode_tiles`` or ``select_prefill_tiles``.
        down_tiles: Down launch configuration, defaulted the same way.
        out: Optional ``(tokens, hidden)`` output buffer.

    Returns:
        ``(tokens, hidden)`` output.
    """
    num_experts = int(w13_weight.shape[0])
    hidden = int(hidden_states.shape[-1])
    intermediate = int(w13_weight.shape[-1]) // 2
    if decode:
        gate_up_tiles = gate_up_tiles or select_decode_tiles(hidden)
        down_tiles = down_tiles or select_decode_tiles(intermediate)
    else:
        rows = topk_ids.numel()
        gate_up_tiles = gate_up_tiles or select_prefill_tiles(
            True, rows, num_experts, hidden
        )
        down_tiles = down_tiles or select_prefill_tiles(
            False, rows, num_experts, intermediate
        )
    tiles_type = DecodeTiles if decode else PrefillTiles
    if not isinstance(gate_up_tiles, tiles_type) or not isinstance(
        down_tiles, tiles_type
    ):
        raise TypeError(f"decode={decode} takes {tiles_type.__name__} tiles")
    if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError("topk_ids and topk_weights must share a rank-2 shape")
    topk_ids = topk_ids.to(device=hidden_states.device, dtype=torch.int32).contiguous()
    topk_weights = topk_weights.to(
        device=hidden_states.device, dtype=torch.float32
    ).contiguous()
    ragged_metadata, gather_indx, scatter_indx, _ = _precomputed_topk_route(
        topk_weights, topk_ids, num_experts
    )
    x_fp8 = _quantize_fp8_activation(hidden_states, w13_weight.act_scale)
    if decode:
        intermediate = launch_gluon_moe_decode_warp_pipelined_gfx1250(
            x_fp8,
            w13_weight,
            w13_mx_scale,
            w13_bias,
            ragged_metadata,
            gather_indx,
            None,
            x_global_scale=w13_weight.act_scale,
            y_global_scale=w2_weight.act_scale,
            activation=activation,
            out_dtype=torch.float8_e4m3fn,
            tiles=gate_up_tiles,
        )
        flat = launch_gluon_moe_decode_warp_pipelined_gfx1250(
            intermediate,
            w2_weight,
            w2_mx_scale,
            w2_bias,
            ragged_metadata,
            None,
            scatter_indx,
            x_global_scale=w2_weight.act_scale,
            y_global_scale=None,
            activation=None,
            out_dtype=out_dtype,
            tiles=down_tiles,
        )
    else:
        intermediate = launch_gluon_moe_gate_up_warp_pipelined_gfx1250(
            x_fp8,
            w13_weight,
            w13_mx_scale,
            w13_bias,
            ragged_metadata,
            gather_indx,
            x_global_scale=w13_weight.act_scale,
            y_global_scale=w2_weight.act_scale,
            activation=activation,
            tiles=gate_up_tiles,
        )
        flat = launch_gluon_moe_down_warp_pipelined_gfx1250(
            intermediate,
            w2_weight,
            w2_mx_scale,
            w2_bias,
            ragged_metadata,
            scatter_indx,
            x_global_scale=w2_weight.act_scale,
            out_dtype=out_dtype,
            tiles=down_tiles,
        )
    return _weighted_topk_reduce_gfx1250(
        flat, topk_weights, out=out, out_dtype=out_dtype
    )


__all__ = [
    "DECODE_TILES",
    "PREFILL_DOWN_TILES",
    "PREFILL_GATE_UP_TILES",
    "DecodeTiles",
    "PrefillTiles",
    "gluon_mxfp_precomputed_mxfp4_warp_pipelined_moe",
    "launch_gluon_moe_decode_warp_pipelined_gfx1250",
    "launch_gluon_moe_down_warp_pipelined_gfx1250",
    "launch_gluon_moe_gate_up_warp_pipelined_gfx1250",
    "launch_warp_pipelined_down",
    "launch_warp_pipelined_gate_up",
    "make_activation",
    "select_decode_tiles",
    "select_prefill_tiles",
    "select_warp_pipelined_gemms",
]
