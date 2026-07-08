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


from __future__ import annotations

from typing import Any, Optional

import torch
from tokenspeed_kernel_amd._triton import aggregate, gl, gluon, tl, triton
from tokenspeed_kernel_amd.ops.moe.utils import (
    FnSpecs,
    FusedActivation,
    RaggedTensorMetadata,
    make_ragged_tensor_metadata,
    swiglu_fn,
    topk,
)


def _as_int32(t):
    if t is None or t.dtype == torch.int32:
        return t
    return t.to(torch.int32)


def _wrapped_tensor_data(obj):
    storage = getattr(obj, "storage", None)
    data = getattr(storage, "data", None)
    return data if isinstance(data, torch.Tensor) else None


_BLOCK_SIZES_TUPLE = tuple(RaggedTensorMetadata.block_sizes())
_BLOCK_SIZES_FROZEN = frozenset(_BLOCK_SIZES_TUPLE)
_BLOCK_SIZE_TO_IDX = {bs: i for i, bs in enumerate(_BLOCK_SIZES_TUPLE)}


def _ragged_block_offs(metadata, block_size: int):
    return metadata.block_offs_data[_BLOCK_SIZE_TO_IDX[block_size]]


def _ragged_block_schedule(metadata, block_size: int):
    return metadata.block_schedule_data[_BLOCK_SIZE_TO_IDX[block_size]]


def composition(cls):
    """A decorator lets aggregate type to directly access attributes from its aggregate member."""

    def __getattr__(self, name):
        if name in self.__dict__:
            return object.__getattribute__(self, name)
        for member in self.__dict__.values():
            if getattr(member, "__triton_aggregate__", False) and hasattr(member, name):
                return getattr(member, name)
        raise AttributeError(f"{type(self).__name__} object has no attribute '{name}'")

    cls.__getattr__ = __getattr__
    return cls


def _estimate_pipeline_lds_per_buffer(
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    x_format: str,
    w_format: str = "e2m1",
    has_x_block_scale: bool | None = None,
    has_w_block_scale: bool = True,
    scale_load_mode: str = "transpose",
) -> int:
    if has_x_block_scale is None:
        has_x_block_scale = x_format == "e2m1"
    x_bytes = block_m * block_k
    if x_format == "e2m1":
        x_bytes //= 2
    w_bytes = block_n * block_k
    if w_format == "e2m1":
        w_bytes //= 2
    scale_bytes = 0
    if scale_load_mode != "bypass":
        if has_x_block_scale:
            scale_bytes += block_m * (block_k // 32)
        if has_w_block_scale:
            scale_bytes += block_n * (block_k // 32)
    return x_bytes + w_bytes + scale_bytes


_CDNA4_NUM_CUS = 256
_PERSISTENT_OVERSUBSCRIBE = 2
_PERSISTENT_TILES_THRESHOLD = _CDNA4_NUM_CUS * 3

_GLUON_DOT_K_WIDTH = 16
_GLUON_DOT_N_LANE = 16
_GLUON_DOT_K_QUAD = 4
_GLUON_DOT_SUB_TILE_K = _GLUON_DOT_K_QUAD * _GLUON_DOT_K_WIDTH  # = 64

_TCP_INFLIGHT_CAP_BYTES = 32 * 1024  # gfx9 L1/TCP per-CU in-flight cap
_CDNA4_NUM_XCDS = 8  # MI355X has 8 XCDs (chiplets) per device.


def shuffle_weight_for_gluon_dot_layout(
    w: torch.Tensor,
    *,
    block_k_pk: int = 128,
    block_n: int = 128,
) -> torch.Tensor:
    K_pk, N = w.shape[-2], w.shape[-1]

    if block_k_pk <= 0 or block_k_pk % _GLUON_DOT_SUB_TILE_K != 0:
        raise ValueError(
            f"shuffle_weight_for_gluon_dot_layout requires block_k_pk "
            f"to be a positive multiple of {_GLUON_DOT_SUB_TILE_K} "
            f"(MFMA SUB_TILE_K); got {block_k_pk}."
        )
    if block_n <= 0 or block_n % _GLUON_DOT_N_LANE != 0:
        raise ValueError(
            f"shuffle_weight_for_gluon_dot_layout requires block_n to "
            f"be a positive multiple of {_GLUON_DOT_N_LANE} (MFMA "
            f"N_LANE); got {block_n}."
        )
    # The preshuffled W path uses tile-level predicates instead of a full
    # per-element N mask, so N must be block_n-aligned. The combine GEMM pads
    # W + W-scale at the backend before this helper sees the tensor; we still
    # assert here to catch unaligned callers.
    if N % block_n != 0:
        raise ValueError(
            f"shuffle_weight_for_gluon_dot_layout requires N "
            f"divisible by block_n={block_n} (got N={N}); the kernel's "
            f"preshuffled W path assumes block_n-aligned N. Pad the raw W "
            f"and its e8m0 W-scale at the backend layer (W with zeros, "
            f"scale with 127 = identity) BEFORE calling "
            f"``swizzle_mxfp4`` and this helper; trim the kernel "
            f"output back to the logical N in the high-level launcher."
        )
    k_tile_bytes = block_k_pk * block_n
    # Zero-pad K_pk to a multiple of block_k_pk (kernel's k_limit_w
    # masks the tail); supports gpt-oss-120b H=2880 etc.
    K_pk_padded = (K_pk + block_k_pk - 1) // block_k_pk * block_k_pk
    N_CTA_TILES = N // block_n

    # In-tile dims: (n_block, k_block, k_quad, n_in_sub, k_within).
    k_block_dim = block_k_pk // _GLUON_DOT_SUB_TILE_K
    stride_n_in_sub = _GLUON_DOT_K_WIDTH
    stride_k_quad = _GLUON_DOT_N_LANE * _GLUON_DOT_K_WIDTH
    stride_k_block = _GLUON_DOT_K_QUAD * stride_k_quad
    stride_n_block = k_block_dim * stride_k_block

    # (k, n) -> shuffled HBM byte offset within one CTA tile.
    k = torch.arange(block_k_pk, dtype=torch.int64).view(-1, 1)
    n = torch.arange(block_n, dtype=torch.int64).view(1, -1)
    k_within = k % _GLUON_DOT_K_WIDTH
    k_quad = (k // _GLUON_DOT_K_WIDTH) % _GLUON_DOT_K_QUAD
    k_block = k // _GLUON_DOT_SUB_TILE_K
    n_in_sub = n % _GLUON_DOT_N_LANE
    n_block_in_tile = n // _GLUON_DOT_N_LANE
    in_tile_offset = (
        n_block_in_tile * stride_n_block
        + k_block * stride_k_block
        + k_quad * stride_k_quad
        + n_in_sub * stride_n_in_sub
        + k_within
    )

    # Across CTA tiles: tile (kt, nt) -> byte (kt * N_CTA_TILES + nt) * k_tile_bytes.
    K_grid_full = torch.arange(K_pk_padded, dtype=torch.int64).view(-1, 1)
    N_grid_full = torch.arange(N, dtype=torch.int64).view(1, -1)
    kt = K_grid_full // block_k_pk
    nt = N_grid_full // block_n
    k_in_tile = K_grid_full % block_k_pk
    n_in_tile = N_grid_full % block_n
    in_tile_2d = in_tile_offset[k_in_tile, n_in_tile]  # [K_pk_padded, N]
    P = (kt * N_CTA_TILES + nt) * k_tile_bytes + in_tile_2d

    leading_shape = list(w.shape[:-2])
    leading = 1
    for s in leading_shape:
        leading *= s
    w_kn = w.reshape(leading, K_pk, N)
    # Zero-pad K_pk -> K_pk_padded; the kernel's k_limit_w masks the tail.
    if K_pk_padded != K_pk:
        pad = torch.zeros(
            leading, K_pk_padded - K_pk, N, dtype=w.dtype, device=w.device
        )
        w_kn_padded = torch.cat([w_kn, pad], dim=-2)
    else:
        w_kn_padded = w_kn
    # K-innermost flat source: src[e, n*K_pk_padded + k] = W[e, k, n].
    w_nk_contig = w_kn_padded.transpose(-1, -2).contiguous()
    src_flat = w_nk_contig.reshape(leading, K_pk_padded * N)

    K_grid = (
        torch.arange(K_pk_padded, dtype=torch.int64).view(-1, 1).expand(K_pk_padded, N)
    )
    N_grid = torch.arange(N, dtype=torch.int64).view(1, -1).expand(K_pk_padded, N)
    src_idx_kn = (N_grid * K_pk_padded + K_grid).flatten().to(w.device)
    P_flat = P.flatten().to(w.device)

    src_in_kn_order = src_flat.index_select(-1, src_idx_kn)
    out_flat = torch.empty(leading, K_pk_padded * N, dtype=w.dtype, device=w.device)
    out_flat.scatter_(
        -1,
        P_flat.unsqueeze(0).expand_as(out_flat),
        src_in_kn_order,
    )

    # Shape is (..., K_pk_padded, N); ``k_limit_w`` (= original K_pk)
    # masks the padded tail. Launcher must pass logical K from X.
    out = out_flat.view(*leading_shape, K_pk_padded, N)
    out.is_shuffled_for_gluon_dot = True
    out.original_k_pk = K_pk
    out.gluon_dot_block_k_pk = block_k_pk
    out.gluon_dot_block_n = block_n
    return out


# ---------------------------------------------------------------------------
# Layout factories (gluon constexpr functions)
# ---------------------------------------------------------------------------


@gluon.constexpr_function
def _store_layout(
    num_warps: int,
    block_m: int = 0,
    w_via_vgpr: bool = False,
    use_narrow_n_layout: bool = False,
):
    # Mirrors the warps_m policy in get_mfma_layout so the MFMA acc
    # and store layouts stay convert-compatible.
    if w_via_vgpr and num_warps >= 4:
        warps_m = 2
    elif block_m and block_m <= 32 and num_warps >= 4:
        warps_m = 1
    else:
        warps_m = 2 if num_warps >= 4 else 1
    warps_n = num_warps // warps_m
    # Selected W-via-VGPR combine routes store 16 contiguous N values per
    # thread instead of 32; profiling found this lower-resource store layout
    # faster for those shapes.
    if use_narrow_n_layout and w_via_vgpr and block_m >= 64 and num_warps >= 4:
        return gl.BlockedLayout([1, 16], [8, 8], [warps_m, warps_n], [1, 0])
    if w_via_vgpr and block_m >= 128 and num_warps >= 4:
        return gl.BlockedLayout([1, 16], [4, 16], [warps_m, warps_n], [1, 0])
    return gl.BlockedLayout([1, 32], [8, 8], [warps_m, warps_n], [1, 0])


@gluon.constexpr_function
def _load_layout(
    block_k: int,
    block_nonk: int,
    num_warps: int,
    order: list[int] = [1, 0],
    elem_bits: int = 8,
):
    # CDNA4 direct-to-LDS coalesce: K_PER_THREAD * elem_bits <= 128.
    max_vec = max(1, 128 // elem_bits)
    K_PER_THREAD: gl.constexpr = min(max_vec, block_k)
    LANES_K = block_k // K_PER_THREAD
    LANES_NONK = 64 // LANES_K
    NONK_PER_WARP = LANES_NONK
    if block_nonk >= NONK_PER_WARP:
        WARPS_NONK = block_nonk // NONK_PER_WARP
        if WARPS_NONK > num_warps:
            WARPS_NONK = num_warps
        WARPS_K = num_warps // WARPS_NONK
    else:
        # Narrow tile: more lanes on K so warps_K * warps_NONK == num_warps.
        WARPS_NONK = 1
        WARPS_K = num_warps
    if order == [1, 0]:
        regs = [1, K_PER_THREAD]
        lanes = [LANES_NONK, LANES_K]
        warps = [WARPS_NONK, WARPS_K]
    else:
        regs = [K_PER_THREAD, 1]
        lanes = [LANES_K, LANES_NONK]
        warps = [WARPS_K, WARPS_NONK]
    return gl.BlockedLayout(regs, lanes, warps, order)


# ---------------------------------------------------------------------------
# Software-pipelined Gluon MoE kernel
# ---------------------------------------------------------------------------


@gluon.constexpr_function
def _swiglu_split_layout(
    block_m: int, block_n_full: int, num_warps: int
) -> gl.constexpr:
    THREADS_PER_WARP = 64  # CDNA4 wavefront size.
    return gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, THREADS_PER_WARP // 4],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )


@gluon.jit
def _swiglu_reduce(
    acc,
    alpha: gl.constexpr,
    limit: gl.constexpr,
    OUT_BLOCK_N: gl.constexpr,
    MMA: gl.constexpr,
):
    BLOCK_M: gl.constexpr = acc.shape[0]
    BLOCK_N_FULL: gl.constexpr = acc.shape[1]
    SPLIT_LAYOUT: gl.constexpr = _swiglu_split_layout(
        BLOCK_M, BLOCK_N_FULL, gl.num_warps()
    )
    acc = gl.convert_layout(acc, SPLIT_LAYOUT)
    reshaped = acc.reshape((BLOCK_M, OUT_BLOCK_N, 2))
    gate, linear = gl.split(reshaped)
    if limit > 0.0:
        gate = gl.minimum(gate, limit)
        linear = gl.clamp(linear, -limit, limit)
    s = gate / (1.0 + gl.exp(-alpha * gate))
    return s * (linear + 1.0)


# ---------------------------------------------------------------------------
# Scaled MFMA MoE kernel (mxfp4 / fp8 + e8m0 block scales)
# ---------------------------------------------------------------------------


@gluon.constexpr_function
def get_mfma_layout(
    num_warps: int,
    use_mfma_scaled: bool,
    scale_preshuffle: bool = False,
    block_m: int = 0,
    w_via_vgpr: bool = False,
) -> gl.constexpr:
    # CDNA4 (gfx950): scaled MFMA = 16x16x128 (mxfp/fp8); regular = 16x16x32.
    # ``[2, 2]`` warps_per_cta split keeps W DotOperand per warp at
    # half the ``[num_warps, 1]`` footprint -- the latter spills VGPRs
    # at BN=256. ``w_via_vgpr`` forces ``warps_m=2`` because the host-
    # preshuffled ``LOAD_W_LAYOUT`` assumes that split for the
    # ``assert_trivial=True`` convert; BM<=32 small-tile decode prefers
    # ``warps_m=1`` to keep the fundamental block from over-filling M.
    assert num_warps in (4, 8), "MI355 MoE kernel currently supports 4 or 8 warps."
    if w_via_vgpr and num_warps >= 4:
        warps_m = 2
    elif block_m and block_m <= 32 and num_warps >= 4:
        warps_m = 1
    else:
        warps_m = 2 if num_warps >= 4 else 1
    warps_n = num_warps // warps_m
    instr_shape = [16, 16, 128] if use_mfma_scaled else [16, 16, 32]
    # tpw=[2,2] required when scales preshuffle through LDS (the 5-D
    # unswizzle view absorbs one 2x2 MFMA block per warp per K-iter).
    tiles_per_warp = [2, 2] if scale_preshuffle else [1, 1]
    return gl.amd.cdna4.AMDMFMALayout(
        version=4,
        instr_shape=instr_shape,
        transposed=True,
        warps_per_cta=[warps_m, warps_n],
        tiles_per_warp=tiles_per_warp,
    )


_SCALE_LOAD_MODES = ("bypass", "transpose", "swizzle")
_SCALE_PRESHUFFLE_FACTOR = 32
_SCALE_ASYNC_VEC = 4  # 32-bit, smallest direct-to-LDS unit on CDNA4.

# Constants matching triton_kernels' CDNA4MXScaleLayout.
_NON_K_PRESHUFFLE_BLOCK_SIZE = 32
_ALIGN_K_SCALE_SWIZZLE = 8
_ALIGN_N_SWIZZLE = 32
# Inner reshape factor for the 7-D unswizzle: K_SCALE_pad must be a
# multiple of this for `unswizzle_mx_scale_cdna4` to be well-defined.
_SWIZZLE_K_S_INNER = 8


def _effective_scale_load_mode(
    mode: str,
    block_m: int,
    block_n: int,
    block_k: int,
    scale_block: int,
    has_x_scale: bool,
    has_w_scale: bool,
    k: int | None = None,
    x_format: str | None = None,
    num_buffers: int | None = None,
) -> str:
    del k, x_format, num_buffers
    if mode != "swizzle":
        return mode
    # CDNA4MXScaleLayout requires BLOCK_K_S >= 8 and BLOCK_{M,N} %
    # 32 == 0 when the corresponding scale is present. Hard-assert
    # (no fallback) -- the input scale tensor is already in the
    # upstream swizzled storage.
    bk_s = block_k // scale_block
    assert bk_s >= _SWIZZLE_K_S_INNER, (
        f"swizzle requires BLOCK_K // SCALE_BLOCK >= "
        f"{_SWIZZLE_K_S_INNER} (got BLOCK_K={block_k}, "
        f"SCALE_BLOCK={scale_block} -> BLOCK_K_S={bk_s}). Bump "
        f"BLOCK_K to >= {_SWIZZLE_K_S_INNER * scale_block}."
    )
    if has_x_scale:
        assert block_m % _NON_K_PRESHUFFLE_BLOCK_SIZE == 0, (
            f"swizzle requires BLOCK_M % "
            f"{_NON_K_PRESHUFFLE_BLOCK_SIZE} == 0 when x_scale is "
            f"present (got BLOCK_M={block_m})."
        )
    if has_w_scale:
        assert block_n % _NON_K_PRESHUFFLE_BLOCK_SIZE == 0, (
            f"swizzle requires BLOCK_N % "
            f"{_NON_K_PRESHUFFLE_BLOCK_SIZE} == 0 when w_scale is "
            f"present (got BLOCK_N={block_n})."
        )
    return "swizzle"


@aggregate
class MoEConfig:
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    NUM_WARPS: gl.constexpr

    DIV_FACTOR_X: gl.constexpr
    DIV_FACTOR_W: gl.constexpr
    DTYPE_X: gl.constexpr
    DTYPE_W: gl.constexpr

    W_TRANSPOSE: gl.constexpr
    W_PRESHUFFLED: gl.constexpr
    W_VIA_VGPR: gl.constexpr
    W_PREFETCH: gl.constexpr
    NUM_BUFFERS: gl.constexpr

    SCALE_BLOCK: gl.constexpr
    WITH_X_MX_SCALE: gl.constexpr
    WITH_W_MX_SCALE: gl.constexpr
    SCALE_LOAD_MODE: gl.constexpr
    X_SCALE_VIA_LDS: gl.constexpr
    W_SCALE_VIA_LDS: gl.constexpr
    PRESHUFFLE_FACTOR: gl.constexpr
    BLOCK_M_PRESHUFFLED: gl.constexpr
    BLOCK_N_PRESHUFFLED: gl.constexpr
    BLOCK_K_SCALE_PRESHUFFLED: gl.constexpr
    shared_layout_w_half_n: gl.constexpr
    shared_layout_x_half_m: gl.constexpr

    NUM_SUBTILES: gl.constexpr
    EVEN_K: gl.constexpr
    K_ITERS: gl.constexpr
    USE_GATHER: gl.constexpr
    USE_MFMA_SCALED: gl.constexpr
    NUM_LOADS_IN_BATCH: gl.constexpr

    shared_layout_x: gl.constexpr
    dot_layout_x: gl.constexpr

    shared_layout_w: gl.constexpr
    dot_layout_w: gl.constexpr

    layout_x_scale: gl.constexpr
    layout_w_scale: gl.constexpr

    shared_layout_x_scale: gl.constexpr
    shared_layout_w_scale: gl.constexpr
    load_layout_x_scale: gl.constexpr
    load_layout_w_scale: gl.constexpr

    acc_layout: gl.constexpr

    index_type: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        DTYPE_X,
        DTYPE_W,
        SCALE_BLOCK,
        NUM_BUFFERS,
        W_TRANSPOSE,
        WITH_X_MX_SCALE,
        WITH_W_MX_SCALE,
        SCALE_LOAD_MODE,
        index_type,
        NUM_SUBTILES=(1, 1, 1),
        EVEN_K=True,
        K_ITERS=0,
        USE_GATHER=False,
        NUM_WARPS=4,
        W_PRESHUFFLED=False,
        W_VIA_VGPR=False,
        W_PREFETCH=True,
        X_SCALE_VIA_LDS=None,
        W_SCALE_VIA_LDS=None,
    ):
        if SCALE_LOAD_MODE not in _SCALE_LOAD_MODES:
            raise ValueError(
                f"SCALE_LOAD_MODE must be one of {_SCALE_LOAD_MODES}, "
                f"got {SCALE_LOAD_MODE!r}"
            )
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.BLOCK_K = gl.constexpr(BLOCK_K)
        self.NUM_BUFFERS = gl.constexpr(NUM_BUFFERS)
        self.W_TRANSPOSE = gl.constexpr(W_TRANSPOSE)
        self.W_PRESHUFFLED = gl.constexpr(W_PRESHUFFLED)
        self.W_VIA_VGPR = gl.constexpr(W_VIA_VGPR)
        self.W_PREFETCH = gl.constexpr(W_PREFETCH)
        self.WITH_X_MX_SCALE = gl.constexpr(WITH_X_MX_SCALE)
        self.WITH_W_MX_SCALE = gl.constexpr(WITH_W_MX_SCALE)
        self.SCALE_LOAD_MODE = gl.constexpr(SCALE_LOAD_MODE)
        self.SCALE_BLOCK = gl.constexpr(SCALE_BLOCK)
        self.DIV_FACTOR_X = gl.constexpr(2 if DTYPE_X == "e2m1" else 1)
        self.DIV_FACTOR_W = gl.constexpr(2 if DTYPE_W == "e2m1" else 1)
        self.DTYPE_X = gl.constexpr(DTYPE_X)
        self.DTYPE_W = gl.constexpr(DTYPE_W)

        if X_SCALE_VIA_LDS is None:
            X_SCALE_VIA_LDS = SCALE_LOAD_MODE == "swizzle" and WITH_X_MX_SCALE
        if W_SCALE_VIA_LDS is None:
            W_SCALE_VIA_LDS = SCALE_LOAD_MODE == "swizzle" and WITH_W_MX_SCALE
        _scale_via_lds = X_SCALE_VIA_LDS or W_SCALE_VIA_LDS
        self.X_SCALE_VIA_LDS = gl.constexpr(X_SCALE_VIA_LDS)
        self.W_SCALE_VIA_LDS = gl.constexpr(W_SCALE_VIA_LDS)
        self.PRESHUFFLE_FACTOR = gl.constexpr(_SCALE_PRESHUFFLE_FACTOR)
        self.BLOCK_M_PRESHUFFLED = gl.constexpr(BLOCK_M // _SCALE_PRESHUFFLE_FACTOR)
        self.BLOCK_N_PRESHUFFLED = gl.constexpr(BLOCK_N // _SCALE_PRESHUFFLE_FACTOR)
        self.BLOCK_K_SCALE_PRESHUFFLED = gl.constexpr(
            (BLOCK_K // SCALE_BLOCK) * _SCALE_PRESHUFFLE_FACTOR
        )

        self.NUM_SUBTILES = gl.constexpr(NUM_SUBTILES)
        self.EVEN_K = gl.constexpr(EVEN_K)
        self.K_ITERS = gl.constexpr(K_ITERS)
        self.USE_GATHER = gl.constexpr(USE_GATHER)
        _SCALED_FORMATS = ("e2m1", "e4m3", "e5m2")
        self.USE_MFMA_SCALED = gl.constexpr(
            DTYPE_X in _SCALED_FORMATS and DTYPE_W in _SCALED_FORMATS
        )
        self.NUM_WARPS = gl.constexpr(NUM_WARPS)

        num_loads = 1  # x
        if not W_VIA_VGPR:
            num_loads += 1  # w (LDS path)
        if _scale_via_lds:
            if X_SCALE_VIA_LDS:
                num_loads += 1
            if W_SCALE_VIA_LDS:
                num_loads += 1
        self.NUM_LOADS_IN_BATCH = gl.constexpr(num_loads)

        BLOCK_K_SCALE = BLOCK_K // SCALE_BLOCK
        self.index_type = gl.constexpr(index_type)

        MFMA_LAYOUT: gl.constexpr = get_mfma_layout(
            NUM_WARPS,
            self.USE_MFMA_SCALED,
            scale_preshuffle=_scale_via_lds,
            block_m=BLOCK_M,
            w_via_vgpr=W_VIA_VGPR or W_PRESHUFFLED,
        )

        DOT_K_WIDTH_X: gl.constexpr = 16 if self.USE_MFMA_SCALED else 8
        DOT_K_WIDTH_W: gl.constexpr = 16 if self.USE_MFMA_SCALED else 8

        NUM_SUBTILES_M = self.NUM_SUBTILES[0]
        NUM_SUBTILES_N = self.NUM_SUBTILES[1]
        NUM_SUBTILES_K = self.NUM_SUBTILES[2]

        self.dot_layout_x = gl.constexpr(
            gl.DotOperandLayout(
                operand_index=0, parent=MFMA_LAYOUT, k_width=DOT_K_WIDTH_X
            )
        )
        self.dot_layout_w = gl.constexpr(
            gl.DotOperandLayout(
                operand_index=1, parent=MFMA_LAYOUT, k_width=DOT_K_WIDTH_W
            )
        )
        if self.USE_MFMA_SCALED:
            self.layout_x_scale = gl.constexpr(
                gl.amd.cdna4.get_mfma_scale_layout(
                    self.dot_layout_x,
                    [BLOCK_M // NUM_SUBTILES_M, BLOCK_K_SCALE // NUM_SUBTILES_K],
                )
            )
            self.layout_w_scale = gl.constexpr(
                gl.amd.cdna4.get_mfma_scale_layout(
                    self.dot_layout_w,
                    [BLOCK_N // NUM_SUBTILES_N, BLOCK_K_SCALE // NUM_SUBTILES_K],
                )
            )
        else:
            self.layout_x_scale = gl.constexpr(0)
            self.layout_w_scale = gl.constexpr(0)
        self.acc_layout = gl.constexpr(MFMA_LAYOUT)

        BLOCK_K_PACKED_X_HOST = BLOCK_K // self.DIV_FACTOR_X
        BLOCK_K_PACKED_W_HOST = BLOCK_K // self.DIV_FACTOR_W

        def _row_major_offsets(H, W):
            H = int(H)
            W = int(W)
            inner = [[0, 1 << i] for i in range(W.bit_length() - 1)]
            outer = [[1 << i, 0] for i in range(H.bit_length() - 1)]
            return inner + outer

        self.shared_layout_x = gl.constexpr(
            gl.PaddedSharedLayout(
                [[1024, 32]],
                _row_major_offsets(BLOCK_M, BLOCK_K_PACKED_X_HOST),
                [],
                [BLOCK_M, BLOCK_K_PACKED_X_HOST],
            )
        )
        if W_PRESHUFFLED:
            w_shape = [BLOCK_N // 16, BLOCK_K_PACKED_W_HOST * 16]
        elif W_TRANSPOSE:
            w_shape = [BLOCK_N, BLOCK_K_PACKED_W_HOST]
        else:
            w_shape = [BLOCK_K_PACKED_W_HOST, BLOCK_N]
        self.shared_layout_w = gl.constexpr(
            gl.PaddedSharedLayout(
                [[1024, 32]],
                _row_major_offsets(w_shape[0], w_shape[1]),
                [],
                w_shape,
            )
        )

        if W_PRESHUFFLED:
            w_half_shape = [BLOCK_N // 2 // 16, BLOCK_K_PACKED_W_HOST * 16]
        elif W_TRANSPOSE:
            w_half_shape = [BLOCK_N // 2, BLOCK_K_PACKED_W_HOST]
        else:
            w_half_shape = [BLOCK_K_PACKED_W_HOST, BLOCK_N // 2]
        if (BLOCK_N // 2) >= 1 and BLOCK_K_PACKED_W_HOST >= 1:
            self.shared_layout_w_half_n = gl.constexpr(
                gl.PaddedSharedLayout(
                    [[1024, 32]],
                    _row_major_offsets(w_half_shape[0], w_half_shape[1]),
                    [],
                    w_half_shape,
                )
            )
        else:
            self.shared_layout_w_half_n = gl.constexpr(0)

        if (BLOCK_M // 2) >= 1 and BLOCK_K_PACKED_X_HOST >= 1:
            self.shared_layout_x_half_m = gl.constexpr(
                gl.PaddedSharedLayout(
                    [[1024, 32]],
                    _row_major_offsets(BLOCK_M // 2, BLOCK_K_PACKED_X_HOST),
                    [],
                    [BLOCK_M // 2, BLOCK_K_PACKED_X_HOST],
                )
            )
        else:
            self.shared_layout_x_half_m = gl.constexpr(0)

        if _scale_via_lds:
            self.shared_layout_x_scale = gl.constexpr(
                gl.SwizzledSharedLayout(4, 1, 1, order=[1, 0])
            )
            self.shared_layout_w_scale = gl.constexpr(
                gl.SwizzledSharedLayout(4, 1, 1, order=[1, 0])
            )
            self.load_layout_x_scale = gl.constexpr(
                _scale_async_blocked_layout(
                    BLOCK_M // _SCALE_PRESHUFFLE_FACTOR,
                    (BLOCK_K // SCALE_BLOCK) * _SCALE_PRESHUFFLE_FACTOR,
                    NUM_WARPS,
                )
            )
            self.load_layout_w_scale = gl.constexpr(
                _scale_async_blocked_layout(
                    BLOCK_N // _SCALE_PRESHUFFLE_FACTOR,
                    (BLOCK_K // SCALE_BLOCK) * _SCALE_PRESHUFFLE_FACTOR,
                    NUM_WARPS,
                )
            )
        else:
            self.shared_layout_x_scale = gl.constexpr(0)
            self.shared_layout_w_scale = gl.constexpr(0)
            self.load_layout_x_scale = gl.constexpr(0)
            self.load_layout_w_scale = gl.constexpr(0)


@aggregate
class MoEProgramBase:

    @gluon.constexpr_function
    def __init__(self):
        pass

    @gluon.jit
    def mfma(self, x, scale_x, w, scale_w, accumulator):
        cfg = self.cfg
        if cfg.USE_MFMA_SCALED:
            return gl.amd.cdna4.mfma_scaled(
                x, scale_x, cfg.DTYPE_X, w, scale_w, cfg.DTYPE_W, accumulator
            )
        else:
            return gl.amd.cdna4.mfma(x, w, accumulator)

    @gluon.jit
    def issue_global_loads(self, load_idx, pred=1, USE_MASK: gl.constexpr = -1):
        cfg = self.cfg
        self.x_desc.issue_async_load(load_idx, self.x_buffer, pred, USE_MASK=USE_MASK)
        if not cfg.W_VIA_VGPR:
            self.w_desc.issue_async_load(
                load_idx, self.w_buffer, pred, USE_MASK=USE_MASK
            )
        scale_via_lds: gl.constexpr = cfg.X_SCALE_VIA_LDS or cfg.W_SCALE_VIA_LDS
        if scale_via_lds:
            if cfg.X_SCALE_VIA_LDS:
                self.x_scale_desc.issue_async_load(
                    load_idx, self.x_scale_buffer, pred, USE_MASK=USE_MASK
                )
            if cfg.W_SCALE_VIA_LDS:
                self.w_scale_desc.issue_async_load(
                    load_idx, self.w_scale_buffer, pred, USE_MASK=USE_MASK
                )
        return load_idx + 1

    @gluon.jit
    def async_wait(self, waitcnt):
        gl.amd.cdna4.async_copy.wait_group(waitcnt * self.cfg.NUM_LOADS_IN_BATCH)


@gluon.constexpr_function
def get_bitwidth(dtype):
    if isinstance(dtype, gl.pointer_type):
        dtype = dtype.element_ty
    return dtype.primitive_bitwidth


@gluon.constexpr_function
def get_blocked_layout(num_warps: gl.constexpr, dtype: gl.constexpr, order):
    bitwidth = get_bitwidth(dtype)
    vector_size = (
        [1, max(1, 128 // bitwidth)] if order[1] == 0 else [max(1, 128 // bitwidth), 1]
    )
    warps_per_cta = [num_warps // 2, 2] if order[1] == 0 else [2, num_warps // 2]
    return gl.BlockedLayout(vector_size, [8, 8], warps_per_cta, order)


@gluon.constexpr_function
def get_scale_blocked_layout(num_warps: gl.constexpr):
    return gl.BlockedLayout([1, 8], [1, 64], [num_warps // 2, 2], [1, 0])


@gluon.constexpr_function
def _scale_async_blocked_layout(
    BLOCK_NONK_PS: gl.constexpr, BLOCK_K_PS: gl.constexpr, NUM_WARPS: gl.constexpr
):
    vec = 4
    lanes_k = max(1, min(64, BLOCK_K_PS // vec))
    lanes_nonk = max(1, 64 // lanes_k)
    warps_nonk = max(1, min(NUM_WARPS, BLOCK_NONK_PS // lanes_nonk))
    warps_k = max(1, NUM_WARPS // warps_nonk)
    return gl.BlockedLayout(
        [1, vec],
        [lanes_nonk, lanes_k],
        [warps_nonk, warps_k],
        [1, 0],
    )


@gluon.aggregate
class AsyncCopyDescriptor:
    cfg: MoEConfig
    op_idx: gl.constexpr
    ptr: gl.tensor
    dtype: gl.constexpr
    stride_k: gl.tensor
    stride_nonk: gl.tensor
    offsets: gl.tensor
    off_k: gl.tensor
    off_nonk: gl.tensor
    masks_nonk: gl.tensor
    k_limit: gl.tensor
    base_offset: gl.tensor
    BLOCK_K: gl.constexpr
    cache_modifier: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        cfg: MoEConfig,
        op_idx,
        BLOCK_K,
        ptr,
        dtype,
        stride_k,
        stride_nonk,
        offsets,
        off_k,
        off_nonk,
        masks_nonk,
        k_limit,
        base_offset,
        cache_modifier="",
    ):
        self.cfg = cfg
        self.op_idx = gl.constexpr(op_idx)
        self.BLOCK_K = gl.constexpr(BLOCK_K)
        self.ptr = ptr
        self.dtype = gl.constexpr(dtype)
        self.stride_k = stride_k
        self.stride_nonk = stride_nonk
        self.offsets = offsets
        self.off_k = off_k
        self.off_nonk = off_nonk
        self.masks_nonk = masks_nonk
        self.k_limit = k_limit
        self.base_offset = base_offset
        self.cache_modifier = gl.constexpr(cache_modifier)

    @gluon.jit
    def initialize(
        cfg: MoEConfig,
        op_idx: gl.constexpr,
        BLOCK_K: gl.constexpr,
        ptr,
        off_nonk,
        off_k,
        stride_nonk,
        stride_k,
        masks_nonk,
        k_limit,
        base_offset=0,
        cache_modifier: gl.constexpr = "",
    ):
        offsets = (
            gl.expand_dims(off_k, op_idx) * stride_k
            + gl.expand_dims(off_nonk, 1 - op_idx) * stride_nonk
            + base_offset
        )
        dtype: gl.constexpr = ptr.dtype.element_ty
        stride_k_t = gl.to_tensor(stride_k)
        stride_nonk_t = gl.to_tensor(stride_nonk)
        base_offset_t = gl.to_tensor(base_offset)
        return AsyncCopyDescriptor(
            cfg,
            op_idx,
            BLOCK_K,
            ptr,
            dtype,
            stride_k_t,
            stride_nonk_t,
            offsets,
            off_k,
            off_nonk,
            masks_nonk,
            k_limit,
            base_offset_t,
            cache_modifier,
        )

    @gluon.jit
    def issue_async_load(
        self,
        idx,
        buffer,
        pred=1,
        USE_MASK: gl.constexpr = -1,
        COMMIT: gl.constexpr = 1,
    ):
        NUM_BUFFERS: gl.constexpr = self.cfg.NUM_BUFFERS
        EVEN_K: gl.constexpr = self.cfg.EVEN_K
        if USE_MASK == -1:
            USE_MASK_RESOLVED: gl.constexpr = 0 if EVEN_K else 1
        else:
            USE_MASK_RESOLVED: gl.constexpr = USE_MASK
        CACHE_MODIFIER: gl.constexpr = self.cache_modifier
        off_k_step = idx * self.BLOCK_K
        offsets = self.offsets + off_k_step * self.stride_k
        if USE_MASK_RESOLVED == 0:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                buffer.index(idx % NUM_BUFFERS),
                self.ptr,
                offsets,
                cache_modifier=CACHE_MODIFIER,
            )
        else:
            # IMPORTANT: do not pass ``other=0`` here. A non-null
            # ``other`` causes the lowering to emit per-element
            # branches around each ``buffer.load.async.lds`` which
            # break ``SIInsertWaitcnts`` static counting and collapse
            # the async pipeline to ``s_waitcnt vmcnt(0)``. We rely on
            # the buffer descriptor's ``numRecords`` OOB check to zero
            # masked-out lanes in LDS.
            mask_k = gl.expand_dims(off_k_step + self.off_k, self.op_idx) < self.k_limit
            mask = mask_k & self.masks_nonk
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                buffer.index(idx % NUM_BUFFERS),
                self.ptr,
                offsets,
                mask=mask,
                cache_modifier=CACHE_MODIFIER,
            )
        if COMMIT == 1:
            gl.amd.cdna4.async_copy.commit_group()

    @gluon.jit
    def issue_local_load(
        self, idx, buffer, layout: gl.constexpr, do_permute: gl.constexpr = False
    ):
        NUM_BUFFERS: gl.constexpr = self.cfg.NUM_BUFFERS
        slot = buffer.index(idx % NUM_BUFFERS)
        if do_permute:
            slot = slot.permute([1, 0])
        return gl.amd.cdna4.async_copy.load_shared_relaxed(slot, layout)

    @gluon.jit
    def issue_local_load_m_swizzle(
        self,
        idx,
        buffer,
        layout: gl.constexpr,
        BLOCK_M: gl.constexpr,
    ):
        if BLOCK_M == 32:
            GROUPS_M: gl.constexpr = 2
        else:
            gl.static_assert(
                BLOCK_M == 64 or BLOCK_M == 128,
                "M-swizzled local load supports BLOCK_M in {32, 64, 128}",
            )
            GROUPS_M: gl.constexpr = 4
        ROWS_PER_GROUP: gl.constexpr = BLOCK_M // GROUPS_M
        NUM_BUFFERS: gl.constexpr = self.cfg.NUM_BUFFERS
        slot = buffer.index(idx % NUM_BUFFERS)
        slot_view = (
            slot.reshape((ROWS_PER_GROUP, GROUPS_M, self.BLOCK_K))
            .permute((1, 0, 2))
            .reshape((BLOCK_M, self.BLOCK_K))
        )
        return gl.amd.cdna4.async_copy.load_shared_relaxed(slot_view, layout)

    @gluon.jit
    def issue_local_load_unswizzle(
        self,
        idx,
        buffer,
        layout: gl.constexpr,
        BLOCK_NONK_PS: gl.constexpr,
        BLOCK_NONK: gl.constexpr,
        BLOCK_K_SCALE: gl.constexpr,
    ):
        NUM_BUFFERS: gl.constexpr = self.cfg.NUM_BUFFERS
        slot = buffer.index(idx % NUM_BUFFERS)
        slot_7d = slot.reshape((BLOCK_NONK_PS, BLOCK_K_SCALE // 8, 4, 16, 2, 2, 1))
        slot_perm = slot_7d.permute((0, 5, 3, 1, 4, 2, 6))
        slot_2d = slot_perm.reshape((BLOCK_NONK, BLOCK_K_SCALE))
        return gl.amd.cdna4.async_copy.load_shared_relaxed(slot_2d, layout)

    @gluon.jit
    def issue_local_load_unswizzle_sub(
        self,
        idx,
        buffer,
        layout: gl.constexpr,
        BLOCK_NONK_PS: gl.constexpr,
        BLOCK_NONK: gl.constexpr,
        BLOCK_K_SCALE: gl.constexpr,
        SUBTILE_NONK: gl.constexpr,
        subtile_start_nonk: gl.constexpr,
    ):
        NUM_BUFFERS: gl.constexpr = self.cfg.NUM_BUFFERS
        slot = buffer.index(idx % NUM_BUFFERS)
        slot_view = (
            slot.reshape((BLOCK_NONK_PS, BLOCK_K_SCALE // 8, 4, 16, 2, 2, 1))
            .permute((0, 5, 3, 1, 4, 2, 6))
            .reshape((BLOCK_NONK, BLOCK_K_SCALE))
        )
        return gl.amd.cdna4.async_copy.load_shared_relaxed(
            slot_view.slice(subtile_start_nonk, SUBTILE_NONK, 0), layout
        )


@gluon.aggregate
class WVgprDescriptor:
    cfg: MoEConfig
    ptr: gl.tensor
    stride_k: gl.tensor  # = N (bytes between consecutive K-slabs)
    offsets: gl.tensor  # [LOAD_BN//N_LANE, BLOCK_K*N_LANE]
    pred: gl.tensor  # int1 scalar (broadcast to a per-element mask)
    BLOCK_K: gl.constexpr  # = BLOCK_K_W; mirrors AsyncCopyDescriptor
    LOAD_BN: gl.constexpr  # N width per load; SUB_BN under sliceN

    @gluon.constexpr_function
    def __init__(
        self, cfg: MoEConfig, BLOCK_K, ptr, stride_k, offsets, pred, LOAD_BN=None
    ):
        self.cfg = cfg
        self.BLOCK_K = gl.constexpr(BLOCK_K)
        self.LOAD_BN = gl.constexpr(LOAD_BN if LOAD_BN is not None else cfg.BLOCK_N)
        self.ptr = ptr
        self.stride_k = stride_k
        self.offsets = offsets
        self.pred = pred

    @gluon.jit
    def issue_global_load_to_vgpr(self, idx, dot_layout: gl.constexpr):
        BLOCK_K_W: gl.constexpr = self.BLOCK_K
        LOAD_BN: gl.constexpr = self.LOAD_BN

        # idx-th K-slab; per-iter shift folds into the scalar ptr so
        # ``offsets`` stays compile-time constant.
        k_iter_offset = idx * BLOCK_K_W * self.stride_k
        ptr_iter = self.ptr + k_iter_offset

        # ``mask`` is a scalar bool; buffer_load broadcasts it to the
        # offsets layout. Hardware OOB masking returns 0 for masked
        # lanes, which is what we want when ``pred=False``.
        tile_flat = gl.amd.cdna4.buffer_load(
            ptr=ptr_iter, offsets=self.offsets, mask=self.pred
        )

        # 5-D HBM layout -> (BLOCK_K_W, LOAD_BN) MFMA-ready.
        tile_5d = tile_flat.reshape(
            LOAD_BN // 16,
            BLOCK_K_W // 64,
            4,
            16,
            16,
        )
        tile_perm = tile_5d.permute(0, 3, 1, 2, 4)
        tile_2d = tile_perm.reshape(LOAD_BN, BLOCK_K_W)
        tile_t = tile_2d.trans(1, 0)

        return gl.convert_layout(tile_t, dot_layout, assert_trivial=True)


@gluon.aggregate
class WPreshuffledLdsDescriptor:
    cfg: MoEConfig
    ptr: gl.tensor
    dtype: gl.constexpr
    stride_k: gl.tensor  # = N bytes between consecutive K slabs.
    offsets: gl.tensor  # [LOAD_BN//16, BLOCK_K*16] in preshuffled tile order.
    pred: gl.tensor
    BLOCK_K: gl.constexpr
    LOAD_BN: gl.constexpr
    load_layout: gl.constexpr
    cache_modifier: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        cfg: MoEConfig,
        BLOCK_K,
        ptr,
        dtype,
        stride_k,
        offsets,
        pred,
        load_layout,
        LOAD_BN=None,
        cache_modifier="",
    ):
        self.cfg = cfg
        self.BLOCK_K = gl.constexpr(BLOCK_K)
        self.LOAD_BN = gl.constexpr(LOAD_BN if LOAD_BN is not None else cfg.BLOCK_N)
        self.ptr = ptr
        self.dtype = gl.constexpr(dtype)
        self.stride_k = stride_k
        self.offsets = offsets
        self.pred = pred
        self.load_layout = gl.constexpr(load_layout)
        self.cache_modifier = gl.constexpr(cache_modifier)

    @gluon.jit
    def issue_async_load(
        self,
        idx,
        buffer,
        pred=1,
        USE_MASK: gl.constexpr = -1,
        COMMIT: gl.constexpr = 1,
    ):
        NUM_BUFFERS: gl.constexpr = self.cfg.NUM_BUFFERS
        k_iter_offset = idx * self.BLOCK_K * self.stride_k
        offsets = self.offsets + k_iter_offset
        CACHE_MODIFIER: gl.constexpr = self.cache_modifier
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            buffer.index(idx % NUM_BUFFERS),
            self.ptr,
            offsets,
            mask=self.pred,
            cache_modifier=CACHE_MODIFIER,
        )
        if COMMIT == 1:
            gl.amd.cdna4.async_copy.commit_group()

    @gluon.jit
    def issue_local_load(
        self, idx, buffer, layout: gl.constexpr, do_permute: gl.constexpr = False
    ):
        NUM_BUFFERS: gl.constexpr = self.cfg.NUM_BUFFERS
        BLOCK_K_W: gl.constexpr = self.BLOCK_K
        LOAD_BN: gl.constexpr = self.LOAD_BN
        slot = buffer.index(idx % NUM_BUFFERS)
        slot_5d = slot.reshape((LOAD_BN // 16, BLOCK_K_W // 64, 4, 16, 16))
        slot_perm = slot_5d.permute((0, 3, 1, 2, 4))
        slot_2d = slot_perm.reshape((LOAD_BN, BLOCK_K_W))
        slot_t = slot_2d.permute((1, 0))
        return gl.amd.cdna4.async_copy.load_shared_relaxed(slot_t, layout)


@gluon.jit
def _load_scale_tile_via_gl_load(desc, mfma_idx):
    EVEN_K: gl.constexpr = desc.cfg.EVEN_K
    off_k_step = mfma_idx * desc.BLOCK_K
    base = desc.ptr + off_k_step * desc.stride_k
    if EVEN_K:
        mask = desc.masks_nonk
    else:
        mask_k = gl.expand_dims(off_k_step + desc.off_k, desc.op_idx) < desc.k_limit
        mask = mask_k & desc.masks_nonk
    return gl.load(base + desc.offsets, mask=mask, other=0)


@gluon.jit
def _load_scale_subtile_via_gl_load(
    desc, mfma_idx, subtile_start_nonk: gl.constexpr, SUBTILE_NONK: gl.constexpr
):
    EVEN_K: gl.constexpr = desc.cfg.EVEN_K
    off_k_step = mfma_idx * desc.BLOCK_K
    base = desc.ptr + off_k_step * desc.stride_k
    offsets = desc.offsets.slice(subtile_start_nonk, SUBTILE_NONK, 0)
    masks_nonk = desc.masks_nonk.slice(subtile_start_nonk, SUBTILE_NONK, 0)
    if EVEN_K:
        mask = masks_nonk
    else:
        mask_k = gl.expand_dims(off_k_step + desc.off_k, desc.op_idx) < desc.k_limit
        mask = mask_k & masks_nonk
    return gl.load(base + offsets, mask=mask, other=0)


@composition
@gluon.aggregate
class MoEPipelinedProgram:
    base: MoEProgramBase
    cfg: MoEConfig
    x_buffer: gl.shared_memory_descriptor
    w_buffer: gl.shared_memory_descriptor | gl.constexpr
    x_scale_buffer: gl.shared_memory_descriptor | gl.constexpr
    w_scale_buffer: gl.shared_memory_descriptor | gl.constexpr
    x_desc: AsyncCopyDescriptor
    w_desc: AsyncCopyDescriptor | WVgprDescriptor | WPreshuffledLdsDescriptor
    x_scale_desc: AsyncCopyDescriptor | gl.constexpr
    w_scale_desc: AsyncCopyDescriptor | gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        cfg: MoEConfig,
        x_buffer,
        w_buffer,
        x_scale_buffer,
        w_scale_buffer,
        x_desc,
        w_desc,
        x_scale_desc,
        w_scale_desc,
    ):
        self.cfg = cfg
        self.x_buffer = x_buffer
        self.w_buffer = w_buffer if not cfg.W_VIA_VGPR else gl.constexpr(0)
        self.x_scale_buffer = x_scale_buffer if cfg.X_SCALE_VIA_LDS else gl.constexpr(0)
        self.w_scale_buffer = w_scale_buffer if cfg.W_SCALE_VIA_LDS else gl.constexpr(0)
        self.x_desc = x_desc
        self.w_desc = w_desc
        self.x_scale_desc = x_scale_desc if cfg.WITH_X_MX_SCALE else gl.constexpr(0)
        self.w_scale_desc = w_scale_desc if cfg.WITH_W_MX_SCALE else gl.constexpr(0)
        self.base = MoEProgramBase()

    @gluon.jit
    def initialize(cfg: MoEConfig, x_desc, w_desc, x_scale_desc, w_scale_desc):
        NUM_BUFFERS: gl.constexpr = cfg.NUM_BUFFERS

        BLOCK_K_PACKED_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
        BLOCK_K_PACKED_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W

        x_buffer = gl.allocate_shared_memory(
            x_desc.dtype,
            shape=[NUM_BUFFERS, cfg.BLOCK_M, BLOCK_K_PACKED_X],
            layout=cfg.shared_layout_x,
        )
        # W_VIA_VGPR: skip W's LDS slot; K-loop does HBM->VGPR direct.
        if cfg.W_VIA_VGPR:
            w_buffer = gl.constexpr(0)
        elif cfg.W_PRESHUFFLED:
            w_buffer = gl.allocate_shared_memory(
                w_desc.dtype,
                shape=[NUM_BUFFERS, cfg.BLOCK_N // 16, BLOCK_K_PACKED_W * 16],
                layout=cfg.shared_layout_w,
            )
        else:
            w_buffer = gl.allocate_shared_memory(
                w_desc.dtype,
                shape=(
                    [NUM_BUFFERS, cfg.BLOCK_N, BLOCK_K_PACKED_W]
                    if cfg.W_TRANSPOSE
                    else [NUM_BUFFERS, BLOCK_K_PACKED_W, cfg.BLOCK_N]
                ),
                layout=cfg.shared_layout_w,
            )

        if cfg.X_SCALE_VIA_LDS:
            x_scale_buffer = gl.allocate_shared_memory(
                gl.uint8,
                shape=[
                    NUM_BUFFERS,
                    cfg.BLOCK_M_PRESHUFFLED,
                    cfg.BLOCK_K_SCALE_PRESHUFFLED,
                ],
                layout=cfg.shared_layout_x_scale,
            )
        else:
            x_scale_buffer = gl.constexpr(0)

        if cfg.W_SCALE_VIA_LDS:
            w_scale_buffer = gl.allocate_shared_memory(
                gl.uint8,
                shape=[
                    NUM_BUFFERS,
                    cfg.BLOCK_N_PRESHUFFLED,
                    cfg.BLOCK_K_SCALE_PRESHUFFLED,
                ],
                layout=cfg.shared_layout_w_scale,
            )
        else:
            w_scale_buffer = gl.constexpr(0)

        return MoEPipelinedProgram(
            cfg,
            x_buffer,
            w_buffer,
            x_scale_buffer,
            w_scale_buffer,
            x_desc,
            w_desc,
            x_scale_desc,
            w_scale_desc,
        )

    @gluon.jit
    def _issue_w_vgpr(self, mfma_idx):
        cfg = self.cfg
        return self.w_desc.issue_global_load_to_vgpr(
            mfma_idx,
            cfg.dot_layout_w,
        )

    @gluon.jit
    def _load_x(self, mfma_idx):
        cfg = self.cfg
        return self.x_desc.issue_local_load(
            mfma_idx,
            self.x_buffer,
            cfg.dot_layout_x,
        )

    @gluon.jit
    def _load_w(self, mfma_idx):
        cfg = self.cfg
        if cfg.W_VIA_VGPR:
            w = self._issue_w_vgpr(mfma_idx)
        else:
            w = self.w_desc.issue_local_load(
                mfma_idx,
                self.w_buffer,
                cfg.dot_layout_w,
                do_permute=cfg.W_TRANSPOSE,
            )
        return w

    @gluon.jit
    def _load_xw(self, mfma_idx):
        w = self._load_w(mfma_idx)
        x = self._load_x(mfma_idx)
        return x, w

    @gluon.jit
    def _load_xw_decode(self, mfma_idx):
        x = self._load_x(mfma_idx)
        w = self._load_w(mfma_idx)
        return x, w

    @gluon.jit
    def _load_x_scales(self, mfma_idx):
        cfg = self.cfg
        x = self.x_desc.issue_local_load(
            mfma_idx,
            self.x_buffer,
            cfg.dot_layout_x,
        )

        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK
        if cfg.USE_MFMA_SCALED:
            if cfg.WITH_X_MX_SCALE:
                if cfg.X_SCALE_VIA_LDS:
                    scale_x = self.x_scale_desc.issue_local_load_unswizzle(
                        mfma_idx,
                        self.x_scale_buffer,
                        cfg.layout_x_scale,
                        cfg.BLOCK_M_PRESHUFFLED,
                        cfg.BLOCK_M,
                        BLOCK_K_SCALE,
                    )
                else:
                    scale_x = _load_scale_tile_via_gl_load(self.x_scale_desc, mfma_idx)
            else:
                scale_x = gl.full(
                    [cfg.BLOCK_M, BLOCK_K_SCALE],
                    127,
                    gl.uint8,
                    layout=cfg.layout_x_scale,
                )
            if cfg.WITH_W_MX_SCALE:
                if cfg.W_SCALE_VIA_LDS:
                    scale_w = self.w_scale_desc.issue_local_load_unswizzle(
                        mfma_idx,
                        self.w_scale_buffer,
                        cfg.layout_w_scale,
                        cfg.BLOCK_N_PRESHUFFLED,
                        cfg.BLOCK_N,
                        BLOCK_K_SCALE,
                    )
                else:
                    scale_w = _load_scale_tile_via_gl_load(self.w_scale_desc, mfma_idx)
            else:
                scale_w = gl.full(
                    [cfg.BLOCK_N, BLOCK_K_SCALE],
                    127,
                    gl.uint8,
                    layout=cfg.layout_w_scale,
                )
        else:
            scale_x: gl.constexpr = 0
            scale_w: gl.constexpr = 0

        return x, scale_x, scale_w

    @gluon.jit
    def _load_scales(self, mfma_idx):
        cfg = self.cfg

        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK

        if cfg.USE_MFMA_SCALED:
            if cfg.WITH_X_MX_SCALE:
                if cfg.X_SCALE_VIA_LDS:
                    scale_x = self.x_scale_desc.issue_local_load_unswizzle(
                        mfma_idx,
                        self.x_scale_buffer,
                        cfg.layout_x_scale,
                        cfg.BLOCK_M_PRESHUFFLED,
                        cfg.BLOCK_M,
                        BLOCK_K_SCALE,
                    )
                else:
                    scale_x = _load_scale_tile_via_gl_load(self.x_scale_desc, mfma_idx)
            else:
                scale_x = gl.full(
                    [cfg.BLOCK_M, BLOCK_K_SCALE],
                    127,
                    gl.uint8,
                    layout=cfg.layout_x_scale,
                )

            if cfg.WITH_W_MX_SCALE:
                if cfg.W_SCALE_VIA_LDS:
                    scale_w = self.w_scale_desc.issue_local_load_unswizzle(
                        mfma_idx,
                        self.w_scale_buffer,
                        cfg.layout_w_scale,
                        cfg.BLOCK_N_PRESHUFFLED,
                        cfg.BLOCK_N,
                        BLOCK_K_SCALE,
                    )
                else:
                    scale_w = _load_scale_tile_via_gl_load(self.w_scale_desc, mfma_idx)
            else:
                scale_w = gl.full(
                    [cfg.BLOCK_N, BLOCK_K_SCALE],
                    127,
                    gl.uint8,
                    layout=cfg.layout_w_scale,
                )
        else:
            scale_x: gl.constexpr = 0
            scale_w: gl.constexpr = 0

        return scale_x, scale_w

    @gluon.jit
    def issue_local_loads(self, mfma_idx):
        scale_x, scale_w = self._load_scales(mfma_idx)
        x, w = self._load_xw(mfma_idx)

        return x, w, scale_x, scale_w

    @gluon.jit
    def issue_decode_local_loads(self, mfma_idx):
        x, w = self._load_xw_decode(mfma_idx)
        scale_x, scale_w = self._load_scales(mfma_idx)

        return x, w, scale_x, scale_w

    @gluon.jit
    def decode_pipeline(self, loop_k):
        cfg = self.cfg
        EVEN_K: gl.constexpr = cfg.EVEN_K
        load_idx = 0
        mfma_idx = 0

        accumulator = gl.zeros(
            (cfg.BLOCK_M, cfg.BLOCK_N), dtype=gl.float32, layout=cfg.acc_layout
        )
        K_iters = gl.cdiv(loop_k, cfg.BLOCK_K)

        W_PREFETCH: gl.constexpr = cfg.W_VIA_VGPR and cfg.W_PREFETCH

        for _ in gl.static_range(cfg.NUM_BUFFERS - 1):
            load_idx = self.issue_global_loads(load_idx, USE_MASK=0)

        if W_PREFETCH:
            w_curr = self._issue_w_vgpr(0)

        # EVEN_K: K_iters - (NUM_BUFFERS-1) all-unmasked main iters.
        # !EVEN_K: one less unmasked iter; the last is the masked tail below.
        main_iters = K_iters - (cfg.NUM_BUFFERS - 1 if EVEN_K else cfg.NUM_BUFFERS)
        gl.assume(main_iters >= 0)

        for _ in range(0, main_iters):
            load_idx = self.issue_global_loads(load_idx, USE_MASK=0)
            self.async_wait(cfg.NUM_BUFFERS - 1)

            if W_PREFETCH:
                x, scale_x, scale_w = self._load_x_scales(mfma_idx)
                accumulator = self.mfma(x, scale_x, w_curr, scale_w, accumulator)
                w_curr = self._issue_w_vgpr(mfma_idx + 1)
            else:
                x, w, scale_x, scale_w = self.issue_decode_local_loads(mfma_idx)
                accumulator = self.mfma(x, scale_x, w, scale_w, accumulator)
            mfma_idx += 1

        if not EVEN_K:
            # Masked tail iter (one more iter still has W to prefetch).
            load_idx = self.issue_global_loads(load_idx, USE_MASK=1)
            self.async_wait(cfg.NUM_BUFFERS - 1)
            if W_PREFETCH:
                x, scale_x, scale_w = self._load_x_scales(mfma_idx)
                accumulator = self.mfma(x, scale_x, w_curr, scale_w, accumulator)
                w_curr = self._issue_w_vgpr(mfma_idx + 1)
            else:
                x, w, scale_x, scale_w = self.issue_decode_local_loads(mfma_idx)
                accumulator = self.mfma(x, scale_x, w, scale_w, accumulator)
            mfma_idx += 1

        # Epilogue: drain remaining in-flight buffers; no new global loads.
        for i in gl.static_range(cfg.NUM_BUFFERS - 1):
            self.async_wait(cfg.NUM_BUFFERS - 2 - i)
            if W_PREFETCH:
                x, scale_x, scale_w = self._load_x_scales(mfma_idx)
                accumulator = self.mfma(x, scale_x, w_curr, scale_w, accumulator)
                if i < cfg.NUM_BUFFERS - 2:
                    w_curr = self._issue_w_vgpr(mfma_idx + 1)
            else:
                x, w, scale_x, scale_w = self.issue_decode_local_loads(mfma_idx)
                accumulator = self.mfma(x, scale_x, w, scale_w, accumulator)
            mfma_idx += 1

        return accumulator

    @gluon.jit
    def pipeline(self, loop_k):
        cfg = self.cfg
        gl.static_assert(
            cfg.NUM_BUFFERS == 2,
            "current local-prefetch pipeline requires exactly two LDS buffers",
        )
        load_idx = 0
        mfma_idx = 0

        accumulator = gl.zeros(
            (cfg.BLOCK_M, cfg.BLOCK_N), dtype=gl.float32, layout=cfg.acc_layout
        )
        if cfg.K_ITERS:
            K_iters: gl.constexpr = cfg.K_ITERS
        else:
            K_iters = gl.cdiv(loop_k, cfg.BLOCK_K)

        # Two-buffer local-prefetch pipeline, unrolled by 2:
        #   async_copy(k + 2) -> freed LDS buffer
        #   mfma(k)
        #   wait for k + 1, leaving k + 2 in flight
        #   local_load(k + 1) -> VGPR
        #
        # The unroll alternates between two explicit operand register sets.
        # This avoids rotating the freshly loaded "next" operands into a
        # single canonical x/w variable at the loop backedge.
        for _ in gl.static_range(cfg.NUM_BUFFERS):
            load_idx = self.issue_global_loads(load_idx, USE_MASK=-1)

        main_iters = K_iters - cfg.NUM_BUFFERS
        gl.assume(main_iters >= 0)

        self.async_wait(cfg.NUM_BUFFERS - 1)
        x0, w0, scale_x0, scale_w0 = self.issue_local_loads(mfma_idx)
        mfma_idx += 1

        unroll_pairs = main_iters // 2
        odd_main = main_iters - unroll_pairs * 2

        for _ in range(0, unroll_pairs):
            # All waves must finish reading the previous contents before any
            # wave overwrites this LDS slot with the future async copy.
            gl.barrier()
            load_idx = self.issue_global_loads(load_idx, USE_MASK=-1)
            accumulator = self.mfma(x0, scale_x0, w0, scale_w0, accumulator)
            self.async_wait(cfg.NUM_BUFFERS - 1)
            x1, w1, scale_x1, scale_w1 = self.issue_local_loads(mfma_idx)
            mfma_idx += 1

            gl.barrier()
            load_idx = self.issue_global_loads(load_idx, USE_MASK=-1)
            accumulator = self.mfma(x1, scale_x1, w1, scale_w1, accumulator)
            self.async_wait(cfg.NUM_BUFFERS - 1)
            x0, w0, scale_x0, scale_w0 = self.issue_local_loads(mfma_idx)
            mfma_idx += 1

        if odd_main:
            gl.barrier()
            load_idx = self.issue_global_loads(load_idx, USE_MASK=-1)
            accumulator = self.mfma(x0, scale_x0, w0, scale_w0, accumulator)
            self.async_wait(cfg.NUM_BUFFERS - 1)
            x1, w1, scale_x1, scale_w1 = self.issue_local_loads(mfma_idx)
            mfma_idx += 1

            # Drain remaining prefetched K tiles; no new global loads.
            accumulator = self.mfma(x1, scale_x1, w1, scale_w1, accumulator)
            self.async_wait(0)
            x0, w0, scale_x0, scale_w0 = self.issue_local_loads(mfma_idx)
            accumulator = self.mfma(x0, scale_x0, w0, scale_w0, accumulator)
        else:
            # Drain remaining prefetched K tiles; no new global loads.
            accumulator = self.mfma(x0, scale_x0, w0, scale_w0, accumulator)
            self.async_wait(0)
            x1, w1, scale_x1, scale_w1 = self.issue_local_loads(mfma_idx)
            accumulator = self.mfma(x1, scale_x1, w1, scale_w1, accumulator)

        return accumulator

    @gluon.jit
    def warp_pipeline(self, loop_k):
        cfg = self.cfg
        gl.static_assert(
            cfg.NUM_BUFFERS >= 3,
            "warp_pipeline requires NUM_BUFFERS >= 3",
        )
        load_idx = 0
        mfma_idx = 0

        for _ in gl.static_range(cfg.NUM_BUFFERS - 1):
            load_idx = self.issue_global_loads(load_idx)

        accumulator = gl.zeros(
            (cfg.BLOCK_M, cfg.BLOCK_N), dtype=gl.float32, layout=cfg.acc_layout
        )
        main_iters = gl.cdiv(loop_k, cfg.BLOCK_K) - (cfg.NUM_BUFFERS - 1)
        gl.assume(main_iters >= 0)

        # Drain oldest prologue batch into LDS; rest remain in flight.
        self.async_wait(cfg.NUM_BUFFERS - 2)

        for _ in range(0, main_iters):
            with gl.amd.warp_pipeline_stage("lds+tdm", priority=1):
                x, w, scale_x, scale_w = self.issue_local_loads(mfma_idx)
                mfma_idx += 1
                load_idx = self.issue_global_loads(load_idx)

            self.async_wait(cfg.NUM_BUFFERS - 2)

            with gl.amd.warp_pipeline_stage("mfma", priority=0):
                accumulator = self.mfma(x, scale_x, w, scale_w, accumulator)

        self.async_wait(0)
        for _ in gl.static_range(cfg.NUM_BUFFERS - 1):
            x, w, scale_x, scale_w = self.issue_local_loads(mfma_idx)
            mfma_idx += 1
            accumulator = self.mfma(x, scale_x, w, scale_w, accumulator)

        return accumulator


@composition
@gluon.aggregate
class MoESliceMNProgram:
    base: MoEProgramBase
    cfg: MoEConfig
    x_buffer_top: gl.shared_memory_descriptor
    x_buffer_bot: gl.shared_memory_descriptor
    w_buffer_left: gl.shared_memory_descriptor
    w_buffer_right: gl.shared_memory_descriptor
    x_scale_buffer: gl.shared_memory_descriptor | gl.constexpr
    w_scale_buffer: gl.shared_memory_descriptor | gl.constexpr
    x_desc_top: AsyncCopyDescriptor
    x_desc_bot: AsyncCopyDescriptor
    w_desc_left: AsyncCopyDescriptor
    w_desc_right: AsyncCopyDescriptor
    x_scale_desc: AsyncCopyDescriptor | gl.constexpr
    w_scale_desc: AsyncCopyDescriptor | gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        cfg: MoEConfig,
        x_buffer_top,
        x_buffer_bot,
        w_buffer_left,
        w_buffer_right,
        x_scale_buffer,
        w_scale_buffer,
        x_desc_top,
        x_desc_bot,
        w_desc_left,
        w_desc_right,
        x_scale_desc,
        w_scale_desc,
    ):
        self.cfg = cfg
        self.x_buffer_top = x_buffer_top
        self.x_buffer_bot = x_buffer_bot
        self.w_buffer_left = w_buffer_left
        self.w_buffer_right = w_buffer_right
        self.x_scale_buffer = x_scale_buffer if cfg.X_SCALE_VIA_LDS else gl.constexpr(0)
        self.w_scale_buffer = w_scale_buffer if cfg.W_SCALE_VIA_LDS else gl.constexpr(0)
        self.x_desc_top = x_desc_top
        self.x_desc_bot = x_desc_bot
        self.w_desc_left = w_desc_left
        self.w_desc_right = w_desc_right
        self.x_scale_desc = x_scale_desc if cfg.WITH_X_MX_SCALE else gl.constexpr(0)
        self.w_scale_desc = w_scale_desc if cfg.WITH_W_MX_SCALE else gl.constexpr(0)
        self.base = MoEProgramBase()

    @gluon.jit
    def initialize(
        cfg: MoEConfig,
        x_desc_top,
        x_desc_bot,
        w_desc_left,
        w_desc_right,
        x_scale_desc,
        w_scale_desc,
    ):
        NUM_BUFFERS: gl.constexpr = cfg.NUM_BUFFERS
        BLOCK_K_PACKED_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
        BLOCK_K_PACKED_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W

        x_buffer_top = gl.allocate_shared_memory(
            x_desc_top.dtype,
            shape=[NUM_BUFFERS, cfg.BLOCK_M // 2, BLOCK_K_PACKED_X],
            layout=cfg.shared_layout_x_half_m,
        )
        x_buffer_bot = gl.allocate_shared_memory(
            x_desc_bot.dtype,
            shape=[NUM_BUFFERS, cfg.BLOCK_M // 2, BLOCK_K_PACKED_X],
            layout=cfg.shared_layout_x_half_m,
        )
        w_buffer_left = gl.allocate_shared_memory(
            w_desc_left.dtype,
            shape=(
                [NUM_BUFFERS, cfg.BLOCK_N // 2, BLOCK_K_PACKED_W]
                if cfg.W_TRANSPOSE
                else [NUM_BUFFERS, BLOCK_K_PACKED_W, cfg.BLOCK_N // 2]
            ),
            layout=cfg.shared_layout_w_half_n,
        )
        w_buffer_right = gl.allocate_shared_memory(
            w_desc_right.dtype,
            shape=(
                [NUM_BUFFERS, cfg.BLOCK_N // 2, BLOCK_K_PACKED_W]
                if cfg.W_TRANSPOSE
                else [NUM_BUFFERS, BLOCK_K_PACKED_W, cfg.BLOCK_N // 2]
            ),
            layout=cfg.shared_layout_w_half_n,
        )

        if cfg.X_SCALE_VIA_LDS:
            x_scale_buffer = gl.allocate_shared_memory(
                gl.uint8,
                shape=[
                    NUM_BUFFERS,
                    cfg.BLOCK_M_PRESHUFFLED,
                    cfg.BLOCK_K_SCALE_PRESHUFFLED,
                ],
                layout=cfg.shared_layout_x_scale,
            )
        else:
            x_scale_buffer = gl.constexpr(0)

        if cfg.W_SCALE_VIA_LDS:
            w_scale_buffer = gl.allocate_shared_memory(
                gl.uint8,
                shape=[
                    NUM_BUFFERS,
                    cfg.BLOCK_N_PRESHUFFLED,
                    cfg.BLOCK_K_SCALE_PRESHUFFLED,
                ],
                layout=cfg.shared_layout_w_scale,
            )
        else:
            w_scale_buffer = gl.constexpr(0)

        return MoESliceMNProgram(
            cfg,
            x_buffer_top,
            x_buffer_bot,
            w_buffer_left,
            w_buffer_right,
            x_scale_buffer,
            w_scale_buffer,
            x_desc_top,
            x_desc_bot,
            w_desc_left,
            w_desc_right,
            x_scale_desc,
            w_scale_desc,
        )

    @gluon.jit
    def issue_local_load_x_sub(self, mfma_idx, subtile_idx_m: gl.constexpr):
        cfg = self.cfg
        SUBTILE_M: gl.constexpr = cfg.BLOCK_M // 2
        subtile_start_m: gl.constexpr = subtile_idx_m * SUBTILE_M
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK

        if subtile_idx_m == 0:
            slot = self.x_buffer_top.index(mfma_idx % cfg.NUM_BUFFERS)
        else:
            slot = self.x_buffer_bot.index(mfma_idx % cfg.NUM_BUFFERS)
        x = gl.amd.cdna4.async_copy.load_shared_relaxed(slot, cfg.dot_layout_x)

        if cfg.USE_MFMA_SCALED:
            if cfg.WITH_X_MX_SCALE:
                if cfg.X_SCALE_VIA_LDS:
                    scale_x = self.x_scale_desc.issue_local_load_unswizzle_sub(
                        mfma_idx,
                        self.x_scale_buffer,
                        cfg.layout_x_scale,
                        cfg.BLOCK_M_PRESHUFFLED,
                        cfg.BLOCK_M,
                        BLOCK_K_SCALE,
                        SUBTILE_M,
                        subtile_start_m,
                    )
                else:
                    scale_x = _load_scale_subtile_via_gl_load(
                        self.x_scale_desc,
                        mfma_idx,
                        subtile_start_m,
                        SUBTILE_M,
                    )
            else:
                scale_x = gl.full(
                    [SUBTILE_M, BLOCK_K_SCALE],
                    127,
                    gl.uint8,
                    layout=cfg.layout_x_scale,
                )
        else:
            scale_x: gl.constexpr = 0

        return x, scale_x

    @gluon.jit
    def issue_local_load_w_sub(self, mfma_idx, subtile_idx_n: gl.constexpr):
        cfg = self.cfg
        SUBTILE_N: gl.constexpr = cfg.BLOCK_N // 2
        subtile_start_n: gl.constexpr = subtile_idx_n * SUBTILE_N
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK

        if subtile_idx_n == 0:
            slot = self.w_buffer_left.index(mfma_idx % cfg.NUM_BUFFERS)
        else:
            slot = self.w_buffer_right.index(mfma_idx % cfg.NUM_BUFFERS)
        if cfg.W_TRANSPOSE:
            w = gl.amd.cdna4.async_copy.load_shared_relaxed(
                slot.permute([1, 0]),
                cfg.dot_layout_w,
            )
        else:
            w = gl.amd.cdna4.async_copy.load_shared_relaxed(slot, cfg.dot_layout_w)

        if cfg.USE_MFMA_SCALED:
            if cfg.WITH_W_MX_SCALE:
                if cfg.W_SCALE_VIA_LDS:
                    scale_w = self.w_scale_desc.issue_local_load_unswizzle_sub(
                        mfma_idx,
                        self.w_scale_buffer,
                        cfg.layout_w_scale,
                        cfg.BLOCK_N_PRESHUFFLED,
                        cfg.BLOCK_N,
                        BLOCK_K_SCALE,
                        SUBTILE_N,
                        subtile_start_n,
                    )
                else:
                    scale_w = _load_scale_subtile_via_gl_load(
                        self.w_scale_desc,
                        mfma_idx,
                        subtile_start_n,
                        SUBTILE_N,
                    )
            else:
                scale_w = gl.full(
                    [SUBTILE_N, BLOCK_K_SCALE],
                    127,
                    gl.uint8,
                    layout=cfg.layout_w_scale,
                )
        else:
            scale_w: gl.constexpr = 0

        return w, scale_w

    @gluon.jit
    def issue_w_left(self, load_idx, pred=1, USE_MASK: gl.constexpr = -1):
        self.w_desc_left.issue_async_load(
            load_idx, self.w_buffer_left, pred, USE_MASK=USE_MASK, COMMIT=1
        )
        return load_idx

    @gluon.jit
    def issue_x_top(self, load_idx, pred=1, USE_MASK: gl.constexpr = -1):
        cfg = self.cfg
        self.x_desc_top.issue_async_load(
            load_idx, self.x_buffer_top, pred, USE_MASK=USE_MASK, COMMIT=0
        )
        scale_via_lds: gl.constexpr = cfg.X_SCALE_VIA_LDS or cfg.W_SCALE_VIA_LDS
        if scale_via_lds:
            if cfg.X_SCALE_VIA_LDS:
                self.x_scale_desc.issue_async_load(
                    load_idx,
                    self.x_scale_buffer,
                    pred,
                    USE_MASK=USE_MASK,
                    COMMIT=0,
                )
            if cfg.W_SCALE_VIA_LDS:
                self.w_scale_desc.issue_async_load(
                    load_idx,
                    self.w_scale_buffer,
                    pred,
                    USE_MASK=USE_MASK,
                    COMMIT=0,
                )
        gl.amd.cdna4.async_copy.commit_group()
        return load_idx

    @gluon.jit
    def issue_x_bot(self, load_idx, pred=1, USE_MASK: gl.constexpr = -1):
        self.x_desc_bot.issue_async_load(
            load_idx, self.x_buffer_bot, pred, USE_MASK=USE_MASK, COMMIT=1
        )
        return load_idx

    @gluon.jit
    def issue_w_right(self, load_idx, pred=1, USE_MASK: gl.constexpr = -1):
        self.w_desc_right.issue_async_load(
            load_idx, self.w_buffer_right, pred, USE_MASK=USE_MASK, COMMIT=1
        )
        return load_idx + 1

    @gluon.jit
    def issue_global_loads(self, load_idx, pred=1, USE_MASK: gl.constexpr = -1):
        load_idx = self.issue_w_left(load_idx, pred, USE_MASK=USE_MASK)
        load_idx = self.issue_x_top(load_idx, pred, USE_MASK=USE_MASK)
        load_idx = self.issue_x_bot(load_idx, pred, USE_MASK=USE_MASK)
        load_idx = self.issue_w_right(load_idx, pred, USE_MASK=USE_MASK)
        return load_idx

    @gluon.jit
    def async_wait(self, waitcnt):
        gl.amd.cdna4.async_copy.wait_group(waitcnt * 4)

    @gluon.jit
    def pipeline(self, loop_k):
        cfg = self.cfg
        NB: gl.constexpr = cfg.NUM_BUFFERS
        gl.static_assert(
            (cfg.NUM_SUBTILES[0] == 2)
            and (cfg.NUM_SUBTILES[1] == 2)
            and (cfg.NUM_SUBTILES[2] == 1),
            "MoESliceMNProgram requires NUM_SUBTILES=(2,2,1)",
        )
        gl.static_assert(NB >= 2, "MoESliceMNProgram requires NUM_BUFFERS >= 2")

        SUBTILE_M: gl.constexpr = cfg.BLOCK_M // 2
        SUBTILE_N: gl.constexpr = cfg.BLOCK_N // 2

        load_idx = 0
        mfma_idx = 0

        # Prologue: NB iters in flight (region 2/3 of iter 0 ds_read
        # iter 1 W_left / X_top, so NB not NB-1).
        for _ in gl.static_range(NB):
            load_idx = self.issue_global_loads(load_idx, USE_MASK=0)

        c_tl = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)
        c_bl = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)
        c_tr = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)
        c_br = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)

        if cfg.K_ITERS:
            K_iters: gl.constexpr = cfg.K_ITERS
        else:
            K_iters = gl.cdiv(loop_k, cfg.BLOCK_K)
        # K-tail mask absorbed via USE_MASK=-1 in-loop (no dedicated peel).
        main_iters = K_iters - NB
        gl.assume(main_iters >= 2)

        # Drain iter 0's first 2 commits so the first MFMA has data.
        gl.amd.cdna4.async_copy.wait_group(4 * NB - 2)
        x_top, sx_top = self.issue_local_load_x_sub(mfma_idx, 0)
        w_left, sw_left = self.issue_local_load_w_sub(mfma_idx, 0)

        # USE_MASK=-1 + in-loop mask drops the dedicated K-tail peel.
        # Region order ``mfma -> issue -> wait -> ds_read`` lets the
        # vmem coalesce start in parallel with the wait's s_barrier
        # (raising the wait target by 1 to compensate).
        unroll_pairs = main_iters // 2
        odd_main = main_iters - unroll_pairs * 2
        for _ in range(0, unroll_pairs):
            # iter k: 4 regions (consume buffer (m % NB), refill same).
            c_tl = self.mfma(x_top, sx_top, w_left, sw_left, c_tl)
            load_idx = self.issue_w_left(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 2)
            x_bot, sx_bot = self.issue_local_load_x_sub(mfma_idx, 1)

            c_bl = self.mfma(x_bot, sx_bot, w_left, sw_left, c_bl)
            # issue_x_top also refills the scale LDS slot. Read the
            # current right-W scale before that slot is reused.
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            w_right, sw_right = self.issue_local_load_w_sub(mfma_idx, 1)
            load_idx = self.issue_x_top(load_idx, USE_MASK=-1)

            c_tr = self.mfma(x_top, sx_top, w_right, sw_right, c_tr)
            mfma_idx += 1
            load_idx = self.issue_x_bot(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            w_left, sw_left = self.issue_local_load_w_sub(mfma_idx, 0)

            c_br = self.mfma(x_bot, sx_bot, w_right, sw_right, c_br)
            load_idx = self.issue_w_right(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 2)
            x_top, sx_top = self.issue_local_load_x_sub(mfma_idx, 0)

            # iter k+1: same 4 regions, ping-ponged buffer slot.
            c_tl = self.mfma(x_top, sx_top, w_left, sw_left, c_tl)
            load_idx = self.issue_w_left(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 2)
            x_bot, sx_bot = self.issue_local_load_x_sub(mfma_idx, 1)

            c_bl = self.mfma(x_bot, sx_bot, w_left, sw_left, c_bl)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            w_right, sw_right = self.issue_local_load_w_sub(mfma_idx, 1)
            load_idx = self.issue_x_top(load_idx, USE_MASK=-1)

            c_tr = self.mfma(x_top, sx_top, w_right, sw_right, c_tr)
            mfma_idx += 1
            load_idx = self.issue_x_bot(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            w_left, sw_left = self.issue_local_load_w_sub(mfma_idx, 0)

            c_br = self.mfma(x_bot, sx_bot, w_right, sw_right, c_br)
            load_idx = self.issue_w_right(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 2)
            x_top, sx_top = self.issue_local_load_x_sub(mfma_idx, 0)

        # Odd peel; same USE_MASK=-1 handles the K-tail iter.
        if odd_main:
            c_tl = self.mfma(x_top, sx_top, w_left, sw_left, c_tl)
            load_idx = self.issue_w_left(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 2)
            x_bot, sx_bot = self.issue_local_load_x_sub(mfma_idx, 1)

            c_bl = self.mfma(x_bot, sx_bot, w_left, sw_left, c_bl)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            w_right, sw_right = self.issue_local_load_w_sub(mfma_idx, 1)
            load_idx = self.issue_x_top(load_idx, USE_MASK=-1)

            c_tr = self.mfma(x_top, sx_top, w_right, sw_right, c_tr)
            mfma_idx += 1
            load_idx = self.issue_x_bot(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            w_left, sw_left = self.issue_local_load_w_sub(mfma_idx, 0)

            c_br = self.mfma(x_bot, sx_bot, w_right, sw_right, c_br)
            load_idx = self.issue_w_right(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 2)
            x_top, sx_top = self.issue_local_load_x_sub(mfma_idx, 0)

        # Drain epilogue: NB iters of MFMA, no further async issues.
        # Mirrors v8's "iterMax-2 / iterMax-1" tail with the trailing
        # ds_reads guarded by ``i < NB - 1`` (the last-iter MFMAs use
        # the final x_top / w_left already in regs).
        gl.amd.cdna4.async_copy.wait_group(0)
        for i in gl.static_range(NB):
            c_tl = self.mfma(x_top, sx_top, w_left, sw_left, c_tl)
            x_bot, sx_bot = self.issue_local_load_x_sub(mfma_idx, 1)

            c_bl = self.mfma(x_bot, sx_bot, w_left, sw_left, c_bl)
            w_right, sw_right = self.issue_local_load_w_sub(mfma_idx, 1)

            c_tr = self.mfma(x_top, sx_top, w_right, sw_right, c_tr)
            mfma_idx += 1
            if i < NB - 1:
                w_left, sw_left = self.issue_local_load_w_sub(mfma_idx, 0)

            c_br = self.mfma(x_bot, sx_bot, w_right, sw_right, c_br)
            if i < NB - 1:
                x_top, sx_top = self.issue_local_load_x_sub(mfma_idx, 0)

        # Stitch the 4 quadrants and re-anchor to cfg.acc_layout.
        acc_top = gl.join(c_tl, c_tr).permute(0, 2, 1).reshape((SUBTILE_M, cfg.BLOCK_N))
        acc_bot = gl.join(c_bl, c_br).permute(0, 2, 1).reshape((SUBTILE_M, cfg.BLOCK_N))
        accumulator = (
            gl.join(acc_top, acc_bot)
            .permute(2, 0, 1)
            .reshape((cfg.BLOCK_M, cfg.BLOCK_N))
        )
        accumulator = gl.convert_layout(accumulator, cfg.acc_layout)

        return accumulator


@composition
@gluon.aggregate
class MoESliceNProgram:
    base: MoEProgramBase
    cfg: MoEConfig
    x_buffer: gl.shared_memory_descriptor
    w_buffer_top: gl.shared_memory_descriptor | gl.constexpr
    w_buffer_bot: gl.shared_memory_descriptor | gl.constexpr
    x_scale_buffer: gl.shared_memory_descriptor | gl.constexpr
    w_scale_buffer: gl.shared_memory_descriptor | gl.constexpr
    x_desc: AsyncCopyDescriptor
    w_desc_top: AsyncCopyDescriptor | WVgprDescriptor | WPreshuffledLdsDescriptor
    w_desc_bot: AsyncCopyDescriptor | WVgprDescriptor | WPreshuffledLdsDescriptor
    x_scale_desc: AsyncCopyDescriptor | gl.constexpr
    w_scale_desc: AsyncCopyDescriptor | gl.constexpr
    bottom_valid: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        cfg: MoEConfig,
        x_buffer,
        w_buffer_top,
        w_buffer_bot,
        x_scale_buffer,
        w_scale_buffer,
        x_desc,
        w_desc_top,
        w_desc_bot,
        x_scale_desc,
        w_scale_desc,
        bottom_valid,
    ):
        self.cfg = cfg
        self.x_buffer = x_buffer
        self.w_buffer_top = w_buffer_top if not cfg.W_VIA_VGPR else gl.constexpr(0)
        self.w_buffer_bot = w_buffer_bot if not cfg.W_VIA_VGPR else gl.constexpr(0)
        self.x_scale_buffer = x_scale_buffer if cfg.X_SCALE_VIA_LDS else gl.constexpr(0)
        self.w_scale_buffer = w_scale_buffer if cfg.W_SCALE_VIA_LDS else gl.constexpr(0)
        self.x_desc = x_desc
        self.w_desc_top = w_desc_top
        self.w_desc_bot = w_desc_bot
        self.x_scale_desc = x_scale_desc if cfg.WITH_X_MX_SCALE else gl.constexpr(0)
        self.w_scale_desc = w_scale_desc if cfg.WITH_W_MX_SCALE else gl.constexpr(0)
        self.bottom_valid = bottom_valid
        self.base = MoEProgramBase()

    @gluon.jit
    def initialize(
        cfg: MoEConfig,
        x_desc,
        w_desc_top,
        w_desc_bot,
        x_scale_desc,
        w_scale_desc,
        bottom_valid,
    ):
        NUM_BUFFERS: gl.constexpr = cfg.NUM_BUFFERS
        BLOCK_K_PACKED_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
        BLOCK_K_PACKED_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W

        x_buffer = gl.allocate_shared_memory(
            x_desc.dtype,
            shape=[NUM_BUFFERS, cfg.BLOCK_M, BLOCK_K_PACKED_X],
            layout=cfg.shared_layout_x,
        )
        if cfg.W_VIA_VGPR:
            w_buffer_top = gl.constexpr(0)
            w_buffer_bot = gl.constexpr(0)
        elif cfg.W_PRESHUFFLED:
            w_buffer_top = gl.allocate_shared_memory(
                w_desc_top.dtype,
                shape=[
                    NUM_BUFFERS,
                    cfg.BLOCK_N // 2 // 16,
                    BLOCK_K_PACKED_W * 16,
                ],
                layout=cfg.shared_layout_w_half_n,
            )
            w_buffer_bot = gl.allocate_shared_memory(
                w_desc_bot.dtype,
                shape=[
                    NUM_BUFFERS,
                    cfg.BLOCK_N // 2 // 16,
                    BLOCK_K_PACKED_W * 16,
                ],
                layout=cfg.shared_layout_w_half_n,
            )
        else:
            w_buffer_top = gl.allocate_shared_memory(
                w_desc_top.dtype,
                shape=(
                    [NUM_BUFFERS, cfg.BLOCK_N // 2, BLOCK_K_PACKED_W]
                    if cfg.W_TRANSPOSE
                    else [NUM_BUFFERS, BLOCK_K_PACKED_W, cfg.BLOCK_N // 2]
                ),
                layout=cfg.shared_layout_w_half_n,
            )
            w_buffer_bot = gl.allocate_shared_memory(
                w_desc_bot.dtype,
                shape=(
                    [NUM_BUFFERS, cfg.BLOCK_N // 2, BLOCK_K_PACKED_W]
                    if cfg.W_TRANSPOSE
                    else [NUM_BUFFERS, BLOCK_K_PACKED_W, cfg.BLOCK_N // 2]
                ),
                layout=cfg.shared_layout_w_half_n,
            )

        if cfg.X_SCALE_VIA_LDS:
            x_scale_buffer = gl.allocate_shared_memory(
                gl.uint8,
                shape=[
                    NUM_BUFFERS,
                    cfg.BLOCK_M_PRESHUFFLED,
                    cfg.BLOCK_K_SCALE_PRESHUFFLED,
                ],
                layout=cfg.shared_layout_x_scale,
            )
        else:
            x_scale_buffer = gl.constexpr(0)

        if cfg.W_SCALE_VIA_LDS:
            w_scale_buffer = gl.allocate_shared_memory(
                gl.uint8,
                shape=[
                    NUM_BUFFERS,
                    cfg.BLOCK_N_PRESHUFFLED,
                    cfg.BLOCK_K_SCALE_PRESHUFFLED,
                ],
                layout=cfg.shared_layout_w_scale,
            )
        else:
            w_scale_buffer = gl.constexpr(0)

        return MoESliceNProgram(
            cfg,
            x_buffer,
            w_buffer_top,
            w_buffer_bot,
            x_scale_buffer,
            w_scale_buffer,
            x_desc,
            w_desc_top,
            w_desc_bot,
            x_scale_desc,
            w_scale_desc,
            bottom_valid,
        )

    @gluon.jit
    def issue_local_load_x(self, mfma_idx):
        cfg = self.cfg
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK
        if not cfg.W_VIA_VGPR and (
            cfg.BLOCK_M == 32 or cfg.BLOCK_M == 64 or cfg.BLOCK_M == 128
        ):
            x = self.x_desc.issue_local_load_m_swizzle(
                mfma_idx,
                self.x_buffer,
                cfg.dot_layout_x,
                cfg.BLOCK_M,
            )
        else:
            x = self.x_desc.issue_local_load(
                mfma_idx,
                self.x_buffer,
                cfg.dot_layout_x,
            )

        if cfg.USE_MFMA_SCALED:
            if cfg.WITH_X_MX_SCALE:
                if cfg.X_SCALE_VIA_LDS:
                    scale_x = self.x_scale_desc.issue_local_load_unswizzle(
                        mfma_idx,
                        self.x_scale_buffer,
                        cfg.layout_x_scale,
                        cfg.BLOCK_M_PRESHUFFLED,
                        cfg.BLOCK_M,
                        BLOCK_K_SCALE,
                    )
                else:
                    scale_x = _load_scale_tile_via_gl_load(self.x_scale_desc, mfma_idx)
            else:
                # fp8 X path: identity scale (e8m0=127 == 2^0).
                scale_x = gl.full(
                    [cfg.BLOCK_M, BLOCK_K_SCALE],
                    127,
                    gl.uint8,
                    layout=cfg.layout_x_scale,
                )
        else:
            scale_x: gl.constexpr = 0

        return x, scale_x

    @gluon.jit
    def issue_global_load_top(self, load_idx, pred=1, USE_MASK: gl.constexpr = -1):
        cfg = self.cfg
        self.x_desc.issue_async_load(
            load_idx, self.x_buffer, pred, USE_MASK=USE_MASK, COMMIT=0
        )
        scale_via_lds: gl.constexpr = cfg.X_SCALE_VIA_LDS or cfg.W_SCALE_VIA_LDS
        if scale_via_lds:
            if cfg.X_SCALE_VIA_LDS:
                self.x_scale_desc.issue_async_load(
                    load_idx,
                    self.x_scale_buffer,
                    pred,
                    USE_MASK=USE_MASK,
                    COMMIT=0,
                )
            if cfg.W_SCALE_VIA_LDS:
                self.w_scale_desc.issue_async_load(
                    load_idx,
                    self.w_scale_buffer,
                    pred,
                    USE_MASK=USE_MASK,
                    COMMIT=0,
                )
        if not cfg.W_VIA_VGPR:
            self.w_desc_top.issue_async_load(
                load_idx, self.w_buffer_top, pred, USE_MASK=USE_MASK, COMMIT=0
            )
        gl.amd.cdna4.async_copy.commit_group()
        return load_idx

    @gluon.jit
    def issue_global_load_bot(self, load_idx, pred=1, USE_MASK: gl.constexpr = -1):
        cfg = self.cfg
        if cfg.W_VIA_VGPR:
            gl.amd.cdna4.async_copy.commit_group()
        else:
            self.w_desc_bot.issue_async_load(
                load_idx, self.w_buffer_bot, pred, USE_MASK=USE_MASK, COMMIT=1
            )
        return load_idx + 1

    @gluon.jit
    def issue_global_loads(self, load_idx, pred=1, USE_MASK: gl.constexpr = -1):
        load_idx = self.issue_global_load_top(load_idx, pred, USE_MASK=USE_MASK)
        load_idx = self.issue_global_load_bot(load_idx, pred, USE_MASK=USE_MASK)
        return load_idx

    @gluon.jit
    def async_wait(self, waitcnt):
        gl.amd.cdna4.async_copy.wait_group(waitcnt * 2)

    @gluon.jit
    def issue_local_load_w_sub(self, mfma_idx, subtile_idx_n: gl.constexpr):
        cfg = self.cfg
        SUBTILE_N: gl.constexpr = cfg.BLOCK_N // cfg.NUM_SUBTILES[1]
        subtile_start_n: gl.constexpr = subtile_idx_n * SUBTILE_N
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK

        if cfg.W_VIA_VGPR:
            if subtile_idx_n == 0:
                w = self.w_desc_top.issue_global_load_to_vgpr(
                    mfma_idx, cfg.dot_layout_w
                )
            else:
                w = self.w_desc_bot.issue_global_load_to_vgpr(
                    mfma_idx, cfg.dot_layout_w
                )
        elif cfg.W_PRESHUFFLED:
            if subtile_idx_n == 0:
                w = self.w_desc_top.issue_local_load(
                    mfma_idx,
                    self.w_buffer_top,
                    cfg.dot_layout_w,
                )
            else:
                w = self.w_desc_bot.issue_local_load(
                    mfma_idx,
                    self.w_buffer_bot,
                    cfg.dot_layout_w,
                )
        else:
            if subtile_idx_n == 0:
                slot = self.w_buffer_top.index(mfma_idx % cfg.NUM_BUFFERS)
            else:
                slot = self.w_buffer_bot.index(mfma_idx % cfg.NUM_BUFFERS)

            if cfg.W_TRANSPOSE:
                w = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    slot.permute([1, 0]),
                    cfg.dot_layout_w,
                )
            else:
                w = gl.amd.cdna4.async_copy.load_shared_relaxed(slot, cfg.dot_layout_w)

        if cfg.USE_MFMA_SCALED:
            if cfg.WITH_W_MX_SCALE:
                scale_w = self.w_scale_desc.issue_local_load_unswizzle_sub(
                    mfma_idx,
                    self.w_scale_buffer,
                    cfg.layout_w_scale,
                    cfg.BLOCK_N_PRESHUFFLED,
                    cfg.BLOCK_N,
                    BLOCK_K_SCALE,
                    SUBTILE_N,
                    subtile_start_n,
                )
            else:
                scale_w = gl.full(
                    [SUBTILE_N, BLOCK_K_SCALE],
                    127,
                    gl.uint8,
                    layout=cfg.layout_w_scale,
                )
        else:
            scale_w: gl.constexpr = 0

        return w, scale_w

    @gluon.jit
    def _finish_accumulator(self, c0, c1):
        cfg = self.cfg
        accumulator = (
            gl.join(c0, c1).permute(0, 2, 1).reshape((cfg.BLOCK_M, cfg.BLOCK_N))
        )
        accumulator = gl.convert_layout(accumulator, cfg.acc_layout)
        return accumulator

    @gluon.jit
    def _pipeline_top_only(self, loop_k):
        cfg = self.cfg
        NB: gl.constexpr = cfg.NUM_BUFFERS
        SUBTILE_N: gl.constexpr = cfg.BLOCK_N // 2

        load_idx = 0
        mfma_idx = 0

        for _ in gl.static_range(NB):
            load_idx = self.issue_global_loads(load_idx, USE_MASK=0)

        c0 = gl.zeros((cfg.BLOCK_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)
        c1 = gl.zeros((cfg.BLOCK_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)

        if cfg.K_ITERS:
            K_iters: gl.constexpr = cfg.K_ITERS
        else:
            K_iters = gl.cdiv(loop_k, cfg.BLOCK_K)
        main_iters = K_iters - NB
        gl.assume(main_iters >= 0)

        gl.amd.cdna4.async_copy.wait_group(2 * NB - 1)
        w00, sw00 = self.issue_local_load_w_sub(mfma_idx, 0)
        x0, sx0 = self.issue_local_load_x(mfma_idx)
        mfma_idx += 1

        unroll_pairs = main_iters // 2
        odd_main = main_iters - unroll_pairs * 2

        for _ in range(0, unroll_pairs):
            gl.barrier()
            load_idx = self.issue_global_loads(load_idx, USE_MASK=-1)
            c0 = self.mfma(x0, sx0, w00, sw00, c0)
            gl.amd.cdna4.async_copy.wait_group(2 * NB - 1)
            w10, sw10 = self.issue_local_load_w_sub(mfma_idx, 0)
            x1, sx1 = self.issue_local_load_x(mfma_idx)
            mfma_idx += 1

            gl.barrier()
            load_idx = self.issue_global_loads(load_idx, USE_MASK=-1)
            c0 = self.mfma(x1, sx1, w10, sw10, c0)
            gl.amd.cdna4.async_copy.wait_group(2 * NB - 1)
            w00, sw00 = self.issue_local_load_w_sub(mfma_idx, 0)
            x0, sx0 = self.issue_local_load_x(mfma_idx)
            mfma_idx += 1

        if odd_main:
            gl.barrier()
            load_idx = self.issue_global_loads(load_idx, USE_MASK=-1)
            c0 = self.mfma(x0, sx0, w00, sw00, c0)
            gl.amd.cdna4.async_copy.wait_group(2 * NB - 1)
            w10, sw10 = self.issue_local_load_w_sub(mfma_idx, 0)
            x1, sx1 = self.issue_local_load_x(mfma_idx)
            mfma_idx += 1

            c0 = self.mfma(x1, sx1, w10, sw10, c0)
            gl.amd.cdna4.async_copy.wait_group(1)
            w00, sw00 = self.issue_local_load_w_sub(mfma_idx, 0)
            x0, sx0 = self.issue_local_load_x(mfma_idx)
            c0 = self.mfma(x0, sx0, w00, sw00, c0)
            gl.amd.cdna4.async_copy.wait_group(0)
        else:
            c0 = self.mfma(x0, sx0, w00, sw00, c0)
            gl.amd.cdna4.async_copy.wait_group(1)
            w10, sw10 = self.issue_local_load_w_sub(mfma_idx, 0)
            x1, sx1 = self.issue_local_load_x(mfma_idx)
            c0 = self.mfma(x1, sx1, w10, sw10, c0)
            gl.amd.cdna4.async_copy.wait_group(0)

        return self._finish_accumulator(c0, c1)

    @gluon.jit
    def _pipeline_full(self, loop_k):
        cfg = self.cfg
        NB: gl.constexpr = cfg.NUM_BUFFERS
        SUBTILE_N: gl.constexpr = cfg.BLOCK_N // 2

        load_idx = 0
        mfma_idx = 0

        for _ in gl.static_range(NB):
            load_idx = self.issue_global_loads(load_idx, USE_MASK=0)

        c0 = gl.zeros((cfg.BLOCK_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)
        c1 = gl.zeros((cfg.BLOCK_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)

        if cfg.K_ITERS:
            K_iters: gl.constexpr = cfg.K_ITERS
        else:
            K_iters = gl.cdiv(loop_k, cfg.BLOCK_K)
        main_iters = K_iters - NB
        gl.assume(main_iters >= 0)

        # Drain iter 0's top async-copy group first. For SliceN, the top group
        # contains X, W-top, and scales; the bottom group contains W-bottom.
        # Loading X before the bottom MFMA gives the next X tile latency slack.
        # The hot loop delays W-top until after the current bottom MFMA so the
        # pre-bottom wait does not also cover an unused W-top LDS read.
        gl.amd.cdna4.async_copy.wait_group(2 * NB - 1)
        w00, sw00 = self.issue_local_load_w_sub(mfma_idx, 0)
        x0, sx0 = self.issue_local_load_x(mfma_idx)
        gl.amd.cdna4.async_copy.wait_group(2 * NB - 2)
        w01, sw01 = self.issue_local_load_w_sub(mfma_idx, 1)
        mfma_idx += 1

        unroll_pairs = main_iters // 2
        odd_main = main_iters - unroll_pairs * 2

        for _ in range(0, unroll_pairs):
            # The future copy reuses the slot just local-loaded into VGPRs.
            # Synchronize the CTA before any producer wave overwrites it.
            c0 = self.mfma(x0, sx0, w00, sw00, c0)
            gl.barrier()
            load_idx = self.issue_global_load_top(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(2 * (NB - 1))
            x1, sx1 = self.issue_local_load_x(mfma_idx)
            c1 = self.mfma(x0, sx0, w01, sw01, c1)
            load_idx = self.issue_global_load_bot(load_idx, USE_MASK=-1)
            w10, sw10 = self.issue_local_load_w_sub(mfma_idx, 0)
            gl.amd.cdna4.async_copy.wait_group(2 * (NB - 1))
            w11, sw11 = self.issue_local_load_w_sub(mfma_idx, 1)
            mfma_idx += 1

            c0 = self.mfma(x1, sx1, w10, sw10, c0)
            gl.barrier()
            load_idx = self.issue_global_load_top(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(2 * (NB - 1))
            x0, sx0 = self.issue_local_load_x(mfma_idx)
            c1 = self.mfma(x1, sx1, w11, sw11, c1)
            load_idx = self.issue_global_load_bot(load_idx, USE_MASK=-1)
            w00, sw00 = self.issue_local_load_w_sub(mfma_idx, 0)
            gl.amd.cdna4.async_copy.wait_group(2 * (NB - 1))
            w01, sw01 = self.issue_local_load_w_sub(mfma_idx, 1)
            mfma_idx += 1

        if odd_main:
            c0 = self.mfma(x0, sx0, w00, sw00, c0)
            gl.barrier()
            load_idx = self.issue_global_load_top(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(2 * (NB - 1))
            x1, sx1 = self.issue_local_load_x(mfma_idx)
            c1 = self.mfma(x0, sx0, w01, sw01, c1)
            load_idx = self.issue_global_load_bot(load_idx, USE_MASK=-1)
            w10, sw10 = self.issue_local_load_w_sub(mfma_idx, 0)
            gl.amd.cdna4.async_copy.wait_group(2 * (NB - 1))
            w11, sw11 = self.issue_local_load_w_sub(mfma_idx, 1)
            mfma_idx += 1

            # Drain + final NB iters of MFMAs (no more async_copy).
            c0 = self.mfma(x1, sx1, w10, sw10, c0)
            gl.amd.cdna4.async_copy.wait_group(1)
            w00, sw00 = self.issue_local_load_w_sub(mfma_idx, 0)
            x0, sx0 = self.issue_local_load_x(mfma_idx)
            c1 = self.mfma(x1, sx1, w11, sw11, c1)
            gl.amd.cdna4.async_copy.wait_group(0)
            w01, sw01 = self.issue_local_load_w_sub(mfma_idx, 1)
            c0 = self.mfma(x0, sx0, w00, sw00, c0)
            c1 = self.mfma(x0, sx0, w01, sw01, c1)
        else:
            # Drain + final NB iters of MFMAs (no more async_copy).
            c0 = self.mfma(x0, sx0, w00, sw00, c0)
            gl.amd.cdna4.async_copy.wait_group(1)
            w10, sw10 = self.issue_local_load_w_sub(mfma_idx, 0)
            x1, sx1 = self.issue_local_load_x(mfma_idx)
            c1 = self.mfma(x0, sx0, w01, sw01, c1)
            gl.amd.cdna4.async_copy.wait_group(0)
            w11, sw11 = self.issue_local_load_w_sub(mfma_idx, 1)
            c0 = self.mfma(x1, sx1, w10, sw10, c0)
            c1 = self.mfma(x1, sx1, w11, sw11, c1)

        return self._finish_accumulator(c0, c1)

    @gluon.jit
    def pipeline(self, loop_k):
        cfg = self.cfg
        NB: gl.constexpr = cfg.NUM_BUFFERS
        gl.static_assert(
            (cfg.NUM_SUBTILES[0] == 1)
            and (cfg.NUM_SUBTILES[1] == 2)
            and (cfg.NUM_SUBTILES[2] == 1),
            "MoESliceNProgram requires NUM_SUBTILES=(1,2,1)",
        )
        gl.static_assert(
            NB == 2,
            "current SliceN local-prefetch pipeline requires exactly two LDS buffers",
        )

        if self.bottom_valid:
            return self._pipeline_full(loop_k)
        return self._pipeline_top_only(loop_k)


@gluon.jit
def _make_moe_x_desc(
    cfg,
    x_ptr,
    rows_m_x,
    offs_xk,
    stride_xm,
    stride_xk,
    x_mask_nonk,
    k_limit_x,
    BLOCK_K_X: gl.constexpr,
):
    return AsyncCopyDescriptor.initialize(
        cfg,
        0,
        BLOCK_K_X,
        x_ptr,
        rows_m_x,
        offs_xk,
        stride_xm,
        stride_xk,
        x_mask_nonk,
        k_limit_x,
    )


@gluon.jit
def _make_slice_mn_x_descs(
    cfg,
    x_ptr,
    gather_idx_ptr,
    stride_xm,
    stride_xk,
    M_X,
    off_m,
    m_limit,
    k_limit_x,
    BLOCK_M: gl.constexpr,
    BLOCK_K_X: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    HAS_GATHER: gl.constexpr,
    X_ELEM_BITS: gl.constexpr,
):
    SUB_BM_MN: gl.constexpr = BLOCK_M // 2

    LOAD_X_SUB_LAYOUT_MN: gl.constexpr = _load_layout(
        BLOCK_K_X, SUB_BM_MN, NUM_WARPS, [1, 0], X_ELEM_BITS
    )
    offs_xm_sub_mn = gl.arange(
        0, SUB_BM_MN, layout=gl.SliceLayout(1, LOAD_X_SUB_LAYOUT_MN)
    )
    offs_xk_sub_mn = gl.arange(
        0, BLOCK_K_X, layout=gl.SliceLayout(0, LOAD_X_SUB_LAYOUT_MN)
    )
    rows_m_top = off_m + offs_xm_sub_mn
    rows_m_bot = off_m + SUB_BM_MN + offs_xm_sub_mn
    pre_gather_mask_top = rows_m_top < m_limit
    pre_gather_mask_bot = rows_m_bot < m_limit
    if HAS_GATHER:
        rows_m_top_safe = gl.where(
            pre_gather_mask_top, rows_m_top, gl.zeros_like(rows_m_top)
        )
        rows_m_bot_safe = gl.where(
            pre_gather_mask_bot, rows_m_bot, gl.zeros_like(rows_m_bot)
        )
        rows_m_top = gl.load(
            gather_idx_ptr + rows_m_top_safe,
            mask=pre_gather_mask_top,
            other=0,
        ).to(gl.int32)
        rows_m_bot = gl.load(
            gather_idx_ptr + rows_m_bot_safe,
            mask=pre_gather_mask_bot,
            other=0,
        ).to(gl.int32)
        mask_m_top = pre_gather_mask_top & (rows_m_top < M_X)
        mask_m_bot = pre_gather_mask_bot & (rows_m_bot < M_X)
    else:
        rows_m_top = gl.where(
            pre_gather_mask_top, rows_m_top, gl.zeros_like(rows_m_top)
        )
        rows_m_bot = gl.where(
            pre_gather_mask_bot, rows_m_bot, gl.zeros_like(rows_m_bot)
        )
        mask_m_top = pre_gather_mask_top
        mask_m_bot = pre_gather_mask_bot
    x_desc_top_mn = AsyncCopyDescriptor.initialize(
        cfg,
        0,
        BLOCK_K_X,
        x_ptr,
        rows_m_top,
        offs_xk_sub_mn,
        stride_xm,
        stride_xk,
        mask_m_top[:, None],
        k_limit_x,
    )
    x_desc_bot_mn = AsyncCopyDescriptor.initialize(
        cfg,
        0,
        BLOCK_K_X,
        x_ptr,
        rows_m_bot,
        offs_xk_sub_mn,
        stride_xm,
        stride_xk,
        mask_m_bot[:, None],
        k_limit_x,
    )
    return x_desc_top_mn, x_desc_bot_mn


@gluon.jit
def _make_nonpreshuffled_w_desc(
    cfg,
    w_ptr,
    rows_n,
    offs_wk,
    stride_wn,
    stride_wk,
    mask_n,
    k_limit_w,
    w_base_offset,
    OP_IDX: gl.constexpr,
    BLOCK_K_W: gl.constexpr,
    W_CACHE_MODIFIER: gl.constexpr,
):
    return AsyncCopyDescriptor.initialize(
        cfg,
        OP_IDX,
        BLOCK_K_W,
        w_ptr,
        rows_n,
        offs_wk,
        stride_wn,
        stride_wk,
        mask_n,
        k_limit_w,
        base_offset=w_base_offset,
        cache_modifier=W_CACHE_MODIFIER,
    )


@gluon.jit
def _make_nonpreshuffled_w_half_descs(
    cfg,
    w_ptr,
    stride_wn,
    stride_wk,
    N,
    off_n,
    k_limit_w,
    w_base_offset,
    SUB_BN: gl.constexpr,
    BLOCK_K_W: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    W_TRANSPOSE: gl.constexpr,
    W_ELEM_BITS: gl.constexpr,
    W_CACHE_MODIFIER: gl.constexpr,
):
    if W_TRANSPOSE:
        LOAD_W_LAYOUT: gl.constexpr = _load_layout(
            BLOCK_K_W, SUB_BN, NUM_WARPS, [1, 0], W_ELEM_BITS
        )
        offs_wn = gl.arange(0, SUB_BN, layout=gl.SliceLayout(1, LOAD_W_LAYOUT))
        offs_wk = gl.arange(0, BLOCK_K_W, layout=gl.SliceLayout(0, LOAD_W_LAYOUT))
        mask_n_first = (off_n + offs_wn) < N
        mask_n_second = (off_n + SUB_BN + offs_wn) < N
        w_desc_first = _make_nonpreshuffled_w_desc(
            cfg,
            w_ptr,
            off_n + offs_wn,
            offs_wk,
            stride_wn,
            stride_wk,
            mask_n_first[:, None],
            k_limit_w,
            w_base_offset,
            0,
            BLOCK_K_W,
            W_CACHE_MODIFIER,
        )
        w_desc_second = _make_nonpreshuffled_w_desc(
            cfg,
            w_ptr,
            off_n + SUB_BN + offs_wn,
            offs_wk,
            stride_wn,
            stride_wk,
            mask_n_second[:, None],
            k_limit_w,
            w_base_offset,
            0,
            BLOCK_K_W,
            W_CACHE_MODIFIER,
        )
    else:
        LOAD_W_LAYOUT: gl.constexpr = _load_layout(
            SUB_BN, BLOCK_K_W, NUM_WARPS, [1, 0], W_ELEM_BITS
        )
        offs_wn = gl.arange(0, SUB_BN, layout=gl.SliceLayout(0, LOAD_W_LAYOUT))
        offs_wk = gl.arange(0, BLOCK_K_W, layout=gl.SliceLayout(1, LOAD_W_LAYOUT))
        mask_n_first = (off_n + offs_wn) < N
        mask_n_second = (off_n + SUB_BN + offs_wn) < N
        w_desc_first = _make_nonpreshuffled_w_desc(
            cfg,
            w_ptr,
            off_n + offs_wn,
            offs_wk,
            stride_wn,
            stride_wk,
            mask_n_first[None, :],
            k_limit_w,
            w_base_offset,
            1,
            BLOCK_K_W,
            W_CACHE_MODIFIER,
        )
        w_desc_second = _make_nonpreshuffled_w_desc(
            cfg,
            w_ptr,
            off_n + SUB_BN + offs_wn,
            offs_wk,
            stride_wn,
            stride_wk,
            mask_n_second[None, :],
            k_limit_w,
            w_base_offset,
            1,
            BLOCK_K_W,
            W_CACHE_MODIFIER,
        )
    return w_desc_first, w_desc_second


@gluon.jit
def _make_nonpreshuffled_w_full_desc(
    cfg,
    w_ptr,
    stride_wn,
    stride_wk,
    N,
    off_n,
    k_limit_w,
    w_base_offset,
    BLOCK_N: gl.constexpr,
    BLOCK_K_W: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    W_TRANSPOSE: gl.constexpr,
    W_ELEM_BITS: gl.constexpr,
    W_CACHE_MODIFIER: gl.constexpr,
):
    if W_TRANSPOSE:
        LOAD_W_LAYOUT: gl.constexpr = _load_layout(
            BLOCK_K_W, BLOCK_N, NUM_WARPS, [1, 0], W_ELEM_BITS
        )
        offs_wn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, LOAD_W_LAYOUT))
        offs_wk = gl.arange(0, BLOCK_K_W, layout=gl.SliceLayout(0, LOAD_W_LAYOUT))
        mask_n = (off_n + offs_wn) < N
        w_desc = _make_nonpreshuffled_w_desc(
            cfg,
            w_ptr,
            off_n + offs_wn,
            offs_wk,
            stride_wn,
            stride_wk,
            mask_n[:, None],
            k_limit_w,
            w_base_offset,
            0,
            BLOCK_K_W,
            W_CACHE_MODIFIER,
        )
    else:
        LOAD_W_LAYOUT: gl.constexpr = _load_layout(
            BLOCK_N, BLOCK_K_W, NUM_WARPS, [1, 0], W_ELEM_BITS
        )
        offs_wn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, LOAD_W_LAYOUT))
        offs_wk = gl.arange(0, BLOCK_K_W, layout=gl.SliceLayout(1, LOAD_W_LAYOUT))
        mask_n = (off_n + offs_wn) < N
        w_desc = _make_nonpreshuffled_w_desc(
            cfg,
            w_ptr,
            off_n + offs_wn,
            offs_wk,
            stride_wn,
            stride_wk,
            mask_n[None, :],
            k_limit_w,
            w_base_offset,
            1,
            BLOCK_K_W,
            W_CACHE_MODIFIER,
        )
    return w_desc


@gluon.constexpr_function
def _preshuffled_w_read_layout(
    block_n_units: int,
    block_k_w: int,
    scale_via_lds: bool,
):
    if scale_via_lds:
        return gl.DistributedLinearLayout(
            reg_bases=[
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 1024],
                [1, 0],
                [4, 0],
            ],
            lane_bases=[
                [0, 16],
                [0, 32],
                [0, 64],
                [0, 128],
                [0, 256],
                [0, 512],
            ],
            warp_bases=[[2, 0], [0, 0]],
            block_bases=[],
            shape=[block_n_units, block_k_w * 16],
        )
    return gl.DistributedLinearLayout(
        reg_bases=[
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 1024],
            [2, 0],
            [4, 0],
        ],
        lane_bases=[
            [0, 16],
            [0, 32],
            [0, 64],
            [0, 128],
            [0, 256],
            [0, 512],
        ],
        warp_bases=[[1, 0], [0, 0]],
        block_bases=[],
        shape=[block_n_units, block_k_w * 16],
    )


@gluon.constexpr_function
def _preshuffled_w_copy_layout(
    block_n_units: int,
    block_k_w: int,
    scale_via_lds: bool,
    use_all_waves_for_copy: bool,
):
    if not use_all_waves_for_copy:
        return _preshuffled_w_read_layout(block_n_units, block_k_w, scale_via_lds)
    if scale_via_lds:
        return gl.DistributedLinearLayout(
            reg_bases=[
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 1024],
                [1, 0],
            ],
            lane_bases=[
                [0, 16],
                [0, 32],
                [0, 64],
                [0, 128],
                [0, 256],
                [0, 512],
            ],
            warp_bases=[[2, 0], [4, 0]],
            block_bases=[],
            shape=[block_n_units, block_k_w * 16],
        )
    return gl.DistributedLinearLayout(
        reg_bases=[
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 1024],
            [2, 0],
        ],
        lane_bases=[
            [0, 16],
            [0, 32],
            [0, 64],
            [0, 128],
            [0, 256],
            [0, 512],
        ],
        warp_bases=[[1, 0], [4, 0]],
        block_bases=[],
        shape=[block_n_units, block_k_w * 16],
    )


@gluon.jit
def _make_preshuffled_w_x_desc(
    cfg,
    x_ptr,
    rows_m_x,
    offs_xk,
    stride_xm,
    stride_xk,
    M_X,
    k_limit_x,
    BLOCK_K_X: gl.constexpr,
    HAS_GATHER: gl.constexpr,
):
    if HAS_GATHER:
        # Invalid expert-tail rows were already clamped to gather row 0.
        # Keep the global token bound check, but avoid carrying the
        # per-expert tail predicate through every X async-copy element.
        x_mask_nonk = (rows_m_x < M_X)[:, None]
    else:
        x_mask_nonk = gl.to_tensor(True)
    return _make_moe_x_desc(
        cfg,
        x_ptr,
        rows_m_x,
        offs_xk,
        stride_xm,
        stride_xk,
        x_mask_nonk,
        k_limit_x,
        BLOCK_K_X,
    )


@gluon.jit
def _make_preshuffled_w_slice_offsets(
    w_base_offset,
    pid_n,
    N,
    LOAD_W_COPY_LAYOUT: gl.constexpr,
    N_LIMIT: gl.constexpr,
    SUB_BN: gl.constexpr,
    BLOCK_K_W: gl.constexpr,
):
    offs_wn_h = gl.arange(0, SUB_BN // 16, layout=gl.SliceLayout(1, LOAD_W_COPY_LAYOUT))
    offs_wk_h = gl.arange(
        0, BLOCK_K_W * 16, layout=gl.SliceLayout(0, LOAD_W_COPY_LAYOUT)
    )
    offsets_h = gl.expand_dims(offs_wk_h, 0) + gl.expand_dims(offs_wn_h, 1) * (
        BLOCK_K_W * 16
    )
    TILE_BYTES_HALF: gl.constexpr = 128 * 128
    if N_LIMIT:
        n_block_count: gl.constexpr = (N_LIMIT + 127) // 128
        w_k_stride = gl.to_tensor(N_LIMIT)
    else:
        n_block_count = (N + 127) // 128
        w_k_stride = gl.to_tensor(N)
    bottom_valid = (2 * pid_n + 1) < n_block_count
    base_off_top = w_base_offset + 2 * pid_n * TILE_BYTES_HALF
    base_off_bot = base_off_top + TILE_BYTES_HALF
    return offsets_h, base_off_top, base_off_bot, w_k_stride, bottom_valid


@gluon.jit
def _make_preshuffled_w_full_offsets(
    w_base_offset,
    pid_n,
    LOAD_W_COPY_LAYOUT: gl.constexpr,
    BLOCK_N_LAYOUT: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K_W: gl.constexpr,
):
    offs_wn = gl.arange(
        0, BLOCK_N_LAYOUT // 16, layout=gl.SliceLayout(1, LOAD_W_COPY_LAYOUT)
    )
    offs_wk = gl.arange(0, BLOCK_K_W * 16, layout=gl.SliceLayout(0, LOAD_W_COPY_LAYOUT))
    offsets_b = gl.expand_dims(offs_wk, 0) + gl.expand_dims(offs_wn, 1) * (
        BLOCK_K_W * 16
    )
    TILE_BYTES: gl.constexpr = BLOCK_K_W * BLOCK_N
    base_off_b = w_base_offset + pid_n * TILE_BYTES
    return offsets_b, base_off_b


@gluon.jit
def _run_moe_tile_w_via_vgpr(
    cfg,
    x_ptr,
    w_ptr,
    x_scale_desc,
    w_scale_desc,
    gather_idx_ptr,
    stride_xm,
    stride_xk,
    M_X,
    N,
    K,
    off_m,
    m_limit,
    rows_m_x,
    offs_xk,
    k_limit_x,
    k_limit_w,
    w_base_offset,
    pid_n,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K_X: gl.constexpr,
    BLOCK_K_W: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    HAS_GATHER: gl.constexpr,
    USE_SLICE_MN: gl.constexpr,
    USE_SLICE_N: gl.constexpr,
    USE_WARP_PIPELINE: gl.constexpr,
    N_LIMIT: gl.constexpr,
    W_CACHE_MODIFIER: gl.constexpr,
):
    gl.static_assert(
        cfg.W_PRESHUFFLED,
        "W_VIA_VGPR consumes the preshuffled Gluon-dot W layout.",
    )
    gl.static_assert(
        BLOCK_K_W == 128 and NUM_WARPS == 4 and not USE_SLICE_MN,
        "W_VIA_VGPR layout bases assume BLOCK_K_W=128, NUM_WARPS=4, "
        "and USE_SLICE_MN=False. Re-derive bases for other shapes.",
    )

    x_desc = _make_preshuffled_w_x_desc(
        cfg,
        x_ptr,
        rows_m_x,
        offs_xk,
        stride_xm,
        stride_xk,
        M_X,
        k_limit_x,
        BLOCK_K_X,
        HAS_GATHER,
    )

    if USE_SLICE_N:
        SUB_BN: gl.constexpr = BLOCK_N // 2
        gl.static_assert(
            SUB_BN == 128 and BLOCK_K_W == 128 and NUM_WARPS == 4,
            "USE_SLICE_N + W_VIA_VGPR requires SUB_BN=BLOCK_K_W=128 "
            "and NUM_WARPS=4; the half-tile LOAD_W_LAYOUT bases assume "
            "this shape (re-derive otherwise).",
        )
        LOAD_W_HALF_COPY_LAYOUT: gl.constexpr = _preshuffled_w_copy_layout(
            SUB_BN // 16, BLOCK_K_W, cfg.W_SCALE_VIA_LDS, False
        )
        (
            offsets_h,
            base_off_top,
            base_off_bot,
            w_k_stride,
            bottom_valid,
        ) = _make_preshuffled_w_slice_offsets(
            w_base_offset,
            pid_n,
            N,
            LOAD_W_HALF_COPY_LAYOUT,
            N_LIMIT,
            SUB_BN,
            BLOCK_K_W,
        )
        w_desc_top = WVgprDescriptor(
            cfg,
            BLOCK_K_W,
            w_ptr,
            w_k_stride,
            offsets_h + base_off_top,
            pred=gl.to_tensor(True),
            LOAD_BN=SUB_BN,
        )
        w_desc_bot = WVgprDescriptor(
            cfg,
            BLOCK_K_W,
            w_ptr,
            w_k_stride,
            offsets_h + base_off_bot,
            pred=bottom_valid,
            LOAD_BN=SUB_BN,
        )
        pgm = MoESliceNProgram.initialize(
            cfg,
            x_desc,
            w_desc_top,
            w_desc_bot,
            x_scale_desc,
            w_scale_desc,
            bottom_valid,
        )
        return pgm.pipeline(K)
    else:
        gl.static_assert(
            BLOCK_N == 128,
            "W_VIA_VGPR full-tile layout bases assume BLOCK_N=128. "
            "Re-derive bases for other shapes.",
        )
        BLOCK_N_LAYOUT: gl.constexpr = BLOCK_N
        LOAD_W_COPY_LAYOUT: gl.constexpr = _preshuffled_w_copy_layout(
            BLOCK_N_LAYOUT // 16, BLOCK_K_W, cfg.W_SCALE_VIA_LDS, False
        )
        offsets_b_vgpr, base_off_b_vgpr = _make_preshuffled_w_full_offsets(
            w_base_offset,
            pid_n,
            LOAD_W_COPY_LAYOUT,
            BLOCK_N_LAYOUT,
            BLOCK_N,
            BLOCK_K_W,
        )
        w_desc = WVgprDescriptor(
            cfg,
            BLOCK_K_W,
            w_ptr,
            gl.to_tensor(N),  # K-iter advance step: idx * BK_W * N
            offsets_b_vgpr + base_off_b_vgpr,
            pred=gl.to_tensor(True),  # full-tile path: always in-bounds
            LOAD_BN=BLOCK_N_LAYOUT,
        )
        pgm = MoEPipelinedProgram.initialize(
            cfg, x_desc, w_desc, x_scale_desc, w_scale_desc
        )
        if USE_WARP_PIPELINE:
            return pgm.warp_pipeline(K)
        return pgm.pipeline(K)


@gluon.jit
def _run_moe_tile_preshuffled_lds_w(
    cfg,
    x_ptr,
    w_ptr,
    x_scale_desc,
    w_scale_desc,
    gather_idx_ptr,
    stride_xm,
    stride_xk,
    M_X,
    N,
    K,
    off_m,
    m_limit,
    rows_m_x,
    offs_xk,
    k_limit_x,
    k_limit_w,
    w_base_offset,
    pid_n,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K_X: gl.constexpr,
    BLOCK_K_W: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    HAS_GATHER: gl.constexpr,
    USE_SLICE_MN: gl.constexpr,
    USE_SLICE_N: gl.constexpr,
    USE_WARP_PIPELINE: gl.constexpr,
    N_LIMIT: gl.constexpr,
    W_CACHE_MODIFIER: gl.constexpr,
):
    gl.static_assert(
        BLOCK_K_W == 128
        and (BLOCK_N == 128 or USE_SLICE_N)
        and NUM_WARPS == 4
        and not USE_SLICE_MN,
        "preshuffled W layout bases assume BLOCK_K_W=128, "
        "BLOCK_N=128 (or USE_SLICE_N=True for half-tile path), "
        "NUM_WARPS=4, and USE_SLICE_MN=False. Re-derive bases for "
        "other shapes.",
    )

    x_desc = _make_preshuffled_w_x_desc(
        cfg,
        x_ptr,
        rows_m_x,
        offs_xk,
        stride_xm,
        stride_xk,
        M_X,
        k_limit_x,
        BLOCK_K_X,
        HAS_GATHER,
    )

    if USE_SLICE_N:
        SUB_BN: gl.constexpr = BLOCK_N // 2
        gl.static_assert(
            SUB_BN == 128 and BLOCK_K_W == 128 and NUM_WARPS == 4,
            "USE_SLICE_N + preshuffled W requires SUB_BN=BLOCK_K_W=128 "
            "and NUM_WARPS=4; the half-tile LOAD_W_LAYOUT bases assume "
            "this shape (re-derive otherwise).",
        )
        LOAD_W_HALF_LAYOUT: gl.constexpr = _preshuffled_w_read_layout(
            SUB_BN // 16, BLOCK_K_W, cfg.W_SCALE_VIA_LDS
        )
        LOAD_W_HALF_COPY_LAYOUT: gl.constexpr = _preshuffled_w_copy_layout(
            SUB_BN // 16, BLOCK_K_W, cfg.W_SCALE_VIA_LDS, True
        )
        (
            offsets_h,
            base_off_top,
            base_off_bot,
            w_k_stride,
            bottom_valid,
        ) = _make_preshuffled_w_slice_offsets(
            w_base_offset,
            pid_n,
            N,
            LOAD_W_HALF_COPY_LAYOUT,
            N_LIMIT,
            SUB_BN,
            BLOCK_K_W,
        )
        w_desc_top = WPreshuffledLdsDescriptor(
            cfg,
            BLOCK_K_W,
            w_ptr,
            w_ptr.dtype.element_ty,
            w_k_stride,
            offsets_h + base_off_top,
            pred=gl.to_tensor(True),
            load_layout=LOAD_W_HALF_LAYOUT,
            LOAD_BN=SUB_BN,
            cache_modifier=W_CACHE_MODIFIER,
        )
        w_desc_bot = WPreshuffledLdsDescriptor(
            cfg,
            BLOCK_K_W,
            w_ptr,
            w_ptr.dtype.element_ty,
            w_k_stride,
            offsets_h + base_off_bot,
            pred=bottom_valid,
            load_layout=LOAD_W_HALF_LAYOUT,
            LOAD_BN=SUB_BN,
            cache_modifier=W_CACHE_MODIFIER,
        )
        pgm = MoESliceNProgram.initialize(
            cfg,
            x_desc,
            w_desc_top,
            w_desc_bot,
            x_scale_desc,
            w_scale_desc,
            bottom_valid,
        )
        return pgm.pipeline(K)

    # Gluon still type-checks the code below when USE_SLICE_N returns above.
    # Keep the original half-tile layout in that specialization so the
    # preshuffled copy/read layouts remain valid during compilation.
    BLOCK_N_LAYOUT: gl.constexpr = (BLOCK_N // 2) if USE_SLICE_N else BLOCK_N
    LOAD_W_LAYOUT: gl.constexpr = _preshuffled_w_read_layout(
        BLOCK_N_LAYOUT // 16, BLOCK_K_W, cfg.W_SCALE_VIA_LDS
    )
    LOAD_W_COPY_LAYOUT: gl.constexpr = _preshuffled_w_copy_layout(
        BLOCK_N_LAYOUT // 16, BLOCK_K_W, cfg.W_SCALE_VIA_LDS, True
    )
    offsets_b_vgpr, base_off_b_vgpr = _make_preshuffled_w_full_offsets(
        w_base_offset,
        pid_n,
        LOAD_W_COPY_LAYOUT,
        BLOCK_N_LAYOUT,
        BLOCK_N,
        BLOCK_K_W,
    )
    w_desc = WPreshuffledLdsDescriptor(
        cfg,
        BLOCK_K_W,
        w_ptr,
        w_ptr.dtype.element_ty,
        gl.to_tensor(N),
        offsets_b_vgpr + base_off_b_vgpr,
        pred=gl.to_tensor(True),
        load_layout=LOAD_W_LAYOUT,
        cache_modifier=W_CACHE_MODIFIER,
    )
    pgm = MoEPipelinedProgram.initialize(
        cfg, x_desc, w_desc, x_scale_desc, w_scale_desc
    )
    if USE_WARP_PIPELINE:
        return pgm.warp_pipeline(K)
    return pgm.pipeline(K)


@gluon.jit
def _run_moe_tile_transposed_w(
    cfg,
    x_ptr,
    w_ptr,
    x_scale_desc,
    w_scale_desc,
    gather_idx_ptr,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    M_X,
    N,
    K,
    off_m,
    off_n,
    m_limit,
    rows_m_x,
    mask_m_x,
    offs_xk,
    k_limit_x,
    k_limit_w,
    w_base_offset,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K_X: gl.constexpr,
    BLOCK_K_W: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    HAS_GATHER: gl.constexpr,
    USE_SLICE_MN: gl.constexpr,
    USE_SLICE_N: gl.constexpr,
    USE_WARP_PIPELINE: gl.constexpr,
    X_ELEM_BITS: gl.constexpr,
    W_ELEM_BITS: gl.constexpr,
    W_CACHE_MODIFIER: gl.constexpr,
):
    x_desc = _make_moe_x_desc(
        cfg,
        x_ptr,
        rows_m_x,
        offs_xk,
        stride_xm,
        stride_xk,
        mask_m_x[:, None],
        k_limit_x,
        BLOCK_K_X,
    )

    if USE_SLICE_MN:
        SUB_BN_MN: gl.constexpr = BLOCK_N // 2
        x_desc_top_mn, x_desc_bot_mn = _make_slice_mn_x_descs(
            cfg,
            x_ptr,
            gather_idx_ptr,
            stride_xm,
            stride_xk,
            M_X,
            off_m,
            m_limit,
            k_limit_x,
            BLOCK_M,
            BLOCK_K_X,
            NUM_WARPS,
            HAS_GATHER,
            X_ELEM_BITS,
        )

        w_desc_left_mn, w_desc_right_mn = _make_nonpreshuffled_w_half_descs(
            cfg,
            w_ptr,
            stride_wn,
            stride_wk,
            N,
            off_n,
            k_limit_w,
            w_base_offset,
            SUB_BN_MN,
            BLOCK_K_W,
            NUM_WARPS,
            True,
            W_ELEM_BITS,
            W_CACHE_MODIFIER,
        )
        slice_mn_pgm = MoESliceMNProgram.initialize(
            cfg,
            x_desc_top_mn,
            x_desc_bot_mn,
            w_desc_left_mn,
            w_desc_right_mn,
            x_scale_desc,
            w_scale_desc,
        )
        return slice_mn_pgm.pipeline(K)

    if USE_SLICE_N:
        SUB_BN: gl.constexpr = BLOCK_N // 2
        bottom_valid = gl.to_tensor(True)
        w_desc_top, w_desc_bot = _make_nonpreshuffled_w_half_descs(
            cfg,
            w_ptr,
            stride_wn,
            stride_wk,
            N,
            off_n,
            k_limit_w,
            w_base_offset,
            SUB_BN,
            BLOCK_K_W,
            NUM_WARPS,
            True,
            W_ELEM_BITS,
            W_CACHE_MODIFIER,
        )
        pgm = MoESliceNProgram.initialize(
            cfg,
            x_desc,
            w_desc_top,
            w_desc_bot,
            x_scale_desc,
            w_scale_desc,
            bottom_valid,
        )
        return pgm.pipeline(K)

    w_desc = _make_nonpreshuffled_w_full_desc(
        cfg,
        w_ptr,
        stride_wn,
        stride_wk,
        N,
        off_n,
        k_limit_w,
        w_base_offset,
        BLOCK_N,
        BLOCK_K_W,
        NUM_WARPS,
        True,
        W_ELEM_BITS,
        W_CACHE_MODIFIER,
    )
    pgm = MoEPipelinedProgram.initialize(
        cfg, x_desc, w_desc, x_scale_desc, w_scale_desc
    )
    if USE_WARP_PIPELINE:
        return pgm.warp_pipeline(K)
    return pgm.pipeline(K)


@gluon.jit
def _run_moe_tile_ncontig_w(
    cfg,
    x_ptr,
    w_ptr,
    x_scale_desc,
    w_scale_desc,
    gather_idx_ptr,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    M_X,
    N,
    K,
    off_m,
    off_n,
    m_limit,
    rows_m_x,
    mask_m_x,
    offs_xk,
    k_limit_x,
    k_limit_w,
    w_base_offset,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K_X: gl.constexpr,
    BLOCK_K_W: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    HAS_GATHER: gl.constexpr,
    USE_SLICE_MN: gl.constexpr,
    USE_SLICE_N: gl.constexpr,
    USE_WARP_PIPELINE: gl.constexpr,
    X_ELEM_BITS: gl.constexpr,
    W_ELEM_BITS: gl.constexpr,
    W_CACHE_MODIFIER: gl.constexpr,
):
    x_desc = _make_moe_x_desc(
        cfg,
        x_ptr,
        rows_m_x,
        offs_xk,
        stride_xm,
        stride_xk,
        mask_m_x[:, None],
        k_limit_x,
        BLOCK_K_X,
    )

    if USE_SLICE_MN:
        SUB_BN_MN: gl.constexpr = BLOCK_N // 2
        x_desc_top_mn, x_desc_bot_mn = _make_slice_mn_x_descs(
            cfg,
            x_ptr,
            gather_idx_ptr,
            stride_xm,
            stride_xk,
            M_X,
            off_m,
            m_limit,
            k_limit_x,
            BLOCK_M,
            BLOCK_K_X,
            NUM_WARPS,
            HAS_GATHER,
            X_ELEM_BITS,
        )

        w_desc_left_mn, w_desc_right_mn = _make_nonpreshuffled_w_half_descs(
            cfg,
            w_ptr,
            stride_wn,
            stride_wk,
            N,
            off_n,
            k_limit_w,
            w_base_offset,
            SUB_BN_MN,
            BLOCK_K_W,
            NUM_WARPS,
            False,
            W_ELEM_BITS,
            W_CACHE_MODIFIER,
        )
        slice_mn_pgm = MoESliceMNProgram.initialize(
            cfg,
            x_desc_top_mn,
            x_desc_bot_mn,
            w_desc_left_mn,
            w_desc_right_mn,
            x_scale_desc,
            w_scale_desc,
        )
        return slice_mn_pgm.pipeline(K)

    if USE_SLICE_N:
        SUB_BN: gl.constexpr = BLOCK_N // 2
        bottom_valid = gl.to_tensor(True)
        w_desc_top, w_desc_bot = _make_nonpreshuffled_w_half_descs(
            cfg,
            w_ptr,
            stride_wn,
            stride_wk,
            N,
            off_n,
            k_limit_w,
            w_base_offset,
            SUB_BN,
            BLOCK_K_W,
            NUM_WARPS,
            False,
            W_ELEM_BITS,
            W_CACHE_MODIFIER,
        )
        pgm = MoESliceNProgram.initialize(
            cfg,
            x_desc,
            w_desc_top,
            w_desc_bot,
            x_scale_desc,
            w_scale_desc,
            bottom_valid,
        )
        return pgm.pipeline(K)

    w_desc = _make_nonpreshuffled_w_full_desc(
        cfg,
        w_ptr,
        stride_wn,
        stride_wk,
        N,
        off_n,
        k_limit_w,
        w_base_offset,
        BLOCK_N,
        BLOCK_K_W,
        NUM_WARPS,
        False,
        W_ELEM_BITS,
        W_CACHE_MODIFIER,
    )
    pgm = MoEPipelinedProgram.initialize(
        cfg, x_desc, w_desc, x_scale_desc, w_scale_desc
    )
    if USE_WARP_PIPELINE:
        return pgm.warp_pipeline(K)
    return pgm.pipeline(K)


@gluon.jit
def _moe_masked_store(out, y_ptr, y_offs, mask, USE_BUFFER_STORE: gl.constexpr):
    """Shared masked store for the prefill and medium-decode epilogues.

    ``USE_BUFFER_STORE`` selects the AMD ``buffer_store`` intrinsic (the medium
    dispatch path's fast store) vs generic ``gl.store`` (prefill + medium
    combine). The caller owns all address/mask computation so each path keeps
    its own addressing; only the final emit is shared.
    """
    if USE_BUFFER_STORE:
        gl.amd.cdna4.buffer_store(out, y_ptr, y_offs.to(gl.int32), mask=mask)
    else:
        gl.store(y_ptr + y_offs, out, mask=mask)


@gluon.jit
def _pipelined_moe_tile_compute(
    # Tensors --------------------------------------------------------
    x_ptr,
    w_ptr,
    x_scale_ptr,
    w_scale_ptr,
    bias_ptr,
    y_ptr,
    gather_idx_ptr,
    scatter_idx_ptr,
    gate_scal_ptr,
    slice_offs_ptr,
    slice_sizes_ptr,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_xsm,
    stride_xsk,
    stride_wse,
    stride_wsn,
    stride_wsk,
    stride_yn,
    stride_ym,
    stride_be,
    stride_bn,
    M,
    M_X,
    N,
    K,
    x_global_scale_ptr,
    out_quant_scale_ptr,
    compact_idx,
    block_in_expert,
    pid_n,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCKS_PER_EXPERT: gl.constexpr,
    X_FORMAT: gl.constexpr,
    W_FORMAT: gl.constexpr,
    UPCAST_INDICES: gl.constexpr,
    HAS_X_BLOCK_SCALE: gl.constexpr,
    HAS_W_BLOCK_SCALE: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    HAS_GATHER: gl.constexpr,
    HAS_SCATTER: gl.constexpr,
    DO_SWIGLU: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    OUT_BLOCK_N: gl.constexpr,
    APPLY_GATE_SCAL: gl.constexpr,
    HAS_RAGGED_OFFS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    SCALE_LOAD_MODE: gl.constexpr,
    W_TRANSPOSE: gl.constexpr = False,
    NUM_SUBTILES: gl.constexpr = (1, 1, 1),
    EVEN_K: gl.constexpr = True,
    K_ITERS: gl.constexpr = 0,
    N_CONST: gl.constexpr = 0,
    Y_N_CONST: gl.constexpr = 0,
    APPLY_X_GLOBAL_SCALE: gl.constexpr = True,
    USE_WARP_PIPELINE: gl.constexpr = False,
    USE_SLICE_MN: gl.constexpr = False,
    USE_SLICE_N: gl.constexpr = False,
    HAS_FP8_QUANT_OUT: gl.constexpr = False,
    W_PRESHUFFLED: gl.constexpr = False,
    W_VIA_VGPR: gl.constexpr = False,
    W_PREFETCH: gl.constexpr = True,
    W_CACHE_CG: gl.constexpr = False,
    X_SCALE_VIA_LDS: gl.constexpr = False,
    W_SCALE_VIA_LDS: gl.constexpr = False,
    USE_NARROW_N_STORE_LAYOUT: gl.constexpr = False,
):
    expert_id = compact_idx

    USE_GATHER: gl.constexpr = HAS_GATHER

    BLOCK_SCALE_FACTOR: gl.constexpr = 32
    BLOCK_K_SCALE: gl.constexpr = BLOCK_K // BLOCK_SCALE_FACTOR

    if HAS_RAGGED_OFFS:
        # X experts are packed back-to-back at slice_offs[expert_id];
        # boundary is slice_sizes[expert_id] (NOT padded to BLOCK_M).
        m_base = gl.load(slice_offs_ptr + expert_id).to(gl.int32)
        m_size = gl.load(slice_sizes_ptr + expert_id).to(gl.int32)
        off_m = m_base + block_in_expert * BLOCK_M
        m_limit = m_base + m_size
    else:
        off_m = compact_idx * BLOCKS_PER_EXPERT * BLOCK_M + block_in_expert * BLOCK_M
        m_limit = M
    off_n = pid_n * BLOCK_N
    w_base_offset = expert_id * stride_we
    ws_base_offset = expert_id * stride_wse
    N_LIMIT: gl.constexpr = N_CONST if N_CONST else 0

    STORE: gl.constexpr = _store_layout(
        NUM_WARPS,
        block_m=BLOCK_M,
        w_via_vgpr=W_VIA_VGPR or W_PRESHUFFLED,
        use_narrow_n_layout=USE_NARROW_N_STORE_LAYOUT,
    )

    index_type: gl.constexpr = gl.int64 if UPCAST_INDICES else gl.int32
    cfg = MoEConfig(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        X_FORMAT,
        W_FORMAT,
        BLOCK_SCALE_FACTOR,
        NUM_BUFFERS,
        W_TRANSPOSE,
        HAS_X_BLOCK_SCALE,
        HAS_W_BLOCK_SCALE,
        SCALE_LOAD_MODE,
        index_type,
        NUM_SUBTILES,
        EVEN_K,
        K_ITERS,
        USE_GATHER,
        NUM_WARPS,
        W_PRESHUFFLED=W_PRESHUFFLED,
        W_VIA_VGPR=W_VIA_VGPR,
        W_PREFETCH=W_PREFETCH,
        X_SCALE_VIA_LDS=X_SCALE_VIA_LDS,
        W_SCALE_VIA_LDS=W_SCALE_VIA_LDS,
    )
    BLOCK_K_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
    BLOCK_K_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W

    W_CACHE_MODIFIER: gl.constexpr = ".cg" if W_CACHE_CG else ""

    X_ELEM_BITS: gl.constexpr = x_ptr.dtype.element_ty.primitive_bitwidth
    W_ELEM_BITS: gl.constexpr = w_ptr.dtype.element_ty.primitive_bitwidth
    LOAD_X_LAYOUT: gl.constexpr = _load_layout(
        BLOCK_K_X, BLOCK_M, NUM_WARPS, [1, 0], X_ELEM_BITS
    )

    offs_xm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, LOAD_X_LAYOUT))
    offs_xk = gl.arange(0, BLOCK_K_X, layout=gl.SliceLayout(0, LOAD_X_LAYOUT))

    rows_m = off_m + offs_xm
    if not cfg.W_VIA_VGPR and USE_SLICE_N and BLOCK_M == 128:
        src_offs_xm = (offs_xm % 4) * 32 + (offs_xm // 4)
    elif not cfg.W_VIA_VGPR and USE_SLICE_N and BLOCK_M == 64:
        src_offs_xm = (offs_xm % 4) * 16 + (offs_xm // 4)
    elif not cfg.W_VIA_VGPR and USE_SLICE_N and BLOCK_M == 32:
        src_offs_xm = (offs_xm % 2) * 16 + (offs_xm // 2)
    else:
        src_offs_xm = offs_xm
    rows_m_x = off_m + src_offs_xm
    # m_limit = per-expert tail (HAS_RAGGED_OFFS) or global M.
    pre_gather_mask = rows_m < m_limit
    pre_gather_mask_x = rows_m_x < m_limit
    if HAS_GATHER:
        rows_m_safe = gl.where(pre_gather_mask, rows_m, gl.zeros_like(rows_m))
        rows_m = gl.load(
            gather_idx_ptr + rows_m_safe, mask=pre_gather_mask, other=0
        ).to(gl.int32)
        rows_m_x_safe = gl.where(pre_gather_mask_x, rows_m_x, gl.zeros_like(rows_m_x))
        rows_m_x = gl.load(
            gather_idx_ptr + rows_m_x_safe, mask=pre_gather_mask_x, other=0
        ).to(gl.int32)
        # Post-gather rows_m is in global token-id space (size M_X);
        # mask out junk gather_idx values too. Don't conflate M_X with
        # ``M`` (= dispatched tile count, can exceed M_X for top-k>1).
        mask_m_x = pre_gather_mask_x & (rows_m_x < M_X)
    else:
        # Clamp OOB lanes to 0 so the buffer_load address stays in
        # bounds during HIP graph warm-up; mask still filters.
        rows_m = gl.where(pre_gather_mask, rows_m, gl.zeros_like(rows_m))
        rows_m_x = gl.where(pre_gather_mask_x, rows_m_x, gl.zeros_like(rows_m_x))
        mask_m_x = pre_gather_mask_x

    k_limit_x = gl.multiple_of(K // cfg.DIV_FACTOR_X, 16)
    k_limit_w = gl.multiple_of(K // cfg.DIV_FACTOR_W, 16)

    # Swizzled scale loads use post-swizzle HBM shape via buffer_load_to_shared;
    # direct scale loads use G->VGPR gl.load and can follow gathered X rows.
    if HAS_X_BLOCK_SCALE:
        if cfg.X_SCALE_VIA_LDS:
            BLOCK_M_PS: gl.constexpr = cfg.BLOCK_M_PRESHUFFLED
            BLOCK_K_S_PS: gl.constexpr = cfg.BLOCK_K_SCALE_PRESHUFFLED
            LX_S: gl.constexpr = cfg.load_layout_x_scale
            offs_xs_m = gl.arange(0, BLOCK_M_PS, layout=gl.SliceLayout(1, LX_S))
            offs_xs_k = gl.arange(0, BLOCK_K_S_PS, layout=gl.SliceLayout(0, LX_S))
            row_base_x_s = off_m // cfg.PRESHUFFLE_FACTOR
            rows_m_scale = row_base_x_s + offs_xs_m
            row_limit_x_s = (M_X + cfg.PRESHUFFLE_FACTOR - 1) // cfg.PRESHUFFLE_FACTOR
            # Suppress the K-mask: the swizzle packs K with N so a
            # K-mask on the packed column scrambles both. The host
            # pads with e8m0=0 and the W K-mask zeros the OOB product
            # regardless of scale value.
            k_limit_xs_load = (
                (K // cfg.SCALE_BLOCK + 7) // 8 * 8
            ) * cfg.PRESHUFFLE_FACTOR
            x_scale_desc = AsyncCopyDescriptor.initialize(
                cfg,
                0,
                BLOCK_K_S_PS,
                x_scale_ptr,
                rows_m_scale,
                offs_xs_k,
                stride_xsm,
                stride_xsk,
                rows_m_scale[:, None] < row_limit_x_s,
                k_limit_xs_load,
            )
        else:
            offs_xs_m = gl.arange(
                0, BLOCK_M, layout=gl.SliceLayout(1, cfg.layout_x_scale)
            )
            offs_xs_k = gl.arange(
                0, BLOCK_K_SCALE, layout=gl.SliceLayout(0, cfg.layout_x_scale)
            )
            rows_m_scale = off_m + offs_xs_m
            if HAS_GATHER:
                pre_gather_mask_scale = rows_m_scale < m_limit
                rows_m_scale_safe = gl.where(
                    pre_gather_mask_scale,
                    rows_m_scale,
                    gl.zeros_like(rows_m_scale),
                )
                rows_m_scale = gl.load(
                    gather_idx_ptr + rows_m_scale_safe,
                    mask=pre_gather_mask_scale,
                    other=0,
                ).to(gl.int32)
                mask_m_scale = pre_gather_mask_scale & (rows_m_scale < M_X)
            else:
                mask_m_scale = rows_m_scale < m_limit
                rows_m_scale = gl.where(
                    mask_m_scale,
                    rows_m_scale,
                    gl.zeros_like(rows_m_scale),
                )
            x_scale_desc = AsyncCopyDescriptor.initialize(
                cfg,
                0,
                BLOCK_K_SCALE,
                x_scale_ptr,
                rows_m_scale,
                offs_xs_k,
                stride_xsm,
                stride_xsk,
                mask_m_scale[:, None],
                K // cfg.SCALE_BLOCK,
            )
    else:
        x_scale_desc: gl.constexpr = 0

    if HAS_W_BLOCK_SCALE:
        if cfg.W_SCALE_VIA_LDS:
            BLOCK_N_PS: gl.constexpr = cfg.BLOCK_N_PRESHUFFLED
            BLOCK_K_S_PS_W: gl.constexpr = cfg.BLOCK_K_SCALE_PRESHUFFLED
            LW_S: gl.constexpr = cfg.load_layout_w_scale
            offs_ws_n = gl.arange(0, BLOCK_N_PS, layout=gl.SliceLayout(1, LW_S))
            offs_ws_k = gl.arange(0, BLOCK_K_S_PS_W, layout=gl.SliceLayout(0, LW_S))
            row_base_w_s = off_n // cfg.PRESHUFFLE_FACTOR
            rows_n_scale = row_base_w_s + offs_ws_n
            if N_LIMIT:
                row_limit_w_s: gl.constexpr = (
                    N_LIMIT + cfg.PRESHUFFLE_FACTOR - 1
                ) // cfg.PRESHUFFLE_FACTOR
            else:
                row_limit_w_s = (N + cfg.PRESHUFFLE_FACTOR - 1) // cfg.PRESHUFFLE_FACTOR
            # See x_scale: suppress K-mask, OOB product is zero.
            k_limit_ws_load = (
                (K // cfg.SCALE_BLOCK + 7) // 8 * 8
            ) * cfg.PRESHUFFLE_FACTOR
            w_scale_desc = AsyncCopyDescriptor.initialize(
                cfg,
                0,
                BLOCK_K_S_PS_W,
                w_scale_ptr,
                rows_n_scale,
                offs_ws_k,
                stride_wsn,
                stride_wsk,
                rows_n_scale[:, None] < row_limit_w_s,
                k_limit_ws_load,
                base_offset=ws_base_offset,
            )
        else:
            offs_ws_n = gl.arange(
                0, BLOCK_N, layout=gl.SliceLayout(1, cfg.layout_w_scale)
            )
            offs_ws_k = gl.arange(
                0, BLOCK_K_SCALE, layout=gl.SliceLayout(0, cfg.layout_w_scale)
            )
            w_scale_desc = AsyncCopyDescriptor.initialize(
                cfg,
                0,
                BLOCK_K_SCALE,
                w_scale_ptr,
                off_n + offs_ws_n,
                offs_ws_k,
                stride_wsn,
                stride_wsk,
                (off_n + offs_ws_n)[:, None] < N,
                K // cfg.SCALE_BLOCK,
                base_offset=ws_base_offset,
            )
    else:
        w_scale_desc: gl.constexpr = 0

    if cfg.W_VIA_VGPR:
        acc = _run_moe_tile_w_via_vgpr(
            cfg,
            x_ptr,
            w_ptr,
            x_scale_desc,
            w_scale_desc,
            gather_idx_ptr,
            stride_xm,
            stride_xk,
            M_X,
            N,
            K,
            off_m,
            m_limit,
            rows_m_x,
            offs_xk,
            k_limit_x,
            k_limit_w,
            w_base_offset,
            pid_n,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K_X,
            BLOCK_K_W,
            NUM_WARPS,
            HAS_GATHER,
            USE_SLICE_MN,
            USE_SLICE_N,
            USE_WARP_PIPELINE,
            N_LIMIT,
            W_CACHE_MODIFIER,
        )
    elif cfg.W_PRESHUFFLED:
        acc = _run_moe_tile_preshuffled_lds_w(
            cfg,
            x_ptr,
            w_ptr,
            x_scale_desc,
            w_scale_desc,
            gather_idx_ptr,
            stride_xm,
            stride_xk,
            M_X,
            N,
            K,
            off_m,
            m_limit,
            rows_m_x,
            offs_xk,
            k_limit_x,
            k_limit_w,
            w_base_offset,
            pid_n,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K_X,
            BLOCK_K_W,
            NUM_WARPS,
            HAS_GATHER,
            USE_SLICE_MN,
            USE_SLICE_N,
            USE_WARP_PIPELINE,
            N_LIMIT,
            W_CACHE_MODIFIER,
        )
    elif W_TRANSPOSE:
        acc = _run_moe_tile_transposed_w(
            cfg,
            x_ptr,
            w_ptr,
            x_scale_desc,
            w_scale_desc,
            gather_idx_ptr,
            stride_xm,
            stride_xk,
            stride_wn,
            stride_wk,
            M_X,
            N,
            K,
            off_m,
            off_n,
            m_limit,
            rows_m_x,
            mask_m_x,
            offs_xk,
            k_limit_x,
            k_limit_w,
            w_base_offset,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K_X,
            BLOCK_K_W,
            NUM_WARPS,
            HAS_GATHER,
            USE_SLICE_MN,
            USE_SLICE_N,
            USE_WARP_PIPELINE,
            X_ELEM_BITS,
            W_ELEM_BITS,
            W_CACHE_MODIFIER,
        )
    else:
        acc = _run_moe_tile_ncontig_w(
            cfg,
            x_ptr,
            w_ptr,
            x_scale_desc,
            w_scale_desc,
            gather_idx_ptr,
            stride_xm,
            stride_xk,
            stride_wn,
            stride_wk,
            M_X,
            N,
            K,
            off_m,
            off_n,
            m_limit,
            rows_m_x,
            mask_m_x,
            offs_xk,
            k_limit_x,
            k_limit_w,
            w_base_offset,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K_X,
            BLOCK_K_W,
            NUM_WARPS,
            HAS_GATHER,
            USE_SLICE_MN,
            USE_SLICE_N,
            USE_WARP_PIPELINE,
            X_ELEM_BITS,
            W_ELEM_BITS,
            W_CACHE_MODIFIER,
        )

    if APPLY_X_GLOBAL_SCALE and not HAS_X_BLOCK_SCALE:
        x_global_scale = gl.load(x_global_scale_ptr)
        acc = acc * x_global_scale

    if HAS_BIAS:
        bias_offs = off_n + gl.arange(0, BLOCK_N, gl.SliceLayout(0, cfg.acc_layout))
        if Y_N_CONST and not DO_SWIGLU:
            BIAS_N: gl.constexpr = Y_N_CONST
            bias_mask = bias_offs < BIAS_N
        else:
            bias_mask = bias_offs < N
        # Masked lanes still need in-bounds addresses; W2 preshuffle can
        # tile over padded physical N while bias remains logical N.
        bias_offs_safe = gl.where(bias_mask, bias_offs, gl.zeros_like(bias_offs))
        bias = gl.load(
            bias_ptr + expert_id * stride_be + bias_offs_safe,
            mask=bias_mask,
            other=0.0,
        )
        acc = acc + bias[None, :].to(gl.float32)

    if DO_SWIGLU:
        out = _swiglu_reduce(
            acc, SWIGLU_ALPHA, SWIGLU_LIMIT, OUT_BLOCK_N, cfg.acc_layout
        )
        if HAS_FP8_QUANT_OUT:
            scale = gl.load(out_quant_scale_ptr).to(gl.float32)
            inv_scale = 1.0 / scale
            out = out * inv_scale
        out = out.to(y_ptr.dtype.element_ty)
        STORE_LAYOUT: gl.constexpr = out.type.layout
    else:
        out = acc.to(y_ptr.dtype.element_ty)
        STORE_LAYOUT: gl.constexpr = STORE
        out = gl.convert_layout(out, STORE_LAYOUT)

    offs_y_m = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, STORE_LAYOUT))
    off_n_out = pid_n * OUT_BLOCK_N
    offs_y_n = off_n_out + gl.arange(0, OUT_BLOCK_N, gl.SliceLayout(0, STORE_LAYOUT))

    # Clamp offs_y_m to m_limit before any pointer arithmetic; AMD/HIP
    # faults on the masked-off lanes if the address goes OOB even
    # under a predicated load.
    y_m_in_bounds = offs_y_m < m_limit
    offs_y_m_safe = gl.where(y_m_in_bounds, offs_y_m, gl.zeros_like(offs_y_m))

    if APPLY_GATE_SCAL:
        scal = gl.load(
            gate_scal_ptr + offs_y_m_safe,
            mask=y_m_in_bounds,
            other=1.0,
        )
        out = out * scal[:, None].to(out.dtype)

    if Y_N_CONST:
        ACTUAL_N: gl.constexpr = Y_N_CONST
    elif N_LIMIT:
        ACTUAL_N: gl.constexpr = (N_LIMIT // 2) if DO_SWIGLU else N_LIMIT
    else:
        actual_n = (N // 2) if DO_SWIGLU else N
    if Y_N_CONST or N_LIMIT:
        n_in_bounds = offs_y_n < ACTUAL_N
    else:
        n_in_bounds = offs_y_n < actual_n
    # Clamp masked-off N lanes before pointer arithmetic; masked GPU
    # memory ops may still fault on OOB addresses.
    offs_y_n_safe = gl.where(n_in_bounds, offs_y_n, gl.zeros_like(offs_y_n))
    if HAS_SCATTER:
        rows_y = gl.load(scatter_idx_ptr + offs_y_m_safe, mask=y_m_in_bounds, other=M)
        rows_y_in_bounds = y_m_in_bounds & (rows_y < M)
        mask_y = rows_y_in_bounds[:, None] & n_in_bounds[None, :]
        rows_y_safe = gl.where(rows_y_in_bounds, rows_y, gl.zeros_like(rows_y))
        y_offs = rows_y_safe[:, None] * stride_ym + offs_y_n_safe[None, :] * stride_yn
    else:
        mask_y = y_m_in_bounds[:, None] & n_in_bounds[None, :]
        offs_y_m_2d_safe = offs_y_m_safe[:, None]
        y_offs = offs_y_m_2d_safe * stride_ym + offs_y_n_safe[None, :] * stride_yn

    gl.store(y_ptr + y_offs, out, mask=mask_y)


@gluon.jit
def _xcd_chiplet_swizzle(pid, num_pids, XCD_SWIZZLE: gl.constexpr):
    if XCD_SWIZZLE == 1:
        return pid
    pids_per_xcd = num_pids // XCD_SWIZZLE
    extra = num_pids % XCD_SWIZZLE
    xcd = pid % XCD_SWIZZLE
    local = pid // XCD_SWIZZLE
    return xcd * pids_per_xcd + gl.minimum(xcd, extra) + local


@gluon.jit
def _group_m_swizzle(
    pid_mn,
    grid_m,
    grid_n,
    GROUP_M: gl.constexpr,
):
    if GROUP_M == 1:
        pid_m = pid_mn // grid_n
        pid_n = pid_mn % grid_n
    else:
        width = GROUP_M * grid_n
        group_id = pid_mn // width
        group_size = gl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        intra = pid_mn % width
        pid_m = group_id * GROUP_M + (intra % group_size)
        pid_n = intra // group_size
    return pid_m, pid_n


@gluon.jit
def _decode_block_schedule(block_schedule_ptr, pid_m):
    """Unpack the packed ``block_in_expert << 16 | expert`` schedule word."""
    schedule_raw = gl.load(block_schedule_ptr + pid_m).to(gl.uint32, bitcast=True)
    expert = (schedule_raw & 0x0000FFFF).to(gl.int32)
    block_in_expert = (schedule_raw >> 16).to(gl.int32)
    return expert, block_in_expert


@gluon.constexpr_function
def _medium_decode_moe_config(BLOCK_M, BLOCK_N, BLOCK_K):
    """Frozen single-buffer direct-load profile shared by the medium-decode stages."""
    return MoEConfig(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        "e4m3",
        "e2m1",
        32,
        1,
        True,
        False,
        True,
        "swizzle",
        gl.int32,
        (1, 1, 1),
        False,
        True,
        4,
        W_VIA_VGPR=False,
        W_PREFETCH=False,
    )


@gluon.jit
def _medium_decode_mfma_tile(
    X,
    W,
    WScale,
    token_m_l_addr,
    expert,
    kt,
    off_n,
    lk,
    lwk,
    lwn,
    acc,
    a_scale,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_wse,
    stride_wsn,
    stride_wsk,
    K,
    N,
    cfg: gl.constexpr,
    DIRECT_SHARED_X: gl.constexpr,
    DIRECT_SHARED_W: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCK_K_PACKED: gl.constexpr,
    MASK_K: gl.constexpr,
    MASK_N: gl.constexpr,
):
    s = _load_w_scale_tile_direct_cdna4(
        WScale, expert, kt, off_n, stride_wse, stride_wsk, stride_wsn, cfg
    )
    k_elem_l = kt * BLOCK_K + lk
    k_pack_l = kt * BLOCK_K_PACKED + lwk
    n_cols_l = off_n + lwn
    x_off_l = (
        gl.expand_dims(token_m_l_addr, 1).to(gl.int64) * stride_xm
        + gl.expand_dims(k_elem_l, 0).to(gl.int64) * stride_xk
    )
    w_off_l = (
        expert.to(gl.int64) * stride_we
        + k_pack_l.to(gl.int64) * stride_wk
        + n_cols_l.to(gl.int64) * stride_wn
    )
    # MASK_K guards the ragged K tail; MASK_N guards N tiles that do not divide
    # BLOCK_N. The full-K stage1 tile (N a multiple of BLOCK_N) takes neither,
    # which keeps its loads bit-identical to the unmasked baseline.
    if MASK_K:
        a_l = gl.amd.cdna4.buffer_load(
            ptr=X,
            offsets=x_off_l.to(gl.int32),
            mask=gl.expand_dims(k_elem_l, 0) < K,
            other=0.0,
        )
    else:
        a_l = gl.amd.cdna4.buffer_load(ptr=X, offsets=x_off_l.to(gl.int32))
    if MASK_N and MASK_K:
        b_l = gl.amd.cdna4.buffer_load(
            ptr=W,
            offsets=w_off_l.to(gl.int32),
            mask=(n_cols_l < N) & (k_pack_l < (K // 2)),
            other=0,
        )
    elif MASK_N:
        b_l = gl.amd.cdna4.buffer_load(
            ptr=W,
            offsets=w_off_l.to(gl.int32),
            mask=n_cols_l < N,
            other=0,
        )
    else:
        b_l = gl.amd.cdna4.buffer_load(ptr=W, offsets=w_off_l.to(gl.int32))
    a_smem = gl.allocate_shared_memory(
        X.dtype.element_ty, [BLOCK_M, BLOCK_K], DIRECT_SHARED_X, a_l
    )
    a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_smem, cfg.dot_layout_x)
    b_smem = gl.allocate_shared_memory(
        W.dtype.element_ty, [BLOCK_K_PACKED, BLOCK_N], DIRECT_SHARED_W, b_l
    )
    b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_smem, cfg.dot_layout_w)
    return gl.amd.cdna4.mfma_scaled(
        a=a,
        a_scale=a_scale,
        a_format="e4m3",
        b=b,
        b_scale=s,
        b_format="e2m1",
        acc=acc,
    )


@gluon.jit
def _medium_decode_body(
    X,
    W,
    WScale,
    Gather,
    Scatter,
    Gate,
    SliceSizes,
    SliceOffs,
    BlockOffs,
    BlockSchedule,
    Y,
    M,
    M_X,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_wse,
    stride_wsn,
    stride_wsk,
    stride_ym,
    stride_yn,
    stride_be,
    stride_bn,
    x_global_scale_ptr,
    out_quant_scale_ptr,
    bias_ptr,
    N_EXPERTS: gl.constexpr,
    NUM_TILES: gl.constexpr,
    GRID_N: gl.constexpr,
    GROUP_M: gl.constexpr,
    XCD_SWIZZLE: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    Y_N_CONST: gl.constexpr,
    MEDIUM_COMBINE: gl.constexpr,
):
    """Shared AITER-style direct-load grouped body for M=8/16 decode.

    ``MEDIUM_COMBINE=False`` is stage1 (gather X rows, SwiGLU + fp8-quant store);
    ``MEDIUM_COMBINE=True`` is stage2/combine (contiguous X rows, gate + scatter
    store). This is selected only for the additive medium-decode path; the
    existing prefill/default pipeline remains the fallback for all other shapes.
    """
    pid_raw = gl.program_id(axis=0)
    grid_n: gl.constexpr = GRID_N
    if XCD_SWIZZLE > 1:
        pid = _xcd_chiplet_swizzle(pid_raw, NUM_TILES, XCD_SWIZZLE)
    else:
        pid = pid_raw
    grid_m_padded: gl.constexpr = NUM_TILES // GRID_N
    pid_m, pid_n = _group_m_swizzle(pid, grid_m_padded, grid_n, GROUP_M)

    unpadded_m = gl.load(BlockOffs + N_EXPERTS).to(gl.int32)
    if pid_m >= unpadded_m:
        return

    expert, block_in_expert = _decode_block_schedule(BlockSchedule, pid_m)
    gl.assume(M >= 0)
    if not MEDIUM_COMBINE:
        gl.assume(M_X >= 0)
    gl.assume(N >= 0)
    gl.assume(K >= 0)
    gl.assume(expert >= 0)
    gl.assume(block_in_expert >= 0)
    gl.assume(stride_xm >= 0)
    gl.assume(stride_xk >= 0)
    gl.assume(stride_we >= 0)
    gl.assume(stride_wn >= 0)
    gl.assume(stride_wk >= 0)
    gl.assume(stride_wse >= 0)
    gl.assume(stride_wsn >= 0)
    gl.assume(stride_wsk >= 0)
    gl.assume(stride_ym >= 0)
    gl.assume(stride_yn >= 0)
    gl.assume(stride_be >= 0)
    gl.assume(stride_bn >= 0)

    cfg = _medium_decode_moe_config(BLOCK_M, BLOCK_N, BLOCK_K)
    BLOCK_K_PACKED: gl.constexpr = BLOCK_K // 2
    BLOCK_K_SCALE: gl.constexpr = BLOCK_K // 32
    gl.static_assert(
        BLOCK_K_SCALE == 8, "M=8/16 direct WScale path assumes BLOCK_K=256"
    )
    OUT_BLOCK_N: gl.constexpr = BLOCK_N // 2

    X_ELEM_BITS: gl.constexpr = X.dtype.element_ty.primitive_bitwidth
    W_ELEM_BITS: gl.constexpr = W.dtype.element_ty.primitive_bitwidth
    LOAD_X_LAYOUT: gl.constexpr = _load_layout(BLOCK_K, BLOCK_M, 4, [1, 0], X_ELEM_BITS)
    LOAD_W_LAYOUT: gl.constexpr = _load_layout(
        BLOCK_K_PACKED, BLOCK_N, 4, [0, 1], W_ELEM_BITS
    )
    DIRECT_SHARED_X: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 16, order=[1, 0])
    DIRECT_SHARED_W: gl.constexpr = gl.SwizzledSharedLayout(16, 2, 8, order=[0, 1])

    m_base = gl.load(SliceOffs + expert).to(gl.int32)
    m_size = gl.load(SliceSizes + expert).to(gl.int32)
    gl.assume(m_base >= 0)
    gl.assume(m_size > 0)
    if not MEDIUM_COMBINE:
        off_m = m_base + block_in_expert * BLOCK_M
        gl.assume(off_m >= 0)
    off_n = pid_n * BLOCK_N
    gl.assume(off_n >= 0)

    lk = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, LOAD_X_LAYOUT))
    lwk = gl.arange(0, BLOCK_K_PACKED, layout=gl.SliceLayout(1, LOAD_W_LAYOUT))[:, None]
    lwn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, LOAD_W_LAYOUT))[None, :]

    local_m_l = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, LOAD_X_LAYOUT))
    valid_m_l = local_m_l < m_size
    safe_local_m_l = gl.where(valid_m_l, local_m_l, gl.zeros_like(local_m_l))
    sorted_l = m_base + safe_local_m_l
    if MEDIUM_COMBINE:
        # Combine reads the sorted (contiguous) X rows directly.
        row_addr = sorted_l
    else:
        # Dispatch gathers the routed token rows for this expert block.
        token_m_l = gl.amd.cdna4.buffer_load(
            ptr=Gather,
            offsets=sorted_l.to(gl.int32),
            mask=valid_m_l,
            other=0,
        ).to(gl.int32)
        row_addr = token_m_l.to(gl.uint32)

    a_scale = gl.full(
        (BLOCK_M, BLOCK_K_SCALE), 127, gl.uint8, layout=cfg.layout_x_scale
    )
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=cfg.acc_layout)

    total_kt = gl.cdiv(K, BLOCK_K)
    gl.assume(total_kt >= 0)
    num_full = K // BLOCK_K
    gl.assume(num_full >= 0)
    for kt in range(0, num_full):
        gl.assume(kt >= 0)
        acc = _medium_decode_mfma_tile(
            X,
            W,
            WScale,
            row_addr,
            expert,
            kt,
            off_n,
            lk,
            lwk,
            lwn,
            acc,
            a_scale,
            stride_xm,
            stride_xk,
            stride_we,
            stride_wn,
            stride_wk,
            stride_wse,
            stride_wsn,
            stride_wsk,
            K,
            N,
            cfg,
            DIRECT_SHARED_X,
            DIRECT_SHARED_W,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            BLOCK_K_PACKED,
            MASK_K=False,
            MASK_N=MEDIUM_COMBINE,
        )

    if total_kt > num_full:
        kt = num_full
        gl.assume(kt >= 0)
        acc = _medium_decode_mfma_tile(
            X,
            W,
            WScale,
            row_addr,
            expert,
            kt,
            off_n,
            lk,
            lwk,
            lwn,
            acc,
            a_scale,
            stride_xm,
            stride_xk,
            stride_we,
            stride_wn,
            stride_wk,
            stride_wse,
            stride_wsn,
            stride_wsk,
            K,
            N,
            cfg,
            DIRECT_SHARED_X,
            DIRECT_SHARED_W,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            BLOCK_K_PACKED,
            MASK_K=True,
            MASK_N=True,
        )

    acc = acc * gl.load(x_global_scale_ptr).to(gl.float32)
    if MEDIUM_COMBINE:
        if HAS_BIAS:
            bias_n = off_n + gl.arange(0, BLOCK_N, gl.SliceLayout(0, cfg.acc_layout))
            if Y_N_CONST:
                bias_bound = bias_n < Y_N_CONST
            else:
                bias_bound = bias_n < N
            bias = gl.load(
                bias_ptr
                + expert.to(gl.int64) * stride_be
                + bias_n.to(gl.int64) * stride_bn,
                mask=bias_bound,
                other=0.0,
            ).to(gl.float32)
            bias = gl.convert_layout(bias, gl.SliceLayout(0, cfg.acc_layout))
            acc = acc + bias[None, :]

        out = acc.to(Y.dtype.element_ty)
        STORE_LAYOUT: gl.constexpr = out.type.layout
        n_out = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(0, STORE_LAYOUT))
        local_store_m = gl.arange(0, BLOCK_M, gl.SliceLayout(1, STORE_LAYOUT))
        valid_store = local_store_m < m_size
        safe_store_m = gl.where(
            valid_store, local_store_m, gl.zeros_like(local_store_m)
        )
        sorted_store = m_base + safe_store_m
        scatter_row = gl.load(Scatter + sorted_store, mask=valid_store, other=0).to(
            gl.int32
        )
        gate = gl.load(Gate + sorted_store, mask=valid_store, other=0.0).to(
            Y.dtype.element_ty
        )
        out = out * gate[:, None]
        y_offs = (
            scatter_row[:, None].to(gl.int64) * stride_ym
            + n_out[None, :].to(gl.int64) * stride_yn
        )
        if Y_N_CONST:
            store_mask = valid_store[:, None] & (n_out[None, :] < Y_N_CONST)
        else:
            store_mask = valid_store[:, None] & (n_out[None, :] < N)
        _moe_masked_store(out, Y, y_offs, store_mask, USE_BUFFER_STORE=False)
    else:
        if HAS_BIAS:
            bias_n = off_n + gl.arange(0, BLOCK_N, gl.SliceLayout(0, cfg.acc_layout))
            if Y_N_CONST:
                bias_bound = bias_n < Y_N_CONST
            else:
                bias_bound = bias_n < N
            bias = gl.load(
                bias_ptr
                + expert.to(gl.int64) * stride_be
                + bias_n.to(gl.int64) * stride_bn,
                mask=bias_bound,
                other=0.0,
            ).to(gl.float32)
            bias = gl.convert_layout(bias, gl.SliceLayout(0, cfg.acc_layout))
            acc = acc + bias[None, :]

        out = _swiglu_reduce(
            acc, SWIGLU_ALPHA, SWIGLU_LIMIT, OUT_BLOCK_N, cfg.acc_layout
        )
        out_inv_scale = 1.0 / gl.load(out_quant_scale_ptr).to(gl.float32)
        out = (out * out_inv_scale).to(Y.dtype.element_ty)
        STORE_LAYOUT: gl.constexpr = out.type.layout
        sm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, STORE_LAYOUT))
        n_out = pid_n * OUT_BLOCK_N + gl.arange(
            0, OUT_BLOCK_N, gl.SliceLayout(0, STORE_LAYOUT)
        )
        local_store_m = block_in_expert * BLOCK_M + sm
        sorted_store = m_base + local_store_m
        valid_store = local_store_m < m_size
        y_off = (
            sorted_store[:, None].to(gl.int64) * stride_ym
            + n_out[None, :].to(gl.int64) * stride_yn
        )
        if Y_N_CONST:
            store_mask = valid_store[:, None] & (n_out[None, :] < (Y_N_CONST // 2))
        else:
            store_mask = valid_store[:, None] & (n_out[None, :] < (N // 2))
        _moe_masked_store(out, Y, y_off, store_mask, USE_BUFFER_STORE=True)


def _pipelined_moe_kernel_repr(specialization) -> str:
    """Distinct rocprof names for schedule vs no-schedule specialization."""
    if bool(specialization.constants.get("IS_MEDIUM_DECODE", False)):
        if bool(specialization.constants.get("MEDIUM_COMBINE", False)):
            return "_moe_medium_decode_combine"
        return "_moe_medium_decode_dispatch"
    use_block_schedule = bool(specialization.constants.get("USE_BLOCK_SCHEDULE", False))
    if use_block_schedule:
        return "_pipelined_moe_kernel_scaled_block_schedule"
    return "_pipelined_moe_kernel_scaled"


@gluon.jit(repr=_pipelined_moe_kernel_repr)
def _pipelined_moe_kernel_scaled(
    x_ptr,
    w_ptr,
    x_scale_ptr,
    w_scale_ptr,
    bias_ptr,
    y_ptr,
    gather_idx_ptr,
    scatter_idx_ptr,
    gate_scal_ptr,
    slice_offs_ptr,
    slice_sizes_ptr,
    block_offs_ptr,
    block_schedule_ptr,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_xsm,
    stride_xsk,
    stride_wse,
    stride_wsn,
    stride_wsk,
    stride_yn,
    stride_ym,
    stride_be,
    stride_bn,
    M,
    M_X,
    N,
    K,
    x_global_scale_ptr,
    out_quant_scale_ptr,
    NUM_TILES,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCKS_PER_EXPERT: gl.constexpr,
    X_FORMAT: gl.constexpr,
    W_FORMAT: gl.constexpr,
    UPCAST_INDICES: gl.constexpr,
    HAS_X_BLOCK_SCALE: gl.constexpr,
    HAS_W_BLOCK_SCALE: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    HAS_GATHER: gl.constexpr,
    HAS_SCATTER: gl.constexpr,
    DO_SWIGLU: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    OUT_BLOCK_N: gl.constexpr,
    APPLY_GATE_SCAL: gl.constexpr,
    HAS_RAGGED_OFFS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    SCALE_LOAD_MODE: gl.constexpr,
    W_TRANSPOSE: gl.constexpr = False,
    NUM_SUBTILES: gl.constexpr = (1, 1, 1),
    EVEN_K: gl.constexpr = True,
    K_ITERS: gl.constexpr = 0,
    N_CONST: gl.constexpr = 0,
    Y_N_CONST: gl.constexpr = 0,
    APPLY_X_GLOBAL_SCALE: gl.constexpr = True,
    USE_WARP_PIPELINE: gl.constexpr = False,
    USE_SLICE_MN: gl.constexpr = False,
    USE_SLICE_N: gl.constexpr = False,
    HAS_FP8_QUANT_OUT: gl.constexpr = False,
    USE_BLOCK_SCHEDULE: gl.constexpr = False,
    N_EXPTS_TOT: gl.constexpr = 0,
    GRID_N: gl.constexpr = 0,
    GROUP_M: gl.constexpr = 1,
    XCD_SWIZZLE: gl.constexpr = 1,
    W_PRESHUFFLED: gl.constexpr = False,
    W_VIA_VGPR: gl.constexpr = False,
    W_PREFETCH: gl.constexpr = True,
    W_CACHE_CG: gl.constexpr = False,
    X_SCALE_VIA_LDS: gl.constexpr = False,
    W_SCALE_VIA_LDS: gl.constexpr = False,
    USE_NARROW_N_STORE_LAYOUT: gl.constexpr = False,
    IS_MEDIUM_DECODE: gl.constexpr = False,
    MEDIUM_COMBINE: gl.constexpr = False,
):
    if IS_MEDIUM_DECODE:
        # Medium-decode (M=8/16) reuses this kernel's signature but runs the
        # single-buffer direct-load body instead of the pipelined prefill loop.
        # The constexpr guard DCEs this branch for the existing/default path.
        _medium_decode_body(
            x_ptr,
            w_ptr,
            w_scale_ptr,
            gather_idx_ptr,
            scatter_idx_ptr,
            gate_scal_ptr,
            slice_sizes_ptr,
            slice_offs_ptr,
            block_offs_ptr,
            block_schedule_ptr,
            y_ptr,
            M,
            M_X,
            N,
            K,
            stride_xm,
            stride_xk,
            stride_we,
            stride_wn,
            stride_wk,
            stride_wse,
            stride_wsn,
            stride_wsk,
            stride_ym,
            stride_yn,
            stride_be,
            stride_bn,
            x_global_scale_ptr,
            out_quant_scale_ptr,
            bias_ptr,
            N_EXPERTS=N_EXPTS_TOT,
            NUM_TILES=NUM_TILES,
            GRID_N=GRID_N,
            GROUP_M=GROUP_M,
            XCD_SWIZZLE=XCD_SWIZZLE,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            HAS_BIAS=HAS_BIAS,
            SWIGLU_ALPHA=SWIGLU_ALPHA,
            SWIGLU_LIMIT=SWIGLU_LIMIT,
            Y_N_CONST=Y_N_CONST,
            MEDIUM_COMBINE=MEDIUM_COMBINE,
        )
        return

    if GRID_N > 0:
        grid_n: gl.constexpr = GRID_N
        tiles_per_expert: gl.constexpr = BLOCKS_PER_EXPERT * GRID_N
    else:
        grid_n = (N + BLOCK_N - 1) // BLOCK_N
        tiles_per_expert = BLOCKS_PER_EXPERT * grid_n

    if USE_BLOCK_SCHEDULE:
        unpadded_m = gl.load(block_offs_ptr + N_EXPTS_TOT).to(gl.int32)
        loop_tiles = unpadded_m * grid_n
    else:
        loop_tiles = NUM_TILES

    for tile_idx in range(gl.program_id(0), loop_tiles, gl.num_programs(0)):
        if USE_BLOCK_SCHEDULE:
            swizzled = _xcd_chiplet_swizzle(tile_idx, loop_tiles, XCD_SWIZZLE)
            pid_m, pid_n = _group_m_swizzle(swizzled, unpadded_m, grid_n, GROUP_M)
            schedule_raw = gl.load(block_schedule_ptr + pid_m).to(
                gl.uint32, bitcast=True
            )
            compact_idx = (schedule_raw & 0x0000FFFF).to(gl.int32)
            block_in_expert = (schedule_raw >> 16).to(gl.int32)
        else:
            # Dense path: tile_idx packs (compact_idx, intra-expert pid);
            # GROUP_M applies WITHIN one expert (W only reusable per expert).
            swizzled = _xcd_chiplet_swizzle(tile_idx, NUM_TILES, XCD_SWIZZLE)
            compact_idx = swizzled // tiles_per_expert
            local = swizzled % tiles_per_expert
            block_in_expert, pid_n = _group_m_swizzle(
                local, BLOCKS_PER_EXPERT, grid_n, GROUP_M
            )

        _pipelined_moe_tile_compute(
            x_ptr,
            w_ptr,
            x_scale_ptr,
            w_scale_ptr,
            bias_ptr,
            y_ptr,
            gather_idx_ptr,
            scatter_idx_ptr,
            gate_scal_ptr,
            slice_offs_ptr,
            slice_sizes_ptr,
            stride_xm,
            stride_xk,
            stride_we,
            stride_wn,
            stride_wk,
            stride_xsm,
            stride_xsk,
            stride_wse,
            stride_wsn,
            stride_wsk,
            stride_yn,
            stride_ym,
            stride_be,
            stride_bn,
            M,
            M_X,
            N,
            K,
            x_global_scale_ptr,
            out_quant_scale_ptr,
            compact_idx,
            block_in_expert,
            pid_n,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            BLOCKS_PER_EXPERT=BLOCKS_PER_EXPERT,
            X_FORMAT=X_FORMAT,
            W_FORMAT=W_FORMAT,
            UPCAST_INDICES=UPCAST_INDICES,
            HAS_X_BLOCK_SCALE=HAS_X_BLOCK_SCALE,
            HAS_W_BLOCK_SCALE=HAS_W_BLOCK_SCALE,
            HAS_BIAS=HAS_BIAS,
            HAS_GATHER=HAS_GATHER,
            HAS_SCATTER=HAS_SCATTER,
            DO_SWIGLU=DO_SWIGLU,
            SWIGLU_ALPHA=SWIGLU_ALPHA,
            SWIGLU_LIMIT=SWIGLU_LIMIT,
            OUT_BLOCK_N=OUT_BLOCK_N,
            APPLY_GATE_SCAL=APPLY_GATE_SCAL,
            HAS_RAGGED_OFFS=HAS_RAGGED_OFFS,
            NUM_WARPS=NUM_WARPS,
            NUM_BUFFERS=NUM_BUFFERS,
            SCALE_LOAD_MODE=SCALE_LOAD_MODE,
            W_TRANSPOSE=W_TRANSPOSE,
            NUM_SUBTILES=NUM_SUBTILES,
            EVEN_K=EVEN_K,
            K_ITERS=K_ITERS,
            N_CONST=N_CONST,
            Y_N_CONST=Y_N_CONST,
            APPLY_X_GLOBAL_SCALE=APPLY_X_GLOBAL_SCALE,
            USE_WARP_PIPELINE=USE_WARP_PIPELINE,
            USE_SLICE_MN=USE_SLICE_MN,
            USE_SLICE_N=USE_SLICE_N,
            HAS_FP8_QUANT_OUT=HAS_FP8_QUANT_OUT,
            W_PRESHUFFLED=W_PRESHUFFLED,
            W_VIA_VGPR=W_VIA_VGPR,
            W_PREFETCH=W_PREFETCH,
            W_CACHE_CG=W_CACHE_CG,
            X_SCALE_VIA_LDS=X_SCALE_VIA_LDS,
            W_SCALE_VIA_LDS=W_SCALE_VIA_LDS,
            USE_NARROW_N_STORE_LAYOUT=USE_NARROW_N_STORE_LAYOUT,
        )


# ---------------------------------------------------------------------------
# Static profile helper (sgpr/vgpr spill detection)
# ---------------------------------------------------------------------------


def _parse_amdgcn_metric(amdgcn: str, key: str) -> int | None:
    """Look for ``.<key>: N`` or ``;  Key: N`` in the AMDGCN dump."""
    import re

    m = re.search(rf"\.{key}:\s+(\d+)", amdgcn)
    if m is not None:
        return int(m.group(1))
    m = re.search(rf";\s+{key}\s*[:=]?\s+(\d+)", amdgcn)
    if m is not None:
        return int(m.group(1))
    return None


def static_profile(kernel: Any, *, label: str = "") -> dict:
    amdgcn = kernel.asm.get("amdgcn", "")
    fields = [
        "sgpr_count",
        "sgpr_spill_count",
        "vgpr_count",
        "vgpr_spill_count",
        "ScratchSize",
        "codeLenInByte",
        "Occupancy",
    ]
    profile = {f: _parse_amdgcn_metric(amdgcn, f) for f in fields}
    if label:
        profile["label"] = label
    return profile


_LAST_KERNEL_PROFILE: dict | None = None
_PROFILE_BY_KERNEL_ID: dict[int, dict] = {}


def _capture_launch_profile(k: Any) -> None:
    global _LAST_KERNEL_PROFILE
    key = id(k)
    prof = _PROFILE_BY_KERNEL_ID.get(key)
    if prof is None:
        prof = static_profile(k)
        name = getattr(k, "name", None)
        if name is None:
            md = getattr(k, "metadata", None)
            name = getattr(md, "name", None) if md is not None else None
        if name is not None:
            prof["kernel_name"] = str(name)
        md = getattr(k, "metadata", None)
        if md is not None:
            shared = getattr(md, "shared", None)
            if shared is not None:
                prof["shared"] = int(shared)
        _PROFILE_BY_KERNEL_ID[key] = prof
    _LAST_KERNEL_PROFILE = prof


def last_kernel_profile() -> dict | None:
    return _LAST_KERNEL_PROFILE


def assert_no_spills(profile: dict, *, allow_scratch: int = 0) -> None:
    sgpr_spill = profile.get("sgpr_spill_count") or 0
    vgpr_spill = profile.get("vgpr_spill_count") or 0
    scratch = profile.get("ScratchSize") or 0
    msg = []
    if sgpr_spill:
        msg.append(f"sgpr_spill={sgpr_spill}")
    if vgpr_spill:
        msg.append(f"vgpr_spill={vgpr_spill}")
    if scratch > allow_scratch:
        msg.append(f"scratch={scratch} (allowed={allow_scratch})")
    if msg:
        raise AssertionError(
            f"Gluon MoE kernel '{profile.get('label', '?')}' "
            f"shows static spills: {', '.join(msg)}"
        )


def _dense_grid_dims(M: int, block_m: int) -> tuple[int, int]:
    """Return ``(num_active, blocks_per_expert)`` for the no-ragged
    (dense / gating GEMM) path."""
    return 1, (M + block_m - 1) // block_m


def _make_dummy(device, dtype=torch.int32, n: int = 0) -> torch.Tensor:
    return torch.empty(max(n, 0), device=device, dtype=dtype)


def _swizzle_scales_cdna4(s: torch.Tensor) -> torch.Tensor:
    assert s.dtype == torch.uint8, (
        f"_swizzle_scales_cdna4: expected uint8 e8m0 scales, " f"got {s.dtype}"
    )
    # gluon convention -> upstream convention.
    s = s.transpose(-2, -1).contiguous()
    *leading_shape, K_SCALE, N = s.shape
    B = 1
    for d in leading_shape:
        B *= d
    ALIGN_K_S = _ALIGN_K_SCALE_SWIZZLE
    ALIGN_N = _ALIGN_N_SWIZZLE
    K_SCALE_pad = ((K_SCALE + ALIGN_K_S - 1) // ALIGN_K_S) * ALIGN_K_S
    N_pad = ((N + ALIGN_N - 1) // ALIGN_N) * ALIGN_N
    # repack is identity for uint8 (only re-orders e2m1 nibbles).
    s = s.mT.contiguous().mT
    s = torch.nn.functional.pad(s, (0, N_pad - N, 0, K_SCALE_pad - K_SCALE))
    s = s.transpose(-1, -2)  # (..., N_pad, K_SCALE_pad)
    s = s.reshape(B, N_pad, K_SCALE_pad)
    s = s.view(B, N_pad // 32, 2, 16, K_SCALE_pad // 8, 2, 4, 1)
    s = s.permute(0, 1, 4, 6, 3, 5, 2, 7).contiguous()
    s = s.reshape(B, N_pad // 32, K_SCALE_pad * 32)
    s = s.transpose(-1, -2)  # (B, K_SCALE_pad*32, N_pad/32)
    return s


def _is_scale_swizzled_cdna4(s: torch.Tensor) -> bool:
    """``stride(-2) == 1`` (the contig K_S*32 axis) is the upstream
    swizzle's signature; cheap check."""
    return s.stride(-2) == 1 and s.stride(-1) >= s.shape[-2]


def _preprocess_scale(data: torch.Tensor | None, mode: str) -> torch.Tensor | None:
    if data is None:
        return None
    if mode not in _SCALE_LOAD_MODES:
        raise ValueError(
            f"_preprocess_scale: SCALE_LOAD_MODE must be one of "
            f"{_SCALE_LOAD_MODES}, got {mode!r}"
        )
    if mode == "swizzle":
        if _is_scale_swizzled_cdna4(data):
            return data
        return _swizzle_scales_cdna4(data)
    return data


# ---------------------------------------------------------------------------
# Public launcher: software-pipelined ragged matmul (scaled-MFMA only)
# ---------------------------------------------------------------------------


def _scale_strides(scale: torch.Tensor | None, mode: str = "bypass") -> tuple[int, int]:
    if scale is None:
        return 0, 0
    if mode == "swizzle":
        return scale.stride(-1), scale.stride(-2)
    return scale.stride(-2), scale.stride(-1)


_SCALED_FORMATS = {"e2m1", "e4m3", "e5m2"}


def _launch_kernel(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    y: torch.Tensor,
    bias: torch.Tensor | None,
    gather_indx,
    scatter_indx,
    gate_scal: torch.Tensor | None,
    a_ragged_metadata,
    swiglu: tuple[float, float] | None,
    out_block_n: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_buffers: int = 2,
    x_format: str,
    w_format: str = "e2m1",
    x_scale: torch.Tensor | None = None,
    w_scale: torch.Tensor | None = None,
    x_global_scale: torch.Tensor | float | None = 1.0,
    scale_load_mode: str = "bypass",
    w_transpose: bool = False,
    apply_x_global_scale: bool | None = None,
    use_warp_pipeline: bool = False,
    use_slice_mn: bool = False,
    use_slice_n: bool = False,
    persistent: bool | None = False,
    num_ctas: int | None = None,
    group_m: int | None = None,
    xcd_swizzle: int | None = None,
    out_quant_scale: torch.Tensor | float | None = None,
    w_preshuffle: bool = False,
    y_n_const: int = 0,
    w_cache_cg: bool | None = None,
    use_narrow_n_store_layout: bool = False,
    medium_decode_dispatch: bool = False,
    medium_decode_combine: bool = False,
):
    assert x_format in _SCALED_FORMATS, f"unknown x_format={x_format!r}"
    assert w_format in _SCALED_FORMATS, f"unknown w_format={w_format!r}"
    if apply_x_global_scale is None:
        apply_x_global_scale = True
    assert scale_load_mode in _SCALE_LOAD_MODES, (
        f"scale_load_mode must be one of {_SCALE_LOAD_MODES}, "
        f"got {scale_load_mode!r}"
    )
    has_x_block_scale = x_format == "e2m1"
    has_w_block_scale = w_format == "e2m1"
    if has_x_block_scale:
        assert x_scale is not None, "mxfp4 A requires a block-scale tensor"
    if has_w_block_scale:
        assert w_scale is not None, "mxfp4 W requires a block-scale tensor"

    M_X = x.shape[-2]
    if gather_indx is not None:
        gather_buf_for_m = gather_indx.src_indx
        M = int(gather_buf_for_m.shape[0])
    else:
        M = M_X
    K_phys = x.shape[-1]
    div_x = 2 if x_format == "e2m1" else 1
    div_w = 2 if w_format == "e2m1" else 1
    K = K_phys * div_x

    scale_load_mode = _effective_scale_load_mode(
        scale_load_mode,
        block_m,
        block_n,
        block_k,
        scale_block=32,
        has_x_scale=has_x_block_scale and gather_indx is None,
        has_w_scale=has_w_block_scale,
        k=K,
        x_format=x_format,
        num_buffers=num_buffers,
    )

    if w.ndim == 3:
        _, K_w_phys, N_w_phys = w.shape
    else:
        K_w_phys, N_w_phys = w.shape
    K_w = K_w_phys * div_w
    if w_preshuffle and getattr(w, "is_shuffled_for_gluon_dot", False):
        # Host pre-shuffle zero-pads K_pk to a multiple of 128 and W
        # scale to padded N (combine launcher trims output back).
        original_k_pk = getattr(w, "original_k_pk", K_w_phys)
        assert (
            K == original_k_pk * div_w
        ), f"K mismatch: A logical K={K} vs W original logical K={original_k_pk * div_w}"
        assert (
            K_w_phys >= original_k_pk and K_w_phys % 128 == 0
        ), f"shuffled W K_pk ({K_w_phys}) must be K_pk_padded (multiple of 128)"
        N = N_w_phys
    else:
        assert K == K_w, f"K mismatch: A logical K={K} vs W logical K={K_w}"
        N = N_w_phys

    assert (
        block_k % _MFMA_SCALED_K == 0
    ), f"BLOCK_K={block_k} must be a multiple of MFMA K dim ({_MFMA_SCALED_K})"
    assert (
        block_k >= _MFMA_SCALED_K
    ), f"scaled MFMA requires BLOCK_K >= {_MFMA_SCALED_K} (got {block_k})"
    assert block_m % _MFMA_M == 0
    if w_preshuffle:
        packed_block_n = _preshuffled_layout_block_n(w)
        expected_packed_block_n = block_n // 2 if use_slice_n else block_n
        assert not use_slice_mn, "preshuffled W LDS path does not support USE_SLICE_MN"
        assert num_warps == 4, "preshuffled W LDS load layout requires NUM_WARPS=4"
        assert packed_block_n == 128, (
            f"preshuffled W LDS path currently supports only block_n=128 "
            f"packed layouts; got block_n={packed_block_n}"
        )
        assert expected_packed_block_n == packed_block_n, (
            f"preshuffled W packed block_n={packed_block_n} is incompatible with "
            f"execution BLOCK_N={block_n}, USE_SLICE_N={use_slice_n}"
        )
        assert (
            N % packed_block_n == 0
        ), f"preshuffled W N={N} must be divisible by packed block_n={packed_block_n}"

    grid_n = (N + block_n - 1) // block_n

    # Per-expert ragged offsets needed when per-expert size < BLOCK_M
    # (else off_m would walk past the expert tail into the next one).
    has_ragged_offs = a_ragged_metadata is not None
    if has_ragged_offs:
        slice_offs_buf = _as_int32(a_ragged_metadata.slice_offs)
        slice_sizes_buf = _as_int32(a_ragged_metadata.slice_sizes)
    else:
        slice_offs_buf = _make_dummy(x.device, torch.int32)
        slice_sizes_buf = _make_dummy(x.device, torch.int32)

    # Block-schedule path: host picks grid_m as an integer upper bound
    # (no D2H sync, graph-capturable) and the kernel decodes
    # (expert_id, block_in_expert) from block_schedule[pid_m]. The
    # dense fallback is only valid when ``a_ragged_metadata is None``.
    use_block_schedule = (
        has_ragged_offs
        and block_m in _BLOCK_SIZES_FROZEN
        and getattr(a_ragged_metadata, "block_offs_data", None) is not None
        and getattr(a_ragged_metadata, "block_schedule_data", None) is not None
    )

    if use_block_schedule:
        n_slices = int(a_ragged_metadata.slice_sizes.shape[0])
        grid_m_upper = RaggedTensorMetadata.n_blocks(n_slices, M, block_m)
        num_tiles_total = grid_m_upper * grid_n
        block_offs_buf = _as_int32(_ragged_block_offs(a_ragged_metadata, block_m))
        block_schedule_buf = _as_int32(
            _ragged_block_schedule(a_ragged_metadata, block_m)
        )
        blocks_per_expert = 1  # unused constexpr sentinel in schedule mode
    else:
        # Only ``a_ragged_metadata is None`` (dense GEMM) is accepted;
        # hand-built ragged metadata without schedule tables is rejected
        # to avoid the historical D2H ``counts.tolist()`` path.
        assert not has_ragged_offs, (
            f"_launch_kernel requires a_ragged_metadata to either be None "
            f"(dense / gating GEMM) or to have populated "
            f"block_offs_data + block_schedule_data and "
            f"block_m={block_m} in {sorted(_BLOCK_SIZES_FROZEN)}. Build "
            f"the metadata via triton_kernels' make_ragged_tensor_metadata."
        )
        _, blocks_per_expert = _dense_grid_dims(M, block_m)
        num_tiles_total = blocks_per_expert * grid_n
        block_offs_buf = _make_dummy(x.device, torch.int32)
        block_schedule_buf = _make_dummy(x.device, torch.int32)
        n_slices = 0

    if persistent:
        if num_ctas is None:
            num_ctas = _persistent_grid_size(num_tiles_total)
        else:
            num_ctas = max(1, min(num_ctas, num_tiles_total))
    else:
        num_ctas = max(1, num_tiles_total)
    grid = (num_ctas, 1)

    grid_m_for_swizzle = num_tiles_total // grid_n
    auto_group_m, auto_xcd = _autotune_pid_swizzle(
        num_tiles_total=num_tiles_total,
        grid_n=grid_n,
        grid_m_padded=grid_m_for_swizzle,
        block_m=block_m,
    )
    if group_m is None:
        group_m = auto_group_m
    if xcd_swizzle is None:
        xcd_swizzle = auto_xcd
    if group_m > 1 and grid_m_for_swizzle % group_m != 0:
        group_m = 1
    if xcd_swizzle > 1 and num_tiles_total % xcd_swizzle != 0:
        xcd_swizzle = 1
    if w_cache_cg is None:
        w_cache_cg = block_m <= 32

    bias_buf = bias if bias is not None else _make_dummy(x.device, torch.float32)
    gather_buf = (
        gather_indx.src_indx
        if gather_indx is not None
        else _make_dummy(x.device, torch.int32)
    )
    scatter_buf = (
        scatter_indx.dst_indx
        if scatter_indx is not None
        else _make_dummy(x.device, torch.int32)
    )
    gate_scal_buf = (
        gate_scal if gate_scal is not None else _make_dummy(x.device, torch.float32)
    )

    swiglu_alpha = swiglu[0] if swiglu is not None else 0.0
    swiglu_limit = swiglu[1] if swiglu is not None else 0.0

    w3 = w if w.ndim == 3 else w.unsqueeze(0)

    if w_preshuffle:
        # Host pre-shuffled into 5-D HBM byte layout. The preshuffled
        # descriptor uses N directly for the K-iter stride and stages the
        # tile through LDS; .contiguous() would clobber the HBM layout.
        # stride_wn/stride_wk are not consulted; only stride_we matters.
        # w_transpose is irrelevant on this path.
        stride_wn, stride_wk = w3.stride(-2), w3.stride(-1)
    elif w_transpose:
        # K-contig W staged as [BN, BK] in LDS; view permuted for dot.
        w3 = w3.transpose(-1, -2).contiguous()
        stride_wn, stride_wk = w3.stride(-2), w3.stride(-1)
    else:
        # N-contig W staged as [BK, BN] in LDS.
        stride_wn, stride_wk = w3.stride(-1), w3.stride(-2)

    x_scale_load_mode = scale_load_mode
    if has_x_block_scale and gather_indx is not None:
        x_scale_load_mode = "transpose"
    w_scale_load_mode = scale_load_mode
    x_scale_via_lds = x_scale_load_mode == "swizzle" and has_x_block_scale
    w_scale_via_lds = w_scale_load_mode == "swizzle" and has_w_block_scale

    if has_w_block_scale:
        w_scale3 = w_scale if w_scale.ndim == 3 else w_scale.unsqueeze(0)
        w_scale_proc3 = _preprocess_scale(w_scale3, w_scale_load_mode)
        stride_wse = w_scale_proc3.stride(0)
        stride_wsn, stride_wsk = _scale_strides(w_scale_proc3, w_scale_load_mode)
        w_scale_buf = w_scale_proc3
    else:
        stride_wse = stride_wsn = stride_wsk = 0
        w_scale_buf = _make_dummy(x.device, torch.uint8)

    x_scale_proc = (
        _preprocess_scale(x_scale, x_scale_load_mode) if has_x_block_scale else None
    )
    stride_xsm, stride_xsk = _scale_strides(x_scale_proc, x_scale_load_mode)

    x_scale_buf = (
        x_scale_proc if x_scale_proc is not None else _make_dummy(x.device, torch.uint8)
    )

    if use_slice_mn:
        NUM_SUBTILES = (2, 2, 1)
    elif use_slice_n:
        NUM_SUBTILES = (1, 2, 1)
    else:
        NUM_SUBTILES = (1, 1, 1)
    EVEN_K = K % block_k == 0
    K_ITERS = (K + block_k - 1) // block_k

    needs_scale_load = apply_x_global_scale and not has_x_block_scale
    if not needs_scale_load:
        x_global_scale_buf = _make_dummy(x.device, torch.float32)
    elif isinstance(x_global_scale, torch.Tensor):
        # Production: zero-copy passthrough of the precision config scale.
        scale_view = x_global_scale.detach().reshape(-1)[:1]
        if scale_view.device == x.device and scale_view.dtype == torch.float32:
            x_global_scale_buf = scale_view
        else:
            x_global_scale_buf = scale_view.to(device=x.device, dtype=torch.float32)
    else:
        x_global_scale_buf = torch.tensor(
            [float(x_global_scale)], dtype=torch.float32, device=x.device
        )

    has_fp8_quant_out = out_quant_scale is not None
    if has_fp8_quant_out:
        if isinstance(out_quant_scale, torch.Tensor):
            qscale_view = out_quant_scale.detach().reshape(-1)[:1]
            if qscale_view.device == x.device and qscale_view.dtype == torch.float32:
                out_quant_scale_buf = qscale_view
            else:
                out_quant_scale_buf = qscale_view.to(
                    device=x.device, dtype=torch.float32
                )
        else:
            out_quant_scale_buf = torch.tensor(
                [float(out_quant_scale)], dtype=torch.float32, device=x.device
            )
        assert y.dtype == torch.float8_e4m3fn, (
            f"out_quant_scale requires a float8_e4m3fn output buffer, "
            f"got y.dtype={y.dtype}"
        )
        if not swiglu:
            raise ValueError(
                "out_quant_scale is currently only wired through the SwiGLU "
                "epilogue (GEMM1 fused quant). For combine-GEMM (DO_SWIGLU=False) "
                "quant, see follow-up P0-1 step 5."
            )
    else:
        out_quant_scale_buf = _make_dummy(x.device, torch.float32)

    # Common args / constexprs shared by both kernel entries.
    common_args = (
        x,
        w3,
        x_scale_buf,
        w_scale_buf,
        bias_buf,
        y,
        gather_buf,
        scatter_buf,
        gate_scal_buf,
        slice_offs_buf,
        slice_sizes_buf,
    )
    common_strides = (
        x.stride(-2),
        x.stride(-1),
        w3.stride(0),
        stride_wn,
        stride_wk,
        stride_xsm,
        stride_xsk,
        stride_wse,
        stride_wsn,
        stride_wsk,
        y.stride(-1),
        y.stride(-2),
        bias.stride(0) if bias is not None else 0,
        bias.stride(-1) if bias is not None else 0,
    )
    common_dims = (
        M,
        M_X,
        N,
        K,
        x_global_scale_buf,
        out_quant_scale_buf,
        num_tiles_total,
    )
    common_kwargs = dict(
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCKS_PER_EXPERT=blocks_per_expert,
        X_FORMAT=x_format,
        W_FORMAT=w_format,
        UPCAST_INDICES=False,
        HAS_X_BLOCK_SCALE=has_x_block_scale,
        HAS_W_BLOCK_SCALE=has_w_block_scale,
        HAS_BIAS=bias is not None,
        HAS_GATHER=gather_indx is not None,
        HAS_SCATTER=scatter_indx is not None,
        DO_SWIGLU=swiglu is not None,
        SWIGLU_ALPHA=float(swiglu_alpha),
        SWIGLU_LIMIT=float(swiglu_limit),
        OUT_BLOCK_N=out_block_n,
        APPLY_GATE_SCAL=gate_scal is not None,
        HAS_RAGGED_OFFS=has_ragged_offs,
        NUM_WARPS=num_warps,
        NUM_BUFFERS=num_buffers,
        SCALE_LOAD_MODE=scale_load_mode,
        W_TRANSPOSE=w_transpose,
        NUM_SUBTILES=NUM_SUBTILES,
        EVEN_K=EVEN_K,
        K_ITERS=K_ITERS,
        N_CONST=N if w_preshuffle else 0,
        Y_N_CONST=int(y_n_const),
        APPLY_X_GLOBAL_SCALE=apply_x_global_scale,
        USE_WARP_PIPELINE=use_warp_pipeline,
        USE_SLICE_MN=use_slice_mn,
        USE_SLICE_N=use_slice_n,
        HAS_FP8_QUANT_OUT=has_fp8_quant_out,
        W_PRESHUFFLED=w_preshuffle,
        W_VIA_VGPR=False,
        W_PREFETCH=False,
        W_CACHE_CG=bool(w_cache_cg),
        X_SCALE_VIA_LDS=bool(x_scale_via_lds),
        W_SCALE_VIA_LDS=bool(w_scale_via_lds),
        USE_NARROW_N_STORE_LAYOUT=bool(use_narrow_n_store_layout),
        GRID_N=grid_n,
        GROUP_M=group_m,
        XCD_SWIZZLE=xcd_swizzle,
        num_warps=num_warps,
    )

    common_kwargs["waves_per_eu"] = num_warps // 4

    if medium_decode_dispatch or medium_decode_combine:
        # M=8/16 decode reuses the regular launcher preprocessing, but selects
        # the single-buffer direct-load body under a constexpr. Keep the
        # identity swizzles/grid shape the direct kernels were tuned for and
        # avoid the occupancy hint that bloats the medium body's VGPR footprint.
        medium_kwargs = dict(common_kwargs)
        medium_kwargs["GROUP_M"] = 1
        medium_kwargs["XCD_SWIZZLE"] = 1
        medium_kwargs["num_stages"] = 1
        medium_kwargs.pop("waves_per_eu", None)
        k = _pipelined_moe_kernel_scaled[(num_tiles_total,)](
            *common_args,
            block_offs_buf,
            block_schedule_buf,
            *common_strides,
            *common_dims,
            USE_BLOCK_SCHEDULE=use_block_schedule,
            N_EXPTS_TOT=n_slices,
            IS_MEDIUM_DECODE=True,
            MEDIUM_COMBINE=medium_decode_combine,
            **medium_kwargs,
        )
        _capture_launch_profile(k)
        return

    k = _pipelined_moe_kernel_scaled[grid](
        *common_args,
        block_offs_buf,
        block_schedule_buf,
        *common_strides,
        *common_dims,
        USE_BLOCK_SCHEDULE=use_block_schedule,
        N_EXPTS_TOT=n_slices,
        **common_kwargs,
    )

    _capture_launch_profile(k)


# CDNA4 MFMA scaled = 16x16x128.
_MFMA_SCALED_K = 128
_MFMA_M = 16


def _round_up_int(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _clamp_block_m(block_m: int, M: int) -> int:
    target = max(_MFMA_M, min(block_m, _round_up_int(M, _MFMA_M)))
    return 1 << (target.bit_length() - 1)


def _ragged_slice_size(a_ragged_metadata, M: int) -> int | None:
    """Per-expert M hint for autotune (mirrors upstream
    ``opt_flags_amd``'s formula). Returns ``None`` on no metadata."""
    if a_ragged_metadata is None:
        return None
    expected = getattr(a_ragged_metadata, "expected_slice_size", None)
    if expected is not None:
        return int(expected)
    try:
        n_slices = int(a_ragged_metadata.slice_sizes.shape[0])
    except (AttributeError, IndexError):
        return None
    return max(1, M // max(1, n_slices))


def _autotune_block(
    M: int,
    N: int,
    K: int,
    *,
    do_swiglu: bool = False,
    ragged: bool = False,
    x_format: str = "e2m1",
    scale_load_mode: str = "transpose",
    slice_size: int | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    use_slice_n: bool | None = None,
    large_slice_size: int | None = None,
    large_m: int | None = None,
) -> tuple[int, int, int, int, bool | None, bool]:
    """Pick the scaled-MFMA tile route.

    Sweep-tuned on gpt-oss-120b (H=I=2880, E=128, top_k=4) at MI355.
    Tiers off logical ``M`` and the per-expert ``slice_size`` hint;
    ``BLOCK_K`` must be a multiple of 128 (MFMA 16x16x128).
    """
    del ragged
    is_fp8 = x_format == "e4m3"
    if slice_size is not None and slice_size <= 16:
        # tinny ragged decode
        bm, bn, bk, nw = 64, 128, 256, 4
    elif slice_size is not None and slice_size <= 64 and M <= 8192:
        # mid ragged decode
        bm, bn, bk, nw = 64, 128, 256, 4
    elif M <= 512:
        bm, bn, bk, nw = 64, 128, 512, 8
    elif is_fp8:
        # fp8 X is 1 byte/elem (lower VGPR pressure); prefill promotes
        # to (128, 256, 256, NW=4) -- sliceMN sweet spot for dispatch.
        if M <= 8192:
            # combine + preshuffled W requires NW=4 (LinearLayout bases);
            # dispatch tolerates NW=8 since OUT_BLOCK_N halving sidesteps
            # the BN=256 / SLICE_N constraint at the BN=256 tile.
            bm, bn, bk, nw = (64, 256, 128, 8) if do_swiglu else (64, 256, 128, 4)
        elif do_swiglu:
            # Preshuffled dispatch may lower BM in the launcher so BN=256 can
            # use SliceN over two 128-wide packed half-tiles.
            bm, bn, bk, nw = 128, 256, 128, 4
        else:
            # combine path: keep BN=256 throughput but force BM<=64
            # so ``_resolve_use_slice_n`` enables USE_SLICE_N=True
            # (half-tile path), which the preshuffled-W static_assert
            # explicitly accepts. NW=4 also required.
            bm, bn, bk, nw = 64, 256, 128, 4
    else:
        # mxfp4 X dequant adds VGPR pressure; same sliceMN sweet spot
        # at the prefill tier.
        if M <= 8192:
            bm, bn, bk, nw = 64, 256, 256, 4
        elif do_swiglu:
            bm, bn, bk, nw = 128, 256, 256, 4
        else:
            bm, bn, bk, nw = 64, 256, 256, 4
    # Clamp tile to actual shape (avoid over-tile + NaN-padded
    # reduction on tiny test shapes).
    bm = _clamp_block_m(bm, M)
    bn = max(_MFMA_M, min(bn, _round_up_int(N, _MFMA_M)))
    bk = max(_MFMA_SCALED_K, min(bk, _round_up_int(K, _MFMA_SCALED_K)))
    # Swizzle unswizzle reshape requires BLOCK_K_S >= 8 (= BLOCK_K
    # >= 256 with SCALE_BLOCK=32).
    if scale_load_mode == "swizzle":
        bk = max(bk, 256)
        bk = min(bk, _round_up_int(K, _MFMA_SCALED_K))

    requested_block_m = block_m
    requested_block_n = block_n
    block_m = block_m or bm
    block_n = block_n or bn
    block_k = block_k or bk

    use_small_m = (
        requested_block_m is None
        and slice_size is not None
        and slice_size < 16
        and 1024 <= M < 2048
    )
    use_medium_m = (
        requested_block_m is None
        and slice_size is not None
        and slice_size <= 16
        and 2048 <= M < 4096
    )
    use_large_m = (
        requested_block_m is None
        and slice_size is not None
        and large_slice_size is not None
        and large_m is not None
        and slice_size >= large_slice_size
        and M >= large_m
    )
    if use_small_m:
        block_m = 16
    elif use_medium_m:
        block_m = 32
    elif use_large_m:
        block_m = 128
        if use_slice_n is None:
            use_slice_n = True

    if requested_block_n is None and block_n == 128 and N >= 256:
        # The tuned prefill route consumes a 256-wide execution tile through the
        # SliceN pipeline.  Preshuffled W reads two 128-wide packed half-tiles;
        # non-preshuffled W reads the same two half-tiles from the normal LDS
        # layout, keeping the compute schedule aligned across both variants.
        block_n = 256

    if block_n == 256 and block_m > 64 and not use_large_m:
        block_m = 64

    return block_m, block_n, block_k, nw, use_slice_n, use_small_m


def _autotune_pid_swizzle(
    num_tiles_total: int,
    grid_n: int,
    grid_m_padded: int,
    block_m: int,
) -> tuple[int, int]:
    if num_tiles_total < 256:
        return 1, 1
    if block_m < 32:
        return 1, 1
    if grid_m_padded >= 8 and grid_m_padded % 4 == 0:
        group_m = 4
    elif grid_m_padded >= 2 and grid_m_padded % 2 == 0:
        group_m = 2
    else:
        group_m = 1
    xcd_swizzle = _CDNA4_NUM_XCDS if num_tiles_total % _CDNA4_NUM_XCDS == 0 else 1
    return group_m, xcd_swizzle


_LaunchTuning = tuple[int | None, int | None, bool | None, bool]


def _default_prefill_launch_tuning() -> _LaunchTuning:
    return 1, None, None, False


def _dispatch_prefill_launch_tuning(m: int) -> _LaunchTuning:
    if m <= 1024:
        return 1, _CDNA4_NUM_XCDS, None, False
    if m <= 2048:
        return 1, 4, None, False
    if m <= 4096:
        return 1, _CDNA4_NUM_XCDS, True, False
    return _default_prefill_launch_tuning()


def _combine_prefill_launch_tuning(m: int) -> _LaunchTuning:
    if m <= 1024:
        return 1, _CDNA4_NUM_XCDS, None, False
    if m <= 2048:
        return 1, 4, None, False
    if m <= 4096:
        return 1, _CDNA4_NUM_XCDS, True, False
    if m < 16384:
        return 1, 4, None, False
    return 1, 4, None, True


def _prefill_launch_tuning(
    op: str,
    *,
    m: int,
    use_slice_mn: bool,
) -> _LaunchTuning:
    if use_slice_mn:
        return _default_prefill_launch_tuning()

    if op == "dispatch":
        return _dispatch_prefill_launch_tuning(m)
    if op == "combine":
        return _combine_prefill_launch_tuning(m)
    return _default_prefill_launch_tuning()


def _persistent_grid_size(num_tiles_total: int) -> int:
    if num_tiles_total <= 0:
        return 1
    return max(1, min(num_tiles_total, _CDNA4_NUM_CUS * _PERSISTENT_OVERSUBSCRIBE))


def _needs_scale_lds(
    x_format: str, has_x_block_scale: bool, has_w_block_scale: bool
) -> bool:
    return (has_x_block_scale and x_format == "e2m1") or has_w_block_scale


def _can_use_slice_n(
    bm: int,
    bn: int,
    *,
    scale_load_mode: str,
    x_format: str,
    has_x_block_scale: bool,
    has_w_block_scale: bool,
) -> bool:
    if bn < 256 or bm < 16 or (bn // 2) % 64 != 0:
        return False
    if _needs_scale_lds(x_format, has_x_block_scale, has_w_block_scale):
        return scale_load_mode == "swizzle"
    return True


def _resolve_use_slice_n(
    use_slice_n: bool | None,
    bm: int,
    bn: int,
    *,
    scale_load_mode: str,
    x_format: str,
    has_x_block_scale: bool,
    has_w_block_scale: bool,
    bk: int,
) -> bool:
    if use_slice_n is not None:
        enabled = bool(use_slice_n)
    else:
        w_bytes = (bn * bk) // 2
        enabled = bn >= 256 and bm <= 64 and w_bytes >= _TCP_INFLIGHT_CAP_BYTES
    return enabled and _can_use_slice_n(
        bm,
        bn,
        scale_load_mode=scale_load_mode,
        x_format=x_format,
        has_x_block_scale=has_x_block_scale,
        has_w_block_scale=has_w_block_scale,
    )


def _can_use_slice_mn(
    bm: int,
    bn: int,
    *,
    num_buffers: int,
    scale_load_mode: str,
    x_format: str,
    has_x_block_scale: bool,
    has_w_block_scale: bool,
) -> bool:
    if bm < 128 or bn < 128:
        return False
    if (bm // 2) % 64 != 0 or (bn // 2) % 64 != 0:
        return False
    if num_buffers < 2:
        return False
    if _needs_scale_lds(x_format, has_x_block_scale, has_w_block_scale):
        return scale_load_mode == "swizzle"
    return True


def _resolve_use_slice_mn(
    use_slice_mn: bool | None,
    bm: int,
    bn: int,
    *,
    num_buffers: int,
    scale_load_mode: str,
    x_format: str,
    has_x_block_scale: bool,
    has_w_block_scale: bool,
    use_slice_n: bool = False,
    bk: int,
) -> bool:
    if use_slice_n:
        return False
    if use_slice_mn is not None:
        enabled = bool(use_slice_mn)
    else:
        w_bytes = (bn * bk) // 2 if x_format == "e2m1" else bn * bk
        enabled = bm >= 128 and bn >= 128 and w_bytes >= 16 * 1024 and (bm + bn) >= 384
    return enabled and _can_use_slice_mn(
        bm,
        bn,
        num_buffers=num_buffers,
        scale_load_mode=scale_load_mode,
        x_format=x_format,
        has_x_block_scale=has_x_block_scale,
        has_w_block_scale=has_w_block_scale,
    )


def _preshuffled_layout_block_n(w: torch.Tensor) -> int:
    block_n = int(getattr(w, "gluon_dot_block_n", 128))
    if block_n <= 0 or block_n % _GLUON_DOT_N_LANE != 0:
        raise ValueError(
            f"invalid preshuffled Gluon W layout block_n={block_n}; "
            f"expected a positive multiple of {_GLUON_DOT_N_LANE}"
        )
    return block_n


def _align_block_n_to_preshuffled_layout(
    w: torch.Tensor,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    scale_load_mode: str,
    x_format: str,
    has_x_block_scale: bool,
    has_w_block_scale: bool,
    use_slice_mn: bool | None,
    use_slice_n: bool | None,
) -> tuple[int, bool, bool | None]:
    """Constrain the execution tile to the host-preshuffled W layout.

    The current preshuffled kernel supports the 128-wide packed layout. A
    256-wide execution tile is still legal by consuming two adjacent 128-wide
    packed half-tiles through USE_SLICE_N.
    """
    packed_block_n = _preshuffled_layout_block_n(w)
    if packed_block_n != 128:
        raise ValueError(
            f"preshuffled Gluon W layout block_n={packed_block_n} is not "
            "supported by the current preshuffled W load layout"
        )

    # USE_SLICE_MN is not wired for the preshuffled W descriptor.
    use_slice_mn = False

    if block_n == packed_block_n:
        return block_n, use_slice_mn, use_slice_n

    auto_slice_n = (
        block_n >= 256
        and block_m <= 64
        and ((block_n * block_k) // 2) >= _TCP_INFLIGHT_CAP_BYTES
    )
    wants_slice_n = use_slice_n is True or (use_slice_n is None and auto_slice_n)
    can_use_slice_n = wants_slice_n and _can_use_slice_n(
        block_m,
        block_n,
        scale_load_mode=scale_load_mode,
        x_format=x_format,
        has_x_block_scale=has_x_block_scale,
        has_w_block_scale=has_w_block_scale,
    )

    if block_n == 2 * packed_block_n and can_use_slice_n:
        return block_n, use_slice_mn, use_slice_n

    return packed_block_n, use_slice_mn, False


def _resolve_prefill_slice_modes(
    *,
    use_slice_mn: bool | None,
    use_slice_n: bool | None,
    block_m: int,
    block_n: int,
    block_k: int,
    num_buffers: int,
    scale_load_mode: str,
    x_format: str,
    has_x_block_scale: bool,
    has_w_block_scale: bool,
) -> tuple[bool, bool]:
    if use_slice_mn is True:
        use_slice_mn_resolved = _resolve_use_slice_mn(
            True,
            block_m,
            block_n,
            num_buffers=num_buffers,
            scale_load_mode=scale_load_mode,
            x_format=x_format,
            has_x_block_scale=has_x_block_scale,
            has_w_block_scale=has_w_block_scale,
            bk=block_k,
        )
        if use_slice_mn_resolved:
            return True, False

    use_slice_n_resolved = _resolve_use_slice_n(
        use_slice_n,
        block_m,
        block_n,
        scale_load_mode=scale_load_mode,
        x_format=x_format,
        has_x_block_scale=has_x_block_scale,
        has_w_block_scale=has_w_block_scale,
        bk=block_k,
    )
    use_slice_mn_resolved = _resolve_use_slice_mn(
        use_slice_mn,
        block_m,
        block_n,
        num_buffers=num_buffers,
        scale_load_mode=scale_load_mode,
        x_format=x_format,
        has_x_block_scale=has_x_block_scale,
        has_w_block_scale=has_w_block_scale,
        use_slice_n=use_slice_n_resolved,
        bk=block_k,
    )
    return use_slice_mn_resolved, use_slice_n_resolved


def gluon_mxfp_dispatch_swiglu(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    x_scale: torch.Tensor | None = None,
    x_format: str = "e2m1",
    x_global_scale: torch.Tensor | float = 1.0,
    bias: torch.Tensor | None,
    a_ragged_metadata,
    gather_indx,
    out_dtype: torch.dtype = torch.bfloat16,
    swiglu_alpha: float = 1.0,
    swiglu_limit: float = 0.0,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
    num_buffers: int = 2,
    use_warp_pipeline: bool | None = None,
    use_slice_mn: bool | None = None,
    use_slice_n: bool | None = None,
    scale_load_mode: str = "transpose",
    w_transpose: bool = False,
    persistent: bool | None = None,
    num_ctas: int | None = None,
    out_quant_scale: torch.Tensor | float | None = None,
    w_preshuffle: bool = False,
) -> torch.Tensor:
    assert w.ndim == 3 and w.shape[-1] % 2 == 0
    M_X = int(x.shape[-2])
    if gather_indx is not None:
        gather_t = (
            gather_indx.src_indx if hasattr(gather_indx, "src_indx") else gather_indx
        )
        M = int(gather_t.shape[0])
    else:
        M = x.shape[-2]
    N = w.shape[-1]
    div_x = 2 if x_format == "e2m1" else 1
    K = x.shape[-1] * div_x
    requested_block_m = block_m
    requested_block_n = block_n
    requested_block_k = block_k
    slice_size = _ragged_slice_size(a_ragged_metadata, M)
    (
        block_m,
        block_n,
        block_k,
        nw,
        use_slice_n,
        use_small_prefill_m,
    ) = _autotune_block(
        M,
        N,
        K,
        do_swiglu=True,
        x_format=x_format,
        scale_load_mode=scale_load_mode,
        slice_size=slice_size,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        use_slice_n=use_slice_n,
        large_slice_size=128,
        large_m=16384,
    )
    medium_decode_dispatch_shape = (
        8 <= M_X <= 16
        and a_ragged_metadata is not None
        and gather_indx is not None
        and out_quant_scale is not None
        and x_format == "e4m3"
        and x_scale is None
        and scale_load_mode == "swizzle"
        and w_transpose
        and requested_block_m is None
        and requested_block_n is None
        and requested_block_k is None
        and not w_preshuffle
        and N % 128 == 0
    )
    if medium_decode_dispatch_shape:
        block_m, block_n, block_k, nw = 16, 128, 256, 4
        use_slice_n = False
        use_slice_mn = False
        use_small_prefill_m = False
    if w_preshuffle:
        block_n, use_slice_mn, use_slice_n = _align_block_n_to_preshuffled_layout(
            w,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            scale_load_mode=scale_load_mode,
            x_format=x_format,
            has_x_block_scale=x_format == "e2m1",
            has_w_block_scale=True,
            use_slice_mn=use_slice_mn,
            use_slice_n=use_slice_n,
        )
    num_warps = num_warps or nw
    if w_preshuffle:
        num_warps = 4
    use_warp_pipeline = (
        bool(use_warp_pipeline) if use_warp_pipeline is not None else False
    )
    use_slice_mn, use_slice_n = _resolve_prefill_slice_modes(
        use_slice_mn=use_slice_mn,
        use_slice_n=use_slice_n,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_buffers=num_buffers,
        scale_load_mode=scale_load_mode,
        x_format=x_format,
        has_x_block_scale=x_format == "e2m1",
        has_w_block_scale=True,
    )
    medium_decode_dispatch_eligible = (
        medium_decode_dispatch_shape
        and block_m == 16
        and block_n == 128
        and block_k == 256
        and not use_slice_mn
        and not use_slice_n
        and not w_preshuffle
    )
    if persistent is None and use_small_prefill_m:
        persistent = False
    if persistent is None and use_slice_n:
        grid_n = (N + block_n - 1) // block_n
        if a_ragged_metadata is not None:
            n_slices = int(a_ragged_metadata.slice_sizes.shape[0])
            grid_m_upper = RaggedTensorMetadata.n_blocks(n_slices, M, block_m)
        else:
            grid_m_upper, _ = _dense_grid_dims(M, block_m)
        persistent = (grid_m_upper * grid_n) >= _PERSISTENT_TILES_THRESHOLD
        if persistent and num_ctas is None:
            num_ctas = _CDNA4_NUM_CUS
    group_m, xcd_swizzle, w_cache_cg, _ = _prefill_launch_tuning(
        "dispatch",
        m=M,
        use_slice_mn=use_slice_mn,
    )
    out_block_n = block_n // 2
    y_dtype = torch.float8_e4m3fn if out_quant_scale is not None else out_dtype
    y = torch.empty((M, N // 2), device=x.device, dtype=y_dtype)
    _launch_kernel(
        x,
        w,
        y=y,
        bias=bias,
        gather_indx=gather_indx,
        scatter_indx=None,
        gate_scal=None,
        a_ragged_metadata=a_ragged_metadata,
        swiglu=(float(swiglu_alpha), float(swiglu_limit)),
        out_block_n=out_block_n,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        x_format=x_format,
        w_format="e2m1",
        x_scale=x_scale,
        w_scale=w_scale,
        x_global_scale=x_global_scale,
        scale_load_mode=scale_load_mode,
        w_transpose=w_transpose,
        num_buffers=num_buffers,
        use_warp_pipeline=use_warp_pipeline,
        use_slice_mn=use_slice_mn,
        use_slice_n=use_slice_n,
        persistent=persistent,
        num_ctas=num_ctas,
        group_m=group_m,
        xcd_swizzle=xcd_swizzle,
        out_quant_scale=out_quant_scale,
        w_preshuffle=w_preshuffle,
        w_cache_cg=w_cache_cg,
        medium_decode_dispatch=medium_decode_dispatch_eligible,
    )
    return y


def gluon_mxfp_combine(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    x_scale: torch.Tensor | None = None,
    x_format: str = "e2m1",
    x_global_scale: torch.Tensor | float = 1.0,
    bias: torch.Tensor | None,
    a_ragged_metadata,
    scatter_indx,
    gate_scal: torch.Tensor | None = None,
    n_tokens: int | None = None,
    n_expts_act: int | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
    num_buffers: int = 2,
    use_warp_pipeline: bool | None = None,
    use_slice_mn: bool | None = None,
    use_slice_n: bool | None = None,
    scale_load_mode: str = "transpose",
    w_transpose: bool = False,
    persistent: bool | None = None,
    num_ctas: int | None = None,
    w_preshuffle: bool = False,
) -> torch.Tensor:
    assert w.ndim == 3
    M = x.shape[-2]
    N = w.shape[-1]
    div_x = 2 if x_format == "e2m1" else 1
    K = x.shape[-1] * div_x
    requested_block_m = block_m
    requested_block_n = block_n
    requested_block_k = block_k
    slice_size = _ragged_slice_size(a_ragged_metadata, M)
    (
        block_m,
        block_n,
        block_k,
        nw,
        use_slice_n,
        use_small_prefill_m,
    ) = _autotune_block(
        M,
        N,
        K,
        ragged=a_ragged_metadata is not None,
        x_format=x_format,
        scale_load_mode=scale_load_mode,
        slice_size=slice_size,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        use_slice_n=use_slice_n,
        large_slice_size=256,
        large_m=32768,
    )
    medium_decode_combine_shape = (
        a_ragged_metadata is not None
        and scatter_indx is not None
        and gate_scal is not None
        and n_tokens in (8, 16)
        and n_expts_act is not None
        and x_format == "e4m3"
        and x_scale is None
        and scale_load_mode == "swizzle"
        and w_transpose
        and requested_block_m is None
        and requested_block_n is None
        and requested_block_k is None
        and not w_preshuffle
        and N % 128 == 0
    )
    if medium_decode_combine_shape:
        block_m, block_n, block_k, nw = 16, 128, 256, 4
        use_slice_n = False
        use_slice_mn = False
        use_small_prefill_m = False
    if w_preshuffle:
        block_n, use_slice_mn, use_slice_n = _align_block_n_to_preshuffled_layout(
            w,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            scale_load_mode=scale_load_mode,
            x_format=x_format,
            has_x_block_scale=x_format == "e2m1",
            has_w_block_scale=True,
            use_slice_mn=use_slice_mn,
            use_slice_n=use_slice_n,
        )
    num_warps = num_warps or nw
    if w_preshuffle:
        num_warps = 4
    use_warp_pipeline = (
        bool(use_warp_pipeline) if use_warp_pipeline is not None else False
    )
    use_slice_mn, use_slice_n = _resolve_prefill_slice_modes(
        use_slice_mn=use_slice_mn,
        use_slice_n=use_slice_n,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_buffers=num_buffers,
        scale_load_mode=scale_load_mode,
        x_format=x_format,
        has_x_block_scale=x_format == "e2m1",
        has_w_block_scale=True,
    )
    medium_decode_combine_eligible = (
        medium_decode_combine_shape
        and block_m == 16
        and block_n == 128
        and block_k == 256
        and not use_slice_mn
        and not use_slice_n
        and not w_preshuffle
    )
    if persistent is None and use_small_prefill_m:
        persistent = False
    if persistent is None:
        grid_n = (N + block_n - 1) // block_n
        if a_ragged_metadata is not None:
            n_slices = int(a_ragged_metadata.slice_sizes.shape[0])
            grid_m_upper = RaggedTensorMetadata.n_blocks(n_slices, M, block_m)
        else:
            grid_m_upper, _ = _dense_grid_dims(M, block_m)
        persistent = (grid_m_upper * grid_n) >= _PERSISTENT_TILES_THRESHOLD
        if persistent and num_ctas is None:
            num_ctas = _CDNA4_NUM_CUS
    group_m, xcd_swizzle, w_cache_cg, use_narrow_n_store_layout = (
        _prefill_launch_tuning(
            "combine",
            m=M,
            use_slice_mn=use_slice_mn,
        )
    )
    n_act_eff = int(n_expts_act) if n_expts_act is not None else 1
    if n_tokens is None:
        n_rows = M
        n_tokens_eff = M
    else:
        n_tokens_eff = int(n_tokens)
        n_rows = n_tokens_eff * n_act_eff
    # W may be padded in N to satisfy the packed layout. Keep padded N for
    # tiling/W-scale reads, but store only the caller-visible width.
    logical_n = int(getattr(w, "original_n", N))
    y_n = logical_n if logical_n < N else N
    y = torch.empty((n_rows, y_n), device=x.device, dtype=out_dtype)
    _launch_kernel(
        x,
        w,
        y=y,
        bias=bias,
        gather_indx=None,
        scatter_indx=scatter_indx,
        gate_scal=gate_scal,
        a_ragged_metadata=a_ragged_metadata,
        swiglu=None,
        out_block_n=block_n,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        x_format=x_format,
        w_format="e2m1",
        x_scale=x_scale,
        w_scale=w_scale,
        x_global_scale=x_global_scale,
        scale_load_mode=scale_load_mode,
        w_transpose=w_transpose,
        num_buffers=num_buffers,
        use_warp_pipeline=use_warp_pipeline,
        use_slice_mn=use_slice_mn,
        use_slice_n=use_slice_n,
        persistent=persistent,
        num_ctas=num_ctas,
        group_m=group_m,
        xcd_swizzle=xcd_swizzle,
        w_preshuffle=w_preshuffle,
        y_n_const=y_n if y_n != N else 0,
        w_cache_cg=w_cache_cg,
        use_narrow_n_store_layout=use_narrow_n_store_layout,
        medium_decode_combine=medium_decode_combine_eligible,
    )
    if n_act_eff > 1:
        if medium_decode_combine_eligible:
            # Fused top-k reduction over the scatter rows (graph-capturable).
            # The TOPK partials for a token are consecutive rows of y.
            y_reduced = torch.empty(
                (n_tokens_eff, y_n), device=x.device, dtype=out_dtype
            )
            R_BLOCK_N = 256
            r_grid = (n_tokens_eff * ((y_n + R_BLOCK_N - 1) // R_BLOCK_N),)
            _moe_partial_reduce[r_grid](
                y,
                y_reduced,
                n_tokens_eff,
                y_n,
                y.stride(0),
                n_act_eff * y.stride(0),
                y.stride(1),
                y_reduced.stride(0),
                y_reduced.stride(1),
                SPLIT_K=n_act_eff,
                BLOCK_N=R_BLOCK_N,
                num_warps=1,
            )
            y = y_reduced
        else:
            y = y.view(n_tokens_eff, n_act_eff, y_n).sum(dim=1)
    # Unpad N if the caller padded W for w_preshuffle. Padded W bytes
    # are 0 and padded scales are 127 so acc[:, N:N_padded] == 0.
    if logical_n != y.shape[-1]:
        y = y[..., :logical_n].contiguous()
    return y


_TUNING_KW = frozenset(
    {"block_m", "block_n", "block_k", "num_warps", "num_buffers", "dtype"}
)

# Gluon-only kwargs; explicitly stripped before forwarding upstream.
_GLUON_PRIVATE_KW = frozenset({"out_quant_scale"})


def _extract_gluon_raw_w(w):
    """Return the raw ``(E, K_packed, N) uint8`` W tensor.

    The upstream wrapper's ``storage.data`` is already K-contiguous
    so we pass it through. If a ``_gluon_shuffled`` attribute is
    attached (set by the backend's preshuffle hook) we return the
    shuffled view instead -- ``is_shuffled_for_gluon_dot=True`` then
    triggers the kernel's preshuffled W path.
    """
    if isinstance(w, torch.Tensor):
        shuffled = getattr(w, "_gluon_shuffled", None)
        if shuffled is not None:
            return shuffled
        return w
    raw = _wrapped_tensor_data(w)
    if raw is None:
        return w
    shuffled = getattr(raw, "_gluon_shuffled", None)
    if shuffled is not None:
        return shuffled
    return raw


def _extract_gluon_raw_w_unshuffled(w):
    """Return the canonical K-contiguous W storage, ignoring preshuffle attrs.

    M=8/16 medium-decode uses the direct-load body and must not be routed to the
    default preshuffled-W path. This helper preserves the main path's
    ``_extract_gluon_raw_w`` behavior by being opt-in at the call site.
    """
    if isinstance(w, torch.Tensor):
        return w
    raw = _wrapped_tensor_data(w)
    return w if raw is None else raw


def _extract_gluon_raw_s(s):
    """Return the raw uint8 scale tensor for Gluon's ``swizzle`` mode
    (bit-equivalent to upstream CDNA4MXScaleLayout.swizzle_data)."""
    if isinstance(s, torch.Tensor):
        return s
    raw = _wrapped_tensor_data(s)
    return s if raw is None else raw


def _maybe_extract_swiglu_args(fused_activation):
    """Pull ``(alpha, limit)`` from an upstream ``FusedActivation`` object
    representing SwiGLU. Returns ``None`` for any other activation."""
    if fused_activation is None:
        return None
    specs = getattr(fused_activation, "specs", None)
    fn_name = getattr(specs, "name", None) if specs is not None else None
    if fn_name != "swiglu":
        return None
    args = getattr(fused_activation, "fn_args", None)
    if args is None:
        args = getattr(fused_activation, "args", None)
    if args is None or len(args) < 2:
        return None
    return float(args[0]), float(args[1])


def _global_scale_passthrough(scale):
    """Return the flex scale in a form the launcher can take without
    a host ``.item()`` (keeps HIP-graph capture working)."""
    if scale is None:
        return 1.0
    if isinstance(scale, torch.Tensor):
        return scale
    return float(scale)


def gluon_mxfp_ragged_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    w_mx_scale: torch.Tensor,
    x_global_scale: Optional[torch.Tensor] = None,
    x_mx_scale: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    x_format: str = "e4m3",
    a_ragged_metadata=None,
    gather_indx=None,
    scatter_indx=None,
    fused_activation=None,
    n_tokens=None,
    n_expts_act=None,
    **extra_kwargs,
) -> torch.Tensor | None:
    assert w_mx_scale is not None

    if x_format == "e4m3":
        x_global_scale = _global_scale_passthrough(x_global_scale)
        x_view = x.view(torch.uint8) if x.dtype != torch.uint8 else x
        x_scale = None
    elif x_format == "e2m1":
        if x_mx_scale is None:
            raise RuntimeError("x_mx_scale is required for e2m1 input X")
        x_global_scale = 1.0
        x_view = x.view(torch.uint8) if x.dtype != torch.uint8 else x
        x_scale = _extract_gluon_raw_s(x_mx_scale)
        assert isinstance(x_scale, torch.Tensor)
    else:
        raise RuntimeError(f"unsupported input X format: {x_format}")

    if out_dtype is None and x.dtype.is_floating_point:
        out_dtype = x.dtype
    elif out_dtype is None:
        out_dtype = torch.bfloat16

    prefer_unshuffled_w = bool(extra_kwargs.get("prefer_unshuffled_w", False))
    w_raw = (
        _extract_gluon_raw_w_unshuffled(w)
        if prefer_unshuffled_w
        else _extract_gluon_raw_w(w)
    )
    s_raw = _extract_gluon_raw_s(w_mx_scale)

    assert isinstance(w_raw, torch.Tensor) and isinstance(s_raw, torch.Tensor)
    assert w_raw.ndim == 3

    # Wrap bare tensors into ``.<attr>``-typed adapters; the launcher
    # consults gather_indx.src_indx / scatter_indx.dst_indx.
    def _adapt_indx(obj, attr):
        if obj is None:
            return None
        if hasattr(obj, attr):
            return obj
        if isinstance(obj, torch.Tensor):
            return type("IndxAdapter", (), {attr: obj})()
        return obj

    gather_indx = _adapt_indx(gather_indx, "src_indx")
    scatter_indx = _adapt_indx(scatter_indx, "dst_indx")

    swiglu_args = _maybe_extract_swiglu_args(fused_activation)
    has_gather = gather_indx is not None
    has_scatter = scatter_indx is not None

    if fused_activation is not None:
        assert swiglu_args is not None, "SwiGLU activation requires swiglu_args"

    gammas = extra_kwargs.get("gammas")
    out_quant_scale = extra_kwargs.get("out_quant_scale")

    if has_scatter and not has_gather:
        # gemm + combine
        w_preshuffle = bool(getattr(w_raw, "is_shuffled_for_gluon_dot", False))
        out = gluon_mxfp_combine(
            x_view,
            w_raw,
            s_raw,
            x_scale=x_scale,
            x_format=x_format,
            x_global_scale=x_global_scale,
            bias=bias,
            a_ragged_metadata=a_ragged_metadata,
            scatter_indx=scatter_indx,
            gate_scal=gammas,
            n_tokens=n_tokens,
            n_expts_act=n_expts_act,
            out_dtype=out_dtype,
            scale_load_mode="swizzle",
            w_transpose=True,
            w_preshuffle=w_preshuffle,
        )
        return out

    if not has_scatter and swiglu_args is not None:
        swiglu_alpha, swiglu_limit = swiglu_args
        w_preshuffle = bool(getattr(w_raw, "is_shuffled_for_gluon_dot", False))
        out = gluon_mxfp_dispatch_swiglu(
            x_view,
            w_raw,
            s_raw,
            x_scale=x_scale,
            x_format=x_format,
            x_global_scale=x_global_scale,
            bias=bias,
            a_ragged_metadata=a_ragged_metadata,
            gather_indx=gather_indx,
            out_dtype=out_dtype,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            scale_load_mode="swizzle",
            w_transpose=True,
            out_quant_scale=out_quant_scale,
            w_preshuffle=w_preshuffle,
        )
        return out


def _gluon_mxfp4_fp8_warp_decode_moe(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w13_weight,
    w2_weight,
    *,
    w13_bias=None,
    w2_bias=None,
    w13_mx_scale: torch.Tensor,
    w2_mx_scale: torch.Tensor,
    w13_act_scale: torch.Tensor,
    w2_act_scale: torch.Tensor,
    out_dtype: torch.dtype,
    top_k: int,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
) -> torch.Tensor | None:
    """Small-M direct warp-decode MoE for GPT-OSS FP8 x MXFP4 path."""
    assert hidden_states.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)

    if hidden_states.ndim != 2 or router_logits.ndim != 2:
        return None
    n_tokens = int(router_logits.shape[0])
    n_experts = int(router_logits.shape[1])
    if n_tokens > WARP_DECODE_MAX_M:
        return None
    if not gluon_route_supported(router_logits, top_k, router_logits.dtype):
        return None

    # Use the optional Gluon dot-layout preshuffled attachments when they match
    # the layout this warp-decode path knows how to consume.
    w13_raw_candidate = _extract_gluon_raw_w(w13_weight)
    w13_preshuffled = (
        isinstance(w13_raw_candidate, torch.Tensor)
        and bool(getattr(w13_raw_candidate, "is_shuffled_for_gluon_dot", False))
        and int(getattr(w13_raw_candidate, "gluon_dot_block_k_pk", 0)) == 128
        and int(getattr(w13_raw_candidate, "gluon_dot_block_n", 0)) == 128
    )
    w13_raw = w13_raw_candidate if w13_preshuffled else _extract_gluon_raw_s(w13_weight)
    w2_raw_candidate = _extract_gluon_raw_w(w2_weight)
    w2_preshuffled = (
        isinstance(w2_raw_candidate, torch.Tensor)
        and bool(getattr(w2_raw_candidate, "is_shuffled_for_gluon_dot", False))
        and int(getattr(w2_raw_candidate, "gluon_dot_block_k_pk", 0)) == 128
        and int(getattr(w2_raw_candidate, "gluon_dot_block_n", 0)) == 128
    )
    w2_raw = w2_raw_candidate if w2_preshuffled else _extract_gluon_raw_s(w2_weight)
    w13_scale = _extract_gluon_raw_s(w13_mx_scale)
    w2_scale = _extract_gluon_raw_s(w2_mx_scale)
    if not all(
        isinstance(t, torch.Tensor) for t in (w13_raw, w2_raw, w13_scale, w2_scale)
    ):
        return None
    if w13_raw.ndim != 3 or w2_raw.ndim != 3:
        return None
    if w13_raw.dtype != torch.uint8 or w2_raw.dtype != torch.uint8:
        return None
    if w13_scale.dtype != torch.uint8 or w2_scale.dtype != torch.uint8:
        return None

    D = int(hidden_states.shape[1])
    w13_k_pk = int(getattr(w13_raw, "original_k_pk", int(w13_raw.shape[1])))
    if w13_k_pk * 2 != D:
        return None
    two_i = int(w13_raw.shape[2])
    if two_i % 2 != 0:
        return None
    if w13_preshuffled and two_i % 128 != 0:
        return None
    i_dim = two_i // 2
    w2_k_pk = int(getattr(w2_raw, "original_k_pk", int(w2_raw.shape[1])))
    if w2_k_pk * 2 != i_dim:
        return None
    w2_n_phys = int(w2_raw.shape[2])
    N = int(getattr(w2_raw, "original_n", w2_n_phys))
    if w2_preshuffled and (N > w2_n_phys or w2_n_phys % 128 != 0):
        return None

    # Stage1 computes the dense top-k inside the kernel; allocate its outputs.
    router_logits_c = router_logits.contiguous()
    topk_ids = torch.empty(
        (n_tokens, top_k), dtype=torch.int32, device=router_logits.device
    )
    topk_weights = torch.empty(
        (n_tokens, top_k), dtype=router_logits.dtype, device=router_logits.device
    )

    # Current GPT-OSS path uses FP8 E4M3 activations with per-tensor scale.
    x_fp8 = hidden_states
    # Pass the FP8 tensor straight to Gluon.  ``view(torch.uint8)`` materializes a
    # copy for float8 tensors on this stack and dominates small-M latency.

    out = torch.empty((n_tokens, N), dtype=out_dtype, device=hidden_states.device)

    # The kernels only read the bias pointer when HAS_BIAS; allocate the
    # placeholder solely for the absent ones.
    dummy_bias = (
        _make_dummy(hidden_states.device, torch.float32, 1)
        if (w13_bias is None or w2_bias is None)
        else None
    )
    b13 = w13_bias if w13_bias is not None else dummy_bias
    b2 = w2_bias if w2_bias is not None else dummy_bias

    BLOCK_K = 128
    S2_BLOCK_N = 32 if n_tokens >= 16 else (8 if n_tokens <= 1 else 16)
    S2_M_DUP = 4
    # Tuned stage2 K=I split factor by batch size (8/4 raise small-M occupancy, off for M>=5).
    if n_tokens <= 2:
        s2_split_k = 8
    elif n_tokens <= 4:
        s2_split_k = 4
    else:
        s2_split_k = 1

    inter = torch.empty(
        (n_tokens * top_k, i_dim), dtype=x_fp8.dtype, device=hidden_states.device
    )
    # Cooperative-LDS, num_warps=4, software-pipelined stage1 -- the smallest-M
    # decode path (this wrapper is only entered for n_tokens <= WARP_DECODE_MAX_M).
    # BLOCK_N=64 maximizes CTAs/CU for non-preshuffled W. Preshuffled W13 is
    # laid out as 128-wide CTA tiles, so the first decode preshuffle path
    # consumes full 128-wide tiles and lets SwiGLU reduce them to 64 columns.
    COOP_NUM_WARPS = 4
    COOP_BLOCK_N = 128 if w13_preshuffled else 64
    COOP_BLOCK_K = 256
    coop_k_iters = (D + COOP_BLOCK_K - 1) // COOP_BLOCK_K
    coop_even_k = D % COOP_BLOCK_K == 0
    # decode_pipeline() prefetches NUM_BUFFERS-1 tiles before the main loop.
    # For short synthetic shapes (for example D=256) the fixed 3-buffer GPT-OSS
    # schedule over-prefetches past the only real K tile. Keep the production
    # 3-buffer schedule when possible, but shrink the pipeline for short K.
    COOP_NUM_BUFFERS = min(3, coop_k_iters + (1 if coop_even_k else 0))
    coop_grid = (n_tokens * ((2 * i_dim + COOP_BLOCK_N - 1) // COOP_BLOCK_N) * top_k,)
    # X is stored as raw i8 in LDS and bitcast to e4m3 in mfma_scaled; pass the
    # uint8 view (an fp8 LDS buffer fails to lower).
    x_uint8 = x_fp8.view(torch.uint8)
    # fmt: off
    _warp_decode_topk_stage1_coop_kernel[coop_grid](
        x_uint8, router_logits_c, w13_raw, w13_scale, topk_ids, topk_weights, inter,
        n_tokens, n_experts, D, i_dim,
        x_uint8.stride(0), x_uint8.stride(1),
        router_logits_c.stride(0), topk_ids.stride(0), topk_weights.stride(0),
        w13_raw.stride(0), w13_raw.stride(-2), w13_raw.stride(-1),
        w13_scale.stride(0), w13_scale.stride(-2), w13_scale.stride(-1),
        inter.stride(0), inter.stride(1),
        w13_act_scale, w2_act_scale, b13,
        D_PACKED=D // 2, TOPK=top_k,
        # EP/TKP: padded widths of the [tokens, experts] logits tile and the
        # top-k selection tile; >= 64*num_warps keeps the blocked layout valid.
        EP=max(_route_next_pow2(n_experts), 64 * COOP_NUM_WARPS), TKP=64 * COOP_NUM_WARPS,
        X_DTYPE=_ROUTE_GL_DTYPE[router_logits.dtype],
        BLOCK_K=COOP_BLOCK_K, BLOCK_N=COOP_BLOCK_N, BLOCK_M=16,
        NUM_BUFFERS=COOP_NUM_BUFFERS, NUM_WARPS=COOP_NUM_WARPS,
        W_PRESHUFFLED=w13_preshuffled,
        EVEN_K=coop_even_k,
        HAS_BIAS=w13_bias is not None,
        SWIGLU_ALPHA=float(swiglu_alpha), SWIGLU_LIMIT=float(swiglu_limit),
        num_warps=COOP_NUM_WARPS,
    )
    # fmt: on

    n_tiles2 = (N + S2_BLOCK_N - 1) // S2_BLOCK_N
    if s2_split_k > 1:
        out_partial = torch.empty(
            (s2_split_k, n_tokens, N), dtype=torch.float32, device=hidden_states.device
        )
        s2_dst = out_partial
        s2_stride_om = out_partial.stride(1)
        s2_stride_on = out_partial.stride(2)
        s2_stride_ok = out_partial.stride(0)
        s2_grid = (n_tokens * n_tiles2 * s2_split_k,)
    else:
        s2_dst = out
        s2_stride_om = out.stride(0)
        s2_stride_on = out.stride(1)
        s2_stride_ok = 0
        s2_grid = (n_tokens * n_tiles2,)
    # fmt: off
    _warp_decode_stage2_fp8_mxfp4_kernel[s2_grid](
        inter, w2_raw, w2_scale, topk_ids, topk_weights, s2_dst,
        n_tokens, N, w2_n_phys, i_dim,
        inter.stride(0), inter.stride(1),
        w2_raw.stride(0), w2_raw.stride(-2), w2_raw.stride(-1),
        w2_scale.stride(0), w2_scale.stride(-2), w2_scale.stride(-1),
        s2_stride_om, s2_stride_on, s2_stride_ok,
        w2_act_scale, b2,
        I_PACKED=i_dim // 2, TOPK=top_k,
        BLOCK_K=BLOCK_K, BLOCK_N=S2_BLOCK_N, M_DUP=S2_M_DUP,
        W_PRESHUFFLED=w2_preshuffled,
        HAS_BIAS=w2_bias is not None, SPLIT_K=s2_split_k,
        num_warps=1,
    )
    # fmt: on
    if s2_split_k > 1:
        R_BLOCK_N = 256
        r_grid = (n_tokens * ((N + R_BLOCK_N - 1) // R_BLOCK_N),)
        # fmt: off
        _moe_partial_reduce[r_grid](
            out_partial, out, n_tokens, N,
            out_partial.stride(0), out_partial.stride(1), out_partial.stride(2),
            out.stride(0), out.stride(1),
            SPLIT_K=s2_split_k, BLOCK_N=R_BLOCK_N, num_warps=1,
        )
        # fmt: on
    return out


def gluon_mxfp_fused_moe(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    *,
    w13_mx_scale: torch.Tensor,
    w2_mx_scale: torch.Tensor,
    w13_act_scale: torch.Tensor,
    w2_act_scale: torch.Tensor,
    top_k: int,
    w13_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    enable_warp_decode: bool = True,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
) -> torch.Tensor:
    """Route + dispatch GEMM + SwiGLU + combine GEMM, all fused for the
    gluon mxfp4 / fp8-activation path.

    Inputs:
        hidden_states: ``(n_tokens, hidden)`` activation in bf16/fp16.
        router_logits: ``(n_tokens, num_experts)`` raw router logits.
        w13_weight, w2_weight: gluon-swizzled MXFP4 expert weights
            (``RaggedTensorMetadata``-compatible wrapped tensors).
        w13_bias, w2_bias: optional float32 expert biases.
        w13_mx_scale, w2_mx_scale: gluon-swizzled MXFP4 expert weight
            scales for the two GEMMs.
        w13_act_scale, w2_act_scale: per-tensor FP8 activation scales
            for the two GEMMs.
        out_dtype: output dtype for the final combine output.
        top_k: routing top_k.
        swiglu_alpha / swiglu_limit: SwiGLU activation parameters.

        enable_warp_decode: Whether to try the gfx950 small-M warp-decode path.
    """
    x_fp8 = fp8_quantize(hidden_states, w13_act_scale)

    n_tokens = router_logits.shape[0]
    use_medium_decode = int(n_tokens) in (8, 16)

    # Warp-decode small-M MoE is only the fastest path for M<=4. M=8/16
    # intentionally falls through to the medium-decode direct path below.
    # It self-guards (returns None) for any shape it does not cover; the
    # tokenspeed-kernel registration wrapper owns the environment/platform gate.
    if enable_warp_decode:
        out = _gluon_mxfp4_fp8_warp_decode_moe(
            x_fp8,
            router_logits,
            w13_weight,
            w2_weight,
            w13_bias=w13_bias,
            w2_bias=w2_bias,
            w13_mx_scale=w13_mx_scale,
            w2_mx_scale=w2_mx_scale,
            w13_act_scale=w13_act_scale,
            w2_act_scale=w2_act_scale,
            out_dtype=out_dtype,
            top_k=top_k,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
        )
        if out is not None:
            return out

    # Decode-small GPT-OSS routing is launch-overhead dominated. Prefer the
    # single-kernel Gluon route for the M<=16 single-block-collapse regime;
    # fall back to the generic Triton route for larger/unsupported shapes.
    if n_tokens <= SMALLM_MAX_M and gluon_route_supported(
        router_logits, top_k, router_logits.dtype
    ):
        ragged_metadata, gather_indx, scatter_indx, gate_scal = gluon_fused_route(
            router_logits,
            top_k,
            dtype=router_logits.dtype,
        )
    else:
        ragged_metadata, gather_indx, scatter_indx, gate_scal = default_route(
            router_logits,
            top_k,
            dtype=router_logits.dtype,
        )

    act = FusedActivation(
        FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
        (swiglu_alpha, swiglu_limit),
    )

    gemm1_input = x_fp8

    intermediate_cache = gluon_mxfp_ragged_matmul(
        gemm1_input,
        w13_weight,
        w13_bias,
        w_mx_scale=w13_mx_scale,
        x_global_scale=w13_act_scale,
        out_dtype=out_dtype,
        a_ragged_metadata=ragged_metadata,
        gather_indx=gather_indx,
        fused_activation=act,
        out_quant_scale=w2_act_scale,
        prefer_unshuffled_w=use_medium_decode,
    )

    gemm2_input = intermediate_cache

    return gluon_mxfp_ragged_matmul(
        gemm2_input,
        w2_weight,
        w2_bias,
        w_mx_scale=w2_mx_scale,
        x_global_scale=w2_act_scale,
        out_dtype=out_dtype,
        a_ragged_metadata=ragged_metadata,
        scatter_indx=scatter_indx,
        gammas=gate_scal,
        n_tokens=n_tokens,
        n_expts_act=top_k,
        prefer_unshuffled_w=use_medium_decode,
    )


# ===========================================================================
# Small-M (decode) fused MoE routing in Gluon.
#
# Decode routing is launch-overhead bound. For ``M <= SMALLM_MAX_M`` this
# replaces the generic ``triton_kernels_routing`` pipeline (~12 kernel
# launches) with a single Gluon kernel, producing output bit-for-bit identical
# to the generic path. Larger M falls back; the caller gates on the bound.
#
# Why M <= 16 makes this exact: 16 is the smallest RaggedTensorMetadata block
# size, so every nonzero expert holds exactly one block (single-block collapse)
# and the gather/scatter placement is stable. The kernel fuses the in-kernel
# top-k, histogram/cumsum, single-block schedule, and a register-only counting
# sort, reproducing ``moe_route(traits={"output_type": "ragged_metadata"})``:
# ``RaggedTensorMetadata`` + gather_indx/scatter_indx/gate_scal of length
# ``G = M*topk``. Metadata shapes are queried from ``RaggedTensorMetadata`` so
# they match ``make_ragged_tensor_metadata`` on HIP and non-HIP alike.
# ===========================================================================

# Number of block-size rows in RaggedTensorMetadata for the active platform
# ([16,32,64,128,256] -> 5 on HIP, [16,32,64,128] -> 4 otherwise). Derived
# from the library so the metadata shapes match make_ragged_tensor_metadata
# exactly on every target.
_ROUTE_NB = len(RaggedTensorMetadata.block_sizes())

# Token-count bound for the small-M fused route. 16 == the smallest
# RaggedTensorMetadata block size, so for M <= 16 every expert's token count is
# ``col_sum <= M <= 16``, i.e. exactly one block, and the single-block schedule
# collapse is exact. The caller dispatches only ``M <= SMALLM_MAX_M`` to the
# Gluon kernel (decode, where it wins ~6x on routing) and keeps the generic
# ``triton_kernels_routing`` pipeline for larger M.
SMALLM_MAX_M = 16
# Warp-decode is only for the smallest decode regime. M>=8 should use the
# medium-decode direct path selected by the ragged matmul path below.
WARP_DECODE_MAX_M = 4
# Backwards-compatible alias for the small-M bound.
FUSED_ROUTE_MAX_M = SMALLM_MAX_M

# Configs the Gluon routing path supports; everything else falls back to the
# generic triton_kernels_routing pipeline.
GLUON_ROUTE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
GLUON_ROUTE_MAX_E = 1024  # next_pow2(E) bins / EP-wide tiles stay bounded
# Upper bound on G = M*topk. The stable-sort rank tile is [GP, GP] and the
# kernel's layouts assume the single-wavefront regime (GP <= 64); configs that
# would exceed it fall back to the generic pipeline.
GLUON_ROUTE_MAX_G = 64

# torch gate dtype -> gluon element type (for the in-kernel softmax cast that
# reproduces topk_forward's ``softmax(...).to(x_dtype)`` rounding exactly).
_ROUTE_GL_DTYPE = {
    torch.float16: gl.float16,
    torch.bfloat16: gl.bfloat16,
    torch.float32: gl.float32,
}


@gluon.jit
def _route_add(a, b):
    return a + b


@gluon.jit
def _fused_topk(
    Logits,  # [M, E]   X_DTYPE   (raw routing logits)
    stride_lm,  # logits row stride
    gmask,  # [GP]   bool     g < G
    tok,  # [GP]      int32    g // TOPK
    slot,  # [GP]     int32    g %  TOPK
    M: gl.constexpr,
    E: gl.constexpr,
    TOPK: gl.constexpr,
    MP: gl.constexpr,  # next_pow2(M)
    EP: gl.constexpr,  # next_pow2(E)
    GP: gl.constexpr,  # next_pow2(M*topk)
    TKP: gl.constexpr,  # next_pow2(topk)
    X_DTYPE: gl.constexpr,  # gate element type (logits dtype)
    L1: gl.constexpr,  # 1D blocked layout used by the consuming kernel
    LT: gl.constexpr,  # 2D blocked layout for the [MP, EP] logits tile
):
    """Fused in-kernel top-k matching ``topk_forward(apply_softmax=True)``.

    Selects, per token row, the top ``TOPK`` experts by logit value (ties to
    the smaller expert id, descending value order) and     the softmax gate over
    the selected logits -- reproducing the triton kernels ``_topk_forward``
    semantics without a separate launch or a ``y_vals``/``y_indx`` global
    round-trip.
    Returns flat ``(idx[GP] int32, vals[GP] X_DTYPE)`` in token-major gate
    order (``g = token*TOPK + slot``), ready for the counting sort.
    """
    NEG: gl.constexpr = float("-inf")
    # ---- load the [MP, EP] logits tile (invalid lanes -> -inf) -------------
    row = gl.expand_dims(gl.arange(0, MP, layout=gl.SliceLayout(1, LT)), 1)  # [MP,1]
    col = gl.expand_dims(gl.arange(0, EP, layout=gl.SliceLayout(0, LT)), 0)  # [1,EP]
    lmask = (row < M) & (col < E)
    cur = gl.load(Logits + row * stride_lm + col, mask=lmask, other=NEG).to(gl.float32)

    # ---- iterative arg-max top-k (descending value, smaller-id tie-break) --
    # Equivalent to streaming_topk's packed sort: max value wins, ties resolve
    # to the smaller expert index; the iteration emits experts in descending
    # value order, matching topk_forward's output slot order. Results are
    # written column-by-column into [MP, TKP] tiles (no python lists, which
    # gluon tracing does not support).
    tcol = gl.expand_dims(gl.arange(0, TKP, layout=gl.SliceLayout(0, LT)), 0)  # [1,TKP]
    val_t = gl.full([MP, TKP], -1e30, gl.float32, layout=LT)  # finite -inf-ish
    idx_t = gl.zeros([MP, TKP], gl.int32, layout=LT)
    live = lmask
    topmask = gl.full([MP, EP], 0x80000000, gl.uint32, layout=LT)
    fullmask = gl.full([MP, EP], 0xFFFFFFFF, gl.uint32, layout=LT)
    zero_pack = gl.full([MP, EP], 0, gl.uint64, layout=LT)
    for _r in gl.static_range(TOPK):
        # Match the generic Triton top-k strategy: rank a packed key that
        # carries both the float value ordering and the expert index. This keeps
        # the selected index valid even for NaN/inf logits, without remapping the
        # original selected value used by the softmax below.
        raw = cur.to(gl.uint32, bitcast=True)
        value_key = raw ^ gl.where((raw & topmask) != 0, fullmask, topmask)
        index_key = (EP - col).to(gl.uint32)
        packed = (value_key.to(gl.uint64) << 16) | index_key.to(gl.uint64)
        packed = gl.where(live, packed, zero_pack)
        best = gl.max(packed, axis=1, keep_dims=True)
        amax_key = (best & 0xFFFF).to(gl.int32)
        amax = (EP - amax_key).to(gl.int32)  # [MP,1]
        chosen = live & (col == amax)
        vmax = gl.sum(gl.where(chosen, cur, gl.zeros_like(cur)), axis=1, keep_dims=True)
        sel = tcol == _r  # [1,TKP]
        val_t = gl.where(sel, vmax, val_t)  # write column _r
        idx_t = gl.where(sel, amax, idx_t)
        live = live & (col != amax)  # drop chosen expert

    # ---- softmax over the selected logits (matches tl.softmax in fp32) -----
    # z = x - max(x); num = exp(z); den = sum(num); gate = fdiv(num, den).
    # Padding columns (TOPK..TKP) hold -1e30 -> exp(-) == 0 -> ignored.
    rmax = gl.max(val_t, axis=1, keep_dims=True)  # [MP,1]
    num = gl.exp(val_t - rmax)  # [MP,TKP]
    den = gl.sum(num, axis=1, keep_dims=True)  # [MP,1]
    gate_t = gl.fdiv(num, den)  # [MP,TKP] fp32

    # ---- flatten per-slot columns into the flat [GP] gate order -----------
    z_i = gl.zeros([MP, TKP], gl.int32, layout=LT)
    z_f = gl.zeros([MP, TKP], gl.float32, layout=LT)
    idx = gl.zeros([GP], gl.int32, layout=L1)
    valsf = gl.zeros([GP], gl.float32, layout=L1)
    for _r in gl.static_range(TOPK):
        sel = tcol == _r  # [1,TKP]
        idx_r = gl.convert_layout(gl.sum(gl.where(sel, idx_t, z_i), axis=1), L1)
        gat_r = gl.convert_layout(gl.sum(gl.where(sel, gate_t, z_f), axis=1), L1)
        take = (slot == _r) & gmask
        idx = gl.where(take, gl.gather(idx_r, tok, axis=0), idx)
        valsf = gl.where(take, gl.gather(gat_r, tok, axis=0), valsf)
    # cast like topk_forward's softmax(...).to(x_dtype) before the gate store.
    return idx, valsf.to(X_DTYPE)


# ===========================================================================
# Small-M (M <= 16): single-workgroup, stable-order, single-block collapse.
# ===========================================================================
@gluon.jit
def _fused_route_small_m(
    Logits,  # [M, E]       X_DTYPE (raw routing logits)
    SliceSizes,  # [E]          int32
    SliceOffs,  # [E+1]         int32
    BlockOffs,  # [NB, E+1]     int32
    BlockSched,  # [NB, MAXBLK] int32
    GatherIndx,  # [G]          int32
    ScatterIndx,  # [G]         int32
    GateScal,  # [G]           dtype
    stride_lm,  # logits row stride
    M: gl.constexpr,
    E: gl.constexpr,
    TOPK: gl.constexpr,
    MP: gl.constexpr,  # next_pow2(M)
    GP: gl.constexpr,  # next_pow2(M*topk)
    EP: gl.constexpr,  # next_pow2(E)
    TKP: gl.constexpr,  # next_pow2(topk)
    MAXBLK: gl.constexpr,  # == M*topk
    MAXBLKP: gl.constexpr,  # next_pow2(MAXBLK)
    NB_C: gl.constexpr,  # number of block-size rows (NB)
    X_DTYPE: gl.constexpr,  # gate element type (logits dtype)
    NW_C: gl.constexpr,  # num_warps (1 for the M<=2 decode hot path, else 4)
    bo_stride: gl.constexpr,  # block_offs row stride  == E+1
    bs_stride: gl.constexpr,  # block_sched row stride == MAXBLK
):
    G: gl.constexpr = M * TOPK
    # Layouts are parametric in NW_C. At M<=2 a single warp (NW_C=1) removes the
    # cross-warp s_barrier stalls (LDS reductions over 4 warps) that dominated
    # the decode hot path; for larger small-M the O(G^2) rank tile + top-k want
    # 4 warps, so NW_C=4 there.
    LE: gl.constexpr = gl.BlockedLayout([1], [64], [NW_C], [0])  # 1D (EP)
    LG: gl.constexpr = gl.BlockedLayout([1], [64], [NW_C], [0])  # 1D (GP)
    LB: gl.constexpr = gl.BlockedLayout([1], [64], [NW_C], [0])  # 1D (MAXBLKP)
    LT: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [NW_C, 1], [1, 0])  # 2D

    # ---- fused top-k: compute (expert id, softmax gate) per gate in-kernel,
    # replacing the separate topk_forward launch + y_vals/y_indx round-trip.
    g = gl.arange(0, GP, layout=LG)
    gmask = g < G
    tok = (g // TOPK).to(gl.int32)
    slot = (g % TOPK).to(gl.int32)
    idx, vals = _fused_topk(
        Logits,
        stride_lm,
        gmask,
        tok,
        slot,
        M,
        E,
        TOPK,
        MP,
        EP,
        GP,
        TKP,
        X_DTYPE,
        LG,
        LT,
    )

    # ---- histogram -> slice_sizes -----------------------------------------
    e = gl.arange(0, EP, layout=LE)
    emask = e < E
    hist = gl.histogram(idx, EP, mask=gmask, layout=LE)
    gl.store(SliceSizes + e, hist, mask=emask)

    # ---- slice_offs = [0] + cumsum(slice_sizes) ---------------------------
    # Store exclusive prefixes at 0..E-1; index E (the total) is the only entry
    # the inclusive scan uniquely supplies, so write just that one element
    # rather than re-writing 1..E-1 with identical values.
    incl = gl.associative_scan(hist, 0, _route_add)
    col_offs = incl - hist
    last = e == (E - 1)
    gl.store(SliceOffs + e, col_offs, mask=emask)
    gl.store(SliceOffs + e + 1, incl, mask=emask & last)

    # ---- block_offs_data / block_schedule_data ----------------------------
    # Single-block collapse: at M <= 16 every nonzero expert is exactly one
    # block at every block size, so all NB rows are identical and the packed
    # block value is just the expert id.
    n_blk = (hist > 0).to(gl.int32)
    blk_incl = gl.associative_scan(n_blk, 0, _route_add)
    blk_excl = blk_incl - n_blk
    n_total = gl.sum(n_blk, 0)
    jb = gl.arange(0, MAXBLKP, layout=LB)
    jbmask = jb < MAXBLK
    neg_fill = gl.full([MAXBLKP], -1, gl.int32, layout=LB)
    for k in gl.static_range(NB_C):
        gl.store(BlockOffs + k * bo_stride + e, blk_excl, mask=emask)
        gl.store(BlockOffs + k * bo_stride + e + 1, blk_incl, mask=emask & last)
        # Fill -1 only in the tail (jb >= n_total). It is disjoint from the
        # scatter targets [0, n_total) below, so the compiler cannot reorder
        # the two stores into an alias that clobbers scattered ids.
        gl.store(
            BlockSched + k * bs_stride + jb,
            neg_fill,
            mask=jbmask & (jb >= n_total),
        )
        # Packed value is the bare expert id (single block, so block index 0).
        gl.store(
            BlockSched + k * bs_stride + blk_excl,
            e,
            mask=(hist > 0) & emask,
        )

    # ---- stable per-expert rank -------------------------------------------
    # rank[g] = #{j<g : idx[j]==idx[g]}. idx is in registers post-fuse, so use
    # a [GP,GP] compare tile reduced over j; cheap since GP <= 64.
    idx_row = gl.expand_dims(gl.convert_layout(idx, gl.SliceLayout(1, LT)), 1)
    idx_col = gl.expand_dims(gl.convert_layout(idx, gl.SliceLayout(0, LT)), 0)
    g_row = gl.expand_dims(gl.arange(0, GP, layout=gl.SliceLayout(1, LT)), 1)
    g_col = gl.expand_dims(gl.arange(0, GP, layout=gl.SliceLayout(0, LT)), 0)
    match = ((idx_row == idx_col) & (g_col < g_row)).to(gl.int32)
    rank = gl.convert_layout(gl.sum(match, axis=1), LG)

    # ---- scatter to destination = slice_offs[expert] + rank ---------------
    pos = gl.gather(col_offs, idx, axis=0) + rank
    gl.store(GatherIndx + pos, tok, mask=gmask)
    gl.store(ScatterIndx + pos, g.to(gl.int32), mask=gmask)
    gl.store(GateScal + pos, vals, mask=gmask)


# ===========================================================================
# Host wrappers for the small-M fused route
# ===========================================================================
def _route_next_pow2(x: int) -> int:
    return 1 << (max(1, x) - 1).bit_length()


@gluon.jit
def _add_expert_bias(acc, bias_base, col, bound, mfma_layout: gl.constexpr):
    """Broadcast-add a per-expert column bias into an MFMA accumulator.

    The bias is loaded along N then converted into the accumulator's column
    slice layout, which keeps the broadcast-add convert-compatible with acc.
    """
    b = gl.load(bias_base + col, mask=bound, other=0.0).to(gl.float32)
    b = gl.convert_layout(b, gl.SliceLayout(0, mfma_layout))
    return acc + b[None, :]


@gluon.constexpr_function
def _warp_decode_mfma_layouts(m_dup, block_n, block_k_scale):
    """MFMA + dot-operand + e8m0 scale layouts shared by the warp-decode kernels.

    get_mfma_layout is not reused: it asserts num_warps in (4, 8), whereas warp
    decode runs a single warp ([1, 1] warps_per_cta).
    """
    mfma = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[1, 1]
    )
    dot_a = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=16)
    dot_b = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=16)
    a_scale = gl.amd.cdna4.get_mfma_scale_layout(dot_a, [m_dup, block_k_scale])
    b_scale = gl.amd.cdna4.get_mfma_scale_layout(dot_b, [block_n, block_k_scale])
    return mfma, dot_a, dot_b, a_scale, b_scale


@gluon.jit
def _mxfp4_scale_offset(n_idx, k_scale_idx, stride_wsk, stride_wsn):
    """Byte offset into a CDNA4-swizzled MXFP4 scale tensor.

    Storage is (..., K_SCALE_PAD*32, N_PAD/32); the swizzle packs the 32-wide N
    block and the K-scale position into one linear axis.
    """
    row = n_idx.to(gl.uint32)
    # CDNA4 e8m0 swizzle: K-scale group stride 256, (k%4) stride 64. Using
    # 128/32 would alias K-scale offsets with the N-part (wrong scale read).
    lin = (
        (k_scale_idx // 8) * 256
        + (k_scale_idx % 4) * 64
        + (row % 16) * 4
        + ((k_scale_idx % 8) // 4) * 2
        + ((row % 32) // 16)
    )
    return (row // 32).to(gl.int64) * stride_wsn + lin.to(gl.int64) * stride_wsk


@gluon.jit
def _load_w_scale_tile_direct_cdna4(
    WScale,
    expert,
    kt,
    off_n,
    stride_wse,
    stride_wsk,
    stride_wsn,
    cfg,
):
    """Load W e8m0 scales in AITER's physical CDNA4-swizzled layout."""
    BLOCK_N: gl.constexpr = cfg.BLOCK_N
    BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK
    BLOCK_N_PS: gl.constexpr = cfg.BLOCK_N_PRESHUFFLED
    BLOCK_K_S_PS: gl.constexpr = cfg.BLOCK_K_SCALE_PRESHUFFLED
    LW_S: gl.constexpr = cfg.load_layout_w_scale

    offs_ws_n = gl.arange(0, BLOCK_N_PS, layout=gl.SliceLayout(1, LW_S))[:, None]
    offs_ws_k = gl.arange(0, BLOCK_K_S_PS, layout=gl.SliceLayout(0, LW_S))[None, :]
    rows_n_scale = off_n // cfg.PRESHUFFLE_FACTOR + offs_ws_n
    scale_k_base = kt * BLOCK_K_S_PS
    raw_off = (
        expert.to(gl.int64) * stride_wse
        + (scale_k_base + offs_ws_k).to(gl.int64) * stride_wsk
        + rows_n_scale.to(gl.int64) * stride_wsn
    )
    raw = gl.amd.cdna4.buffer_load(
        ptr=WScale, offsets=raw_off.to(gl.int32), cache=".cg"
    )

    raw_7d = raw.reshape((BLOCK_N_PS, BLOCK_K_SCALE // 8, 4, 16, 2, 2, 1))
    raw_perm = raw_7d.permute((0, 5, 3, 1, 4, 2, 6))
    logical = raw_perm.reshape((BLOCK_N, BLOCK_K_SCALE))
    return gl.convert_layout(logical, cfg.layout_w_scale)


@gluon.jit
def _swiglu_gate_up(gate, linear, alpha: gl.constexpr, limit: gl.constexpr):
    """SwiGLU on separate gate/up MFMA accumulators (pre-split form)."""
    if limit > 0.0:
        gate = gl.minimum(gate, limit)
        linear = gl.clamp(linear, -limit, limit)
    sigmoid = 1.0 / (1.0 + gl.exp(-alpha * gate))
    return (gate * sigmoid) * (linear + 1.0)


@gluon.jit
def _warp_decode_stage1_coop_compute(
    token,
    slot,
    expert,
    pid_n,
    X,
    W,
    WScale,
    Y,
    M,
    D,
    i_dim,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wk,
    stride_wn,
    stride_wse,
    stride_wsk,
    stride_wsn,
    stride_ym,
    stride_yn,
    x_global_scale_ptr,
    out_quant_scale_ptr,
    w13_bias,
    TOPK: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    W_PRESHUFFLED: gl.constexpr,
    EVEN_K: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
):
    """Cooperative gate_up GEMM + bias + SwiGLU + fp8-quant + store for one
    (token, slot, expert).  N runs over the INTERLEAVED gate_up rows (2*I);
    ``_swiglu_reduce`` splits even=gate / odd=up.  Mirrors the plain path of
    ``_pipelined_moe_tile_compute`` (W_TRANSPOSE=False, swizzled w-scale,
    per-tensor x scale) but specialized to a single decode token (row 0 of the
    BLOCK_M tile).
    """
    N = 2 * i_dim
    off_n = pid_n * BLOCK_N
    # Keep base offsets int32 (buffer_load_to_shared requires int32/uint32
    # offsets); expert * stride fits int32 for GPT-OSS shapes.
    w_base_offset = expert * stride_we
    ws_base_offset = expert * stride_wse

    cfg = MoEConfig(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        "e4m3",  # X format (fp8 activations)
        "e2m1",  # W format (mxfp4 weights)
        32,  # SCALE_BLOCK
        NUM_BUFFERS,
        not W_PRESHUFFLED,  # W_TRANSPOSE for non-preshuffled K-packed-contiguous W
        False,  # WITH_X_MX_SCALE (per-tensor x scale only)
        True,  # WITH_W_MX_SCALE (e8m0 block scales)
        "swizzle",  # SCALE_LOAD_MODE -> W_SCALE_VIA_LDS unswizzle
        gl.int32,
        (1, 1, 1),  # NUM_SUBTILES
        EVEN_K,
        False,  # USE_GATHER
        NUM_WARPS,
        W_PRESHUFFLED=W_PRESHUFFLED,
        W_VIA_VGPR=False,
        W_PREFETCH=True,
    )

    BLOCK_K_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
    BLOCK_K_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W
    OUT_BLOCK_N: gl.constexpr = BLOCK_N // 2
    W_CACHE_MODIFIER: gl.constexpr = ".cg" if BLOCK_M <= 32 else ""

    X_ELEM_BITS: gl.constexpr = X.dtype.element_ty.primitive_bitwidth
    W_ELEM_BITS: gl.constexpr = W.dtype.element_ty.primitive_bitwidth
    LOAD_X_LAYOUT: gl.constexpr = _load_layout(
        BLOCK_K_X, BLOCK_M, NUM_WARPS, [1, 0], X_ELEM_BITS
    )
    offs_xm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, LOAD_X_LAYOUT))
    offs_xk = gl.arange(0, BLOCK_K_X, layout=gl.SliceLayout(0, LOAD_X_LAYOUT))

    valid = (token < M) & (expert >= 0)
    # One decode token per CTA: row 0 of the BLOCK_M tile carries the token,
    # the remaining rows are clamped/masked (buffer OOB -> 0 in LDS).
    rows_m = gl.where(offs_xm == 0, token, gl.zeros_like(offs_xm))
    mask_m = (offs_xm == 0) & valid

    k_limit_x = gl.multiple_of(D // cfg.DIV_FACTOR_X, 16)
    k_limit_w = gl.multiple_of(D // cfg.DIV_FACTOR_W, 16)

    x_desc = AsyncCopyDescriptor.initialize(
        cfg,
        0,
        BLOCK_K_X,
        X,
        rows_m,
        offs_xk,
        stride_xm,
        stride_xk,
        mask_m[:, None],
        k_limit_x,
    )
    if W_PRESHUFFLED:
        gl.static_assert(
            BLOCK_N == 128 and BLOCK_K_W == 128 and NUM_WARPS == 4,
            "warp_decode preshuffled W13 path assumes 128x128 W tiles "
            "and NUM_WARPS=4; re-derive the copy/read layouts for other shapes.",
        )
        LOAD_W_LAYOUT: gl.constexpr = _preshuffled_w_read_layout(
            BLOCK_N // 16, BLOCK_K_W, cfg.W_SCALE_VIA_LDS
        )
        LOAD_W_COPY_LAYOUT: gl.constexpr = _preshuffled_w_copy_layout(
            BLOCK_N // 16, BLOCK_K_W, cfg.W_SCALE_VIA_LDS, True
        )
        offsets_w, base_off_w = _make_preshuffled_w_full_offsets(
            w_base_offset,
            pid_n,
            LOAD_W_COPY_LAYOUT,
            BLOCK_N,
            BLOCK_N,
            BLOCK_K_W,
        )
        w_desc = WPreshuffledLdsDescriptor(
            cfg,
            BLOCK_K_W,
            W,
            W.dtype.element_ty,
            gl.to_tensor(N),
            offsets_w + base_off_w,
            pred=gl.to_tensor(True),
            load_layout=LOAD_W_LAYOUT,
            cache_modifier=W_CACHE_MODIFIER,
        )
    else:
        # K-contig W (W_TRANSPOSE=True): vectorise the contiguous K_packed axis
        # (mirrors the W_TRANSPOSE branch of _pipelined_moe_tile_compute).
        LOAD_W_LAYOUT: gl.constexpr = _load_layout(
            BLOCK_K_W, BLOCK_N, NUM_WARPS, [1, 0], W_ELEM_BITS
        )
        offs_wn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, LOAD_W_LAYOUT))
        offs_wk = gl.arange(0, BLOCK_K_W, layout=gl.SliceLayout(0, LOAD_W_LAYOUT))
        mask_n = (off_n + offs_wn) < N
        w_desc = AsyncCopyDescriptor.initialize(
            cfg,
            0,
            BLOCK_K_W,
            W,
            off_n + offs_wn,
            offs_wk,
            stride_wn,
            stride_wk,
            mask_n[:, None],
            k_limit_w,
            base_offset=w_base_offset,
            cache_modifier=W_CACHE_MODIFIER,
        )

    # W e8m0 scales -> LDS in the post-swizzle HBM shape; issue_local_load_unswizzle
    # reconstructs [BLOCK_N, BLOCK_K_SCALE] (the 7-D reshape/permute).
    BLOCK_N_PS: gl.constexpr = cfg.BLOCK_N_PRESHUFFLED
    BLOCK_K_S_PS_W: gl.constexpr = cfg.BLOCK_K_SCALE_PRESHUFFLED
    LW_S: gl.constexpr = cfg.load_layout_w_scale
    offs_ws_n = gl.arange(0, BLOCK_N_PS, layout=gl.SliceLayout(1, LW_S))
    offs_ws_k = gl.arange(0, BLOCK_K_S_PS_W, layout=gl.SliceLayout(0, LW_S))
    rows_n_scale = off_n // cfg.PRESHUFFLE_FACTOR + offs_ws_n
    row_limit_w_s = (N + cfg.PRESHUFFLE_FACTOR - 1) // cfg.PRESHUFFLE_FACTOR
    # Suppress the K-mask: the swizzle packs K with N; the W K-mask already
    # zeroes the OOB product regardless of scale value.
    k_limit_ws_load = ((D // cfg.SCALE_BLOCK + 7) // 8 * 8) * cfg.PRESHUFFLE_FACTOR
    w_scale_desc = AsyncCopyDescriptor.initialize(
        cfg,
        0,
        BLOCK_K_S_PS_W,
        WScale,
        rows_n_scale,
        offs_ws_k,
        stride_wsn,
        stride_wsk,
        rows_n_scale[:, None] < row_limit_w_s,
        k_limit_ws_load,
        base_offset=ws_base_offset,
    )

    pgm = MoEPipelinedProgram.initialize(cfg, x_desc, w_desc, 0, w_scale_desc)
    # Preserve the upstream small-M decode schedule: it uses a three-buffer
    # local-prefetch pipeline and different tail masking from the optimized
    # prefill GEMM pipeline above.
    acc = pgm.decode_pipeline(D)

    # Per-tensor activation scale.
    x_scale = gl.load(x_global_scale_ptr).to(gl.float32)
    acc = acc * x_scale

    if HAS_BIAS:
        # Bias is laid out [E, 2*I] (interleaved gate/up rows); add before the
        # SwiGLU even/odd split, matching the num_warps=1 path.
        bias_offs = off_n + gl.arange(0, BLOCK_N, gl.SliceLayout(0, cfg.acc_layout))
        bias_mask = bias_offs < N
        bias = gl.load(
            w13_bias + expert.to(gl.int64) * N + bias_offs,
            mask=bias_mask,
            other=0.0,
        )
        acc = acc + bias[None, :].to(gl.float32)

    out = _swiglu_reduce(acc, SWIGLU_ALPHA, SWIGLU_LIMIT, OUT_BLOCK_N, cfg.acc_layout)
    out_inv_scale = 1.0 / gl.load(out_quant_scale_ptr).to(gl.float32)
    out = (out * out_inv_scale).to(Y.dtype.element_ty)
    STORE_LAYOUT: gl.constexpr = out.type.layout

    offs_y_m = gl.arange(0, BLOCK_M, gl.SliceLayout(1, STORE_LAYOUT))
    off_n_out = pid_n * OUT_BLOCK_N
    offs_y_n = off_n_out + gl.arange(0, OUT_BLOCK_N, gl.SliceLayout(0, STORE_LAYOUT))
    row = token * TOPK + slot
    # Only tile-row 0 holds the token's result; all valid columns map to the
    # single Y row (row*stride_ym).
    y_offs = (
        row.to(gl.int64) * stride_ym
        + offs_y_n[None, :].to(gl.int64) * stride_yn
        + offs_y_m[:, None].to(gl.int64) * 0
    )
    mask_y = (offs_y_m[:, None] == 0) & valid & (offs_y_n[None, :] < i_dim)
    gl.store(Y + y_offs, out, mask=mask_y)


@gluon.jit
def _warp_decode_topk_stage1_coop_kernel(
    X,
    Logits,
    W,
    WScale,
    TopkIdsOut,
    TopkWeightsOut,
    Y,
    M,
    E,
    D,
    i_dim,
    stride_xm,
    stride_xk,
    stride_lm,
    stride_tim,
    stride_twm,
    stride_we,
    stride_wk,
    stride_wn,
    stride_wse,
    stride_wsk,
    stride_wsn,
    stride_ym,
    stride_yn,
    x_global_scale_ptr,
    out_quant_scale_ptr,
    w13_bias,
    D_PACKED: gl.constexpr,
    TOPK: gl.constexpr,
    EP: gl.constexpr,
    TKP: gl.constexpr,
    X_DTYPE: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_M: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    W_PRESHUFFLED: gl.constexpr,
    EVEN_K: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
):
    """Cooperative (multi-warp) fused dense top-k + gate_up stage1.

    The slot dimension is folded into the grid -- one gate_up GEMM (one
    MoEPipelinedProgram / LDS buffer set) per program, so LDS is not multiplied
    by TOPK. Routing layouts span all warps (EP/TKP padded to 64*NUM_WARPS).
    """
    pid = gl.program_id(axis=0)
    num_pid_n = gl.cdiv(2 * i_dim, BLOCK_N)
    slot = pid % TOPK
    rest = pid // TOPK
    pid_n = rest % num_pid_n
    token = rest // num_pid_n

    # ---- direct top-k for this token (replicated per (N tile, slot)) ----
    LE: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    LT: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    e = gl.arange(0, EP, layout=LE)
    emask = e < E
    cur = gl.load(
        Logits + token.to(gl.int64) * stride_lm + e,
        mask=(token < M) & emask,
        other=float("-inf"),
    ).to(gl.float32)
    t = gl.arange(0, TKP, layout=LT)
    val_t = gl.full([TKP], -1e30, gl.float32, layout=LT)
    idx_t = gl.zeros([TKP], gl.int32, layout=LT)
    live = (token < M) & emask
    topmask = gl.full([EP], 0x80000000, gl.uint32, layout=LE)
    fullmask = gl.full([EP], 0xFFFFFFFF, gl.uint32, layout=LE)
    zero_pack = gl.full([EP], 0, gl.uint64, layout=LE)
    for r in gl.static_range(TOPK):
        raw = cur.to(gl.uint32, bitcast=True)
        value_key = raw ^ gl.where((raw & topmask) != 0, fullmask, topmask)
        index_key = (EP - e).to(gl.uint32)
        packed = (value_key.to(gl.uint64) << 16) | index_key.to(gl.uint64)
        packed = gl.where(live, packed, zero_pack)
        best = gl.max(packed, axis=0)
        amax_key = (best & 0xFFFF).to(gl.int32)
        amax = (EP - amax_key).to(gl.int32)
        chosen = live & (e == amax)
        vmax = gl.sum(gl.where(chosen, cur, gl.zeros_like(cur)), axis=0)
        sel = t == r
        val_t = gl.where(sel, vmax, val_t)
        idx_t = gl.where(sel, amax, idx_t)
        live = live & (e != amax)
    rmax = gl.max(val_t, axis=0)
    num = gl.exp(val_t - rmax)
    den = gl.sum(num, axis=0)
    gate_t = gl.fdiv(num, den)
    if (pid_n == 0) & (slot == 0):
        gl.store(
            TopkIdsOut + token.to(gl.int64) * stride_tim + t,
            idx_t,
            mask=(token < M) & (t < TOPK),
        )
        gl.store(
            TopkWeightsOut + token.to(gl.int64) * stride_twm + t,
            gate_t.to(TopkWeightsOut.dtype.element_ty),
            mask=(token < M) & (t < TOPK),
        )

    slot_sel = t == slot
    expert = gl.sum(
        gl.where(slot_sel, idx_t, gl.zeros([TKP], gl.int32, layout=LT)), axis=0
    )
    # Grouped by role: coords / tensors / shapes / strides / scalars / constexpr.
    # fmt: off
    _warp_decode_stage1_coop_compute(
        token, slot, expert, pid_n,
        X, W, WScale, Y,
        M, D, i_dim,
        stride_xm, stride_xk,
        stride_we, stride_wk, stride_wn,
        stride_wse, stride_wsk, stride_wsn,
        stride_ym, stride_yn,
        x_global_scale_ptr, out_quant_scale_ptr, w13_bias,
        TOPK, BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, NUM_WARPS,
        W_PRESHUFFLED, EVEN_K, HAS_BIAS, SWIGLU_ALPHA, SWIGLU_LIMIT,
    )
    # fmt: on


@gluon.jit
def _warp_decode_stage2_preshuffled_w_offset(
    w_expert_off,
    k_pack,
    n_col,
    N_PHYS,
):
    k_in_block = k_pack % 128
    n_in_block = n_col % 128

    k_within = k_in_block % 16
    k_quad = (k_in_block // 16) % 4
    k_block = k_in_block // 64
    n_in_sub = n_in_block % 16
    n_block = n_in_block // 16

    in_tile = (
        n_block.to(gl.int64) * 2048
        + k_block.to(gl.int64) * 1024
        + k_quad.to(gl.int64) * 256
        + n_in_sub.to(gl.int64) * 16
        + k_within.to(gl.int64)
    )
    n_tiles = N_PHYS // 128
    tile_id = (k_pack // 128).to(gl.int64) * n_tiles + (n_col // 128).to(gl.int64)
    return (w_expert_off + tile_id * (128 * 128) + in_tile).to(gl.int32)


@gluon.jit
def _warp_decode_stage2_load_tile(
    kt,
    ak,
    bk,
    bsk,
    am,
    X,
    W,
    WScale,
    x_row_off,
    w_expert_off,
    w_n_off,
    ws_expert_off,
    scale_row_off,
    n_cols,
    stride_xk,
    stride_wk,
    stride_wsk,
    N_PHYS,
    i_dim,
    BLOCK_K: gl.constexpr,
    BLOCK_K_PACKED: gl.constexpr,
    BLOCK_K_SCALE: gl.constexpr,
    I_PACKED: gl.constexpr,
    W_PRESHUFFLED: gl.constexpr,
    MASK_TAIL: gl.constexpr = False,
):
    k_elem = kt * BLOCK_K + ak
    k_pack = kt * BLOCK_K_PACKED + bk
    a_off = (x_row_off + k_elem.to(gl.int64) * stride_xk + am.to(gl.int64) * 0).to(
        gl.int32
    )
    if W_PRESHUFFLED:
        b_off = _warp_decode_stage2_preshuffled_w_offset(
            w_expert_off, k_pack, n_cols, N_PHYS
        )
    else:
        b_off = (w_n_off + k_pack.to(gl.int64) * stride_wk).to(gl.int32)
    if BLOCK_K_SCALE == 4:
        scale_k_lin = (kt // 2) * 256 + (kt % 2) * 2 + bsk * 64
    else:
        sk = kt * BLOCK_K_SCALE + bsk
        scale_k_lin = (sk // 8) * 256 + (sk % 4) * 64 + ((sk % 8) // 4) * 2
    scale_k_off = scale_k_lin.to(gl.int64) * stride_wsk
    s_off = (ws_expert_off + scale_row_off + scale_k_off).to(gl.int32)
    if MASK_TAIL:
        # Partial / odd final K-tile (K = intermediate dim I): mask out-of-range
        # K lanes to 0 so they contribute nothing and never over-read.
        sk_valid = (kt * BLOCK_K_SCALE + bsk) < (i_dim // 32)
        a = gl.amd.cdna4.buffer_load(
            ptr=X, offsets=a_off, mask=k_elem < i_dim, other=0.0
        )
        b_mask = k_pack < I_PACKED
        b = gl.amd.cdna4.buffer_load(ptr=W, offsets=b_off, mask=b_mask, other=0)
        s = gl.amd.cdna4.buffer_load(ptr=WScale, offsets=s_off, mask=sk_valid, other=0)
    else:
        a = gl.amd.cdna4.buffer_load(ptr=X, offsets=a_off)
        b = gl.amd.cdna4.buffer_load(ptr=W, offsets=b_off)
        s = gl.amd.cdna4.buffer_load(ptr=WScale, offsets=s_off)
    return a, b, s


@gluon.jit
def _warp_decode_stage2_load_pair(
    kt,
    ak,
    bk,
    bsk,
    am,
    X,
    W,
    WScale,
    x_row_off,
    w_expert_off,
    w_n_off,
    ws_expert_off,
    scale_row_off,
    n_cols,
    stride_xk,
    stride_wk,
    stride_wsk,
    N_PHYS,
    i_dim,
    BLOCK_K: gl.constexpr,
    BLOCK_K_PACKED: gl.constexpr,
    BLOCK_K_SCALE: gl.constexpr,
    I_PACKED: gl.constexpr,
    W_PRESHUFFLED: gl.constexpr,
):
    """Load the even (kt) and odd (kt+1) K-tiles of one pipeline step."""
    # fmt: off
    a_even, b_even, s_even = _warp_decode_stage2_load_tile(
        kt, ak, bk, bsk, am, X, W, WScale,
        x_row_off, w_expert_off, w_n_off, ws_expert_off, scale_row_off,
        n_cols, stride_xk, stride_wk, stride_wsk, N_PHYS, i_dim,
        BLOCK_K, BLOCK_K_PACKED, BLOCK_K_SCALE, I_PACKED, W_PRESHUFFLED,
    )
    a_odd, b_odd, s_odd = _warp_decode_stage2_load_tile(
        kt + 1, ak, bk, bsk, am, X, W, WScale,
        x_row_off, w_expert_off, w_n_off, ws_expert_off, scale_row_off,
        n_cols, stride_xk, stride_wk, stride_wsk, N_PHYS, i_dim,
        BLOCK_K, BLOCK_K_PACKED, BLOCK_K_SCALE, I_PACKED, W_PRESHUFFLED,
    )
    # fmt: on
    return a_even, b_even, s_even, a_odd, b_odd, s_odd


@gluon.jit
def _warp_decode_stage2_mfma_pair(
    acc, a_even, b_even, s_even, a_odd, b_odd, s_odd, a_scale
):
    """Accumulate the scaled-MFMA of one even+odd K-tile pair (fp8 x mxfp4)."""
    # fmt: off
    acc = gl.amd.cdna4.mfma_scaled(
        a=a_even, a_scale=a_scale, a_format="e4m3",
        b=b_even, b_scale=s_even, b_format="e2m1", acc=acc,
    )
    acc = gl.amd.cdna4.mfma_scaled(
        a=a_odd, a_scale=a_scale, a_format="e4m3",
        b=b_odd, b_scale=s_odd, b_format="e2m1", acc=acc,
    )
    # fmt: on
    return acc


@gluon.jit
def _warp_decode_stage2_fp8_mxfp4_kernel(
    X,
    W,
    WScale,
    TopkIds,
    TopkWeights,
    Out,
    M,
    N,
    N_PHYS,
    i_dim,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wk,
    stride_wn,
    stride_wse,
    stride_wsk,
    stride_wsn,
    stride_om,
    stride_on,
    stride_ok,
    x_global_scale_ptr,
    w2_bias,
    I_PACKED: gl.constexpr,
    TOPK: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCK_N: gl.constexpr,
    M_DUP: gl.constexpr,
    W_PRESHUFFLED: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    SPLIT_K: gl.constexpr,
):
    """Direct top-k stage2: FP8 intermediate x MXFP4 W2 -> BF16 output.

    With SPLIT_K > 1 the K (intermediate) reduction is partitioned across
    SPLIT_K CTAs per output tile; each writes an fp32 partial into slice
    ``pid_k`` of the destination, reduced by ``_moe_partial_reduce``.
    Bias is added only by the first slice so it is not counted SPLIT_K times.
    """
    BLOCK_K_PACKED: gl.constexpr = BLOCK_K // 2
    BLOCK_K_SCALE: gl.constexpr = BLOCK_K // 32
    if W_PRESHUFFLED:
        gl.static_assert(
            128 % BLOCK_N == 0 and BLOCK_K_PACKED == 64,
            "warp_decode preshuffled W2 expects BLOCK_N to divide the "
            "128-wide shuffled tile and BLOCK_K_PACKED=64 so two stage2 "
            "iterations cover one 128-packed-byte K tile.",
        )
    pid = gl.program_id(axis=0)
    num_n = gl.cdiv(N, BLOCK_N)
    if SPLIT_K == 1:
        pid_k = 0
        pid_token = pid // num_n
        pid_n = pid % num_n
    else:
        per_k = M * num_n
        pid_k = pid // per_k
        rem = pid % per_k
        pid_token = rem // num_n
        pid_n = rem % num_n
    # Full + partial K-tile coverage (K = intermediate dim I). The old
    # `num_kt = I // BLOCK_K` dropped the partial final tile, miscomputing any
    # I not a multiple of BLOCK_K (GPT-OSS I=2880 lost K=2816..2879).
    num_full = i_dim // BLOCK_K
    total_kt = (i_dim + BLOCK_K - 1) // BLOCK_K
    kt_per = (total_kt + SPLIT_K - 1) // SPLIT_K
    kt_start = pid_k * kt_per
    kt_stop = gl.minimum(kt_start + kt_per, total_kt)
    full_stop = gl.minimum(kt_stop, num_full)
    _layouts: gl.constexpr = _warp_decode_mfma_layouts(M_DUP, BLOCK_N, BLOCK_K_SCALE)
    mfma_layout: gl.constexpr = _layouts[0]
    dot_a_layout: gl.constexpr = _layouts[1]
    dot_b_layout: gl.constexpr = _layouts[2]
    a_scale_layout: gl.constexpr = _layouts[3]
    b_scale_layout: gl.constexpr = _layouts[4]
    am = gl.arange(0, M_DUP, layout=gl.SliceLayout(1, dot_a_layout))[:, None]
    ak = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, dot_a_layout))[None, :]
    bk = gl.arange(0, BLOCK_K_PACKED, layout=gl.SliceLayout(1, dot_b_layout))[:, None]
    bn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, dot_b_layout))[None, :]
    bsn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, b_scale_layout))[:, None]
    bsk = gl.arange(0, BLOCK_K_SCALE, layout=gl.SliceLayout(0, b_scale_layout))[None, :]
    n_cols = pid_n * BLOCK_N + bn
    n_cols_s = pid_n * BLOCK_N + bsn
    a_scale = gl.full((M_DUP, BLOCK_K_SCALE), 127, gl.uint8, layout=a_scale_layout)
    acc_total = gl.zeros((M_DUP, BLOCK_N), dtype=gl.float32, layout=mfma_layout)
    if pid_token < M:
        for slot in gl.static_range(0, TOPK):
            expert = gl.load(
                TopkIds + pid_token * TOPK + slot, mask=pid_token < M, other=-1
            )
            gate = gl.load(
                TopkWeights + pid_token * TOPK + slot,
                mask=pid_token < M,
                other=0.0,
            ).to(gl.float32)
            if expert >= 0:
                row = pid_token * TOPK + slot
                x_row_off = row.to(gl.int64) * stride_xm
                w_expert_off = expert.to(gl.int64) * stride_we
                ws_expert_off = expert.to(gl.int64) * stride_wse
                w_n_off = w_expert_off + n_cols.to(gl.int64) * stride_wn
                scale_row = n_cols_s.to(gl.uint32)
                scale_row_off = (scale_row // 32).to(gl.int64) * stride_wsn + (
                    (scale_row % 16) * 4 + ((scale_row % 32) // 16)
                ).to(gl.int64) * stride_wsk
                acc = gl.zeros((M_DUP, BLOCK_N), dtype=gl.float32, layout=mfma_layout)
                main_end = kt_start + ((full_stop - kt_start) // 2) * 2

                # Software-pipeline the main paired K-loop one step ahead:
                # prefetch the first pair, then each iteration loads the next pair
                # before MFMA-ing the current one (prefetch depth 2).
                main_kt = main_end - kt_start
                # fmt: off
                if main_kt > 0:
                    (a_even, b_even, s_even,
                     a_odd, b_odd, s_odd) = _warp_decode_stage2_load_pair(
                        kt_start, ak, bk, bsk, am, X, W, WScale,
                        x_row_off, w_expert_off, w_n_off, ws_expert_off, scale_row_off,
                        n_cols, stride_xk, stride_wk, stride_wsk, N_PHYS, i_dim,
                        BLOCK_K, BLOCK_K_PACKED, BLOCK_K_SCALE, I_PACKED, W_PRESHUFFLED,
                    )
                    for kt in range(kt_start, main_end - 2, 2):
                        (nxt_a_even, nxt_b_even, nxt_s_even,
                         nxt_a_odd, nxt_b_odd, nxt_s_odd) = _warp_decode_stage2_load_pair(
                            kt + 2, ak, bk, bsk, am, X, W, WScale,
                            x_row_off, w_expert_off, w_n_off, ws_expert_off, scale_row_off,
                            n_cols, stride_xk, stride_wk, stride_wsk, N_PHYS, i_dim,
                            BLOCK_K, BLOCK_K_PACKED, BLOCK_K_SCALE, I_PACKED, W_PRESHUFFLED,
                        )
                        acc = _warp_decode_stage2_mfma_pair(
                            acc, a_even, b_even, s_even, a_odd, b_odd, s_odd, a_scale
                        )
                        a_even, b_even, s_even, a_odd, b_odd, s_odd = (
                            nxt_a_even, nxt_b_even, nxt_s_even,
                            nxt_a_odd, nxt_b_odd, nxt_s_odd,
                        )
                    # Epilogue: MFMA the final prefetched pair.
                    acc = _warp_decode_stage2_mfma_pair(
                        acc, a_even, b_even, s_even, a_odd, b_odd, s_odd, a_scale
                    )
                # Masked remainder: leftover odd/partial K-tile(s) in this split.
                for kt in range(main_end, kt_stop):
                    a_t, b_t, s_t = _warp_decode_stage2_load_tile(
                        kt, ak, bk, bsk, am, X, W, WScale,
                        x_row_off, w_expert_off, w_n_off, ws_expert_off, scale_row_off,
                        n_cols, stride_xk, stride_wk, stride_wsk, N_PHYS, i_dim,
                        BLOCK_K, BLOCK_K_PACKED, BLOCK_K_SCALE, I_PACKED, W_PRESHUFFLED,
                        MASK_TAIL=True,
                    )
                    acc = gl.amd.cdna4.mfma_scaled(
                        a=a_t, a_scale=a_scale, a_format="e4m3",
                        b=b_t, b_scale=s_t, b_format="e2m1", acc=acc,
                    )
                # fmt: on
                acc = acc * gl.load(x_global_scale_ptr).to(gl.float32)
                if HAS_BIAS:
                    bias_n = pid_n * BLOCK_N + gl.arange(
                        0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout)
                    )
                    w2_base = w2_bias + expert.to(gl.int64) * N
                    if SPLIT_K == 1:
                        bias_bound = bias_n < N
                    else:
                        bias_bound = (bias_n < N) & (pid_k == 0)
                    acc = _add_expert_bias(
                        acc, w2_base, bias_n, bias_bound, mfma_layout
                    )
                acc_total += gate * acc
    sm = gl.arange(0, M_DUP, layout=gl.SliceLayout(1, mfma_layout))[:, None]
    sn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))[None, :]
    col = pid_n * BLOCK_N + sn
    out_base = (
        Out
        + pid_token.to(gl.int64) * stride_om
        + col.to(gl.int64) * stride_on
        + sm.to(gl.int64) * 0
    )
    if SPLIT_K > 1:
        out_base = out_base + pid_k.to(gl.int64) * stride_ok
    gl.store(
        out_base,
        acc_total.to(Out.dtype.element_ty),
        mask=(pid_token < M) & (sm == 0) & (col < N),
    )


@gluon.jit
def _moe_partial_reduce(
    Partial,
    Out,
    M,
    N,
    stride_pk,
    stride_pm,
    stride_pn,
    stride_om,
    stride_on,
    SPLIT_K: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    """Sum SPLIT_K partials per (m, n) into Out in one launch.

    Shared by the warp-decode split-K stage2 ([SPLIT_K, M, N] partials) and the
    medium-decode top-k combine (consecutive-row partials, mapped by passing
    stride_pk = row stride and stride_pm = TOPK * row stride). The float32 cast
    is a no-op for f32 partials and upcasts bf16 combine partials.
    """
    pid = gl.program_id(axis=0)
    num_n = gl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n
    pid_n = pid % num_n
    LAYOUT: gl.constexpr = gl.BlockedLayout([4], [64], [1], [0])
    n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=LAYOUT)
    bound = (pid_m < M) & (n < N)
    acc = gl.zeros([BLOCK_N], gl.float32, layout=LAYOUT)
    for k in gl.static_range(SPLIT_K):
        acc += gl.load(
            Partial
            + k * stride_pk
            + pid_m.to(gl.int64) * stride_pm
            + n.to(gl.int64) * stride_pn,
            mask=bound,
            other=0.0,
        ).to(gl.float32)
    gl.store(
        Out + pid_m.to(gl.int64) * stride_om + n.to(gl.int64) * stride_on,
        acc.to(Out.dtype.element_ty),
        mask=bound,
    )


def _route_small_m(logits, topk, dtype):
    """M <= 16: 1-kernel stable-order fused route (top-k fused in-kernel)."""
    M, E = logits.shape
    G = M * topk
    device = logits.device
    logits = logits.contiguous()

    slice_sizes = torch.empty(E, dtype=torch.int32, device=device)
    slice_offs = torch.empty(E + 1, dtype=torch.int32, device=device)
    block_offs_data = torch.empty(_ROUTE_NB, E + 1, dtype=torch.int32, device=device)
    # Query the library for the block-schedule width so it stays exact on any
    # platform rather than hardcoding the small-M value.
    maxblk = RaggedTensorMetadata.max_n_blocks(E, G)
    block_schedule_data = torch.empty(
        _ROUTE_NB, maxblk, dtype=torch.int32, device=device
    )
    gather_indx = torch.empty(G, dtype=torch.int32, device=device)
    scatter_indx = torch.empty(G, dtype=torch.int32, device=device)
    gate_scal = torch.empty(G, dtype=dtype, device=device)

    # M<=2 is the launch-bound decode hot path: a single warp removes the
    # cross-warp s_barrier stalls. Larger small-M has enough work (O(G^2) rank
    # tile + top-k) to benefit from 4 warps.
    nw = 1 if M <= 2 else 4

    _fused_route_small_m[(1,)](
        logits,
        slice_sizes,
        slice_offs,
        block_offs_data,
        block_schedule_data,
        gather_indx,
        scatter_indx,
        gate_scal,
        logits.stride(0),
        M=M,
        E=E,
        TOPK=topk,
        MP=_route_next_pow2(M),
        GP=_route_next_pow2(G),
        EP=_route_next_pow2(E),
        TKP=_route_next_pow2(topk),
        MAXBLK=maxblk,
        MAXBLKP=_route_next_pow2(maxblk),
        NB_C=_ROUTE_NB,
        X_DTYPE=_ROUTE_GL_DTYPE[logits.dtype],
        NW_C=nw,
        bo_stride=block_offs_data.stride(0),
        bs_stride=block_schedule_data.stride(0),
        num_warps=nw,
    )

    ragged = RaggedTensorMetadata(
        slice_sizes, slice_offs, block_offs_data, block_schedule_data
    )
    return ragged, gather_indx, scatter_indx, gate_scal


def gluon_route_supported(
    logits: torch.Tensor,
    topk: int,
    dtype: torch.dtype | None = None,
) -> bool:
    """Whether the unified Gluon routing path supports this configuration.

    Guards the structural assumptions the Gluon kernels make so unsupported
    configs fall back to the generic ``triton_kernels_routing`` pipeline:
    a 2D float ``logits`` tensor,     a supported gate ``dtype``, a sane ``topk``
    and an expert count whose ``next_pow2`` keeps the histogram bins / EP-wide
    tiles bounded.
    """
    if logits.ndim != 2:
        return False
    if dtype is None:
        dtype = logits.dtype
    if logits.dtype not in GLUON_ROUTE_DTYPES or dtype not in GLUON_ROUTE_DTYPES:
        return False
    M, E = logits.shape
    if topk < 1 or topk > E:
        return False
    if E < 1 or E > GLUON_ROUTE_MAX_E:
        return False
    # G = M*topk drives the [GP, GP] rank tile / single-wavefront layouts.
    if M * topk > GLUON_ROUTE_MAX_G:
        return False
    return True


def gluon_fused_route(
    logits: torch.Tensor,
    topk: int,
    dtype: torch.dtype | None = None,
) -> tuple[RaggedTensorMetadata, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Small-M (decode) fused MoE routing.

    Reproduces ``moe_route(traits={"output_type": "ragged_metadata"})`` in a
    single Gluon kernel, returning ``(ragged_metadata, gather_indx,
    scatter_indx, gate_scal)`` bit-for-bit identical to the generic pipeline.
    Valid for ``M <= SMALLM_MAX_M`` (the single-block-collapse regime); callers
    gate on that bound and fall back to the generic pipeline for larger ``M``.
    """
    if dtype is None:
        dtype = logits.dtype
    M = logits.shape[0]
    if M > SMALLM_MAX_M:
        raise ValueError(
            f"gluon_fused_route requires M <= {SMALLM_MAX_M} "
            f"(single-block-collapse regime); got M={M}. Route larger M "
            "through the generic triton_kernels_routing pipeline."
        )
    return _route_small_m(logits, topk, dtype)


def default_route(
    logits: torch.Tensor,
    n_expts_act: int,
    dtype: torch.dtype | None = None,
) -> tuple[RaggedTensorMetadata, torch.Tensor, torch.Tensor, torch.Tensor]:
    if dtype is None:
        dtype = logits.dtype

    assert logits.ndim == 2, "router_logits must be (n_tokens, n_expts_tot)"
    n_tokens, _ = logits.shape

    sparse = topk(logits, n_expts_act, apply_softmax=True)
    mask_metadata = sparse.mask_metadata

    col_sorted = mask_metadata.col_sorted_indx
    gather_indx = col_sorted // n_expts_act
    scatter_indx = col_sorted

    vals_flat = sparse.vals.reshape(-1)
    if dtype is not None and vals_flat.dtype != dtype:
        vals_flat = vals_flat.to(dtype)
    gate_scal = vals_flat[scatter_indx]

    n_total_rows = n_tokens * n_expts_act
    ragged_metadata = make_ragged_tensor_metadata(mask_metadata.col_sum, n_total_rows)

    return ragged_metadata, gather_indx, scatter_indx, gate_scal


@triton.jit
def _fp8_quantize_kernel(
    x_ptr,
    out_ptr,
    scale,
    M,
    N,
    x_row_stride,
    out_row_stride,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
    FP8_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_SCALE_TENSOR: tl.constexpr,
):
    pid = tl.program_id(0)
    m_idx = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_idx < M
    n_idx = tl.arange(0, BLOCK_N)

    if EVEN_N:
        load_mask = m_mask[:, None]
    else:
        load_mask = m_mask[:, None] & (n_idx[None, :] < N)

    x_off = m_idx[:, None] * x_row_stride + n_idx[None, :]
    x = tl.load(x_ptr + x_off, mask=load_mask)

    x = x.to(tl.float32)
    if HAS_SCALE:
        if HAS_SCALE_TENSOR:
            scale = tl.load(scale)
        x = x * (1.0 / scale)
    x_fp8 = x.to(FP8_DTYPE)

    out_off = m_idx[:, None] * out_row_stride + n_idx[None, :]
    tl.store(out_ptr + out_off, x_fp8, mask=load_mask)


def _flatten_to_2d(x: torch.Tensor):
    assert x.stride(-1) == 1, f"expected stride-1 inner dim, got stride={x.stride(-1)}"
    N = x.shape[-1]
    if x.ndim == 1:
        return 1, N, N
    M = x.numel() // N
    row_stride = x.stride(-2)
    # Validate that every leading dim packs onto the next.
    for d in range(x.ndim - 2):
        expected = x.shape[d + 1] * x.stride(d + 1)
        if x.stride(d) != expected:
            raise ValueError(
                f"cannot flatten dim {d}: stride={x.stride(d)} but expected "
                f"shape[{d+1}]*stride[{d+1}]={expected}. Tensor shape={tuple(x.shape)}, "
                f"stride={tuple(x.stride())}."
            )
    return M, N, row_stride


def fp8_quantize(
    x: torch.Tensor,
    scale: float | torch.Tensor | None = None,
    out: Optional[torch.Tensor] = None,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    assert x.dtype in (
        torch.bfloat16,
        torch.float16,
    ), f"fp8_quantize input must be bf16/fp16, got {x.dtype}"
    assert fp8_dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e4m3fnuz,
    ), f"fp8_quantize unsupported fp8 dtype: {fp8_dtype}"
    has_scale = scale is not None
    has_scale_tensor = isinstance(scale, torch.Tensor)
    if has_scale_tensor:
        assert scale.numel() == 1, "scale tensor must be scalar"
        scale = scale.contiguous()

    M, N, x_row_stride = _flatten_to_2d(x)

    if out is None:
        out = torch.empty(x.shape, dtype=fp8_dtype, device=x.device)
    else:
        assert out.shape == x.shape and out.dtype == fp8_dtype
    out_M, _, out_row_stride = _flatten_to_2d(out)
    assert out_M == M

    if fp8_dtype is torch.float8_e4m3fn:
        fp8_dtype_const = tl.float8e4nv
    elif fp8_dtype is torch.float8_e5m2:
        fp8_dtype_const = tl.float8e5
    else:
        fp8_dtype_const = tl.float8e4b8

    if M <= 2048:
        block_m = 4
    elif M <= 16384:
        block_m = 16
    else:
        block_m = 32
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(M, block_m),)

    block_n = max(1, triton.next_power_of_2(N))
    even_n = block_n == N

    _fp8_quantize_kernel[grid](
        x,
        out,
        1.0 if scale is None else scale,
        M,
        N,
        x_row_stride,
        out_row_stride,
        BLOCK_N=block_n,
        EVEN_N=even_n,
        FP8_DTYPE=fp8_dtype_const,
        BLOCK_M=block_m,
        HAS_SCALE=has_scale,
        HAS_SCALE_TENSOR=has_scale_tensor,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


__all__ = [
    "gluon_mxfp_fused_moe",
    "gluon_mxfp_ragged_matmul",
    "gluon_mxfp_combine",
    "gluon_mxfp_dispatch_swiglu",
    "gluon_fused_route",
]
