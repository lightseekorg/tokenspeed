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

import os
from typing import Any

import torch
from tokenspeed_kernel._triton import (
    aggregate,
    gl,
    gluon,
    redirect_triton_to_tokenspeed_triton,
)
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import KernelRegistry, Priority, register_kernel
from tokenspeed_kernel.signature import (
    ScaleFormat,
    dense_tensor_format,
    format_signature,
    format_signatures,
    tensor_format,
)

with redirect_triton_to_tokenspeed_triton():
    from triton_kernels.matmul import FnSpecs, FusedActivation
    from triton_kernels.swiglu import swiglu_fn
    from triton_kernels.tensor import RaggedTensorMetadata, Tensor

__all__ = [
    "_gluon_mxfp_fused_moe",
    "_gluon_mxfp_ragged_matmul",
    "gluon_mxfp_combine",
    "gluon_mxfp_dispatch_swiglu",
    "FUSED_ROUTE_MAX_M",
    "SMALLM_MAX_M",
    "GLUON_ROUTE_DTYPES",
    "GLUON_ROUTE_MAX_E",
    "GLUON_ROUTE_MAX_G",
    "gluon_fused_route",
    "gluon_route_supported",
    "gluon_decode_routing_gfx950",
]

_GLUON_DISABLE_VALUES = frozenset({"0", "false", "no", "off", "disable", "disabled"})
_GLUON_DISABLED_ENV = (
    os.environ.get("TOKENSPEED_MOE_GLUON", "").strip().lower() in _GLUON_DISABLE_VALUES
)


def _as_int32(t):
    if t is None or t.dtype == torch.int32:
        return t
    return t.to(torch.int32)


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


def _default_num_buffers(
    K: int,
    block_k: int,
    *,
    block_m: int | None = None,
    block_n: int | None = None,
    x_format: str = "e2m1",
    w_format: str = "e2m1",
    has_x_block_scale: bool | None = None,
    has_w_block_scale: bool = True,
    scale_load_mode: str = "transpose",
) -> int:
    K_iters = max(1, (K + block_k - 1) // block_k)
    nb = 3 if K_iters >= 3 else 2
    if block_m is not None and block_n is not None:
        per_buf = _estimate_pipeline_lds_per_buffer(
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            x_format=x_format,
            w_format=w_format,
            has_x_block_scale=has_x_block_scale,
            has_w_block_scale=has_w_block_scale,
            scale_load_mode=scale_load_mode,
        )
        if per_buf > 0:
            max_nb = max(1, 160 * 1024 // per_buf)
            nb = min(nb, max_nb)
    return nb


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
    # W_VIA_VGPR drops the n-mask, so N must be block_n-aligned. The
    # combine GEMM pads W + W-scale at the backend before this helper
    # sees the tensor; we still assert here to catch unaligned callers.
    if N % block_n != 0:
        raise ValueError(
            f"shuffle_weight_for_gluon_dot_layout requires N "
            f"divisible by block_n={block_n} (got N={N}); the kernel's "
            f"W_VIA_VGPR path assumes block_n-aligned N. Pad the raw W "
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
    return out


# ---------------------------------------------------------------------------
# Layout factories (gluon constexpr functions)
# ---------------------------------------------------------------------------


@gluon.constexpr_function
def _store_layout(num_warps: int, block_m: int = 0, w_via_vgpr: bool = False):
    # Mirrors the warps_m policy in get_mfma_layout so the MFMA acc
    # and store layouts stay convert-compatible.
    if w_via_vgpr and num_warps >= 4:
        warps_m = 2
    elif block_m and block_m <= 32 and num_warps >= 4:
        warps_m = 1
    else:
        warps_m = 2 if num_warps >= 4 else 1
    warps_n = num_warps // warps_m
    return gl.BlockedLayout([1, 8], [2, 32], [warps_m, warps_n], [1, 0])


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
        size_per_thread=[1, 4],
        threads_per_warp=[2, THREADS_PER_WARP // 2],
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
    W_VIA_VGPR: gl.constexpr
    W_PREFETCH: gl.constexpr
    NUM_BUFFERS: gl.constexpr

    SCALE_BLOCK: gl.constexpr
    WITH_X_MX_SCALE: gl.constexpr
    WITH_W_MX_SCALE: gl.constexpr
    SCALE_LOAD_MODE: gl.constexpr
    SCALE_VIA_LDS: gl.constexpr
    PRESHUFFLE_FACTOR: gl.constexpr
    BLOCK_M_PRESHUFFLED: gl.constexpr
    BLOCK_N_PRESHUFFLED: gl.constexpr
    BLOCK_K_SCALE_PRESHUFFLED: gl.constexpr
    shared_layout_w_half_n: gl.constexpr
    shared_layout_x_half_m: gl.constexpr

    NUM_SUBTILES: gl.constexpr
    EVEN_K: gl.constexpr
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
        USE_GATHER=False,
        NUM_WARPS=4,
        W_VIA_VGPR=False,
        W_PREFETCH=True,
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

        _scale_via_lds = SCALE_LOAD_MODE == "swizzle" and (
            WITH_X_MX_SCALE or WITH_W_MX_SCALE
        )
        self.SCALE_VIA_LDS = gl.constexpr(_scale_via_lds)
        self.PRESHUFFLE_FACTOR = gl.constexpr(_SCALE_PRESHUFFLE_FACTOR)
        self.BLOCK_M_PRESHUFFLED = gl.constexpr(BLOCK_M // _SCALE_PRESHUFFLE_FACTOR)
        self.BLOCK_N_PRESHUFFLED = gl.constexpr(BLOCK_N // _SCALE_PRESHUFFLE_FACTOR)
        self.BLOCK_K_SCALE_PRESHUFFLED = gl.constexpr(
            (BLOCK_K // SCALE_BLOCK) * _SCALE_PRESHUFFLE_FACTOR
        )

        self.NUM_SUBTILES = gl.constexpr(NUM_SUBTILES)
        self.EVEN_K = gl.constexpr(EVEN_K)
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
            if WITH_X_MX_SCALE:
                num_loads += 1
            if WITH_W_MX_SCALE:
                num_loads += 1
        self.NUM_LOADS_IN_BATCH = gl.constexpr(num_loads)

        BLOCK_K_SCALE = BLOCK_K // SCALE_BLOCK
        self.index_type = gl.constexpr(index_type)

        MFMA_LAYOUT: gl.constexpr = get_mfma_layout(
            NUM_WARPS,
            self.USE_MFMA_SCALED,
            scale_preshuffle=_scale_via_lds,
            block_m=BLOCK_M,
            w_via_vgpr=W_VIA_VGPR,
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
        if W_TRANSPOSE:
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

        if W_TRANSPOSE:
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
        if cfg.SCALE_VIA_LDS:
            if cfg.WITH_X_MX_SCALE:
                self.x_scale_desc.issue_async_load(
                    load_idx, self.x_scale_buffer, pred, USE_MASK=USE_MASK
                )
            if cfg.WITH_W_MX_SCALE:
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
    w_desc: AsyncCopyDescriptor | WVgprDescriptor
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
        self.x_scale_buffer = (
            x_scale_buffer
            if (cfg.SCALE_VIA_LDS and cfg.WITH_X_MX_SCALE)
            else gl.constexpr(0)
        )
        self.w_scale_buffer = (
            w_scale_buffer
            if (cfg.SCALE_VIA_LDS and cfg.WITH_W_MX_SCALE)
            else gl.constexpr(0)
        )
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

        if cfg.SCALE_VIA_LDS and cfg.WITH_X_MX_SCALE:
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

        if cfg.SCALE_VIA_LDS and cfg.WITH_W_MX_SCALE:
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
    def _load_xw(self, mfma_idx):
        cfg = self.cfg
        x = self.x_desc.issue_local_load(
            mfma_idx,
            self.x_buffer,
            cfg.dot_layout_x,
        )
        if cfg.W_VIA_VGPR:
            w = self._issue_w_vgpr(mfma_idx)
        else:
            w = self.w_desc.issue_local_load(
                mfma_idx,
                self.w_buffer,
                cfg.dot_layout_w,
                do_permute=cfg.W_TRANSPOSE,
            )
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
                if cfg.SCALE_VIA_LDS:
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
                if cfg.SCALE_VIA_LDS:
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
    def issue_local_loads(self, mfma_idx):
        cfg = self.cfg
        x, w = self._load_xw(mfma_idx)

        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK

        if cfg.USE_MFMA_SCALED:
            if cfg.WITH_X_MX_SCALE:
                if cfg.SCALE_VIA_LDS:
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
                if cfg.SCALE_VIA_LDS:
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

        return x, w, scale_x, scale_w

    @gluon.jit
    def pipeline(self, loop_k):
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

        for i in range(0, main_iters):
            load_idx = self.issue_global_loads(load_idx, USE_MASK=0)
            self.async_wait(cfg.NUM_BUFFERS - 1)

            if W_PREFETCH:
                x, scale_x, scale_w = self._load_x_scales(mfma_idx)
                accumulator = self.mfma(x, scale_x, w_curr, scale_w, accumulator)
                w_curr = self._issue_w_vgpr(mfma_idx + 1)
            else:
                x, w, scale_x, scale_w = self.issue_local_loads(mfma_idx)
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
                x, w, scale_x, scale_w = self.issue_local_loads(mfma_idx)
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
                x, w, scale_x, scale_w = self.issue_local_loads(mfma_idx)
                accumulator = self.mfma(x, scale_x, w, scale_w, accumulator)
            mfma_idx += 1

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
        self.x_scale_buffer = (
            x_scale_buffer
            if (cfg.SCALE_VIA_LDS and cfg.WITH_X_MX_SCALE)
            else gl.constexpr(0)
        )
        self.w_scale_buffer = (
            w_scale_buffer
            if (cfg.SCALE_VIA_LDS and cfg.WITH_W_MX_SCALE)
            else gl.constexpr(0)
        )
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

        if cfg.SCALE_VIA_LDS and cfg.WITH_X_MX_SCALE:
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

        if cfg.SCALE_VIA_LDS and cfg.WITH_W_MX_SCALE:
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
        if cfg.SCALE_VIA_LDS:
            if cfg.WITH_X_MX_SCALE:
                self.x_scale_desc.issue_async_load(
                    load_idx,
                    self.x_scale_buffer,
                    pred,
                    USE_MASK=USE_MASK,
                    COMMIT=0,
                )
            if cfg.WITH_W_MX_SCALE:
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
        EVEN_K: gl.constexpr = cfg.EVEN_K
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
            load_idx = self.issue_x_top(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 2)
            w_right, sw_right = self.issue_local_load_w_sub(mfma_idx, 1)

            c_tr = self.mfma(x_top, sx_top, w_right, sw_right, c_tr)
            mfma_idx += 1
            load_idx = self.issue_x_bot(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 2)
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
            load_idx = self.issue_x_top(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 2)
            w_right, sw_right = self.issue_local_load_w_sub(mfma_idx, 1)

            c_tr = self.mfma(x_top, sx_top, w_right, sw_right, c_tr)
            mfma_idx += 1
            load_idx = self.issue_x_bot(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 2)
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
            load_idx = self.issue_x_top(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 2)
            w_right, sw_right = self.issue_local_load_w_sub(mfma_idx, 1)

            c_tr = self.mfma(x_top, sx_top, w_right, sw_right, c_tr)
            mfma_idx += 1
            load_idx = self.issue_x_bot(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 2)
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
    w_desc_top: AsyncCopyDescriptor | WVgprDescriptor
    w_desc_bot: AsyncCopyDescriptor | WVgprDescriptor
    x_scale_desc: AsyncCopyDescriptor | gl.constexpr
    w_scale_desc: AsyncCopyDescriptor | gl.constexpr

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
    ):
        self.cfg = cfg
        self.x_buffer = x_buffer
        self.w_buffer_top = w_buffer_top if not cfg.W_VIA_VGPR else gl.constexpr(0)
        self.w_buffer_bot = w_buffer_bot if not cfg.W_VIA_VGPR else gl.constexpr(0)
        self.x_scale_buffer = (
            x_scale_buffer
            if (cfg.SCALE_VIA_LDS and cfg.WITH_X_MX_SCALE)
            else gl.constexpr(0)
        )
        self.w_scale_buffer = (
            w_scale_buffer
            if (cfg.SCALE_VIA_LDS and cfg.WITH_W_MX_SCALE)
            else gl.constexpr(0)
        )
        self.x_desc = x_desc
        self.w_desc_top = w_desc_top
        self.w_desc_bot = w_desc_bot
        self.x_scale_desc = x_scale_desc if cfg.WITH_X_MX_SCALE else gl.constexpr(0)
        self.w_scale_desc = w_scale_desc if cfg.WITH_W_MX_SCALE else gl.constexpr(0)
        self.base = MoEProgramBase()

    @gluon.jit
    def initialize(
        cfg: MoEConfig,
        x_desc,
        w_desc_top,
        w_desc_bot,
        x_scale_desc,
        w_scale_desc,
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

        if cfg.SCALE_VIA_LDS and cfg.WITH_X_MX_SCALE:
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

        if cfg.SCALE_VIA_LDS and cfg.WITH_W_MX_SCALE:
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
        )

    @gluon.jit
    def issue_local_load_x(self, mfma_idx):
        cfg = self.cfg
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK
        x = self.x_desc.issue_local_load(
            mfma_idx,
            self.x_buffer,
            cfg.dot_layout_x,
        )

        if cfg.USE_MFMA_SCALED:
            if cfg.WITH_X_MX_SCALE:
                if cfg.SCALE_VIA_LDS:
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
        if not cfg.W_VIA_VGPR:
            self.w_desc_top.issue_async_load(
                load_idx, self.w_buffer_top, pred, USE_MASK=USE_MASK, COMMIT=0
            )
        if cfg.SCALE_VIA_LDS:
            if cfg.WITH_X_MX_SCALE:
                self.x_scale_desc.issue_async_load(
                    load_idx,
                    self.x_scale_buffer,
                    pred,
                    USE_MASK=USE_MASK,
                    COMMIT=0,
                )
            if cfg.WITH_W_MX_SCALE:
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
    def pipeline(self, loop_k):
        cfg = self.cfg
        EVEN_K: gl.constexpr = cfg.EVEN_K
        NB: gl.constexpr = cfg.NUM_BUFFERS
        gl.static_assert(
            (cfg.NUM_SUBTILES[0] == 1)
            and (cfg.NUM_SUBTILES[1] == 2)
            and (cfg.NUM_SUBTILES[2] == 1),
            "MoESliceNProgram requires NUM_SUBTILES=(1,2,1)",
        )

        SUBTILE_N: gl.constexpr = cfg.BLOCK_N // 2

        load_idx = 0
        mfma_idx = 0

        for _ in gl.static_range(NB):
            load_idx = self.issue_global_loads(load_idx, USE_MASK=0)

        c0 = gl.zeros((cfg.BLOCK_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)
        c1 = gl.zeros((cfg.BLOCK_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)

        K_iters = gl.cdiv(loop_k, cfg.BLOCK_K)
        main_iters = K_iters - NB
        gl.assume(main_iters >= 2)

        # Drain iter 0's top half so the first ds_read has data.
        gl.amd.cdna4.async_copy.wait_group(2 * NB - 1)
        x, sx = self.issue_local_load_x(mfma_idx)
        w0, sw0 = self.issue_local_load_w_sub(mfma_idx, 0)

        unroll_pairs = main_iters // 2
        odd_main = main_iters - unroll_pairs * 2
        for _ in range(0, unroll_pairs):
            # iter k regions 0+1.
            c0 = self.mfma(x, sx, w0, sw0, c0)
            gl.amd.cdna4.async_copy.wait_group(2 * NB - 2)
            w1, sw1 = self.issue_local_load_w_sub(mfma_idx, 1)
            load_idx = self.issue_global_load_top(load_idx, USE_MASK=-1)

            c1 = self.mfma(x, sx, w1, sw1, c1)
            mfma_idx += 1
            gl.amd.cdna4.async_copy.wait_group(2 * NB - 2)
            x, sx = self.issue_local_load_x(mfma_idx)
            w0, sw0 = self.issue_local_load_w_sub(mfma_idx, 0)
            load_idx = self.issue_global_load_bot(load_idx, USE_MASK=-1)

            # iter k+1 regions 2+3 (LDS slot ping-ponged via mfma_idx parity).
            c0 = self.mfma(x, sx, w0, sw0, c0)
            gl.amd.cdna4.async_copy.wait_group(2 * NB - 2)
            w1, sw1 = self.issue_local_load_w_sub(mfma_idx, 1)
            load_idx = self.issue_global_load_top(load_idx, USE_MASK=-1)

            c1 = self.mfma(x, sx, w1, sw1, c1)
            mfma_idx += 1
            gl.amd.cdna4.async_copy.wait_group(2 * NB - 2)
            x, sx = self.issue_local_load_x(mfma_idx)
            w0, sw0 = self.issue_local_load_w_sub(mfma_idx, 0)
            load_idx = self.issue_global_load_bot(load_idx, USE_MASK=-1)

        # Odd peel; USE_MASK=-1 covers the K-tail iter.
        if odd_main:
            c0 = self.mfma(x, sx, w0, sw0, c0)
            gl.amd.cdna4.async_copy.wait_group(2 * NB - 2)
            w1, sw1 = self.issue_local_load_w_sub(mfma_idx, 1)
            load_idx = self.issue_global_load_top(load_idx, USE_MASK=-1)

            c1 = self.mfma(x, sx, w1, sw1, c1)
            mfma_idx += 1
            gl.amd.cdna4.async_copy.wait_group(2 * NB - 2)
            x, sx = self.issue_local_load_x(mfma_idx)
            w0, sw0 = self.issue_local_load_w_sub(mfma_idx, 0)
            load_idx = self.issue_global_load_bot(load_idx, USE_MASK=-1)

        # Drain + final NB iters of MFMAs (no more async_copy).
        gl.amd.cdna4.async_copy.wait_group(0)
        for i in gl.static_range(NB):
            c0 = self.mfma(x, sx, w0, sw0, c0)
            w1, sw1 = self.issue_local_load_w_sub(mfma_idx, 1)

            c1 = self.mfma(x, sx, w1, sw1, c1)
            mfma_idx += 1

            if i < NB - 1:
                x, sx = self.issue_local_load_x(mfma_idx)
                w0, sw0 = self.issue_local_load_w_sub(mfma_idx, 0)

        accumulator = (
            gl.join(c0, c1).permute(0, 2, 1).reshape((cfg.BLOCK_M, cfg.BLOCK_N))
        )
        accumulator = gl.convert_layout(accumulator, cfg.acc_layout)

        return accumulator


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
    APPLY_X_GLOBAL_SCALE: gl.constexpr = True,
    USE_WARP_PIPELINE: gl.constexpr = False,
    USE_SLICE_MN: gl.constexpr = False,
    USE_SLICE_N: gl.constexpr = False,
    HAS_FP8_QUANT_OUT: gl.constexpr = False,
    W_VIA_VGPR: gl.constexpr = False,
    W_PREFETCH: gl.constexpr = True,
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

    STORE: gl.constexpr = _store_layout(
        NUM_WARPS, block_m=BLOCK_M, w_via_vgpr=W_VIA_VGPR
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
        USE_GATHER,
        NUM_WARPS,
        W_VIA_VGPR=W_VIA_VGPR,
        W_PREFETCH=W_PREFETCH,
    )

    BLOCK_K_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
    BLOCK_K_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W

    W_CACHE_MODIFIER: gl.constexpr = ".cg" if BLOCK_M <= 32 else ""

    X_ELEM_BITS: gl.constexpr = x_ptr.dtype.element_ty.primitive_bitwidth
    W_ELEM_BITS: gl.constexpr = w_ptr.dtype.element_ty.primitive_bitwidth
    LOAD_X_LAYOUT: gl.constexpr = _load_layout(
        BLOCK_K_X, BLOCK_M, NUM_WARPS, [1, 0], X_ELEM_BITS
    )
    if cfg.W_VIA_VGPR:
        gl.static_assert(
            BLOCK_K_W == 128 and (BLOCK_N == 128 or USE_SLICE_N) and NUM_WARPS == 4,
            "W_VIA_VGPR LinearLayout bases assume BLOCK_K_W=128, "
            "BLOCK_N=128 (or USE_SLICE_N=True for half-tile path), "
            "NUM_WARPS=4. Re-derive bases for other shapes.",
        )
        BLOCK_N_LAYOUT: gl.constexpr = (BLOCK_N // 2) if USE_SLICE_N else BLOCK_N
        if cfg.SCALE_VIA_LDS:
            # tpw=[2,2]
            LOAD_W_LAYOUT: gl.constexpr = gl.DistributedLinearLayout(
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
                warp_bases=[
                    [2, 0],
                    [0, 0],
                ],
                block_bases=[],
                shape=[BLOCK_N_LAYOUT // 16, BLOCK_K_W * 16],
            )
        else:
            # tpw=[1,1]
            LOAD_W_LAYOUT: gl.constexpr = gl.DistributedLinearLayout(
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
                warp_bases=[
                    [1, 0],
                    [0, 0],
                ],
                block_bases=[],
                shape=[BLOCK_N_LAYOUT // 16, BLOCK_K_W * 16],
            )
    elif W_TRANSPOSE:
        # LDS path, K-contig: offsets shape (BLOCK_NONK, BLOCK_K).
        LOAD_W_LAYOUT: gl.constexpr = _load_layout(
            BLOCK_K_W, BLOCK_N, NUM_WARPS, [1, 0], W_ELEM_BITS
        )
    else:
        # LDS path, N-contig: W is [K_packed, N]; vectorise contig axis.
        LOAD_W_LAYOUT: gl.constexpr = _load_layout(
            BLOCK_N, BLOCK_K_W, NUM_WARPS, [1, 0], W_ELEM_BITS
        )

    offs_xm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, LOAD_X_LAYOUT))
    offs_xk = gl.arange(0, BLOCK_K_X, layout=gl.SliceLayout(0, LOAD_X_LAYOUT))
    if cfg.W_VIA_VGPR:
        # Virtual (n_block, k_flat); half-tile (BLOCK_N//2//16) under sliceN.
        _BN_W_LAYOUT: gl.constexpr = (BLOCK_N // 2) if USE_SLICE_N else BLOCK_N
        offs_wn = gl.arange(
            0, _BN_W_LAYOUT // 16, layout=gl.SliceLayout(1, LOAD_W_LAYOUT)
        )
        offs_wk = gl.arange(0, BLOCK_K_W * 16, layout=gl.SliceLayout(0, LOAD_W_LAYOUT))
    elif W_TRANSPOSE:
        offs_wn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, LOAD_W_LAYOUT))
        offs_wk = gl.arange(0, BLOCK_K_W, layout=gl.SliceLayout(0, LOAD_W_LAYOUT))
    else:
        offs_wn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, LOAD_W_LAYOUT))
        offs_wk = gl.arange(0, BLOCK_K_W, layout=gl.SliceLayout(1, LOAD_W_LAYOUT))

    rows_m = off_m + offs_xm
    # m_limit = per-expert tail (HAS_RAGGED_OFFS) or global M.
    pre_gather_mask = rows_m < m_limit
    if HAS_GATHER:
        rows_m_safe = gl.where(pre_gather_mask, rows_m, gl.zeros_like(rows_m))
        rows_m = gl.load(
            gather_idx_ptr + rows_m_safe, mask=pre_gather_mask, other=0
        ).to(gl.int32)
        # Post-gather rows_m is in global token-id space (size M_X);
        # mask out junk gather_idx values too. Don't conflate M_X with
        # ``M`` (= dispatched tile count, can exceed M_X for top-k>1).
        mask_m = pre_gather_mask & (rows_m < M_X)
    else:
        # Clamp OOB lanes to 0 so the buffer_load address stays in
        # bounds during HIP graph warm-up; mask still filters.
        rows_m = gl.where(pre_gather_mask, rows_m, gl.zeros_like(rows_m))
        mask_m = pre_gather_mask
    if cfg.W_VIA_VGPR:
        # W_VIA_VGPR skips the n-mask (launcher enforces N aligned).
        # Half-tile width under sliceN so the dummy mask matches the
        # actual offs_wn extent.
        _BN_MASK: gl.constexpr = (BLOCK_N // 2) if USE_SLICE_N else BLOCK_N
        mask_n = offs_wn < (_BN_MASK // 16)
    else:
        mask_n = (off_n + offs_wn) < N

    k_limit_x = gl.multiple_of(K // cfg.DIV_FACTOR_X, 16)
    k_limit_w = gl.multiple_of(K // cfg.DIV_FACTOR_W, 16)
    x_desc = AsyncCopyDescriptor.initialize(
        cfg,
        0,
        BLOCK_K_X,
        x_ptr,
        rows_m,
        offs_xk,
        stride_xm,
        stride_xk,
        mask_m[:, None],
        k_limit_x,
    )
    if cfg.W_VIA_VGPR:
        # Host-preshuffled W -> VGPR direct; bypasses LDS staging.
        TILE_BYTES: gl.constexpr = BLOCK_K_W * BLOCK_N
        offsets_b_vgpr = gl.expand_dims(offs_wk, 0) + gl.expand_dims(offs_wn, 1) * (
            BLOCK_K_W * 16
        )
        base_off_b_vgpr = w_base_offset + pid_n * TILE_BYTES
        w_desc = WVgprDescriptor(
            cfg,
            BLOCK_K_W,
            w_ptr,
            gl.to_tensor(N),  # K-iter advance step: idx * BK_W * N
            offsets_b_vgpr + base_off_b_vgpr,
            pred=gl.to_tensor(True),  # full-tile path: always in-bounds
        )
    elif W_TRANSPOSE:
        w_desc = AsyncCopyDescriptor.initialize(
            cfg,
            0,
            BLOCK_K_W,
            w_ptr,
            off_n + offs_wn,
            offs_wk,
            stride_wn,
            stride_wk,
            mask_n[:, None],
            k_limit_w,
            base_offset=w_base_offset,
            cache_modifier=W_CACHE_MODIFIER,
        )
    else:
        w_desc = AsyncCopyDescriptor.initialize(
            cfg,
            1,
            BLOCK_K_W,
            w_ptr,
            off_n + offs_wn,
            offs_wk,
            stride_wn,
            stride_wk,
            mask_n[None, :],
            k_limit_w,
            base_offset=w_base_offset,
            cache_modifier=W_CACHE_MODIFIER,
        )
    # SCALE_VIA_LDS uses post-swizzle HBM shape via buffer_load_to_shared;
    # other modes load scales G->VGPR via gl.load.
    if HAS_X_BLOCK_SCALE:
        if cfg.SCALE_VIA_LDS:
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
                rows_m_scale = rows_m
            x_scale_desc = AsyncCopyDescriptor.initialize(
                cfg,
                0,
                BLOCK_K_SCALE,
                x_scale_ptr,
                rows_m_scale,
                offs_xs_k,
                stride_xsm,
                stride_xsk,
                rows_m_scale[:, None] < M_X,
                K // cfg.SCALE_BLOCK,
            )
    else:
        x_scale_desc: gl.constexpr = 0

    if HAS_W_BLOCK_SCALE:
        if cfg.SCALE_VIA_LDS:
            BLOCK_N_PS: gl.constexpr = cfg.BLOCK_N_PRESHUFFLED
            BLOCK_K_S_PS_W: gl.constexpr = cfg.BLOCK_K_SCALE_PRESHUFFLED
            LW_S: gl.constexpr = cfg.load_layout_w_scale
            offs_ws_n = gl.arange(0, BLOCK_N_PS, layout=gl.SliceLayout(1, LW_S))
            offs_ws_k = gl.arange(0, BLOCK_K_S_PS_W, layout=gl.SliceLayout(0, LW_S))
            row_base_w_s = off_n // cfg.PRESHUFFLE_FACTOR
            rows_n_scale = row_base_w_s + offs_ws_n
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

    if USE_SLICE_MN:
        SUB_BM_MN: gl.constexpr = BLOCK_M // 2
        SUB_BN_MN: gl.constexpr = BLOCK_N // 2

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

        if W_TRANSPOSE:
            LOAD_W_SUB_LAYOUT_MN: gl.constexpr = _load_layout(
                BLOCK_K_W, SUB_BN_MN, NUM_WARPS, [1, 0], W_ELEM_BITS
            )
            offs_wn_sub_mn = gl.arange(
                0, SUB_BN_MN, layout=gl.SliceLayout(1, LOAD_W_SUB_LAYOUT_MN)
            )
            offs_wk_sub_mn = gl.arange(
                0, BLOCK_K_W, layout=gl.SliceLayout(0, LOAD_W_SUB_LAYOUT_MN)
            )
            mask_n_left_mn = (off_n + offs_wn_sub_mn) < N
            mask_n_right_mn = (off_n + SUB_BN_MN + offs_wn_sub_mn) < N
            w_desc_left_mn = AsyncCopyDescriptor.initialize(
                cfg,
                0,
                BLOCK_K_W,
                w_ptr,
                off_n + offs_wn_sub_mn,
                offs_wk_sub_mn,
                stride_wn,
                stride_wk,
                mask_n_left_mn[:, None],
                k_limit_w,
                base_offset=w_base_offset,
                cache_modifier=W_CACHE_MODIFIER,
            )
            w_desc_right_mn = AsyncCopyDescriptor.initialize(
                cfg,
                0,
                BLOCK_K_W,
                w_ptr,
                off_n + SUB_BN_MN + offs_wn_sub_mn,
                offs_wk_sub_mn,
                stride_wn,
                stride_wk,
                mask_n_right_mn[:, None],
                k_limit_w,
                base_offset=w_base_offset,
                cache_modifier=W_CACHE_MODIFIER,
            )
        else:
            LOAD_W_SUB_LAYOUT_MN: gl.constexpr = _load_layout(
                SUB_BN_MN, BLOCK_K_W, NUM_WARPS, [1, 0], W_ELEM_BITS
            )
            offs_wn_sub_mn = gl.arange(
                0, SUB_BN_MN, layout=gl.SliceLayout(0, LOAD_W_SUB_LAYOUT_MN)
            )
            offs_wk_sub_mn = gl.arange(
                0, BLOCK_K_W, layout=gl.SliceLayout(1, LOAD_W_SUB_LAYOUT_MN)
            )
            mask_n_left_mn = (off_n + offs_wn_sub_mn) < N
            mask_n_right_mn = (off_n + SUB_BN_MN + offs_wn_sub_mn) < N
            w_desc_left_mn = AsyncCopyDescriptor.initialize(
                cfg,
                1,
                BLOCK_K_W,
                w_ptr,
                off_n + offs_wn_sub_mn,
                offs_wk_sub_mn,
                stride_wn,
                stride_wk,
                mask_n_left_mn[None, :],
                k_limit_w,
                base_offset=w_base_offset,
                cache_modifier=W_CACHE_MODIFIER,
            )
            w_desc_right_mn = AsyncCopyDescriptor.initialize(
                cfg,
                1,
                BLOCK_K_W,
                w_ptr,
                off_n + SUB_BN_MN + offs_wn_sub_mn,
                offs_wk_sub_mn,
                stride_wn,
                stride_wk,
                mask_n_right_mn[None, :],
                k_limit_w,
                base_offset=w_base_offset,
                cache_modifier=W_CACHE_MODIFIER,
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
        acc = slice_mn_pgm.pipeline(K)
    elif USE_SLICE_N:
        SUB_BN: gl.constexpr = BLOCK_N // 2
        if cfg.W_VIA_VGPR:
            gl.static_assert(
                SUB_BN == 128 and BLOCK_K_W == 128 and NUM_WARPS == 4,
                "USE_SLICE_N + W_VIA_VGPR requires SUB_BN=BLOCK_K_W=128 "
                "and NUM_WARPS=4; the half-tile LOAD_W_LAYOUT bases assume "
                "this shape (re-derive otherwise).",
            )
            if cfg.SCALE_VIA_LDS:
                LOAD_W_HALF_LAYOUT: gl.constexpr = gl.DistributedLinearLayout(
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
                    shape=[SUB_BN // 16, BLOCK_K_W * 16],
                )
            else:
                LOAD_W_HALF_LAYOUT: gl.constexpr = gl.DistributedLinearLayout(
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
                    shape=[SUB_BN // 16, BLOCK_K_W * 16],
                )
            offs_wn_h = gl.arange(
                0, SUB_BN // 16, layout=gl.SliceLayout(1, LOAD_W_HALF_LAYOUT)
            )
            offs_wk_h = gl.arange(
                0, BLOCK_K_W * 16, layout=gl.SliceLayout(0, LOAD_W_HALF_LAYOUT)
            )
            offsets_h = gl.expand_dims(offs_wk_h, 0) + gl.expand_dims(offs_wn_h, 1) * (
                BLOCK_K_W * 16
            )
            TILE_BYTES_HALF: gl.constexpr = 128 * 128
            n_block_count = (N + 127) // 128
            bot_valid = (2 * pid_n + 1) < n_block_count
            base_off_top = w_base_offset + 2 * pid_n * TILE_BYTES_HALF
            base_off_bot = base_off_top + TILE_BYTES_HALF
            w_desc_top = WVgprDescriptor(
                cfg,
                BLOCK_K_W,
                w_ptr,
                gl.to_tensor(N),
                offsets_h + base_off_top,
                pred=gl.to_tensor(True),
                LOAD_BN=SUB_BN,
            )
            w_desc_bot = WVgprDescriptor(
                cfg,
                BLOCK_K_W,
                w_ptr,
                gl.to_tensor(N),
                offsets_h + base_off_bot,
                pred=bot_valid,
                LOAD_BN=SUB_BN,
            )
        elif W_TRANSPOSE:
            # LDS path, K-contig W tiles.
            LOAD_W_SUB_LAYOUT: gl.constexpr = _load_layout(
                BLOCK_K_W, SUB_BN, NUM_WARPS, [1, 0], W_ELEM_BITS
            )
            offs_wn_sub = gl.arange(
                0, SUB_BN, layout=gl.SliceLayout(1, LOAD_W_SUB_LAYOUT)
            )
            offs_wk_sub = gl.arange(
                0, BLOCK_K_W, layout=gl.SliceLayout(0, LOAD_W_SUB_LAYOUT)
            )
            mask_n_top = (off_n + offs_wn_sub) < N
            mask_n_bot = (off_n + SUB_BN + offs_wn_sub) < N
            w_desc_top = AsyncCopyDescriptor.initialize(
                cfg,
                0,
                BLOCK_K_W,
                w_ptr,
                off_n + offs_wn_sub,
                offs_wk_sub,
                stride_wn,
                stride_wk,
                mask_n_top[:, None],
                k_limit_w,
                base_offset=w_base_offset,
                cache_modifier=W_CACHE_MODIFIER,
            )
            w_desc_bot = AsyncCopyDescriptor.initialize(
                cfg,
                0,
                BLOCK_K_W,
                w_ptr,
                off_n + SUB_BN + offs_wn_sub,
                offs_wk_sub,
                stride_wn,
                stride_wk,
                mask_n_bot[:, None],
                k_limit_w,
                base_offset=w_base_offset,
                cache_modifier=W_CACHE_MODIFIER,
            )
        else:
            # LDS path, N-contig W tiles.
            LOAD_W_SUB_LAYOUT: gl.constexpr = _load_layout(
                SUB_BN, BLOCK_K_W, NUM_WARPS, [1, 0], W_ELEM_BITS
            )
            offs_wn_sub = gl.arange(
                0, SUB_BN, layout=gl.SliceLayout(0, LOAD_W_SUB_LAYOUT)
            )
            offs_wk_sub = gl.arange(
                0, BLOCK_K_W, layout=gl.SliceLayout(1, LOAD_W_SUB_LAYOUT)
            )
            mask_n_top = (off_n + offs_wn_sub) < N
            mask_n_bot = (off_n + SUB_BN + offs_wn_sub) < N
            w_desc_top = AsyncCopyDescriptor.initialize(
                cfg,
                1,
                BLOCK_K_W,
                w_ptr,
                off_n + offs_wn_sub,
                offs_wk_sub,
                stride_wn,
                stride_wk,
                mask_n_top[None, :],
                k_limit_w,
                base_offset=w_base_offset,
                cache_modifier=W_CACHE_MODIFIER,
            )
            w_desc_bot = AsyncCopyDescriptor.initialize(
                cfg,
                1,
                BLOCK_K_W,
                w_ptr,
                off_n + SUB_BN + offs_wn_sub,
                offs_wk_sub,
                stride_wn,
                stride_wk,
                mask_n_bot[None, :],
                k_limit_w,
                base_offset=w_base_offset,
                cache_modifier=W_CACHE_MODIFIER,
            )
        pgm = MoESliceNProgram.initialize(
            cfg,
            x_desc,
            w_desc_top,
            w_desc_bot,
            x_scale_desc,
            w_scale_desc,
        )
        acc = pgm.pipeline(K)
    else:
        pgm = MoEPipelinedProgram.initialize(
            cfg, x_desc, w_desc, x_scale_desc, w_scale_desc
        )
        if USE_WARP_PIPELINE:
            acc = pgm.warp_pipeline(K)
        else:
            acc = pgm.pipeline(K)

    if APPLY_X_GLOBAL_SCALE and not HAS_X_BLOCK_SCALE:
        x_global_scale = gl.load(x_global_scale_ptr)
        acc = acc * x_global_scale

    if HAS_BIAS:
        bias_offs = off_n + gl.arange(0, BLOCK_N, gl.SliceLayout(0, cfg.acc_layout))
        bias_mask = bias_offs < N
        bias = gl.load(
            bias_ptr + expert_id * stride_be + bias_offs,
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
            out = out / scale
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

    actual_n = (N // 2) if DO_SWIGLU else N
    if HAS_SCATTER:
        rows_y = gl.load(scatter_idx_ptr + offs_y_m_safe, mask=y_m_in_bounds, other=M)
        mask_y = (rows_y[:, None] < M) & (offs_y_n[None, :] < actual_n)
        rows_y_safe = gl.where(y_m_in_bounds, rows_y, gl.zeros_like(rows_y))
        y_offs = rows_y_safe[:, None] * stride_ym + offs_y_n[None, :] * stride_yn
    else:
        mask_y = (offs_y_m[:, None] < m_limit) & (offs_y_n[None, :] < actual_n)
        offs_y_m_2d_safe = offs_y_m_safe[:, None]
        y_offs = offs_y_m_2d_safe * stride_ym + offs_y_n[None, :] * stride_yn

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


def _pipelined_moe_kernel_repr(specialization) -> str:
    """Distinct rocprof names for schedule vs no-schedule specialization."""
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
    W_VIA_VGPR: gl.constexpr = False,
    W_PREFETCH: gl.constexpr = True,
):
    if GRID_N > 0:
        grid_n: gl.constexpr = GRID_N
        tiles_per_expert: gl.constexpr = BLOCKS_PER_EXPERT * GRID_N
    else:
        grid_n = (N + BLOCK_N - 1) // BLOCK_N
        tiles_per_expert = BLOCKS_PER_EXPERT * grid_n

    if USE_BLOCK_SCHEDULE:
        unpadded_m = gl.load(block_offs_ptr + N_EXPTS_TOT).to(gl.int32)

    for tile_idx in range(gl.program_id(0), NUM_TILES, gl.num_programs(0)):
        if USE_BLOCK_SCHEDULE:
            swizzled = _xcd_chiplet_swizzle(tile_idx, NUM_TILES, XCD_SWIZZLE)
            grid_m_padded = NUM_TILES // grid_n
            pid_m, pid_n = _group_m_swizzle(swizzled, grid_m_padded, grid_n, GROUP_M)
            do_tile = pid_m < unpadded_m
        else:
            # Dense path: tile_idx packs (compact_idx, intra-expert pid);
            # GROUP_M applies WITHIN one expert (W only reusable per expert).
            swizzled = _xcd_chiplet_swizzle(tile_idx, NUM_TILES, XCD_SWIZZLE)
            compact_idx = swizzled // tiles_per_expert
            local = swizzled % tiles_per_expert
            block_in_expert, pid_n = _group_m_swizzle(
                local, BLOCKS_PER_EXPERT, grid_n, GROUP_M
            )
            do_tile = True

        if do_tile:
            if USE_BLOCK_SCHEDULE:
                schedule_raw = gl.load(block_schedule_ptr + pid_m).to(
                    gl.uint32, bitcast=True
                )
                compact_idx = (schedule_raw & 0x0000FFFF).to(gl.int32)
                block_in_expert = (schedule_raw >> 16).to(gl.int32)

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
                APPLY_X_GLOBAL_SCALE=APPLY_X_GLOBAL_SCALE,
                USE_WARP_PIPELINE=USE_WARP_PIPELINE,
                USE_SLICE_MN=USE_SLICE_MN,
                USE_SLICE_N=USE_SLICE_N,
                HAS_FP8_QUANT_OUT=HAS_FP8_QUANT_OUT,
                W_VIA_VGPR=W_VIA_VGPR,
                W_PREFETCH=W_PREFETCH,
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
        has_x_scale=has_x_block_scale,
        has_w_scale=has_w_block_scale,
        k=K,
        x_format=x_format,
        num_buffers=num_buffers,
    )

    if w.ndim == 3:
        E, K_w_phys, N_w_phys = w.shape
    else:
        K_w_phys, N_w_phys = w.shape
        E = 1
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
        # Host pre-shuffled into 5-D HBM byte layout (W_VIA_VGPR path);
        # .contiguous() would clobber it. The descriptor reads N
        # directly for the K-iter stride, so stride_wn/stride_wk
        # aren't consulted -- only stride_we matters at launcher level.
        # ``w_transpose`` is irrelevant on this path.
        stride_wn, stride_wk = w3.stride(-2), w3.stride(-1)
    elif w_transpose:
        # K-contig W staged as [BN, BK] in LDS; view permuted for dot.
        w3 = w3.transpose(-1, -2).contiguous()
        stride_wn, stride_wk = w3.stride(-2), w3.stride(-1)
    else:
        # N-contig W staged as [BK, BN] in LDS.
        stride_wn, stride_wk = w3.stride(-1), w3.stride(-2)

    if has_w_block_scale:
        w_scale3 = w_scale if w_scale.ndim == 3 else w_scale.unsqueeze(0)
        w_scale_proc3 = _preprocess_scale(w_scale3, scale_load_mode)
        stride_wse = w_scale_proc3.stride(0)
        stride_wsn, stride_wsk = _scale_strides(w_scale_proc3, scale_load_mode)
        w_scale_buf = w_scale_proc3
    else:
        stride_wse = stride_wsn = stride_wsk = 0
        w_scale_buf = _make_dummy(x.device, torch.uint8)

    x_scale_proc = (
        _preprocess_scale(x_scale, scale_load_mode) if has_x_block_scale else None
    )
    stride_xsm, stride_xsk = _scale_strides(x_scale_proc, scale_load_mode)

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

    needs_scale_load = apply_x_global_scale and not has_x_block_scale
    if not needs_scale_load:
        x_global_scale_buf = _make_dummy(x.device, torch.float32)
    elif isinstance(x_global_scale, torch.Tensor):
        # Production: zero-copy passthrough of upstream FlexCtx scale.
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
        APPLY_X_GLOBAL_SCALE=apply_x_global_scale,
        USE_WARP_PIPELINE=use_warp_pipeline,
        USE_SLICE_MN=use_slice_mn,
        USE_SLICE_N=use_slice_n,
        HAS_FP8_QUANT_OUT=has_fp8_quant_out,
        W_VIA_VGPR=w_preshuffle,
        W_PREFETCH=False,
        GRID_N=grid_n,
        GROUP_M=group_m,
        XCD_SWIZZLE=xcd_swizzle,
        num_warps=num_warps,
    )

    common_kwargs["waves_per_eu"] = num_warps // 4

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
) -> tuple[int, int, int, int]:
    """Pick ``(BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS)`` for scaled-MFMA tiles.

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
            # combine + W_VIA_VGPR requires NW=4 (LinearLayout bases);
            # dispatch tolerates NW=8 since OUT_BLOCK_N halving sidesteps
            # the BN=256 / SLICE_N constraint at the BN=256 tile.
            bm, bn, bk, nw = (64, 256, 128, 8) if do_swiglu else (64, 256, 128, 4)
        elif do_swiglu:
            # dispatch+swiglu writes BLOCK_N//2 so the W_VIA_VGPR
            # LinearLayout static_assert (expects BN=128 or
            # USE_SLICE_N) is satisfied via OUT_BLOCK_N halving.
            bm, bn, bk, nw = 128, 256, 128, 4
        else:
            # combine path: keep BN=256 throughput but force BM<=64
            # so ``_resolve_use_slice_n`` enables USE_SLICE_N=True
            # (half-tile path), which the W_VIA_VGPR static_assert
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
    bm = max(_MFMA_M, min(bm, _round_up_int(M, _MFMA_M)))
    bn = max(_MFMA_M, min(bn, _round_up_int(N, _MFMA_M)))
    bk = max(_MFMA_SCALED_K, min(bk, _round_up_int(K, _MFMA_SCALED_K)))
    # Swizzle unswizzle reshape requires BLOCK_K_S >= 8 (= BLOCK_K
    # >= 256 with SCALE_BLOCK=32).
    if scale_load_mode == "swizzle":
        bk = max(bk, 256)
        bk = min(bk, _round_up_int(K, _MFMA_SCALED_K))
    return bm, bn, bk, nw


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
    num_buffers: int | None = None,
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
    bm, bn, bk, nw = _autotune_block(
        M,
        N,
        K,
        do_swiglu=True,
        x_format=x_format,
        scale_load_mode=scale_load_mode,
        slice_size=_ragged_slice_size(a_ragged_metadata, M),
    )
    block_m = block_m or bm
    block_n = block_n or bn
    block_k = block_k or bk
    num_warps = num_warps or nw
    num_buffers = (
        num_buffers
        if num_buffers is not None
        else _default_num_buffers(
            K,
            block_k,
            block_m=block_m,
            block_n=block_n,
            x_format=x_format,
            w_format="e2m1",
            has_x_block_scale=x_format == "e2m1",
            has_w_block_scale=True,
            scale_load_mode=scale_load_mode,
        )
    )
    use_warp_pipeline = (
        bool(use_warp_pipeline) if use_warp_pipeline is not None else False
    )
    use_slice_mn = _resolve_use_slice_mn(
        use_slice_mn,
        block_m,
        block_n,
        num_buffers=num_buffers,
        scale_load_mode=scale_load_mode,
        x_format=x_format,
        has_x_block_scale=x_format == "e2m1",
        has_w_block_scale=True,
        bk=block_k,
    )
    use_slice_n = _resolve_use_slice_n(
        use_slice_n,
        block_m,
        block_n,
        scale_load_mode=scale_load_mode,
        x_format=x_format,
        has_x_block_scale=x_format == "e2m1",
        has_w_block_scale=True,
        bk=block_k,
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
        out_quant_scale=out_quant_scale,
        w_preshuffle=w_preshuffle,
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
    num_buffers: int | None = None,
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
    bm, bn, bk, nw = _autotune_block(
        M,
        N,
        K,
        ragged=a_ragged_metadata is not None,
        x_format=x_format,
        scale_load_mode=scale_load_mode,
        slice_size=_ragged_slice_size(a_ragged_metadata, M),
    )
    block_m = block_m or bm
    block_n = block_n or bn
    block_k = block_k or bk
    num_warps = num_warps or nw
    num_buffers = (
        num_buffers
        if num_buffers is not None
        else _default_num_buffers(
            K,
            block_k,
            block_m=block_m,
            block_n=block_n,
            x_format=x_format,
            w_format="e2m1",
            has_x_block_scale=x_format == "e2m1",
            has_w_block_scale=True,
            scale_load_mode=scale_load_mode,
        )
    )
    use_warp_pipeline = (
        bool(use_warp_pipeline) if use_warp_pipeline is not None else False
    )
    use_slice_mn = _resolve_use_slice_mn(
        use_slice_mn,
        block_m,
        block_n,
        num_buffers=num_buffers,
        scale_load_mode=scale_load_mode,
        x_format=x_format,
        has_x_block_scale=x_format == "e2m1",
        has_w_block_scale=True,
        bk=block_k,
    )
    use_slice_n = _resolve_use_slice_n(
        use_slice_n,
        block_m,
        block_n,
        scale_load_mode=scale_load_mode,
        x_format=x_format,
        has_x_block_scale=x_format == "e2m1",
        has_w_block_scale=True,
        bk=block_k,
    )
    n_act_eff = int(n_expts_act) if n_expts_act is not None else 1
    if n_tokens is None:
        n_rows = M
        n_tokens_eff = M
    else:
        n_tokens_eff = int(n_tokens)
        n_rows = n_tokens_eff * n_act_eff
    y = torch.empty((n_rows, N), device=x.device, dtype=out_dtype)
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
        w_preshuffle=w_preshuffle,
    )
    if n_act_eff > 1:
        y = y.view(n_tokens_eff, n_act_eff, N).sum(dim=1)
    # Unpad N if the caller padded W for w_preshuffle. Padded W bytes
    # are 0 and padded scales are 127 so acc[:, N:N_padded] == 0.
    logical_n = int(getattr(w, "original_n", N))
    if logical_n != N:
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
    triggers the kernel's W_VIA_VGPR path.
    """
    if isinstance(w, torch.Tensor):
        shuffled = getattr(w, "_gluon_shuffled", None)
        if shuffled is not None:
            return shuffled
        return w
    if not isinstance(w, Tensor):
        return w
    raw = w.storage.data
    shuffled = getattr(raw, "_gluon_shuffled", None)
    if shuffled is not None:
        return shuffled
    return raw


def _extract_gluon_raw_s(s):
    """Return the raw uint8 scale tensor for Gluon's ``swizzle`` mode
    (bit-equivalent to upstream CDNA4MXScaleLayout.swizzle_data)."""
    if isinstance(s, torch.Tensor):
        return s
    if not isinstance(s, Tensor):
        return s
    return s.storage.data


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


def _kernel_priority() -> int:
    if _GLUON_DISABLED_ENV:
        return Priority.PORTABLE + 1  # 5
    return Priority.SPECIALIZED + 2  # 14


_MXFP4_SCALE_FORMAT = ScaleFormat(
    storage_dtype=torch.uint8,
    granularity="block",
    block_shape=(32,),
)
_FP8_PER_TENSOR_SCALE_FORMAT = ScaleFormat(
    storage_dtype=torch.float32,
    granularity="tensor",
)

_FUSED_WEIGHT_MXFP4 = tensor_format("mxfp4", torch.uint8, scale=_MXFP4_SCALE_FORMAT)


def _experts_fp8_mxfp4_signatures() -> frozenset:
    return frozenset(
        {
            format_signature(
                x=tensor_format(
                    "scaled-fp8",
                    torch.float8_e4m3fn,
                    scale=_FP8_PER_TENSOR_SCALE_FORMAT,
                ),
                weight=_FUSED_WEIGHT_MXFP4,
            ),
            format_signature(
                x=tensor_format(
                    "scaled-fp8",
                    torch.float8_e4m3fnuz,
                    scale=_FP8_PER_TENSOR_SCALE_FORMAT,
                ),
                weight=_FUSED_WEIGHT_MXFP4,
            ),
            format_signature(
                x=tensor_format("mxfp4", torch.uint8, scale=_MXFP4_SCALE_FORMAT),
                weight=_FUSED_WEIGHT_MXFP4,
            ),
        }
    )


_common = dict(
    solution="gluon",
    signatures=_experts_fp8_mxfp4_signatures(),
    capability=CapabilityRequirement(
        vendors=frozenset({"amd"}),
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
    ),
    priority=_kernel_priority(),
    tags={"throughput", "latency"},
)


@register_kernel(
    "moe",
    "experts",
    name="gluon_dispatch_gemm",
    features={"ragged_metadata", "dispatch_gemm"},
    **_common,
)
@register_kernel(
    "moe",
    "experts",
    name="gluon_gemm_combine",
    features={"ragged_metadata", "gemm_combine"},
    **_common,
)
def _gluon_mxfp_ragged_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    a_ragged_metadata=None,
    gather_indx=None,
    scatter_indx=None,
    precision_config=None,
    fused_activation=None,
    n_tokens=None,
    n_expts_act=None,
    **extra_kwargs,
) -> torch.Tensor | None:
    assert precision_config is not None
    w_mx_scale = getattr(precision_config, "b_mx_scale", None)
    assert w_mx_scale is not None

    flex = getattr(precision_config, "flex_ctx", None)
    lhs = getattr(flex, "lhs_data", None) if flex is not None else None
    fp8_dtype = getattr(lhs, "dtype", None) if lhs is not None else None
    fp8_scale = getattr(lhs, "scale", None) if lhs is not None else None

    x_mx_scale = getattr(precision_config, "a_mx_scale", None)
    if fp8_dtype is not None and x_mx_scale is not None:
        return

    if fp8_dtype is not None:
        x_format = "e4m3"
        x_global_scale = _global_scale_passthrough(fp8_scale)
        x_view = x.view(torch.uint8) if x.dtype != torch.uint8 else x
        x_scale = None
    elif x_mx_scale is not None:
        x_format = "e2m1"
        x_global_scale = 1.0
        x_view = x.view(torch.uint8) if x.dtype != torch.uint8 else x
        x_scale = _extract_gluon_raw_s(x_mx_scale)
        if not isinstance(x_scale, torch.Tensor):
            return
    else:
        return

    if precision_config.out_dtype is not None:
        out_dtype = precision_config.out_dtype
    elif x.dtype.is_floating_point:
        out_dtype = x.dtype
    else:
        out_dtype = torch.bfloat16

    w_raw = _extract_gluon_raw_w(w)
    s_raw = _extract_gluon_raw_s(w_mx_scale)

    if not isinstance(w_raw, torch.Tensor) or not isinstance(s_raw, torch.Tensor):
        return
    if w_raw.ndim != 3:
        return

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

    try:
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

    except Exception as exc:  # noqa: BLE001
        import logging
        import traceback

        logger = logging.getLogger("tokenspeed_kernel.ops.moe.gluon")
        logger.warning(
            "_gluon_mxfp_ragged_matmul falling back to upstream: %s: %s",
            type(exc).__name__,
            exc,
        )
        logger.warning(
            "  full chain:\n%s",
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
        return

    return


_GLUON_FUSED_SIGNATURES = frozenset(
    {
        format_signature(
            x=dense_tensor_format(torch.bfloat16),
            weight=_FUSED_WEIGHT_MXFP4,
        ),
        format_signature(
            x=dense_tensor_format(torch.float16),
            weight=_FUSED_WEIGHT_MXFP4,
        ),
        format_signature(
            x=tensor_format(
                "scaled-fp8",
                torch.float8_e4m3fn,
                scale=_FP8_PER_TENSOR_SCALE_FORMAT,
            ),
            weight=_FUSED_WEIGHT_MXFP4,
        ),
        format_signature(
            x=tensor_format(
                "scaled-fp8",
                torch.float8_e4m3fnuz,
                scale=_FP8_PER_TENSOR_SCALE_FORMAT,
            ),
            weight=_FUSED_WEIGHT_MXFP4,
        ),
    }
)


@register_kernel(
    "moe",
    "fused",
    name="gluon_mxfp4_fp8_fused_moe",
    features={"self_routing"},
    solution="gluon",
    capability=CapabilityRequirement(
        vendors=frozenset({"amd"}),
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
    ),
    signatures=_GLUON_FUSED_SIGNATURES,
    traits={"activation_dtype": frozenset({"fp8"})},
    priority=_kernel_priority(),
    tags={"throughput", "latency"},
)
def _gluon_mxfp_fused_moe(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w13_weight,
    w2_weight,
    *,
    w13_bias=None,
    w2_bias=None,
    w13_precision_config=None,
    w2_precision_config=None,
    w13_act_scale: torch.Tensor,
    w2_act_scale: torch.Tensor,
    top_k: int,
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
        w13_precision_config, w2_precision_config: ``PrecisionConfig``
            built by the backend (encodes fp8 LHS scale + mxfp4 RHS).
        w13_act_scale, w2_act_scale: per-tensor FP8 activation scales
            for the two GEMMs.
        top_k: routing top_k.
        swiglu_alpha / swiglu_limit: SwiGLU activation parameters.

    Lazy-imports the top-level ``moe_route`` / ``moe_experts`` /
    ``quantize_fp8`` to avoid a circular import (this module is imported
    by ``tokenspeed_kernel.ops.moe.__init__`` which defines those).
    """
    # Lazy imports to avoid the circular dependency described above.
    from tokenspeed_kernel import quantize_fp8
    from tokenspeed_kernel.ops.moe import moe_experts, moe_route

    n_tokens = router_logits.shape[0]

    ragged_metadata, gather_indx, scatter_indx, gate_scal = moe_route(
        router_logits,
        top_k,
        sm_first=False,
        dtype=router_logits.dtype,
        traits={"output_type": "ragged_metadata"},
        expected_kernel_name="triton_kernels_routing",
    )

    act = FusedActivation(
        FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
        (swiglu_alpha, swiglu_limit),
    )

    if hidden_states.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        gemm1_input = hidden_states
    else:
        gemm1_input = quantize_fp8(
            hidden_states,
            scale=w13_act_scale,
            solution="triton",
        )

    gluon_traits = {"weight_dtype": "mxfp4"}

    intermediate_cache = moe_experts(
        gemm1_input,
        w13_weight,
        w13_bias,
        a_ragged_metadata=ragged_metadata,
        gather_indx=gather_indx,
        precision_config=w13_precision_config,
        fused_activation=act,
        dtype=gemm1_input.dtype,
        weight_format="mxfp4",
        fp8_scale_granularity="tensor",
        features={"ragged_metadata", "dispatch_gemm"},
        traits=gluon_traits,
        expected_kernel_name="gluon_dispatch_gemm",
        out_quant_scale=w2_act_scale,
    )

    # Skip the redundant quantise when the fused ``out_quant_scale``
    # epilogue already wrote FP8.
    if intermediate_cache.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    ):
        gemm2_input = intermediate_cache
    else:
        gemm2_input = quantize_fp8(
            intermediate_cache,
            scale=w2_act_scale,
            solution="triton",
        )

    return moe_experts(
        gemm2_input,
        w2_weight,
        w2_bias,
        a_ragged_metadata=ragged_metadata,
        scatter_indx=scatter_indx,
        precision_config=w2_precision_config,
        gammas=gate_scal,
        n_tokens=n_tokens,
        n_expts_act=top_k,
        dtype=gemm2_input.dtype,
        weight_format="mxfp4",
        fp8_scale_granularity="tensor",
        features={"ragged_metadata", "gemm_combine"},
        traits=gluon_traits,
        expected_kernel_name="gluon_gemm_combine",
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
    big = gl.full([MP, EP], E, gl.int32, layout=LT)
    tcol = gl.expand_dims(gl.arange(0, TKP, layout=gl.SliceLayout(0, LT)), 0)  # [1,TKP]
    val_t = gl.full([MP, TKP], -1e30, gl.float32, layout=LT)  # finite -inf-ish
    idx_t = gl.zeros([MP, TKP], gl.int32, layout=LT)
    for _r in gl.static_range(TOPK):
        vmax = gl.max(cur, axis=1, keep_dims=True)  # [MP,1]
        ismax = (cur == vmax) & (col < E)
        amax = gl.min(gl.where(ismax, col, big), axis=1, keep_dims=True)  # [MP,1]
        sel = tcol == _r  # [1,TKP]
        val_t = gl.where(sel, vmax, val_t)  # write column _r
        idx_t = gl.where(sel, amax, idx_t)
        cur = gl.where(col == amax, NEG, cur)  # drop chosen expert

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
    # The kernel's BlockedLayouts assume a 64-lane wavefront, so the path is
    # gfx950 (CDNA4) only; every other arch falls back to the generic pipeline.
    if not current_platform().is_cdna4:
        return False
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


@register_kernel(
    "moe",
    "route",
    name="gluon_decode_routing_gfx950",
    solution="gluon",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    ),
    signatures=format_signatures(
        "logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
    ),
    traits={"output_type": frozenset({"ragged_metadata"})},
    # Narrowly gated on gfx950 + small-M decode shapes -> SPECIALIZED. Selection
    # prefers this over the portable triton route on gfx950; the runtime guard
    # below degrades to the generic route for shapes it does not cover.
    # TOKENSPEED_MOE_GLUON=0 demotes it below the triton route, matching the
    # other gluon moe kernels' disable switch.
    priority=Priority.PORTABLE if _GLUON_DISABLED_ENV else Priority.SPECIALIZED,
)
def gluon_decode_routing_gfx950(
    logits: torch.Tensor,
    n_expts_act: int,
    sm_first: bool = False,
    dtype: torch.dtype | None = None,
) -> tuple[RaggedTensorMetadata, torch.Tensor, torch.Tensor, torch.Tensor]:
    """gfx950 small-M decode route, registered alongside the generic Triton route.

    The single-block-collapse fused kernel only covers the decode regime
    (``M <= SMALLM_MAX_M`` and the bounds ``gluon_route_supported`` checks); the
    bound is dynamic, so it cannot be a static selection trait. Shapes outside
    it fall back to the registered ``triton_kernels_routing`` generic pipeline,
    keeping that kernel free of any gluon coupling.
    """
    if dtype is None:
        dtype = logits.dtype
    n_tokens = logits.shape[0]
    if (
        not sm_first
        and n_tokens <= SMALLM_MAX_M
        and gluon_route_supported(logits, n_expts_act, dtype)
    ):
        return gluon_fused_route(logits, n_expts_act, dtype=dtype)

    generic = KernelRegistry.get().get_impl("triton_kernels_routing")
    return generic(logits, n_expts_act, sm_first, dtype)
