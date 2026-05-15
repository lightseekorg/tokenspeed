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

import tokenspeed_kernel.thirdparty.triton_kernels  # noqa: F401  (side effect)
import tokenspeed_triton  # noqa: F401
import torch
from tokenspeed_kernel._triton import tl, triton  # noqa: F401  (kept for parity)
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_triton.experimental import gluon
from tokenspeed_triton.experimental.gluon import language as gl
from tokenspeed_triton.language.core import _aggregate as aggregate
from triton_kernels.matmul import FlexCtx as _UpstreamFlexCtx  # noqa: F401
from triton_kernels.matmul import (  # noqa: F401
    FusedActivation as _UpstreamFusedActivation,
)
from triton_kernels.matmul import (  # noqa: F401
    PrecisionConfig as _UpstreamPrecisionConfig,
)
from triton_kernels.matmul import matmul as _upstream_matmul
from triton_kernels.tensor import RaggedTensorMetadata  # noqa: F401
from triton_kernels.tensor import Tensor as _UpstreamWrappedTensor  # noqa: F401
from triton_kernels.tensor import convert_layout as _upstream_convert_layout
from triton_kernels.tensor_details.layout_details.strided import (  # noqa: F401
    StridedLayout as _UpstreamStridedLayout,
)

# ---------------------------------------------------------------------------
# Env knob
# ---------------------------------------------------------------------------

# The Gluon MoE kernels target throughput/latency on AMD MI355 (CDNA4 / gfx950)
# and outperform the upstream ``triton_kernels`` MoE GEMM there. They are
# therefore enabled by default on that platform; set
# ``TOKENSPEED_MOE_GLUON=0`` (or false/no/off) to fall back to the upstream
# triton_kernels path -- useful for A/B comparisons and for working around a
# regression without rebuilding.
_GLUON_DISABLE_VALUES = {"0", "false", "no", "off", "disable", "disabled"}
_GLUON_DISABLED_ENV = (
    os.environ.get("TOKENSPEED_MOE_GLUON", "").strip().lower() in _GLUON_DISABLE_VALUES
)


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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CDNA4 LDS budget (per CU). 160 KB total; Triton scratch needs slack so
# the practical ceiling is around 144 KB.
_CDNA4_LDS_BYTES = 160 * 1024

# Software-pipelined defaults. Picked so 2 * (BM * BK + BK * BN) * 2 bytes
# (bf16) fits comfortably in 160 KB:
#   2 * (128 * 64 + 64 * 128) * 2 = 65536 bytes ~= 64 KB.
# This leaves the remaining ~80 KB for other allocations (scales when we
# add the mxfp4 path, etc.).
_DEFAULT_BLOCK_M = 128
_DEFAULT_BLOCK_N = 128
_DEFAULT_BLOCK_K = 64
_DEFAULT_NUM_WARPS = 4
_DEFAULT_NUM_BUFFERS = 2

# CDNA4 per-CU LDS budget (gfx950 = 160 KB).
_CDNA4_LDS_BUDGET = 163840

# MI355 (CDNA4 / gfx950) has 256 CUs. We pick a persistent grid that
# slightly over-subscribes -- 2x CUs -- so the runtime can hide tail-tile
# imbalance across waves while still avoiding launch-overhead bloat.
_CDNA4_NUM_CUS = 256
_PERSISTENT_OVERSUBSCRIBE = 2
# Persistent kernel is helpful when the launch grid is smaller than
# ~3x the CU count (CTAs queue up on each CU and pay launch overhead);
# above that the regular grid amortises launch better and persistent
# adds bookkeeping cost. Tuned for gpt-oss-120b decode (B=1..32).
_PERSISTENT_TILES_THRESHOLD = _CDNA4_NUM_CUS * 3


# ---------------------------------------------------------------------------
# Layout factories (gluon constexpr functions)
# ---------------------------------------------------------------------------


@gluon.constexpr_function
def _store_layout(num_warps: int):
    # Output store layout (shared between bf16/fp16 and scaled paths).
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
    # K_PER_THREAD * elem_bits <= 128 (CDNA4 direct-to-LDS coalesce cap).
    max_vec = max(1, 128 // elem_bits)
    K_PER_THREAD: gl.constexpr = min(max_vec, block_k)
    LANES_K = block_k // K_PER_THREAD
    LANES_NONK = 64 // LANES_K
    # How many non-K elements one warp covers without warps along non-K.
    NONK_PER_WARP = LANES_NONK
    # Split the warps so that ``warps_K * warps_NONK == num_warps`` and
    # the per-CTA tile equals exactly ``[block_K, block_NONK]``.
    if block_nonk >= NONK_PER_WARP:
        WARPS_NONK = block_nonk // NONK_PER_WARP
        if WARPS_NONK > num_warps:
            WARPS_NONK = num_warps
        WARPS_K = num_warps // WARPS_NONK
    else:
        # Tile is narrower than one warp's natural NONK footprint.
        # Shrink the per-warp NONK footprint and put more lanes on K.
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
    """Build a ``BlockedLayout`` whose last-axis registers come in pairs,
    so that ``reshape([..., 2]) -> split`` lowers to a free lane swap.

    ``block_n_full = 2 * OUT_BLOCK_N`` is the GEMM accumulator N width
    BEFORE the swiglu halve. We use ``size_per_thread = [1, 4]`` so each
    thread holds two ``(gate, linear)`` pairs in adjacent registers; the
    new innermost axis after reshape lives entirely in registers and the
    split can lower to a zero-cost SliceLayout pick.

    The fixed natural per-CTA tile is ``[2 * num_warps, 128]`` (i.e.
    8 warps -> [16, 128]); both BLOCK_M and BLOCK_N replicate over this
    tile when bigger, which is fine for split (each replica holds an
    independent fragment with its own (gate, linear) pairs).
    """
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
    """SwiGLU (gated GELU) reduction along the inner-most axis.

    The accumulator has shape ``[BLOCK_M, 2 * OUT_BLOCK_N]``; pairs of
    adjacent N-columns are interpreted as ``(gate, linear)``. We split
    them along the last axis and apply the canonical
    ``s = gate / (1 + exp(-alpha * gate))`` mixing rule.

    The MFMA-scaled accumulator layout on CDNA4 places adjacent N
    elements in DIFFERENT lanes, so a naive
    ``reshape -> split`` fails to infer a SliceLayout for the new
    innermost axis (``gl.split`` requires the split axis to live in
    registers). We pre-convert the accumulator to a ``BlockedLayout``
    with ``size_per_thread = [1, 4]`` along N so adjacent N-elements
    sit in the same thread's registers; the reshape then folds the
    inner ``2`` axis into registers and ``gl.split`` becomes a free
    lane pick.
    """
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
        linear = gl.maximum(gl.minimum(linear, limit), -limit)
    s = gate / (1.0 + gl.exp(-alpha * gate))
    return s * (linear + 1.0)


# ---------------------------------------------------------------------------
# Scaled MFMA MoE kernel (mxfp4 / fp8 / bf16 / fp16 + optional e8m0 scales)
# ---------------------------------------------------------------------------


@gluon.constexpr_function
def get_mfma_layout(
    num_warps: int, use_mfma_scaled: bool, scale_preshuffle: bool = False
) -> gl.constexpr:
    # CDNA4 (gfx950): scaled MFMA = [16, 16, 128] (mxfp/fp8); regular = [16, 16, 32].
    # tiles_per_warp=[2,2] when scales are preshuffled+LDS-staged: lets a single
    # warp issue 2x2 MFMA tiles per K step so the per-tile mfma_scale_layout
    # absorbs the 5-D unswizzle view cleanly.
    assert num_warps in (4, 8), "MI355 MoE kernel currently supports 4 or 8 warps."
    warps_m = 2 if num_warps >= 4 else 1
    warps_n = num_warps // warps_m
    instr_shape = [16, 16, 128] if use_mfma_scaled else [16, 16, 32]
    tiles_per_warp = [2, 2] if scale_preshuffle else [1, 1]
    return gl.amd.cdna4.AMDMFMALayout(
        version=4,
        instr_shape=instr_shape,
        transposed=True,
        warps_per_cta=[warps_m, warps_n],
        tiles_per_warp=tiles_per_warp,
    )


_SCALE_LOAD_MODES = ("bypass", "transpose", "swizzle", "cdna4_upstream")
_SCALE_PRESHUFFLE_FACTOR = 32
_SCALE_KWIDTH = 4
_SCALE_ASYNC_VEC = 4  # 32-bit, smallest direct-to-LDS unit on CDNA4.

# Constants matching triton_kernels' CDNA4MXScaleLayout.
_NON_K_PRESHUFFLE_BLOCK_SIZE = 32
_ALIGN_K_SCALE_CDNA4_UPSTREAM = 8
_ALIGN_N_CDNA4_UPSTREAM = 32
# Inner reshape factor for the 7-D unswizzle: K_SCALE_pad must be a
# multiple of this for `unswizzle_mx_scale_cdna4` to be well-defined.
_CDNA4_UPSTREAM_K_S_INNER = 8


def _effective_scale_load_mode(
    mode: str,
    block_m: int,
    block_n: int,
    block_k: int,
    scale_block: int,
    has_x_scale: bool,
    has_w_scale: bool,
    k: int | None = None,
    a_format: str | None = None,
    num_buffers: int | None = None,
) -> str:
    # swizzle -> bypass fallback when the post-swizzle tile is too narrow
    # along NONK for canCoalesceWriteIntoSharedMemory to succeed, or when
    # K_S = K/scale_block is not a multiple of SCALE_KWIDTH (host swizzle
    # reshape requires K_S % KWIDTH == 0).
    if mode == "cdna4_upstream":
        # Upstream `unswizzle_mx_scale_cdna4` reshapes the K_S-block dim
        # into `(BLOCK_K_S // 8, 4, 16, 2, 2, 1)`; this requires
        # ``BLOCK_K_S = BLOCK_K // SCALE_BLOCK >= 8`` (i.e. BLOCK_K >= 256
        # for SCALE_BLOCK=32). N-side preshuffle factor is 32 so BLOCK_N
        # (and BLOCK_M when X has a scale) must be a multiple of 32.
        # Unlike "swizzle", cdna4_upstream cannot silently fall back to
        # "bypass": the input scale tensor is already in the upstream
        # storage layout (K_S_pad*32 contig) which "bypass" cannot read.
        # So we hard-assert.
        bk_s = block_k // scale_block
        assert bk_s >= _CDNA4_UPSTREAM_K_S_INNER, (
            f"cdna4_upstream requires BLOCK_K // SCALE_BLOCK >= "
            f"{_CDNA4_UPSTREAM_K_S_INNER} (got BLOCK_K={block_k}, "
            f"SCALE_BLOCK={scale_block} -> BLOCK_K_S={bk_s}). Bump "
            f"BLOCK_K to >= {_CDNA4_UPSTREAM_K_S_INNER * scale_block}."
        )
        if has_x_scale:
            assert block_m % _NON_K_PRESHUFFLE_BLOCK_SIZE == 0, (
                f"cdna4_upstream requires BLOCK_M % "
                f"{_NON_K_PRESHUFFLE_BLOCK_SIZE} == 0 when x_scale is "
                f"present (got BLOCK_M={block_m})."
            )
        if has_w_scale:
            assert block_n % _NON_K_PRESHUFFLE_BLOCK_SIZE == 0, (
                f"cdna4_upstream requires BLOCK_N % "
                f"{_NON_K_PRESHUFFLE_BLOCK_SIZE} == 0 when w_scale is "
                f"present (got BLOCK_N={block_n})."
            )
        return "cdna4_upstream"
    if mode != "swizzle":
        return mode
    if k is not None:
        k_s = k // scale_block
        if k_s % _SCALE_KWIDTH != 0:
            return "bypass"
    PF = _SCALE_PRESHUFFLE_FACTOR
    bk_s_ps = (block_k // scale_block) * PF
    lanes_nonk = max(1, _SCALE_ASYNC_VEC * 64 // bk_s_ps)
    if has_x_scale and (block_m // PF) < lanes_nonk:
        return "bypass"
    if has_w_scale and (block_n // PF) < lanes_nonk:
        return "bypass"
    # LDS-budget fallback: with very large tiles (e.g. BM*BK + BN*BK +
    # both scale LDS chunks * NB), swizzle can bust CDNA4's 160 KB LDS
    # budget while the same tile without scales-in-LDS still fits.
    # TASKS.md update-8 specifically calls this corner case out: if
    # "with-scale" overflows LDS but "without-scale" fits, route the
    # scales through VGPR (`bypass`) rather than refusing to compile.
    if (
        a_format is not None
        and num_buffers is not None
        and (has_x_scale or has_w_scale)
    ):
        bytes_with = _scaled_lds_bytes(
            block_m,
            block_n,
            block_k,
            has_x_block_scale=has_x_scale,
            has_w_block_scale=has_w_scale,
            a_format=a_format,
            num_buffers=num_buffers,
        )
        if bytes_with > _CDNA4_LDS_BUDGET:
            bytes_without = _scaled_lds_bytes(
                block_m,
                block_n,
                block_k,
                has_x_block_scale=False,
                has_w_block_scale=False,
                a_format=a_format,
                num_buffers=num_buffers,
            )
            if bytes_without <= _CDNA4_LDS_BUDGET:
                return "bypass"
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
    NUM_BUFFERS: gl.constexpr
    NUM_LOADS_IN_BATCH: gl.constexpr

    SCALE_BLOCK: gl.constexpr
    WITH_X_MX_SCALE: gl.constexpr
    WITH_W_MX_SCALE: gl.constexpr
    SCALE_LOAD_MODE: gl.constexpr
    SCALE_VIA_LDS: gl.constexpr
    IS_CDNA4_UPSTREAM: gl.constexpr
    PRESHUFFLE_FACTOR: gl.constexpr
    SCALE_KWIDTH: gl.constexpr
    SCALE_K_INNER_CDNA4_UP: gl.constexpr
    BLOCK_M_PRESHUFFLED: gl.constexpr
    BLOCK_N_PRESHUFFLED: gl.constexpr
    BLOCK_K_SCALE_PRESHUFFLED: gl.constexpr

    NUM_SUBTILES: gl.constexpr
    EVEN_K: gl.constexpr
    USE_GATHER: gl.constexpr
    USE_MFMA_SCALED: gl.constexpr

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
        self.WITH_X_MX_SCALE = gl.constexpr(WITH_X_MX_SCALE)
        self.WITH_W_MX_SCALE = gl.constexpr(WITH_W_MX_SCALE)
        self.SCALE_LOAD_MODE = gl.constexpr(SCALE_LOAD_MODE)
        self.SCALE_BLOCK = gl.constexpr(SCALE_BLOCK)
        self.DIV_FACTOR_X = gl.constexpr(2 if DTYPE_X == "e2m1" else 1)
        self.DIV_FACTOR_W = gl.constexpr(2 if DTYPE_W == "e2m1" else 1)
        self.DTYPE_X = gl.constexpr(DTYPE_X)
        self.DTYPE_W = gl.constexpr(DTYPE_W)

        _scale_via_lds = SCALE_LOAD_MODE in ("swizzle", "cdna4_upstream") and (
            WITH_X_MX_SCALE or WITH_W_MX_SCALE
        )
        self.SCALE_VIA_LDS = gl.constexpr(_scale_via_lds)
        # IS_CDNA4_UPSTREAM controls the in-LDS unswizzle reshape pattern:
        # True  -> upstream's 7-D `unswizzle_mx_scale_cdna4`
        # False -> AITer's 5-D pattern (when SCALE_LOAD_MODE == "swizzle")
        self.IS_CDNA4_UPSTREAM = gl.constexpr(SCALE_LOAD_MODE == "cdna4_upstream")
        self.PRESHUFFLE_FACTOR = gl.constexpr(_SCALE_PRESHUFFLE_FACTOR)
        self.SCALE_KWIDTH = gl.constexpr(_SCALE_KWIDTH)
        # cdna4_upstream uses an inner-8 split for the K_S axis instead
        # of an inner-KWIDTH=4; expose it as a constexpr so the unswizzle
        # path can use it directly.
        self.SCALE_K_INNER_CDNA4_UP = gl.constexpr(_CDNA4_UPSTREAM_K_S_INNER)
        self.BLOCK_M_PRESHUFFLED = gl.constexpr(BLOCK_M // _SCALE_PRESHUFFLE_FACTOR)
        self.BLOCK_N_PRESHUFFLED = gl.constexpr(BLOCK_N // _SCALE_PRESHUFFLE_FACTOR)
        self.BLOCK_K_SCALE_PRESHUFFLED = gl.constexpr(
            (BLOCK_K // SCALE_BLOCK) * _SCALE_PRESHUFFLE_FACTOR
        )

        # NUM_LOADS_IN_BATCH unused (wait_group counts commit groups); kept
        # informational. With SCALE_VIA_LDS we issue 4 buffer_load_to_shared
        # per commit; otherwise just 2 (X, W).
        self.NUM_LOADS_IN_BATCH = gl.constexpr(
            2 + (1 if WITH_X_MX_SCALE else 0) + (1 if WITH_W_MX_SCALE else 0)
            if _scale_via_lds
            else 2
        )
        self.NUM_SUBTILES = gl.constexpr(NUM_SUBTILES)
        self.EVEN_K = gl.constexpr(EVEN_K)
        self.USE_GATHER = gl.constexpr(USE_GATHER)
        _SCALED_FORMATS = ("e2m1", "e4m3", "e5m2")
        self.USE_MFMA_SCALED = gl.constexpr(
            DTYPE_X in _SCALED_FORMATS and DTYPE_W in _SCALED_FORMATS
        )
        self.NUM_WARPS = gl.constexpr(NUM_WARPS)

        BLOCK_K_SCALE = BLOCK_K // SCALE_BLOCK
        self.index_type = gl.constexpr(index_type)

        MFMA_LAYOUT: gl.constexpr = get_mfma_layout(
            NUM_WARPS,
            self.USE_MFMA_SCALED,
            scale_preshuffle=_scale_via_lds,
        )

        # k_width = per-thread K extent: 16 for mfma_scaled, 8 for mfma.
        DOT_K_WIDTH: gl.constexpr = 16 if self.USE_MFMA_SCALED else 8

        NUM_SUBTILES_M = self.NUM_SUBTILES[0]
        NUM_SUBTILES_N = self.NUM_SUBTILES[1]
        NUM_SUBTILES_K = self.NUM_SUBTILES[2]

        self.dot_layout_x = gl.constexpr(
            gl.DotOperandLayout(
                operand_index=0, parent=MFMA_LAYOUT, k_width=DOT_K_WIDTH
            )
        )
        self.dot_layout_w = gl.constexpr(
            gl.DotOperandLayout(
                operand_index=1, parent=MFMA_LAYOUT, k_width=DOT_K_WIDTH
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

        # vec=16 elements (=128 bits for uint8) matches the
        # buffer_load_to_shared 128-bit coalesce requirement on CDNA4.
        vec: gl.constexpr = 16
        per_phase: gl.constexpr = 2
        max_phase: gl.constexpr = 8
        self.shared_layout_x = gl.constexpr(
            gl.SwizzledSharedLayout(vec, per_phase, max_phase, order=[1, 0])
        )
        self.shared_layout_w = gl.constexpr(
            gl.SwizzledSharedLayout(vec, per_phase, max_phase, order=[1, 0])
        )

        # Scale LDS layout (only used when SCALE_VIA_LDS): vec=4 (32-bit) is
        # the smallest direct-to-LDS vector on CDNA4. max_phase=1 = no swizzle
        # so the 5-D unswizzle view on the LDS slot is a pure address remap.
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
    def async_wait(self, waitcnt):
        gl.amd.cdna4.async_copy.wait_group(waitcnt)


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
    # Layout for buffer_load_to_shared of a swizzled scale tile shape
    # [BLOCK_NONK_PS, BLOCK_K_PS] (uint8). vec=4 (32-bit) is CDNA4's smallest
    # supported direct-to-LDS vector. Threads spread along K first; remaining
    # warps either tile NONK or replicate K (over-cover) -- both are valid
    # under canCoalesceWriteIntoSharedMemory which only checks that
    # coalesced = id(vec)*id(64) divides srcToShared.
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
    offsets: gl.tensor
    off_k: gl.tensor
    masks_nonk: gl.tensor
    k_limit: gl.tensor
    BLOCK_K: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        cfg: MoEConfig,
        op_idx,
        BLOCK_K,
        ptr,
        dtype,
        stride_k,
        offsets,
        off_k,
        masks_nonk,
        k_limit,
    ):
        self.cfg = cfg
        self.op_idx = gl.constexpr(op_idx)
        self.BLOCK_K = gl.constexpr(BLOCK_K)
        self.ptr = ptr
        self.dtype = gl.constexpr(dtype)
        self.stride_k = stride_k
        self.offsets = offsets
        self.off_k = off_k
        self.masks_nonk = masks_nonk
        self.k_limit = k_limit

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
    ):
        # ``base_offset`` is folded into per-thread offsets (not added to
        # ``ptr``); a runtime ptr+gep base trips ``unrealized_conversion_cast``
        # in BufferLoadToLocalOpConversion.
        offsets = (
            gl.expand_dims(off_k, op_idx) * stride_k
            + gl.expand_dims(off_nonk, 1 - op_idx) * stride_nonk
            + base_offset
        )
        dtype: gl.constexpr = ptr.dtype.element_ty
        stride_k_t = gl.to_tensor(stride_k)
        return AsyncCopyDescriptor(
            cfg,
            op_idx,
            BLOCK_K,
            ptr,
            dtype,
            stride_k_t,
            offsets,
            off_k,
            masks_nonk,
            k_limit,
        )

    @gluon.jit
    def issue_async_load(self, idx, buffer, pred=1):
        # Fold the per-iteration K step into ``offsets`` and keep the base ptr
        # equal to ``self.ptr``. A runtime ``tt.addptr`` on the base trips
        # ``unrealized_conversion_cast`` in BufferLoadToLocalOpConversion for
        # certain tile shapes (e.g. mxfp4 X with BLOCK_K_X=128).
        NUM_BUFFERS: gl.constexpr = self.cfg.NUM_BUFFERS
        EVEN_K: gl.constexpr = self.cfg.EVEN_K
        off_k_step = idx * self.BLOCK_K
        offsets = self.offsets + off_k_step * self.stride_k
        pred_b = pred != 0
        if EVEN_K:
            mask = (
                gl.full(offsets.shape, True, gl.int1, layout=offsets.type.layout)
                & pred_b
            )
        else:
            mask_k = gl.expand_dims(off_k_step + self.off_k, self.op_idx) < self.k_limit
            mask = mask_k & self.masks_nonk & pred_b
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            buffer.index(idx % NUM_BUFFERS),
            self.ptr,
            offsets,
            mask=mask,
            other=0,
        )

    @gluon.jit
    def issue_local_load(
        self, idx, buffer, layout: gl.constexpr, do_permute: gl.constexpr = False
    ):
        NUM_BUFFERS: gl.constexpr = self.cfg.NUM_BUFFERS
        slot = buffer.index(idx % NUM_BUFFERS)
        if do_permute:
            slot = slot.permute([1, 0])
        return slot.load(layout=layout)

    @gluon.jit
    def issue_local_load_unswizzle(
        self,
        idx,
        buffer,
        layout: gl.constexpr,
        BLOCK_NONK_PS: gl.constexpr,
        BLOCK_NONK: gl.constexpr,
        BLOCK_K_SCALE: gl.constexpr,
        PRESHUFFLE_FACTOR: gl.constexpr,
        SCALE_KWIDTH: gl.constexpr,
    ):
        # AITer CDNA4 swizzle: LDS holds [BLOCK_NONK_PS, BLOCK_K_S * PF] uint8;
        # the 5-D unswizzle view (reshape + permute + reshape) restores the
        # natural [BLOCK_NONK, BLOCK_K_S] layout before local_load.
        NUM_BUFFERS: gl.constexpr = self.cfg.NUM_BUFFERS
        slot = buffer.index(idx % NUM_BUFFERS)
        slot_5d = slot.reshape(
            (
                BLOCK_NONK_PS,
                BLOCK_K_SCALE // SCALE_KWIDTH,
                PRESHUFFLE_FACTOR // 4,
                4,
                SCALE_KWIDTH,
            )
        )
        slot_perm = slot_5d.permute((0, 3, 2, 1, 4))
        slot_2d = slot_perm.reshape((BLOCK_NONK, BLOCK_K_SCALE))
        return slot_2d.load(layout=layout)

    @gluon.jit
    def issue_local_load_unswizzle_cdna4_upstream(
        self,
        idx,
        buffer,
        layout: gl.constexpr,
        BLOCK_NONK_PS: gl.constexpr,
        BLOCK_NONK: gl.constexpr,
        BLOCK_K_SCALE: gl.constexpr,
    ):
        # Upstream CDNA4MXScaleLayout swizzle: LDS holds the same
        # [BLOCK_NONK_PS=BLOCK_N/32, BLOCK_K_S * 32] uint8 tile shape as
        # the AITer path, but the in-tile reordering follows upstream's
        # 7-D pattern (`triton_kernels.tensor_details.layout_details.cdna4_scale.unswizzle_mx_scale_cdna4`).
        # That requires BLOCK_K_SCALE % 8 == 0; the host preprocess pads
        # the global K_SCALE up to a multiple of 8 for us.
        NUM_BUFFERS: gl.constexpr = self.cfg.NUM_BUFFERS
        slot = buffer.index(idx % NUM_BUFFERS)
        slot_7d = slot.reshape((BLOCK_NONK_PS, BLOCK_K_SCALE // 8, 4, 16, 2, 2, 1))
        slot_perm = slot_7d.permute((0, 5, 3, 1, 4, 2, 6))
        slot_2d = slot_perm.reshape((BLOCK_NONK, BLOCK_K_SCALE))
        return slot_2d.load(layout=layout)


@gluon.jit
def _load_scale_tile_via_gl_load(desc, mfma_idx, scale_layout: gl.constexpr):
    # G->VGPR scale load via gl.load. Direct-to-LDS for scales is structurally
    # impossible on CDNA4 for many tile shapes. K-mask the load so the OOB
    # scale slots become 0; otherwise junk reads (e.g. 0xFF e8m0 == NaN)
    # poison the mfma_scaled accumulator on the partial last K tile.
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
    w_buffer: gl.shared_memory_descriptor
    x_scale_buffer: gl.shared_memory_descriptor | gl.constexpr
    w_scale_buffer: gl.shared_memory_descriptor | gl.constexpr
    x_desc: AsyncCopyDescriptor
    w_desc: AsyncCopyDescriptor
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
        self.w_buffer = w_buffer
        # constexpr fallback set here (Python ctx) so the type check is happy;
        # initialize() in @gluon.jit ctx can't make a real constexpr.
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
    def _load_xw(self, mfma_idx):
        cfg = self.cfg
        x = self.x_desc.issue_local_load(
            mfma_idx,
            self.x_buffer,
            cfg.dot_layout_x,
        )
        w = self.w_desc.issue_local_load(
            mfma_idx,
            self.w_buffer,
            cfg.dot_layout_w,
            do_permute=cfg.W_TRANSPOSE,
        )
        return x, w

    @gluon.jit
    def issue_global_loads(self, load_idx, pred=1):
        # X / W always go through LDS via async copy. Scales go through LDS
        # only when SCALE_VIA_LDS (== swizzle mode).
        cfg = self.cfg
        self.x_desc.issue_async_load(load_idx, self.x_buffer, pred)
        self.w_desc.issue_async_load(load_idx, self.w_buffer, pred)
        if cfg.SCALE_VIA_LDS:
            if cfg.WITH_X_MX_SCALE:
                self.x_scale_desc.issue_async_load(load_idx, self.x_scale_buffer, pred)
            if cfg.WITH_W_MX_SCALE:
                self.w_scale_desc.issue_async_load(load_idx, self.w_scale_buffer, pred)
        gl.amd.cdna4.async_copy.commit_group()
        return load_idx + 1

    @gluon.jit
    def issue_local_loads(self, mfma_idx):
        cfg = self.cfg
        x, w = self._load_xw(mfma_idx)

        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK

        # Scales only consumed by mfma_scaled; bf16/fp16 path returns 0.
        if cfg.USE_MFMA_SCALED:
            # Dummy scales use e8m0=127 (== 2^0 = 1.0) so the dot is identity-
            # scaled when the operand has no real block scale (fp8 path).
            if cfg.WITH_X_MX_SCALE:
                if cfg.SCALE_VIA_LDS:
                    if cfg.IS_CDNA4_UPSTREAM:
                        scale_x = (
                            self.x_scale_desc.issue_local_load_unswizzle_cdna4_upstream(
                                mfma_idx,
                                self.x_scale_buffer,
                                cfg.layout_x_scale,
                                cfg.BLOCK_M_PRESHUFFLED,
                                cfg.BLOCK_M,
                                BLOCK_K_SCALE,
                            )
                        )
                    else:
                        scale_x = self.x_scale_desc.issue_local_load_unswizzle(
                            mfma_idx,
                            self.x_scale_buffer,
                            cfg.layout_x_scale,
                            cfg.BLOCK_M_PRESHUFFLED,
                            cfg.BLOCK_M,
                            BLOCK_K_SCALE,
                            cfg.PRESHUFFLE_FACTOR,
                            cfg.SCALE_KWIDTH,
                        )
                else:
                    scale_x = _load_scale_tile_via_gl_load(
                        self.x_scale_desc, mfma_idx, cfg.layout_x_scale
                    )
            else:
                scale_x = gl.full(
                    [cfg.BLOCK_M, BLOCK_K_SCALE],
                    127,
                    gl.uint8,
                    layout=cfg.layout_x_scale,
                )

            if cfg.WITH_W_MX_SCALE:
                if cfg.SCALE_VIA_LDS:
                    if cfg.IS_CDNA4_UPSTREAM:
                        scale_w = (
                            self.w_scale_desc.issue_local_load_unswizzle_cdna4_upstream(
                                mfma_idx,
                                self.w_scale_buffer,
                                cfg.layout_w_scale,
                                cfg.BLOCK_N_PRESHUFFLED,
                                cfg.BLOCK_N,
                                BLOCK_K_SCALE,
                            )
                        )
                    else:
                        scale_w = self.w_scale_desc.issue_local_load_unswizzle(
                            mfma_idx,
                            self.w_scale_buffer,
                            cfg.layout_w_scale,
                            cfg.BLOCK_N_PRESHUFFLED,
                            cfg.BLOCK_N,
                            BLOCK_K_SCALE,
                            cfg.PRESHUFFLE_FACTOR,
                            cfg.SCALE_KWIDTH,
                        )
                else:
                    scale_w = _load_scale_tile_via_gl_load(
                        self.w_scale_desc, mfma_idx, cfg.layout_w_scale
                    )
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
        # X / W / scales all go through LDS via async copy + commit_group.
        # Classical (multi-buffer / single-stage-per-iter) software pipeline:
        # prologue issues NUM_BUFFERS - 1 batches, then each main-iter
        # issues one batch, waits for the oldest, local-loads + MFMA.
        cfg = self.cfg
        load_idx = 0
        mfma_idx = 0

        for _ in gl.static_range(cfg.NUM_BUFFERS - 1):
            load_idx = self.issue_global_loads(load_idx)

        accumulator = gl.zeros(
            (cfg.BLOCK_M, cfg.BLOCK_N), dtype=gl.float32, layout=cfg.acc_layout
        )
        loop_ub = gl.cdiv(loop_k, cfg.BLOCK_K)
        gl.assume(loop_ub > 0)
        epilogue_lb = loop_ub - (cfg.NUM_BUFFERS - 1)

        for i in range(0, loop_ub):
            pred = i - epilogue_lb
            pred = (pred >> 31) & 1
            load_idx = self.issue_global_loads(load_idx, pred=pred)
            self.async_wait(cfg.NUM_BUFFERS - 1)

            x, w, scale_x, scale_w = self.issue_local_loads(mfma_idx)
            mfma_idx += 1

            accumulator = self.mfma(x, scale_x, w, scale_w, accumulator)

        return accumulator

    @gluon.jit
    def warp_pipeline(self, loop_k):
        """Warp-pipelined variant: same compute as ``pipeline`` but marks
        the loop body with ``gl.amd.warp_pipeline_stage(...)`` so the LLVM
        backend can interleave LDS-load and MFMA across consecutive
        iterations on different warps.

        Mirrors the gfx1250 warp_pipeline reference at
        ``triton-450/third_party/amd/python/examples/gluon/moe_gfx1250.py::518``,
        ported to CDNA4 ``async_copy.{commit_group,wait_group}``.

        Hard requirements (caller / autotuner must enforce):
          * ``NUM_BUFFERS >= 3``: with NB=2 the `local_load` and the
            next `async_copy` would race on the same LDS slot.
          * ``cdiv(K, BLOCK_K) >= NUM_BUFFERS``: prologue alone needs
            ``NB - 1`` valid loads.
        """
        cfg = self.cfg
        gl.static_assert(
            cfg.NUM_BUFFERS >= 3,
            "warp_pipeline requires NUM_BUFFERS >= 3 (LDS slot reuse race)",
        )
        load_idx = 0
        mfma_idx = 0

        # Prologue: issue NB - 1 batches without waiting.
        for _ in gl.static_range(cfg.NUM_BUFFERS - 1):
            load_idx = self.issue_global_loads(load_idx)

        accumulator = gl.zeros(
            (cfg.BLOCK_M, cfg.BLOCK_N), dtype=gl.float32, layout=cfg.acc_layout
        )
        loop_ub = gl.cdiv(loop_k, cfg.BLOCK_K) - (cfg.NUM_BUFFERS - 1)
        gl.assume(loop_ub >= 0)

        # Wait so the oldest of the NB - 1 prologue batches is in LDS.
        # Followed by NB - 2 still in-flight; matches the per-iter wait
        # at the bottom of the loop.
        self.async_wait(cfg.NUM_BUFFERS - 2)

        for _ in range(0, loop_ub):
            # Stage 1 ("lds+tdm"): LDS local-load of THIS iter's tile +
            # async issue of the next K-tile. Both touch LDS but along
            # different addresses, so the scheduler is free to interleave
            # them across warps.
            with gl.amd.warp_pipeline_stage("lds+tdm", priority=1):
                x, w, scale_x, scale_w = self.issue_local_loads(mfma_idx)
                mfma_idx += 1
                load_idx = self.issue_global_loads(load_idx)

            self.async_wait(cfg.NUM_BUFFERS - 2)

            # Stage 2 ("mfma"): pure VGPR/MFMA compute. Lower priority
            # so the scheduler prefers to issue the load stage first
            # when there is contention.
            with gl.amd.warp_pipeline_stage("mfma", priority=0):
                accumulator = self.mfma(x, scale_x, w, scale_w, accumulator)

        # Epilogue: drain the NB - 1 remaining unread tiles. We pay one
        # full ``async_wait(0)`` here (vs. the gfx1250 epilogue's
        # interleaved ``wait_group(NB-1-i)``) because CDNA4's commit_group
        # tracking can't safely overlap the final tile's local_load with
        # the previous tile's MFMA in the warp_pipeline cluster: the
        # MFMA + local_load instructions sit in two different stages and
        # the scheduler may reorder, exposing the just-issued in-flight
        # tile to a premature local_load. Strict drain is correct and
        # the cost is one async_wait per kernel invocation, not per-iter.
        self.async_wait(0)
        for _ in gl.static_range(cfg.NUM_BUFFERS - 1):
            x, w, scale_x, scale_w = self.issue_local_loads(mfma_idx)
            mfma_idx += 1
            accumulator = self.mfma(x, scale_x, w, scale_w, accumulator)

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
    expert_remap_ptr,
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
    HAS_EXPERT_REMAP: gl.constexpr,
    HAS_RAGGED_OFFS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    SCALE_LOAD_MODE: gl.constexpr,
    W_TRANSPOSE: gl.constexpr = False,
    NUM_SUBTILES: gl.constexpr = (1, 1, 1),
    EVEN_K: gl.constexpr = True,
    APPLY_X_GLOBAL_SCALE: gl.constexpr = True,
    USE_WARP_PIPELINE: gl.constexpr = False,
):
    """Compute one (compact_idx, block_in_expert, pid_n) output tile.

    Extracted from ``_pipelined_moe_kernel_scaled`` so the persistent
    variant can call it in a per-CTA loop while the non-persistent
    grid kernel still invokes it once. The LDS allocations inside
    ``MoEPipelinedProgram.initialize`` are hoisted to the enclosing
    kernel's prologue by Triton, so calling this helper many times
    from a runtime-bounded ``for`` loop reuses the same buffers and
    keeps occupancy unchanged.
    """
    if HAS_EXPERT_REMAP:
        expert_id = gl.load(expert_remap_ptr + compact_idx).to(gl.int32)
    else:
        expert_id = compact_idx

    # HAS_*/USE_GATHER must come from the launcher; an
    # ``is not None`` test on tensor ptrs always returns True under JIT.
    USE_GATHER: gl.constexpr = HAS_GATHER

    BLOCK_SCALE_FACTOR: gl.constexpr = 32
    BLOCK_K_SCALE: gl.constexpr = BLOCK_K // BLOCK_SCALE_FACTOR

    if HAS_RAGGED_OFFS:
        # Honour the actual ragged packing of x: experts live back-to-back
        # in HBM at ``slice_offs[expert_id]``; per-expert block boundary
        # is ``slice_sizes[expert_id]`` (NOT padded up to BLOCK_M).
        # This is required when per-expert size < BLOCK_M (otherwise the
        # naive ``compact_idx * BLOCKS_PER_EXPERT * BLOCK_M`` reads rows
        # belonging to the next expert and corrupts the GEMM).
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

    STORE: gl.constexpr = _store_layout(NUM_WARPS)

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
    )

    # e2m1 packs 2 elements / byte along K.
    BLOCK_K_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
    BLOCK_K_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W

    X_ELEM_BITS: gl.constexpr = x_ptr.dtype.element_ty.primitive_bitwidth
    W_ELEM_BITS: gl.constexpr = w_ptr.dtype.element_ty.primitive_bitwidth
    LOAD_X_LAYOUT: gl.constexpr = _load_layout(
        BLOCK_K_X, BLOCK_M, NUM_WARPS, [1, 0], X_ELEM_BITS
    )
    if W_TRANSPOSE:
        LOAD_W_LAYOUT: gl.constexpr = _load_layout(
            BLOCK_K_W, BLOCK_N, NUM_WARPS, [1, 0], W_ELEM_BITS
        )
    else:
        # HBM W is [K_packed, N] with N contiguous. Vectorise along the
        # contig axis by passing BLOCK_N as the "k" arg (= contig-axis size).
        LOAD_W_LAYOUT: gl.constexpr = _load_layout(
            BLOCK_N, BLOCK_K_W, NUM_WARPS, [1, 0], W_ELEM_BITS
        )

    offs_xm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, LOAD_X_LAYOUT))
    offs_xk = gl.arange(0, BLOCK_K_X, layout=gl.SliceLayout(0, LOAD_X_LAYOUT))
    if W_TRANSPOSE:
        offs_wn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, LOAD_W_LAYOUT))
        offs_wk = gl.arange(0, BLOCK_K_W, layout=gl.SliceLayout(0, LOAD_W_LAYOUT))
    else:
        offs_wn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, LOAD_W_LAYOUT))
        offs_wk = gl.arange(0, BLOCK_K_W, layout=gl.SliceLayout(1, LOAD_W_LAYOUT))

    rows_m = off_m + offs_xm
    # ``m_limit`` is the per-expert tail when HAS_RAGGED_OFFS=True
    # (= slice_offs[expert_id] + slice_sizes[expert_id]); otherwise it
    # is the global ``M``. The pre-gather position lives in the
    # ragged-dispatched order so we use ``m_limit`` to bound it.
    pre_gather_mask = rows_m < m_limit
    if HAS_GATHER:
        rows_m_safe = gl.where(pre_gather_mask, rows_m, gl.zeros_like(rows_m))
        rows_m = gl.load(
            gather_idx_ptr + rows_m_safe, mask=pre_gather_mask, other=0
        ).to(gl.int32)
        # Post-gather ``rows_m`` lives in the *global* token-id space
        # of x_orig (size = ``M_X`` = x.shape[-2] = n_tokens), so the
        # x-load mask must use the pre-gather mask combined with the
        # global ``rows_m < M_X`` bound (the latter guards a junk
        # gather_idx value). NOTE: ``M`` is the dispatched / output
        # tile count (= gather_indx.numel() in production) which can be
        # > M_X for top-k>1 dispatches; do NOT mix the two.
        mask_m = pre_gather_mask & (rows_m < M_X)
    else:
        mask_m = pre_gather_mask
    mask_n = (off_n + offs_wn) < N

    # Hint divisibility on K_phys: the launcher requires K % 32 == 0, so
    # K_phys = K / DIV_FACTOR has div(K_phys) >= 32 // DIV_FACTOR (>= 16 for
    # mxfp4 packed). Without this hint the K-mask alignment for non-even-K
    # caps the direct-to-LDS vec at 64b on mxfp4, which CDNA4 rejects.
    # ``multiple_of`` on a func arg is a silent no-op; apply it on the
    # ``K // DIV_FACTOR`` SSA which has an arith.divsi defining op.
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
    if W_TRANSPOSE:
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
        )
    # Scale offsets: SCALE_VIA_LDS (swizzle) uses the post-swizzle HBM shape
    # [..., NONK/PF, K_S*PF] with BLOCK_NONK_PRESHUFFLED rows and
    # BLOCK_K_SCALE_PRESHUFFLED cols, and issues buffer_load_to_shared.
    # Otherwise scales load G->VGPR directly via gl.load using the
    # mfma_scale_layout (uniform across bypass/transpose).
    if HAS_X_BLOCK_SCALE:
        if cfg.SCALE_VIA_LDS:
            BLOCK_M_PS: gl.constexpr = cfg.BLOCK_M_PRESHUFFLED
            BLOCK_K_S_PS: gl.constexpr = cfg.BLOCK_K_SCALE_PRESHUFFLED
            LX_S: gl.constexpr = cfg.load_layout_x_scale
            offs_xs_m = gl.arange(0, BLOCK_M_PS, layout=gl.SliceLayout(1, LX_S))
            offs_xs_k = gl.arange(0, BLOCK_K_S_PS, layout=gl.SliceLayout(0, LX_S))
            row_base_x_s = off_m // cfg.PRESHUFFLE_FACTOR
            rows_m_scale = row_base_x_s + offs_xs_m
            # x_scale's row bound follows x's physical row count (M_X),
            # not the dispatched tile count M. With HAS_GATHER, off_m
            # is in dispatched space but the x_scale read is gated by
            # mask_m (post-gather) below; this row_limit only exists to
            # guard the SCALE_VIA_LDS bulk load.
            row_limit_x_s = (M_X + cfg.PRESHUFFLE_FACTOR - 1) // cfg.PRESHUFFLE_FACTOR
            # cdna4_upstream: the in-tile K-axis is packed with the N-axis,
            # so a K-mask on the packed column would scramble both. We rely
            # on the host-side padding (e8m0=0 for OOB microblocks)
            # combined with the W operand's own K-mask which zeros the OOB
            # W elements -- making the OOB scale-x weight contribution 0
            # regardless of the loaded scale value. Suppress the kernel-
            # side K-mask by setting k_limit = K_S_pad * PF (full post-pad
            # storage extent), which is guaranteed >= every loaded offset.
            # K is a runtime arg so we cannot use constexpr ternaries here.
            k_limit_xs_load_nopad = (K // cfg.SCALE_BLOCK) * cfg.PRESHUFFLE_FACTOR
            k_limit_xs_load_pad = (
                (K // cfg.SCALE_BLOCK + 7) // 8 * 8
            ) * cfg.PRESHUFFLE_FACTOR
            if cfg.IS_CDNA4_UPSTREAM:
                k_limit_xs_load = k_limit_xs_load_pad
            else:
                k_limit_xs_load = k_limit_xs_load_nopad
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
            # See x_scale comment above for the rationale.
            k_limit_ws_load_nopad = (K // cfg.SCALE_BLOCK) * cfg.PRESHUFFLE_FACTOR
            k_limit_ws_load_pad = (
                (K // cfg.SCALE_BLOCK + 7) // 8 * 8
            ) * cfg.PRESHUFFLE_FACTOR
            if cfg.IS_CDNA4_UPSTREAM:
                k_limit_ws_load = k_limit_ws_load_pad
            else:
                k_limit_ws_load = k_limit_ws_load_nopad
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

    pgm = MoEPipelinedProgram.initialize(
        cfg, x_desc, w_desc, x_scale_desc, w_scale_desc
    )

    if USE_WARP_PIPELINE:
        acc = pgm.warp_pipeline(K)
    else:
        acc = pgm.pipeline(K)

    if APPLY_X_GLOBAL_SCALE and not HAS_X_BLOCK_SCALE:
        # Read the per-tensor flex scale from device memory (1-element
        # f32 tensor). Passing this as a pointer rather than a Python
        # scalar lets the kernel be CUDA/HIP-graph capturable -- callers
        # no longer have to .item() the scale on the host stream, which
        # would break graph capture (see _try_dispatch_mxfp).
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
    else:
        out = acc

    out = out.to(y_ptr.dtype.element_ty)
    out = gl.convert_layout(out, STORE)

    offs_y_m = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, STORE))
    off_n_out = pid_n * OUT_BLOCK_N
    offs_y_n = off_n_out + gl.arange(0, OUT_BLOCK_N, gl.SliceLayout(0, STORE))

    if APPLY_GATE_SCAL:
        scal = gl.load(
            gate_scal_ptr + offs_y_m,
            mask=offs_y_m < m_limit,
            other=1.0,
        )
        out = out * scal[:, None].to(out.dtype)

    actual_n = (N // 2) if DO_SWIGLU else N
    if HAS_SCATTER:
        rows_y = gl.load(scatter_idx_ptr + offs_y_m, mask=offs_y_m < m_limit, other=M)
        mask_y = (rows_y[:, None] < M) & (offs_y_n[None, :] < actual_n)
        y_offs = rows_y[:, None] * stride_ym + offs_y_n[None, :] * stride_yn
    else:
        mask_y = (offs_y_m[:, None] < m_limit) & (offs_y_n[None, :] < actual_n)
        y_offs = offs_y_m[:, None] * stride_ym + offs_y_n[None, :] * stride_yn

    gl.store(y_ptr + y_offs, out, mask=mask_y)


@gluon.jit
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
    expert_remap_ptr,
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
    HAS_EXPERT_REMAP: gl.constexpr,
    HAS_RAGGED_OFFS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    SCALE_LOAD_MODE: gl.constexpr,
    W_TRANSPOSE: gl.constexpr = False,
    NUM_SUBTILES: gl.constexpr = (1, 1, 1),
    EVEN_K: gl.constexpr = True,
    APPLY_X_GLOBAL_SCALE: gl.constexpr = True,
    USE_WARP_PIPELINE: gl.constexpr = False,
    PERSISTENT: gl.constexpr = False,
    USE_BLOCK_SCHEDULE: gl.constexpr = False,
    N_EXPTS_TOT: gl.constexpr = 0,
):
    """Top-level MoE GEMM kernel.

    Three scheduling modes, picked by the launcher to match the host
    invocation context (CUDA-graph capture vs eager) and the workload.

    1. ``USE_BLOCK_SCHEDULE=False, PERSISTENT=False`` (legacy):
       2-D grid ``(blocks_per_expert * grid_n, num_active)``; each CTA
       computes one tile ``(compact_idx, block_in_expert, pid_n)``.
       Requires host-side ``_expert_layout`` which does a ``.tolist()``
       on ``slice_sizes`` -- NOT graph-capturable.

    2. ``USE_BLOCK_SCHEDULE=False, PERSISTENT=True`` (legacy persistent):
       1-D grid sized to occupy the GPU; each CTA strides through tiles
       ``0..NUM_TILES-1`` where ``NUM_TILES = num_active *
       blocks_per_expert * grid_n``. Same host-side blocker as (1).

    3. ``USE_BLOCK_SCHEDULE=True`` (graph-capturable, this CL):
       Borrows triton_kernels' device-side ragged schedule. Host picks
       ``grid_m`` = ``RaggedTensorMetadata.n_blocks(E, M, BLOCK_M)`` (a
       pure-integer upper bound; no D2H sync), and ``block_schedule_ptr``
       points at the device-side schedule table where entry ``pid_m``
       packs ``(block_in_expert << 16) | expert_id``. ``block_offs_ptr +
       N_EXPTS_TOT`` holds the exact number of valid blocks so each CTA
       early-returns on padded entries -- the cost of a padded tile is
       a single ``gl.load`` plus a branch, ~zero compared to the prologue
       of a real tile.

       Combined with ``a_global_scale_ptr`` (the per-tensor flex scale
       read via ``gl.load`` rather than ``.item()``-ed on the host),
       this is the path that lets Gluon participate in the production
       gpt-oss-120b decode CUDA / HIP graph instead of being silently
       fall-back'd to upstream triton_kernels.
    """
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    tiles_per_expert = BLOCKS_PER_EXPERT * grid_n
    if USE_BLOCK_SCHEDULE:
        # Schedule-driven path: grid is 1-D, tiles are dispatched via the
        # pre-built schedule table. Schedule entry ``pid_m`` is a packed
        # ``(block_in_expert << 16) | expert_id`` int32; padding entries
        # past ``actual_unpadded_m`` are ``0xFFFFFFFF`` and we cheap-skip
        # them by checking ``pid_m < actual_unpadded_m`` before any
        # ``_pipelined_moe_tile_compute`` work.
        actual_unpadded_m = gl.load(block_offs_ptr + N_EXPTS_TOT).to(gl.int32)
        actual_total = actual_unpadded_m * grid_n
        if PERSISTENT:
            pid = gl.program_id(0)
            num_pids = gl.num_programs(0)
            for tile_idx in range(pid, NUM_TILES, num_pids):
                if tile_idx < actual_total:
                    pid_m = tile_idx // grid_n
                    pid_n = tile_idx % grid_n
                    schedule_raw = gl.load(block_schedule_ptr + pid_m).to(
                        gl.uint32, bitcast=True
                    )
                    expert_id = (schedule_raw & 0x0000FFFF).to(gl.int32)
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
                        expert_remap_ptr,
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
                        expert_id,
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
                        HAS_EXPERT_REMAP=False,
                        HAS_RAGGED_OFFS=HAS_RAGGED_OFFS,
                        NUM_WARPS=NUM_WARPS,
                        NUM_BUFFERS=NUM_BUFFERS,
                        SCALE_LOAD_MODE=SCALE_LOAD_MODE,
                        W_TRANSPOSE=W_TRANSPOSE,
                        NUM_SUBTILES=NUM_SUBTILES,
                        EVEN_K=EVEN_K,
                        APPLY_X_GLOBAL_SCALE=APPLY_X_GLOBAL_SCALE,
                        USE_WARP_PIPELINE=USE_WARP_PIPELINE,
                    )
        else:
            pid = gl.program_id(0)
            if pid < actual_total:
                pid_m = pid // grid_n
                pid_n = pid % grid_n
                schedule_raw = gl.load(block_schedule_ptr + pid_m).to(
                    gl.uint32, bitcast=True
                )
                expert_id = (schedule_raw & 0x0000FFFF).to(gl.int32)
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
                    expert_remap_ptr,
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
                    expert_id,
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
                    HAS_EXPERT_REMAP=False,
                    HAS_RAGGED_OFFS=HAS_RAGGED_OFFS,
                    NUM_WARPS=NUM_WARPS,
                    NUM_BUFFERS=NUM_BUFFERS,
                    SCALE_LOAD_MODE=SCALE_LOAD_MODE,
                    W_TRANSPOSE=W_TRANSPOSE,
                    NUM_SUBTILES=NUM_SUBTILES,
                    EVEN_K=EVEN_K,
                    APPLY_X_GLOBAL_SCALE=APPLY_X_GLOBAL_SCALE,
                    USE_WARP_PIPELINE=USE_WARP_PIPELINE,
                )
    elif PERSISTENT:
        pid = gl.program_id(0)
        num_pids = gl.num_programs(0)
        for tile_idx in range(pid, NUM_TILES, num_pids):
            compact_idx = tile_idx // tiles_per_expert
            local = tile_idx % tiles_per_expert
            block_in_expert = local // grid_n
            pid_n = local % grid_n
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
                expert_remap_ptr,
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
                HAS_EXPERT_REMAP=HAS_EXPERT_REMAP,
                HAS_RAGGED_OFFS=HAS_RAGGED_OFFS,
                NUM_WARPS=NUM_WARPS,
                NUM_BUFFERS=NUM_BUFFERS,
                SCALE_LOAD_MODE=SCALE_LOAD_MODE,
                W_TRANSPOSE=W_TRANSPOSE,
                NUM_SUBTILES=NUM_SUBTILES,
                EVEN_K=EVEN_K,
                APPLY_X_GLOBAL_SCALE=APPLY_X_GLOBAL_SCALE,
                USE_WARP_PIPELINE=USE_WARP_PIPELINE,
            )
    else:
        compact_idx = gl.program_id(1)
        block_pid = gl.program_id(0)
        block_in_expert = block_pid // grid_n
        pid_n = block_pid % grid_n
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
            expert_remap_ptr,
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
            HAS_EXPERT_REMAP=HAS_EXPERT_REMAP,
            HAS_RAGGED_OFFS=HAS_RAGGED_OFFS,
            NUM_WARPS=NUM_WARPS,
            NUM_BUFFERS=NUM_BUFFERS,
            SCALE_LOAD_MODE=SCALE_LOAD_MODE,
            W_TRANSPOSE=W_TRANSPOSE,
            NUM_SUBTILES=NUM_SUBTILES,
            EVEN_K=EVEN_K,
            APPLY_X_GLOBAL_SCALE=APPLY_X_GLOBAL_SCALE,
            USE_WARP_PIPELINE=USE_WARP_PIPELINE,
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
    """Return a structured GPR / scratch / occupancy profile for ``kernel``.

    Mirrors the helper from
    ``triton-450/third_party/amd/python/examples/gluon/gfx1250_utils.py``,
    but tolerant of the slightly different MI355 AMDGCN dump format.
    """
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


def assert_no_spills(profile: dict, *, allow_scratch: int = 0) -> None:
    """Raise if the static profile shows any GPR spill or excess scratch."""
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


# ---------------------------------------------------------------------------
# Helpers shared by all three Python-side launchers
# ---------------------------------------------------------------------------


def _expert_layout(
    a_ragged_metadata: Any | None,
    block_m: int,
    M: int,
) -> tuple[int, int, torch.Tensor | None]:

    if a_ragged_metadata is None:
        return 1, (M + block_m - 1) // block_m, None
    counts = a_ragged_metadata.slice_sizes
    counts_list = counts.tolist()
    active = [i for i, c in enumerate(counts_list) if int(c) > 0]
    num_active = len(active)
    if num_active == 0:
        return 1, 0, None
    max_blocks = max((int(counts_list[i]) + block_m - 1) // block_m for i in active)
    if num_active == counts.numel():
        # All experts active: identity remap, no need to materialise.
        return num_active, max_blocks, None
    expert_remap = torch.tensor(active, device=counts.device, dtype=torch.int32)
    return num_active, max_blocks, expert_remap


def _make_dummy(device, dtype=torch.int32, n: int = 0) -> torch.Tensor:
    return torch.empty(max(n, 0), device=device, dtype=dtype)


def _swizzle_scales_cdna4(s: torch.Tensor) -> torch.Tensor:
    # AITer CDNA4 preshuffle: [..., NONK, K_S] -> [..., NONK/PF, K_S*PF].
    # 5-D split: NONK = (NONK/PF) * 4 * (PF/4); K_S = (K_S/KWIDTH) * KWIDTH;
    # permute (..., 0, 3, 2, 1, 4) (last-5 axes) interleaves so the kernel-side
    # 5-D unswizzle view restores the natural [NONK, K_S] layout in LDS.
    PF = _SCALE_PRESHUFFLE_FACTOR
    KW = _SCALE_KWIDTH
    nonk = s.shape[-2]
    k_s = s.shape[-1]
    assert nonk % PF == 0, f"swizzle: NONK={nonk} not divisible by PF={PF}"
    assert k_s % KW == 0, f"swizzle: K_S={k_s} not divisible by KWIDTH={KW}"
    batch = s.shape[:-2]
    v = s.reshape(*batch, nonk // PF, 4, PF // 4, k_s // KW, KW)
    rank = v.ndim
    last5 = (rank - 5, rank - 2, rank - 3, rank - 4, rank - 1)
    perm = (*range(rank - 5), *last5)
    return v.permute(*perm).contiguous().reshape(*batch, nonk // PF, k_s * PF)


def _swizzle_scales_cdna4_upstream(s: torch.Tensor) -> torch.Tensor:
    """Pure-python mirror of triton_kernels' ``CDNA4MXScaleLayout.swizzle_data``
    for uint8 (e8m0) MX block scales.

    Input convention: ``s`` has shape ``(*leading, NONK=N, K_S)`` -- the
    same layout gluon's launcher passes to ``_preprocess_scale``. This
    helper internally transposes to upstream's ``(*leading, K_SCALE, N)``
    convention before applying the swizzle.

    Output: ``(*leading, K_SCALE_pad*32, N_pad/32)`` with ``stride(-2)==1``
    (i.e. the K_SCALE_pad*32 axis is contiguous in memory). This matches
    upstream's wrapped scale tensor exactly so a tensor produced by
    ``triton_kernels.tensor_details.layout.CDNA4MXScaleLayout`` can be
    consumed zero-copy by the ``cdna4_upstream`` mode.
    """
    assert s.dtype == torch.uint8, (
        f"_swizzle_scales_cdna4_upstream: expected uint8 e8m0 scales, " f"got {s.dtype}"
    )
    # gluon convention -> upstream convention.
    s = s.transpose(-2, -1).contiguous()
    *leading_shape, K_SCALE, N = s.shape
    B = 1
    for d in leading_shape:
        B *= d
    ALIGN_K_S = _ALIGN_K_SCALE_CDNA4_UPSTREAM
    ALIGN_N = _ALIGN_N_CDNA4_UPSTREAM
    K_SCALE_pad = ((K_SCALE + ALIGN_K_S - 1) // ALIGN_K_S) * ALIGN_K_S
    N_pad = ((N + ALIGN_N - 1) // ALIGN_N) * ALIGN_N
    # repack is identity for uint8 (only re-orders e2m1 nibbles).
    s = s.mT.contiguous().mT
    s = torch.nn.functional.pad(s, (0, N_pad - N, 0, K_SCALE_pad - K_SCALE))
    s = s.transpose(-1, -2)  # (..., N_pad, K_SCALE_pad)
    # `view` requires the shape to be reachable without a copy from
    # current strides. The chain above ensures (..., N_pad, K_SCALE_pad)
    # is contiguous on the inner two axes; collapse leading dims via
    # `reshape` (which may copy if non-contig leading slabs).
    s = s.reshape(B, N_pad, K_SCALE_pad)
    s = s.view(B, N_pad // 32, 2, 16, K_SCALE_pad // 8, 2, 4, 1)
    s = s.permute(0, 1, 4, 6, 3, 5, 2, 7).contiguous()
    s = s.reshape(B, N_pad // 32, K_SCALE_pad * 32)
    s = s.transpose(-1, -2)  # (B, K_SCALE_pad*32, N_pad/32)
    # Match upstream's output shape (collapsed leading dim into B).
    # Restoring multi-leading would force a non-trivial reshape that can
    # invalidate `stride(-2) == 1`, which the kernel relies on.
    return s


def _is_cdna4_upstream_swizzled(s: torch.Tensor) -> bool:
    """Heuristic: a tensor is already cdna4_upstream-swizzled iff its
    inner 2-D has ``stride(-2) == 1`` (the contiguous K_S*32 axis) and
    ``shape[-2] % 32 == 0`` and ``shape[-1] % 1 == 0`` etc. We take the
    cheap path and just check ``stride(-2) == 1`` since the upstream
    swizzle asserts that explicitly."""
    return s.stride(-2) == 1 and s.stride(-1) >= s.shape[-2]


def _preprocess_scale(data: torch.Tensor | None, mode: str) -> torch.Tensor | None:
    # "bypass" / "transpose": no-op (kernel uses gl.load directly).
    # "swizzle": AITer 5-D preshuffle so contig K dim post-swizzle is large
    # enough for buffer_load_to_shared canCoalesce to succeed (see Update 9).
    # "cdna4_upstream": apply (or pass-through) upstream's CDNA4MXScaleLayout
    # so the kernel can buffer_load_to_shared and unswizzle in LDS using
    # the upstream 7-D pattern. Already-swizzled tensors are detected by
    # `stride(-2) == 1` and forwarded zero-copy.
    if data is None:
        return None
    if mode not in _SCALE_LOAD_MODES:
        raise ValueError(
            f"_preprocess_scale: SCALE_LOAD_MODE must be one of "
            f"{_SCALE_LOAD_MODES}, got {mode!r}"
        )
    if mode == "swizzle":
        return _swizzle_scales_cdna4(data)
    if mode == "cdna4_upstream":
        if _is_cdna4_upstream_swizzled(data):
            return data
        return _swizzle_scales_cdna4_upstream(data)
    return data


def _supports_pure_bf16(precision_config, fused_activation) -> bool:
    """Return True iff this call can take the pure-bf16 fast path."""
    if precision_config is None:
        return True
    if getattr(precision_config, "b_mx_scale", None) is not None:
        return False
    flex = getattr(precision_config, "flex_ctx", None)
    lhs = getattr(flex, "lhs_data", None) if flex is not None else None
    if lhs is not None and getattr(lhs, "dtype", None) is not None:
        return False
    return True


# ---------------------------------------------------------------------------
# Public launcher: software-pipelined ragged matmul (unified driver)
# ---------------------------------------------------------------------------


def _scale_strides(scale: torch.Tensor | None, mode: str = "bypass") -> tuple[int, int]:
    """Return ``(stride_nonk, stride_k)`` matching the kernel's address math
    ``base + r * stride_nonk + c * stride_k`` where ``r`` ranges over the
    NONK axis (M for x_scale, N for w_scale, possibly preshuffled) and
    ``c`` ranges over the K_S axis (possibly *PF post-swizzle).

    For ``bypass`` / ``transpose`` / ``swizzle`` the gluon convention is
    ``[..., NONK, K_S]`` with K_S contiguous, so we return the natural
    ``(stride(-2), stride(-1))``.

    For ``cdna4_upstream`` the post-swizzle storage shape is
    ``[..., K_SCALE_pad*32, N_pad/32]`` with ``stride(-2)==1`` (the
    K_SCALE_pad*32 axis contiguous). The kernel still treats it as
    ``[..., NONK_PS=N/32, K_S_PS=K_S*32]`` so we swap and return
    ``(stride(-1), stride(-2))``.
    """
    if scale is None:
        return 0, 0
    if mode == "cdna4_upstream":
        return scale.stride(-1), scale.stride(-2)
    return scale.stride(-2), scale.stride(-1)


_PLAIN_DTYPE_STR = {torch.bfloat16: "bfloat16", torch.float16: "float16"}
_SCALED_FORMATS = {"e2m1", "e4m3", "e5m2"}
_PLAIN_FORMATS = set(_PLAIN_DTYPE_STR.values())


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
    a_format: str | None = None,
    b_format: str | None = None,
    x_scale: torch.Tensor | None = None,
    w_scale: torch.Tensor | None = None,
    a_global_scale: torch.Tensor | float | None = 1.0,
    scale_load_mode: str = "bypass",
    w_transpose: bool = False,
    apply_x_global_scale: bool | None = None,
    scaled_mfma: bool | None = None,
    use_warp_pipeline: bool = False,
    persistent: bool | None = None,
    num_ctas: int | None = None,
):
    # Unified launcher for both scaled (mxfp4/fp8) and plain (bf16/fp16) paths.
    # ``a_format``/``b_format`` default to dtype-derived strings for plain
    # tensors; callers using packed-uint8 scaled formats must pass them
    # explicitly. ``USE_MFMA_SCALED`` in MoEConfig drives the kernel-body
    # selection (scaled-mfma 16x16x128 vs regular 16x16x32).
    if a_format is None:
        a_format = _PLAIN_DTYPE_STR[x.dtype]
    if b_format is None:
        b_format = _PLAIN_DTYPE_STR[w.dtype]
    assert (
        a_format in _SCALED_FORMATS | _PLAIN_FORMATS
    ), f"unknown a_format={a_format!r}"
    assert (
        b_format in _SCALED_FORMATS | _PLAIN_FORMATS
    ), f"unknown b_format={b_format!r}"
    is_scaled = a_format in _SCALED_FORMATS or b_format in _SCALED_FORMATS
    # ``scaled_mfma`` override exists only for the assertion-only test; the
    # kernel body itself dispatches on the actual format constexpr.
    enforce_scaled_k = scaled_mfma if scaled_mfma is not None else is_scaled
    if apply_x_global_scale is None:
        apply_x_global_scale = is_scaled
    assert scale_load_mode in _SCALE_LOAD_MODES, (
        f"scale_load_mode must be one of {_SCALE_LOAD_MODES}, "
        f"got {scale_load_mode!r}"
    )
    has_a_block_scale = a_format == "e2m1"
    has_w_block_scale = b_format == "e2m1"
    if has_a_block_scale:
        assert x_scale is not None, "mxfp4 A requires a block-scale tensor"
    if has_w_block_scale:
        assert w_scale is not None, "mxfp4 W requires a block-scale tensor"

    # ``M`` is the dispatched / output tile count consumed by the kernel
    # for ragged & writeback bookkeeping. With a non-None ``gather_indx``
    # production passes ``x`` *un-permuted* (shape ``(n_tokens, H)``) and
    # the kernel walks ``gather_indx[i]`` for ``i in [0, gather_indx.numel())``
    # -- so M must reflect the dispatched count, not ``n_tokens``.
    # ``M_X`` stays bound to ``x.shape[-2]`` so the post-gather safety
    # check (``rows_m < M_X``) keeps catching out-of-range gather idx.
    M_X = x.shape[-2]
    if gather_indx is not None:
        gather_buf_for_m = gather_indx.src_indx
        M = int(gather_buf_for_m.shape[0])
    else:
        M = M_X
    K_phys = x.shape[-1]
    div_a = 2 if a_format == "e2m1" else 1
    div_b = 2 if b_format == "e2m1" else 1
    K = K_phys * div_a

    scale_load_mode = _effective_scale_load_mode(
        scale_load_mode,
        block_m,
        block_n,
        block_k,
        scale_block=32,
        has_x_scale=has_a_block_scale,
        has_w_scale=has_w_block_scale,
        k=K,
        a_format=a_format,
        num_buffers=num_buffers,
    )

    if w.ndim == 3:
        E, K_w_phys, N = w.shape
    else:
        K_w_phys, N = w.shape
        E = 1
    K_w = K_w_phys * div_b
    assert K == K_w, f"K mismatch: A logical K={K} vs W logical K={K_w}"

    mfma_k = _MFMA_SCALED_K if enforce_scaled_k else _MFMA_K
    assert block_k % mfma_k == 0, (
        f"BLOCK_K={block_k} must be a multiple of MFMA K dim ({mfma_k}); "
        f"scaled_mfma={enforce_scaled_k}"
    )
    if enforce_scaled_k:
        assert (
            block_k >= _MFMA_SCALED_K
        ), f"scaled MFMA requires BLOCK_K >= {_MFMA_SCALED_K} (got {block_k})"
    assert block_m % _MFMA_M == 0

    grid_n = (N + block_n - 1) // block_n

    # Per-expert ragged offsets (slice_offs) + sizes (slice_sizes).
    # Required when per-expert size < BLOCK_M -- otherwise the kernel's
    # naive ``compact_idx * BLOCKS_PER_EXPERT * BLOCK_M`` row offset
    # would step PAST the per-expert tail and load rows belonging to
    # the NEXT expert (corrupting the GEMM, see check_gluon_mxfp_fp8
    # n_active=8 per_expert=32 case).
    # DEBUG knob: ``TOKENSPEED_MOE_GLUON_RAGGED_OFFS=0`` disables the
    # per-expert slice_offs / slice_sizes mask path even when ragged
    # metadata is provided. Used to bisect whether a regression comes
    # from the new ragged-offs code or from elsewhere.
    _ragged_offs_disabled = os.environ.get(
        "TOKENSPEED_MOE_GLUON_RAGGED_OFFS", ""
    ).strip().lower() in {"0", "false", "no", "off"}
    has_ragged_offs = a_ragged_metadata is not None and not _ragged_offs_disabled
    if has_ragged_offs:
        slice_offs_buf = a_ragged_metadata.slice_offs.to(torch.int32)
        slice_sizes_buf = a_ragged_metadata.slice_sizes.to(torch.int32)
    else:
        slice_offs_buf = _make_dummy(x.device, torch.int32)
        slice_sizes_buf = _make_dummy(x.device, torch.int32)

    # Decide whether to take the schedule-driven (graph-capturable) path.
    # Schedule mode borrows triton_kernels' device-side per-block-size
    # ragged schedule: host picks ``grid_m`` from ``n_blocks(E, M, BLOCK_M)``
    # (a pure-integer upper bound; no D2H sync) and the kernel decodes
    # ``(expert_id, block_in_expert)`` from ``block_schedule[pid_m]``,
    # cheap-skipping padded entries. This is what unlocks CUDA / HIP
    # graph capture for the Gluon MoE kernel; the legacy ``_expert_layout``
    # branch is kept for shapes the schedule pre-computation doesn't
    # cover (BLOCK_M outside ``RaggedTensorMetadata.block_sizes()`` or
    # callers that pass a hand-built metadata without the schedule
    # tables populated) and for the dense / gating-GEMM paths where
    # ``a_ragged_metadata is None``.
    _schedule_disabled = os.environ.get(
        "TOKENSPEED_MOE_GLUON_SCHEDULE", ""
    ).strip().lower() in {"0", "false", "no", "off"}
    _supported_schedule_block_ms = set(RaggedTensorMetadata.block_sizes())
    use_block_schedule = (
        has_ragged_offs
        and block_m in _supported_schedule_block_ms
        and not _schedule_disabled
        and getattr(a_ragged_metadata, "block_offs_data", None) is not None
        and getattr(a_ragged_metadata, "block_schedule_data", None) is not None
    )

    if use_block_schedule:
        n_slices = int(a_ragged_metadata.slice_sizes.shape[0])
        grid_m_upper = RaggedTensorMetadata.n_blocks(n_slices, M, block_m)
        num_tiles_total = grid_m_upper * grid_n
        block_offs_buf = a_ragged_metadata.block_offs(block_m).to(torch.int32)
        block_schedule_buf = a_ragged_metadata.block_schedule(block_m).to(torch.int32)
        # BLOCKS_PER_EXPERT is unused in schedule mode (the off_m math
        # comes from slice_offs + schedule's block_in_expert), but the
        # constexpr still needs a sentinel value.
        blocks_per_expert = 1
        num_active = n_slices
        expert_remap = None
    else:
        num_active, blocks_per_expert, expert_remap = _expert_layout(
            a_ragged_metadata, block_m, M
        )
        num_tiles_total = num_active * blocks_per_expert * grid_n
        block_offs_buf = _make_dummy(x.device, torch.int32)
        block_schedule_buf = _make_dummy(x.device, torch.int32)
        n_slices = 0  # unused constexpr below when USE_BLOCK_SCHEDULE=False

    # Persistent grid: launch one CTA per (num_cus * max_occ) and stride
    # through tiles in-kernel. Falls back to the historical 2-D grid when
    # ``persistent`` is False / None and the total tile count is large
    # enough to fully populate the GPU.
    if persistent is None:
        persistent = _should_use_persistent(num_tiles_total)
    if persistent:
        if num_ctas is None:
            num_ctas = _persistent_grid_size(num_tiles_total)
        else:
            num_ctas = max(1, min(num_ctas, num_tiles_total))
        grid = (num_ctas, 1)
    elif use_block_schedule:
        # Schedule mode keeps a 1-D grid; padded ``pid_m >= unpadded_m``
        # CTAs cheap-skip via the kernel's entry early-return.
        grid = (max(1, num_tiles_total), 1)
    else:
        grid = (blocks_per_expert * grid_n, num_active)

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
    expert_remap_buf = (
        expert_remap if expert_remap is not None else _make_dummy(x.device, torch.int32)
    )

    swiglu_alpha = swiglu[0] if swiglu is not None else 0.0
    swiglu_limit = swiglu[1] if swiglu is not None else 0.0

    w3 = w if w.ndim == 3 else w.unsqueeze(0)

    if w_transpose:
        # W -> [E, N, K_packed]: K contig in HBM; kernel stages [BN, BK]
        # in LDS and permute([1,0])s the LDS view for the dot operand.
        w3 = w3.transpose(-1, -2).contiguous()
        stride_wn, stride_wk = w3.stride(-2), w3.stride(-1)
    else:
        # W stays [E, K_packed, N]: kernel stages [BK, BN] in LDS.
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
        _preprocess_scale(x_scale, scale_load_mode) if has_a_block_scale else None
    )
    stride_xsm, stride_xsk = _scale_strides(x_scale_proc, scale_load_mode)

    x_scale_buf = (
        x_scale_proc if x_scale_proc is not None else _make_dummy(x.device, torch.uint8)
    )

    NUM_SUBTILES = (1, 1, 1)
    EVEN_K = K % block_k == 0

    # Materialise ``a_global_scale`` as a device pointer the kernel can
    # ``gl.load`` at run time. This keeps the launcher CUDA/HIP-graph
    # capturable -- the previous ``float(scale.item())`` path forced a
    # device->host sync that aborted graph capture and is why
    # ``_try_dispatch_mxfp`` had to bail out under capture.
    #
    # The kernel only reads this pointer when
    # ``APPLY_X_GLOBAL_SCALE and not HAS_X_BLOCK_SCALE`` (i.e. the fp8 x
    # mxfp4 path). Other paths (bf16 x bf16, mxfp4 x mxfp4) skip the
    # load entirely, so we hand them a size-0 dummy and avoid the
    # per-launch host->device transfer that ``torch.tensor`` would do.
    needs_scale_load = apply_x_global_scale and not has_a_block_scale
    if not needs_scale_load:
        x_global_scale_buf = _make_dummy(x.device, torch.float32)
    elif isinstance(a_global_scale, torch.Tensor):
        # Production fp8 path: the upstream FlexCtx hands us a 1-element
        # f32 tensor already on ``x.device``, so the view + dtype guard
        # is a zero-copy passthrough (``Tensor.to`` returns self when
        # device and dtype already match).
        scale_view = a_global_scale.detach().reshape(-1)[:1]
        if scale_view.device == x.device and scale_view.dtype == torch.float32:
            x_global_scale_buf = scale_view
        else:
            x_global_scale_buf = scale_view.to(device=x.device, dtype=torch.float32)
    else:
        # Tests / microbenchmarks pass a Python float here. This path
        # materialises the scale via ``torch.tensor``, which is *not*
        # graph-safe -- production never lands here (it always passes a
        # tensor through ``_global_scale_passthrough``).
        x_global_scale_buf = torch.tensor(
            [float(a_global_scale)], dtype=torch.float32, device=x.device
        )

    _pipelined_moe_kernel_scaled[grid](
        x,
        w3,
        x_scale_buf,
        w_scale_buf,
        bias_buf,
        y,
        gather_buf,
        scatter_buf,
        gate_scal_buf,
        expert_remap_buf,
        slice_offs_buf,
        slice_sizes_buf,
        block_offs_buf,
        block_schedule_buf,
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
        M,
        M_X,
        N,
        K,
        x_global_scale_buf,
        num_tiles_total,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCKS_PER_EXPERT=blocks_per_expert,
        X_FORMAT=a_format,
        W_FORMAT=b_format,
        UPCAST_INDICES=False,
        HAS_X_BLOCK_SCALE=has_a_block_scale,
        HAS_W_BLOCK_SCALE=has_w_block_scale,
        HAS_BIAS=bias is not None,
        HAS_GATHER=gather_indx is not None,
        HAS_SCATTER=scatter_indx is not None,
        DO_SWIGLU=swiglu is not None,
        SWIGLU_ALPHA=float(swiglu_alpha),
        SWIGLU_LIMIT=float(swiglu_limit),
        OUT_BLOCK_N=out_block_n,
        APPLY_GATE_SCAL=gate_scal is not None,
        HAS_EXPERT_REMAP=expert_remap is not None,
        HAS_RAGGED_OFFS=has_ragged_offs,
        NUM_WARPS=num_warps,
        NUM_BUFFERS=num_buffers,
        SCALE_LOAD_MODE=scale_load_mode,
        W_TRANSPOSE=w_transpose,
        NUM_SUBTILES=NUM_SUBTILES,
        EVEN_K=EVEN_K,
        APPLY_X_GLOBAL_SCALE=apply_x_global_scale,
        USE_WARP_PIPELINE=use_warp_pipeline,
        PERSISTENT=persistent,
        USE_BLOCK_SCHEDULE=use_block_schedule,
        N_EXPTS_TOT=n_slices,
        num_warps=num_warps,
    )


# ---------------------------------------------------------------------------
# Public Python entry points (one per kernel that TASKS.md asks for)
# ---------------------------------------------------------------------------


# CDNA4 MFMA: regular = 16x16x32, scaled = 16x16x128 (BLOCK_K constraint).
_MFMA_K = 32
_MFMA_SCALED_K = 128
_MFMA_M = 16


def _round_up_int(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _autotune_block(
    M: int,
    N: int,
    K: int,
    *,
    do_swiglu: bool = False,
    ragged: bool = False,
    scaled_mfma: bool = False,
    a_format: str = "e2m1",
    scale_load_mode: str = "transpose",
) -> tuple[int, int, int, int]:
    """Pick ``(BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS)`` for given shape.

    Heuristic obtained by sweeping ``benchmarks/moe_gluon_perf_sweep.py``
    on MI355 with the gpt-oss-120b MoE dimensions (``H=I=2880, E=128,
    topk=4``). Each candidate must be (a) spill-free per ``static_profile``
    and (b) the highest-TFLOPs config at its shape.

    Plain bf16/fp16 path
    ~~~~~~~~~~~~~~~~~~~~

    * **Dense gating GEMM** (``do_swiglu=False, ragged=False``).
      Output ``N=128`` (``num_local_experts``) so we keep ``BLOCK_N=64``
      to give us ``grid_n=2`` and rely on growing ``grid_m`` for fill:
        - M <= 1024 : 64x64x64,  8 warps  (decode + small prefill)
        - M <= 2048 : 128x64x64, 8 warps
        - M  > 2048 : 128x64x64, 4 warps  (prefill, more CTAs)

    * **Fused SwiGLU 1st GEMM** (``do_swiglu=True``). Internal
      ``BLOCK_N`` covers the ``gate || linear`` width (``2*OUT_BLOCK_N``);
      ``BLOCK_K=32`` keeps VGPR pressure for the swiglu reduce manageable:
        - M <= 8192 :  64x128x32, 4 warps
        - M  > 8192 : 128x128x32, 4 warps

    * **Ragged 2nd GEMM + scatter combine** (``ragged=True``).
      Same per-tile MFMA flow as the gating GEMM but with E experts'
      worth of CTAs in the launch grid; benefits from larger blocks at
      prefill scale to amortise scatter epilogue:
        - M <= 8192 :  64x128x32, 4 warps
        - M  > 8192 : 128x128x32, 4 warps  (256x256 saturates VGPRs
          on MI355 -> spills; 128x128 stays under the 256-VGPR limit
          while still hitting 273 TFLOPs).

    Scaled MFMA path  (``scaled_mfma=True``)
    ~~~~~~~~~~~~~~~~~

    ``gl.amd.cdna4.mfma_scaled`` is 16x16x128 (M x N x K), so
    ``BLOCK_K`` must be a multiple of 128 and ``>= 128``. The heuristic
    tiers off the logical ``M`` (which already includes per-expert
    padding for the MoE wrappers):

    * tiny decode (``M <= 512``, e.g. B=1 with E=128): tall-thin
      ``64 x 128 x 512`` with 8 warps -- the few CTAs need every MFMA
      they can hide under the LDS prefetch.

    * medium (``M <= 16384``, decode B=32..1024 or chunked prefill):
      ``64 x 256 x BK`` with low BM keeps grid_m large; on mxfp4 inputs
      ``BK=256`` wins (lower per-tile mfma latency, BK=128 is bandwidth
      bound on the half-byte X load); fp8 keeps ``BK=128`` and bumps
      warps to 8 to recover occupancy.

    * large (``M > 16384``, prefill B=4096..8192): the M_d=32768 grid
      can amortise huge tiles. mxfp4 hits its peak at ``256 x 256 x 128
      / NW=4`` (occ=4, vgpr~122); fp8 has higher VGPR pressure per
      tile, so the sweet spot is ``128 x 256 x 128 / NW=4``.

    Older heuristic comment about ``BLOCK_K=128 hurting bf16``: that
    was for the bf16 pipeline (register-staged, no async-copy). The
    scaled path is LDS-staged, so larger BK is cheap there.
    """
    if scaled_mfma:
        is_fp8 = a_format == "e4m3"
        if M <= 512:
            bm, bn, bk, nw = 64, 128, 512, 8
        elif is_fp8:
            # fp8 X tiles are 1 byte/elem so VGPR pressure is lower; we
            # promote to the large BM=128 tier as soon as M > 8192.
            if M <= 8192:
                bm, bn, bk, nw = 64, 256, 128, 8
            else:
                bm, bn, bk, nw = 128, 256, 128, 4
        else:
            # mxfp4 X is 4b packed but the dequant adds VGPR pressure;
            # BM=64 with BK=256 stays the sweet spot until M_d > 16384.
            if M <= 16384:
                bm, bn, bk, nw = 64, 256, 256, 4
            else:
                bm, bn, bk, nw = 256, 256, 128, 4
        # Clamp tile to actual shape (rounded up to the MFMA tile / scaled
        # K granularity). Tiny test shapes like 32x32x256 would otherwise
        # over-tile and yield NaN-padded reductions.
        bm = max(_MFMA_M, min(bm, _round_up_int(M, _MFMA_M)))
        bn = max(_MFMA_M, min(bn, _round_up_int(N, _MFMA_M)))
        bk = max(_MFMA_SCALED_K, min(bk, _round_up_int(K, _MFMA_SCALED_K)))
        # cdna4_upstream's in-LDS unswizzle reshapes the K_S-block dim
        # into `(BLOCK_K_S//8, 4, 16, 2, 2, 1)` which is only valid when
        # BLOCK_K_S = BLOCK_K // SCALE_BLOCK >= 8 (=> BLOCK_K >= 256 for
        # SCALE_BLOCK=32). Bump the autotune pick up to 256 so the mode
        # is usable. This matches what upstream's AMD opt_flags pick for
        # CDNA4 mxfp4 (block_k = 128 // 0.5 = 256).
        if scale_load_mode == "cdna4_upstream":
            bk = max(bk, 256)
            bk = min(bk, _round_up_int(K, _MFMA_SCALED_K))
        return bm, bn, bk, nw
    if do_swiglu:
        bm, bn, bk, nw = (64, 128, 32, 4) if M <= 8192 else (128, 128, 32, 4)
    elif ragged:
        bm, bn, bk, nw = (64, 128, 32, 4) if M <= 8192 else (128, 128, 32, 4)
    elif M <= 1024:
        bm, bn, bk, nw = (64, 64, 64, 8)
    elif M <= 2048:
        bm, bn, bk, nw = (128, 64, 64, 8)
    else:
        bm, bn, bk, nw = (128, 64, 64, 4)
    return bm, bn, bk, nw


def _scaled_lds_bytes(
    bm: int,
    bn: int,
    bk: int,
    *,
    has_x_block_scale: bool,
    has_w_block_scale: bool,
    a_format: str,
    num_buffers: int,
) -> int:
    """Bytes of LDS the scaled-MFMA pipeline allocates for a given tile +
    NUM_BUFFERS. Used by ``_autotune_pipeline`` to decide whether we can
    bump NUM_BUFFERS to 3 (which warp_pipeline needs) without busting
    CDNA4's 160 KB LDS budget.

    Matches ``MoEPipelinedProgram.initialize`` exactly: X / W / X-scale /
    W-scale buffers, each NB-deep. Scales are only allocated when
    ``SCALE_VIA_LDS`` (== swizzle / cdna4_upstream); we always assume
    they are for the worst-case budget here.
    """
    div_x = 2 if a_format == "e2m1" else 1
    # X: BM x (BK/div_x) bytes per slot; uint8 storage either way.
    x_bytes = bm * (bk // div_x)
    # W is always mxfp4 packed: BN x (BK/2).
    w_bytes = bn * (bk // 2)
    # Scale LDS shapes mirror MoEConfig.BLOCK_*_PRESHUFFLED:
    #   X-scale: (BM // PF) x (BK_S * PF) bytes  -> simplifies to BM*BK/32
    #   W-scale: (BN // PF) x (BK_S * PF) bytes  -> simplifies to BN*BK/32
    # (PF = _SCALE_PRESHUFFLE_FACTOR = 32, SCALE_BLOCK = 32, so the two
    # factors of 32 cancel; BK doesn't need a div_factor here because
    # mxfp4 scales are addressed in logical-K not packed-K.)
    x_scale_bytes = (bm * bk // 32) if has_x_block_scale else 0
    w_scale_bytes = (bn * bk // 32) if has_w_block_scale else 0
    per_buffer = x_bytes + w_bytes + x_scale_bytes + w_scale_bytes
    return num_buffers * per_buffer


def _autotune_pipeline(
    bm: int,
    bn: int,
    bk: int,
    *,
    K: int,
    scaled_mfma: bool,
    a_format: str,
    has_x_block_scale: bool,
    has_w_block_scale: bool,
    scale_load_mode: str,
) -> tuple[int, bool]:
    """Pick ``(NUM_BUFFERS, USE_WARP_PIPELINE)`` for the given tile.

    Current heuristic: default to ``(NB=2, USE_WARP_PIPELINE=False)``
    everywhere. ``warp_pipeline`` is plumbed end-to-end (see
    ``MoEPipelinedProgram.warp_pipeline``) and can be force-enabled
    by passing ``use_warp_pipeline=True`` to the launcher; empirically
    on gpt-oss-120b shapes the gain is washed out by the NB=3 VGPR
    pressure hit (occupancy drops 4 -> 3, see
    ``benchmarks/check_gluon_decode_perf.py --force-warp-pipeline``),
    so we ship it as an opt-in rather than default-on. The LDS-budget
    helper below stays so the autotuner can be made smarter in a future
    revision without re-introducing the dead-code risk.
    """
    del bm, bn, bk, K, scaled_mfma, a_format, scale_load_mode  # autotune-off
    del has_x_block_scale, has_w_block_scale
    return _DEFAULT_NUM_BUFFERS, False


_PERSISTENT_FORCE_ENV = "TOKENSPEED_MOE_GLUON_PERSISTENT"


def _persistent_env_override() -> bool | None:
    """Read ``TOKENSPEED_MOE_GLUON_PERSISTENT`` env (1/0/auto).

    ``1`` / ``true`` / ``on``  -> force persistent kernel,
    ``0`` / ``false`` / ``off`` -> force non-persistent,
    anything else (or unset) -> ``None`` (let the heuristic decide).
    """
    raw = os.environ.get(_PERSISTENT_FORCE_ENV, "").strip().lower()
    if raw in {"1", "true", "yes", "on", "force"}:
        return True
    if raw in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return None


def _should_use_persistent(num_tiles_total: int) -> bool:
    """Decide persistent vs traditional grid based on total tile count.

    Empirically on gpt-oss-120b MoE decode shapes (B=1..32, grid_n in
    [4, 23], n_active <= 4 -> 16..92 tiles total) the per-tile
    bookkeeping (compact_idx / pid_n div+mod + induction var) costs
    more SGPR/VGPR than the launch overhead it saves, dropping
    occupancy 4 -> 3 (dispatch) or 4 -> 2 (combine) and either tying
    or losing on wall time. We therefore default the heuristic to
    *off*; the env override and ``persistent=True`` kwarg let callers
    opt in for workloads where the launch overhead truly dominates
    (e.g. tiny per-tile MFMA work or very low tile count).
    """
    override = _persistent_env_override()
    if override is not None:
        return override
    del num_tiles_total
    return False


def _persistent_grid_size(num_tiles_total: int) -> int:
    """Pick how many CTAs the persistent kernel should launch.

    Cap at ``num_tiles_total`` (no point launching more CTAs than work
    items) and at ``_CDNA4_NUM_CUS * _PERSISTENT_OVERSUBSCRIBE`` (over-
    subscribing past this just queues CTAs on the same CU).
    """
    if num_tiles_total <= 0:
        return 1
    return max(1, min(num_tiles_total, _CDNA4_NUM_CUS * _PERSISTENT_OVERSUBSCRIBE))


def _can_use_warp_pipeline(
    bm: int,
    bn: int,
    bk: int,
    *,
    K: int,
    scaled_mfma: bool,
    a_format: str,
    has_x_block_scale: bool,
    has_w_block_scale: bool,
    scale_load_mode: str,
    num_buffers: int = 3,
) -> bool:
    """Return True iff ``warp_pipeline`` is *safely* runnable with
    ``num_buffers`` on the requested tile: enough K-iters and the LDS
    budget fits under ``_CDNA4_LDS_BUDGET``.

    Use this from test / bench harnesses that want to force-enable
    warp_pipeline without triggering an OOR at compile time.
    """
    if not scaled_mfma:
        return False
    if num_buffers < 3:
        return False
    if (K + bk - 1) // bk < num_buffers:
        return False
    via_lds = scale_load_mode in ("swizzle", "cdna4_upstream")
    bytes_used = _scaled_lds_bytes(
        bm,
        bn,
        bk,
        has_x_block_scale=has_x_block_scale and via_lds,
        has_w_block_scale=has_w_block_scale and via_lds,
        a_format=a_format,
        num_buffers=num_buffers,
    )
    return bytes_used <= _CDNA4_LDS_BUDGET


def gluon_bf16_gating_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
    num_buffers: int = _DEFAULT_NUM_BUFFERS,
    persistent: bool | None = None,
    num_ctas: int | None = None,
) -> torch.Tensor:
    """bf16/fp16 dense GEMM ``y = x @ w`` (gating projection).

    Special-cases the non-MoE path of :func:`_pipelined_moe_kernel_scaled`
    -- no gather, no scatter, no swiglu, no per-expert metadata -- but still
    uses the same software-pipelined kernel body.

    Block size defaults are picked by :func:`_autotune_block` based on
    the input shape; pass explicit overrides to bench-tune.
    """
    assert x.dim() == 2 and w.dim() == 2
    M, K = x.shape
    K_W, N = w.shape
    assert K == K_W
    bm, bn, bk, nw = _autotune_block(M, N, K)
    block_m = block_m or bm
    block_n = block_n or bn
    block_k = block_k or bk
    num_warps = num_warps or nw
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    _launch_kernel(
        x,
        w,
        y=y,
        bias=bias,
        gather_indx=None,
        scatter_indx=None,
        gate_scal=None,
        a_ragged_metadata=None,
        swiglu=None,
        out_block_n=block_n,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_buffers=num_buffers,
        persistent=persistent,
        num_ctas=num_ctas,
    )
    return y


def gluon_bf16_dispatch_swiglu(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    bias: torch.Tensor | None,
    a_ragged_metadata,
    gather_indx,
    swiglu_alpha: float = 1.0,
    swiglu_limit: float = 0.0,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
    num_buffers: int = _DEFAULT_NUM_BUFFERS,
    persistent: bool | None = None,
    num_ctas: int | None = None,
) -> torch.Tensor:
    """Dispatch + 1st GEMM + fused SwiGLU for MoE.

    The output dtype is ``x.dtype``; the output N is ``w.shape[-1] // 2``
    because SwiGLU consumes pairs of (gate, linear) along the N axis.
    """
    assert w.ndim == 3 and w.shape[-1] % 2 == 0
    M = x.shape[-2]
    N = w.shape[-1]
    bm, bn, bk, nw = _autotune_block(M, N, w.shape[-2], do_swiglu=True)
    block_m = block_m or bm
    block_n = block_n or bn
    block_k = block_k or bk
    num_warps = num_warps or nw
    out_block_n = block_n // 2
    y = torch.empty((M, N // 2), device=x.device, dtype=x.dtype)
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
        num_buffers=num_buffers,
        persistent=persistent,
        num_ctas=num_ctas,
    )
    return y


def gluon_bf16_combine(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    bias: torch.Tensor | None,
    a_ragged_metadata,
    scatter_indx,
    gate_scal: torch.Tensor | None = None,
    n_tokens: int | None = None,
    n_expts_act: int | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
    num_buffers: int = _DEFAULT_NUM_BUFFERS,
    persistent: bool | None = None,
    num_ctas: int | None = None,
) -> torch.Tensor:
    """2nd GEMM + scatter combine for MoE.

    Accumulates ``y[token] = sum_{e in topk(token)} gate_scal[e] *
    (x_e @ w_e)`` via the kernel's optional scatter+gate_scal post-write.
    The combine across the top-k axis is a final ``view + sum`` on host.
    """
    assert w.ndim == 3
    M = x.shape[-2]
    N = w.shape[-1]
    if n_tokens is None:
        n_tokens = M
    bm, bn, bk, nw = _autotune_block(
        M, N, w.shape[-2], ragged=a_ragged_metadata is not None
    )
    block_m = block_m or bm
    block_n = block_n or bn
    block_k = block_k or bk
    num_warps = num_warps or nw
    y = torch.zeros((n_tokens, N), device=x.device, dtype=x.dtype)
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
        num_buffers=num_buffers,
        persistent=persistent,
        num_ctas=num_ctas,
    )
    if n_expts_act is not None and n_expts_act > 1:
        y = y.view(n_tokens, n_expts_act, N).sum(dim=1)
    return y


def gluon_mxfp_gating_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    x_scale: torch.Tensor | None = None,
    a_format: str = "e2m1",
    a_global_scale: torch.Tensor | float = 1.0,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
    num_buffers: int | None = None,
    use_warp_pipeline: bool | None = None,
    scale_load_mode: str = "transpose",
    w_transpose: bool = False,
    persistent: bool | None = None,
    num_ctas: int | None = None,
) -> torch.Tensor:
    # Scaled-MFMA dense GEMM y = (a_scale * x) @ (w_scale * w).
    # See _launch_kernel for scale_load_mode and tensor layouts.
    assert x.dim() == 2 and w.dim() == 2
    M = x.shape[0]
    N = w.shape[-1]
    div_a = 2 if a_format == "e2m1" else 1
    K = x.shape[-1] * div_a
    bm, bn, bk, nw = _autotune_block(
        M,
        N,
        K,
        scaled_mfma=True,
        a_format=a_format,
        scale_load_mode=scale_load_mode,
    )
    block_m = block_m or bm
    block_n = block_n or bn
    block_k = block_k or bk
    num_warps = num_warps or nw
    nb_auto, wp_auto = _autotune_pipeline(
        block_m,
        block_n,
        block_k,
        K=K,
        scaled_mfma=True,
        a_format=a_format,
        has_x_block_scale=a_format == "e2m1",
        has_w_block_scale=True,
        scale_load_mode=scale_load_mode,
    )
    num_buffers = num_buffers if num_buffers is not None else nb_auto
    use_warp_pipeline = use_warp_pipeline if use_warp_pipeline is not None else wp_auto
    y = torch.empty((M, N), device=x.device, dtype=out_dtype)
    _launch_kernel(
        x,
        w,
        y=y,
        bias=bias,
        gather_indx=None,
        scatter_indx=None,
        gate_scal=None,
        a_ragged_metadata=None,
        swiglu=None,
        out_block_n=block_n,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        a_format=a_format,
        b_format="e2m1",
        x_scale=x_scale,
        w_scale=w_scale,
        a_global_scale=a_global_scale,
        scale_load_mode=scale_load_mode,
        w_transpose=w_transpose,
        num_buffers=num_buffers,
        use_warp_pipeline=use_warp_pipeline,
        persistent=persistent,
        num_ctas=num_ctas,
    )
    return y


def gluon_mxfp_dispatch_swiglu(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    x_scale: torch.Tensor | None = None,
    a_format: str = "e2m1",
    a_global_scale: torch.Tensor | float = 1.0,
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
    scale_load_mode: str = "transpose",
    w_transpose: bool = False,
    persistent: bool | None = None,
    num_ctas: int | None = None,
) -> torch.Tensor:
    """Scaled-MFMA dispatch + 1st GEMM + fused SwiGLU."""
    assert w.ndim == 3 and w.shape[-1] % 2 == 0
    # Output rows: dispatched tile count (= ``gather_indx.numel()`` when
    # ``x`` is the un-permuted (n_tokens, H) input from production), or
    # ``x.shape[-2]`` for pre-permuted bench-style calls.
    if gather_indx is not None:
        gather_t = (
            gather_indx.src_indx if hasattr(gather_indx, "src_indx") else gather_indx
        )
        M = int(gather_t.shape[0])
    else:
        M = x.shape[-2]
    N = w.shape[-1]
    div_a = 2 if a_format == "e2m1" else 1
    K = x.shape[-1] * div_a
    bm, bn, bk, nw = _autotune_block(
        M,
        N,
        K,
        do_swiglu=True,
        scaled_mfma=True,
        a_format=a_format,
        scale_load_mode=scale_load_mode,
    )
    block_m = block_m or bm
    block_n = block_n or bn
    block_k = block_k or bk
    num_warps = num_warps or nw
    nb_auto, wp_auto = _autotune_pipeline(
        block_m,
        block_n,
        block_k,
        K=K,
        scaled_mfma=True,
        a_format=a_format,
        has_x_block_scale=a_format == "e2m1",
        has_w_block_scale=True,
        scale_load_mode=scale_load_mode,
    )
    num_buffers = num_buffers if num_buffers is not None else nb_auto
    use_warp_pipeline = use_warp_pipeline if use_warp_pipeline is not None else wp_auto
    out_block_n = block_n // 2
    y = torch.empty((M, N // 2), device=x.device, dtype=out_dtype)
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
        a_format=a_format,
        b_format="e2m1",
        x_scale=x_scale,
        w_scale=w_scale,
        a_global_scale=a_global_scale,
        scale_load_mode=scale_load_mode,
        w_transpose=w_transpose,
        num_buffers=num_buffers,
        use_warp_pipeline=use_warp_pipeline,
        persistent=persistent,
        num_ctas=num_ctas,
    )
    return y


def gluon_mxfp_combine(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    x_scale: torch.Tensor | None = None,
    a_format: str = "e2m1",
    a_global_scale: torch.Tensor | float = 1.0,
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
    scale_load_mode: str = "transpose",
    w_transpose: bool = False,
    persistent: bool | None = None,
    num_ctas: int | None = None,
) -> torch.Tensor:
    """Scaled-MFMA 2nd GEMM + scatter combine for MoE."""
    assert w.ndim == 3
    M = x.shape[-2]
    N = w.shape[-1]
    div_a = 2 if a_format == "e2m1" else 1
    K = x.shape[-1] * div_a
    bm, bn, bk, nw = _autotune_block(
        M,
        N,
        K,
        ragged=a_ragged_metadata is not None,
        scaled_mfma=True,
        a_format=a_format,
        scale_load_mode=scale_load_mode,
    )
    block_m = block_m or bm
    block_n = block_n or bn
    block_k = block_k or bk
    num_warps = num_warps or nw
    nb_auto, wp_auto = _autotune_pipeline(
        block_m,
        block_n,
        block_k,
        K=K,
        scaled_mfma=True,
        a_format=a_format,
        has_x_block_scale=a_format == "e2m1",
        has_w_block_scale=True,
        scale_load_mode=scale_load_mode,
    )
    num_buffers = num_buffers if num_buffers is not None else nb_auto
    use_warp_pipeline = use_warp_pipeline if use_warp_pipeline is not None else wp_auto
    # Scatter writeback produces a *flat* ``(n_tokens * n_expts_act, N)``
    # buffer (matching upstream ``triton_kernels.matmul`` semantics). The
    # caller-supplied ``n_tokens`` is the post-combine token count; with
    # ``n_expts_act > 1`` we still need ``n_tokens * n_expts_act`` rows
    # in the kernel's HBM output so each expert's slot is independently
    # writable, then we reduce over the top-k axis below. With
    # ``n_tokens=None`` (bench callers) we default to ``M`` rows.
    n_act_eff = int(n_expts_act) if n_expts_act is not None else 1
    if n_tokens is None:
        n_rows = M
        n_tokens_eff = M
    else:
        n_tokens_eff = int(n_tokens)
        n_rows = n_tokens_eff * n_act_eff
    y = torch.zeros((n_rows, N), device=x.device, dtype=out_dtype)
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
        a_format=a_format,
        b_format="e2m1",
        x_scale=x_scale,
        w_scale=w_scale,
        a_global_scale=a_global_scale,
        scale_load_mode=scale_load_mode,
        w_transpose=w_transpose,
        num_buffers=num_buffers,
        use_warp_pipeline=use_warp_pipeline,
        persistent=persistent,
        num_ctas=num_ctas,
    )
    if n_act_eff > 1:
        y = y.view(n_tokens_eff, n_act_eff, N).sum(dim=1)
    return y


# ---------------------------------------------------------------------------
# Adapter that matches ``triton_kernels.matmul`` signature (for the
# kernel registry / selector)
# ---------------------------------------------------------------------------


_TUNING_KW = frozenset(
    {"block_m", "block_n", "block_k", "num_warps", "num_buffers", "dtype"}
)


def _extract_gluon_raw_w(w):
    """Return the raw ``(E, K_packed, N) uint8`` weight tensor that the
    Gluon scaled MoE launcher expects, given a possibly upstream-wrapped
    ``triton_kernels.tensor.Tensor``.

    On AMD CDNA4 the value layout used by ``swizzle_mxfp4`` is
    ``StridedLayout(major_dim=-2)``: the production runtime calls
    ``wrap_torch_tensor`` on a *non-contiguous* ``transpose(-2, -1)`` of
    the loaded weight, so ``wrap_torch_tensor`` already infers the
    target layout and ``convert_layout`` is a no-op -- the underlying
    ``storage.data`` is the original transposed view of shape
    ``(E, K_packed, N)`` with K contiguous (``stride == (..., 1,
    K_packed)``). The Gluon kernel only consults ``.stride()`` for its
    address math, so we can pass this through zero-copy.

    For tensors that are not upstream-wrapped (``torch.Tensor`` already
    in the right shape), pass through unchanged.
    """
    if isinstance(w, torch.Tensor):
        return w
    if not isinstance(w, _UpstreamWrappedTensor):
        return w
    return w.storage.data


def _extract_gluon_raw_s(s):
    """Return the raw uint8 scale tensor that Gluon's ``cdna4_upstream``
    mode consumes. ``CDNA4MXScaleLayout.swizzle_data`` and our host
    ``_swizzle_scales_cdna4_upstream`` are bit-equivalent (verified by
    ``benchmarks/check_cdna4_upstream_scale.py``), so an upstream-
    wrapped scale's underlying bytes can be passed straight through.
    """
    if isinstance(s, torch.Tensor):
        return s
    if not isinstance(s, _UpstreamWrappedTensor):
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
    """Return the per-tensor flex scale in a form the Gluon launcher can
    take without going through ``.item()`` on the host stream.

    The launcher accepts ``Tensor | float | None`` for ``a_global_scale``;
    passing the upstream flex-scale tensor unchanged keeps the launch
    CUDA/HIP-graph capturable (the launcher itself materialises a
    1-element f32 device tensor and the kernel ``gl.load``s it).
    """
    if scale is None:
        return 1.0
    if isinstance(scale, torch.Tensor):
        return scale
    return float(scale)


def _try_dispatch_mxfp(
    x: torch.Tensor,
    w,
    bias: torch.Tensor | None,
    *,
    a_ragged_metadata,
    gather_indx,
    scatter_indx,
    precision_config,
    fused_activation,
    n_tokens,
    n_expts_act,
    passthrough,
) -> tuple[torch.Tensor | None, bool]:
    """Try to route an upstream-shaped ``moe_experts`` call to the
    Gluon scaled-MFMA path (mxfp4 weight + fp8 / mxfp4 activation).

    Returns ``(out, True)`` on success, ``(None, False)`` if anything in
    the call shape is outside what we currently support, in which case
    the caller falls back to ``triton_kernels.matmul``.
    """
    if precision_config is None:
        return None, False
    b_mx_scale = getattr(precision_config, "b_mx_scale", None)
    if b_mx_scale is None:
        return None, False

    # NOTE: previously this guard fell back to upstream whenever the
    # current stream was capturing into a CUDA/HIP graph, because the
    # Gluon launcher took ``a_global_scale: float`` and ``.item()``-ed
    # the upstream fp8 flex-scale tensor on the host stream (forbidden
    # during capture). Now the launcher accepts a tensor and the kernel
    # ``gl.load``s the scale at run time, so this path is graph-safe and
    # Gluon participates in decode CUDA-graph replay just like upstream.

    # KILL-SWITCH: until production-shape gluon swiglu compile lands and
    # the W_TRANSPOSE+scatter NaN is fixed, allow disabling the gluon
    # dispatcher entirely via env knob so we can isolate baseline /
    # upstream-only behaviour.
    import os as _os

    if _os.environ.get("TOKENSPEED_MOE_GLUON_DISABLE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None, False

    flex = getattr(precision_config, "flex_ctx", None)
    lhs = getattr(flex, "lhs_data", None) if flex is not None else None
    fp8_dtype = getattr(lhs, "dtype", None) if lhs is not None else None
    fp8_scale = getattr(lhs, "scale", None) if lhs is not None else None

    a_mx_scale = getattr(precision_config, "a_mx_scale", None)
    if fp8_dtype is not None and a_mx_scale is not None:
        return None, False

    if fp8_dtype is not None:
        a_format = "e4m3"
        a_global_scale = _global_scale_passthrough(fp8_scale)
        x_view = x.view(torch.uint8) if x.dtype != torch.uint8 else x
        x_scale = None
    elif a_mx_scale is not None:
        a_format = "e2m1"
        a_global_scale = 1.0
        x_view = x.view(torch.uint8) if x.dtype != torch.uint8 else x
        x_scale = _extract_gluon_raw_s(a_mx_scale)
        if not isinstance(x_scale, torch.Tensor):
            return None, False
    else:
        return None, False

    if precision_config.out_dtype is not None:
        out_dtype = precision_config.out_dtype
    elif x.dtype.is_floating_point:
        out_dtype = x.dtype
    else:
        out_dtype = torch.bfloat16

    w_raw = _extract_gluon_raw_w(w)
    s_raw = _extract_gluon_raw_s(b_mx_scale)

    if not isinstance(w_raw, torch.Tensor) or not isinstance(s_raw, torch.Tensor):
        return None, False
    if w_raw.ndim != 3:
        return None, False

    # Gluon's launcher consults ``gather_indx.src_indx`` / ``scatter_indx.dst_indx``
    # whereas the upstream-routed callers (production + tests) often pass a
    # plain ``torch.Tensor``. Accept both by lazily wrapping bare tensors.
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

    if fused_activation is not None and swiglu_args is None:
        return None, False

    gammas = passthrough.get("gammas")
    betas = passthrough.get("betas")
    out_alpha = passthrough.get("out_alpha")
    if betas is not None or out_alpha is not None:
        return None, False
    epilogue = passthrough.get("epilogue")
    if epilogue is not None:
        return None, False
    fused_comm = passthrough.get("fused_comm")
    if fused_comm is not None:
        return None, False
    c_in = passthrough.get("c") or passthrough.get("c_acc_in")
    if c_in is not None:
        return None, False

    try:
        # Per-path kill-switches (debug aid for bisecting end-to-end
        # crashes). Set ``TOKENSPEED_MOE_GLUON_DISPATCH=0`` /
        # ``TOKENSPEED_MOE_GLUON_COMBINE=0`` to fall back to upstream
        # for one specific kernel without disabling gluon entirely.
        _disable_dispatch = _os.environ.get(
            "TOKENSPEED_MOE_GLUON_DISPATCH", ""
        ).strip().lower() in {"0", "false", "no", "off"}
        _disable_combine = _os.environ.get(
            "TOKENSPEED_MOE_GLUON_COMBINE", ""
        ).strip().lower() in {"0", "false", "no", "off"}

        if has_scatter and not has_gather:
            if _disable_combine:
                return None, False
            # gemm + scatter combine path. The pre-ragged-offs bug
            # (off_m past per-expert tail loaded NEXT expert's rows)
            # used to taint this path's writeback with NaNs whenever
            # per-expert size < BLOCK_M; the HAS_RAGGED_OFFS fix
            # bounds writes to the actual per-expert tail and now
            # produces bit-identical output vs upstream.
            out = gluon_mxfp_combine(
                x_view,
                w_raw,
                s_raw,
                x_scale=x_scale,
                a_format=a_format,
                a_global_scale=a_global_scale,
                bias=bias,
                a_ragged_metadata=a_ragged_metadata,
                scatter_indx=scatter_indx,
                gate_scal=gammas,
                # ``n_tokens`` / ``n_expts_act`` arrive as direct
                # parameters on ``_try_dispatch_mxfp`` (see
                # ``_gluon_bf16_ragged_matmul``), NOT in ``passthrough``;
                # falling back to the latter silently dropped them and
                # left the kernel allocating a ``(M, N)`` slab without
                # the post-kernel top-k reduction, producing a
                # ``(top_k * n_tokens, N)`` output that crashed the
                # downstream layernorm with a shape mismatch on decode
                # batches.
                n_tokens=n_tokens,
                n_expts_act=n_expts_act,
                out_dtype=out_dtype,
                scale_load_mode="cdna4_upstream",
                w_transpose=True,
            )
            return out, True

        if not has_scatter and swiglu_args is not None:
            if _disable_dispatch:
                return None, False
            # Covers both gather_indx-driven dispatch+swiglu (production
            # path from triton_kernel_fp8.py) AND the pre-permuted
            # bench-style call where ``gather_indx=None``.
            swiglu_alpha, swiglu_limit = swiglu_args
            out = gluon_mxfp_dispatch_swiglu(
                x_view,
                w_raw,
                s_raw,
                x_scale=x_scale,
                a_format=a_format,
                a_global_scale=a_global_scale,
                bias=bias,
                a_ragged_metadata=a_ragged_metadata,
                gather_indx=gather_indx,
                out_dtype=out_dtype,
                swiglu_alpha=swiglu_alpha,
                swiglu_limit=swiglu_limit,
                scale_load_mode="cdna4_upstream",
                w_transpose=True,
            )
            return out, True

        if not has_gather and not has_scatter and swiglu_args is None:
            if x.ndim != 2 or w_raw.shape[0] != 1:
                return None, False
            out = gluon_mxfp_gating_gemm(
                x_view,
                w_raw.squeeze(0),
                s_raw if s_raw.ndim == 2 else s_raw.squeeze(0),
                x_scale=x_scale,
                a_format=a_format,
                a_global_scale=a_global_scale,
                bias=bias,
                out_dtype=out_dtype,
                scale_load_mode="cdna4_upstream",
                w_transpose=True,
            )
            return out, True
    except Exception as exc:  # noqa: BLE001
        # Defensive: if anything in the gluon launcher trips, fall back
        # to upstream rather than aborting the whole MoE forward. The
        # ``TOKENSPEED_MOE_GLUON_DEBUG`` env knob promotes the silent
        # warning into a stack-trace re-raise so wiring regressions can
        # be diagnosed quickly during development.
        import logging
        import os as _os

        logging.getLogger("tokenspeed_kernel.ops.moe.gluon").warning(
            "_try_dispatch_mxfp falling back to upstream: %s: %s",
            type(exc).__name__,
            exc,
        )
        if _os.environ.get("TOKENSPEED_MOE_GLUON_DEBUG", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
            "debug",
        }:
            raise
        return None, False

    return None, False


def _gluon_bf16_ragged_matmul(
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
    block_m: int = _DEFAULT_BLOCK_M,
    block_n: int = _DEFAULT_BLOCK_N,
    block_k: int = _DEFAULT_BLOCK_K,
    num_warps: int = _DEFAULT_NUM_WARPS,
    num_buffers: int = _DEFAULT_NUM_BUFFERS,
    **passthrough,
) -> torch.Tensor:
    """Selector-facing entry: matches the upstream ``matmul`` signature.

    Falls back to ``triton_kernels.matmul`` for unsupported precisions
    (mxfp4 weight scales / fp8 activation flex data) so we never break
    the gpt-oss-120b path while we land scaled-MFMA features.

    Extra keyword arguments (``gammas``, ``betas``, ``out_alpha``,
    ``c``/``c_acc_in``, ``fused_comm``, ``epilogue``,
    ``b_ragged_metadata``, ...) the runtime backends pass to
    ``tokenspeed_kernel.moe_experts`` are forwarded transparently to the
    upstream ``triton_kernels.matmul`` so combine-side scaling
    (``gammas``) and other epilogue knobs are preserved when we fall
    back.
    """
    if not _supports_pure_bf16(precision_config, fused_activation):
        # Fast path: route mxfp4 weight (+ fp8 / mxfp4 activation) calls
        # to our scaled-MFMA Gluon launchers. The launchers already do
        # the top-k combine reduction internally, so we return as-is on
        # success.
        out, ok = _try_dispatch_mxfp(
            x,
            w,
            bias,
            a_ragged_metadata=a_ragged_metadata,
            gather_indx=gather_indx,
            scatter_indx=scatter_indx,
            precision_config=precision_config,
            fused_activation=fused_activation,
            n_tokens=n_tokens,
            n_expts_act=n_expts_act,
            passthrough=passthrough,
        )
        if ok:
            return out

        out = _upstream_matmul(
            x,
            w,
            bias,
            a_ragged_metadata=a_ragged_metadata,
            gather_indx=gather_indx,
            scatter_indx=scatter_indx,
            precision_config=precision_config,
            fused_activation=fused_activation,
            **{k: v for k, v in passthrough.items() if k not in _TUNING_KW},
        )
        # Mirror the post-processing applied by the registered
        # ``triton_kernels`` wrapper (see ``ops.moe.triton_kernels._matmul``):
        # when the caller passes ``scatter_indx`` together with
        # ``n_expts_act > 1``, the upstream matmul writes a flat
        # ``(n_tokens * n_expts_act, N)`` slab that has to be folded back
        # into ``(n_tokens, N)`` via top-k expert summation. The previous
        # adapter skipped this step, which surfaced the moment we won
        # selection over ``triton_kernels_gemm_combine``.
        if scatter_indx is not None and n_expts_act is not None and n_expts_act > 1:
            assert (
                n_tokens is not None
            ), "n_tokens required when n_expts_act > 1 for top-k reduction"
            out = out.view(n_tokens, n_expts_act, out.shape[-1]).sum(dim=1)
        return out

    # bf16 / fp16 path
    M = x.shape[-2]
    if w.ndim == 3:
        N = w.shape[-1]
        K = w.shape[-2]
    else:
        K, N = w.shape

    # Apply the shape-aware autotune if the caller did not pin block sizes.
    if (
        block_m == _DEFAULT_BLOCK_M
        and block_n == _DEFAULT_BLOCK_N
        and block_k == _DEFAULT_BLOCK_K
    ):
        bm, bn, bk, nw = _autotune_block(M, N, K, ragged=a_ragged_metadata is not None)
        block_m, block_n, block_k = bm, bn, bk
        if num_warps == _DEFAULT_NUM_WARPS:
            num_warps = nw

    out_dtype = (precision_config.out_dtype if precision_config else None) or x.dtype
    if scatter_indx is not None:
        y = torch.zeros((n_tokens or M, N), device=x.device, dtype=out_dtype)
    else:
        y = torch.empty((M, N), device=x.device, dtype=out_dtype)
    _launch_kernel(
        x,
        w,
        y=y,
        bias=bias,
        gather_indx=gather_indx,
        scatter_indx=scatter_indx,
        gate_scal=None,
        a_ragged_metadata=a_ragged_metadata,
        swiglu=None,
        out_block_n=block_n,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_buffers=num_buffers,
    )
    if scatter_indx is not None and n_expts_act and n_expts_act > 1:
        y = y.view(n_tokens, n_expts_act, N).sum(dim=1)
    return y


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _kernel_priority() -> int:
    # The Gluon MoE GEMM is a throughput/latency-optimized kernel for MI355
    # (gfx950). When enabled (default), sit clearly above the upstream
    # ``triton_kernels`` MoE GEMM (PERFORMANT + 2 = 10) so that the selector
    # picks us on MI355 without requiring an explicit override.
    if _GLUON_DISABLED_ENV:
        # Drop below triton_kernels so the upstream path wins. Kept as a
        # candidate so that an explicit ``TOKENSPEED_KERNEL_OVERRIDE_MOE_EXPERTS``
        # can still target the Gluon kernel by name.
        return Priority.PORTABLE + 1  # 5
    return Priority.SPECIALIZED + 2  # 14


# Tag the kernels with the SelectionObjective categories they actually serve
# (``throughput`` / ``latency``). They are intentionally **not** tagged as
# ``portability`` -- portability falls back to ``triton_kernels``.
_common = dict(
    solution="triton",
    dtypes={torch.bfloat16, torch.float16, torch.uint8},
    capability=CapabilityRequirement(
        vendors=frozenset({"amd"}),
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
    ),
    priority=_kernel_priority(),
    tags={"throughput", "latency", "gluon"},
)

register_kernel(
    "moe",
    "experts",
    name="triton_kernels_gluon_dispatch_gemm",
    features={"ragged_metadata", "dispatch_gemm"},
    **_common,
)(_gluon_bf16_ragged_matmul)

register_kernel(
    "moe",
    "experts",
    name="triton_kernels_gluon_gemm_combine",
    features={"ragged_metadata", "gemm_combine"},
    **_common,
)(_gluon_bf16_ragged_matmul)

register_kernel(
    "moe",
    "experts",
    name="triton_kernels_gluon_matmul_ogs",
    features={"ragged_metadata"},
    **_common,
)(_gluon_bf16_ragged_matmul)


__all__ = [
    "_gluon_bf16_ragged_matmul",
    "assert_no_spills",
    "gluon_bf16_combine",
    "gluon_bf16_dispatch_swiglu",
    "gluon_bf16_gating_gemm",
    "gluon_mxfp_combine",
    "gluon_mxfp_dispatch_swiglu",
    "gluon_mxfp_gating_gemm",
    "static_profile",
]
