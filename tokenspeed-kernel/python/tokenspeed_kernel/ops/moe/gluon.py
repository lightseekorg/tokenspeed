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

_GLUON_DISABLE_VALUES = {"0", "false", "no", "off", "disable", "disabled"}
_GLUON_DISABLED_ENV = (
    os.environ.get("TOKENSPEED_MOE_GLUON", "").strip().lower() in _GLUON_DISABLE_VALUES
)


def _env_int(name: str, default: int) -> int:
    """Read an int from env. Returns ``default`` on missing / unparsable."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


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

_DEFAULT_NUM_BUFFERS = 2


# X / W SwizzledSharedLayout params (vec, per_phase, max_phase).
#
# Both can be overridden at process start via env
# ``TOKENSPEED_MOE_GLUON_SWIZZLE_{X,W}="vec,per_phase,max_phase"`` for
# closed-loop tuning. Env is read once at module import (not per
# kernel-launch) because the Triton JIT cache-key AST walker can't
# follow a regular Python function call inside a
# ``@gluon.constexpr_function`` body. Module-level constants are pure
# ``ast.Name`` references which the walker handles.
# def _parse_swizzle_env(
#     name: str, default: tuple[int, int, int]
# ) -> tuple[int, int, int]:
#     raw = os.environ.get(name, "").strip()
#     if not raw:
#         return default
#     parts = raw.split(",")
#     if len(parts) != 3:
#         raise ValueError(f"{name} must be 'vec,per_phase,max_phase' (got {raw!r})")
#     try:
#         v, p, m = (int(x) for x in parts)
#     except ValueError as e:
#         raise ValueError(f"{name}={raw!r} is not 3 ints") from e
#     return (v, p, m)


# _SWIZZLE_X = _parse_swizzle_env("TOKENSPEED_MOE_GLUON_SWIZZLE_X", (16, 1, 1))
# _SWIZZLE_W = _parse_swizzle_env("TOKENSPEED_MOE_GLUON_SWIZZLE_W", (16, 2, 8))

_CDNA4_LDS_BUDGET = 163840

_CDNA4_NUM_CUS = 256
_PERSISTENT_OVERSUBSCRIBE = 2
_PERSISTENT_TILES_THRESHOLD = _CDNA4_NUM_CUS * 3


# ---------------------------------------------------------------------------
# Layout factories (gluon constexpr functions)
# ---------------------------------------------------------------------------


@gluon.constexpr_function
def _store_layout(num_warps: int):
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
    a_format: str | None = None,
    num_buffers: int | None = None,
) -> str:
    del k, a_format, num_buffers  # legacy fallback knobs no longer used
    if mode != "swizzle":
        return mode
    # "swizzle" is the upstream CDNA4MXScaleLayout preshuffle path:
    # `unswizzle_mx_scale_cdna4` reshapes the K_S-block dim into
    # `(BLOCK_K_S // 8, 4, 16, 2, 2, 1)`; this requires
    # ``BLOCK_K_S = BLOCK_K // SCALE_BLOCK >= 8`` (i.e. BLOCK_K >= 256
    # for SCALE_BLOCK=32). N-side preshuffle factor is 32 so BLOCK_N
    # (and BLOCK_M when X has a scale) must be a multiple of 32. We
    # cannot silently fall back to "bypass" because the input scale
    # tensor is already in the upstream storage layout (K_S_pad*32
    # contig) which "bypass" cannot read; so we hard-assert.
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

        num_loads = 2  # x and w
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

        # X / W shared layouts: PaddedSharedLayout (Update 11) -- mirrors
        # the gfx950 a8w8 tutorial's padding pattern (the tutorial uses
        # plain ``mfma`` with k_width=32; we use ``mfma_scaled`` with
        # k_width=16 -- see DOT_K_WIDTH_{X,W} above).
        #
        # Padding stride ``[[1024, 16], [2048, 32]]`` is borrowed
        # verbatim from a8w8.  This pattern is tuned for that tutorial's
        # ``[BM/2, BK]=[128, 128]`` slot; on our auto-tuned tile range
        # (decode BM=32 / prefill BM=128 BN=256) it is *not* universally
        # optimal -- prefill currently regresses vs swizzled because the
        # padding inflates LDS and forces occ=1 + sgpr spills.  The next
        # step is a per-tile sweep over ``[[stride, pad]]`` (e.g. just
        # ``[[256, 16]]`` for prefill's 256B-row slots) before locking
        # the parameters in.
        #
        # Offset list and block_shape must match the allocated slot
        # shape exactly: X = ``[BM, BK_packed]``, W depends on
        # ``W_TRANSPOSE`` (True -> ``[BN, BK_packed]``,
        # False -> ``[BK_packed, BN]``).  Packed-K accounts for e2m1's
        # 2-per-byte storage.
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
                [[1024, 16], [2048, 32]],
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
                [[1024, 16], [2048, 32]],
                _row_major_offsets(w_shape[0], w_shape[1]),
                [],
                w_shape,
            )
        )

        # Half-N variant of ``shared_layout_w`` -- used by
        # :class:`MoESliceNProgram` to allocate two LDS slots, one
        # per W half (top = N[:BN/2], bot = N[BN/2:]). Splitting the
        # G->LDS path along N keeps each ``buffer_load_to_shared`` train
        # under the gfx950 TCP/L1 32 KB in-flight cap (a full BN=256
        # K-tile is ~32 KB and saturates that cap on its own). The
        # layout is identical in shape rule -- just halved along the
        # nonk axis -- so the row_major_offsets pattern stays valid.
        # See ``v7_sliceN`` in the gfx950-gluon-tutorials for the
        # canonical pattern.
        if W_TRANSPOSE:
            w_half_shape = [BLOCK_N // 2, BLOCK_K_PACKED_W_HOST]
        else:
            w_half_shape = [BLOCK_K_PACKED_W_HOST, BLOCK_N // 2]
        if (BLOCK_N // 2) >= 1 and BLOCK_K_PACKED_W_HOST >= 1:
            self.shared_layout_w_half_n = gl.constexpr(
                gl.PaddedSharedLayout(
                    [[1024, 16], [2048, 32]],
                    _row_major_offsets(w_half_shape[0], w_half_shape[1]),
                    [],
                    w_half_shape,
                )
            )
        else:
            # Fallback so the type-check is happy on tiny tiles (sliceN
            # is gated by ``_can_use_slice_n`` to BN >= 128 anyway).
            self.shared_layout_w_half_n = gl.constexpr(0)

        # Half-M variant of ``shared_layout_x`` -- used by
        # :class:`MoESliceMNProgram` to allocate two LDS slots, one per
        # X half along M (top = M[:BM/2], bot = M[BM/2:]). Splitting the
        # G->LDS X path along M complements the half-N W split so each
        # of the four ``buffer_load_to_shared`` half-tiles fits comfortably
        # under the gfx950 TCP/L1 32 KB in-flight cap and the four issues
        # can be staggered across the v8-style 4-region pipeline.
        if (BLOCK_M // 2) >= 1 and BLOCK_K_PACKED_X_HOST >= 1:
            self.shared_layout_x_half_m = gl.constexpr(
                gl.PaddedSharedLayout(
                    [[1024, 16], [2048, 32]],
                    _row_major_offsets(BLOCK_M // 2, BLOCK_K_PACKED_X_HOST),
                    [],
                    [BLOCK_M // 2, BLOCK_K_PACKED_X_HOST],
                )
            )
        else:
            self.shared_layout_x_half_m = gl.constexpr(0)

        # --- Previous Swizzled layout (kept for fallback / A-B testing).
        # self.shared_layout_x = gl.constexpr(
        #     gl.SwizzledSharedLayout(
        #         _SWIZZLE_X[0], _SWIZZLE_X[1], _SWIZZLE_X[2], order=[1, 0]
        #     )
        # )
        # self.shared_layout_w = gl.constexpr(
        #     gl.SwizzledSharedLayout(
        #         _SWIZZLE_W[0], _SWIZZLE_W[1], _SWIZZLE_W[2], order=[1, 0]
        #     )
        # )

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
    def issue_global_loads(self, load_idx, pred=1, USE_MASK: gl.constexpr = -1):
        cfg = self.cfg
        self.x_desc.issue_async_load(load_idx, self.x_buffer, pred, USE_MASK=USE_MASK)
        self.w_desc.issue_async_load(load_idx, self.w_buffer, pred, USE_MASK=USE_MASK)
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
        off_k_step = idx * self.BLOCK_K
        offsets = self.offsets + off_k_step * self.stride_k
        if USE_MASK_RESOLVED == 0:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                buffer.index(idx % NUM_BUFFERS),
                self.ptr,
                offsets,
            )
        else:
            # NOTE: do NOT pass ``other=0``. Gluon's ``buffer_load_to_shared``
            # routes ``other`` straight into ``ttag.BufferLoadToLocalOp``
            # (gluon_ir.cc:create_buffer_load_to_local), bypassing the
            # ``ConvertTritonLoadToBufferLoad`` ``isZeroConst(other)`` strip.
            # With a non-null ``other`` operand, the LLVM lowering
            # (LoadStoreOpToLLVM.cpp BufferLoadToLocalOpConversion) emits
            # ``cond = threadPred AND maskElem`` and ``emitBranch(cond)`` per
            # vector load -- per-element ``br i1`` blocks around every
            # ``buffer.load.async.lds``. Those branches prevent LLVM's
            # ``SIInsertWaitcnts`` from statically counting in-flight vmem,
            # so ``wait_asyncmark(N)`` collapses to ``s_waitcnt vmcnt(0)``
            # and kills the async pipeline depth on CDNA4. With ``other``
            # omitted the lowering uses ``cond = threadPred`` (warp-uniform),
            # leaving straight-line loads; masked-out lanes still issue the
            # load and rely on the buffer-descriptor's ``numRecords`` OOB
            # check to write 0 to LDS for out-of-range offsets.
            mask_k = gl.expand_dims(off_k_step + self.off_k, self.op_idx) < self.k_limit
            mask = mask_k & self.masks_nonk
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                buffer.index(idx % NUM_BUFFERS),
                self.ptr,
                offsets,
                mask=mask,
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
        # Upstream CDNA4MXScaleLayout swizzle: LDS holds the same
        # [BLOCK_NONK_PS=BLOCK_N/32, BLOCK_K_S * 32] uint8 tile shape;
        # the in-tile reordering follows upstream's 7-D pattern
        # (`triton_kernels.tensor_details.layout_details.cdna4_scale.unswizzle_mx_scale_cdna4`).
        # That requires BLOCK_K_SCALE % 8 == 0; the host preprocess pads
        # the global K_SCALE up to a multiple of 8 for us.
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
        # Upstream CDNA4MXScaleLayout 7-D unswizzle on the per-K_S slot,
        # followed by an in-LDS slice along the NONK axis to extract the
        # requested sub-tile (used by sliceN / sliceMN sub-MFMA paths).
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
    def issue_local_loads(self, mfma_idx):
        cfg = self.cfg
        x, w = self._load_xw(mfma_idx)

        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK

        # Scales only consumed by mfma_scaled; the plain-MFMA branch is
        # dead code on this scaled-only kernel but kept for the constexpr.
        if cfg.USE_MFMA_SCALED:
            # Dummy scales use e8m0=127 (== 2^0 = 1.0) so the dot is identity-
            # scaled when the operand has no real block scale (fp8 path).
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
                    scale_w = self.w_scale_desc.issue_local_load_unswizzle(
                        mfma_idx,
                        self.w_scale_buffer,
                        cfg.layout_w_scale,
                        cfg.BLOCK_N_PRESHUFFLED,
                        cfg.BLOCK_N,
                        BLOCK_K_SCALE,
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
        cfg = self.cfg
        EVEN_K: gl.constexpr = cfg.EVEN_K
        load_idx = 0
        mfma_idx = 0

        accumulator = gl.zeros(
            (cfg.BLOCK_M, cfg.BLOCK_N), dtype=gl.float32, layout=cfg.acc_layout
        )
        K_iters = gl.cdiv(loop_k, cfg.BLOCK_K)

        if EVEN_K:
            # No tail to peel: every iter is in-bounds, no mask anywhere.
            for _ in gl.static_range(cfg.NUM_BUFFERS - 1):
                load_idx = self.issue_global_loads(load_idx, USE_MASK=0)

            main_iters = K_iters - (cfg.NUM_BUFFERS - 1)
            gl.assume(main_iters >= 0)

            for i in range(0, main_iters):
                load_idx = self.issue_global_loads(load_idx, USE_MASK=0)
                self.async_wait(cfg.NUM_BUFFERS - 1)

                x, w, scale_x, scale_w = self.issue_local_loads(mfma_idx)
                mfma_idx += 1

                accumulator = self.mfma(x, scale_x, w, scale_w, accumulator)

            # epilogue
            for i in gl.static_range(cfg.NUM_BUFFERS - 1):
                self.async_wait(cfg.NUM_BUFFERS - 2 - i)
                x, w, scale_x, scale_w = self.issue_local_loads(mfma_idx)
                mfma_idx += 1
                accumulator = self.mfma(x, scale_x, w, scale_w, accumulator)
        else:
            # Peel last K-iter as masked; main loop stays unmasked.
            # Precondition: K_iters >= NUM_BUFFERS so the prologue + one
            # peeled tail fit (main_iters = K_iters - NB >= 0). The
            # launcher (gluon_mxfp_*; ``_autotune_block``) enforces
            # this; ``gl.assume`` propagates it to LLVM.
            for _ in gl.static_range(cfg.NUM_BUFFERS - 1):
                load_idx = self.issue_global_loads(load_idx, USE_MASK=0)

            main_iters = K_iters - cfg.NUM_BUFFERS
            gl.assume(main_iters >= 0)

            for i in range(0, main_iters):
                load_idx = self.issue_global_loads(load_idx, USE_MASK=0)
                self.async_wait(cfg.NUM_BUFFERS - 1)

                x, w, scale_x, scale_w = self.issue_local_loads(mfma_idx)
                mfma_idx += 1

                accumulator = self.mfma(x, scale_x, w, scale_w, accumulator)

            # Peeled tail: 1 masked issue + drain oldest + MFMA.
            load_idx = self.issue_global_loads(load_idx, USE_MASK=1)
            self.async_wait(cfg.NUM_BUFFERS - 1)
            x, w, scale_x, scale_w = self.issue_local_loads(mfma_idx)
            mfma_idx += 1
            accumulator = self.mfma(x, scale_x, w, scale_w, accumulator)

            # epilogue: same NUM_BUFFERS - 1 drain pattern as EVEN_K path.
            for i in gl.static_range(cfg.NUM_BUFFERS - 1):
                self.async_wait(cfg.NUM_BUFFERS - 2 - i)
                x, w, scale_x, scale_w = self.issue_local_loads(mfma_idx)
                mfma_idx += 1
                accumulator = self.mfma(x, scale_x, w, scale_w, accumulator)

        return accumulator

    @gluon.jit
    def pipeline_3stage(self, loop_k):
        """3-stage software pipeline: do the LDS ``local_load`` for the
        NEXT iter's tile at the BOTTOM of the current iter, then carry
        the loaded ``(x, w, scale_x, scale_w)`` registers into the next
        iter so the top-of-loop MFMA consumes a value already in VGPR
        (no ``ds_read`` latency in the critical path).

        Layout::

            issue prologue (NB-1)
            prime    : wait + local_load  -> x0, w0, sx0, sw0
            iter 0   : issue, wait, MFMA(x0..)  , local_load -> x1...
            iter 1   : issue, wait, MFMA(x1..)  , local_load -> x2...
            ...
            iter N-1 : issue, wait, MFMA(x_N-1..), local_load -> x_N
            final    :                MFMA(x_N..)
            epi      : (drain NB-2 leftover slots and consume them)

        This hides the ``ds_read`` latency that the 2-stage pipeline
        leaves exposed at the top of each iter -- a win when the MFMA
        chain is too short to absorb it (decode BM=32). The cost is one
        extra tile's worth of live VGPR (~+8 for fp8 BM=32, well under
        the ~120 budget headroom we measured at occ=3).

        Requires ``NUM_BUFFERS >= 2``. The pred-mask the 2-stage
        pipeline uses for tail issues is replaced here by a static-range
        epilogue, which the compiler unrolls into the steady-state
        instruction window (no dynamic predicate live across the loop).
        """
        cfg = self.cfg
        gl.static_assert(
            cfg.NUM_BUFFERS >= 2,
            "pipeline_3stage requires NUM_BUFFERS >= 2",
        )
        load_idx = 0
        mfma_idx = 0

        for _ in gl.static_range(cfg.NUM_BUFFERS - 1):
            load_idx = self.issue_global_loads(load_idx)

        accumulator = gl.zeros(
            (cfg.BLOCK_M, cfg.BLOCK_N), dtype=gl.float32, layout=cfg.acc_layout
        )
        K_iters = gl.cdiv(loop_k, cfg.BLOCK_K)
        gl.assume(K_iters > 0)
        main_iters = K_iters - (cfg.NUM_BUFFERS - 1)
        gl.assume(main_iters >= 0)

        # Prime: drain the oldest in-flight load and local_load it so
        # the first MFMA at the loop top has its operands in VGPR.
        self.async_wait(cfg.NUM_BUFFERS - 2)
        x, w, scale_x, scale_w = self.issue_local_loads(mfma_idx)
        mfma_idx += 1

        for _ in range(0, main_iters):
            load_idx = self.issue_global_loads(load_idx)
            self.async_wait(cfg.NUM_BUFFERS - 1)
            accumulator = self.mfma(x, scale_x, w, scale_w, accumulator)
            x, w, scale_x, scale_w = self.issue_local_loads(mfma_idx)
            mfma_idx += 1

        # Final main-loop MFMA consumes the last local-loaded carry.
        accumulator = self.mfma(x, scale_x, w, scale_w, accumulator)

        # Epilogue: drain remaining in-flight loads and consume them.
        # After main_iters issues+drains we still have NB-2 in flight
        # (the prime drained 1 from the NB-1 prologue, no extra issue
        # since main_iters last issue was already balanced by its wait).
        self.async_wait(0)
        for _ in gl.static_range(cfg.NUM_BUFFERS - 2):
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
        main_iters = gl.cdiv(loop_k, cfg.BLOCK_K) - (cfg.NUM_BUFFERS - 1)
        gl.assume(main_iters >= 0)

        # Wait so the oldest of the NB - 1 prologue batches is in LDS.
        # Followed by NB - 2 still in-flight; matches the per-iter wait
        # at the bottom of the loop.
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
    """v8-style 4-region split-shmem 2x2 M-N subtile interleave program.

    Mirrors ``v8_sliceMN`` from the gfx950-gluon-tutorials. The X tile
    is split along M into ``[BM/2, BK]`` halves (top / bot), W along N
    into ``[BK, BN/2]`` halves (left / right), each with its own LDS
    slot (4 ``allocate_shared_memory`` calls instead of 2). Each K-iter
    fires **4** ``buffer_load_to_shared`` commit_groups -- one per
    half-tile -- staggered across the 4-region MFMA pipeline. Two
    benefits stack:

      * **Half-tile fits cleanly under TCP/L1 32 KB cap.** Each
        ``buffer_load`` train transfers ~half of the full W tile, so a
        single train no longer saturates the cap on its own (same
        principle as :class:`MoESliceNProgram`, just extended to X).
      * **Per-region staggering hides HBM latency at large K.** The 4
        issues are spread across ~3 regions of MFMA (~1500 cycles)
        instead of clustered together. The earliest issue retires from
        the TCP before new ones arrive, resolving the large-K stall
        analysed in ``v8_sliceMN/README.md`` §4.5.

    The full ``[BLOCK_M, BLOCK_N]`` accumulator decomposes into four
    ``[BLOCK_M/2, BLOCK_N/2]`` sub-accumulators -- 4 sub-MFMAs
    interleaved with 4 sub-``ds_read``s per K iter. Each sub-MFMA
    depends on only one sub-X / sub-W, so the LLVM scheduler overlaps
    MFMA ``N`` with the ``ds_read`` of MFMA ``N+1`` -- a single-acc
    :class:`MoEPipelinedProgram` is held off by the long accumulator
    dep chain. Numerics: bit-exact against
    :class:`MoEPipelinedProgram` baseline (verified by
    ``test_slice_mn.py``).

    Constraints (caller / autotuner must enforce):

      * ``cfg.NUM_SUBTILES == (2, 2, 1)``.
      * ``cfg.NUM_BUFFERS >= 2`` (the 4-region prologue / wait sequence
        breaks at NB=1).
      * ``BLOCK_M >= 128`` / ``BLOCK_N >= 128`` (sub-tile floor from
        ``tiles_per_warp=[2,2] * warps_{m,n}=2`` = 64).
      * ``cfg.SCALE_VIA_LDS`` whenever ``cfg.WITH_*_MX_SCALE`` (sub-load
        slices the unswizzled scale view via
        ``issue_local_load_unswizzle_sub``).
    """

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
        """Sub-M X load. ``subtile_idx_m=0`` -> ``x_buffer_top`` (rows
        [0:BM/2)), ``=1`` -> ``x_buffer_bot`` (rows [BM/2:BM)). Each
        slot is already half-sized so no in-LDS slicing is needed.
        Scales are still loaded full-by-half via the existing
        unswizzle-sub path (the X-scale buffer is the full
        ``[BM_PRE, BK_S_PRE]`` allocation).
        """
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
        """Sub-N W load. ``subtile_idx_n=0`` -> ``w_buffer_left`` (cols
        [0:BN/2)), ``=1`` -> ``w_buffer_right`` (cols [BN/2:BN)).
        """
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
        """Region-0 issue: ``W_left`` only. 1 commit_group (~half W tile)."""
        self.w_desc_left.issue_async_load(
            load_idx, self.w_buffer_left, pred, USE_MASK=USE_MASK, COMMIT=1
        )
        return load_idx

    @gluon.jit
    def issue_x_top(self, load_idx, pred=1, USE_MASK: gl.constexpr = -1):
        """Region-1 issue: ``X_top`` + ``X_scale`` + ``W_scale`` folded
        into ONE commit_group. Scales are <= 4 KB combined and each
        lives in its own LDS slot, so packing them with X_top doesn't
        grow any in-flight train past the gfx950 TCP/L1 32 KB cap. The
        scales are issued by the time the next K-iter's Region 0
        consumes them (3 regions of slack at NB=2, 7 at NB=3).
        """
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
        """Region-2 issue: ``X_bot`` only. 1 commit_group."""
        self.x_desc_bot.issue_async_load(
            load_idx, self.x_buffer_bot, pred, USE_MASK=USE_MASK, COMMIT=1
        )
        return load_idx

    @gluon.jit
    def issue_w_right(self, load_idx, pred=1, USE_MASK: gl.constexpr = -1):
        """Region-3 issue: ``W_right`` only. 1 commit_group. Advances
        ``load_idx`` (= one K-iter consumed).
        """
        self.w_desc_right.issue_async_load(
            load_idx, self.w_buffer_right, pred, USE_MASK=USE_MASK, COMMIT=1
        )
        return load_idx + 1

    @gluon.jit
    def issue_global_loads(self, load_idx, pred=1, USE_MASK: gl.constexpr = -1):
        """Combined 4-region issue (= 4 commit_groups, 1 K-iter). Used
        in the prologue to spin up ``NB`` iters worth of in-flight
        commits before the main loop's per-region drain begins.
        """
        load_idx = self.issue_w_left(load_idx, pred, USE_MASK=USE_MASK)
        load_idx = self.issue_x_top(load_idx, pred, USE_MASK=USE_MASK)
        load_idx = self.issue_x_bot(load_idx, pred, USE_MASK=USE_MASK)
        load_idx = self.issue_w_right(load_idx, pred, USE_MASK=USE_MASK)
        return load_idx

    @gluon.jit
    def async_wait(self, waitcnt):
        """Coarse drain: 4 commit_groups per K-iter (one per region).
        Used by callers that batch wait at iter granularity; the main
        loop body uses raw ``async_copy.wait_group(4*NB - 3)`` for
        per-region one-commit drain.
        """
        gl.amd.cdna4.async_copy.wait_group(waitcnt * 4)

    @gluon.jit
    def pipeline(self, loop_k):
        """K-loop in v8_sliceMN's 4-region unrolled-by-2 form.

        Each main-loop body covers **2 K-iters** (= 8 regions). Within
        each K-iter we have 4 sub-MFMAs (one per quadrant) interleaved
        with 4 commit_groups (W_left, X_top + scales, X_bot, W_right).
        Each region: MFMA -> wait_group(drain 1) -> ds_read NEXT operand
        -> issue refill of CURRENT buffer.

        Per-region structure (mfma_idx = m, ahead = m + NB)::

            Region 0:  MFMA(x_top, w_left)   -> c_tl
                       wait_group(4*NB - 3)  # drain X_bot[m]
                       ds_read x_bot[m]
                       async_copy W_left[ahead]   (1 commit)

            Region 1:  MFMA(x_bot, w_left)   -> c_bl
                       wait_group(4*NB - 3)  # drain W_right[m]
                       ds_read w_right[m]
                       async_copy X_top + scales[ahead]  (1 commit)

            Region 2:  MFMA(x_top, w_right)  -> c_tr; mfma_idx++
                       wait_group(4*NB - 3)  # drain W_left[m+1]
                       ds_read w_left[m+1]
                       async_copy X_bot[ahead]   (1 commit)

            Region 3:  MFMA(x_bot, w_right)  -> c_br
                       wait_group(4*NB - 3)  # drain X_top[m+1]
                       ds_read x_top[m+1]
                       async_copy W_right[ahead]   (1 commit)

        With NB=2 the wait_group(5) keeps 6 commits in flight after
        each region (~ 1 1/2 iters ahead). Each region drains exactly
        ONE commit_group; 4 regions per iter aligns exactly with the
        per-iter 4-commit budget.

        Loop is unrolled by 2 so the LDS slot ping-pong (mfma_idx % NB)
        is statically resolved for NB=2 and the LLVM scheduler can
        keep all four operands (x_top, x_bot, w_left, w_right) alive
        concurrently across the iter boundary with minimal AGPR<->VGPR
        copies (see v8_sliceMN README §3.3).
        """
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

        # Prologue: issue NB iters worth (= 4*NB commit_groups). NB
        # iters not NB-1: the v8 4-region pattern needs both iter 0
        # AND iter 1 already in flight at main-loop entry so region 2
        # of iter 0 can ds_read iter 1 W_left (and region 3 ds_read
        # iter 1 X_top).
        for _ in gl.static_range(NB):
            load_idx = self.issue_global_loads(load_idx, USE_MASK=0)

        c_tl = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)
        c_bl = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)
        c_tr = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)
        c_br = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)

        K_iters = gl.cdiv(loop_k, cfg.BLOCK_K)
        if EVEN_K:
            main_iters = K_iters - NB
        else:
            main_iters = K_iters - NB - 1
        gl.assume(main_iters >= 2)

        # Drain iter 0's first 2 commits (W_left[0] + X_top[0]) so the
        # very first MFMA (Region 0: c_tl = mfma(x_top, w_left)) has
        # data. After this, in-flight = 4*NB - 2.
        gl.amd.cdna4.async_copy.wait_group(4 * NB - 2)
        x_top, sx_top = self.issue_local_load_x_sub(mfma_idx, 0)
        w_left, sw_left = self.issue_local_load_w_sub(mfma_idx, 0)

        unroll_pairs = main_iters // 2
        odd_main = main_iters - unroll_pairs * 2
        for _ in range(0, unroll_pairs):
            # ============================================================
            # iter k: 4 regions (consume buffer (m % NB), refill same)
            # ============================================================
            # Region 0: MFMA c_tl; drain X_bot[k]; ds_read; issue W_left[ahead]
            c_tl = self.mfma(x_top, sx_top, w_left, sw_left, c_tl)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            x_bot, sx_bot = self.issue_local_load_x_sub(mfma_idx, 1)
            load_idx = self.issue_w_left(load_idx, USE_MASK=0)

            # Region 1: MFMA c_bl; drain W_right[k]; ds_read; issue X_top+scales[ahead]
            c_bl = self.mfma(x_bot, sx_bot, w_left, sw_left, c_bl)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            w_right, sw_right = self.issue_local_load_w_sub(mfma_idx, 1)
            load_idx = self.issue_x_top(load_idx, USE_MASK=0)

            # Region 2: MFMA c_tr; mfma_idx++; drain W_left[k+1]; ds_read; issue X_bot[ahead]
            c_tr = self.mfma(x_top, sx_top, w_right, sw_right, c_tr)
            mfma_idx += 1
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            w_left, sw_left = self.issue_local_load_w_sub(mfma_idx, 0)
            load_idx = self.issue_x_bot(load_idx, USE_MASK=0)

            # Region 3: MFMA c_br; drain X_top[k+1]; ds_read; issue W_right[ahead]
            c_br = self.mfma(x_bot, sx_bot, w_right, sw_right, c_br)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            x_top, sx_top = self.issue_local_load_x_sub(mfma_idx, 0)
            load_idx = self.issue_w_right(load_idx, USE_MASK=0)

            # ============================================================
            # iter k+1: same 4 regions, ping-ponged buffer slot
            # ============================================================
            c_tl = self.mfma(x_top, sx_top, w_left, sw_left, c_tl)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            x_bot, sx_bot = self.issue_local_load_x_sub(mfma_idx, 1)
            load_idx = self.issue_w_left(load_idx, USE_MASK=0)

            c_bl = self.mfma(x_bot, sx_bot, w_left, sw_left, c_bl)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            w_right, sw_right = self.issue_local_load_w_sub(mfma_idx, 1)
            load_idx = self.issue_x_top(load_idx, USE_MASK=0)

            c_tr = self.mfma(x_top, sx_top, w_right, sw_right, c_tr)
            mfma_idx += 1
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            w_left, sw_left = self.issue_local_load_w_sub(mfma_idx, 0)
            load_idx = self.issue_x_bot(load_idx, USE_MASK=0)

            c_br = self.mfma(x_bot, sx_bot, w_right, sw_right, c_br)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            x_top, sx_top = self.issue_local_load_x_sub(mfma_idx, 0)
            load_idx = self.issue_w_right(load_idx, USE_MASK=0)

        # Odd-iter peel: 4 regions if main_iters is odd.
        if odd_main:
            c_tl = self.mfma(x_top, sx_top, w_left, sw_left, c_tl)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            x_bot, sx_bot = self.issue_local_load_x_sub(mfma_idx, 1)
            load_idx = self.issue_w_left(load_idx, USE_MASK=0)

            c_bl = self.mfma(x_bot, sx_bot, w_left, sw_left, c_bl)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            w_right, sw_right = self.issue_local_load_w_sub(mfma_idx, 1)
            load_idx = self.issue_x_top(load_idx, USE_MASK=0)

            c_tr = self.mfma(x_top, sx_top, w_right, sw_right, c_tr)
            mfma_idx += 1
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            w_left, sw_left = self.issue_local_load_w_sub(mfma_idx, 0)
            load_idx = self.issue_x_bot(load_idx, USE_MASK=0)

            c_br = self.mfma(x_bot, sx_bot, w_right, sw_right, c_br)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            x_top, sx_top = self.issue_local_load_x_sub(mfma_idx, 0)
            load_idx = self.issue_w_right(load_idx, USE_MASK=0)

        # K-tail peel for not-EVEN_K: 4 regions with USE_MASK=1.
        if not EVEN_K:
            c_tl = self.mfma(x_top, sx_top, w_left, sw_left, c_tl)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            x_bot, sx_bot = self.issue_local_load_x_sub(mfma_idx, 1)
            load_idx = self.issue_w_left(load_idx, USE_MASK=1)

            c_bl = self.mfma(x_bot, sx_bot, w_left, sw_left, c_bl)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            w_right, sw_right = self.issue_local_load_w_sub(mfma_idx, 1)
            load_idx = self.issue_x_top(load_idx, USE_MASK=1)

            c_tr = self.mfma(x_top, sx_top, w_right, sw_right, c_tr)
            mfma_idx += 1
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            w_left, sw_left = self.issue_local_load_w_sub(mfma_idx, 0)
            load_idx = self.issue_x_bot(load_idx, USE_MASK=1)

            c_br = self.mfma(x_bot, sx_bot, w_right, sw_right, c_br)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            x_top, sx_top = self.issue_local_load_x_sub(mfma_idx, 0)
            load_idx = self.issue_w_right(load_idx, USE_MASK=1)

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

        # Stitch: 4 quadrants -> [BM, BN] via join along quadrant axis,
        # permute, reshape.
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
    """N-only subtile interleave variant of :class:`MoEPipelinedProgram`.

    Splits the ``[BLOCK_M, BLOCK_N]`` accumulator into 2
    ``[BLOCK_M, BLOCK_N/2]`` sub-accumulators and emits 2 sub-MFMAs per
    K-loop iter interleaved with 2 sub-W ``ds_read``s + 1 full-X
    ``ds_read``. Same MFMA-vs-``ds_read`` overlap idea as the 2x2
    :class:`MoESliceMNProgram` but keeps the M dim intact, so it works
    on small-BM decode tiles (BM=16/32) where the 2x2 split is
    structurally illegal (the ``BM/2`` sub-tile would fall below the
    64-element warp-tile floor).

    **W is split-shmem'd**: each half of N (``[0:BN/2]`` and
    ``[BN/2:BN]``) gets its own LDS buffer (``w_buffer_top`` /
    ``w_buffer_bot``) and its own :class:`AsyncCopyDescriptor`
    (``w_desc_top`` / ``w_desc_bot``). The K-loop body issues each W
    half via a separate ``buffer_load_to_shared`` train (~16 KB
    each), so a single K-iter no longer fires a 32 KB transfer that
    would saturate the gfx950 TCP/L1 32 KB in-flight cap. Pattern
    mirrors v7_sliceN in ``gfx950-gluon-tutorials/kernels/gemm/a16w16``.

    Compared to plain :class:`MoEPipelinedProgram`:

      * 2 sub-MFMAs per iter instead of 1 full MFMA; each sub-MFMA's
        accumulator is half-width along N, so its dep chain is
        shorter and the LLVM scheduler can interleave the sub-W
        ``ds_read`` (next iter's sub-W) with the in-flight sub-MFMA.
      * 5 commit_groups per iter instead of 4 (extra W half).
        ``async_wait_split`` accounts for that.
      * X / X-scale / W-scale stay whole (X is small, W-scale is
        small AND its LDS unswizzle slices in-place via
        ``issue_local_load_unswizzle_sub``).

    Constraints (caller / autotuner must enforce):

      * ``cfg.NUM_SUBTILES == (1, 2, 1)``.
      * ``cfg.SCALE_VIA_LDS`` whenever ``cfg.WITH_W_MX_SCALE``: the
        sub-W-scale load slices the unswizzled view; the G->VGPR
        path doesn't yet have per-subtile global addressing.
      * ``BLOCK_N >= 128`` (sub-N >= 64 = warp-tile floor with
        ``warps_n = 2``, ``tiles_per_warp = 2``).
      * ``BLOCK_M >= 16`` (no constraint from N-only slicing).

    Numerics: bit-exact against :class:`MoEPipelinedProgram` baseline
    when both run on the same tile + scale-load mode (verified by
    ``test_slice_n.py``).
    """

    base: MoEProgramBase
    cfg: MoEConfig
    x_buffer: gl.shared_memory_descriptor
    w_buffer_top: gl.shared_memory_descriptor
    w_buffer_bot: gl.shared_memory_descriptor
    x_scale_buffer: gl.shared_memory_descriptor | gl.constexpr
    w_scale_buffer: gl.shared_memory_descriptor | gl.constexpr
    x_desc: AsyncCopyDescriptor
    w_desc_top: AsyncCopyDescriptor
    w_desc_bot: AsyncCopyDescriptor
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
        self.w_buffer_top = w_buffer_top
        self.w_buffer_bot = w_buffer_bot
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
        # X / X-scale / W-scale buffers identical to
        # :class:`MoEPipelinedProgram`. The
        # **W** buffer is split: two ``[NB, BN/2, BK_packed]`` slots,
        # one per N half. Each sub-G->LDS train hits its own LDS slot,
        # which lets the K-loop body issue them as independent
        # ``buffer_load_to_shared`` commit_groups so each train is ~16
        # KB rather than 32 KB (= TCP/L1 in-flight cap on gfx9).
        NUM_BUFFERS: gl.constexpr = cfg.NUM_BUFFERS
        BLOCK_K_PACKED_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
        BLOCK_K_PACKED_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W

        x_buffer = gl.allocate_shared_memory(
            x_desc.dtype,
            shape=[NUM_BUFFERS, cfg.BLOCK_M, BLOCK_K_PACKED_X],
            layout=cfg.shared_layout_x,
        )
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
        """Full-M X load + X-scale load. Same shape as
        :meth:`MoEPipelinedProgram.issue_local_loads` but returns just
        the X half (W is sliced per sub-MFMA via
        :meth:`issue_local_load_w_sub`).

        ``cfg.layout_x_scale`` is sized ``[BLOCK_M, BLOCK_K_SCALE]``
        because ``NUM_SUBTILES_M == 1`` -- no sub-tile slicing on M.
        """
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
                    scale_x = _load_scale_tile_via_gl_load(
                        self.x_scale_desc, mfma_idx, cfg.layout_x_scale
                    )
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
        """Issue the *top* half of a K-iter's loads as ONE commit_group::

            X         + W_top + X_scale + W_scale     (1 commit, ~22 KB)

        Together with :meth:`issue_global_load_bot` (which fires
        ``W_bot`` as a second commit_group), each K-iter produces
        exactly **2 commit_groups** -- the bookkeeping that lets the
        v7-style 4-region main loop drain one commit per region via
        ``wait_group(2*NB - 2)``.

        ``X / X_scale / W_scale`` are folded into the same commit (not
        split): they're individually small (<= 4 KB each) and live in
        separate LDS slots, so packing them into one commit_group
        doesn't grow any single in-flight train past the gfx950 TCP/L1
        32 KB cap. Splitting them further would only multiply
        ``commit_group`` overhead.

        Returns ``load_idx`` unchanged -- :meth:`issue_global_load_bot`
        is what advances the iter counter.
        """
        cfg = self.cfg
        self.x_desc.issue_async_load(
            load_idx, self.x_buffer, pred, USE_MASK=USE_MASK, COMMIT=0
        )
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
        """Issue the *bot* half (``W_bot``) of a K-iter's loads as
        ONE commit_group. See :meth:`issue_global_load_top`.
        """
        self.w_desc_bot.issue_async_load(
            load_idx, self.w_buffer_bot, pred, USE_MASK=USE_MASK, COMMIT=1
        )
        return load_idx + 1

    @gluon.jit
    def issue_global_loads(self, load_idx, pred=1, USE_MASK: gl.constexpr = -1):
        """Combined top + bot issue for prologue / single-iter peel.
        Produces 2 commit_groups (= 1 K-iter).
        """
        load_idx = self.issue_global_load_top(load_idx, pred, USE_MASK=USE_MASK)
        load_idx = self.issue_global_load_bot(load_idx, pred, USE_MASK=USE_MASK)
        return load_idx

    @gluon.jit
    def async_wait(self, waitcnt):
        """SliceN-aware ``async_wait``: 2 commit_groups per K-iter (top
        + bot). ``waitcnt`` semantics match
        :meth:`MoEProgramBase.async_wait` (= number of full K-iter
        batches still in-flight). Used for the prologue's coarse
        per-iter drain; the main loop uses raw
        ``async_copy.wait_group(2*NB - 2)`` for per-region one-commit
        drain.
        """
        gl.amd.cdna4.async_copy.wait_group(waitcnt * 2)

    @gluon.jit
    def issue_local_load_w_sub(self, mfma_idx, subtile_idx_n: gl.constexpr):
        """Sub-N W load. Picks the per-half LDS slot by
        ``subtile_idx_n``: ``0`` -> ``w_buffer_top`` (rows [0:BN/2)),
        ``1`` -> ``w_buffer_bot`` (rows [BN/2:BN)). The slot is
        already half-sized, so no in-LDS slicing is needed.
        """
        cfg = self.cfg
        SUBTILE_N: gl.constexpr = cfg.BLOCK_N // cfg.NUM_SUBTILES[1]
        subtile_start_n: gl.constexpr = subtile_idx_n * SUBTILE_N
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK

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
        """K-loop in v7_sliceN's 4-region unrolled-by-2 form.

        Each main-loop body covers **2 K-iters** (= 4 regions). Within
        each K-iter we have 2 sub-MFMAs (top, bot) interleaved with 2
        commit_groups (top half, bot half), and the issue of the next
        iter's halves is staggered ACROSS the 2 sub-MFMAs of the
        current iter so MFMA + ds_read sit between the two
        ``buffer_load_to_shared`` trains. That gap is what lets the
        gfx950 TCP/L1 in-flight cap (32 KB) drain between the two W
        halves -- back-to-back issues would still saturate it. Mirrors
        ``v7_sliceN`` in ``gfx950-gluon-tutorials``.

        Per-region structure (mfma_idx = m, ahead = m+NB)::

            Region 0:  MFMA[0](x[m], w_top[m]) -> c0
                       wait_group(2*NB-2)   # drain w_bot[m]
                       ds_read   w_bot[m]
                       async_copy iter[ahead] top  (1 commit)

            Region 1:  MFMA[1](x[m], w_bot[m]) -> c1
                       wait_group(2*NB-2)   # drain x[m+1]/w_top[m+1]
                       ds_read   x[m+1], w_top[m+1]
                       async_copy iter[ahead] bot  (1 commit)

        With NB=2 the wait_group(2) keeps 2 commits in flight after
        each region (= 1 iter ahead). Each region drains exactly ONE
        commit_group, so the per-iter 2-commit budget aligns exactly
        with the 2 regions per iter.

        Loop is unrolled by 2 so the 4 regions form one body and the
        ping-pong of LDS slots (mfma_idx % NB) and per-half register
        names (a/b_left vs a_next/b_left) is explicit -- the LLVM
        scheduler can then keep a, b_left, b_right alive concurrently
        across the boundary with minimal AGPR<->VGPR copies (see v7
        README §4.3).
        """
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

        # Prologue: issue NB iters worth (= 2*NB commit_groups).
        # NB iters not NB-1: the v7 4-region pattern needs both
        # iter 0 AND iter 1 already in flight at main-loop entry so
        # region 1 of iter 0 can ds_read iter 1 X/W_top.
        for _ in gl.static_range(NB):
            load_idx = self.issue_global_loads(load_idx, USE_MASK=0)

        c0 = gl.zeros((cfg.BLOCK_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)
        c1 = gl.zeros((cfg.BLOCK_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)

        K_iters = gl.cdiv(loop_k, cfg.BLOCK_K)
        # Number of K-iters whose loads are still pending issuance
        # *during* the main loop. The last one (= iter K_iters-1)
        # is peeled to a tail iter when not EVEN_K so its USE_MASK=1
        # cost stays out of the hot body.
        if EVEN_K:
            main_iters = K_iters - NB
        else:
            main_iters = K_iters - NB - 1
        gl.assume(main_iters >= 2)

        # Drain iter 0's top half so the first ds_read X/W_top has
        # data. After this, in-flight = 2*NB - 1 commits.
        gl.amd.cdna4.async_copy.wait_group(2 * NB - 1)
        x, sx = self.issue_local_load_x(mfma_idx)
        w0, sw0 = self.issue_local_load_w_sub(mfma_idx, 0)

        # Even portion of main loop: unroll by 2 (4 regions / body).
        # ``main_iters`` is a runtime int -- unroll structure is a
        # regular ``range`` with step 1 over (main_iters // 2) pairs,
        # plus a post-loop odd-iter peel below for the case
        # main_iters % 2 == 1.
        unroll_pairs = main_iters // 2
        odd_main = main_iters - unroll_pairs * 2
        for _ in range(0, unroll_pairs):
            # ============================================================
            # iter k: regions 0 + 1
            # ============================================================
            # Region 0: MFMA[0]; drain W_bot[k]; ds_read W_bot[k]; issue
            # top of (k+NB)
            c0 = self.mfma(x, sx, w0, sw0, c0)
            gl.amd.cdna4.async_copy.wait_group(2 * NB - 2)
            w1, sw1 = self.issue_local_load_w_sub(mfma_idx, 1)
            load_idx = self.issue_global_load_top(load_idx, USE_MASK=0)

            # Region 1: MFMA[1]; mfma_idx++; drain X/W_top[k+1]; ds_read
            # X/W_top[k+1]; issue bot of (k+NB)
            c1 = self.mfma(x, sx, w1, sw1, c1)
            mfma_idx += 1
            gl.amd.cdna4.async_copy.wait_group(2 * NB - 2)
            x, sx = self.issue_local_load_x(mfma_idx)
            w0, sw0 = self.issue_local_load_w_sub(mfma_idx, 0)
            load_idx = self.issue_global_load_bot(load_idx, USE_MASK=0)

            # ============================================================
            # iter k+1: regions 2 + 3 (== regions 0 + 1 with ping-ponged
            # LDS slot via mfma_idx odd vs even)
            # ============================================================
            c0 = self.mfma(x, sx, w0, sw0, c0)
            gl.amd.cdna4.async_copy.wait_group(2 * NB - 2)
            w1, sw1 = self.issue_local_load_w_sub(mfma_idx, 1)
            load_idx = self.issue_global_load_top(load_idx, USE_MASK=0)

            c1 = self.mfma(x, sx, w1, sw1, c1)
            mfma_idx += 1
            gl.amd.cdna4.async_copy.wait_group(2 * NB - 2)
            x, sx = self.issue_local_load_x(mfma_idx)
            w0, sw0 = self.issue_local_load_w_sub(mfma_idx, 0)
            load_idx = self.issue_global_load_bot(load_idx, USE_MASK=0)

        # Odd-iter peel: 2 regions if main_iters is odd.
        if odd_main:
            c0 = self.mfma(x, sx, w0, sw0, c0)
            gl.amd.cdna4.async_copy.wait_group(2 * NB - 2)
            w1, sw1 = self.issue_local_load_w_sub(mfma_idx, 1)
            load_idx = self.issue_global_load_top(load_idx, USE_MASK=0)

            c1 = self.mfma(x, sx, w1, sw1, c1)
            mfma_idx += 1
            gl.amd.cdna4.async_copy.wait_group(2 * NB - 2)
            x, sx = self.issue_local_load_x(mfma_idx)
            w0, sw0 = self.issue_local_load_w_sub(mfma_idx, 0)
            load_idx = self.issue_global_load_bot(load_idx, USE_MASK=0)

        # K-tail peel for not-EVEN_K: same 2-region body but the
        # issued iter (K_iters - 1) uses USE_MASK=1 so the per-element
        # K-mask is paid only once (instead of every main-loop iter).
        if not EVEN_K:
            c0 = self.mfma(x, sx, w0, sw0, c0)
            gl.amd.cdna4.async_copy.wait_group(2 * NB - 2)
            w1, sw1 = self.issue_local_load_w_sub(mfma_idx, 1)
            load_idx = self.issue_global_load_top(load_idx, USE_MASK=1)

            c1 = self.mfma(x, sx, w1, sw1, c1)
            mfma_idx += 1
            gl.amd.cdna4.async_copy.wait_group(2 * NB - 2)
            x, sx = self.issue_local_load_x(mfma_idx)
            w0, sw0 = self.issue_local_load_w_sub(mfma_idx, 0)
            load_idx = self.issue_global_load_bot(load_idx, USE_MASK=1)

        # Epilogue: drain everything, then run NB iters' MFMAs without
        # any further ``async_copy`` issues (all loads have already
        # been emitted). Mirrors v7's "iterMax-2 / iterMax-1" tail.
        gl.amd.cdna4.async_copy.wait_group(0)
        for i in gl.static_range(NB):
            c0 = self.mfma(x, sx, w0, sw0, c0)
            w1, sw1 = self.issue_local_load_w_sub(mfma_idx, 1)

            c1 = self.mfma(x, sx, w1, sw1, c1)
            mfma_idx += 1

            if i < NB - 1:
                x, sx = self.issue_local_load_x(mfma_idx)
                w0, sw0 = self.issue_local_load_w_sub(mfma_idx, 0)

        # Stitch: c0 is [BM, BN/2], c1 is [BM, BN/2] -- interleave along N.
        # ``join`` produces [BM, BN/2, 2]; ``permute(0, 2, 1)`` then
        # ``reshape((BM, BN))`` gives the expected logical
        # [BM, n0_0, n0_1, ..., n1_0, n1_1, ...] order. Final
        # ``convert_layout`` matches the downstream acc_layout.
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
    USE_SLICE_MN: gl.constexpr = False,
    USE_SLICE_N: gl.constexpr = False,
    USE_3STAGE_PIPELINE: gl.constexpr = False,
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
        # Clamp rows_m to a safe in-bounds value (0) on OOB lanes so
        # the AsyncCopyDescriptor's per-lane pointer arithmetic
        # ``x_ptr + rows_m * stride_xm`` stays inside the x allocation
        # even when ``off_m + BLOCK_M > m_limit`` (per-expert size <
        # BLOCK_M case in the ragged combine GEMM). The mask still
        # filters out the bogus rows; we just avoid handing the
        # AMD/HIP buffer-load path an OOB address (which has been
        # observed to fault with "Memory access fault by GPU node"
        # during cuda graph capture warm-up).
        rows_m = gl.where(pre_gather_mask, rows_m, gl.zeros_like(rows_m))
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
            # swizzle: the in-tile K-axis is packed with the N-axis,
            # so a K-mask on the packed column would scramble both. We rely
            # on the host-side padding (e8m0=0 for OOB microblocks)
            # combined with the W operand's own K-mask which zeros the OOB
            # W elements -- making the OOB scale-x weight contribution 0
            # regardless of the loaded scale value. Suppress the kernel-
            # side K-mask by setting k_limit = K_S_pad * PF (full post-pad
            # storage extent), which is guaranteed >= every loaded offset.
            # K is a runtime arg so we cannot use constexpr ternaries here.
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
            # See x_scale comment above for the rationale.
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
        if W_TRANSPOSE:
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
            )
        else:
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
        elif USE_3STAGE_PIPELINE:
            acc = pgm.pipeline_3stage(K)
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
    else:
        out = acc

    out = out.to(y_ptr.dtype.element_ty)
    out = gl.convert_layout(out, STORE)

    offs_y_m = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, STORE))
    off_n_out = pid_n * OUT_BLOCK_N
    offs_y_n = off_n_out + gl.arange(0, OUT_BLOCK_N, gl.SliceLayout(0, STORE))

    # Bound ``offs_y_m`` BEFORE using it as a pointer offset. ``gate_scal``
    # and ``scatter_idx`` are sized to ``M`` (= sum(slice_sizes)), but
    # ``offs_y_m`` can extend past the per-expert tail (``m_limit < off_m
    # + BLOCK_M``) AND past the global ``M`` (when per-expert size <
    # BLOCK_M in the ragged schedule). Even with masked ``gl.load``,
    # AMD/HIP can surface a "Memory access fault by GPU node" on the
    # masked-off lanes because pointer arithmetic happens before the
    # predicate. Mirror the ``HAS_GATHER`` branch above which already
    # does this clamp pattern.
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


# ---------------------------------------------------------------------------
# Grid-walk swizzles (Update 9 / 10): XCD chiplet swizzle + GROUP_M for L2 reuse
# ---------------------------------------------------------------------------


@gluon.jit
def _xcd_chiplet_swizzle(pid, num_pids, XCD_SWIZZLE: gl.constexpr):
    """Reorder a linear ``pid`` so consecutive original pids land on the
    same XCD (chiplet).

    MI355X has 8 XCDs that the hardware scheduler round-robins
    (sequential pid 0,1,2,...,7 → XCD 0,1,2,...,7). That spreads
    sequential tiles across DIFFERENT L2 partitions, costing W/X
    re-fetches. With ``XCD_SWIZZLE = N`` we permute the linear domain
    so that pids ``[0..pids_per_xcd)`` go to XCD 0, ``[pids_per_xcd..2
    *pids_per_xcd)`` to XCD 1, etc.; combined with the GROUP_M walk
    that follows, consecutive original pids land on the same XCD and
    reuse the same N-tile of W in that XCD's L2.

    Mirrors ``triton_kernels`` ``xcd_swizzle`` in
    ``matmul_details/_common.py``.
    """
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
    """Map a linear ``pid_mn`` (= row-major over ``grid_m * grid_n``)
    to ``(pid_m, pid_n)`` so that ``GROUP_M`` consecutive M-tiles share
    the same N-tile of W.

    For ``GROUP_M == 1`` this is the trivial row-major mapping
    ``(pid_mn // grid_n, pid_mn % grid_n)``. Boundary tiles where the
    last M-group is shorter than ``GROUP_M`` use the ``group_size`` =
    ``min(grid_m - group_id*GROUP_M, GROUP_M)`` formula from
    ``triton_kernels`` ``swizzle2d``.
    """
    if GROUP_M == 1:
        pid_m = pid_mn // grid_n
        pid_n = pid_mn % grid_n
    else:
        width = GROUP_M * grid_n
        group_id = pid_mn // width
        group_size = gl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        # NB: ``(pid_mn % width) % group_size`` -- the modulo against
        # ``width`` is REQUIRED, not redundant. Without it, the formula
        # mixes the intra-group index with the group_id and produces
        # ``pid_m >= grid_m`` for tiles in any group past the first,
        # which then dereferences out-of-bounds pointers (GPU memfault).
        intra = pid_mn % width
        pid_m = group_id * GROUP_M + (intra % group_size)
        pid_n = intra // group_size
    return pid_m, pid_n


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
    USE_SLICE_MN: gl.constexpr = False,
    USE_SLICE_N: gl.constexpr = False,
    USE_3STAGE_PIPELINE: gl.constexpr = False,
    GRID_N: gl.constexpr = 0,
    GROUP_M: gl.constexpr = 1,
    XCD_SWIZZLE: gl.constexpr = 1,
):
    """Non-persistent MoE GEMM kernel: legacy 2-D grid, one tile per CTA.

    The kernel is straight-line -- no constexpr-collapsed for-loop, no
    schedule-padding ``do_tile`` guard around the body -- so LLVM sees
    a single basic block per launch and register-allocates accordingly.
    Iterative launch flavours (persistent / block-schedule) live in
    :mod:`.gluon_persistent` to keep this entry point lean.

    Grid: ``(blocks_per_expert * grid_n, num_active)``.
      * ``program_id(0)`` is the intra-(expert) tile id.
      * ``program_id(1)`` is the compact expert id (in
        ``[0, num_active)`` -- ``HAS_EXPERT_REMAP`` then maps to the
        real expert id inside :func:`_pipelined_moe_tile_compute`).

    ``GRID_N`` is passed as constexpr (``= ceil_div(N, BLOCK_N)``) so
    ``tiles_per_expert`` lowers to a compile-time constant; ``GRID_N=0``
    falls back to the runtime compute for legacy call sites.

    ``GROUP_M`` / ``XCD_SWIZZLE`` are launcher-side knobs that re-order
    ``tile_idx`` for L2 reuse and XCD locality respectively; both
    default to no-op (1).
    """
    if GRID_N > 0:
        grid_n: gl.constexpr = GRID_N
        tiles_per_expert: gl.constexpr = BLOCKS_PER_EXPERT * GRID_N
    else:
        grid_n = (N + BLOCK_N - 1) // BLOCK_N
        tiles_per_expert = BLOCKS_PER_EXPERT * grid_n

    # tile_idx packs (compact_idx, intra-expert pid). XCD swizzle on the
    # FULL domain so the permutation crosses experts; GROUP_M then
    # re-orders within an expert's (BLOCKS_PER_EXPERT, grid_n) sub-grid
    # for W-tile L2 reuse.
    tile_idx = gl.program_id(1) * tiles_per_expert + gl.program_id(0)
    swizzled = _xcd_chiplet_swizzle(tile_idx, NUM_TILES, XCD_SWIZZLE)
    compact_idx = swizzled // tiles_per_expert
    local = swizzled % tiles_per_expert
    block_in_expert, pid_n = _group_m_swizzle(local, BLOCKS_PER_EXPERT, grid_n, GROUP_M)

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
        USE_SLICE_MN=USE_SLICE_MN,
        USE_SLICE_N=USE_SLICE_N,
        USE_3STAGE_PIPELINE=USE_3STAGE_PIPELINE,
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


_LAST_KERNEL_PROFILE: dict | None = None
_PROFILE_BY_KERNEL_ID: dict[int, dict] = {}


def _capture_launch_profile(k: Any) -> None:
    """Internal: snapshot the static GPR/occupancy profile of ``k``."""
    global _LAST_KERNEL_PROFILE
    key = id(k)
    prof = _PROFILE_BY_KERNEL_ID.get(key)
    if prof is None:
        prof = static_profile(k)
        # Capture the underlying kernel function name (e.g.
        # ``_pipelined_moe_kernel_scaled`` vs
        # ``_pipelined_moe_kernel_scaled_persistent``) so callers that
        # bypass autotune via ``--tiles``/``--persistent`` can confirm
        # which kernel function actually got launched without grepping
        # the IR dump. ``CompiledKernel`` exposes the function name via
        # ``.name`` (Triton) or as a fallback through ``.metadata.name``;
        # be defensive so a future Triton bump doesn't break the bench.
        name = getattr(k, "name", None)
        if name is None:
            md = getattr(k, "metadata", None)
            name = getattr(md, "name", None) if md is not None else None
        if name is not None:
            prof["kernel_name"] = str(name)
        _PROFILE_BY_KERNEL_ID[key] = prof
    _LAST_KERNEL_PROFILE = prof


def last_kernel_profile() -> dict | None:
    """Return the static profile (sgpr/vgpr/scratch/occupancy) of the
    most recent :func:`_launch_kernel` invocation, or ``None`` if no MoE
    launch has happened yet on this process.
    """
    return _LAST_KERNEL_PROFILE


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
    consumed zero-copy by the ``swizzle`` mode.
    """
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


def _is_scale_swizzled_cdna4(s: torch.Tensor) -> bool:
    """Heuristic: a tensor is already swizzled iff its inner 2-D has
    ``stride(-2) == 1`` (the contiguous K_S*32 axis) and
    ``shape[-2] % 32 == 0`` and ``shape[-1] % 1 == 0`` etc. We take
    the cheap path and just check ``stride(-2) == 1`` since the upstream
    swizzle asserts that explicitly."""
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
    """Return ``(stride_nonk, stride_k)`` matching the kernel's address math
    ``base + r * stride_nonk + c * stride_k`` where ``r`` ranges over the
    NONK axis (M for x_scale, N for w_scale, possibly preshuffled) and
    ``c`` ranges over the K_S axis (possibly *PF post-swizzle).

    For ``bypass`` / ``transpose`` the gluon convention is
    ``[..., NONK, K_S]`` with K_S contiguous, so we return the natural
    ``(stride(-2), stride(-1))``.

    For ``swizzle`` the post-swizzle storage shape is
    ``[..., K_SCALE_pad*32, N_pad/32]`` with ``stride(-2)==1`` (the
    K_SCALE_pad*32 axis contiguous). The kernel still treats it as
    ``[..., NONK_PS=N/32, K_S_PS=K_S*32]`` so we swap and return
    ``(stride(-1), stride(-2))``.
    """
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
    a_format: str,
    b_format: str = "e2m1",
    x_scale: torch.Tensor | None = None,
    w_scale: torch.Tensor | None = None,
    a_global_scale: torch.Tensor | float | None = 1.0,
    scale_load_mode: str = "bypass",
    w_transpose: bool = False,
    apply_x_global_scale: bool | None = None,
    use_warp_pipeline: bool = False,
    use_slice_mn: bool = False,
    use_slice_n: bool = False,
    use_3stage_pipeline: bool | None = None,
    persistent: bool | None = None,
    num_ctas: int | None = None,
    group_m: int | None = None,
    xcd_swizzle: int | None = None,
):
    # Scaled-MFMA-only launcher (mxfp4 / fp8 x mxfp4). ``a_format`` and
    # ``b_format`` are required and must be in ``_SCALED_FORMATS``.
    assert a_format in _SCALED_FORMATS, f"unknown a_format={a_format!r}"
    assert b_format in _SCALED_FORMATS, f"unknown b_format={b_format!r}"
    if apply_x_global_scale is None:
        apply_x_global_scale = True
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

    assert (
        block_k % _MFMA_SCALED_K == 0
    ), f"BLOCK_K={block_k} must be a multiple of MFMA K dim ({_MFMA_SCALED_K})"
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
        # DEBUG (set via env): dump kernel launch parameters once per
        # unique (M, block_m, has_gather, has_scatter) signature so we
        # can correlate runtime shapes with crashes.
        if os.environ.get("TOKENSPEED_MOE_GLUON_LAUNCH_DEBUG", "").strip() in {
            "1",
            "true",
        }:
            # Avoid D2H sync (e.g. .item()/.tolist()): unsafe inside
            # cuda graph capture. Print purely host-side metadata.
            _key = (
                M,
                block_m,
                block_n,
                gather_indx is not None,
                scatter_indx is not None,
                int(a_ragged_metadata.slice_sizes.shape[0]),
                int(block_schedule_buf.shape[-1]),
                grid_m_upper,
                tuple(x.shape),
                x.stride(),
            )
            _seen = _launch_kernel.__dict__.setdefault("_seen_keys", set())
            if _key not in _seen:
                _seen.add(_key)
                import sys as _sys

                _scatter_shape = (
                    tuple(scatter_indx.dst_indx.shape)
                    if scatter_indx is not None
                    else None
                )
                _gather_shape = (
                    tuple(gather_indx.src_indx.shape)
                    if gather_indx is not None
                    else None
                )
                _sys.stderr.write(
                    f"[gluon-launch] M={M} M_X={M_X} block_m={block_m} "
                    f"block_n={block_n} block_k={block_k} "
                    f"grid_m_upper={grid_m_upper} grid_n={grid_n} "
                    f"n_slices={n_slices} "
                    f"sched.shape={tuple(block_schedule_buf.shape)} "
                    f"x.shape={tuple(x.shape)} x.stride={x.stride()} "
                    f"x.dtype={x.dtype} y.shape={tuple(y.shape)} "
                    f"y.stride={y.stride()} y.dtype={y.dtype} "
                    f"gather.shape={_gather_shape} "
                    f"scatter.shape={_scatter_shape} "
                    f"slice_offs.shape={tuple(a_ragged_metadata.slice_offs.shape)}\n"
                )
                _sys.stderr.flush()
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

    # GROUP_M + XCD_SWIZZLE grid-walk permutation. Default picked by
    # ``_autotune_pid_swizzle`` (no-op on decode / small-tile shapes,
    # ``(GROUP_M=8|4|2, XCD_SWIZZLE=8)`` on prefill). Env overrides
    # ``TOKENSPEED_MOE_GLUON_GROUP_M`` / ``..._XCD_SWIZZLE`` win over
    # both the heuristic and the caller's kwargs -- match the rest of
    # the gluon tuning knobs (env always wins for closed-loop sweeps).
    grid_m_for_swizzle = (
        (num_tiles_total // grid_n) if use_block_schedule else blocks_per_expert
    )
    auto_group_m, auto_xcd = _autotune_pid_swizzle(
        num_tiles_total=num_tiles_total,
        grid_n=grid_n,
        grid_m_padded=grid_m_for_swizzle,
        block_m=block_m,
    )
    env_group_m = _env_int("TOKENSPEED_MOE_GLUON_GROUP_M", -1)
    env_xcd = _env_int("TOKENSPEED_MOE_GLUON_XCD_SWIZZLE", -1)
    if env_group_m >= 1:
        group_m = env_group_m
    elif group_m is None:
        group_m = auto_group_m
    if env_xcd >= 1:
        xcd_swizzle = env_xcd
    elif xcd_swizzle is None:
        xcd_swizzle = auto_xcd
    # Clamp to legal values: GROUP_M must divide grid_m_padded (or 1) so
    # the swizzle stays a permutation; otherwise fall back to 1.
    if group_m > 1 and grid_m_for_swizzle % group_m != 0:
        group_m = 1
    # XCD_SWIZZLE must divide the 1-D domain it operates on; otherwise
    # ``_xcd_chiplet_swizzle`` is identity, so a non-divisor would be a
    # silent no-op. Clamp to 1 for clarity.
    if xcd_swizzle > 1 and num_tiles_total % xcd_swizzle != 0:
        xcd_swizzle = 1

    # 3-stage software pipeline. Resolution order:
    #   env TOKENSPEED_MOE_GLUON_3STAGE > caller kwarg > default(False).
    # The 3-stage variant trades +1 tile of live VGPR for hiding the
    # ``ds_read`` latency in the MFMA critical path. Empirically on
    # MI355 the carry cost is too high to be a net win on the current
    # tile menu (BM=32 fp8 decode: 136 -> 246 VGPRs, occ 3 -> 2,
    # +6-9% on dispatch / +5-12% on combine). The method is kept and
    # plumbed end-to-end so future tile shapes (smaller BK or
    # BM=16) can opt in via env without code edits. ``warp_pipeline``
    # already covers the NB>=3 case via cross-warp interleave, so the
    # two are mutex (warp_pipeline wins if both are set). It is also
    # mutex with sliceMN whose own subtile scheduler doesn't compose
    # with the carry chain.
    _3stage_env = os.environ.get("TOKENSPEED_MOE_GLUON_3STAGE", "").strip().lower()
    if _3stage_env in {"1", "true", "yes", "on"}:
        use_3stage_pipeline = True
    elif _3stage_env in {"0", "false", "no", "off"}:
        use_3stage_pipeline = False
    elif use_3stage_pipeline is None:
        use_3stage_pipeline = False
    # Hard constraints: NB>=2 (already enforced by autotune) and
    # the program-side gluon.static_assert. Mutex with warp_pipeline
    # and sliceMN.
    if use_3stage_pipeline and (use_warp_pipeline or use_slice_mn):
        use_3stage_pipeline = False
    if use_3stage_pipeline and num_buffers < 2:
        use_3stage_pipeline = False

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

    # SliceMN halves M and N (v8-style 4-region split-shmem); the
    # K-loop body emits 4 sub-acc MFMA chains that the scheduler
    # interleaves with 4 sub-ds_reads. SliceN halves only N so the
    # body emits 2 sub-acc MFMAs (full-M); used on small-BM tiles
    # where the 2x2 sliceMN split is structurally illegal. See
    # ``MoESliceMNProgram`` / ``MoESliceNProgram`` for shape
    # constraints.
    if use_slice_mn:
        NUM_SUBTILES = (2, 2, 1)
    elif use_slice_n:
        NUM_SUBTILES = (1, 2, 1)
    else:
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
    # mxfp4 path). The mxfp4 x mxfp4 path skips the load entirely, so we
    # hand it a size-0 dummy and avoid the per-launch host->device
    # transfer that ``torch.tensor`` would do.
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
        expert_remap_buf,
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
    common_dims = (M, M_X, N, K, x_global_scale_buf, num_tiles_total)
    common_kwargs = dict(
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
        USE_SLICE_MN=use_slice_mn,
        USE_SLICE_N=use_slice_n,
        USE_3STAGE_PIPELINE=use_3stage_pipeline,
        GRID_N=grid_n,
        GROUP_M=group_m,
        XCD_SWIZZLE=xcd_swizzle,
        num_warps=num_warps,
    )

    if persistent or use_block_schedule:
        # Persistent / block-schedule path lives in a sibling module so
        # the non-persistent kernel stays straight-line. Both flavours
        # share `_pipelined_moe_tile_compute` and the swizzle helpers.
        from .gluon_persistent import _pipelined_moe_kernel_scaled_persistent

        k = _pipelined_moe_kernel_scaled_persistent[grid](
            *common_args,
            block_offs_buf,
            block_schedule_buf,
            *common_strides,
            *common_dims,
            USE_BLOCK_SCHEDULE=use_block_schedule,
            N_EXPTS_TOT=n_slices,
            **common_kwargs,
        )
    else:
        # Non-persistent: no block_offs / block_schedule / N_EXPTS_TOT /
        # USE_BLOCK_SCHEDULE constexprs needed; the kernel is a single
        # straight-line tile compute.
        k = _pipelined_moe_kernel_scaled[grid](
            *common_args,
            *common_strides,
            *common_dims,
            **common_kwargs,
        )

    # Snapshot the just-launched kernel's static GPR / occupancy profile
    # for bench-side ``last_kernel_profile()`` readers. id(k)-keyed cache
    # keeps the regex cost out of the hot launch loop.
    _capture_launch_profile(k)


# ---------------------------------------------------------------------------
# Public Python entry points (one per kernel that TASKS.md asks for)
# ---------------------------------------------------------------------------


# CDNA4 MFMA: regular = 16x16x32, scaled = 16x16x128 (BLOCK_K constraint).
_MFMA_K = 32
_MFMA_SCALED_K = 128
_MFMA_M = 16


def _round_up_int(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _ragged_slice_size(a_ragged_metadata, M: int) -> int | None:
    """Per-expert M hint for autotune. Mirrors triton_kernels'
    ``opt_flags_amd.make_default_opt_flags_amd`` slice-size formula so
    our BLOCK_M picks match what upstream's kernel sees on the same
    workload. Returns ``None`` (=> use the dense fallback) when no
    ragged metadata is available."""
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
    a_format: str = "e2m1",
    scale_load_mode: str = "transpose",
    slice_size: int | None = None,
) -> tuple[int, int, int, int]:
    """Pick ``(BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS)`` for scaled-MFMA tiles.

    Heuristic obtained by sweeping ``benchmarks/moe_gluon_perf_sweep.py``
    on MI355 with the gpt-oss-120b MoE dimensions (``H=I=2880, E=128,
    topk=4``). Each candidate must be (a) spill-free per ``static_profile``
    and (b) the highest-TFLOPs config at its shape.

    ``gl.amd.cdna4.mfma_scaled`` is 16x16x128 (M x N x K), so ``BLOCK_K``
    must be a multiple of 128 and ``>= 128``. The heuristic tiers off
    the logical ``M`` (which already includes per-expert padding for
    the MoE wrappers):

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
    """
    del do_swiglu, ragged  # tile menu is shape-driven; swiglu/ragged flags
    # currently don't change the pick. Kept in the signature so callers
    # (and the unit-test pinning the contract) stay stable.
    is_fp8 = a_format == "e4m3"
    # NOTE: a BM=16 sub-tier (for fp8 X + slice_size<=2, where the
    # swizzle scale layout's BM%32 constraint doesn't apply because
    # there's no per-block X scale) was tried and reverted: microbench
    # (dispatch B=1, prod-block-m=32) showed 71.9us at BM=16 vs 75.0us
    # at BM=32, but E2E c=1 TPOT regressed from 7.33ms -> 7.56ms over
    # 3 stable runs. The bench shape uses slice_sizes[i]=per_expert_padded,
    # so the microbench measures only MFMA pipeline efficiency at the
    # smaller tile; production has slice_sizes[i]=actual which apparently
    # exposes a different bottleneck (LDS pressure / gather path /
    # num_buffers). Profile c=1 before re-enabling.
    if slice_size is not None and slice_size <= 16:
        # Tiny ragged decode (per-expert M <= 16, i.e. B=1..~8 served
        # alone over E=128 experts). Empirically swept across
        # ``check_gluon_decode_perf.py --H 2880 --I 2880 --prod-block-m
        # 32`` on MI355: NW=4 + BK=256 wins by 5-9us vs the legacy
        # 64/128/512/8 tier across B=4..8. BM=16 sub-tier was tried
        # and reverted (see above comment block).
        bm, bn, bk, nw = 32, 128, 256, 4
    elif M <= 512:
        bm, bn, bk, nw = 64, 128, 512, 8
    elif is_fp8:
        # fp8 X tiles are 1 byte/elem so VGPR pressure is lower; we
        # promote to the large BM=128 tier as soon as M > 8192. Prefill
        # tier (M > 8192) lands on (128, 256, 256, NW=4) which is
        # :class:`MoESliceMNProgram`'s sweet spot (W tile 64 KB / K-iter,
        # 4-region staggering hides HBM at 32 KB TCP cap; auto-fired via
        # ``_should_auto_use_slice_mn`` since BM+BN=384 and W bytes >=
        # 16 KB).
        if M <= 8192:
            bm, bn, bk, nw = 64, 256, 128, 8
        else:
            bm, bn, bk, nw = 128, 256, 128, 4
    else:
        # mxfp4 X is 4b packed but the dequant adds VGPR pressure;
        # decode-y tier (M <= 8192) keeps BM=64 BK=256 to maximise
        # grid_m. Prefill tier (M > 8192) jumps to (128, 256, 256, NW=4)
        # -- same sliceMN sweet spot as the fp8 branch. Empirically
        # (sweep on B=1024..8192 dispatch+swiglu prefill) this beats the
        # older (64, 256, 256) by ~24% (M_d=16384) and the older
        # (256, 256, 128) by ~11% (M_d=32768) since 256x256 sets
        # vgpr=512 -> occ=1 with a high spill rate while 128x256 stays
        # at vgpr=312 occ=1 spill-light.
        if M <= 8192:
            bm, bn, bk, nw = 64, 256, 256, 4
        else:
            bm, bn, bk, nw = 128, 256, 256, 4
    # Clamp tile to actual shape (rounded up to the MFMA tile / scaled
    # K granularity). Tiny test shapes like 32x32x256 would otherwise
    # over-tile and yield NaN-padded reductions.
    bm = max(_MFMA_M, min(bm, _round_up_int(M, _MFMA_M)))
    bn = max(_MFMA_M, min(bn, _round_up_int(N, _MFMA_M)))
    bk = max(_MFMA_SCALED_K, min(bk, _round_up_int(K, _MFMA_SCALED_K)))
    # swizzle's in-LDS unswizzle reshapes the K_S-block dim into
    # `(BLOCK_K_S//8, 4, 16, 2, 2, 1)` which is only valid when
    # BLOCK_K_S = BLOCK_K // SCALE_BLOCK >= 8 (=> BLOCK_K >= 256 for
    # SCALE_BLOCK=32). Bump the autotune pick up to 256 so the mode is
    # usable. This matches what upstream's AMD opt_flags pick for CDNA4
    # mxfp4 (block_k = 128 // 0.5 = 256).
    if scale_load_mode == "swizzle":
        bk = max(bk, 256)
        bk = min(bk, _round_up_int(K, _MFMA_SCALED_K))
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
    ``SCALE_VIA_LDS`` (== swizzle); we always assume they are for the
    worst-case budget here.
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
    del bm, bn, bk, K, a_format, scale_load_mode  # autotune-off
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


_CDNA4_NUM_XCDS = 8  # MI355X has 8 XCDs (chiplets) per device.


def _autotune_pid_swizzle(
    num_tiles_total: int,
    grid_n: int,
    grid_m_padded: int,
    block_m: int,
) -> tuple[int, int]:
    """Pick ``(GROUP_M, XCD_SWIZZLE)`` heuristic defaults.

    Only the prefill-shaped regime benefits empirically: decode tile
    counts (``num_tiles_total < ~256``) are small enough that the
    natural hardware-rotation order already populates all XCDs evenly,
    and adding a logical permutation costs more than it saves
    (measured 1.43e2 -> 1.74e2 us on decode B=1 dispatch+swiglu).
    For larger tile counts we want to:

    1. **XCD swizzle** so consecutive ``pid`` land on the same XCD --
       neighbouring CTAs in the launch order then share the same XCD's
       L2 slice for the W-tile they fetch.
    2. **GROUP_M** so the same W column tile (selected by ``pid_n``)
       is reused across ``GROUP_M`` M-tiles before moving on -- the
       classic GEMM L2-reuse swizzle.

    The combination requires:
      * ``grid_m_padded % GROUP_M == 0`` (so GROUP_M is a valid 2-D
        permutation; otherwise ``_group_m_swizzle`` boundary code path
        leaves orphan tiles).
      * ``num_tiles_total % XCD_SWIZZLE == 0`` (same reasoning for
        the linear-domain permutation).
    Returns ``(1, 1)`` (no-op) on any mismatch.
    """
    # Decode / tiny prefill: no swizzle. ``256`` is the empirical knee
    # on MI355 (8 XCDs * ~32 CUs each); below that L2 reuse is already
    # high and swizzle overhead dominates.
    if num_tiles_total < 256:
        return 1, 1
    # block_m=32 is the decode autotune tier; even if the tile count is
    # large (long ragged decode), the per-tile MFMA chain is too short
    # to amortise the prologue, so disable swizzle there too.
    if block_m <= 32:
        return 1, 1
    # Pick the largest GROUP_M in {8, 4, 2} that divides grid_m_padded.
    group_m = 1
    for g in (8, 4, 2):
        if grid_m_padded % g == 0:
            group_m = g
            break
    # XCD swizzle: only enable if it divides the full domain.
    xcd_swizzle = _CDNA4_NUM_XCDS if num_tiles_total % _CDNA4_NUM_XCDS == 0 else 1
    return group_m, xcd_swizzle


def _persistent_grid_size(num_tiles_total: int) -> int:
    """Pick how many CTAs the persistent kernel should launch.

    Cap at ``num_tiles_total`` (no point launching more CTAs than work
    items) and at ``_CDNA4_NUM_CUS * _PERSISTENT_OVERSUBSCRIBE`` (over-
    subscribing past this just queues CTAs on the same CU).
    """
    if num_tiles_total <= 0:
        return 1
    return max(1, min(num_tiles_total, _CDNA4_NUM_CUS * _PERSISTENT_OVERSUBSCRIBE))


_SLICE_N_FORCE_ENV = "TOKENSPEED_MOE_GLUON_SLICE_N"


def _slice_n_env_override() -> bool | None:
    """Read ``TOKENSPEED_MOE_GLUON_SLICE_N`` env (1/0/auto).

    ``1`` / ``true`` / ``on``  -> force sliceN,
    ``0`` / ``false`` / ``off`` -> force regular pipeline,
    anything else (or unset)   -> ``None`` (let auto-heuristic decide).
    """
    raw = os.environ.get(_SLICE_N_FORCE_ENV, "").strip().lower()
    if raw in {"1", "true", "yes", "on", "force"}:
        return True
    if raw in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return None


def _can_use_slice_n(
    bm: int,
    bn: int,
    *,
    scale_load_mode: str,
    a_format: str,
    has_x_block_scale: bool,
    has_w_block_scale: bool,
) -> bool:
    """Return True iff :class:`MoESliceNProgram` is structurally legal
    on the requested tile.

    Constraints (mirror the docstring on :class:`MoESliceNProgram`):
      * ``bn >= 256`` (sub-N >= 128). At ``BN=128`` the half-N
        ``[BLOCK_K, 64]`` G->LDS tile under-fills the per-warp NONK
        footprint that ``_load_layout`` selects for ``BLOCK_K`` >=
        256 with ``NUM_WARPS=4`` -- the LDS unswizzle then rejects
        the layout (compile fail in test_moe_gluon
        ``[64-128-2880-swizzle]``). Empirically BN=128 also gives no
        TCP-overlap win because the half-tile (~16 KB) is already
        under the gfx9 32 KB cap.
      * ``(bn // 2) % 64 == 0`` (sub-tile must be a clean MFMA stripe).
      * ``bm >= 16`` (no further constraint from N-only slicing).
      * Same scale-LDS gating as sliceMN.

    Notice ``bm`` is intentionally NOT capped at 64 / 128 the way the
    2x2 sliceMN split is -- that's the entire point of having sliceN:
    cover the BM=16 / BM=32 decode tiles that the 2x2 split refuses.
    """
    if bn < 256 or bm < 16:
        return False
    if (bn // 2) % 64 != 0:
        return False
    needs_scale_lds = (has_x_block_scale and a_format == "e2m1") or has_w_block_scale
    if needs_scale_lds and scale_load_mode != "swizzle":
        return False
    return True


_TCP_INFLIGHT_CAP_BYTES = 32 * 1024  # gfx9 L1/TCP per-CU in-flight cap


def _should_auto_use_slice_n(bm: int, bn: int, bk: int) -> bool:
    """Auto-enable heuristic for sliceN.

    Returns True when ALL of the following hold:
      1. ``BN >= 256`` -- empirically the only regime where sliceN's
         per-region wait_group + extra commit_group overhead is
         dominated by the TCP-overlap benefit. At ``BN=128`` the
         half-N ``buffer_load_to_shared`` train (~16 KB or less)
         is well under the gfx9 32 KB cap, so splitting it adds
         scheduling overhead without any in-flight headroom gain
         (see e4m3 BN=128 BK=256 microbench: +3.2% regression).
         The half-N async_copy + LDS layout shapes also have to
         tile cleanly across ``num_warps`` warps; at BN=128 with
         NW=4 the ``[BLOCK_K, BN/2=64]`` G->LDS tile under-fills
         a warp's natural NONK footprint and the layout selector
         emits a degenerate config that the LDS unswizzle rejects
         (test_moe_gluon ``[64-128-2880-swizzle]`` repro).
      2. ``W tile >= 32 KB`` per K-iter (= gfx9 TCP/L1 in-flight
         cap). Below this, plain ``buffer_load_to_shared`` doesn't
         saturate the cap and there's no overlap window to win
         back via the 4-region staggered issue.
      3. ``BM <= 64`` -- sliceMN already covers BM>=128 with a
         tighter MFMA-sub-tile schedule; don't double up.

    Empirical wins on MI355 gpt-oss-120b dispatch (e2m1) at
    ``check_gluon_decode_perf.py``::

        BM=64 BN=256 BK=256 NB=2:
            B=  8  : 182.12 us -> 152.49 us (-16.3%)
            B= 16  : 230.88 us -> 205.90 us (-10.8%)
            B= 32  : 307.93 us -> 281.02 us (-8.7%)
            B= 64  : 307.90 us -> 280.44 us (-8.9%)
            B=128  : 309.83 us -> 281.90 us (-9.0%)
    """
    if bn < 256:
        return False
    # mxfp4 W (always packed 0.5 byte/elem in our kernels).
    w_bytes = (bn * bk) // 2
    if w_bytes < _TCP_INFLIGHT_CAP_BYTES:
        return False
    if bm > 64:
        return False
    return True


def _resolve_use_slice_n(
    user: bool | None,
    bm: int,
    bn: int,
    *,
    scale_load_mode: str,
    a_format: str,
    has_x_block_scale: bool,
    has_w_block_scale: bool,
    bk: int | None = None,
) -> bool:
    """Combine user kwarg + env + auto-heuristic + structural feasibility.

    Precedence (high -> low):
      1. ``user is not None``: caller explicit.
      2. ``TOKENSPEED_MOE_GLUON_SLICE_N`` env (force on/off).
      3. Auto-heuristic: enable iff W tile per K-iter saturates the
         gfx9 TCP 32 KB cap on small-BM tiles where sliceMN cannot
         apply (``BM<=64``). See :func:`_should_auto_use_slice_n`.

    sliceN is the small-BM counterpart of sliceMN; the two have
    disjoint structural domains (sliceN covers BM<=64, sliceMN
    requires BM>=128) so no host-side mutex is needed.
    """
    if user is not None:
        decision = bool(user)
    else:
        env = _slice_n_env_override()
        if env is not None:
            decision = env
        elif bk is not None:
            decision = _should_auto_use_slice_n(bm, bn, bk)
        else:
            # Caller didn't pass BK (legacy entry point) -- preserve
            # the original opt-in default to avoid changing behaviour
            # silently on paths that haven't been audited.
            decision = False
    if not decision:
        return False
    return _can_use_slice_n(
        bm,
        bn,
        scale_load_mode=scale_load_mode,
        a_format=a_format,
        has_x_block_scale=has_x_block_scale,
        has_w_block_scale=has_w_block_scale,
    )


_SLICE_MN_FORCE_ENV = "TOKENSPEED_MOE_GLUON_SLICE_MN"


def _slice_mn_env_override() -> bool | None:
    """Read ``TOKENSPEED_MOE_GLUON_SLICE_MN`` env (1/0/auto).

    ``1`` / ``true`` / ``on``  -> force sliceMN,
    ``0`` / ``false`` / ``off`` -> force regular pipeline,
    anything else (or unset)   -> ``None`` (let auto-heuristic decide).
    """
    raw = os.environ.get(_SLICE_MN_FORCE_ENV, "").strip().lower()
    if raw in {"1", "true", "yes", "on", "force"}:
        return True
    if raw in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return None


def _can_use_slice_mn(
    bm: int,
    bn: int,
    *,
    num_buffers: int,
    scale_load_mode: str,
    a_format: str,
    has_x_block_scale: bool,
    has_w_block_scale: bool,
) -> bool:
    """Return True iff :class:`MoESliceMNProgram` is structurally
    legal on the requested tile.

    Constraints (mirror the docstring on :class:`MoESliceMNProgram`):
      * ``BM >= 128`` / ``BN >= 128`` (sub-tile floor from
        ``tiles_per_warp=[2,2] * warps_{m,n}=2`` = 64).
      * ``(BM // 2) % 64 == 0`` and ``(BN // 2) % 64 == 0``.
      * ``NUM_BUFFERS >= 2`` (the 4-region prologue / drain dance
        breaks at NB=1 -- ``gluon.static_assert`` in the program).
      * Same scale-LDS gating as sliceN.
    """
    if bm < 128 or bn < 128:
        return False
    if (bm // 2) % 64 != 0 or (bn // 2) % 64 != 0:
        return False
    if num_buffers < 2:
        return False
    needs_scale_lds = (has_x_block_scale and a_format == "e2m1") or has_w_block_scale
    if needs_scale_lds and scale_load_mode != "swizzle":
        return False
    return True


def _should_auto_use_slice_mn(bm: int, bn: int, bk: int, a_format: str) -> bool:
    """Auto-enable heuristic for :class:`MoESliceMNProgram`.

    Returns True when the tile is in the regime where the 4-region
    staggered async-copy schedule pays off vs the plain pipelined
    single ``buffer_load_to_shared`` train:

      1. ``BM >= 128`` AND ``BN >= 128`` -- structural prereq for
         the 2x2 sub-tile MFMA grid.
      2. **W tile per K-iter >= 16 KB** -- the half-W tile (= region's
         own ``buffer_load`` train) only meaningfully benefits from
         per-region staggering once the full-W issue is comfortably
         within the gfx9 32 KB TCP/L1 cap. Below 16 KB the 4-commit
         overhead of split-issue can outweigh the staggering win.
         (e2m1 BN=128 BK=128 = 8 KB W tile; e4m3 BN=128 BK=128 = 16
         KB; e2m1 BN=256 BK=128 = 16 KB. So this triggers above
         BN=128 BK=128 e2m1 / BN=128 BK=128 e4m3 prefill tiles.)
      3. ``BM + BN >= 384`` -- the per-region MFMA budget needs to be
         long enough (~3 regions ~= 1500 cycles) to hide HBM at large
         K. Tiny tiles (BM=128 BN=128) have only ~32 MFMAs/region
         which is borderline; we require BM+BN >= 384 = roughly 48
         MFMAs/region.

    These thresholds will be tuned once the prefill sweep results
    come in. Conservative for now -- caller can force-enable via
    ``TOKENSPEED_MOE_GLUON_SLICE_MN=1`` if the heuristic misses
    a win.
    """
    if bm < 128 or bn < 128:
        return False
    # Approximate W tile bytes per K-iter assuming packed mxfp4 (0.5
    # byte/elem) for e2m1 and 1 byte/elem for fp8.
    if a_format == "e2m1":
        w_bytes = (bn * bk) // 2
    else:
        w_bytes = bn * bk
    if w_bytes < 16 * 1024:
        return False
    if (bm + bn) < 384:
        return False
    return True


def _resolve_use_slice_mn(
    user: bool | None,
    bm: int,
    bn: int,
    *,
    num_buffers: int,
    scale_load_mode: str,
    a_format: str,
    has_x_block_scale: bool,
    has_w_block_scale: bool,
    use_slice_n: bool = False,
    bk: int | None = None,
) -> bool:
    """Combine user kwarg + env + auto-heuristic + structural feasibility.

    Precedence (high -> low):
      1. ``user is not None``: caller explicit.
      2. ``TOKENSPEED_MOE_GLUON_SLICE_MN`` env (force on/off).
      3. Auto-heuristic: enable iff the tile is in the v8 sweet
         spot (``BM/BN >= 128``, large enough W half-tile, long
         enough per-region MFMA budget). See
         :func:`_should_auto_use_slice_mn`.

    sliceN covers small-BM tiles (BM<=64) where sliceMN would
    require ``BM >= 128`` anyway -- they don't overlap, but if a
    caller explicitly forced sliceN we still bail out here so the
    kernel picks ONE subtile-strategy flavour.
    """
    if use_slice_n:
        return False
    if user is not None:
        decision = bool(user)
    else:
        env = _slice_mn_env_override()
        if env is not None:
            decision = env
        elif bk is not None:
            decision = _should_auto_use_slice_mn(bm, bn, bk, a_format)
        else:
            decision = False
    if not decision:
        return False
    return _can_use_slice_mn(
        bm,
        bn,
        num_buffers=num_buffers,
        scale_load_mode=scale_load_mode,
        a_format=a_format,
        has_x_block_scale=has_x_block_scale,
        has_w_block_scale=has_w_block_scale,
    )


def _can_use_warp_pipeline(
    bm: int,
    bn: int,
    bk: int,
    *,
    K: int,
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
    if num_buffers < 3:
        return False
    if (K + bk - 1) // bk < num_buffers:
        return False
    via_lds = scale_load_mode == "swizzle"
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
    use_slice_mn: bool | None = None,
    use_slice_n: bool | None = None,
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
        a_format=a_format,
        has_x_block_scale=a_format == "e2m1",
        has_w_block_scale=True,
        scale_load_mode=scale_load_mode,
    )
    num_buffers = num_buffers if num_buffers is not None else nb_auto
    use_warp_pipeline = use_warp_pipeline if use_warp_pipeline is not None else wp_auto
    use_slice_mn = _resolve_use_slice_mn(
        use_slice_mn,
        block_m,
        block_n,
        num_buffers=num_buffers,
        scale_load_mode=scale_load_mode,
        a_format=a_format,
        has_x_block_scale=a_format == "e2m1",
        has_w_block_scale=True,
        bk=block_k,
    )
    use_slice_n = _resolve_use_slice_n(
        use_slice_n,
        block_m,
        block_n,
        scale_load_mode=scale_load_mode,
        a_format=a_format,
        has_x_block_scale=a_format == "e2m1",
        has_w_block_scale=True,
        bk=block_k,
    )
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
        use_slice_mn=use_slice_mn,
        use_slice_n=use_slice_n,
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
    use_slice_mn: bool | None = None,
    use_slice_n: bool | None = None,
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
        a_format=a_format,
        scale_load_mode=scale_load_mode,
        slice_size=_ragged_slice_size(a_ragged_metadata, M),
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
        a_format=a_format,
        has_x_block_scale=a_format == "e2m1",
        has_w_block_scale=True,
        scale_load_mode=scale_load_mode,
    )
    num_buffers = num_buffers if num_buffers is not None else nb_auto
    use_warp_pipeline = use_warp_pipeline if use_warp_pipeline is not None else wp_auto
    use_slice_mn = _resolve_use_slice_mn(
        use_slice_mn,
        block_m,
        block_n,
        num_buffers=num_buffers,
        scale_load_mode=scale_load_mode,
        a_format=a_format,
        has_x_block_scale=a_format == "e2m1",
        has_w_block_scale=True,
        bk=block_k,
    )
    use_slice_n = _resolve_use_slice_n(
        use_slice_n,
        block_m,
        block_n,
        scale_load_mode=scale_load_mode,
        a_format=a_format,
        has_x_block_scale=a_format == "e2m1",
        has_w_block_scale=True,
        bk=block_k,
    )
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
        use_slice_mn=use_slice_mn,
        use_slice_n=use_slice_n,
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
    use_slice_mn: bool | None = None,
    use_slice_n: bool | None = None,
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
        a_format=a_format,
        scale_load_mode=scale_load_mode,
        slice_size=_ragged_slice_size(a_ragged_metadata, M),
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
        a_format=a_format,
        has_x_block_scale=a_format == "e2m1",
        has_w_block_scale=True,
        scale_load_mode=scale_load_mode,
    )
    num_buffers = num_buffers if num_buffers is not None else nb_auto
    use_warp_pipeline = use_warp_pipeline if use_warp_pipeline is not None else wp_auto
    use_slice_mn = _resolve_use_slice_mn(
        use_slice_mn,
        block_m,
        block_n,
        num_buffers=num_buffers,
        scale_load_mode=scale_load_mode,
        a_format=a_format,
        has_x_block_scale=a_format == "e2m1",
        has_w_block_scale=True,
        bk=block_k,
    )
    use_slice_n = _resolve_use_slice_n(
        use_slice_n,
        block_m,
        block_n,
        scale_load_mode=scale_load_mode,
        a_format=a_format,
        has_x_block_scale=a_format == "e2m1",
        has_w_block_scale=True,
        bk=block_k,
    )
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
        use_slice_mn=use_slice_mn,
        use_slice_n=use_slice_n,
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
    """Return the raw uint8 scale tensor that Gluon's ``swizzle``
    mode consumes. ``CDNA4MXScaleLayout.swizzle_data`` and our host
    ``_swizzle_scales_cdna4`` are bit-equivalent (verified by
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
                # ``_gluon_mxfp_ragged_matmul``), NOT in ``passthrough``;
                # falling back to the latter silently dropped them and
                # left the kernel allocating a ``(M, N)`` slab without
                # the post-kernel top-k reduction, producing a
                # ``(top_k * n_tokens, N)`` output that crashed the
                # downstream layernorm with a shape mismatch on decode
                # batches.
                n_tokens=n_tokens,
                n_expts_act=n_expts_act,
                out_dtype=out_dtype,
                scale_load_mode="swizzle",
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
                scale_load_mode="swizzle",
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
                scale_load_mode="swizzle",
                w_transpose=True,
            )
            return out, True
    except Exception as exc:  # noqa: BLE001
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
    **passthrough,
) -> torch.Tensor:
    """Selector-facing entry: matches the upstream ``matmul`` signature.

    Routes mxfp4 weight + (fp8 / mxfp4) activation calls to our scaled-MFMA
    Gluon launchers via :func:`_try_dispatch_mxfp`. Any call shape that our
    launchers don't support (e.g. plain bf16 x bf16 with no block scale,
    unsupported epilogue knobs, missing ``b_mx_scale``) is transparently
    forwarded to ``triton_kernels.matmul`` so the gpt-oss-120b path keeps
    working.

    Extra keyword arguments (``gammas``, ``betas``, ``out_alpha``,
    ``c``/``c_acc_in``, ``fused_comm``, ``epilogue``,
    ``b_ragged_metadata``, ...) the runtime backends pass to
    ``tokenspeed_kernel.moe_experts`` are forwarded transparently to the
    upstream ``triton_kernels.matmul`` so combine-side scaling
    (``gammas``) and other epilogue knobs are preserved when we fall back.
    """
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
)(_gluon_mxfp_ragged_matmul)

register_kernel(
    "moe",
    "experts",
    name="triton_kernels_gluon_gemm_combine",
    features={"ragged_metadata", "gemm_combine"},
    **_common,
)(_gluon_mxfp_ragged_matmul)

register_kernel(
    "moe",
    "experts",
    name="triton_kernels_gluon_matmul_ogs",
    features={"ragged_metadata"},
    **_common,
)(_gluon_mxfp_ragged_matmul)


__all__ = [
    "_gluon_mxfp_ragged_matmul",
    "assert_no_spills",
    "gluon_mxfp_combine",
    "gluon_mxfp_dispatch_swiglu",
    "gluon_mxfp_gating_gemm",
    "static_profile",
]
