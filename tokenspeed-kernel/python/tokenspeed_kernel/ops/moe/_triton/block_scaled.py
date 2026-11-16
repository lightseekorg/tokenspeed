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

"""Portable block-scaled expert arithmetic; weights stay compressed in HBM."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel._triton import TensorDescriptor, libdevice, tl, triton
from tokenspeed_kernel.ops.communication.mailbox import MAILBOX_ROWS, MailboxTarget
from tokenspeed_kernel.ops.gemm.triton_skinny import _prefetch_weight_rows
from tokenspeed_kernel.ops.moe._triton.common import _route_expert, _routing_block
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    pdl_enabled,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

if TYPE_CHECKING:
    from tokenspeed_kernel.ops.moe import ExpertDispatch


@triton.jit
def _fp4_decode(code):
    magnitude = code & 7
    value = tl.where(magnitude < 4, magnitude.to(tl.float32) * 0.5, 2.0)
    value = tl.where(magnitude == 5, 3.0, value)
    value = tl.where(magnitude == 6, 4.0, value)
    value = tl.where(magnitude == 7, 6.0, value)
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def _fp4_encode(x):
    magnitude = tl.abs(x)
    code = (
        (magnitude > 0.25).to(tl.int32)
        + (magnitude >= 0.75).to(tl.int32)
        + (magnitude > 1.25).to(tl.int32)
        + (magnitude >= 1.75).to(tl.int32)
        + (magnitude > 2.5).to(tl.int32)
        + (magnitude >= 3.5).to(tl.int32)
        + (magnitude > 5.0).to(tl.int32)
    )
    sign = (x.to(tl.int32, bitcast=True) < 0).to(tl.int32)
    return code | (sign << 3)


@triton.jit
def _fp4_round(x):
    """``(_fp4_encode(x), _fp4_decode(code))``: the e2m1 code of FP32 ``x``,
    rounded to nearest even and saturated at 6, and the value it encodes."""
    if tl.target_info.cuda_capability_geq(10, 0):
        # sm100 converts in hardware; finite inputs give the same codes.
        code, value = tl.inline_asm_elementwise(
            """
            {
            .reg .b8 q;
            .reg .b32 h;
            .reg .b16 lo, hi;
            cvt.rn.satfinite.e2m1x2.f32 q, 0f00000000, $2;
            cvt.u32.u8 $0, q;
            cvt.rn.f16x2.e2m1x2 h, q;
            mov.b32 {lo, hi}, h;
            cvt.f32.f16 $1, lo;
            }
            """,
            "=r,=f,f",
            [x],
            dtype=(tl.int32, tl.float32),
            is_pure=True,
            pack=1,
        )
    else:
        code = _fp4_encode(x)
        value = _fp4_decode(code)
    return code, value


@triton.jit
def _block_error(x, reconstructed, GROUPS: tl.constexpr):
    error = tl.abs(x - reconstructed)
    for width in tl.static_range(3, -1, -1):
        pairs = tl.reshape(error, (GROUPS, 1 << width, 2))
        low, high = tl.split(pairs)
        error = low + high
    return tl.reshape(error, (GROUPS,))


@triton.jit
def _scale_offset(row, col, cols: tl.constexpr, layout):
    """Offset of block scale ``(row, col)`` of ``[rows, cols]`` scales in a
    TRT-LLM-gen layout: 0 row-major; 1 row-major 8x4 tiles; 3 row-major
    128x4 tiles whose rows 0-31, 32-63, 64-95 and 96-127 interleave."""
    if layout == 1:
        offset = ((row // 8) * (cols // 4) + col // 4) * 32 + (row % 8) * 4 + col % 4
    elif layout == 3:
        offset = (
            ((row // 128) * (cols // 4) + col // 4) * 512
            + (row % 32) * 16
            + ((row % 128) // 32) * 4
            + col % 4
        )
    else:
        offset = row * cols + col
    return offset


@triton.jit
def _nvfp4_chunk_codes(x, amax, BLOCK: tl.constexpr):
    """NVFP4 codes of FP32 ``x`` (``BLOCK`` columns of a row whose absolute
    maximum is ``amax``): packed E2M1 pairs (low nibble first) and their
    per-16 E4M3 scales, four or six chosen by reconstruction MAE."""
    packed, scale, _ = _nvfp4_chunk_codes_and_values(x, amax, BLOCK)
    return packed, scale


@triton.jit
def _nvfp4_chunk_codes_and_values(x, amax, BLOCK: tl.constexpr):
    """``_nvfp4_chunk_codes`` and the FP32 values the codes encode: each E2M1
    value times its E4M3 scale, without the row's outer factor. A GEMM that
    dequantizes the codes reads exactly these values, and they fit BF16."""
    x = tl.reshape(x, (BLOCK // 16, 16))
    encoder = tl.minimum(tl.div_rn(1536.0, amax), 3.4028234663852886e38)
    decoder = tl.div_rn(1.0, encoder)
    outer = amax * (1.0 / 1536.0)
    maximum = tl.max(tl.abs(x), 1)
    scale6 = tl.minimum(maximum * (encoder * (1.0 / 6.0)), 448.0)
    scale4 = tl.minimum(maximum * (encoder * 0.25), 448.0)
    scale6 = scale6.to(tl.float8e4nv).to(tl.float32)
    scale4 = scale4.to(tl.float8e4nv).to(tl.float32)
    inverse6 = tl.minimum(tl.div_rn(1.0, scale6 * decoder), 3.4028234663852886e38)
    inverse4 = tl.minimum(tl.div_rn(1.0, scale4 * decoder), 3.4028234663852886e38)
    code6, value6 = _fp4_round(x * inverse6[:, None])
    code4, value4 = _fp4_round(x * inverse4[:, None])
    reconstructed6 = (value6 * scale6[:, None]) * outer
    reconstructed4 = (value4 * scale4[:, None]) * outer
    error6 = _block_error(x, reconstructed6, BLOCK // 16)
    error4 = _block_error(x, reconstructed4, BLOCK // 16)
    choose4 = error4 < error6
    code = tl.reshape(tl.where(choose4[:, None], code4, code6), (BLOCK // 2, 2))
    low, high = tl.split(code)
    packed = (low | (high << 4)).to(tl.uint8)
    scale = tl.where(choose4, scale4, scale6).to(tl.float8e4nv)
    values = tl.where(
        choose4[:, None], value4 * scale4[:, None], value6 * scale6[:, None]
    )
    return packed, scale, tl.reshape(values, (BLOCK,))


@triton.jit
def _store_nvfp4_chunk(
    Values,
    Scales,
    row,
    chunk,
    packed,
    scale,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
    layout,
):
    """Store ``_nvfp4_chunk_codes`` at columns ``chunk * BLOCK`` of ``row``,
    the scales in ``layout`` (``_scale_offset``)."""
    packed_col = chunk * (BLOCK // 2) + tl.arange(0, BLOCK // 2)
    scale_col = chunk * (BLOCK // 16) + tl.arange(0, BLOCK // 16)
    tl.store(Values + row * (width // 2) + packed_col, packed, packed_col < width // 2)
    tl.store(
        Scales + _scale_offset(row, scale_col, width // 16, layout),
        scale,
        scale_col < width // 16,
    )


@triton.jit
def _quantize_nvfp4_chunk(
    Values,
    Scales,
    Dequant,
    row,
    chunk,
    x,
    amax,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
    layout,
    STORE_DEQUANT: tl.constexpr,
):
    """Quantize FP32 columns ``chunk * BLOCK + [0, BLOCK)`` of one row, its
    scales in ``layout`` (``_scale_offset``). With ``STORE_DEQUANT`` also
    store the BF16 values the codes encode in ``Dequant`` (``[rows, width]``),
    for a GEMM that reads its activations as BF16."""
    packed, scale, values = _nvfp4_chunk_codes_and_values(x, amax, BLOCK)
    _store_nvfp4_chunk(Values, Scales, row, chunk, packed, scale, width, BLOCK, layout)
    if STORE_DEQUANT:
        col = chunk * BLOCK + tl.arange(0, BLOCK)
        tl.store(Dequant + row * width + col, values.to(tl.bfloat16), col < width)


@triton.jit
def _mxfp8_chunk_codes(x, BLOCK: tl.constexpr):
    """MXFP8 codes of FP32 ``x`` (``BLOCK`` columns): E4M3 values and their
    per-32 UE8M0 exponents."""
    x = tl.reshape(x, (BLOCK // 32, 32))
    raw = tl.max(tl.abs(x), 1) * (1.0 / 448.0)
    exponent = ((raw.to(tl.int32, bitcast=True) + 0x007FFFFF) & 0x7F800000) >> 23
    exponent = tl.where(raw <= 2.0**-127, 0, tl.minimum(exponent, 254))
    exponent = tl.where(raw != raw, 255, exponent)
    scale = tl.maximum((exponent << 23).to(tl.float32, bitcast=True), 2.0**-127)
    inverse = tl.where(exponent == 0, 2.0**127, tl.div_rn(1.0, scale))
    inverse = tl.where(exponent == 255, 0.0, inverse)
    values = x * inverse[:, None]
    values = tl.minimum(
        tl.maximum(values, -448.0, propagate_nan=tl.PropagateNan.ALL),
        448.0,
        propagate_nan=tl.PropagateNan.ALL,
    )
    return tl.reshape(values, (BLOCK,)).to(tl.float8e4nv), exponent.to(tl.uint8)


@triton.jit
def _store_mxfp8_chunk(
    Values,
    Scales,
    row,
    chunk,
    values,
    exponent,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
    layout,
):
    """Store ``_mxfp8_chunk_codes`` at columns ``chunk * BLOCK`` of ``row``,
    the scales in ``layout`` (``_scale_offset``)."""
    col = chunk * BLOCK + tl.arange(0, BLOCK)
    group = chunk * (BLOCK // 32) + tl.arange(0, BLOCK // 32)
    tl.store(Values + row * width + col, values, col < width)
    tl.store(
        Scales + _scale_offset(row, group, width // 32, layout),
        exponent,
        group < width // 32,
    )


@triton.jit
def _quantize_mxfp8_chunk(
    Values, Scales, row, chunk, x, width: tl.constexpr, BLOCK: tl.constexpr, layout
):
    """Quantize FP32 columns ``chunk * BLOCK + [0, BLOCK)`` of one row, its
    scales in ``layout`` (``_scale_offset``)."""
    values, exponent = _mxfp8_chunk_codes(x, BLOCK)
    _store_mxfp8_chunk(
        Values, Scales, row, chunk, values, exponent, width, BLOCK, layout
    )


@triton.jit
def _quantize_row(
    X,
    Values,
    Scales,
    Amax,
    Dequant,
    row,
    chunk_program,
    width: tl.constexpr,
    NVFP4: tl.constexpr,
    BLOCK: tl.constexpr,
    ROW_PROGRAMS: tl.constexpr,
    STORE_DEQUANT: tl.constexpr,
):
    """Quantize ``row``: all of it (``ROW_PROGRAMS``) or column chunk
    ``chunk_program``. Every program takes its row's NVFP4 absolute maximum
    itself."""
    chunks: tl.constexpr = tl.cdiv(width, BLOCK)
    amax = 0.0
    if NVFP4:
        # One wide load per program: the row's maximum is exact in any order.
        rcol = tl.arange(0, triton.next_power_of_2(width))
        values = tl.load(X + row * width + rcol, rcol < width, other=0.0)
        amax = tl.max(tl.abs(values.to(tl.float32)), 0)
        if ROW_PROGRAMS:
            tl.store(Amax + row, amax)
        elif chunk_program == 0:
            tl.store(Amax + row, amax)
    first = 0 if ROW_PROGRAMS else chunk_program
    for chunk in range(first, chunks if ROW_PROGRAMS else first + 1):
        col = chunk * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + row * width + col, col < width, other=0.0).to(tl.float32)
        if NVFP4:
            _quantize_nvfp4_chunk(
                Values,
                Scales,
                Dequant,
                row,
                chunk,
                x,
                amax,
                width,
                BLOCK,
                0,
                STORE_DEQUANT,
            )
        else:
            _quantize_mxfp8_chunk(Values, Scales, row, chunk, x, width, BLOCK, 0)


@triton.constexpr_function
def _exact_reciprocal(divisor: float) -> float:
    """``1 / divisor`` when that is exact (a power of two), else 0."""
    return 1.0 / divisor if math.frexp(divisor)[0] == 0.5 else 0.0


@triton.jit
def _capped_divide(x, divisor: tl.constexpr):
    """``div_rn(x, divisor)``; a power-of-two divisor multiplies by its
    exact reciprocal instead, which rounds the same quotient once."""
    reciprocal: tl.constexpr = _exact_reciprocal(divisor)
    if reciprocal > 0:
        return x * reciprocal
    return tl.div_rn(x, divisor)


@triton.jit
def _swiglu_sigmoid(x):
    """The capped SwiGLU's sigmoid. On NVIDIA it is the checkpoint reference
    kernels' ``rcp.approx(1 + ex2.approx(x * -log2(e)))`` with subnormals
    flushed; elsewhere ``1 / (1 + exp(-x))``."""
    if tl.target_info.is_cuda():
        sigmoid = tl.inline_asm_elementwise(
            "{ .reg .f32 t; mul.rn.f32 t, $1, 0fBFB8AA3B; "
            "ex2.approx.ftz.f32 t, t; add.rn.f32 t, t, 0f3F800000; "
            "rcp.approx.ftz.f32 $0, t; }",
            "=f,f",
            [x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    else:
        sigmoid = 1.0 / (1.0 + tl.exp(-x))
    return sigmoid


@triton.jit
def _swiglu_tanh(x):
    """The capped SwiGLU's tanh: the reference kernels' ``tanh.approx.f32``
    on NVIDIA, libdevice's elsewhere."""
    if tl.target_info.is_cuda():
        y = tl.inline_asm_elementwise(
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            [x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    else:
        y = libdevice.tanh(x)
    return y


@triton.jit
def _capped_swiglu(
    GateUp,
    Gamma,
    route,
    row,
    col,
    width: tl.constexpr,
    gate_cap: tl.constexpr,
    up_cap: tl.constexpr,
):
    """BF16-rounded capped SwiGLU of one route's columns ``col``, read from
    ``row`` of the first projection's output."""
    mask = col < width
    gate = tl.load(GateUp + row * (2 * width) + 2 * col, mask, other=0.0).to(tl.float32)
    up = tl.load(GateUp + row * (2 * width) + 2 * col + 1, mask, other=0.0).to(
        tl.float32
    )
    sigmoid = _swiglu_sigmoid(gate)
    if gate_cap > 0:
        gate = gate_cap * _swiglu_tanh(_capped_divide(gate, gate_cap))
    if up_cap > 0:
        up = up_cap * _swiglu_tanh(_capped_divide(up, up_cap))
    gamma = tl.load(Gamma + route)
    result = ((gate * sigmoid) * up) * gamma
    return tl.where(mask, result.to(tl.bfloat16).to(tl.float32), 0.0)


@triton.jit
def _trigger_after_stores():
    """Release PDL dependents once every thread's stores are visible across
    the GPU: a GEMM after this kernel may read its inputs before its own
    wait, which alone orders only the kernel's completion."""
    tl.inline_asm_elementwise(
        "fence.acq_rel.gpu;\n\tmov.u32 $0, 0;",
        "=r",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    tl.debug_barrier()
    tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _projection_row(RouteRows, route, ROW_MAP: tl.constexpr):
    """The route's row in the projections' buffers: the route itself, or
    with ``ROW_MAP`` its expert-tiled row (``_flashinfer/route_tiles.py``)."""
    row = route
    if ROW_MAP:
        row = tl.load(RouteRows + route).to(tl.int64)
    return row


@triton.jit
def _store_row_amax(Amax, WAmax, WDen, row, expert, amax, ROW_MAP: tl.constexpr):
    """Store an NVFP4 row's maximum, or with ``ROW_MAP`` the second
    projection's FP32 output scale ``(amax * w_amax) * rcp(1536 * den)``
    that an expert-tiled GEMM applies per row."""
    if ROW_MAP:
        rcp = tl.div_rn(1.0, 1536.0 * tl.load(WDen))
        tl.store(Amax + row, (amax * tl.load(WAmax + expert)) * rcp)
    else:
        tl.store(Amax + row, amax)


@triton.jit(do_not_specialize=["layout"])
def _swiglu_quantize(
    GateUp,
    Gamma,
    Ids,
    RouteRows,
    Values,
    Scales,
    Amax,
    Dequant,
    WAmax,
    WDen,
    Output,
    expert_start,
    num_experts,
    layout,
    width: tl.constexpr,
    hidden: tl.constexpr,
    gate_cap: tl.constexpr,
    up_cap: tl.constexpr,
    NVFP4: tl.constexpr,
    BLOCK: tl.constexpr,
    ROW_PROGRAMS: tl.constexpr,
    ZERO_REMOTE: tl.constexpr,
    ROW_MAP: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    STORE_DEQUANT: tl.constexpr,
):
    """Capped SwiGLU, rounded to BF16, then quantized for the second
    projection, per route (``ROW_PROGRAMS``) or per (route, column chunk).
    No local projection reads a remote expert's route: with ``ZERO_REMOTE``
    its programs instead zero that route's ``[hidden]`` output row for the EP
    reduction. An NVFP4 row program evaluates the SwiGLU once: it keeps the
    BF16 row in the route's output row, which the second projection
    overwrites, and quantizes from there once it has the row maximum.
    With ``ROW_MAP`` a route reads and writes its expert-tiled row
    (``_projection_row``), its scales in ``layout`` (``_scale_offset``).
    Under PDL it waits before releasing the second projection, which loads
    its routes, or with ``ROW_MAP`` its tile metadata and weights, before
    its own wait: both come from the routing before the first projection.
    The route's expert id loads before this kernel's wait."""
    route = tl.program_id(0).to(tl.int64)
    expert = tl.load(Ids + route) - expert_start
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if expert < 0 or expert >= num_experts:
        if ZERO_REMOTE:
            # Each program zeroes its share of the row.
            span: tl.constexpr = (
                hidden if ROW_PROGRAMS else tl.cdiv(hidden, tl.cdiv(width, BLOCK))
            )
            offs = tl.arange(0, triton.next_power_of_2(span))
            start = 0 if ROW_PROGRAMS else tl.program_id(1) * span
            col = start + offs
            zeros = tl.zeros(offs.shape, Output.dtype.element_ty)
            tl.store(
                Output + route * hidden + col, zeros, (offs < span) & (col < hidden)
            )
    elif NVFP4 and ROW_PROGRAMS and hidden >= width:
        row = _projection_row(RouteRows, route, ROW_MAP)
        chunks: tl.constexpr = tl.cdiv(width, BLOCK)
        scratch = Output + route * hidden
        amax = 0.0
        for chunk in range(chunks):
            col = chunk * BLOCK + tl.arange(0, BLOCK)
            x = _capped_swiglu(GateUp, Gamma, route, row, col, width, gate_cap, up_cap)
            amax = tl.maximum(amax, tl.max(tl.abs(x), 0))
            tl.store(scratch + col, x.to(tl.bfloat16), col < width)
        _store_row_amax(Amax, WAmax, WDen, row, expert, amax, ROW_MAP)
        # Every thread's stores land before any thread reloads the row.
        tl.debug_barrier()
        for chunk in range(chunks):
            col = chunk * BLOCK + tl.arange(0, BLOCK)
            x = tl.load(scratch + col, col < width, other=0.0).to(tl.float32)
            _quantize_nvfp4_chunk(
                Values,
                Scales,
                Dequant,
                row,
                chunk,
                x,
                amax,
                width,
                BLOCK,
                layout,
                STORE_DEQUANT,
            )
    else:
        row = _projection_row(RouteRows, route, ROW_MAP)
        chunks: tl.constexpr = tl.cdiv(width, BLOCK)
        amax = 0.0
        if NVFP4:
            rcol = tl.arange(0, triton.next_power_of_2(width))
            values = _capped_swiglu(
                GateUp, Gamma, route, row, rcol, width, gate_cap, up_cap
            )
            amax = tl.max(tl.abs(values), 0)
            if ROW_PROGRAMS:
                _store_row_amax(Amax, WAmax, WDen, row, expert, amax, ROW_MAP)
            elif tl.program_id(1) == 0:
                _store_row_amax(Amax, WAmax, WDen, row, expert, amax, ROW_MAP)
        first = 0 if ROW_PROGRAMS else tl.program_id(1)
        for chunk in range(first, chunks if ROW_PROGRAMS else first + 1):
            col = chunk * BLOCK + tl.arange(0, BLOCK)
            x = _capped_swiglu(GateUp, Gamma, route, row, col, width, gate_cap, up_cap)
            if NVFP4:
                _quantize_nvfp4_chunk(
                    Values,
                    Scales,
                    Dequant,
                    row,
                    chunk,
                    x,
                    amax,
                    width,
                    BLOCK,
                    layout,
                    STORE_DEQUANT,
                )
            else:
                _quantize_mxfp8_chunk(
                    Values, Scales, row, chunk, x, width, BLOCK, layout
                )


@triton.jit
def _swiglu_chunk_max(
    GateUp,
    Gamma,
    Ids,
    RouteRows,
    Output,
    ChunkMax,
    expert_start,
    num_experts,
    width: tl.constexpr,
    hidden: tl.constexpr,
    gate_cap: tl.constexpr,
    up_cap: tl.constexpr,
    BLOCK: tl.constexpr,
    ROW_MAP: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    """First half of the decode-sized NVFP4 SwiGLU quantizer: one program per
    local (route, column chunk) evaluates the chunk's capped SwiGLU once,
    keeps the BF16 values in the route's output row (the second projection
    overwrites it) and stores their absolute maximum to ``ChunkMax``.
    Remote routes are left to ``_quantize_swiglu_chunks``.
    Under PDL both halves release their successor before they wait, so the
    second projection launches and asks L2 for its weights while the SwiGLU
    runs: every read before a wait down this chain (the expert ids, the
    second projection's routes) is older than the first projection, which
    waits before it releases this kernel. With ``ROW_MAP`` (a route's
    expert-tiled row, ``_projection_row``) the projections are TRT-LLM-gen
    GEMMs whose release points are their own, so this half waits first: an
    earlier release would launch the second GEMM while the first still holds
    the SMs. The second half, which starts once the first GEMM is done,
    releases before it waits: the second GEMM reads only its weights and the
    tile metadata, written before the first projection, before its own
    wait."""
    route = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    expert = tl.load(Ids + route) - expert_start
    if ENABLE_PDL:
        if ROW_MAP:
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
        else:
            tl.extra.cuda.gdc_launch_dependents()
            tl.extra.cuda.gdc_wait()
    if expert >= 0 and expert < num_experts:
        row = _projection_row(RouteRows, route, ROW_MAP)
        col = chunk * BLOCK + tl.arange(0, BLOCK)
        x = _capped_swiglu(GateUp, Gamma, route, row, col, width, gate_cap, up_cap)
        tl.store(Output + route * hidden + col, x.to(tl.bfloat16), col < width)
        chunks: tl.constexpr = tl.cdiv(width, BLOCK)
        tl.store(ChunkMax + route * chunks + chunk, tl.max(tl.abs(x), 0))


@triton.jit(do_not_specialize=["layout"])
def _quantize_swiglu_chunks(
    Ids,
    RouteRows,
    ChunkMax,
    Values,
    Scales,
    Amax,
    Dequant,
    WAmax,
    WDen,
    Output,
    expert_start,
    num_experts,
    layout,
    width: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK: tl.constexpr,
    ZERO_REMOTE: tl.constexpr,
    ROW_MAP: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    STORE_DEQUANT: tl.constexpr,
):
    """Second half: a local route's row maximum is the maximum of its chunk
    maxima, so each chunk program quantizes its BF16 chunk to NVFP4 without
    re-evaluating the row. With ``ZERO_REMOTE`` remote routes' programs zero
    their share of the route's output row. Under PDL it releases the second
    projection before it waits (see ``_swiglu_chunk_max``)."""
    route = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    expert = tl.load(Ids + route) - expert_start
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    chunks: tl.constexpr = tl.cdiv(width, BLOCK)
    if expert < 0 or expert >= num_experts:
        if ZERO_REMOTE:
            span: tl.constexpr = tl.cdiv(hidden, chunks)
            offs = tl.arange(0, triton.next_power_of_2(span))
            zero_col = chunk * span + offs
            zeros = tl.zeros(offs.shape, Output.dtype.element_ty)
            tl.store(
                Output + route * hidden + zero_col,
                zeros,
                (offs < span) & (zero_col < hidden),
            )
    else:
        row = _projection_row(RouteRows, route, ROW_MAP)
        maxima = tl.arange(0, triton.next_power_of_2(chunks))
        amax = tl.max(
            tl.load(ChunkMax + route * chunks + maxima, maxima < chunks, other=0.0),
            0,
        )
        if chunk == 0:
            _store_row_amax(Amax, WAmax, WDen, row, expert, amax, ROW_MAP)
        col = chunk * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(Output + route * hidden + col, col < width, other=0.0)
        _quantize_nvfp4_chunk(
            Values,
            Scales,
            Dequant,
            row,
            chunk,
            x.to(tl.float32),
            amax,
            width,
            BLOCK,
            layout,
            STORE_DEQUANT,
        )


# Up to this many rows each column chunk gets a program; beyond it, each row one.
_CHUNK_PROGRAM_MAX_ROWS = 1024


def _quantized_buffers(
    rows: int, width: int, nvfp4: bool, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    values = torch.empty(
        (rows, width // 2 if nvfp4 else width),
        device=device,
        dtype=torch.uint8 if nvfp4 else torch.float8_e4m3fn,
    )
    scales = torch.empty(
        (rows, width // (16 if nvfp4 else 32)),
        device=device,
        dtype=torch.float8_e4m3fn if nvfp4 else torch.uint8,
    )
    amax = torch.empty((rows,), device=device, dtype=torch.float32) if nvfp4 else None
    return values, scales, amax


@triton.jit
def _route_and_block_quantize(
    Ids,
    ExpertRoutes,
    ExpertCounts,
    num_routes,
    expert_start,
    X,
    Values,
    Scales,
    Amax,
    Dequant,
    EXPERTS: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
    width: tl.constexpr,
    NVFP4: tl.constexpr,
    BLOCK: tl.constexpr,
    ROW_PROGRAMS: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    STORE_DEQUANT: tl.constexpr,
):
    """The local experts' route lists (``_routing``, one program per expert)
    and the input quantization (``_quantize_row``, the other programs) in
    one launch; neither reads the other's output."""
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    if pid < EXPERTS:
        _route_expert(
            Ids, ExpertRoutes, ExpertCounts, num_routes, expert_start, pid, BLOCK_ROUTES
        )
    else:
        programs: tl.constexpr = 1 if ROW_PROGRAMS else tl.cdiv(width, BLOCK)
        program = pid - EXPERTS
        _quantize_row(
            X,
            Values,
            Scales,
            Amax,
            Dequant,
            (program // programs).to(tl.int64),
            program % programs,
            width,
            NVFP4,
            BLOCK,
            ROW_PROGRAMS,
            STORE_DEQUANT,
        )


def _routed_quantized_input(
    x: torch.Tensor,
    ids: torch.Tensor,
    experts: int,
    expert_start: int,
    fmt: str,
    dequant: torch.Tensor | None,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
]:
    """``_routing(ids, experts, expert_start)`` and the block quantization of
    ``x`` to ``fmt`` in one launch. A BF16 ``dequant`` (``x``'s shape, NVFP4
    only) also receives the values the codes encode."""
    ids = ids.to(torch.int32).contiguous()
    rows, width = x.shape
    num_routes = ids.numel()
    routes = torch.empty((experts, num_routes), device=x.device, dtype=torch.int32)
    counts = torch.empty(experts, device=x.device, dtype=torch.int32)
    nvfp4 = fmt == "nvfp4"
    if dequant is not None and (not nvfp4 or dequant.shape != x.shape):
        raise ValueError("a BF16 dequant copy needs NVFP4 and the input's shape")
    values, scales, amax = _quantized_buffers(rows, width, nvfp4, x.device)
    row_programs = rows > _CHUNK_PROGRAM_MAX_ROWS
    programs = rows * (1 if row_programs else triton.cdiv(width, 256))
    enable_pdl = pdl_enabled()
    _route_and_block_quantize[(experts + programs,)](
        ids,
        routes,
        counts,
        num_routes,
        expert_start,
        x,
        values,
        scales,
        values if amax is None else amax,
        values if dequant is None else dequant,
        experts,
        _routing_block(num_routes),
        width,
        nvfp4,
        # Row programs walk 2048-column chunks on 8 warps (2304 x 4608 NVFP4
        # rows with the routing on B300: 32.5 -> 23.2 us).
        2048 if row_programs else 256,
        row_programs,
        enable_pdl,
        dequant is not None,
        num_warps=8 if row_programs else 4,
        enable_fp_fusion=False,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return (routes, counts), (values, scales, amax)


@triton.jit
def _load_block(
    Values, Scales, rows, cols, mask, width: tl.constexpr, NVFP4: tl.constexpr
):
    if NVFP4:
        packed = tl.load(Values + rows * (width // 2) + cols // 2, mask=mask, other=0)
        code = (packed.to(tl.int32) >> ((cols % 2) * 4)) & 15
        scale = tl.load(
            Scales + rows * (width // 16) + cols // 16, mask=mask, other=0.0
        ).to(tl.float32)
        value = _fp4_decode(code) * scale
    else:
        data = tl.load(Values + rows * width + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        exponent = tl.load(
            Scales + rows * (width // 32) + cols // 32, mask=mask, other=0
        ).to(tl.int32)
        scale = tl.maximum((exponent << 23).to(tl.float32, bitcast=True), 2.0**-127)
        scale = tl.where(exponent == 255, float("nan"), scale)
        value = data * scale
    return value.to(tl.bfloat16)


@triton.jit(do_not_specialize=["route_count", "num_programs"])
def _expert_mm(
    X,
    XS,
    XA,
    W,
    WS,
    WA,
    WD,
    Y,
    Routes,
    Counts,
    route_count,
    num_programs,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    TOP_K: tl.constexpr,
    INPUT_PER_TOKEN: tl.constexpr,
    NVFP4: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tile = tl.program_id(0)
    problem_start = 0
    for expert in range(E):
        count = tl.load(Counts + expert)
        tiles_n = tl.cdiv(N, BN)
        problem_tiles = tl.cdiv(count, BM) * tiles_n
        while tile >= problem_start and tile < problem_start + problem_tiles:
            local = tile - problem_start
            m = (local // tiles_n) * BM + tl.arange(0, BM)
            n = (local % tiles_n) * BN + tl.arange(0, BN)
            route = tl.load(Routes + expert * route_count + m, m < count, other=0).to(
                tl.int64
            )
            xrow = route // TOP_K if INPUT_PER_TOKEN else route
            wrow = expert.to(tl.int64) * N + n.to(tl.int64)
            acc = tl.zeros((BM, BN), tl.float32)
            for start in range(tl.cdiv(K, BK)):
                k = start * BK + tl.arange(0, BK)
                xv = _load_block(
                    X,
                    XS,
                    xrow[:, None],
                    k[None, :],
                    (m < count)[:, None] & (k < K)[None, :],
                    K,
                    NVFP4,
                )
                wv = _load_block(
                    W,
                    WS,
                    wrow[:, None],
                    k[None, :],
                    (n < N)[:, None] & (k < K)[None, :],
                    K,
                    NVFP4,
                )
                acc = tl.dot(xv, tl.trans(wv), acc)
            if NVFP4:
                xa = tl.load(XA + xrow, m < count, other=0.0)
                wa = tl.load(WA + expert)
                denominator = 1536.0 * tl.load(WD)
                scale = (xa * wa) * tl.div_rn(1.0, denominator)
                acc = acc * scale[:, None]
            tl.store(
                Y + route[:, None] * N + n[None, :],
                acc,
                (m < count)[:, None] & (n < N)[None, :],
            )
            tile += num_programs
        problem_start += problem_tiles


@triton.jit(do_not_specialize=["route_count", "num_programs"])
def _expert_mm_scaled(
    X,
    XS,
    XA,
    W,
    WS,
    WA,
    WD,
    Y,
    Routes,
    Counts,
    WRows,
    WSRows,
    route_count,
    num_programs,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    TOP_K: tl.constexpr,
    INPUT_PER_TOKEN: tl.constexpr,
    NVFP4: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BK: tl.constexpr,
    TMA: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    ROUTES_BEFORE_WAIT: tl.constexpr,
):
    """``_expert_mm`` on native block-scaled tensor cores: the payloads and
    scales feed ``dot_scaled`` without a BF16 decode.

    A tile is ``BW`` weight rows by ``BT`` routed rows, the weights on the
    MMA's M side: an expert routed a handful of tokens fills a 128-row MMA
    with its weights instead of masked token rows. Every element sums the
    same K blocks in the same order as the token-major tile would. Tiles are
    numbered expert by expert; a program finds its tile's expert from the
    per-expert tile counts held in registers. With ``TMA``, ``W`` and ``WS``
    are 2-D ``[experts * N, ...]`` tensor descriptors; ``WRows`` and
    ``WSRows`` are their tensors.

    With ``ROUTES_BEFORE_WAIT`` (under PDL, once a kernel launched after the
    routes' producer has waited before releasing its successor, as the first
    projection does) the routes come from an older kernel: a program maps
    its first tile and asks L2 for its weight rows before it waits for
    ``X``.
    """
    # Packed payload bytes and scale columns along K.
    KP: tl.constexpr = K // 2 if NVFP4 else K
    PK: tl.constexpr = BK // 2 if NVFP4 else BK
    G: tl.constexpr = 16 if NVFP4 else 32
    tiles_n: tl.constexpr = (N + BW - 1) // BW
    if ENABLE_PDL and not ROUTES_BEFORE_WAIT:
        tl.extra.cuda.gdc_wait()
        # Every input is ready: dependents may launch and set up meanwhile.
        tl.extra.cuda.gdc_launch_dependents()
    experts = tl.arange(0, triton.next_power_of_2(E))
    counts = tl.load(Counts + experts, experts < E, other=0)
    expert_tiles = tl.cdiv(counts, BT) * tiles_n
    ends = tl.cumsum(expert_tiles, 0)
    tile = tl.program_id(0)
    while tile < tl.sum(expert_tiles, 0):
        expert = tl.sum((ends <= tile).to(tl.int32), 0)
        local = tile - tl.sum(tl.where(experts == expert, ends - expert_tiles, 0), 0)
        count = tl.sum(tl.where(experts == expert, counts, 0), 0)
        m = (local // tiles_n) * BT + tl.arange(0, BT)
        n = (local % tiles_n) * BW + tl.arange(0, BW)
        route = tl.load(Routes + expert * route_count + m, m < count, other=0).to(
            tl.int64
        )
        xrow = route // TOP_K if INPUT_PER_TOKEN else route
        wrow = expert.to(tl.int64) * N + n.to(tl.int64)
        if ENABLE_PDL and ROUTES_BEFORE_WAIT:
            # Later tiles find the wait already passed.
            first = (expert * N + (local % tiles_n) * BW).to(tl.int64)
            # Bulk L2 prefetches take 16-byte aligned rows.
            if KP % 16 == 0:
                _prefetch_weight_rows(WRows, first, 0, E * N, KP, BW, KP)
            if (K // G) % 16 == 0:
                _prefetch_weight_rows(WSRows, first, 0, E * N, K // G, BW, K // G)
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
        acc = tl.zeros((BW, BT), tl.float32)
        for start in range(tl.cdiv(K, BK)):
            kp = start * PK + tl.arange(0, PK)
            ks = start * (BK // G) + tl.arange(0, BK // G)
            # Masked columns load zero codes, which add nothing.
            xmask = (m < count)[:, None] & (kp < KP)[None, :]
            xsmask = (m < count)[:, None] & (ks < K // G)[None, :]
            xv = tl.load(X + xrow[:, None] * KP + kp[None, :], xmask, other=0.0)
            xs = tl.load(XS + xrow[:, None] * (K // G) + ks[None, :], xsmask)
            if TMA:
                row = expert * N + (local % tiles_n) * BW
                wv = W.load([row, start * PK])
                ws = WS.load([row, start * (BK // G)])
            else:
                wmask = (n < N)[:, None] & (kp < KP)[None, :]
                wsmask = (n < N)[:, None] & (ks < K // G)[None, :]
                wv = tl.load(W + wrow[:, None] * KP + kp[None, :], wmask, other=0.0)
                ws = tl.load(WS + wrow[:, None] * (K // G) + ks[None, :], wsmask)
            if NVFP4:
                acc = tl.dot_scaled(wv, ws, "e2m1", tl.trans(xv), xs, "e2m1", acc)
            else:
                acc = tl.dot_scaled(wv, ws, "e4m3", tl.trans(xv), xs, "e4m3", acc)
        if NVFP4:
            xa = tl.load(XA + xrow, m < count, other=0.0)
            wa = tl.load(WA + expert)
            denominator = 1536.0 * tl.load(WD)
            scale = (xa * wa) * tl.div_rn(1.0, denominator)
            acc = acc * scale[None, :]
        tl.store(
            Y + route[None, :] * N + n[:, None],
            acc,
            (m < count)[None, :] & (n < N)[:, None],
        )
        tile += num_programs


def _scaled_tile_config(
    nvfp4: bool, k: int, route_count: int
) -> tuple[int, int, int, int, int, int, bool]:
    """(routed rows, weight rows, K, warps, stages, programs per SM, TMA) of
    an ``_expert_mm_scaled`` launch, measured on B300 decode and prefill
    shapes.

    Every choice sums each element alike, so the rows-dependent tier keeps
    results batch-invariant. Weight tiles stay at 128 rows (the scaled MMA's
    M) and routed tiles at 16 or more. MXFP8 scale tiles are too narrow for
    TMA.
    """
    decode = route_count <= 1024
    if not nvfp4:
        return (16 if decode else 64, 128, 256, 8 if decode else 4, 3, 2, False)
    if k >= 4096:
        return (16 if decode else 64, 128, 512, 4, 3, 2, True)
    return (32 if decode else 64, 128, 256, 4, 4, 3 if decode else 2, True)


def _launch_expert_mm(
    matmul: triton.JITFunction,
    tiles: tuple[int, int, int] | None,
    x: torch.Tensor,
    x_scale: torch.Tensor,
    x_amax: torch.Tensor | None,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    w_amax: torch.Tensor | None,
    w_denominator: torch.Tensor | None,
    out: torch.Tensor,
    routes: torch.Tensor,
    counts: torch.Tensor,
    route_count: int,
    programs: int,
    n: int,
    k: int,
    top_k: int,
    input_per_token: bool,
    fmt: str,
    routes_before_wait: bool,
    bf16_x: bool,
) -> None:
    """Launch ``matmul`` over the routes. With ``bf16_x``, ``x`` holds the BF16
    values of the quantized input (``x_scale`` is unused) and the kernel takes
    a ``BF16_X`` flag; only kernels with ``_expert_mm``'s tile arguments do."""
    nvfp4 = fmt == "nvfp4"
    experts = w.shape[0]
    extra = ()
    rows = ()
    pdl = False
    if bf16_x and matmul in (_expert_mm, _expert_mm_scaled):
        raise ValueError(f"{matmul.fn.__name__} reads quantized activations")
    if matmul is _expert_mm_scaled:
        bt, bn, bk, warps, stages, occupancy, tma = _scaled_tile_config(
            nvfp4, k, route_count
        )
        programs = programs // 2 * occupancy
        # TMA rows need 16-byte strides and bases.
        tma = tma and all(
            t.is_contiguous() and t.shape[-1] % 16 == 0 and t.data_ptr() % 16 == 0
            for t in (w, w_scale)
        )
        rows = (w, w_scale)
        if tma:
            # Payload and scale tiles of the [experts * N, K bytes] rows.
            w = TensorDescriptor.from_tensor(w.view(-1, w.shape[-1]), [bn, bk // 2])
            w_scale = TensorDescriptor.from_tensor(
                w_scale.view(-1, w_scale.shape[-1]), [bn, bk // 16]
            )
        pdl = pdl_enabled()
        extra = (tma, pdl, routes_before_wait)
    else:
        (bt, bn, bk), warps, stages = tiles, 4, 2
        if matmul is not _expert_mm:
            extra = (bf16_x,)
    matmul[(programs,)](
        x,
        x_scale,
        x_amax,
        w,
        w_scale,
        w_amax,
        w_denominator,
        out,
        routes,
        counts,
        *rows,
        route_count,
        programs,
        n,
        k,
        experts,
        top_k,
        input_per_token,
        nvfp4,
        bt,
        bn,
        bk,
        *extra,
        num_warps=warps,
        num_stages=stages,
        enable_fp_fusion=False,
        **({"launch_pdl": True} if pdl else {}),
    )


def _swiglu_quantize_routes(
    gate_up: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    output: torch.Tensor,
    second: torch.Tensor,
    second_scale: torch.Tensor,
    second_amax: torch.Tensor | None,
    second_dequant: torch.Tensor | None,
    *,
    second_format: str,
    expert_start: int,
    num_experts: int,
    gate_cap: float,
    up_cap: float,
    zero_remote_routes: bool,
    route_rows: torch.Tensor | None,
    w_amax: torch.Tensor | None,
    w_denominator: torch.Tensor | None,
    layout: int,
) -> None:
    """Launch every route's capped SwiGLU and second-input quantization.

    ``output`` ``[routes, hidden]`` holds the BF16 SwiGLU rows until the
    second projection overwrites them. ``route_rows`` (int32 ``[routes]``)
    maps routes to their expert-tiled rows of ``gate_up`` and of the
    quantized buffers, whose scales then go in ``layout`` (``_scale_offset``)
    and whose NVFP4 maxima become the per-row output scales of the second
    projection (weights ``w_amax`` and ``w_denominator``); None keeps route
    rows, maxima and row-major scales. A BF16 ``second_dequant`` (``second``'s
    unpacked shape, NVFP4 only) also receives the values the codes encode.
    """
    route_count, hidden = output.shape
    intermediate = gate_up.shape[1] // 2
    second_nvfp4 = second_format == "nvfp4"
    store_dequant = second_dequant is not None
    if store_dequant and not second_nvfp4:
        raise ValueError("a BF16 dequant copy needs an NVFP4 second input")
    # Unused pointer without a copy.
    second_dequant = second if second_dequant is None else second_dequant
    row_map = route_rows is not None
    row_programs = route_count > _CHUNK_PROGRAM_MAX_ROWS
    chunks = triton.cdiv(intermediate, 256)
    enable_pdl = pdl_enabled()
    pdl_launch = {"launch_pdl": True} if enable_pdl else {}
    ids = ids.contiguous()
    # Unused pointers of the route-row form.
    route_rows = ids if route_rows is None else route_rows
    w_amax = ids if w_amax is None else w_amax
    w_denominator = ids if w_denominator is None else w_denominator
    if second_nvfp4 and not row_programs and hidden >= intermediate:
        # Two launches evaluate each chunk once (64 x 2304 on B300: 5.6 -> 3.0 us).
        chunk_max = torch.empty(
            (route_count, chunks), device=output.device, dtype=torch.float32
        )
        _swiglu_chunk_max[(route_count, chunks)](
            gate_up,
            weights,
            ids,
            route_rows,
            output,
            chunk_max,
            expert_start,
            num_experts,
            intermediate,
            hidden,
            gate_cap,
            up_cap,
            256,
            row_map,
            enable_pdl,
            num_warps=4,
            enable_fp_fusion=False,
            **pdl_launch,
        )
        _quantize_swiglu_chunks[(route_count, chunks)](
            ids,
            route_rows,
            chunk_max,
            second,
            second_scale,
            second_amax,
            second_dequant,
            w_amax,
            w_denominator,
            output,
            expert_start,
            num_experts,
            layout,
            intermediate,
            hidden,
            256,
            zero_remote_routes,
            row_map,
            enable_pdl,
            store_dequant,
            num_warps=4,
            enable_fp_fusion=False,
            **pdl_launch,
        )
    else:
        _swiglu_quantize[(route_count, 1 if row_programs else chunks)](
            gate_up,
            weights,
            ids,
            route_rows,
            second,
            second_scale,
            second if second_amax is None else second_amax,
            second_dequant,
            w_amax,
            w_denominator,
            output,
            expert_start,
            num_experts,
            layout,
            intermediate,
            hidden,
            gate_cap,
            up_cap,
            second_nvfp4,
            256,
            row_programs,
            zero_remote_routes,
            row_map,
            enable_pdl,
            store_dequant,
            # Row programs walk 256-column chunks: two warps run them fastest.
            num_warps=2 if row_programs else 8,
            enable_fp_fusion=False,
            **pdl_launch,
        )


@triton.jit(do_not_specialize=["tokens"])
def _fold_routes_kernel(
    Rows,
    RouteRows,
    Ids,
    Output,
    tokens,
    expert_start,
    num_experts,
    hidden: tl.constexpr,
    TOP_K: tl.constexpr,
    TOKENS: tl.constexpr,
    BLOCK: tl.constexpr,
    ROW_MAP: tl.constexpr,
    PUBLISH: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    """Sum ``TOKENS`` tokens' local route rows over columns ``BLOCK`` in FP32
    slot order from +0.0 and round once to BF16; a route of another rank's
    expert, or of none, adds nothing. With ``ROW_MAP`` a route's row is
    ``RouteRows[route]`` (an expert-tiled GEMM row), else the route itself.
    With ``PUBLISH`` ``Output`` is a mailbox slot (``ops.communication.mailbox``)
    and ``-0.0`` sums are stored as ``+0.0``."""
    token = tl.program_id(0).to(tl.int64) * TOKENS + tl.arange(0, TOKENS)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = token < tokens
    slot = tl.arange(0, triton.next_power_of_2(TOP_K))
    routes = token[:, None] * TOP_K + slot[None, :]
    # The routing is older than the expert GEMMs: read it before the wait.
    experts = (
        tl.load(Ids + routes, valid[:, None] & (slot < TOP_K)[None, :], other=-1)
        - expert_start
    )
    if ROW_MAP:
        # So are the routes' expert-tiled rows.
        tiled = tl.load(
            RouteRows + routes,
            valid[:, None] & (experts >= 0) & (experts < num_experts),
            other=0,
        )
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    acc = tl.zeros((TOKENS, BLOCK), tl.float32)
    for k in tl.static_range(TOP_K):
        expert = tl.sum(tl.where(slot[None, :] == k, experts, 0), 1)
        local = valid & (expert >= 0) & (expert < num_experts)
        row = token * TOP_K + k
        if ROW_MAP:
            row = tl.sum(tl.where(slot[None, :] == k, tiled, 0), 1).to(tl.int64)
        acc += tl.load(
            Rows + row[:, None] * hidden + col[None, :],
            local[:, None] & (col < hidden)[None, :],
            other=0.0,
        ).to(tl.float32)
    out = acc.to(tl.bfloat16)
    if PUBLISH:
        # An armed mailbox slot holds -0.0: producers never publish it.
        bits = out.to(tl.int16, bitcast=True)
        out = tl.where(bits == -32768, 0, bits).to(tl.bfloat16, bitcast=True)
    tl.store(
        Output + token[:, None] * hidden + col[None, :],
        out,
        valid[:, None] & (col < hidden)[None, :],
    )


def _fold_routes(
    rows: torch.Tensor,
    route_rows: torch.Tensor | None,
    ids: torch.Tensor,
    *,
    expert_start: int,
    num_experts: int,
    fold_into: MailboxTarget | torch.Tensor | None,
) -> torch.Tensor | None:
    """BF16 ``[T,1,H]``: each token's local route rows summed in FP32 slot
    order and rounded once. ``rows`` holds the routes' rows at their route
    index, or with ``route_rows`` at ``route_rows[route]``. A mailbox
    ``fold_into`` has the sums published into that parity instead (returns
    None); a tensor ``fold_into`` (contiguous BF16 ``[T,H]``) receives them
    (returns its ``[T,1,H]`` view)."""
    tokens, top_k = ids.shape
    hidden = rows.shape[-1]
    publish = isinstance(fold_into, MailboxTarget)
    if publish and (
        fold_into.mailbox.width != hidden or not 0 < tokens <= MAILBOX_ROWS
    ):
        raise ValueError(
            f"a published fold takes 1-{MAILBOX_ROWS} tokens of the mailbox's width"
        )
    if isinstance(fold_into, torch.Tensor):
        if (
            fold_into.shape != (tokens, hidden)
            or fold_into.dtype != torch.bfloat16
            or not fold_into.is_contiguous()
        ):
            raise ValueError("a fold destination is a contiguous BF16 [T, H] tensor")
        output = fold_into.view(tokens, 1, hidden)
        if not tokens:
            return output
    elif fold_into is None:
        output = rows.new_empty((tokens, 1, hidden))
        if not tokens:
            return output
    else:
        # This rank's slot of every rank's mailbox, through the multicast address.
        mailbox = fold_into.mailbox
        output = mailbox.multicast_slots(fold_into.parity)[mailbox.rank]
    enable_pdl = pdl_enabled()
    # Two tokens by 512 columns per program keeps a decode batch's fold on
    # enough SMs to hide the row loads' latency, and suits prefill as well.
    _fold_routes_kernel[(triton.cdiv(tokens, 2), triton.cdiv(hidden, 512))](
        rows,
        rows if route_rows is None else route_rows,
        ids.contiguous(),
        output,
        tokens,
        expert_start,
        num_experts,
        hidden,
        top_k,
        2,
        512,
        route_rows is not None,
        publish,
        enable_pdl,
        num_warps=4,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return None if publish else output


def _block_scaled_moe_routes(
    x: torch.Tensor,
    w13: torch.Tensor,
    s13: torch.Tensor,
    a13: torch.Tensor | None,
    d13: torch.Tensor | None,
    w2: torch.Tensor,
    s2: torch.Tensor,
    a2: torch.Tensor | None,
    d2: torch.Tensor | None,
    weights: torch.Tensor,
    ids: torch.Tensor,
    *,
    formats: tuple[str, str],
    expert_start: int,
    gate_cap: float,
    up_cap: float,
    zero_remote_routes: bool,
    fold_slots: bool,
    fold_into: MailboxTarget | torch.Tensor | None,
    matmul: triton.JITFunction,
    tiles: tuple[int, int, int] | None,
    bf16_activations: bool,
) -> torch.Tensor | None:
    """Share quantization and BF16 boundaries across expert GEMM solutions.

    ``matmul`` is ``_expert_mm_scaled``, which picks its own tiles (``tiles``
    is None), or a kernel with ``_expert_mm``'s arguments run with
    ``tiles = (BM, BN, BK)``. With ``bf16_activations`` (NVFP4 experts, a
    kernel that takes ``BF16_X``) the quantizers also store the exact BF16
    values of both quantized inputs, and ``matmul`` reads those instead of
    decoding the NVFP4 codes for every output tile.
    """
    if not x.is_cuda:
        raise ValueError("Triton block-scaled experts require GPU tensors")
    tokens, hidden = x.shape
    top_k = ids.shape[1]
    route_count = ids.numel()
    intermediate = w13.shape[1] // 2
    if not tokens:
        return x.new_empty((0, 1 if fold_slots else top_k, hidden))
    if bf16_activations and formats != ("nvfp4", "nvfp4"):
        raise ValueError(f"BF16 activations need NVFP4 experts, not {formats}")
    first_dequant = torch.empty_like(x) if bf16_activations else None
    (routes, counts), (first, first_scale, first_amax) = _routed_quantized_input(
        x, ids, w13.shape[0], expert_start, formats[0], first_dequant
    )
    # Remote routes skip gate_up; zero_remote_routes zeroes their output rows.
    gate_up = torch.empty(
        (route_count, 2 * intermediate), device=x.device, dtype=torch.bfloat16
    )
    output = torch.empty((route_count, hidden), device=x.device, dtype=torch.bfloat16)
    programs = torch.cuda.get_device_properties(x.device).multi_processor_count * 2
    _launch_expert_mm(
        matmul,
        tiles,
        first if first_dequant is None else first_dequant,
        first_scale,
        first_amax,
        w13,
        s13,
        a13,
        d13,
        gate_up,
        routes,
        counts,
        route_count,
        programs,
        2 * intermediate,
        hidden,
        top_k,
        True,
        formats[0],
        False,
        bf16_activations,
    )
    second, second_scale, second_amax = _quantized_buffers(
        route_count, intermediate, formats[1] == "nvfp4", x.device
    )
    second_dequant = (
        torch.empty((route_count, intermediate), device=x.device, dtype=torch.bfloat16)
        if bf16_activations
        else None
    )
    _swiglu_quantize_routes(
        gate_up,
        weights,
        ids,
        output,
        second,
        second_scale,
        second_amax,
        second_dequant,
        second_format=formats[1],
        expert_start=expert_start,
        num_experts=w13.shape[0],
        gate_cap=gate_cap,
        up_cap=up_cap,
        zero_remote_routes=zero_remote_routes,
        route_rows=None,
        w_amax=None,
        w_denominator=None,
        layout=0,
    )
    _launch_expert_mm(
        matmul,
        tiles,
        second if second_dequant is None else second_dequant,
        second_scale,
        second_amax,
        w2,
        s2,
        a2,
        d2,
        output,
        routes,
        counts,
        route_count,
        programs,
        hidden,
        intermediate,
        top_k,
        False,
        formats[1],
        # The first projection waits before it releases the SwiGLU kernels.
        True,
        bf16_activations,
    )
    if fold_slots:
        return _fold_routes(
            output,
            None,
            ids,
            expert_start=expert_start,
            num_experts=w13.shape[0],
            fold_into=fold_into,
        )
    return output.view(tokens, top_k, hidden)


_ROUTE_TRAITS = {
    "w13_weight_dtype": frozenset({"nvfp4", "mxfp8"}),
    "w2_weight_dtype": frozenset({"nvfp4", "mxfp8"}),
}


@register_kernel(
    "moe",
    "block_scaled_routes",
    name="triton_block_scaled_moe_routes",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset({format_signature(x=dense_tensor_format(torch.bfloat16))}),
    traits=_ROUTE_TRAITS,
    priority=Priority.PORTABLE,
)
def triton_block_scaled_moe_routes(
    x: torch.Tensor,
    w13: torch.Tensor,
    s13: torch.Tensor,
    a13: torch.Tensor | None,
    d13: torch.Tensor | None,
    w2: torch.Tensor,
    s2: torch.Tensor,
    a2: torch.Tensor | None,
    d2: torch.Tensor | None,
    weights: torch.Tensor,
    ids: torch.Tensor,
    *,
    formats: tuple[str, str],
    experts: int,
    expert_start: int,
    gate_cap: float,
    up_cap: float,
    zero_remote_routes: bool,
    fold_slots: bool,
    fold_into: MailboxTarget | torch.Tensor | None,
) -> torch.Tensor | None:
    del experts  # per-expert tiles need no global count
    return _block_scaled_moe_routes(
        x,
        w13,
        s13,
        a13,
        d13,
        w2,
        s2,
        a2,
        d2,
        weights,
        ids,
        formats=formats,
        expert_start=expert_start,
        gate_cap=gate_cap,
        up_cap=up_cap,
        zero_remote_routes=zero_remote_routes,
        fold_slots=fold_slots,
        fold_into=fold_into,
        matmul=_expert_mm,
        tiles=(16, 64, 64),
        bf16_activations=False,
    )


@register_kernel(
    "moe",
    "block_scaled_routes",
    name="triton_scaled_dot_block_scaled_moe_routes",
    solution="triton",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
    ),
    signatures=frozenset({format_signature(x=dense_tensor_format(torch.bfloat16))}),
    traits=_ROUTE_TRAITS,
    priority=Priority.PERFORMANT,
)
def triton_scaled_dot_block_scaled_moe_routes(
    x: torch.Tensor,
    w13: torch.Tensor,
    s13: torch.Tensor,
    a13: torch.Tensor | None,
    d13: torch.Tensor | None,
    w2: torch.Tensor,
    s2: torch.Tensor,
    a2: torch.Tensor | None,
    d2: torch.Tensor | None,
    weights: torch.Tensor,
    ids: torch.Tensor,
    *,
    formats: tuple[str, str],
    experts: int,
    expert_start: int,
    gate_cap: float,
    up_cap: float,
    zero_remote_routes: bool,
    fold_slots: bool,
    fold_into: MailboxTarget | torch.Tensor | None,
) -> torch.Tensor | None:
    """The same contract on Blackwell block-scaled tensor cores."""
    del experts  # per-expert tiles need no global count
    return _block_scaled_moe_routes(
        x,
        w13,
        s13,
        a13,
        d13,
        w2,
        s2,
        a2,
        d2,
        weights,
        ids,
        formats=formats,
        expert_start=expert_start,
        gate_cap=gate_cap,
        up_cap=up_cap,
        zero_remote_routes=zero_remote_routes,
        fold_slots=fold_slots,
        fold_into=fold_into,
        matmul=_expert_mm_scaled,
        tiles=None,
        bf16_activations=False,
    )


@triton.jit
def _topk(
    Logits,
    Bias,
    Replicas,
    NumReplicas,
    Weights,
    Ids,
    replica_width,
    E: tl.constexpr,
    TOP_K: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
    PAD_K: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    row = tl.program_id(0)
    expert = tl.arange(0, BLOCK)
    if Replicas is not None:
        # Placement tables change only between forwards: read them before the wait.
        first = tl.load(Replicas + expert * replica_width, expert < E, other=0)
        counts = tl.load(NumReplicas + expert, expert < E, other=1)
    # The bias is a constant: read it before the wait.
    bias = tl.load(Bias + expert, expert < E, other=0.0)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    logits = tl.load(Logits + row * E + expert, expert < E, other=-float("inf"))
    # torch.sigmoid's expf and correctly rounded divide, as the reference selects.
    score = tl.div_rn(1.0, 1.0 + libdevice.exp(-logits)) + bias
    score = tl.where(expert < E, score, -float("inf"))
    score = tl.where(score == 0.0, 0.0, score)
    bits = score.to(tl.uint32, bitcast=True)
    ordered = bits ^ tl.where((bits & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000).to(
        tl.uint32
    )
    keys = (ordered.to(tl.uint64) << 32) | (BLOCK - expert).to(tl.uint64)
    keys = tl.where(expert < E, keys, 0)
    selected = tl.topk(keys, PAD_K, dim=0)
    ids = (BLOCK - (selected & 0xFFFFFFFF).to(tl.int32)).to(tl.int32)
    slot = tl.arange(0, PAD_K)
    # The selected logits come from registers, not a second dependent load.
    picked = tl.where(expert[None, :] == ids[:, None], logits[None, :], 0.0)
    selected_logits = tl.where(slot < TOP_K, tl.sum(picked, 1), -float("inf"))
    log_probs = tl.minimum(selected_logits, 0.0) - libdevice.log1p(
        libdevice.exp(-tl.abs(selected_logits))
    )
    maximum = tl.max(log_probs, 0)
    shifted = libdevice.exp(log_probs - maximum)
    summed = tl.sum(tl.where(slot < TOP_K, shifted, 0.0), 0)
    lse = maximum + libdevice.log(summed)
    weights = libdevice.exp(log_probs - lse) * SCALE
    if Replicas is not None:
        # Replica (row + slot) % count as in dispatch_topk_ids; only later ones load.
        hit = expert[None, :] == ids[:, None]
        physical = tl.sum(tl.where(hit, first[None, :], 0), 1)
        if replica_width > 1:
            count = tl.sum(tl.where(hit, counts[None, :], 0), 1)
            choice = (row + slot) % count
            later = tl.load(
                Replicas + ids * replica_width + choice,
                (slot < TOP_K) & (choice != 0),
                other=0,
            )
            physical = tl.where(choice != 0, later, physical)
        ids = physical
    tl.store(Weights + row * TOP_K + slot, weights, slot < TOP_K)
    tl.store(Ids + row * TOP_K + slot, ids, slot < TOP_K)


@register_kernel(
    "moe",
    "block_scaled_topk",
    name="triton_block_scaled_topk",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset({format_signature(logits=dense_tensor_format(torch.float32))}),
    priority=Priority.PORTABLE,
)
def triton_block_scaled_topk(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    top_k: int,
    route_scale: float,
    dispatch: ExpertDispatch | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Under PDL the routing that follows reads nothing before its wait.
    if not logits.is_cuda or not logits.is_contiguous() or not bias.is_contiguous():
        raise ValueError("Triton router requires contiguous GPU operands")
    tokens, experts = logits.shape
    weights = torch.empty((tokens, top_k), device=logits.device, dtype=torch.float32)
    ids = torch.empty((tokens, top_k), device=logits.device, dtype=torch.int32)
    if tokens:
        enable_pdl = pdl_enabled()
        _topk[(tokens,)](
            logits,
            bias,
            None if dispatch is None else dispatch.replicas,
            None if dispatch is None else dispatch.num_replicas,
            weights,
            ids,
            1 if dispatch is None else dispatch.max_replicas,
            experts,
            top_k,
            route_scale,
            triton.next_power_of_2(experts),
            triton.next_power_of_2(top_k),
            enable_pdl,
            enable_fp_fusion=False,
            **({"launch_pdl": True} if enable_pdl else {}),
        )
    return weights, ids
