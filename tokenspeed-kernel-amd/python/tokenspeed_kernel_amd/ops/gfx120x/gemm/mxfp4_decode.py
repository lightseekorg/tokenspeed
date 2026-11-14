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

"""MXFP4 decode GEMM for 1-3 rows on RDNA4.

The activations are quantized to MXFP4 once, with E2M1 expanded exactly to
integers (2x); the GEMM expands the weights the same way, reduces with packed
int8 dot products, and scales each 32-element group by its E8M0 exponents.
"""

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton


def _launch_metadata(grid, kernel, args):
    """Expose algorithmic FLOPs and tensor traffic to Proton."""
    m, n, k = args["M"], args["N"], args["K"]
    return {
        "name": kernel.name,
        "flops4": 2 * m * n * k,
        "bytes": m * k + n * k // 2 + (m + n) * k // 32 + m * n * 2,
    }


@gluon.jit
def _perm(first, second, index):
    return gl.inline_asm_elementwise(
        "v_perm_b32 $0, $1, $2, $3;",
        constraints="=v,v,v,v",
        args=[first, second, index],
        dtype=gl.uint32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _signed_e2m1_x2(nibbles):
    index = nibbles & 0x07070707
    # E2M1 * 2: [0, 1, 2, 3, 4, 6, 8, 12], including signed zero.
    positive = _perm(0x0C080604, 0x03020100, index)
    negative = _perm(0xF4F8FAFC, 0xFDFEFF00, index)
    sign = ((nibbles & 0x08080808) >> 3) * 255
    return (positive & ~sign) | (negative & sign)


@gluon.jit
def _unpack(word):
    return _signed_e2m1_x2(word & 0x0F0F0F0F), _signed_e2m1_x2((word >> 4) & 0x0F0F0F0F)


@gluon.jit
def _dot8(a0, a1, w0, w1):
    return gl.inline_asm_elementwise(
        "v_dot4_i32_iu8 $0, $1, $2, 0 neg_lo:[1,1,0];\n"
        "v_dot4_i32_iu8 $0, $3, $4, $0 neg_lo:[1,1,0];",
        constraints="=&v,v,v,v,v",
        args=[a0, w0, a1, w1],
        dtype=gl.int32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _quantize(vals):
    """``[G, 4, 8]`` FP32 groups to (``[G, 4]`` packed E2M1 words, ``[G]``
    UE8M0), bit for bit as ``mxfp4_quantize``: the scale exponent is the group
    amax rounded up at the third mantissa bit, minus two, clamped to [-127,
    127]; values round to nearest even E2M1 and saturate at 6."""
    amax = gl.max(gl.max(gl.abs(vals), 2), 1)
    amax = (
        (amax.to(gl.int32, bitcast=True) + 0x200000).to(gl.uint32, bitcast=True)
        & 0xFF800000
    ).to(gl.float32, bitcast=True)
    e = gl.minimum(gl.maximum(gl.floor(gl.log2(amax)) - 2, -127.0), 127.0)
    qx = (vals * gl.exp2(-e)[:, None, None]).to(gl.uint32, bitcast=True)
    sign = qx & 0x80000000
    qx = qx ^ sign
    qf = qx.to(gl.float32, bitcast=True)
    saturate = qf >= 6.0
    denormal = (not saturate) & (qf < 1.0)
    normal = not (saturate | denormal)
    sub = ((qf + 4194304.0).to(gl.uint32, bitcast=True) - (149 << 23)) & 0xFF
    nor = ((qx + 0xC11FFFFF + ((qx >> 22) & 1)) >> 22) & 0xFF
    code = gl.where(normal, nor, gl.where(denormal, sub, 7)) | (sign >> 28)
    shift = gl.arange(0, 8, gl.SliceLayout(0, gl.SliceLayout(1, vals.type.layout)))
    words = gl.sum((code & 0xF) << (shift * 4)[None, None, :], 2)
    return words, (e + 127.0).to(gl.int32)


# The row count and the input's row stride (row views of wider tensors)
# vary, so they stay runtime arguments.
@gluon.jit(do_not_specialize=("M", "XM"))
def gluon_mm_mxfp4_decode_quantize_gfx120x(
    X,
    Out,
    Scales,
    M,
    XM,
    K: gl.constexpr,
    OM: gl.constexpr,
    SM: gl.constexpr,
    GROUPS: gl.constexpr,
):
    """Quantize ``GROUPS`` groups of one row of BF16 ``X`` to MXFP4 (as
    ``mxfp4_quantize``) and store them expanded, ``[lo | hi]`` int8 words,
    as the decode GEMM's unpacked mode reads them."""
    L: gl.constexpr = gl.BlockedLayout(
        [1, 4, 8], [32, 1, 1], [gl.num_warps(), 1, 1], [2, 1, 0]
    )
    WL: gl.constexpr = gl.SliceLayout(2, L)
    row = gl.program_id(1)
    g = gl.program_id(0) * GROUPS + gl.arange(0, GROUPS, gl.SliceLayout(1, WL))
    w = gl.arange(0, 4, gl.SliceLayout(0, WL))
    i = gl.arange(0, 8, gl.SliceLayout(0, gl.SliceLayout(1, L)))
    valid = g < K // 32
    x = gl.load(
        X
        + row.to(gl.int64) * XM
        + (g[:, None] * 32 + w[None, :] * 8)[:, :, None]
        + i[None, None, :],
        mask=valid[:, None, None],
        other=0.0,
    )
    words, scales = _quantize(x.to(gl.float32))
    index = g[:, None] * 4 + w[None, :]
    low, high = _unpack(words)
    gl.store(Out + row * OM + index, low, valid[:, None])
    gl.store(Out + row * OM + K // 8 + index, high, valid[:, None])
    gl.store(Scales + row * SM + g, scales.to(gl.uint8), valid)


@gluon.jit
def _accumulate(
    A,
    AS,
    m: gl.constexpr,
    words,
    groups,
    w0,
    w1,
    ws,
    acc,
    K: gl.constexpr,
    AM: gl.constexpr,
    ASM: gl.constexpr,
):
    a0 = gl.load(A + m * AM + words, words < K // 8, 0).to(gl.uint32)
    a1 = gl.load(A + m * AM + K // 8 + words, words < K // 8, 0).to(gl.uint32)
    scale = gl.load(AS + m * ASM + groups, groups < K // 32, 127).to(gl.int32)
    dot32 = gl.sum(_dot8(a0[None, :, :], a1[None, :, :], w0, w1), 2).to(gl.float32)
    # The 2x of both operands is removed with the -2 in the bias (127 + 127 + 2).
    value = gl.inline_asm_elementwise(
        "v_ldexp_f32 $0, $1, $2;",
        constraints="=v,v,v",
        args=[dot32, scale[None, :] + ws - 256],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
    return acc + gl.where((scale[None, :] == 255) | (ws == 255), float("nan"), value)


@gluon.jit
def _mxfp4_decode_rows(
    A,
    W,
    AS,
    WS,
    C,
    N: gl.constexpr,
    K: gl.constexpr,
    AM: gl.constexpr,
    WN: gl.constexpr,
    ASM: gl.constexpr,
    WSN: gl.constexpr,
    CM: gl.constexpr,
    CN: gl.constexpr,
    ROWS: gl.constexpr,
    WORDS: gl.constexpr,
    M: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 4], [1, 32, 1], [4, 1, 1], [2, 1, 0])
    group_layout: gl.constexpr = gl.SliceLayout(2, layout)
    word_layout: gl.constexpr = gl.SliceLayout(0, layout)
    rows = gl.program_id(0) * ROWS + gl.arange(
        0, ROWS, layout=gl.SliceLayout(1, group_layout)
    )
    group_index = gl.arange(0, WORDS // 4, layout=gl.SliceLayout(0, group_layout))
    local_words = gl.arange(0, 4, layout=gl.SliceLayout(0, word_layout))
    acc0 = gl.full((ROWS, WORDS // 4), 0, gl.float32, group_layout)
    acc1 = gl.full((ROWS, WORDS // 4), 0, gl.float32, group_layout)
    acc2 = gl.full((ROWS, WORDS // 4), 0, gl.float32, group_layout)
    for base in range(0, K // 32, WORDS // 4):
        groups = base + group_index
        word_groups = gl.convert_layout(groups, gl.SliceLayout(1, word_layout))
        words = word_groups[:, None] * 4 + local_words[None, :]
        packed = gl.load(
            W + rows[:, None, None] * WN + words[None, :, :],
            (rows[:, None, None] < N) & (words[None, :, :] < K // 8),
            0,
        ).to(gl.uint32)
        ws = gl.load(
            WS + rows[:, None] * WSN + groups[None, :],
            (rows[:, None] < N) & (groups[None, :] < K // 32),
            127,
        ).to(gl.int32)
        w0, w1 = _unpack(packed)
        acc0 = _accumulate(A, AS, 0, words, groups, w0, w1, ws, acc0, K, AM, ASM)
        if M >= 2:
            acc1 = _accumulate(A, AS, 1, words, groups, w0, w1, ws, acc1, K, AM, ASM)
        if M >= 3:
            acc2 = _accumulate(A, AS, 2, words, groups, w0, w1, ws, acc2, K, AM, ASM)
    gl.store(C + rows * CN, gl.sum(acc0, 1), rows < N)
    if M >= 2:
        gl.store(C + CM + rows * CN, gl.sum(acc1, 1), rows < N)
    if M >= 3:
        gl.store(C + 2 * CM + rows * CN, gl.sum(acc2, 1), rows < N)


@gluon.jit(launch_metadata=_launch_metadata, do_not_specialize=("M",))
def gluon_mm_mxfp4_decode_gfx120x(
    A,
    W,
    AS,
    WS,
    C,
    M,
    N: gl.constexpr,
    K: gl.constexpr,
    AM: gl.constexpr,
    WN: gl.constexpr,
    ASM: gl.constexpr,
    WSN: gl.constexpr,
    CM: gl.constexpr,
    CN: gl.constexpr,
    ROWS: gl.constexpr,
    WORDS: gl.constexpr,
):
    # One binary serves 1-3 rows: the runtime row count selects a
    # specialization, so a new batch size never recompiles while serving.
    if M == 1:
        _mxfp4_decode_rows(
            A, W, AS, WS, C, N, K, AM, WN, ASM, WSN, CM, CN, ROWS, WORDS, 1
        )
    elif M == 2:
        _mxfp4_decode_rows(
            A, W, AS, WS, C, N, K, AM, WN, ASM, WSN, CM, CN, ROWS, WORDS, 2
        )
    else:
        _mxfp4_decode_rows(
            A, W, AS, WS, C, N, K, AM, WN, ASM, WSN, CM, CN, ROWS, WORDS, 3
        )


def launch_gluon_mm_mxfp4_decode_gfx120x(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``mxfp4(x) @ weight.T`` for 1-3 BF16 rows on RDNA4.

    One kernel quantizes ``x`` as ``mxfp4_quantize`` does, straight into
    E2M1 expanded to int8 (``[lo | hi]`` words), then the GEMM reads them.

    Args:
        x: BF16 activations ``[M, K]``, ``1 <= M <= 3``, unit inner stride,
            ``K`` a multiple of 32.
        weight: Packed E2M1 weight ``[N, K/2]`` uint8.
        weight_scale: UE8M0 weight scales ``[N, K/32]`` uint8.
        out: Optional ``[M, N]`` BF16 destination.

    Returns:
        ``[M, N]`` BF16; ``out`` when provided.
    """
    m, k = x.shape
    n = weight.shape[0]
    if not 1 <= m <= 3 or k != weight.shape[1] * 2 or k % 32 or x.stride(-1) != 1:
        raise ValueError("MXFP4 decode requires 1-3 unit-stride rows and matching K")
    if weight.dtype != torch.uint8 or weight_scale.dtype != torch.uint8:
        raise TypeError("MXFP4 decode expects packed uint8 weights and E8M0 scales")
    words = torch.empty((m, k // 4), dtype=torch.int32, device=x.device)
    scales = torch.empty((m, k // 32), dtype=torch.uint8, device=x.device)
    groups = 16
    gluon_mm_mxfp4_decode_quantize_gfx120x[(triton.cdiv(k // 32, groups), m)](
        x,
        words,
        scales,
        m,
        x.stride(0),
        k,
        words.stride(0),
        scales.stride(0),
        groups,
        num_warps=2,
    )
    result = (
        out if out is not None else torch.empty((m, n), dtype=x.dtype, device=x.device)
    )
    wv = weight.view(torch.int32)
    gluon_mm_mxfp4_decode_gfx120x[(triton.cdiv(n, 4),)](
        words,
        wv,
        scales,
        weight_scale,
        result,
        m,
        n,
        k,
        words.stride(0),
        wv.stride(0),
        scales.stride(0),
        weight_scale.stride(0),
        result.stride(0),
        result.stride(1),
        ROWS=4,
        WORDS=128,
        num_warps=4,
    )
    return result
