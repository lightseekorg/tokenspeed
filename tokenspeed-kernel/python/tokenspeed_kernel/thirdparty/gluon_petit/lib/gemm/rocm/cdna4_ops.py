# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""CDNA4 operations with Petit's existing per-lane register views."""

import triton.experimental.gluon as g
from triton.experimental.gluon import language as l
from triton.experimental.gluon.language.amd import cdna4


@g.jit
def join_words(words):
    """View four consecutive registers as the last tensor dimension."""
    return l.join(l.join(words[0], words[2]), l.join(words[1], words[3]))


@g.jit
def split_words(value, lane_layout: l.constexpr):
    """Unpack the last four-value dimension into four tensors in lane_layout."""
    even, odd = l.split(value.reshape([value.numel // 4, 2, 2]))
    x, z = l.split(even)
    y, w = l.split(odd)
    return (
        l.convert_layout(x, lane_layout),
        l.convert_layout(y, lane_layout),
        l.convert_layout(z, lane_layout),
        l.convert_layout(w, lane_layout),
    )


@g.constexpr_function
def _log2(value):
    assert value > 0 and value & (value - 1) == 0
    return value.bit_length() - 1


@g.jit
def _join_fragments(fragments):
    NW: l.constexpr = l.num_warps()
    parts = ()
    for i in l.static_range(len(fragments)):
        parts += (join_words(fragments[i]).reshape([NW * 64, 1, 4]),)
    for step in l.static_range(_log2(len(fragments))):
        joined = ()
        for i in l.static_range(len(parts) // 2):
            joined += (
                l.join(parts[2 * i], parts[2 * i + 1])
                .permute([0, 3, 1, 2])
                .reshape([NW * 64, 2 << step, 4]),
            )
        parts = joined
    return parts[0]


@g.jit
def _tile_operand(fragments, operand: l.constexpr, parent: l.constexpr):
    NW: l.constexpr = l.num_warps()
    repeats: l.constexpr = len(fragments) // 2
    packed = _join_fragments(fragments)
    value = l.join(
        l.join(packed.to(l.uint8), (packed >> 16).to(l.uint8)),
        l.join((packed >> 8).to(l.uint8), (packed >> 24).to(l.uint8)),
    ).reshape([NW, 4, 16, repeats, 2, 16])
    if operand == 0:
        value = value.permute([0, 3, 2, 4, 1, 5]).reshape([NW, repeats * 16, 128])
    else:
        value = value.permute([0, 4, 1, 5, 3, 2]).reshape([NW, 128, repeats * 16])
    return l.convert_layout(value, l.DotOperandLayout(operand, parent, 16))


@g.constexpr_function
def _tile_scale_layout(warps, rows):
    # CDNA4 packs K repeats before row repeats. Keep the wave batch explicit;
    # Triton 3.8's scale-layout factory only handles unbatched matrices.
    return l.DistributedLinearLayout(
        reg_bases=[[0, 0, 4], [0, 16, 0]]
        + [[0, 32 << i, 0] for i in range((rows // 32).bit_length() - 1)],
        lane_bases=[[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 0, 1], [0, 0, 2]],
        warp_bases=[[1 << i, 0, 0] for i in range(warps.bit_length() - 1)],
        block_bases=[],
        shape=[warps, rows, 8],
    )


@g.jit
def _tile_scales(scales):
    NW: l.constexpr = l.num_warps()
    parts = ()
    for i in l.static_range(len(scales)):
        parts += (scales[i].reshape([NW * 64, 1]),)
    for step in l.static_range(_log2(len(scales))):
        joined = ()
        for i in l.static_range(len(parts) // 2):
            joined += (
                l.join(parts[2 * i], parts[2 * i + 1])
                .permute([0, 2, 1])
                .reshape([NW * 64, 2 << step]),
            )
        parts = joined
    packed = parts[0]
    value = l.join(
        l.join(packed.to(l.uint8), (packed >> 16).to(l.uint8)),
        l.join((packed >> 8).to(l.uint8), (packed >> 24).to(l.uint8)),
    ).reshape([NW, 4, 16, len(scales), 2, 2])
    value = value.permute([0, 3, 5, 2, 4, 1]).reshape([NW, len(scales) * 32, 8])
    target: l.constexpr = _tile_scale_layout(NW, len(scales) * 32)
    return l.convert_layout(value, target)


@g.jit
def scaled_mfma_tile(acc, w, x, scale_x, scale_w, M: l.constexpr, N: l.constexpr):
    """Multiply the existing per-wave tile through one tensor MFMA operation.

    M and N count 16-element activation and weight fragments. The batch axis
    assigns one independent tile to each wave; K remains 256 in its original
    accumulation order. Each fragment contains four per-lane register tensors:
    acc has N*M FP32 fragments, w has two K groups of N packed FP4 fragments,
    and x has M*2 packed FP4 fragments. scale_x and scale_w contain M/2 and N/2
    packed E8M0 scale tensors, respectively. Return FP32 fragments with the
    same tuple indexing and lane layout as acc.
    """
    NW: l.constexpr = l.num_warps()
    parent: l.constexpr = cdna4.AMDMFMALayout(4, [16, 16, 128], False, [NW, 1, 1])
    weight = ()
    for n in l.static_range(N):
        for k in l.static_range(2):
            weight += (w[k][n],)
    lhs = _tile_operand(weight, 0, parent)
    rhs = _tile_operand(x, 1, parent)
    sa = _tile_scales(scale_w)
    sb = _tile_scales(scale_x)
    c = (
        _join_fragments(acc)
        .reshape([NW, 4, 16, N, M, 4])
        .permute([0, 3, 1, 5, 4, 2])
        .reshape([NW, N * 16, M * 16])
    )
    c = l.convert_layout(c, parent)
    result = cdna4.mfma_scaled(lhs, sa, "e2m1", rhs, sb, "e2m1", c)
    result = (
        result.reshape([NW, N, 4, 4, M, 16])
        .permute([0, 2, 5, 1, 4, 3])
        .reshape([NW * 64, N * M, 4])
    )
    result = l.convert_layout(
        result, l.BlockedLayout([1, N * M, 4], [64, 1, 1], [NW, 1, 1], [2, 1, 0])
    )
    pieces = (result,)
    for step in l.static_range(_log2(N * M)):
        split = ()
        for i in l.static_range(len(pieces)):
            even, odd = l.split(
                pieces[i]
                .reshape([NW * 64, 2, (N * M) >> (step + 1), 4])
                .permute([0, 2, 3, 1])
            )
            split += (even, odd)
        pieces = split
    out = ()
    for i in l.static_range(N * M):
        out += (
            split_words(
                pieces[i].reshape([NW * 64, 4]), l.BlockedLayout([1], [64], [NW], [0])
            ),
        )
    return out
