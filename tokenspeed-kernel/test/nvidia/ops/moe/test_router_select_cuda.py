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

"""``triton_router_route_lanepar`` on an NVIDIA device: Beam's route in one launch
against ``torch_grouped_route``, byte for byte (ids strict; weights strict on random
rows and under the NaN rule on the tie and edge corpora), in eager runs and
CUDA-graph replays. Two pins come
first and fail rather than skip: Torch's FP32 CUDA sigmoid against the kernel's
recipe (``_sigmoid_rn``) on an enumerated edge corpus, and Torch's CUDA K8 sum
against the reduction tree the kernel spells.
"""

from __future__ import annotations

import math
import struct

import pytest
import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.moe._triton.grouped_select import _LAUNCH_OPTIONS
from tokenspeed_kernel.ops.moe._triton.route_epilogue import (
    k8_reduction_tree,
    reduction_order_fixture,
    reduction_order_mismatch,
)
from tokenspeed_kernel.ops.moe._triton.router_select import (
    _sigmoid_rn,
    triton_router_route_lanepar,
)
from tokenspeed_kernel.ops.moe.grouped_route import torch_grouped_route
from tokenspeed_kernel.platform import Platform

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or not Platform.get().is_nvidia,
    reason="requires an NVIDIA CUDA device",
)
pytestmark = requires_cuda

DEVICE = torch.device("cuda", 0)
SEED = 1234
EXPERTS, NUM_GROUPS, GROUP_TOPK, TOPK_GROUPS, TOPK = 256, 8, 2, 4, 8
GROUP_SIZE = EXPERTS // NUM_GROUPS
TIE_KINDS = ("group_boundary", "expert_boundary", "all_equal", "neg_inf_heavy", "zeros")
RANDOM_ROWS = (1, 8, 33, 1024, 4096)
TIE_ROWS = (1, 8, 33, 1024)
EDGE_ROWS = (1, 8)
ROWS_GRAPH = (4,)
PROBE_BLOCK = 256  # one row of lanes per program, as the router kernel's x
CANONICAL_NAN_BITS = 0x7FC00000

# ---------------------------------------------------------------------------------------
# Input builders: the random kind, the five tie kinds and the edge kind.
# ---------------------------------------------------------------------------------------

LOGIT_STD, BIAS_STD = 0.6, 0.1
TIE_ROW, NEGATIVE_ROW = 1, 2
TIE_EXPERTS = (0, 1, 2, 32, 33, 64, 65, 96, 97)
TIE_LOGIT, TIE_BIAS = 6.0, BIAS_STD
TIE_BACKGROUND_LOGIT, TIE_BACKGROUND_STD = -3.0, 0.3
NEGATIVE_LOGIT = -50.0
BACKGROUND_LOGIT, BACKGROUND_STD = -3.0, 0.3
GROUP_BOUNDARY_HIGH_LOGIT, GROUP_BOUNDARY_HIGH_GROUPS = 6.0, 3
GROUP_BOUNDARY_TIED_LOGIT, GROUP_BOUNDARY_TIED_GROUPS = 4.0, 2
EXPERT_BOUNDARY_HIGH_LOGITS = (8.0, 7.9, 7.8, 7.7, 7.6, 7.5, 7.4)
EXPERT_BOUNDARY_TIED_LOGIT = 5.0
ALL_EQUAL_LOGIT, ALL_EQUAL_ROW_STEP = 0.5, 0.01
NEG_INF_FINITE_LANE, NEG_INF_LOGIT_STD = 0, 1.0
ZEROS_LOGIT, ZEROS_NEGATIVE_BIAS = -200.0, -0.5
ZEROS_NEGATIVE_LOGIT_MEAN, ZEROS_NEGATIVE_LOGIT_STD = -12.0, 1.0
ZEROS_GROUPS = ((0, 3), (1, 3), (2, 2), (3, 2), (4, 2))
EDGE_BIAS_STD = 0.1
EDGE_NAN_LANE = 17
FLT_MAX = 3.4028234663852886e38


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


# (lane, logit bits, bias bits) of the edge block; the NaN lane's logit bits are per row.
EDGE_BLOCK = (
    (0, _f32_bits(0.0), _f32_bits(-0.5)),
    (1, _f32_bits(-0.0), _f32_bits(-0.5)),
    (2, _f32_bits(2.0**-149), _f32_bits(-0.5 + 2.0**-25)),
    (3, _f32_bits(-(2.0**-149)), _f32_bits(0.0)),
    (4, _f32_bits(2.0**-126 - 2.0**-149), _f32_bits(-0.0)),
    (5, _f32_bits(-(2.0**-126 - 2.0**-149)), _f32_bits(0.5)),
    (6, _f32_bits(float("inf")), _f32_bits(-1.0)),
    (7, _f32_bits(float("-inf")), _f32_bits(-0.0)),
    (8, _f32_bits(100.0), _f32_bits(-(1.0 - 2.0**-24))),
    (9, _f32_bits(-100.0), _f32_bits(2.0**-149)),
    (10, _f32_bits(-88.0), _f32_bits(-(2.0**-126))),
    (11, _f32_bits(-87.5), _f32_bits(2.0**-126)),
    (12, _f32_bits(88.5), _f32_bits(-1.0)),
    (13, _f32_bits(-104.0), _f32_bits(-1e30)),
    (14, _f32_bits(FLT_MAX), _f32_bits(FLT_MAX)),
    (15, _f32_bits(-FLT_MAX), _f32_bits(-FLT_MAX)),
    (16, _f32_bits(50.0), _f32_bits(FLT_MAX)),
    (EDGE_NAN_LANE, 0x7FC00000, _f32_bits(1e30)),
)


def _int32_from_unsigned(bits: int) -> int:
    bits &= 0xFFFFFFFF
    return bits - (1 << 32) if bits >= (1 << 31) else bits


def _rotated_group(row: int, offset: int) -> int:
    return (row + offset) % NUM_GROUPS


def _background(rows, generator, mean, std) -> torch.Tensor:
    return mean + torch.randn(rows, EXPERTS, generator=generator) * std


def _random_inputs(rows: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Random FP32 ``(logits [rows, 256], bias [256])`` on CPU
    (row 1 the nine-equal-choices tie row, row 2 the all-negative row)."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, EXPERTS, generator=generator, dtype=torch.float32)
    logits = logits * LOGIT_STD
    bias = torch.randn(EXPERTS, generator=generator, dtype=torch.float32) * BIAS_STD
    bias[list(TIE_EXPERTS)] = TIE_BIAS
    if rows > TIE_ROW:
        logits[TIE_ROW] = (
            TIE_BACKGROUND_LOGIT
            + torch.randn(EXPERTS, generator=generator, dtype=torch.float32)
            * TIE_BACKGROUND_STD
        )
        logits[TIE_ROW, list(TIE_EXPERTS)] = TIE_LOGIT
    if rows > NEGATIVE_ROW:
        logits[NEGATIVE_ROW] = NEGATIVE_LOGIT
    return logits.contiguous(), bias.contiguous()


def _tie_inputs(kind: str, rows: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The five tie kinds: FP32 ``(logits, bias)`` on CPU."""
    generator = torch.Generator().manual_seed(seed)
    bias = torch.zeros(EXPERTS, dtype=torch.float32)
    if kind == "group_boundary":
        logits = _background(rows, generator, BACKGROUND_LOGIT, BACKGROUND_STD)
        for row in range(rows):
            for offset in range(GROUP_BOUNDARY_HIGH_GROUPS):
                start = _rotated_group(row, offset) * GROUP_SIZE
                logits[row, start : start + 2] = GROUP_BOUNDARY_HIGH_LOGIT
            for offset in range(GROUP_BOUNDARY_TIED_GROUPS):
                start = (
                    _rotated_group(row, GROUP_BOUNDARY_HIGH_GROUPS + offset)
                    * GROUP_SIZE
                )
                logits[row, start : start + 2] = GROUP_BOUNDARY_TIED_LOGIT
    elif kind == "expert_boundary":
        logits = _background(rows, generator, BACKGROUND_LOGIT, BACKGROUND_STD)
        high = EXPERT_BOUNDARY_HIGH_LOGITS
        for row in range(rows):
            kept = [_rotated_group(row, offset) * GROUP_SIZE for offset in range(4)]
            logits[row, kept[0] : kept[0] + 2] = torch.tensor(high[0:2])
            logits[row, kept[0] + 2] = EXPERT_BOUNDARY_TIED_LOGIT
            logits[row, kept[1] : kept[1] + 2] = torch.tensor(high[2:4])
            logits[row, kept[2] : kept[2] + 2] = torch.tensor(high[4:6])
            logits[row, kept[3]] = high[6]
            logits[row, kept[3] + 1 : kept[3] + 3] = EXPERT_BOUNDARY_TIED_LOGIT
    elif kind == "all_equal":
        logits = torch.empty(rows, EXPERTS, dtype=torch.float32)
        for row in range(rows):
            logits[row] = ALL_EQUAL_LOGIT + ALL_EQUAL_ROW_STEP * row
    elif kind == "neg_inf_heavy":
        logits = torch.randn(rows, EXPERTS, generator=generator) * NEG_INF_LOGIT_STD
        bias = torch.full((EXPERTS,), -float("inf"), dtype=torch.float32)
        bias[NEG_INF_FINITE_LANE::GROUP_SIZE] = 0.0
    elif kind == "zeros":
        logits = _background(
            rows, generator, ZEROS_NEGATIVE_LOGIT_MEAN, ZEROS_NEGATIVE_LOGIT_STD
        )
        bias = torch.full((EXPERTS,), ZEROS_NEGATIVE_BIAS, dtype=torch.float32)
        for group, count in ZEROS_GROUPS:
            start = group * GROUP_SIZE
            logits[:, start : start + count] = ZEROS_LOGIT
            bias[start : start + count] = 0.0
    else:
        raise ValueError(f"unknown tie kind {kind!r}")
    return logits.contiguous(), bias.contiguous()


def _edge_nan_bits(row: int) -> int:
    return (0x7FC00000 | (row + 1)) | ((row & 1) << 31)


def _edge_inputs(rows: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The edge kind: the tie kinds' background and a N(0, 0.1) bias, then per
    row the 18-lane block written into group ``row % 8`` (one NaN lane per row)."""
    generator = torch.Generator().manual_seed(seed)
    logits = _background(rows, generator, BACKGROUND_LOGIT, BACKGROUND_STD)
    bias = torch.randn(EXPERTS, generator=generator) * EDGE_BIAS_STD
    logit_bits = logits.contiguous().view(torch.int32)
    bias_bits = bias.contiguous().view(torch.int32)
    for row in range(rows):
        start = _rotated_group(row, 0) * GROUP_SIZE
        for lane, logit_word, bias_word in EDGE_BLOCK:
            bits = _edge_nan_bits(row) if lane == EDGE_NAN_LANE else logit_word
            logit_bits[row, start + lane] = _int32_from_unsigned(bits)
            bias_bits[start + lane] = _int32_from_unsigned(bias_word)
    return (
        logit_bits.view(torch.float32).contiguous(),
        bias_bits.view(torch.float32).contiguous(),
    )


# ---------------------------------------------------------------------------------------
# The sigmoid pin corpus: the enumerated edge values.
# ---------------------------------------------------------------------------------------

MANTISSA_BITS = 23
MANTISSA_MASK = (1 << MANTISSA_BITS) - 1
NAN_PAYLOADS = (
    0x7FC00000,
    0xFFC00000,
    0x7FC00001,
    0xFFC00001,
    0x7F800001,
    0xFF800001,
    0x7FBFFFFF,
    0xFFBFFFFF,
)
W_PLUS = (0x42B00A3D, 0x42B175C3)  # every FP32 value of [88.02, 88.73]: 93,063
W_MINUS = (0xC2AE999A, 0xC2CFF0A4)  # every FP32 value of [-103.97, -87.3]: 2,184,971
SEEDED_SUBNORMALS = 64
LN2_K_MAX = 150
EDGE_VALUES = (
    2
    + 4
    + 2 * SEEDED_SUBNORMALS
    + 2
    + 2
    + len(NAN_PAYLOADS)
    + (W_PLUS[1] - W_PLUS[0] + 1)
    + (W_MINUS[1] - W_MINUS[0] + 1)
    + 2 * (LN2_K_MAX + 1)
)  # 2,278,482


def _unsigned_to_int32(bits: torch.Tensor) -> torch.Tensor:
    wrapped = torch.where(bits >= 2**31, bits - 2**32, bits)
    return wrapped.to(torch.int32)


def _edge_corpus(seed: int = SEED):
    """``(x float32 [2,278,482], categories)``: ``+-0``, the extreme and 64 seeded
    subnormals of both signs, ``+-FLT_MIN``, ``+-inf``, the 8 NaN payloads, every FP32
    value of the two ``expf`` overflow windows, ``+-k ln 2`` for ``k = 0..150``."""
    generator = torch.Generator().manual_seed(seed)
    seeded = torch.randint(
        2, MANTISSA_MASK - 1, (SEEDED_SUBNORMALS,), generator=generator
    )
    subnormal = [1, 0x80000001, 0x007FFFFF, 0x807FFFFF]
    for mantissa in seeded.tolist():
        subnormal.extend((mantissa, 0x80000000 | mantissa))
    ln2k: list[int] = []
    for k in range(LN2_K_MAX + 1):
        value = struct.unpack("<f", struct.pack("<f", k * math.log(2)))[0]
        ln2k.extend((_f32_bits(value), _f32_bits(-value)))
    groups = [
        ("zero", [0x00000000, 0x80000000]),
        ("subnormal", subnormal),
        ("flt_min", [0x00800000, 0x80800000]),
        ("inf", [0x7F800000, 0xFF800000]),
        ("nan", list(NAN_PAYLOADS)),
        ("W+", torch.arange(W_PLUS[0], W_PLUS[1] + 1, dtype=torch.int64)),
        ("W-", torch.arange(W_MINUS[0], W_MINUS[1] + 1, dtype=torch.int64)),
        ("ln2k", ln2k),
    ]
    pieces: list[torch.Tensor] = []
    categories: dict[str, tuple[int, int]] = {}
    start = 0
    for label, bits in groups:
        tensor = (
            bits
            if isinstance(bits, torch.Tensor)
            else torch.tensor(bits, dtype=torch.int64)
        )
        pieces.append(_unsigned_to_int32(tensor))
        categories[label] = (start, int(tensor.numel()))
        start += int(tensor.numel())
    x = torch.cat(pieces).view(torch.float32)
    assert x.numel() == EDGE_VALUES
    return x, categories


def _nan_rule_mismatch(torch_values: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """The per-element mismatch mask under the NaN rule: equal ``isnan`` masks and
    byte-equal non-NaN elements (payloads not compared)."""
    assert torch_values.shape == other.shape
    t_nan, c_nan = torch.isnan(torch_values), torch.isnan(other)
    t_bits = torch_values.contiguous().view(torch.int32)
    c_bits = other.contiguous().view(torch.int32)
    return (t_nan != c_nan) | (~t_nan & (t_bits != c_bits))


def _canonical(scores: torch.Tensor) -> torch.Tensor:
    """The NaN rule's canonical form: every NaN lane rewritten to ``0x7FC00000``."""
    bits = scores.detach().contiguous().view(torch.int32)
    return torch.where(
        torch.isnan(scores), torch.full_like(bits, CANONICAL_NAN_BITS), bits
    )


# ---------------------------------------------------------------------------------------
# The probe jit (the product helper by import)
# ---------------------------------------------------------------------------------------


@triton.jit
def _sigmoid_probe_kernel(x_ptr, s_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(s_ptr + offsets, _sigmoid_rn(x), mask=mask)


def _sigmoid_probe(x: torch.Tensor):
    """``_sigmoid_rn`` alone on a 1-D FP32 CUDA tensor (256 lanes per program, the
    product launch options); returns ``(s, CompiledKernel)``."""
    assert x.dtype is torch.float32 and x.ndim == 1 and x.is_cuda
    s = torch.empty_like(x)
    n = int(x.numel())
    grid = (-(-n // PROBE_BLOCK),)
    compiled = _sigmoid_probe_kernel[grid](
        x, s, n, BLOCK=PROBE_BLOCK, **_LAUNCH_OPTIONS
    )
    torch.cuda.synchronize(DEVICE)
    return s, compiled


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------


def _on_device(pair: tuple[torch.Tensor, torch.Tensor]):
    return pair[0].to(DEVICE), pair[1].to(DEVICE)


def _bits(value: torch.Tensor) -> torch.Tensor:
    return value.contiguous().view(torch.int32)


def test_sigmoid_pin_edge_corpus_through_the_product_helper():
    """The enumerated edge corpus (2,278,482 values: signed zeros, subnormals, the
    infinities, 8 NaN payloads, every FP32 value of the two ``expf`` overflow windows,
    ``+-k ln 2``) -> 0 mismatches under the NaN rule; the 8 NaN inputs are NaN in both.
    """
    x_cpu, categories = _edge_corpus(SEED)
    assert x_cpu.numel() == EDGE_VALUES
    assert categories["nan"][1] == len(NAN_PAYLOADS)
    x = x_cpu.to(DEVICE)
    reference = torch.sigmoid(x)
    probe, _ = _sigmoid_probe(x)
    mask = _nan_rule_mismatch(reference, probe)
    mismatches = int(mask.sum())
    start, count = categories["nan"]
    assert bool(torch.isnan(reference[start : start + count]).all())
    assert bool(torch.isnan(probe[start : start + count]).all())
    per_category = {
        name: int(mask[s : s + c].sum()) for name, (s, c) in categories.items()
    }
    assert mismatches == 0, {k: v for k, v in per_category.items() if v}


def test_torch_k8_sum_matches_the_pinned_reduction_tree():
    """``torch.sum(dim=-1, keepdim=True)`` over the pin rows equals the tree the
    kernel spells, bit for bit; the sequential and adjacent-pairwise orders each
    differ on at least one row, so the agreement is not vacuous."""
    rows = reduction_order_fixture().to(DEVICE)
    torch_sum = rows.sum(dim=-1, keepdim=True)
    assert torch.equal(_bits(torch_sum), _bits(k8_reduction_tree(rows)))
    assert reduction_order_mismatch(DEVICE) == ""
    v = rows.unbind(-1)
    sequential = v[0]
    for value in v[1:]:
        sequential = sequential + value
    pairwise = ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]))
    assert not torch.equal(_bits(sequential.unsqueeze(-1)), _bits(torch_sum))
    assert not torch.equal(_bits(pairwise.unsqueeze(-1)), _bits(torch_sum))


# (renormalize, scale): Beam's vendor route (normalized, routed scale 2.5), BeamMoE's
# (normalized, unscaled) and the unnormalized variants.
ROUTE_VARIANTS = ((True, 2.5), (True, None), (False, 2.5), (False, None))
ARGS = (NUM_GROUPS, GROUP_TOPK, TOPK_GROUPS, TOPK)


def _route_one_launch(logits, bias, renormalize, scale):
    out = triton_router_route_lanepar(logits, bias, *ARGS, renormalize, scale)
    torch.cuda.synchronize(DEVICE)
    return out


def _route_torch(logits, bias, renormalize, scale):
    out = torch_grouped_route(logits, bias, *ARGS, renormalize, scale)
    torch.cuda.synchronize(DEVICE)
    return out


def _assert_route_equal(got, expected, kind, context):
    """ids strict; weights strict on random rows, else under the NaN rule (Torch and
    the kernel's recipe may write different NaN payloads)."""
    assert torch.equal(got[1], expected[1]), context
    if kind == "random":
        assert not bool(torch.isnan(got[0]).any()), context
        assert torch.equal(_bits(got[0]), _bits(expected[0])), context
    else:
        assert torch.equal(_canonical(got[0]), _canonical(expected[0])), context


def test_route_in_one_launch_equals_the_torch_route():
    """Every corpus and route variant; the one-launch route is deterministic (NaN
    payloads included)."""
    reason = reduction_order_mismatch(DEVICE)
    assert not reason, reason
    seed = SEED
    corpus = (
        [("random", rows, _random_inputs(rows, seed + rows)) for rows in RANDOM_ROWS]
        + [
            (kind, rows, _tie_inputs(kind, rows, seed))
            for kind in TIE_KINDS
            for rows in TIE_ROWS
        ]
        + [("edge", rows, _edge_inputs(rows, seed)) for rows in EDGE_ROWS]
    )
    for kind, rows, pair in corpus:
        logits, bias = _on_device(pair)
        for renormalize, scale in ROUTE_VARIANTS:
            context = (kind, rows, renormalize, scale)
            weights, ids = _route_one_launch(logits, bias, renormalize, scale)
            assert weights.dtype is torch.float32 and ids.dtype is torch.int32
            assert weights.shape == ids.shape == (rows, TOPK)
            assert weights.is_contiguous() and ids.is_contiguous()
            expected = _route_torch(logits, bias, renormalize, scale)
            _assert_route_equal((weights, ids), expected, kind, context)
            again = _route_one_launch(logits, bias, renormalize, scale)
            assert torch.equal(_bits(again[0]), _bits(weights)), context
            assert torch.equal(again[1], ids), context


@pytest.mark.parametrize("rows", ROWS_GRAPH)
def test_route_in_one_launch_replays_in_a_graph_on_live_inputs(rows):
    """Captured once, replayed on new logits and bias copied into the same buffers
    (a tie pattern, an edge row, fresh random rows): the torch route's answer on
    the new inputs."""
    logits, bias = _on_device(_random_inputs(rows, SEED + rows))
    _route_one_launch(logits, bias, True, 2.5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = triton_router_route_lanepar(logits, bias, *ARGS, True, 2.5)
    for kind, (new_logits, new_bias) in (
        ("random", _random_inputs(rows, SEED + rows)),
        ("zeros", _tie_inputs("zeros", rows, SEED + 1)),
        ("edge", _edge_inputs(rows, SEED)),
        ("random", _random_inputs(rows, SEED + 1 + rows)),
    ):
        logits.copy_(new_logits)
        bias.copy_(new_bias)
        graph.replay()
        torch.cuda.synchronize(DEVICE)
        expected = _route_torch(logits, bias, True, 2.5)
        _assert_route_equal(captured, expected, kind, kind)
    graph.reset()
