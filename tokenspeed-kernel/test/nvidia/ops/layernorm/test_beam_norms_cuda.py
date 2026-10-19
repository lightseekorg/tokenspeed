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

"""Beam's fused norm kernels on an NVIDIA device: byte for byte the statements.

The residual boundary: the shared ``rmsnorm``'s residual arm with
``round_residual_sum_bf16`` rounds the FP32 sum ``x + residual`` once to BF16 inside
``_rmsnorm_kernel`` (and, with the depth-scale pair, each scaled input first with an
inline ``mul.rn.f32``), the roundings Torch's BF16 ops perform, so it must equal the
Torch BF16 add followed by the plain ``rmsnorm``. ``triton_fused_qk_rmsnorm_ssmax``
rounds the affine output once to BF16 and multiplies the Q rows by the SSMax scale
computed in the kernel from the position, so it must equal its torch kernel. Both
bit for bit on every lane -- signed zeros, subnormals, exact cancellations and
midpoint ties included -- on random rows and on edge corpora; a captured launch must
read its live inputs on CUDA-graph replay.
"""

from __future__ import annotations

import itertools
import math
import zlib

import pytest
import torch
from tokenspeed_kernel.ops.layernorm.qk_rmsnorm_ssmax import (
    BEAM_HEAD_DIM,
    BEAM_HEADS,
    TORCH_QK_RMSNORM_SSMAX,
    TRITON_FUSED_QK_RMSNORM_SSMAX,
    qk_rmsnorm_ssmax,
    select_qk_rmsnorm_ssmax_kernel,
)
from tokenspeed_kernel.ops.layernorm.triton import (
    rmsnorm,
)
from tokenspeed_kernel.platform import ArchVersion, Platform
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import kernel_override

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not Platform.get().is_nvidia,
    reason="requires an NVIDIA CUDA device",
)
# Beam calls the fused kernels there, and the Q/K kernel's guard admits only these
# devices even by name.
SM100_TO_SM103 = (
    torch.cuda.is_available()
    and Platform.get().is_nvidia
    and ArchVersion(10, 0) <= Platform.get().arch_version <= ArchVersion(10, 3)
)
requires_sm100_to_sm103 = pytest.mark.skipif(
    not SM100_TO_SM103, reason="requires an NVIDIA SM100 to SM103 device"
)

DEVICE = torch.device("cuda", 0)
SEED = 1234
EPS = 1e-6  # Beam rms_norm_eps
HIDDEN = 5120  # Beam's hidden size, the width Beam's residual boundary calls it at
HEAD_DIM = BEAM_HEAD_DIM  # 128
NUM_LAYERS = 52  # Beam depth -> 104 depth-scale pairs (attention and MLP per layer)
# Positive BF16 edge patterns without the two max-finite entries: signed zero, the
# subnormal range and its top, the smallest normals, the unit neighbours and their
# ties, 2 and its neighbour.
EDGES = tuple(
    int(word, 16)
    for word in "0000 0001 0002 003F 0040 007E 007F 0080 0081 00FF 0100 3A80 3E80 "
    "3EFF 3F00 3F7E 3F7F 3F80 3F81 3F82 3FFF 4000 4001".split()
)
SIGNED_EDGES = EDGES + tuple(bits | 0x8000 for bits in EDGES)
PAIRS_EDGES = tuple(itertools.product(SIGNED_EDGES, SIGNED_EDGES))
SIGNED_ZERO_PAIRS = (
    (0x0000, 0x0000),
    (0x0000, 0x8000),
    (0x8000, 0x0000),
    (0x8000, 0x8000),
)
TIE_MULTIPLIERS = (1.0, 1.0 + 2.0**-7, 1.0 - 2.0**-8, 3.0, 3.0 + 2.0**-6, 3.0 - 2.0**-6)
TIE_EXPONENT_RANGE = (-20, 20)
SUBNORMAL_ROTATION = 37
SUBNORMAL_PATTERNS = tuple(range(0x0001, 0x0080)) + tuple(range(0x8001, 0x8080))
NEAR_MIDPOINT_MAX_EXPONENT = 32
# Positions whose SSMax scale sits on a boundary of the float formats.
BOUNDARIES = (1048575, 65536, 65535, 8191, 2048, 2047, 512, 511, 1, 0)
BF16_ONE, BF16_MINUS_ONE = 0x3F80, 0xBF80
EDGE_SUBNORMALS = (0x0001, 0x0021, 0x007F, 0x8001, 0x8021, 0x807F, 0x0040, 0x8040)


def depth_scale_coefficients(layer_id: int, is_attention: bool) -> tuple[float, float]:
    """Beam's ``depth_scale_coefficients``: (sublayer, residual)."""
    depth = 2 * (layer_id + 1) + (0 if is_attention else 1)
    return 1.0 / math.sqrt(float(depth)), math.sqrt(float(depth - 1)) / math.sqrt(
        float(depth)
    )


PAIRS = tuple(
    depth_scale_coefficients(layer, is_attention)
    for layer in range(NUM_LAYERS)
    for is_attention in (True, False)
)


@pytest.fixture(autouse=True)
def selection_state():
    KernelRegistry.get().clear_cache()
    yield
    KernelRegistry.get().clear_cache()


def _bits(values) -> torch.Tensor:
    raw = torch.as_tensor(values, dtype=torch.int32)
    raw = torch.where(raw >= 0x8000, raw - 0x10000, raw)
    return raw.to(torch.int16).view(torch.bfloat16)


def _bits16(value: torch.Tensor) -> torch.Tensor:
    return value.contiguous().view(torch.int16)


def _mismatches(on, off) -> list[int]:
    return [int((_bits16(a) != _bits16(b)).sum()) for a, b in zip(on, off, strict=True)]


def _saved(inputs) -> dict[str, torch.Tensor]:
    return {name: value.clone() for name, value in inputs.items()}


def _assert_untouched(inputs, saved) -> None:
    for name, value in inputs.items():
        assert torch.equal(_bits16(value), _bits16(saved[name])), name


def _assert_fresh(result, inputs) -> None:
    """Contiguous BF16 outputs on the inputs' device, on storage of their own."""
    input_ptrs = {value.untyped_storage().data_ptr() for value in inputs.values()}
    for tensor in result:
        assert tensor.dtype is torch.bfloat16 and tensor.is_contiguous()
        assert tensor.device == DEVICE
        assert tensor.untyped_storage().data_ptr() not in input_ptrs


# ---------------------------------------------------------------------------------------
# The residual boundary: the corpus
# ---------------------------------------------------------------------------------------


def random_inputs(rows: int, seed: int) -> dict[str, torch.Tensor]:
    """``randn * 4`` BF16 ``rows`` x 5120 with a ``1e-3`` and a ``1e3`` row, a
    ``rand + 0.5`` weight, on the device."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, HIDDEN, generator=generator) * 4.0
    if rows >= 2:
        x[0] *= 1e-3
        x[1] *= 1e3
    weight = torch.rand(HIDDEN, generator=generator) + 0.5
    residual = torch.randn(rows, HIDDEN, generator=generator) * 4.0
    return {
        "x": x.to(torch.bfloat16).to(DEVICE),
        "residual": residual.to(torch.bfloat16).to(DEVICE),
        "weight": weight.to(torch.bfloat16).to(DEVICE),
    }


def _pair_rows(x_row, residual_row, x_second, residual_second, seed):
    return {
        "x": torch.stack([x_row, x_second]).contiguous().to(DEVICE),
        "residual": torch.stack([residual_row, residual_second])
        .contiguous()
        .to(DEVICE),
        "weight": random_inputs(2, seed)["weight"],
    }


def edge_inputs(seed: int) -> dict[str, torch.Tensor]:
    """The 2116 signed edge pairs Cartesian in rows 0/1 (roles swapped in the middle
    third), exact cancellation and one-ulp sums in row 2, the four signed-zero pairs
    cycling in row 3."""
    pairs = len(PAIRS_EDGES)
    x0, r0 = [], []
    for k in range(HIDDEN):
        a, b = PAIRS_EDGES[k % pairs]
        if pairs <= k < 2 * pairs:
            a, b = b, a
        x0.append(a)
        r0.append(b)
    x_row0, r_row0 = _bits(x0), _bits(r0)
    base = random_inputs(4, seed)["x"][2].cpu()
    flipped = (base.view(torch.int16) ^ 1).view(torch.bfloat16)
    r_row2 = torch.where(torch.arange(HIDDEN) % 2 == 0, -base, -flipped)
    zero_x = _bits([SIGNED_ZERO_PAIRS[k % 4][0] for k in range(HIDDEN)])
    zero_r = _bits([SIGNED_ZERO_PAIRS[k % 4][1] for k in range(HIDDEN)])
    return {
        "x": torch.stack([x_row0, r_row0, base, zero_x]).contiguous().to(DEVICE),
        "residual": torch.stack([r_row0, x_row0, r_row2, zero_r])
        .contiguous()
        .to(DEVICE),
        "weight": random_inputs(4, seed)["weight"],
    }


def tie_inputs(seed: int) -> dict[str, torch.Tensor]:
    """Sums on (M in {1, 3}) and just off BF16 midpoints."""
    generator = torch.Generator().manual_seed(seed)
    low, high = TIE_EXPONENT_RANGE
    exponent = torch.randint(low, high + 1, (HIDDEN,), generator=generator)
    mantissa = torch.randint(0, 128, (HIDDEN,), generator=generator)
    sign = torch.randint(0, 2, (HIDDEN,), generator=generator)
    x = _bits((sign * 0x8000 + (exponent + 127) * 128 + mantissa).tolist())
    half_ulp = torch.ldexp(
        torch.ones(HIDDEN, dtype=torch.float64), (exponent - 8).to(torch.int32)
    )
    signs = torch.where(torch.arange(HIDDEN) % 2 == 0, 1.0, -1.0).to(torch.float64)
    multipliers = torch.tensor(
        [TIE_MULTIPLIERS[k % len(TIE_MULTIPLIERS)] for k in range(HIDDEN)],
        dtype=torch.float64,
    )
    residual = (signs * half_ulp * multipliers).to(torch.bfloat16)
    return _pair_rows(x, residual, residual, x, seed)


def subnormal_inputs(seed: int) -> dict[str, torch.Tensor]:
    """The 254 signed BF16 subnormal patterns cycling, the residual the same list
    rotated; row 1 ``residual = -x`` (the FTZ detector)."""
    count = len(SUBNORMAL_PATTERNS)
    x0 = _bits([SUBNORMAL_PATTERNS[k % count] for k in range(HIDDEN)])
    r0 = _bits(
        [SUBNORMAL_PATTERNS[(k + SUBNORMAL_ROTATION) % count] for k in range(HIDDEN)]
    )
    return _pair_rows(x0, r0, x0, -x0, seed)


def nearest_to_midpoint_values(scale: float) -> torch.Tensor:
    """The 5120 finite BF16 values of magnitude below ``2 ** 32`` whose FP32 product
    with ``fl32(scale)`` lies closest to an RNE midpoint."""
    values = _bits(list(range(1 << 16)))
    values = values[torch.isfinite(values)]
    values = values[values.float().abs() < 2.0**NEAR_MIDPOINT_MAX_EXPONENT]
    product = values.float() * torch.tensor(scale, dtype=torch.float32)
    distance = ((product.view(torch.int32).to(torch.int64) & 0xFFFF) - 0x8000).abs()
    distance = torch.where(product != 0, distance, torch.full_like(distance, 1 << 20))
    pattern = values.view(torch.int16).to(torch.int64) & 0xFFFF
    order = torch.argsort(distance * (1 << 16) + pattern)
    return values[order[:HIDDEN]].contiguous()


def near_midpoint_inputs(alpha: float, beta: float, seed: int) -> dict:
    for_alpha = nearest_to_midpoint_values(alpha)
    for_beta = nearest_to_midpoint_values(beta)
    return _pair_rows(for_alpha, for_beta, for_beta, for_alpha, seed)


def _fused(inputs, scales=(None, None)):
    """The residual boundary in one launch, as ``BeamRMSNorm`` calls it."""
    x_scale, residual_scale = scales
    result = rmsnorm(
        inputs["x"],
        inputs["weight"],
        EPS,
        residual=inputs["residual"],
        out=None,
        enable_pdl=None,
        round_residual_sum_bf16=True,
        x_scale=x_scale,
        residual_scale=residual_scale,
    )
    torch.cuda.synchronize(DEVICE)
    return result


def _statements(inputs, scales=(None, None)):
    """The boundary's statements: the two BF16 products, the BF16 add, the norm."""
    x, residual = inputs["x"], inputs["residual"]
    if scales[0] is not None:
        x, residual = x * scales[0], residual * scales[1]
    combined = x + residual
    out = rmsnorm(
        combined, inputs["weight"], EPS, residual=None, out=None, enable_pdl=None
    )
    torch.cuda.synchronize(DEVICE)
    return out, combined


# ---------------------------------------------------------------------------------------
# The residual boundary: tests
# ---------------------------------------------------------------------------------------


def test_residual_fused_equals_the_statements():
    """Without the depth-scale pair on random rows of decode, odd and prefill counts
    and the edge and tie corpora; with it on all 104 pairs at 8 rows, a 1024-row
    prefill, and the edge, tie, subnormal and near-midpoint corpora at the first and
    the last pair. The fused launch is deterministic and leaves its inputs as
    they were."""
    unscaled = (None, None)
    corpus = [(f"rows{n}", random_inputs(n, SEED), unscaled) for n in (1, 33, 4096)]
    corpus += [
        ("edges", edge_inputs(SEED), unscaled),
        ("ties", tie_inputs(SEED), unscaled),
    ]
    corpus += [
        (f"rows8_pair{index}", random_inputs(8, SEED + index), pair)
        for index, pair in enumerate(PAIRS)
    ]
    corpus.append(("rows1024_pair0", random_inputs(1024, SEED), PAIRS[0]))
    for index in (0, len(PAIRS) - 1):
        alpha, beta = PAIRS[index]
        corpus += [
            (f"edges_pair{index}", edge_inputs(SEED), PAIRS[index]),
            (f"ties_pair{index}", tie_inputs(SEED), PAIRS[index]),
            (f"subnormal_pair{index}", subnormal_inputs(SEED), PAIRS[index]),
            (
                f"near_mid_pair{index}",
                near_midpoint_inputs(alpha, beta, SEED),
                (alpha, beta),
            ),
        ]
    failures = []
    for name, inputs, scales in corpus:
        saved = _saved(inputs)
        fold = _fused(inputs, scales=scales)
        _assert_fresh(fold, inputs)
        mismatches = _mismatches(fold, _statements(inputs, scales=scales))
        mismatches += _mismatches(fold, _fused(inputs, scales=scales))
        if any(mismatches):
            failures.append((name, mismatches))
        _assert_untouched(inputs, saved)
    assert failures == []


# ---------------------------------------------------------------------------------------
# layernorm.qk_rmsnorm_ssmax
# ---------------------------------------------------------------------------------------


def edge_rows() -> torch.Tensor:
    """Seven fixed head rows: subnormals, signed zeros, ones, a tiny and the smallest
    normal, a mixed-magnitude and a huge row."""
    half = HEAD_DIM // 2
    ones = [BF16_ONE if lane % 2 == 0 else BF16_MINUS_ONE for lane in range(half)]
    rows = (
        [EDGE_SUBNORMALS[lane % len(EDGE_SUBNORMALS)] for lane in range(half)] + ones,
        [0x0000 if lane % 2 == 0 else 0x8000 for lane in range(half)] + ones,
        [BF16_ONE] * HEAD_DIM,
        [0x2180] * HEAD_DIM,
        [0x0080 if lane % 2 == 0 else 0x8080 for lane in range(half)] + ones,
        [0x3A83] * half + [0x447A] * half,
        [0x467A if lane % 2 == 0 else 0xC67A for lane in range(HEAD_DIM)],
    )
    return _bits([list(row) for row in rows])


def qk_inputs(
    tokens: int,
    heads: tuple[int, int],
    *,
    kind: str = "scatter",
    position: int = 0,
    layout: str = "split",
    positions_dtype: torch.dtype = torch.int64,
    seed: int = SEED,
) -> dict[str, torch.Tensor]:
    """``randn * 1.5`` BF16 ``qkv`` with a ``1e-3`` and a ``1e3`` row (``edge``: the
    edge rows written into the q and k columns), ``rand + 0.5`` weights, ``q`` / ``k``
    column views of ``qkv`` (or contiguous copies), and positions per ``kind``:
    ``scatter`` random in ``[0, 2^20)`` with the boundary tail, ``uniform`` all
    ``position``, ``edge`` the first eight boundaries."""
    q_heads, kv_heads = heads
    q_size, kv_size = q_heads * HEAD_DIM, kv_heads * HEAD_DIM
    generator = torch.Generator().manual_seed(seed)
    qkv = torch.randn(tokens, q_size + 2 * kv_size, generator=generator) * 1.5
    if tokens >= 2:
        qkv[0] *= 1e-3
        qkv[1] *= 1e3
    qkv = qkv.to(torch.bfloat16)
    if kind == "edge":
        rows = edge_rows()
        for token in range(min(tokens, rows.shape[0])):
            qkv[token, :q_size] = rows[token].repeat(q_heads)
            qkv[token, q_size : q_size + kv_size] = rows[token].repeat(kv_heads)
    qkv = qkv.to(DEVICE)
    q, k, _ = qkv.split([q_size, kv_size, kv_size], dim=-1)
    if layout == "contiguous":
        q, k = q.contiguous(), k.contiguous()
    weights = [(torch.rand(HEAD_DIM, generator=generator) + 0.5) for _ in range(2)]
    if kind == "scatter":
        position_generator = torch.Generator().manual_seed(
            seed ^ zlib.crc32(f"{tokens}x{heads}/positions".encode())
        )
        positions = torch.randint(0, 2**20, (tokens,), generator=position_generator)
        tail = min(tokens, len(BOUNDARIES))
        positions[tokens - tail :] = torch.tensor(BOUNDARIES[:tail])
    elif kind == "uniform":
        positions = torch.full((tokens,), position, dtype=torch.int64)
    else:
        positions = torch.tensor(BOUNDARIES[:tokens], dtype=torch.int64)
    return {
        "qkv": qkv,
        "q": q,
        "k": k,
        "q_weight": weights[0].to(torch.bfloat16).to(DEVICE),
        "k_weight": weights[1].to(torch.bfloat16).to(DEVICE),
        "positions": positions.to(positions_dtype).to(DEVICE),
    }


def _qk(name: str | None, inputs) -> tuple[torch.Tensor, torch.Tensor]:
    """``qk_rmsnorm_ssmax`` by kernel name, or the ranked default for ``None``."""
    arguments = [inputs[key] for key in ("q", "k", "q_weight", "k_weight")]
    arguments += [EPS, inputs["positions"]]
    if name is None:
        result = qk_rmsnorm_ssmax(*arguments, solution=None, override=None)
    else:
        with kernel_override("layernorm", "qk_rmsnorm_ssmax", name):
            result = qk_rmsnorm_ssmax(*arguments, solution=None, override=None)
    torch.cuda.synchronize(DEVICE)
    return result


QK_CELLS = (
    [dict(tokens=t, heads=h) for t in (1, 8, 1024) for h in BEAM_HEADS]
    + [
        dict(tokens=8, heads=(6, 1), layout="contiguous"),
        dict(tokens=8, heads=(48, 8), positions_dtype=torch.int32),
        dict(tokens=8, heads=(48, 8), kind="edge"),
        dict(tokens=8, heads=(6, 1), kind="edge", positions_dtype=torch.int32),
    ]
    + [
        dict(tokens=1, heads=h, kind="uniform", position=p, seed=SEED + p % 97)
        for p in BOUNDARIES
        for h in BEAM_HEADS
    ]
)


@requires_sm100_to_sm103
def test_qk_triton_equals_torch_byte_for_byte():
    """Every cell: Beam's head pairs at decode and prefill token counts on ``qkv.split``
    views, a contiguous and an int32 cell, the edge rows (subnormal lanes stay non-zero,
    signed zeros keep their sign) and the ten boundary positions; each kernel is
    deterministic, the inputs are untouched, and the ranked default (the fused kernel
    here) gives the same bytes."""
    failures = []
    for cell in QK_CELLS:
        inputs = qk_inputs(**cell)
        saved = _saved(inputs)
        torch_result = _qk(TORCH_QK_RMSNORM_SSMAX, inputs)
        triton_result = _qk(TRITON_FUSED_QK_RMSNORM_SSMAX, inputs)
        _assert_fresh(triton_result, inputs)
        mismatches = _mismatches(torch_result, triton_result)
        mismatches += _mismatches(
            triton_result, _qk(TRITON_FUSED_QK_RMSNORM_SSMAX, inputs)
        )
        mismatches += _mismatches(torch_result, _qk(None, inputs))
        if any(mismatches):
            failures.append((cell, mismatches))
        arguments = [inputs[key] for key in ("q", "k", "q_weight", "k_weight")]
        default = select_qk_rmsnorm_ssmax_kernel(
            *arguments,
            EPS,
            inputs["positions"],
            dequant=False,
            solution=None,
            override=None,
        )
        assert default.name == TRITON_FUSED_QK_RMSNORM_SSMAX
        if cell.get("kind") == "edge":
            # Token 0's subnormal lanes stay non-zero (no flush to zero).
            row = edge_rows()[0].view(torch.int16)
            lanes = (((row & 0x7F80) == 0) & (row != 0)).to(DEVICE)
            heads = cell["heads"][0]
            q0 = triton_result[0][0].reshape(heads, HEAD_DIM)[:, lanes]
            assert (q0.view(torch.int16) != 0).all()
        _assert_untouched(inputs, saved)
    assert failures == []


@requires_sm100_to_sm103
def test_qk_graph_replay_follows_live_inputs():
    """A captured fused launch reads the live ``qkv`` views and positions on replay (the
    NoPE path runs under CUDA-graph capture in decode): the edge rows with the boundary
    positions, then fresh random inputs, copied into the same buffers replay to the
    torch kernel's answer on the new inputs."""
    inputs = qk_inputs(8, (6, 1), seed=SEED + 8)
    expected = _qk(TORCH_QK_RMSNORM_SSMAX, inputs)
    arguments = [inputs[key] for key in ("q", "k", "q_weight", "k_weight")]
    arguments += [EPS, inputs["positions"]]
    with kernel_override(
        "layernorm", "qk_rmsnorm_ssmax", TRITON_FUSED_QK_RMSNORM_SSMAX
    ):
        warmup = qk_rmsnorm_ssmax(*arguments, solution=None, override=None)
        torch.cuda.synchronize(DEVICE)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = qk_rmsnorm_ssmax(*arguments, solution=None, override=None)
    graph.replay()
    torch.cuda.synchronize(DEVICE)
    assert _mismatches(expected, warmup) == [0, 0]
    assert _mismatches(expected, captured) == [0, 0]
    for new in (
        qk_inputs(8, (6, 1), kind="edge", seed=SEED + 1),
        qk_inputs(8, (6, 1), seed=SEED + 9),
    ):
        inputs["qkv"].copy_(new["qkv"])
        inputs["positions"].copy_(new["positions"])
        expected_new = _qk(TORCH_QK_RMSNORM_SSMAX, inputs)
        assert _mismatches(expected_new, expected) != [0, 0]
        graph.replay()
        torch.cuda.synchronize(DEVICE)
        assert _mismatches(expected_new, captured) == [0, 0]
    graph.reset()
