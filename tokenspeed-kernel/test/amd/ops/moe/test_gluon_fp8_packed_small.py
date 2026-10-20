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

"""Packed-layout and shared routed-numerics tests for the FP8 MoE schedules."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("tokenspeed_kernel_amd", reason="AMD kernel package is optional")


@pytest.fixture
def gfx950_only() -> None:
    from utils import is_cdna4

    if not is_cdna4():
        pytest.skip("requires AMD gfx950 GPU")


@pytest.mark.parametrize(
    ("experts", "rows", "columns"),
    [(2, 128, 256), (1, 1024, 512), (1, 512, 512), (1, 1024, 4096), (1, 4096, 512)],
)
def test_packed_fp8_pack_matches_aiter_shuffle_bytes(
    experts: int, rows: int, columns: int
) -> None:
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.weight_preprocess import (
        pack_fp8_weight_n16k32,
    )

    raw = torch.randint(0, 256, (experts, rows, columns), dtype=torch.uint8)
    weight = raw.view(torch.float8_e4m3fn)

    # Independent element-to-byte mapping for AITER shuffle_weight(layout=(16, 16))
    # with one-byte FP8 elements; include gate/up and down projection shapes.
    n = torch.arange(rows, dtype=torch.int64)[:, None]
    k = torch.arange(columns, dtype=torch.int64)[None, :]
    packed_offset = (
        (((n // 16) * (columns // 32) + k // 32) * 2 + (k % 32) // 16) * 16 + n % 16
    ) * 16 + k % 16
    groups = packed_offset.reshape(rows, -1, 16)
    assert (groups[..., 0] % 16 == 0).all()
    assert torch.equal(groups - groups[..., :1], torch.arange(16).expand_as(groups))
    expected = torch.empty_like(raw)
    expected.reshape(experts, -1)[:, packed_offset.flatten()] = raw.reshape(experts, -1)

    packed = pack_fp8_weight_n16k32(weight)
    assert packed.dtype == weight.dtype
    assert packed.shape == weight.shape
    assert torch.equal(packed.view(torch.uint8), expected)


def _quantize_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    groups = x.float().reshape(x.shape[0], -1, 128)
    scales = groups.abs().amax(dim=-1).clamp_min(1.0e-10) * (1.0 / 448.0)
    quantized = (
        (groups * scales.reciprocal()[..., None])
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
    )
    return quantized.reshape_as(x), scales


def _reference(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    s13: torch.Tensor,
    s2: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    expert_start: int,
    limit: float,
    *,
    quantized_activations: bool,
) -> tuple[torch.Tensor, bool]:
    if quantized_activations:
        qx, sx = _quantize_reference(x)
        dequantized_x = qx.float() * sx.repeat_interleave(128, dim=1)
    else:
        dequantized_x = x.float()
    partials = torch.zeros(
        (*ids.shape, x.shape[1]),
        device=x.device,
        dtype=torch.bfloat16 if quantized_activations else torch.float32,
    )
    clamp_active = False
    for expert in range(w13.shape[0]):
        tokens, slots = torch.where(ids == expert_start + expert)
        if tokens.numel() == 0:
            continue
        gate_up = w13[expert].float() * s13[expert].repeat_interleave(
            128, dim=0
        ).repeat_interleave(128, dim=1)
        gate, up = F.linear(dequantized_x[tokens], gate_up).chunk(2, dim=-1)
        clamp_active |= bool(((gate > limit) | (up.abs() > limit)).any().item())
        intermediate = (F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)).to(
            torch.bfloat16
        )
        if quantized_activations:
            qi, si = _quantize_reference(intermediate)
            dequantized_i = qi.float() * si.repeat_interleave(128, dim=1)
        else:
            dequantized_i = intermediate.float()
        down_weight = w2[expert].float() * s2[expert].repeat_interleave(
            128, dim=0
        ).repeat_interleave(128, dim=1)
        down = F.linear(dequantized_i, down_weight)
        partials[tokens, slots] = (down * weights[tokens, slots, None]).to(
            partials.dtype
        )
    return partials.float().sum(dim=1).to(torch.bfloat16), clamp_active


def _routes(
    m: int, local_experts: int, ep_size: int, expert_start: int, variant: int
) -> torch.Tensor:
    topk = 8
    if ep_size == 1:
        return torch.tensor(
            [
                [(token + slot + variant) % local_experts for slot in range(topk)]
                for token in range(m)
            ],
            device="cuda",
            dtype=torch.int32,
        )
    remote = [
        expert
        for expert in range(local_experts * ep_size)
        if not expert_start <= expert < expert_start + local_experts
    ]
    rows = []
    for token in range(m):
        other = [remote[(token + slot + variant) % len(remote)] for slot in range(topk)]
        if token % 4 != 2 or variant:
            other[:2] = [
                expert_start + (token + variant) % local_experts,
                expert_start + (token + variant + 1) % local_experts,
            ]
        rows.append(other)
    return torch.tensor(rows, device="cuda", dtype=torch.int32)


def _fixture(
    m: int,
    ep_size: int,
    rank: int,
    outlier: bool,
    hidden: int,
    intermediate: int,
) -> tuple[torch.Tensor, ...]:
    local_experts = 8 if ep_size == 1 else 4
    expert_start = rank * local_experts
    generator = torch.Generator(device="cuda").manual_seed(71)
    x = (torch.randn(m, hidden, device="cuda", generator=generator) * 0.5).to(
        torch.bfloat16
    )
    w13 = (
        torch.randn(
            local_experts, 2 * intermediate, hidden, device="cuda", generator=generator
        )
        * 0.5
    ).to(torch.float8_e4m3fn)
    w2 = (
        torch.randn(
            local_experts, hidden, intermediate, device="cuda", generator=generator
        )
        * 0.5
    ).to(torch.float8_e4m3fn)
    s13 = (
        torch.rand(
            local_experts,
            2 * intermediate // 128,
            hidden // 128,
            device="cuda",
            generator=generator,
        )
        * 0.01
        + 0.04
    )
    s2 = (
        torch.rand(
            local_experts,
            hidden // 128,
            intermediate // 128,
            device="cuda",
            generator=generator,
        )
        * 0.01
        + 0.04
    )
    ids = _routes(m, local_experts, ep_size, expert_start, variant=0)
    weights = torch.softmax(
        torch.randn(m, 8, device="cuda", generator=generator), dim=-1
    )
    if outlier:
        x[0, :128] = 20.0
        w13[0, 0, :128] = 1.0
        w13[0, intermediate, :128] = 1.0
    return x, w13, w2, s13, s2, ids, weights


def _assert_numerics(
    actual: torch.Tensor,
    expected: torch.Tensor,
    record_property: Callable[[str, float], None],
) -> None:
    difference = (actual.float() - expected.float()).flatten()
    reference = expected.float().flatten()
    relative_l2 = (difference.norm() / reference.norm().clamp_min(1.0e-12)).item()
    cosine = F.cosine_similarity(actual.float().flatten(), reference, dim=0).item()
    record_property("relative_l2", relative_l2)
    record_property("cosine", cosine)
    assert torch.isfinite(actual).all()
    assert (
        relative_l2 <= 0.03 and cosine >= 0.9995
    ), f"relative L2={relative_l2:.6g}, cosine={cosine:.8g}"


_SMALL_CASES = [
    (16, 1, 0, False, 4096, 512),
    (16, 1, 0, True, 4096, 512),
    (16, 4, 0, False, 4096, 2048),
    (32, 1, 0, False, 4096, 512),
    (16, 1, 0, False, 512, 512),  # Factor-one pipeline's two-tile drain.
]
_SPLIT_CASES = [
    (1, 4, 0, False, 4096, 2048),
    (2, 4, 3, False, 4096, 2048),
    (2, 4, 3, True, 4096, 2048),
]
_LARGE_CASES = [
    (63, 1, 0, False, 4096, 512),
    (64, 1, 0, False, 4096, 512),
    (65, 1, 0, False, 4096, 512),
    (65, 1, 0, True, 4096, 512),
    (65, 4, 3, False, 4096, 2048),
]
_FULL_CASES = [
    (schedule, *case)
    for schedule, cases in (
        ("small", _SMALL_CASES),
        ("split", _SPLIT_CASES),
        ("large", _LARGE_CASES),
    )
    for case in cases
]


@pytest.mark.usefixtures("gfx950_only")
@pytest.mark.parametrize(
    ("schedule", "m", "ep_size", "rank", "outlier", "hidden", "intermediate"),
    _FULL_CASES,
    ids=[
        f"{schedule}-m{m}-ep{ep_size}-rank{rank}-k{hidden}-i{intermediate}-{'stress' if outlier else 'normal'}"
        for schedule, m, ep_size, rank, outlier, hidden, intermediate in _FULL_CASES
    ],
)
def test_packed_fp8_moe_matches_clamped_routed_reference(
    schedule: str,
    m: int,
    ep_size: int,
    rank: int,
    outlier: bool,
    hidden: int,
    intermediate: int,
    record_property: Callable[[str, float], None],
) -> None:
    from tokenspeed_kernel.ops.moe.gluon import _quantize_packed_activation
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8 import (
        launch_gluon_fp8_packed_moe_gfx950,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.weight_preprocess import (
        pack_fp8_weight_n16k32,
    )

    x, w13, w2, s13, s2, ids, weights = _fixture(
        m, ep_size, rank, outlier, hidden, intermediate
    )
    expert_start = rank * w13.shape[0]
    if schedule == "large" and ep_size > 1:
        # Keep both local experts above the second 64-route tile boundary.
        ids[:, 0] = expert_start
        ids[:, 1] = expert_start + 1
    limit = 10.0 if schedule in ("split", "warp") else 7.0
    expected, clamp_active = _reference(
        x,
        w13,
        w2,
        s13,
        s2,
        ids,
        weights,
        expert_start,
        limit,
        quantized_activations=schedule != "warp",
    )

    if schedule == "large" and hidden == 4096:
        counts = torch.stack(
            [(ids == expert_start + expert).sum() for expert in range(w13.shape[0])]
        )
        assert bool(((counts > 64) & (counts % 64 != 0)).any()) == (
            m > 64 and m % 64 != 0
        )
    packed13, packed2 = pack_fp8_weight_n16k32(w13), pack_fp8_weight_n16k32(w2)
    del w13, w2
    actual = launch_gluon_fp8_packed_moe_gfx950(
        x,
        packed13,
        packed2,
        s13,
        s2,
        ids,
        weights,
        swiglu_limit=limit,
        expert_start=expert_start,
        num_experts_total=packed13.shape[0] * ep_size,
        quantize_activation=_quantize_packed_activation,
    )
    if schedule in ("split", "warp"):
        if outlier:
            assert clamp_active
    else:
        assert clamp_active == outlier
    if schedule == "large":
        assert actual.dtype == torch.bfloat16
    if schedule == "small" and ep_size > 1:
        local = (ids >= expert_start) & (ids < expert_start + packed13.shape[0])
        assert not actual[~local.any(dim=1)].any()
    _assert_numerics(actual, expected, record_property)
