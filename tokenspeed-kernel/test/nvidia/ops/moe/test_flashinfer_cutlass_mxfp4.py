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

"""Numerical check: FlashInfer CUTLASS MXFP4 MoE on SM90 vs a dequant reference.

The W4A16 kernel dequantizes E2M1 weights in the mainloop and must track the
bf16 reference like Marlin does. The W4A8 (Humming) kernel also rounds the
activations to FP8, so it is held to a looser bound and additionally checked
to sit closer to the clamped reference than to an unclamped one, which pins
both the residual scales and the SwiGLU clamp. SM90 only: the interleaved
layouts feed FlashInfer's Hopper mixed-input module.
"""

from __future__ import annotations

from importlib.util import find_spec
from types import SimpleNamespace

import pytest
import torch
from test_marlin_mxfp4_apply import _clamped_swiglu_moe_reference
from tokenspeed_kernel.ops.tuning import autotune
from tokenspeed_kernel.platform import ArchVersion, current_platform
from utils import make_mxfp4_moe_weights


def _requires_sm90_flashinfer() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not current_platform().is_nvidia:
        pytest.skip("NVIDIA required")
    if current_platform().arch_version != ArchVersion(9, 0):
        pytest.skip("FlashInfer SM90 mixed-input MoE is Hopper-only")
    if find_spec("flashinfer") is None:
        pytest.skip("requires flashinfer")


def _weights(raw, num_local_experts, ep_size, ep_rank, limit):
    return SimpleNamespace(
        w13_weight=torch.nn.Parameter(raw["w13_weight"], requires_grad=False),
        w13_weight_scale=torch.nn.Parameter(raw["w13_scale"], requires_grad=False),
        w2_weight=torch.nn.Parameter(raw["w2_weight"], requires_grad=False),
        w2_weight_scale=torch.nn.Parameter(raw["w2_scale"], requires_grad=False),
        num_local_experts=num_local_experts,
        ep_size=ep_size,
        ep_rank=ep_rank,
        tp_size=1,
        tp_rank=0,
        activation="swiglu",
        swiglu_arg=SimpleNamespace(alpha=None, limit=limit),
        swiglu_beta=None,
        w13_input_layout="concatenated",
    )


def _routing(num_tokens, num_experts, top_k, generator):
    # Round-robin ids keep every expert busy; weights are normalized random.
    topk_ids = (
        torch.arange(num_tokens * top_k, device="cuda").reshape(num_tokens, top_k)
        % num_experts
    ).to(torch.int32)
    topk_weights = torch.rand(
        num_tokens, top_k, generator=generator, device="cuda", dtype=torch.float32
    )
    return topk_ids, topk_weights / topk_weights.sum(-1, keepdim=True)


def _case(num_tokens, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    num_experts, top_k = 8, 2
    hidden_size, intermediate_size = 256, 128
    limit = 10.0
    x = torch.randn(num_tokens, hidden_size, generator=generator, device="cuda").to(
        torch.bfloat16
    )
    # Scales near 1.0 make gate/up exceed the clamp, so a dropped clamp shows.
    raw = make_mxfp4_moe_weights(
        num_experts,
        hidden_size,
        intermediate_size,
        generator,
        device="cuda",
        scale_range=(126, 127),
    )
    topk_ids, topk_weights = _routing(num_tokens, num_experts, top_k, generator)
    expected = _clamped_swiglu_moe_reference(x, raw, topk_ids, topk_weights, limit)
    unclamped = _clamped_swiglu_moe_reference(x, raw, topk_ids, topk_weights, None)
    magnitude = expected.float().abs().max()
    assert (
        expected.float() - unclamped.float()
    ).abs().max() > 0.5 * magnitude, (
        "the clamp must be active for this test to discriminate"
    )
    return x, raw, topk_ids, topk_weights, expected, unclamped, limit, magnitude


def _relative_l2(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


@pytest.mark.parametrize("num_tokens", [1, 8, 64])
def test_w4a16_matches_clamped_reference(num_tokens) -> None:
    _requires_sm90_flashinfer()
    from tokenspeed_kernel.ops.moe.flashinfer.cutlass_mxfp4 import (
        flashinfer_cutlass_mxfp4_w4a16_moe_apply,
        flashinfer_cutlass_mxfp4_w4a16_moe_weights,
    )

    x, raw, topk_ids, topk_weights, expected, _, limit, magnitude = _case(
        num_tokens, seed=11
    )
    w = _weights({k: v.clone() for k, v in raw.items()}, 8, 1, 0, limit)
    plan = {"activation": "swiglu"}
    flashinfer_cutlass_mxfp4_w4a16_moe_weights(plan, w)
    with autotune():
        actual = flashinfer_cutlass_mxfp4_w4a16_moe_apply(
            plan, x, w, None, topk_weights=topk_weights, topk_ids=topk_ids
        )
    assert torch.isfinite(actual).all()
    # bf16 GEMM outputs at this magnitude carry ~2^-8 relative rounding.
    torch.testing.assert_close(
        actual.float(), expected.float(), atol=float(2e-2 * magnitude), rtol=2e-2
    )


@pytest.mark.parametrize("num_tokens", [8, 64])
def test_w4a8_tracks_clamped_reference_within_fp8_noise(num_tokens) -> None:
    _requires_sm90_flashinfer()
    from tokenspeed_kernel.ops.moe.flashinfer.cutlass_mxfp4 import (
        flashinfer_cutlass_mxfp4_w4a8_moe_apply,
        flashinfer_cutlass_mxfp4_w4a8_moe_weights,
    )

    x, raw, topk_ids, topk_weights, expected, unclamped, limit, _ = _case(
        num_tokens, seed=17
    )
    w = _weights({k: v.clone() for k, v in raw.items()}, 8, 1, 0, limit)
    plan = {"activation": "swiglu"}
    flashinfer_cutlass_mxfp4_w4a8_moe_weights(plan, w)
    with autotune():
        actual = flashinfer_cutlass_mxfp4_w4a8_moe_apply(
            plan, x, w, None, topk_weights=topk_weights, topk_ids=topk_ids
        )
    assert torch.isfinite(actual).all()
    # E4M3 activations round at ~2^-4 per element; the dot products average
    # that down to a few percent. A 10% bound still fails on a dropped clamp or
    # a wrong residual scale, which both move the output by far more.
    assert _relative_l2(actual, expected) < 0.10
    assert _relative_l2(actual, expected) < _relative_l2(actual, unclamped)


def test_w4a16_ep_masks_nonlocal_experts() -> None:
    """EP: each rank owns half the experts; summed ranks == full reference."""
    _requires_sm90_flashinfer()
    from tokenspeed_kernel.ops.moe.flashinfer.cutlass_mxfp4 import (
        flashinfer_cutlass_mxfp4_w4a16_moe_apply,
        flashinfer_cutlass_mxfp4_w4a16_moe_weights,
    )

    x, raw, topk_ids, topk_weights, expected, _, limit, magnitude = _case(16, seed=23)
    ep_size = 2
    num_local = raw["w13_weight"].shape[0] // ep_size
    plan = {"activation": "swiglu"}
    acc = torch.zeros_like(x, dtype=torch.float32)
    for ep_rank in range(ep_size):
        lo = ep_rank * num_local
        shard = {k: v[lo : lo + num_local].clone() for k, v in raw.items()}
        w = _weights(shard, num_local, ep_size, ep_rank, limit)
        flashinfer_cutlass_mxfp4_w4a16_moe_weights(plan, w)
        acc += flashinfer_cutlass_mxfp4_w4a16_moe_apply(
            plan, x, w, None, topk_weights=topk_weights, topk_ids=topk_ids
        ).float()
    torch.testing.assert_close(
        acc, expected.float(), atol=float(2e-2 * magnitude), rtol=2e-2
    )
