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

"""Correctness tests for Kimi K3 prefill Gluon kernels on AMD CDNA4/CDNA5."""

from __future__ import annotations

import pytest
import torch
from utils import is_cdna4, is_cdna5

if not (is_cdna4() or is_cdna5()):
    pytest.skip(
        "AMD CDNA4 or CDNA5 is required for Kimi K3 prefill Gluon tests",
        allow_module_level=True,
    )


import tokenspeed_kernel.ops.attn_res as attn_res  # noqa: E402
from tokenspeed_kernel.ops.attn_res import (  # noqa: E402
    attn_res_fwd,
    attn_res_fwd_available,
)
from tokenspeed_kernel.ops.moe import moe_sigmoid_bias_topk  # noqa: E402
from tokenspeed_kernel.ops.moe.sigmoid_topk import _gluon_eligible  # noqa: E402

if is_cdna4():
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda.attn_res import (  # noqa: E402
        attn_res_rmsnorm_gfx950 as attn_res_rmsnorm_amd,
    )
else:
    from tokenspeed_kernel_amd.ops.gfx1250.attention.kda.attn_res import (  # noqa: E402
        attn_res_rmsnorm_gfx1250 as attn_res_rmsnorm_amd,
    )


def _attn_res_reference(
    layer: torch.Tensor,
    history: torch.Tensor,
    res_weight: torch.Tensor,
    score_weight: torch.Tensor,
    output_weight: torch.Tensor,
    valid_blocks: int,
    score_eps: float,
    output_eps: float,
) -> torch.Tensor:
    values = torch.cat((history[:, :valid_blocks], layer.unsqueeze(1)), dim=1).float()
    inverse_rms = torch.rsqrt(values.square().mean(-1, keepdim=True) + score_eps)
    logits = values.mul(inverse_rms) @ (score_weight * res_weight.float())
    mixed = torch.matmul(logits.softmax(-1).unsqueeze(1), values).squeeze(1)
    mixed = mixed.to(torch.bfloat16).float()
    return (
        mixed
        * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + output_eps)
        * output_weight
    ).to(torch.bfloat16)


def _bf16_add_rne(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Match the fused kernel's FP32 add followed by BF16 rounding."""
    return (lhs.float() + rhs.float()).to(torch.bfloat16)


def test_attn_res_public_block_major_dispatch_matches_reference() -> None:
    tokens, valid_blocks = 256, 8
    score_eps, output_eps = 1e-6, 2e-6
    generator = torch.Generator(device="cuda").manual_seed(91)
    layer = torch.randn(
        tokens, 7168, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    history = torch.randn(
        valid_blocks,
        tokens,
        7168,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    res_weight = torch.randn(
        7168, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    score_weight = torch.randn(7168, device="cuda", generator=generator)
    output_weight = torch.randn(7168, device="cuda", generator=generator)

    actual = attn_res_fwd(
        layer,
        history,
        res_weight,
        score_weight,
        eps=score_eps,
        out_norm_weight=output_weight,
        out_norm_eps=output_eps,
    )
    expected = _attn_res_reference(
        layer,
        history.transpose(0, 1),
        res_weight,
        score_weight,
        output_weight,
        valid_blocks,
        score_eps,
        output_eps,
    )
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=1.6e-2)


def test_attn_res_large_prefill_dispatch_boundary(monkeypatch) -> None:
    selected_solutions = []

    def capture_selection(*args, **kwargs):
        selected_solutions.append(kwargs["solution"])
        return lambda **kwargs: None

    monkeypatch.setattr(attn_res, "select_kernel", capture_selection)
    cases = (
        (16384, 7168, True),
        (16385, 7168, True),
        (65536, 7168, True),
        (65537, 7168, True),
        (32768, 7168, False),
        (64, 4096, True),
    )
    for tokens, hidden, fuse_output_norm in cases:
        weight = torch.empty(hidden, device="meta", dtype=torch.bfloat16)
        layer = torch.empty(tokens, hidden, device="meta", dtype=torch.bfloat16)
        history = torch.empty(11, tokens, hidden, device="meta", dtype=torch.bfloat16)
        attn_res.attn_res_fwd(
            layer,
            history,
            weight,
            weight,
            out_norm_weight=weight if fuse_output_norm else None,
        )

    assert selected_solutions == [None, None, None, "torch", "torch", "torch"]


def test_attn_res_amd_entrypoint_accepts_legacy_keywords() -> None:
    tokens, valid_blocks = 2, 3
    score_eps, output_eps = 1e-6, 2e-6
    generator = torch.Generator(device="cuda").manual_seed(92)
    layer = torch.randn(
        tokens, 7168, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    history = torch.randn(
        tokens,
        valid_blocks,
        7168,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    res_weight = torch.randn(
        7168, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    score_weight = torch.randn(
        7168, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    output_weight = torch.randn(
        7168, device="cuda", dtype=torch.bfloat16, generator=generator
    )

    actual = attn_res_rmsnorm_amd(
        layer_residual=layer,
        block_residual=history,
        res_weight=res_weight,
        score_rms_weight=score_weight,
        score_eps=score_eps,
        output_rms_weight=output_weight,
        output_eps=output_eps,
        num_valid_blocks=valid_blocks,
    )
    expected = _attn_res_reference(
        layer,
        history,
        res_weight,
        score_weight,
        output_weight,
        valid_blocks,
        score_eps,
        output_eps,
    )
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=1.6e-2)


def test_attn_res_first_layer_snapshot_write() -> None:
    tokens = 2
    generator = torch.Generator(device="cuda").manual_seed(93)
    prefix = torch.randn(
        tokens, 7168, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    blocks = torch.empty(1, tokens, 7168, device="cuda", dtype=torch.bfloat16)
    res_weight = torch.randn(
        7168, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    score_weight = torch.randn(
        7168, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    output_weight = torch.randn(
        7168, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    original_prefix = prefix.clone()

    assert attn_res_fwd_available(
        prefix,
        blocks,
        res_weight,
        score_weight,
        out_norm_weight=output_weight,
        num_valid_blocks=0,
        block_write_idx=0,
    )
    actual = attn_res_fwd(
        prefix,
        blocks,
        res_weight,
        score_weight,
        out_norm_weight=output_weight,
        num_valid_blocks=0,
        block_write_idx=0,
    )
    expected = _attn_res_reference(
        original_prefix,
        blocks.transpose(0, 1),
        res_weight,
        score_weight,
        output_weight,
        0,
        1e-6,
        1e-6,
    )

    torch.testing.assert_close(prefix, original_prefix, rtol=0, atol=0)
    torch.testing.assert_close(blocks[0], original_prefix, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=1.6e-2)


@pytest.mark.parametrize("tokens", [2, 4, 8, 16, 32, 64, 65, 128, 256, 512])
def test_attn_res_delta_and_block_write_batches(tokens: int) -> None:
    valid_blocks = 3
    score_eps, output_eps = 1e-6, 2e-6
    generator = torch.Generator(device="cuda").manual_seed(100 + tokens)
    prefix = torch.randn(
        tokens, 7168, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    delta = torch.randn(
        tokens, 7168, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    blocks = torch.randn(
        valid_blocks + 1,
        tokens,
        7168,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    res_weight = torch.randn(
        7168, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    score_weight = torch.randn(
        7168, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    output_weight = torch.randn(
        7168, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    original_prefix = prefix.clone()
    original_blocks = blocks.clone()

    assert attn_res_fwd_available(
        prefix,
        blocks,
        res_weight,
        score_weight,
        eps=score_eps,
        out_norm_weight=output_weight,
        out_norm_eps=output_eps,
        delta=delta,
        num_valid_blocks=valid_blocks,
        block_write_idx=valid_blocks,
    )
    actual = attn_res_fwd(
        prefix,
        blocks,
        res_weight,
        score_weight,
        eps=score_eps,
        out_norm_weight=output_weight,
        out_norm_eps=output_eps,
        delta=delta,
        num_valid_blocks=valid_blocks,
        block_write_idx=valid_blocks,
    )

    updated_prefix = _bf16_add_rne(original_prefix, delta)
    expected = _attn_res_reference(
        updated_prefix,
        original_blocks.transpose(0, 1),
        res_weight,
        score_weight,
        output_weight,
        valid_blocks,
        score_eps,
        output_eps,
    )
    torch.testing.assert_close(prefix, updated_prefix, rtol=0, atol=0)
    torch.testing.assert_close(blocks[valid_blocks], updated_prefix, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=1.6e-2)


def test_attn_res_noncontiguous_delta_uses_fallback() -> None:
    tokens, valid_blocks = 2, 3
    generator = torch.Generator(device="cuda").manual_seed(177)
    prefix = torch.randn(
        tokens, 7168, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    delta_storage = torch.randn(
        tokens, 2 * 7168, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    delta = delta_storage[:, ::2]
    blocks = torch.randn(
        valid_blocks,
        tokens,
        7168,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    res_weight = torch.randn(
        7168, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    score_weight = torch.randn(
        7168, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    output_weight = torch.randn(
        7168, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    original_prefix = prefix.clone()

    assert not attn_res_fwd_available(
        prefix,
        blocks,
        res_weight,
        score_weight,
        out_norm_weight=output_weight,
        delta=delta,
        num_valid_blocks=valid_blocks,
    )
    actual = attn_res_fwd(
        prefix,
        blocks,
        res_weight,
        score_weight,
        out_norm_weight=output_weight,
        delta=delta,
        num_valid_blocks=valid_blocks,
    )
    updated_prefix = (original_prefix + delta).to(torch.bfloat16)
    expected = _attn_res_reference(
        updated_prefix,
        blocks.transpose(0, 1),
        res_weight,
        score_weight,
        output_weight,
        valid_blocks,
        1e-6,
        1e-6,
    )
    torch.testing.assert_close(prefix, updated_prefix, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=1.6e-2)


@pytest.mark.parametrize("tokens", [2, 64, 256])
@pytest.mark.parametrize(
    ("use_delta", "write_block"),
    [(False, True), (True, False)],
)
def test_attn_res_model_update_modes_graph_replay(
    tokens: int, use_delta: bool, write_block: bool
) -> None:
    valid_blocks = 3
    generator = torch.Generator(device="cuda").manual_seed(200 + tokens)
    prefix = torch.randn(
        tokens, 7168, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    delta = torch.randn(
        prefix.shape,
        device=prefix.device,
        dtype=prefix.dtype,
        generator=generator,
    )
    blocks = torch.randn(
        valid_blocks + 1,
        tokens,
        7168,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    res_weight = torch.randn(
        7168, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    score_weight = torch.randn(
        7168, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    output_weight = torch.randn(
        7168, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    original_prefix = prefix.clone()
    original_blocks = blocks.clone()
    call_delta = delta if use_delta else None
    block_write_idx = valid_blocks if write_block else -1

    # Compile before capture and retain the eager result as the replay oracle.
    eager_prefix = original_prefix.clone()
    eager_blocks = original_blocks.clone()
    eager_output = attn_res_fwd(
        eager_prefix,
        eager_blocks,
        res_weight,
        score_weight,
        out_norm_weight=output_weight,
        delta=call_delta,
        num_valid_blocks=valid_blocks,
        block_write_idx=block_write_idx,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = attn_res_fwd(
            prefix,
            blocks,
            res_weight,
            score_weight,
            out_norm_weight=output_weight,
            delta=call_delta,
            num_valid_blocks=valid_blocks,
            block_write_idx=block_write_idx,
        )

    prefix.copy_(original_prefix)
    blocks.copy_(original_blocks)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(prefix, eager_prefix, rtol=0, atol=0)
    torch.testing.assert_close(blocks, eager_blocks, rtol=0, atol=0)
    torch.testing.assert_close(actual, eager_output, rtol=0, atol=0)


@pytest.mark.skipif(not is_cdna4(), reason="Gluon sigmoid top-k is gfx950-only")
@pytest.mark.parametrize("tokens", [1, 17, 8192])
def test_kimi_topk_prefill_matches_reference(tokens: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(41 + tokens)
    logits = torch.randn(tokens, 896, device="cuda", generator=generator)
    bias = torch.randn(896, device="cuda", generator=generator) * 0.1
    scores = logits.sigmoid()
    _, expected_ids = torch.topk(scores + bias, 16, dim=-1, sorted=True)
    expected_weights = scores.gather(1, expected_ids)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

    actual_weights, actual_ids = moe_sigmoid_bias_topk(
        logits,
        bias,
        16,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
    )
    torch.testing.assert_close(actual_ids, expected_ids.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=2e-6, atol=2e-7)


@pytest.mark.skipif(not is_cdna4(), reason="Gluon sigmoid top-k is gfx950-only")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_kimi_topk_prefill_scales_beyond_8k(dtype: torch.dtype) -> None:
    tokens = 16384
    generator = torch.Generator(device="cuda").manual_seed(73)
    logits = torch.randn(tokens, 896, device="cuda", dtype=dtype, generator=generator)
    bias = torch.randn(896, device="cuda", generator=generator) * 0.1
    scores = logits.float().sigmoid().to(dtype)
    _, expected_ids = torch.topk(scores.float() + bias, 16, dim=-1, sorted=True)
    expected_weights = scores.gather(1, expected_ids)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

    assert _gluon_eligible(logits, bias, 16)
    actual_weights, actual_ids = moe_sigmoid_bias_topk(
        logits,
        bias,
        16,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
    )

    torch.testing.assert_close(actual_ids, expected_ids.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(
        actual_weights,
        expected_weights.float(),
        rtol=5e-3,
        atol=5e-4,
    )


@pytest.mark.skipif(not is_cdna4(), reason="Gluon sigmoid top-k is gfx950-only")
def test_kimi_topk_prefill_ties_choose_smaller_expert_id() -> None:
    logits = torch.zeros(3, 896, device="cuda")
    bias = torch.zeros(896, device="cuda")
    weights, ids = moe_sigmoid_bias_topk(logits, bias, 16)
    expected_ids = torch.arange(16, device="cuda", dtype=torch.int32).expand(3, -1)
    assert torch.equal(ids, expected_ids)
    torch.testing.assert_close(weights, torch.full_like(weights, 1 / 16))
