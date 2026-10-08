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

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from utils import (
    assert_no_triton_compile,
    int_specialization_class,
    is_cdna5,
    warm_specialization_classes,
)

if not is_cdna5():
    pytest.skip(
        "AMD CDNA5 is required for gfx1250 Gluon block-FP8 MoE tests",
        allow_module_level=True,
    )

from tokenspeed_kernel.ops.moe import (  # noqa: E402
    moe_apply,
    moe_plan,
    moe_process_weights,
)
from tokenspeed_kernel.ops.moe.triton import _common  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx1250.moe.fp8 import block_experts  # noqa: E402

_KERNEL_NAME = "gluon_fp8_block_precomputed_moe_apply_gfx1250"
_HIDDEN = 512
_INTERMEDIATE = 512


def _block_fp8(
    shape: tuple[int, int, int], generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_experts, rows, cols = shape
    weight = torch.randn(shape, device="cuda", generator=generator).to(
        torch.float8_e4m3fn
    )
    scale = (
        torch.rand(
            (num_experts, rows // 128, cols // 128),
            device="cuda",
            generator=generator,
        )
        + 0.5
    ) / cols**0.5
    dequantized = weight.float() * scale.repeat_interleave(128, 1).repeat_interleave(
        128, 2
    )
    return weight, scale, dequantized


def _experts(
    num_experts: int, generator: torch.Generator, swiglu_limit: float | None = None
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    w13, w13_scale, w13_dequant = _block_fp8(
        (num_experts, 2 * _INTERMEDIATE, _HIDDEN), generator
    )
    w2, w2_scale, w2_dequant = _block_fp8(
        (num_experts, _HIDDEN, _INTERMEDIATE), generator
    )
    weights = torch.nn.Module()
    weights.w13_weight = w13
    weights.w13_weight_scale_inv = w13_scale
    weights.w2_weight = w2
    weights.w2_weight_scale_inv = w2_scale
    weights.ep_size = 1
    weights.swiglu_arg = SimpleNamespace(alpha=None, limit=swiglu_limit)
    return weights, w13_dequant, w2_dequant


def _plan(activation: str, swiglu_limit: float | None) -> dict:
    plan = moe_plan(
        "fp8",
        input_dtype=torch.bfloat16,
        activation=activation,
        routing_mode="precomputed_topk",
        ispp=_INTERMEDIATE,
        fp8_scale_block_shape=(128, 128),
        internal_activation_dtype="input",
        hidden=_HIDDEN,
        swiglu_form=("standard" if activation == "swiglu" else None),
        activation_clamped=swiglu_limit is not None,
        expert_id_repeats=False,
        fast_math=True,
        combine_order="rank",
    )
    assert plan["apply_kernel_name"] == _KERNEL_NAME
    return plan


def _topk_ids(
    num_tokens: int, num_experts: int, top_k: int, generator: torch.Generator
) -> torch.Tensor:
    return (
        torch.rand(num_tokens, num_experts, device="cuda", generator=generator)
        .argsort(dim=1)[:, :top_k]
        .to(torch.int32)
    )


def _reference(
    x: torch.Tensor,
    w13_dequant: torch.Tensor,
    w2_dequant: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    swiglu_limit: float | None,
    expert_start: int = 0,
) -> torch.Tensor:
    expected = torch.zeros_like(x, dtype=torch.float32)
    for local_expert in range(w13_dequant.shape[0]):
        token_ids, slots = torch.where(topk_ids == expert_start + local_expert)
        gate, up = F.linear(x[token_ids].float(), w13_dequant[local_expert]).chunk(
            2, dim=-1
        )
        if swiglu_limit is not None:
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(-swiglu_limit, swiglu_limit)
        intermediate = (F.silu(gate) * up).to(torch.bfloat16)
        expert_output = F.linear(intermediate.float(), w2_dequant[local_expert]).to(
            torch.bfloat16
        )
        expected.index_add_(
            0,
            token_ids,
            expert_output.float() * topk_weights[token_ids, slots, None],
        )
    return expected


@pytest.mark.parametrize("num_tokens", [1, 33, 2048])
@pytest.mark.parametrize(
    ("activation", "swiglu_limit"), [("silu", None), ("swiglu", 1.5)]
)
def test_gluon_fp8_moe_gfx1250_matches_torch(
    num_tokens: int, activation: str, swiglu_limit: float | None
) -> None:
    num_experts, top_k = 6, 3
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(
        num_tokens, _HIDDEN, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    weights, w13_dequant, w2_dequant = _experts(num_experts, generator, swiglu_limit)
    topk_ids = _topk_ids(num_tokens, num_experts, top_k, generator)
    topk_ids[0, 1] = -1
    topk_ids[-1, 0] = num_experts
    topk_weights = torch.rand(
        num_tokens, top_k, device="cuda", dtype=torch.float32, generator=generator
    )

    plan = _plan(activation, swiglu_limit)
    moe_process_weights(plan, weights)
    actual = moe_apply(
        plan,
        x,
        weights,
        torch.empty((num_tokens, num_experts), device="cuda"),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )

    expected = _reference(
        x, w13_dequant, w2_dequant, topk_ids, topk_weights, swiglu_limit
    )
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.01)


def test_gluon_fp8_moe_gfx1250_ep_filters_nonlocal_routes() -> None:
    from tokenspeed_kernel.ops.moe.gluon.fp8 import (
        gluon_fp8_block_precomputed_moe_apply_gfx1250,
    )

    num_tokens, num_local_experts, top_k, ep_rank = 9, 3, 2, 1
    generator = torch.Generator(device="cuda").manual_seed(19)
    x = torch.randn(
        num_tokens, _HIDDEN, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    weights, w13_dequant, w2_dequant = _experts(num_local_experts, generator, 7.0)
    weights.ep_size = 2
    weights.ep_rank = ep_rank
    weights.num_local_experts = num_local_experts
    topk_ids = _topk_ids(num_tokens, 2 * num_local_experts, top_k, generator)
    topk_weights = torch.rand(
        num_tokens, top_k, device="cuda", dtype=torch.float32, generator=generator
    )

    actual = gluon_fp8_block_precomputed_moe_apply_gfx1250(
        {"activation": "swiglu"},
        x,
        weights,
        torch.empty((num_tokens, 2 * num_local_experts), device="cuda"),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )

    expected = _reference(
        x,
        w13_dequant,
        w2_dequant,
        topk_ids,
        topk_weights,
        7.0,
        expert_start=ep_rank * num_local_experts,
    )
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.01)


def test_gluon_fp8_moe_gfx1250_token_count() -> None:
    num_experts, top_k = 4, 2
    max_tokens = 1500
    generator = torch.Generator(device="cuda").manual_seed(0)
    weights, _, _ = _experts(num_experts, generator)
    plan = _plan("silu", None)
    moe_process_weights(plan, weights)
    x = torch.randn(
        max_tokens, _HIDDEN, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    topk_ids = _topk_ids(max_tokens, num_experts, top_k, generator)
    topk_weights = torch.rand(
        max_tokens, top_k, device="cuda", dtype=torch.float32, generator=generator
    )

    def run(tokens):
        return moe_apply(
            plan,
            x[:tokens],
            weights,
            torch.empty((tokens, num_experts), device="cuda"),
            topk_weights=topk_weights[:tokens],
            topk_ids=topk_ids[:tokens],
        )

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    def tile(tokens):
        return block_experts._select_tiles(
            tokens * top_k, num_experts, _INTERMEDIATE, num_sms
        )

    def key(tokens):
        routes = tokens * top_k
        counts = (tokens, routes)
        # Routing and combine still key on the token and route counts.
        return (*map(int_specialization_class, counts), tile(tokens), routes <= 128)

    # Tokens are independent: a shorter batch is a prefix of the longest one
    # that uses the same tiles.
    sweep = (3, 7, 12, 24, 100, 300, 1000, 1483)
    longest = {}
    for tokens in (*sweep, max_tokens):
        longest[tile(tokens)] = max(longest.get(tile(tokens), 0), tokens)
    expected = {bucket: run(tokens) for bucket, tokens in longest.items()}
    warm_specialization_classes(run, key, sweep, range(1, max_tokens))
    with assert_no_triton_compile(
        block_experts._fp8_block_gate_up_gfx1250,
        block_experts._fp8_block_down_gfx1250,
        _common._routing_kernel,
        _common._combine_kernel,
    ):
        for tokens in sweep:
            torch.testing.assert_close(
                run(tokens), expected[tile(tokens)][:tokens], rtol=0, atol=0
            )


def test_gluon_fp8_block_experts_launch_metadata() -> None:
    routes, hidden, inter, experts = 6, 512, 1024, 8
    bf16 = torch.empty(0, dtype=torch.bfloat16)
    e4m3 = torch.empty(0, dtype=torch.float8_e4m3fn)
    sizes = {
        "num_routes": routes,
        "HIDDEN": hidden,
        "INTERMEDIATE": inter,
        "NUM_EXPERTS": experts,
    }
    gate_up = block_experts._gate_up_launch_metadata(
        None,
        SimpleNamespace(name="gate_up"),
        {**sizes, "x": bf16, "w13": e4m3, "intermediate": bf16},
    )
    down = block_experts._down_launch_metadata(
        None,
        SimpleNamespace(name="down"),
        {**sizes, "intermediate": bf16, "w2": e4m3, "route_output": bf16},
    )

    # Six routes reach at most six of the eight experts.
    assert gate_up == {
        "name": "gate_up",
        "flops16": 2 * routes * 2 * inter * hidden,
        "bytes": routes * hidden * 2 + 6 * 2 * inter * hidden + routes * inter * 2,
    }
    assert down == {
        "name": "down",
        "flops16": 2 * routes * hidden * inter,
        "bytes": routes * inter * 2 + 6 * hidden * inter + routes * hidden * 2,
    }
    assert (
        block_experts._fp8_block_gate_up_gfx1250.launch_metadata
        is block_experts._gate_up_launch_metadata
    )
    assert (
        block_experts._fp8_block_down_gfx1250.launch_metadata
        is block_experts._down_launch_metadata
    )
