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
from kimi3_reference import dequantize_mxfp4
from tokenspeed_kernel import (
    dsv4_select_experts,
    moe_apply,
    moe_plan,
    moe_process_weights,
)
from tokenspeed_kernel.platform import PlatformInfo
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import SelectionObjective, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def test_router_selection(
    mi350_platform: PlatformInfo, mi450_platform: PlatformInfo
) -> None:
    for platform in (mi350_platform, mi450_platform):
        for tokens in (1, 17):
            kernel = select_kernel(
                "moe",
                "dsv4_select_experts",
                format_signature(router_logits=dense_tensor_format(torch.float32)),
                traits={
                    "tokens": tokens,
                    "experts": 256,
                    "top_k": 6,
                    "renormalize": True,
                    "routing_kind": "bias",
                },
                features=None,
                platform=platform,
                objective=SelectionObjective.DEFAULT,
                solution=None,
                override=None,
            )
            specialized = "gluon_dsv4_select_experts_gfx950"
            expected = (
                specialized
                if platform.is_cdna4
                and tokens == 1
                and KernelRegistry.get().get_by_name(specialized) is not None
                else "triton_dsv4_select_experts"
            )
            assert kernel.name == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("kind", ["plain", "bias", "hash"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("renormalize,need_scores", [(False, True), (True, False)])
def test_router_matches_reference(
    kind: str, dtype: torch.dtype, renormalize: bool, need_scores: bool
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(35)
    # Non-contiguous logits also cover a prefill batch beyond the gfx950 specialization.
    logits = torch.randn((17, 768), device="cuda", dtype=dtype, generator=generator)[
        :, ::2
    ]
    logits[0] = torch.linspace(-100, 100, 384, device="cuda").to(dtype)
    bias = (
        torch.randn((384,), device="cuda", dtype=torch.float32, generator=generator)
        if kind == "bias"
        else None
    )
    table = (
        torch.stack(
            [
                torch.randperm(384, device="cuda", generator=generator)[:6]
                for _ in range(20)
            ]
        ).to(torch.int32)
        if kind == "hash"
        else None
    )
    input_ids = (
        torch.arange(17, device="cuda", dtype=torch.int64)
        if table is not None
        else None
    )
    weights, ids, scores = dsv4_select_experts(
        logits,
        top_k=6,
        renormalize=renormalize,
        correction_bias=bias,
        hash_indices_table=table,
        input_ids=input_ids,
        need_scores=need_scores,
        override=None,
        solution="triton",
    )
    expected_scores = F.softplus(logits.float()).sqrt()
    expected_ids = (
        table[input_ids].long()
        if table is not None
        else (expected_scores if bias is None else expected_scores + bias).argsort(
            dim=-1, descending=True, stable=True
        )[:, :6]
    )
    expected_weights = expected_scores.gather(1, expected_ids)
    if renormalize:
        expected_weights /= expected_weights.sum(-1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).tiny
        )
    torch.testing.assert_close(ids.long(), expected_ids)
    torch.testing.assert_close(weights, expected_weights, atol=1e-6, rtol=2e-6)
    if need_scores:
        torch.testing.assert_close(scores, expected_scores, atol=1e-6, rtol=2e-6)
    else:
        assert scores is logits


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_router_ties_and_graph_replay() -> None:
    logits = torch.zeros((2, 256), device="cuda", dtype=torch.float32)

    def run():
        return dsv4_select_experts(
            logits,
            top_k=6,
            renormalize=True,
            correction_bias=None,
            hash_indices_table=None,
            input_ids=None,
            need_scores=True,
            override=None,
            solution=None,
        )

    weights, ids, _ = run()
    torch.testing.assert_close(
        ids, torch.arange(6, device="cuda", dtype=torch.int32).repeat(2, 1)
    )
    torch.testing.assert_close(weights, torch.full_like(weights, 1 / 6))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    logits.copy_(
        torch.arange(512, device="cuda", dtype=torch.float32).reshape(2, 256) / 32
    )
    graph.replay()
    eager = run()
    for actual, expected in zip(captured, eager, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("tokens", [1, 17])
def test_router_to_mxfp4_experts(tokens: int) -> None:
    """Exercise DSv4's BF16-input, concatenated-weight, clipped SwiGLU path."""
    generator = torch.Generator(device="cuda").manual_seed(95)
    experts, hidden, intermediate, top_k = 8, 128, 64, 6
    x = torch.randn(
        (tokens, hidden), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    logits = torch.randn(
        (tokens, experts), device="cuda", dtype=torch.float32, generator=generator
    )
    bias = torch.randn(
        (experts,), device="cuda", dtype=torch.float32, generator=generator
    )
    weights = torch.nn.Module()
    for name, rows, cols in (
        ("w13", 2 * intermediate, hidden),
        ("w2", hidden, intermediate),
    ):
        setattr(
            weights,
            f"{name}_weight",
            torch.randint(
                0,
                256,
                (experts, rows, cols // 2),
                device="cuda",
                dtype=torch.uint8,
                generator=generator,
            ),
        )
        setattr(
            weights,
            f"{name}_weight_scale",
            torch.full(
                (experts, rows, cols // 32), 123, device="cuda", dtype=torch.uint8
            ),
        )
    weights.top_k = top_k
    weights.ep_size = 1
    weights.activation = "swiglu"
    weights.w13_input_layout = "concatenated"
    weights.swiglu_arg = SimpleNamespace(alpha=1.0, limit=0.75)
    weights.swiglu_beta = 0.0
    plan = moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        requires_deferred_finalize=False,
        routing_mode="precomputed_topk",
        a2a_backend=None,
        ep_size=1,
        ispp=intermediate,
        fp8_scale_block_shape=None,
        internal_activation_dtype="input",
        with_bias=False,
        deepep_group=None,
        deepep_mode=None,
        deepep_low_latency_max_num_tokens_per_gpu=None,
        solution="triton",
    )
    assert plan["apply_kernel_name"] == "triton_mxfp4_precomputed_moe_apply"
    moe_process_weights(plan, weights)
    w13 = dequantize_mxfp4(weights.w13_weight, weights.w13_weight_scale, group_size=32)
    w2 = dequantize_mxfp4(weights.w2_weight, weights.w2_weight_scale, group_size=32)

    def run() -> torch.Tensor:
        route_weights, route_ids, _ = dsv4_select_experts(
            logits,
            top_k=top_k,
            renormalize=True,
            correction_bias=bias,
            hash_indices_table=None,
            input_ids=None,
            need_scores=False,
            override=None,
            solution="triton",
        )
        return moe_apply(
            plan,
            x,
            weights,
            logits,
            topk_weights=route_weights,
            topk_ids=route_ids,
            num_tokens_global=None,
            max_num_tokens_per_gpu=None,
            do_finalize=True,
            low_latency=None,
            overlap_fn=None,
            shared_input=None,
            shared_weight=None,
            shared_out=None,
        )

    def reference() -> torch.Tensor:
        scores = F.softplus(logits).sqrt()
        route_ids = (scores + bias).argsort(dim=-1, descending=True, stable=True)[
            :, :top_k
        ]
        route_weights = scores.gather(1, route_ids)
        route_weights /= route_weights.sum(-1, keepdim=True)
        expected = torch.zeros_like(x, dtype=torch.float32)
        for expert in range(experts):
            token_ids, routes = torch.where(route_ids == expert)
            gate, up = F.linear(x[token_ids].float(), w13[expert].float()).chunk(
                2, dim=-1
            )
            activation = (
                F.silu(gate.clamp_max(0.75)) * up.clamp(-0.75, 0.75)
            ).bfloat16()
            output = F.linear(activation.float(), w2[expert].float()).bfloat16()
            expected.index_add_(
                0, token_ids, output.float() * route_weights[token_ids, routes, None]
            )
        return expected.to(x.dtype)

    torch.testing.assert_close(run(), reference(), atol=4e-3, rtol=1e-2)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    x.mul_(1.25)
    logits.neg_()
    graph.replay()
    torch.testing.assert_close(captured, reference(), atol=4e-3, rtol=1e-2)
