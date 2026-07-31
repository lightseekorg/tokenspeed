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

import pytest
import torch
from kimi3_reference import (
    a16w4_mxfp4_moe_reference,
)
from utils import make_mxfp4_moe_weights, make_round_robin_topk


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return "gfx950" in arch


if not _is_gfx950():
    pytest.skip("MXFP4 SiTU Gluon tests require gfx950", allow_module_level=True)

import tokenspeed_kernel  # noqa: E402
from tokenspeed_kernel.ops.moe.gluon.mxfp4 import (  # noqa: E402
    _use_route_direct_decode,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.situ_decode import (  # noqa: E402
    _stage2_num_warps,
)


def _make_mxfp4_module(
    *,
    num_experts: int,
    latent_size: int,
    intermediate_size: int,
    top_k: int,
    generator: torch.Generator,
) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    raw = make_mxfp4_moe_weights(
        num_experts,
        latent_size,
        intermediate_size,
        generator,
    )

    module = torch.nn.Module()
    module.w13_weight = torch.nn.Parameter(raw["w13_weight"], requires_grad=False)
    module.w13_weight_scale = torch.nn.Parameter(raw["w13_scale"], requires_grad=False)
    module.w2_weight = torch.nn.Parameter(raw["w2_weight"], requires_grad=False)
    module.w2_weight_scale = torch.nn.Parameter(raw["w2_scale"], requires_grad=False)
    module.top_k = top_k
    module.num_experts = num_experts
    module.ep_size = 1
    # The selected plan is authoritative for the activation. Keep this
    # direct-kernel test independent of the runtime MoELayer attribute.
    module.activation_situ_beta = 4.0
    module.activation_situ_linear_beta = 25.0
    return module, raw


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [(1, True), (4, True), (5, False), (8, False), (16, False)],
)
def test_gluon_route_direct_decode_dispatch_boundary_gfx950(
    num_tokens: int,
    expected: bool,
) -> None:
    w13 = torch.empty((2, 512, 256), dtype=torch.uint8)

    assert _use_route_direct_decode(torch.empty((num_tokens, 512)), w13) is expected


@pytest.mark.parametrize(
    ("num_tokens", "fuse_shared_down", "expected"),
    [(1, False, 8), (2, False, 8), (4, False, 4), (1, True, 8)],
)
def test_gluon_route_direct_stage2_warps_gfx950(
    num_tokens: int,
    fuse_shared_down: bool,
    expected: int,
) -> None:
    assert _stage2_num_warps(num_tokens, fuse_shared_down) == expected


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 16])
def test_ep_decode_matches_kimi_k3_shape_gfx950(
    num_tokens: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260720 + num_tokens)
    num_local_experts = 2
    num_experts = 16
    ep_size = 8
    ep_rank = 3
    top_k = 16
    latent_size = 3584
    intermediate_size = 3072
    module, raw = _make_mxfp4_module(
        num_experts=num_local_experts,
        latent_size=latent_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        generator=generator,
    )
    module.num_experts = num_experts
    module.num_local_experts = num_local_experts
    module.ep_size = ep_size
    module.ep_rank = ep_rank
    hidden_states = (
        torch.randn(
            (num_tokens, latent_size),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.1
    )
    expert_ids = torch.arange(num_experts, dtype=torch.int32, device="cuda")
    topk_ids = torch.stack(
        [torch.roll(expert_ids, token) for token in range(num_tokens)]
    )
    topk_weights = torch.softmax(
        torch.randn(
            (num_tokens, top_k),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        ),
        dim=-1,
    )
    router_logits = torch.zeros(
        (num_tokens, num_experts), dtype=torch.float32, device="cuda"
    )
    plan = tokenspeed_kernel.moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="situ",
        routing_mode="precomputed_topk",
        ep_size=ep_size,
        ispp=intermediate_size,
        internal_activation_dtype="input",
        solution="gluon",
    )
    assert (
        plan["apply_kernel_name"] == "gluon_mxfp4_a16w4_situ_ep_precomputed_moe_apply"
    )
    tokenspeed_kernel.moe_process_weights(plan, module)
    actual = tokenspeed_kernel.moe_apply(
        plan,
        hidden_states,
        module,
        router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )

    expert_start = ep_rank * num_local_experts
    local_ids = topk_ids - expert_start
    local_mask = (local_ids >= 0) & (local_ids < num_local_experts)
    local_ids = torch.where(local_mask, local_ids, torch.full_like(local_ids, -1))
    local_weights = torch.where(
        local_mask, topk_weights, torch.zeros_like(topk_weights)
    )
    expected = a16w4_mxfp4_moe_reference(
        hidden_states,
        raw["w13_weight"],
        raw["w13_scale"],
        raw["w2_weight"],
        raw["w2_scale"],
        local_ids,
        local_weights,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )

    assert actual.shape == (num_tokens, latent_size)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 16])
def test_ep_decode_all_remote_routes_return_zero_gfx950(
    num_tokens: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260722)
    module, _ = _make_mxfp4_module(
        num_experts=4,
        latent_size=512,
        intermediate_size=512,
        top_k=16,
        generator=generator,
    )
    module.num_experts = 32
    module.num_local_experts = 4
    module.ep_size = 8
    module.ep_rank = 7
    hidden_states = torch.randn(
        (num_tokens, 512),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    topk_ids = torch.arange(16, dtype=torch.int32, device="cuda").repeat(num_tokens, 1)
    topk_weights = torch.full(
        (num_tokens, 16), 1.0 / 16, dtype=torch.float32, device="cuda"
    )
    plan = tokenspeed_kernel.moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="situ",
        routing_mode="precomputed_topk",
        ep_size=8,
        ispp=512,
        internal_activation_dtype="input",
        solution="gluon",
    )
    tokenspeed_kernel.moe_process_weights(plan, module)
    actual = tokenspeed_kernel.moe_apply(
        plan,
        hidden_states,
        module,
        torch.zeros((num_tokens, 32), dtype=torch.float32, device="cuda"),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0.0, rtol=0.0)


def test_gluon_grouped_a16w4_situ_matches_kimi_k3_shape_gfx950() -> None:
    generator = torch.Generator(device="cuda").manual_seed(124)
    num_tokens = 17
    num_experts = 2
    top_k = 1
    latent_size = 3584
    intermediate_size = 3072
    module, raw = _make_mxfp4_module(
        num_experts=num_experts,
        latent_size=latent_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        generator=generator,
    )
    module.num_local_experts = num_experts
    module.ep_rank = 0
    hidden_states = (
        torch.randn(
            (num_tokens, latent_size),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.1
    )
    topk_weights, topk_ids = make_round_robin_topk(num_tokens, num_experts, top_k)
    router_logits = torch.zeros(
        (num_tokens, num_experts), dtype=torch.float32, device="cuda"
    )
    plan = tokenspeed_kernel.moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="situ",
        routing_mode="precomputed_topk",
        internal_activation_dtype="input",
        solution="gluon",
    )
    tokenspeed_kernel.moe_process_weights(plan, module)
    actual = tokenspeed_kernel.moe_apply(
        plan,
        hidden_states,
        module,
        router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    expected = a16w4_mxfp4_moe_reference(
        hidden_states,
        raw["w13_weight"],
        raw["w13_scale"],
        raw["w2_weight"],
        raw["w2_scale"],
        topk_ids,
        topk_weights,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def _make_local_ep_module(
    raw: dict[str, torch.Tensor],
    *,
    ep_rank: int,
    ep_size: int,
    top_k: int,
) -> torch.nn.Module:
    num_experts = int(raw["w13_weight"].shape[0])
    num_local = num_experts // ep_size
    start = ep_rank * num_local
    stop = start + num_local
    module = torch.nn.Module()
    module.w13_weight = torch.nn.Parameter(
        raw["w13_weight"][start:stop].clone(), requires_grad=False
    )
    module.w13_weight_scale = torch.nn.Parameter(
        raw["w13_scale"][start:stop].clone(), requires_grad=False
    )
    module.w2_weight = torch.nn.Parameter(
        raw["w2_weight"][start:stop].clone(), requires_grad=False
    )
    module.w2_weight_scale = torch.nn.Parameter(
        raw["w2_scale"][start:stop].clone(), requires_grad=False
    )
    module.top_k = top_k
    module.num_experts = num_experts
    module.num_local_experts = num_local
    module.ep_rank = ep_rank
    module.ep_size = ep_size
    module.activation_situ_beta = 4.0
    module.activation_situ_linear_beta = 25.0
    return module


def test_gluon_grouped_device_align_localizes_global_ep_routes_gfx950() -> None:
    """The larger-M fallback consumes global IDs without torch localization."""
    generator = torch.Generator(device="cuda").manual_seed(20260723)
    num_tokens = 33
    num_experts = 8
    ep_size = 8
    ep_rank = 3
    top_k = 4
    latent_size = 512
    intermediate_size = 512
    _, raw = _make_mxfp4_module(
        num_experts=num_experts,
        latent_size=latent_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        generator=generator,
    )
    module = _make_local_ep_module(
        raw,
        ep_rank=ep_rank,
        ep_size=ep_size,
        top_k=top_k,
    )
    hidden_states = (
        torch.randn(
            (num_tokens, latent_size),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    )
    topk_weights, topk_ids = make_round_robin_topk(
        num_tokens,
        num_experts,
        top_k,
    )
    plan = tokenspeed_kernel.moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="situ",
        routing_mode="precomputed_topk",
        ep_size=ep_size,
        ispp=intermediate_size,
        internal_activation_dtype="input",
        solution="gluon",
    )
    tokenspeed_kernel.moe_process_weights(plan, module)
    actual = tokenspeed_kernel.moe_apply(
        plan,
        hidden_states,
        module,
        torch.zeros(
            (num_tokens, num_experts),
            dtype=torch.float32,
            device="cuda",
        ),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )

    expert_start = ep_rank * module.num_local_experts
    local_ids = topk_ids - expert_start
    local_mask = (local_ids >= 0) & (local_ids < module.num_local_experts)
    local_ids = torch.where(local_mask, local_ids, torch.full_like(local_ids, -1))
    local_weights = torch.where(
        local_mask,
        topk_weights,
        torch.zeros_like(topk_weights),
    )
    expected = a16w4_mxfp4_moe_reference(
        hidden_states,
        module.w13_weight,
        module.w13_weight_scale,
        module.w2_weight,
        module.w2_weight_scale,
        local_ids,
        local_weights,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("top_k", [1, 4])
def test_mxfp4_situ_virtual_ep_sum_matches_global_reference_gfx950(
    top_k: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(321)
    num_experts = 8
    num_tokens = 8
    ep_size = 8
    _, raw = _make_mxfp4_module(
        num_experts=num_experts,
        latent_size=512,
        intermediate_size=512,
        top_k=top_k,
        generator=generator,
    )
    hidden_states = (
        torch.randn(
            num_tokens,
            512,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    )
    topk_weights, topk_ids = make_round_robin_topk(num_tokens, num_experts, top_k)
    router_logits = torch.zeros(
        num_tokens, num_experts, device="cuda", dtype=torch.float32
    )
    plan = tokenspeed_kernel.moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="situ",
        routing_mode="precomputed_topk",
        ep_size=ep_size,
        ispp=512,
        internal_activation_dtype="input",
        solution="gluon",
    )

    partials = []
    for ep_rank in range(ep_size):
        module = _make_local_ep_module(
            raw,
            ep_rank=ep_rank,
            ep_size=ep_size,
            top_k=top_k,
        )
        tokenspeed_kernel.moe_process_weights(plan, module)
        partials.append(
            tokenspeed_kernel.moe_apply(
                plan,
                hidden_states,
                module,
                router_logits,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
            )
        )
    actual = torch.stack([partial.float() for partial in partials]).sum(0)
    actual = actual.to(torch.bfloat16)
    expected = a16w4_mxfp4_moe_reference(
        hidden_states,
        raw["w13_weight"],
        raw["w13_scale"],
        raw["w2_weight"],
        raw["w2_scale"],
        topk_ids,
        topk_weights,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8])
def test_mxfp4_situ_ep_paths_are_cuda_graph_capturable_gfx950(
    num_tokens: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260718)
    top_k = 4
    _, raw = _make_mxfp4_module(
        num_experts=8,
        latent_size=512,
        intermediate_size=512,
        top_k=top_k,
        generator=generator,
    )
    ep_size = 8
    module = _make_local_ep_module(
        raw,
        ep_rank=1,
        ep_size=ep_size,
        top_k=top_k,
    )
    hidden_states = torch.randn(
        (num_tokens, 512),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    topk_weights, topk_ids = make_round_robin_topk(num_tokens, 8, top_k)
    router_logits = torch.zeros((num_tokens, 8), dtype=torch.float32, device="cuda")
    plan = tokenspeed_kernel.moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="situ",
        routing_mode="precomputed_topk",
        ep_size=ep_size,
        ispp=512,
        internal_activation_dtype="input",
        solution="gluon",
    )
    tokenspeed_kernel.moe_process_weights(plan, module)

    def apply() -> torch.Tensor:
        return tokenspeed_kernel.moe_apply(
            plan,
            hidden_states,
            module,
            router_logits,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )

    eager = apply().clone()
    warmup_stream = torch.cuda.Stream()
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            apply()
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = apply()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, eager, atol=3e-2, rtol=3e-2)
