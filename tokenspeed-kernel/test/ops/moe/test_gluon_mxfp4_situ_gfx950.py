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
    mxfp4_moe_reference,
)
from utils import is_cdna4, make_mxfp4_moe_weights, make_round_robin_topk

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for Gluon MXFP4 SiTU tests",
        allow_module_level=True,
    )

import tokenspeed_kernel  # noqa: E402


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
    expected = mxfp4_moe_reference(
        hidden_states,
        raw["w13_weight"],
        raw["w13_scale"],
        raw["w2_weight"],
        raw["w2_scale"],
        local_ids,
        local_weights,
        activation_dtype=torch.bfloat16,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )

    assert actual.shape == (num_tokens, latent_size)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-2)


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
    expected = mxfp4_moe_reference(
        hidden_states,
        raw["w13_weight"],
        raw["w13_scale"],
        raw["w2_weight"],
        raw["w2_scale"],
        topk_ids,
        topk_weights,
        activation_dtype=torch.bfloat16,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-2)


def test_grouped_atomic_combine_matches_partial_reduction_gfx950(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The FP32 atomic stage-2 epilogue must match the default partials path.

    ``GROUPED_ATOMIC_COMBINE_MAX_TOKENS`` is 0 because atomics are slower on
    every reachable batch, so this is the only coverage the atomic branch gets.
    Keep it: the constant exists so the crossover can be re-measured, and that
    is only safe while the branch is known to be correct.
    """
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4 import situ_grouped

    generator = torch.Generator(device="cuda").manual_seed(20260809)
    num_tokens = 12
    num_experts = 8
    top_k = 4
    latent_size = 3584
    intermediate_size = 512
    raw = make_mxfp4_moe_weights(num_experts, latent_size, intermediate_size, generator)
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

    def run() -> torch.Tensor:
        return situ_grouped.gluon_a16w4_situ_grouped_ep_gfx950(
            hidden_states,
            raw["w13_weight"],
            raw["w13_scale"],
            raw["w2_weight"],
            raw["w2_scale"],
            topk_weights,
            topk_ids,
            situ_beta=4.0,
            situ_linear_beta=25.0,
            block_m=64,
            expert_start=0,
        )

    monkeypatch.setattr(
        situ_grouped, "GROUPED_ATOMIC_COMBINE_MAX_TOKENS", 0, raising=True
    )
    partials = run()
    monkeypatch.setattr(
        situ_grouped, "GROUPED_ATOMIC_COMBINE_MAX_TOKENS", num_tokens, raising=True
    )
    atomic = run()

    expected = mxfp4_moe_reference(
        hidden_states,
        raw["w13_weight"],
        raw["w13_scale"],
        raw["w2_weight"],
        raw["w2_scale"],
        topk_ids,
        topk_weights,
        activation_dtype=torch.bfloat16,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )
    torch.testing.assert_close(partials, expected, atol=2e-4, rtol=2e-2)
    torch.testing.assert_close(atomic, expected, atol=2e-4, rtol=2e-2)


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
    expected = mxfp4_moe_reference(
        hidden_states,
        module.w13_weight,
        module.w13_weight_scale,
        module.w2_weight,
        module.w2_weight_scale,
        local_ids,
        local_weights,
        activation_dtype=torch.bfloat16,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )

    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-2)


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
    expected = mxfp4_moe_reference(
        hidden_states,
        raw["w13_weight"],
        raw["w13_scale"],
        raw["w2_weight"],
        raw["w2_scale"],
        topk_ids,
        topk_weights,
        activation_dtype=torch.bfloat16,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )

    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-2)


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 16, 32])
def test_mxfp4_situ_ep_paths_are_cuda_graph_capturable_gfx950(
    num_tokens: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tokenspeed_kernel.ops.moe.gluon import mxfp4 as mxfp4_module

    grouped = mxfp4_module.gluon_a16w4_situ_grouped_ep_gfx950
    grouped_calls = 0

    def tracked_grouped(*args, **kwargs):
        nonlocal grouped_calls
        grouped_calls += 1
        return grouped(*args, **kwargs)

    monkeypatch.setattr(
        mxfp4_module,
        "gluon_a16w4_situ_grouped_ep_gfx950",
        tracked_grouped,
    )
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
    expected = tokenspeed_kernel.moe_apply(
        plan,
        hidden_states,
        module,
        router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    ).clone()
    assert bool(grouped_calls) == (num_tokens >= 16)
    output = torch.empty_like(hidden_states)
    module._situ_output_buffer = output

    def apply() -> torch.Tensor:
        result = tokenspeed_kernel.moe_apply(
            plan,
            hidden_states,
            module,
            router_logits,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )
        assert result.data_ptr() == output.data_ptr()
        return result

    eager = apply().clone()
    torch.testing.assert_close(eager, expected, atol=0.0, rtol=0.0)
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
    torch.testing.assert_close(captured, eager, atol=3e-4, rtol=3e-2)


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 32])
def test_tp_situ_selects_a8w4_and_matches_reference_gfx950(
    num_tokens: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260818 + num_tokens)
    num_experts, latent_size, intermediate_size, top_k = 16, 3584, 384, 16
    module, raw = _make_mxfp4_module(
        num_experts=num_experts,
        latent_size=latent_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        generator=generator,
    )
    hidden_states = 0.1 * torch.randn(
        num_tokens,
        latent_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
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
        ep_size=1,
        ispp=intermediate_size,
        internal_activation_dtype="input",
        solution="gluon",
    )
    assert plan["apply_kernel_name"] == "gluon_mxfp4_a8w4_situ_precomputed_moe_apply"
    tokenspeed_kernel.moe_process_weights(plan, module)
    actual = tokenspeed_kernel.moe_apply(
        plan,
        hidden_states,
        module,
        router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    expected = mxfp4_moe_reference(
        hidden_states,
        raw["w13_weight"],
        raw["w13_scale"],
        raw["w2_weight"],
        raw["w2_scale"],
        topk_ids,
        topk_weights,
        activation_dtype=torch.bfloat16,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )

    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=8e-2)
    if num_tokens <= 4:
        output = torch.empty_like(actual)
        module._situ_output_buffer = output

        buffered = tokenspeed_kernel.moe_apply(
            plan,
            hidden_states,
            module,
            router_logits,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )
        assert buffered.data_ptr() == output.data_ptr()
        torch.testing.assert_close(buffered, expected, atol=2e-3, rtol=8e-2)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = tokenspeed_kernel.moe_apply(
                plan,
                hidden_states,
                module,
                router_logits,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
            )
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, atol=2e-3, rtol=8e-2)


@pytest.mark.parametrize(
    ("num_tokens", "num_experts", "top_k"),
    [(257, 17, 4), (4097, 3, 1)],
)
def test_package_prefill_sort_contract_gfx950(
    num_tokens: int,
    num_experts: int,
    top_k: int,
) -> None:
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.moe_sorting import (
        _max_padded_route_capacity,
        gluon_moe_sorting,
    )

    generator = torch.Generator().manual_seed(20260829 + num_tokens)
    topk_ids_cpu = torch.randint(
        num_experts,
        (num_tokens, top_k),
        dtype=torch.int32,
        generator=generator,
    )
    topk_ids_cpu.view(-1)[::13] = -1
    topk_weights_cpu = torch.arange(
        num_tokens * top_k,
        dtype=torch.float32,
    ).view(num_tokens, top_k)
    topk_ids = topk_ids_cpu.to("cuda")
    topk_weights = topk_weights_cpu.to("cuda")
    block_m = 64

    sorted_ids, sorted_weights, sorted_experts, num_valid, _ = gluon_moe_sorting(
        topk_ids,
        topk_weights,
        num_experts,
        1,
        torch.bfloat16,
        block_m,
    )
    valid_extent, reported_tokens = num_valid.cpu().tolist()
    assert reported_tokens == num_tokens
    assert sorted_ids.numel() == _max_padded_route_capacity(
        num_tokens * top_k,
        num_experts,
        block_m,
    )
    expected_extent = sum(
        ((int((topk_ids_cpu == expert).sum()) + block_m - 1) // block_m) * block_m
        for expert in range(num_experts)
    )
    assert valid_extent == expected_extent

    ids = sorted_ids[:valid_extent].cpu()
    weights = sorted_weights[:valid_extent].cpu()
    experts = sorted_experts[: valid_extent // block_m].cpu()
    expected_routes = {
        (slot << 24) | token
        for token in range(num_tokens)
        for slot in range(top_k)
        if int(topk_ids_cpu[token, slot]) >= 0
    }
    actual_routes: set[int] = set()
    actual_route_count = 0
    for row, packed_tensor in enumerate(ids):
        packed = int(packed_tensor)
        token = packed & 0xFFFFFF
        if token == num_tokens:
            continue
        slot = packed >> 24
        expert = int(experts[row // block_m])
        assert int(topk_ids_cpu[token, slot]) == expert
        assert float(weights[row]) == float(topk_weights_cpu[token, slot])
        actual_routes.add(packed)
        actual_route_count += 1
    assert actual_route_count == len(expected_routes)
    assert actual_routes == expected_routes


def test_package_prefill_low_density_route_capacity_gfx950() -> None:
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.moe_sorting import (
        _max_padded_route_capacity,
        gluon_moe_sorting,
    )

    num_tokens, top_k, num_experts, block_m = 9, 16, 896, 128
    topk_ids = torch.arange(
        num_tokens * top_k,
        dtype=torch.int32,
        device="cuda",
    ).view(num_tokens, top_k)
    topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
    sorted_ids, _, sorted_experts, num_valid, _ = gluon_moe_sorting(
        topk_ids,
        topk_weights,
        num_experts,
        1,
        torch.bfloat16,
        block_m,
    )

    expected_capacity = _max_padded_route_capacity(
        num_tokens * top_k,
        num_experts,
        block_m,
    )
    assert expected_capacity == 18_432
    assert sorted_ids.numel() == expected_capacity
    assert sorted_experts.numel() == expected_capacity // block_m
    assert int(num_valid[0].cpu()) == expected_capacity


def test_stage1_quantized_output_rejects_partial_scale_panel_gfx950() -> None:
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.prefill_stage1 import (
        invoke_gluon_mxfp4_moe_stage1,
    )

    hidden_states = torch.empty((64, 64), dtype=torch.uint8, device="cuda")
    w1 = torch.empty((1, 256, 64), dtype=torch.uint8, device="cuda")
    sorted_ids = torch.zeros(64, dtype=torch.int32, device="cuda")
    sorted_experts = torch.zeros(1, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([64, 64], dtype=torch.int32, device="cuda")
    out = torch.empty((64, 64), dtype=torch.uint8, device="cuda")
    out_scale = torch.empty((64, 4), dtype=torch.uint8, device="cuda")
    a_scale = torch.empty((64, 4), dtype=torch.uint8, device="cuda")
    w_scale = torch.empty((1, 256, 4), dtype=torch.uint8, device="cuda")

    with pytest.raises(ValueError, match="complete CDNA4 scale panels"):
        invoke_gluon_mxfp4_moe_stage1(
            hidden_states,
            w1,
            None,
            sorted_ids,
            sorted_experts,
            num_valid,
            out,
            1,
            w1_scale=w_scale,
            a1_scale=a_scale,
            block_m=64,
            sorted_weights=None,
            b_preshuffled=True,
            output_sorted=True,
            output_quantized=True,
            out_scale=out_scale,
            output_k=128,
        )


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [(128, 128), (129, 64), (192, 64), (193, 128)],
)
def test_package_prefill_block_m_selection_gfx950(
    num_tokens: int,
    expected: int,
) -> None:
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.moe import (
        _select_package_prefill_block_m,
    )

    assert _select_package_prefill_block_m(num_tokens, 16, 16) == expected


def test_tp_situ_package_prefill_block64_matches_block128_gfx950(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import moe as fused_moe

    generator = torch.Generator(device="cuda").manual_seed(20260828)
    num_tokens, num_experts, top_k = 160, 16, 16
    latent_size, intermediate_size = 3584, 384
    module, _ = _make_mxfp4_module(
        num_experts=num_experts,
        latent_size=latent_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        generator=generator,
    )
    hidden_states = torch.randn(
        num_tokens,
        latent_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    topk_weights, topk_ids = make_round_robin_topk(
        num_tokens,
        num_experts,
        top_k,
    )
    router_logits = torch.empty((num_tokens, 0), dtype=torch.float32, device="cuda")
    plan = tokenspeed_kernel.moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="situ",
        routing_mode="precomputed_topk",
        ep_size=1,
        ispp=intermediate_size,
        internal_activation_dtype="input",
        solution="gluon",
    )
    tokenspeed_kernel.moe_process_weights(plan, module)

    block64 = tokenspeed_kernel.moe_apply(
        plan,
        hidden_states,
        module,
        router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    monkeypatch.setattr(
        fused_moe,
        "_select_package_prefill_block_m",
        lambda *_args: 128,
    )
    block128 = tokenspeed_kernel.moe_apply(
        plan,
        hidden_states,
        module,
        router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )

    torch.testing.assert_close(block64, block128, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4])
def test_tp_situ_joint_shared_projection_gfx950(num_tokens: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260818 + num_tokens)
    module, _ = _make_mxfp4_module(
        num_experts=16,
        latent_size=3584,
        intermediate_size=384,
        top_k=16,
        generator=generator,
    )
    hidden_states = torch.randn(
        num_tokens,
        3584,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    topk_weights, topk_ids = make_round_robin_topk(num_tokens, 16, 16)
    router_logits = torch.zeros(num_tokens, 16, dtype=torch.float32, device="cuda")
    plan = tokenspeed_kernel.moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="situ",
        routing_mode="precomputed_topk",
        ep_size=1,
        ispp=384,
        internal_activation_dtype="input",
        solution="gluon",
    )
    tokenspeed_kernel.moe_process_weights(plan, module)
    routed_reference = tokenspeed_kernel.moe_apply(
        plan,
        hidden_states,
        module,
        router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    shared_input = torch.randn(
        num_tokens, 768, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    shared_weight = torch.randn(
        7168, 768, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    shared_reference = shared_input @ shared_weight.T
    shared_out = torch.empty_like(shared_reference)

    routed, shared = tokenspeed_kernel.moe_apply(
        plan,
        hidden_states,
        module,
        router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_input=shared_input,
        shared_weight=shared_weight,
        shared_out=shared_out,
    )

    torch.testing.assert_close(routed, routed_reference, atol=0, rtol=0)
    assert shared.data_ptr() == shared_out.data_ptr()
    torch.testing.assert_close(shared, shared_reference, atol=0.125, rtol=2e-2)

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_routed, captured_shared = tokenspeed_kernel.moe_apply(
            plan,
            hidden_states,
            module,
            router_logits,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_input=shared_input,
            shared_weight=shared_weight,
            shared_out=shared_out,
        )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured_routed, routed_reference, atol=0, rtol=0)
    torch.testing.assert_close(captured_shared, shared_reference, atol=0.125, rtol=2e-2)
