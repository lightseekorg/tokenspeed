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
from tokenspeed_kernel.numerics.torch.activation import torch_silu_and_mul
from tokenspeed_kernel.numerics.torch.moe import torch_moe_experts
from tokenspeed_kernel.ops.moe.native import torch_softmax_topk
from utils import assert_no_triton_compile, is_cdna4

pytest.importorskip(
    "tokenspeed_kernel_amd.ops.gfx950.moe.fp16",
    reason="tokenspeed-kernel-amd is required for Gluon BF16 MoE tests",
)

requires_cdna4 = pytest.mark.skipif(
    not is_cdna4(), reason="AMD CDNA4 is required for BF16-weight Gluon MoE tests"
)

from tokenspeed_kernel_amd.ops.gfx950.moe.fp16 import gluon_bf16_moe  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.moe_align_device import (  # noqa: E402
    moe_align_block_size_device,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.moe_align_fused import (  # noqa: E402
    _fused_align_kernel,
    moe_align_block_size_fused,
)

# DeepSeek-V3 TP=8 MoE reference shape.
E = 256
D = 7168
I_R = 256
TOPK = 8
REL_TOL = 2e-2


def _routing_softmax_topk(logits: torch.Tensor, topk: int):
    """Softmax router with renormalised top-k weights."""
    weights, ids = torch_softmax_topk(
        router_logits=logits,
        topk=topk,
        topk_indices_dtype=torch.int32,
        renormalize=True,
        routed_scaling_factor=1.0,
    )
    return ids, weights


def _torch_moe_ref(x, w13, w2, topk_ids, topk_weights) -> torch.Tensor:
    """Golden fp32 MoE FFN: y[t] = sum_s w * (silu(h@wg) * (h@wu)) @ wd."""
    return torch_moe_experts(
        x,
        w13,
        w2,
        topk_ids,
        topk_weights,
        activation=lambda gate_up: torch_silu_and_mul(gate_up, None),
        w13_bias=None,
        w2_bias=None,
        quantize_input=None,
        quantize_intermediate=None,
        round_expert_output=None,
        first_expert=0,
    )


def _build(num_tokens: int, seed: int = 0, i_r: int = I_R, hidden_size: int = D):
    g = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda", generator=g
    )
    # Scale weights down so the fp32 reference stays in a comparable range.
    w1 = (
        torch.randn(
            E, 2 * i_r, hidden_size, dtype=torch.bfloat16, device="cuda", generator=g
        )
        * 0.05
    )
    w2 = (
        torch.randn(
            E, hidden_size, i_r, dtype=torch.bfloat16, device="cuda", generator=g
        )
        * 0.05
    )
    logits = torch.randn(num_tokens, E, dtype=torch.float32, device="cuda", generator=g)
    topk_ids, topk_weights = _routing_softmax_topk(logits, TOPK)
    return hidden, w1, w2, topk_ids, topk_weights


@requires_cdna4
def test_small_route_device_alignment_contract() -> None:
    """The two-launch route path preserves every packed route and block."""
    num_tokens, num_experts, top_k, block_m = 64, 4, 2, 16
    topk_ids = (
        torch.arange(num_tokens * top_k, device="cuda", dtype=torch.int32)
        .reshape(num_tokens, top_k)
        .remainder(num_experts)
    )
    topk_weights = torch.arange(
        num_tokens * top_k, device="cuda", dtype=torch.float32
    ).reshape(num_tokens, top_k)

    outputs = moe_align_block_size_device(topk_ids, topk_weights, num_experts, block_m)
    _assert_production_alignment_contract(
        topk_ids,
        topk_weights,
        outputs,
        num_experts=num_experts,
        block_m=block_m,
        expert_start=0,
    )


def _assert_production_alignment_contract(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    num_experts: int,
    block_m: int,
    expert_start: int,
) -> None:
    sorted_ids, sorted_experts, sorted_weights, num_valid = outputs
    num_tokens, top_k = topk_ids.shape
    local_ids = topk_ids.reshape(-1) - expert_start
    local_mask = (local_ids >= 0) & (local_ids < num_experts)
    counts = torch.bincount(
        local_ids[local_mask].to(torch.int64), minlength=num_experts
    )
    blocks_per_expert = (counts + block_m - 1) // block_m
    expected_experts = torch.arange(
        num_experts, device="cuda", dtype=torch.int32
    ).repeat_interleave(blocks_per_expert)
    valid_rows = int((blocks_per_expert.sum() * block_m).item())

    assert int(num_valid.item()) == valid_rows
    torch.testing.assert_close(
        sorted_experts[: expected_experts.numel()], expected_experts, rtol=0, atol=0
    )
    assert torch.all(sorted_experts[expected_experts.numel() :] == -1)

    row_experts = expected_experts.repeat_interleave(block_m)
    packed = sorted_ids[:valid_rows]
    weights = sorted_weights[:valid_rows]
    routed = packed != num_tokens
    tokens = packed[routed] & 0xFFFFFF
    slots = packed[routed] >> 24
    actual_flat_slots = torch.sort(tokens * top_k + slots).values
    expected_flat_slots = (
        torch.nonzero(local_mask, as_tuple=False).flatten().to(torch.int32)
    )
    torch.testing.assert_close(actual_flat_slots, expected_flat_slots, rtol=0, atol=0)
    torch.testing.assert_close(
        topk_ids[tokens, slots] - expert_start,
        row_experts[routed],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        weights[routed], topk_weights[tokens, slots], rtol=0, atol=0
    )
    assert torch.all(packed[~routed] == num_tokens)
    assert torch.all(weights[~routed] == 0)
    assert torch.all(sorted_ids[valid_rows:] == num_tokens)
    assert torch.all(sorted_weights[valid_rows:] == 0)


@requires_cdna4
@pytest.mark.parametrize(
    "case", ["no_local_routes", "maximum_small_route", "filtered_routes"]
)
def test_small_route_boundary_alignment_contract(case: str) -> None:
    """The direct-fill bound handles empty, filtered, and 1,024-route inputs."""
    num_experts, top_k, block_m = 288, 8, 16
    if case == "no_local_routes":
        num_tokens = 1
        expert_start = 100
        topk_ids = torch.full(
            (num_tokens, top_k),
            expert_start - 1,
            dtype=torch.int32,
            device="cuda",
        )
    elif case == "filtered_routes":
        num_tokens = 64
        expert_start = 100
        route_pattern = torch.tensor(
            [99, 100, 101, 200, 387, 388, 42, 999],
            device="cuda",
            dtype=torch.int32,
        )
        topk_ids = route_pattern.repeat(num_tokens).reshape(num_tokens, top_k)
    else:
        num_tokens = 128
        expert_start = 0
        topk_ids = torch.full(
            (num_tokens, top_k),
            num_experts - 1,
            dtype=torch.int32,
            device="cuda",
        )
    topk_weights = torch.arange(
        num_tokens * top_k, dtype=torch.float32, device="cuda"
    ).reshape(num_tokens, top_k)
    if case == "filtered_routes":
        topk_weights = topk_weights + 0.25

    outputs = moe_align_block_size_device(
        topk_ids,
        topk_weights,
        num_experts,
        block_m,
        expert_start=expert_start,
    )
    torch.cuda.synchronize()
    _assert_production_alignment_contract(
        topk_ids,
        topk_weights,
        outputs,
        num_experts=num_experts,
        block_m=block_m,
        expert_start=expert_start,
    )


@requires_cdna4
def test_small_route_production_alignment_graph_replay() -> None:
    """Graph replay updates the live block prefix and padded suffix."""
    num_tokens, num_experts, top_k, block_m = 64, 288, 8, 16
    flat_slots = torch.arange(num_tokens * top_k, device="cuda", dtype=torch.int32)
    topk_ids = torch.zeros((num_tokens, top_k), device="cuda", dtype=torch.int32)
    topk_weights = (flat_slots.float() + 0.25).reshape(num_tokens, top_k)

    moe_align_block_size_device(
        topk_ids,
        topk_weights,
        num_experts,
        block_m,
        expert_start=0,
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = moe_align_block_size_device(
            topk_ids,
            topk_weights,
            num_experts,
            block_m,
            expert_start=0,
        )
    for _ in range(32):
        graph.replay()
    torch.cuda.synchronize()
    _assert_production_alignment_contract(
        topk_ids,
        topk_weights,
        outputs,
        num_experts=num_experts,
        block_m=block_m,
        expert_start=0,
    )
    first_experts = outputs[1].clone()

    topk_ids.copy_((flat_slots * 37).remainder(num_experts).reshape_as(topk_ids))
    topk_weights.copy_(torch.flip(topk_weights, dims=(0, 1)))
    for _ in range(32):
        graph.replay()
    torch.cuda.synchronize()
    _assert_production_alignment_contract(
        topk_ids,
        topk_weights,
        outputs,
        num_experts=num_experts,
        block_m=block_m,
        expert_start=0,
    )
    assert not torch.equal(outputs[1], first_experts)

    topk_ids.zero_()
    for _ in range(32):
        graph.replay()
    torch.cuda.synchronize()
    _assert_production_alignment_contract(
        topk_ids,
        topk_weights,
        outputs,
        num_experts=num_experts,
        block_m=block_m,
        expert_start=0,
    )
    torch.testing.assert_close(outputs[1], first_experts, rtol=0, atol=0)


@requires_cdna4
def test_large_route_count_reset_and_graph_replay() -> None:
    """The shared large sorter resets counts when local routes disappear."""
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16 import moe_align_device

    num_tokens, top_k, num_experts, block_m, expert_start = 130, 8, 4, 16, 12
    ids = torch.full(
        (num_tokens, top_k), expert_start, dtype=torch.int32, device="cuda"
    )
    weights = torch.ones((num_tokens, top_k), dtype=torch.float32, device="cuda")
    kernels = (
        moe_align_device._chunk_count_kernel,
        moe_align_device._chunk_scatter_kernel,
    )

    # Both views have more than 1,024 routes and share the same capacity bucket.
    moe_align_block_size_device(
        ids[:129], weights[:129], num_experts, block_m, expert_start=expert_start
    )
    with assert_no_triton_compile(*kernels):
        moe_align_block_size_device(
            ids, weights, num_experts, block_m, expert_start=expert_start
        )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = moe_align_block_size_device(
            ids, weights, num_experts, block_m, expert_start=expert_start
        )
    for route_id in (expert_start, 0, expert_start + 3):
        ids.fill_(route_id)
        graph.replay()
        torch.cuda.synchronize()
        _assert_production_alignment_contract(
            ids,
            weights,
            outputs,
            num_experts=num_experts,
            block_m=block_m,
            expert_start=expert_start,
        )


def _top8_routes(
    num_tokens: int, routing: str, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distinct top-8 routes over 288 global experts, or eight shared hot ones."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if routing == "hot":
        hot = torch.arange(TOPK, device="cuda", dtype=torch.int32) * 37
        topk_ids = hot.expand(num_tokens, TOPK).contiguous()
    else:
        logits = torch.randn(num_tokens, 288, device="cuda", generator=generator)
        topk_ids = torch.topk(logits, TOPK, dim=1).indices.to(torch.int32)
    topk_weights = torch.rand(num_tokens, TOPK, device="cuda", generator=generator)
    return topk_ids, topk_weights


@requires_cdna4
@pytest.mark.parametrize(
    ("num_tokens", "num_experts", "expert_start", "routing"),
    [
        (129, 288, 0, "random"),
        (848, 288, 0, "random"),
        (8192, 288, 0, "random"),
        (8192, 288, 0, "hot"),
        (848, 72, 216, "random"),
    ],
)
def test_large_route_alignment_contract(
    num_tokens: int, num_experts: int, expert_start: int, routing: str
) -> None:
    """Chunked counting and scatter keep every route for one or many chunks."""
    topk_ids, topk_weights = _top8_routes(num_tokens, routing, seed=num_tokens)
    outputs = moe_align_block_size_device(
        topk_ids, topk_weights, num_experts, 64, expert_start=expert_start
    )
    torch.cuda.synchronize()
    _assert_production_alignment_contract(
        topk_ids,
        topk_weights,
        outputs,
        num_experts=num_experts,
        block_m=64,
        expert_start=expert_start,
    )


@requires_cdna4
@pytest.mark.parametrize("num_tokens", [6, 24, 129, 8192])
def test_alignment_clears_output_to_zero(num_tokens: int) -> None:
    """Every alignment path clears the caller's output and nothing past it."""
    topk_ids, topk_weights = _top8_routes(num_tokens, "random", seed=7)
    storage = torch.full(
        (num_tokens + 1, 4096), -7.0, dtype=torch.bfloat16, device="cuda"
    )
    output = storage[:num_tokens]
    block_m = 16 if num_tokens <= 32 else 64
    # 6 tokens use the fused aligner, 24 the small route path, and 129 and
    # 8192 the chunked large-route path.
    align = (
        moe_align_block_size_fused if num_tokens <= 16 else moe_align_block_size_device
    )
    outputs = align(topk_ids, topk_weights, 288, block_m, output_to_zero=output)
    torch.cuda.synchronize()
    assert torch.count_nonzero(output).item() == 0
    assert torch.all(storage[num_tokens:] == -7.0)
    _assert_production_alignment_contract(
        topk_ids,
        topk_weights,
        outputs,
        num_experts=288,
        block_m=block_m,
        expert_start=0,
    )


@requires_cdna4
def test_alignment_rejects_strided_output_to_zero() -> None:
    topk_ids, topk_weights = _top8_routes(4, "random", seed=3)
    strided = torch.empty((4, 8192), dtype=torch.bfloat16, device="cuda")[:, :4096]
    with pytest.raises(ValueError, match="contiguous"):
        moe_align_block_size_device(
            topk_ids, topk_weights, 288, 16, output_to_zero=strided
        )


@requires_cdna4
def test_large_route_chunk_counts_do_not_recompile() -> None:
    """Route counts that change the chunk count and buckets reuse one compile."""
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16 import moe_align_device

    topk_ids, topk_weights = _top8_routes(8192, "random", seed=11)
    output = torch.empty((8192, 4096), dtype=torch.bfloat16, device="cuda")
    moe_align_block_size_device(
        topk_ids[:200], topk_weights[:200], 288, 64, output_to_zero=output[:200]
    )
    with assert_no_triton_compile(
        moe_align_device._chunk_count_kernel, moe_align_device._chunk_scatter_kernel
    ):
        for tokens in (300, 848, 4096, 8192):
            moe_align_block_size_device(
                topk_ids[:tokens],
                topk_weights[:tokens],
                288,
                64,
                output_to_zero=output[:tokens],
            )


@requires_cdna4
@pytest.mark.parametrize("num_tokens", [1, 8, 32, 64, 65, 256])
def test_fp16_weight_moe_matches_fp32_reference(num_tokens):
    """End-to-end output matches the fp32 oracle (auto decode/prefill path)."""
    hidden, w1, w2, topk_ids, topk_weights = _build(num_tokens)
    out = gluon_bf16_moe(hidden, w1, w2, topk_ids, topk_weights)
    torch.cuda.synchronize()
    ref = _torch_moe_ref(hidden, w1, w2, topk_ids, topk_weights)
    peak = ref.abs().max().item()
    max_abs = (out.float() - ref).abs().max().item()
    assert max_abs <= REL_TOL * peak + 1e-2, f"max_abs={max_abs}, peak={peak}"


@requires_cdna4
@pytest.mark.parametrize("num_tokens", [1, 16, 64, 256])
def test_fp16_weight_moe_64_aligned_intermediate_matches_fp32_reference(num_tokens):
    """An intermediate size that is a multiple of 64 but not 128 (a TP shard
    such as 640 / 2) runs on both the decode and prefill paths."""
    hidden, w1, w2, topk_ids, topk_weights = _build(num_tokens, i_r=320)
    out = gluon_bf16_moe(hidden, w1, w2, topk_ids, topk_weights)
    torch.cuda.synchronize()
    ref = _torch_moe_ref(hidden, w1, w2, topk_ids, topk_weights)
    peak = ref.abs().max().item()
    max_abs = (out.float() - ref).abs().max().item()
    assert max_abs <= REL_TOL * peak + 1e-2, f"max_abs={max_abs}, peak={peak}"


@requires_cdna4
@pytest.mark.parametrize("num_tokens", [1, 4, 8])
def test_fp16_weight_moe_decode_hidden_with_odd_k_tiles(num_tokens):
    """A hidden size whose 128-wide K-tile count is not a multiple of the
    default decode split (1280 -> 10 tiles) still runs the split-K decode
    path (I = 320 keeps the warp-decode shortcut off)."""
    hidden, w1, w2, topk_ids, topk_weights = _build(
        num_tokens, i_r=320, hidden_size=1280
    )
    out = gluon_bf16_moe(hidden, w1, w2, topk_ids, topk_weights)
    torch.cuda.synchronize()
    ref = _torch_moe_ref(hidden, w1, w2, topk_ids, topk_weights)
    peak = ref.abs().max().item()
    max_abs = (out.float() - ref).abs().max().item()
    assert max_abs <= REL_TOL * peak + 1e-2, f"max_abs={max_abs}, peak={peak}"


@requires_cdna4
@pytest.mark.parametrize("num_tokens", [1, 4, 16])
def test_fp16_weight_moe_splitk_consistency(num_tokens):
    """The decode split-K stage-1 path matches the single-launch path."""
    hidden, w1, w2, topk_ids, topk_weights = _build(num_tokens)
    a = gluon_bf16_moe(
        hidden, w1, w2, topk_ids, topk_weights, split_k=1, warp_decode=False
    )
    b = gluon_bf16_moe(
        hidden, w1, w2, topk_ids, topk_weights, split_k=8, warp_decode=False
    )
    torch.cuda.synchronize()
    peak = a.float().abs().max().item()
    max_abs = (a.float() - b.float()).abs().max().item()
    assert max_abs <= 1e-2 * peak + 1e-2, f"max_abs={max_abs}"


@requires_cdna4
@pytest.mark.parametrize("num_tokens", [1, 8])
def test_fp16_weight_moe_warp_decode_matches_default(num_tokens):
    """The warp-reduce GEMV decode path matches the split-K/reduce path."""
    hidden, w1, w2, topk_ids, topk_weights = _build(num_tokens)
    warp = gluon_bf16_moe(hidden, w1, w2, topk_ids, topk_weights, warp_decode=True)
    base = gluon_bf16_moe(hidden, w1, w2, topk_ids, topk_weights, warp_decode=False)
    torch.cuda.synchronize()
    peak = base.float().abs().max().item()
    max_abs = (warp.float() - base.float()).abs().max().item()
    assert max_abs <= 1e-2 * peak + 1e-2, f"max_abs={max_abs}"


@requires_cdna4
def test_fp16_weight_moe_prefill_ep_filters_nonlocal_routes() -> None:
    """The prefill aligner maps global EP routes to contiguous local weights."""
    generator = torch.Generator(device="cuda").manual_seed(29)
    num_tokens = 64
    num_local_experts = 2
    expert_start = 4
    top_k = 2
    hidden = torch.randn(
        num_tokens, D, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    w1 = (
        torch.randn(
            num_local_experts,
            2 * I_R,
            D,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.05
    )
    w2 = (
        torch.randn(
            num_local_experts,
            D,
            I_R,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.05
    )
    topk_ids = torch.randint(
        0,
        8,
        (num_tokens, top_k),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, device="cuda", generator=generator), dim=-1
    )

    actual = gluon_bf16_moe(
        hidden,
        w1,
        w2,
        topk_ids,
        topk_weights,
        decode=False,
        expert_start=expert_start,
        expert_parallel=True,
    )
    expected = torch.zeros_like(hidden, dtype=torch.float32)
    for local_expert in range(num_local_experts):
        token_ids, slots = torch.where(topk_ids == expert_start + local_expert)
        projected = hidden[token_ids].float() @ w1[local_expert].float().T
        intermediate = (
            torch.nn.functional.silu(projected[:, :I_R]) * projected[:, I_R:]
        ).to(torch.bfloat16)
        expert_output = (intermediate.float() @ w2[local_expert].float().T).to(
            torch.bfloat16
        )
        expected.index_add_(
            0,
            token_ids,
            expert_output * topk_weights[token_ids, slots, None],
        )

    peak = expected.abs().max().item()
    max_abs = (actual.float() - expected).abs().max().item()
    assert max_abs <= REL_TOL * peak + 1e-2, f"max_abs={max_abs}, peak={peak}"


@requires_cdna4
def test_decode_align_batch_size_does_not_recompile() -> None:
    """Token counts inside one power-of-two route bucket share a compile."""
    topk_ids = torch.randint(0, E, (32, TOPK), dtype=torch.int32, device="cuda")
    topk_weights = torch.rand(32, TOPK, device="cuda")
    # 9..15 tokens route 72..120 slots at top-8, all in the 128-slot bucket.
    # Warm the route count with and without the divisible-by-16
    # specialization.
    for tokens in (9, 10):
        moe_align_block_size_fused(topk_ids[:tokens], topk_weights[:tokens], E, 32)
    with assert_no_triton_compile(_fused_align_kernel):
        for tokens in (11, 12, 13, 15):
            moe_align_block_size_fused(topk_ids[:tokens], topk_weights[:tokens], E, 32)
