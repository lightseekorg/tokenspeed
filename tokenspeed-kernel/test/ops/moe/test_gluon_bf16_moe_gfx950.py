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
from utils import is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for BF16-weight Gluon MoE tests",
        allow_module_level=True,
    )

from tokenspeed_kernel_amd.ops.gfx950.moe.fp16 import gluon_bf16_moe  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.moe_align_device import (  # noqa: E402
    moe_align_block_size_device,
)

# DeepSeek-V3 TP=8 MoE reference shape.
E = 256
D = 7168
I_R = 256
TOPK = 8
REL_TOL = 2e-2


def _routing_softmax_topk(logits: torch.Tensor, topk: int):
    """Reference router: softmax over experts then renormalised top-k."""
    probs = torch.softmax(logits.float(), dim=-1)
    weights, ids = torch.topk(probs, topk, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return ids.to(torch.int32), weights.to(torch.float32)


def _torch_moe_ref(hidden, w1, w2, topk_ids, topk_weights) -> torch.Tensor:
    """Golden fp32 MoE FFN: y[t] = sum_s w * (silu(h@wg) * (h@wu)) @ wd."""
    hf, w1f, w2f = hidden.float(), w1.float(), w2.float()
    ids, wts = topk_ids.cpu(), topk_weights.float().cpu()
    out = torch.zeros(hidden.shape[0], D, dtype=torch.float32, device=hidden.device)
    for t in range(hidden.shape[0]):
        for s in range(TOPK):
            e = int(ids[t, s])
            g = hf[t] @ w1f[e].T
            inter = torch.nn.functional.silu(g[:I_R]) * g[I_R:]
            out[t] += float(wts[t, s]) * (inter @ w2f[e].T)
    return out


def _build(num_tokens: int, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn(
        num_tokens, D, dtype=torch.bfloat16, device="cuda", generator=g
    )
    # Scale weights down so the fp32 reference stays in a comparable range.
    w1 = (
        torch.randn(E, 2 * I_R, D, dtype=torch.bfloat16, device="cuda", generator=g)
        * 0.05
    )
    w2 = torch.randn(E, D, I_R, dtype=torch.bfloat16, device="cuda", generator=g) * 0.05
    logits = torch.randn(num_tokens, E, dtype=torch.float32, device="cuda", generator=g)
    topk_ids, topk_weights = _routing_softmax_topk(logits, TOPK)
    return hidden, w1, w2, topk_ids, topk_weights


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

    sorted_ids, sorted_experts, sorted_weights, num_valid = moe_align_block_size_device(
        topk_ids, topk_weights, num_experts, block_m
    )
    valid_rows = int(num_valid.item())
    assert valid_rows == num_tokens * top_k
    assert sorted_experts[: valid_rows // block_m].tolist() == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]

    packed = sorted_ids[:valid_rows]
    tokens = packed & 0xFFFFFF
    slots = packed >> 24
    experts = sorted_experts[: valid_rows // block_m].repeat_interleave(block_m)
    assert torch.equal(topk_ids[tokens, slots], experts)
    assert torch.equal(topk_weights[tokens, slots], sorted_weights[:valid_rows])
    assert torch.equal(
        torch.sort(tokens * top_k + slots).values,
        torch.arange(num_tokens * top_k, device="cuda"),
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


@pytest.mark.parametrize("routing", ["concentrated", "distributed", "filtered"])
def test_small_route_production_alignment_contract(routing: str) -> None:
    """Production-sized concentrated, distributed, and EP routes stay exact."""
    num_tokens, num_experts, top_k, block_m = 64, 288, 8, 16
    flat_slots = torch.arange(num_tokens * top_k, device="cuda", dtype=torch.int32)
    if routing == "concentrated":
        expert_start = 0
        topk_ids = torch.zeros_like(flat_slots)
    elif routing == "distributed":
        expert_start = 0
        topk_ids = (flat_slots * 37).remainder(num_experts)
    else:
        expert_start = 100
        route_pattern = torch.tensor(
            [99, 100, 101, 200, 387, 388, 42, 999],
            device="cuda",
            dtype=torch.int32,
        )
        topk_ids = route_pattern.repeat(num_tokens)
    topk_ids = topk_ids.reshape(num_tokens, top_k)
    topk_weights = (flat_slots.float() + 0.25).reshape(num_tokens, top_k)
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


@pytest.mark.parametrize("case", ["no_local_routes", "maximum_small_route"])
def test_small_route_boundary_alignment_contract(case: str) -> None:
    """The direct-fill bound handles empty routing and its 1,024-route edge."""
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


def test_small_route_production_alignment_graph_replay() -> None:
    """Graph replay preserves cross-wave block-table initialization ordering."""
    num_tokens, num_experts, top_k, block_m = 64, 288, 8, 16
    flat_slots = torch.arange(num_tokens * top_k, device="cuda", dtype=torch.int32)
    topk_ids = ((flat_slots.remainder(3) + 1) * 64).reshape(num_tokens, top_k)
    topk_weights = (flat_slots.float() + 0.25).reshape(num_tokens, top_k)

    moe_align_block_size_device(topk_ids, topk_weights, num_experts, block_m)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = moe_align_block_size_device(
            topk_ids, topk_weights, num_experts, block_m
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

    topk_ids.copy_((((flat_slots + 1).remainder(3) + 1) * 64 + 1).reshape_as(topk_ids))
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


@pytest.mark.parametrize("num_tokens", [1, 8, 32, 64, 256])
def test_fp16_weight_moe_matches_fp32_reference(num_tokens):
    """End-to-end output matches the fp32 oracle (auto decode/prefill path)."""
    hidden, w1, w2, topk_ids, topk_weights = _build(num_tokens)
    out = gluon_bf16_moe(hidden, w1, w2, topk_ids, topk_weights)
    torch.cuda.synchronize()
    ref = _torch_moe_ref(hidden, w1, w2, topk_ids, topk_weights)
    peak = ref.abs().max().item()
    max_abs = (out.float() - ref).abs().max().item()
    assert max_abs <= REL_TOL * peak + 1e-2, f"max_abs={max_abs}, peak={peak}"


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
