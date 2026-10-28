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

"""Native MXFP8 Marlin MoE arithmetic, routing, and graph replay on SM90+."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from tokenspeed_kernel.ops.moe._marlin.mxfp8 import (
    marlin_mxfp8_moe_weights,
    marlin_mxfp8_precomputed_moe_apply,
)
from tokenspeed_kernel.platform import ArchVersion, current_platform
from tokenspeed_kernel.thirdparty.cuda.marlin_moe import moe_align_block_size

BLOCK = 32
PLAN = {
    "activation": "swiglu",
    "combine_order": "rank",
    "deterministic_finalize": True,
}


def _requires_sm90() -> None:
    if not torch.cuda.is_available() or not current_platform().is_nvidia:
        pytest.skip("NVIDIA CUDA required")
    if current_platform().arch_version < ArchVersion(9, 0):
        pytest.skip("SM90+ required")


@dataclass(frozen=True)
class _SwigluConfig:
    """The kernel's activation configuration protocol, without runtime imports."""

    alpha: float | None
    limit: float | None


class _Experts(torch.nn.Module):
    def __init__(self, raw, ep_size: int, ep_rank: int, limit: float | None):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(raw["w13_weight"], requires_grad=False)
        self.w13_weight_scale_inv = torch.nn.Parameter(
            raw["w13_weight_scale_inv"], requires_grad=False
        )
        self.w2_weight = torch.nn.Parameter(raw["w2_weight"], requires_grad=False)
        self.w2_weight_scale_inv = torch.nn.Parameter(
            raw["w2_weight_scale_inv"], requires_grad=False
        )
        self.num_local_experts = raw["w13_weight"].shape[0]
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.tp_size = 1
        self.tp_rank = 0
        self.activation = "swiglu"
        self.swiglu_arg = _SwigluConfig(alpha=1.0, limit=limit)
        self.swiglu_beta = 0.0
        self._moe_backend_state = None


def _make_weights(experts, hidden, intermediate, generator, scale_range=(118, 124)):
    device = generator.device

    def weight(*shape):
        return (
            (torch.randn(*shape, generator=generator, device=device) * 2)
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
        )

    def scale(*shape):
        low, high = scale_range
        return torch.randint(
            low, high + 1, shape, generator=generator, device=device, dtype=torch.uint8
        )

    return {
        "w13_weight": weight(experts, 2 * intermediate, hidden),
        "w13_weight_scale_inv": scale(experts, 2 * intermediate, hidden // BLOCK),
        "w2_weight": weight(experts, hidden, intermediate),
        "w2_weight_scale_inv": scale(experts, hidden, intermediate // BLOCK),
    }


def _dequant(weight, scale):
    scales = torch.exp2(scale.double() - 127).repeat_interleave(BLOCK, dim=-1)
    return (weight.double() * scales).to(torch.bfloat16)


def _reference(x, raw, ids, weights, limit, *, expert_start=0):
    """FP64 products with the model's BF16 stage boundaries and slot-order sum."""
    tokens, top_k = ids.shape
    hidden = x.shape[-1]
    routes = torch.zeros((tokens, top_k, hidden), dtype=torch.bfloat16, device=x.device)
    for local_expert in range(raw["w13_weight"].shape[0]):
        token_ids, slots = (ids == expert_start + local_expert).nonzero(as_tuple=True)
        if token_ids.numel() == 0:
            continue
        w13 = _dequant(
            raw["w13_weight"][local_expert], raw["w13_weight_scale_inv"][local_expert]
        )
        w2 = _dequant(
            raw["w2_weight"][local_expert], raw["w2_weight_scale_inv"][local_expert]
        )
        gate_up = (x[token_ids].double() @ w13.double().T).bfloat16()
        gate, up = gate_up.float().chunk(2, dim=-1)
        if limit is not None:
            gate = gate.clamp(max=limit)
            up = up.clamp(-limit, limit)
        intermediate = (torch.nn.functional.silu(gate) * up).bfloat16()
        down = (intermediate.double() @ w2.double().T).bfloat16()
        route_weights = weights[token_ids, slots].bfloat16().float()
        routes[token_ids, slots] = (down.float() * route_weights[:, None]).bfloat16()
    total = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(top_k):
        total += routes[:, slot].float()
    return total.bfloat16()


def _routing(tokens, experts, top_k, generator):
    scores = torch.rand(tokens, experts, generator=generator, device=generator.device)
    weights, ids = scores.topk(top_k, dim=-1)
    return ids.int(), (weights / weights.sum(dim=-1, keepdim=True)).float()


def _apply(w, x, ids, weights):
    return marlin_mxfp8_precomputed_moe_apply(
        PLAN, x, w, None, topk_weights=weights, topk_ids=ids
    )


def _assert_close(actual, expected):
    actual, expected = actual.float(), expected.float()
    scale = expected.abs().amax().item()
    torch.testing.assert_close(actual, expected, atol=scale / 64, rtol=1 / 64)
    relative_l2 = (actual - expected).norm() / expected.norm().clamp_min(1e-30)
    assert relative_l2.item() < 0.006, f"relative L2 error {relative_l2.item()}"


@pytest.mark.parametrize("tokens", [1, 7, 64, 300])
@pytest.mark.parametrize("ep_size", [1, 2])
def test_native_marlin_matches_bf16_stage_reference(tokens, ep_size):
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(7)
    experts, hidden, intermediate, top_k = 8, 512, 256, 3
    raw = _make_weights(experts, hidden, intermediate, generator)
    x = torch.randn(tokens, hidden, generator=generator, device="cuda").bfloat16()
    ids, weights = _routing(tokens, experts, top_k, generator)
    num_local = experts // ep_size
    for rank in range(ep_size):
        start = rank * num_local
        shard = {
            name: value[start : start + num_local].clone()
            for name, value in raw.items()
        }
        expected = _reference(x, shard, ids, weights, None, expert_start=start)
        w = _Experts(shard, ep_size, rank, None)
        marlin_mxfp8_moe_weights(PLAN, w)
        _assert_close(_apply(w, x, ids, weights), expected)


@pytest.mark.parametrize("tokens", [4, 32])
def test_native_marlin_applies_routed_swiglu_clamp(tokens):
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(13)
    raw = _make_weights(8, 256, 128, generator, scale_range=(124, 126))
    x = torch.randn(tokens, 256, generator=generator, device="cuda").bfloat16()
    ids, weights = _routing(tokens, 8, 2, generator)
    expected = _reference(x, raw, ids, weights, 3.0)
    unclamped = _reference(x, raw, ids, weights, None)
    assert (expected.float() - unclamped.float()).norm() > expected.float().norm() * 0.1
    w = _Experts(raw, 1, 0, 3.0)
    marlin_mxfp8_moe_weights(PLAN, w)
    _assert_close(_apply(w, x, ids, weights), expected)


@pytest.mark.parametrize("tokens", [2, 300])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_native_marlin_ignores_invalid_routes_and_preserves_repeated_slots(
    tokens, ids_dtype
):
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(37)
    raw = _make_weights(4, 256, 128, generator)
    x = torch.randn(tokens, 256, generator=generator, device="cuda").bfloat16()
    ids, weights = _routing(tokens, 8, 4, generator)
    ids = ids.to(ids_dtype)
    ids[:, 0] = 4
    ids[:, 1] = 4
    ids[::2, 2] = torch.iinfo(ids_dtype).min
    ids[1] = torch.tensor(
        [-1, 0, 8, torch.iinfo(ids_dtype).max], device="cuda", dtype=ids_dtype
    )
    weights[1].fill_(float("nan"))
    expected = _reference(x, raw, ids, weights, None, expert_start=4)
    w = _Experts(raw, 2, 1, None)
    marlin_mxfp8_moe_weights(PLAN, w)
    actual = _apply(w, x, ids, weights)
    _assert_close(actual, expected)
    assert torch.count_nonzero(actual[1]).item() == 0


@pytest.mark.parametrize("local_experts", [8, 300, 1100])
@pytest.mark.parametrize("tokens", [0, 1, 100, 129, 300, 4096])
def test_stable_alignment_matches_exact_route_order(local_experts, tokens):
    _requires_sm90()
    generator = torch.Generator().manual_seed(5)
    start, top_k, block = 11, 4, 16
    ids = torch.randint(
        start,
        start + local_experts,
        (tokens, top_k),
        generator=generator,
        dtype=torch.int32,
    )
    if tokens:
        ids[:, 1] = ids[:, 0]
        ids[0] = torch.tensor(
            [torch.iinfo(torch.int32).min, -1, start - 1, start + local_experts],
            dtype=torch.int32,
        )
    flat = ids.flatten().tolist()
    lanes = [[] for _ in range(local_experts + 1)]
    for route, expert in enumerate(flat):
        lane = expert - start + 1 if start <= expert < start + local_experts else 0
        lanes[lane].append(route)
    expected_routes, expected_experts = [], []
    for lane, routes in enumerate(lanes):
        padded = ((len(routes) + block - 1) // block) * block
        expected_routes.extend(routes + [len(flat)] * (padded - len(routes)))
        expected_experts.extend([lane - 1] * (padded // block))
    device_ids = ids.cuda()
    for _ in range(2):
        aligned, expert_ids, count = moe_align_block_size(
            device_ids, block, local_experts, expert_start=start, deterministic=True
        )
        total = count.item()
        assert total == len(expected_routes)
        assert aligned[:total].cpu().tolist() == expected_routes
        assert expert_ids[: total // block].cpu().tolist() == expected_experts


@pytest.mark.parametrize("tokens", [5, 256, 1024])
def test_native_marlin_repeated_apply_is_bitwise_stable(tokens):
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(17)
    raw = _make_weights(8, 512, 256, generator)
    w = _Experts(raw, 1, 0, 7.0)
    marlin_mxfp8_moe_weights(PLAN, w)
    x = torch.randn(tokens, 512, generator=generator, device="cuda").bfloat16()
    ids, weights = _routing(tokens, 8, 4, generator)
    first = _apply(w, x, ids, weights)
    for _ in range(3):
        torch.testing.assert_close(_apply(w, x, ids, weights), first, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [4, 256])
def test_native_marlin_hy4_expert_geometry(tokens):
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(19)
    global_experts, ep_size, ep_rank = 256, 32, 5
    local_experts, top_k, hidden, intermediate = 8, 8, 6144, 2048
    raw = _make_weights(local_experts, hidden, intermediate, generator)
    x = torch.randn(tokens, hidden, generator=generator, device="cuda").bfloat16()
    ids, weights = _routing(tokens, global_experts, top_k, generator)
    ids[0, 0] = ep_rank * local_experts
    expected = _reference(
        x, raw, ids, weights, 7.0, expert_start=ep_rank * local_experts
    )
    w = _Experts(raw, ep_size, ep_rank, 7.0)
    marlin_mxfp8_moe_weights(PLAN, w)
    _assert_close(_apply(w, x, ids, weights), expected)


@pytest.mark.parametrize("tokens", [0, 2, 32, 1024])
def test_native_marlin_handles_empty_and_strided_inputs(tokens):
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(47)
    raw = _make_weights(4, 256, 128, generator)
    w = _Experts(raw, 2, 1, None)
    marlin_mxfp8_moe_weights(PLAN, w)
    x = torch.randn(tokens, 512, generator=generator, device="cuda").bfloat16()[:, ::2]
    ids, weights = _routing(tokens, 8, 2, generator)
    weight_storage = torch.full((tokens, 4), float("nan"), device="cuda")
    ids_storage = torch.full((tokens, 4), -1, device="cuda", dtype=torch.int32)
    weight_storage[:, ::2] = weights
    ids_storage[:, ::2] = ids
    expected = _apply(w, x.contiguous(), ids, weights)
    actual = _apply(w, x, ids_storage[:, ::2], weight_storage[:, ::2])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (tokens, 256)


@pytest.mark.parametrize("code", [0, 255])
@pytest.mark.parametrize("name", ["w13_weight_scale_inv", "w2_weight_scale_inv"])
def test_invalid_scale_does_not_partially_prepare_parameters(code, name):
    generator = torch.Generator().manual_seed(53)
    raw = _make_weights(2, 256, 128, generator)
    raw[name].view(-1)[0] = code
    w = _Experts(raw, 1, 0, None)
    original = dict(w.named_parameters())
    original_bytes = {
        key: value.detach().view(torch.uint8).clone() for key, value in original.items()
    }
    with pytest.raises(ValueError, match="unsupported E8M0"):
        marlin_mxfp8_moe_weights(PLAN, w)
    assert w._moe_backend_state is None
    for key, parameter in w.named_parameters():
        assert parameter is original[key]
        torch.testing.assert_close(
            parameter.detach().view(torch.uint8), original_bytes[key], rtol=0, atol=0
        )


def test_malformed_scale_shape_is_rejected_before_preparation():
    generator = torch.Generator().manual_seed(3)
    raw = _make_weights(2, 256, 128, generator)
    raw["w2_weight_scale_inv"] = raw["w2_weight_scale_inv"][..., :-1].contiguous()
    w = _Experts(raw, 1, 0, None)
    with pytest.raises(ValueError, match="1x32 blocks"):
        marlin_mxfp8_moe_weights(PLAN, w)
    assert w._moe_backend_state is None
    assert w.w13_weight.dtype == torch.float8_e4m3fn
    assert w.w2_weight.dtype == torch.float8_e4m3fn


def test_native_marlin_cuda_graph_reads_updated_inputs_and_routing():
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(61)
    raw = _make_weights(4, 256, 128, generator)
    w = _Experts(raw, 1, 0, 3.0)
    marlin_mxfp8_moe_weights(PLAN, w)
    x = torch.randn(7, 256, generator=generator, device="cuda").bfloat16()
    ids, weights = _routing(7, 4, 3, generator)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _apply(w, x, ids, weights)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = _apply(w, x, ids, weights)
    graph.replay()
    torch.testing.assert_close(result, _apply(w, x, ids, weights), rtol=0, atol=0)
    first = result.clone()
    x.neg_()
    weights.copy_(weights.roll(1, dims=1))
    ids.copy_((ids + 1) % 4)
    graph.replay()
    torch.testing.assert_close(result, _apply(w, x, ids, weights), rtol=0, atol=0)
    assert not torch.equal(result, first)
