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

import math
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.ops.moe import moe_apply, moe_plan, moe_process_weights, moe_topk
from tokenspeed_kernel.platform import pdl_enabled
from tokenspeed_kernel.selection import NoKernelFoundError

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="requires SM90",
)


class _Experts(torch.nn.Module):
    def __init__(
        self, hidden: int, intermediate: int, experts: int, ep_size: int, ep_rank: int
    ):
        super().__init__()
        self.activation = "silu"
        self.swiglu_arg = None
        self.swiglu_beta = None
        self.w13_input_layout = "concatenated"
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.num_local_experts = experts
        self.num_experts = experts * ep_size
        self.w13_weight = torch.nn.Parameter(
            torch.randn(experts, 2 * intermediate, hidden, device="cuda").to(
                torch.float8_e4m3fn
            ),
            requires_grad=False,
        )
        self.w2_weight = torch.nn.Parameter(
            torch.randn(experts, hidden, intermediate, device="cuda").to(
                torch.float8_e4m3fn
            ),
            requires_grad=False,
        )
        # Non-power-of-two scales vary on every expert/N/K block, including
        # across the gate/up boundary. This catches transposed or ignored scales.
        self.w13_weight_scale_inv = torch.nn.Parameter(
            (
                torch.rand(
                    experts, 2 * intermediate // 128, hidden // 128, device="cuda"
                )
                + 0.5
            )
            / math.sqrt(hidden),
            requires_grad=False,
        )
        self.w2_weight_scale_inv = torch.nn.Parameter(
            (
                torch.rand(experts, hidden // 128, intermediate // 128, device="cuda")
                + 0.5
            )
            / math.sqrt(intermediate),
            requires_grad=False,
        )


def _plan(w: _Experts, **overrides) -> dict:
    args = dict(
        weight_dtype="fp8",
        input_dtype=torch.bfloat16,
        activation=w.activation,
        requires_deferred_finalize=False,
        routing_mode="precomputed_topk",
        a2a_backend="none",
        ep_size=w.ep_size,
        ispp=w.w2_weight.shape[2],
        hidden=w.w2_weight.shape[1],
        swiglu_form="standard",
        activation_clamped=False,
        expert_id_repeats=True,
        fp8_scale_block_shape=(128, 128),
        internal_activation_dtype="input",
        with_bias=False,
        fast_math=False,
        combine_order="rank",
        solution="triton",
    )
    args.update(overrides)
    return moe_plan(**args)


def _routes(tokens: int, experts: int) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.randn(tokens, experts, device="cuda")
    bias = torch.randn(experts, device="cuda") * 0.05
    weights, ids = moe_topk(
        logits,
        8,
        score_function="sigmoid",
        selection_method="topk",
        renormalize=True,
        routed_scaling_factor=2.5,
        correction_bias=bias,
        topk_weights_dtype=torch.float32,
        solution="triton",
    )
    scores = logits.sigmoid()
    expected_ids = (scores + bias).topk(8, dim=-1).indices
    assert torch.equal(ids.sort(dim=-1).values.long(), expected_ids.sort(dim=-1).values)
    expected_weights = scores.gather(1, ids.long())
    expected_weights = (
        expected_weights / expected_weights.sum(dim=-1, keepdim=True) * 2.5
    )
    torch.testing.assert_close(weights, expected_weights, atol=1e-6, rtol=1e-6)
    return weights, ids


def _reference(
    x: torch.Tensor, w: _Experts, weights: torch.Tensor, ids: torch.Tensor
) -> torch.Tensor:
    def dequant(weight, scale):
        return weight.float() * scale.repeat_interleave(128, 0).repeat_interleave(
            128, 1
        )

    route_output = torch.zeros(
        (*ids.shape, x.shape[1]), device=x.device, dtype=torch.float32
    )
    local_ids = ids - w.ep_rank * w.num_local_experts
    # Only the test oracle materializes dequantized weights, one expert at a time.
    for expert in local_ids.unique().tolist():
        if not 0 <= expert < w.num_local_experts:
            continue
        rows, slots = torch.where(local_ids == expert)
        gate_up = (
            x[rows].float()
            @ dequant(w.w13_weight[expert], w.w13_weight_scale_inv[expert]).T
        )
        gate, up = gate_up.chunk(2, dim=-1)
        activated = (torch.nn.functional.silu(gate) * up).bfloat16()
        route_output[rows, slots] = (
            activated.float()
            @ dequant(w.w2_weight[expert], w.w2_weight_scale_inv[expert]).T
        )
    return (route_output * weights.float().unsqueeze(-1)).sum(dim=1).bfloat16()


def _check(actual, expected):
    assert actual.dtype == torch.bfloat16
    # Different FP32 accumulation orders can straddle a BF16 rounding boundary.
    # Allow one output ULP, not an FP8-sized tolerance.
    torch.testing.assert_close(
        actual, expected, atol=1e-3, rtol=torch.finfo(torch.bfloat16).eps
    )
    if expected.numel() and expected.float().square().mean() > 0:
        relative_rms = (
            (actual.float() - expected.float()).square().mean()
            / expected.float().square().mean()
        ).sqrt()
        assert relative_rms < 1e-3, f"relative RMS error: {relative_rms.item()}"


@pytest.fixture(autouse=True)
def _precise_reference():
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old


@pytest.fixture(scope="module", params=[(1536, 32, 8, 3), (384, 256, 1, 0)])
def full_experts(request):
    torch.manual_seed(42)
    intermediate, experts, ep_size, ep_rank = request.param
    return _Experts(5120, intermediate, experts, ep_size, ep_rank)


@pytest.mark.parametrize("tokens", [1, 17, 65, 257])
def test_public_api_full_shapes(full_experts, tokens):
    w = full_experts
    plan = _plan(w)
    assert plan["apply_kernel_name"] == "triton_fp8_precomputed_moe_apply"
    assert plan["supports_precomputed_topk"] and not plan["support_routing"]
    pointers = [p.data_ptr() for p in w.parameters()]
    versions = [p._version for p in w.parameters()]
    moe_process_weights(plan, w)
    x = torch.randn(tokens, 5120, device="cuda", dtype=torch.bfloat16)
    weights, ids = _routes(tokens, 256)
    # Guarantee both endpoints of this rank's range are exercised even at M=1.
    ids[0, 0] = w.ep_rank * w.num_local_experts
    ids[0, 1] = (w.ep_rank + 1) * w.num_local_experts - 1
    actual = moe_apply(plan, x, w, None, topk_weights=weights, topk_ids=ids)
    _check(actual, _reference(x, w, weights, ids))
    assert [p.data_ptr() for p in w.parameters()] == pointers
    assert [p._version for p in w.parameters()] == versions
    assert w.w13_weight.dtype == w.w2_weight.dtype == torch.float8_e4m3fn


@pytest.fixture(params=[False, True])
def pdl_mode(request):
    old = pdl_enabled()
    pdl_enabled(request.param)
    yield request.param
    pdl_enabled(old)


@pytest.mark.parametrize("ep_rank", [0, 7])
def test_graph_replay_padding_and_local_experts(ep_rank, pdl_mode):
    torch.manual_seed(13)
    w = _Experts(256, 384, 32, 8, ep_rank)
    plan = _plan(w)
    moe_process_weights(plan, w)
    x = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
    weights, ids = _routes(17, 256)
    ids = ids.long()
    start = ep_rank * 32
    ids[0] = torch.tensor(
        [
            start,
            start + 31,
            start,
            -1,
            2**32 + start,
            -(2**32) + start,
            start + 32,
            start + 1,
        ],
        device="cuda",
    )
    moe_apply(plan, x, w, None, topk_weights=weights, topk_ids=ids)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = moe_apply(plan, x, w, None, topk_weights=weights, topk_ids=ids)
    pointer = actual.data_ptr()
    graph.replay()
    _check(actual, _reference(x, w, weights, ids))
    for live in (17, 3, 0, 17):
        x.normal_()
        weights.uniform_()
        ids.random_(start, start + 32)
        ids[0, :4] = torch.tensor([start, start + 31, start, -1], device="cuda")
        ids[live:] = -1
        weights[live:] = 0
        graph.replay()
        _check(actual, _reference(x, w, weights, ids))
        assert actual.data_ptr() == pointer
        assert torch.count_nonzero(actual[live:]) == 0
    # A rank with no local assignments must not reuse stale route outputs.
    ids.fill_((start + 32) % 256)
    graph.replay()
    assert torch.count_nonzero(actual) == 0


def test_full_shape_graph(full_experts):
    w = full_experts
    plan = _plan(w)
    moe_process_weights(plan, w)
    x = torch.randn(17, 5120, device="cuda", dtype=torch.bfloat16)
    logits = torch.randn(17, 256, device="cuda")
    bias = torch.randn(256, device="cuda") * 0.05

    def run():
        weights, ids = moe_topk(
            logits,
            8,
            score_function="sigmoid",
            selection_method="topk",
            renormalize=True,
            routed_scaling_factor=2.5,
            correction_bias=bias,
            topk_weights_dtype=torch.float32,
            solution="triton",
        )
        output = moe_apply(plan, x, w, None, topk_weights=weights, topk_ids=ids)
        return output, weights, ids

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, weights, ids = run()
    for _ in range(3):
        x.normal_()
        logits.normal_()
        bias.normal_(std=0.05)
        graph.replay()
        _check(actual, _reference(x, w, weights, ids))
        scores = logits.sigmoid()
        expected_ids = (scores + bias).topk(8, dim=-1).indices
        assert torch.equal(
            ids.sort(dim=-1).values.long(), expected_ids.sort(dim=-1).values
        )
    x.zero_()
    graph.replay()
    assert torch.count_nonzero(actual) == 0


def test_concentrated_prefill_routes():
    torch.manual_seed(7)
    w = _Experts(256, 384, 32, 8, 7)
    plan = _plan(w)
    moe_process_weights(plan, w)
    x = torch.randn(257, 256, device="cuda", dtype=torch.bfloat16)
    # More than two routing blocks and many M tiles for one local expert.
    ids = torch.full((257, 8), 255, device="cuda", dtype=torch.int32)
    weights = torch.rand(257, 8, device="cuda")
    weights /= weights.sum(dim=-1, keepdim=True)
    ids[0, 0] = 223
    ids[-1] = -1
    weights[-2] = 0
    actual = moe_apply(plan, x, w, None, topk_weights=weights, topk_ids=ids)
    _check(actual, _reference(x, w, weights, ids))
    assert torch.count_nonzero(actual[-2:]) == 0


def test_empty_tokens_and_noncontiguous_routing():
    w = _Experts(128, 128, 8, 8, 0)
    plan = _plan(w)
    moe_process_weights(plan, w)
    x = torch.empty(0, 128, device="cuda", dtype=torch.bfloat16)
    ids = torch.empty(0, 8, device="cuda", dtype=torch.int32)
    weights = torch.empty(0, 8, device="cuda")
    assert (
        moe_apply(plan, x, w, None, topk_weights=weights, topk_ids=ids).shape == x.shape
    )
    x = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(0, 8, (3, 16), device="cuda")[:, ::2]
    weights = torch.rand(3, 16, device="cuda", dtype=torch.bfloat16)[:, ::2]
    actual = moe_apply(plan, x, w, None, topk_weights=weights, topk_ids=ids)
    _check(actual, _reference(x, w, weights, ids))


@pytest.mark.parametrize(
    "unsupported",
    [
        {"routing_mode": "kernel_routing"},
        {"a2a_backend": "deepep"},
        {"requires_deferred_finalize": True},
        {"with_bias": True},
        {"activation": "situ"},
        {"activation_clamped": True},
        {"swiglu_form": "generalized"},
        {"fp8_scale_block_shape": (64, 128)},
        {"internal_activation_dtype": "fp8"},
        {"input_dtype": torch.float16},
        {"ispp": 192},
        {"hidden": 192},
        {"persistent_max_num_tokens_per_gpu": 32},
    ],
)
def test_plan_rejects_unsupported_modes(unsupported):
    # The mainline non-EP implementation also supports generalized SwiGLU.
    # EP requires this commit's standard-SwiGLU implementation.
    w = _Experts(128, 128, 8, 8, 0)
    with pytest.raises(NoKernelFoundError):
        _plan(w, **unsupported)


def test_preprocessor_and_apply_reject_invalid_contracts():
    w = _Experts(128, 128, 8, 8, 0)
    plan = _plan(w)
    w.swiglu_arg = SimpleNamespace(alpha=1.0, limit=7.0)
    with pytest.raises(ValueError, match="clamped"):
        moe_process_weights(plan, w)
    w.swiglu_arg = None
    x = torch.zeros(1, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="precomputed"):
        moe_apply(plan, x, w, None)
    with pytest.raises(ValueError, match="finalization"):
        moe_apply(plan, x, w, None, do_finalize=False)
    w.w2_weight_scale_inv = torch.nn.Parameter(
        torch.ones(8, 1, 2, device="cuda"), requires_grad=False
    )
    with pytest.raises(ValueError, match="block shape"):
        moe_process_weights(plan, w)
