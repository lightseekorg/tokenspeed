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

"""gfx950 SiLU experts with MXFP8 activations and MXFP4 weights."""

from __future__ import annotations

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for the Gluon MXFP8 SiLU MoE tests",
        allow_module_level=True,
    )

from tokenspeed_kernel.ops.moe import moe_process_weights  # noqa: E402
from tokenspeed_kernel.ops.moe import moe_apply, moe_plan  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4 import mxfp8_gemm  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.n16_weights import (  # noqa: E402
    n16_mxfp4_shape,
)

# Qwen3.8-Flash-Next TP1 geometry with fewer experts: a 640-wide expert
# intermediate the weight preprocessor pads to the pipeline's 768.
EXPERTS = 32
HIDDEN = 2560
INTER = 640
PADDED_INTER = 768
TOP_K = 10
_KERNEL = "gluon_mxfp4_a8w4_silu_precomputed_moe_apply"
_E2M1 = (0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6)


def _weights(seed: int = 0) -> torch.nn.Module:
    g = torch.Generator(device="cuda").manual_seed(seed)

    def codes(*shape):
        return torch.randint(
            0, 256, shape, device="cuda", dtype=torch.uint8, generator=g
        )

    def scales(*shape):
        return torch.randint(
            118, 124, shape, device="cuda", dtype=torch.uint8, generator=g
        )

    w = torch.nn.Module()
    w.w13_weight = torch.nn.Parameter(
        codes(EXPERTS, 2 * INTER, HIDDEN // 2), requires_grad=False
    )
    w.w13_weight_scale = torch.nn.Parameter(
        scales(EXPERTS, 2 * INTER, HIDDEN // 32), requires_grad=False
    )
    w.w2_weight = torch.nn.Parameter(
        codes(EXPERTS, HIDDEN, INTER // 2), requires_grad=False
    )
    w.w2_weight_scale = torch.nn.Parameter(
        scales(EXPERTS, HIDDEN, INTER // 32), requires_grad=False
    )
    w.top_k = TOP_K
    w.w13_input_layout = "concatenated"
    return w


def _plan() -> dict:
    return moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="silu",
        routing_mode="precomputed_topk",
        ispp=INTER,
        internal_activation_dtype="mxfp8",
        solution=None,
        hidden=HIDDEN,
        swiglu_form=None,
        activation_clamped=False,
        expert_id_repeats=False,
        fast_math=True,
        combine_order="rank",
        features=frozenset(),
    )


def _routes(m: int, top_k: int, seed: int = 1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=g)
    ids = torch.rand(m, EXPERTS, device="cuda", generator=g).topk(top_k, dim=-1)
    weights = torch.softmax(torch.randn(m, top_k, device="cuda", generator=g), dim=-1)
    return x, weights, ids.indices.to(torch.int32)


def _dequant(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(_E2M1, device=packed.device, dtype=torch.float32)
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2)
    return table[codes.long()] * torch.exp2(scale.float() - 127).repeat_interleave(
        32, dim=-1
    )


def _mxfp8_round_trip(x: torch.Tensor) -> torch.Tensor:
    """E4M3 RNE values under the per-32 ``amax / 448`` (rounded up) scale."""
    groups = x.float().reshape(x.shape[0], -1, 32)
    ratio = groups.abs().amax(-1).clamp_min(1e-10) * torch.tensor(
        1 / 448, dtype=torch.float32
    )
    bits = ratio.view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    scale = (exponent.clamp(1, 254) << 23).view(torch.float32).unsqueeze(-1)
    values = (groups / scale).to(torch.float8_e4m3fn).float()
    return (values * scale).reshape(x.shape)


def _route_outputs(x, w13, w2, topk_ids):
    """FP32 ``[M, top_k, H]`` per-route outputs of the MXFP8 SiLU experts."""
    xq = _mxfp8_round_trip(x)
    out = torch.zeros((*topk_ids.shape, HIDDEN), device=x.device, dtype=torch.float32)
    for expert in topk_ids.unique().tolist():
        token, slot = (topk_ids == expert).nonzero(as_tuple=True)
        gate_up = xq[token] @ w13[expert].T
        gate = gate_up[:, :INTER].bfloat16().float()
        up = gate_up[:, INTER:].bfloat16().float()
        h = (gate * (1.0 / (1.0 + torch.exp(-gate))) * up).bfloat16()
        out[token, slot] = _mxfp8_round_trip(h) @ w2[expert].T
    return out


def _assert_top_k_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # Ten routes per token sum through BF16 atomics, each add rounding to
    # BF16 in arrival order; the error stays within a few BF16 ULPs.
    error = (actual.float() - expected).abs()
    assert error.max() <= 8 * 2**-8 * expected.abs().max()
    assert error.norm() <= 2**-7 * expected.norm()


def _dequantized_bank(w: torch.nn.Module):
    return (
        _dequant(w.w13_weight, w.w13_weight_scale),
        _dequant(w.w2_weight, w.w2_weight_scale),
    )


def test_plan_selects_gluon_and_pads_the_intermediate() -> None:
    plan = _plan()
    assert plan["apply_kernel_name"] == _KERNEL
    w = _weights()
    moe_process_weights(plan, w)
    assert n16_mxfp4_shape(
        w.w13_weight, w.w13_weight_scale, w.w2_weight, w.w2_weight_scale
    ) == (EXPERTS, HIDDEN, PADDED_INTER)


@pytest.mark.parametrize("m", [1, 2, 4, 37, 300])
def test_single_route_matches_reference(m: int) -> None:
    # One route per token: the BF16 atomic output add rounds once. Most
    # elements match the FP32 reference to one BF16 rounding; FP32 summation
    # order can flip an E4M3 rounding of the requantized intermediate, which
    # moves a whole output row by about one E4M3 step of one of its 640
    # inputs, so rows are compared by norm.
    w = _weights()
    w13, w2 = _dequantized_bank(w)
    plan = _plan()
    moe_process_weights(plan, w)
    x, weights, ids = _routes(m, 1)
    actual = moe_apply(plan, x, w, None, topk_weights=weights, topk_ids=ids).float()
    expected = (_route_outputs(x, w13, w2, ids) * weights[..., None]).sum(1)
    close = torch.isclose(actual, expected, atol=1e-3, rtol=2**-8)
    assert close.float().mean() >= 0.99
    row_error = (actual - expected).norm(dim=-1) / expected.norm(dim=-1)
    assert row_error.max() <= 2**-7


@pytest.mark.parametrize("m", [1, 3, 4, 64, 1100])
def test_top_k_matches_reference(m: int) -> None:
    w = _weights()
    w13, w2 = _dequantized_bank(w)
    plan = _plan()
    moe_process_weights(plan, w)
    x, weights, ids = _routes(m, TOP_K)
    actual = moe_apply(plan, x, w, None, topk_weights=weights, topk_ids=ids)
    expected = (_route_outputs(x, w13, w2, ids) * weights[..., None]).sum(1)
    _assert_top_k_close(actual, expected)


def test_padded_intermediate_adds_exactly_zero() -> None:
    # The padded columns must not change the result: a bank built at the
    # padded width with explicit zero codes and unit scales matches.
    w = _weights()
    padded = _weights()
    for name, value in (("w13_weight", 0), ("w13_weight_scale", 127)):
        tensor = getattr(w, name).reshape(EXPERTS, 2, INTER, -1)
        wide = tensor.new_full((EXPERTS, 2, PADDED_INTER, tensor.shape[-1]), value)
        wide[:, :, :INTER] = tensor
        setattr(
            padded,
            name,
            torch.nn.Parameter(wide.reshape(EXPERTS, 2 * PADDED_INTER, -1), False),
        )
    for name, value, group in (("w2_weight", 0, 2), ("w2_weight_scale", 127, 32)):
        tensor = getattr(w, name)
        wide = tensor.new_full((EXPERTS, HIDDEN, PADDED_INTER // group), value)
        wide[..., : INTER // group] = tensor
        setattr(padded, name, torch.nn.Parameter(wide, requires_grad=False))
    plan = _plan()
    moe_process_weights(plan, w)
    moe_process_weights(plan, padded)
    for name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
        assert torch.equal(getattr(w, name), getattr(padded, name))


def test_batch_sizes_do_not_recompile() -> None:
    w = _weights()
    plan = _plan()
    moe_process_weights(plan, w)
    # One batch per expert-tile height (32 rows up to 1024 tokens, 128 above).
    for m in (1, 1025):
        x, weights, ids = _routes(m, TOP_K)
        moe_apply(plan, x, w, None, topk_weights=weights, topk_ids=ids)
    with assert_no_triton_compile(mxfp8_gemm._mxfp8_stage1, mxfp8_gemm._mxfp8_stage2):
        for m in (2, 3, 4, 17, 300, 1024, 1500):
            x, weights, ids = _routes(m, TOP_K)
            moe_apply(plan, x, w, None, topk_weights=weights, topk_ids=ids)


def test_graph_replay_follows_new_inputs() -> None:
    w = _weights()
    w13, w2 = _dequantized_bank(w)
    plan = _plan()
    moe_process_weights(plan, w)
    x, weights, ids = _routes(4, TOP_K)
    eager = moe_apply(plan, x, w, None, topk_weights=weights, topk_ids=ids)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        moe_apply(plan, x, w, None, topk_weights=weights, topk_ids=ids)
        with torch.cuda.graph(graph, stream=stream):
            out = moe_apply(plan, x, w, None, topk_weights=weights, topk_ids=ids)
    stream.synchronize()
    new_x, new_weights, new_ids = _routes(4, TOP_K, seed=7)
    x.copy_(new_x)
    weights.copy_(new_weights)
    ids.copy_(new_ids)
    graph.replay()
    torch.cuda.synchronize()
    expected = (_route_outputs(x, w13, w2, ids) * weights[..., None]).sum(1)
    _assert_top_k_close(out, expected)
    assert not torch.equal(out, eager)
