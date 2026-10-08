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

"""flashinfer TRTLLM-Gen NVFP4 squared-ReLU MoE (non-gated) vs a dequantized reference.

Squared ReLU is Nemotron-H's expert activation: GEMM1 holds only the up
projection (``w13`` is ``[E, I, H]``), and ``relu(x)**2`` is not homogeneous
of degree one, so the GEMM1 dequant must happen before the activation. The
reference runs the MoE in fp32 over the exactly-dequantized weights with both
w4a4 activation quantizations modeled, as flashinfer's own reference does.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

HIDDEN = 256
ISPP = 256

_FP8_E4M3_MAX = 448.0
_FP4_E2M1_MAX = 6.0
_E2M1_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_E2M1_VALUES = torch.tensor(_E2M1_LUT + [-v for v in _E2M1_LUT])


def _relu2_runtime_reason() -> str | None:
    if not torch.cuda.is_available():
        return "requires CUDA"
    import tokenspeed_kernel.ops.moe.flashinfer.trtllm_nvfp4  # noqa: F401
    from tokenspeed_kernel.platform import current_platform
    from tokenspeed_kernel.registry import KernelRegistry

    spec = KernelRegistry.get().get_by_name(
        "flashinfer_trtllm_nvfp4_relu2_routed_moe_apply"
    )
    if spec is None or not spec.capability.satisfied_by(current_platform()):
        return "outside the kernel's registered capability range"
    return None


_reason = _relu2_runtime_reason()
requires_relu2 = pytest.mark.skipif(_reason is not None, reason=str(_reason))


def _nvfp4_quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from flashinfer import fp4_quantize

    global_scale = (_FP8_E4M3_MAX * _FP4_E2M1_MAX / w.abs().amax().clamp(min=1e-12)).to(
        torch.float32
    )
    packed, sf = fp4_quantize(
        w.cuda().to(torch.bfloat16), global_scale.cuda(), is_sf_swizzled_layout=False
    )
    m, k = w.shape
    sf = sf.reshape(-1)[: m * (k // 16)].view(m, k // 16)
    return packed, sf.view(torch.float8_e4m3fn), (1.0 / global_scale).cpu()


def _nvfp4_dequant(
    packed: torch.Tensor, sf: torch.Tensor, weight_scale_2: torch.Tensor
) -> torch.Tensor:
    lut = _E2M1_VALUES.to(packed.device)
    lo = lut[(packed & 0x0F).long()]
    hi = lut[(packed >> 4).long()]
    vals = torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], -1)
    return (
        vals
        * sf.float().repeat_interleave(16, dim=1)
        * weight_scale_2.to(packed.device)
    )


def _quant_dequant_activation(
    t: torch.Tensor, global_quant_scale: float
) -> torch.Tensor:
    from flashinfer import fp4_quantize

    packed, sf = fp4_quantize(
        t.bfloat16().cuda(),
        torch.tensor(global_quant_scale, dtype=torch.float32, device="cuda"),
        is_sf_swizzled_layout=False,
    )
    m, k = t.shape
    sf = sf.reshape(-1)[: m * (k // 16)].view(m, k // 16).view(torch.float8_e4m3fn)
    return _nvfp4_dequant(packed, sf, torch.tensor(1.0 / global_quant_scale))


class _MoEWeights(torch.nn.Module):
    """Minimal module carrying what the preprocessor and apply consume."""

    def __init__(self, raw: dict[str, torch.Tensor], num_experts: int, top_k: int):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(raw["w13_weight"], requires_grad=False)
        self.w13_weight_scale = torch.nn.Parameter(
            raw["w13_weight_scale"], requires_grad=False
        )
        self.w2_weight = torch.nn.Parameter(raw["w2_weight"], requires_grad=False)
        self.w2_weight_scale = torch.nn.Parameter(
            raw["w2_weight_scale"], requires_grad=False
        )
        self.w13_weight_scale_2 = torch.nn.Parameter(
            raw["w13_weight_scale_2"], requires_grad=False
        )
        self.w2_weight_scale_2 = torch.nn.Parameter(
            raw["w2_weight_scale_2"], requires_grad=False
        )
        self.w13_input_scale = torch.nn.Parameter(
            raw["w13_input_scale"], requires_grad=False
        )
        self.w2_input_scale = torch.nn.Parameter(
            raw["w2_input_scale"], requires_grad=False
        )
        self._spec = SimpleNamespace(
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            ep_rank=0,
        )
        # In-kernel routing reads these, as an MoE layer declares them.
        self._correction_bias: torch.Tensor | None = None
        self._routing_method_type = 0
        self._routing_logits_dtype = torch.bfloat16
        self._n_group = 0
        self._topk_group = 0
        self._routed_scaling_factor = 1.0


def _make_weights(
    generator: torch.Generator, num_experts: int, input_scales: tuple[float, float]
) -> dict[str, torch.Tensor]:
    """Non-gated NVFP4 experts in the loader layout: w13 ``[E, I, H]``, w2 ``[E, H, I]``."""
    w13, w13_scale, w13_s2 = [], [], []
    w2, w2_scale, w2_s2 = [], [], []
    for _ in range(num_experts):
        packed, sf, s2 = _nvfp4_quantize(
            torch.randn(ISPP, HIDDEN, generator=generator) * 0.5
        )
        w13.append(packed), w13_scale.append(sf), w13_s2.append(s2)
        packed, sf, s2 = _nvfp4_quantize(
            torch.randn(HIDDEN, ISPP, generator=generator) * 0.5
        )
        w2.append(packed), w2_scale.append(sf), w2_s2.append(s2)
    return {
        "w13_weight": torch.stack(w13),
        "w13_weight_scale": torch.stack(w13_scale),
        "w13_weight_scale_2": torch.stack(w13_s2).reshape(num_experts),
        "w2_weight": torch.stack(w2),
        "w2_weight_scale": torch.stack(w2_scale),
        "w2_weight_scale_2": torch.stack(w2_s2).reshape(num_experts),
        "w13_input_scale": torch.tensor([input_scales[0]]),
        "w2_input_scale": torch.tensor([input_scales[1]]),
    }


def _reference_moe(
    hidden_states: torch.Tensor,
    raw: dict[str, torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    x = _quant_dequant_activation(
        hidden_states.float(), 1.0 / raw["w13_input_scale"].item()
    )
    out = torch.zeros_like(x)
    for e in topk_ids.unique().tolist():
        w13 = _nvfp4_dequant(
            raw["w13_weight"][e].cuda(),
            raw["w13_weight_scale"][e].cuda(),
            raw["w13_weight_scale_2"][e],
        )
        w2 = _nvfp4_dequant(
            raw["w2_weight"][e].cuda(),
            raw["w2_weight_scale"][e].cuda(),
            raw["w2_weight_scale_2"][e],
        )
        act = torch.relu(x @ w13.t()).square()
        act = _quant_dequant_activation(act, 1.0 / raw["w2_input_scale"].item())
        weight = torch.where(topk_ids == e, topk_weights.float(), 0.0).sum(dim=-1)
        out += weight[:, None] * (act @ w2.t())
    return out


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


def _random_topk(generator, num_tokens, num_experts, top_k):
    ids = torch.stack(
        [
            torch.randperm(num_experts, generator=generator)[:top_k]
            for _ in range(num_tokens)
        ]
    )
    weights = torch.rand(num_tokens, top_k, generator=generator).softmax(dim=-1)
    return ids.to(torch.int32).cuda(), weights.bfloat16().cuda()


@requires_relu2
@pytest.mark.parametrize("input_scales", [(1.0, 1.0), (0.05, 2.0)])
@pytest.mark.parametrize("num_tokens", [1, 37])
def test_relu2_routed_moe_matches_dequant_reference(input_scales, num_tokens):
    from tokenspeed_kernel.ops.moe.flashinfer.trtllm_nvfp4 import (
        flashinfer_trtllm_nvfp4_relu2_moe_weights,
        flashinfer_trtllm_nvfp4_relu2_routed_moe_apply,
    )

    num_experts, top_k = 16, 6
    generator = torch.Generator().manual_seed(20260930)
    raw = _make_weights(generator, num_experts, input_scales)
    hidden_states = (
        (torch.randn(num_tokens, HIDDEN, generator=generator) * 0.2).bfloat16().cuda()
    )
    topk_ids, topk_weights = _random_topk(generator, num_tokens, num_experts, top_k)

    w = _MoEWeights({k: v.clone() for k, v in raw.items()}, num_experts, top_k).cuda()
    flashinfer_trtllm_nvfp4_relu2_moe_weights({}, w)
    assert w.intermediate_size_per_partition == ISPP
    actual = flashinfer_trtllm_nvfp4_relu2_routed_moe_apply(
        {},
        hidden_states,
        w,
        router_logits=None,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    torch.cuda.synchronize()
    expected = _reference_moe(hidden_states, raw, topk_ids, topk_weights)
    err = _rel_l2(actual, expected)
    assert err < 0.06, f"{err=:.4f}"


@requires_relu2
@pytest.mark.parametrize("num_tokens", [3, 64])
def test_relu2_kernel_routing_matches_sigmoid_bias_top22(num_tokens):
    """Nemotron-H routing: 512 experts, sigmoid + correction bias, one group, top-22, scale 5."""
    from flashinfer.fused_moe import RoutingMethodType
    from tokenspeed_kernel.ops.moe.flashinfer.trtllm_nvfp4 import (
        flashinfer_trtllm_nvfp4_relu2_moe_apply,
        flashinfer_trtllm_nvfp4_relu2_moe_weights,
    )

    num_experts, top_k, scale = 512, 22, 5.0
    generator = torch.Generator().manual_seed(88)
    raw = _make_weights(generator, num_experts, (1.0, 1.0))
    hidden_states = (
        (torch.randn(num_tokens, HIDDEN, generator=generator) * 0.2).bfloat16().cuda()
    )
    logits = torch.randn(num_tokens, num_experts, generator=generator).cuda()
    bias = (torch.randn(num_experts, generator=generator) * 0.1).cuda()

    w = _MoEWeights({k: v.clone() for k, v in raw.items()}, num_experts, top_k).cuda()
    w._correction_bias = bias.float()
    w._routing_method_type = RoutingMethodType.DeepSeekV3
    w._routing_logits_dtype = torch.float32
    w._n_group = 1
    w._topk_group = 1
    w._routed_scaling_factor = scale
    flashinfer_trtllm_nvfp4_relu2_moe_weights({}, w)
    actual = flashinfer_trtllm_nvfp4_relu2_moe_apply({}, hidden_states, w, logits)
    torch.cuda.synchronize()

    scores = logits.sigmoid()
    topk_ids = (scores + bias).topk(top_k, dim=-1).indices
    picked = scores.gather(1, topk_ids)
    topk_weights = picked / picked.sum(dim=-1, keepdim=True) * scale
    expected = _reference_moe(hidden_states, raw, topk_ids, topk_weights)
    err = _rel_l2(actual, expected)
    assert err < 0.06, f"{err=:.4f}"


@requires_relu2
@pytest.mark.parametrize("routing_mode", [None, "precomputed_topk"])
def test_moe_plan_selects_the_relu2_kernels(routing_mode):
    import tokenspeed_kernel

    plan = tokenspeed_kernel.moe_plan(
        "nvfp4",
        input_dtype=torch.bfloat16,
        activation="relu2",
        routing_mode=routing_mode,
        ep_size=1,
        ispp=2688,
        internal_activation_dtype="input",
        hidden=None,
        swiglu_form=None,
        activation_clamped=False,
        expert_id_repeats=False,
        fast_math=True,
        combine_order="rank",
    )
    expected = {
        None: "flashinfer_trtllm_nvfp4_relu2_moe_apply",
        "precomputed_topk": "flashinfer_trtllm_nvfp4_relu2_routed_moe_apply",
    }[routing_mode]
    assert plan["apply_kernel_name"] == expected
    assert (
        plan["weight_preprocessor"].__name__
        == "flashinfer_trtllm_nvfp4_relu2_moe_weights"
    )


@requires_relu2
def test_moe_plan_rejects_an_intermediate_width_off_128_rows():
    import tokenspeed_kernel
    from tokenspeed_kernel.selection import NoKernelFoundError

    with pytest.raises(NoKernelFoundError):
        tokenspeed_kernel.moe_plan(
            "nvfp4",
            input_dtype=torch.bfloat16,
            activation="relu2",
            routing_mode=None,
            ep_size=1,
            ispp=2624,
            internal_activation_dtype="input",
            hidden=None,
            swiglu_form=None,
            activation_clamped=False,
            expert_id_repeats=False,
            fast_math=True,
            combine_order="rank",
        )
