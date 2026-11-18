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

"""flashinfer CuTe DSL NVFP4 SiTU MoE (w4a4) on NVFP4 checkpoint experts.

The preprocessor's layout is checked bytewise against FlashInfer's own
``prepare_cute_dsl_weights``; the registered apply against an fp32 SiTU MoE
over the exactly-dequantized weights (both activation quantizations modeled),
anchored to the TRT-LLM-Gen SiTU kernel on the same checkpoint.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.numerics.torch.activation import torch_situ_and_mul
from tokenspeed_kernel.numerics.torch.moe import DequantizedExperts, torch_moe_experts
from tokenspeed_kernel.numerics.torch.quantization import dequantize_nvfp4_blocks
from tokenspeed_kernel.ops.moe import moe_plan as kernel_moe_plan

NUM_EXPERTS = 16
TOP_K = 8
HIDDEN = 256
ISPP = 256  # the CuTe DSL tactic tiles 2 * ISPP and HIDDEN by 256
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0
_FP8_E4M3_MAX = 448.0
_FP4_E2M1_MAX = 6.0


def _runtime_reason() -> str | None:
    if not torch.cuda.is_available():
        return "requires CUDA"
    import tokenspeed_kernel.ops.moe.flashinfer_cutedsl  # noqa: F401
    from tokenspeed_kernel.platform import current_platform
    from tokenspeed_kernel.registry import KernelRegistry

    spec = KernelRegistry.get().get_by_name(
        "flashinfer_cutedsl_nvfp4_situ_routed_moe_apply"
    )
    if spec is None or not spec.capability.satisfied_by(current_platform()):
        return "outside the kernel's registered capability range"
    return None


_reason = _runtime_reason()
requires_cutedsl_situ = pytest.mark.skipif(_reason is not None, reason=str(_reason))


def _quantize(w: torch.Tensor, unit_scale: bool):
    """NVFP4 of a 2-D weight: packed codes, linear E4M3 scales, weight_scale_2."""
    from flashinfer import fp4_quantize

    w = w.cuda().to(torch.bfloat16)
    global_scale = (
        torch.ones((), device="cuda")
        if unit_scale
        else _FP8_E4M3_MAX * _FP4_E2M1_MAX / w.float().abs().amax().clamp(min=1e-12)
    ).reshape(1)
    packed, sf = fp4_quantize(w, global_scale, is_sf_swizzled_layout=False)
    m, k = w.shape
    sf = sf.reshape(-1)[: m * (k // 16)].view(m, k // 16).view(torch.float8_e4m3fn)
    return packed.view(m, k // 2), sf, (1.0 / global_scale).reshape(())


def _checkpoint(w13_bf16, w2_bf16, *, unit_scale: bool) -> dict[str, torch.Tensor]:
    """Loader-layout NVFP4 experts: W13 rows [gate | up], [E, 2] gate/up scale_2."""
    q13 = [_quantize(w, unit_scale) for w in w13_bf16]
    q2 = [_quantize(w, unit_scale) for w in w2_bf16]
    return {
        "w13_weight": torch.stack([p for p, _, _ in q13]),
        "w13_weight_scale": torch.stack([s for _, s, _ in q13]),
        "w13_weight_scale_2": torch.stack([torch.stack((g, g)) for _, _, g in q13]),
        "w2_weight": torch.stack([p for p, _, _ in q2]),
        "w2_weight_scale": torch.stack([s for _, s, _ in q2]),
        "w2_weight_scale_2": torch.stack([g for _, _, g in q2]),
        "w13_input_scale": torch.ones(1, device="cuda"),
        "w2_input_scale": torch.ones(NUM_EXPERTS, device="cuda"),
    }


class _Experts(torch.nn.Module):
    """Minimal module carrying what the preprocessors and applies consume."""

    def __init__(self, raw: dict[str, torch.Tensor]) -> None:
        from types import SimpleNamespace

        from flashinfer.fused_moe import RoutingMethodType

        super().__init__()
        for name, value in raw.items():
            self.register_parameter(
                name, torch.nn.Parameter(value.clone(), requires_grad=False)
            )
        self.activation_situ_beta = SITU_BETA
        self.activation_situ_linear_beta = SITU_LINEAR_BETA
        self._routing_method_type = RoutingMethodType.RenormalizeNaive
        self._routing_logits_dtype = torch.bfloat16
        self._spec = SimpleNamespace(
            num_experts=NUM_EXPERTS,
            num_local_experts=NUM_EXPERTS,
            top_k=TOP_K,
            ep_rank=0,
        )


def _weights(generator: torch.Generator):
    w13 = torch.randn(NUM_EXPERTS, 2 * ISPP, HIDDEN, generator=generator) * 0.5
    w2 = torch.randn(NUM_EXPERTS, HIDDEN, ISPP, generator=generator) * 0.5
    return w13.bfloat16(), w2.bfloat16()


def _routing(generator: torch.Generator, num_tokens: int):
    # TOP_K distinct experts per token.
    ids = torch.rand(num_tokens, NUM_EXPERTS, generator=generator).argsort(dim=1)
    ids = ids[:, :TOP_K]
    weights = torch.rand(num_tokens, TOP_K, generator=generator).softmax(dim=-1)
    return ids.to(torch.int32).cuda(), weights.cuda()


def _quant_dequant(t: torch.Tensor) -> torch.Tensor:
    """The kernels' NVFP4 activation quantization (unit global scale) in fp32."""
    packed, sf, scale_2 = _quantize(t, unit_scale=True)
    return dequantize_nvfp4_blocks(packed, sf) * scale_2


def _dequantize(packed, sf, scale_2) -> torch.Tensor:
    """One expert's weight; W13 scale_2 holds one value per [gate | up] half."""
    w = dequantize_nvfp4_blocks(packed, sf)
    halves = w.unflatten(0, (scale_2.numel(), -1))
    return (halves * scale_2.view(-1, 1, 1)).flatten(0, 1)


def _reference(hidden_states, raw, topk_ids, topk_weights) -> torch.Tensor:
    def experts(prefix: str) -> DequantizedExperts:
        return DequantizedExperts(
            _dequantize,
            raw[f"{prefix}_weight"],
            raw[f"{prefix}_weight_scale"],
            raw[f"{prefix}_weight_scale_2"],
        )

    return torch_moe_experts(
        hidden_states,
        experts("w13"),
        experts("w2"),
        topk_ids,
        topk_weights,
        activation=lambda gate_up: torch_situ_and_mul(
            gate_up, SITU_BETA, SITU_LINEAR_BETA
        ),
        w13_bias=None,
        w2_bias=None,
        quantize_input=lambda t: _quant_dequant(t.float()),
        quantize_intermediate=_quant_dequant,
        round_expert_output=None,
        first_expert=0,
    )


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


@requires_cutedsl_situ
def test_preprocessor_layout_matches_flashinfer_prepare() -> None:
    """Checkpoint experts land exactly where FlashInfer's own BF16 preparation
    puts them: W13 [up | gate] interleaved by 64 rows, swizzled scales."""
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    from flashinfer.fused_moe.api import QuantConfig, QuantFormat
    from flashinfer.fused_moe.prepare import prepare_cute_dsl_weights
    from tokenspeed_kernel.ops.moe.flashinfer_cutedsl import (
        flashinfer_cutedsl_nvfp4_situ_moe_weights,
    )

    w13, w2 = _weights(torch.Generator().manual_seed(20261116))
    w = _Experts(_checkpoint(w13, w2, unit_scale=True)).cuda()
    flashinfer_cutedsl_nvfp4_situ_moe_weights({}, w)
    up_gate = torch.cat([w13[:, ISPP:], w13[:, :ISPP]], dim=1).cuda()
    ref = prepare_cute_dsl_weights(
        up_gate,
        w2.cuda(),
        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
        num_local_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=ISPP,
    )

    def mma_view(sf: torch.Tensor, m: int, k: int) -> torch.Tensor:
        return convert_sf_to_mma_layout(
            sf.view(NUM_EXPERTS * m, -1), m=m, k=k, num_groups=NUM_EXPERTS
        )

    assert torch.equal(w.cutedsl_w1_weight, ref["w1_weight"].view(torch.uint8))
    assert torch.equal(w.cutedsl_w2_weight, ref["w2_weight"].view(torch.uint8))
    assert torch.equal(
        mma_view(w.cutedsl_w1_weight_sf, 2 * ISPP, HIDDEN).view(torch.uint8),
        ref["w1_weight_sf"].view(torch.uint8),
    )
    assert torch.equal(
        mma_view(w.cutedsl_w2_weight_sf, HIDDEN, ISPP).view(torch.uint8),
        ref["w2_weight_sf"].view(torch.uint8),
    )


@requires_cutedsl_situ
def test_routed_moe_matches_dequant_reference_and_trtllm() -> None:
    from flashinfer import fp4_quantize
    from tokenspeed_kernel.ops.moe.flashinfer import (
        flashinfer_trtllm_nvfp4_situ_moe_weights,
        flashinfer_trtllm_nvfp4_situ_routed_moe_apply,
    )
    from tokenspeed_kernel.ops.moe.flashinfer_cutedsl import (
        flashinfer_cutedsl_nvfp4_situ_moe_weights,
        flashinfer_cutedsl_nvfp4_situ_routed_moe_apply,
    )

    generator = torch.Generator().manual_seed(20261117)
    num_tokens = 64
    w13, w2 = _weights(generator)
    raw = _checkpoint(w13, w2, unit_scale=False)
    hidden_states = (
        (torch.randn(num_tokens, HIDDEN, generator=generator) * 0.2).bfloat16().cuda()
    )
    topk_ids, topk_weights = _routing(generator, num_tokens)

    cute = _Experts(raw).cuda()
    flashinfer_cutedsl_nvfp4_situ_moe_weights({}, cute)
    trtllm = _Experts(raw).cuda()
    flashinfer_trtllm_nvfp4_situ_moe_weights({}, trtllm)

    def run_cute(x):
        return flashinfer_cutedsl_nvfp4_situ_routed_moe_apply(
            {},
            x,
            cute,
            router_logits=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )

    actual = run_cute(hidden_states)
    serving = flashinfer_trtllm_nvfp4_situ_routed_moe_apply(
        {},
        hidden_states,
        trtllm,
        router_logits=None,
        topk_weights=topk_weights.bfloat16(),
        topk_ids=topk_ids,
    )
    torch.cuda.synchronize()
    expected = _reference(hidden_states, raw, topk_ids, topk_weights)
    cute_err, serving_err = _rel_l2(actual, expected), _rel_l2(serving, expected)
    assert cute_err < 0.06, f"{cute_err=:.4f}"
    assert cute_err < 1.5 * serving_err + 0.005, f"{cute_err=:.4f} {serving_err=:.4f}"

    # An NVFP4 pair from the latent projection gives the BF16-input result.
    # The finalize adds experts with BF16 atomics in arrival order, so runs
    # differ by rounding (measured 4.5e-3 here); a wrong layout is O(1).
    x_fp4, x_sf = fp4_quantize(
        hidden_states, cute.w13_input_scale_quant, is_sf_swizzled_layout=False
    )
    prequantized = run_cute((x_fp4, x_sf))
    torch.cuda.synchronize()
    assert _rel_l2(prequantized, actual) < 1e-2


@requires_cutedsl_situ
def test_routed_moe_compiles_once_for_any_row_count() -> None:
    """The fixed tactic keys no compile on the row count."""
    from tokenspeed_kernel.ops.moe.flashinfer_cutedsl import (
        flashinfer_cutedsl_nvfp4_situ_moe_weights,
        flashinfer_cutedsl_nvfp4_situ_routed_moe_apply,
    )
    from utils import assert_no_cute_compile

    generator = torch.Generator().manual_seed(20261119)
    w13, w2 = _weights(generator)
    cute = _Experts(_checkpoint(w13, w2, unit_scale=False)).cuda()
    flashinfer_cutedsl_nvfp4_situ_moe_weights({}, cute)

    def run(num_tokens: int) -> torch.Tensor:
        hidden_states = torch.randn(num_tokens, HIDDEN, generator=generator)
        topk_ids, topk_weights = _routing(generator, num_tokens)
        return flashinfer_cutedsl_nvfp4_situ_routed_moe_apply(
            {},
            hidden_states.bfloat16().cuda(),
            cute,
            router_logits=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )

    run(64)
    with assert_no_cute_compile():
        for num_tokens in (1, 37, 300):
            run(num_tokens)
        # An idle attention-DP rank routes no tokens.
        assert run(0).shape == (0, HIDDEN)
    torch.cuda.synchronize()


@requires_cutedsl_situ
def test_preprocessor_rejects_per_expert_gemm2_input_scales() -> None:
    from tokenspeed_kernel.ops.moe.flashinfer_cutedsl import (
        flashinfer_cutedsl_nvfp4_situ_moe_weights,
    )

    w13, w2 = _weights(torch.Generator().manual_seed(20261118))
    raw = _checkpoint(w13, w2, unit_scale=False)
    raw["w2_input_scale"] = torch.linspace(0.5, 1.0, NUM_EXPERTS, device="cuda")
    with pytest.raises(RuntimeError, match="one global"):
        flashinfer_cutedsl_nvfp4_situ_moe_weights({}, _Experts(raw).cuda())


@requires_cutedsl_situ
@pytest.mark.parametrize(
    "solution,kernel",
    [
        ("flashinfer_cutedsl", "flashinfer_cutedsl_nvfp4_situ_routed_moe_apply"),
        (None, "flashinfer_trtllm_nvfp4_situ_routed_moe_apply"),
    ],
)
def test_plan_selects_cutedsl_only_when_requested(solution, kernel) -> None:
    plan = kernel_moe_plan(
        "nvfp4",
        input_dtype=torch.bfloat16,
        activation="situ",
        routing_mode="precomputed_topk",
        ep_size=1,
        ispp=3072,
        internal_activation_dtype="input",
        hidden=3584,
        swiglu_form=None,
        activation_clamped=False,
        expert_id_repeats=False,
        num_experts=896,
        top_k=16,
        fast_math=True,
        combine_order="rank",
        features=frozenset(),
        solution=solution,
    )
    assert plan["apply_kernel_name"] == kernel
