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

"""What Beam computes, on its real modules.

On CPU: the residual boundaries and the depth-scale fold, the output gate,
the NoPE Q/K norm with SSMax, the router and its route weights, and the strict
loader. The shared RMSNorm kernel needs a GPU: here it is its FP32 statement.
On one CUDA GPU: the module equations of the tiny model.
"""

from __future__ import annotations

import math
import os
import sys
from copy import copy

import pytest
import tokenspeed_kernel.ops.layernorm.qk_rmsnorm_ssmax as qk_facade
import torch
from torch.nn import functional as F

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.attention.gating import apply_attention_output_gate
from tokenspeed.runtime.models import beam
from tokenspeed.runtime.models.beam import (
    BeamForCausalLM,
    BeamLogitsProcessor,
    BeamResidualFoldRMSNorm,
    BeamRMSNorm,
    beam_checkpoint_name,
    beam_route_weights,
    depth_scale_coefficients,
)

# The CI runner executes each registered test file as a standalone script.
_TEST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))
from test.runtime.models.beam_fixture import tiny_config, tiny_weights

from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="runtime-1gpu", disabled_on_runners=["amd-*"])

EPS = 1e-6


def _norm(x, weight, eps, *, residual=None, out=None, enable_pdl=None, **_):
    """The shared RMSNorm kernel's statement: normalize and scale in FP32, one
    rounding to the input dtype."""
    assert residual is None
    x32 = x.float()
    scale = torch.rsqrt(x32.square().mean(-1, keepdim=True) + eps)
    result = (x32 * scale * weight.float()).to(x.dtype)
    return result if out is None else out.copy_(result)


@pytest.fixture
def cpu_rmsnorm(monkeypatch):
    """The shared RMSNorm kernel (Triton, GPU only) as its FP32 statement, where
    Beam's norms and the NoPE Q/K norm's torch kernel call it."""
    for module in (beam, qk_facade):
        monkeypatch.setattr(module, "rmsnorm", _norm)


def _tiny_model():
    return BeamForCausalLM(tiny_config(), Mapping(rank=0, world_size=1), None)


def _bf16(*shape, seed=0, scale=1.0):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=generator) * scale).to(torch.bfloat16)


# ---------------------------------------------------------------- residual and fold
def test_the_depth_scale_pairs():
    """Depth 2 .. 105 for Beam's 52 layers, (0, attention), (0, MLP), (1, ...)."""
    pairs = [
        depth_scale_coefficients(layer, is_attention)
        for layer in range(52)
        for is_attention in (True, False)
    ]
    assert pairs == [
        (1.0 / math.sqrt(float(d)), math.sqrt(float(d - 1)) / math.sqrt(float(d)))
        for d in range(2, 106)
    ]
    assert depth_scale_coefficients(0, False) == (
        0.5773502691896258,
        0.8164965809277261,
    )


def test_residual_products_and_sum_round_at_the_activation_dtype(cpu_rmsnorm):
    """Both products and the sum are BF16 (not one FP32 sum), and the norm runs on
    that BF16 sum."""
    torch.manual_seed(0)
    x, residual = torch.randn(2, 8).bfloat16(), torch.randn(2, 8).bfloat16()
    scales = (1 / math.sqrt(5), math.sqrt(4 / 5))
    norm = BeamRMSNorm(8, EPS, torch.bfloat16)
    normed, combined = norm(x, residual, x_scale=scales[0], residual_scale=scales[1])
    expected = x * scales[0] + residual * scales[1]
    assert torch.equal(combined, expected)
    assert not torch.equal(
        expected, (x.float() * scales[0] + residual.float() * scales[1]).bfloat16()
    )
    assert torch.equal(normed, _norm(expected, norm.weight, EPS))
    with pytest.raises(ValueError, match="need a residual"):
        norm(x, None, x_scale=0.5, residual_scale=0.5)


def test_a_fold_norm_adds_the_previous_layers_pair(cpu_rmsnorm):
    """The norm consuming layer i's MLP boundary applies layer i's pair itself:
    the explicit-scale call's bytes. It refuses a second source of the pair, a
    call without a residual and a pair it cannot fold."""
    pair = depth_scale_coefficients(0, False)
    fold = BeamResidualFoldRMSNorm(5120, EPS, torch.bfloat16, residual_fold=pair)
    plain = BeamRMSNorm(5120, EPS, torch.bfloat16)
    weight = _bf16(5120, seed=1) + 1
    fold.weight.data.copy_(weight)
    plain.weight.data.copy_(weight)
    output, residual = _bf16(8, 5120, seed=2, scale=4), _bf16(8, 5120, seed=3, scale=4)
    folded = fold(output, residual)
    explicit = plain(output, residual, x_scale=pair[0], residual_scale=pair[1])
    assert all(torch.equal(a, b) for a, b in zip(folded, explicit, strict=True))
    with pytest.raises(ValueError, match="two sources"):
        fold(output, residual, x_scale=pair[0], residual_scale=pair[1])
    with pytest.raises(ValueError, match="consumes a residual pair"):
        fold(output)
    with pytest.raises(ValueError):
        BeamResidualFoldRMSNorm(5120, EPS, torch.bfloat16, residual_fold=(pair[0], 1))


# ---------------------------------------------------------------- attention
@pytest.mark.parametrize(
    "dtype, floor, temperature",
    [(torch.bfloat16, 0.0, 1.0), (torch.float32, 0.125, 0.7)],
    ids=["beam", "floor-temperature"],
)
def test_the_output_gate_rounds_once(dtype, floor, temperature):
    """Bias, sigmoid, floor and product in FP32 with one rounding (Beam's gate
    is floor 0, temperature 1), into a new tensor; the checkpoint reference's
    three BF16 roundings differ."""
    generator = torch.Generator().manual_seed(0)
    output = torch.randn(6, 4, 5, generator=generator).to(dtype)
    gate = (torch.randn(6, 4, 5, generator=generator) * 4).to(dtype)
    bias = torch.randn(4, generator=generator).to(dtype)
    original = output.clone()
    logits = (gate.double() + bias.double()[:, None]) / temperature
    expected = output.double() * (floor + (1 - floor) / (1 + (-logits).exp()))
    actual = apply_attention_output_gate(output, gate, bias, floor, temperature)
    tolerance = 1e-6 if dtype == torch.float32 else 0
    torch.testing.assert_close(actual, expected.to(dtype), rtol=tolerance, atol=0)
    assert torch.equal(output, original)
    if dtype == torch.bfloat16:
        reference = output * torch.sigmoid(gate + bias[None, :, None])
        assert not torch.equal(actual, reference)


def test_the_nope_qk_norm_scales_q_by_ssmax(cpu_rmsnorm):
    """A NoPE layer: the per-head Q and K norms with the modules' weights, then
    Q times the FP32 SSMax scale ``exp(0.1 * log1p((p + 1) / 2048))``, one
    rounding; K unscaled."""
    attention = _tiny_model().model.layers[1].self_attn
    assert attention.rotary_emb is None
    attention.q_norm.weight.data.copy_(_bf16(128, seed=6) + 1)
    attention.k_norm.weight.data.copy_(_bf16(128, seed=7) + 1)
    qkv = _bf16(3, 16 * 128, seed=8)
    q, k, _ = qkv.split([8 * 128, 4 * 128, 4 * 128], dim=-1)
    positions = torch.tensor([0, 511, 2047])
    actual_q, actual_k = attention.qk_norm(q, k, positions)
    q_norm = _norm(q.reshape(3, 8, 128), attention.q_norm.weight, EPS)
    k_norm = _norm(k.reshape(3, 4, 128), attention.k_norm.weight, EPS)
    scale = torch.exp(0.1 * torch.log1p((positions.float() + 1) / 2048.0))
    expected_q = (q_norm * scale[:, None, None]).to(torch.bfloat16)
    assert torch.equal(actual_q.reshape(3, 8, 128), expected_q)
    assert torch.equal(actual_k.reshape(3, 4, 128), k_norm)


# ---------------------------------------------------------------- router
def test_the_router_projects_bf16_rows_onto_its_fp32_weight():
    router = _tiny_model().model.layers[1].mlp.gate
    generator = torch.Generator().manual_seed(9)
    router.weight.data.copy_(torch.randn(8, 128, generator=generator))
    hidden = torch.randn(3, 128, generator=generator).bfloat16()
    logits = router(hidden)
    assert logits.dtype == torch.float32
    assert torch.equal(logits, hidden.float() @ router.weight.t())


def test_the_route_weights_are_fp32_and_take_the_routed_scale_once():
    """The sorted grouped top-k of sigmoid + bias; the FP32 weights are the
    gathered sigmoid over (their sum + 1e-20), unscaled, or times the routed
    scale rounded to FP32, once."""
    logits = torch.full((3, 8), -50.0)
    bias = torch.arange(8).float() * 0.01
    weights, ids = beam_route_weights(logits, bias, 2, True, 2, 1)
    assert torch.equal(ids, torch.tensor([[7, 6]]).expand(3, -1).int())
    scores = logits.sigmoid().gather(1, ids.long())
    assert weights.dtype == torch.float32
    assert torch.equal(weights, scores / (scores.sum(-1, keepdim=True) + 1e-20))
    assert torch.all(weights.sum(-1) < 0.1)
    generator = torch.Generator().manual_seed(10)
    logits = torch.randn(5, 8, generator=generator)
    unscaled, ids = beam_route_weights(logits, bias, 2, True, 2, 1)
    scaled, scaled_ids = beam_route_weights(logits, bias, 2, True, 2, 1, 2.5)
    assert torch.equal(scaled_ids, ids)
    assert torch.equal(scaled, unscaled * torch.tensor(2.5, dtype=torch.float32))


# ---------------------------------------------------------------- loader
def test_the_loader_maps_beams_norm_names_and_checks_the_checkpoint():
    """Each canonical norm tensor lands in the module that plays its role, and
    the strict loader refuses the module names as checkpoint names, a padded
    LM head and a missing tensor."""
    weights = tiny_weights(tiny_config())
    model = _tiny_model()
    assert model.load_weights(weights.items()) == weights.keys()
    assert beam_checkpoint_name("model.layers.0.pre_feedforward_layernorm.weight") == (
        "model.layers.0.post_attention_layernorm.weight"
    )
    for name, value in weights.items():
        if "layernorm.weight" in name:
            assert torch.equal(model.get_parameter(beam_checkpoint_name(name)), value)
    with pytest.raises(KeyError, match="obsolete checkpoint norm name"):
        beam_checkpoint_name("model.layers.0.attn_output_layernorm.weight")
    padded = dict(weights)
    padded["lm_head.weight"] = F.pad(weights["lm_head.weight"], (0, 8))
    with pytest.raises(ValueError, match="unpadded"):
        _tiny_model().load_weights(padded.items())
    missing = {k: v for k, v in weights.items() if k != "model.norm.weight"}
    with pytest.raises(ValueError, match="Missing"):
        _tiny_model().load_weights(missing.items())


# ---------------------------------------------------------------- one CUDA GPU
@pytest.fixture
def loaded(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires one CUDA GPU")
    from tokenspeed.runtime.layers.moe import utils

    monkeypatch.setattr(utils, "MOE_BACKEND", utils.MoeBackend.TRITON)
    config = tiny_config()
    weights = tiny_weights(config)
    with torch.device("cuda"):
        model = BeamForCausalLM(
            config, Mapping(rank=0, world_size=1), quant_config=None
        )
    assert model.load_weights(weights.items()) == weights.keys()
    return model.eval(), {name: value.cuda() for name, value in weights.items()}


def norm_equation(x, weight, epsilon):
    x32 = x.float()
    normalized = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + epsilon)
    return (normalized * weight.float()).to(x.dtype)


def swiglu_equation(x, weights, prefix):
    gate = F.linear(x, weights[f"{prefix}.gate_proj.weight"])
    up = F.linear(x, weights[f"{prefix}.up_proj.weight"])
    activated = (F.silu(gate.float()) * up.float()).to(x.dtype)
    return F.linear(activated, weights[f"{prefix}.down_proj.weight"])


def test_norm_and_qk_boundaries(loaded):
    model, weights = loaded
    hidden = torch.linspace(-1, 1, 4 * 128, device="cuda").reshape(4, 128).bfloat16()
    positions = torch.tensor([0, 63, 64, 511], device="cuda")
    eps = torch.finfo(torch.bfloat16).eps
    for layer_id in range(2):
        layer = model.model.layers[layer_id]
        prefix = f"model.layers.{layer_id}"
        expected = norm_equation(
            hidden, weights[f"{prefix}.input_layernorm.weight"], 1e-6
        )
        # Layer 1's input norm holds layer 0's depth-scale pair as its residual_fold
        # and refuses a residual-free call; the weight role is checked through the
        # plain forward on the same module (layer 0's plain norm answers the call).
        if layer_id == 0:
            torch.testing.assert_close(
                layer.input_layernorm(hidden), expected, rtol=eps, atol=eps
            )
        else:
            with pytest.raises(ValueError, match="consumes a residual pair"):
                layer.input_layernorm(hidden)
        torch.testing.assert_close(
            BeamRMSNorm.forward(layer.input_layernorm, hidden),
            expected,
            rtol=eps,
            atol=eps,
        )
        q, k, v, gate = layer.self_attn.prepare_qkv(hidden, positions, None)
        attention = f"{prefix}.self_attn"
        qr = norm_equation(
            F.linear(hidden, weights[f"{attention}.q_proj.weight"]).reshape(4, 8, 128),
            weights[f"{attention}.q_norm.weight"],
            1e-6,
        )
        kr = norm_equation(
            F.linear(hidden, weights[f"{attention}.k_proj.weight"]).reshape(4, 4, 128),
            weights[f"{attention}.k_norm.weight"],
            1e-6,
        )
        if layer_id == 0:
            inv_freq = 1.0 / (
                10000.0 ** (torch.arange(0, 128, 2, device="cuda").float() / 128)
            )
            phases = positions.float()[:, None] * inv_freq[None, :]
            cos = torch.cat((phases, phases), dim=-1).cos()[:, None, :]
            sin = torch.cat((phases, phases), dim=-1).sin()[:, None, :]

            def rotate(x):
                left, right = x.float().chunk(2, dim=-1)
                return (x.float() * cos + torch.cat((-right, left), -1) * sin).to(
                    x.dtype
                )

            qr, kr = rotate(qr), rotate(kr)
        else:
            scale = torch.exp(0.1 * torch.log1p((positions.float() + 1) / 2048))
            qr = (qr * scale[:, None, None]).to(qr.dtype)
        torch.testing.assert_close(q, qr, rtol=2 * eps, atol=2 * eps)
        torch.testing.assert_close(k, kr, rtol=2 * eps, atol=2 * eps)
        torch.testing.assert_close(
            v,
            F.linear(hidden, weights[f"{attention}.v_proj.weight"]).reshape_as(v),
            rtol=eps,
            atol=eps,
        )
        torch.testing.assert_close(
            gate,
            F.linear(hidden, weights[f"{attention}.gate_proj.weight"]).reshape_as(gate),
            rtol=eps,
            atol=eps,
        )


def test_shared_dense_router_and_moe(loaded):
    model, weights = loaded
    hidden = torch.linspace(-1, 1, 2 * 128, device="cuda").reshape(2, 128).bfloat16()
    eps = torch.finfo(torch.bfloat16).eps
    dense = model.model.layers[0].mlp(hidden)
    expected_dense = swiglu_equation(hidden, weights, "model.layers.0.mlp")
    torch.testing.assert_close(dense, expected_dense, rtol=2 * eps, atol=2 * eps)
    prefix = "model.layers.1.mlp"
    scores = F.linear(hidden.float(), weights[f"{prefix}.gate.weight"]).sigmoid()
    choice = scores + weights[f"{prefix}.gate.e_score_correction_bias"]
    groups = choice.reshape(2, 2, 4).topk(2, -1).values.sum(-1)
    selected_group = groups.argmax(-1)
    allowed = torch.arange(8, device="cuda")[None, :] // 4 == selected_group[:, None]
    indices = (
        choice.masked_fill(~allowed, float("-inf")).topk(2, -1, sorted=True).indices
    )
    coefficients = scores.gather(1, indices)
    coefficients = coefficients / (coefficients.sum(-1, keepdim=True) + 1e-20)
    moe = model.model.layers[1].mlp
    routes = moe.topk(hidden, moe.gate(hidden))
    assert torch.equal(routes.topk_ids.long(), indices)
    torch.testing.assert_close(routes.topk_weights, coefficients, rtol=1e-5, atol=1e-6)
    expected = torch.zeros_like(hidden, dtype=torch.float32)
    for expert in range(8):
        token, slot = torch.where(indices == expert)
        if token.numel():
            output = swiglu_equation(
                hidden[token], weights, f"{prefix}.experts.{expert}"
            )
            weighted = (output * coefficients[token, slot, None]).to(hidden.dtype)
            expected.index_add_(0, token, weighted.float())
    expected = (expected * 2.5).to(hidden.dtype)
    moe.experts.process_weights_after_loading(moe.experts)
    actual = moe(hidden, hidden.shape[0], hidden.shape[0])
    torch.testing.assert_close(actual, expected, rtol=2 * eps, atol=2 * eps)


@pytest.mark.parametrize("softcap", [None, 30.0])
def test_logits_projection_and_softcap_dtypes(loaded, softcap):
    model, weights = loaded
    config = copy(model.config)
    config.final_logit_softcapping = softcap
    processor = BeamLogitsProcessor(config, model.mapping)
    hidden = torch.linspace(-1, 1, 2 * 128, device="cuda").reshape(2, 128).bfloat16()
    actual = processor._get_logits(
        hidden,
        model.lm_head,
        None,
        embedding_bias=None,
        plan=None,
        require_full_vocab=False,
    )
    expected = F.linear(hidden, weights["lm_head.weight"])
    if softcap is not None:
        expected = (expected.float() / softcap).tanh() * softcap
    assert actual.dtype == expected.dtype
    # On SM100 to SM103 the one-pass softcap kernel: the statement's bytes.
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("softcap", [None, 30.0])
def test_logits_processor_fuses_argmax_only_without_softcap(softcap):
    # Greedy sampling asks the target head to fuse its argmax; the soft cap
    # runs after the shared processor, which keeps it out of a capped head.
    config = tiny_config()
    config.final_logit_softcapping = softcap
    processor = BeamLogitsProcessor(config, Mapping(rank=0, world_size=1))
    processor.fuse_argmax()
    assert processor.do_argmax is (softcap is None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
