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

"""flashinfer TRTLLM-Gen NVFP4 SiTU MoE (w4a4) vs a dequantized reference.

Exercises the full in-repo chain for Kimi-K3-NVFP4 -- the SiTU weight
preprocessor (concatenated [gate|up] loader layout -> shuffled TRTLLM
[up|gate], SiTU-specific GEMM1 output scales) and the registered routed apply
-- against an fp32 SiTU MoE over the exactly-dequantized NVFP4 weights, with
both w4a4 activation quantizations (input and GEMM1 output) modeled the way
flashinfer's own reference does. The remaining deviation is in-kernel
quantization detail, so the SiTU error is additionally anchored to the
production-validated SwiGLU/silu path on the same weights. Also asserts the
registry resolves the (nvfp4, situ, precomputed_topk) plan to this kernel.
"""

from __future__ import annotations

from importlib.util import find_spec
from types import SimpleNamespace

import pytest
import torch

NUM_EXPERTS = 16
TOP_K = 10  # scaled-down test shape (real Kimi-K3 num_experts_per_token=16)
HIDDEN = 256
ISPP = 192  # Kimi-K3 intermediate size per partition
SITU_BETA = 4.0  # K3 activation_situ_beta (gate branch)
SITU_LINEAR_BETA = 25.0  # K3 activation_situ_linear_beta (up branch)

_FP8_E4M3_MAX = 448.0
_FP4_E2M1_MAX = 6.0
# e2m1 nibble decode table (bit 3 = sign).
_E2M1_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_E2M1_VALUES = torch.tensor(_E2M1_LUT + [-v for v in _E2M1_LUT])


def _situ_runtime_reason() -> str | None:
    if not torch.cuda.is_available():
        return "requires CUDA"
    if not (10, 0) <= torch.cuda.get_device_capability() <= (10, 3):
        return "flashinfer TRTLLM-Gen SiTU targets the sm_100 family"
    if find_spec("flashinfer") is None:
        return "requires flashinfer"
    from tokenspeed_kernel.ops.moe.flashinfer.trtllm_mxfp4 import (
        situ_moe_unavailable_reason,
    )

    return situ_moe_unavailable_reason()


_reason = _situ_runtime_reason()
requires_flashinfer_situ = pytest.mark.skipif(_reason is not None, reason=str(_reason))


def _nvfp4_quantize(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-tensor-global + per-16-group NVFP4 quantization of a 2-D weight.

    Returns (packed fp4 [m, k/2] uint8, scale [m, k/16] float8_e4m3fn,
    weight_scale_2 scalar = 1/global_scale, i.e. the ModelOpt global dequant
    multiplier).
    """
    from flashinfer import fp4_quantize

    global_scale = (_FP8_E4M3_MAX * _FP4_E2M1_MAX / w.abs().amax().clamp(min=1e-12)).to(
        torch.float32
    )
    packed, sf = fp4_quantize(
        w.cuda().to(torch.bfloat16),
        global_scale.cuda(),
        is_sf_swizzled_layout=False,
    )
    m, k = w.shape
    sf = sf.reshape(-1)[: m * (k // 16)].view(m, k // 16)
    return packed, sf.view(torch.float8_e4m3fn), (1.0 / global_scale).cpu()


def _nvfp4_dequant(
    packed: torch.Tensor, sf: torch.Tensor, weight_scale_2: torch.Tensor
) -> torch.Tensor:
    """Exact fp32 dequantization of `_nvfp4_quantize` outputs."""
    lut = _E2M1_VALUES.to(packed.device)
    lo = lut[(packed & 0x0F).long()]
    hi = lut[(packed >> 4).long()]
    vals = torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], -1)
    scales = sf.float().repeat_interleave(16, dim=1)
    return vals * scales * weight_scale_2.to(packed.device).float()


class _MoEWeights(torch.nn.Module):
    """Minimal module carrying what the preprocessor and apply consume."""

    def __init__(self, raw: dict[str, torch.Tensor]) -> None:
        super().__init__()
        for name in (
            "w13_weight",
            "w13_weight_scale",
            "w2_weight",
            "w2_weight_scale",
            "w13_weight_scale_2",
            "w2_weight_scale_2",
            "w13_input_scale",
            "w2_input_scale",
        ):
            setattr(self, name, torch.nn.Parameter(raw[name], requires_grad=False))
        self.activation_situ_beta = SITU_BETA
        self.activation_situ_linear_beta = SITU_LINEAR_BETA
        self._spec = SimpleNamespace(
            num_experts=NUM_EXPERTS,
            num_local_experts=NUM_EXPERTS,
            top_k=TOP_K,
            ep_rank=0,
        )


def _make_nvfp4_moe_weights(generator: torch.Generator) -> dict[str, torch.Tensor]:
    """K3-style [gate|up]-concatenated NVFP4 expert weights (loader layout)."""
    w13, w13_scale, w13_s2 = [], [], []
    w2, w2_scale, w2_s2 = [], [], []
    for _ in range(NUM_EXPERTS):
        w13_bf16 = torch.randn(2 * ISPP, HIDDEN, generator=generator) * 0.5
        w2_bf16 = torch.randn(HIDDEN, ISPP, generator=generator) * 0.5
        packed, sf, s2 = _nvfp4_quantize(w13_bf16)
        w13.append(packed), w13_scale.append(sf), w13_s2.append(s2)
        packed, sf, s2 = _nvfp4_quantize(w2_bf16)
        w2.append(packed), w2_scale.append(sf), w2_s2.append(s2)
    return {
        "w13_weight": torch.stack(w13),
        "w13_weight_scale": torch.stack(w13_scale),
        # One global scale per expert covers both the gate and up halves --
        # the layout the SiTU kernel requires (and what the Kimi-K3-NVFP4
        # checkpoint ships: w1.weight_scale_2 == w3.weight_scale_2).
        "w13_weight_scale_2": torch.stack(w13_s2).reshape(NUM_EXPERTS),
        "w2_weight": torch.stack(w2),
        "w2_weight_scale": torch.stack(w2_scale),
        "w2_weight_scale_2": torch.stack(w2_s2).reshape(NUM_EXPERTS),
        # Kimi-K3-NVFP4 ships input_scale == 1.0 for every expert projection.
        "w13_input_scale": torch.ones(NUM_EXPERTS),
        "w2_input_scale": torch.ones(NUM_EXPERTS),
    }


def _situ(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    left = SITU_LINEAR_BETA * torch.tanh(up / SITU_LINEAR_BETA)
    right = SITU_BETA * torch.tanh(gate / SITU_BETA) * torch.sigmoid(gate)
    return left * right


def _quant_dequant_activation(t: torch.Tensor, global_quant_scale: float):
    """Model the kernel's NVFP4 activation quantization (w4a4) in fp32."""
    from flashinfer import fp4_quantize

    packed, sf = fp4_quantize(
        t.bfloat16().cuda(),
        torch.tensor(global_quant_scale, dtype=torch.float32, device="cuda"),
        is_sf_swizzled_layout=False,
    )
    m, k = t.shape
    sf = sf.reshape(-1)[: m * (k // 16)].view(m, k // 16).view(torch.float8_e4m3fn)
    return _nvfp4_dequant(packed, sf, torch.tensor(1.0 / global_quant_scale))


def _reference_moe(
    hidden_states: torch.Tensor,
    raw: dict[str, torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    situ: bool,
) -> torch.Tensor:
    """fp32 MoE over the exactly-dequantized NVFP4 weights (w4a4 modeled).

    Both activation quantizations of the fused kernel are modeled, as in
    flashinfer's own reference: the bf16 input is quant-dequanted with the
    checkpoint's w13 input scale, and the GEMM1 activation output with the
    GEMM2 input scale (Kimi-K3-NVFP4 ships both as 1.0).
    """
    x = _quant_dequant_activation(hidden_states.float(), 1.0)
    out = torch.zeros_like(x)
    for e in range(NUM_EXPERTS):
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
        gate, up = w13[:ISPP], w13[ISPP:]
        if situ:
            act = _situ(x @ gate.t(), x @ up.t())
        else:
            act = torch.nn.functional.silu(x @ gate.t()) * (x @ up.t())
        act = _quant_dequant_activation(act, 1.0)
        expert_out = act @ w2.t()
        weight = torch.where(topk_ids == e, topk_weights.float(), 0.0).sum(dim=-1)
        out += weight[:, None] * expert_out
    return out


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


@requires_flashinfer_situ
def test_flashinfer_nvfp4_situ_routed_moe_matches_dequant_reference() -> None:
    from tokenspeed_kernel.ops.moe.flashinfer.trtllm_nvfp4 import (
        flashinfer_trtllm_nvfp4_moe_weights,
        flashinfer_trtllm_nvfp4_routed_moe_apply,
        flashinfer_trtllm_nvfp4_situ_moe_weights,
        flashinfer_trtllm_nvfp4_situ_routed_moe_apply,
    )

    generator = torch.Generator().manual_seed(20260817)
    num_tokens = 16

    raw = _make_nvfp4_moe_weights(generator)
    hidden_states = (
        (torch.randn(num_tokens, HIDDEN, generator=generator) * 0.2).bfloat16().cuda()
    )
    topk_ids = (
        torch.stack(
            [
                torch.randperm(NUM_EXPERTS, generator=generator)[:TOP_K]
                for _ in range(num_tokens)
            ]
        )
        .to(dtype=torch.int32)
        .cuda()
    )
    topk_weights = (
        torch.rand(num_tokens, TOP_K, generator=generator)
        .softmax(dim=-1)
        .bfloat16()
        .cuda()
    )

    w = _MoEWeights({k: v.clone() for k, v in raw.items()}).cuda()
    flashinfer_trtllm_nvfp4_situ_moe_weights({}, w)
    # SiTU-specific GEMM1 output scales: gate dequant happens inside the
    # activation (output1_scale_gate_scalar), output1_scale_scalar carries
    # only the GEMM2-input requant factor (no up-half folding), and the
    # actual-domain K3 constants pass through unchanged.
    assert torch.allclose(
        w.g1_scale_c.data.cpu(),
        (1.0 / raw["w2_input_scale"]).float(),
    )
    assert torch.allclose(w.gemm1_alpha.data.cpu(), torch.full((NUM_EXPERTS,), 4.0))
    assert torch.allclose(w.gemm1_beta.data.cpu(), torch.full((NUM_EXPERTS,), 25.0))

    actual = flashinfer_trtllm_nvfp4_situ_routed_moe_apply(
        {},
        hidden_states,
        w,
        router_logits=None,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    torch.cuda.synchronize()
    expected = _reference_moe(hidden_states, raw, topk_ids, topk_weights, situ=True)
    situ_err = _rel_l2(actual, expected)

    # Caller-owned output buffer (K3 fused-AR lane): the kernel must write
    # the identical result in place (zero-copy join).
    w._situ_output_buffer = torch.empty(
        num_tokens, HIDDEN, dtype=torch.bfloat16, device="cuda"
    )
    buffered = flashinfer_trtllm_nvfp4_situ_routed_moe_apply(
        {},
        hidden_states,
        w,
        router_logits=None,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    torch.cuda.synchronize()
    assert buffered.data_ptr() == w._situ_output_buffer.data_ptr()
    assert torch.equal(buffered, actual)
    del w._situ_output_buffer

    # Anchor: the SwiGLU/silu variant of the same kernel on the same weights
    # is the production-validated path; SiTU must sit in the same in-kernel
    # quantization-detail envelope (measured ~0.03 for both).
    w_silu = _MoEWeights({k: v.clone() for k, v in raw.items()}).cuda()
    flashinfer_trtllm_nvfp4_moe_weights({}, w_silu)
    actual_silu = flashinfer_trtllm_nvfp4_routed_moe_apply(
        {},
        hidden_states,
        w_silu,
        router_logits=None,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    torch.cuda.synchronize()
    silu_err = _rel_l2(
        actual_silu,
        _reference_moe(hidden_states, raw, topk_ids, topk_weights, situ=False),
    )

    assert situ_err < 0.06, f"{situ_err=:.4f}"
    assert situ_err < 2.0 * silu_err + 0.01, f"{situ_err=:.4f} vs {silu_err=:.4f}"


@requires_flashinfer_situ
@pytest.mark.parametrize("routing_mode", [None, "precomputed_topk"])
def test_moe_plan_selects_nvfp4_situ_routed_kernel(
    routing_mode: str | None,
) -> None:
    import tokenspeed_kernel

    plan = tokenspeed_kernel.moe_plan(
        "nvfp4",
        input_dtype=torch.bfloat16,
        activation="situ",
        routing_mode=routing_mode,
        ep_size=1,
        ispp=ISPP,
        internal_activation_dtype="input",
    )
    assert plan["apply_kernel_name"] == "flashinfer_trtllm_nvfp4_situ_routed_moe_apply"
    assert plan["support_routing"] is False
    preprocessor = plan["weight_preprocessor"]
    assert (
        getattr(preprocessor, "__name__", None)
        == "flashinfer_trtllm_nvfp4_situ_moe_weights"
    )


@requires_flashinfer_situ
def test_situ_preprocessor_rejects_mismatched_gate_up_global_scales() -> None:
    from tokenspeed_kernel.ops.moe.flashinfer.trtllm_nvfp4 import (
        flashinfer_trtllm_nvfp4_situ_moe_weights,
    )

    generator = torch.Generator().manual_seed(0)
    raw = _make_nvfp4_moe_weights(generator)
    # Separate per-half global scales that differ -> must fail fast: the
    # kernel dequantizes both halves with the single gate scale.
    ws2 = raw["w13_weight_scale_2"]
    raw["w13_weight_scale_2"] = torch.stack([ws2, ws2 * 2.0], dim=1)
    w = _MoEWeights(raw).cuda()
    with pytest.raises(RuntimeError, match="equal gate/up weight_scale_2"):
        flashinfer_trtllm_nvfp4_situ_moe_weights({}, w)


def test_registry_deferred_finalize_bit_matches_wiring() -> None:
    """K3's latent-tail arming reads this bit. It must only include True
    while the situ apply actually serves do_finalize=False (the deferred
    triple); a bit/behavior mismatch crashes the experts layer at the first
    MoE forward of a TAIL_FUSION deployment."""
    import tokenspeed_kernel.ops.moe.flashinfer.trtllm_nvfp4  # noqa: F401
    from tokenspeed_kernel.registry import KernelRegistry

    spec = KernelRegistry.get().get_by_name(
        "flashinfer_trtllm_nvfp4_situ_routed_moe_apply"
    )
    if spec is None:
        pytest.skip("nvfp4 SiTU kernel not registered on this platform")
    assert spec.traits["supports_deferred_finalize"] == frozenset({True, False})


@requires_flashinfer_situ
def test_nvfp4_situ_deferred_triple_matches_finalized() -> None:
    """do_finalize=False must hand back the exact raw materials of the
    finalized result: manually finalizing the (permuted rows, echoed bf16
    expert weights, expanded_idx) triple with the latent tail's recipe
    (fp32 ascending-k accumulate, single bf16 round) reproduces the
    do_finalize=True output."""
    from tokenspeed_kernel.ops.moe.flashinfer.trtllm_nvfp4 import (
        flashinfer_trtllm_nvfp4_situ_moe_weights,
        flashinfer_trtllm_nvfp4_situ_routed_moe_apply,
    )

    generator = torch.Generator().manual_seed(20260818)
    num_tokens = 16
    raw = _make_nvfp4_moe_weights(generator)
    hidden_states = (
        (torch.randn(num_tokens, HIDDEN, generator=generator) * 0.2).bfloat16().cuda()
    )
    topk_ids = (
        torch.stack(
            [
                torch.randperm(NUM_EXPERTS, generator=generator)[:TOP_K]
                for _ in range(num_tokens)
            ]
        )
        .to(dtype=torch.int32)
        .cuda()
    )
    topk_weights = (
        torch.rand(num_tokens, TOP_K, generator=generator)
        .softmax(dim=-1)
        .bfloat16()
        .cuda()
    )
    w = _MoEWeights({k: v.clone() for k, v in raw.items()}).cuda()
    flashinfer_trtllm_nvfp4_situ_moe_weights({}, w)

    finalized = flashinfer_trtllm_nvfp4_situ_routed_moe_apply(
        {},
        hidden_states,
        w,
        router_logits=None,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    gemm2_out, expert_weights, expanded_idx = (
        flashinfer_trtllm_nvfp4_situ_routed_moe_apply(
            {},
            hidden_states,
            w,
            router_logits=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            do_finalize=False,
        )
    )
    torch.cuda.synchronize()

    # Triple contract, as consumed by KimiK3LatentTailOp.call_deferred.
    assert gemm2_out.dtype == torch.bfloat16
    assert expert_weights.dtype == torch.bfloat16
    assert torch.equal(expert_weights, topk_weights)  # routed variant echoes
    assert expanded_idx.dtype == torch.int32
    assert expanded_idx.shape == (num_tokens * TOP_K,)
    assert int(expanded_idx.max()) < gemm2_out.shape[0]
    assert int(expanded_idx.min()) >= -1

    # The tail's finalize recipe: fp32 ascending-k accumulate per token,
    # one bf16 round at the end. -1 entries are padding and contribute 0.
    idx = expanded_idx.view(num_tokens, TOP_K).long()
    acc = torch.zeros(num_tokens, HIDDEN, dtype=torch.float32, device="cuda")
    for k in range(TOP_K):
        valid = idx[:, k] >= 0
        rows = gemm2_out[idx[:, k].clamp(min=0)].float()
        acc += torch.where(
            valid[:, None],
            expert_weights[:, k].float()[:, None] * rows,
            torch.zeros_like(rows),
        )
    manual = acc.to(torch.bfloat16)

    assert _rel_l2(manual, finalized) < 5e-3, f"{_rel_l2(manual, finalized)=}"
