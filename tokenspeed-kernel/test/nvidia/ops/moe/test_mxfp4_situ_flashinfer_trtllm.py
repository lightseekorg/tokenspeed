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

"""FlashInfer TRTLLM-Gen hybrid-routing SiTU MoE vs the portable K3 reference.

Exercises the full in-repo chain -- the weight preprocessor (concatenated
[gate|up] loader layout -> shuffled TRTLLM [up|gate]) and the registered
apply -- against ``mxfp4_moe_reference`` on the same MXFP4 weights.
"""

from __future__ import annotations

import pytest
import torch
from kimi3_reference import mxfp4_moe_reference
from utils import make_mxfp4_moe_weights

NUM_EXPERTS = 8
TOP_K = 2
HIDDEN = 256  # multiple of 256: no hidden padding path
ISPP = 128  # multiple of 128: no intermediate padding path
SITU_BETA = 4.0  # K3 activation_situ_beta
SITU_LINEAR_BETA = 25.0  # K3 activation_situ_linear_beta
ROUTING_SCALE = 1.25


def _situ_runtime_reason() -> str | None:
    if not torch.cuda.is_available():
        return "requires CUDA"
    if not (10, 0) <= torch.cuda.get_device_capability() <= (10, 3):
        return "flashinfer TRTLLM-Gen SiTU targets the sm_100 family"
    return None


_reason = _situ_runtime_reason()
requires_flashinfer_situ = pytest.mark.skipif(_reason is not None, reason=str(_reason))


class _MoEWeights(torch.nn.Module):
    """Minimal module carrying what the preprocessor and apply consume."""

    def __init__(self, raw: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.w13_weight = torch.nn.Parameter(raw["w13_weight"], requires_grad=False)
        self.w13_weight_scale = torch.nn.Parameter(
            raw["w13_scale"], requires_grad=False
        )
        self.w2_weight = torch.nn.Parameter(raw["w2_weight"], requires_grad=False)
        self.w2_weight_scale = torch.nn.Parameter(raw["w2_scale"], requires_grad=False)
        self.w13_input_layout = "concatenated"
        self.num_experts = NUM_EXPERTS
        self.num_local_experts = NUM_EXPERTS
        self.top_k = TOP_K
        self.hidden_size = HIDDEN
        self.activation_situ_beta = SITU_BETA
        self.activation_situ_linear_beta = SITU_LINEAR_BETA


def _kernel_routing_case(seed: int, num_tokens: int):
    generator = torch.Generator().manual_seed(seed)
    raw = make_mxfp4_moe_weights(NUM_EXPERTS, HIDDEN, ISPP, generator, device="cpu")
    hidden_states = (
        torch.randn(num_tokens, HIDDEN, generator=generator) * 0.2
    ).bfloat16()
    router_logits = torch.randn(
        num_tokens, NUM_EXPERTS, generator=generator, dtype=torch.float32
    )
    correction_bias = (
        torch.randn(NUM_EXPERTS, generator=generator, dtype=torch.float32) * 0.1
    )
    scores = router_logits.sigmoid()
    topk_ids = torch.topk(
        scores + correction_bias.unsqueeze(0), TOP_K, dim=-1, sorted=False
    ).indices.to(torch.int32)
    topk_weights = scores.gather(1, topk_ids.long())
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights * ROUTING_SCALE
    return raw, hidden_states, router_logits, correction_bias, topk_ids, topk_weights


def _prepare_kernel_routing_weights(raw, correction_bias):
    from tokenspeed_kernel.ops.moe.flashinfer.trtllm_mxfp4 import (
        flashinfer_trtllm_mxfp4_situ_moe_weights,
    )

    w = _MoEWeights({k: v.clone() for k, v in raw.items()}).cuda()
    w.routing_config = {
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": ROUTING_SCALE,
        "normalize_topk_weights": True,
        "correction_bias": correction_bias.cuda(),
        "routing_method_type": 2,  # DeepSeekV3
    }
    flashinfer_trtllm_mxfp4_situ_moe_weights({}, w)
    return w


@requires_flashinfer_situ
def test_flashinfer_situ_kernel_routing_matches_portable_reference() -> None:
    from tokenspeed_kernel.ops.moe.flashinfer.trtllm_mxfp4 import (
        flashinfer_trtllm_mxfp4_situ_moe_apply,
    )

    num_tokens = 16
    raw, hidden_states, router_logits, bias, topk_ids, topk_weights = (
        _kernel_routing_case(20260825, num_tokens)
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
        situ_beta=SITU_BETA,
        situ_linear_beta=SITU_LINEAR_BETA,
    )

    w = _prepare_kernel_routing_weights(raw, bias)
    actual = flashinfer_trtllm_mxfp4_situ_moe_apply(
        {}, hidden_states.cuda(), w, router_logits.cuda()
    )

    torch.testing.assert_close(
        actual.cpu().float(), expected.float(), atol=8e-2, rtol=8e-2
    )


@requires_flashinfer_situ
def test_flashinfer_situ_precomputed_routing_matches_portable_reference() -> None:
    from tokenspeed_kernel.ops.moe.flashinfer.trtllm_mxfp4 import (
        flashinfer_trtllm_mxfp4_situ_moe_apply,
    )

    raw, hidden_states, router_logits, bias, topk_ids, topk_weights = (
        _kernel_routing_case(20260827, 16)
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
        situ_beta=SITU_BETA,
        situ_linear_beta=SITU_LINEAR_BETA,
    )

    w = _prepare_kernel_routing_weights(raw, bias)
    actual = flashinfer_trtllm_mxfp4_situ_moe_apply(
        {},
        hidden_states.cuda(),
        w,
        router_logits.cuda(),
        topk_weights=topk_weights.cuda(),
        topk_ids=topk_ids.cuda(),
    )

    torch.testing.assert_close(
        actual.cpu().float(), expected.float(), atol=8e-2, rtol=8e-2
    )


@requires_flashinfer_situ
def test_flashinfer_situ_kernel_routing_deferred_matches_finalized() -> None:
    from tokenspeed_kernel.ops.moe.flashinfer.trtllm_mxfp4 import (
        flashinfer_trtllm_mxfp4_situ_moe_apply,
    )

    num_tokens = 16
    raw, hidden_states, router_logits, bias, _, _ = _kernel_routing_case(
        20260826, num_tokens
    )
    hidden_states = hidden_states.cuda()
    router_logits = router_logits.cuda()
    w = _prepare_kernel_routing_weights(raw, bias)

    finalized = flashinfer_trtllm_mxfp4_situ_moe_apply(
        {}, hidden_states, w, router_logits
    )
    gemm2_out, expert_weights, expanded_idx = flashinfer_trtllm_mxfp4_situ_moe_apply(
        {}, hidden_states, w, router_logits, do_finalize=False
    )
    torch.cuda.synchronize()

    assert gemm2_out.dtype == torch.bfloat16
    assert expert_weights.dtype == torch.bfloat16
    assert expert_weights.shape == (num_tokens, TOP_K)
    assert expanded_idx.dtype == torch.int32
    assert expanded_idx.shape == (num_tokens * TOP_K,)
    assert int(expanded_idx.max()) < gemm2_out.shape[0]
    assert int(expanded_idx.min()) >= -1

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
    torch.testing.assert_close(manual, finalized, atol=1e-2, rtol=1e-2)


@requires_flashinfer_situ
@pytest.mark.parametrize("num_tokens", [0, 4, 1025])
@pytest.mark.parametrize("precomputed", [False, True])
def test_swiglu_deferred_finalize(num_tokens, precomputed, monkeypatch) -> None:
    import tokenspeed_kernel
    from tokenspeed_kernel.ops.moe.cuda import moe_finalize_fuse_shared
    from tokenspeed_kernel.platform import pdl_enabled

    raw, x, logits, _, ids, weights = _kernel_routing_case(31, num_tokens)
    w = _MoEWeights(raw).cuda()
    plan = tokenspeed_kernel.moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        ep_size=1,
        ispp=ISPP,
        internal_activation_dtype="input",
        solution="flashinfer_trtllm",
        hidden=None,
        swiglu_form="standard",
        activation_clamped=False,
        expert_id_repeats=False,
        fast_math=True,
    )
    assert plan["supports_deferred_finalize"]
    assert plan["topk_weights_dtype"] == torch.bfloat16
    tokenspeed_kernel.moe_process_weights(plan, w)
    x, logits = x.cuda(), logits.cuda()
    routes = (
        dict(topk_weights=weights.cuda(), topk_ids=ids.cuda()) if precomputed else {}
    )
    from tokenspeed_kernel.ops.moe.flashinfer import trtllm_mxfp4

    # Compare CuTe and CUDA quantization through the expert kernels.
    with monkeypatch.context() as context:
        context.setattr(trtllm_mxfp4, "is_cute_dsl_available", lambda: False)
        expected = tokenspeed_kernel.moe_apply(
            plan, x, w, logits, do_finalize=True, **routes
        )
    seen = []
    original_quantizer = trtllm_mxfp4.mxfp8_quantize

    def observe_quantizer(*args, **kwargs):
        seen.append(kwargs["backend"])
        return original_quantizer(*args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(trtllm_mxfp4, "mxfp8_quantize", observe_quantizer)
        actual_finalized = tokenspeed_kernel.moe_apply(
            plan, x, w, logits, do_finalize=True, **routes
        )
    expected_backend = "cute-dsl" if trtllm_mxfp4.is_cute_dsl_available() else "cuda"
    assert seen == ([expected_backend] if num_tokens else [])
    torch.testing.assert_close(actual_finalized, expected, rtol=0, atol=0)
    if precomputed:
        bf16_routes = dict(
            topk_weights=routes["topk_weights"].bfloat16(), topk_ids=routes["topk_ids"]
        )
        bf16_result = tokenspeed_kernel.moe_apply(
            plan, x, w, logits, do_finalize=True, **bf16_routes
        )
        torch.testing.assert_close(bf16_result, expected, rtol=0, atol=0)
    gemm, weights, idx = tokenspeed_kernel.moe_apply(
        plan, x, w, logits, do_finalize=False, **routes
    )
    assert weights.shape == (num_tokens, TOP_K)
    assert weights.dtype == torch.bfloat16
    assert idx.shape == (num_tokens * TOP_K,)
    shared = torch.randn_like(expected)

    def finalize_reference(expert_output, row_indices, route_weights):
        acc = torch.zeros(num_tokens, HIDDEN, device="cuda", dtype=torch.float32)
        for k in range(TOP_K):
            rows = row_indices.view(num_tokens, TOP_K)[:, k].long()
            values = expert_output[rows.clamp_min(0)].double()
            weight = torch.where(rows >= 0, route_weights[:, k], 0).double()
            acc = (acc.double() + weight[:, None] * values).float()
        return (acc + shared.float()).bfloat16()

    actual = moe_finalize_fuse_shared(
        gemm,
        idx,
        weights,
        shared,
        top_k=TOP_K,
        enable_pdl=pdl_enabled(),
        hidden_dim=HIDDEN,
    )
    torch.testing.assert_close(
        actual, finalize_reference(gemm, idx, weights), atol=0, rtol=0
    )
    routed_only = moe_finalize_fuse_shared(
        gemm,
        idx,
        weights,
        None,
        top_k=TOP_K,
        enable_pdl=pdl_enabled(),
        hidden_dim=HIDDEN,
    )
    torch.testing.assert_close(routed_only, expected, atol=2e-3, rtol=8e-3)
    if num_tokens:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = tokenspeed_kernel.moe_apply(
                plan, x, w, logits, do_finalize=True, **routes
            )
            gemm, route_weights, row_map = tokenspeed_kernel.moe_apply(
                plan, x, w, logits, do_finalize=False, **routes
            )
            captured_fused = moe_finalize_fuse_shared(
                gemm,
                row_map,
                route_weights,
                shared,
                top_k=TOP_K,
                enable_pdl=pdl_enabled(),
                hidden_dim=HIDDEN,
            )
        x.normal_()
        with monkeypatch.context() as context:
            context.setattr(trtllm_mxfp4, "is_cute_dsl_available", lambda: False)
            expected_replay = tokenspeed_kernel.moe_apply(
                plan, x, w, logits, do_finalize=True, **routes
            )
        graph.replay()
        torch.testing.assert_close(captured, expected_replay, atol=0, rtol=0)
        torch.testing.assert_close(
            captured_fused,
            finalize_reference(gemm, row_map, route_weights),
            atol=0,
            rtol=0,
        )


@requires_flashinfer_situ
def test_moe_plan_selects_mxfp4_situ_hybrid_routing() -> None:
    import tokenspeed_kernel

    plan = tokenspeed_kernel.moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="situ",
        ep_size=1,
        ispp=ISPP,
        internal_activation_dtype="fp8",
        solution="flashinfer_trtllm",
        hidden=None,
        swiglu_form=None,
        activation_clamped=False,
        expert_id_repeats=False,
        fast_math=True,
    )
    assert plan["apply_kernel_name"] == "flashinfer_trtllm_mxfp4_situ_moe_apply"
    assert plan["support_routing"] is True
    assert plan["supports_precomputed_topk"] is True


@requires_flashinfer_situ
def test_swiglu_quantizer_keeps_native_padding(monkeypatch) -> None:
    import tokenspeed_kernel
    from tokenspeed_kernel.ops.moe.flashinfer import trtllm_mxfp4

    raw, x, logits, _, ids, weights = _kernel_routing_case(37, 4)
    w = _MoEWeights(raw).cuda()
    plan = tokenspeed_kernel.moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        ep_size=1,
        ispp=ISPP,
        internal_activation_dtype="input",
        solution="flashinfer_trtllm",
        hidden=None,
        swiglu_form="standard",
        activation_clamped=False,
        expert_id_repeats=False,
        fast_math=True,
    )
    tokenspeed_kernel.moe_process_weights(plan, w)
    original_hidden = HIDDEN - 32
    w.hidden_size_original = original_hidden
    x = x[:, :original_hidden].cuda().contiguous()
    logits, ids, weights = logits.cuda(), ids.cuda(), weights.cuda()
    padded = torch.nn.functional.pad(x, (0, HIDDEN - original_hidden))
    expected = tokenspeed_kernel.moe_apply(
        plan,
        padded,
        w,
        logits,
        topk_weights=weights,
        topk_ids=ids,
        do_finalize=True,
    )
    seen = []
    original_quantizer = trtllm_mxfp4.mxfp8_quantize

    def observe_quantizer(*args, **kwargs):
        seen.append(kwargs["backend"])
        return original_quantizer(*args, **kwargs)

    monkeypatch.setattr(trtllm_mxfp4, "mxfp8_quantize", observe_quantizer)
    actual = tokenspeed_kernel.moe_apply(
        plan,
        x,
        w,
        logits,
        topk_weights=weights,
        topk_ids=ids,
        do_finalize=True,
    )
    assert seen == ["cuda"]
    assert actual.shape == (4, original_hidden)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
