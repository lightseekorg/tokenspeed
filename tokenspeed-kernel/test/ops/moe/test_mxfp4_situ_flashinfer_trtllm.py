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
apply -- against ``a16w4_mxfp4_moe_reference`` on the same MXFP4 weights.
"""

from __future__ import annotations

from importlib.util import find_spec

import pytest
import torch
from kimi3_reference import a16w4_mxfp4_moe_reference
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
    if find_spec("flashinfer") is None:
        return "requires flashinfer"
    from tokenspeed_kernel.ops.moe.flashinfer.trtllm_mxfp4 import (
        situ_moe_unavailable_reason,
    )

    return situ_moe_unavailable_reason()


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
    expected = a16w4_mxfp4_moe_reference(
        hidden_states,
        raw["w13_weight"],
        raw["w13_scale"],
        raw["w2_weight"],
        raw["w2_scale"],
        topk_ids,
        topk_weights,
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
    expected = a16w4_mxfp4_moe_reference(
        hidden_states,
        raw["w13_weight"],
        raw["w13_scale"],
        raw["w2_weight"],
        raw["w2_scale"],
        topk_ids,
        topk_weights,
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
    )
    assert plan["apply_kernel_name"] == "flashinfer_trtllm_mxfp4_situ_moe_apply"
    assert plan["support_routing"] is True
    assert plan["supports_precomputed_topk"] is True
