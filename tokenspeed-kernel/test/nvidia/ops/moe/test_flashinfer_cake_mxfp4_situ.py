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

"""FlashInfer Cake MXFP4 SiTU routed experts vs the K3 references.

Runs the registered plan on one rank holding every expert: the weight
preprocessor (checkpoint layout rewritten in place) and the dense chain, at
token counts the chain serves directly and at counts it reaches by padding.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.numerics.torch.moe import mxfp4_situ_moe_experts
from tokenspeed_kernel.ops.moe import moe_apply, moe_plan, moe_process_weights
from tokenspeed_kernel.ops.tuning import kernel_warmup
from tokenspeed_kernel.thirdparty.flashinfer import cake_mxfp4_moe
from utils import make_mxfp4_moe_weights

NUM_EXPERTS = 896
TOP_K = 16
HIDDEN = 3584
INTERMEDIATE = 3072
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0
# The portable reference keeps BF16 activations where the kernel quantizes them
# to MXFP8; the TRT-LLM kernel quantizes them the same way.
PORTABLE_TOLERANCE = 0.1
# The Cake chain rounds each expert's output to BF16 and adds the top-k of them
# in BF16, in no fixed order; TRT-LLM sums them in FP32 and rounds once. That
# is about one BF16 ulp per element (6e-3 of the norm here), and 2e-3 between
# two runs of the same inputs.
SAME_PRECISION_TOLERANCE = 8e-3
REPEAT_TOLERANCE = 3e-3


def _runtime_reason() -> str | None:
    if not torch.cuda.is_available():
        return "requires CUDA"
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        return "FlashInfer's Cake MXFP4 MoE targets SM100/SM103"
    if not cake_mxfp4_moe.available():
        return "FlashInfer without the Cake MXFP4 MoE kernels"
    return None


_reason = _runtime_reason()
pytestmark = pytest.mark.skipif(_reason is not None, reason=str(_reason))


class _Experts(torch.nn.Module):
    """What the preprocessors and the kernels read from an MoE layer."""

    def __init__(self, raw: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.w13_weight = torch.nn.Parameter(raw["w13_weight"], requires_grad=False)
        self.w13_weight_scale = torch.nn.Parameter(
            raw["w13_scale"], requires_grad=False
        )
        self.w2_weight = torch.nn.Parameter(raw["w2_weight"], requires_grad=False)
        self.w2_weight_scale = torch.nn.Parameter(raw["w2_scale"], requires_grad=False)
        self.tp_size = 1
        self.w13_input_layout = "concatenated"
        self.num_experts = NUM_EXPERTS
        self.num_local_experts = NUM_EXPERTS
        self.top_k = TOP_K
        self.hidden_size = HIDDEN
        self.intermediate_size = INTERMEDIATE
        self.activation_situ_beta = SITU_BETA
        self.activation_situ_linear_beta = SITU_LINEAR_BETA


def _inputs(num_tokens: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden_states = torch.randn(
        num_tokens, HIDDEN, device="cuda", generator=generator
    ).bfloat16()
    scores = torch.rand(num_tokens, NUM_EXPERTS, device="cuda", generator=generator)
    topk_weights, topk_ids = torch.topk(scores, TOP_K, dim=-1, sorted=False)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return hidden_states, topk_ids.to(torch.int32), topk_weights.contiguous()


def _trtllm_experts(raw):
    from tokenspeed_kernel.ops.moe.flashinfer import (
        flashinfer_trtllm_mxfp4_situ_moe_weights,
    )

    w = _Experts({name: tensor.clone() for name, tensor in raw.items()})
    w.routing_config = {
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 1.0,
        "normalize_topk_weights": True,
        "correction_bias": torch.zeros(NUM_EXPERTS, device="cuda"),
        "routing_method_type": 2,  # DeepSeekV3
    }
    flashinfer_trtllm_mxfp4_situ_moe_weights({}, w)
    return w


@pytest.fixture(scope="module")
def experts():
    generator = torch.Generator(device="cuda").manual_seed(20261126)
    raw = make_mxfp4_moe_weights(
        NUM_EXPERTS, HIDDEN, INTERMEDIATE, generator, scale_range=(123, 126)
    )
    plan = moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="situ",
        requires_deferred_finalize=False,
        routing_mode="precomputed_topk",
        a2a_backend="none",
        ep_size=1,
        ispp=INTERMEDIATE,
        hidden=HIDDEN,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        swiglu_form=None,
        activation_clamped=False,
        expert_id_repeats=False,
        internal_activation_dtype="fp8",
        process_group=None,
        persistent_max_num_tokens_per_gpu=None,
        fast_math=True,
        combine_order="rank",
        features=frozenset(),
        solution="flashinfer_cake",
    )
    assert plan["apply_kernel_name"] == "flashinfer_cake_mxfp4_situ_routed_moe_apply"
    assert plan["supports_peer_weights"]
    w = _Experts({name: tensor.clone() for name, tensor in raw.items()})
    prepared = moe_process_weights(plan, w)
    return plan, w, raw, _trtllm_experts(raw), prepared


def _apply(plan, w, hidden_states, topk_ids, topk_weights):
    return moe_apply(
        plan,
        hidden_states,
        w,
        None,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_tokens_global=hidden_states.shape[0],
        max_num_tokens_per_gpu=hidden_states.shape[0],
        do_finalize=True,
    )


def _trtllm(w, hidden_states, topk_ids, topk_weights):
    from tokenspeed_kernel.ops.moe.flashinfer import (
        flashinfer_trtllm_mxfp4_situ_moe_apply,
    )

    return flashinfer_trtllm_mxfp4_situ_moe_apply(
        {},
        hidden_states,
        w,
        torch.zeros(hidden_states.shape[0], NUM_EXPERTS, device="cuda"),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = actual.float(), expected.float()
    return ((actual - expected).norm() / expected.norm()).item()


def test_weights_are_expert_major_for_the_peer_store(experts) -> None:
    _, w, _, _, prepared = experts
    assert prepared.layout == "flashinfer-cake-mxfp4-situ-dense-v1"
    assert [shard.name for shard in prepared.shards] == [
        "cake_w1",
        "cake_w1_sf",
        "cake_w2",
        "cake_w2_sf",
    ]
    for shard in prepared.shards:
        assert shard.tensor.shape[0] == NUM_EXPERTS
        assert shard.tensor.is_contiguous()
        assert not shard.replicate
        assert shard.tensor is w.get_parameter(shard.name)


# 2048 and 8192 are served directly (one and two tile sizes); 1, 300 and 1024
# pad to 1025, 7300 pads to 8192.
@pytest.mark.parametrize("num_tokens", [1, 300, 1024, 2048, 7300, 8192])
def test_matches_the_trtllm_kernel(experts, num_tokens) -> None:
    plan, w, _, trtllm, _ = experts
    hidden_states, topk_ids, topk_weights = _inputs(num_tokens, 7 + num_tokens)
    actual = _apply(plan, w, hidden_states, topk_ids, topk_weights)
    assert actual.shape == (num_tokens, HIDDEN)
    assert actual.dtype == torch.bfloat16
    expected = _trtllm(trtllm, hidden_states, topk_ids, topk_weights)
    assert _relative_error(actual, expected) < SAME_PRECISION_TOLERANCE


def test_matches_the_portable_reference(experts) -> None:
    plan, w, raw, _, _ = experts
    hidden_states, topk_ids, topk_weights = _inputs(1500, 99)
    actual = _apply(plan, w, hidden_states, topk_ids, topk_weights)
    expected = mxfp4_situ_moe_experts(
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
    assert _relative_error(actual, expected) < PORTABLE_TOLERANCE


def test_startup_forward_leaves_the_result_unchanged(experts) -> None:
    plan, w, _, _, _ = experts
    inputs = _inputs(1200, 5)
    expected = _apply(plan, w, *inputs).clone()
    with kernel_warmup(enabled=True):
        warm = _apply(plan, w, *inputs)
    assert _relative_error(warm, expected) < REPEAT_TOLERANCE


def test_no_tokens(experts) -> None:
    plan, w, _, _, _ = experts
    hidden_states, topk_ids, topk_weights = _inputs(0, 1)
    assert _apply(plan, w, hidden_states, topk_ids, topk_weights).shape == (0, HIDDEN)


def test_warmup_counts_cover_every_kernel_set() -> None:
    geometry = dict(
        device_index=torch.cuda.current_device(),
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        hidden=HIDDEN,
        intermediate=INTERMEDIATE,
        num_local_experts=NUM_EXPERTS,
        local_expert_offset=0,
    )
    max_tokens = 24576
    counts = cake_mxfp4_moe.warmup_token_counts(max_tokens, **geometry)
    warmed = {cake_mxfp4_moe.launch_forms(count, **geometry) for count in counts}
    assert None not in warmed
    for tokens in (1, 700, 1025, 5000, 7168, 7200, 8192, 12288, 16110, 20000, 24576):
        served = cake_mxfp4_moe.dense_tokens(tokens, **geometry)
        assert cake_mxfp4_moe.launch_forms(served, **geometry) in warmed
