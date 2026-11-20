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

"""FlashInfer Kimi K3 MegaMoE for MXFP4 experts vs the K3 references.

Runs the registered plan on one rank: the weight preprocessor (checkpoint
layout rewritten in place) and the fused kernel, eagerly and from a CUDA graph.
"""

from __future__ import annotations

import tempfile
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from tokenspeed_kernel.numerics.torch.moe import mxfp4_situ_moe_experts
from tokenspeed_kernel.ops.moe import moe_apply, moe_plan, moe_process_weights
from utils import assert_no_cute_compile, make_mxfp4_moe_weights

NUM_EXPERTS = 896
TOP_K = 16
HIDDEN = 3584
INTERMEDIATE = 3072
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0
MAX_TOKENS = 256
# The portable reference keeps BF16 activations where the kernel quantizes them
# to MXFP8; the non-fused TRT-LLM kernel quantizes them the same way.
PORTABLE_TOLERANCE = 0.1
SAME_PRECISION_TOLERANCE = 5e-3


def _runtime_reason() -> str | None:
    if not torch.cuda.is_available():
        return "requires CUDA"
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        return "FlashInfer's Kimi K3 MegaMoE targets SM100/SM103"
    return None


_reason = _runtime_reason()
pytestmark = pytest.mark.skipif(_reason is not None, reason=str(_reason))


class _Experts(torch.nn.Module):
    """What the preprocessor and the kernel read from an MoE layer."""

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
        self._moe_backend_state: object | None = None


def _inputs(num_tokens: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden_states = torch.randn(
        num_tokens, HIDDEN, device="cuda", generator=generator
    ).bfloat16()
    scores = torch.rand(num_tokens, NUM_EXPERTS, device="cuda", generator=generator)
    topk_weights, topk_ids = torch.topk(scores, TOP_K, dim=-1, sorted=False)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return hidden_states, topk_ids.to(torch.int32), topk_weights.contiguous()


@pytest.fixture(scope="module")
def experts():
    with tempfile.TemporaryDirectory() as tmp:
        dist.init_process_group(
            "nccl",
            init_method=f"file://{tmp}/rendezvous",
            rank=0,
            world_size=1,
            timeout=timedelta(minutes=10),
            device_id=torch.cuda.current_device(),
        )
        try:
            generator = torch.Generator(device="cuda").manual_seed(20261120)
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
                process_group=dist.group.WORLD,
                persistent_max_num_tokens_per_gpu=MAX_TOKENS,
                fast_math=True,
                combine_order="rank",
                features=frozenset(),
                solution="mega_moe",
            )
            assert plan["apply_kernel_name"] == "flashinfer_mxfp4_situ_mega_moe_apply"
            w = _Experts({name: tensor.clone() for name, tensor in raw.items()})
            moe_process_weights(plan, w)
            yield plan, w, raw, _non_fused_experts(raw)
        finally:
            dist.destroy_process_group()


def _apply(plan, w, hidden_states, topk_ids, topk_weights, capacity):
    return moe_apply(
        plan,
        hidden_states,
        w,
        None,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_tokens_global=capacity,
        max_num_tokens_per_gpu=capacity,
        do_finalize=True,
    )


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = actual.float(), expected.float()
    return ((actual - expected).norm() / expected.norm()).item()


def _non_fused_experts(raw):
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


def _non_fused(w, hidden_states, topk_ids, topk_weights):
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


def _reference(raw, hidden_states, topk_ids, topk_weights):
    return mxfp4_situ_moe_experts(
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


@pytest.mark.parametrize(
    ("num_tokens", "capacity"), [(17, 128), (128, 128), (200, 256)]
)
def test_matches_the_references(experts, num_tokens, capacity) -> None:
    plan, w, raw, non_fused = experts
    hidden_states, topk_ids, topk_weights = _inputs(num_tokens, 7 + num_tokens)
    # Masked routes contribute nothing.
    topk_ids[0] = -1
    topk_ids[1, 0] = -1
    actual = _apply(plan, w, hidden_states, topk_ids, topk_weights, capacity).clone()
    assert actual.shape == (num_tokens, HIDDEN)
    expected = _reference(raw, hidden_states, topk_ids, topk_weights)
    assert _relative_error(actual, expected) < PORTABLE_TOLERANCE
    same_precision = _non_fused(non_fused, hidden_states, topk_ids, topk_weights)
    assert _relative_error(actual, same_precision) < SAME_PRECISION_TOLERANCE
    assert not actual[0].any()


def test_cuda_graph_replay_follows_its_inputs(experts) -> None:
    plan, w, _, non_fused = experts
    hidden_states, topk_ids, topk_weights = _inputs(64, 31)
    eager = _apply(plan, w, hidden_states, topk_ids, topk_weights, 128).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _apply(plan, w, hidden_states, topk_ids, topk_weights, 128)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, eager, atol=0, rtol=0)

    changed = _inputs(64, 32)
    for tensor, value in zip((hidden_states, topk_ids, topk_weights), changed):
        tensor.copy_(value)
    graph.replay()
    torch.cuda.synchronize()
    expected = _non_fused(non_fused, hidden_states, topk_ids, topk_weights)
    assert _relative_error(captured, expected) < SAME_PRECISION_TOLERANCE


def test_token_counts_of_one_capacity_share_one_compile(experts) -> None:
    plan, w, _, _ = experts
    _apply(plan, w, *_inputs(128, 41), 128)
    with assert_no_cute_compile():
        for num_tokens in (1, 5, 33, 80, 128):
            _apply(plan, w, *_inputs(num_tokens, 42 + num_tokens), 128)
    torch.cuda.synchronize()
