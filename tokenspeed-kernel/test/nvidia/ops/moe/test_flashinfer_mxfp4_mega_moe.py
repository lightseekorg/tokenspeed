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

Runs the registered plan directly or under torchrun: the weight preprocessor,
direct input buffers and fused kernel, eagerly and from a CUDA graph.
"""

from __future__ import annotations

import os
import tempfile
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from tokenspeed_kernel.numerics.torch.moe import mxfp4_situ_moe_experts
from tokenspeed_kernel.ops.moe import (
    moe_apply,
    moe_plan,
    moe_prepare_input_buffers,
    moe_process_weights,
)
from utils import assert_no_cute_compile, make_mxfp4_moe_weights

TOP_K = 16
INTERMEDIATE = 3072
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0
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

    def __init__(self, raw: dict[str, torch.Tensor], num_experts: int) -> None:
        super().__init__()
        self.w13_weight = torch.nn.Parameter(raw["w13_weight"], requires_grad=False)
        self.w13_weight_scale = torch.nn.Parameter(
            raw["w13_scale"], requires_grad=False
        )
        self.w2_weight = torch.nn.Parameter(raw["w2_weight"], requires_grad=False)
        self.w2_weight_scale = torch.nn.Parameter(raw["w2_scale"], requires_grad=False)
        self.tp_size = 1
        self.w13_input_layout = "concatenated"
        self.num_experts = num_experts
        self.num_local_experts = raw["w13_weight"].shape[0]
        self.top_k = TOP_K
        self.hidden_size = raw["w2_weight"].shape[1]
        self.intermediate_size = raw["w2_weight"].shape[2] * 2
        self.activation_situ_beta = SITU_BETA
        self.activation_situ_linear_beta = SITU_LINEAR_BETA
        self._moe_backend_state: object | None = None


def _inputs(num_tokens: int, seed: int, w: _Experts):
    generator = torch.Generator(device="cuda").manual_seed(
        seed + 1000 * dist.get_rank()
    )
    hidden_states = torch.randn(
        num_tokens, w.hidden_size, device="cuda", generator=generator
    ).bfloat16()
    scores = torch.rand(num_tokens, w.num_experts, device="cuda", generator=generator)
    topk_weights, topk_ids = torch.topk(scores, w.top_k, dim=-1, sorted=False)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return hidden_states, topk_ids.to(torch.int32), topk_weights.contiguous()


@pytest.fixture(scope="module")
def process_group():
    with tempfile.TemporaryDirectory() as tmp:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group(
            "nccl",
            init_method="env://" if world_size > 1 else f"file://{tmp}/rendezvous",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=10),
            device_id=torch.cuda.current_device(),
        )
        try:
            yield dist.group.WORLD
        finally:
            dist.destroy_process_group()


@pytest.fixture(scope="module", params=[(896, 3584), (128, 3584)])
def experts(request, process_group):
    num_experts, hidden_size = request.param
    generator = torch.Generator(device="cuda").manual_seed(20261120)
    raw = make_mxfp4_moe_weights(
        num_experts, hidden_size, INTERMEDIATE, generator, scale_range=(123, 126)
    )
    plan = moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="situ",
        requires_deferred_finalize=False,
        routing_mode="precomputed_topk",
        a2a_backend="none",
        ep_size=process_group.size(),
        ispp=INTERMEDIATE,
        hidden=hidden_size,
        num_experts=num_experts,
        top_k=TOP_K,
        swiglu_form=None,
        activation_clamped=False,
        expert_id_repeats=False,
        internal_activation_dtype="fp8",
        process_group=process_group,
        fast_math=True,
        combine_order="rank",
        features=frozenset(),
        solution="mega_moe",
    )
    assert plan["apply_kernel_name"] == "flashinfer_mxfp4_situ_mega_moe_apply"
    count = num_experts // process_group.size()
    local = slice(process_group.rank() * count, (process_group.rank() + 1) * count)
    w = _Experts(
        {name: tensor[local].clone() for name, tensor in raw.items()}, num_experts
    )
    moe_process_weights(plan, w)
    return plan, w, raw, _non_fused_experts(raw)


def _apply(plan, w, hidden_states, topk_ids, topk_weights, capacity):
    buffers = moe_prepare_input_buffers(plan, w)
    _, _, ids, scores = buffers(hidden_states.shape[0], capacity)
    ids.copy_(topk_ids)
    scores.copy_(topk_weights)
    return moe_apply(
        plan,
        hidden_states,
        w,
        None,
        topk_weights=scores,
        topk_ids=ids,
        num_tokens_global=capacity * dist.get_world_size(),
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

    w = _Experts(
        {name: tensor.clone() for name, tensor in raw.items()},
        raw["w13_weight"].shape[0],
    )
    w.routing_config = {
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 1.0,
        "normalize_topk_weights": True,
        "correction_bias": torch.zeros(w.num_experts, device="cuda"),
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
        torch.zeros(hidden_states.shape[0], w.num_experts, device="cuda"),
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
    if (
        num_tokens == 17
        and dist.get_world_size() > 1
        and dist.get_rank() == dist.get_world_size() - 1
    ):
        num_tokens = 0
    hidden_states, topk_ids, topk_weights = _inputs(num_tokens, 7 + num_tokens, w)
    # Masked routes contribute nothing.
    if num_tokens:
        topk_ids[0] = -1
        topk_ids[1, 0] = -1
    actual = _apply(plan, w, hidden_states, topk_ids, topk_weights, capacity).clone()
    assert actual.shape == (num_tokens, w.hidden_size)
    if num_tokens == 0:
        return
    expected = _reference(raw, hidden_states, topk_ids, topk_weights)
    assert _relative_error(actual, expected) < PORTABLE_TOLERANCE
    same_precision = _non_fused(non_fused, hidden_states, topk_ids, topk_weights)
    assert _relative_error(actual, same_precision) < SAME_PRECISION_TOLERANCE
    assert not actual[0].any()


def test_cuda_graph_replay_follows_its_inputs(experts) -> None:
    plan, w, _, non_fused = experts
    hidden_states, topk_ids, topk_weights = _inputs(64, 31, w)
    eager = _apply(plan, w, hidden_states, topk_ids, topk_weights, 128).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _apply(plan, w, hidden_states, topk_ids, topk_weights, 128)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, eager, atol=0, rtol=0)

    changed = _inputs(64, 32, w)
    for tensor, value in zip((hidden_states, topk_ids, topk_weights), changed):
        tensor.copy_(value)
    graph.replay()
    torch.cuda.synchronize()
    expected = _non_fused(non_fused, hidden_states, topk_ids, topk_weights)
    assert _relative_error(captured, expected) < SAME_PRECISION_TOLERANCE


def test_token_counts_of_one_capacity_share_one_compile(experts) -> None:
    plan, w, _, _ = experts
    _apply(plan, w, *_inputs(128, 41, w), 128)
    with assert_no_cute_compile():
        for num_tokens in (1, 5, 33, 80, 128):
            _apply(plan, w, *_inputs(num_tokens, 42 + num_tokens, w), 128)
    torch.cuda.synchronize()
