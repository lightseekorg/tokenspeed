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

"""SM100 DeepGEMM BF16 fused router GEMM + sqrt-softplus top-k tests."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.moe import deep_gemm as deep_gemm_moe
from tokenspeed_kernel.ops.moe import moe_topk, set_mega_gate_workspace_region
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import NoKernelFoundError
from tokenspeed_kernel.thirdparty.cuda.mega_gate import (
    bf16_mega_gate,
    is_bf16_mega_gate_available,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform().is_nvidia
    or current_platform().arch_version.major != 10
    or not is_bf16_mega_gate_available(),
    reason="requires SM100-family GPU and deep_gemm.bf16_mega_gate",
)


def _reference_sqrt_softplus_topk(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    top_k: int,
    *,
    routed_scaling_factor: float,
    correction_bias: torch.Tensor | None = None,
    image_bias: torch.Tensor | None = None,
    image_token_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = hidden_states.float() @ gate_weight.float().T
    scores = F.softplus(logits).sqrt()
    selection_scores = scores
    if correction_bias is not None:
        bias = correction_bias.unsqueeze(0).expand(hidden_states.shape[0], -1)
        if image_bias is not None and image_token_mask is not None:
            bias = torch.where(
                image_token_mask.unsqueeze(-1),
                image_bias.unsqueeze(0).expand_as(bias),
                bias,
            )
        selection_scores = scores + bias
    elif image_bias is not None and image_token_mask is not None:
        bias = torch.where(
            image_token_mask.unsqueeze(-1),
            image_bias.unsqueeze(0).expand(hidden_states.shape[0], -1),
            torch.zeros(
                hidden_states.shape[0],
                gate_weight.shape[0],
                device=hidden_states.device,
                dtype=scores.dtype,
            ),
        )
        selection_scores = scores + bias
    topk_ids = torch.topk(selection_scores, k=top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, topk_ids)
    weights = (weights / weights.sum(dim=-1, keepdim=True)) * routed_scaling_factor
    return weights, topk_ids


def _canonicalize_routes(weights: torch.Tensor, ids: torch.Tensor):
    order = torch.argsort(ids, dim=-1)
    return weights.gather(1, order), ids.gather(1, order)


@pytest.mark.parametrize("tokens", [0, 1, 17, 128, 300])
@pytest.mark.parametrize("experts", [64, 256, 384])
@pytest.mark.parametrize("top_k", [1, 8])
@pytest.mark.parametrize("with_bias", [False, True])
def test_bf16_mega_gate_matches_fp32_reference(
    tokens: int,
    experts: int,
    top_k: int,
    with_bias: bool,
) -> None:
    torch.manual_seed(tokens + experts + top_k + int(with_bias))
    hidden = 1024
    # Construct structured rows with separated expert scores to verify exact ID
    # selection alongside FP32 weight accuracy across SM100 TensorCore GEMMs.
    gate_weight = (torch.randn(experts, hidden, device="cuda") * 0.1).bfloat16()
    gate_weight[:, :experts] += (
        torch.eye(experts, device="cuda", dtype=torch.bfloat16) * 1.5
    )
    hidden_states = (torch.randn(tokens, hidden, device="cuda") * 0.05).bfloat16()
    for row in range(tokens):
        primary = (row * 13) % experts
        hidden_states[row, primary] = 2.0
        if top_k > 1:
            for offset in range(1, top_k):
                hidden_states[row, (primary + offset * 7) % experts] = (
                    1.8 - offset * 0.1
                )
    bias = torch.randn(experts, device="cuda") * 0.02 if with_bias else None

    got_weights, got_ids = moe_topk(
        None,
        top_k,
        "sqrt_softplus",
        "topk",
        True,
        2.5,
        correction_bias=bias,
        hidden_states=hidden_states,
        gate_weight=gate_weight,
        topk_weights_dtype=torch.float32,
        topk_indices_dtype=torch.int64,
        override="deep_gemm_bf16_mega_gate",
    )
    ref_weights, ref_ids = _reference_sqrt_softplus_topk(
        hidden_states,
        gate_weight,
        top_k,
        routed_scaling_factor=2.5,
        correction_bias=bias,
    )
    got_weights, got_ids = _canonicalize_routes(got_weights, got_ids.to(torch.int64))
    ref_weights, ref_ids = _canonicalize_routes(ref_weights, ref_ids)
    torch.testing.assert_close(got_ids, ref_ids, atol=0, rtol=0)
    torch.testing.assert_close(got_weights, ref_weights, atol=2e-3, rtol=2e-3)


def test_bf16_mega_gate_image_bias_and_out_buffers() -> None:
    torch.manual_seed(7)
    tokens, hidden, experts, top_k = 12, 512, 32, 4
    gate_weight = torch.zeros(experts, hidden, device="cuda", dtype=torch.bfloat16)
    gate_weight[:, :experts] = torch.eye(experts, device="cuda", dtype=torch.bfloat16)
    hidden_states = torch.full(
        (tokens, hidden), 0.1, device="cuda", dtype=torch.bfloat16
    )
    text_bias = torch.arange(experts, device="cuda", dtype=torch.float32) * 0.25
    image_bias = (
        torch.arange(experts - 1, -1, -1, device="cuda", dtype=torch.float32) * 0.25
    )
    image_mask = torch.tensor(
        [i % 2 == 1 for i in range(tokens)], device="cuda", dtype=torch.bool
    )
    out_weights = torch.empty(tokens, top_k, device="cuda", dtype=torch.float32)
    out_ids = torch.empty(tokens, top_k, device="cuda", dtype=torch.int64)

    got_weights, got_ids = moe_topk(
        None,
        top_k,
        "sqrt_softplus",
        "topk",
        True,
        1.25,
        correction_bias=text_bias,
        hidden_states=hidden_states,
        gate_weight=gate_weight,
        image_bias=image_bias,
        image_token_mask=image_mask,
        out=(out_weights, out_ids),
        topk_weights_dtype=torch.float32,
        topk_indices_dtype=torch.int64,
        override="deep_gemm_bf16_mega_gate",
    )
    assert got_weights.data_ptr() == out_weights.data_ptr()
    assert got_ids.data_ptr() == out_ids.data_ptr()

    ref_weights, ref_ids = _reference_sqrt_softplus_topk(
        hidden_states,
        gate_weight,
        top_k,
        routed_scaling_factor=1.25,
        correction_bias=text_bias,
        image_bias=image_bias,
        image_token_mask=image_mask,
    )
    got_weights, got_ids = _canonicalize_routes(got_weights, got_ids)
    ref_weights, ref_ids = _canonicalize_routes(ref_weights, ref_ids)
    torch.testing.assert_close(got_ids, ref_ids, atol=0, rtol=0)
    torch.testing.assert_close(got_weights, ref_weights, atol=2e-3, rtol=2e-3)


def test_bf16_mega_gate_static_replica_placement() -> None:
    tokens, hidden, experts, top_k = 10, 512, 16, 2
    gate_weight = torch.zeros(experts, hidden, device="cuda", dtype=torch.bfloat16)
    gate_weight[:, :experts] = torch.eye(experts, device="cuda", dtype=torch.bfloat16)
    hidden_states = torch.zeros(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    # Every token prefers logical experts 0 and 1.
    hidden_states[:, 0] = 4.0
    hidden_states[:, 1] = 3.0
    to_physical = torch.full((experts, 4), -1, device="cuda", dtype=torch.int32)
    logical_count = torch.ones(experts, device="cuda", dtype=torch.int32)
    for expert in range(experts):
        to_physical[expert, 0] = expert
    # Replicate logical experts 0 and 1 across physical slots >= experts.
    to_physical[0, :3] = torch.tensor([0, 16, 17], device="cuda", dtype=torch.int32)
    logical_count[0] = 3
    to_physical[1, :2] = torch.tensor([1, 18], device="cuda", dtype=torch.int32)
    logical_count[1] = 2

    for ep_rank in (0, 2):
        weights, ids = moe_topk(
            None,
            top_k,
            "sqrt_softplus",
            "topk",
            True,
            1.0,
            hidden_states=hidden_states,
            gate_weight=gate_weight,
            logical_to_physical_map=to_physical,
            logical_replica_counts=logical_count,
            ep_rank=ep_rank,
            topk_weights_dtype=torch.float32,
            topk_indices_dtype=torch.int64,
            override="deep_gemm_bf16_mega_gate",
        )
        expected = torch.empty(tokens, top_k, device="cuda", dtype=torch.int64)
        for token_idx in range(tokens):
            expected[token_idx, 0] = to_physical[0, (ep_rank + token_idx * 23333) % 3]
            expected[token_idx, 1] = to_physical[1, (ep_rank + token_idx * 23333) % 2]
        assert weights.shape == (tokens, top_k)
        torch.testing.assert_close(ids, expected, atol=0, rtol=0)


def test_bf16_mega_gate_validation_and_registry_guards() -> None:
    hidden = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(NoKernelFoundError):
        moe_topk(
            None,
            4,
            "sqrt_softplus",
            "topk",
            False,
            1.0,
            hidden_states=hidden,
            gate_weight=weight,
            override="deep_gemm_bf16_mega_gate",
        )
    with pytest.raises(ValueError, match="normalized"):
        bf16_mega_gate(
            hidden,
            weight,
            4,
            renormalize=False,
            routed_scaling_factor=1.0,
            ep_rank=0,
        )
    with pytest.raises(ValueError, match="INT64"):
        bf16_mega_gate(
            hidden,
            weight,
            4,
            renormalize=True,
            routed_scaling_factor=1.0,
            ep_rank=0,
            out=(
                torch.empty(4, 4, device="cuda", dtype=torch.float32),
                torch.empty(4, 4, device="cuda", dtype=torch.int32),
            ),
        )
    with pytest.raises(NoKernelFoundError):
        moe_topk(
            None,
            4,
            "sqrt_softplus",
            "topk",
            True,
            1.0,
            hidden_states=hidden[:, :128].contiguous(),
            gate_weight=weight[:, :128].contiguous(),
            override="deep_gemm_bf16_mega_gate",
        )
    # Selection resolves to deep_gemm_bf16_mega_gate by default on SM100.
    w, idx = moe_topk(
        None,
        4,
        "sqrt_softplus",
        "topk",
        True,
        1.0,
        hidden_states=hidden,
        gate_weight=weight,
    )
    assert w.shape == (4, 4) and idx.shape == (4, 4)


def test_bf16_mega_gate_cuda_graph_replay_and_workspace_region() -> None:
    torch.manual_seed(11)
    tokens, hidden, experts, top_k = 16, 512, 64, 8
    weight = (torch.randn(experts, hidden, device="cuda") * 0.1).bfloat16()
    hidden_states = (torch.randn(tokens, hidden, device="cuda") * 0.1).bfloat16()
    bias = torch.randn(experts, device="cuda") * 0.05
    out_w = torch.empty(tokens, top_k, device="cuda", dtype=torch.float32)
    out_idx = torch.empty(tokens, top_k, device="cuda", dtype=torch.int64)

    entered: list[bool] = []
    from contextlib import contextmanager

    @contextmanager
    def _track_workspace():
        entered.append(True)
        yield

    set_mega_gate_workspace_region(_track_workspace)
    stream = torch.cuda.Stream()
    try:
        # First uncaptured call on a fresh stream invokes the workspace region;
        # later calls on the same stream do not re-enter it.
        with torch.cuda.stream(stream):
            moe_topk(
                None,
                top_k,
                "sqrt_softplus",
                "topk",
                True,
                1.0,
                correction_bias=bias,
                hidden_states=hidden_states,
                gate_weight=weight,
                out=(out_w, out_idx),
                topk_weights_dtype=torch.float32,
                topk_indices_dtype=torch.int64,
                override="deep_gemm_bf16_mega_gate",
            )
        assert len(entered) == 1

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            moe_topk(
                None,
                top_k,
                "sqrt_softplus",
                "topk",
                True,
                1.0,
                correction_bias=bias,
                hidden_states=hidden_states,
                gate_weight=weight,
                out=(out_w, out_idx),
                topk_weights_dtype=torch.float32,
                topk_indices_dtype=torch.int64,
                override="deep_gemm_bf16_mega_gate",
            )
        assert len(entered) == 1

        hidden_states.normal_()
        stream.synchronize()
        graph.replay()
        stream.synchronize()
        eager_w, eager_idx = bf16_mega_gate(
            hidden_states,
            weight,
            top_k,
            renormalize=True,
            routed_scaling_factor=1.0,
            ep_rank=0,
            correction_bias=bias,
        )
        torch.testing.assert_close(out_idx, eager_idx, atol=0, rtol=0)
        torch.testing.assert_close(out_w, eager_w, atol=0, rtol=0)
        assert deep_gemm_moe._HAS_MEGA_GATE
    finally:
        from contextlib import nullcontext

        set_mega_gate_workspace_region(nullcontext)
