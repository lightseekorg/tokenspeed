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

"""K3 MoE preserves stream scheduling independently of stage-2 kernel selection.

CPU-only checks cover stream selection and tail-stage ordering around the join.
GPU correctness and concurrent collective progress require distributed tests.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.models.kimi_k3 import KimiLinearMoE

register_cuda_ci(est_time=5, suite="runtime-1gpu")


class _SpyFork:
    """Records the ``scope()`` arguments and behaves like an inactive fork."""

    def __init__(self) -> None:
        self.calls: list[dict[str, bool]] = []
        self.events: list[str] = []
        self.inside_scope = False
        self.inside_branch = False
        self._active = False

    @contextmanager
    def scope(self, *, enable: bool, overlap: bool = True):
        self.calls.append({"enable": enable, "overlap": overlap})
        self.events.append("fork")
        self.inside_scope = True
        try:
            yield self
        finally:
            self.inside_scope = False
            self.events.append("join")

    @contextmanager
    def branch(self):
        self.inside_branch = True
        try:
            yield
        finally:
            self.inside_branch = False


def _make_moe(fork: _SpyFork, *, num_tokens: int) -> SimpleNamespace:
    """Minimal stand-in exposing only what the fork path of forward() touches."""
    hidden = torch.zeros(num_tokens, 4)
    shard = torch.zeros(num_tokens, 2)

    def shared_rs(shared_partial):
        assert fork.inside_scope and fork.inside_branch
        assert num_tokens <= 32
        fork.events.append("shared_rs")
        return shard

    def routed_ar_fusion(routed_out, num_tokens):
        assert fork.inside_scope and not fork.inside_branch
        fork.events.append("routed_ar_fusion")
        return routed_out

    def up_proj_ag(routed_latent, shared_shard, prefix_sum):
        assert not fork.inside_scope
        assert num_tokens <= 32 and shared_shard is shard
        fork.events.append("up_proj_ag")
        return hidden

    def up_proj_inject_ar(routed_latent, shared_partial, prefix_sum):
        assert not fork.inside_scope
        assert num_tokens > 32 and shared_partial.shape == hidden.shape
        fork.events.append("up_proj_inject_ar")
        return hidden

    comm = SimpleNamespace(
        defer_finalize=False,
        shared_rs=shared_rs,
        routed_ar_fusion=routed_ar_fusion,
        up_proj_ag=up_proj_ag,
        up_proj_inject_ar=up_proj_inject_ar,
    )
    return SimpleNamespace(
        mapping=SimpleNamespace(attn=SimpleNamespace(dp_size=1)),
        execution_plan=SimpleNamespace(use_native=False),
        native_latent_moe=None,
        stream_fork=fork,
        _topk_ready=None,
        routed_hidden=4,
        comm=comm,
        # Stand in for TopKOutputFormat so the fake does not have to track the
        # enum; only is_standard() is consulted on this path.
        _routing_output_format=lambda ctx: SimpleNamespace(is_standard=lambda: True),
        gate=lambda hs: torch.zeros(num_tokens, 2),
        topk=lambda hs, logits, output_format=None: (
            torch.zeros(num_tokens, 1),
            torch.zeros(num_tokens, 1),
        ),
        # None keeps this on the separate per-module projections, which is the
        # composition whose fork structure these tests pin.
        _latent_input_projections=lambda hs, shared_out=None: None,
        shared_experts=lambda hs, down_out=None: hs,
        routed_expert_down_proj=lambda hs: (hs, None),
        experts=SimpleNamespace(_situ_output_buffer=None),
        _routed_experts=lambda *a, **k: hidden,
    )


def _run(*, graph_phase: bool, capture_mode: bool, num_tokens: int) -> dict[str, bool]:
    fork = _SpyFork()
    moe = _make_moe(fork, num_tokens=num_tokens)
    with (
        mock.patch(
            "tokenspeed.runtime.models.kimi_k3.get_is_cuda_graph_phase",
            return_value=graph_phase,
        ),
        mock.patch(
            "tokenspeed.runtime.models.kimi_k3.get_is_capture_mode",
            return_value=capture_mode,
        ),
    ):
        KimiLinearMoE.forward(
            moe,
            torch.zeros(num_tokens, 4),
            torch.zeros(num_tokens, 4),
            num_global_tokens=num_tokens,
            max_num_tokens_per_gpu=num_tokens,
        )
    assert len(fork.calls) == 1
    assert fork.events == (
        ["fork"]
        + (["shared_rs"] if num_tokens <= 32 else [])
        + ["routed_ar_fusion", "join"]
        + (["up_proj_ag"] if num_tokens <= 32 else ["up_proj_inject_ar"])
    )
    return fork.calls[0]


@pytest.mark.parametrize("num_tokens", [32, 33])
@pytest.mark.parametrize(
    "graph_phase,capture_mode", [(False, False), (True, False), (True, True)]
)
def test_moe_preserves_stream_scheduling_in_eager_warmup_and_capture(
    graph_phase, capture_mode, num_tokens
):
    call = _run(
        graph_phase=graph_phase, capture_mode=capture_mode, num_tokens=num_tokens
    )
    assert call == {"enable": graph_phase, "overlap": capture_mode}


@pytest.mark.parametrize("prequantized", [False, True])
def test_moe_passes_projection_payload_to_experts(prequantized):
    hidden = torch.zeros(8, 4)
    payload = (
        (torch.empty(8, 2, dtype=torch.uint8), torch.empty(8, 1))
        if prequantized
        else hidden
    )
    projection = mock.Mock(spec=["__call__"], return_value=(payload, None))
    moe = _make_moe(_SpyFork(), num_tokens=8)
    moe.routed_expert_down_proj = projection
    moe._routed_experts = mock.Mock(return_value=hidden)
    with (
        mock.patch(
            "tokenspeed.runtime.models.kimi_k3.get_is_cuda_graph_phase",
            return_value=False,
        ),
        mock.patch(
            "tokenspeed.runtime.models.kimi_k3.get_is_capture_mode", return_value=False
        ),
    ):
        KimiLinearMoE.forward(
            moe,
            hidden,
            hidden,
            num_global_tokens=8,
            max_num_tokens_per_gpu=8,
        )
    projection.assert_called_once_with(hidden)
    assert moe._routed_experts.call_args.args[0] is payload


@pytest.mark.parametrize(
    "weight_dtype,solution,enabled",
    [
        ("nvfp4", "flashinfer_trtllm", True),
        ("mxfp4", "flashinfer_trtllm", False),
        ("nvfp4", "flashinfer_cutlass", False),
    ],
)
def test_nvfp4_projection_setup_uses_processed_expert_scale(
    weight_dtype, solution, enabled
):
    experts = SimpleNamespace(plan={"weight_dtype": weight_dtype, "solution": solution})
    scale = torch.nn.Parameter(torch.tensor(128.0), requires_grad=False)

    def process_weights(module):
        module.w13_input_scale_quant = scale

    experts.process_weights_after_loading = mock.Mock(side_effect=process_weights)
    projection = mock.Mock(spec=["prepare_nvfp4_output"])
    moe = SimpleNamespace(experts=experts, routed_expert_down_proj=projection)
    KimiLinearMoE.process_weights_after_loading(moe, moe)
    if enabled:
        experts.process_weights_after_loading.assert_called_once_with(experts)
        projection.prepare_nvfp4_output.assert_called_once_with(scale)
    else:
        experts.process_weights_after_loading.assert_not_called()
        projection.prepare_nvfp4_output.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
