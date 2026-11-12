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
        decode_moe=None,
        iris_state=None,
        stream_fork=fork,
        _topk_ready=None,
        _fused_routing_scratch=None,
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
        _latent_input_projections=lambda hs, shared_out=None, **kwargs: None,
        packed_input_projection_weight=None,
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
            prefix_is_sharded=False,
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
            prefix_is_sharded=False,
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


def _make_amd_moe(fork, *, routed, shared, projection, norm, mapping):
    moe = _make_moe(fork, num_tokens=routed.shape[0])
    moe.mapping = mapping
    moe.execution_plan = SimpleNamespace(use_native=True)
    moe.comm = None
    moe.routed_hidden = routed.shape[1]
    moe.routed_expert_norm = norm
    moe.routed_expert_up_proj = projection
    moe.shared_experts = mock.Mock(return_value=shared)
    moe._routed_experts = mock.Mock(return_value=routed)
    moe._forward_amd = KimiLinearMoE._forward_amd.__get__(moe)
    moe._replicated_moe_experts = KimiLinearMoE._replicated_moe_experts.__get__(moe)
    return moe


@pytest.mark.parametrize(
    "rows,producer_direct,tp,ep,narrowed,solution,accepted,attempted,pp_size",
    [
        (1, True, 8, 1, False, "auto", True, False, 1),
        (32, True, 8, 1, False, "auto", True, False, 1),
        (39, True, 8, 1, False, "auto", True, False, 1),
        (40, True, 8, 1, False, "auto", True, True, 1),
        (41, True, 8, 1, False, "auto", False, True, 1),
        (48, True, 8, 1, False, "auto", True, True, 1),
        (64, True, 8, 1, False, "auto", True, True, 1),
        (504, True, 8, 1, False, "auto", True, True, 1),
        (512, True, 8, 1, False, "auto", True, True, 1),
        (848, True, 8, 1, False, "auto", True, True, 1),
        (8192, True, 8, 1, False, "auto", True, True, 1),
        (8200, True, 8, 1, False, "auto", True, False, 1),
        (8192, False, 8, 1, False, "auto", True, False, 1),
        (8192, True, 1, 8, False, "auto", True, False, 1),
        (8192, True, 8, 1, True, "auto", True, False, 1),
        (8192, True, 8, 1, False, "torch", True, False, 1),
        (8192, True, 8, 1, False, "auto", False, True, 1),
        (8192, True, 8, 1, False, "auto", True, False, 2),
    ],
)
@pytest.mark.parametrize("has_norm", [False, True])
@pytest.mark.parametrize(
    "graph_phase,capture_mode", [(False, False), (True, False), (True, True)]
)
def test_row_sharded_moe_tail_selection_and_fallback(
    monkeypatch,
    rows,
    producer_direct,
    tp,
    ep,
    narrowed,
    solution,
    accepted,
    attempted,
    pp_size,
    has_norm,
    graph_phase,
    capture_mode,
):
    from tokenspeed.runtime.models import kimi_k3 as mod

    routed = torch.empty((rows, 3584), dtype=torch.bfloat16, device="meta")
    shared = torch.empty((rows, 7168), dtype=torch.bfloat16, device="meta")
    prefix = torch.empty_like(shared)
    expected = torch.empty_like(shared)
    fallback = torch.empty_like(shared)
    group = tuple(range(8))
    norm = (
        mock.Mock(
            return_value=routed,
            weight=torch.empty(3584, dtype=torch.bfloat16, device="meta"),
            variance_epsilon=1e-5,
        )
        if has_norm
        else None
    )
    projection = SimpleNamespace(
        narrowed=narrowed,
        solution=solution,
        weight=torch.empty((7168, 3584), dtype=torch.bfloat16, device="meta"),
        forward_add3=mock.Mock(return_value=fallback),
    )
    fork = _SpyFork()
    moe = _make_amd_moe(
        fork,
        routed=routed,
        shared=shared,
        projection=projection,
        norm=norm,
        mapping=SimpleNamespace(
            pp_size=pp_size,
            attn=SimpleNamespace(dp_size=1, tp_size=8, tp_group=group),
            moe=SimpleNamespace(tp_size=tp, ep_size=ep, tp_ep_group=group),
        ),
    )

    def run_tail(*args, **kwargs):
        assert not fork.inside_scope and fork.events[-1] == "join"
        return expected if accepted else None

    def reduce_partials(partials, actual_group):
        assert not fork.inside_scope and actual_group == group
        return partials

    candidate = mock.Mock(side_effect=run_tail)
    joined = mock.Mock(side_effect=reduce_partials)
    moe.iris_state = SimpleNamespace(
        moe_tail=candidate,
        all_reduce=SimpleNamespace(owns_outputs=lambda tensors: producer_direct),
        reduce_moe=lambda tensors, shared_owned_columns=False: joined(tensors, group),
        can_moe_gather_tail=lambda *args: False,
        can_reduce_moe_topk_partials=lambda shapes: False,
        can_latent_input_gather=lambda *args: False,
    )
    monkeypatch.setattr(mod, "current_platform", lambda: SimpleNamespace(is_cdna4=True))
    monkeypatch.setattr(mod, "all_reduce", joined)
    monkeypatch.setattr(mod, "get_is_cuda_graph_phase", lambda: graph_phase)
    monkeypatch.setattr(mod, "get_is_capture_mode", lambda: capture_mode)
    monkeypatch.setattr(
        mod, "can_acquire_all_reduce_outputs", lambda *args: producer_direct
    )
    acquire = mock.Mock(return_value=(routed, shared))
    monkeypatch.setattr(mod, "acquire_all_reduce_outputs", acquire)
    monkeypatch.setattr(mod, "_amd_moe_join_lane", lambda *args: None)

    output = KimiLinearMoE.forward(
        moe, prefix, prefix, rows, rows, prefix_is_sharded=False
    )

    assert fork.calls == [
        {
            "enable": graph_phase and mod.MOE_ROUTING_STREAM_FORK,
            "overlap": capture_mode,
        }
    ]
    assert moe.experts._situ_output_buffer is (routed if producer_direct else None)
    moe.shared_experts.assert_called_once_with(
        prefix, down_out=shared if producer_direct else None
    )
    assert moe._routed_experts.call_args.kwargs["do_finalize"] is True
    if attempted:
        candidate.assert_called_once_with(
            routed,
            shared,
            prefix,
            projection.weight,
            prefix_is_sharded=False,
            norm_weight=norm.weight if has_norm else None,
            eps=norm.variance_epsilon if has_norm else None,
            shared_reduced=None,
        )
    else:
        candidate.assert_not_called()
    if attempted and accepted:
        assert output is expected
        joined.assert_not_called()
        projection.forward_add3.assert_not_called()
        if has_norm:
            norm.assert_not_called()
    else:
        assert output._base is fallback and output.shape == shared.shape
        joined.assert_called_once_with((routed, shared), group)
        projection.forward_add3.assert_called_once_with(routed, prefix, shared)
        if has_norm:
            norm.assert_called_once_with(routed)


def test_row_sharded_moe_tail_skips_iris_import_on_other_platform(monkeypatch):
    from tokenspeed.runtime.models import kimi_k3 as mod

    group = tuple(range(8))
    routed = torch.empty((512, 3584), dtype=torch.bfloat16, device="meta")
    shared = torch.empty((512, 7168), dtype=torch.bfloat16, device="meta")
    prefix = torch.empty_like(shared)
    fallback = torch.empty_like(shared)
    moe = _make_amd_moe(
        _SpyFork(),
        routed=routed,
        shared=shared,
        projection=SimpleNamespace(forward_add3=mock.Mock(return_value=fallback)),
        norm=None,
        mapping=SimpleNamespace(
            attn=SimpleNamespace(dp_size=1), moe=SimpleNamespace(tp_ep_group=group)
        ),
    )
    monkeypatch.setattr(
        mod, "current_platform", lambda: SimpleNamespace(is_cdna4=False)
    )
    monkeypatch.setitem(
        sys.modules, "tokenspeed_kernel.ops.communication.iris", SimpleNamespace()
    )
    monkeypatch.setattr(mod, "can_acquire_all_reduce_outputs", lambda *args: True)
    monkeypatch.setattr(
        mod, "acquire_all_reduce_outputs", lambda *args: (routed, shared)
    )
    monkeypatch.setattr(mod, "all_reduce", mock.Mock(return_value=(routed, shared)))
    monkeypatch.setattr(mod, "get_is_cuda_graph_phase", lambda: False)
    monkeypatch.setattr(mod, "get_is_capture_mode", lambda: False)

    result = KimiLinearMoE.forward(
        moe, prefix, prefix, 512, 512, prefix_is_sharded=False
    )
    assert result._base is fallback


@pytest.mark.parametrize(
    "producer_direct,accepted", [(True, True), (True, False), (False, False)]
)
def test_sharded_attention_residual_is_gathered_before_moe_fallback(
    monkeypatch, producer_direct, accepted
):
    from tokenspeed.runtime.models import kimi_k3 as mod

    group = tuple(range(8))
    routed = torch.empty((4096, 3584), dtype=torch.bfloat16, device="meta")
    shared = torch.empty((4096, 7168), dtype=torch.bfloat16, device="meta")
    shard = torch.empty((512, 7168), dtype=torch.bfloat16, device="meta")
    full = torch.empty_like(shared)
    expected = torch.empty_like(shared)
    fork = _SpyFork()

    def project(routed, prefix, shared):
        assert prefix is full
        fork.events.append("project")
        return expected

    moe = _make_amd_moe(
        fork,
        routed=routed,
        shared=shared,
        projection=SimpleNamespace(
            narrowed=False,
            solution="auto",
            weight=torch.empty((7168, 3584), dtype=torch.bfloat16, device="meta"),
            forward_add3=mock.Mock(side_effect=project),
        ),
        norm=None,
        mapping=SimpleNamespace(
            pp_size=1,
            attn=SimpleNamespace(dp_size=1, tp_size=8, tp_group=group),
            moe=SimpleNamespace(tp_size=8, ep_size=1, tp_ep_group=group),
        ),
    )

    def gather_prefix(*args, **kwargs):
        assert not fork.inside_scope
        fork.events.append("gather")
        return full

    def reduce_partials(*args, **kwargs):
        fork.events.append("reduce")
        return routed, shared

    candidate = mock.Mock(return_value=expected if accepted else None)
    gather = mock.Mock(side_effect=gather_prefix)
    joined = mock.Mock(side_effect=reduce_partials)
    moe.iris_state = SimpleNamespace(
        moe_tail=candidate,
        all_reduce=SimpleNamespace(owns_outputs=lambda tensors: producer_direct),
        reduce_moe=lambda tensors, shared_owned_columns=False: joined(tensors, group),
        can_moe_gather_tail=lambda *args: False,
        can_reduce_moe_topk_partials=lambda shapes: False,
        can_latent_input_gather=lambda *args: False,
    )
    monkeypatch.setattr(mod, "current_platform", lambda: SimpleNamespace(is_cdna4=True))
    monkeypatch.setattr(mod, "all_gather", gather)
    monkeypatch.setattr(mod, "all_reduce", joined)
    monkeypatch.setattr(mod, "_get_process_group", lambda _: "owner")
    monkeypatch.setattr(mod, "get_is_cuda_graph_phase", lambda: False)
    monkeypatch.setattr(mod, "get_is_capture_mode", lambda: False)
    monkeypatch.setattr(
        mod, "can_acquire_all_reduce_outputs", lambda *args: producer_direct
    )
    monkeypatch.setattr(
        mod, "acquire_all_reduce_outputs", lambda *args: (routed, shared)
    )
    monkeypatch.setattr(mod, "_amd_moe_join_lane", lambda *args: None)

    result = KimiLinearMoE.forward(moe, full, shard, 4096, 4096, prefix_is_sharded=True)

    if producer_direct:
        assert candidate.call_args.kwargs["prefix_is_sharded"] is True
        assert candidate.call_args.args[2] is shard
    else:
        candidate.assert_not_called()
    if accepted:
        assert result is expected
        gather.assert_not_called()
        joined.assert_not_called()
        moe.routed_expert_up_proj.forward_add3.assert_not_called()
    else:
        assert result._base is expected
        gather.assert_called_once_with(shard, group, dim=0, backend=None)
        moe.routed_expert_up_proj.forward_add3.assert_called_once_with(
            routed, full, shared
        )
        assert fork.events == ["fork", "join", "gather", "reduce", "project"]


@pytest.mark.parametrize("gathering", [False, True])
@pytest.mark.parametrize("publishes", [False, True])
@pytest.mark.parametrize(
    "rows,declined,accepted",
    [
        (4096, False, True),
        (8192, False, False),
        (4096, True, True),
        (2048, False, True),
    ],
)
def test_row_sharded_moe_inputs_feed_experts_and_tail(
    monkeypatch, rows, declined, accepted, publishes, gathering
):
    from tokenspeed.runtime.models import kimi_k3 as mod

    group = tuple(range(8))
    attempted = rows >= mod._IRIS_MOE_ROW_INPUT_MIN_TOKENS
    row_inputs = attempted and not declined
    routed = torch.empty((rows, 3584), dtype=torch.bfloat16, device="meta")
    shared = torch.empty((rows, 7168), dtype=torch.bfloat16, device="meta")
    shard = torch.empty((rows // 8, 7168), dtype=torch.bfloat16, device="meta")
    full = torch.empty_like(shared)
    expected = torch.empty_like(shared)
    fork = _SpyFork()
    moe = _make_amd_moe(
        fork,
        routed=routed,
        shared=shared,
        projection=SimpleNamespace(
            narrowed=False,
            solution="auto",
            weight=torch.empty((7168, 3584), dtype=torch.bfloat16, device="meta"),
            forward_add3=mock.Mock(return_value=expected),
        ),
        norm=None,
        mapping=SimpleNamespace(
            pp_size=1,
            attn=SimpleNamespace(dp_size=1, tp_size=8, tp_group=group),
            moe=SimpleNamespace(tp_size=8, ep_size=1, tp_ep_group=group),
        ),
    )
    moe.packed_input_projection_weight = torch.empty(
        (6016, 7168), dtype=torch.bfloat16, device="meta"
    )
    packed = (torch.empty((rows, 1792), dtype=torch.uint8, device="meta"), None)
    weights = torch.empty((rows, 16), device="meta")
    ids = torch.empty((rows, 16), dtype=torch.int32, device="meta")
    shared_input = torch.empty((rows, 768), dtype=torch.bfloat16, device="meta")
    if row_inputs:
        moe.routed_expert_down_proj = SimpleNamespace(narrowed=False)
        moe.shared_experts = SimpleNamespace(
            act_fn=SimpleNamespace(beta=4.0, linear_beta=25.0),
            down_proj=SimpleNamespace(weight=torch.empty(7168, 768, device="meta")),
        )
    else:
        moe.shared_experts = mock.Mock(
            return_value=shared, act_fn=SimpleNamespace(beta=4.0, linear_beta=25.0)
        )
        moe.routed_expert_down_proj = mock.Mock(
            return_value=(full, None), narrowed=False
        )

    def project_shared(rows_in, weight, *, out):
        assert rows_in is shared_input and out is shared
        fork.events.append("shared_down")
        return out

    def experts(rows_in, topk_output, *args, **kwargs):
        fork.events.append("experts")
        if row_inputs:
            assert rows_in is packed
            assert topk_output.topk_weights is weights
            assert topk_output.topk_ids is ids
        return routed

    def tail(*args, **kwargs):
        assert not fork.inside_scope
        fork.events.append("tail")
        return expected if accepted else None

    candidate = mock.Mock(side_effect=tail)

    def reduce_moe(tensors, shared_owned_columns=False):
        fork.events.append("reduce")
        return (routed, shared)

    def project_moe_shared(rows_in, weight, outputs):
        assert rows_in is shared_input and outputs == (routed, shared)
        assert weight is moe.shared_experts.down_proj.weight
        if record is not None:
            fork.events.append("shared_down+reduce")
        return record

    record = object() if publishes else None
    joined = mock.Mock(side_effect=reduce_moe)
    moe._routed_experts = mock.Mock(side_effect=experts)
    moe.iris_state = SimpleNamespace(
        moe_tail=candidate,
        project_moe_shared=mock.Mock(side_effect=project_moe_shared),
        join_moe_shared=mock.Mock(
            side_effect=lambda *args: fork.events.append("join_shared")
        ),
        join_gather=mock.Mock(),
        moe_row_sharded_inputs=mock.Mock(
            return_value=(packed, weights, ids, shared_input) if row_inputs else None
        ),
        all_reduce=SimpleNamespace(owns_outputs=lambda tensors: True),
        reduce_moe=joined,
        can_moe_gather_tail=lambda *args: False,
        can_reduce_moe_topk_partials=lambda shapes: False,
        can_latent_input_gather=lambda *args: False,
    )
    monkeypatch.setattr(mod, "current_platform", lambda: SimpleNamespace(is_cdna4=True))
    monkeypatch.setattr(mod, "kimi3_shared_down_projection", project_shared)
    monkeypatch.setattr(mod, "all_gather", mock.Mock(return_value=full))
    monkeypatch.setattr(mod, "get_is_cuda_graph_phase", lambda: False)
    monkeypatch.setattr(mod, "get_is_capture_mode", lambda: False)
    monkeypatch.setattr(mod, "can_acquire_all_reduce_outputs", lambda *args: True)
    monkeypatch.setattr(
        mod, "acquire_all_reduce_outputs", lambda *args: (routed, shared)
    )
    monkeypatch.setattr(mod, "_amd_moe_join_lane", lambda *args: None)

    pending = object() if gathering else None
    result = KimiLinearMoE.forward(
        moe, full, shard, rows, rows, prefix_is_sharded=True, attention_gather=pending
    )

    if not attempted:
        moe.iris_state.moe_row_sharded_inputs.assert_not_called()
        # Every row is read below: the attention gather joins first.
        if gathering:
            moe.iris_state.join_gather.assert_called_once_with(pending)
        else:
            moe.iris_state.join_gather.assert_not_called()
        assert fork.events == ["fork", "experts", "join", "tail"]
        return
    call = moe.iris_state.moe_row_sharded_inputs.call_args
    assert call.args == (full, moe.packed_input_projection_weight)
    # The row-sharded projection joins the gather itself after the owned rows.
    assert call.kwargs["attention_gather"] is pending
    moe.iris_state.join_gather.assert_not_called()
    assert call.kwargs["beta"] == 4.0 and call.kwargs["linear_beta"] == 25.0
    # The router allocates its outputs; the gather copies the (weights, ids) pair.
    route = call.kwargs["route"]
    routed_rows = torch.empty(rows // 8, 7168, device="meta")
    moe.topk = mock.Mock(return_value=mod.StandardTopKOutput(weights, ids, None))
    assert route(routed_rows, full) == (weights, ids)
    assert moe.topk.call_args.args == (routed_rows, full)
    assert "out" not in moe.topk.call_args.kwargs
    declined_tail = [] if accepted else (["join_shared"] if publishes else [])
    declined_tail += [] if accepted else ["reduce"]
    if row_inputs:
        # Experts overlap a publishing shared GEMM's reduce; other reduces must join it.
        shared_down = "shared_down+reduce" if publishes else "shared_down"
        assert fork.events == [shared_down, "experts", "tail", *declined_tail]
        assert candidate.call_args.kwargs["shared_reduced"] is record
        if publishes and not accepted:
            moe.iris_state.join_moe_shared.assert_called_once_with(record, shared)
    else:
        assert fork.events == ["fork", "experts", "join", "tail", *declined_tail[-1:]]
        moe.shared_experts.assert_called_once_with(full, down_out=shared)
        moe.iris_state.project_moe_shared.assert_not_called()
        assert candidate.call_args.kwargs["shared_reduced"] is None
    assert candidate.call_args.kwargs["prefix_is_sharded"] is True
    if accepted:
        assert result is expected
        joined.assert_not_called()
    else:
        joined.assert_called_once_with((routed, shared), shared_owned_columns=False)
        assert result._base is expected


@pytest.mark.parametrize(
    "rows,gather_tail_available,gathered",
    [(40, True, True), (48, True, True), (64, True, True), (48, False, False)]
    + [(72, True, False)],
)
def test_replicated_decode_rows_prefer_the_gather_tail(
    monkeypatch, rows, gather_tail_available, gathered
):
    from tokenspeed.runtime.models import kimi_k3 as mod

    group = tuple(range(8))
    routed = torch.empty((rows, 3584), dtype=torch.bfloat16, device="meta")
    shared = torch.empty((rows, 7168), dtype=torch.bfloat16, device="meta")
    prefix = torch.empty_like(shared)
    row_sharded, gathered_out = torch.empty_like(shared), torch.empty_like(shared)
    projection = SimpleNamespace(
        narrowed=False,
        solution="auto",
        weight=torch.empty((7168, 3584), dtype=torch.bfloat16, device="meta"),
        forward_add3=mock.Mock(side_effect=AssertionError("replicated projection")),
    )
    moe = _make_amd_moe(
        _SpyFork(),
        routed=routed,
        shared=shared,
        projection=projection,
        norm=None,
        mapping=SimpleNamespace(
            pp_size=1,
            attn=SimpleNamespace(dp_size=1, tp_size=8, tp_group=group),
            moe=SimpleNamespace(tp_size=8, ep_size=1, tp_ep_group=group),
        ),
    )
    can_gather = mock.Mock(
        side_effect=lambda routed_rows, *args: gather_tail_available
        and routed_rows.shape[0] <= 64
    )
    moe.iris_state = SimpleNamespace(
        moe_tail=mock.Mock(return_value=row_sharded),
        all_reduce=SimpleNamespace(owns_outputs=lambda tensors: True),
        reduce_moe=mock.Mock(return_value=(routed, shared)),
        can_moe_gather_tail=can_gather,
        moe_gather_tail=mock.Mock(return_value=gathered_out),
        can_reduce_moe_topk_partials=lambda shapes: False,
        can_latent_input_gather=lambda *args: False,
    )
    monkeypatch.setattr(mod, "current_platform", lambda: SimpleNamespace(is_cdna4=True))
    monkeypatch.setattr(mod, "get_is_cuda_graph_phase", lambda: True)
    monkeypatch.setattr(mod, "get_is_capture_mode", lambda: True)
    monkeypatch.setattr(mod, "can_acquire_all_reduce_outputs", lambda *args: True)
    monkeypatch.setattr(
        mod, "acquire_all_reduce_outputs", lambda *args: (routed, shared)
    )
    monkeypatch.setattr(mod, "_amd_moe_join_lane", lambda *args: None)

    output = KimiLinearMoE.forward(
        moe, prefix, prefix, rows, rows, prefix_is_sharded=False
    )

    # To 64 rows the gather tail replaces the row-sharded one, seeing the shapes first.
    if gathered:
        assert output is gathered_out
        moe.iris_state.moe_tail.assert_not_called()
        moe.iris_state.reduce_moe.assert_called_once_with(
            (routed, shared), shared_owned_columns=True
        )
    else:
        assert output is row_sharded
        moe.iris_state.moe_tail.assert_called_once()
        moe.iris_state.moe_gather_tail.assert_not_called()


@pytest.mark.parametrize(
    "rows,gathered", [(8, True), (32, True), (64, True), (48, False)]
)
def test_decode_top_k_slot_rows_reduce_in_the_moe_all_reduce(
    monkeypatch, rows, gathered
):
    """Where the MoE all-reduce sums the top-k slot rows, the experts return
    them unreduced: the tail decision reads the producer's routed output, the
    row-sharded tail never takes slot rows, and the reduction gets them."""
    from tokenspeed.runtime.models import kimi_k3 as mod

    group = tuple(range(8))
    routed = torch.empty((rows, 3584), dtype=torch.bfloat16, device="meta")
    slots = torch.empty((rows * 16, 3584), dtype=torch.bfloat16, device="meta")
    shared = torch.empty((rows, 7168), dtype=torch.bfloat16, device="meta")
    prefix = torch.empty_like(shared)
    gathered_out = torch.empty_like(shared)
    projection = SimpleNamespace(
        narrowed=False,
        solution="auto",
        weight=torch.empty((7168, 3584), dtype=torch.bfloat16, device="meta"),
        forward_add3=mock.Mock(return_value=prefix),
    )
    moe = _make_amd_moe(
        _SpyFork(),
        routed=routed,
        shared=shared,
        projection=projection,
        norm=None,
        mapping=SimpleNamespace(
            pp_size=1,
            attn=SimpleNamespace(dp_size=1, tp_size=8, tp_group=group),
            moe=SimpleNamespace(tp_size=8, ep_size=1, tp_ep_group=group),
        ),
    )
    moe._routed_experts = mock.Mock(return_value=slots)
    can_gather = mock.Mock(
        side_effect=lambda routed_rows, *args: gathered and routed_rows.shape[0] <= 64
    )
    moe.iris_state = SimpleNamespace(
        moe_tail=mock.Mock(side_effect=AssertionError("row-sharded tail")),
        all_reduce=SimpleNamespace(owns_outputs=lambda tensors: True),
        reduce_moe=mock.Mock(return_value=(routed, shared)),
        can_moe_gather_tail=can_gather,
        moe_gather_tail=mock.Mock(return_value=gathered_out),
        can_reduce_moe_topk_partials=lambda shapes: True,
        can_latent_input_gather=lambda *args: False,
    )
    monkeypatch.setattr(mod, "current_platform", lambda: SimpleNamespace(is_cdna4=True))
    monkeypatch.setattr(mod, "get_is_cuda_graph_phase", lambda: True)
    monkeypatch.setattr(mod, "get_is_capture_mode", lambda: True)
    monkeypatch.setattr(mod, "can_acquire_all_reduce_outputs", lambda *args: True)
    monkeypatch.setattr(
        mod, "acquire_all_reduce_outputs", lambda *args: (routed, shared)
    )
    monkeypatch.setattr(mod, "_amd_moe_join_lane", lambda *args: None)

    output = KimiLinearMoE.forward(
        moe, prefix, prefix, rows, rows, prefix_is_sharded=False
    )

    assert moe.experts._situ_defer_topk_reduce is True
    assert all(call.args[0] is routed for call in can_gather.call_args_list)
    moe.iris_state.reduce_moe.assert_called_once_with(
        (routed, shared), routed_partials=slots, shared_owned_columns=gathered
    )
    moe.iris_state.moe_tail.assert_not_called()
    if gathered:
        assert output is gathered_out
    else:
        projection.forward_add3.assert_called_once()


@pytest.mark.parametrize("rows", [32, 64])
def test_non_iris_moe_tail_materializes_a_sharded_residual(monkeypatch, rows):
    from tokenspeed.runtime.models import kimi_k3 as mod

    group = tuple(range(8))
    fork = _SpyFork()
    moe = _make_moe(fork, num_tokens=rows)
    moe.mapping = SimpleNamespace(
        attn=SimpleNamespace(dp_size=1, tp_size=8, tp_group=group),
        moe=SimpleNamespace(tp_size=8, ep_size=1, tp_ep_group=group),
    )
    shard = torch.empty((rows // 8, 4))
    full = torch.empty((rows, 4))

    def gather_prefix(*args, **kwargs):
        assert not fork.inside_scope and fork.events[-1] == "join"
        return full

    gather = mock.Mock(side_effect=gather_prefix)
    monkeypatch.setattr(mod, "all_gather", gather)
    monkeypatch.setattr(mod, "get_is_cuda_graph_phase", lambda: False)
    monkeypatch.setattr(mod, "get_is_capture_mode", lambda: False)
    moe.comm.up_proj_ag = mock.Mock(side_effect=moe.comm.up_proj_ag)
    moe.comm.up_proj_inject_ar = mock.Mock(side_effect=moe.comm.up_proj_inject_ar)

    result = KimiLinearMoE.forward(moe, full, shard, rows, rows, prefix_is_sharded=True)

    assert result.shape == full.shape
    gather.assert_called_once_with(shard, group, dim=0, backend=None)
    stage2 = moe.comm.up_proj_ag if rows <= 32 else moe.comm.up_proj_inject_ar
    assert stage2.call_count == 1 and stage2.call_args.args[2] is full


class _Gate:
    """The router gate: fixed FP32 logits and the selection bias."""

    def __init__(self, logits: torch.Tensor, bias: torch.Tensor) -> None:
        self.logits = logits
        self.e_score_correction_bias = bias

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="the fused K3 routing sorts for the gfx950 A8W4 MoE",
)
@pytest.mark.parametrize("rows", [4, 8, 64, 192])
def test_fused_routing_hands_its_sort_to_the_experts(monkeypatch, rows):
    from tokenspeed_kernel.ops.moe import kimi3_sorted_routing_scratch
    from tokenspeed_kernel.ops.moe._triton.kimi3_sigmoid_topk import (
        kimi3_sigmoid_bias_topk,
    )

    from tokenspeed.runtime.models import kimi_k3 as mod

    monkeypatch.setattr(mod, "_FUSED_ROUTING_MAX_ROWS", 192)
    generator = torch.Generator(device="cuda").manual_seed(rows)
    logits = torch.randn((rows, 896), generator=generator, device="cuda")
    bias = torch.randn((896,), generator=generator, device="cuda") * 0.01
    fallback = (torch.zeros(rows, 16), torch.zeros(rows, 16))
    moe = _make_moe(_SpyFork(), num_tokens=rows)
    moe.gate = _Gate(logits, bias)
    moe.config = SimpleNamespace(moe_renormalize=True)
    moe.routed_scaling_factor = 2.827
    moe.topk = mock.Mock(return_value=fallback, simulate_routing=False)
    moe._fused_routing_scratch = kimi3_sorted_routing_scratch(torch.device("cuda"))
    moe.experts._situ_sorted_routes = ("stale",)
    handed = []

    def routed_experts(routed_in, topk_output, *args, **kwargs):
        handed.append((topk_output, moe.experts._situ_sorted_routes))
        return routed_in

    moe._routed_experts = routed_experts
    with (
        mock.patch.object(mod, "get_is_cuda_graph_phase", return_value=False),
        mock.patch.object(mod, "get_is_capture_mode", return_value=False),
    ):
        KimiLinearMoE._replicated_moe_experts(
            moe,
            torch.zeros(rows, 4),
            None,
            mod.TopKOutputFormat.STANDARD,
            rows,
            rows,
        )
    ((topk_output, sorted_routes),) = handed
    if rows < 8:
        assert topk_output is fallback and sorted_routes is None
        return
    moe.topk.assert_not_called()
    weights, ids = kimi3_sigmoid_bias_topk(
        logits, bias, routed_scaling_factor=2.827, normalize_topk_weights=True
    )
    assert torch.equal(topk_output.topk_ids, ids)
    assert torch.equal(
        topk_output.topk_weights.view(torch.int32), weights.view(torch.int32)
    )
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = sorted_routes
    assert int(num_valid_ids[1]) == rows
    real = sorted_ids[: int(num_valid_ids[0])]
    assert int(((real >> 24) < 16).sum()) == rows * 16


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
