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
    process_group = object()
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
    resolve = mock.Mock(return_value=process_group)
    monkeypatch.setitem(
        sys.modules,
        "tokenspeed_kernel.ops.communication.iris",
        SimpleNamespace(iris_kimi3_moe_tail=candidate),
    )
    monkeypatch.setattr(mod, "current_platform", lambda: SimpleNamespace(is_cdna4=True))
    monkeypatch.setattr(mod, "all_reduce", joined)
    monkeypatch.setattr(mod, "_get_process_group", resolve)
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

    assert fork.calls == [{"enable": graph_phase, "overlap": capture_mode}]
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
            group=process_group,
        )
        resolve.assert_called_once_with(group)
    else:
        candidate.assert_not_called()
        resolve.assert_not_called()
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
    monkeypatch.setitem(
        sys.modules,
        "tokenspeed_kernel.ops.communication.iris",
        SimpleNamespace(iris_kimi3_moe_tail=candidate),
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
