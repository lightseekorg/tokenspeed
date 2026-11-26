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

"""Attention-DP MoE ownership and collective ordering."""

import os
import sys
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardMode
from tokenspeed.runtime.layers.moe.latent import Kimi3LatentProjection
from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput, TopKOutputFormat
from tokenspeed.runtime.layers.moe.utils import All2AllBackend, MoeBackend
from tokenspeed.runtime.models import kimi_k3
from tokenspeed.runtime.models.kimi_k3 import KimiLinearMoE


@pytest.mark.parametrize("dp,ep,world", [(4, 2, 4), (4, 4, 8)])
def test_attn_dp_rejects_partial_world_layout_before_backend_setup(
    monkeypatch, dp: int, ep: int, world: int
) -> None:
    backend = mock.Mock(side_effect=AssertionError("backend setup reached"))
    monkeypatch.setattr(kimi_k3, "get_moe_backend", backend)
    with pytest.raises(ValueError, match="attention DP == MoE EP == world size"):
        KimiLinearMoE(
            config=SimpleNamespace(),
            mapping=Mapping(
                rank=0, world_size=world, attn_dp_size=dp, moe_tp_size=1, moe_ep_size=ep
            ),
            layer_index=0,
            model_scope="test",
            moe_block_count=1,
            quant_config=None,
            prefix="moe",
            alt_stream=None,
        )
    backend.assert_not_called()


def test_non_mega_moe_rejects_hybrid_attn_dp_tp_layout(monkeypatch) -> None:
    monkeypatch.setattr(
        kimi_k3, "get_moe_backend", lambda: MoeBackend("flashinfer_trtllm")
    )
    monkeypatch.setattr(kimi_k3, "get_all2all_backend", lambda: All2AllBackend("none"))
    with pytest.raises(ValueError, match="attention DP == MoE EP == world size"):
        KimiLinearMoE(
            config=SimpleNamespace(),
            mapping=Mapping(
                rank=0, world_size=4, attn_dp_size=2, moe_tp_size=1, moe_ep_size=4
            ),
            layer_index=0,
            model_scope="test",
            moe_block_count=1,
            quant_config=None,
            prefix="moe",
            alt_stream=None,
        )


@pytest.mark.parametrize(
    "moe_backend,backend",
    [
        (moe, a2a)
        for moe in ("flashinfer_trtllm", "mega_moe")
        for a2a in ("none", "agrs", "flashinfer")
    ]
    + [("gluon_petit", a2a) for a2a in ("none", "agrs", "flashinfer", "gluon_petit")],
)
@pytest.mark.parametrize("fabric_available", [False, True])
@pytest.mark.parametrize("weights_dtype", [torch.float32, torch.bfloat16])
def test_attn_dp_replicates_dense_weights_and_selects_transport(
    monkeypatch,
    backend: str,
    fabric_available: bool,
    moe_backend: str,
    weights_dtype: torch.dtype,
) -> None:
    mega_moe = moe_backend in {"mega_moe", "gluon_petit"}
    expected_a2a = "gluon_petit" if moe_backend == "gluon_petit" else "none"

    class Experts(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.supports_precomputed_topk = True
            self.topk_output_format = TopKOutputFormat.STANDARD
            self.plan = {"a2a_backend": expected_a2a}
            self.topk_weights_dtype = weights_dtype
            self.sorted_route_rows = frozenset()

    monkeypatch.setattr(
        kimi_k3,
        "get_moe_backend",
        lambda: MoeBackend(moe_backend),
    )
    plan = kimi_k3.Kimi3MoEExecutionPlan(
        use_mega_moe=moe_backend == "mega_moe",
        owns_ep_communication=mega_moe,
        use_native=False,
        use_trtllm=not mega_moe,
        use_cutedsl=False,
        overlap_shared_experts=False,
        joint_moe_reduce=False,
        decode_chain=False,
    )
    forbidden = mock.Mock(side_effect=AssertionError("TP-only setup reached"))
    monkeypatch.setattr(
        kimi_k3.Kimi3MoEExecutionPlan, "build", mock.Mock(return_value=plan)
    )
    monkeypatch.setattr(kimi_k3.KimiK3LatentDownOp, "initialize", forbidden)
    monkeypatch.setattr(kimi_k3, "K3MoeTailComm", forbidden)
    monkeypatch.setattr(kimi_k3, "LatentMoELayer", forbidden)
    monkeypatch.setattr(kimi_k3, "MoELayer", Experts)
    transport = object() if fabric_available else None
    factory = mock.Mock(return_value=transport)
    monkeypatch.setattr(kimi_k3, "get_flashinfer_moe_alltoall", factory)
    monkeypatch.setattr(kimi_k3, "get_all2all_backend", lambda: All2AllBackend(backend))
    monkeypatch.setattr(
        "tokenspeed.runtime.distributed.process_group_manager.process_group_manager.get_device_process_group",
        mock.Mock(return_value=object()),
    )
    monkeypatch.setitem(kimi_k3.global_server_args_dict, "enforce_eager", False)
    monkeypatch.setitem(kimi_k3.global_server_args_dict, "max_prefill_tokens", 8192)
    monkeypatch.setitem(kimi_k3.global_server_args_dict, "max_num_seqs", 128)
    monkeypatch.setitem(
        kimi_k3.global_server_args_dict, "speculative_num_draft_tokens", 4
    )
    mapping = Mapping(
        rank=1, world_size=2, attn_dp_size=2, moe_tp_size=1, moe_ep_size=2
    )
    expected_error = (mega_moe and backend != expected_a2a) or (
        not mega_moe and backend == "flashinfer" and not fabric_available
    )
    with (
        pytest.raises(
            ValueError,
            match="owns dispatch/combine" if mega_moe else "sharing a CUDA fabric",
        )
        if expected_error
        else nullcontext()
    ):
        layer = KimiLinearMoE(
            config=KimiLinearConfig(
                hidden_size=64,
                routed_expert_hidden_size=32,
                moe_intermediate_size=32,
                num_experts=8,
                num_experts_per_token=2,
                num_shared_experts=1,
            ),
            mapping=mapping,
            layer_index=1,
            model_scope="test",
            moe_block_count=1,
            quant_config=None,
            prefix="moe",
            alt_stream=None,
        )
    if expected_error:
        return
    if mega_moe or backend == "agrs":
        factory.assert_not_called()
        assert layer.moe_alltoall is None
    else:
        factory.assert_called_once()
        assert layer.moe_alltoall is transport
    forbidden.assert_not_called()
    assert not layer._shard_latent_projections
    assert not layer.routed_expert_down_proj.narrowed
    assert not layer.routed_expert_up_proj.narrowed
    assert layer.routed_expert_down_proj.weight.shape == (32, 64)
    assert layer.routed_expert_up_proj.weight.shape == (64, 32)
    assert layer.shared_experts.gate_up_proj.weight.shape == (64, 64)
    assert layer.shared_experts.down_proj.weight.shape == (64, 32)
    assert layer.shared_experts.down_proj.tp_size == 1
    assert layer.shared_experts.down_proj.tp_group is None
    assert layer.experts.kwargs["routing_mode"] == "precomputed_topk"
    assert layer.topk.topk_config.topk_weights_dtype == weights_dtype
    assert layer.comm is None
    assert layer.native_latent_moe is None


@pytest.mark.parametrize(
    "backend,dp,match",
    [
        ("agrs", 1, "requires attention DP"),
        ("flashinfer", 1, "requires attention DP"),
    ],
)
def test_k3_rejects_unsupported_transport_layout(monkeypatch, backend, dp, match):
    monkeypatch.setattr(kimi_k3, "get_all2all_backend", lambda: All2AllBackend(backend))
    with pytest.raises(ValueError, match=match):
        KimiLinearMoE(
            config=SimpleNamespace(),
            mapping=SimpleNamespace(
                world_size=2,
                attn=SimpleNamespace(dp_size=dp, tp_size=2 // dp),
                moe=SimpleNamespace(ep_size=2, tp_ep_size=2),
            ),
            layer_index=0,
            model_scope="test",
            moe_block_count=1,
            quant_config=None,
            prefix="moe",
            alt_stream=None,
        )


@pytest.mark.parametrize("counts", [(1, 1), (0, 1), (1, 3)])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("weights_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_norm", [False, True])
@pytest.mark.parametrize("use_alltoall", [False, True])
@pytest.mark.parametrize("nvfp4", [False, True])
@pytest.mark.parametrize("shared_tp", [False, True])
@pytest.mark.parametrize(
    "graph_phase,capture_mode", [(False, False), (True, False), (True, True)]
)
def test_attn_dp_exchanges_latents_and_returns_reduced_local_rows(
    monkeypatch,
    counts: tuple[int, int],
    rank: int,
    weights_dtype: torch.dtype,
    with_norm: bool,
    use_alltoall: bool,
    nvfp4: bool,
    shared_tp: bool,
    graph_phase: bool,
    capture_mode: bool,
) -> None:
    rows, capacity = counts[rank], max(counts)
    events = []

    @contextmanager
    def scope(*, enable, overlap):
        assert enable == graph_phase
        assert overlap == capture_mode
        events.append("fork")
        yield fork
        events.append("join")

    @contextmanager
    def branch():
        events.append("branch")
        yield
        events.append("branch_done")

    @contextmanager
    def branch_after_main():
        events.append("branch_after_main")
        with branch():
            yield

    def shared(value, *, down_out):
        events.append("shared")
        return value * 3

    def shared_gather(value, physical_counts):
        events.append("shared_gather")
        assert physical_counts == list(counts)
        return value

    def shared_reduce(value, local_rows):
        events.append("shared_reduce")
        assert local_rows == rows
        return value

    fork = SimpleNamespace(
        scope=scope,
        branch=branch,
        branch_after_main=branch_after_main,
        join=lambda: events.append("join_branch"),
        record_checkpoint=lambda: events.append("record_gather"),
        join_checkpoint=lambda: events.append("join_gather"),
    )
    monkeypatch.setattr(kimi_k3, "get_is_cuda_graph_phase", lambda: graph_phase)
    monkeypatch.setattr(kimi_k3, "get_is_capture_mode", lambda: capture_mode)
    group = (0, 1)
    width = 32 if nvfp4 else 2
    latent = torch.zeros(2, capacity, width, dtype=torch.bfloat16)
    ids = torch.zeros(2, capacity, 2, dtype=torch.int32)
    weights = torch.zeros(2, capacity, 2, dtype=weights_dtype)
    for owner, count in enumerate(counts):
        latent[owner, :count] = torch.tensor([owner + 1, owner + 2]).repeat(width // 2)
        ids[owner, :count] = torch.tensor([1, 0], dtype=torch.int32)
        weights[owner, :count] = torch.tensor([0.271828, 0.728172], dtype=weights_dtype)
    hidden = torch.cat((latent[rank, :rows], latent[rank, :rows]), dim=-1)
    prefix = hidden + 10
    packed = torch.zeros(2, capacity, width // 2, dtype=torch.uint8)
    scales = torch.zeros(2, capacity, width // 16, dtype=torch.uint8)
    for owner, count in enumerate(counts):
        packed[owner, :count] = owner + 3
        scales[owner, :count] = owner + 7
    activation_payloads = (
        [("AG_packed", packed), ("AG_scales", scales)]
        if nvfp4
        else [("AG_latent", latent)]
    )
    gathers = iter([*activation_payloads, ("AG_ids", ids), ("AG_weights", weights)])
    quant_scale = torch.tensor(0.125)

    def quantize(tensor, scale, *, is_sf_swizzled_layout, enable_pdl, out=None):
        events.append("quantize")
        assert out is None
        assert scale is quant_scale and not is_sf_swizzled_layout
        torch.testing.assert_close(tensor, latent[rank, :rows])
        return packed[rank, :rows], scales[rank, :rows]

    quantizer = mock.Mock(side_effect=quantize)
    monkeypatch.setattr(kimi_k3, "fp4_quantize", quantizer)

    def gather(tensor, group, *, dim):
        name, expected = next(gathers)
        events.append(name)
        assert group == (0, 1)
        assert dim == 0
        torch.testing.assert_close(tensor, expected[rank])
        if rows == capacity:
            assert tensor.data_ptr() == expected[rank].data_ptr()
        return expected.reshape(2 * capacity, expected.shape[-1])

    def experts(
        routed,
        routing,
        *,
        num_global_tokens,
        max_num_tokens_per_gpu,
        do_finalize,
    ):
        events.append("experts")
        assert num_global_tokens == 2 * capacity
        assert max_num_tokens_per_gpu == capacity
        assert do_finalize
        assert routing.router_logits is None
        if nvfp4:
            torch.testing.assert_close(routed[0], packed.reshape(-1, width // 2))
            torch.testing.assert_close(routed[1], scales.reshape(-1, width // 16))
            routed = latent.reshape(-1, width)
        torch.testing.assert_close(routed, latent.reshape(-1, width))
        torch.testing.assert_close(routing.topk_ids, ids.reshape(-1, 2))
        torch.testing.assert_close(routing.topk_weights, weights.reshape(-1, 2))
        return routed * (rank + 1)

    def scatter(partial, *, group):
        events.append("RS")
        assert group == (0, 1)
        torch.testing.assert_close(partial, latent.reshape(-1, width) * (rank + 1))
        return latent[rank] * 3

    def dispatch(local_hidden, local_ids, local_weights, max_tokens):
        events.append("dispatch")
        assert max_tokens == capacity
        if nvfp4:
            torch.testing.assert_close(local_hidden[0], packed[rank, :rows])
            torch.testing.assert_close(local_hidden[1], scales[rank, :rows])
            received = (packed.reshape(-1, width // 2), scales.reshape(-1, width // 16))
        else:
            torch.testing.assert_close(local_hidden, latent[rank, :rows])
            received = latent.reshape(-1, width)
        torch.testing.assert_close(local_ids, ids[rank, :rows])
        torch.testing.assert_close(local_weights, weights[rank, :rows])
        return received, ids.reshape(-1, 2), weights.reshape(-1, 2)

    def combine(partial):
        events.append("combine")
        torch.testing.assert_close(partial, latent.reshape(-1, width) * (rank + 1))
        return latent[rank, :rows] * 3

    def norm(value):
        events.append("norm")
        torch.testing.assert_close(value, latent[rank, :rows] * 3)
        return torch.nn.functional.rms_norm(value.float(), (width,), eps=1e-6).to(
            value.dtype
        )

    def up(value, residual, shared, *, norm_weight, eps):
        events.append("up")
        assert norm_weight is None and eps is None
        torch.testing.assert_close(shared, hidden * 3)
        return residual + torch.cat((value, value), dim=-1) + shared

    topk = mock.Mock(
        return_value=StandardTopKOutput(weights[rank, :rows], ids[rank, :rows], None)
    )
    topk.topk_config = SimpleNamespace(
        topk_weights_dtype=weights_dtype, topk_indices_dtype=torch.int32
    )
    layer = SimpleNamespace(
        mapping=Mapping(
            rank=rank, world_size=2, attn_dp_size=2, moe_tp_size=1, moe_ep_size=2
        ),
        execution_plan=SimpleNamespace(use_mega_moe=False, owns_ep_communication=False),
        moe_alltoall=(
            SimpleNamespace(dispatch=dispatch, combine=combine)
            if use_alltoall
            else None
        ),
        stream_fork=fork,
        routed_hidden=width,
        top_k=2,
        topk=topk,
        expert_dispatch_info=None,
        experts=SimpleNamespace(
            plan={"weight_dtype": "nvfp4" if nvfp4 else "unquant"},
            w13_input_scale_quant=quant_scale,
        ),
        gate=mock.Mock(return_value=torch.empty(rows, 2)),
        routed_expert_down_proj=mock.Mock(return_value=(latent[rank, :rows], None)),
        shared_experts=mock.Mock(side_effect=shared),
        _routed_experts=experts,
        routed_expert_norm=norm if with_norm else None,
        routed_expert_up_proj=SimpleNamespace(forward_add3=up),
    )
    monkeypatch.setattr(kimi_k3, "all_gather", gather)
    layer.shared_experts.shared_parallel = object() if shared_tp else None
    layer.shared_experts.shared_communication = SimpleNamespace(
        gather_inputs=shared_gather, reduce_outputs=shared_reduce
    )
    monkeypatch.setattr(kimi_k3, "reduce_scatter", scatter)
    monkeypatch.setattr(
        kimi_k3, "all_reduce", mock.Mock(side_effect=AssertionError("all-reduce"))
    )
    ctx = SimpleNamespace(
        collective_global_num_tokens=list(counts),
        global_num_tokens=[99, 99],
        forward_mode=ForwardMode.EXTEND,
        attn_backend=None,
    )
    result = KimiLinearMoE._forward_attn_dp(layer, hidden, prefix, ctx)

    expected_latent = latent[rank, :rows] * 3
    if with_norm:
        expected_latent = torch.nn.functional.rms_norm(
            expected_latent.float(), (width,), eps=1e-6
        ).to(hidden.dtype)
    expected = (
        prefix + torch.cat((expected_latent, expected_latent), dim=-1) + hidden * 3
    )
    torch.testing.assert_close(result, expected)
    expected_events = (
        ["dispatch", "experts", "combine"]
        if use_alltoall
        else [name for name, _ in activation_payloads]
        + ["AG_ids", "AG_weights", "experts", "RS"]
    )
    if nvfp4 and rows:
        expected_events.insert(0, "quantize")
        quantizer.assert_called_once()
    else:
        quantizer.assert_not_called()
    if rows and with_norm:
        expected_events.append("norm")
    if shared_tp:
        expected_events.insert(1 if nvfp4 and rows else 0, "join_gather")
        before_bmm = expected_events.index("experts")
        expected_events[before_bmm:before_bmm] = [
            "join_branch",
            "branch_after_main",
            "branch",
            "shared_reduce",
            "branch_done",
        ]
        expected_events.insert(expected_events.index("experts") + 1, "join_branch")
        expected_events = [
            "fork",
            "branch",
            "shared_gather",
            "record_gather",
            "shared",
            "branch_done",
            *expected_events,
            "join",
        ]
    else:
        if rows:
            before_bmm = expected_events.index("experts")
            expected_events[before_bmm:before_bmm] = ["branch", "shared", "branch_done"]
        expected_events = ["fork", *expected_events, "join"]
    if rows:
        expected_events.append("up")
    assert events == expected_events
    if rows:
        layer.gate.assert_called_once_with(hidden)
        layer.routed_expert_down_proj.assert_called_once_with(hidden)
        layer.shared_experts.assert_called_once_with(hidden, down_out=None)
    else:
        layer.gate.assert_not_called()
        topk.assert_not_called()
        layer.routed_expert_down_proj.assert_not_called()
        if shared_tp:
            layer.shared_experts.assert_called_once_with(hidden, down_out=None)
        else:
            layer.shared_experts.assert_not_called()


def test_attn_dp_all_idle_skips_collectives(monkeypatch) -> None:
    forbidden = mock.Mock(side_effect=AssertionError("collective on all-idle batch"))
    monkeypatch.setattr(kimi_k3, "all_gather", forbidden)
    monkeypatch.setattr(kimi_k3, "reduce_scatter", forbidden)
    hidden = torch.empty(0, 4)
    layer = SimpleNamespace(
        mapping=Mapping(
            rank=0, world_size=2, attn_dp_size=2, moe_tp_size=1, moe_ep_size=2
        )
    )
    ctx = SimpleNamespace(
        collective_global_num_tokens=None,
        global_num_tokens=[0, 0],
        forward_mode=ForwardMode.IDLE,
        attn_backend=None,
    )
    result = KimiLinearMoE._forward_attn_dp(layer, hidden, hidden, ctx)
    assert result is hidden
    forbidden.assert_not_called()


def test_attn_dp_forward_bypasses_tp_tail() -> None:
    hidden = torch.ones(1, 4)
    ctx = SimpleNamespace()
    dp_forward = mock.Mock(return_value=hidden)
    layer = SimpleNamespace(
        mapping=Mapping(
            rank=0, world_size=2, attn_tp_size=1, moe_tp_size=1, moe_ep_size=2
        ),
        _forward_attn_dp=dp_forward,
    )
    result = KimiLinearMoE.forward(
        layer,
        hidden,
        hidden,
        num_global_tokens=2,
        max_num_tokens_per_gpu=1,
        ctx=ctx,
        prefix_is_sharded=False,
    )
    assert result is hidden
    dp_forward.assert_called_once_with(hidden, hidden, ctx)


@pytest.mark.parametrize(
    "counts,prefix_width", [(None, 4), ([1], 4), ([2, 1], 4), ([1, 1], 3)]
)
def test_attn_dp_rejects_invalid_token_metadata_before_collectives(
    monkeypatch, counts: list[int] | None, prefix_width: int
) -> None:
    forbidden = mock.Mock(
        side_effect=AssertionError("collective with invalid metadata")
    )
    monkeypatch.setattr(kimi_k3, "all_gather", forbidden)
    monkeypatch.setattr(kimi_k3, "reduce_scatter", forbidden)
    layer = SimpleNamespace(
        mapping=Mapping(
            rank=0, world_size=2, attn_dp_size=2, moe_tp_size=1, moe_ep_size=2
        )
    )
    ctx = SimpleNamespace(
        collective_global_num_tokens=counts,
        global_num_tokens=None,
        forward_mode=ForwardMode.EXTEND,
        attn_backend=None,
    )
    with pytest.raises(ValueError, match="matching collective token counts"):
        KimiLinearMoE._forward_attn_dp(
            layer, torch.ones(1, 4), torch.ones(1, prefix_width), ctx
        )
    forbidden.assert_not_called()


def test_attn_dp_forward_requires_context() -> None:
    layer = SimpleNamespace(
        mapping=Mapping(
            rank=0, world_size=2, attn_tp_size=1, moe_tp_size=1, moe_ep_size=2
        )
    )
    hidden = torch.ones(1, 4)
    with pytest.raises(ValueError, match="requires a ForwardContext"):
        KimiLinearMoE.forward(
            layer,
            hidden,
            hidden,
            num_global_tokens=2,
            max_num_tokens_per_gpu=1,
            ctx=None,
            prefix_is_sharded=False,
        )


@pytest.mark.parametrize("rows", [0, 2])
@pytest.mark.parametrize("shared_tp", [False, True])
@pytest.mark.parametrize("weight_dtype", ["nvfp4", "mxfp4"])
def test_attn_dp_fused_moe_keeps_inputs_local_and_owns_combine(
    monkeypatch, rows, shared_tp, weight_dtype
):
    hidden = torch.randn(rows, 8, dtype=torch.bfloat16)
    prefix = torch.randn_like(hidden)
    latent = hidden[:, :4].contiguous()
    ids = torch.zeros(rows, 2, dtype=torch.int32)
    weights = torch.full((rows, 2), 0.5, dtype=torch.bfloat16)
    payload = (
        torch.ones(rows, 2, dtype=torch.uint8),
        torch.ones(rows, 1, dtype=torch.uint8),
    )
    quantize = mock.Mock(return_value=payload)
    monkeypatch.setattr(kimi_k3, "fp4_quantize", quantize)
    for name in ("all_gather", "reduce_scatter", "all_reduce"):
        monkeypatch.setattr(kimi_k3, name, mock.Mock(side_effect=AssertionError(name)))

    events = []

    @contextmanager
    def scope(**kwargs):
        yield SimpleNamespace(
            branch=nullcontext,
            branch_after_main=nullcontext,
            record_checkpoint=lambda: events.append("record_gather"),
            join_checkpoint=lambda: events.append("join_gather"),
            join=lambda: events.append("join"),
        )

    routed = mock.Mock(
        side_effect=lambda *args, **kwargs: (events.append("experts"), latent)[1]
    )
    up = Kimi3LatentProjection(4, 8, params_dtype=torch.bfloat16, solution="torch")
    up.weight.data.normal_(std=0.1)
    layer = SimpleNamespace(
        execution_plan=SimpleNamespace(use_mega_moe=True, owns_ep_communication=True),
        mapping=Mapping(
            rank=0, world_size=2, attn_dp_size=2, moe_tp_size=1, moe_ep_size=2
        ),
        stream_fork=SimpleNamespace(scope=scope),
        topk=mock.Mock(return_value=StandardTopKOutput(weights, ids, None)),
        expert_dispatch_info=None,
        gate=mock.Mock(return_value=torch.zeros(rows, 2)),
        routed_expert_down_proj=mock.Mock(return_value=(latent, None)),
        experts=SimpleNamespace(
            plan={"weight_dtype": weight_dtype},
            w13_input_scale_quant=torch.ones(1),
            input_buffers=lambda num_tokens, max_tokens: (*payload, ids, weights),
        ),
        moe_alltoall=None,
        _routed_experts=routed,
        routed_expert_norm=None,
        routed_expert_up_proj=up,
        shared_experts=mock.Mock(return_value=hidden),
        routed_hidden=4,
        top_k=2,
    )
    layer.shared_experts.shared_parallel = object() if shared_tp else None
    layer.shared_experts.shared_communication = SimpleNamespace(
        gather_inputs=mock.Mock(
            side_effect=lambda *args: (events.append("gather"), hidden)[1]
        ),
        reduce_outputs=mock.Mock(
            side_effect=lambda *args: (events.append("reduce"), hidden)[1]
        ),
    )
    layer.topk.topk_config = SimpleNamespace(
        topk_weights_dtype=torch.bfloat16, topk_indices_dtype=torch.int32
    )
    ctx = SimpleNamespace(
        collective_global_num_tokens=[rows, 3],
        global_num_tokens=None,
        forward_mode=ForwardMode.EXTEND,
        attn_backend=None,
    )
    result = KimiLinearMoE._forward_attn_dp(layer, hidden, prefix, ctx)
    if shared_tp:
        assert events == [
            "gather",
            "record_gather",
            "join_gather",
            "join",
            "join",
            "experts",
            "reduce",
        ]
    else:
        assert events == ["join", "experts"]
    routed.assert_called_once()
    call = routed.call_args
    expert_input = call.args[0][0] if weight_dtype == "nvfp4" else call.args[0]
    assert expert_input.shape[0] == rows
    assert call.args[1].topk_ids.shape == (rows, 2)
    assert call.kwargs["max_num_tokens_per_gpu"] == 3
    assert call.kwargs["num_global_tokens"] == 6
    if weight_dtype == "mxfp4":
        quantize.assert_not_called()
    if rows:
        assert call.args[0] is (payload if weight_dtype == "nvfp4" else latent)
        torch.testing.assert_close(result, up.forward_add3(latent, prefix, hidden))
    else:
        assert result is prefix


@pytest.mark.parametrize(
    "tp,dp,pp",
    [(8, 1, 1), (16, 1, 1), (8, 2, 1), (4, 4, 1), (8, 1, 2)],
)
def test_mega_moe_accepts_tp_ep_and_dep_layouts_and_shards_tail(
    monkeypatch, tp: int, dp: int, pp: int
) -> None:
    ep = tp * dp
    world = ep * pp

    class Experts(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.supports_precomputed_topk = True
            self.topk_output_format = TopKOutputFormat.STANDARD
            self.plan = {"a2a_backend": "none", "weight_dtype": "mxfp4"}
            self.topk_weights_dtype = torch.float32
            self.sorted_route_rows = frozenset()

    monkeypatch.setattr(kimi_k3, "get_moe_backend", lambda: MoeBackend("mega_moe"))
    monkeypatch.setattr(kimi_k3, "get_all2all_backend", lambda: All2AllBackend("none"))
    monkeypatch.setattr(kimi_k3, "MoELayer", Experts)
    forbidden = mock.Mock(side_effect=AssertionError("TP-only setup reached"))
    monkeypatch.setattr(kimi_k3.KimiK3LatentDownOp, "initialize", forbidden)
    monkeypatch.setattr(kimi_k3, "K3MoeTailComm", forbidden)
    monkeypatch.setattr(kimi_k3, "LatentMoELayer", forbidden)

    mapping = Mapping(
        rank=min(3, world - 1),
        world_size=world,
        attn_tp_size=tp,
        attn_dp_size=dp,
        pp_size=pp,
        moe_tp_size=1,
        moe_ep_size=ep,
    )
    layer = KimiLinearMoE(
        config=KimiLinearConfig(
            hidden_size=64,
            routed_expert_hidden_size=32,
            moe_intermediate_size=32,
            num_experts=64,
            num_experts_per_token=2,
            num_shared_experts=1,
        ),
        mapping=mapping,
        layer_index=1,
        model_scope="test",
        moe_block_count=1,
        quant_config=None,
        prefix="moe",
        alt_stream=None,
    )
    forbidden.assert_not_called()
    assert layer.comm is None
    assert layer.native_latent_moe is None
    assert not layer._shard_latent_projections
    assert layer._shard_up_projection is True
    assert not layer.routed_expert_down_proj.narrowed
    assert layer.routed_expert_down_proj.weight.shape == (32, 64)
    assert layer.routed_expert_up_proj.narrowed
    assert layer.routed_expert_up_proj.shard_group == mapping.attn.tp_group
    assert layer.routed_expert_up_proj.weight.shape == (64 // tp, 32)
    assert layer.shared_experts.down_proj.tp_size == tp
    assert layer.shared_experts.down_proj.tp_group == mapping.attn.tp_group
    assert layer.experts.kwargs["routing_mode"] == "precomputed_topk"
    assert layer.experts.kwargs["token_scatter_size"] == tp
    layer.pack_input_projection_weights()
    assert layer.packed_input_projection_weight is None


def test_mega_moe_rejects_invalid_ep_layout(monkeypatch) -> None:
    monkeypatch.setattr(kimi_k3, "get_moe_backend", lambda: MoeBackend("mega_moe"))
    monkeypatch.setattr(kimi_k3, "get_all2all_backend", lambda: All2AllBackend("none"))
    with pytest.raises(ValueError, match="EP > 1"):
        KimiLinearMoE(
            config=SimpleNamespace(),
            mapping=Mapping(
                rank=0, world_size=8, attn_tp_size=8, moe_tp_size=8, moe_ep_size=1
            ),
            layer_index=0,
            model_scope="test",
            moe_block_count=1,
            quant_config=None,
            prefix="moe",
            alt_stream=None,
        )


@pytest.mark.parametrize(
    "tp,dp,dp_token_counts",
    [
        (4, 1, [0]),
        (4, 1, [2]),
        (4, 1, [7]),
        (4, 1, [8]),
        (4, 2, [6, 2]),
        (4, 2, [0, 3]),
    ],
)
def test_mega_moe_tp_ep_and_dep_forward_matches_dense_reference(
    monkeypatch, tp: int, dp: int, dp_token_counts: list[int]
) -> None:
    """Simulate every rank of TP/EP and DEP MegaMoE and verify exact equivalence."""
    ep = tp * dp
    hidden_size = 16
    routed_hidden = 8
    top_k = 2
    torch.manual_seed(17)
    down_w = torch.randn(routed_hidden, hidden_size, dtype=torch.bfloat16) * 0.2
    up_w = torch.randn(hidden_size, routed_hidden, dtype=torch.bfloat16) * 0.2
    shared_w = torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16) * 0.2
    norm_w = torch.ones(routed_hidden, dtype=torch.bfloat16)

    dp_hiddens = [
        torch.randn(n, hidden_size, dtype=torch.bfloat16) for n in dp_token_counts
    ]
    dp_prefixes = [
        torch.randn(n, hidden_size, dtype=torch.bfloat16) for n in dp_token_counts
    ]
    global_counts = [
        dp_token_counts[dp_rank] for dp_rank in range(dp) for _ in range(tp)
    ]

    def expert_transform(x: torch.Tensor) -> torch.Tensor:
        return x * 1.5

    expected_max_slice = max((n + tp - 1) // tp for n in dp_token_counts)

    # Precompute full routed output per DP group for simulated token_all_gather.
    dp_expected_routed = [expert_transform(h @ down_w.T) for h in dp_hiddens]
    # Split shared_w by columns across TP ranks so sum(partial) == h @ shared_w.T.
    shard_cols = hidden_size // tp

    for rank in range(ep):
        mapping = Mapping(
            rank=rank,
            world_size=ep,
            attn_tp_size=tp,
            attn_dp_size=dp,
            moe_tp_size=1,
            moe_ep_size=ep,
        )
        dp_rank = mapping.attn.dp_rank
        tp_rank = mapping.attn.tp_rank
        hidden = dp_hiddens[dp_rank]
        prefix = dp_prefixes[dp_rank]
        n_tok = dp_token_counts[dp_rank]
        expected_tp_counts = [
            n_tok // tp + (1 if r < n_tok % tp else 0) for r in range(tp)
        ]
        local_n = expected_tp_counts[tp_rank]
        start = sum(expected_tp_counts[:tp_rank])

        events: list[str] = []

        @contextmanager
        def scope(**kwargs):
            yield SimpleNamespace(
                branch=nullcontext,
                branch_after_main=nullcontext,
                record_checkpoint=lambda: events.append("record"),
                join_checkpoint=lambda: events.append("join_ckpt"),
                join=lambda: events.append("join"),
            )

        up_proj = Kimi3LatentProjection(
            routed_hidden,
            hidden_size,
            params_dtype=torch.bfloat16,
            solution="torch",
            shard_group=mapping.attn.tp_group,
            shard_rank=tp_rank,
            shard_size=tp,
        )
        up_proj.weight.data.copy_(
            up_w[tp_rank * shard_cols : (tp_rank + 1) * shard_cols]
        )

        def norm(value: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.rms_norm(
                value.float(), (routed_hidden,), weight=norm_w.float(), eps=1e-6
            ).to(value.dtype)

        buf_calls: list[tuple[int, int]] = []

        def input_buffers(tok: int, max_tok: int):
            buf_calls.append((tok, max_tok))
            return (
                torch.empty((tok, routed_hidden // 2), dtype=torch.uint8),
                torch.empty((tok, max(1, routed_hidden // 16)), dtype=torch.uint8),
                torch.full((tok, top_k), -1, dtype=torch.int32),
                torch.zeros((tok, top_k), dtype=torch.float32),
            )

        expert_calls: list[tuple[int, int, int]] = []

        def run_experts(
            routed_in,
            routing,
            *,
            num_global_tokens,
            max_num_tokens_per_gpu,
            do_finalize,
        ):
            events.append("experts")
            expert_calls.append(
                (routed_in.shape[0], max_num_tokens_per_gpu, num_global_tokens)
            )
            assert do_finalize
            return expert_transform(routed_in)

        def fake_token_all_gather(tensor, *, group, scattered_num_tokens):
            events.append("token_all_gather")
            assert group == mapping.attn.tp_group
            assert scattered_num_tokens == expected_tp_counts
            torch.testing.assert_close(
                tensor, dp_expected_routed[dp_rank][start : start + local_n]
            )
            return dp_expected_routed[dp_rank]

        def fake_all_reduce(tensor, group):
            events.append("all_reduce")
            assert group == mapping.attn.tp_group
            # Reconstruct exact sum across all TP ranks in this DP group:
            # sum_r (shared_partial_r + column_shard_r(up_proj(norm(routed)))).
            routed_normed = norm(dp_expected_routed[dp_rank])
            expected_local = (
                hidden[:, tp_rank * shard_cols : (tp_rank + 1) * shard_cols]
                @ shared_w[:, tp_rank * shard_cols : (tp_rank + 1) * shard_cols].T
            )
            expected_local[:, tp_rank * shard_cols : (tp_rank + 1) * shard_cols] += (
                routed_normed
                @ up_w[tp_rank * shard_cols : (tp_rank + 1) * shard_cols].T
            )
            torch.testing.assert_close(tensor, expected_local)
            return hidden @ shared_w.T + routed_normed @ up_w.T

        def run_shared(x, *, down_out):
            return (
                x[:, tp_rank * shard_cols : (tp_rank + 1) * shard_cols]
                @ shared_w[:, tp_rank * shard_cols : (tp_rank + 1) * shard_cols].T
            )

        def run_topk(
            src,
            logits,
            *,
            output_format,
            num_token_non_padded,
            expert_location_dispatch_info,
            out,
        ):
            w_out, ids_out = out
            ids_out.fill_(1)
            w_out.fill_(0.5)
            return StandardTopKOutput(w_out, ids_out, None)

        monkeypatch.setattr(kimi_k3, "token_all_gather", fake_token_all_gather)
        monkeypatch.setattr(kimi_k3, "all_reduce", fake_all_reduce)
        monkeypatch.setattr(
            kimi_k3, "all_gather", mock.Mock(side_effect=AssertionError("all_gather"))
        )
        monkeypatch.setattr(
            kimi_k3,
            "reduce_scatter",
            mock.Mock(side_effect=AssertionError("reduce_scatter")),
        )

        shared_mod = mock.Mock(side_effect=run_shared)
        shared_mod.shared_parallel = None
        shared_mod.shared_communication = None

        layer = SimpleNamespace(
            execution_plan=SimpleNamespace(
                use_mega_moe=True, owns_ep_communication=True
            ),
            mapping=mapping,
            stream_fork=SimpleNamespace(scope=scope),
            gate=lambda src: torch.zeros((src.shape[0], 8), dtype=torch.bfloat16),
            topk=run_topk,
            expert_dispatch_info=None,
            routed_expert_down_proj=lambda src: (src @ down_w.T, None),
            experts=SimpleNamespace(
                plan={"weight_dtype": "mxfp4"},
                input_buffers=input_buffers,
            ),
            moe_alltoall=None,
            _routed_experts=run_experts,
            routed_expert_norm=norm,
            routed_expert_up_proj=up_proj,
            _shard_up_projection=True,
            shared_experts=shared_mod,
            routed_hidden=routed_hidden,
            top_k=top_k,
        )
        layer._forward_attn_dp = lambda h, p, c: KimiLinearMoE._forward_attn_dp(
            layer, h, p, c
        )

        ctx = SimpleNamespace(
            collective_global_num_tokens=global_counts if dp > 1 else None,
            global_num_tokens=None,
            forward_mode=ForwardMode.EXTEND,
            attn_backend=None,
        )
        out = KimiLinearMoE.forward(
            layer,
            hidden,
            prefix,
            num_global_tokens=sum(dp_token_counts),
            max_num_tokens_per_gpu=expected_max_slice,
            ctx=ctx,
            prefix_is_sharded=False,
        )

        if expected_max_slice == 0:
            assert out is prefix
            assert expert_calls == []
            assert events == []
        else:
            assert buf_calls == [(local_n, expected_max_slice)]
            assert expert_calls == [
                (local_n, expected_max_slice, ep * expected_max_slice)
            ]
            if n_tok == 0:
                assert out is prefix
                assert events == ["join", "experts"]
            else:
                assert events == ["join", "experts", "token_all_gather", "all_reduce"]
                expected = prefix + (
                    hidden @ shared_w.T + norm(dp_expected_routed[dp_rank]) @ up_w.T
                )
                torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_mega_moe_tp_ep_masks_cuda_graph_decode_padding_rows(monkeypatch) -> None:
    """Padded decode rows (decode_window_locations == 0) must be masked to id=-1, weight=0."""
    tp = 4
    num_tokens = 7  # ragged counts: [2, 2, 2, 1]
    locations = torch.tensor([11, 12, 13, 0, 0, 0, 0], dtype=torch.int32)
    hidden = torch.randn(num_tokens, 8, dtype=torch.bfloat16)
    prefix = torch.zeros_like(hidden)

    for rank in range(tp):
        mapping = Mapping(
            rank=rank, world_size=tp, attn_tp_size=tp, moe_tp_size=1, moe_ep_size=tp
        )
        counts = [2, 2, 2, 1]
        start = sum(counts[:rank])
        local_n = counts[rank]
        expected_live = locations[start : start + local_n] > 0

        @contextmanager
        def scope(**kwargs):
            yield SimpleNamespace(
                branch=nullcontext,
                branch_after_main=nullcontext,
                join=lambda: None,
            )

        captured_routing: list[tuple[torch.Tensor, torch.Tensor]] = []

        def run_experts(
            routed_in,
            routing,
            *,
            num_global_tokens,
            max_num_tokens_per_gpu,
            do_finalize,
        ):
            captured_routing.append(
                (routing.topk_ids.clone(), routing.topk_weights.clone())
            )
            return routed_in

        def run_topk(
            src,
            logits,
            *,
            output_format,
            num_token_non_padded,
            expert_location_dispatch_info,
            out,
        ):
            w_out, ids_out = out
            ids_out.fill_(5)
            w_out.fill_(0.75)
            return StandardTopKOutput(w_out, ids_out, None)

        monkeypatch.setattr(
            kimi_k3,
            "token_all_gather",
            lambda t, *, group, scattered_num_tokens: torch.zeros(
                (num_tokens, 4), dtype=t.dtype
            ),
        )
        monkeypatch.setattr(kimi_k3, "all_reduce", lambda t, group: t)

        up_proj = Kimi3LatentProjection(
            4,
            8,
            params_dtype=torch.bfloat16,
            solution="torch",
            shard_group=mapping.attn.tp_group,
            shard_rank=rank,
            shard_size=tp,
        )
        shared_mod = mock.Mock(return_value=torch.zeros_like(hidden))
        shared_mod.shared_parallel = None
        shared_mod.shared_communication = None

        layer = SimpleNamespace(
            execution_plan=SimpleNamespace(
                use_mega_moe=True, owns_ep_communication=True
            ),
            mapping=mapping,
            stream_fork=SimpleNamespace(scope=scope),
            gate=lambda src: torch.zeros((src.shape[0], 8), dtype=torch.bfloat16),
            topk=run_topk,
            expert_dispatch_info=None,
            routed_expert_down_proj=lambda src: (src[:, :4].contiguous(), None),
            experts=SimpleNamespace(
                plan={"weight_dtype": "mxfp4"},
                input_buffers=lambda tok, max_tok: (
                    torch.empty((tok, 2), dtype=torch.uint8),
                    torch.empty((tok, 1), dtype=torch.uint8),
                    torch.empty((tok, 2), dtype=torch.int32),
                    torch.empty((tok, 2), dtype=torch.float32),
                ),
            ),
            moe_alltoall=None,
            _routed_experts=run_experts,
            routed_expert_norm=None,
            routed_expert_up_proj=up_proj,
            _shard_up_projection=True,
            shared_experts=shared_mod,
            routed_hidden=4,
            top_k=2,
        )
        ctx = SimpleNamespace(
            collective_global_num_tokens=None,
            global_num_tokens=None,
            forward_mode=ForwardMode.DECODE,
            attn_backend=SimpleNamespace(decode_window_locations=lambda: locations),
        )
        KimiLinearMoE._forward_attn_dp(layer, hidden, prefix, ctx)
        assert len(captured_routing) == 1
        ids, weights = captured_routing[0]
        for row in range(local_n):
            if expected_live[row]:
                assert torch.equal(ids[row], torch.tensor([5, 5], dtype=torch.int32))
                assert torch.equal(weights[row], torch.tensor([0.75, 0.75]))
            else:
                assert torch.equal(ids[row], torch.tensor([-1, -1], dtype=torch.int32))
                assert torch.equal(weights[row], torch.tensor([0.0, 0.0]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
