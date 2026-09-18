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

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput, TopKOutputFormat
from tokenspeed.runtime.layers.moe.utils import All2AllBackend
from tokenspeed.runtime.models import kimi_k3
from tokenspeed.runtime.models.kimi_k3 import KimiLinearMoE


@pytest.mark.parametrize("dp,ep,world", [(2, 1, 2), (2, 4, 4), (4, 4, 8)])
def test_attn_dp_rejects_partial_world_layout_before_backend_setup(
    monkeypatch, dp: int, ep: int, world: int
) -> None:
    backend = mock.Mock(side_effect=AssertionError("backend setup reached"))
    monkeypatch.setattr(kimi_k3, "get_moe_backend", backend)
    with pytest.raises(ValueError, match="attention DP == MoE EP == world size"):
        KimiLinearMoE(
            config=SimpleNamespace(),
            mapping=SimpleNamespace(
                world_size=world,
                attn=SimpleNamespace(dp_size=dp),
                moe=SimpleNamespace(ep_size=ep),
            ),
            layer_index=0,
            model_scope="test",
            moe_block_count=1,
            quant_config=None,
            prefix="moe",
            alt_stream=None,
        )
    backend.assert_not_called()


@pytest.mark.parametrize("backend", ["none", "agrs", "flashinfer"])
@pytest.mark.parametrize("fabric_available", [False, True])
@pytest.mark.parametrize("mega_moe", [False, True])
def test_attn_dp_replicates_dense_weights_and_selects_transport(
    monkeypatch, backend: str, fabric_available: bool, mega_moe: bool
) -> None:
    class Experts(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.supports_precomputed_topk = True
            self.topk_output_format = TopKOutputFormat.STANDARD
            self.plan = {"a2a_backend": "none"}

    monkeypatch.setattr(
        kimi_k3,
        "get_moe_backend",
        lambda: SimpleNamespace(value="mega_moe" if mega_moe else "flashinfer_trtllm"),
    )
    plan = kimi_k3.Kimi3MoEExecutionPlan(
        use_mega_moe=mega_moe,
        use_native=False,
        use_trtllm=not mega_moe,
        overlap_shared_experts=False,
        joint_moe_reduce=False,
    )
    forbidden = mock.Mock(side_effect=AssertionError("TP-only setup reached"))
    monkeypatch.setattr(
        kimi_k3.Kimi3MoEExecutionPlan, "build", mock.Mock(return_value=plan)
    )
    monkeypatch.setattr(
        kimi_k3.Kimi3MoEExecutionPlan, "prepare_latent_fusion", forbidden
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
    monkeypatch.setattr(kimi_k3, "load_packaged_flashinfer_tuning_cache", mock.Mock())
    monkeypatch.setitem(kimi_k3.global_server_args_dict, "enforce_eager", False)
    monkeypatch.setitem(kimi_k3.global_server_args_dict, "max_prefill_tokens", 8192)
    monkeypatch.setitem(kimi_k3.global_server_args_dict, "max_num_seqs", 128)
    monkeypatch.setitem(
        kimi_k3.global_server_args_dict, "speculative_num_draft_tokens", 4
    )
    mapping = SimpleNamespace(
        world_size=2,
        rank=1,
        attn=SimpleNamespace(dp_size=2, dp_rank=1, tp_size=1, cp_size=1),
        moe=SimpleNamespace(
            ep_size=2,
            ep_rank=1,
            ep_group=(0, 1),
            tp_size=1,
            tp_rank=0,
            tp_group=(1,),
            tp_ep_size=2,
            tp_ep_rank=1,
            tp_ep_group=(0, 1),
            dp_size=1,
        ),
    )
    expected_error = (mega_moe and backend != "none") or (
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
    assert layer.shared_experts.down_proj.tp_group == (1,)
    assert layer.experts.kwargs["routing_mode"] == "precomputed_topk"
    assert not hasattr(layer, "comm")
    assert not hasattr(layer, "native_latent_moe")


@pytest.mark.parametrize(
    "backend,dp,match",
    [
        ("deepep", 2, "does not support DeepEP"),
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

    def shared(value, *, down_out):
        events.append("shared")
        return value * 3

    fork = SimpleNamespace(scope=scope, branch=branch)
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

    def quantize(tensor, scale, *, is_sf_swizzled_layout, enable_pdl):
        events.append("quantize")
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
        routed, routing, *, num_global_tokens, max_num_tokens_per_gpu, do_finalize
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
        return received, ids.reshape(-1, 2), weights.reshape(-1, 2), 1234

    def combine(partial, local_tokens, max_tokens, combine_offset):
        events.append("combine")
        assert local_tokens == rows and max_tokens == capacity
        assert combine_offset == 1234
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
        mapping=SimpleNamespace(
            world_size=2,
            attn=SimpleNamespace(dp_rank=rank),
            moe=SimpleNamespace(ep_group=group),
        ),
        execution_plan=SimpleNamespace(use_mega_moe=False),
        moe_alltoall=(
            SimpleNamespace(dispatch=dispatch, combine=combine)
            if use_alltoall
            else None
        ),
        stream_fork=fork,
        routed_hidden=width,
        top_k=2,
        topk=topk,
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
    monkeypatch.setattr(kimi_k3, "reduce_scatter", scatter)
    monkeypatch.setattr(
        kimi_k3, "all_reduce", mock.Mock(side_effect=AssertionError("all-reduce"))
    )
    ctx = SimpleNamespace(
        collective_global_num_tokens=list(counts), global_num_tokens=[99, 99]
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
    expected_events = ["fork", *expected_events, "branch"]
    if rows:
        expected_events.append("shared")
    expected_events.extend(["branch_done", "join"])
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
        layer.shared_experts.assert_not_called()


def test_attn_dp_all_idle_skips_collectives(monkeypatch) -> None:
    forbidden = mock.Mock(side_effect=AssertionError("collective on all-idle batch"))
    monkeypatch.setattr(kimi_k3, "all_gather", forbidden)
    monkeypatch.setattr(kimi_k3, "reduce_scatter", forbidden)
    hidden = torch.empty(0, 4)
    layer = SimpleNamespace(
        mapping=SimpleNamespace(world_size=2, attn=SimpleNamespace(dp_rank=0))
    )
    ctx = SimpleNamespace(collective_global_num_tokens=None, global_num_tokens=[0, 0])
    result = KimiLinearMoE._forward_attn_dp(layer, hidden, hidden, ctx)
    assert result is hidden
    forbidden.assert_not_called()


def test_attn_dp_forward_bypasses_tp_tail() -> None:
    hidden = torch.ones(1, 4)
    ctx = SimpleNamespace()
    dp_forward = mock.Mock(return_value=hidden)
    layer = SimpleNamespace(
        mapping=SimpleNamespace(attn=SimpleNamespace(dp_size=2)),
        _forward_attn_dp=dp_forward,
    )
    result = KimiLinearMoE.forward(
        layer, hidden, hidden, num_global_tokens=2, max_num_tokens_per_gpu=1, ctx=ctx
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
        mapping=SimpleNamespace(world_size=2, attn=SimpleNamespace(dp_rank=0))
    )
    ctx = SimpleNamespace(collective_global_num_tokens=counts, global_num_tokens=None)
    with pytest.raises(ValueError, match="matching collective token counts"):
        KimiLinearMoE._forward_attn_dp(
            layer, torch.ones(1, 4), torch.ones(1, prefix_width), ctx
        )
    forbidden.assert_not_called()


def test_attn_dp_forward_requires_context() -> None:
    layer = SimpleNamespace(mapping=SimpleNamespace(attn=SimpleNamespace(dp_size=2)))
    hidden = torch.ones(1, 4)
    with pytest.raises(ValueError, match="requires a ForwardContext"):
        KimiLinearMoE.forward(
            layer,
            hidden,
            hidden,
            num_global_tokens=2,
            max_num_tokens_per_gpu=1,
            ctx=None,
        )


@pytest.mark.parametrize("rows", [0, 2])
def test_attn_dp_megamoe_keeps_inputs_local_and_owns_combine(monkeypatch, rows):
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

    @contextmanager
    def scope(**kwargs):
        yield SimpleNamespace(branch=nullcontext)

    routed = mock.Mock(return_value=latent)
    up = mock.Mock(return_value=hidden + prefix)
    layer = SimpleNamespace(
        execution_plan=SimpleNamespace(use_mega_moe=True),
        mapping=SimpleNamespace(world_size=2, attn=SimpleNamespace(dp_rank=0)),
        stream_fork=SimpleNamespace(scope=scope),
        topk=mock.Mock(return_value=StandardTopKOutput(weights, ids, None)),
        gate=mock.Mock(return_value=torch.zeros(rows, 2)),
        routed_expert_down_proj=mock.Mock(return_value=(latent, None)),
        experts=SimpleNamespace(
            plan={"weight_dtype": "nvfp4"}, w13_input_scale_quant=torch.ones(1)
        ),
        moe_alltoall=None,
        _routed_experts=routed,
        routed_expert_norm=None,
        routed_expert_up_proj=SimpleNamespace(forward_add3=up),
        shared_experts=mock.Mock(return_value=hidden),
        routed_hidden=4,
        top_k=2,
    )
    layer.topk.topk_config = SimpleNamespace(
        topk_weights_dtype=torch.bfloat16, topk_indices_dtype=torch.int32
    )
    ctx = SimpleNamespace(
        collective_global_num_tokens=[rows, 3], global_num_tokens=None
    )
    result = KimiLinearMoE._forward_attn_dp(layer, hidden, prefix, ctx)
    routed.assert_called_once()
    call = routed.call_args
    assert call.args[0][0].shape[0] == rows
    assert call.args[1].topk_ids.shape == (rows, 2)
    assert call.kwargs["max_num_tokens_per_gpu"] == 3
    assert call.kwargs["num_global_tokens"] == 6
    if rows:
        assert call.args[0] is payload
        torch.testing.assert_close(result, hidden + prefix)
    else:
        assert result is prefix
        up.assert_not_called()
