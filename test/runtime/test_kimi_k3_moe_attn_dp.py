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

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput, TopKOutputFormat
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


def test_attn_dp_replicates_dense_weights_and_skips_tp_tail_setup(monkeypatch) -> None:
    class Experts(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.supports_precomputed_topk = True
            self.topk_output_format = TopKOutputFormat.STANDARD
            self.plan = {"a2a_backend": "none"}

    monkeypatch.setattr(
        kimi_k3, "get_moe_backend", lambda: SimpleNamespace(value="flashinfer_trtllm")
    )
    plan = kimi_k3.Kimi3MoEExecutionPlan(
        use_native=False,
        use_trtllm=True,
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
    monkeypatch.setattr(
        kimi_k3, "situ_moe_unavailable_reason", mock.Mock(return_value=None)
    )
    monkeypatch.setattr(kimi_k3, "load_packaged_flashinfer_tuning_cache", mock.Mock())
    monkeypatch.setitem(kimi_k3.global_server_args_dict, "enforce_eager", False)
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


@pytest.mark.parametrize("counts", [(1, 1), (0, 1), (1, 3)])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("weights_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_norm", [False, True])
def test_attn_dp_exchanges_latents_and_returns_reduced_local_rows(
    monkeypatch,
    counts: tuple[int, int],
    rank: int,
    weights_dtype: torch.dtype,
    with_norm: bool,
) -> None:
    rows, capacity = counts[rank], max(counts)
    events = []
    group = (0, 1)
    latent = torch.zeros(2, capacity, 2, dtype=torch.bfloat16)
    ids = torch.zeros(2, capacity, 2, dtype=torch.int32)
    weights = torch.zeros(2, capacity, 2, dtype=weights_dtype)
    for owner, count in enumerate(counts):
        latent[owner, :count] = torch.tensor([owner + 1, owner + 2])
        ids[owner, :count] = torch.tensor([1, 0], dtype=torch.int32)
        weights[owner, :count] = torch.tensor([0.271828, 0.728172], dtype=weights_dtype)
    hidden = torch.cat((latent[rank, :rows], latent[rank, :rows]), dim=-1)
    prefix = hidden + 10
    gathers = iter((("AG_latent", latent), ("AG_ids", ids), ("AG_weights", weights)))

    def gather(tensor, group, *, dim):
        name, expected = next(gathers)
        events.append(name)
        assert group == (0, 1)
        assert dim == 0
        torch.testing.assert_close(tensor, expected[rank])
        if rows == capacity:
            assert tensor.data_ptr() == expected[rank].data_ptr()
        return expected.reshape(2 * capacity, 2)

    def experts(
        routed, routing, *, num_global_tokens, max_num_tokens_per_gpu, do_finalize
    ):
        events.append("experts")
        assert num_global_tokens == max_num_tokens_per_gpu == 2 * capacity
        assert do_finalize
        assert routing.router_logits is None
        torch.testing.assert_close(routed, latent.reshape(-1, 2))
        torch.testing.assert_close(routing.topk_ids, ids.reshape(-1, 2))
        torch.testing.assert_close(routing.topk_weights, weights.reshape(-1, 2))
        return routed * (rank + 1)

    def scatter(partial, *, group):
        events.append("RS")
        assert group == (0, 1)
        torch.testing.assert_close(partial, latent.reshape(-1, 2) * (rank + 1))
        return latent[rank] * 3

    def norm(value):
        events.append("norm")
        torch.testing.assert_close(value, latent[rank, :rows] * 3)
        return torch.nn.functional.rms_norm(value.float(), (2,), eps=1e-6).to(
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
        routed_hidden=2,
        top_k=2,
        topk=topk,
        gate=mock.Mock(return_value=torch.empty(rows, 2)),
        routed_expert_down_proj=mock.Mock(return_value=(latent[rank, :rows], None)),
        shared_experts=mock.Mock(return_value=hidden * 3),
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
            expected_latent.float(), (2,), eps=1e-6
        ).to(hidden.dtype)
    expected = (
        prefix + torch.cat((expected_latent, expected_latent), dim=-1) + hidden * 3
    )
    torch.testing.assert_close(result, expected)
    expected_events = ["AG_latent", "AG_ids", "AG_weights", "experts", "RS"]
    if rows:
        if with_norm:
            expected_events.append("norm")
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
