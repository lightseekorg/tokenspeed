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

from __future__ import annotations

import inspect
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from tokenspeed_kernel import (
    gated_residual_combine,
    gated_residual_combine_norm,
    gated_residual_mix,
    grouped_gemma_rmsnorm,
)

import tokenspeed.runtime.distributed.comm_manager as comm_manager_module
import tokenspeed.runtime.layers.hyperconnection as hyperconnection_module
import tokenspeed.runtime.models.qwen4_exp as qwen4_exp_module
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.hyperconnection import (
    GatedResidualSimple,
    GatedResidualUpdate,
    HyperConnectionConfig,
)
from tokenspeed.runtime.models.qwen4_exp import Qwen4ExpModel, _Qwen4ExpDecoderMixin
from tokenspeed.runtime.models.qwen4_exp_nextn import Qwen4ExpDraftAttentionDecoderLayer


def test_runtime_uses_tokenspeed_kernel_boundary() -> None:
    source = inspect.getsource(hyperconnection_module)
    assert "from tokenspeed_kernel import" in source
    assert "import triton" not in source
    assert "@triton.jit" not in source


def test_kernel_boundary_is_gpu_only() -> None:
    normalized = torch.empty(1, 8)
    projection = torch.empty(4, 8)
    up = torch.empty(8, 2)
    with pytest.raises(ValueError, match="requires GPU tensors"):
        gated_residual_mix(
            normalized, projection, up, 2, 4, 2, weights_independent=False
        )
    with pytest.raises(ValueError, match="requires GPU tensors"):
        gated_residual_combine(torch.empty(1, 4), normalized, torch.empty(1, 2), 2, 4)
    with pytest.raises(ValueError, match="requires GPU tensors"):
        grouped_gemma_rmsnorm(normalized, torch.empty(8), 4, 1e-6)
    with pytest.raises(ValueError, match="requires GPU tensors"):
        gated_residual_combine_norm(
            torch.empty(1, 4),
            normalized,
            torch.empty(1, 2),
            torch.empty(8),
            2,
            4,
            1e-6,
            preload_residual=False,
        )


def test_up_weight_loader_prepares_kernel_cache(monkeypatch) -> None:
    prepare = mock.Mock(return_value=True)
    monkeypatch.setattr(
        hyperconnection_module, "prepare_gated_residual_weight_cache", prepare
    )
    lowrank = 3
    mixer = GatedResidualSimple(
        HyperConnectionConfig(hc_count=2, hidden_size=4, hc_lowrank=lowrank)
    )
    param = mixer.input_mix_weight_up.weight
    loaded = torch.randn_like(param)

    param.weight_loader(param, loaded)

    torch.testing.assert_close(param, loaded)
    prepare.assert_called_once_with(param, lowrank)


def test_up_weight_loader_rejects_shape_change(monkeypatch) -> None:
    prepare = mock.Mock(return_value=True)
    monkeypatch.setattr(
        hyperconnection_module, "prepare_gated_residual_weight_cache", prepare
    )
    mixer = GatedResidualSimple(
        HyperConnectionConfig(hc_count=2, hidden_size=4, hc_lowrank=3)
    )
    param = mixer.input_mix_weight_up.weight

    with pytest.raises(ValueError, match="shape mismatch"):
        param.weight_loader(param, torch.empty(8, 4))

    prepare.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("per_branch_norm", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_mix_fuses_previous_combine_with_its_own_norm(
    per_branch_norm: bool, dtype: torch.dtype
) -> None:
    config = HyperConnectionConfig(
        hc_count=3,
        hidden_size=13,
        hc_lowrank=7,
        rms_norm_eps=1e-6,
        params_dtype=dtype,
        hc_per_branch_norm=per_branch_norm,
    )
    previous = GatedResidualSimple(config, use_mix=True, use_combine=True).to(
        device="cuda", dtype=dtype
    )
    current = GatedResidualSimple(config, use_mix=True, use_combine=True).to(
        device="cuda", dtype=dtype
    )
    with torch.no_grad():
        previous.hc_norm.weight.fill_(0.25)
        current.hc_norm.weight.fill_(-0.125)
    residual = torch.randn(5, 39, device="cuda", dtype=dtype)
    block = torch.randn(5, 13, device="cuda", dtype=dtype)
    _, previous_residuals = previous.mix(
        residual, block_output=None, inject_logits=None, preload_residual=False
    )
    combined = previous.combine(block, previous_residuals)
    expected_mix, expected_residuals = current.mix(
        combined, block_output=None, inject_logits=None, preload_residual=False
    )

    with mock.patch.object(
        current.hc_norm,
        "forward",
        side_effect=AssertionError("unexpected separate norm"),
    ):
        actual_mix, actual_residuals = current.mix(
            residual,
            block_output=block,
            inject_logits=previous_residuals[2],
            preload_residual=True,
        )

    torch.testing.assert_close(actual_mix, expected_mix, rtol=0, atol=0)
    for actual, expected in zip(actual_residuals, expected_residuals, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # The next injection must retain the unnormalized residual from the fused op.
    next_block = torch.randn_like(block)
    torch.testing.assert_close(
        current.combine(next_block, actual_residuals),
        current.combine(next_block, expected_residuals),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("communication", ["idle", "unchanged", "row_slice"])
def test_attention_to_mlp_fusion_after_communication(communication: str) -> None:
    config = HyperConnectionConfig(
        hc_count=4,
        hidden_size=8,
        hc_lowrank=6,
        rms_norm_eps=1e-6,
        params_dtype=torch.float32,
        hc_per_branch_norm=True,
    )
    attention = GatedResidualSimple(config, use_mix=True, use_combine=True).cuda()
    mlp = GatedResidualSimple(config, use_mix=True, use_combine=True).cuda()
    rows = 0 if communication == "idle" else 6
    residual = torch.randn(rows, 32, device="cuda")
    output = torch.randn(rows, 8, device="cuda")
    _, residuals = attention.mix(
        residual, block_output=None, inject_logits=None, preload_residual=False
    )
    row_slice = slice(2, 5) if communication == "row_slice" else slice(None)
    communicated_residual = residual[row_slice]
    communicated_output = output[row_slice]
    aligned_residuals = attention.norm_for(communicated_residual, residuals)
    combined = attention.combine(communicated_output, aligned_residuals)
    expected_mix, expected_residuals = mlp.mix(
        combined, block_output=None, inject_logits=None, preload_residual=False
    )
    post_attn_comm = mock.Mock(
        return_value=(communicated_output, communicated_residual)
    )
    layer = SimpleNamespace(
        attn_hyper_connection=attention,
        mlp_hyper_connection=mlp,
        comm_manager=SimpleNamespace(post_attn_comm=post_attn_comm),
    )
    ctx = SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: communication == "idle")
    )

    with mock.patch.object(
        attention, "combine", side_effect=AssertionError("unexpected separate combine")
    ):
        actual_mix, actual_residuals = _Qwen4ExpDecoderMixin._finish_attention(
            layer, output, residuals, ctx
        )

    torch.testing.assert_close(actual_mix, expected_mix, rtol=0, atol=0)
    for actual, expected in zip(actual_residuals, expected_residuals, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert post_attn_comm.call_count == (0 if communication == "idle" else 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_non_power_of_two_hc_scales_projection_results() -> None:
    hc_count, hidden_size, lowrank = 3, 8, 6
    mixer = GatedResidualSimple(
        HyperConnectionConfig(
            hc_count=hc_count,
            hidden_size=hidden_size,
            hc_lowrank=lowrank,
            params_dtype=torch.float32,
        )
    ).cuda()
    down_weight = torch.randn(lowrank, hc_count * hidden_size, device="cuda")
    inject_weight = torch.randn(hc_count, hc_count * hidden_size, device="cuda")
    param = mixer.mix_inject_proj.weight
    loader = param.weight_loader
    loader(param, down_weight, "mix")
    loader(param, inject_weight, "inject")

    torch.testing.assert_close(param[:lowrank], down_weight)
    torch.testing.assert_close(param[lowrank:], inject_weight)
    assert mixer._projection_scale == pytest.approx(1.0 / hc_count)

    hyper_input = torch.randn(5, hc_count * hidden_size, device="cuda")
    block_output = torch.randn(5, hidden_size, device="cuda")
    mixed, residuals = mixer.mix(
        hyper_input, block_output=None, inject_logits=None, preload_residual=False
    )
    combined = mixer.combine(block_output, residuals)
    normalized = residuals[1]

    down = torch.nn.functional.linear(normalized, down_weight) / hc_count
    gate = mixer.input_mix_weight_up(torch.nn.functional.silu(down))
    expected_mixed = (
        torch.sigmoid(gate).unflatten(-1, (hc_count, hidden_size))
        * normalized.unflatten(-1, (hc_count, hidden_size))
    ).mean(dim=-2)
    torch.testing.assert_close(mixed, expected_mixed)

    inject = 2 * torch.sigmoid(
        torch.nn.functional.linear(normalized, inject_weight) / hc_count
    )
    expected_combined = hyper_input.unflatten(
        -1, (hc_count, hidden_size)
    ) + block_output.unsqueeze(-2) * inject.unsqueeze(-1)
    torch.testing.assert_close(combined, expected_combined.flatten(-2))


def test_runtime_declares_loaded_hc_weights_independent(monkeypatch) -> None:
    mixer = GatedResidualSimple(
        HyperConnectionConfig(hc_count=4, hidden_size=8, hc_lowrank=3)
    )
    value = torch.randn(2, 32)
    mixed = torch.randn(2, 8)
    inject = torch.randn(2, 4)
    monkeypatch.setattr(mixer, "_normalize", lambda x: x)
    call = mock.Mock(return_value=(mixed, inject))
    monkeypatch.setattr(hyperconnection_module, "gated_residual_mix", call)
    mixer.mix(value, block_output=None, inject_logits=None, preload_residual=False)
    assert call.call_args.kwargs["weights_independent"] is True


@pytest.mark.parametrize("layer_id", [0, 1])
@pytest.mark.parametrize("attn_tp", [1, 4])
@pytest.mark.parametrize("other_tp", [1, 4])
@pytest.mark.parametrize("is_moe", [False, True])
def test_residual_fusion_gather_boundaries_match_communication(
    monkeypatch, layer_id: int, attn_tp: int, other_tp: int, is_moe: bool
) -> None:
    mapping = SimpleNamespace(
        has_attn_tp=attn_tp > 1,
        attn=SimpleNamespace(tp_size=attn_tp, tp_group=list(range(attn_tp))),
        dense=SimpleNamespace(tp_size=other_tp),
        moe=SimpleNamespace(tp_ep_size=other_tp),
    )
    manager = CommManager(
        mapping=mapping,
        layer_id=layer_id,
        is_moe=is_moe,
        prev_is_moe=is_moe,
        input_layernorm=None,
        post_attn_layernorm=None,
    )
    x = torch.arange(12).reshape(3, 4)
    gather = mock.Mock(side_effect=lambda value, **kwargs: value.repeat(2, 1))
    monkeypatch.setattr(comm_manager_module, "token_all_gather", gather)
    monkeypatch.setattr(
        manager, "attn_tp_group_scattered_num_tokens", lambda ctx: [3] * attn_tp
    )
    expected_final = attn_tp > 1 and attn_tp != other_tp
    expected_pre = layer_id > 0 and expected_final
    assert manager.needs_pre_attn_all_gather() == expected_pre
    assert manager.needs_final_all_gather() == expected_final
    for operation, expected in (
        (manager.pre_attn_comm, expected_pre),
        (manager.gather_residual, expected_pre),
    ):
        gather.reset_mock()
        result = operation(x, None)
        assert gather.call_count == int(expected)
        assert (result is x) == (not expected)
    gather.reset_mock()
    result, residual = manager.post_final_norm_comm(x, x, None)
    assert gather.call_count == int(expected_final)
    assert residual is x and (result is x) == (not expected_final)


class _TailFusionLayer(torch.nn.Module, _Qwen4ExpDecoderMixin):
    """Exercise the real residual chain around small deterministic sublayers."""

    def __init__(
        self,
        config: HyperConnectionConfig,
        pre_gather: bool,
        final_gather: bool,
        ple: bool,
    ):
        super().__init__()
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.attn_hyper_connection = GatedResidualSimple(
            config, use_mix=True, use_combine=True
        )
        self.mlp_hyper_connection = GatedResidualSimple(
            config, use_mix=True, use_combine=True
        )
        self.mlp = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.is_moe = False
        self.materialize_tail = False
        self.ple = (lambda value, ids, ctx: torch.tanh(value)) if ple else None
        self.comm_manager = SimpleNamespace(
            needs_pre_attn_all_gather=lambda: pre_gather,
            needs_final_all_gather=lambda: final_gather,
            pre_attn_comm=lambda value, ctx: (
                torch.cat((value, value.flip(0)), dim=0) if pre_gather else value
            ),
            post_attn_comm=lambda value, residual, ctx: (value, residual),
            pre_mlp_comm=lambda value, ctx: value,
            post_mlp_comm=lambda value, residual, ctx: (value, residual),
            post_final_norm_comm=lambda value, residual, ctx: (
                torch.cat((value.flip(0), value), dim=0),
                residual,
            ),
            get_num_tokens=lambda ctx: (ctx.input_num_tokens, ctx.input_num_tokens),
        )

    def forward(self, positions, hidden_states, residual, ctx, input_ids):
        del positions, residual
        mixed, residuals = self._prepare_attention(hidden_states, input_ids, ctx)
        attention_output = mixed if ctx.forward_mode.is_idle() else mixed * 0.125
        mixed, residuals = self._finish_attention(attention_output, residuals, ctx)
        update = self._run_mlp(mixed, residuals, ctx)
        assert isinstance(update, GatedResidualUpdate)
        return update.materialize() if self.materialize_tail else update, None


def _tail_fusion_model(
    dtype: torch.dtype, boundary: str, layer_count: int
) -> Qwen4ExpModel:
    config = HyperConnectionConfig(
        hc_count=4,
        hidden_size=32,
        hc_lowrank=16,
        rms_norm_eps=1e-6,
        params_dtype=dtype,
        hc_per_branch_norm=True,
    )
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    torch.nn.Module.__init__(model)
    model.config = config
    model.hidden_size = config.hidden_size
    model.embed_tokens = torch.nn.Embedding(32, config.hidden_size)
    model.layers = torch.nn.ModuleList(
        _TailFusionLayer(
            config,
            pre_gather=boundary == "pre_gather" and index == 1,
            final_gather=boundary == "final_gather" and index == layer_count - 1,
            ple=boundary == "ple" and index == 1,
        )
        for index in range(layer_count)
    )
    # The fake PLE has no cache state; the residual ordering is exercised above.
    model.ple_layers = ()
    model.hyper_connection_mixer = GatedResidualSimple(
        config, use_mix=True, use_combine=False
    )
    model = model.to(device="cuda", dtype=dtype)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, GatedResidualSimple):
                module.hc_norm.weight.uniform_(-0.25, 0.25)
    return model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "boundary", ["none", "ple", "pre_gather", "final_gather", "deepstack"]
)
def test_mlp_tail_fusion_preserves_intervening_operations(
    monkeypatch, dtype: torch.dtype, boundary: str
) -> None:
    monkeypatch.setattr(
        qwen4_exp_module,
        "get_global_expert_distribution_recorder",
        lambda: SimpleNamespace(with_current_layer=lambda index: nullcontext()),
    )
    torch.manual_seed(103)
    model = _tail_fusion_model(dtype, boundary, 3)
    ids = torch.arange(4, device="cuda")
    positions = ids.clone()
    ctx = SimpleNamespace(forward_mode=ForwardMode.DECODE, input_num_tokens=4)
    deepstack = (
        torch.randn(4, 96, device="cuda", dtype=dtype)
        if boundary == "deepstack"
        else None
    )

    def run():
        return model(
            input_ids=ids,
            positions=positions,
            ctx=ctx,
            input_embeds=None,
            pp_proxy_tensors=None,
            input_deepstack_embeds=deepstack,
        )

    for layer in model.layers:
        layer.materialize_tail = True
    expected, expected_hc = run()
    for layer in model.layers:
        layer.materialize_tail = False
    with mock.patch.object(
        hyperconnection_module,
        "gated_residual_combine",
        wraps=hyperconnection_module.gated_residual_combine,
    ) as combine, mock.patch.object(
        hyperconnection_module,
        "gated_residual_combine_norm",
        wraps=hyperconnection_module.gated_residual_combine_norm,
    ) as fused:
        actual, actual_hc = run()
    materialized_tails = {
        "none": 0,
        "ple": 1,
        "pre_gather": 1,
        "final_gather": 1,
        "deepstack": 3,
    }[boundary]
    assert combine.call_count == materialized_tails
    assert fused.call_count == 6 - materialized_tails
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_hc[0], expected_hc[0], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("rows", [0, 1, 4])
@pytest.mark.parametrize("layer_count", [1, 3])
def test_mlp_tail_fusion_graph_replay_keeps_current_residual(
    monkeypatch, rows: int, layer_count: int
) -> None:
    monkeypatch.setattr(
        qwen4_exp_module,
        "get_global_expert_distribution_recorder",
        lambda: SimpleNamespace(with_current_layer=lambda index: nullcontext()),
    )
    torch.manual_seed(109)
    model = _tail_fusion_model(torch.bfloat16, "none", layer_count)
    ids = torch.arange(rows, device="cuda")
    ctx = SimpleNamespace(
        forward_mode=ForwardMode.IDLE if rows == 0 else ForwardMode.DECODE,
        input_num_tokens=rows,
    )
    value = torch.randn(rows, 32, device="cuda", dtype=torch.bfloat16)

    def run():
        return model(
            input_ids=ids,
            positions=ids,
            ctx=ctx,
            input_embeds=value,
            pp_proxy_tensors=None,
            input_deepstack_embeds=None,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, actual_hc = run()
    for seed in (113, 127):
        torch.manual_seed(seed)
        value.copy_(torch.randn_like(value))
        for layer in model.layers:
            layer.materialize_tail = True
        expected, expected_hc = run()
        for layer in model.layers:
            layer.materialize_tail = False
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_hc[0], expected_hc[0], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_mtp_tail_fusion_preserves_narrowed_residual_rows() -> None:
    torch.manual_seed(131)
    model = _tail_fusion_model(torch.bfloat16, "none", 1)
    layer = model.layers[0]
    ids = torch.arange(4, device="cuda")
    value = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    ctx = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        input_num_tokens=2,
        draft_narrowing=object(),
        gather_ids=torch.tensor([3, 0], device="cuda"),
    )
    layer.self_attention = (
        lambda positions, mixed, ctx: mixed.index_select(0, ctx.gather_ids) * 0.125
    )
    mixed, residuals = layer._prepare_attention(value, ids, ctx)
    output = layer.self_attention(ids, mixed, ctx)
    residuals = tuple(tensor.index_select(0, ctx.gather_ids) for tensor in residuals)
    combined = layer.attn_hyper_connection.combine(output, residuals)
    mixed, residuals = layer.mlp_hyper_connection.mix(
        combined, block_output=None, inject_logits=None, preload_residual=False
    )
    expected_hc = layer.mlp_hyper_connection.combine(layer.mlp(mixed), residuals)
    expected, _ = model.hyper_connection_mixer.mix(
        expected_hc, block_output=None, inject_logits=None, preload_residual=False
    )

    with mock.patch.object(
        hyperconnection_module,
        "gated_residual_combine",
        side_effect=AssertionError("unexpected separate combine"),
    ):
        update, _ = Qwen4ExpDraftAttentionDecoderLayer.forward(
            layer, positions=ids, hidden_states=value, input_ids=ids, ctx=ctx
        )
        actual, actual_residuals = qwen4_exp_module._mix_gated_residual(
            model.hyper_connection_mixer, update
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_residuals[0], expected_hc, rtol=0, atol=0)
