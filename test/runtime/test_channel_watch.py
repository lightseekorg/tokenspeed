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

"""Sparse channel observation through real Qwen3 MLP modules."""

import pytest
import torch
import torch.nn.functional as F

from tokenspeed.runtime.configs.qwen3_config import Qwen3Config
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.inspection.channel_watch import ChannelWatch, RowIdentity
from tokenspeed.runtime.models.qwen3 import Qwen3ForCausalLM
from tokenspeed.runtime.utils.env import global_server_args_dict


def _model(monkeypatch, *, dtype=torch.float32):
    monkeypatch.setitem(global_server_args_dict, "numerics", "auto")
    monkeypatch.setitem(global_server_args_dict, "layer_boundary_norm", "unfused")
    monkeypatch.setitem(global_server_args_dict, "logprob_order", "torch")
    monkeypatch.setitem(global_server_args_dict, "moe_combine_order", "rank")
    monkeypatch.setitem(global_server_args_dict, "enable_allreduce_fusion", False)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
    )
    model = Qwen3ForCausalLM(config, Mapping(rank=0, world_size=1)).to(
        device="cpu", dtype=dtype
    )
    torch.manual_seed(91)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 1:
                parameter.fill_(1)
            else:
                parameter.normal_(mean=0, std=0.15)
    model.eval()

    def mm(x, weight, *, bias=None, **kwargs):
        return F.linear(x, weight, bias)

    monkeypatch.setattr("tokenspeed.runtime.layers.dense.unquant.kernel_mm", mm)
    return model


def _watchlist(entries, identity="test-model"):
    return {
        "schema": "channel-watchlist-v1",
        "model_identity": identity,
        "entries": entries,
    }


def _watch(model, entries, *, identity="test-model", **choices):
    return ChannelWatch(
        model,
        watchlist=_watchlist(entries, identity),
        model_identity=identity,
        execution=choices.get("execution", "eager"),
        speculative=choices.get("speculative", False),
        monokernel=choices.get("monokernel", False),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_all_layer_sparse_samples_match_real_mlp_and_preserve_output(
    monkeypatch, dtype
):
    model = _model(monkeypatch, dtype=dtype)
    torch.manual_seed(73)
    inputs = [torch.randn(2, 64, dtype=dtype) for _ in range(2)]
    rows = [
        RowIdentity(request_id="request-a", token_position=17),
        RowIdentity(request_id="request-b", token_position=4),
    ]
    entries = [
        {"layer": 0, "channel": 1, "contribution_limit": 1e-6},
        {"layer": 0, "channel": 3, "contribution_limit": 1e9},
        {"layer": 1, "channel": 4, "contribution_limit": 1e-6},
    ]
    with torch.no_grad():
        baseline = [layer.mlp(x) for layer, x in zip(model.model.layers, inputs)]
    with _watch(model, entries) as watch:
        with watch.record(rows):
            actual = [layer.mlp(x) for layer, x in zip(model.model.layers, inputs)]
        snapshot = watch.snapshot()
        events = watch.events(snapshot)

    assert all(torch.equal(a, b) for a, b in zip(actual, baseline))
    assert snapshot.rows == tuple(rows)
    assert snapshot.selected_rows == (0, 1)
    assert len(snapshot.samples) == 2
    for sample, x in zip(snapshot.samples, inputs):
        layer = model.model.layers[sample.layer]
        gate_up = F.linear(x, layer.mlp.gate_up_proj.weight).float()
        gate, up = gate_up.chunk(2, dim=-1)
        full_activation = (F.silu(gate) * up).to(dtype)
        expected = full_activation[:, sample.channels]
        assert torch.equal(sample.activations, expected)
        norms = torch.linalg.vector_norm(
            layer.mlp.down_proj.weight[:, sample.channels].float(), dim=0
        )
        expected_contribution = expected.float().abs() * norms
        assert torch.allclose(sample.contribution_estimates, expected_contribution)
        assert sample.activations.shape == (2, len(sample.channels))
        assert sample.activations.data_ptr() != full_activation.data_ptr()

    assert events
    assert all(event.channel != 3 for event in events)
    assert {event.layer for event in events} <= {0, 1}
    assert all(
        event.request_id == rows[event.physical_row].request_id for event in events
    )
    assert all(
        event.token_position == rows[event.physical_row].token_position
        for event in events
    )
    assert all(
        event.contribution_estimate > event.contribution_limit for event in events
    )


def test_selected_rows_preserve_physical_identity_and_output(monkeypatch):
    model = _model(monkeypatch)
    layer = model.model.layers[1]
    rows = [
        RowIdentity("request-0", 10),
        RowIdentity("request-1", 11),
        RowIdentity("request-2", 12),
    ]
    entries = [
        {"layer": 1, "channel": 1, "contribution_limit": 1e-12},
        {"layer": 1, "channel": 4, "contribution_limit": 1e-12},
    ]
    x = torch.tensor([[1.0] * 64, [2.0] * 64, [3.0] * 64])
    with torch.no_grad():
        baseline = layer.mlp(x)
    with _watch(model, entries) as watch:
        with watch.record(rows, selected_rows=(2, 0)):
            actual = layer.mlp(x)
        snapshot = watch.snapshot()
        events = watch.events(snapshot)

    assert torch.equal(actual, baseline)
    assert snapshot.rows == tuple(rows)
    assert snapshot.selected_rows == (2, 0)
    sample = snapshot.samples[0]
    assert sample.activations.shape == (2, 2)
    gate_up = F.linear(x, layer.mlp.gate_up_proj.weight)
    gate, up = gate_up.chunk(2, dim=-1)
    expected = F.silu(gate) * up
    assert torch.equal(sample.activations, expected[[2, 0]][:, [1, 4]])
    assert events
    assert {event.physical_row for event in events} <= {0, 2}
    assert all(
        event.request_id == rows[event.physical_row].request_id for event in events
    )
    assert all(
        event.token_position == rows[event.physical_row].token_position
        for event in events
    )


def test_event_export_skips_quiet_host_copies_and_transfers_only_hits(monkeypatch):
    model = _model(monkeypatch)
    layer = model.model.layers[1]
    x = torch.stack((torch.zeros(64), torch.ones(64)))
    rows = [RowIdentity("quiet", 4), RowIdentity("active", 5)]
    with torch.no_grad():
        gate_up = F.linear(x, layer.mlp.gate_up_proj.weight)
        gate, up = gate_up.chunk(2, dim=-1)
        activation = F.silu(gate) * up
        norm = torch.linalg.vector_norm(layer.mlp.down_proj.weight[:, 1].float())
        active_contribution = float(activation[1, 1].abs() * norm)
    assert active_contribution > 0
    entries = [{"layer": 1, "channel": 1, "contribution_limit": 1e10}]
    with _watch(model, entries) as watch:
        with watch.record(rows):
            layer.mlp(x)
        quiet_snapshot = watch.snapshot()

        with monkeypatch.context() as patch:

            def unexpected_copy(self, *args, **kwargs):
                raise AssertionError("quiet samples must not be copied to host")

            patch.setattr(torch.Tensor, "cpu", unexpected_copy)
            assert watch.events(quiet_snapshot) == ()

    entries[0]["contribution_limit"] = active_contribution / 2
    with _watch(model, entries) as watch:
        with watch.record(rows):
            layer.mlp(x)
        snapshot = watch.snapshot()
        original_cpu = torch.Tensor.cpu
        copied_shapes = []

        def record_copy(tensor, *args, **kwargs):
            copied_shapes.append((tuple(tensor.shape), tensor.dtype))
            return original_cpu(tensor, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", record_copy)
            events = watch.events(snapshot)

    assert len(events) == 1
    assert events[0].physical_row == 1
    assert events[0].request_id == "active"
    assert copied_shapes
    assert all(
        shape != (2, 1) for shape, dtype in copied_shapes if dtype.is_floating_point
    )


@pytest.mark.parametrize("selected_rows", [[], [-1], [2], [True], [1, 1], ["1"]])
def test_invalid_selected_rows_are_rejected(monkeypatch, selected_rows):
    model = _model(monkeypatch)
    entry = {"layer": 0, "channel": 1, "contribution_limit": 1.0}
    rows = [RowIdentity("request-0", 0), RowIdentity("request-1", 0)]
    with _watch(model, [entry]) as watch:
        with pytest.raises(ValueError, match="selected_rows"):
            with watch.record(rows, selected_rows=selected_rows):
                model.model.layers[0].mlp(torch.ones(2, 64))


def test_snapshot_survives_later_forward_and_failed_record(monkeypatch):
    model = _model(monkeypatch)
    entry = {"layer": 1, "channel": 5, "contribution_limit": 1e-6}
    row = [RowIdentity(request_id="one", token_position=1)]
    layer = model.model.layers[1]
    with _watch(model, [entry]) as watch:
        with watch.record(row):
            layer.mlp(torch.ones(1, 64))
        first = watch.snapshot()
        first_values = first.samples[0].activations.clone()
        with watch.record([RowIdentity(request_id="two", token_position=2)]):
            layer.mlp(torch.ones(1, 64) * 2)
        second = watch.snapshot()
        assert torch.equal(first.samples[0].activations, first_values)
        assert first.rows[0].request_id == "one"
        assert second.rows[0].request_id == "two"
        with pytest.raises(RuntimeError, match="bypassed watched MLP hooks"):
            with watch.record(row):
                pass
        with pytest.raises(RuntimeError, match="No completed channel record"):
            watch.snapshot()
        assert watch.events(first) == watch.events(first)


def test_invalid_identity_schema_configuration_and_changed_weights_fail(monkeypatch):
    model = _model(monkeypatch)
    entry = {"layer": 0, "channel": 2, "contribution_limit": 0.1}
    with pytest.raises(ValueError, match="model identity"):
        ChannelWatch(
            model,
            watchlist=_watchlist([entry], identity="other-model"),
            model_identity="test-model",
            execution="eager",
            speculative=False,
            monokernel=False,
        )
    with pytest.raises(ValueError, match="unique and sorted"):
        _watch(model, [entry, entry])
    with pytest.raises(ValueError, match="positive and finite"):
        _watch(model, [{**entry, "contribution_limit": float("nan")}])
    with pytest.raises(ValueError, match="finite in float32"):
        _watch(model, [{**entry, "contribution_limit": 1e100}])
    with pytest.raises(ValueError, match="outside its MLP"):
        _watch(model, [{**entry, "channel": 128}])
    with pytest.raises(ValueError, match="requires eager"):
        _watch(model, [entry], execution="graph")
    with pytest.raises(ValueError, match="requires eager"):
        _watch(model, [entry], monokernel=True)

    with _watch(model, [entry]) as watch:
        with torch.no_grad():
            model.model.layers[0].mlp.down_proj.weight[:, 2].add_(1)
        with pytest.raises(RuntimeError, match="parameters changed"):
            with watch.record([RowIdentity("one", 0)]):
                model.model.layers[0].mlp(torch.ones(1, 64))


def test_unrelated_parameter_change_invalidates_watch(monkeypatch):
    model = _model(monkeypatch)
    entry = {"layer": 1, "channel": 2, "contribution_limit": 1.0}
    with _watch(model, [entry]) as watch:
        with torch.no_grad():
            model.model.layers[0].input_layernorm.weight.add_(1)
        with pytest.raises(RuntimeError, match="parameters changed"):
            with watch.record([RowIdentity("one", 0)]):
                model.model.layers[1].mlp(torch.ones(1, 64))


def test_nonfinite_down_column_norm_is_rejected(monkeypatch):
    model = _model(monkeypatch)
    with torch.no_grad():
        model.model.layers[1].mlp.down_proj.weight[0, 2] = float("inf")
    entry = {"layer": 1, "channel": 2, "contribution_limit": 1.0}
    with pytest.raises(ValueError, match="norms must be finite"):
        _watch(model, [entry])


def test_row_shape_and_repeated_forward_are_rejected(monkeypatch):
    model = _model(monkeypatch)
    entry = {"layer": 0, "channel": 2, "contribution_limit": 1.0}
    layer = model.model.layers[0]
    with _watch(model, [entry]) as watch:
        layer.mlp(torch.ones(1, 64))
        with pytest.raises(RuntimeError, match="No completed channel record"):
            watch.snapshot()
        with pytest.raises(ValueError, match="physical row mapping"):
            with watch.record([RowIdentity("one", 0)]):
                layer.mlp(torch.ones(2, 64))
        with pytest.raises(ValueError, match="physical row mapping"):
            with watch.record([RowIdentity("one", 0)]):
                layer.mlp.act_fn(torch.ones(1, 8))
        with pytest.raises(RuntimeError, match="repeated layer"):
            with watch.record([RowIdentity("one", 0)]):
                layer.mlp(torch.ones(1, 64))
                layer.mlp(torch.ones(1, 64))
