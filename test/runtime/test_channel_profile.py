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

"""Activity profiling through real Qwen3 MLPs and their activation modules."""

import json
import os
import statistics
import sys

import pytest
import torch
import torch.nn.functional as F

from tokenspeed.runtime.configs.qwen3_config import Qwen3Config
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.inspection.channel_profile import ChannelActivityProfiler
from tokenspeed.runtime.layers.mlp import GatedMLPFusion
from tokenspeed.runtime.models.qwen3 import Qwen3ForCausalLM
from tokenspeed.runtime.utils.env import global_server_args_dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")


def _model(monkeypatch, device="cpu"):
    monkeypatch.setitem(global_server_args_dict, "numerics", "auto")
    monkeypatch.setitem(global_server_args_dict, "layer_boundary_norm", "fused")
    monkeypatch.setitem(global_server_args_dict, "logprob_order", "torch")
    monkeypatch.setitem(global_server_args_dict, "moe_combine_order", "rank")
    monkeypatch.setitem(global_server_args_dict, "enable_allreduce_fusion", False)
    hidden_size = 64 if device == "cuda" else 32
    intermediate_size = 128 if device == "cuda" else 48
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden_size // 4,
        max_position_embeddings=32,
    )
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = Qwen3ForCausalLM(config, Mapping(rank=0, world_size=1)).to(
        device=device, dtype=dtype
    )
    torch.manual_seed(41)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 1:
                parameter.fill_(1)
            else:
                parameter.normal_(mean=0, std=0.15)
    model.eval()
    if device == "cpu":
        # Only the unavailable GPU GEMM entry point is replaced. The model,
        # MLP, activation and projections are the actual runtime classes.
        def mm(x, weight, *, bias=None, **kwargs):
            return F.linear(x, weight, bias)

        monkeypatch.setattr("tokenspeed.runtime.layers.dense.unquant.kernel_mm", mm)
    return model


def _profiler(
    model,
    *,
    activation_threshold=0.02,
    histogram_edges=(0.01, 0.1, 1.0),
):
    return ChannelActivityProfiler(
        model,
        model_identity="test-checkpoint-v1",
        execution="eager",
        speculative=False,
        monokernel=False,
        activation_threshold=activation_threshold,
        histogram_edges=histogram_edges,
    )


@torch.no_grad()
def _run_mlps(model, hidden):
    for layer in model.model.layers:
        hidden = layer.mlp(hidden)
    return hidden


@torch.no_grad()
def _independent_activations(model, hidden):
    activations = []
    for layer in model.model.layers:
        gate_up = F.linear(hidden, layer.mlp.gate_up_proj.weight)
        gate, up = gate_up.float().chunk(2, dim=-1)
        activation = (F.silu(gate) * up).to(gate_up.dtype)
        activations.append(activation)
        hidden = F.linear(activation, layer.mlp.down_proj.weight)
    return activations


def test_all_layer_statistics_match_independent_channel_calculation(monkeypatch):
    model = _model(monkeypatch)
    torch.manual_seed(83)
    first = torch.randn(2, 32)
    second = torch.randn(3, 32)
    expected_outputs = [_run_mlps(model, x) for x in (first, second)]
    independent = [_independent_activations(model, x) for x in (first, second)]

    with _profiler(model) as profiler:
        actual_outputs = []
        for hidden in (first, second):
            with profiler.record():
                actual_outputs.append(_run_mlps(model, hidden))
        profile = profiler.profile()

    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(actual_outputs, expected_outputs)
    )
    assert profile["record_count"] == 2
    assert profile["schema"] == "channel-activity-profile-v1"
    assert (
        profile["histogram_edges"]
        == torch.tensor([0.01, 0.1, 1.0], dtype=torch.float32).tolist()
    )
    assert (
        profile["activation_threshold"]
        == torch.tensor(0.02, dtype=torch.float32).item()
    )
    assert len(profile["layers"]) == 2
    json.dumps(profile)

    for layer_index, layer_profile in enumerate(profile["layers"]):
        activity = torch.cat([run[layer_index] for run in independent]).float().abs()
        down = model.model.layers[layer_index].mlp.down_proj.weight.float()
        norms = torch.linalg.vector_norm(down, dim=0).tolist()
        norm_median = statistics.median(norms)
        contribution = activity * torch.linalg.vector_norm(down, dim=0)
        bins = torch.bucketize(activity, torch.tensor(profile["histogram_edges"]))
        assert layer_profile["down_norm_median"] == pytest.approx(norm_median)
        assert len(layer_profile["channels"]) == activity.shape[1]
        for channel_index, row in enumerate(layer_profile["channels"]):
            assert row["layer"] == layer_index
            assert row["channel"] == channel_index
            assert row["sample_count"] == 5
            assert row["mean_abs"] == pytest.approx(
                activity[:, channel_index].mean().item(), abs=1e-7
            )
            assert row["rms"] == pytest.approx(
                activity[:, channel_index].square().mean().sqrt().item(), abs=1e-7
            )
            assert row["max_abs"] == pytest.approx(
                activity[:, channel_index].max().item(), abs=1e-7
            )
            assert row["active_count"] == int(
                (activity[:, channel_index] > 0.02).sum().item()
            )
            assert row["active_fraction"] == pytest.approx(
                (activity[:, channel_index] > 0.02).float().mean().item()
            )
            assert row["down_column_norm"] == pytest.approx(
                torch.linalg.vector_norm(down[:, channel_index]).item(), abs=1e-7
            )
            assert row["down_norm_ratio_to_layer_median"] == pytest.approx(
                norms[channel_index] / norm_median
            )
            assert row["mean_contribution"] == pytest.approx(
                contribution[:, channel_index].mean().item(), abs=1e-7
            )
            assert row["max_contribution"] == pytest.approx(
                contribution[:, channel_index].max().item(), abs=1e-7
            )
            assert (
                row["histogram_counts"]
                == torch.bincount(bins[:, channel_index], minlength=4).tolist()
            )


def test_watchlist_is_frozen_from_profile_without_target_labels(monkeypatch):
    model = _model(monkeypatch)
    with torch.no_grad():
        for layer_index, channel_index in ((0, 0), (1, 3)):
            gate_up = model.model.layers[layer_index].mlp.gate_up_proj.weight
            width = model.model.layers[layer_index].mlp.down_proj.weight.shape[1]
            gate_up[channel_index].zero_()
            gate_up[channel_index + width].zero_()
    hidden = torch.randn(5, 32)
    with _profiler(model) as profiler:
        with profiler.record():
            _run_mlps(model, hidden)
        ranked = profiler.rank_candidates(limit=4)
        assert [(row["layer"], row["channel"]) for row in ranked[:2]] == [
            (0, 0),
            (1, 3),
        ]
        assert ranked == profiler.rank_candidates(limit=None)[:4]
        watchlist = profiler.freeze_watchlist(
            max_active_fraction=0.0,
            max_mean_contribution=0.0,
            max_observed_contribution=0.0,
            min_down_norm_ratio=1000.0,
            max_outlier_active_fraction=0.5,
            min_observed_rows=5,
            runtime_multiplier=2.0,
            contribution_floor=0.25,
        )
    assert watchlist == {
        "schema": "channel-watchlist-v1",
        "model_identity": "test-checkpoint-v1",
        "entries": [
            {"layer": 0, "channel": 0, "contribution_limit": 0.25},
            {"layer": 1, "channel": 3, "contribution_limit": 0.25},
        ],
    }
    json.dumps(watchlist)


def test_large_down_column_norm_can_select_a_moderately_active_channel(monkeypatch):
    model = _model(monkeypatch)
    mlp = model.model.layers[0].mlp
    width = mlp.down_proj.weight.shape[1]
    with torch.no_grad():
        mlp.gate_up_proj.weight[5].zero_()
        mlp.gate_up_proj.weight[width + 5].zero_()
        mlp.gate_up_proj.weight[5, 0] = 1.0
        mlp.gate_up_proj.weight[width + 5, 0] = 1.0
        mlp.down_proj.weight[:, 5].fill_(10.0)
    hidden = torch.zeros(4, 32)
    hidden[:, 0] = torch.tensor([0.1, 0.1, 0.2, 0.2])
    with _profiler(model) as profiler:
        with profiler.record():
            _run_mlps(model, hidden)
        profile = profiler.profile()
        row = profile["layers"][0]["channels"][5]
        assert row["active_fraction"] == pytest.approx(0.5)
        assert row["down_norm_ratio_to_layer_median"] > 10
        assert row["max_contribution"] > 0
        watchlist = profiler.freeze_watchlist(
            max_active_fraction=0.0,
            max_mean_contribution=0.0,
            max_observed_contribution=0.0,
            min_down_norm_ratio=10.0,
            max_outlier_active_fraction=0.5,
            min_observed_rows=4,
            runtime_multiplier=2.0,
            contribution_floor=0.25,
        )
    selected = {
        (entry["layer"], entry["channel"]): entry for entry in watchlist["entries"]
    }
    assert (0, 5) in selected
    assert selected[(0, 5)]["contribution_limit"] == pytest.approx(
        row["max_contribution"] * 2
    )


def test_zero_layer_median_has_undefined_ratio_and_skips_outlier_gate(monkeypatch):
    model = _model(monkeypatch)
    mlp = model.model.layers[0].mlp
    with torch.no_grad():
        mlp.down_proj.weight.zero_()
        mlp.down_proj.weight[:, 5].fill_(10.0)
    with _profiler(model) as profiler:
        with profiler.record():
            _run_mlps(model, torch.randn(4, 32))
        layer = profiler.profile()["layers"][0]
        assert layer["down_norm_median"] == 0
        assert layer["channels"][5]["down_norm_ratio_to_layer_median"] is None
        assert layer["channels"][5]["mean_contribution"] > 0
        watchlist = profiler.freeze_watchlist(
            max_active_fraction=0.0,
            max_mean_contribution=0.0,
            max_observed_contribution=0.0,
            min_down_norm_ratio=10.0,
            max_outlier_active_fraction=1.0,
            min_observed_rows=4,
            runtime_multiplier=2.0,
            contribution_floor=0.25,
        )
    assert (0, 5) not in {
        (entry["layer"], entry["channel"]) for entry in watchlist["entries"]
    }


@pytest.mark.parametrize(
    "ratio,fraction",
    [(0.0, 0.5), (-1.0, 0.5), (float("inf"), 0.5), (1.0, -0.1), (1.0, 1.1)],
)
def test_invalid_outlier_thresholds_are_rejected(monkeypatch, ratio, fraction):
    model = _model(monkeypatch)
    with _profiler(model) as profiler:
        with profiler.record():
            _run_mlps(model, torch.randn(4, 32))
        with pytest.raises(ValueError, match="norm ratio|fraction|min_down_norm_ratio"):
            profiler.freeze_watchlist(
                max_active_fraction=0.0,
                max_mean_contribution=0.0,
                max_observed_contribution=0.0,
                min_down_norm_ratio=ratio,
                max_outlier_active_fraction=fraction,
                min_observed_rows=4,
                runtime_multiplier=2.0,
                contribution_floor=0.25,
            )


def test_selected_physical_rows_are_measured_without_changing_forward(monkeypatch):
    model = _model(monkeypatch)
    torch.manual_seed(99)
    hidden = torch.randn(4, 32)
    expected = _run_mlps(model, hidden)
    independent = _independent_activations(model, hidden)

    with _profiler(model) as profiler:
        with profiler.record(row_indices=(3,)):
            actual = _run_mlps(model, hidden)
        profile = profiler.profile()
        assert torch.equal(actual, expected)
        assert profile["physical_rows_seen"] == 4
        assert profile["selected_row_records"] == 1
        for layer_index, layer in enumerate(profile["layers"]):
            measured = independent[layer_index][3].float().abs()
            for channel_index, row in enumerate(layer["channels"]):
                assert row["sample_count"] == 1
                assert row["mean_abs"] == pytest.approx(
                    measured[channel_index].item(), abs=1e-7
                )

        with profiler.record():
            _run_mlps(model, hidden[:2])
        mixed = profiler.profile()
        assert mixed["physical_rows_seen"] == 6
        assert mixed["selected_row_records"] == 1
        assert mixed["layers"][0]["channels"][0]["sample_count"] == 3


@pytest.mark.parametrize("rows", [(), (-1,), (2, 1), (1, 1), (True,)])
def test_invalid_row_selection_is_rejected(monkeypatch, rows):
    model = _model(monkeypatch)
    with _profiler(model) as profiler:
        with pytest.raises(ValueError, match="row_indices"):
            with profiler.record(row_indices=rows):
                _run_mlps(model, torch.randn(3, 32))
        with pytest.raises(RuntimeError, match="complete"):
            profiler.profile()


def test_row_selection_outside_forward_fails_without_committing(monkeypatch):
    model = _model(monkeypatch)
    hidden = torch.randn(3, 32)
    with _profiler(model) as profiler:
        with profiler.record():
            _run_mlps(model, hidden)
        before = profiler.profile()
        with pytest.raises(ValueError, match="outside the physical forward"):
            with profiler.record(row_indices=(3,)):
                _run_mlps(model, hidden)
        assert profiler.profile() == before


def test_layers_with_different_physical_row_counts_fail(monkeypatch):
    model = _model(monkeypatch)
    hidden = torch.randn(3, 32)
    with _profiler(model) as profiler:
        with pytest.raises(ValueError, match="different physical row counts"):
            with profiler.record(row_indices=(0,)):
                model.model.layers[0].mlp(hidden)
                model.model.layers[1].mlp(hidden[:2])
        with pytest.raises(RuntimeError, match="complete"):
            profiler.profile()


def test_incomplete_record_does_not_change_previous_statistics(monkeypatch):
    model = _model(monkeypatch)
    hidden = torch.randn(3, 32)
    with _profiler(model) as profiler:
        with profiler.record():
            _run_mlps(model, hidden)
        before = profiler.profile()
        with pytest.raises(RuntimeError, match="bypassed"):
            with profiler.record():
                model.model.layers[0].mlp(hidden)
        assert profiler.profile() == before
        with pytest.raises(RuntimeError, match="repeated"):
            with profiler.record():
                model.model.layers[0].mlp(hidden)
                model.model.layers[0].mlp(hidden)
        assert profiler.profile() == before
    with pytest.raises(RuntimeError, match="attached"):
        with profiler.record():
            pass


@pytest.mark.parametrize(
    "choices", [{"execution": "captured"}, {"speculative": True}, {"monokernel": True}]
)
def test_unsupported_execution_is_rejected(monkeypatch, choices):
    model = _model(monkeypatch)
    args = {
        "model_identity": "test-checkpoint-v1",
        "execution": "eager",
        "speculative": False,
        "monokernel": False,
        "activation_threshold": 0.02,
        "histogram_edges": (0.01, 0.1),
    }
    args.update(choices)
    with pytest.raises(ValueError, match="requires eager"):
        ChannelActivityProfiler(model, **args)


def test_fused_layer_is_rejected_before_installing_hooks(monkeypatch):
    model = _model(monkeypatch)
    model.model.layers[0].mlp.fusion = GatedMLPFusion.MXFP4_GEMM_SWIGLU
    with pytest.raises(ValueError, match="unfused"):
        _profiler(model)


@pytest.mark.parametrize("edges", [(1e100,), (1e-50,), (1.0, 1.0 + 1e-9)])
def test_histogram_edges_must_remain_distinct_in_float32(monkeypatch, edges):
    model = _model(monkeypatch)
    with pytest.raises(ValueError, match="float32"):
        _profiler(model, histogram_edges=edges)


@pytest.mark.parametrize("threshold", [1e100, 1e-50])
def test_activation_threshold_must_fit_in_float32(monkeypatch, threshold):
    model = _model(monkeypatch)
    with pytest.raises(ValueError, match="float32"):
        _profiler(model, activation_threshold=threshold)


def test_profile_requires_complete_data_and_explicit_thresholds(monkeypatch):
    model = _model(monkeypatch)
    profiler = _profiler(model)
    with pytest.raises(RuntimeError, match="complete"):
        profiler.profile()
    with pytest.raises(RuntimeError, match="attached"):
        with profiler.record():
            pass
    with profiler:
        with profiler.record():
            _run_mlps(model, torch.randn(2, 32))
    with pytest.raises(ValueError, match="min_observed_rows"):
        profiler.freeze_watchlist(
            max_active_fraction=0.1,
            max_mean_contribution=0.1,
            max_observed_contribution=0.1,
            min_down_norm_ratio=10.0,
            max_outlier_active_fraction=0.5,
            min_observed_rows=0,
            runtime_multiplier=2.0,
            contribution_floor=0.1,
        )
    with pytest.raises(ValueError, match="fewer rows"):
        profiler.freeze_watchlist(
            max_active_fraction=0.1,
            max_mean_contribution=0.1,
            max_observed_contribution=0.1,
            min_down_norm_ratio=10.0,
            max_outlier_active_fraction=0.5,
            min_observed_rows=3,
            runtime_multiplier=2.0,
            contribution_floor=0.1,
        )


@pytest.mark.parametrize("changed", ["mlp", "attention"])
def test_parameter_update_invalidates_profile_and_future_records(monkeypatch, changed):
    model = _model(monkeypatch)
    hidden = torch.randn(2, 32)
    with _profiler(model) as profiler:
        with profiler.record():
            _run_mlps(model, hidden)
        parameter = (
            model.model.layers[1].mlp.down_proj.weight
            if changed == "mlp"
            else model.model.layers[0].self_attn.qkv_proj.weight
        )
        with torch.no_grad():
            parameter.view(-1)[0].add_(0.1)
        with pytest.raises(RuntimeError, match="parameters changed"):
            profiler.profile()
        with pytest.raises(RuntimeError, match="parameters changed"):
            with profiler.record():
                _run_mlps(model, hidden)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_gpu_selected_kernels_preserve_outputs(monkeypatch):
    model = _model(monkeypatch, device="cuda")
    hidden = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)
    expected = _run_mlps(model, hidden)
    with _profiler(model) as profiler:
        with profiler.record():
            actual = _run_mlps(model, hidden)
        profile = profiler.profile()
    assert torch.equal(actual, expected)
    assert all(layer["channels"][0]["sample_count"] == 3 for layer in profile["layers"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
