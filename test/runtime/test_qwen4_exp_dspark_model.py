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

"""Projection and capture-contract coverage for Qwen4-Exp DSpark drafts."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.execution.drafter.dflash import DFlash
from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
from tokenspeed.runtime.models import dflash as dflash_model
from tokenspeed.runtime.models.dspark import (
    DSparkDraftModel,
    HyperDSparkDraftModel,
    _get_markov_params,
)
from tokenspeed.runtime.utils.env import global_server_args_dict


class _TorchRMSNorm(nn.RMSNorm):
    def forward(self, x):
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(x.dtype)


@pytest.fixture
def make_draft():
    # The projector and real weight loader need no distributed decoder or GPU.
    # Leave the FC as a real ReplicatedLinear and replace only its GEMM leaf.
    with (
        mock.patch.object(
            dflash_model.DFlashDraftModel,
            "decoder_layer_cls",
            side_effect=lambda **kwargs: nn.Identity(),
        ),
        mock.patch.object(dflash_model, "RMSNorm", _TorchRMSNorm),
        mock.patch.object(
            UnquantizedLinearMethod,
            "apply",
            new=lambda self, layer, x, bias=None: F.linear(x, layer.weight, bias),
        ),
    ):

        def build(model_cls, *, dtype=torch.float32, **overrides):
            fields = dict(
                hidden_size=8,
                num_hidden_layers=1,
                vocab_size=32,
                rms_norm_eps=1e-6,
                block_size=7,
                target_hidden_size=8,
                dflash_config=dict(
                    target_layer_ids=[1, 3],
                    hc_count=4,
                    hc_lowrank=3,
                    markov_rank=4,
                    markov_head_type="vanilla",
                ),
            )
            fields.update(overrides)
            return model_cls(SimpleNamespace(**fields), SimpleNamespace()).to(dtype)

        yield build


def _reference_projection(model, hidden):
    reduced = []
    width = model.hc_count * model.config.hidden_size
    for reducer, tap in zip(model.hc_reducers, hidden.split(width, dim=-1)):
        branches = []
        for branch in tap.split(model.config.hidden_size, dim=-1):
            branches.append(
                branch.float()
                / torch.sqrt(
                    branch.float().square().mean(-1, keepdim=True) + reducer.eps
                )
            )
        normalized = (
            torch.cat(branches, dim=-1) * (1.0 + reducer.hc_norm_weight.float())
        ).to(hidden.dtype)
        lowrank = F.linear(normalized, reducer.input_mix_weight_down.weight)
        gates = torch.sigmoid(
            F.linear(
                F.silu(lowrank / model.hc_count), reducer.input_mix_weight_up.weight
            )
        )
        weighted = (normalized * gates).reshape(
            *hidden.shape[:-1], model.hc_count, model.config.hidden_size
        )
        reduced.append(weighted.mean(dim=-2))
    projected = F.linear(torch.cat(reduced, dim=-1), model.fc.weight)
    return model.hidden_norm(projected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(8 * 4 * 2,), (5, 8 * 4 * 2)])
def test_hyper_projection_matches_trained_reducer(make_draft, dtype, shape):
    torch.manual_seed(17)
    model = make_draft(HyperDSparkDraftModel, dtype=dtype)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.3)
    hidden = torch.randn(shape, dtype=dtype)
    # Different branch magnitudes distinguish grouped RMS from one wide RMS.
    hidden[..., :8] *= 12
    expected = _reference_projection(model, hidden)
    actual = model.project_target_hidden(hidden)
    tolerance = 2e-3 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert actual.shape == (*shape[:-1], 8)
    assert model.fc.input_size == 16
    assert model.context_in_features == 64
    assert not model.supports_incremental_target_projection


def test_hyper_checkpoint_reducers_and_markov_weights_load(make_draft):
    model = make_draft(HyperDSparkDraftModel)
    generator = torch.Generator().manual_seed(23)
    expected = {
        name: torch.randn(parameter.shape, generator=generator)
        for name, parameter in model.named_parameters()
    }
    # Both exported wrapper prefixes are supported by the inherited loader.
    checkpoint = [
        (f"model.{name}" if index % 2 else name, weight)
        for index, (name, weight) in enumerate(expected.items())
    ]
    model.load_weights(reversed(checkpoint))
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, expected[name], atol=0, rtol=0)
    hidden = torch.randn(3, 64, generator=generator)
    torch.testing.assert_close(
        model.project_target_hidden(hidden), _reference_projection(model, hidden)
    )


@pytest.mark.parametrize("model_cls", [DSparkDraftModel, HyperDSparkDraftModel])
def test_optional_checkpoint_vocabulary_tables_load_streaming(make_draft, model_cls):
    model = make_draft(
        model_cls, dtype=torch.bfloat16, vocab_size=128, logit_scale=0.75
    )
    model.mapping = SimpleNamespace(
        attn=SimpleNamespace(tp_rank=1, tp_size=2, tp_group=(0, 1), has_dp=True),
        lm_head=SimpleNamespace(tp_rank=0, tp_size=2, tp_group=(0, 2), has_tp=True),
    )
    embedding = torch.arange(128 * 8, dtype=torch.float32).reshape(128, 8) / 128
    head = embedding.flip(0).contiguous()
    projector = torch.randn_like(model.fc.weight)

    def checkpoint():
        assert model.embed_tokens is None and model.lm_head is None
        yield "model.fc.weight", projector
        # The preceding tensor must be consumed before advancing the stream.
        torch.testing.assert_close(model.fc.weight, projector, atol=0, rtol=0)
        yield "model.embed_tokens.weight", embedding
        torch.testing.assert_close(model.embed_tokens.weight, embedding[64:].bfloat16())
        yield "lm_head.weight", head
        torch.testing.assert_close(model.lm_head.weight, head[:64].bfloat16())
        yield "t2d", torch.arange(128)

    with mock.patch.dict(global_server_args_dict, {"logprob_order": "torch"}):
        model.load_weights(checkpoint())
    assert model.embed_tokens.tp_group == (0, 1)
    assert model.lm_head.tp_group == (0, 2)
    assert model.embed_tokens.weight.device == model.fc.weight.device
    assert model.lm_head.weight.device == model.fc.weight.device
    assert model.logits_processor.tp_group == (0, 2)
    assert model.logits_processor.dp_lm_head_tp
    assert model.logits_processor.logit_scale == 0.75


@pytest.mark.parametrize("name", ["d2t", "model.draft_id_to_target_id"])
def test_reduced_vocabulary_checkpoint_mapping_is_rejected(make_draft, name):
    model = make_draft(DSparkDraftModel)
    with pytest.raises(ValueError, match="reduced vocabulary mapping"):
        model.load_weights([(name, torch.arange(16))])


def test_optional_vocabulary_table_requires_full_checkpoint_vocab(make_draft):
    model = make_draft(DSparkDraftModel)
    with pytest.raises(ValueError, match="full-vocabulary shape"):
        model.load_weights([("lm_head.weight", torch.randn(16, 8))])
    assert model.lm_head is None


@pytest.mark.parametrize("fused", [False, True])
def test_native_context_write_projects_raw_hc_before_writing_kv(make_draft, fused):
    model = make_draft(HyperDSparkDraftModel)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.3)
    hidden = torch.randn(3, 64, dtype=torch.float64)
    positions = torch.tensor([11, 12, 13])
    locations = torch.tensor([24, 25, 26])
    expected = _reference_projection(model, hidden.to(model.context_dtype))
    native_write = mock.Mock()
    fused_write = mock.Mock()
    pool = object()
    drafter = SimpleNamespace(
        draft_model_runner=SimpleNamespace(model=model),
        device="cpu",
        _fused_kv_enabled=True,
        _write_native_cache_fused=fused_write,
        token_to_kv_pool=pool,
    )
    with mock.patch.object(model, "write_context_kv", native_write):
        DFlash._write_native_cache(
            drafter, hidden, positions, locations, decode_only=fused
        )
    writer = fused_write if fused else native_write
    writer.assert_called_once()
    projected, actual_positions, actual_locations, *actual_pool = writer.call_args.args
    torch.testing.assert_close(projected, expected)
    assert projected.shape == (3, 8)
    assert actual_positions is positions
    assert actual_locations is locations
    if fused:
        native_write.assert_not_called()
    else:
        assert actual_pool == [pool]
        fused_write.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_hyper_reducer_cuda_graph_replay_matches_reference(make_draft, dtype):
    """Exercise reducer graph compatibility with the fixture's PyTorch GEMM leaf."""
    model = make_draft(HyperDSparkDraftModel, dtype=dtype).cuda()
    for parameter in model.parameters():
        parameter.normal_(mean=0.0, std=0.3)
    hidden = torch.randn(5, 64, device="cuda", dtype=dtype)
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            model.project_target_hidden(hidden)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        projected = model.project_target_hidden(hidden)
    hidden.copy_(torch.randn_like(hidden))
    graph.replay()
    expected = _reference_projection(model, hidden)
    tolerance = 2e-3 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(projected, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize(
    ("model_cls", "capture_hc"),
    [(DSparkDraftModel, False), (HyperDSparkDraftModel, True)],
)
@pytest.mark.parametrize("multimodal", [False, True])
def test_qwen4_capture_uses_checkpoint_taps_including_last_layer(
    make_draft, model_cls, capture_hc, multimodal
):
    model = make_draft(model_cls)
    text_config = SimpleNamespace(
        model_type="qwen4_exp_text",
        hidden_size=8,
        hc_count=4,
        num_hidden_layers=4,
        vocab_size=32,
    )
    config = (
        SimpleNamespace(model_type="qwen4_exp", text_config=text_config)
        if multimodal
        else text_config
    )
    target = SimpleNamespace(set_dspark_layers_to_capture=mock.Mock())
    model.configure_target(target, config)
    target.set_dspark_layers_to_capture.assert_called_once_with(
        [1, 3], capture_hc=capture_hc
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("hidden_size", 12, "hidden_size"), ("hc_count", 2, "hc_count")],
)
def test_hyper_capture_rejects_incompatible_target_geometry(
    make_draft, field, value, message
):
    model = make_draft(HyperDSparkDraftModel)
    fields = dict(model_type="qwen4_exp_text", hidden_size=8, hc_count=4, vocab_size=32)
    fields[field] = value
    target = SimpleNamespace(set_dspark_layers_to_capture=mock.Mock())
    with pytest.raises(ValueError, match=message):
        model.configure_target(target, SimpleNamespace(**fields))
    target.set_dspark_layers_to_capture.assert_not_called()


def test_qwen4_capture_rejects_incompatible_residual_stream(make_draft):
    model = make_draft(
        DSparkDraftModel,
        dflash_config=dict(
            target_layer_ids=[1, 3], markov_rank=4, aux_hidden_stream="attn_res"
        ),
    )
    target = SimpleNamespace(set_dspark_layers_to_capture=mock.Mock())
    with pytest.raises(ValueError, match="aux_hidden_stream"):
        model.configure_target(
            target, SimpleNamespace(model_type="qwen4_exp_text", hidden_size=8)
        )
    target.set_dspark_layers_to_capture.assert_not_called()


@pytest.mark.parametrize(
    ("checkpoint_fields", "target_vocab_size", "message"),
    [
        ({"target_hidden_size": 12}, 32, "target_hidden_size"),
        ({}, 64, "vocab_size"),
        ({"draft_vocab_size": 16}, 32, "draft_vocab_size"),
        ({"num_target_layers": 5}, 32, "num_target_layers"),
    ],
)
def test_qwen4_capture_rejects_incompatible_checkpoint_contract(
    make_draft, checkpoint_fields, target_vocab_size, message
):
    model = make_draft(DSparkDraftModel, **checkpoint_fields)
    target = SimpleNamespace(set_dspark_layers_to_capture=mock.Mock())
    config = SimpleNamespace(
        model_type="qwen4_exp_text",
        hidden_size=8,
        hc_count=4,
        vocab_size=target_vocab_size,
        num_hidden_layers=4,
    )
    with pytest.raises(ValueError, match=message):
        model.configure_target(target, config)
    target.set_dspark_layers_to_capture.assert_not_called()


def test_markov_params_read_dflash_config_with_dspark_precedence():
    config = SimpleNamespace(
        markov_rank=2,
        dflash_config=dict(markov_rank=4, markov_head_type="VANILLA"),
        dspark_config=dict(markov_rank=8),
    )
    assert _get_markov_params(config) == (8, "vanilla")
    config.dspark_config.clear()
    assert _get_markov_params(config) == (4, "vanilla")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
