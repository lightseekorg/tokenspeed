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

"""DFlash table sizing and the executor's raw-position bound."""

import functools
from test.ci_system.ci_register import register_cuda_ci
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.models.dflash import DFlashDraftModel, _draft_rope_cache_limit

register_cuda_ci(est_time=5, suite="runtime-1gpu")


@pytest.mark.parametrize(
    ("scaling", "context", "width", "expected"),
    [
        ({"rope_type": "deepseek_yarn"}, 512000, 8, 512032),
        ({"type": "deepseek_yarn"}, 512000, 8, 512032),
        ({"rope_type": "default", "type": "deepseek_yarn"}, 512000, 8, None),
        (None, 512000, 8, None),
        ({"type": "deepseek_yarn"}, None, 8, None),
        ({"type": "deepseek_yarn"}, 512000, None, None),
    ],
)
def test_draft_rope_cache_limit(monkeypatch, scaling, context, width, expected):
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "max_model_len", context)
    monkeypatch.setitem(global_server_args_dict, "speculative_num_draft_tokens", width)
    assert _draft_rope_cache_limit(scaling) == expected


def _draft(layers: int, rope_parameters: dict | None = None) -> DFlashDraftModel:
    config = PretrainedConfig(
        hidden_size=256,
        num_hidden_layers=layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=128,
        intermediate_size=512,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=4096,
        dflash_config={"target_layer_ids": [0], "block_size": 8},
    )
    if rope_parameters is not None:
        config.rope_parameters = rope_parameters
    return DFlashDraftModel(config, Mapping(rank=0, world_size=1, attn_tp_size=1))


def test_layerless_draft_reports_no_rope_table():
    from tokenspeed.runtime.execution.model_executor import RopeTableModel

    model = _draft(layers=0)
    assert isinstance(model, RopeTableModel)
    assert model.rope_positions is None


@pytest.mark.parametrize(("context", "rows"), [(32, 64), (None, 128)])
def test_draft_reports_its_yarn_table_to_the_executor(
    monkeypatch, caplog, context, rows
):
    import tokenspeed.runtime.execution.model_executor as executor
    import tokenspeed.runtime.layers.rotary_embedding as rotary
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setattr(rotary, "_ROPE_DICT", {})
    if not torch.cuda.is_available():
        monkeypatch.setattr(
            rotary,
            "DeepseekScalingRotaryEmbedding",
            functools.partial(rotary.DeepseekScalingRotaryEmbedding, device="cpu"),
        )
    monkeypatch.setitem(global_server_args_dict, "yarn_ramp_mask_device", "cpu")
    monkeypatch.setitem(global_server_args_dict, "yarn_frequency_device", "cpu")
    monkeypatch.setitem(global_server_args_dict, "max_model_len", context)
    monkeypatch.setitem(global_server_args_dict, "speculative_num_draft_tokens", 8)
    model = _draft(
        layers=2,
        rope_parameters={
            "rope_theta": 10000.0,
            "rope_type": "deepseek_yarn",
            "original_max_position_embeddings": 64,
            "factor": 2.0,
        },
    )
    assert isinstance(model, executor.RopeTableModel)
    for layer in model.layers:
        assert layer.self_attn.rotary_emb.cos_sin_cache.shape[0] == rows
    assert model.rope_positions == rows

    monkeypatch.setattr(executor.logger, "propagate", True)
    config = SimpleNamespace(hf_text_config=model.config, is_draft_worker=True)
    caplog.clear()
    with caplog.at_level("WARNING"):
        for physical in (rows - 1, rows):
            executor.warn_past_rope_table(model, config, 32, physical)
        assert caplog.text == ""
        executor.warn_past_rope_table(model, config, 32, rows + 1)
    assert f"{rows} positions of the draft model's rope tables" in caplog.text


def test_draft_reports_addressable_longrope_positions(monkeypatch):
    import tokenspeed.runtime.layers.rotary_embedding as rotary

    monkeypatch.setattr(rotary, "_ROPE_DICT", {})
    model = _draft(
        layers=1,
        rope_parameters={
            "rope_type": "longrope",
            "rope_theta": 10000.0,
            "original_max_position_embeddings": 64,
            "short_factor": [1.0] * 64,
            "long_factor": [2.0] * 64,
        },
    )
    assert model.rope_positions == model.config.max_position_embeddings
