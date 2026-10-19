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

"""Read a cached Muse Spark checkpoint's config and shard headers, never its data."""

from __future__ import annotations

import glob
import json
import os
import sys

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.configs.muse_spark_config import (
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    MuseSparkConfig,
)
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.quantization.nvfp4 import Nvfp4Config
from tokenspeed.runtime.models.muse_spark import (
    MuseSparkForConditionalGeneration,
    resolve_parameter_name,
)

_TEXT_OVERRIDES = {
    "vocab_size": 256,
    "hidden_size": 256,
    "head_dim": 32,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "moe_hidden_size": 128,
    "moe_intermediate_size": 128,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
}
_VISION_OVERRIDES = {"hidden_size": 64, "intermediate_size": 32}
_AUDIO_OVERRIDES = {"hidden_sizes": [8, 16, 32, 64], "out_hidden_size": 256}


def _single_rank_mapping() -> Mapping:
    return Mapping(
        rank=0,
        world_size=1,
        attn_tp_size=1,
        attn_kvp_size=1,
        attn_qcp_size=1,
        attn_dp_size=1,
        dense_tp_size=1,
        dense_dp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        moe_dp_size=1,
        vision_tp_size=1,
        vision_dp_size=1,
        linear_attn_tp_size=1,
        pp_size=1,
        pp_layer_partition=None,
        nprocs_per_node=1,
        nnodes=1,
        base_gpu_id=0,
        gpu_id_step=1,
    )


def _snapshot() -> str:
    """Return a cached Muse Spark snapshot directory, skipping the test without one."""
    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    pattern = os.path.join(home, "hub", "models--*Muse-Spark*", "snapshots", "*")
    for candidate in sorted(glob.glob(pattern)):
        config_path = os.path.join(candidate, "config.json")
        if not os.path.exists(config_path):
            continue
        with open(config_path) as handle:
            if json.load(handle).get("model_type") == "muse_spark":
                return candidate
    pytest.skip("no Muse Spark checkpoint in the Hugging Face cache")


def _raw_config(snapshot: str) -> dict:
    with open(os.path.join(snapshot, "config.json")) as handle:
        return json.load(handle)


def _checkpoint_tensor_names(snapshot: str) -> list[str]:
    """Return every tensor name in the snapshot, reading shard headers only."""
    from safetensors import safe_open

    names: list[str] = []
    for shard in sorted(glob.glob(os.path.join(snapshot, "*.safetensors"))):
        with safe_open(shard, framework="pt") as handle:
            names.extend(handle.keys())
    return names


def test_the_checkpoint_config_builds_the_runtime_view() -> None:
    snapshot = _snapshot()
    raw = _raw_config(snapshot)
    config = MuseSparkConfig(**raw)
    text = raw["text_config"]
    assert config.architectures == raw["architectures"]
    assert config.num_hidden_layers == text["num_hidden_layers"]
    assert config.head_dim == text["head_dim"]
    assert config.layer_types == text["layer_types"]
    assert config.num_local_experts == text["num_local_experts"]
    assert config.num_experts_per_tok == text["num_experts_per_tok"]
    assert config.moe_intermediate_size == text["moe_intermediate_size"]
    assert config.output_multiplier == text["output_multiplier"]
    windows = {
        FULL_ATTENTION: None,
        SLIDING_ATTENTION: text["sliding_window"],
    }
    assert config.sliding_window == tuple(
        windows[layer_type] for layer_type in text["layer_types"]
    )
    assert [config.uses_rope(i) for i in range(config.num_hidden_layers)] == [
        theta != 0 for theta in text["layer_rope_theta"]
    ]
    assert len(config.cache_layer_types) == config.num_hidden_layers


def test_the_quantization_config_keeps_the_modules_it_excludes() -> None:
    raw = _raw_config(_snapshot())["quantization_config"]
    quant_config = Nvfp4Config.from_config(raw)
    assert quant_config.exclude_modules == raw["exclude_modules"]
    assert quant_config.group_size == raw["group_size"]


def _narrow_config(raw: dict) -> MuseSparkConfig:
    """Keep the checkpoint's layer counts and schedule, shrink every width."""
    return MuseSparkConfig(
        text_config={**raw["text_config"], **_TEXT_OVERRIDES},
        vision_config={**raw["vision_config"], **_VISION_OVERRIDES},
        audio_config={**raw["audio_config"], **_AUDIO_OVERRIDES},
        image_token_id=raw["image_token_id"],
        video_token_id=raw["video_token_id"],
        audio_token_id=raw["audio_token_id"],
        out_hidden_size=_VISION_OVERRIDES["hidden_size"]
        * raw["vision_config"]["merge_size"] ** 2,
        projector_hidden_size=64,
        projector_hidden_act=raw["projector_hidden_act"],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_every_checkpoint_tensor_fills_exactly_one_parameter() -> None:
    snapshot = _snapshot()
    raw = _raw_config(snapshot)
    quant_config = Nvfp4Config.from_config(raw["quantization_config"])
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            model = MuseSparkForConditionalGeneration(
                config=_narrow_config(raw),
                mapping=_single_rank_mapping(),
                quant_config=quant_config,
                is_multimodal_active=True,
                mm_attention_backend="triton_attn",
            )
    finally:
        torch.set_default_dtype(previous_dtype)

    parameters = set(dict(model.named_parameters()))
    filled: dict[str, list[str | None]] = {}
    for name in _checkpoint_tensor_names(snapshot):
        parameter_name, shard_id = resolve_parameter_name(name)
        filled.setdefault(parameter_name, []).append(shard_id)
    assert set(filled) == parameters
    for parameter_name, shards in filled.items():
        if ".qkv_proj." in parameter_name:
            # The decoder packs its output gate into the same projection.
            gated = parameter_name.startswith("model.")
            assert sorted(shards) == (
                ["g", "k", "q", "v"] if gated else ["k", "q", "v"]
            )
            continue
        assert shards == [None], parameter_name


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
