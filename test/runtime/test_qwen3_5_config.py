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

from tokenspeed.runtime.configs.qwen3_5_config import Qwen3_5Config, Qwen3_5TextConfig


def test_layer_types_default_full_attention_interval() -> None:
    """Without an explicit schedule, layer 4 (1-indexed) is full attention."""
    config = Qwen3_5TextConfig(num_hidden_layers=6)
    assert config.layer_types == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
        "linear_attention",
    ]


def test_layer_types_derived_from_full_attention_interval_kwarg() -> None:
    config = Qwen3_5TextConfig(num_hidden_layers=4, full_attention_interval=2)
    assert config.layer_types == [
        "linear_attention",
        "full_attention",
        "linear_attention",
        "full_attention",
    ]


def test_layer_types_explicit_schedule_preserved() -> None:
    """An explicit checkpoint schedule drives the config verbatim."""
    config = Qwen3_5TextConfig(
        num_hidden_layers=3,
        layer_types=["full_attention", "linear_attention", "full_attention"],
    )
    assert config.layer_types == [
        "full_attention",
        "linear_attention",
        "full_attention",
    ]


def test_layer_types_without_interval_does_not_raise() -> None:
    """Explicit layer_types must not require full_attention_interval."""
    config = Qwen3_5TextConfig(
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
    )
    assert config.layer_types == ["linear_attention", "full_attention"]


def test_layer_types_legacy_labels_remapped() -> None:
    """Legacy spellings normalize to the paged-cache vocabulary."""
    config = Qwen3_5TextConfig(
        num_hidden_layers=4,
        layer_types=["attention", "mamba", "conv", "full_attention"],
    )
    assert config.layer_types == [
        "full_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]


def test_layers_block_type_maps_full_attention_to_attention() -> None:
    """Model dispatch reads the checkpoint vocabulary reversed to 'attention'."""
    config = Qwen3_5TextConfig(
        num_hidden_layers=2,
        layer_types=["full_attention", "linear_attention"],
    )
    assert config.layers_block_type == ["attention", "linear_attention"]
    assert config.full_attention_layer_ids == [0]
    assert config.linear_layer_ids == [1]


def test_layer_types_round_trip_serialization() -> None:
    config = Qwen3_5TextConfig(
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
    )
    assert config.to_dict()["layer_types"] == [
        "linear_attention",
        "full_attention",
    ]
    reloaded = Qwen3_5TextConfig.from_dict(config.to_dict())
    assert reloaded.layer_types == config.layer_types


def test_composite_config_forwards_layer_types_to_text_config() -> None:
    config = Qwen3_5Config(
        text_config={
            "num_hidden_layers": 3,
            "layer_types": ["full_attention", "linear_attention", "full_attention"],
        }
    )
    assert config.layer_types == [
        "full_attention",
        "linear_attention",
        "full_attention",
    ]
