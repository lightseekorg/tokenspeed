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

"""KDA transfer schemas shared with the Rust planner byte-level tests."""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from test.runtime.cache_pd_test_utils import group, layout, segment

import pytest

from tokenspeed.runtime.layers.attention.kv_cache.recipes.transfer import (
    build_cache_transfer_schema,
)


@dataclass(frozen=True)
class TextConfig:
    linear_attn_config: dict


@dataclass(frozen=True)
class ModelFacts:
    num_attention_layers: int = 1
    hf_text_config: TextConfig = TextConfig(dict(num_heads=64, head_dim=128))


def conv_field(axis):
    return "layer.0.conv_state_channels_last" if axis == 1 else "layer.0.conv_state"


def contract(tp, axis, *, padding=256):
    channels = 3 * 64 * 128 // tp
    shape = (3, channels) if axis == 1 else (channels, 3)
    raw = layout(
        group(
            "state",
            segment(
                conv_field(axis),
                shape=shape,
                dtype="bfloat16",
                stride=3 * channels * 2 + padding,
            ),
            segment(
                "layer.0.recurrent_state",
                shape=(64 // tp, 128, 128),
                dtype="float32",
                offset=1 << 24,
            ),
            attn_kind="state",
        ),
        page_bytes=1 << 20,
    )
    kwargs = dict(model_config=ModelFacts())
    schema = build_cache_transfer_schema(raw.plan, **kwargs)
    schema.validate(raw.plan)
    return replace(raw, transfer_schema=schema)


@pytest.mark.parametrize("tp", [4, 8, 16])
@pytest.mark.parametrize("axis", [0, 1], ids=["channel-major", "channels-last"])
@pytest.mark.parametrize("padding", [256, 384])
def test_schema_matches_the_rust_planner_fixture(tp, axis, padding):
    # Both languages consume the same contract. The Rust tests execute the
    # actual planner and check every Q/K/V byte, stride and destination guard.
    path = (
        Path(__file__).resolve().parents[2]
        / "tokenspeed-engine/crates/ts-pd/tests/fixtures/planning/kda_conv.json"
    )
    fixtures = json.loads(path.read_text())
    actual = contract(tp, axis, padding=padding)
    partition = actual.transfer_schema.partition_for(conv_field(axis))
    assert partition.axis == axis
    assert partition.global_parts == (8192, 8192, 8192)
    assert (
        json.loads(actual.to_wire_bytes())
        == fixtures[f"tp{tp}-axis{axis}-padding{padding}"]
    )


def test_real_tp8_conv_and_recurrent_schema():
    value = contract(8, 1)
    assert value.plan.field(conv_field(1)).shape == (3, 3072)
    assert value.plan.field(conv_field(1)).payload_bytes == 18432
    assert value.plan.field("layer.0.recurrent_state").shape == (8, 128, 128)
    assert value.transfer_schema.partition_for("layer.0.recurrent_state").axis == 0
    assert (
        value.transfer_schema.partition_for("layer.0.recurrent_state").global_extent
        == 64
    )


def test_channels_last_axis_must_exist_in_field_shape():
    value = contract(8, 1)
    fields = tuple(
        replace(f, shape=(3072,)) if f.field_id == conv_field(1) else f
        for f in value.plan.fields
    )
    with pytest.raises(ValueError):
        malformed = replace(value.plan, fields=fields)
        build_cache_transfer_schema(malformed, model_config=ModelFacts()).validate(
            malformed
        )
