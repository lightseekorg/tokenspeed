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

"""CPU byte-level coverage of actual KDA raw-conv checkpoint dimensions."""

from dataclasses import dataclass, replace
from test.runtime.cache_pd_test_utils import group, layout, segment

import numpy as np
import pytest

from tokenspeed.runtime.layers.attention.kv_cache.recipes.transfer import (
    build_cache_transfer_schema,
)
from tokenspeed.runtime.pd.transfer_plan import (
    CacheTransferPlanner,
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


def raw_bytes(tp, rank, axis):
    # Exact uint16 bit patterns, no BF16/FP32 conversion in the copy check.
    global_rows = np.arange(3 * 3 * 8192, dtype=np.uint16).reshape(3, 3, 8192)
    local_width = 8192 // tp
    rows = global_rows[:, :, rank * local_width : (rank + 1) * local_width].reshape(
        3, -1
    )
    return (rows if axis == 1 else rows.T).copy().tobytes()


@pytest.mark.parametrize("scope", ["all-fields", "conv-only"])
@pytest.mark.parametrize("axis", [0, 1], ids=["channel-major", "channels-last"])
@pytest.mark.parametrize(
    "source_tp,destination_tp", [(8, 8), (8, 4), (4, 8), (8, 16), (16, 8)]
)
def test_qkv_raw_history_reassembled_exactly_with_strides(
    source_tp, destination_tp, axis, scope
):
    source, destination = (
        contract(source_tp, axis),
        contract(destination_tp, axis, padding=384),
    )
    planner = CacheTransferPlanner(
        prefill_tp_size=source_tp,
        decode_tp_size=destination_tp,
        prefill_layout=source,
        decode_layout=destination,
        prefill_field_ids=(
            None if scope == "all-fields" else frozenset({conv_field(axis)})
        ),
    )
    source_field = source.plan.field(conv_field(axis))
    destination_field = destination.plan.field(conv_field(axis))
    assert source.transfer_schema.partition_for(source_field.field_id).axis == axis
    assert source.transfer_schema.partition_for(source_field.field_id).global_parts == (
        8192,
        8192,
        8192,
    )
    for destination_rank in range(destination_tp):
        route = planner.plan_for_decode_rank(destination_rank)
        guard = 31
        stride = destination_field.page_stride_bytes
        begin = guard + 3 * stride
        output = bytearray([0xC3]) * (begin + destination_field.payload_bytes + guard)
        counts = np.zeros(destination_field.payload_bytes, dtype=np.int32)
        for source_rank, fragments in route.fragments_by_prefill_rank.items():
            payload = raw_bytes(source_tp, source_rank, axis)
            source_begin = guard + 2 * source_field.page_stride_bytes
            input_bytes = bytearray([0xA5]) * (source_begin + len(payload) + guard)
            input_bytes[source_begin : source_begin + len(payload)] = payload
            untouched = bytes(input_bytes)
            if not fragments:
                # Equal TP with every field resident uses the actual planner's
                # whole-field route, encoded by an empty fragment tuple.
                assert source_tp == destination_tp and scope == "all-fields"
                assert source_rank == destination_rank
                output[begin : begin + len(payload)] = input_bytes[
                    source_begin : source_begin + len(payload)
                ]
                counts += 1
            for fragment in fragments:
                if fragment.field_id != source_field.field_id:
                    continue
                for row in range(fragment.rows_per_page):
                    src = (
                        source_begin
                        + fragment.src_byte_offset
                        + row * fragment.src_row_stride_bytes
                    )
                    dst = fragment.dst_byte_offset + row * fragment.dst_row_stride_bytes
                    width = fragment.bytes_per_row
                    output[begin + dst : begin + dst + width] = input_bytes[
                        src : src + width
                    ]
                    counts[dst : dst + width] += 1
            assert bytes(input_bytes) == untouched
        assert np.all(
            counts == 1
        ), "Q/K/V and all three raw-history rows must be covered exactly once"
        assert output[begin : begin + destination_field.payload_bytes] == raw_bytes(
            destination_tp, destination_rank, axis
        )
        assert output[:begin] == bytes([0xC3]) * begin
        assert (
            output[begin + destination_field.payload_bytes :] == bytes([0xC3]) * guard
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
