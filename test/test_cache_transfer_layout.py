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

from types import SimpleNamespace

import pytest

from tokenspeed.runtime.cache.transfer.layout import (
    CacheField,
    CacheGroupLayout,
    CacheTransferLayout,
    combine_cache_transfer_layouts,
    layout_from_lcm_plan,
)


def _field(field_id: str, *, stride: int = 64, payload: int = 48):
    return CacheField(
        field_id=field_id,
        device_buffer_index=0,
        device_block_zero_offset_bytes=0,
        block_stride_bytes=stride,
        payload_bytes=payload,
    )


def test_layout_rejects_duplicate_group_ids():
    group = CacheGroupLayout(
        group_id="full",
        cache_blocks_per_lcm_block=16,
        fields=(_field("k"),),
    )

    with pytest.raises(ValueError, match="duplicate group"):
        CacheTransferLayout(
            num_lcm_blocks=10,
            groups=(group, group),
            buffers=(object(),),
            consumers=(("k",),),
        )


def test_layout_rejects_duplicate_field_ids_across_groups():
    groups = (
        CacheGroupLayout("full", 16, (_field("shared"),)),
        CacheGroupLayout("state", 1, (_field("shared"),)),
    )

    with pytest.raises(ValueError, match="duplicate field"):
        CacheTransferLayout(10, groups, (object(),), (("shared",),))


def test_layout_rejects_payload_larger_than_device_stride():
    with pytest.raises(ValueError, match="payload_bytes"):
        _field("bad", stride=31, payload=32)


def test_layout_rejects_unknown_consumer_field():
    group = CacheGroupLayout("full", 16, (_field("k"),))

    with pytest.raises(ValueError, match="unknown field"):
        CacheTransferLayout(10, (group,), (object(),), (("v",),))


def test_layout_requires_a_positive_num_lcm_blocks():
    group = CacheGroupLayout("full", 16, (_field("k"),))

    with pytest.raises(ValueError, match="num_lcm_blocks"):
        CacheTransferLayout(
            0,
            (group,),
            (object(),),
            (("k",),),
        )


def test_lcm_layout_is_derived_from_planned_field_offsets():
    plane = SimpleNamespace(
        plane_id="kv", bytes_per_lcm_block=4096, arena_offset_bytes=8192
    )
    plan = SimpleNamespace(
        logical_block_tokens=128,
        num_lcm_blocks=10,
        planes=(plane,),
        groups=(
            SimpleNamespace(
                group_id="full",
                cache_blocks_per_lcm_block=16,
                page_count=161,
            ),
        ),
        fields=(
            SimpleNamespace(
                group_id="full",
                field_id="layer.0.k",
                plane_id="kv",
                field_offset_bytes=64,
                page_stride_bytes=256,
                payload_bytes=16,
            ),
        ),
    )
    backing = object()

    layout = layout_from_lcm_plan(
        plan,
        backing,
        consumers=(("layer.0.k",),),
    )

    assert layout.buffers == (backing,)
    assert layout.num_lcm_blocks == 10
    assert layout.groups[0].fields == (
        CacheField(
            field_id="layer.0.k",
            device_buffer_index=0,
            device_block_zero_offset_bytes=8192 + 4096 - 256 + 64,
            block_stride_bytes=256,
            payload_bytes=16,
        ),
    )


def test_target_and_draft_layouts_share_scheduler_groups_but_keep_both_payloads():
    target = CacheTransferLayout(
        num_lcm_blocks=10,
        groups=(
            CacheGroupLayout("full", 16, (_field("layer.0.k"),)),
            CacheGroupLayout("state", 1, (_field("layer.1.state"),)),
        ),
        buffers=("target",),
        consumers=(("layer.0.k",), ("layer.1.state",)),
    )
    draft = CacheTransferLayout(
        num_lcm_blocks=10,
        groups=(CacheGroupLayout("full", 16, (_field("layer.0.k"),)),),
        buffers=("draft",),
        consumers=(("layer.0.k",),),
    )

    combined = combine_cache_transfer_layouts(target, draft)

    assert tuple(group.group_id for group in combined.groups) == ("full", "state")
    assert combined.buffers == ("target", "draft")
    assert tuple(field.field_id for field in combined.groups[0].fields) == (
        "target:layer.0.k",
        "draft:layer.0.k",
    )
    assert combined.groups[0].fields[1].device_buffer_index == 1
    assert combined.consumers == (
        ("target:layer.0.k",),
        ("target:layer.1.state",),
        ("draft:layer.0.k",),
    )


def test_draft_layout_must_use_target_block_geometry():
    target = CacheTransferLayout(
        10,
        (CacheGroupLayout("full", 16, (_field("target"),)),),
        (object(),),
        (("target",),),
    )
    draft = CacheTransferLayout(
        10,
        (CacheGroupLayout("full", 8, (_field("draft"),)),),
        (object(),),
        (("draft",),),
    )

    with pytest.raises(ValueError, match="geometry"):
        combine_cache_transfer_layouts(target, draft)
