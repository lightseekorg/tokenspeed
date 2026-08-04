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

from tokenspeed.runtime.cache.layout import (
    CacheGroupLayout,
    CacheSegment,
    CacheTransferLayout,
    combine_cache_transfer_layouts,
    layout_from_lcm_plan,
)


def _segment(segment_id: str, *, stride: int = 64, payload: int = 48):
    return CacheSegment(
        segment_id=segment_id,
        buffer_index=0,
        page_zero_offset=0,
        page_stride_bytes=stride,
        payload_bytes=payload,
    )


def test_layout_rejects_duplicate_group_ids():
    group = CacheGroupLayout(
        group_id="full",
        cache_blocks_per_lcm_block=16,
        page_count=161,
        segments=(_segment("k"),),
    )

    with pytest.raises(ValueError, match="duplicate group"):
        CacheTransferLayout(
            logical_block_tokens=128,
            groups=(group, group),
            buffers=(object(),),
            consumers=(("k",),),
        )


def test_layout_rejects_duplicate_segment_ids_across_groups():
    groups = (
        CacheGroupLayout("full", 16, 161, (_segment("shared"),)),
        CacheGroupLayout("state", 1, 11, (_segment("shared"),)),
    )

    with pytest.raises(ValueError, match="duplicate segment"):
        CacheTransferLayout(128, groups, (object(),), (("shared",),))


def test_layout_rejects_payload_larger_than_device_stride():
    with pytest.raises(ValueError, match="payload_bytes"):
        _segment("bad", stride=31, payload=32)


def test_layout_rejects_unknown_consumer_segment():
    group = CacheGroupLayout("full", 16, 161, (_segment("k"),))

    with pytest.raises(ValueError, match="unknown segment"):
        CacheTransferLayout(128, (group,), (object(),), (("v",),))


def test_layout_rejects_parent_count_that_disagrees_with_group_geometry():
    group = CacheGroupLayout("full", 16, 161, (_segment("k"),))

    with pytest.raises(ValueError, match="disagrees"):
        CacheTransferLayout(
            128,
            (group,),
            (object(),),
            (("k",),),
            lcm_block_count=11,
        )


def test_qwen_style_packing_excludes_device_padding():
    full = CacheGroupLayout(
        group_id="full",
        cache_blocks_per_lcm_block=16,
        page_count=161,
        segments=(
            _segment("full.k", stride=96, payload=40),
            _segment("full.v", stride=96, payload=40),
        ),
    )
    state = CacheGroupLayout(
        group_id="state",
        cache_blocks_per_lcm_block=1,
        page_count=11,
        segments=(
            _segment("state.ssm", stride=2048, payload=1000),
            _segment("state.conv", stride=1024, payload=200),
        ),
    )

    packed = CacheTransferLayout(
        logical_block_tokens=128,
        groups=(full, state),
        buffers=(object(),),
        consumers=(("full.k", "full.v"), ("state.ssm", "state.conv")),
    ).pack(alignment=16)

    assert packed.child_bytes == (96, 1216)
    assert packed.segment_offsets == ((0, 48), (0, 1008))
    assert packed.parent_bytes == 1536


def test_lcm_layout_is_derived_from_planned_field_offsets():
    plane = SimpleNamespace(
        plane_id="kv", bytes_per_lcm_block=4096, arena_offset_bytes=8192
    )
    plan = SimpleNamespace(
        logical_block_tokens=128,
        num_lcm_blocks=10,
        planes=(plane,),
        groups=(SimpleNamespace(group_id="full", cache_blocks_per_lcm_block=16, page_count=161),),
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
    assert layout.groups[0].segments == (
        CacheSegment(
            segment_id="layer.0.k",
            buffer_index=0,
            page_zero_offset=8192 + 4096 - 256 + 64,
            page_stride_bytes=256,
            payload_bytes=16,
        ),
    )


def test_target_and_draft_layouts_share_scheduler_groups_but_keep_both_payloads():
    target = CacheTransferLayout(
        logical_block_tokens=128,
        groups=(
            CacheGroupLayout("full", 16, 161, (_segment("layer.0.k"),)),
            CacheGroupLayout("state", 1, 11, (_segment("layer.1.state"),)),
        ),
        buffers=("target",),
        consumers=(("layer.0.k",), ("layer.1.state",)),
    )
    draft = CacheTransferLayout(
        logical_block_tokens=128,
        groups=(CacheGroupLayout("full", 16, 161, (_segment("layer.0.k"),)),),
        buffers=("draft",),
        consumers=(("layer.0.k",),),
    )

    combined = combine_cache_transfer_layouts(target, draft)

    assert tuple(group.group_id for group in combined.groups) == ("full", "state")
    assert combined.buffers == ("target", "draft")
    assert tuple(
        segment.segment_id for segment in combined.groups[0].segments
    ) == ("target:layer.0.k", "draft:layer.0.k")
    assert combined.groups[0].segments[1].buffer_index == 1
    assert combined.consumers == (
        ("target:layer.0.k",),
        ("target:layer.1.state",),
        ("draft:layer.0.k",),
    )


def test_draft_layout_must_use_target_page_geometry():
    target = CacheTransferLayout(
        128,
        (CacheGroupLayout("full", 16, 161, (_segment("target"),)),),
        (object(),),
        (("target",),),
    )
    draft = CacheTransferLayout(
        128,
        (CacheGroupLayout("full", 8, 81, (_segment("draft"),)),),
        (object(),),
        (("draft",),),
    )

    with pytest.raises(ValueError, match="geometry"):
        combine_cache_transfer_layouts(target, draft)
