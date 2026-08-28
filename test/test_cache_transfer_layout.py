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
    select_layer_fields,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    cache_field_layer_id,
)


def _field(field_id: str, *, stride: int = 64, payload: int = 48):
    return CacheField(
        field_id=field_id,
        device_buffer_index=0,
        device_block_zero_offset_bytes=0,
        block_stride_bytes=stride,
        payload_bytes=payload,
    )


def _single_group_layout(buffer, *fields):
    return CacheTransferLayout(
        10,
        (CacheGroupLayout("full", 16, fields),),
        (buffer,),
        tuple((field.field_id,) for field in fields),
    )


def test_cache_field_layer_id_parses_layer_owned_field():
    assert cache_field_layer_id("layer.12.k") == 12


@pytest.mark.parametrize(
    "field_id",
    ("attention.k", "layer.bad.k", "layer.-1.k", "layer.0"),
)
def test_cache_field_layer_id_rejects_invalid_field(field_id):
    with pytest.raises(ValueError, match="cache field"):
        cache_field_layer_id(field_id)


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
        prefix_granularity=128,
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


def test_lcm_layout_uses_scheduler_group_order():
    plane = SimpleNamespace(
        plane_id="shared", bytes_per_lcm_block=4096, arena_offset_bytes=0
    )
    plan = SimpleNamespace(
        num_lcm_blocks=4,
        planes=(plane,),
        # The memory planner sorts by group id, while the scheduler assigns
        # numeric group ids in first-appearance order.
        groups=(
            SimpleNamespace(
                group_id="full",
                cache_blocks_per_lcm_block=32,
                page_count=129,
            ),
            SimpleNamespace(
                group_id="state",
                cache_blocks_per_lcm_block=1,
                page_count=5,
            ),
        ),
        fields=(
            SimpleNamespace(
                group_id="full",
                field_id="layer.1.k",
                plane_id="shared",
                field_offset_bytes=0,
                page_stride_bytes=128,
                payload_bytes=128,
            ),
            SimpleNamespace(
                group_id="state",
                field_id="layer.0.state",
                plane_id="shared",
                field_offset_bytes=0,
                page_stride_bytes=4096,
                payload_bytes=4096,
            ),
        ),
    )

    layout = layout_from_lcm_plan(
        plan,
        object(),
        consumers=(("layer.0.state",), ("layer.1.k",)),
        group_ids=("state", "full"),
    )

    assert tuple(group.group_id for group in layout.groups) == ("state", "full")
    assert tuple(group.cache_blocks_per_lcm_block for group in layout.groups) == (
        1,
        32,
    )


def test_merged_plan_views_filter_and_remap_fields_before_combining():
    plane = SimpleNamespace(
        plane_id="shared", bytes_per_lcm_block=4096, arena_offset_bytes=0
    )
    plan = SimpleNamespace(
        num_lcm_blocks=4,
        planes=(plane,),
        groups=(
            SimpleNamespace(group_id="full", cache_blocks_per_lcm_block=16),
            SimpleNamespace(group_id="state", cache_blocks_per_lcm_block=1),
            SimpleNamespace(group_id="draft_swa", cache_blocks_per_lcm_block=4),
        ),
        fields=(
            SimpleNamespace(
                group_id="full",
                field_id="layer.0.k",
                plane_id="shared",
                field_offset_bytes=0,
                page_stride_bytes=256,
                payload_bytes=128,
            ),
            SimpleNamespace(
                group_id="state",
                field_id="layer.1.state",
                plane_id="shared",
                field_offset_bytes=256,
                page_stride_bytes=4096,
                payload_bytes=1024,
            ),
            SimpleNamespace(
                group_id="draft_swa",
                field_id="layer.2.k",
                plane_id="shared",
                field_offset_bytes=1280,
                page_stride_bytes=1024,
                payload_bytes=512,
            ),
        ),
    )

    target_fields, target_consumers = select_layer_fields(
        plan.fields, first_layer=0, num_layers=2
    )
    draft_fields, draft_consumers = select_layer_fields(
        plan.fields, first_layer=2, num_layers=1
    )
    target = layout_from_lcm_plan(
        plan,
        "shared-buffer",
        consumers=target_consumers,
        group_ids=("state", "full"),
        field_ids=target_fields,
    )
    draft = layout_from_lcm_plan(
        plan,
        "shared-buffer",
        consumers=draft_consumers,
        group_ids=("draft_swa",),
        field_ids=draft_fields,
    )

    combined = combine_cache_transfer_layouts(
        target,
        draft,
        group_ids=("state", "full", "draft_swa"),
    )

    assert target.consumers == (("layer.0.k",), ("layer.1.state",))
    assert draft.consumers == (("layer.2.k",),)
    assert tuple(group.group_id for group in combined.groups) == (
        "state",
        "full",
        "draft_swa",
    )
    assert combined.buffers == ("shared-buffer",)
    assert tuple(field.field_id for field in combined.groups[2].fields) == (
        "layer.2.k",
    )
    assert combined.consumers == (
        ("layer.0.k",),
        ("layer.1.state",),
        ("layer.2.k",),
    )


def test_target_and_draft_layouts_must_share_one_arena():
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

    with pytest.raises(ValueError, match="share one arena"):
        combine_cache_transfer_layouts(target, draft)


def test_aliased_target_and_draft_layout_is_transferred_once():
    buffer = object()
    target = _single_group_layout(buffer, _field("layer.0.k"))
    draft = _single_group_layout(buffer, _field("layer.0.k"))

    combined = combine_cache_transfer_layouts(target, draft)

    assert combined.buffers == (buffer,)
    assert combined.groups[0].fields == (_field("layer.0.k"),)
    assert combined.consumers == (("layer.0.k",),)


def test_merged_owner_keeps_aliased_field_on_target():
    buffer = object()
    target = _single_group_layout(buffer, _field("layer.0.k"), _field("layer.1.k"))
    draft = _single_group_layout(buffer, _field("layer.1.k"))

    combined = combine_cache_transfer_layouts(target, draft)

    assert combined.buffers == (buffer,)
    assert combined.groups == target.groups
    assert combined.consumers == (("layer.0.k",), ("layer.1.k",), ())


@pytest.mark.parametrize(
    ("draft_fields", "error"),
    (
        ((_field("layer.1.k", stride=96),), "conflicting layouts"),
        ((_field("layer.1.k"), _field("layer.2.k")), "ambiguous overlapping"),
    ),
)
def test_invalid_target_draft_field_overlap_is_rejected(draft_fields, error):
    buffer = object()
    target = _single_group_layout(buffer, _field("layer.0.k"), _field("layer.1.k"))
    draft = _single_group_layout(buffer, *draft_fields)

    with pytest.raises(ValueError, match=error):
        combine_cache_transfer_layouts(target, draft)


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
