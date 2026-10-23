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

import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    pack,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    SLIDING_ATTENTION,
    auto_group,
    group,
    layer_group_ids,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")


def _packing(groups):
    return {spec.group_id: 1 for spec, _ in groups}


def _groups(labels, windows, widths, *, automatic):
    def fields(layer, gid, occurrence):
        return tuple(
            CacheFieldSpec(
                f"layer.{layer}.{plane}",
                f"unit.{occurrence}.{plane}",
                (64, widths[layer]),
                dtype,
                exact_page_stride=False,
            )
            for plane, dtype in (
                ("k", "bfloat16"),
                ("v", "uint8"),
                ("scale", "float32"),
            )
        )

    kwargs = dict(
        layer_types=labels,
        group_ids=layer_group_ids(layer_types=labels, sliding_window_tokens=windows),
        sliding_window_tokens=windows,
        prefix_granularity=64,
        fields_for_layer=fields,
        pd_disaggregation_enabled=True,
    )
    if automatic:
        return auto_group(
            **kwargs,
            packing=_packing,
            alignment=256,
            max_padding_fraction=float("inf"),
        )
    return group(**kwargs)


def _layout(groups):
    return pack(
        groups,
        prefix_granularity=64,
        cache_blocks_per_lcm_block=_packing(groups),
        alignment=256,
        max_padding_fraction=float("inf"),
    )


@pytest.mark.parametrize(
    "full,sliding,expected",
    [
        (9, 43, [9, 9, 9, 8, 8]),
        (14, 43, [11, 11, 11, 10]),
        (11, 55, [11] * 5),
        (1, 4, [1] * 4),
        (8, 8, [8]),
        (9, 10, [10]),
        (8, 0, []),
        (0, 8, [8]),
    ],
)
def test_byte_equal_schedules(full, sliding, expected):
    labels = [FULL_ATTENTION] * full + [SLIDING_ATTENTION] * sliding
    widths = [32] * len(labels)
    before = _groups(labels, 128, widths, automatic=False)
    after = _groups(labels, 128, widths, automatic=True)
    assert after == _groups(labels, 128, widths, automatic=True)
    assert [
        len(fields) // 3 for spec, fields in after if spec.attn_kind == "sliding_window"
    ] == expected
    old, new = _layout(before), _layout(after)
    assert new.lcm_block_bytes <= old.lcm_block_bytes
    assert new.lcm_block_bytes * len(after) <= old.lcm_block_bytes * len(before)
    assert Counter(f.field_id for _, fields in after for f in fields) == Counter(
        f.field_id for _, fields in before for f in fields
    )
    plan = new.bind(20)
    for spec, fields in after:
        if spec.attn_kind == "sliding_window":
            assert spec.sliding_window_tokens == 128
        assert spec.transfer_policy == "full_suffix"
        spans = sorted(
            (
                plan.field_page_byte_offset(f.field_id, 1),
                plan.field_page_byte_offset(f.field_id, 1) + f.payload_bytes,
            )
            for f in fields
        )
        assert all(end <= start for (_, end), (start, _) in zip(spans, spans[1:]))


def test_multiple_windows_are_never_merged():
    labels = [FULL_ATTENTION] * 2 + [SLIDING_ATTENTION] * 14
    windows = [None] * 2 + [64] * 8 + [256] * 6
    groups = _groups(labels, windows, [32] * 16, automatic=True)
    for spec, fields in groups:
        assert all(
            windows[int(field.field_id.split(".")[1])] == spec.sliding_window_tokens
            for field in fields
        )
    assert len(groups) == 8
    assert (
        _layout(groups).lcm_block_bytes
        < _layout(_groups(labels, windows, [32] * 16, automatic=False)).lcm_block_bytes
    )


def test_heterogeneous_draft_and_scale_bytes_change_the_decision():
    labels = [FULL_ATTENTION] * 9 + [SLIDING_ATTENTION] * 43 + [FULL_ATTENTION] * 5
    narrow = _groups(labels, 128, [32] * 57, automatic=True)
    wide = _groups(labels, 128, [32] * 52 + [96] * 5, automatic=True)
    assert len(narrow) == 5
    assert len(wide) == 4
    assert _layout(wide).lcm_block_bytes > _layout(narrow).lcm_block_bytes


@pytest.mark.parametrize(
    "labels",
    [
        [LINEAR_ATTENTION] * 3 + [FULL_ATTENTION],
        [FULL_ATTENTION, LINEAR_ATTENTION] * 6,
        [FULL_ATTENTION] * 4
        + ["sliding_attention_0"] * 5
        + ["sliding_attention_1"] * 5,
    ],
)
def test_recurrent_and_explicit_groups_are_preserved(labels):
    widths = [32] * len(labels)
    assert _groups(labels, 128, widths, automatic=True) == _groups(
        labels, 128, widths, automatic=False
    )


def test_group_proliferation_is_bounded():
    labels = [FULL_ATTENTION] + [SLIDING_ATTENTION] * 100
    groups = _groups(labels, 128, [32] * len(labels), automatic=True)
    assert len(groups) <= 16
    assert (
        _layout(groups).lcm_block_bytes
        < _layout(
            _groups(labels, 128, [32] * len(labels), automatic=False)
        ).lcm_block_bytes
    )


@pytest.mark.parametrize("mixed_labels", [False, True])
def test_explicit_group_ids_with_semantic_labels_are_preserved(mixed_labels):
    kwargs = dict(
        layer_types=[FULL_ATTENTION] * 2 + [SLIDING_ATTENTION] * 8,
        group_ids=[FULL_ATTENTION] * 2 + ["window_a"] * 4 + ["window_b"] * 4,
        sliding_window_tokens=128,
        prefix_granularity=64,
        fields_for_layer=lambda layer, gid, occurrence: (
            CacheFieldSpec(
                f"layer.{layer}.k", f"unit.{occurrence}", (64, 16), "bfloat16"
            ),
        ),
        pd_disaggregation_enabled=False,
    )
    if mixed_labels:
        kwargs["layer_types"][-1] = "sliding_attention_0"
        kwargs["group_ids"][2:] = ["window_a"] * 8
    assert auto_group(
        **kwargs, packing=_packing, alignment=256, max_padding_fraction=float("inf")
    ) == group(**kwargs)


def test_split_can_repair_an_unsplit_padding_limit_violation():
    labels = [FULL_ATTENTION] * 9 + [SLIDING_ATTENTION] * 43
    groups = auto_group(
        layer_types=labels,
        group_ids=labels,
        sliding_window_tokens=128,
        prefix_granularity=64,
        fields_for_layer=lambda layer, gid, occurrence: (
            CacheFieldSpec(
                f"layer.{layer}.k", f"unit.{occurrence}", (64, 16), "bfloat16"
            ),
        ),
        pd_disaggregation_enabled=False,
        packing=_packing,
        alignment=256,
        max_padding_fraction=0.25,
    )
    assert len(groups) == 6
    assert _layout(groups).lcm_block_bytes == 9 * 64 * 16 * 2


@pytest.mark.parametrize("draft_width", [4, 8, 16, 32])
@pytest.mark.parametrize("mxfp8", [False, True])
@pytest.mark.parametrize("full,sliding,drafts", [(9, 43, 5), (1, 2, 1)])
def test_layout_and_capacity_use_identical_merged_bytes(
    draft_width, mxfp8, full, sliding, drafts
):
    from test.runtime.test_cache_setup import _mha_config

    import torch

    from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
        OrdinaryRecipe,
    )

    target = _mha_config()
    target = replace(
        target,
        prefix_granularity=128,
        kernel_page_size=128,
        kv_cache_mxfp8=mxfp8,
        components=(
            replace(
                target.component(MHAConfig),
                head_dim=128,
                cache_layer_types=(FULL_ATTENTION,) * full
                + (SLIDING_ATTENTION,) * sliding,
                sliding_window_tokens=(None,) * full + (256,) * sliding,
            ),
        ),
    )
    draft = replace(
        target,
        kv_cache_mxfp8=False,
        kv_cache_dtype=torch.bfloat16,
        components=(
            replace(
                target.component(MHAConfig),
                head_dim=draft_width * 16,
                cache_layer_types=(FULL_ATTENTION,) * drafts,
                sliding_window_tokens=None,
            ),
        ),
    )
    budget = 1 << 28
    recipe = OrdinaryRecipe(
        family="mha",
        server_args=SimpleNamespace(max_total_tokens=None, kv_offload_config=None),
        model_config=SimpleNamespace(num_attention_layers=full + sliding),
        attn_config=target,
        draft_model_config=SimpleNamespace(num_attention_layers=drafts),
        draft_attn_config=draft,
        cache_budget_bytes=budget,
        probe_batch_rows=None,
        decode_input_tokens=16,
        overlap_schedule_depth=1,
    )
    setup = recipe.setup()
    plan = setup.spec.memory_plan
    assert plan.num_lcm_blocks == budget // plan.lcm_block_bytes - 1
    assert plan.arena_bytes <= budget
    assert budget - plan.arena_bytes < plan.lcm_block_bytes
    assert len(plan.fields) == (full + sliding) * (4 if mxfp8 else 2) + drafts * 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
