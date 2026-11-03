from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types
from collections.abc import Mapping

import pytest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

_KV_CACHE_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python"
    / "tokenspeed"
    / "runtime"
    / "layers"
    / "attention"
    / "kv_cache"
)
_RECIPE_DIR = _KV_CACHE_DIR / "recipes"


@pytest.fixture(scope="class", autouse=True)
def cache_plan_module(request):
    module_name = "tokenspeed.runtime.layers.attention.kv_cache.recipes.plan"
    spec = importlib.util.spec_from_file_location(module_name, _RECIPE_DIR / "plan.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, module_name, module)
        spec.loader.exec_module(module)
        request.cls.plan_module = module
        yield


def _tagged_field(group_id, field_id, plane_id, shape, dtype, **kwargs):
    """``(group_id, CacheFieldSpec)`` -- the tag says which group declares it.

    Byte-layout tests care about the tag only to route the field into a
    group, so they keep one readable line per field; _plan_fields turns the
    tags into declarations.
    """
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
        CacheFieldSpec,
    )

    return group_id, CacheFieldSpec(field_id, plane_id, shape, dtype, **kwargs)


def _group(group_id: str, *fields, **spec_kwargs):
    """Declare one cache group the way a recipe does: id once, fields under it.

    Row geometry defaults so a byte-layout test need not restate scheduler
    semantics; ``rows_per_page`` is the group's CacheBlock token span.
    """
    # A duck-typed spec: the planner reads only spec.group_id.
    return types.SimpleNamespace(group_id=group_id, **spec_kwargs), tuple(fields)


class TestCacheMemoryPlanTest:
    def _plan_fields(
        self,
        fields,
        *,
        prefix_granularity,
        budget_bytes=None,
        num_lcm_blocks=None,
        **kwargs,
    ):
        """Solve a layout and bind capacity the way the recipes do.

        ``fields`` is either the ``{group_id: fields}`` map a recipe field
        builder returns, or a sequence of ``(group_id, CacheFieldSpec)``
        pairs from :func:`_tagged_field`. Both are grouped into declarations
        here, the same join the recipes do.
        """
        if isinstance(fields, Mapping):
            by_group = {group_id: tuple(group) for group_id, group in fields.items()}
        else:
            by_group = {}
            for group_id, field in fields:
                by_group[group_id] = by_group.get(group_id, ()) + (field,)
        layout = self.plan_module.pack(
            tuple(_group(group_id, *group) for group_id, group in by_group.items()),
            prefix_granularity=prefix_granularity,
            **kwargs,
        )
        if budget_bytes is not None:
            # Parent 0 backs logical null page 0 and is never schedulable.
            num_lcm_blocks = budget_bytes // layout.lcm_block_bytes - 1
        return layout.bind(num_lcm_blocks)

    def test_planner_expresses_byte_ratio_as_per_group_packing(self):
        field = _tagged_field
        fields = (
            field("history", "history.k", "unit.0.a", (128, 1, 8), "bfloat16"),
            field("history", "history.v", "unit.0.b", (128, 1, 8), "bfloat16"),
            field("state", "state.ssm", "unit.0.a", (1, 1, 128), "bfloat16"),
            field(
                "state",
                "state.conv",
                "unit.0.b",
                (1, 128),
                "bfloat16",
                exact_page_stride=False,
            ),
        )

        plan = self._plan_fields(
            fields,
            prefix_granularity=128,
            budget_bytes=8192,
            alignment=256,
        )

        assert (plan.prefix_granularity) == (128)
        assert (plan.group("history").cache_blocks_per_lcm_block) == (1)
        assert (plan.group("state").cache_blocks_per_lcm_block) == (8)

    def test_layout_is_capacity_independent(self):
        from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
            CacheFieldSpec,
        )

        layout = self.plan_module.pack(
            (
                _group(
                    "full",
                    CacheFieldSpec("full.k", "unit.0.k", (128, 8), "bfloat16"),
                    CacheFieldSpec("full.v", "unit.0.v", (128, 8), "bfloat16"),
                ),
            ),
            prefix_granularity=128,
            alignment=16,
        )
        small = layout.bind(2)
        large = layout.bind(5)

        assert (small.lcm_block_bytes) == (large.lcm_block_bytes)
        assert ([field.page_stride_bytes for field in small.fields]) == (
            [field.page_stride_bytes for field in large.fields]
        )
        assert (small.group("full").page_count) == (3)
        assert (large.group("full").page_count) == (6)
        assert (small.planes[1].arena_offset_bytes) != (
            large.planes[1].arena_offset_bytes
        )

    def test_flexible_field_preserves_required_page_stride_alignment(self):
        field = _tagged_field
        plan = self._plan_fields(
            (
                field(
                    "compressed",
                    "compressed.kv",
                    "unit.0",
                    (64, 9),
                    "uint8",
                    exact_page_stride=False,
                    page_stride_alignment_bytes=576,
                ),
                field(
                    "wide",
                    "wide.state",
                    "unit.0",
                    (6000,),
                    "uint8",
                    exact_page_stride=False,
                ),
            ),
            prefix_granularity=256,
            num_lcm_blocks=2,
            cache_blocks_per_lcm_block={"compressed": 8, "wide": 1},
            alignment=256,
            max_padding_fraction=float("inf"),
        )

        compressed = plan.field("compressed.kv")
        assert (compressed.page_stride_bytes) > (compressed.payload_bytes)
        assert (compressed.page_stride_bytes % 576) == (0)

    def test_layout_constructors_derive_sizes_from_geometry(self):
        group = self.plan_module.CacheGroupLayout(
            group_id="history",
            cache_blocks_per_lcm_block=4,
            page_count=17,
        )
        field = self.plan_module.CacheFieldLayout(
            group_id="history",
            field_id="history.k",
            plane_id="plane.a",
            shape=(8, 16),
            dtype="bfloat16",
            field_offset_bytes=0,
            page_stride_bytes=256,
        )
        plan = self.plan_module.CacheMemoryPlan(
            prefix_granularity=128,
            lcm_block_bytes=1024,
            num_lcm_blocks=4,
            groups=(group,),
            fields=(field,),
        )

        assert (field.payload_bytes) == (256)
        assert (plan.arena_bytes) == (5120)

    def test_duplicate_field_ids_are_rejected(self):
        field = _tagged_field
        fields = (
            field("a", "duplicate", "plane.a", (16,), "bfloat16"),
            field("b", "duplicate", "plane.b", (16,), "bfloat16"),
        )

        with pytest.raises((ValueError), match=("field ids must be unique")):
            self._plan_fields(
                fields,
                prefix_granularity=16,
                budget_bytes=4096,
            )

    def test_fixed_parent_count_and_explicit_packing(self):
        field = _tagged_field
        fields = (
            field("history", "history.k", "plane.k", (128, 8), "bfloat16"),
            field("state", "state.ssm", "plane.k", (128, 2), "bfloat16"),
        )

        plan = self._plan_fields(
            fields,
            prefix_granularity=128,
            num_lcm_blocks=7,
            cache_blocks_per_lcm_block={"history": 1, "state": 4},
            alignment=256,
        )

        assert (plan.num_lcm_blocks) == (7)
        assert (plan.group("history").cache_blocks_per_lcm_block) == (1)
        assert (plan.group("state").cache_blocks_per_lcm_block) == (4)
        assert (plan.group("history").page_count) == (8)
        assert (plan.group("state").page_count) == (29)

    def test_explicit_packing_is_bounded_by_page_ids_not_a_magic_count(self):
        field = _tagged_field
        plan = self._plan_fields(
            (field("history", "history.k", "plane.k", (1,), "uint8"),),
            prefix_granularity=16,
            num_lcm_blocks=2,
            cache_blocks_per_lcm_block={"history": 256},
        )

        assert (plan.group("history").cache_blocks_per_lcm_block) == (256)
        assert (plan.group("history").page_count) == (513)

    def test_automatic_packing_keeps_large_exact_byte_ratio(self):
        field = _tagged_field
        plan = self._plan_fields(
            (
                field("history", "history.k", "plane.shared", (1,), "uint8"),
                field("state", "state.ssm", "plane.shared", (256,), "uint8"),
            ),
            prefix_granularity=16,
            num_lcm_blocks=2,
        )

        assert (plan.group("history").cache_blocks_per_lcm_block) == (256)
        assert (plan.group("state").cache_blocks_per_lcm_block) == (1)

    def test_explicit_packing_rejects_groups_outside_plan(self):
        field = _tagged_field
        fields = (
            field("history", "history.k", "plane.k", (128, 8), "bfloat16"),
            field("state", "state.ssm", "plane.s", (128, 2), "bfloat16"),
        )

        with pytest.raises((ValueError), match=("outside the plan")):
            self._plan_fields(
                fields,
                prefix_granularity=128,
                num_lcm_blocks=2,
                cache_blocks_per_lcm_block={"history": 1, "state": 4, "extra": 1},
            )

    def test_partial_packing_pins_named_groups_and_solves_the_rest(self):
        # A draft plan pins the groups it shares with the target (page ids
        # must align) while its draft-only groups pack by byte ratio.
        field = _tagged_field
        fields = (
            field("history", "history.k", "plane.k", (128, 8), "bfloat16"),
            field("state", "state.ssm", "plane.s", (128, 2), "bfloat16"),
        )

        unpinned = self._plan_fields(
            fields,
            prefix_granularity=128,
            num_lcm_blocks=2,
            max_padding_fraction=1.0,
        )
        history_count = unpinned.group("history").cache_blocks_per_lcm_block
        pinned = self._plan_fields(
            fields,
            prefix_granularity=128,
            num_lcm_blocks=2,
            max_padding_fraction=1.0,
            cache_blocks_per_lcm_block={"history": history_count},
        )
        assert (pinned.group("history").cache_blocks_per_lcm_block) == (history_count)
        # The unnamed group packs exactly as the fully-unpinned solve does.
        assert (pinned.group("state").cache_blocks_per_lcm_block) == (
            unpinned.group("state").cache_blocks_per_lcm_block
        )

    def test_explicit_packing_preserves_exact_stride_validation(self):
        field = _tagged_field
        fields = (
            field("history", "history.k", "plane.shared", (128, 8), "bfloat16"),
            field("state", "state.ssm", "plane.shared", (128, 3), "bfloat16"),
        )

        with pytest.raises((ValueError), match=("needs page stride")):
            self._plan_fields(
                fields,
                prefix_granularity=128,
                num_lcm_blocks=2,
                cache_blocks_per_lcm_block={"history": 1, "state": 2},
            )


class TestCapacityReportTest:
    """Per-group capacity in its own unit; binding admission = min."""

    def _mixed_plan(self):
        from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
            CacheFieldSpec,
            pack,
        )

        groups = (
            _group(
                "history",
                CacheFieldSpec("history.k", "plane.shared", (128, 8), "bfloat16"),
            ),
            _group(
                "state",
                CacheFieldSpec("state.ssm", "plane.shared", (128, 8), "bfloat16"),
            ),
            _group(
                "swa",
                CacheFieldSpec("swa.kv", "plane.shared", (128, 8), "bfloat16"),
            ),
        )
        return pack(groups, prefix_granularity=128).bind(100)

    def test_units_and_supported_requests(self):
        plan = self._mixed_plan()
        report = plan.capacity_report(
            window_tokens={"swa": 512},
            per_request_blocks={"state": 2},
            max_num_seqs=32,
        )
        state = report["state"]
        assert (state["unit"]) == ("requests")
        state_pages = 100 * plan.group("state").cache_blocks_per_lcm_block
        assert (state["supported_requests"]) == (state_pages // 2)

        swa = report["swa"]
        assert (swa["unit"]) == ("tokens")
        swa_tokens = 100 * plan.group("swa").cache_blocks_per_lcm_block * 128
        assert (swa["supported_requests"]) == (swa_tokens // 512)

        history = report["history"]
        assert (history["supported_requests"]) is None
        assert (history["dead_bytes"]) == (0)

    def test_window_group_dead_bytes_bounded_by_demand(self):
        plan = self._mixed_plan()
        report = plan.capacity_report(
            window_tokens={"swa": 512},
            max_num_seqs=4,
        )
        swa = report["swa"]
        # 4 requests x 512-token windows is the whole active demand; the
        # rest of the group's rows are stranded by the static slab split.
        swa_tokens = 100 * plan.group("swa").cache_blocks_per_lcm_block * 128
        assert (swa["dead_bytes"]) > (0)
        demand_tokens = 4 * 512
        field_bytes = 128 * 8 * 2
        expected = (swa_tokens - demand_tokens) // 128 * field_bytes
        assert (swa["dead_bytes"]) == (expected)


class TestContinuationLayerFieldsTest(TestCacheMemoryPlanTest):
    """One-big-model merged solve: draft fields join as continuation layers
    (renumbered after the target's), sharing group ids and packing."""

    def test_merged_solve_shares_group_packing(self):
        from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
            CacheFieldSpec,
        )

        target = (
            CacheFieldSpec("layer.0.kv", "slot.0", (128, 8), "bfloat16"),
            CacheFieldSpec("layer.1.kv", "slot.1", (128, 8), "bfloat16"),
        )
        # The draft layer is layer 2 of the one big model: global ids from the
        # first walk, so nothing needs renumbering.
        draft = (CacheFieldSpec("layer.2.kv", "slot.2", (128, 8), "bfloat16"),)
        merged = self.plan_module.pack(
            (_group("history", *target, *draft),),
            prefix_granularity=128,
        )
        plan = merged.bind(2)
        # One group, one packing, one page-id space; the draft layer is
        # just layer 2 of the one big model.
        assert (dict(merged.group_packing)) == ({"history": 1})
        assert (plan.field("layer.2.kv").page_stride_bytes) == (
            plan.field("layer.0.kv").page_stride_bytes
        )
        assert (len(plan.planes)) == (3)


class TestBindingUtilizationTest(TestCacheMemoryPlanTest):
    """util(g) = K_g x group_bytes / parent_bytes — the binding-hole metric.

    Aliased slabs are sized by their widest tenant; a narrower group's
    binding uses only its own shape and the rest of the parent is dead
    for that binding's lifetime.
    """

    def test_wide_group_full_narrow_group_partial(self):
        field = _tagged_field
        # Two groups share one plane (aliased): wide 256B/page x1, narrow
        # 100B/page x1 -> parent is sized by the wide tenant.
        plan = self._plan_fields(
            (
                field("wide", "w.kv", "plane.shared", (256,), "uint8"),
                field(
                    "narrow",
                    "n.state",
                    "plane.shared",
                    (100,),
                    "uint8",
                    exact_page_stride=False,
                ),
            ),
            prefix_granularity=16,
            num_lcm_blocks=2,
            cache_blocks_per_lcm_block={"wide": 1, "narrow": 1},
            max_padding_fraction=2.0,
        )
        report = plan.capacity_report()
        _pytest_first_1 = report["wide"]["binding_utilization"]
        _pytest_second_2 = 1.0
        assert (
            _pytest_first_1 == _pytest_second_2
            or round(abs(_pytest_first_1 - _pytest_second_2), 7) == 0
        )
        _pytest_first_3 = report["narrow"]["binding_utilization"]
        _pytest_second_4 = 100 / 256
        _pytest_places_5 = 3
        assert (
            _pytest_first_3 == _pytest_second_4
            or round(abs(_pytest_first_3 - _pytest_second_4), _pytest_places_5) == 0
        )
