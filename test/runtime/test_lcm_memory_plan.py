from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

_CONFIGS_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python"
    / "tokenspeed"
    / "runtime"
    / "configs"
)


def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _CONFIGS_DIR / file_name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_lcm_modules():
    for package_name in (
        "tokenspeed",
        "tokenspeed.runtime",
        "tokenspeed.runtime.configs",
    ):
        sys.modules.setdefault(package_name, types.ModuleType(package_name))
    plan = _load(
        "tokenspeed.runtime.configs.lcm_memory_plan",
        "lcm_memory_plan.py",
    )
    layouts = _load(
        "tokenspeed.runtime.configs.lcm_layouts",
        "lcm_layouts.py",
    )
    return plan, layouts


class LcmMemoryPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan_module, cls.layouts_module = _load_lcm_modules()

    def test_qwen_recipe_keeps_one_logical_block_size(self):
        fields = self.layouts_module.qwen_gdn_lcm_fields(
            layer_types=("linear_attention", "full_attention"),
            layer_group_ids=("linear_attention_0", "full_attention"),
            logical_block_tokens=128,
            kv_shape=(128, 1, 128),
            kv_element_size=2,
            conv_shape=(256, 3),
            conv_element_size=2,
            ssm_shape=(8, 128, 128),
            ssm_element_size=4,
        )

        self.assertEqual(
            {field.group_id for field in fields},
            {"linear_attention_0", "full_attention"},
        )
        by_id = {field.field_id: field for field in fields}
        self.assertEqual(by_id["layer.1.k"].shape[0], 128)
        self.assertEqual(by_id["layer.0.ssm"].shape, (8, 128, 128))

    def test_planner_expresses_byte_ratio_as_per_group_packing(self):
        field = self.plan_module.LcmFieldSpec
        fields = (
            field("history", "history.k", "unit.0.a", (128, 1, 8), 2),
            field("history", "history.v", "unit.0.b", (128, 1, 8), 2),
            field("state", "state.ssm", "unit.0.a", (1, 1, 128), 2),
            field(
                "state",
                "state.conv",
                "unit.0.b",
                (1, 128),
                2,
                exact_page_stride=False,
            ),
        )

        plan = self.plan_module.plan_lcm_fields(
            fields,
            logical_block_tokens=128,
            budget_bytes=8192,
            alignment=256,
        )

        self.assertEqual(plan.logical_block_tokens, 128)
        self.assertEqual(
            plan.group("history").cache_blocks_per_lcm_block,
            1,
        )
        self.assertEqual(
            plan.group("state").cache_blocks_per_lcm_block,
            8,
        )

    def test_layout_constructors_derive_sizes_from_geometry(self):
        group = self.plan_module.LcmGroupLayout(
            group_id="history",
            cache_blocks_per_lcm_block=4,
            page_count=17,
        )
        field = self.plan_module.LcmFieldLayout(
            group_id="history",
            field_id="history.k",
            plane_id="plane.a",
            shape=(8, 16),
            element_size=2,
            field_offset_bytes=0,
            page_stride_bytes=256,
        )
        plan = self.plan_module.LcmMemoryPlan(
            logical_block_tokens=128,
            lcm_block_bytes=1024,
            num_lcm_blocks=4,
            groups=(group,),
            fields=(field,),
        )

        self.assertEqual(field.payload_bytes, 256)
        self.assertEqual(plan.arena_bytes, 5120)

    def test_duplicate_field_ids_are_rejected(self):
        field = self.plan_module.LcmFieldSpec
        fields = (
            field("a", "duplicate", "plane.a", (16,), 2),
            field("b", "duplicate", "plane.b", (16,), 2),
        )

        with self.assertRaisesRegex(ValueError, "field ids must be unique"):
            self.plan_module.plan_lcm_fields(
                fields,
                logical_block_tokens=16,
                budget_bytes=4096,
            )

    def test_budget_must_hold_null_and_usable_parent(self):
        field = self.plan_module.LcmFieldSpec

        with self.assertRaisesRegex(ValueError, "null parent"):
            self.plan_module.plan_lcm_fields(
                (field("history", "history.k", "plane.a", (128, 8), 2),),
                logical_block_tokens=128,
                budget_bytes=2048,
            )

    def test_fixed_parent_count_and_explicit_packing(self):
        field = self.plan_module.LcmFieldSpec
        fields = (
            field("history", "history.k", "plane.k", (128, 8), 2),
            field("state", "state.ssm", "plane.k", (128, 2), 2),
        )

        plan = self.plan_module.plan_lcm_fields(
            fields,
            logical_block_tokens=128,
            num_lcm_blocks=7,
            cache_blocks_per_lcm_block={"history": 1, "state": 4},
            alignment=256,
        )

        self.assertEqual(plan.num_lcm_blocks, 7)
        self.assertEqual(plan.group("history").cache_blocks_per_lcm_block, 1)
        self.assertEqual(plan.group("state").cache_blocks_per_lcm_block, 4)
        self.assertEqual(plan.group("history").page_count, 8)
        self.assertEqual(plan.group("state").page_count, 29)

    def test_explicit_packing_is_bounded_by_page_ids_not_a_magic_count(self):
        field = self.plan_module.LcmFieldSpec
        plan = self.plan_module.plan_lcm_fields(
            (field("history", "history.k", "plane.k", (1,), 1),),
            logical_block_tokens=16,
            num_lcm_blocks=2,
            cache_blocks_per_lcm_block={"history": 256},
        )

        self.assertEqual(
            plan.group("history").cache_blocks_per_lcm_block,
            256,
        )
        self.assertEqual(plan.group("history").page_count, 513)

    def test_automatic_packing_keeps_large_exact_byte_ratio(self):
        field = self.plan_module.LcmFieldSpec
        plan = self.plan_module.plan_lcm_fields(
            (
                field("history", "history.k", "plane.shared", (1,), 1),
                field("state", "state.ssm", "plane.shared", (256,), 1),
            ),
            logical_block_tokens=16,
            num_lcm_blocks=2,
        )

        self.assertEqual(
            plan.group("history").cache_blocks_per_lcm_block,
            256,
        )
        self.assertEqual(plan.group("state").cache_blocks_per_lcm_block, 1)

    def test_requires_exactly_one_capacity_input(self):
        field = self.plan_module.LcmFieldSpec
        fields = (field("history", "history.k", "plane.k", (128, 8), 2),)

        for kwargs in (
            {},
            {"budget_bytes": 4096, "num_lcm_blocks": 2},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(
                    ValueError, "exactly one of budget_bytes and num_lcm_blocks"
                ):
                    self.plan_module.plan_lcm_fields(
                        fields,
                        logical_block_tokens=128,
                        **kwargs,
                    )

    def test_fixed_parent_count_must_be_a_positive_integer(self):
        field = self.plan_module.LcmFieldSpec
        fields = (field("history", "history.k", "plane.k", (128, 8), 2),)

        for count in (0, -1, True, 1.5, "2"):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    self.plan_module.plan_lcm_fields(
                        fields,
                        logical_block_tokens=128,
                        num_lcm_blocks=count,
                    )

    def test_explicit_packing_keys_must_match_groups(self):
        field = self.plan_module.LcmFieldSpec
        fields = (
            field("history", "history.k", "plane.k", (128, 8), 2),
            field("state", "state.ssm", "plane.s", (128, 2), 2),
        )

        for packing in (
            {"history": 1},
            {"history": 1, "state": 4, "extra": 1},
        ):
            with self.subTest(packing=packing):
                with self.assertRaisesRegex(ValueError, "exactly the cache groups"):
                    self.plan_module.plan_lcm_fields(
                        fields,
                        logical_block_tokens=128,
                        num_lcm_blocks=2,
                        cache_blocks_per_lcm_block=packing,
                    )

    def test_explicit_packing_rejects_invalid_count(self):
        field = self.plan_module.LcmFieldSpec
        fields = (field("history", "history.k", "plane.k", (128, 8), 2),)

        for count in (0, -1, True, 1.5, "2"):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "packing"):
                    self.plan_module.plan_lcm_fields(
                        fields,
                        logical_block_tokens=128,
                        num_lcm_blocks=2,
                        cache_blocks_per_lcm_block={"history": count},
                    )

    def test_explicit_packing_preserves_exact_stride_validation(self):
        field = self.plan_module.LcmFieldSpec
        fields = (
            field("history", "history.k", "plane.shared", (128, 8), 2),
            field("state", "state.ssm", "plane.shared", (128, 3), 2),
        )

        with self.assertRaisesRegex(ValueError, "needs page stride"):
            self.plan_module.plan_lcm_fields(
                fields,
                logical_block_tokens=128,
                num_lcm_blocks=2,
                cache_blocks_per_lcm_block={"history": 1, "state": 2},
            )

    def test_draft_history_recipe_emits_only_enabled_kv_fields(self):
        fields = self.layouts_module.draft_history_lcm_fields(
            layer_group_ids=("full", "swa", "unused"),
            enabled_layer_ids=(0, 1),
            logical_block_tokens=128,
            layer_kv_heads=(2, 4, 8),
            head_dim=64,
            kv_element_size=2,
        )

        self.assertEqual(
            {field.field_id for field in fields},
            {"layer.0.k", "layer.0.v", "layer.1.k", "layer.1.v"},
        )
        self.assertEqual({field.group_id for field in fields}, {"full", "swa"})

    def test_inkling_bf16_and_fp8_checkpoint_planes(self):
        bf16 = self.layouts_module.inkling_lcm_fields(
            layer_group_ids=("swa",),
            logical_block_tokens=128,
            layer_kv_heads=(2,),
            head_dim=128,
            kv_element_size=2,
            hidden_size=256,
            checkpoint_rows=3,
            kvconv_element_size=2,
            hiddenconv_element_size=2,
        )
        fp8 = self.layouts_module.inkling_lcm_fields(
            layer_group_ids=("swa",),
            logical_block_tokens=128,
            layer_kv_heads=(2,),
            head_dim=128,
            kv_element_size=1,
            hidden_size=256,
            checkpoint_rows=3,
            kvconv_element_size=2,
            hiddenconv_element_size=1,
            kv_scale_block_size=32,
            kv_scale_element_size=1,
        )

        bf16_by_id = {field.field_id: field for field in bf16}
        fp8_by_id = {field.field_id: field for field in fp8}
        self.assertEqual(
            bf16_by_id["layer.0.attnconv"].plane_id,
            "unit.0.hidden_k",
        )
        self.assertEqual(
            fp8_by_id["layer.0.attnconv"].plane_id,
            "unit.0.k",
        )
        self.assertIn("layer.0.k_scale", fp8_by_id)
        self.assertIn("layer.0.v_scale", fp8_by_id)


if __name__ == "__main__":
    unittest.main()
