from __future__ import annotations

import ast
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


def _load(module_name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_cache_modules():
    for package_name in (
        "tokenspeed",
        "tokenspeed.runtime",
        "tokenspeed.runtime.configs",
        "tokenspeed.runtime.layers",
        "tokenspeed.runtime.layers.attention",
        "tokenspeed.runtime.layers.attention.kv_cache",
        "tokenspeed.runtime.layers.attention.kv_cache.recipes",
    ):
        sys.modules.setdefault(package_name, types.ModuleType(package_name))
    plan = _load(
        "tokenspeed.runtime.layers.attention.kv_cache.plan",
        _KV_CACHE_DIR / "plan.py",
    )
    _load(
        "tokenspeed.runtime.layers.attention.kv_cache.recipes",
        _RECIPE_DIR / "__init__.py",
    )
    recipes = {}
    for recipe in (
        "deepseek_v4",
        "ordinary",
        "inkling",
        "kimi_k3",
        "qwen35",
    ):
        recipes[recipe] = _load(
            f"tokenspeed.runtime.layers.attention.kv_cache.recipes.{recipe}",
            _RECIPE_DIR / f"{recipe}.py",
        )
    layouts = types.SimpleNamespace(
        **{
            name: getattr(module, name)
            for module in recipes.values()
            for name in dir(module)
            if name.endswith("_cache_fields")
        }
    )
    return plan, layouts


class CacheMemoryPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan_module, cls.layouts_module = _load_cache_modules()

    def test_qwen_recipe_keeps_one_logical_block_size(self):
        fields = self.layouts_module.qwen_gdn_cache_fields(
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

    def test_mha_layers_keep_distinct_fields_with_shared_placement(self):
        fields = self.layouts_module.mha_cache_fields(
            layer_group_ids=(
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ),
            logical_block_tokens=128,
            kv_heads=2,
            head_dim=8,
            kv_element_size=2,
        )
        by_id = {field.field_id: field for field in fields}

        self.assertEqual(by_id["layer.0.k"].plane_id, by_id["layer.1.k"].plane_id)
        self.assertEqual(by_id["layer.0.v"].plane_id, by_id["layer.1.v"].plane_id)
        self.assertEqual(by_id["layer.2.k"].plane_id, by_id["layer.3.k"].plane_id)
        self.assertNotEqual(
            by_id["layer.0.k"].plane_id,
            by_id["layer.2.k"].plane_id,
        )

        plan = self.plan_module.plan_cache_fields(
            fields,
            logical_block_tokens=128,
            num_lcm_blocks=2,
            cache_blocks_per_lcm_block={
                "sliding_attention": 1,
                "full_attention": 1,
            },
            alignment=1,
            max_padding_fraction=1.0,
        )
        self.assertEqual(len(plan.planes), 4)
        self.assertEqual(len(plan.fields), 8)

    def test_inkling_pool_exclusively_owns_conv_views(self):
        def class_methods(path, class_name):
            module = ast.parse(path.read_text())
            cls = next(
                node
                for node in module.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            )
            return {
                node.name
                for node in cls.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }

        generic_methods = class_methods(
            _KV_CACHE_DIR / "mha.py", "MHATokenToKVPool"
        ) | class_methods(_KV_CACHE_DIR / "hybrid_mha.py", "HybridMHATokenToKVPool")
        inkling_methods = class_methods(
            _KV_CACHE_DIR / "hybrid_inkling.py",
            "HybridInklingTokenToKVPool",
        )

        conv_methods = {
            "kvconv_checkpoint_buffers",
            "hiddenconv_checkpoint_buffer",
        }
        self.assertTrue(conv_methods.isdisjoint(generic_methods))
        self.assertTrue(conv_methods.issubset(inkling_methods))

    def test_model_fields_live_in_dedicated_recipe_modules(self):
        expected = {
            "ordinary.py": {
                "mha_cache_fields",
                "mla_cache_fields",
                "draft_cache_fields",
            },
            "qwen35.py": {"qwen_gdn_cache_fields"},
            "inkling.py": {"inkling_cache_fields"},
            "kimi_k3.py": {"kimi_k3_cache_fields"},
            "deepseek_v4.py": {"deepseek_v4_cache_fields"},
        }

        for file_name, function_names in expected.items():
            module = ast.parse((_RECIPE_DIR / file_name).read_text())
            definitions = {
                node.name for node in module.body if isinstance(node, ast.FunctionDef)
            }
            self.assertTrue(function_names.issubset(definitions), file_name)

        planner_source = (_KV_CACHE_DIR / "plan.py").read_text().lower()
        for model_name in ("qwen", "inkling", "kimi", "deepseek"):
            self.assertNotIn(model_name, planner_source)

    def test_cache_pool_factory_is_separate_from_setup(self):
        setup_module = ast.parse((_KV_CACHE_DIR / "setup.py").read_text())
        factory_module = ast.parse((_KV_CACHE_DIR / "factory.py").read_text())

        setup_functions = {
            node.name for node in setup_module.body if isinstance(node, ast.FunctionDef)
        }
        factory_functions = {
            node.name
            for node in factory_module.body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertNotIn("create_cache_pool", setup_functions)
        self.assertIn("create_cache_pool", factory_functions)

    def test_registry_only_orchestrates_cache_construction(self):
        registry_path = _KV_CACHE_DIR.parent / "registry.py"
        registry_module = ast.parse(registry_path.read_text())
        create_components = next(
            node
            for node in registry_module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "create_attn_components"
        )
        called_names = {
            node.func.id
            for node in ast.walk(create_components)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("create_cache_pool", called_names)
        self.assertNotIn("plan_cache_fields", called_names)
        self.assertNotIn("profile_max_num_pages", called_names)

    def test_deepseek_v4_recipe_uses_one_p256_scheduler_domain(self):
        fields = self.layouts_module.deepseek_v4_cache_fields(
            layer_ratios=(1, 4, 128),
            logical_block_tokens=256,
            swa_shape=(128,),
            compressed_shapes={4: (64,), 128: (2,)},
            compressor_state_shapes={4: (8, 16), 128: (128, 8)},
            indexer_kv_shape=(64, 4),
            indexer_state_shape=(8, 32),
            kv_page_stride_alignment_bytes=576,
        )

        by_id = {field.field_id: field for field in fields}
        self.assertEqual(by_id["layer.0.swa"].shape, (128,))
        self.assertEqual(by_id["layer.1.compressed_kv"].shape, (64,))
        self.assertEqual(by_id["layer.2.compressor_state"].shape, (128, 8))
        self.assertEqual(
            by_id["layer.0.swa"].plane_id,
            by_id["layer.1.compressed_kv"].plane_id,
        )
        self.assertEqual(by_id["layer.1.indexer_kv"].plane_id, "unit.1")
        self.assertFalse(by_id["layer.0.swa"].exact_page_stride)
        self.assertFalse(by_id["layer.1.compressor_state"].exact_page_stride)
        self.assertEqual(
            by_id["layer.0.swa"].page_stride_alignment_bytes,
            576,
        )
        self.assertEqual(
            by_id["layer.1.compressed_kv"].page_stride_alignment_bytes,
            576,
        )

        plan = self.plan_module.plan_cache_fields(
            fields,
            logical_block_tokens=256,
            num_lcm_blocks=4,
            max_padding_fraction=float("inf"),
        )

        self.assertEqual(plan.logical_block_tokens, 256)
        self.assertEqual(
            {group.group_id for group in plan.groups},
            {
                "v4.swa_kv",
                "v4.c4a.compressed_kv",
                "v4.c4a.compressor_state",
                "v4.c128a.compressed_kv",
                "v4.c128a.compressor_state",
                "v4.c4a.indexer_compressor_state",
            },
        )

    def test_planner_expresses_byte_ratio_as_per_group_packing(self):
        field = self.plan_module.CacheFieldSpec
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

        plan = self.plan_module.plan_cache_fields(
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

    def test_layout_is_capacity_independent(self):
        field = self.plan_module.CacheFieldSpec
        fields = (
            field("full", "full.k", "unit.0.k", (128, 8), 2),
            field("full", "full.v", "unit.0.v", (128, 8), 2),
        )

        layout = self.plan_module.solve_cache_layout(
            fields,
            logical_block_tokens=128,
            alignment=16,
        )
        small = layout.with_num_lcm_blocks(2)
        large = layout.with_num_lcm_blocks(5)

        self.assertEqual(small.lcm_block_bytes, large.lcm_block_bytes)
        self.assertEqual(
            [field.page_stride_bytes for field in small.fields],
            [field.page_stride_bytes for field in large.fields],
        )
        self.assertEqual(small.group("full").page_count, 3)
        self.assertEqual(large.group("full").page_count, 6)
        self.assertNotEqual(
            small.planes[1].arena_offset_bytes,
            large.planes[1].arena_offset_bytes,
        )

    def test_flexible_field_preserves_required_page_stride_alignment(self):
        field = self.plan_module.CacheFieldSpec
        plan = self.plan_module.plan_cache_fields(
            (
                field(
                    "compressed",
                    "compressed.kv",
                    "unit.0",
                    (64, 9),
                    1,
                    exact_page_stride=False,
                    page_stride_alignment_bytes=576,
                ),
                field(
                    "wide",
                    "wide.state",
                    "unit.0",
                    (6000,),
                    1,
                    exact_page_stride=False,
                ),
            ),
            logical_block_tokens=256,
            num_lcm_blocks=2,
            cache_blocks_per_lcm_block={"compressed": 8, "wide": 1},
            alignment=256,
            max_padding_fraction=float("inf"),
        )

        compressed = plan.field("compressed.kv")
        self.assertGreater(compressed.page_stride_bytes, compressed.payload_bytes)
        self.assertEqual(compressed.page_stride_bytes % 576, 0)

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
            element_size=2,
            field_offset_bytes=0,
            page_stride_bytes=256,
        )
        plan = self.plan_module.CacheMemoryPlan(
            logical_block_tokens=128,
            lcm_block_bytes=1024,
            num_lcm_blocks=4,
            groups=(group,),
            fields=(field,),
        )

        self.assertEqual(field.payload_bytes, 256)
        self.assertEqual(plan.arena_bytes, 5120)

    def test_duplicate_field_ids_are_rejected(self):
        field = self.plan_module.CacheFieldSpec
        fields = (
            field("a", "duplicate", "plane.a", (16,), 2),
            field("b", "duplicate", "plane.b", (16,), 2),
        )

        with self.assertRaisesRegex(ValueError, "field ids must be unique"):
            self.plan_module.plan_cache_fields(
                fields,
                logical_block_tokens=16,
                budget_bytes=4096,
            )

    def test_budget_must_hold_null_and_usable_parent(self):
        field = self.plan_module.CacheFieldSpec

        with self.assertRaisesRegex(ValueError, "null parent"):
            self.plan_module.plan_cache_fields(
                (field("history", "history.k", "plane.a", (128, 8), 2),),
                logical_block_tokens=128,
                budget_bytes=2048,
            )

    def test_fixed_parent_count_and_explicit_packing(self):
        field = self.plan_module.CacheFieldSpec
        fields = (
            field("history", "history.k", "plane.k", (128, 8), 2),
            field("state", "state.ssm", "plane.k", (128, 2), 2),
        )

        plan = self.plan_module.plan_cache_fields(
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
        field = self.plan_module.CacheFieldSpec
        plan = self.plan_module.plan_cache_fields(
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
        field = self.plan_module.CacheFieldSpec
        plan = self.plan_module.plan_cache_fields(
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
        field = self.plan_module.CacheFieldSpec
        fields = (field("history", "history.k", "plane.k", (128, 8), 2),)

        for kwargs in (
            {},
            {"budget_bytes": 4096, "num_lcm_blocks": 2},
        ):
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(
                    ValueError, "exactly one of budget_bytes and num_lcm_blocks"
                ),
            ):
                self.plan_module.plan_cache_fields(
                    fields,
                    logical_block_tokens=128,
                    **kwargs,
                )

    def test_fixed_parent_count_must_be_a_positive_integer(self):
        field = self.plan_module.CacheFieldSpec
        fields = (field("history", "history.k", "plane.k", (128, 8), 2),)

        for count in (0, -1, True, 1.5, "2"):
            with (
                self.subTest(count=count),
                self.assertRaisesRegex(ValueError, "positive integer"),
            ):
                self.plan_module.plan_cache_fields(
                    fields,
                    logical_block_tokens=128,
                    num_lcm_blocks=count,
                )

    def test_explicit_packing_keys_must_match_groups(self):
        field = self.plan_module.CacheFieldSpec
        fields = (
            field("history", "history.k", "plane.k", (128, 8), 2),
            field("state", "state.ssm", "plane.s", (128, 2), 2),
        )

        for packing in (
            {"history": 1},
            {"history": 1, "state": 4, "extra": 1},
        ):
            with (
                self.subTest(packing=packing),
                self.assertRaisesRegex(ValueError, "exactly the cache groups"),
            ):
                self.plan_module.plan_cache_fields(
                    fields,
                    logical_block_tokens=128,
                    num_lcm_blocks=2,
                    cache_blocks_per_lcm_block=packing,
                )

    def test_explicit_packing_rejects_invalid_count(self):
        field = self.plan_module.CacheFieldSpec
        fields = (field("history", "history.k", "plane.k", (128, 8), 2),)

        for count in (0, -1, True, 1.5, "2"):
            with (
                self.subTest(count=count),
                self.assertRaisesRegex(ValueError, "packing"),
            ):
                self.plan_module.plan_cache_fields(
                    fields,
                    logical_block_tokens=128,
                    num_lcm_blocks=2,
                    cache_blocks_per_lcm_block={"history": count},
                )

    def test_explicit_packing_preserves_exact_stride_validation(self):
        field = self.plan_module.CacheFieldSpec
        fields = (
            field("history", "history.k", "plane.shared", (128, 8), 2),
            field("state", "state.ssm", "plane.shared", (128, 3), 2),
        )

        with self.assertRaisesRegex(ValueError, "needs page stride"):
            self.plan_module.plan_cache_fields(
                fields,
                logical_block_tokens=128,
                num_lcm_blocks=2,
                cache_blocks_per_lcm_block={"history": 1, "state": 2},
            )

    def test_draft_history_recipe_emits_only_enabled_kv_fields(self):
        fields = self.layouts_module.draft_cache_fields(
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
        bf16 = self.layouts_module.inkling_cache_fields(
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
        fp8 = self.layouts_module.inkling_cache_fields(
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
