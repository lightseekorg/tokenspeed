from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
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


def _load(mod_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, _CONFIGS_DIR / file_name)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: on py3.9 @dataclass + `from __future__ import
    # annotations` resolves field types via sys.modules[cls.__module__].
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_fmp = _load("flat_memory_plan_under_test", "flat_memory_plan.py")
ComponentSpec = _fmp.ComponentSpec
BlockGeometry = _fmp.BlockGeometry
solve_page_geometry = _fmp.solve_page_geometry
plan_tensors = _fmp.plan_tensors
plan_component_tensors = _fmp.plan_component_tensors
components_from_layers = _fmp.components_from_layers
hybrid_lcm_field_specs = _fmp.hybrid_lcm_field_specs
LcmFieldSpec = _fmp.LcmFieldSpec
LcmArena = _fmp.LcmArena
plan_lcm_fields = _fmp.plan_lcm_fields


class EqualizerTest(unittest.TestCase):
    def test_gpt_oss_degenerate_keeps_page_size(self):
        comps = [
            ComponentSpec(
                group_id="full_attention",
                layer=0,
                component="kv",
                bytes_per_slot=1024,
                const_bytes=0,
            ),
            ComponentSpec(
                group_id="sliding_attention",
                layer=1,
                component="kv",
                bytes_per_slot=1024,
                const_bytes=0,
            ),
        ]
        geo = solve_page_geometry(comps, block_size=16, alignment=256)
        self.assertEqual(geo.block_size, 16)
        self.assertEqual(geo.block_bytes, 16 * 1024)

    def test_qwen35_constant_state_inflates_page_size(self):
        comps = [
            ComponentSpec(
                group_id="full_attention",
                layer=0,
                component="kv",
                bytes_per_slot=1024,
                const_bytes=0,
            ),
            ComponentSpec(
                group_id="linear_attention",
                layer=1,
                component="conv",
                bytes_per_slot=0,
                const_bytes=40 * 1024,
            ),
            ComponentSpec(
                group_id="linear_attention",
                layer=1,
                component="ssm",
                bytes_per_slot=0,
                const_bytes=60 * 1024,
            ),
        ]
        geo = solve_page_geometry(comps, block_size=16, alignment=4)
        # A state layer's components pack into ONE page row ([conv|ssm|pad]),
        # so the constant demand is their SUM: ceil((40+60)KiB / 1KiB) = 100.
        self.assertEqual(geo.block_size, 100)
        self.assertEqual(geo.block_bytes, 100 * 1024)

    def test_inflation_rounds_up_to_alignment(self):
        comps = [
            ComponentSpec(
                "full_attention", 0, "kv", bytes_per_slot=1024, const_bytes=0
            ),
            ComponentSpec(
                "linear_attention",
                1,
                "state",
                bytes_per_slot=0,
                const_bytes=101 * 1024,
            ),
        ]
        geo = solve_page_geometry(comps, block_size=16, alignment=16)
        # ceil(101K / 1K) = 101 -> rounded up to the next multiple of 16.
        self.assertEqual(geo.block_size, 112)
        self.assertEqual(geo.block_bytes, 112 * 1024)

    def test_dsv4_linear_rows_pad_not_inflate(self):
        comps = [
            ComponentSpec("full_mla", 0, "latent", bytes_per_slot=1152, const_bytes=0),
            ComponentSpec(
                "full_mla", 0, "indexer_k", bytes_per_slot=132, const_bytes=0
            ),
        ]
        geo = solve_page_geometry(comps, block_size=64, alignment=256)
        self.assertEqual(geo.block_size, 64)
        # Same-layer components pack into one row.
        self.assertEqual(geo.block_bytes, 64 * (1152 + 132))

    def test_constant_components_require_a_linear_row(self):
        comps = [
            ComponentSpec(
                "linear_attention", 0, "state", bytes_per_slot=0, const_bytes=1024
            )
        ]
        with self.assertRaises(ValueError):
            solve_page_geometry(comps, block_size=16, alignment=4)


class PlanTensorsTest(unittest.TestCase):
    def _comps_qwen35(self):
        return [
            ComponentSpec(
                "full_attention",
                layer=0,
                component="kv",
                bytes_per_slot=1024,
                const_bytes=0,
            ),
            ComponentSpec(
                "full_attention",
                layer=1,
                component="kv",
                bytes_per_slot=1024,
                const_bytes=0,
            ),
            ComponentSpec(
                "linear_attention",
                layer=0,
                component="conv",
                bytes_per_slot=0,
                const_bytes=40 * 1024,
            ),
            ComponentSpec(
                "linear_attention",
                layer=0,
                component="ssm",
                bytes_per_slot=0,
                const_bytes=60 * 1024,
            ),
        ]

    def test_slot_pairing_one_layer_per_group_per_slot(self):
        plan = plan_tensors(
            self._comps_qwen35(),
            block_size=16,
            alignment=4,
            budget_bytes=100 * 1024 * 1024,
        )
        self.assertEqual(len(plan.tensors), 2)
        slot0 = plan.tensors[0]
        self.assertEqual(
            {(b.group_id, b.layer) for b in slot0.bindings},
            {("full_attention", 0), ("linear_attention", 0)},
        )
        slot1 = plan.tensors[1]
        self.assertEqual(
            {(b.group_id, b.layer) for b in slot1.bindings},
            {("full_attention", 1)},
        )
        for t in plan.tensors:
            seen = {}
            for b in t.bindings:
                key = (b.slot, b.group_id)
                self.assertEqual(seen.setdefault(key, b.layer), b.layer)

    def test_row_offsets_accumulate_within_a_row(self):
        plan = plan_tensors(
            self._comps_qwen35(),
            block_size=16,
            alignment=4,
            budget_bytes=100 * 1024 * 1024,
        )
        state = [
            b for b in plan.tensors[0].bindings if b.group_id == "linear_attention"
        ]
        by_comp = {b.component: b for b in state}
        self.assertEqual(by_comp["conv"].row_offset, 0)
        self.assertEqual(by_comp["ssm"].row_offset, 40 * 1024)
        full = [b for b in plan.tensors[0].bindings if b.group_id == "full_attention"]
        self.assertEqual(full[0].row_offset, 0)

    def test_num_blocks_from_budget_shared_across_slots(self):
        plan = plan_tensors(
            self._comps_qwen35(),
            block_size=16,
            alignment=4,
            budget_bytes=100 * 1024 * 1024,
        )
        geo = plan.geometry
        self.assertEqual(geo.block_size, 100)
        self.assertEqual(geo.block_bytes, 300 * 1024)
        self.assertEqual(geo.num_blocks, 100 * 1024 * 1024 // (300 * 1024))  # 341
        slot0, slot1 = plan.tensors
        self.assertEqual(slot0.nbytes, geo.num_blocks * 200 * 1024)
        self.assertEqual(slot1.nbytes, geo.num_blocks * 100 * 1024)

    def test_gpt_oss_pairing_matches_hybrid_slab(self):
        comps = [
            ComponentSpec(
                "full_attention",
                layer=i,
                component="kv",
                bytes_per_slot=1024,
                const_bytes=0,
            )
            for i in range(2)
        ]
        comps += [
            ComponentSpec(
                "sliding_attention",
                layer=i,
                component="kv",
                bytes_per_slot=1024,
                const_bytes=0,
            )
            for i in range(2)
        ]
        plan = plan_tensors(
            comps, block_size=16, alignment=4, budget_bytes=64 * 1024 * 1024
        )
        self.assertEqual(len(plan.tensors), 2)
        for t in plan.tensors:
            self.assertEqual(
                {b.group_id for b in t.bindings},
                {"full_attention", "sliding_attention"},
            )

    def test_budget_too_small_raises(self):
        with self.assertRaises(ValueError):
            plan_tensors(
                self._comps_qwen35(),
                block_size=16,
                alignment=4,
                budget_bytes=100 * 1024,
            )

    def test_cross_group_rows_sized_by_own_bindings(self):
        comps = [
            ComponentSpec("full", 0, "kv", 100, 0),
            ComponentSpec("state", 0, "conv", 0, 300),
            ComponentSpec("state", 1, "conv", 0, 300),
        ]
        plan = plan_tensors(comps, block_size=4, alignment=1, budget_bytes=100_000)
        # slot0 packs 100*4 + 300 = 700, slot1 packs 300.
        self.assertEqual(plan.geometry.num_blocks, 100_000 // 1000)
        slot0, slot1 = plan.tensors
        self.assertEqual(slot0.nbytes, plan.geometry.num_blocks * 700)
        self.assertEqual(slot1.nbytes, plan.geometry.num_blocks * 300)


class PlanComponentTensorsTest(unittest.TestCase):
    def test_qwen_shape(self):
        kv_per_slot = 2048
        state = {"conv": 848_256, "ssm": 1_298_048}
        layers = (["linear_attention"] * 3 + ["full_attention"]) * 12
        comps = components_from_layers(
            layer_types=layers,
            kv_bytes_per_slot=kv_per_slot,
            state_const_bytes=state,
        )
        plan = plan_component_tensors(comps, block_size=1088, budget_bytes=10 * 1024**3)
        row_sum = 12 * 1088 * kv_per_slot + 36 * sum(state.values())
        self.assertEqual(plan.geometry.num_blocks, (10 * 1024**3) // row_sum)
        self.assertGreaterEqual(plan.geometry.num_blocks, 100)
        self.assertEqual(len(plan.tensors), 12 + 72)
        for t in plan.tensors:
            (b,) = t.bindings
            self.assertEqual(b.row_offset, 0)
            self.assertEqual(t.nbytes, plan.geometry.num_blocks * b.nbytes_per_block)

    def test_reserved_bytes_shrink_blocks(self):
        comps = components_from_layers(
            layer_types=["full_attention"] * 2,
            kv_bytes_per_slot=100,
            state_const_bytes={},
        )
        base = plan_component_tensors(comps, block_size=4, budget_bytes=10_000)
        tighter = plan_component_tensors(
            comps, block_size=4, budget_bytes=10_000, reserved_bytes_per_block=800
        )
        self.assertEqual(base.geometry.num_blocks, 10_000 // 800)
        self.assertEqual(tighter.geometry.num_blocks, 10_000 // 1600)

    def test_budget_too_small_raises(self):
        comps = components_from_layers(
            layer_types=["full_attention"],
            kv_bytes_per_slot=100,
            state_const_bytes={},
        )
        with self.assertRaises(ValueError):
            plan_component_tensors(comps, block_size=4, budget_bytes=500)


class GptOssCapacityTest(unittest.TestCase):
    def test_plan_counts_every_layer_row(self):
        comps = [
            ComponentSpec(
                "full_attention",
                layer=i,
                component="kv",
                bytes_per_slot=1024,
                const_bytes=0,
            )
            for i in range(24)
        ]
        comps += [
            ComponentSpec(
                "sliding_attention",
                layer=i,
                component="kv",
                bytes_per_slot=1024,
                const_bytes=0,
            )
            for i in range(24)
        ]
        budget = 10 * 1024**3
        plan = plan_tensors(comps, block_size=16, alignment=4, budget_bytes=budget)
        self.assertEqual(plan.geometry.num_blocks, budget // (48 * 16 * 1024))


class ComponentsFromLayersTest(unittest.TestCase):
    def test_qwen35_shape(self):
        comps = components_from_layers(
            layer_types=["linear_attention", "full_attention", "linear_attention"],
            kv_bytes_per_slot=1024,
            state_const_bytes={"conv": 40 * 1024, "ssm": 60 * 1024},
        )
        by_key = {(c.group_id, c.layer, c.component): c for c in comps}
        self.assertIn(("full_attention", 0, "kv"), by_key)
        self.assertIn(("linear_attention", 0, "conv"), by_key)
        self.assertIn(("linear_attention", 1, "ssm"), by_key)
        self.assertEqual(by_key[("full_attention", 0, "kv")].bytes_per_slot, 1024)
        self.assertEqual(by_key[("linear_attention", 1, "conv")].const_bytes, 40 * 1024)

    def test_pure_attention_model_has_no_state_components(self):
        comps = components_from_layers(
            layer_types=["full_attention", "sliding_attention"],
            kv_bytes_per_slot=512,
            state_const_bytes={},
        )
        self.assertTrue(all(c.const_bytes == 0 for c in comps))
        self.assertEqual(len(comps), 2)

    def test_plan_end_to_end_from_layers(self):
        comps = components_from_layers(
            layer_types=["linear_attention", "full_attention"],
            kv_bytes_per_slot=1024,
            state_const_bytes={"conv": 40 * 1024, "ssm": 60 * 1024},
        )
        plan = plan_tensors(
            comps, block_size=16, alignment=4, budget_bytes=100 * 1024 * 1024
        )
        self.assertEqual(plan.geometry.block_size, 100)  # inflated by the state row


class LcmMemoryPlanTest(unittest.TestCase):
    def _qwen35_fields(self):
        return [
            LcmFieldSpec("full_attention", "full", "qwen", (1_000_320,), 1),
            LcmFieldSpec("linear_attention_0", "state0", "qwen", (8_002_560,), 1),
            LcmFieldSpec("linear_attention_1", "state1", "qwen", (8_002_560,), 1),
            LcmFieldSpec("linear_attention_2", "state2", "qwen", (8_002_560,), 1),
        ]

    def test_qwen35_uses_one_logical_p_with_per_group_packing(self):
        budget_bytes = 80_025_600
        plan = plan_lcm_fields(
            self._qwen35_fields(),
            logical_block_tokens=128,
            budget_bytes=budget_bytes,
            alignment=128,
        )

        self.assertEqual(plan.logical_block_tokens, 128)
        self.assertEqual(plan.lcm_block_bytes, 8_002_560)
        self.assertEqual(plan.num_lcm_blocks, 9)
        self.assertEqual(plan.arena_bytes, budget_bytes)

        layouts = {group.group_id: group for group in plan.groups}
        self.assertEqual(
            {
                group_id: layout.cache_blocks_per_lcm_block
                for group_id, layout in layouts.items()
            },
            {
                "full_attention": 8,
                "linear_attention_0": 1,
                "linear_attention_1": 1,
                "linear_attention_2": 1,
            },
        )
        for layout in layouts.values():
            self.assertEqual(
                layout.cache_blocks_per_lcm_block * layout.slot_stride_bytes,
                plan.lcm_block_bytes,
            )
            self.assertGreaterEqual(
                layout.slot_stride_bytes, layout.raw_cache_block_bytes
            )
            self.assertEqual(
                layout.page_count,
                1 + plan.num_lcm_blocks * layout.cache_blocks_per_lcm_block,
            )

        # K is physical packing only. It must not change scheduler/hash P.
        self.assertNotEqual(
            128 * layouts["full_attention"].cache_blocks_per_lcm_block,
            128 * layouts["linear_attention_0"].cache_blocks_per_lcm_block,
        )

    def test_qwen35_hybrid_fields_keep_exact_kv_and_state_strides(self):
        layer_types = [
            layer_type
            for _ in range(15)
            for layer_type in (
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            )
        ]
        layer_group_ids = [
            group_id
            for _ in range(15)
            for group_id in (
                "linear_attention_0",
                "linear_attention_1",
                "linear_attention_2",
                "full_attention",
            )
        ]
        for kv_dtype, kv_element_size, expected_full_packing in (
            ("bf16", 2, 8),
            ("fp8", 1, 16),
        ):
            with self.subTest(kv_dtype=kv_dtype):
                fields = hybrid_lcm_field_specs(
                    layer_types=layer_types,
                    layer_group_ids=layer_group_ids,
                    logical_block_tokens=128,
                    kv_shape=(128, 1, 256),
                    kv_element_size=kv_element_size,
                    conv_shape=(1536, 3),
                    conv_element_size=2,
                    ssm_shape=(8, 128, 128),
                    ssm_element_size=4,
                )

                plan = plan_lcm_fields(
                    fields,
                    logical_block_tokens=128,
                    budget_bytes=2 * 15_728_640,
                    alignment=256,
                    max_padding_fraction=1.0,
                )

                self.assertEqual(plan.lcm_block_bytes, 15_728_640)
                self.assertEqual(len(plan.planes), 30)
                for field in plan.fields:
                    if field.group_id == "full_attention" or field.field_id.endswith(
                        ".ssm"
                    ):
                        self.assertEqual(field.page_stride_bytes, field.payload_bytes)
                for group in plan.groups:
                    self.assertLessEqual(
                        group.padding_bytes_per_slot / group.raw_cache_block_bytes,
                        1.0,
                    )
                groups = {group.group_id: group for group in plan.groups}
                self.assertEqual(
                    groups["full_attention"].padding_bytes_per_slot,
                    0,
                )
                self.assertEqual(
                    {
                        group.group_id: group.cache_blocks_per_lcm_block
                        for group in plan.groups
                    },
                    {
                        "full_attention": expected_full_packing,
                        "linear_attention_0": 1,
                        "linear_attention_1": 1,
                        "linear_attention_2": 1,
                    },
                )

    def test_solve_is_deterministic_across_input_order(self):
        fields = self._qwen35_fields()
        kwargs = dict(
            logical_block_tokens=128,
            budget_bytes=80_025_600,
            alignment=128,
        )
        self.assertEqual(
            plan_lcm_fields(fields, **kwargs),
            plan_lcm_fields(reversed(fields), **kwargs),
        )

    def test_alignment_padding_is_per_slot(self):
        plan = plan_lcm_fields(
            [
                LcmFieldSpec(
                    "small", "small", "shared", (1000,), 1, exact_page_stride=False
                ),
                LcmFieldSpec(
                    "large", "large", "shared", (2048,), 1, exact_page_stride=False
                ),
            ],
            logical_block_tokens=128,
            budget_bytes=4096,
            alignment=256,
        )
        small = plan.group("small")
        self.assertEqual(small.slot_stride_bytes, 1024)
        self.assertEqual(small.padding_bytes_per_slot, 24)
        self.assertEqual(small.cache_blocks_per_lcm_block, 2)
        self.assertEqual(
            small.cache_blocks_per_lcm_block * small.slot_stride_bytes,
            plan.lcm_block_bytes,
        )

    def test_checked_lcm_limit_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "LCM block size"):
            plan_lcm_fields(
                [
                    LcmFieldSpec("a", "a", "shared", (1 << 31,), 1),
                    LcmFieldSpec("b", "b", "shared", (1 << 30,), 1),
                ],
                logical_block_tokens=128,
                budget_bytes=1 << 40,
                alignment=1,
                max_lcm_block_bytes=1 << 30,
            )

    def test_excessive_padding_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "padding"):
            plan_lcm_fields(
                [
                    LcmFieldSpec(
                        "small", "small", "shared", (600,), 1, exact_page_stride=False
                    ),
                    LcmFieldSpec(
                        "large", "large", "shared", (1000,), 1, exact_page_stride=False
                    ),
                ],
                logical_block_tokens=128,
                budget_bytes=1 << 20,
                alignment=1,
                max_padding_fraction=0.25,
            )

    def test_rejects_invalid_or_too_small_plan(self):
        with self.assertRaisesRegex(ValueError, "logical_block_tokens"):
            plan_lcm_fields(
                [LcmFieldSpec("g", "g", "shared", (1024,), 1)],
                logical_block_tokens=0,
                budget_bytes=4096,
            )
        with self.assertRaisesRegex(ValueError, "null parent and one usable"):
            plan_lcm_fields(
                [LcmFieldSpec("g", "g", "shared", (1024,), 1)],
                logical_block_tokens=128,
                budget_bytes=1024,
            )


class InklingLcmFieldSpecsTest(unittest.TestCase):
    @staticmethod
    def _fields():
        builder = getattr(_fmp, "inkling_lcm_field_specs", None)
        if builder is None:
            return None
        return builder(
            layer_group_ids=(
                "sliding_attention_0",
                "sliding_attention_1",
                "sliding_attention_2",
                "sliding_attention_3",
                "sliding_attention_4",
                "full_attention",
            ),
            logical_block_tokens=128,
            layer_kv_heads=(2, 2, 2, 2, 2, 1),
            head_dim=128,
            kv_element_size=2,
            hidden_size=256,
            checkpoint_rows=3,
            kvconv_element_size=2,
            hiddenconv_element_size=1,
        )

    def test_builder_describes_attention_and_boundary_checkpoint_groups(self):
        group_ids = (
            "sliding_attention_0",
            "sliding_attention_1",
            "sliding_attention_2",
            "sliding_attention_3",
            "sliding_attention_4",
            "full_attention",
        )
        fields = self._fields()
        self.assertIsNotNone(fields, "Inkling needs a model-side LCM field builder")

        groups = {field.group_id for field in fields}
        self.assertEqual(groups, {*group_ids, "kvconv", "hiddenconv"})
        self.assertTrue(
            all(
                field.shape[0] == 128 for field in fields if field.group_id in group_ids
            )
        )
        self.assertTrue(
            all(
                field.shape[0] == 3
                for field in fields
                if field.group_id in {"kvconv", "hiddenconv"}
            )
        )

    def test_checkpoint_packing_respects_each_physical_plane(self):
        try:
            plan = plan_lcm_fields(
                self._fields(),
                logical_block_tokens=128,
                budget_bytes=2 * 131_072,
                alignment=256,
                max_padding_fraction=16.0,
            )
        except ValueError as exc:
            self.fail(f"valid Inkling plane packing was rejected: {exc}")

        self.assertEqual(plan.lcm_block_bytes, 131_072)
        self.assertEqual(
            {group.group_id: group.cache_blocks_per_lcm_block for group in plan.groups},
            {
                "full_attention": 2,
                "sliding_attention_0": 1,
                "sliding_attention_1": 1,
                "sliding_attention_2": 1,
                "sliding_attention_3": 1,
                "sliding_attention_4": 1,
                "kvconv": 4,
                "hiddenconv": 8,
            },
        )

    def test_bf16_checkpoints_do_not_change_attention_page_stride(self):
        layer_groups = (
            "sliding_attention_0",
            "sliding_attention_1",
            "sliding_attention_2",
            "sliding_attention_3",
            "sliding_attention_4",
            "full_attention",
        ) * 11
        fields = _fmp.inkling_lcm_field_specs(
            layer_group_ids=layer_groups,
            logical_block_tokens=128,
            layer_kv_heads=(4, 4, 4, 4, 4, 2) * 11,
            head_dim=128,
            kv_element_size=1,
            hidden_size=6144,
            checkpoint_rows=3,
            kvconv_element_size=2,
            hiddenconv_element_size=2,
            kv_scale_block_size=32,
            kv_scale_element_size=1,
        )

        plan = plan_lcm_fields(
            fields,
            logical_block_tokens=128,
            budget_bytes=1 << 30,
            alignment=256,
            max_padding_fraction=16.0,
        )

        full_k = plan.field("layer.5.k")
        self.assertEqual(full_k.page_stride_bytes, full_k.payload_bytes)
        self.assertEqual(
            full_k.plane_id,
            plan.field("layer.5.kvconv_k").plane_id,
        )
        self.assertNotEqual(
            full_k.plane_id,
            plan.field("layer.5.attnconv").plane_id,
        )
        self.assertTrue(
            all(
                plan.lcm_block_bytes % group.cache_blocks_per_lcm_block == 0
                for group in plan.groups
            )
        )

    def test_mxfp8_scale_sidecars_are_part_of_the_same_lcm_parent(self):
        builder = getattr(_fmp, "inkling_lcm_field_specs")
        try:
            fields = builder(
                layer_group_ids=(
                    "sliding_attention_0",
                    "sliding_attention_1",
                    "sliding_attention_2",
                    "sliding_attention_3",
                    "sliding_attention_4",
                    "full_attention",
                ),
                logical_block_tokens=128,
                layer_kv_heads=(2, 2, 2, 2, 2, 1),
                head_dim=128,
                kv_element_size=1,
                kv_scale_block_size=32,
                kv_scale_element_size=1,
                hidden_size=256,
                checkpoint_rows=3,
                kvconv_element_size=2,
                hiddenconv_element_size=1,
            )
        except TypeError as exc:
            self.fail(f"Inkling builder cannot describe MXFP8 scale fields: {exc}")

        scale_fields = [
            field
            for field in fields
            if field.field_id.endswith((".k_scale", ".v_scale"))
        ]
        self.assertEqual(len(scale_fields), 12)
        self.assertEqual(scale_fields[0].shape, (2, 1, 32, 4, 4))
        self.assertEqual(
            sum(
                field.payload_bytes
                for field in scale_fields
                if field.group_id == "sliding_attention_0"
            ),
            2 * 128 * 2 * 128 // 32,
        )

        plan = plan_lcm_fields(
            fields,
            logical_block_tokens=128,
            budget_bytes=2 * 67_584,
            alignment=256,
            max_padding_fraction=16.0,
        )
        self.assertEqual(plan.lcm_block_bytes, 67_584)
        self.assertEqual(
            {group.group_id: group.cache_blocks_per_lcm_block for group in plan.groups},
            {
                "full_attention": 2,
                "sliding_attention_0": 1,
                "sliding_attention_1": 1,
                "sliding_attention_2": 1,
                "sliding_attention_3": 1,
                "sliding_attention_4": 1,
                "kvconv": 2,
                "hiddenconv": 4,
            },
        )


class PerFieldLcmPackingTest(unittest.TestCase):
    """K must come from per-field ratios inside a plane, not the group byte ratio.

    A group's K only has to make its *exact* fields land on an integral page
    stride. Sizing K from the whole-group ratio (the old rule) under-counts
    whenever a group's fields are spread over several planes, which forced the
    history K down and doubled the KV footprint on Qwen3.5.
    """

    # Qwen3.5-397B, TP=8, P=128, fp32 SSM. One repeated hybrid unit.
    KV_PAGE = 128 * 1 * 256 * 2  # 65,536
    SSM = 8 * 128 * 128 * 4  # 524,288
    CONV = 1536 * 3 * 2  # 9,216

    def _unit_fields(self, units=1):
        fields = []
        for unit in range(units):
            fields += [
                LcmFieldSpec(
                    "full_attention", f"l{unit}.k", f"u{unit}.a", (self.KV_PAGE,), 1
                ),
                LcmFieldSpec(
                    "full_attention", f"l{unit}.v", f"u{unit}.b", (self.KV_PAGE,), 1
                ),
                LcmFieldSpec("state", f"l{unit}.ssm", f"u{unit}.a", (self.SSM,), 1),
                LcmFieldSpec(
                    "state",
                    f"l{unit}.conv",
                    f"u{unit}.b",
                    (self.CONV,),
                    1,
                    exact_page_stride=False,
                ),
            ]
        return fields

    def _plan(self, fields, **kwargs):
        kwargs.setdefault("logical_block_tokens", 128)
        kwargs.setdefault("budget_bytes", 8 << 30)
        kwargs.setdefault("alignment", 256)
        kwargs.setdefault("max_padding_fraction", 1.0)
        return plan_lcm_fields(fields, **kwargs)

    def test_packing_comes_from_per_field_ratio(self):
        plan = self._plan(self._unit_fields())
        packing = {g.group_id: g.cache_blocks_per_lcm_block for g in plan.groups}
        # ssm / kv_page == 8, so eight history pages tile one state page.
        self.assertEqual(packing, {"full_attention": 8, "state": 1})

    def test_exact_fields_get_payload_sized_page_stride(self):
        plan = self._plan(self._unit_fields())
        strides = {f.field_id.split(".")[-1]: f for f in plan.fields}
        for name in ("k", "v", "ssm"):
            field = strides[name]
            self.assertEqual(
                field.page_stride_bytes,
                field.payload_bytes,
                f"{name} must be addressable at exactly its payload stride",
            )

    def test_tolerant_field_may_carry_slack(self):
        plan = self._plan(self._unit_fields())
        conv = plan.field("l0.conv")
        # conv rides the plane sized by v; causal_conv1d reads stride(0) so the
        # gap is fine. Asserting it *is* padded keeps the intent explicit.
        self.assertGreater(conv.page_stride_bytes, conv.payload_bytes)

    def test_history_footprint_has_no_waste(self):
        plan = self._plan(self._unit_fields(units=15))
        full = plan.group("full_attention")
        self.assertEqual(full.slot_stride_bytes, full.raw_cache_block_bytes)
        self.assertEqual(full.padding_bytes_per_slot, 0)

    def test_single_group_is_an_identity(self):
        """Plain dense models must not pay anything for the LCM machinery."""
        plan = self._plan(
            [
                LcmFieldSpec("full_attention", "k", "a", (self.KV_PAGE,), 1),
                LcmFieldSpec("full_attention", "v", "b", (self.KV_PAGE,), 1),
            ]
        )
        group = plan.group("full_attention")
        self.assertEqual(group.cache_blocks_per_lcm_block, 1)
        self.assertEqual(group.slot_stride_bytes, group.raw_cache_block_bytes)
        self.assertEqual(group.padding_bytes_per_slot, 0)

    def test_equal_sized_groups_keep_packing_one(self):
        """GPT-OSS shape: full and sliding rows are byte-equal."""
        plan = self._plan(
            [
                LcmFieldSpec("full_attention", "f.k", "a", (self.KV_PAGE,), 1),
                LcmFieldSpec("full_attention", "f.v", "b", (self.KV_PAGE,), 1),
                LcmFieldSpec("sliding_attention", "s.k", "a", (self.KV_PAGE,), 1),
                LcmFieldSpec("sliding_attention", "s.v", "b", (self.KV_PAGE,), 1),
            ]
        )
        self.assertEqual(
            {g.group_id: g.cache_blocks_per_lcm_block for g in plan.groups},
            {"full_attention": 1, "sliding_attention": 1},
        )

    def test_three_state_groups_share_one_history_packing(self):
        """Qwen3.5 splits recurrent layers into three groups on one plane."""
        fields = [
            LcmFieldSpec("full_attention", "k", "a", (self.KV_PAGE,), 1),
            LcmFieldSpec("full_attention", "v", "b", (self.KV_PAGE,), 1),
        ]
        for i in range(3):
            fields += [
                LcmFieldSpec(f"state_{i}", f"s{i}.ssm", "a", (self.SSM,), 1),
                LcmFieldSpec(
                    f"state_{i}",
                    f"s{i}.conv",
                    "b",
                    (self.CONV,),
                    1,
                    exact_page_stride=False,
                ),
            ]
        plan = self._plan(fields)
        packing = {g.group_id: g.cache_blocks_per_lcm_block for g in plan.groups}
        self.assertEqual(packing["full_attention"], 8)
        for i in range(3):
            self.assertEqual(packing[f"state_{i}"], 1)

    def test_unservable_exact_stride_fails_closed(self):
        """Near-coprime payloads have no affordable exact solution.

        The planner falls back to ratio packing, and the verification pass then
        refuses the plan rather than handing a strided page to a kernel that
        assumes contiguity.
        """
        with self.assertRaisesRegex(ValueError, "page stride"):
            self._plan(
                [
                    LcmFieldSpec("full_attention", "k", "a", (self.KV_PAGE,), 1),
                    LcmFieldSpec("state", "ssm", "a", (530_000,), 1),
                ]
            )

    def test_contradictory_ratios_fail_closed(self):
        """One group cannot satisfy two different ratios at once."""
        with self.assertRaisesRegex(ValueError, "page stride"):
            self._plan(
                [
                    LcmFieldSpec("full_attention", "k", "a", (1024,), 1),
                    LcmFieldSpec("full_attention", "v", "b", (1024,), 1),
                    LcmFieldSpec("state", "s1", "a", (2048,), 1),
                    LcmFieldSpec("state", "s2", "b", (4096,), 1),
                ]
            )

    def test_wide_tolerant_cotenant_fails_closed(self):
        """A tolerant field can still size the plane and break an exact one.

        Scaling cannot rescue this -- both sides grow together -- so the plan is
        refused rather than handing the exact field an inflated stride.
        """
        with self.assertRaisesRegex(ValueError, "page stride"):
            self._plan(
                [
                    LcmFieldSpec("full_attention", "k", "a", (1024,), 1),
                    LcmFieldSpec("full_attention", "v", "b", (1024,), 1),
                    LcmFieldSpec("state", "ssm", "a", (1024,), 1),
                    LcmFieldSpec(
                        "state", "conv", "b", (8192,), 1, exact_page_stride=False
                    ),
                ],
                alignment=1,
            )

    def test_tolerant_only_plan_falls_back_without_error(self):
        """With nothing to keep exact, a costly ratio is fine."""
        plan = self._plan(
            [
                LcmFieldSpec("small", "s", "a", (1000,), 1, exact_page_stride=False),
                LcmFieldSpec("large", "l", "a", (2048,), 1, exact_page_stride=False),
            ],
            alignment=256,
            budget_bytes=1 << 20,
        )
        self.assertEqual(plan.group("small").cache_blocks_per_lcm_block, 2)
        self.assertEqual(plan.group("large").cache_blocks_per_lcm_block, 1)


class LcmArenaTest(unittest.TestCase):
    class _Backing:
        def __init__(self, nbytes):
            self.nbytes = nbytes

    def test_field_planes_share_one_backing_without_padding_history_rows(self):
        fields = []
        for unit in range(2):
            fields.extend(
                [
                    LcmFieldSpec("full", f"full.{unit}.k", f"{unit}.a", (64,), 1),
                    LcmFieldSpec("full", f"full.{unit}.v", f"{unit}.b", (64,), 1),
                ]
            )
            for state in ("state0", "state1", "state2"):
                fields.extend(
                    [
                        LcmFieldSpec(
                            state, f"{state}.{unit}.conv", f"{unit}.a", (512,), 1
                        ),
                        LcmFieldSpec(
                            state, f"{state}.{unit}.ssm", f"{unit}.b", (512,), 1
                        ),
                    ]
                )

        plan = plan_lcm_fields(
            fields,
            logical_block_tokens=128,
            budget_bytes=4 * 2048,
            alignment=128,
        )
        self.assertEqual(plan.lcm_block_bytes, 2048)
        self.assertEqual(plan.num_lcm_blocks, 3)
        self.assertEqual(plan.arena_bytes, 4 * 2048)
        self.assertEqual(plan.group("full").cache_blocks_per_lcm_block, 8)
        self.assertEqual(plan.group("state0").cache_blocks_per_lcm_block, 1)
        self.assertEqual(len(plan.planes), 4)

        for field in plan.fields:
            if field.group_id == "full":
                self.assertEqual(field.page_stride_bytes, field.payload_bytes)

        backing = self._Backing(plan.arena_bytes)
        arena = LcmArena(plan, backing)
        full_k = plan.field("full.0.k")
        state_conv = plan.field("state0.0.conv")
        # Same plane, same physical parent 1; the state page overlays all
        # eight full child pages.
        full_begin = arena.field_page_byte_offset(full_k.field_id, 1)
        state_begin = arena.field_page_byte_offset(state_conv.field_id, 1)
        self.assertEqual(full_begin, state_begin)
        self.assertEqual(
            arena.field_page_byte_offset(full_k.field_id, 8) + full_k.payload_bytes,
            state_begin + state_conv.payload_bytes,
        )

        with self.assertRaisesRegex(ValueError, "backing"):
            LcmArena(plan, self._Backing(plan.arena_bytes - 1))
        with self.assertRaisesRegex(IndexError, "page_id"):
            arena.field_page_byte_offset(full_k.field_id, plan.group("full").page_count)


if __name__ == "__main__":
    unittest.main()
