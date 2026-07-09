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
    spec = importlib.util.spec_from_file_location(
        mod_name, _CONFIGS_DIR / file_name
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: on py3.9 @dataclass + `from __future__ import
    # annotations` resolves field types via sys.modules[cls.__module__].
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_fmp = _load("flat_memory_plan_under_test", "flat_memory_plan.py")
ComponentSpec = _fmp.ComponentSpec
PageGeometry = _fmp.PageGeometry
solve_page_geometry = _fmp.solve_page_geometry
plan_tensors = _fmp.plan_tensors


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
        geo = solve_page_geometry(comps, page_size_tokens=16, alignment=256)
        self.assertEqual(geo.page_size_tokens, 16)
        self.assertEqual(geo.page_bytes, 16 * 1024)

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
        geo = solve_page_geometry(comps, page_size_tokens=16, alignment=4)
        # A state layer's components pack into ONE page row ([conv|ssm|pad]),
        # so the constant demand is their SUM: ceil((40+60)KiB / 1KiB) = 100.
        self.assertEqual(geo.page_size_tokens, 100)
        self.assertEqual(geo.page_bytes, 100 * 1024)

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
        geo = solve_page_geometry(comps, page_size_tokens=16, alignment=16)
        # ceil(101K / 1K) = 101 -> rounded up to the next multiple of 16.
        self.assertEqual(geo.page_size_tokens, 112)
        self.assertEqual(geo.page_bytes, 112 * 1024)

    def test_dsv4_linear_rows_pad_not_inflate(self):
        comps = [
            ComponentSpec("full_mla", 0, "latent", bytes_per_slot=1152, const_bytes=0),
            ComponentSpec(
                "full_mla", 0, "indexer_k", bytes_per_slot=132, const_bytes=0
            ),
        ]
        geo = solve_page_geometry(comps, page_size_tokens=64, alignment=256)
        self.assertEqual(geo.page_size_tokens, 64)
        # Same-layer components pack into one row.
        self.assertEqual(geo.page_bytes, 64 * (1152 + 132))

    def test_constant_components_require_a_linear_row(self):
        comps = [
            ComponentSpec(
                "linear_attention", 0, "state", bytes_per_slot=0, const_bytes=1024
            )
        ]
        with self.assertRaises(ValueError):
            solve_page_geometry(comps, page_size_tokens=16, alignment=4)


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
            page_size_tokens=16,
            alignment=4,
            budget_bytes=100 * 1024 * 1024,
        )
        self.assertEqual(len(plan.tensors), 2)  # max group layer count = full 的 2 层
        # 每槽每组至多一层:槽 0 绑 full L0 + state L0(conv+ssm 两 binding),槽 1 只绑 full L1
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
            # 每槽每组唯一层(同层多 component 允许:conv+ssm 各占一 binding)。
            seen = {}
            for b in t.bindings:
                key = (b.slot, b.group_id)
                self.assertEqual(seen.setdefault(key, b.layer), b.layer)

    def test_row_offsets_accumulate_within_a_row(self):
        plan = plan_tensors(
            self._comps_qwen35(),
            page_size_tokens=16,
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

    def test_num_pages_from_budget_uniform_across_slots(self):
        plan = plan_tensors(
            self._comps_qwen35(),
            page_size_tokens=16,
            alignment=4,
            budget_bytes=100 * 1024 * 1024,
        )
        geo = plan.geometry
        self.assertEqual(geo.page_size_tokens, 100)  # 等化继承 B2:state 100KiB / 1KiB
        self.assertEqual(geo.page_bytes, 100 * 1024)
        self.assertEqual(
            geo.num_pages, 100 * 1024 * 1024 // (2 * 100 * 1024)
        )  # 512
        for t in plan.tensors:
            self.assertEqual(t.nbytes, geo.num_pages * geo.page_bytes)

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
            comps, page_size_tokens=16, alignment=4, budget_bytes=64 * 1024 * 1024
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
                page_size_tokens=16,
                alignment=4,
                budget_bytes=100 * 1024,
            )


if __name__ == "__main__":
    unittest.main()
