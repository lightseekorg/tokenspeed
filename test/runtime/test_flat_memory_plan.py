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


if __name__ == "__main__":
    unittest.main()
