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

_PYTHON_DIR = pathlib.Path(__file__).resolve().parents[2] / "python"


def _load(module_name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_lcm_modules():
    package_paths = {
        "tokenspeed": _PYTHON_DIR / "tokenspeed",
        "tokenspeed.runtime": _PYTHON_DIR / "tokenspeed" / "runtime",
        "tokenspeed.runtime.configs": (
            _PYTHON_DIR / "tokenspeed" / "runtime" / "configs"
        ),
        "tokenspeed.runtime.layers": (
            _PYTHON_DIR / "tokenspeed" / "runtime" / "layers"
        ),
        "tokenspeed.runtime.layers.attention": (
            _PYTHON_DIR / "tokenspeed" / "runtime" / "layers" / "attention"
        ),
        "tokenspeed.runtime.layers.attention.kv_cache": (
            _PYTHON_DIR
            / "tokenspeed"
            / "runtime"
            / "layers"
            / "attention"
            / "kv_cache"
        ),
    }
    for package_name, package_path in package_paths.items():
        package = sys.modules.setdefault(package_name, types.ModuleType(package_name))
        package.__path__ = [str(package_path)]

    plan = _load(
        "tokenspeed.runtime.configs.lcm_memory_plan",
        package_paths["tokenspeed.runtime.configs"] / "lcm_memory_plan.py",
    )
    arena = _load(
        "tokenspeed.runtime.layers.attention.kv_cache.lcm_arena",
        package_paths["tokenspeed.runtime.layers.attention.kv_cache"]
        / "lcm_arena.py",
    )
    return plan, arena


def _small_plan(plan_module):
    field = plan_module.LcmFieldSpec
    return plan_module.plan_lcm_fields(
        (
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
        ),
        logical_block_tokens=128,
        budget_bytes=8192,
        alignment=256,
    )


class LcmArenaGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan_module, cls.arena_module = _load_lcm_modules()
        cls.plan = _small_plan(cls.plan_module)

    def test_backing_size_must_match_plan(self):
        backing = types.SimpleNamespace(nbytes=self.plan.arena_bytes - 1)

        with self.assertRaisesRegex(ValueError, "plan requires"):
            self.arena_module.LcmArena(self.plan, backing)

    def test_page_segments_map_group_pages_to_exact_byte_ranges(self):
        backing = types.SimpleNamespace(nbytes=self.plan.arena_bytes)
        arena = self.arena_module.LcmArena(self.plan, backing)

        self.assertEqual(
            arena.page_byte_segments("state", [0]),
            [(1792, 256), (5888, 256)],
        )

    def test_page_segments_reject_out_of_range_page(self):
        backing = types.SimpleNamespace(nbytes=self.plan.arena_bytes)
        arena = self.arena_module.LcmArena(self.plan, backing)
        page_count = self.plan.group("state").page_count

        with self.assertRaises(IndexError):
            arena.page_byte_segments("state", [page_count])


class LcmArenaTensorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest("torch is not installed in the local venv") from exc
        cls.torch = torch
        cls.plan_module, cls.arena_module = _load_lcm_modules()
        cls.plan = _small_plan(cls.plan_module)

    def test_field_views_alias_one_backing(self):
        arena = self.arena_module.LcmArena.allocate(self.plan, "cpu")

        history = arena.field_view("history.k", self.torch.bfloat16)
        state = arena.field_view("state.ssm", self.torch.bfloat16)

        self.assertEqual(
            history.untyped_storage().data_ptr(),
            arena.backing.untyped_storage().data_ptr(),
        )
        self.assertEqual(
            state.untyped_storage().data_ptr(),
            arena.backing.untyped_storage().data_ptr(),
        )

    def test_field_view_rejects_wrong_dtype_width(self):
        arena = self.arena_module.LcmArena.allocate(self.plan, "cpu")

        with self.assertRaisesRegex(ValueError, "dtype itemsize"):
            arena.field_view("history.k", self.torch.float32)


if __name__ == "__main__":
    unittest.main()
