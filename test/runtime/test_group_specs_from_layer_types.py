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


_pcs = _load("paged_cache_spec_under_test", "paged_cache_spec.py")
group_specs_from_layer_types = _pcs.group_specs_from_layer_types
layer_group_ids = _pcs.layer_group_ids
PagedCacheGroupSpec = _pcs.PagedCacheGroupSpec


class GroupSpecsFromLayerTypesTest(unittest.TestCase):
    def test_gpt_oss_mixed_shape_yields_two_groups(self):
        layer_types = [
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ]
        specs = group_specs_from_layer_types(
            layer_types=layer_types,
            sliding_window_tokens=128,
            page_size=16,
        )
        self.assertEqual(len(specs), 2)
        by_id = {s.group_id: s for s in specs}
        self.assertIn("full_attention", by_id)
        self.assertIn("sliding_attention", by_id)

        full = by_id["full_attention"]
        self.assertEqual(full.retention, "full_history")
        self.assertIsNone(full.sliding_window_tokens)
        self.assertEqual(full.rows_per_page, 16)
        self.assertEqual(full.entry_stride_tokens, 1)
        self.assertEqual(full.family, "history")

        swa = by_id["sliding_attention"]
        self.assertEqual(swa.retention, "sliding_window")
        self.assertEqual(swa.sliding_window_tokens, 128)
        self.assertEqual(swa.rows_per_page, 16)
        self.assertEqual(swa.family, "history")

    def test_all_full_yields_single_group(self):
        # 等价于现状的单 group,无回归。
        specs = group_specs_from_layer_types(
            layer_types=["full_attention"] * 8,
            sliding_window_tokens=None,
            page_size=16,
        )
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].group_id, "full_attention")
        self.assertEqual(specs[0].retention, "full_history")
        self.assertIsNone(specs[0].sliding_window_tokens)

    def test_group_order_is_first_appearance(self):
        specs = group_specs_from_layer_types(
            layer_types=["sliding_attention", "full_attention", "full_attention"],
            sliding_window_tokens=64,
            page_size=8,
        )
        self.assertEqual(
            [s.group_id for s in specs],
            ["sliding_attention", "full_attention"],
        )

    def test_unknown_layer_type_raises(self):
        with self.assertRaises(ValueError):
            group_specs_from_layer_types(
                layer_types=["full_attention", "banana_attention"],
                sliding_window_tokens=None,
                page_size=16,
            )

    def test_sliding_without_window_raises(self):
        with self.assertRaises(ValueError):
            group_specs_from_layer_types(
                layer_types=["sliding_attention"],
                sliding_window_tokens=None,
                page_size=16,
            )

    def test_sliding_with_nonpositive_window_raises(self):
        with self.assertRaises(ValueError):
            group_specs_from_layer_types(
                layer_types=["sliding_attention"],
                sliding_window_tokens=0,
                page_size=16,
            )

    def test_qwen35_linear_attention_yields_state_group(self):
        layer_types = ["linear_attention", "linear_attention", "full_attention"]
        specs = group_specs_from_layer_types(
            layer_types=layer_types,
            sliding_window_tokens=None,
            page_size=16,
        )
        self.assertEqual(len(specs), 2)
        by_id = {s.group_id: s for s in specs}
        state = by_id["linear_attention"]
        self.assertEqual(state.family, "state")
        self.assertEqual(state.retention, "full_history")
        self.assertIsNone(state.sliding_window_tokens)
        self.assertEqual(by_id["full_attention"].family, "history")

    def test_qwen35_mixed_with_sliding_and_state_layers(self):
        layer_types = ["sliding_attention", "linear_attention", "full_attention"]
        specs = group_specs_from_layer_types(
            layer_types=layer_types,
            sliding_window_tokens=[128, None, None],
            page_size=16,
        )
        by_id = {s.group_id: s for s in specs}
        self.assertEqual(by_id["sliding_attention"].family, "history")
        self.assertEqual(by_id["sliding_attention"].sliding_window_tokens, 128)
        self.assertEqual(by_id["linear_attention"].family, "state")
        self.assertIsNone(by_id["linear_attention"].sliding_window_tokens)


class LayerGroupIdsTest(unittest.TestCase):
    """layer_group_ids 与 group_specs_from_layer_types 共享同一分组核,
    模型侧 PagedAttention(group_id=...) 从这里取值。"""

    def test_single_window_ids_equal_layer_types(self):
        # 回归锚:标量路径 id 必须逐元素等于 layer_types,页表 key 零变动。
        layer_types = ["full_attention", "sliding_attention"] * 12
        self.assertEqual(
            layer_group_ids(layer_types=layer_types, sliding_window_tokens=128),
            list(layer_types),
        )

    def test_multi_window_ids_gain_window_suffix(self):
        self.assertEqual(
            layer_group_ids(
                layer_types=[
                    "full_attention",
                    "sliding_attention",
                    "sliding_attention",
                ],
                sliding_window_tokens=[None, 4, 512],
            ),
            ["full_attention", "sliding_attention_4", "sliding_attention_512"],
        )

    def test_uniform_window_sequence_keeps_bare_labels(self):
        # 序列形式但只有一种 window -> 与标量等价,不加后缀。
        self.assertEqual(
            layer_group_ids(
                layer_types=[
                    "full_attention",
                    "sliding_attention",
                    "sliding_attention",
                ],
                sliding_window_tokens=[None, 128, 128],
            ),
            ["full_attention", "sliding_attention", "sliding_attention"],
        )

    def test_repeated_window_layers_share_group_id(self):
        self.assertEqual(
            layer_group_ids(
                layer_types=[
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                ],
                sliding_window_tokens=[4, 512, 4],
            ),
            ["sliding_attention_4", "sliding_attention_512", "sliding_attention_4"],
        )


class MultiWindowGroupSpecsTest(unittest.TestCase):
    def test_two_windows_yield_three_groups_in_first_appearance_order(self):
        specs = group_specs_from_layer_types(
            layer_types=[
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            sliding_window_tokens=[None, 4, 512, None],
            page_size=16,
        )
        self.assertEqual(
            [s.group_id for s in specs],
            ["full_attention", "sliding_attention_4", "sliding_attention_512"],
        )
        by_id = {s.group_id: s for s in specs}
        self.assertEqual(by_id["sliding_attention_4"].sliding_window_tokens, 4)
        self.assertEqual(by_id["sliding_attention_512"].sliding_window_tokens, 512)
        self.assertIsNone(by_id["full_attention"].sliding_window_tokens)
        for s in specs:
            self.assertEqual(s.rows_per_page, 16)

    def test_window_sequence_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            group_specs_from_layer_types(
                layer_types=["full_attention", "sliding_attention"],
                sliding_window_tokens=[None, 4, 512],
                page_size=16,
            )

    def test_sliding_layer_without_window_in_sequence_raises(self):
        with self.assertRaises(ValueError):
            group_specs_from_layer_types(
                layer_types=["full_attention", "sliding_attention"],
                sliding_window_tokens=[None, None],
                page_size=16,
            )

    def test_sliding_layer_nonpositive_window_in_sequence_raises(self):
        with self.assertRaises(ValueError):
            group_specs_from_layer_types(
                layer_types=["sliding_attention"],
                sliding_window_tokens=[0],
                page_size=16,
            )

    def test_full_layer_with_positive_window_in_sequence_raises(self):
        # 疑似标错 layer_type:响亮拒绝而非忽略。
        with self.assertRaises(ValueError):
            group_specs_from_layer_types(
                layer_types=["full_attention", "sliding_attention"],
                sliding_window_tokens=[64, 64],
                page_size=16,
            )

    def test_linear_layer_with_positive_window_in_sequence_raises(self):
        with self.assertRaises(ValueError):
            group_specs_from_layer_types(
                layer_types=["linear_attention", "full_attention"],
                sliding_window_tokens=[128, None],
                page_size=16,
            )

    def test_repeated_window_across_layers_dedups_to_one_group(self):
        specs = group_specs_from_layer_types(
            layer_types=[
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
            ],
            sliding_window_tokens=[None, 4, 512, 4],
            page_size=16,
        )
        self.assertEqual(
            [s.group_id for s in specs],
            ["full_attention", "sliding_attention_4", "sliding_attention_512"],
        )

    def test_bool_window_raises(self):
        with self.assertRaises(ValueError):
            group_specs_from_layer_types(
                layer_types=["sliding_attention"],
                sliding_window_tokens=True,
                page_size=16,
            )
        with self.assertRaises(ValueError):
            group_specs_from_layer_types(
                layer_types=["sliding_attention"],
                sliding_window_tokens=[True],
                page_size=16,
            )

    def test_float_window_raises(self):
        with self.assertRaises(ValueError):
            group_specs_from_layer_types(
                layer_types=["sliding_attention"],
                sliding_window_tokens=[4.7],
                page_size=16,
            )

    def test_scalar_str_window_raises(self):
        with self.assertRaises(ValueError):
            group_specs_from_layer_types(
                layer_types=["sliding_attention"],
                sliding_window_tokens="128",
                page_size=16,
            )

    def test_scalar_float_window_raises(self):
        with self.assertRaises(ValueError):
            group_specs_from_layer_types(
                layer_types=["sliding_attention"],
                sliding_window_tokens=4.5,
                page_size=16,
            )

    def test_scalar_window_with_full_layers_does_not_raise(self):
        # 标量广播不触发 full-layer 守卫(gpt-oss 现状:标量 128 + full 层共存)。
        specs = group_specs_from_layer_types(
            layer_types=["full_attention", "sliding_attention"],
            sliding_window_tokens=128,
            page_size=16,
        )
        self.assertEqual(len(specs), 2)


class PoolToPagedCacheGroupsIntegrationTest(unittest.TestCase):
    """pool_to_paged_cache_groups converts published specs to a multi-group
    scheduler config. Needs torch + the tokenspeed_scheduler ext; skips
    where those are absent."""

    def _import_converter(self):
        try:
            from tokenspeed.runtime.engine.scheduler_utils import (
                pool_to_paged_cache_groups,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(
                f"pool_to_paged_cache_groups unavailable (needs torch + "
                f"tokenspeed_scheduler ext): {exc}"
            )
        return pool_to_paged_cache_groups

    def test_two_group_specs_convert_to_two_scheduler_groups(self):
        from types import SimpleNamespace

        pool_to_paged_cache_groups = self._import_converter()

        specs = group_specs_from_layer_types(
            layer_types=["full_attention", "sliding_attention"],
            sliding_window_tokens=128,
            page_size=16,
        )
        # Duck-typed stand-in: only the two attributes the converter reads.
        fake_pool = SimpleNamespace(
            paged_cache_group_specs=specs,
            paged_cache_group_page_counts={s.group_id: 1024 for s in specs},
        )

        groups = pool_to_paged_cache_groups(fake_pool)

        self.assertEqual(len(groups), 2)
        group_ids = {g.group_id for g in groups}
        self.assertEqual(group_ids, {"full_attention", "sliding_attention"})

    def test_empty_specs_convert_to_no_groups(self):
        pool_to_paged_cache_groups = self._import_converter()

        from types import SimpleNamespace

        fake_pool = SimpleNamespace(
            paged_cache_group_specs=(),
            paged_cache_group_page_counts={},
        )
        self.assertEqual(pool_to_paged_cache_groups(fake_pool), [])


if __name__ == "__main__":
    unittest.main()
