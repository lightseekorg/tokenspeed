from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import unittest
from types import SimpleNamespace

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

_RUNTIME_DIR = (
    pathlib.Path(__file__).resolve().parents[2] / "python" / "tokenspeed" / "runtime"
)
_KV_CACHE_DIR = _RUNTIME_DIR / "layers" / "attention" / "kv_cache"
_RECIPES_DIR = _KV_CACHE_DIR / "recipes"


def _load(mod_name: str, file_path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: on py3.9 @dataclass + `from __future__ import
    # annotations` resolves field types via sys.modules[cls.__module__].
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# spec.py is self-contained: load it from the repo file under a private
# name (no real package import, no sys.modules shadowing needed).
_spec = _load("kv_cache_spec_under_test", _RECIPES_DIR / "spec.py")
layer_group_ids = _spec.layer_group_ids
CacheGroupSpec = _spec.CacheGroupSpec
_spec.CacheFieldSpec = _load(
    "kv_cache_plan_under_test", _RECIPES_DIR / "plan.py"
).CacheFieldSpec


def _group_specs(module, **kwargs):
    """The specs a layer vocabulary produces, via the one-walk ``group``.

    A group must declare fields, so a one-byte placeholder stands in: these
    tests are about scheduler semantics, not bytes.
    """
    return tuple(
        group_spec
        for group_spec, _ in module.group(
            fields_for_layer=lambda layer_id, group_id, occurrence: (
                module.CacheFieldSpec(
                    f"layer.{layer_id}.probe", f"unit.{occurrence}", (1,), "uint8"
                ),
            ),
            **kwargs,
        )
    )


def _specs(**kwargs):
    """Call ``group`` the way the recipes do: group_ids derived from the
    layer types unless the test supplies a finer split."""
    kwargs.setdefault(
        "group_ids",
        layer_group_ids(
            layer_types=kwargs["layer_types"],
            sliding_window_tokens=kwargs["sliding_window_tokens"],
        ),
    )
    return _group_specs(_spec, **kwargs)


class GroupSpecsFromLayerTypesTest(unittest.TestCase):
    def test_group_spec_has_no_producer_boundary_capability(self):
        self.assertNotIn(
            "materializes_all_boundaries",
            CacheGroupSpec.__dataclass_fields__,
        )

    def test_gpt_oss_mixed_shape_yields_two_groups(self):
        layer_types = [
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ]
        specs = _specs(
            layer_types=layer_types,
            sliding_window_tokens=128,
            prefix_granularity=16,
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
        specs = _specs(
            layer_types=["full_attention"] * 8,
            sliding_window_tokens=None,
            prefix_granularity=16,
        )
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].group_id, "full_attention")
        self.assertEqual(specs[0].retention, "full_history")
        self.assertIsNone(specs[0].sliding_window_tokens)

    def test_group_order_is_first_appearance(self):
        specs = _specs(
            layer_types=["sliding_attention", "full_attention", "full_attention"],
            sliding_window_tokens=64,
            prefix_granularity=8,
        )
        self.assertEqual(
            [s.group_id for s in specs],
            ["sliding_attention", "full_attention"],
        )

    def test_sliding_subgroups_yield_one_spec_per_subgroup(self):
        # Inkling step 2.5 shape: 5 sliding sub-groups + full -> 6 specs,
        # first-appearance order, all sliding specs share the one window.
        block = [f"sliding_attention_{k}" for k in range(5)] + ["full_attention"]
        specs = _specs(
            layer_types=block * 2,
            sliding_window_tokens=512,
            prefix_granularity=128,
        )
        self.assertEqual(
            [s.group_id for s in specs],
            block,
        )
        for s in specs[:5]:
            self.assertEqual(s.retention, "sliding_window")
            self.assertEqual(s.sliding_window_tokens, 512)
            self.assertEqual(s.family, "history")
        self.assertEqual(specs[5].retention, "full_history")
        self.assertIsNone(specs[5].sliding_window_tokens)
        # Single window: group ids stay equal to the raw labels.
        self.assertEqual(
            layer_group_ids(layer_types=block * 2, sliding_window_tokens=512),
            block * 2,
        )

    def test_sliding_subgroup_nondigit_suffix_raises(self):
        with self.assertRaises(ValueError):
            _specs(
                layer_types=["full_attention", "sliding_attention_x"],
                sliding_window_tokens=64,
                prefix_granularity=16,
            )

    def test_unknown_layer_type_raises(self):
        with self.assertRaises(ValueError):
            _specs(
                layer_types=["full_attention", "banana_attention"],
                sliding_window_tokens=None,
                prefix_granularity=16,
            )

    def test_sliding_without_window_raises(self):
        with self.assertRaises(ValueError):
            _specs(
                layer_types=["sliding_attention"],
                sliding_window_tokens=None,
                prefix_granularity=16,
            )

    def test_sliding_with_nonpositive_window_raises(self):
        with self.assertRaises(ValueError):
            _specs(
                layer_types=["sliding_attention"],
                sliding_window_tokens=0,
                prefix_granularity=16,
            )

    def test_qwen35_linear_attention_yields_state_group(self):
        layer_types = ["linear_attention", "linear_attention", "full_attention"]
        specs = _specs(
            layer_types=layer_types,
            sliding_window_tokens=None,
            prefix_granularity=16,
        )
        self.assertEqual(len(specs), 2)
        by_id = {s.group_id: s for s in specs}
        state = by_id["linear_attention"]
        self.assertEqual(state.family, "state")
        self.assertEqual(state.retention, "full_history")
        self.assertIsNone(state.sliding_window_tokens)
        # Snapshot-state groups declare checkpoint_granularity, not rows.
        self.assertEqual(state.checkpoint_granularity, 16)
        self.assertIsNone(state.rows_per_page)
        self.assertIsNone(state.entry_stride_tokens)
        self.assertEqual(state.block_granularity, 16)
        with self.assertRaises(TypeError):
            _ = state.page_size
        full = by_id["full_attention"]
        self.assertEqual(full.family, "history")
        self.assertIsNone(full.checkpoint_granularity)
        self.assertEqual(full.page_size, 16)
        self.assertEqual(full.block_granularity, 16)

    def test_qwen35_mixed_with_sliding_and_state_layers(self):
        layer_types = ["sliding_attention", "linear_attention", "full_attention"]
        specs = _specs(
            layer_types=layer_types,
            sliding_window_tokens=[128, None, None],
            prefix_granularity=16,
        )
        by_id = {s.group_id: s for s in specs}
        self.assertEqual(by_id["sliding_attention"].family, "history")
        self.assertEqual(by_id["sliding_attention"].sliding_window_tokens, 128)
        self.assertEqual(by_id["linear_attention"].family, "state")
        self.assertIsNone(by_id["linear_attention"].sliding_window_tokens)


class LayerGroupIdsTest(unittest.TestCase):
    def test_single_window_ids_equal_layer_types(self):
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
        specs = _specs(
            layer_types=[
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            sliding_window_tokens=[None, 4, 512, None],
            prefix_granularity=16,
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
            _specs(
                layer_types=["full_attention", "sliding_attention"],
                sliding_window_tokens=[None, 4, 512],
                prefix_granularity=16,
            )

    def test_sliding_layer_without_window_in_sequence_raises(self):
        with self.assertRaises(ValueError):
            _specs(
                layer_types=["full_attention", "sliding_attention"],
                sliding_window_tokens=[None, None],
                prefix_granularity=16,
            )

    def test_sliding_layer_nonpositive_window_in_sequence_raises(self):
        with self.assertRaises(ValueError):
            _specs(
                layer_types=["sliding_attention"],
                sliding_window_tokens=[0],
                prefix_granularity=16,
            )

    def test_full_layer_with_positive_window_in_sequence_raises(self):
        with self.assertRaises(ValueError):
            _specs(
                layer_types=["full_attention", "sliding_attention"],
                sliding_window_tokens=[64, 64],
                prefix_granularity=16,
            )

    def test_linear_layer_with_positive_window_in_sequence_raises(self):
        with self.assertRaises(ValueError):
            _specs(
                layer_types=["linear_attention", "full_attention"],
                sliding_window_tokens=[128, None],
                prefix_granularity=16,
            )

    def test_repeated_window_across_layers_dedups_to_one_group(self):
        specs = _specs(
            layer_types=[
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
            ],
            sliding_window_tokens=[None, 4, 512, 4],
            prefix_granularity=16,
        )
        self.assertEqual(
            [s.group_id for s in specs],
            ["full_attention", "sliding_attention_4", "sliding_attention_512"],
        )

    def test_bool_window_raises(self):
        with self.assertRaises(ValueError):
            _specs(
                layer_types=["sliding_attention"],
                sliding_window_tokens=True,
                prefix_granularity=16,
            )
        with self.assertRaises(ValueError):
            _specs(
                layer_types=["sliding_attention"],
                sliding_window_tokens=[True],
                prefix_granularity=16,
            )

    def test_float_window_raises(self):
        with self.assertRaises(ValueError):
            _specs(
                layer_types=["sliding_attention"],
                sliding_window_tokens=[4.7],
                prefix_granularity=16,
            )

    def test_scalar_str_window_raises(self):
        with self.assertRaises(ValueError):
            _specs(
                layer_types=["sliding_attention"],
                sliding_window_tokens="128",
                prefix_granularity=16,
            )

    def test_scalar_float_window_raises(self):
        with self.assertRaises(ValueError):
            _specs(
                layer_types=["sliding_attention"],
                sliding_window_tokens=4.5,
                prefix_granularity=16,
            )

    def test_scalar_window_with_full_layers_does_not_raise(self):
        specs = _specs(
            layer_types=["full_attention", "sliding_attention"],
            sliding_window_tokens=128,
            prefix_granularity=16,
        )
        self.assertEqual(len(specs), 2)


class CacheGroupSpecShapeTest(unittest.TestCase):
    """A spec declares exactly one geometry shape: row geometry (paged KV)
    or checkpoint_granularity (snapshot state)."""

    def test_row_geometry_shape(self):
        spec = CacheGroupSpec(
            group_id="kv",
            retention="full_history",
            rows_per_page=16,
            entry_stride_tokens=4,
            sliding_window_tokens=None,
        )
        self.assertEqual(spec.page_size, 64)
        self.assertEqual(spec.block_granularity, 64)
        self.assertIsNone(spec.checkpoint_granularity)

    def test_checkpoint_shape(self):
        spec = CacheGroupSpec(
            group_id="state",
            retention="full_history",
            sliding_window_tokens=None,
            family="state",
            checkpoint_granularity=64,
        )
        self.assertEqual(spec.block_granularity, 64)
        with self.assertRaises(TypeError):
            _ = spec.page_size

    def test_shapes_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            CacheGroupSpec(
                group_id="state",
                retention="full_history",
                rows_per_page=16,
                entry_stride_tokens=1,
                sliding_window_tokens=None,
                family="state",
                checkpoint_granularity=16,
            )

    def test_missing_shape_raises(self):
        with self.assertRaises(ValueError):
            CacheGroupSpec(
                group_id="kv",
                retention="full_history",
                sliding_window_tokens=None,
            )

    def test_partial_row_geometry_raises(self):
        with self.assertRaises(ValueError):
            CacheGroupSpec(
                group_id="kv",
                retention="full_history",
                rows_per_page=16,
                sliding_window_tokens=None,
            )

    def test_checkpoint_requires_state_family(self):
        with self.assertRaises(ValueError):
            CacheGroupSpec(
                group_id="kv",
                retention="full_history",
                sliding_window_tokens=None,
                family="history",
                checkpoint_granularity=16,
            )

    def test_nonpositive_geometry_raises(self):
        with self.assertRaises(ValueError):
            CacheGroupSpec(
                group_id="kv",
                retention="full_history",
                rows_per_page=0,
                entry_stride_tokens=1,
                sliding_window_tokens=None,
            )
        with self.assertRaises(ValueError):
            CacheGroupSpec(
                group_id="state",
                retention="full_history",
                sliding_window_tokens=None,
                family="state",
                checkpoint_granularity=0,
            )

    def test_state_family_may_keep_row_geometry(self):
        # V4-style row-buffer groups are state-family with real rows.
        spec = CacheGroupSpec(
            group_id="v4.compressor",
            retention="sliding_window",
            rows_per_page=16,
            entry_stride_tokens=4,
            sliding_window_tokens=256,
            family="state",
        )
        self.assertEqual(spec.page_size, 64)
        self.assertEqual(spec.block_granularity, 64)


def _fake_pool(specs, *, packing=1) -> SimpleNamespace:
    """A cache view whose arena publishes ``specs`` as its contract.

    Page counts and packing are the plan's facts, carried by the contract
    beside the specs -- the bridge reads them from there, never off a spec.
    """
    return SimpleNamespace(
        arena=SimpleNamespace(
            runtime_contract=SimpleNamespace(
                group_specs=tuple(specs),
                group_page_counts={spec.group_id: 1024 for spec in specs},
                group_packing={spec.group_id: packing for spec in specs},
            )
        )
    )


class PoolToCacheGroupsIntegrationTest(unittest.TestCase):
    """pool_to_cache_groups converts published specs to a multi-group
    scheduler config. Needs torch + the tokenspeed_scheduler ext; skips
    where those are absent."""

    def _import_converter(self):
        try:
            from tokenspeed.runtime.engine.scheduler_utils import (
                pool_to_cache_groups,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(
                f"pool_to_cache_groups unavailable (needs torch + "
                f"tokenspeed_scheduler ext): {exc}"
            )
        return pool_to_cache_groups

    def test_two_group_specs_convert_to_two_scheduler_groups(self):
        from types import SimpleNamespace

        pool_to_cache_groups = self._import_converter()

        specs = _specs(
            layer_types=["full_attention", "sliding_attention"],
            sliding_window_tokens=128,
            prefix_granularity=16,
        )
        # Duck-typed stand-in: a view naming an arena whose contract is
        # what the converter reads.
        fake_pool = _fake_pool(specs)

        groups = pool_to_cache_groups(fake_pool)

        self.assertEqual(len(groups), 2)
        group_ids = {g.group_id for g in groups}
        self.assertEqual(group_ids, {"full_attention", "sliding_attention"})

    def test_checkpoint_spec_folds_to_row_geometry_at_the_bridge(self):
        from types import SimpleNamespace

        pool_to_cache_groups = self._import_converter()

        specs = _specs(
            layer_types=["linear_attention", "full_attention"],
            sliding_window_tokens=None,
            prefix_granularity=16,
        )
        fake_pool = _fake_pool(specs)

        groups = {g.group_id: g for g in pool_to_cache_groups(fake_pool)}

        state = groups["linear_attention"]
        self.assertEqual(state.rows_per_page, 16)
        self.assertEqual(state.entry_stride_tokens, 1)


if __name__ == "__main__":
    unittest.main()
