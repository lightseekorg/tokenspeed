from __future__ import annotations

import importlib.util
import math
import os
import pathlib
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

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


class TestGroupSpecsFromLayerTypesTest:
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
        assert (len(specs)) == (2)
        by_id = {s.group_id: s for s in specs}
        assert ("full_attention") in (by_id)
        assert ("sliding_attention") in (by_id)

        full = by_id["full_attention"]
        assert (full.attn_kind) == ("full_history")
        assert (full.sliding_window_tokens) is None
        assert (full.rows_per_page) == (16)
        assert (full.entry_stride_tokens) == (1)

        swa = by_id["sliding_attention"]
        assert (swa.attn_kind) == ("sliding_window")
        assert (swa.sliding_window_tokens) == (128)
        assert (swa.rows_per_page) == (16)

    def test_all_full_yields_single_group(self):
        specs = _specs(
            layer_types=["full_attention"] * 8,
            sliding_window_tokens=None,
            prefix_granularity=16,
        )
        assert (len(specs)) == (1)
        assert (specs[0].group_id) == ("full_attention")
        assert (specs[0].attn_kind) == ("full_history")
        assert (specs[0].sliding_window_tokens) is None

    def test_group_order_is_first_appearance(self):
        specs = _specs(
            layer_types=["sliding_attention", "full_attention", "full_attention"],
            sliding_window_tokens=64,
            prefix_granularity=8,
        )
        assert ([s.group_id for s in specs]) == (
            ["sliding_attention", "full_attention"]
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
        assert ([s.group_id for s in specs]) == (block)
        for s in specs[:5]:
            assert (s.attn_kind) == ("sliding_window")
            assert (s.sliding_window_tokens) == (512)
        assert (specs[5].attn_kind) == ("full_history")
        assert (specs[5].sliding_window_tokens) is None
        # Single window: group ids stay equal to the raw labels.
        assert (layer_group_ids(layer_types=block * 2, sliding_window_tokens=512)) == (
            block * 2
        )

    def test_sliding_subgroup_nondigit_suffix_raises(self):
        with pytest.raises((ValueError)):
            _specs(
                layer_types=["full_attention", "sliding_attention_x"],
                sliding_window_tokens=64,
                prefix_granularity=16,
            )

    def test_unknown_layer_type_raises(self):
        with pytest.raises((ValueError)):
            _specs(
                layer_types=["full_attention", "banana_attention"],
                sliding_window_tokens=None,
                prefix_granularity=16,
            )

    def test_sliding_without_window_raises(self):
        with pytest.raises((ValueError)):
            _specs(
                layer_types=["sliding_attention"],
                sliding_window_tokens=None,
                prefix_granularity=16,
            )

    def test_qwen35_linear_attention_yields_state_group(self):
        layer_types = ["linear_attention", "linear_attention", "full_attention"]
        specs = _specs(
            layer_types=layer_types,
            sliding_window_tokens=None,
            prefix_granularity=16,
        )
        assert (len(specs)) == (2)
        by_id = {s.group_id: s for s in specs}
        state = by_id["linear_attention"]
        assert (state.attn_kind) == ("state")
        assert (state.sliding_window_tokens) is None
        # State groups declare no rows and cover no token span.
        assert (state.rows_per_page) is None
        assert (state.entry_stride_tokens) is None
        with pytest.raises((TypeError)):
            _ = state.block_granularity
        with pytest.raises((TypeError)):
            _ = state.page_size
        full = by_id["full_attention"]
        assert (full.page_size) == (16)
        assert (full.block_granularity) == (16)

    def test_qwen35_mixed_with_sliding_and_state_layers(self):
        layer_types = ["sliding_attention", "linear_attention", "full_attention"]
        specs = _specs(
            layer_types=layer_types,
            sliding_window_tokens=[128, None, None],
            prefix_granularity=16,
        )
        by_id = {s.group_id: s for s in specs}
        assert (by_id["sliding_attention"].sliding_window_tokens) == (128)
        assert (by_id["linear_attention"].attn_kind) == ("state")
        assert (by_id["linear_attention"].sliding_window_tokens) is None


class TestLayerGroupIdsTest:
    def test_single_window_ids_equal_layer_types(self):
        layer_types = ["full_attention", "sliding_attention"] * 12
        assert (
            layer_group_ids(layer_types=layer_types, sliding_window_tokens=128)
        ) == (list(layer_types))

    def test_multi_window_ids_gain_window_suffix(self):
        assert (
            layer_group_ids(
                layer_types=[
                    "full_attention",
                    "sliding_attention",
                    "sliding_attention",
                ],
                sliding_window_tokens=[None, 4, 512],
            )
        ) == (["full_attention", "sliding_attention_4", "sliding_attention_512"])

    def test_uniform_window_sequence_keeps_bare_labels(self):
        assert (
            layer_group_ids(
                layer_types=[
                    "full_attention",
                    "sliding_attention",
                    "sliding_attention",
                ],
                sliding_window_tokens=[None, 128, 128],
            )
        ) == (["full_attention", "sliding_attention", "sliding_attention"])

    def test_repeated_window_layers_share_group_id(self):
        assert (
            layer_group_ids(
                layer_types=[
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                ],
                sliding_window_tokens=[4, 512, 4],
            )
        ) == (["sliding_attention_4", "sliding_attention_512", "sliding_attention_4"])


class TestMultiWindowGroupSpecsTest:
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
        assert ([s.group_id for s in specs]) == (
            ["full_attention", "sliding_attention_4", "sliding_attention_512"]
        )
        by_id = {s.group_id: s for s in specs}
        assert (by_id["sliding_attention_4"].sliding_window_tokens) == (4)
        assert (by_id["sliding_attention_512"].sliding_window_tokens) == (512)
        assert (by_id["full_attention"].sliding_window_tokens) is None
        for s in specs:
            assert (s.rows_per_page) == (16)

    def test_window_sequence_length_mismatch_raises(self):
        with pytest.raises((ValueError)):
            _specs(
                layer_types=["full_attention", "sliding_attention"],
                sliding_window_tokens=[None, 4, 512],
                prefix_granularity=16,
            )

    def test_sliding_layer_without_window_in_sequence_raises(self):
        with pytest.raises((ValueError)):
            _specs(
                layer_types=["full_attention", "sliding_attention"],
                sliding_window_tokens=[None, None],
                prefix_granularity=16,
            )

    def test_full_layer_with_positive_window_in_sequence_raises(self):
        with pytest.raises((ValueError)):
            _specs(
                layer_types=["full_attention", "sliding_attention"],
                sliding_window_tokens=[64, 64],
                prefix_granularity=16,
            )

    def test_linear_layer_with_positive_window_in_sequence_raises(self):
        with pytest.raises((ValueError)):
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
        assert ([s.group_id for s in specs]) == (
            ["full_attention", "sliding_attention_4", "sliding_attention_512"]
        )


class TestBroadcastWindowAsymmetryTest:
    """The one asymmetry between the two window spellings.

    A scalar reaches every layer, so non-sliding layers must ignore it; the
    same value written at that position in a sequence is a mislabeled layer
    (see ``test_full_layer_with_positive_window_in_sequence_raises``).
    """

    def test_broadcast_scalar_leaves_full_group_windowless(self):
        by_id = {
            spec.group_id: spec
            for spec in _specs(
                layer_types=["full_attention", "sliding_attention"],
                sliding_window_tokens=128,
                prefix_granularity=16,
            )
        }
        assert (by_id["full_attention"].sliding_window_tokens) is None
        assert (by_id["sliding_attention"].sliding_window_tokens) == (128)

    def test_broadcast_scalar_leaves_state_group_windowless(self):
        by_id = {
            spec.group_id: spec
            for spec in _specs(
                layer_types=["linear_attention", "sliding_attention"],
                sliding_window_tokens=128,
                prefix_granularity=16,
            )
        }
        state = by_id["linear_attention"]
        assert (state.attn_kind) == ("state")
        assert (state.sliding_window_tokens) is None


class TestCacheGroupSpecShapeTest:
    """A history spec (full history or sliding window) declares row geometry
    (paged KV); a state spec declares no span at all."""

    def test_row_geometry_shape(self):
        spec = CacheGroupSpec(
            group_id="kv",
            attn_kind="full_history",
            rows_per_page=16,
            entry_stride_tokens=4,
            sliding_window_tokens=None,
            replayable=False,
        )
        assert (spec.page_size) == (64)
        assert (spec.block_granularity) == (64)

    def test_state_group_declares_no_rows(self):
        # Rows of token history -- V4's compressor tails included -- are a
        # sliding-window group, not a state group with a window.
        with pytest.raises((ValueError), match=("no row geometry")):
            CacheGroupSpec(
                group_id="v4.compressor",
                attn_kind="state",
                rows_per_page=16,
                entry_stride_tokens=4,
                replayable=False,
            )
        spec = CacheGroupSpec(
            group_id="v4.compressor",
            attn_kind="sliding_window",
            rows_per_page=16,
            entry_stride_tokens=4,
            sliding_window_tokens=256,
            replayable=False,
        )
        assert (spec.page_size) == (64)
        assert (spec.block_granularity) == (64)

    def test_missing_row_geometry_raises(self):
        with pytest.raises((ValueError)):
            CacheGroupSpec(
                group_id="kv",
                attn_kind="full_history",
                sliding_window_tokens=None,
                replayable=False,
            )

    def test_state_group_has_no_window_and_is_never_replayed(self):
        # A checkpoint summarizes everything before it, so nothing in a state
        # group ever slides out, and nothing of it is regenerated by replay.
        with pytest.raises((ValueError), match=("only a sliding-window group")):
            CacheGroupSpec(
                group_id="state",
                attn_kind="state",
                sliding_window_tokens=256,
                replayable=False,
            )
        with pytest.raises((ValueError), match=("bounded replay")):
            CacheGroupSpec(group_id="state", attn_kind="state", replayable=True)

    def test_window_belongs_to_sliding_groups_only(self):
        with pytest.raises((ValueError), match=("only a sliding-window group")):
            CacheGroupSpec(
                group_id="kv",
                attn_kind="full_history",
                rows_per_page=16,
                entry_stride_tokens=1,
                sliding_window_tokens=256,
                replayable=False,
            )
        with pytest.raises((ValueError), match=("positive sliding_window_tokens")):
            CacheGroupSpec(
                group_id="swa",
                attn_kind="sliding_window",
                rows_per_page=16,
                entry_stride_tokens=1,
                replayable=False,
            )


def _fake_pool(specs, *, packing: int) -> SimpleNamespace:
    """A cache view whose arena publishes ``specs`` as its contract.

    Page counts and packing are the plan's facts, carried by the contract
    beside the specs -- the bridge reads them from there, never off a spec.
    """
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
        CacheRuntimeContract,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
        CacheGroupSpec as RuntimeCacheGroupSpec,
    )

    # The declaration tests use an isolated spec module. The actual contract
    # validates the runtime class, so cross that boundary explicitly here.
    runtime_specs = tuple(RuntimeCacheGroupSpec(**asdict(spec)) for spec in specs)
    prefix_granularity = math.lcm(
        *(spec.block_granularity for spec in specs if spec.attn_kind != "state")
    )
    contract = CacheRuntimeContract(
        prefix_granularity=prefix_granularity,
        num_lcm_blocks=1023,
        token_capacity=1023 * prefix_granularity,
        group_specs=runtime_specs,
        group_page_counts={spec.group_id: 1 + 1023 * packing for spec in specs},
        group_packing={spec.group_id: packing for spec in specs},
    )
    return SimpleNamespace(arena=SimpleNamespace(runtime_contract=contract))


class TestPoolToCacheGroupsIntegrationTest:
    """pool_to_cache_groups converts published specs to a multi-group
    scheduler config. Needs torch + the tokenspeed_scheduler ext; skips
    where those are absent."""

    def _import_converter(self):
        try:
            from tokenspeed.runtime.engine.scheduler_utils import (
                pool_to_cache_groups,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(
                f"pool_to_cache_groups unavailable (needs torch + "
                f"tokenspeed_scheduler ext): {exc}"
            )
        return pool_to_cache_groups

    def test_two_group_specs_convert_to_two_scheduler_groups(self):

        pool_to_cache_groups = self._import_converter()

        specs = _specs(
            layer_types=["full_attention", "sliding_attention"],
            sliding_window_tokens=128,
            prefix_granularity=16,
        )
        # Duck-typed stand-in: a view naming an arena whose contract is
        # what the converter reads.
        fake_pool = _fake_pool(specs, packing=1)

        groups = pool_to_cache_groups(fake_pool)

        assert (len(groups)) == (2)
        group_ids = {g.group_id for g in groups}
        assert (group_ids) == ({"full_attention", "sliding_attention"})

    def test_both_families_cross_the_bridge_with_a_bare_slot_span(self):
        """The scheduler config carries no declaration shape: a row-geometry
        spec arrives as its slot span, a state spec as span 0 (no span)."""
        pool_to_cache_groups = self._import_converter()

        specs = _specs(
            layer_types=["linear_attention", "full_attention"],
            sliding_window_tokens=None,
            prefix_granularity=16,
        )
        fake_pool = _fake_pool(specs, packing=1)

        groups = {g.group_id: g for g in pool_to_cache_groups(fake_pool)}

        state = groups["linear_attention"]
        assert (state.block_granularity) == (0)
        assert not (hasattr(state, "rows_per_page"))
        assert not (hasattr(state, "entry_stride_tokens"))
        full = groups["full_attention"]
        assert (full.block_granularity) == (16)

    def test_sharded_group_exports_virtual_capacity_and_packing(self):
        pool_to_cache_groups = self._import_converter()
        specs = _specs(
            layer_types=["full_attention", "sliding_attention"],
            sliding_window_tokens=128,
            prefix_granularity=16,
        )
        specs = tuple(
            replace(spec, shard_count=4) if spec.group_id == "full_attention" else spec
            for spec in specs
        )
        pool = _fake_pool(specs, packing=2)
        groups = {group.group_id: group for group in pool_to_cache_groups(pool)}
        full = groups["full_attention"]
        assert (full.total_pages) == (1 + 1023 * 2 * 4)
        assert (full.cache_blocks_per_lcm_block) == (8)
        assert (full.shard_count) == (4)
        sliding = groups["sliding_attention"]
        assert (sliding.total_pages) == (1 + 1023 * 2)
        assert (sliding.cache_blocks_per_lcm_block) == (2)
        assert (sliding.shard_count) == (1)
        assert (full.block_granularity) == (16)
        assert (sliding.block_granularity) == (16)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
