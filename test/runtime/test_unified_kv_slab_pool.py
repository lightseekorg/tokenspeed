"""Hybrid slab sizing and buffer-layout tests."""

from __future__ import annotations

import importlib.util
import itertools
import os
import pathlib
import sys
import unittest
from functools import partial

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
_pcs = _load("kv_cache_spec_slab_under_test", _RECIPES_DIR / "spec.py")
hybrid_slab_group_size = _pcs.hybrid_slab_group_size


def _specs_for_layers(**kwargs):
    from cache_pool_test_utils import specs_for_layers

    return specs_for_layers(**kwargs)


def _real_spec():
    # The REAL package module (not the file-loaded shadow above): pool
    # constructions need specs whose class passes the contract's
    # isinstance check.
    from tokenspeed.runtime.layers.attention.kv_cache.recipes import spec

    return spec


GPT_OSS_LAYER_TYPES = ("sliding_attention", "full_attention") * 12

# One Inkling-style layer block: 5 sliding sub-groups + 1 full layer.
# Repeated N times, all 6 groups have equal count N (fully bound slabs).
SUBGROUP_LAYER_BLOCK = tuple(f"sliding_attention_{k}" for k in range(5)) + (
    "full_attention",
)


class HybridSlabGroupSizeTest(unittest.TestCase):
    """Each case pins exactly ONE reason the predicate returns None (or the
    single shape where it activates)."""

    def test_gpt_oss_shape_returns_group_size(self):
        # gpt-oss: 12 sliding + 12 full, alternating -> 12 layers per group.
        self.assertEqual(
            hybrid_slab_group_size(GPT_OSS_LAYER_TYPES),
            12,
        )

    def test_none_when_single_group(self):
        self.assertIsNone(hybrid_slab_group_size(("full_attention",) * 24))

    def test_unequal_groups_return_largest_count(self):
        # Unequal groups (e.g. Inkling: 55 sliding + 11 full): the slab
        # count is the largest group's layer count; slabs past the smaller
        # group's count are single-layer.
        lt = ("sliding_attention",) * 8 + ("full_attention",) * 16
        self.assertEqual(hybrid_slab_group_size(lt), 16)
        lt_inkling = ("sliding_attention",) * 55 + ("full_attention",) * 11
        self.assertEqual(hybrid_slab_group_size(lt_inkling), 55)

    def test_sliding_subgroups_return_group_size(self):
        # Inkling step 2.5: 5 sliding sub-groups + full, all count 11 ->
        # 11 slabs, every slab bound by one layer of each of the 6 groups.
        lt = SUBGROUP_LAYER_BLOCK * 11
        self.assertEqual(
            hybrid_slab_group_size(lt, sliding_window_tokens=512),
            11,
        )

    def test_none_when_subgroup_suffix_not_digit(self):
        lt = GPT_OSS_LAYER_TYPES + ("sliding_attention_x",)
        self.assertIsNone(hybrid_slab_group_size(lt))

    def test_none_when_unknown_label(self):
        # Unknown input degrades to None (safe legacy layout), never raises;
        # loud rejection is spec.group's job.
        lt = GPT_OSS_LAYER_TYPES + ("banana_attention",)
        self.assertIsNone(hybrid_slab_group_size(lt))

    def test_none_when_empty(self):
        # Plain models pass empty or None layer_types.
        self.assertIsNone(hybrid_slab_group_size(()))
        self.assertIsNone(hybrid_slab_group_size(None))

    def test_none_when_multi_window_sequence(self):
        it = itertools.cycle((4, 512))
        windows = [
            next(it) if t == "sliding_attention" else None for t in GPT_OSS_LAYER_TYPES
        ]
        self.assertIsNone(
            hybrid_slab_group_size(
                GPT_OSS_LAYER_TYPES,
                sliding_window_tokens=windows,
            )
        )

    def test_uniform_window_sequence_stays_active(self):
        windows = [None if t == "full_attention" else 128 for t in GPT_OSS_LAYER_TYPES]
        self.assertEqual(
            hybrid_slab_group_size(
                GPT_OSS_LAYER_TYPES,
                sliding_window_tokens=windows,
            ),
            12,
        )

    def test_scalar_window_stays_active(self):
        self.assertEqual(
            hybrid_slab_group_size(
                GPT_OSS_LAYER_TYPES,
                sliding_window_tokens=128,
            ),
            12,
        )

    def test_none_when_window_sequence_length_mismatch(self):
        self.assertIsNone(
            hybrid_slab_group_size(
                GPT_OSS_LAYER_TYPES,
                sliding_window_tokens=[128],
            )
        )

    def test_garbage_elements_ignored_not_raised(self):
        self.assertEqual(
            hybrid_slab_group_size(
                GPT_OSS_LAYER_TYPES,
                sliding_window_tokens=["a"] * len(GPT_OSS_LAYER_TYPES),
            ),
            12,
        )


class MHAPoolSlabLayoutTest(unittest.TestCase):
    """The memory plan may alias fields while MHA keeps one view per layer.

    When placement sharing is active, paired layer views address the same
    bytes without sharing a Python tensor object.
    Constructs a real (tiny, CPU) MHATokenToKVPool; skips without deps.
    """

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.kv_cache.mha import (
                MHATokenToKVPool,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.MHATokenToKVPool = MHATokenToKVPool

    def _pool(self, **overrides):
        from cache_pool_test_utils import (
            make_mha_memory_plan,
            make_pool,
        )

        kwargs = {
            "size": 32,
            "dtype": self.torch.bfloat16,
            "head_num": 1,
            "head_dim": 8,
            "layer_num": 24,
            "device": "cpu",
            "enable_memory_saver": False,
            "prefix_granularity": 16,
            "rank": 0,
            "layer_types": GPT_OSS_LAYER_TYPES,
            "sliding_window_tokens": 128,
        }
        kwargs.update(overrides)
        plan = make_mha_memory_plan(
            size=kwargs["size"],
            prefix_granularity=kwargs["prefix_granularity"],
            layer_num=kwargs["layer_num"],
            kv_heads=kwargs["head_num"],
            head_dim=kwargs["head_dim"],
            dtype=kwargs["dtype"],
            layer_types=kwargs.get("layer_types", ()),
            sliding_window_tokens=kwargs.get("sliding_window_tokens"),
        )
        kwargs.pop("sliding_window_tokens", None)
        # The arena owns the allocation; ``size`` and the plan geometry are
        # its properties, not the view's.
        device = kwargs.pop("device")
        kwargs.pop("size")
        kwargs.pop("prefix_granularity")
        kwargs.pop("enable_memory_saver")
        arena, pool = make_pool(self.MHATokenToKVPool, plan, device=device, **kwargs)
        self.arena = arena
        return pool

    def test_plan_aliases_distinct_layer_views(self):
        pool = self._pool()
        self.assertEqual(len(pool.k_buffer), 24)
        self.assertEqual(len({id(t) for t in pool.k_buffer}), 24)
        self.assertEqual(len({id(t) for t in pool.v_buffer}), 24)
        for i in range(12):
            self.assertIsNot(pool.k_buffer[2 * i], pool.k_buffer[2 * i + 1])
            self.assertIsNot(pool.v_buffer[2 * i], pool.v_buffer[2 * i + 1])
            self.assertEqual(
                pool.k_buffer[2 * i].data_ptr(),
                pool.k_buffer[2 * i + 1].data_ptr(),
            )
        for buffers in (pool.k_buffer, pool.v_buffer):
            address_to_layers: dict[int, list[int]] = {}
            for layer_id, tensor in enumerate(buffers):
                address_to_layers.setdefault(tensor.data_ptr(), []).append(layer_id)
            self.assertEqual(len(address_to_layers), 12)
            for layer_ids in address_to_layers.values():
                sliding = [lid for lid in layer_ids if lid % 2 == 0]
                full = [lid for lid in layer_ids if lid % 2 == 1]
                self.assertEqual(len(sliding), 1)
                self.assertEqual(len(full), 1)
        self.assertEqual(len({t.data_ptr() for t in pool.k_buffer}), 12)
        self.assertEqual(len({t.data_ptr() for t in pool.v_buffer}), 12)

    def test_fallback_matrix_keeps_24_buffers(self):
        cases = {
            "single_group": {
                "layer_types": ("full_attention",) * 24,
                "sliding_window_tokens": None,
            }
        }
        for name, overrides in cases.items():
            with self.subTest(name):
                pool = self._pool(**overrides)
                self.assertEqual(len({id(t) for t in pool.k_buffer}), 24)
                self.assertEqual(len({id(t) for t in pool.v_buffer}), 24)

    def test_unequal_groups_pair_smaller_into_larger(self):
        # 8 sliding + 16 full -> 16 slabs: the 8 sliding layers alias the
        # first 8 full layers' slabs; full layers 8..15 keep solo slabs
        # (the Inkling shape, 55 sliding + 11 full, scaled down).
        pool = self._pool(
            layer_types=("sliding_attention",) * 8 + ("full_attention",) * 16
        )
        self.assertEqual(len(pool.k_buffer), 24)
        self.assertEqual(len({id(t) for t in pool.k_buffer}), 24)
        self.assertEqual(len({id(t) for t in pool.v_buffer}), 24)
        self.assertEqual(len({t.data_ptr() for t in pool.k_buffer}), 16)
        self.assertEqual(len({t.data_ptr() for t in pool.v_buffer}), 16)
        for i in range(8):
            self.assertEqual(
                pool.k_buffer[i].data_ptr(), pool.k_buffer[8 + i].data_ptr()
            )
            self.assertEqual(
                pool.v_buffer[i].data_ptr(), pool.v_buffer[8 + i].data_ptr()
            )
        solo = [t.data_ptr() for t in pool.k_buffer[16:]]
        self.assertEqual(len(set(solo)), 8)

    def test_sliding_subgroups_six_way_binding(self):
        # (s0..s4, full) x 2 -> 6 groups of 2 layers -> 2 slabs, each bound
        # by one layer of every group (the Inkling 5+1 shape, scaled down).
        pool = self._pool(
            layer_num=12,
            layer_types=SUBGROUP_LAYER_BLOCK * 2,
            sliding_window_tokens=512,
        )
        self.assertEqual(len(pool.k_buffer), 12)
        self.assertEqual(len({id(t) for t in pool.k_buffer}), 12)
        self.assertEqual(len({id(t) for t in pool.v_buffer}), 12)
        self.assertEqual(len({t.data_ptr() for t in pool.k_buffer}), 2)
        self.assertEqual(len({t.data_ptr() for t in pool.v_buffer}), 2)
        for i in range(6):
            self.assertEqual(pool.k_buffer[i].data_ptr(), pool.k_buffer[0].data_ptr())
            self.assertEqual(
                pool.k_buffer[6 + i].data_ptr(), pool.k_buffer[6].data_ptr()
            )
            self.assertEqual(pool.v_buffer[i].data_ptr(), pool.v_buffer[0].data_ptr())
            self.assertEqual(
                pool.v_buffer[6 + i].data_ptr(), pool.v_buffer[6].data_ptr()
            )
        self.assertNotEqual(pool.k_buffer[0].data_ptr(), pool.k_buffer[6].data_ptr())

    def test_aliased_sliding_plan_publishes_one_typed_pd_arena(self):
        from cache_pool_test_utils import make_layer_group_ids

        from tokenspeed.runtime.pd.cache_protocol import (
            build_arena_cache_transfer_contract,
        )

        group_ids = make_layer_group_ids(
            layer_num=len(GPT_OSS_LAYER_TYPES),
            layer_types=GPT_OSS_LAYER_TYPES,
            sliding_window_tokens=128,
        )
        specs = _specs_for_layers(
            layer_types=GPT_OSS_LAYER_TYPES,
            group_ids=group_ids,
            sliding_window_tokens=128,
            prefix_granularity=16,
            pd_disaggregation_enabled=True,
        )
        pool = self._pool(
            cache_group_specs=specs,
        )

        layout, base_addr = build_arena_cache_transfer_contract(pool.arena)

        self.assertTrue(pool.arena.supports_disaggregation)
        self.assertEqual(base_addr, pool.arena.buffer.data_ptr())
        self.assertEqual(layout.plan, pool.arena.plan)
        groups = {spec.retention: spec for spec in layout.group_specs}
        self.assertEqual(set(groups), {"full_history", "sliding_window"})
        self.assertIsNone(groups["full_history"].sliding_window_tokens)
        self.assertEqual(groups["sliding_window"].sliding_window_tokens, 128)

    def test_arena_rejects_an_empty_group_spec_set(self):
        # Publication is unconditional: an arena with no groups is one no
        # scheduler can address, so it is rejected at construction rather
        # than left in a contract-less mode for callers to discover.
        from cache_pool_test_utils import make_mha_memory_plan

        from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena

        plan = make_mha_memory_plan(
            size=32,
            prefix_granularity=16,
            layer_num=1,
            kv_heads=1,
            head_dim=8,
            dtype=self.torch.bfloat16,
            layer_types=("full_attention",),
        )

        with self.assertRaisesRegex(ValueError, "at least one cache group spec"):
            CacheArena(plan, "cpu", cache_group_specs=())

    def test_arena_aligns_recipe_specs_with_plan(self):
        # Recipe specs carry default packing; the arena overwrites it (and the
        # page counts) from the memory plan before publishing the contract.
        from cache_pool_test_utils import make_arena, make_mha_memory_plan

        plan = make_mha_memory_plan(
            size=32,
            prefix_granularity=16,
            layer_num=1,
            kv_heads=1,
            head_dim=8,
            dtype=self.torch.bfloat16,
            layer_types=("full_attention",),
        )
        arena = make_arena(
            plan,
            device="cpu",
            cache_group_specs=_specs_for_layers(
                layer_types=("full_attention",),
                group_ids=("full_attention",),
                sliding_window_tokens=None,
                prefix_granularity=16,
            ),
        )

        self.assertIsNotNone(arena.runtime_contract)
        self.assertEqual(
            [spec.group_id for spec in arena.cache_group_specs],
            ["full_attention"],
        )
        self.assertEqual(
            arena.cache_group_page_counts["full_attention"],
            plan.group("full_attention").page_count,
        )
        # No explicit token capacity: the arena falls back to child capacity.
        self.assertEqual(arena.runtime_contract.token_capacity, 32)


class MLAPoolAllocationHookTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.kv_cache.mla import (
                MLATokenToKVPool,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.MLATokenToKVPool = MLATokenToKVPool

    def test_pool_arranges_arena_views_without_allocating(self):
        torch = self.torch
        from cache_pool_test_utils import make_mla_memory_plan, make_pool

        arena, pool = make_pool(
            self.MLATokenToKVPool,
            make_mla_memory_plan(
                size=8,
                prefix_granularity=4,
                layer_num=1,
                latent_width=6,
                dtype=torch.bfloat16,
            ),
            model_dtype=torch.bfloat16,
            dtype=torch.bfloat16,
            quant_method=None,
            kv_lora_rank=4,
            qk_rope_head_dim=2,
            layer_num=1,
            rank=0,
            cache_group_specs=_specs_for_layers(
                layer_types=(),
                group_ids=("full_attention",),
                sliding_window_tokens=None,
                prefix_granularity=4,
            ),
        )

        # The pool allocates nothing: each per-layer buffer is a reshape of
        # the arena view the plan already materialized.
        self.assertEqual(
            pool.kv_buffer[0].untyped_storage().data_ptr(),
            arena.buffer.untyped_storage().data_ptr(),
        )
        self.assertEqual(tuple(pool.kv_buffer[0].shape[1:]), (1, 6))
        self.assertEqual(
            [spec.group_id for spec in pool.arena.cache_group_specs],
            ["full_attention"],
        )
        self.assertGreater(
            pool.arena.cache_group_page_counts["full_attention"],
            1,
        )


class StateCacheGroupPageCountTest(unittest.TestCase):
    """compute_cache_group_page_counts: the family="state" branch is
    positive and bounded by the full-history formula for the same inputs
    (sparse state uses two rolling checkpoints independent of overlap and
    prefill chunk width).
    The direct-loaded module still imports ceil_div from the real package
    at call time, so this skips on a bare interpreter.
    """

    def setUp(self):
        try:
            from tokenspeed.runtime.utils.common import ceil_div  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"page-count math needs the real package: {exc}")

    def _counts(self, **overrides):
        specs = _specs_for_layers(
            layer_types=("linear_attention", "full_attention"),
            group_ids=("linear_attention", "full_attention"),
            sliding_window_tokens=None,
            prefix_granularity=16,
        )
        params = {
            "max_live_requests": 2,
            "max_scheduled_tokens": 64,
            "max_total_tokens": 1024,
            "max_context_len": 4096,
        }
        params.update(overrides)
        return _pcs.compute_cache_group_page_counts(specs, **params)

    def test_state_count_positive_and_bounded_by_full_history(self):
        counts = self._counts()
        self.assertGreater(counts["linear_attention"], 0)
        self.assertLessEqual(counts["linear_attention"], counts["full_attention"])

    def test_state_branch_is_independent_of_chunk_and_total_token_width(self):
        small = self._counts(max_scheduled_tokens=16, max_total_tokens=128)
        large = self._counts(max_scheduled_tokens=4096, max_total_tokens=1 << 20)
        self.assertEqual(small["linear_attention"], large["linear_attention"])
        # R=2: two input/output blocks each plus one allocator dummy page.
        self.assertEqual(small["linear_attention"], 2 * 2 + 1)

    def test_state_branch_is_independent_of_overlap(self):
        baseline = self._counts(overlap_schedule_depth=0, decode_input_tokens=1)
        overlapped = self._counts(overlap_schedule_depth=1, decode_input_tokens=1)
        self.assertEqual(baseline["linear_attention"], overlapped["linear_attention"])


class CachePoolFieldBindingTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            from cache_pool_test_utils import plan_fields

            from tokenspeed.runtime.layers.attention.kv_cache.hybrid_mha import (
                HybridMHATokenToKVPool,
            )
            from tokenspeed.runtime.layers.attention.kv_cache.mha import (
                MHATokenToKVPool,
            )
            from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
                CacheFieldSpec,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.pool_cls = HybridMHATokenToKVPool
        self.mha_pool_cls = MHATokenToKVPool
        # Qwen-shaped: one GDN state layer (ssm + conv) and one KV layer, the
        # state fields aliasing the KV planes.
        fields = {
            "linear_attention_0": (
                CacheFieldSpec("layer.0.ssm", "unit.0.a", (1, 2, 2), "bfloat16"),
                CacheFieldSpec(
                    "layer.0.conv",
                    "unit.0.b",
                    (2, 2),
                    "bfloat16",
                    exact_page_stride=False,
                ),
            ),
            "full_attention": (
                CacheFieldSpec("layer.1.k", "unit.0.a", (4, 1, 2), "bfloat16"),
                CacheFieldSpec("layer.1.v", "unit.0.b", (4, 1, 2), "bfloat16"),
            ),
        }
        self.plan = plan_fields(
            fields,
            prefix_granularity=4,
            budget_bytes=64,
            alignment=2,
        )

    def _pool(self):
        from cache_pool_test_utils import make_pool

        _, pool = make_pool(
            self.pool_cls,
            self.plan,
            dtype=self.torch.bfloat16,
            head_num=1,
            head_dim=2,
            layer_num=2,
            rank=0,
            layer_types=("linear_attention", "full_attention"),
            cache_group_specs=_specs_for_layers(
                layer_types=("linear_attention", "full_attention"),
                group_ids=("linear_attention_0", "full_attention"),
                sliding_window_tokens=None,
                prefix_granularity=4,
            ),
        )
        return pool

    def _ordinary_pool(self):
        from cache_pool_test_utils import make_mha_memory_plan, make_pool

        _, pool = make_pool(
            self.mha_pool_cls,
            make_mha_memory_plan(
                size=8,
                prefix_granularity=4,
                layer_num=2,
                kv_heads=1,
                head_dim=2,
                dtype=self.torch.bfloat16,
                layer_types=("linear_attention", "full_attention"),
            ),
            dtype=self.torch.bfloat16,
            head_num=1,
            head_dim=2,
            layer_num=2,
            rank=0,
            layer_types=("linear_attention", "full_attention"),
        )
        return pool

    def test_pool_binds_history_and_state_views_from_one_arena(self):
        pool = self._pool()

        conv, ssm = pool.get_state_buffers(0)
        self.assertIsNone(pool.k_buffer[0])
        self.assertIsNone(pool.v_buffer[0])
        self.assertIsNotNone(pool.k_buffer[1])
        self.assertIsNotNone(pool.v_buffer[1])
        self.assertEqual(
            conv.untyped_storage().data_ptr(),
            pool.arena.field("layer.0.conv").untyped_storage().data_ptr(),
        )
        self.assertEqual(
            ssm.untyped_storage().data_ptr(),
            pool.arena.field("layer.0.ssm").untyped_storage().data_ptr(),
        )

    def test_pool_publishes_runtime_contract_and_component_mapping(self):
        pool = self._pool()

        self.assertEqual(pool.arena.runtime_contract.prefix_granularity, 4)
        self.assertEqual(pool.arena.runtime_contract.num_lcm_blocks, 1)
        self.assertEqual(
            pool.arena.runtime_contract.group_page_counts,
            {"linear_attention_0": 3, "full_attention": 2},
        )
        self.assertEqual(pool.state_group_by_layer[0], "linear_attention_0")
        conv, ssm = pool.get_state_buffers(0)
        self.assertIs(pool.get_component(0, "conv_state"), conv)
        self.assertIs(pool.get_component(0, "recurrent_state"), ssm)

    def test_pool_rejects_unknown_state_layer_and_field(self):
        pool = self._pool()

        with self.assertRaisesRegex(ValueError, "not a state layer"):
            pool.get_state_buffers(1)
        with self.assertRaisesRegex(ValueError, "not planned"):
            pool.arena.field("missing")

    def test_ordinary_pool_keeps_ordinary_per_layer_kv(self):
        pool = self._ordinary_pool()

        self.assertTrue(all(buffer is not None for buffer in pool.k_buffer))
        self.assertTrue(all(buffer is not None for buffer in pool.v_buffer))
        self.assertFalse(hasattr(pool, "lcm_pool"))
        self.assertFalse(hasattr(pool, "state_slabs"))


class DeriveStateGroupsByLayerTest(unittest.TestCase):
    """The plan's field declarations are the single layer -> group record."""

    def setUp(self):
        try:
            from tokenspeed.runtime.layers.attention.kv_cache.base import (
                derive_state_groups_by_layer,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.derive = partial(
            derive_state_groups_by_layer,
            state_field_suffixes=("conv", "ssm"),
        )

    @staticmethod
    def _arena(fields, families):
        from types import SimpleNamespace

        return SimpleNamespace(
            plan=SimpleNamespace(
                fields=tuple(
                    SimpleNamespace(field_id=field_id, group_id=group_id)
                    for field_id, group_id in fields
                )
            ),
            cache_group_specs=tuple(
                SimpleNamespace(group_id=group_id, family=family)
                for group_id, family in families.items()
            ),
        )

    def test_reads_state_groups_back_from_planned_fields(self):
        arena = self._arena(
            fields=(
                ("layer.0.conv", "linear_attention_0"),
                ("layer.0.ssm", "linear_attention_0"),
                ("layer.0.qwen4_exp.ple.conv", "qwen4_exp_ple"),
                ("layer.1.k", "full_attention"),
                ("layer.1.v", "full_attention"),
            ),
            families={
                "linear_attention_0": "state",
                "qwen4_exp_ple": "state",
                "full_attention": "history",
            },
        )
        mapping = self.derive(arena, first_layer=0, num_layers=2, state_layer_ids=(0,))
        self.assertEqual(mapping, {0: "linear_attention_0"})

    def test_draft_view_window_offsets_layer_ids(self):
        arena = self._arena(
            fields=(
                ("layer.0.k", "full_attention"),
                ("layer.1.conv", "linear_attention_0"),
            ),
            families={"linear_attention_0": "state", "full_attention": "history"},
        )
        mapping = self.derive(arena, first_layer=1, num_layers=1, state_layer_ids=(0,))
        self.assertEqual(mapping, {0: "linear_attention_0"})

    def test_rejects_a_layer_spanning_two_state_groups(self):
        arena = self._arena(
            fields=(
                ("layer.0.conv", "linear_attention_0"),
                ("layer.0.ssm", "linear_attention_1"),
            ),
            families={
                "linear_attention_0": "state",
                "linear_attention_1": "state",
            },
        )
        with self.assertRaisesRegex(ValueError, "more than one cache group"):
            self.derive(arena, first_layer=0, num_layers=1, state_layer_ids=(0,))


if __name__ == "__main__":
    unittest.main()
