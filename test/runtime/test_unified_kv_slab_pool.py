"""Hybrid slab sizing and buffer-layout tests."""

from __future__ import annotations

import importlib.util
import itertools
import os
import pathlib
import sys
import unittest

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
        # loud rejection is group_specs_from_layer_types' job.
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
        from cache_pool_test_utils import make_layer_group_ids, make_mha_memory_plan

        kwargs = {
            "size": 32,
            "dtype": self.torch.bfloat16,
            "head_num": 1,
            "head_dim": 8,
            "layer_num": 24,
            "device": "cpu",
            "enable_memory_saver": False,
            "page_size": 16,
            "rank": 0,
            "layer_types": GPT_OSS_LAYER_TYPES,
            "sliding_window_tokens": 128,
        }
        kwargs.update(overrides)
        kwargs["memory_plan"] = make_mha_memory_plan(
            size=kwargs["size"],
            page_size=kwargs["page_size"],
            layer_num=kwargs["layer_num"],
            kv_heads=kwargs["head_num"],
            head_dim=kwargs["head_dim"],
            dtype=kwargs["dtype"],
            layer_types=kwargs.get("layer_types", ()),
            sliding_window_tokens=kwargs.get("sliding_window_tokens"),
        )
        kwargs.setdefault(
            "layer_group_ids",
            make_layer_group_ids(
                layer_num=kwargs["layer_num"],
                layer_types=kwargs.get("layer_types", ()),
                sliding_window_tokens=kwargs.get("sliding_window_tokens"),
            ),
        )
        kwargs.pop("sliding_window_tokens", None)
        return self.MHATokenToKVPool(**kwargs)

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

    def test_guard_raises_on_pd_with_aliased_plan(self):
        with self.assertRaisesRegex(
            RuntimeError,
            r"aliased MHA cache layout is incompatible with PD disaggregation"
            r".*disaggregation_mode='null'",
        ):
            self._pool(pd_disaggregation_enabled=True)

    def test_constructor_without_specs_publishes_no_contract(self):
        # The recipe is the single source of group specs; a pool constructed
        # without them (tests, partial harnesses) publishes no contract.
        kwargs = {
            "size": 32,
            "dtype": self.torch.bfloat16,
            "head_num": 1,
            "head_dim": 8,
            "layer_num": 1,
            "device": "cpu",
            "enable_memory_saver": False,
            "page_size": 16,
            "rank": 0,
            "layer_types": ("full_attention",),
            "layer_group_ids": ("full_attention",),
        }
        from cache_pool_test_utils import make_mha_memory_plan

        kwargs["memory_plan"] = make_mha_memory_plan(
            size=kwargs["size"],
            page_size=kwargs["page_size"],
            layer_num=kwargs["layer_num"],
            kv_heads=kwargs["head_num"],
            head_dim=kwargs["head_dim"],
            dtype=kwargs["dtype"],
            layer_types=kwargs["layer_types"],
        )
        pool = self.MHATokenToKVPool(**kwargs)

        self.assertEqual(pool.paged_cache_group_specs, ())
        self.assertIsNone(pool.runtime_contract)

    def test_constructor_aligns_recipe_specs_with_plan(self):
        # Recipe specs carry default packing; the pool overwrites it (and the
        # page counts) from the memory plan before publishing the contract.
        kwargs = {
            "size": 32,
            "dtype": self.torch.bfloat16,
            "head_num": 1,
            "head_dim": 8,
            "layer_num": 1,
            "device": "cpu",
            "enable_memory_saver": False,
            "page_size": 16,
            "rank": 0,
            "layer_types": ("full_attention",),
            "layer_group_ids": ("full_attention",),
        }
        from cache_pool_test_utils import make_mha_memory_plan

        kwargs["memory_plan"] = make_mha_memory_plan(
            size=kwargs["size"],
            page_size=kwargs["page_size"],
            layer_num=kwargs["layer_num"],
            kv_heads=kwargs["head_num"],
            head_dim=kwargs["head_dim"],
            dtype=kwargs["dtype"],
            layer_types=kwargs["layer_types"],
        )
        kwargs["paged_cache_group_specs"] = _real_spec().build_paged_cache_group_specs(
            layer_types=kwargs["layer_types"],
            group_ids=kwargs["layer_group_ids"],
            sliding_window_tokens=None,
            page_size=kwargs["page_size"],
        )
        pool = self.MHATokenToKVPool(**kwargs)

        self.assertIsNotNone(pool.runtime_contract)
        self.assertEqual(
            [spec.group_id for spec in pool.paged_cache_group_specs],
            ["full_attention"],
        )
        plan_group = kwargs["memory_plan"].group("full_attention")
        self.assertEqual(
            pool.paged_cache_group_page_counts["full_attention"],
            plan_group.page_count,
        )
        self.assertEqual(pool.runtime_contract.token_capacity, kwargs["size"])


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

    def test_constructor_uses_overridable_buffer_allocation(self):
        torch = self.torch
        from cache_pool_test_utils import make_mla_memory_plan

        class PoolWithCustomAllocation(self.MLATokenToKVPool):
            def _create_buffers(self):
                self.allocation_hook_called = True
                self.kv_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, 1, self.kv_cache_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

        pool = PoolWithCustomAllocation(
            size=8,
            model_dtype=torch.bfloat16,
            dtype=torch.bfloat16,
            quant_method=None,
            kv_lora_rank=4,
            qk_rope_head_dim=2,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
            page_size=4,
            rank=0,
            memory_plan=make_mla_memory_plan(
                size=8,
                page_size=4,
                layer_num=1,
                latent_width=6,
                dtype=torch.bfloat16,
            ),
            layer_group_ids=("full_attention",),
            paged_cache_group_specs=_real_spec().build_paged_cache_group_specs(
                layer_types=(),
                group_ids=("full_attention",),
                sliding_window_tokens=None,
                page_size=4,
            ),
        )

        self.assertTrue(pool.allocation_hook_called)
        self.assertEqual(tuple(pool.kv_buffer[0].shape), (12, 1, 6))
        self.assertEqual(
            [spec.group_id for spec in pool.paged_cache_group_specs],
            ["full_attention"],
        )
        self.assertGreater(
            pool.paged_cache_group_page_counts["full_attention"],
            1,
        )


class StatePagedCacheGroupPageCountTest(unittest.TestCase):
    """compute_paged_cache_group_page_counts: the family="state" branch is
    positive and bounded by the full-history formula for the same inputs
    (state rows keep <= 2 live pages per request -- the W=2 write window --
    and snapshots are bounded by the shared page-id space).
    The direct-loaded module still imports ceil_div from the real package
    at call time, so this skips on a bare interpreter.
    """

    def setUp(self):
        try:
            from tokenspeed.runtime.utils.common import ceil_div  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"page-count math needs the real package: {exc}")

    def _counts(self, **overrides):
        specs = _pcs.group_specs_from_layer_types(
            layer_types=("linear_attention", "full_attention"),
            group_ids=("linear_attention", "full_attention"),
            sliding_window_tokens=None,
            page_size=16,
        )
        params = {
            "max_live_requests": 2,
            "max_scheduled_tokens": 64,
            "max_total_tokens": 1024,
            "max_context_len": 4096,
        }
        params.update(overrides)
        return _pcs.compute_paged_cache_group_page_counts(specs, **params)

    def test_state_count_positive_and_bounded_by_full_history(self):
        counts = self._counts()
        self.assertGreater(counts["linear_attention"], 0)
        self.assertLessEqual(counts["linear_attention"], counts["full_attention"])

    def test_state_branch_departs_from_full_history_formula(self):
        # B=0 with a non-page-multiple T distinguishes the state branch
        # (floor(T/P) + 0 live) from the full-history one (ceil(T/P) + B):
        # 1000/16 -> state 62+1=63 < full 63+1=64.
        counts = self._counts(max_live_requests=0, max_total_tokens=1000)
        self.assertLess(counts["linear_attention"], counts["full_attention"])


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
            from tokenspeed.runtime.layers.attention.kv_cache.recipes.qwen35 import (
                qwen_gdn_cache_fields,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.pool_cls = HybridMHATokenToKVPool
        self.mha_pool_cls = MHATokenToKVPool
        fields = qwen_gdn_cache_fields(
            layer_types=("linear_attention", "full_attention"),
            layer_group_ids=("linear_attention_0", "full_attention"),
            logical_block_tokens=4,
            kv_shape=(4, 1, 2),
            kv_element_size=2,
            conv_shape=(2, 2),
            conv_element_size=2,
            ssm_shape=(1, 2, 2),
            ssm_element_size=2,
        )
        self.plan = plan_fields(
            fields,
            logical_block_tokens=4,
            budget_bytes=64,
            alignment=2,
        )

    def _pool(self):
        return self.pool_cls(
            size=8,
            dtype=self.torch.bfloat16,
            head_num=1,
            head_dim=2,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
            page_size=4,
            rank=0,
            layer_types=("linear_attention", "full_attention"),
            state_field_dtypes={
                "layer.0.conv": self.torch.bfloat16,
                "layer.0.ssm": self.torch.bfloat16,
            },
            memory_plan=self.plan,
            layer_group_ids=("linear_attention_0", "full_attention"),
            paged_cache_group_specs=_real_spec().build_paged_cache_group_specs(
                layer_types=("linear_attention", "full_attention"),
                group_ids=("linear_attention_0", "full_attention"),
                sliding_window_tokens=None,
                page_size=4,
            ),
        )

    def _ordinary_pool(self):
        from cache_pool_test_utils import make_mha_memory_plan

        return self.mha_pool_cls(
            size=8,
            dtype=self.torch.bfloat16,
            head_num=1,
            head_dim=2,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
            page_size=4,
            rank=0,
            layer_types=("linear_attention", "full_attention"),
            layer_group_ids=("linear_attention", "full_attention"),
            memory_plan=make_mha_memory_plan(
                size=8,
                page_size=4,
                layer_num=2,
                kv_heads=1,
                head_dim=2,
                dtype=self.torch.bfloat16,
                layer_types=("linear_attention", "full_attention"),
            ),
        )

    def test_pool_binds_history_and_state_views_from_one_arena(self):
        pool = self._pool()

        conv, ssm = pool.get_state_buffers(0)
        self.assertIsNone(pool.k_buffer[0])
        self.assertIsNone(pool.v_buffer[0])
        self.assertIsNotNone(pool.k_buffer[1])
        self.assertIsNotNone(pool.v_buffer[1])
        self.assertEqual(
            conv.untyped_storage().data_ptr(),
            pool.field("layer.0.conv", self.torch.bfloat16)
            .untyped_storage()
            .data_ptr(),
        )
        self.assertEqual(
            ssm.untyped_storage().data_ptr(),
            pool.field("layer.0.ssm", self.torch.bfloat16).untyped_storage().data_ptr(),
        )

    def test_pool_publishes_runtime_contract_and_component_mapping(self):
        pool = self._pool()

        self.assertEqual(pool.runtime_contract.block_size, 4)
        self.assertEqual(pool.runtime_contract.num_lcm_blocks, 1)
        self.assertEqual(
            pool.runtime_contract.group_page_counts,
            {"linear_attention_0": 3, "full_attention": 2},
        )
        self.assertEqual(pool.group_id_for_layer(0), "linear_attention_0")
        conv, ssm = pool.get_state_buffers(0)
        self.assertIs(pool.get_component(0, "conv_state"), conv)
        self.assertIs(pool.get_component(0, "recurrent_state"), ssm)

    def test_pool_rejects_unknown_state_layer_and_field(self):
        pool = self._pool()

        with self.assertRaisesRegex(ValueError, "not a state layer"):
            pool.get_state_buffers(1)
        with self.assertRaisesRegex(ValueError, "not planned"):
            pool.field("missing", self.torch.bfloat16)

    def test_ordinary_pool_keeps_ordinary_per_layer_kv(self):
        pool = self._ordinary_pool()

        self.assertTrue(all(buffer is not None for buffer in pool.k_buffer))
        self.assertTrue(all(buffer is not None for buffer in pool.v_buffer))
        self.assertFalse(hasattr(pool, "lcm_pool"))
        self.assertFalse(hasattr(pool, "state_slabs"))


if __name__ == "__main__":
    unittest.main()
