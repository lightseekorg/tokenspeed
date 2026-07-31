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


_pcs = _load("paged_cache_spec_slab_under_test", "paged_cache_spec.py")
hybrid_slab_group_size = _pcs.hybrid_slab_group_size

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


class KvProfileLayerDivisorTest(unittest.TestCase):
    """Registry sizing consumer: _kv_profile_layer_divisor charges
    layers-per-group exactly when the predicate activates, all layers
    otherwise. Imports the real registry, so skips on a bare interpreter.
    Patch target is the PACKAGE paged_cache_spec probe -- the path-loaded
    _pcs copy above is a distinct module object the registry never sees.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import tokenspeed.runtime.configs.paged_cache_spec as pkg_pcs
            from tokenspeed.runtime.layers.attention import registry
        except ImportError as exc:
            raise unittest.SkipTest(f"real attention registry unimportable here: {exc}")
        cls._registry = registry
        cls._pkg_pcs = pkg_pcs

    def test_gpt_oss_charges_group_size(self):
        # 24 layers, 12+12 alternating -> charge 12 (per-token bytes halve).
        self.assertEqual(
            self._registry._kv_profile_layer_divisor(24, GPT_OSS_LAYER_TYPES),
            12,
        )

    def test_all_layers_when_no_layer_types(self):
        # () from MHAConfig's default, None from MLA configs via getattr.
        self.assertEqual(
            self._registry._kv_profile_layer_divisor(24, ()),
            24,
        )
        self.assertEqual(
            self._registry._kv_profile_layer_divisor(24, None),
            24,
        )

    def test_all_layers_when_multi_window_sequence(self):
        # M14: the registry must forward sliding_window_tokens so sizing
        # matches the pool's layout decision (divergence is the hazard).
        it = itertools.cycle((4, 512))
        windows = [
            next(it) if t == "sliding_attention" else None for t in GPT_OSS_LAYER_TYPES
        ]
        self.assertEqual(
            self._registry._kv_profile_layer_divisor(
                24,
                GPT_OSS_LAYER_TYPES,
                sliding_window_tokens=windows,
            ),
            24,
        )

    def test_group_size_when_uniform_window_sequence(self):
        windows = [
            128 if t == "sliding_attention" else None for t in GPT_OSS_LAYER_TYPES
        ]
        self.assertEqual(
            self._registry._kv_profile_layer_divisor(
                24,
                GPT_OSS_LAYER_TYPES,
                sliding_window_tokens=windows,
            ),
            12,
        )


class MHAPoolSlabLayoutTest(unittest.TestCase):
    """Layout consumer (kv_cache/mha.py _create_buffers): when the predicate
    activates, paired layers bind to the SAME slab tensors; otherwise the
    legacy per-layer layout holds and the PD guard never fires.
    Constructs a real (tiny, CPU) MHATokenToKVPool; skips without deps.
    Patch target is the PACKAGE paged_cache_spec probe (see above).
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
        kwargs = dict(
            size=32,
            dtype=self.torch.bfloat16,
            head_num=1,
            head_dim=8,
            layer_num=24,
            device="cpu",
            enable_memory_saver=False,
            max_batch_size=2,
            max_context_len=64,
            page_size=16,
            rank=0,
            layer_types=GPT_OSS_LAYER_TYPES,
            sliding_window_tokens=128,
            enable_alt_stream=False,
        )
        kwargs.update(overrides)
        return self.MHATokenToKVPool(**kwargs)

    def test_slab_pairing_binds_same_tensor(self):
        pool = self._pool()
        # 24 layer entries alias 12 slabs: accessors stay layer-indexed.
        self.assertEqual(len(pool.k_buffer), 24)
        self.assertEqual(len({id(t) for t in pool.k_buffer}), 12)
        self.assertEqual(len({id(t) for t in pool.v_buffer}), 12)
        # The i-th sliding layer (2i) pairs the i-th full layer (2i+1) on
        # the SAME tensor object -- shared storage, not a copy or a view.
        for i in range(12):
            self.assertIs(pool.k_buffer[2 * i], pool.k_buffer[2 * i + 1])
            self.assertIs(pool.v_buffer[2 * i], pool.v_buffer[2 * i + 1])
            self.assertEqual(
                pool.k_buffer[2 * i].data_ptr(),
                pool.k_buffer[2 * i + 1].data_ptr(),
            )
        # Every slab is referenced by exactly one layer of EACH group.
        for buffers in (pool.k_buffer, pool.v_buffer):
            slab_to_layers: dict[int, list[int]] = {}
            for layer_id, tensor in enumerate(buffers):
                slab_to_layers.setdefault(id(tensor), []).append(layer_id)
            self.assertEqual(len(slab_to_layers), 12)
            for layer_ids in slab_to_layers.values():
                sliding = [lid for lid in layer_ids if lid % 2 == 0]
                full = [lid for lid in layer_ids if lid % 2 == 1]
                self.assertEqual(len(sliding), 1)
                self.assertEqual(len(full), 1)
        # Distinct slabs own distinct storage.
        self.assertEqual(len({t.data_ptr() for t in pool.k_buffer}), 12)
        self.assertEqual(len({t.data_ptr() for t in pool.v_buffer}), 12)
        # Per-layer host (L2) copies would alias shared slabs, so the slab
        # pool opts out of the hierarchical cache surface.
        self.assertFalse(pool.supports_hierarchical_kv_cache)

    def test_fallback_matrix_keeps_24_buffers(self):
        cases = dict(
            single_group=dict(
                layer_types=("full_attention",) * 24,
                sliding_window_tokens=None,
            )
        )
        for name, overrides in cases.items():
            with self.subTest(name):
                pool = self._pool(**overrides)
                self.assertEqual(len({id(t) for t in pool.k_buffer}), 24)
                self.assertEqual(len({id(t) for t in pool.v_buffer}), 24)
                self.assertTrue(pool.supports_hierarchical_kv_cache)

    def test_unequal_groups_pair_smaller_into_larger(self):
        # 8 sliding + 16 full -> 16 slabs: the 8 sliding layers alias the
        # first 8 full layers' slabs; full layers 8..15 keep solo slabs
        # (the Inkling shape, 55 sliding + 11 full, scaled down).
        pool = self._pool(
            layer_types=("sliding_attention",) * 8 + ("full_attention",) * 16
        )
        self.assertEqual(len(pool.k_buffer), 24)
        self.assertEqual(len({id(t) for t in pool.k_buffer}), 16)
        self.assertEqual(len({id(t) for t in pool.v_buffer}), 16)
        for i in range(8):
            # sliding layer i (id i) pairs full layer i (id 8 + i).
            self.assertIs(pool.k_buffer[i], pool.k_buffer[8 + i])
            self.assertIs(pool.v_buffer[i], pool.v_buffer[8 + i])
        solo = [id(t) for t in pool.k_buffer[16:]]
        self.assertEqual(len(set(solo)), 8)
        self.assertFalse(pool.supports_hierarchical_kv_cache)

    def test_sliding_subgroups_six_way_binding(self):
        # (s0..s4, full) x 2 -> 6 groups of 2 layers -> 2 slabs, each bound
        # by one layer of every group (the Inkling 5+1 shape, scaled down).
        pool = self._pool(
            layer_num=12,
            layer_types=SUBGROUP_LAYER_BLOCK * 2,
            sliding_window_tokens=512,
        )
        self.assertEqual(len(pool.k_buffer), 12)
        self.assertEqual(len({id(t) for t in pool.k_buffer}), 2)
        self.assertEqual(len({id(t) for t in pool.v_buffer}), 2)
        # Occurrence pairing: layers 0..5 (first occurrence of each label)
        # bind slab 0; layers 6..11 bind slab 1.
        for i in range(6):
            self.assertIs(pool.k_buffer[i], pool.k_buffer[0])
            self.assertIs(pool.k_buffer[6 + i], pool.k_buffer[6])
            self.assertIs(pool.v_buffer[i], pool.v_buffer[0])
            self.assertIs(pool.v_buffer[6 + i], pool.v_buffer[6])
        self.assertIsNot(pool.k_buffer[0], pool.k_buffer[6])
        self.assertFalse(pool.supports_hierarchical_kv_cache)

    def test_guard_raises_on_pd_with_slab(self):
        with self.assertRaisesRegex(
            RuntimeError,
            r"hybrid slab KV layout is incompatible with PD disaggregation"
            r".*legacy scheduler",
        ):
            self._pool(pd_disaggregation_enabled=True)

    def test_constructor_uses_overridable_group_publication(self):
        class PoolWithCustomPublication(self.MHATokenToKVPool):
            def _publish_paged_cache_groups(self, **kwargs):
                self.publication_args = kwargs
                return None

        kwargs = dict(
            size=32,
            dtype=self.torch.bfloat16,
            head_num=1,
            head_dim=8,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
            max_batch_size=2,
            max_context_len=64,
            page_size=16,
            rank=0,
            layer_types=("full_attention",),
            enable_alt_stream=False,
        )
        pool = PoolWithCustomPublication(**kwargs)

        self.assertEqual(pool.paged_cache_group_specs, ())
        self.assertEqual(pool.publication_args["layer_types"], ("full_attention",))
        self.assertEqual(pool.publication_args["max_total_tokens"], 32)


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
            max_batch_size=2,
            max_context_len=32,
            page_size=4,
            rank=0,
            enable_alt_stream=False,
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
            sliding_window_tokens=None,
            page_size=16,
        )
        params = dict(
            max_live_requests=2,
            max_scheduled_tokens=64,
            max_total_tokens=1024,
            max_context_len=4096,
        )
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


class LcmPoolFieldBindingTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.configs.lcm_layouts import (
                qwen_gdn_lcm_fields,
            )
            from tokenspeed.runtime.configs.lcm_memory_plan import plan_lcm_fields
            from tokenspeed.runtime.layers.attention.kv_cache.lcm_mha import (
                LcmMHATokenToKVPool,
            )
            from tokenspeed.runtime.layers.attention.kv_cache.mha import (
                MHATokenToKVPool,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.lcm_pool_cls = LcmMHATokenToKVPool
        self.pool_cls = MHATokenToKVPool
        fields = qwen_gdn_lcm_fields(
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
        self.plan = plan_lcm_fields(
            fields,
            logical_block_tokens=4,
            budget_bytes=64,
            alignment=2,
        )

    def _pool(self):
        return self.lcm_pool_cls(
            size=8,
            dtype=self.torch.bfloat16,
            head_num=1,
            head_dim=2,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
            max_batch_size=2,
            max_context_len=32,
            page_size=4,
            rank=0,
            layer_types=("linear_attention", "full_attention"),
            state_field_dtypes={
                "layer.0.conv": self.torch.bfloat16,
                "layer.0.ssm": self.torch.bfloat16,
            },
            memory_plan=self.plan,
            layer_group_ids=("linear_attention_0", "full_attention"),
            enable_alt_stream=False,
        )

    def _non_lcm_pool(self):
        return self.pool_cls(
            size=8,
            dtype=self.torch.bfloat16,
            head_num=1,
            head_dim=2,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
            max_batch_size=2,
            max_context_len=32,
            page_size=4,
            rank=0,
            layer_types=("linear_attention", "full_attention"),
            enable_alt_stream=False,
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
            pool.get_lcm_field("layer.0.conv", self.torch.bfloat16)
            .untyped_storage()
            .data_ptr(),
        )
        self.assertEqual(
            ssm.untyped_storage().data_ptr(),
            pool.get_lcm_field("layer.0.ssm", self.torch.bfloat16)
            .untyped_storage()
            .data_ptr(),
        )

    def test_lcm_pool_publishes_runtime_contract_and_component_mapping(self):
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
            pool.get_lcm_field("missing", self.torch.bfloat16)

    def test_non_lcm_pool_keeps_ordinary_per_layer_kv(self):
        pool = self._non_lcm_pool()

        self.assertTrue(all(buffer is not None for buffer in pool.k_buffer))
        self.assertTrue(all(buffer is not None for buffer in pool.v_buffer))
        self.assertFalse(hasattr(pool, "lcm_pool"))
        self.assertFalse(hasattr(pool, "state_slabs"))
        self.assertTrue(pool.supports_hierarchical_kv_cache)


if __name__ == "__main__":
    unittest.main()
