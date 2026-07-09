"""hybrid_slab_group_size: the single activation predicate for the unified
KV slab pool (M12), and its two consumers (registry sizing divisor and
MHATokenToKVPool buffer layout).

The predicate returns the common layers-per-group count exactly when the
slab layout may activate (flat ext, no spec decode, >= 2 equal-size known
groups) and None otherwise (legacy per-layer layout). The installed ext's
real build flavor must not decide these tests, so the
scheduler_ext_flat_kvcache probe is patched per case.
"""

from __future__ import annotations

import contextlib
import importlib.util
import itertools
import os
import pathlib
import sys
import unittest
from unittest import mock

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


_pcs = _load("paged_cache_spec_slab_under_test", "paged_cache_spec.py")
hybrid_slab_group_size = _pcs.hybrid_slab_group_size

GPT_OSS_LAYER_TYPES = ("sliding_attention", "full_attention") * 12


class HybridSlabGroupSizeTest(unittest.TestCase):
    """Each case pins exactly ONE reason the predicate returns None (or the
    single shape where it activates)."""

    @contextlib.contextmanager
    def _flat_ext(self, value: bool):
        # The predicate resolves the probe from its own module globals at
        # call time, so the patch must target the path-loaded module.
        with mock.patch.object(
            _pcs, "scheduler_ext_flat_kvcache", return_value=value
        ):
            yield

    def test_gpt_oss_shape_returns_group_size(self):
        # gpt-oss: 12 sliding + 12 full, alternating -> 12 layers per group.
        with self._flat_ext(True):
            self.assertEqual(
                hybrid_slab_group_size(
                    GPT_OSS_LAYER_TYPES, speculative_enabled=False
                ),
                12,
            )

    def test_none_when_radix_ext(self):
        with self._flat_ext(False):
            self.assertIsNone(
                hybrid_slab_group_size(
                    GPT_OSS_LAYER_TYPES, speculative_enabled=False
                )
            )

    def test_none_when_speculative(self):
        with self._flat_ext(True):
            self.assertIsNone(
                hybrid_slab_group_size(
                    GPT_OSS_LAYER_TYPES, speculative_enabled=True
                )
            )

    def test_none_when_single_group(self):
        with self._flat_ext(True):
            self.assertIsNone(
                hybrid_slab_group_size(
                    ("full_attention",) * 24, speculative_enabled=False
                )
            )

    def test_none_when_unequal_groups(self):
        lt = ("sliding_attention",) * 8 + ("full_attention",) * 16
        with self._flat_ext(True):
            self.assertIsNone(
                hybrid_slab_group_size(lt, speculative_enabled=False)
            )

    def test_none_when_unknown_label(self):
        # Unknown input degrades to None (safe legacy layout), never raises;
        # loud rejection is group_specs_from_layer_types' job.
        lt = GPT_OSS_LAYER_TYPES + ("banana_attention",)
        with self._flat_ext(True):
            self.assertIsNone(
                hybrid_slab_group_size(lt, speculative_enabled=False)
            )

    def test_none_when_empty(self):
        # Plain models pass empty or None layer_types.
        with self._flat_ext(True):
            self.assertIsNone(
                hybrid_slab_group_size((), speculative_enabled=False)
            )
            self.assertIsNone(
                hybrid_slab_group_size(None, speculative_enabled=False)
            )

    def test_none_when_multi_window_sequence(self):
        # 多种 sliding window:保守退 legacy(M14 spec §2.3)。
        with self._flat_ext(True):
            it = itertools.cycle((4, 512))
            windows = [
                next(it) if t == "sliding_attention" else None
                for t in GPT_OSS_LAYER_TYPES
            ]
            self.assertIsNone(
                hybrid_slab_group_size(
                    GPT_OSS_LAYER_TYPES,
                    speculative_enabled=False,
                    sliding_window_tokens=windows,
                )
            )

    def test_uniform_window_sequence_stays_active(self):
        with self._flat_ext(True):
            windows = [
                None if t == "full_attention" else 128
                for t in GPT_OSS_LAYER_TYPES
            ]
            self.assertEqual(
                hybrid_slab_group_size(
                    GPT_OSS_LAYER_TYPES,
                    speculative_enabled=False,
                    sliding_window_tokens=windows,
                ),
                12,
            )

    def test_scalar_window_stays_active(self):
        with self._flat_ext(True):
            self.assertEqual(
                hybrid_slab_group_size(
                    GPT_OSS_LAYER_TYPES,
                    speculative_enabled=False,
                    sliding_window_tokens=128,
                ),
                12,
            )

    def test_none_when_window_sequence_length_mismatch(self):
        # 谓词永不 raise;畸形输入按 None 降级。
        with self._flat_ext(True):
            self.assertIsNone(
                hybrid_slab_group_size(
                    GPT_OSS_LAYER_TYPES,
                    speculative_enabled=False,
                    sliding_window_tokens=[128],
                )
            )

    def test_garbage_elements_ignored_not_raised(self):
        # 谓词只看有效 window 的 distinct 数,非 int 元素被忽略。
        with self._flat_ext(True):
            self.assertEqual(
                hybrid_slab_group_size(
                    GPT_OSS_LAYER_TYPES,
                    speculative_enabled=False,
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
            raise unittest.SkipTest(
                f"real attention registry unimportable here: {exc}"
            )
        cls._registry = registry
        cls._pkg_pcs = pkg_pcs

    @contextlib.contextmanager
    def _pkg_flat_ext(self, value: bool):
        with mock.patch.object(
            self._pkg_pcs, "scheduler_ext_flat_kvcache", return_value=value
        ):
            yield

    def test_gpt_oss_flat_ext_charges_group_size(self):
        # 24 layers, 12+12 alternating -> charge 12 (per-token bytes halve).
        with self._pkg_flat_ext(True):
            self.assertEqual(
                self._registry._kv_profile_layer_divisor(
                    24, GPT_OSS_LAYER_TYPES, speculative_enabled=False
                ),
                12,
            )

    def test_all_layers_when_radix_ext(self):
        with self._pkg_flat_ext(False):
            self.assertEqual(
                self._registry._kv_profile_layer_divisor(
                    24, GPT_OSS_LAYER_TYPES, speculative_enabled=False
                ),
                24,
            )

    def test_all_layers_when_speculative(self):
        with self._pkg_flat_ext(True):
            self.assertEqual(
                self._registry._kv_profile_layer_divisor(
                    24, GPT_OSS_LAYER_TYPES, speculative_enabled=True
                ),
                24,
            )

    def test_all_layers_when_no_layer_types(self):
        # () from MHAConfig's default, None from MLA configs via getattr.
        with self._pkg_flat_ext(True):
            self.assertEqual(
                self._registry._kv_profile_layer_divisor(
                    24, (), speculative_enabled=False
                ),
                24,
            )
            self.assertEqual(
                self._registry._kv_profile_layer_divisor(
                    24, None, speculative_enabled=False
                ),
                24,
            )

    def test_all_layers_when_multi_window_sequence(self):
        # M14: the registry must forward sliding_window_tokens so sizing
        # matches the pool's layout decision (divergence is the hazard).
        with self._pkg_flat_ext(True):
            it = itertools.cycle((4, 512))
            windows = [
                next(it) if t == "sliding_attention" else None
                for t in GPT_OSS_LAYER_TYPES
            ]
            self.assertEqual(
                self._registry._kv_profile_layer_divisor(
                    24,
                    GPT_OSS_LAYER_TYPES,
                    speculative_enabled=False,
                    sliding_window_tokens=windows,
                ),
                24,
            )

    def test_group_size_when_uniform_window_sequence(self):
        with self._pkg_flat_ext(True):
            windows = [
                128 if t == "sliding_attention" else None
                for t in GPT_OSS_LAYER_TYPES
            ]
            self.assertEqual(
                self._registry._kv_profile_layer_divisor(
                    24,
                    GPT_OSS_LAYER_TYPES,
                    speculative_enabled=False,
                    sliding_window_tokens=windows,
                ),
                12,
            )


_PKG_FLAT_PROBE = (
    "tokenspeed.runtime.configs.paged_cache_spec.scheduler_ext_flat_kvcache"
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

    def _pool(self, *, flat_ext: bool = True, **overrides):
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
        with mock.patch(_PKG_FLAT_PROBE, return_value=flat_ext):
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
            radix_ext=dict(flat_ext=False),
            spec_decode=dict(speculative_enabled=True),
            single_group=dict(
                layer_types=("full_attention",) * 24,
                sliding_window_tokens=None,
            ),
            unequal_groups=dict(
                layer_types=("sliding_attention",) * 8
                + ("full_attention",) * 16
            ),
        )
        for name, overrides in cases.items():
            with self.subTest(name):
                pool = self._pool(**overrides)
                self.assertEqual(len({id(t) for t in pool.k_buffer}), 24)
                self.assertEqual(len({id(t) for t in pool.v_buffer}), 24)
                self.assertTrue(pool.supports_hierarchical_kv_cache)

    def test_kvstore_with_slab_allowed(self):
        # Spec §6 revision: the flat L2 tier mirrors whole slabs byte-blind,
        # so per-slab copies are group-safe and the guard no longer fires.
        pool = self._pool(kvstore_enabled=True)
        self.assertEqual(len({id(t) for t in pool.k_buffer}), 12)

    def test_guard_raises_on_pd_with_slab(self):
        with self.assertRaisesRegex(
            RuntimeError,
            r"hybrid slab KV layout is incompatible with PD disaggregation"
            r".*radix-built",
        ):
            self._pool(pd_disaggregation_enabled=True)

    def test_no_guard_when_fallback(self):
        # The flags only conflict with the slab layout, not the legacy one.
        pool = self._pool(
            flat_ext=False,
            kvstore_enabled=True,
            pd_disaggregation_enabled=True,
        )
        self.assertEqual(len({id(t) for t in pool.k_buffer}), 24)


if __name__ == "__main__":
    unittest.main()
