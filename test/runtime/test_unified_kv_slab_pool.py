"""hybrid_slab_group_size: the single activation predicate for the unified
KV slab pool (M12).

Rule under test (configs/paged_cache_spec.py): the predicate returns the
common layers-per-group count exactly when the hybrid slab layout may
activate — flat scheduler ext, no spec decode, >= 2 known groups of EQUAL
size — and None for every other input, which keeps the legacy per-layer
buffer layout. Both the sizing divisor (registry profile) and the buffer
layout (_create_buffers) consume this one function, so its gating IS the
activation contract.

The installed ext's real build flavor must not decide these tests, so the
scheduler_ext_flat_kvcache probe is patched per case. The predicate looks
the probe up as a module global at call time, so the patch targets the name
inside the path-loaded module's own namespace (patching the import source
package would miss it).
"""

from __future__ import annotations

import contextlib
import importlib.util
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
        # The predicate resolves scheduler_ext_flat_kvcache as a module
        # global at call time; patch it in the loaded module's namespace so
        # the interception actually lands.
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
        # Radix/older ext: no flat BlockPool single-ownership guarantee, so
        # paired layers' live rows could overlap -> keep legacy layout.
        with self._flat_ext(False):
            self.assertIsNone(
                hybrid_slab_group_size(
                    GPT_OSS_LAYER_TYPES, speculative_enabled=False
                )
            )

    def test_none_when_speculative(self):
        # Spec decode gates paged-cache group publication off entirely; the
        # slab layout depends on the flat group machinery being live.
        with self._flat_ext(True):
            self.assertIsNone(
                hybrid_slab_group_size(
                    GPT_OSS_LAYER_TYPES, speculative_enabled=True
                )
            )

    def test_none_when_single_group(self):
        # All-full (llama/qwen shape): one group, nothing to pair, no win.
        with self._flat_ext(True):
            self.assertIsNone(
                hybrid_slab_group_size(
                    ("full_attention",) * 24, speculative_enabled=False
                )
            )

    def test_none_when_unequal_groups(self):
        # 8 sliding + 16 full: unequal fan-out, no clean layer pairing.
        lt = ("sliding_attention",) * 8 + ("full_attention",) * 16
        with self._flat_ext(True):
            self.assertIsNone(
                hybrid_slab_group_size(lt, speculative_enabled=False)
            )

    def test_none_when_unknown_label(self):
        # Unknown label -> None, NOT a raise. Asymmetry with
        # group_specs_from_layer_types (which raises ValueError) is
        # deliberate: the predicate gates an OPTIMIZATION, so unknown input
        # degrades to the safe legacy layout, while spec publication must
        # fail loudly.
        lt = GPT_OSS_LAYER_TYPES + ("banana_attention",)
        with self._flat_ext(True):
            self.assertIsNone(
                hybrid_slab_group_size(lt, speculative_enabled=False)
            )

    def test_none_when_empty(self):
        # Plain models pass empty/None layer_types; both mean "no hybrid
        # grouping" and must keep the legacy layout.
        with self._flat_ext(True):
            self.assertIsNone(
                hybrid_slab_group_size((), speculative_enabled=False)
            )
            self.assertIsNone(
                hybrid_slab_group_size(None, speculative_enabled=False)
            )


class KvProfileLayerDivisorTest(unittest.TestCase):
    """Registry-side sizing consumer (M12 Task 2): _kv_profile_layer_divisor
    must charge layers-per-group exactly when the predicate activates, and
    all layers otherwise. The helper is thin glue; the valuable pin is that
    the REAL registry module routes its KV profile through the predicate.

    Imports the real package registry (torch/triton/tokenspeed_kernel
    stack), so these cases skip on a bare interpreter and exercise in the
    container CI. Patch target: the scheduler_ext_flat_kvcache probe in the
    PACKAGE paged_cache_spec module -- registry's imported
    hybrid_slab_group_size is bound to that module object and resolves the
    probe from its own globals at call time, so patching there intercepts
    the real call chain (patching the path-loaded _pcs copy above would
    miss it: that is a distinct module object the registry never sees).
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
        # 24 layers, 12+12 alternating -> charge 12: per-token bytes halve,
        # so the profiled max_num_tokens/pages double under the slab layout.
        with self._pkg_flat_ext(True):
            self.assertEqual(
                self._registry._kv_profile_layer_divisor(
                    24, GPT_OSS_LAYER_TYPES, False
                ),
                12,
            )

    def test_all_layers_when_radix_ext(self):
        with self._pkg_flat_ext(False):
            self.assertEqual(
                self._registry._kv_profile_layer_divisor(
                    24, GPT_OSS_LAYER_TYPES, False
                ),
                24,
            )

    def test_all_layers_when_speculative(self):
        with self._pkg_flat_ext(True):
            self.assertEqual(
                self._registry._kv_profile_layer_divisor(
                    24, GPT_OSS_LAYER_TYPES, True
                ),
                24,
            )

    def test_all_layers_when_no_layer_types(self):
        # Plain models: MHAConfig defaults to (); MLA configs have no
        # layer_types attribute at all, which the registry call site turns
        # into None via getattr. Both keep the all-layers divisor.
        with self._pkg_flat_ext(True):
            self.assertEqual(
                self._registry._kv_profile_layer_divisor(24, (), False), 24
            )
            self.assertEqual(
                self._registry._kv_profile_layer_divisor(24, None, False), 24
            )


_PKG_FLAT_PROBE = (
    "tokenspeed.runtime.configs.paged_cache_spec.scheduler_ext_flat_kvcache"
)


class MHAPoolSlabLayoutTest(unittest.TestCase):
    """Layout half of M12 (kv_cache/mha.py _create_buffers): when the
    predicate activates, the pool allocates group_size K/V slab pairs and
    binds paired layers (sliding-i <-> full-i, first-appearance group
    order, in-group index alignment) to the SAME tensor objects; every
    other configuration keeps the legacy per-layer layout, and the slab
    guards (kvstore / PD-disagg) never fire there.

    Constructs a real (tiny, CPU) MHATokenToKVPool; skips without deps.
    Patch target is the PACKAGE paged_cache_spec probe: the pool's
    hybrid_slab_group_size is bound to that module object and resolves
    scheduler_ext_flat_kvcache from its own globals at call time, so
    patching there intercepts the real call chain (the path-loaded _pcs
    copy above is a distinct module object the pool never sees).
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
        # gpt-oss shape: 24 layer entries alias 12 slabs, so accessors
        # (get_key_buffer etc.) stay layer-indexed with zero changes.
        self.assertEqual(len(pool.k_buffer), 24)
        self.assertEqual(len({id(t) for t in pool.k_buffer}), 12)
        self.assertEqual(len({id(t) for t in pool.v_buffer}), 12)
        # In-group index alignment: groups form in first-appearance order
        # of layer_types (sliding first here), and the i-th sliding layer
        # (layer 2i) pairs the i-th full layer (layer 2i+1) on the SAME
        # tensor object -- shared storage, not a copy or a view.
        for i in range(12):
            self.assertIs(pool.k_buffer[2 * i], pool.k_buffer[2 * i + 1])
            self.assertIs(pool.v_buffer[2 * i], pool.v_buffer[2 * i + 1])
            self.assertEqual(
                pool.k_buffer[2 * i].data_ptr(),
                pool.k_buffer[2 * i + 1].data_ptr(),
            )
        # Pairing completeness: every slab is referenced by exactly one
        # layer of EACH group (sliding layers sit at even ids here).
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
        # pool opts out of the hierarchical cache surface (event_loop
        # builds a MemoryExecutor for retraction offload even with the
        # kvstore flag off; this is the switch that gates it).
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

    def test_guard_raises_on_kvstore_with_slab(self):
        with self.assertRaisesRegex(
            RuntimeError,
            r"hybrid slab KV layout is incompatible with the kvstore"
            r".*radix-built",
        ):
            self._pool(kvstore_enabled=True)

    def test_guard_raises_on_pd_with_slab(self):
        with self.assertRaisesRegex(
            RuntimeError,
            r"hybrid slab KV layout is incompatible with PD disaggregation"
            r".*radix-built",
        ):
            self._pool(pd_disaggregation_enabled=True)

    def test_no_guard_when_fallback(self):
        # The flags only conflict with the slab layout; the legacy layout
        # keeps serving them, so a radix build must construct fine.
        pool = self._pool(
            flat_ext=False,
            kvstore_enabled=True,
            pd_disaggregation_enabled=True,
        )
        self.assertEqual(len({id(t) for t in pool.k_buffer}), 24)


if __name__ == "__main__":
    unittest.main()
