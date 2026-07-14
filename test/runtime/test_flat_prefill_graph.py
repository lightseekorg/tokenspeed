"""Flat KV-cache x prefill CUDA graph seams.

Prefill-graph replay pads q/k/v rows to the bucket while flat per-group
write locs cover only the real (leading) tokens; the mha KV write must trim
the padded tail or the store kernel walks past the loc array (IAE on the
first padded replay -- reproduced on gpt-oss + flat + default prefill graph).
Capture must also exercise the flat metadata branch via all-zero dummy
tables so capture and replay take the same code path.
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")


class TrimKvToLocsTest(unittest.TestCase):
    """_trim_kv_to_locs slices padded k/v tails to the write-loc count --
    the shared fix point every flat-capable backend's KV write calls.
    Trimming (not loc-padding) keeps the null page 0 all-zero: trtllm does
    not scrub padded tail rows before saving KV."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.backends.flat_groups import (
                FlatCacheGroupsMixin,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.trim = FlatCacheGroupsMixin._trim_kv_to_locs

    def test_padded_tail_trimmed(self):
        k = self.torch.zeros(16, 2, 8)
        v = self.torch.zeros(16, 2, 8)
        locs = self.torch.zeros(5, dtype=self.torch.int32)
        k2, v2 = self.trim(locs, k, v)
        self.assertEqual((k2.shape[0], v2.shape[0]), (5, 5))

    def test_equal_rows_identity(self):
        k = self.torch.zeros(16, 2, 8)
        v = self.torch.zeros(16, 2, 8)
        locs = self.torch.zeros(16, dtype=self.torch.int32)
        k2, v2 = self.trim(locs, k, v)
        self.assertIs(k2, k)
        self.assertIs(v2, v)

    def test_none_kv_passthrough(self):
        locs = self.torch.zeros(4, dtype=self.torch.int32)
        self.assertEqual(self.trim(locs, None, None), (None, None))


class DummyFlatTablesTest(unittest.TestCase):
    """Capture-time dummy tables: one all-zero row per non-state flat group."""

    def setUp(self):
        try:
            import torch  # noqa: F401

            from tokenspeed.runtime.execution.prefill_graph import PrefillGraph
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + runtime deps: {exc}")
        self.PrefillGraph = PrefillGraph

    def _bare(self, backend, pool):
        pg = self.PrefillGraph.__new__(self.PrefillGraph)
        pg.attn_backend = backend
        pg.token_to_kv_pool = pool
        pg.config = SimpleNamespace(device="cpu")
        return pg

    def test_flat_backend_gets_zero_tables_per_group(self):
        backend = SimpleNamespace(
            uses_flat_cache_groups=True,
            page_size=32,
            max_num_pages=0,  # fall back to bucket-derived width
            flat_state_group_ids=frozenset({"linear_attention"}),
        )
        pool = SimpleNamespace(
            paged_cache_group_specs=(
                SimpleNamespace(group_id="full_attention"),
                SimpleNamespace(group_id="sliding_attention"),
                SimpleNamespace(group_id="linear_attention"),  # state: included
            )
        )
        tables = self._bare(backend, pool)._dummy_flat_tables(100)
        self.assertEqual(
            set(tables),
            {"full_attention", "sliding_attention", "linear_attention"},
        )
        for t in tables.values():
            self.assertEqual(t.shape, (1, 4))  # ceil(100/32)
            self.assertEqual(int(t.abs().sum()), 0)  # null block 0 only

    def test_full_width_for_stride_deriving_backends(self):
        # trtllm-style: row stride comes from max_kv_len, so dummy tables
        # must span the full table width, not just the bucket.
        backend = SimpleNamespace(
            uses_flat_cache_groups=True,
            page_size=32,
            max_num_pages=2500,
            flat_state_group_ids=frozenset(),
        )
        pool = SimpleNamespace(
            paged_cache_group_specs=(SimpleNamespace(group_id="full_attention"),)
        )
        tables = self._bare(backend, pool)._dummy_flat_tables(100)
        self.assertEqual(tables["full_attention"].shape, (1, 2500))

    def test_composite_wrapper_resolves_flat_child(self):
        # Hybrid wrappers set the flag but hold the flat KV consumer as
        # full_attn_backend; the helper must not AttributeError (which would
        # silently disable the prefill graph via the capture fallback).
        child = SimpleNamespace(
            page_size=32, max_num_pages=0, flat_state_group_ids=frozenset()
        )
        wrapper = SimpleNamespace(uses_flat_cache_groups=True, full_attn_backend=child)
        pool = SimpleNamespace(
            paged_cache_group_specs=(
                SimpleNamespace(group_id="full_attention"),
                SimpleNamespace(group_id="linear_attention"),
            )
        )
        tables = self._bare(wrapper, pool)._dummy_flat_tables(64)
        self.assertEqual(set(tables), {"full_attention", "linear_attention"})
        self.assertEqual(tables["full_attention"].shape, (1, 2))

    def test_non_flat_backend_empty(self):
        backend = SimpleNamespace(uses_flat_cache_groups=False)
        pool = SimpleNamespace(paged_cache_group_specs=())
        self.assertEqual(self._bare(backend, pool)._dummy_flat_tables(64), {})


class TrtllmPrefillGraphSeamsTest(unittest.TestCase):
    """trtllm under the prefill graph: the extend prewrite must not bake
    capture-time write locs into the graph, and the break's KV write must
    trim padded tails like mha."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.backends import trtllm
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.mod = trtllm

    def _bare_backend(self):
        b = self.mod.TRTLLMMHAAttnBackend.__new__(self.mod.TRTLLMMHAAttnBackend)
        b.kv_cache_dtype = self.torch.bfloat16
        return b

    def test_prewrite_disabled_during_breakable_capture(self):
        from unittest import mock

        b = self._bare_backend()
        self.assertTrue(b.support_kv_cache_prewrite(None))
        with mock.patch.object(
            self.mod, "is_breakable_capture_active", return_value=True
        ):
            self.assertFalse(b.support_kv_cache_prewrite(None))


if __name__ == "__main__":
    unittest.main()
