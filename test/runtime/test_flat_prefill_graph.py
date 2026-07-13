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


class SelectOutCacheLocPadTest(unittest.TestCase):
    """_select_out_cache_loc extends flat group locs to the caller's padded
    row count with dummy slot 0 (the radix tail convention) -- single fix
    point for every flat-capable backend under the prefill graph."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.backends.mha import (
                MHAAttnBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.b = MHAAttnBackend.__new__(MHAAttnBackend)

    def _meta(self, locs):
        return SimpleNamespace(out_cache_locs=locs)

    def _layer(self, gid="full_attention"):
        return SimpleNamespace(group_id=gid)

    def test_padded_caller_extends_locs_with_dummy_slot0(self):
        real = self.torch.arange(64, 69, dtype=self.torch.int32)  # 5 real locs
        meta = self._meta({"full_attention": real})
        caller = self.torch.zeros(16, dtype=self.torch.int32)  # bucket rows
        out = self.b._select_out_cache_loc(self._layer(), meta, caller)
        self.assertEqual(out.shape[0], 16)
        self.assertEqual(out[:5].tolist(), list(range(64, 69)))
        self.assertEqual(out[5:].abs().sum().item(), 0)  # dummy slot 0 tail
        # Memoized: the dict now holds the padded tensor (once per forward).
        self.assertIs(meta.out_cache_locs["full_attention"], out)
        again = self.b._select_out_cache_loc(self._layer(), meta, caller)
        self.assertIs(again, out)

    def test_equal_rows_untouched(self):
        real = self.torch.arange(16, dtype=self.torch.int32)
        meta = self._meta({"full_attention": real})
        caller = self.torch.zeros(16, dtype=self.torch.int32)
        self.assertIs(self.b._select_out_cache_loc(self._layer(), meta, caller), real)

    def test_prefer_caller_wins(self):
        meta = self._meta({"full_attention": self.torch.zeros(5)})
        caller = self.torch.zeros(16, dtype=self.torch.int32)
        self.assertIs(
            self.b._select_out_cache_loc(
                self._layer(), meta, caller, prefer_caller=True
            ),
            caller,
        )


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
            flat_state_group_ids=frozenset({"linear_attention"}),
        )
        pool = SimpleNamespace(
            paged_cache_group_specs=(
                SimpleNamespace(group_id="full_attention"),
                SimpleNamespace(group_id="sliding_attention"),
                SimpleNamespace(group_id="linear_attention"),  # state: skipped
            )
        )
        tables = self._bare(backend, pool)._dummy_flat_tables(100)
        self.assertEqual(set(tables), {"full_attention", "sliding_attention"})
        for t in tables.values():
            self.assertEqual(t.shape, (1, 4))  # ceil(100/32)
            self.assertEqual(int(t.abs().sum()), 0)  # null block 0 only

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
