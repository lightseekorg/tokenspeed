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


class SaveKvPadTrimTest(unittest.TestCase):
    """_save_kv_cache drops padded k/v tail rows beyond the loc count."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.backends.mha import (
                MHAAttnBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.Backend = MHAAttnBackend

    def _bare_backend(self):
        b = self.Backend.__new__(self.Backend)
        b.kv_cache_dtype = self.torch.bfloat16
        return b

    class _RecordingPool:
        def __init__(self):
            self.calls = []

        def set_kv_buffer(self, layer, loc, k, v, k_scale, v_scale):
            self.calls.append((loc.shape[0], k.shape[0], v.shape[0]))

    def test_padded_tail_trimmed_to_loc_rows(self):
        # Bucket-padded replay: 16 k/v rows, 5 real write locs.
        b = self._bare_backend()
        pool = self._RecordingPool()
        k = self.torch.zeros(16, 2, 8, dtype=self.torch.bfloat16)
        v = self.torch.zeros(16, 2, 8, dtype=self.torch.bfloat16)
        locs = self.torch.zeros(5, dtype=self.torch.int32)
        b._save_kv_cache(
            SimpleNamespace(layer_id=0, k_scale=None, v_scale=None), locs, pool, k, v
        )
        self.assertEqual(pool.calls, [(5, 5, 5)])

    def test_matching_rows_untouched(self):
        # Radix / unpadded: loc rows == k rows -> no slicing.
        b = self._bare_backend()
        pool = self._RecordingPool()
        k = self.torch.zeros(16, 2, 8, dtype=self.torch.bfloat16)
        v = self.torch.zeros(16, 2, 8, dtype=self.torch.bfloat16)
        locs = self.torch.zeros(16, dtype=self.torch.int32)
        b._save_kv_cache(
            SimpleNamespace(layer_id=0, k_scale=None, v_scale=None), locs, pool, k, v
        )
        self.assertEqual(pool.calls, [(16, 16, 16)])

    def test_none_k_returns(self):
        b = self._bare_backend()
        pool = self._RecordingPool()
        b._save_kv_cache(
            SimpleNamespace(layer_id=0), self.torch.zeros(4), pool, None, None
        )
        self.assertEqual(pool.calls, [])


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

    def test_save_kv_trims_padded_tail(self):
        b = self._bare_backend()
        calls = []

        class _Pool:
            def set_kv_buffer(self, layer, loc, k, v, k_scale, v_scale):
                calls.append((loc.shape[0], k.shape[0], v.shape[0]))

        k = self.torch.zeros(16, 2, 8, dtype=self.torch.bfloat16)
        v = self.torch.zeros(16, 2, 8, dtype=self.torch.bfloat16)
        q = self.torch.zeros(16, 4 * 8, dtype=self.torch.bfloat16)
        locs = self.torch.zeros(5, dtype=self.torch.int32)
        layer = SimpleNamespace(
            layer_id=0, k_scale=None, v_scale=None, tp_q_head_num=4, head_dim=8
        )
        out_q = b._save_kv_and_prepare_q(q, k, v, layer, locs, _Pool(), True)
        self.assertEqual(calls, [(5, 5, 5)])
        # q keeps the padded rows: the graphed layers expect bucket shape.
        self.assertEqual(out_q.shape[0], 16)


if __name__ == "__main__":
    unittest.main()
