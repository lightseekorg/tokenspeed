"""Paged KV-cache and prefill CUDA-graph seams.

Prefill-graph replay pads q/k/v rows to the bucket while flat per-group
write locs cover only the real (leading) tokens; the mha KV write must trim
the padded tail or the store kernel walks past the loc array (IAE on the
first padded replay -- reproduced on gpt-oss + flat + default prefill graph).
Capture must also exercise the cache metadata branch via dummy block tables
so capture and replay take the same code path.
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


def _spec(group_id: str, *, family: str = "history", **fields) -> SimpleNamespace:
    """A published group-spec double. ``family`` decides state vs history, so
    it is never inferred from the group id."""
    return SimpleNamespace(group_id=group_id, family=family, **fields)


def _fake_pool(*, specs=(), **arena_attrs) -> SimpleNamespace:
    """A cache-view double: the arena publishes, the view just names it."""
    return SimpleNamespace(
        arena=SimpleNamespace(cache_group_specs=tuple(specs), **arena_attrs)
    )


class SliceMhaExtendInputsTest(unittest.TestCase):
    """MHA kernels see exactly the rows covered by live cu-seqlens."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.backends import mha
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.slice_inputs = mha._slice_extend_inputs

    def test_padded_tail_is_not_passed_to_kernel(self):
        metadata = SimpleNamespace(cu_extend_seq_lens_cpu=[0, 3])
        q = self.torch.zeros(4, 2, 8)
        k = self.torch.zeros(4, 2, 8)
        v = self.torch.zeros(4, 2, 8)

        q, k, v = self.slice_inputs(metadata, q, k, v)

        self.assertEqual((q.shape[0], k.shape[0], v.shape[0]), (3, 3, 3))

    def test_unpadded_inputs_are_unchanged(self):
        metadata = SimpleNamespace(cu_extend_seq_lens_cpu=[0, 4])
        q = self.torch.zeros(4, 2, 8)
        self.assertIs(self.slice_inputs(metadata, q, None, None)[0], q)


class TrimKvToLocsTest(unittest.TestCase):
    """_trim_kv_to_locs slices padded k/v tails to the write-loc count --
    the shared fix point every flat-capable backend's KV write calls.
    Trimming (not loc-padding) keeps the null page 0 all-zero: trtllm does
    not scrub padded tail rows before saving KV."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.backends.cache_groups import (
                CacheGroupsMixin,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.trim = CacheGroupsMixin._trim_kv_to_locs

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


class DummyGroupTablesTest(unittest.TestCase):
    """Capture-time dummy tables: null KV pages and writable state pages."""

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

    def test_backend_gets_writable_state_tables(self):
        backend = SimpleNamespace(
            kernel_page_size=32,
            max_num_pages=0,  # fall back to bucket-derived width
            state_group_ids=frozenset({"linear_attention"}),
            cache_active_pages_must_be_real=False,
        )
        pool = _fake_pool(
            specs=(
                _spec("full_attention"),
                _spec("sliding_attention"),
                _spec("linear_attention", family="state"),  # state: included
            )
        )
        tables = self._bare(backend, pool)._dummy_group_tables(100, 1)
        self.assertEqual(
            set(tables),
            {"full_attention", "sliding_attention", "linear_attention"},
        )
        for t in tables.values():
            self.assertEqual(t.shape, (1, 4))  # ceil(100/32)
        self.assertEqual(int(tables["full_attention"].abs().sum()), 0)
        self.assertEqual(int(tables["sliding_attention"].abs().sum()), 0)
        self.assertTrue(
            bool((tables["linear_attention"] == 1).all()),
            "state checkpoints need a writable dummy page during graph capture",
        )

    def test_full_width_for_stride_deriving_backends(self):
        # trtllm-style: row stride comes from max_kv_len, so dummy tables
        # must span the full table width, not just the bucket.
        backend = SimpleNamespace(
            kernel_page_size=32,
            max_num_pages=2500,
            state_group_ids=frozenset(),
            cache_active_pages_must_be_real=False,
        )
        pool = _fake_pool(specs=(_spec("full_attention"),))
        tables = self._bare(backend, pool)._dummy_group_tables(100, 1)
        self.assertEqual(tables["full_attention"].shape, (1, 2500))

    def test_real_active_page_contract_uses_per_group_geometry(self):
        backend = SimpleNamespace(
            cache_active_pages_must_be_real=True,
            prefix_granularity=256,
            max_num_pages=257,
            state_group_ids=frozenset(),
        )
        pool = _fake_pool(
            specs=(
                _spec("fine", block_granularity=4),
                _spec("coarse", block_granularity=256),
            )
        )

        tables = self._bare(backend, pool)._dummy_group_tables(65536, 1)

        self.assertEqual(tables["fine"].shape, (1, 16384))
        self.assertEqual(tables["coarse"].shape, (1, 256))
        self.assertTrue(bool((tables["fine"] == 1).all()))
        self.assertTrue(bool((tables["coarse"] == 1).all()))

    def test_composite_wrapper_resolves_grouped_cache_child(self):
        # Hybrid wrappers set the flag but hold the paged KV consumer as
        # full_attn_backend; the helper must not AttributeError (which would
        # silently disable the prefill graph via the capture fallback).
        child = SimpleNamespace(
            kernel_page_size=32,
            max_num_pages=0,
            state_group_ids=frozenset(),
            cache_active_pages_must_be_real=False,
        )
        wrapper = SimpleNamespace(full_attn_backend=child)
        pool = _fake_pool(
            specs=(
                _spec("full_attention"),
                _spec("linear_attention", family="state"),
            )
        )
        tables = self._bare(wrapper, pool)._dummy_group_tables(64, 1)
        self.assertEqual(set(tables), {"full_attention", "linear_attention"})
        self.assertEqual(tables["full_attention"].shape, (1, 2))

    def test_pool_without_groups_is_empty(self):
        backend = SimpleNamespace(
            kernel_page_size=64, cache_active_pages_must_be_real=False
        )
        pool = _fake_pool(specs=())
        self.assertEqual(self._bare(backend, pool)._dummy_group_tables(64, 1), {})

    def test_runtime_contract_pool_is_eligible_for_capture(self):
        from unittest import mock

        inner_model = SimpleNamespace(embed_tokens=object())
        model_runner = SimpleNamespace(
            model=SimpleNamespace(model=inner_model),
            is_generation=True,
            is_multimodal=False,
        )
        config = SimpleNamespace(
            enforce_eager=False,
            disable_prefill_graph=False,
            data_parallel_size=1,
        )
        pool = _fake_pool(runtime_contract=object())
        with (
            mock.patch(
                "tokenspeed.runtime.execution.prefill_graph.get_prefill_token_buckets",
                return_value=[64],
            ),
            mock.patch.object(self.PrefillGraph, "capture") as capture,
        ):
            graph = self.PrefillGraph(
                model_runner=model_runner,
                attn_backend=object(),
                token_to_kv_pool=pool,
                input_buffers=object(),
                config=config,
                page_table=object(),
            )

        self.assertFalse(graph.disable)
        capture.assert_not_called()


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

    def test_declares_history_contract_family(self):
        self.assertEqual(
            self.mod.TRTLLMMHAAttnBackend.cache_consumer_families,
            frozenset({"history"}),
        )


if __name__ == "__main__":
    unittest.main()
