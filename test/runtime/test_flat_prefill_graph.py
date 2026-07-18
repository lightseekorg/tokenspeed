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
    trim padded tails like mha. The context kernel must likewise receive only
    live query rows, never the larger capture-bucket tensor."""

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

    def test_context_inputs_trimmed_to_cpu_metadata_token_count(self):
        from unittest import mock

        bucket_tokens = 1536
        live_tokens = 1478
        prefix_tokens = 60_222
        b = self._bare_backend()
        b.max_context_len = 100_000
        b.workspace_buffer = self.torch.empty(0, dtype=self.torch.uint8)
        b.dtype = self.torch.bfloat16

        block_tables = self.torch.zeros((1, 1), dtype=self.torch.int32)
        b._init_extend_metadata(
            bs=1,
            req_pool_indices=self.torch.tensor([0], dtype=self.torch.int32),
            seq_lens=self.torch.tensor(
                [prefix_tokens + live_tokens], dtype=self.torch.int32
            ),
            req_to_page=None,
            extend_with_prefix=True,
            extend_prefix_lens=self.torch.tensor(
                [prefix_tokens], dtype=self.torch.int32
            ),
            extend_prefix_lens_cpu=self.torch.tensor(
                [prefix_tokens], dtype=self.torch.int32
            ),
            extend_seq_lens_cpu=self.torch.tensor(
                [live_tokens], dtype=self.torch.int32
            ),
            flat_page_tables={"full_attention": block_tables},
        )
        metadata = b.forward_prefill_metadata
        self.assertEqual(metadata.num_query_tokens, live_tokens)

        layer = SimpleNamespace(
            group_id="full_attention",
            tp_q_head_num=1,
            tp_k_head_num=1,
            head_dim=4,
            sliding_window_size=-1,
        )
        q = (
            self.torch.arange(bucket_tokens * layer.head_dim, dtype=self.torch.float32)
            .to(self.torch.bfloat16)
            .view(bucket_tokens, layer.head_dim)
        )
        k = q + 10_000
        v = q + 20_000
        out_cache_loc = (
            self.torch.arange(bucket_tokens, dtype=self.torch.int32) + 30_000
        )
        k_cache = self.torch.empty(0)
        v_cache = self.torch.empty(0)
        b._get_kv_cache_permuted = mock.Mock(return_value=(k_cache, v_cache))
        b._compute_scales = mock.Mock(return_value=(0.125, 1.0))

        def fake_context_kernel(**kwargs):
            return kwargs["query"].clone()

        with (
            mock.patch.object(
                b,
                "_save_kv_and_prepare_q",
                wraps=b._save_kv_and_prepare_q,
            ) as save_kv,
            mock.patch.object(
                self.mod,
                "trtllm_batch_context_with_kv_cache",
                side_effect=fake_context_kernel,
            ) as context_kernel,
        ):
            out = b.forward_extend(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool=object(),
                bs=1,
                save_kv_cache=False,
            )

        save_args = save_kv.call_args.args
        self.assertEqual([arg.shape[0] for arg in save_args[:3]], [live_tokens] * 3)
        self.assertEqual(save_args[4].shape[0], live_tokens)
        for actual, expected in zip(
            (*save_args[:3], save_args[4]),
            (q, k, v, out_cache_loc),
            strict=True,
        ):
            self.assertTrue(self.torch.equal(actual, expected[:live_tokens]))
        call = context_kernel.call_args.kwargs
        self.assertEqual(call["query"].shape[0], live_tokens)
        self.assertTrue(
            self.torch.equal(
                call["query"], q[:live_tokens].view(live_tokens, 1, layer.head_dim)
            )
        )
        self.assertIs(call["cum_seq_lens_q"], metadata.cu_seqlens_q)
        self.assertIs(call["cum_seq_lens_kv"], metadata.cu_seqlens_k)
        self.assertIs(call["seq_lens"], metadata.cache_seqlens_int32)
        self.assertIs(call["block_tables"], block_tables)
        self.assertEqual(call["max_q_len"], live_tokens)
        self.assertEqual(call["batch_size"], 1)
        self.assertEqual(call["window_left"], -1)
        self.assertEqual(out.shape[0], live_tokens)

    def test_mixed_metadata_does_not_trim_using_extend_only_token_count(self):
        b = self._bare_backend()
        b.max_context_len = 100_000
        block_tables = self.torch.zeros((2, 1), dtype=self.torch.int32)

        b._init_extend_metadata(
            bs=2,
            req_pool_indices=self.torch.tensor([0, 1], dtype=self.torch.int32),
            seq_lens=self.torch.tensor([1478, 1], dtype=self.torch.int32),
            req_to_page=None,
            forward_mode=self.mod.ForwardMode.MIXED,
            extend_seq_lens_cpu=self.torch.tensor([1478], dtype=self.torch.int32),
            flat_page_tables={"full_attention": block_tables},
        )

        self.assertIsNone(b.forward_prefill_metadata.num_query_tokens)


if __name__ == "__main__":
    unittest.main()
