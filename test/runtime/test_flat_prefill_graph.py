"""Flat KV-cache x prefill CUDA graph seams.

Prefill-graph replay pads q/k/v rows to the bucket while flat per-group
write locs cover only the real (leading) tokens; the mha KV write must trim
the padded tail or the store kernel walks past the loc array (IAE on the
first padded replay -- reproduced on gpt-oss + flat + default prefill graph).
Capture must also exercise the flat metadata branch via dummy block tables
so capture and replay take the same code path.
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

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

    def test_flat_backend_gets_writable_state_tables(self):
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
        tables = self._bare(backend, pool)._dummy_flat_tables(100, 1)
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
            uses_flat_cache_groups=True,
            page_size=32,
            max_num_pages=2500,
            flat_state_group_ids=frozenset(),
        )
        pool = SimpleNamespace(
            paged_cache_group_specs=(SimpleNamespace(group_id="full_attention"),)
        )
        tables = self._bare(backend, pool)._dummy_flat_tables(100, 1)
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
        tables = self._bare(wrapper, pool)._dummy_flat_tables(64, 1)
        self.assertEqual(set(tables), {"full_attention", "linear_attention"})
        self.assertEqual(tables["full_attention"].shape, (1, 2))

    def test_non_flat_backend_empty(self):
        backend = SimpleNamespace(uses_flat_cache_groups=False)
        pool = SimpleNamespace(paged_cache_group_specs=())
        self.assertEqual(self._bare(backend, pool)._dummy_flat_tables(64, 1), {})

    def test_v4_state_group_uses_absolute_logical_width(self):
        backend = SimpleNamespace(
            uses_flat_cache_groups=True,
            page_size=64,
            max_num_pages=64,
            flat_state_group_ids=frozenset({"v4.c4a.compressor_state"}),
        )
        pool = SimpleNamespace(
            paged_cache_group_specs=(
                SimpleNamespace(
                    group_id="v4.swa_kv",
                    family="state",
                    rows_per_page=64,
                    entry_stride_tokens=1,
                ),
                SimpleNamespace(
                    group_id="v4.c4a.compressor_state",
                    family="state",
                    rows_per_page=4,
                    entry_stride_tokens=1,
                ),
            )
        )
        tables = self._bare(backend, pool)._dummy_flat_tables(4096, 2)
        self.assertEqual(tables["v4.swa_kv"].shape, (2, 64))
        self.assertEqual(
            tables["v4.c4a.compressor_state"].shape,
            (2, 1024),
        )

    def test_v4_dummy_batch_satisfies_active_real_page_contract(self):
        import torch

        from tokenspeed.runtime.configs.deepseek_v4_cache_spec import (
            build_v4_cache_specs,
        )
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
        from tokenspeed.runtime.layers.attention.backends import deepseek_v4
        from tokenspeed.runtime.layers.attention.backends.deepseek_v4 import (
            DeepseekV4AttentionBackend,
        )

        config = SimpleNamespace(
            page_size=64,
            device="cpu",
            num_attention_heads=64,
            num_kv_heads=1,
            attn_tp_size=1,
            dtype=torch.bfloat16,
            is_draft=False,
            speculative_num_draft_tokens=4,
            speculative_num_steps=1,
            head_dim=512,
            qk_rope_head_dim=64,
            context_len=4096,
            world_size=1,
        )
        specs = tuple(
            build_v4_cache_specs(
                SimpleNamespace(sliding_window=128),
                layer_ratio=(1, 4, 128),
            )
        )
        with mock.patch.object(
            deepseek_v4, "scheduler_ext_flat_kvcache", return_value=True
        ):
            backend = DeepseekV4AttentionBackend(config)
        backend.init_cuda_graph_state(
            2,
            paged_cache_group_specs=specs,
            paged_cache_group_page_counts={spec.group_id: 4096 for spec in specs},
        )
        pool = SimpleNamespace(
            paged_cache_group_specs=specs,
            runtime_contract=None,
        )
        input_buffers = SimpleNamespace(
            seq_lens_buf=torch.zeros(2, dtype=torch.int32),
            input_ids_buf=torch.zeros(256, dtype=torch.int32),
            out_cache_loc_buf=torch.zeros(256, dtype=torch.int32),
            positions_buf=torch.zeros(256, dtype=torch.int64),
            req_pool_indices_buf=torch.zeros(2, dtype=torch.int32),
            extend_seq_lens_buf=torch.zeros(2, dtype=torch.int32),
            extend_seq_lens_cpu=torch.zeros(2, dtype=torch.int32),
            extend_prefix_lens_buf=torch.zeros(2, dtype=torch.int32),
            extend_prefix_lens_cpu=torch.zeros(2, dtype=torch.int32),
            dummy_kv_slot=0,
        )
        graph = self.PrefillGraph.__new__(self.PrefillGraph)
        graph.attn_backend = backend
        graph.token_to_kv_pool = pool
        graph.config = config
        graph.input_buffers = input_buffers
        graph.req_to_page = torch.zeros((2, 64), dtype=torch.int32)
        graph.dp_size = 1
        graph.drafter = None

        context = graph.make_dummy_batch(130, decode_wrapper=None)

        self.assertEqual(context.forward_mode, ForwardMode.EXTEND)
        metadata = backend.forward_metadata
        self.assertIsNotNone(metadata)
        assert metadata is not None
        self.assertEqual(
            tuple(metadata.cache.paged_cache_block_tables),
            tuple(spec.group_id for spec in specs),
        )
        for table in metadata.cache.paged_cache_block_tables.values():
            self.assertTrue(bool((table > 0).all()))

    def test_dual_capable_backend_selects_one_capture_contract(self):
        flat = SimpleNamespace(
            uses_paged_cache_groups=True,
            uses_flat_cache_groups=True,
        )
        radix = SimpleNamespace(
            uses_paged_cache_groups=True,
            uses_flat_cache_groups=False,
        )
        self.assertFalse(self.PrefillGraph._capture_uses_radix_group_tables(flat))
        self.assertTrue(self.PrefillGraph._capture_uses_radix_group_tables(radix))

    def test_runtime_contract_pool_is_eligible_for_capture(self):
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
        pool = SimpleNamespace(runtime_contract=object())
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
                req_to_page=object(),
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
        b = self._bare_backend()
        self.assertTrue(b.support_kv_cache_prewrite(None))
        with mock.patch.object(
            self.mod, "is_breakable_capture_active", return_value=True
        ):
            self.assertFalse(b.support_kv_cache_prewrite(None))

    def test_declares_history_contract_family(self):
        self.assertEqual(
            self.mod.TRTLLMMHAAttnBackend.flat_cache_consumer_families,
            frozenset({"history"}),
        )


if __name__ == "__main__":
    unittest.main()
