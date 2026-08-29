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


def _backend(**attrs) -> SimpleNamespace:
    """An attention-backend double whose ``consumes_cache_metadata`` the
    production property computes, so these doubles cannot drift from it."""
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend

    attrs.setdefault("uses_cache_groups", False)
    attrs.setdefault("_cache_contract_bound", False)
    attrs.setdefault("capture_table_in_block_granularity", False)
    attrs.setdefault("max_num_pages", 0)
    ns = SimpleNamespace(**attrs)
    ns.consumes_cache_metadata = AttentionBackend.consumes_cache_metadata.fget(ns)
    return ns


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

    def test_every_group_gets_a_real_page(self):
        backend = _backend(
            uses_cache_groups=True,
            kernel_page_size=32,
            max_num_pages=0,  # fall back to bucket-derived width
            state_group_ids=frozenset({"linear_attention"}),
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
        for group_id, table in tables.items():
            self.assertTrue(
                bool((table == 1).all()),
                f"{group_id}: capture writes KV, so no group may get the "
                "reserved null page",
            )

    def test_full_width_for_stride_deriving_backends(self):
        # trtllm-style: row stride comes from max_kv_len, so dummy tables
        # must span the full table width, not just the bucket.
        backend = _backend(
            uses_cache_groups=True,
            kernel_page_size=32,
            max_num_pages=2500,
            state_group_ids=frozenset(),
        )
        pool = _fake_pool(specs=(_spec("full_attention"),))
        tables = self._bare(backend, pool)._dummy_group_tables(100, 1)
        self.assertEqual(tables["full_attention"].shape, (1, 2500))

    def test_real_active_page_contract_uses_per_group_geometry(self):
        backend = _backend(
            uses_cache_groups=True,
            capture_table_in_block_granularity=True,
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
        child = _backend(kernel_page_size=32, max_num_pages=0)
        wrapper = _backend(uses_cache_groups=True, full_attn_backend=child)
        pool = _fake_pool(
            specs=(
                _spec("full_attention"),
                _spec("linear_attention", family="state"),
            )
        )
        tables = self._bare(wrapper, pool)._dummy_group_tables(64, 1)
        self.assertEqual(set(tables), {"full_attention", "linear_attention"})
        self.assertEqual(tables["full_attention"].shape, (1, 2))

    def test_backend_without_cache_groups_is_empty(self):
        backend = _backend(uses_cache_groups=False)
        pool = _fake_pool(specs=())
        self.assertEqual(
            self._bare(backend, pool)._dummy_group_tables(64, 1),
            {},
        )

    def test_unbound_contract_is_still_empty(self):
        """An MLA backend the registry never bound keeps the legacy page_table
        path and must not be handed capture metadata."""
        backend = _backend(uses_cache_groups=False, _cache_contract_bound=False)
        pool = _fake_pool(specs=(_spec("full_attention"),))
        self.assertEqual(
            self._bare(backend, pool)._dummy_group_tables(64, 1),
            {},
        )

    def test_contract_bound_mla_target_gets_tables(self):
        """A plain MLA target leaves uses_cache_groups False by design, so
        gating capture on it handed the backend a dummy batch with no metadata,
        which it refuses -- Kimi-K2.5 ran every eval on eager prefill."""
        backend = _backend(
            uses_cache_groups=False,
            _cache_contract_bound=True,
            capture_table_in_block_granularity=True,
            kernel_page_size=32,
            max_num_pages=2500,
            state_group_ids=frozenset(),
        )
        pool = _fake_pool(specs=(_spec("full_attention", block_granularity=128),))
        tables = self._bare(backend, pool)._dummy_group_tables(256, 2)
        self.assertEqual(set(tables), {"full_attention"})
        # Scheduler-table columns span block_granularity, not the backend's
        # kernel pages: 256/128 == 2, where kernel geometry would say 8.
        self.assertEqual(tables["full_attention"].shape, (2, 2))
        self.assertTrue(
            bool((tables["full_attention"] == 1).all()),
            "MLA rejects the null page in live metadata, so capture needs a "
            "real writable page",
        )

    def test_mla_family_capture_avoids_the_reserved_page(self):
        """Capture must not hand an MLA backend page 0: write locations clamp
        into the reserved null page instead of failing, so the graph would
        scribble on the page that padding and table holes read as zero."""
        from tokenspeed.runtime.layers.attention.backends.mla_cache_groups import (
            MlaCacheGroupMixin,
        )
        from tokenspeed.runtime.layers.attention.backends.tokenspeed_mla import (
            CuteDSLMLABackend,
        )

        for cls in (MlaCacheGroupMixin, CuteDSLMLABackend):
            backend = _backend(
                _cache_contract_bound=True,
                capture_table_in_block_granularity=cls.capture_table_in_block_granularity,
                kernel_page_size=32,
                max_num_pages=2500,
                state_group_ids=frozenset(),
            )
            pool = _fake_pool(specs=(_spec("full_attention", block_granularity=128),))
            tables = self._bare(backend, pool)._dummy_group_tables(256, 1)
            self.assertGreater(
                int(tables["full_attention"].min()),
                0,
                f"{cls.__name__} capture would write into the reserved page",
            )

    def test_delegating_wrapper_answers_for_its_child(self):
        """A wrapper that forwards mark_cache_contract must forward the capture
        predicates too, or capture hands its bound child a batch the child
        declares invalid. Asserting the values, not their presence: a property
        that returns the wrong answer is the failure being guarded against."""
        from tokenspeed.runtime.layers.attention.backends.dsa import DSABackend

        wrapper = DSABackend.__new__(DSABackend)
        wrapper._dense_backend = SimpleNamespace(
            consumes_cache_metadata=True, capture_table_in_block_granularity=False
        )
        self.assertTrue(wrapper.consumes_cache_metadata)
        self.assertFalse(wrapper.capture_table_in_block_granularity)

        wrapper._dense_backend = SimpleNamespace(
            consumes_cache_metadata=False, capture_table_in_block_granularity=True
        )
        self.assertFalse(wrapper.consumes_cache_metadata)
        self.assertTrue(wrapper.capture_table_in_block_granularity)

    def test_inkling_width_reaches_the_inner_backend(self):
        """max_num_pages is a base-class attribute now, and a class attribute
        shadows Inkling's __getattr__ -- the mirror block exists for exactly
        this, and without the mirror a stride-deriving inner collapses from
        full width to bucket width."""
        from tokenspeed.runtime.layers.attention.backends.inkling import (
            InklingAttnBackend,
        )

        wrapper = InklingAttnBackend.__new__(InklingAttnBackend)
        wrapper.inner = SimpleNamespace(max_num_pages=2500)
        self.assertEqual(wrapper.max_num_pages, 2500)

    def test_inkling_wrapper_answers_for_its_inner(self):
        """Same contract as the DSA wrapper: Inkling mirrors flags explicitly
        because a class-level default on the base would shadow __getattr__."""
        from tokenspeed.runtime.layers.attention.backends.inkling import (
            InklingAttnBackend,
        )

        wrapper = InklingAttnBackend.__new__(InklingAttnBackend)
        for answer in (True, False):
            wrapper.inner = SimpleNamespace(
                consumes_cache_metadata=answer,
                capture_table_in_block_granularity=not answer,
                uses_cache_groups=answer,
            )
            self.assertIs(wrapper.consumes_cache_metadata, answer)
            self.assertIs(wrapper.capture_table_in_block_granularity, not answer)

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


class CaptureFailureIsLoudTest(unittest.TestCase):
    """A capture the dummy-batch machinery cannot serve must stop the boot.

    Degrading here is what let a whole model family run eager prefill with a
    warning nobody read: the warning was indistinguishable from the families
    that are deliberately eager.
    """

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.execution.prefill_graph import PrefillGraph
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + runtime deps: {exc}")
        self.torch = torch
        self.PrefillGraph = PrefillGraph

    def _bare(self, raises=None):
        pg = self.PrefillGraph.__new__(self.PrefillGraph)
        pg.disable = False
        pg.capture_buckets = [4]
        pg.attn_backend = SimpleNamespace()
        pg.config = SimpleNamespace(device="cpu", world_group=None, world_size=1)
        pg._embed_tokens = SimpleNamespace(
            weight=self.torch.zeros(2, 8, dtype=self.torch.float32)
        )

        def _capture_all_buckets(_decode_wrapper):
            if raises is not None:
                raise raises

        pg._capture_all_buckets = _capture_all_buckets
        return pg

    def test_capture_failure_propagates_untouched(self):
        """Nothing between the backend and the operator: same exception object,
        same traceback. ``capture`` has no handler at all, so a partial ladder
        cannot be left behind -- the boot dies with it."""
        cause = RuntimeError("backend refused the dummy batch")
        pg = self._bare(raises=cause)
        with self.assertRaises(RuntimeError) as caught:
            pg.capture(None)
        self.assertIs(caught.exception, cause)

    def test_successful_capture_does_not_raise(self):
        self._bare().capture(None)

    def test_oom_propagates(self):
        """OOM keeps its own type and message. The capture pool not fitting is
        an operator-visible sizing failure, not something to recover from."""
        pg = self._bare(raises=self.torch.cuda.OutOfMemoryError("no room"))
        with self.assertRaises(self.torch.cuda.OutOfMemoryError):
            pg.capture(None)


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
