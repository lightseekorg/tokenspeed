"""Per-group CUDA-graph pad, capture, and replay core-logic tests.

CPU-only (plain tensors, no graph capture): covers the wrapper's flat
placeholder + padding helpers and the MHA backend's flat capture/replay
branches. Graph runtime semantics (pointer-fixed replay) are validated
separately on GPU via the P0 probe.
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

from tokenspeed.runtime.layers.attention.backends.cache_group_geometry import (
    CacheGroupGeometry,
)

MAX_BS = 4
MAX_NUM_PAGES = 6


def _decode_forward_mode():
    return SimpleNamespace(
        is_extend_or_mixed=lambda: False,
        is_decode_or_idle=lambda: True,
        is_idle=lambda: False,
        is_mixed=lambda: False,
    )


class _TorchCase(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch: {exc}")
        self.torch = torch


class PadBlockTablesTest(_TorchCase):
    def setUp(self):
        super().setUp()
        from tokenspeed.runtime.execution.forward_step import (
            ForwardStepRunner,
        )

        self.pad = ForwardStepRunner._pad_block_tables_to_padded_bs

    def _tables(self):
        torch = self.torch
        return {
            "full_attention": torch.arange(6, dtype=torch.int32).reshape(2, 3),
            "sliding_attention": torch.ones((2, 3), dtype=torch.int32),
        }

    def test_default_pads_tail_rows_with_minus_one(self):
        # single-table/V4 path keeps -1 dummy rows: the backend masks dummy tokens
        # via is_valid_token before any block-table read.
        tables = self._tables()
        out = self.pad(tables, actual_bs=2, padded_bs=4)
        for gid, src in tables.items():
            self.assertEqual(tuple(out[gid].shape), (4, 3))
            self.assertTrue((out[gid][:2] == src).all())
            self.assertTrue((out[gid][2:] == -1).all())

    def test_pads_tail_rows_with_zero(self):
        # Grouped cache replay passes pad_value=0: dummy rows replay with
        # seq_lens=1 and ARE dereferenced, so they must land on the zero-init
        # dummy page 0.
        tables = self._tables()
        out = self.pad(tables, actual_bs=2, padded_bs=4, pad_value=0)
        for gid, src in tables.items():
            self.assertEqual(tuple(out[gid].shape), (4, 3))
            self.assertTrue((out[gid][:2] == src).all())
            self.assertTrue((out[gid][2:] == 0).all())

    def test_noop_when_bs_equal(self):
        torch = self.torch
        tables = {"full_attention": torch.ones((3, 2), dtype=torch.int32)}
        out = self.pad(tables, actual_bs=3, padded_bs=3)
        self.assertIs(out["full_attention"], tables["full_attention"])

    def test_rejects_partial_or_oversized_batch_rows(self):
        torch = self.torch
        for rows in (2, 5):
            with self.subTest(rows=rows), self.assertRaisesRegex(
                RuntimeError, "expected actual_bs=3 or padded_bs=4"
            ):
                self.pad(
                    {"full_attention": torch.ones((rows, 2), dtype=torch.int32)},
                    actual_bs=3,
                    padded_bs=4,
                )


class CacheGroupIdsTest(_TorchCase):
    """Wrapper-side capture contract: group ids only, no fabricated tensors."""

    def setUp(self):
        super().setUp()
        from tokenspeed.runtime.execution.forward_step import (
            ForwardStepRunner,
        )

        self.group_ids = ForwardStepRunner._cache_group_ids

    def _wrapper(self):
        return SimpleNamespace(
            attn_backend=SimpleNamespace(),
        )

    def _pool(self, group_ids):
        # A cache view names its arena; the arena publishes the specs.
        return SimpleNamespace(
            arena=SimpleNamespace(
                cache_group_specs=tuple(
                    SimpleNamespace(group_id=gid) for gid in group_ids
                )
            )
        )

    def test_ids_in_spec_order(self):
        out = self.group_ids(
            self._wrapper(),
            self._pool(["sliding_attention", "full_attention"]),
        )
        self.assertEqual(out, ("sliding_attention", "full_attention"))

    def test_empty_without_specs(self):
        self.assertEqual(self.group_ids(self._wrapper(), self._pool([])), ())


class DraftCacheGroupIdsTest(_TorchCase):
    """DFLASH owns an independent draft page table; EAGLE-style drafts use
    target cache-group tables at matching page ids."""

    def setUp(self):
        super().setUp()
        from tokenspeed.runtime.execution.forward_step import (
            ForwardStepRunner,
        )

        self.group_ids = ForwardStepRunner._draft_cache_group_ids

    def _wrapper(
        self,
        *,
        draft_block_decode,
        families=("history",),
    ):
        return SimpleNamespace(
            draft_attn_backend=SimpleNamespace(
                draft_block_decode=draft_block_decode,
                cache_consumer_families=frozenset(families),
            ),
            draft_token_to_kv_pool=SimpleNamespace(
                arena=SimpleNamespace(
                    cache_group_specs=(
                        SimpleNamespace(group_id="full_attention", family="history"),
                        SimpleNamespace(group_id="state", family="state"),
                    )
                )
            ),
        )

    def test_dflash_does_not_capture_target_group_tables(self):
        self.assertEqual(
            self.group_ids(self._wrapper(draft_block_decode=True)),
            (),
        )

    def test_eagle_draft_uses_published_history_groups(self):
        self.assertEqual(
            self.group_ids(self._wrapper(draft_block_decode=False)),
            ("full_attention",),
        )

    def test_stateful_draft_uses_published_state_groups(self):
        self.assertEqual(
            self.group_ids(
                self._wrapper(
                    draft_block_decode=False,
                    families=("history", "state"),
                )
            ),
            ("full_attention", "state"),
        )


class WrapperReplayGroupedTest(_TorchCase):
    """Call-site wiring: the real _prepare_decode_metadata must row-pad
    grouped tables with 0 (not the -1 default) before handing them to the
    backend's unified refresh."""

    def _run_replay(self, block_tables, padded_bs, actual_bs):
        torch = self.torch
        from tokenspeed.runtime.execution.forward_step import (
            ForwardStepRunner,
        )

        recorded = {}

        def record(bs, actual_bs, req_pool_indices, seq_lens, **kwargs):
            recorded["bs"] = bs
            recorded["actual_bs"] = actual_bs
            recorded.update(kwargs)

        mock = SimpleNamespace(
            max_tokens_per_req=1,
            attn_backend=SimpleNamespace(
                refresh_decode_metadata=record,
                tables_self_padding=False,
            ),
            draft_attn_backend=None,
            # Production helper, so the pinned pad_value is the real one.
            _pad_block_tables_to_padded_bs=(
                ForwardStepRunner._pad_block_tables_to_padded_bs
            ),
        )
        ForwardStepRunner._prepare_decode_metadata(
            mock,
            padded_bs,
            actual_bs,
            torch.arange(padded_bs, dtype=torch.int64),
            torch.ones(padded_bs, dtype=torch.int32),
            torch.zeros((MAX_BS, MAX_NUM_PAGES), dtype=torch.int32),
            _decode_forward_mode(),
            use_graph=True,
            block_tables=block_tables,
        )
        return recorded

    def test_replay_path_pads_with_zero(self):
        torch = self.torch
        src = {
            "sliding_attention": torch.tensor([[3, 4], [5, 6]], dtype=torch.int32),
            "full_attention": torch.tensor([[7, 8], [9, 1]], dtype=torch.int32),
        }
        recorded = self._run_replay(src, padded_bs=4, actual_bs=2)
        self.assertEqual(recorded["bs"], 4)
        out = recorded["block_tables"]
        self.assertEqual(set(out), set(src))
        for gid, table in out.items():
            self.assertEqual(tuple(table.shape), (4, 2))
            self.assertTrue((table[:2] == src[gid]).all())
            # Dummy rows must land on the zero-init dummy page 0, never -1:
            # they replay with seq_lens=1 and their col-0 IS dereferenced.
            self.assertTrue((table[2:] == 0).all())

    def test_replay_path_noop_without_padding(self):
        torch = self.torch
        src = {"full_attention": torch.ones((2, 2), dtype=torch.int32)}
        recorded = self._run_replay(src, padded_bs=2, actual_bs=2)
        self.assertIs(
            recorded["block_tables"]["full_attention"],
            src["full_attention"],
        )

    def test_single_table_target_pads_group_tables_before_draft_routing(self):
        torch = self.torch
        from tokenspeed.runtime.execution.forward_step import (
            ForwardStepRunner,
        )

        draft_call = {}

        def record_draft(bs, actual_bs, req_pool_indices, seq_lens, **kwargs):
            draft_call.update(kwargs)

        mock = SimpleNamespace(
            max_tokens_per_req=1,
            attn_backend=SimpleNamespace(
                refresh_decode_metadata=lambda *args, **kwargs: None,
                tables_self_padding=False,
            ),
            draft_attn_backend=SimpleNamespace(
                refresh_decode_metadata=record_draft,
                tables_self_padding=False,
            ),
            drafter=SimpleNamespace(
                draft_seq_lens_buf=torch.zeros(2, dtype=torch.int32),
                page_staging=SimpleNamespace(
                    table=torch.zeros((2, MAX_NUM_PAGES), dtype=torch.int32)
                ),
            ),
            _draft_group_tables=lambda tables: tables,
            _pad_block_tables_to_padded_bs=(
                ForwardStepRunner._pad_block_tables_to_padded_bs
            ),
        )
        tables = {
            "full_attention": torch.tensor([[3, 4]], dtype=torch.int32),
        }

        ForwardStepRunner._prepare_decode_metadata(
            mock,
            padded_bs=2,
            actual_bs=1,
            req_pool_indices=torch.arange(2, dtype=torch.int64),
            seq_lens=torch.ones(2, dtype=torch.int32),
            page_table=torch.zeros((2, MAX_NUM_PAGES), dtype=torch.int32),
            forward_mode=_decode_forward_mode(),
            use_graph=True,
            block_tables=tables,
        )

        padded = draft_call["block_tables"]
        self.assertEqual(tuple(padded["full_attention"].shape), (2, 2))
        self.assertTrue(
            (padded["full_attention"][:1] == tables["full_attention"]).all()
        )
        self.assertTrue((padded["full_attention"][1:] == 0).all())

    def test_target_and_draft_share_padded_replay_tables(self):
        torch = self.torch
        from tokenspeed.runtime.execution.forward_step import (
            ForwardStepRunner,
        )

        calls = {}

        def record_target(bs, actual_bs, req_pool_indices, seq_lens, **kwargs):
            calls["target"] = kwargs["block_tables"]

        def record_draft(bs, actual_bs, req_pool_indices, seq_lens, **kwargs):
            calls["draft"] = kwargs["block_tables"]

        mock = SimpleNamespace(
            max_tokens_per_req=1,
            attn_backend=SimpleNamespace(
                refresh_decode_metadata=record_target,
                tables_self_padding=False,
            ),
            draft_attn_backend=SimpleNamespace(
                refresh_decode_metadata=record_draft,
                tables_self_padding=False,
            ),
            drafter=SimpleNamespace(
                draft_seq_lens_buf=torch.zeros(4, dtype=torch.int32),
                page_staging=SimpleNamespace(
                    table=torch.zeros((4, MAX_NUM_PAGES), dtype=torch.int32)
                ),
            ),
            _draft_group_tables=lambda tables: {
                "full_attention": tables["full_attention"]
            },
            _pad_block_tables_to_padded_bs=(
                ForwardStepRunner._pad_block_tables_to_padded_bs
            ),
        )
        tables = {
            "full_attention": torch.arange(6, dtype=torch.int32).reshape(3, 2),
            "state": torch.ones((3, 2), dtype=torch.int32),
        }

        ForwardStepRunner._prepare_decode_metadata(
            mock,
            padded_bs=4,
            actual_bs=3,
            req_pool_indices=torch.arange(4, dtype=torch.int64),
            seq_lens=torch.ones(4, dtype=torch.int32),
            page_table=torch.zeros((4, MAX_NUM_PAGES), dtype=torch.int32),
            forward_mode=_decode_forward_mode(),
            use_graph=True,
            block_tables=tables,
        )

        self.assertIs(
            calls["draft"]["full_attention"],
            calls["target"]["full_attention"],
        )
        self.assertEqual(set(calls["target"]), set(tables))
        self.assertEqual(set(calls["draft"]), {"full_attention"})
        for table in calls["target"].values():
            self.assertEqual(tuple(table.shape), (4, 2))
            self.assertTrue((table[3:] == 0).all())


class WrapperCaptureGroupIdsTest(_TorchCase):
    """Call-site wiring: the real _init_capture_metadata must derive
    cache_group_ids from the pool's published specs and pass them to
    the backend capture hook."""

    def _run_capture(self, bs, group_ids, page_table=None):
        torch = self.torch
        from types import MethodType

        from tokenspeed.runtime.execution.forward_step import (
            ForwardStepRunner,
        )

        recorded = {}

        def record(bs, req_pool_indices, seq_lens, forward_mode, **kwargs):
            recorded["bs"] = bs
            recorded["kwargs"] = kwargs

        mock = SimpleNamespace(
            max_tokens_per_req=1,
            input_buffers=SimpleNamespace(
                has_mamba=False,
                req_pool_indices_buf=torch.arange(MAX_BS, dtype=torch.int64),
                seq_lens_buf=torch.ones(MAX_BS, dtype=torch.int32),
            ),
            attn_backend=SimpleNamespace(
                init_forward_metadata_capture_cuda_graph=record,
                cache_active_pages_must_be_real=False,
            ),
            token_to_kv_pool=SimpleNamespace(
                arena=SimpleNamespace(
                    cache_group_specs=tuple(
                        SimpleNamespace(group_id=gid) for gid in group_ids
                    )
                )
            ),
            drafter=None,
            use_target_verify_forward_mode=False,
            draft_attn_backend=None,
            page_table=page_table,
        )
        mock._cache_group_ids = MethodType(ForwardStepRunner._cache_group_ids, mock)
        ForwardStepRunner._init_capture_metadata(mock, bs)
        return recorded

    def test_capture_passes_group_ids_from_pool_specs(self):
        recorded = self._run_capture(2, ["sliding_attention", "full_attention"])
        self.assertEqual(recorded["bs"], 2)
        self.assertEqual(
            recorded["kwargs"]["cache_group_ids"],
            ("sliding_attention", "full_attention"),
        )

    def test_capture_omits_group_ids_without_specs(self):
        recorded = self._run_capture(2, [])
        self.assertNotIn("cache_group_ids", recorded["kwargs"])

    def test_capture_forwards_the_staged_page_table(self):
        # The default capture's idle refresh reads the same address-stable
        # staged table replay passes; the runner must hand it through.
        table = self.torch.zeros((MAX_BS, 3), dtype=self.torch.int32)
        recorded = self._run_capture(2, [], page_table=table)
        self.assertIs(recorded["kwargs"]["page_table"], table)


class WrapperEagerGroupGuardTest(_TorchCase):
    """Eager parity guard: a multi-group pool and group-aware backend
    must not reach the backend's single-table fallback without tables."""

    def _call(self, group_ids, block_tables=None):
        torch = self.torch
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
        from tokenspeed.runtime.execution.forward_step import (
            ForwardStepRunner,
        )

        calls = {}

        def init_forward_metadata(*args, **kwargs):
            calls["init_kwargs"] = kwargs

        mock = SimpleNamespace(
            max_tokens_per_req=1,
            input_buffers=SimpleNamespace(
                seq_lens_buf=torch.ones(MAX_BS, dtype=torch.int32),
                req_pool_indices_buf=torch.arange(MAX_BS, dtype=torch.int64),
            ),
            config=SimpleNamespace(),
            attn_backend=SimpleNamespace(),
            token_to_kv_pool=SimpleNamespace(
                arena=SimpleNamespace(
                    cache_group_specs=tuple(
                        SimpleNamespace(group_id=gid) for gid in group_ids
                    )
                )
            ),
            drafter=None,
            draft_attn_backend=None,
            _can_use_graph=lambda bs, ctx: False,
            _init_forward_metadata=init_forward_metadata,
            _forward_func=lambda **kwargs: (None, None, None),
        )
        ctx = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            num_extends=2,
            input_num_tokens=2,
            global_num_tokens=None,
            all_decode_or_idle=False,
            capture_hidden_mode=None,
        )
        ForwardStepRunner.__call__(
            mock,
            bs=2,
            ctx=ctx,
            sampling_info=None,
            page_table=torch.zeros((MAX_BS, MAX_NUM_PAGES), dtype=torch.int32),
            block_tables=block_tables,
        )
        return calls

    def test_multi_group_eager_without_tables_raises(self):
        with self.assertRaisesRegex(RuntimeError, "block_tables"):
            self._call(["sliding_attention", "full_attention"])

    def test_multi_group_eager_with_tables_passes(self):
        torch = self.torch
        tables = {
            "sliding_attention": torch.ones((2, 2), dtype=torch.int32),
            "full_attention": torch.ones((2, 2), dtype=torch.int32),
        }
        calls = self._call(["sliding_attention", "full_attention"], block_tables=tables)
        self.assertIs(calls["init_kwargs"]["block_tables"], tables)

    def test_single_group_eager_without_tables_falls_back(self):
        # Documented fallback: with one published group the backend's single
        # table IS that group's table, so no tables are required.
        calls = self._call(["full_attention"])
        self.assertIsNone(calls["init_kwargs"]["block_tables"])


class DecodeStaleTableGuardTest(_TorchCase):
    """Decode table-delivery guard: one wrapper-level check covering eager
    and replay for every backend family (replaces the per-backend
    _replay_stale_guard)."""

    def _guard(self, group_ids, actual_bs, block_tables):
        from types import MethodType

        from tokenspeed.runtime.execution.forward_step import (
            ForwardStepRunner,
        )

        mock = SimpleNamespace(
            token_to_kv_pool=SimpleNamespace(
                arena=SimpleNamespace(
                    cache_group_specs=tuple(
                        SimpleNamespace(group_id=gid) for gid in group_ids
                    )
                )
            ),
        )
        mock._cache_group_ids = MethodType(ForwardStepRunner._cache_group_ids, mock)
        ForwardStepRunner._decode_stale_table_guard(mock, actual_bs, block_tables)

    def test_missing_tables_raise(self):
        with self.assertRaisesRegex(RuntimeError, "missing/empty"):
            self._guard(["full_attention"], 2, None)

    def test_missing_group_raises(self):
        tables = {"full_attention": self.torch.ones((2, 2), dtype=self.torch.int32)}
        with self.assertRaisesRegex(RuntimeError, "missing published groups"):
            self._guard(["full_attention", "sliding_attention"], 2, tables)

    def test_full_delivery_passes(self):
        tables = {
            "full_attention": self.torch.ones((2, 2), dtype=self.torch.int32),
            "sliding_attention": self.torch.ones((2, 2), dtype=self.torch.int32),
        }
        self._guard(["full_attention", "sliding_attention"], 2, tables)

    def test_idle_and_capture_rows_skip(self):
        # actual_bs == 0: idle replay / capture seeding synthesize their own
        # placeholder tables downstream.
        self._guard(["full_attention"], 0, None)

    def test_group_less_pool_skips(self):
        self._guard([], 2, None)


class IdleBlockTablesTest(_TorchCase):
    """bs==0 idle replay tables: one col-0 page-0 entry per dummy row."""

    def setUp(self):
        super().setUp()
        from tokenspeed.runtime.execution.forward_step import (
            ForwardStepRunner,
        )

        self.idle = ForwardStepRunner._idle_block_tables

    def _wrapper(self, group_ids):
        return SimpleNamespace(
            token_to_kv_pool=SimpleNamespace(
                arena=SimpleNamespace(
                    cache_group_specs=tuple(
                        SimpleNamespace(group_id=gid) for gid in group_ids
                    )
                )
            ),
            device="cpu",
        )

    def test_page_zero_single_column_per_group(self):
        out = self.idle(self._wrapper(["sliding_attention", "full_attention"]), 3)
        self.assertEqual(set(out), {"sliding_attention", "full_attention"})
        for table in out.values():
            self.assertEqual(tuple(table.shape), (3, 1))
            self.assertEqual(table.dtype, self.torch.int32)
            self.assertTrue((table == 0).all())

    def test_none_without_specs(self):
        self.assertIsNone(self.idle(self._wrapper([]), 3))


class _BackendCase(_TorchCase):
    """Real MHAAttnBackend methods on a __init__-bypassed instance."""

    def setUp(self):
        super().setUp()
        try:
            from tokenspeed.runtime.layers.attention.backends.mha import (
                MHAAttnBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs tokenspeed_kernel: {exc}")
        torch = self.torch
        backend = MHAAttnBackend.__new__(MHAAttnBackend)
        backend.spec_num_tokens = 1
        backend.is_draft = False
        backend.draft_block_decode = False
        backend._geometry = CacheGroupGeometry()
        backend.max_num_pages = MAX_NUM_PAGES
        backend.kernel_page_size = 2
        backend.device = "cpu"
        backend.cuda_graph_decode_metadata = {}
        backend.cuda_graph_page_table = torch.zeros(
            (MAX_BS, MAX_NUM_PAGES), dtype=torch.int32
        )
        # seq_lens 1 (never 0): flat replay recomputes write locs from these
        # (M11), and seq_len 0 would gather at position -1.
        backend.cuda_graph_seq_lens = torch.ones(MAX_BS, dtype=torch.int32)
        backend._init_group_graph_buffers(MAX_BS)
        self.backend = backend

    def _capture(self, bs, cache_group_ids=()):
        torch = self.torch
        self.backend.init_forward_metadata_capture_cuda_graph(
            bs,
            torch.arange(bs, dtype=torch.int64),
            torch.ones(bs, dtype=torch.int32),
            _decode_forward_mode(),
            cache_group_ids=cache_group_ids,
        )
        return self.backend.cuda_graph_decode_metadata[bs]

    def _replay(self, bs, block_tables=None):
        torch = self.torch
        kwargs = {}
        if block_tables is not None:
            kwargs["block_tables"] = block_tables
        self.backend.refresh_decode_metadata(
            bs,
            bs,
            torch.arange(MAX_BS, dtype=torch.int64),
            torch.ones(MAX_BS, dtype=torch.int32),
            forward_mode=_decode_forward_mode(),
            page_table=torch.zeros((MAX_BS, MAX_NUM_PAGES), dtype=torch.int32),
            for_graph_replay=True,
            **kwargs,
        )


_GROUP_IDS = ("sliding_attention", "full_attention")


class BackendCaptureGroupTest(_BackendCase):
    def test_page_tables_none_without_group_ids(self):
        metadata = self._capture(2)
        self.assertIsNone(metadata.page_tables)
        self.assertEqual(self.backend.cuda_graph_page_tables, {})

    def test_single_table_capture_keeps_page_table(self):
        # Single-table capture: page_table stays a live slice of the
        # persistent buffer (replay fills it via the gather path).
        metadata = self._capture(2)
        self.assertIsNotNone(metadata.page_table)
        self.assertEqual(tuple(metadata.page_table.shape), (2, MAX_NUM_PAGES))
        self.assertEqual(
            metadata.page_table.data_ptr(),
            self.backend.cuda_graph_page_table.data_ptr(),
        )

    def test_with_dflash_block_decode_asserts(self):
        self.backend.spec_num_tokens = 2
        self.backend.draft_block_decode = True
        torch = self.torch
        self.backend.cuda_graph_page_table = torch.zeros(
            (MAX_BS * 2, MAX_NUM_PAGES), dtype=torch.int32
        )
        self.backend.cuda_graph_seq_lens = torch.zeros(MAX_BS * 2, dtype=torch.int32)
        with self.assertRaisesRegex(AssertionError, "DFLASH"):
            self._capture(2, _GROUP_IDS)


class BackendConsumedGroupTablesTest(_BackendCase):
    """Positive claim: a backend keeps exactly the delivered groups whose
    family it declared in ``cache_consumer_families``. family="state" groups
    (GDN/mamba pages) must never reach MHA's flat buffers, table copies, or
    write-loc math; the hybrid router still hands the FULL dict to the mamba
    backend (see test_gdn_state_paging)."""

    _HYBRID_IDS = ("full_attention", "linear_attention")
    _FAMILIES = {"full_attention": "history", "linear_attention": "state"}

    def setUp(self):
        super().setUp()
        self.backend._geometry = CacheGroupGeometry(
            granularities=dict(self.backend._geometry.granularities),
            families=dict(self._FAMILIES),
            state_group_ids=frozenset({"linear_attention"}),
        )

    def test_filter_keeps_declared_families_only(self):
        torch = self.torch
        tables = {
            "full_attention": torch.ones((1, 1), dtype=torch.int32),
            "linear_attention": torch.ones((1, 1), dtype=torch.int32),
        }
        kept = self.backend._consumed_group_tables(tables)
        self.assertEqual(set(kept), {"full_attention"})
        self.assertEqual(
            self.backend._consumed_group_ids(), frozenset({"full_attention"})
        )

    def test_filter_unpublished_group_raises(self):
        torch = self.torch
        with self.assertRaisesRegex(RuntimeError, "never published"):
            self.backend._consumed_group_tables(
                {"mystery": torch.ones((1, 1), dtype=torch.int32)}
            )

    def test_filter_all_foreign_or_empty_returns_none(self):
        torch = self.torch
        self.assertIsNone(
            self.backend._consumed_group_tables(
                {"linear_attention": torch.ones((1, 1), dtype=torch.int32)}
            )
        )
        self.assertIsNone(self.backend._consumed_group_tables(None))
        self.assertIsNone(self.backend._consumed_group_tables({}))

    def test_filter_owned_groups_ride_to_the_wrapper(self):
        torch = self.torch
        self.backend.engine_owned_group_ids = frozenset({"full_attention"})
        self.assertIsNone(
            self.backend._consumed_group_tables(
                {"full_attention": torch.ones((1, 1), dtype=torch.int32)}
            )
        )

    def test_filter_without_learned_families_passes_through(self):
        # Pre-contract pools (older draft paths) deliver tables the draft's
        # own geometry never learned; they pass through unfiltered.
        torch = self.torch
        self.backend._geometry = CacheGroupGeometry()
        tables = {"anything": torch.ones((1, 1), dtype=torch.int32)}
        self.assertEqual(set(self.backend._consumed_group_tables(tables)), {"anything"})

    def test_capture_state_only_yields_no_attention_metadata(self):
        metadata = self._capture(2, ("linear_attention",))
        self.assertIsNone(metadata.page_tables)
        self.assertIsNone(metadata.out_cache_locs)
        self.assertEqual(self.backend.cuda_graph_page_tables, {})

    def test_eager_decode_refresh_sheds_state_group(self):
        torch = self.torch
        forward_mode = SimpleNamespace(
            is_mixed=lambda: False,
            is_extend_or_mixed=lambda: False,
        )
        # Unified decode path: state groups never enter the stacked graph
        # buffers (learned at init), and the eager refresh (bs == actual_bs)
        # fills the same persistent buffers replay does.
        self.backend._geometry = CacheGroupGeometry(
            granularities={"full_attention": 2},
            families=dict(self._FAMILIES),
            state_group_ids=frozenset({"linear_attention"}),
        )
        self.backend._init_group_graph_buffers(MAX_BS)
        self.backend.refresh_decode_metadata(
            2,
            2,
            torch.arange(2, dtype=torch.int64),
            torch.tensor([3, 4], dtype=torch.int32),
            forward_mode=forward_mode,
            page_table=torch.zeros((MAX_BS, MAX_NUM_PAGES), dtype=torch.int32),
            block_tables={
                "full_attention": torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
                "linear_attention": torch.tensor([[0, 5], [0, 6]], dtype=torch.int32),
            },
        )
        metadata = self.backend.forward_decode_metadata
        self.assertEqual(set(metadata.page_tables), {"full_attention"})
        self.assertEqual(set(metadata.out_cache_locs), {"full_attention"})
        # seq_lens [3, 4], page_size 2 -> last pos 2, 3 -> page col 1 ->
        # pages 2, 4 -> locs 2*2+0=4, 4*2+1=9.
        self.assertEqual(metadata.out_cache_locs["full_attention"][:2].tolist(), [4, 9])


class BackendReplayNoGroupBuffersTest(_BackendCase):
    def test_replay_without_group_capture_is_a_contract_violation(self):
        # Every LCM pool publishes at least one history group, so the wrapper
        # always passes cache_group_ids at capture; a replay that finds no
        # per-group buffers means the contract was bypassed. The pre-LCM
        # single-table gather fallback is gone — fail loudly instead.
        self._capture(2)
        with self.assertRaisesRegex(RuntimeError, "published no cache groups"):
            self._replay(2)


if __name__ == "__main__":
    unittest.main()
