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

MAX_BS = 4
MAX_NUM_PAGES = 6


def _decode_forward_mode():
    return SimpleNamespace(is_extend_or_mixed=lambda: False)


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
        from tokenspeed.runtime.execution.cuda_graph_wrapper import (
            CudaGraphWrapper,
        )

        self.pad = CudaGraphWrapper._pad_block_tables_to_padded_bs

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
        from tokenspeed.runtime.execution.cuda_graph_wrapper import (
            CudaGraphWrapper,
        )

        self.group_ids = CudaGraphWrapper._cache_group_ids

    def _wrapper(self, uses_cache_groups=True):
        return SimpleNamespace(
            attn_backend=SimpleNamespace(uses_cache_groups=uses_cache_groups),
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

    def test_empty_when_backend_does_not_use_cache_groups(self):
        out = self.group_ids(
            self._wrapper(uses_cache_groups=False), self._pool(["full_attention"])
        )
        self.assertEqual(out, ())


class TreeMaskGraphRoutingTest(_TorchCase):
    """Tree-mask metadata must bypass graphs captured for causal decode."""

    def setUp(self):
        super().setUp()
        from tokenspeed.runtime.execution.cuda_graph_wrapper import (
            CudaGraphWrapper,
        )

        self.has_tree_mask = CudaGraphWrapper._has_custom_tree_mask

    def test_detects_only_mask_bearing_spec_info(self):
        torch = self.torch
        self.assertFalse(self.has_tree_mask(None))
        self.assertFalse(self.has_tree_mask(SimpleNamespace(custom_mask=None)))
        self.assertTrue(
            self.has_tree_mask(
                SimpleNamespace(custom_mask=torch.ones(1, dtype=torch.int8))
            )
        )


class DraftTreeMaskMetadataIsolationTest(_TorchCase):
    """Target verify masks must not become draft attention metadata."""

    def test_mixed_batch_drops_target_spec_info_from_draft_metadata(self):
        torch = self.torch
        from tokenspeed.runtime.execution.cuda_graph_wrapper import (
            CudaGraphWrapper,
        )
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

        captured = {"target": [], "draft": []}

        class FakeBackend:
            needs_group_block_tables = False

            def __init__(self, key):
                self.key = key

            def init_forward_metadata(self, *args, **kwargs):
                captured[self.key].append((args, kwargs))

        wrapper = object.__new__(CudaGraphWrapper)
        wrapper.attn_backend = FakeBackend("target")
        wrapper.draft_attn_backend = FakeBackend("draft")
        wrapper.max_tokens_per_req = 4
        wrapper.drafter = SimpleNamespace(
            cache_view=SimpleNamespace(table=torch.zeros((1, 1), dtype=torch.int32)),
            draft_seq_lens_buf=torch.zeros(2, dtype=torch.int32),
        )
        wrapper.use_v4_mtp_paged_metadata = False

        target_spec_info = SimpleNamespace(custom_mask=torch.ones(4, dtype=torch.int8))
        wrapper._init_forward_metadata(
            padded_bs=2,
            num_extends=1,
            req_pool_indices=torch.zeros(2, dtype=torch.int32),
            seq_lens=torch.tensor([21, 22], dtype=torch.int32),
            page_table=torch.zeros((1, 1), dtype=torch.int32),
            forward_mode=ForwardMode.MIXED,
            spec_info=target_spec_info,
        )

        self.assertIs(captured["target"][0][1]["spec_info"], target_spec_info)
        self.assertNotIn("spec_info", captured["draft"][0][1])


class DraftCacheGroupIdsTest(_TorchCase):
    """DFLASH owns an independent draft page table; EAGLE-style drafts use
    target cache-group tables at matching page ids."""

    def setUp(self):
        super().setUp()
        from tokenspeed.runtime.execution.cuda_graph_wrapper import (
            CudaGraphWrapper,
        )

        self.group_ids = CudaGraphWrapper._draft_cache_group_ids

    def _wrapper(
        self,
        *,
        draft_block_decode,
        families=("history",),
        reads_staged_draft_page_table=False,
    ):
        return SimpleNamespace(
            draft_attn_backend=SimpleNamespace(
                uses_cache_groups=True,
                draft_block_decode=draft_block_decode,
                reads_staged_draft_page_table=reads_staged_draft_page_table,
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

    def test_mla_draft_reads_staged_page_table(self):
        # MLA drafts consume only the batch-ordered staged draft page table, so
        # the wrapper must not dispatch per-group tables to them.
        self.assertEqual(
            self.group_ids(
                self._wrapper(
                    draft_block_decode=False,
                    reads_staged_draft_page_table=True,
                )
            ),
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
    """Call-site wiring: the real _init_replay_metadata must row-pad grouped
    tables with 0 (not the -1 default) before handing them to the backend."""

    def _run_replay(self, block_tables, padded_bs, actual_bs):
        torch = self.torch
        from tokenspeed.runtime.execution.cuda_graph_wrapper import (
            CudaGraphWrapper,
        )

        recorded = {}

        def record(bs, req_pool_indices, seq_lens, **kwargs):
            recorded["bs"] = bs
            recorded.update(kwargs)

        mock = SimpleNamespace(
            attn_backend=SimpleNamespace(
                uses_cache_groups=True,
                needs_group_block_tables=False,
                uses_padded_decode_token_mask=False,
                init_forward_metadata_replay_cuda_graph=record,
            ),
            draft_attn_backend=None,
            # Production helper, so the pinned pad_value is the real one.
            _pad_block_tables_to_padded_bs=(
                CudaGraphWrapper._pad_block_tables_to_padded_bs
            ),
        )
        CudaGraphWrapper._init_replay_metadata(
            mock,
            padded_bs,
            actual_bs,
            torch.arange(padded_bs, dtype=torch.int64),
            torch.ones(padded_bs, dtype=torch.int32),
            torch.zeros((MAX_BS, MAX_NUM_PAGES), dtype=torch.int32),
            _decode_forward_mode(),
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
        from tokenspeed.runtime.execution.cuda_graph_wrapper import (
            CudaGraphWrapper,
        )

        draft_call = {}

        def record_draft(bs, req_pool_indices, seq_lens, **kwargs):
            draft_call.update(kwargs)

        mock = SimpleNamespace(
            attn_backend=SimpleNamespace(
                uses_cache_groups=False,
                needs_group_block_tables=False,
                uses_padded_decode_token_mask=False,
                init_forward_metadata_replay_cuda_graph=lambda *args, **kwargs: None,
            ),
            draft_attn_backend=SimpleNamespace(
                uses_cache_groups=True,
                needs_group_block_tables=False,
                uses_padded_decode_token_mask=False,
                init_forward_metadata_replay_cuda_graph=record_draft,
            ),
            drafter=SimpleNamespace(
                draft_seq_lens_buf=torch.zeros(2, dtype=torch.int32),
                cache_view=SimpleNamespace(
                    table=torch.zeros((2, MAX_NUM_PAGES), dtype=torch.int32)
                ),
            ),
            _draft_group_tables=lambda tables: tables,
            _pad_block_tables_to_padded_bs=(
                CudaGraphWrapper._pad_block_tables_to_padded_bs
            ),
        )
        tables = {
            "full_attention": torch.tensor([[3, 4]], dtype=torch.int32),
        }

        CudaGraphWrapper._init_replay_metadata(
            mock,
            padded_bs=2,
            actual_bs=1,
            req_pool_indices=torch.arange(2, dtype=torch.int64),
            seq_lens=torch.ones(2, dtype=torch.int32),
            page_table=torch.zeros((2, MAX_NUM_PAGES), dtype=torch.int32),
            forward_mode=_decode_forward_mode(),
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
        from tokenspeed.runtime.execution.cuda_graph_wrapper import (
            CudaGraphWrapper,
        )

        calls = {}

        def record_target(bs, req_pool_indices, seq_lens, **kwargs):
            calls["target"] = kwargs["block_tables"]

        def record_draft(bs, req_pool_indices, seq_lens, **kwargs):
            calls["draft"] = kwargs["block_tables"]

        backend_contract = {
            "uses_cache_groups": True,
            "needs_group_block_tables": False,
            "uses_padded_decode_token_mask": True,
        }
        mock = SimpleNamespace(
            attn_backend=SimpleNamespace(
                **backend_contract,
                init_forward_metadata_replay_cuda_graph=record_target,
            ),
            draft_attn_backend=SimpleNamespace(
                **backend_contract,
                init_forward_metadata_replay_cuda_graph=record_draft,
            ),
            drafter=SimpleNamespace(
                draft_seq_lens_buf=torch.zeros(4, dtype=torch.int32),
                cache_view=SimpleNamespace(
                    table=torch.zeros((4, MAX_NUM_PAGES), dtype=torch.int32)
                ),
            ),
            _draft_group_tables=lambda tables: {
                "full_attention": tables["full_attention"]
            },
            _pad_block_tables_to_padded_bs=(
                CudaGraphWrapper._pad_block_tables_to_padded_bs
            ),
        )
        tables = {
            "full_attention": torch.arange(6, dtype=torch.int32).reshape(3, 2),
            "state": torch.ones((3, 2), dtype=torch.int32),
        }

        CudaGraphWrapper._init_replay_metadata(
            mock,
            padded_bs=4,
            actual_bs=3,
            req_pool_indices=torch.arange(4, dtype=torch.int64),
            seq_lens=torch.ones(4, dtype=torch.int32),
            page_table=torch.zeros((4, MAX_NUM_PAGES), dtype=torch.int32),
            forward_mode=_decode_forward_mode(),
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

    def _run_capture(self, bs, group_ids, uses_cache_groups=True):
        torch = self.torch
        from types import MethodType

        from tokenspeed.runtime.execution.cuda_graph_wrapper import (
            CudaGraphWrapper,
        )

        recorded = {}

        def record(bs, req_pool_indices, seq_lens, forward_mode, **kwargs):
            recorded["bs"] = bs
            recorded["kwargs"] = kwargs

        mock = SimpleNamespace(
            input_buffers=SimpleNamespace(
                has_mamba=False,
                req_pool_indices_buf=torch.arange(MAX_BS, dtype=torch.int64),
                seq_lens_buf=torch.ones(MAX_BS, dtype=torch.int32),
            ),
            attn_backend=SimpleNamespace(
                needs_group_block_tables=False,
                uses_cache_groups=uses_cache_groups,
                init_forward_metadata_capture_cuda_graph=record,
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
        )
        mock._cache_group_ids = MethodType(CudaGraphWrapper._cache_group_ids, mock)
        CudaGraphWrapper._init_capture_metadata(mock, bs)
        return recorded

    def test_capture_passes_group_ids_from_pool_specs(self):
        recorded = self._run_capture(2, ["sliding_attention", "full_attention"])
        self.assertEqual(recorded["bs"], 2)
        self.assertEqual(
            recorded["kwargs"]["cache_group_ids"],
            ("sliding_attention", "full_attention"),
        )

    def test_capture_omits_group_ids_when_backend_does_not_use_cache_groups(self):
        recorded = self._run_capture(
            2, ["sliding_attention", "full_attention"], uses_cache_groups=False
        )
        self.assertNotIn("cache_group_ids", recorded["kwargs"])

    def test_capture_omits_group_ids_without_specs(self):
        recorded = self._run_capture(2, [])
        self.assertNotIn("cache_group_ids", recorded["kwargs"])


class WrapperEagerGroupGuardTest(_TorchCase):
    """Eager parity guard: a multi-group pool and group-aware backend
    must not reach the backend's single-table fallback without tables."""

    def _call(self, group_ids, block_tables=None):
        torch = self.torch
        from tokenspeed.runtime.execution.cuda_graph_wrapper import (
            CudaGraphWrapper,
        )
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

        calls = {}

        def init_forward_metadata(*args, **kwargs):
            calls["init_kwargs"] = kwargs

        mock = SimpleNamespace(
            input_buffers=SimpleNamespace(
                seq_lens_buf=torch.ones(MAX_BS, dtype=torch.int32),
                req_pool_indices_buf=torch.arange(MAX_BS, dtype=torch.int64),
            ),
            config=SimpleNamespace(),
            attn_backend=SimpleNamespace(
                uses_cache_groups=True,
                needs_group_block_tables=False,
            ),
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
        mock._any_backend_uses_cache_groups = (
            lambda: CudaGraphWrapper._any_backend_uses_cache_groups(mock)
        )
        ctx = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            num_extends=2,
            global_num_tokens=None,
            all_decode_or_idle=False,
            capture_hidden_mode=None,
        )
        CudaGraphWrapper.__call__(
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


class IdleBlockTablesTest(_TorchCase):
    """bs==0 idle replay tables: one col-0 page-0 entry per dummy row."""

    def setUp(self):
        super().setUp()
        from tokenspeed.runtime.execution.cuda_graph_wrapper import (
            CudaGraphWrapper,
        )

        self.idle = CudaGraphWrapper._idle_block_tables

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
        backend.state_group_ids = frozenset()
        backend.group_block_granularities = {}
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
        backend.cuda_graph_page_tables = {}
        backend.cuda_graph_out_cache_locs = {}
        backend._cuda_graph_max_bs = MAX_BS
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
        self.backend.init_forward_metadata_replay_cuda_graph(
            bs,
            torch.arange(MAX_BS, dtype=torch.int64),
            torch.ones(MAX_BS, dtype=torch.int32),
            torch.zeros((MAX_BS, MAX_NUM_PAGES), dtype=torch.int32),
            _decode_forward_mode(),
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


class BackendStateGroupShedTest(_BackendCase):
    """family="state" groups (GDN/mamba pages) must never reach MHA's flat
    buffers, table copies, or write-loc math; the hybrid router still hands
    the FULL dict to the mamba backend (see test_gdn_state_paging)."""

    _HYBRID_IDS = ("full_attention", "linear_attention")

    def setUp(self):
        super().setUp()
        self.backend.state_group_ids = frozenset({"linear_attention"})

    def test_capture_state_only_yields_no_attention_metadata(self):
        metadata = self._capture(2, ("linear_attention",))
        self.assertIsNone(metadata.page_tables)
        self.assertIsNone(metadata.out_cache_locs)
        self.assertEqual(self.backend.cuda_graph_page_tables, {})

    def test_eager_decode_metadata_sheds_state_group(self):
        torch = self.torch
        forward_mode = SimpleNamespace(
            is_mixed=lambda: False,
            is_extend_or_mixed=lambda: False,
        )
        self.backend.init_forward_metadata(
            bs=2,
            num_extends=0,
            req_pool_indices=torch.arange(2, dtype=torch.int64),
            seq_lens=torch.tensor([3, 4], dtype=torch.int32),
            page_table=torch.zeros((MAX_BS, MAX_NUM_PAGES), dtype=torch.int32),
            forward_mode=forward_mode,
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
        self.assertEqual(metadata.out_cache_locs["full_attention"].tolist(), [4, 9])


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
