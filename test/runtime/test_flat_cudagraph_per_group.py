"""M10: flat per-group CUDA-graph pad / capture / replay core-logic tests.

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
        # Radix/V4 path keeps -1 dummy rows: the backend masks dummy tokens
        # via is_valid_token before any block-table read.
        tables = self._tables()
        out = self.pad(tables, actual_bs=2, padded_bs=4)
        for gid, src in tables.items():
            self.assertEqual(tuple(out[gid].shape), (4, 3))
            self.assertTrue((out[gid][:2] == src).all())
            self.assertTrue((out[gid][2:] == -1).all())

    def test_flat_pads_tail_rows_with_zero(self):
        # Flat path passes pad_value=0: dummy rows replay with seq_lens=1 and
        # ARE dereferenced, so they must land on the zero-init dummy page 0.
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


class FlatCacheGroupIdsTest(_TorchCase):
    """Wrapper-side capture contract: group ids only, no fabricated tensors."""

    def setUp(self):
        super().setUp()
        from tokenspeed.runtime.execution.cuda_graph_wrapper import (
            CudaGraphWrapper,
        )

        self.group_ids = CudaGraphWrapper._resolve_flat_cache_group_ids

    def _wrapper(self, uses_flat=True):
        return SimpleNamespace(
            attn_backend=SimpleNamespace(uses_flat_cache_groups=uses_flat),
        )

    def _pool(self, group_ids):
        return SimpleNamespace(
            paged_cache_group_specs=tuple(
                SimpleNamespace(group_id=gid) for gid in group_ids
            )
        )

    def test_ids_in_spec_order(self):
        wrapper = self._wrapper()
        out = self.group_ids(
            self._pool(["sliding_attention", "full_attention"]),
            wrapper.attn_backend,
        )
        self.assertEqual(out, ("sliding_attention", "full_attention"))

    def test_empty_without_specs(self):
        wrapper = self._wrapper()
        self.assertEqual(self.group_ids(self._pool([]), wrapper.attn_backend), ())

    def test_empty_when_backend_not_flat(self):
        wrapper = self._wrapper(uses_flat=False)
        out = self.group_ids(self._pool(["full_attention"]), wrapper.attn_backend)
        self.assertEqual(out, ())


class OwnerLocalFlatGroupSelectionTest(_TorchCase):
    """Target and draft consume their own plan views from the scheduler union."""

    def setUp(self):
        super().setUp()
        from tokenspeed.runtime.execution.cuda_graph_wrapper import (
            CudaGraphWrapper,
        )

        self.wrapper_type = CudaGraphWrapper

    @staticmethod
    def _pool(*group_ids):
        return SimpleNamespace(
            paged_cache_group_specs=tuple(
                SimpleNamespace(group_id=group_id) for group_id in group_ids
            )
        )

    def _wrapper(self):
        return SimpleNamespace(
            _target_flat_group_ids=("target.history", "target.state"),
            _draft_flat_group_ids=("draft.history", "draft.state", "draft.only"),
        )

    def test_draft_ids_come_from_draft_owner_view_including_state(self):
        wrapper = self._wrapper()
        self.assertEqual(
            wrapper._draft_flat_group_ids,
            ("draft.history", "draft.state", "draft.only"),
        )

    def test_target_and_draft_subset_the_scheduler_union_independently(self):
        wrapper = self._wrapper()
        scheduler_union = {
            "target.history": object(),
            "target.state": object(),
            "draft.history": object(),
            "draft.state": object(),
            "draft.only": object(),
        }
        scheduler_bases = {group_id: object() for group_id in scheduler_union}
        target, target_bases = self.wrapper_type._owner_flat_inputs(
            scheduler_union,
            scheduler_bases,
            wrapper._target_flat_group_ids,
            owner="target",
        )
        draft, draft_bases = self.wrapper_type._owner_flat_inputs(
            scheduler_union,
            scheduler_bases,
            wrapper._draft_flat_group_ids,
            owner="draft",
        )
        self.assertEqual(set(target), {"target.history", "target.state"})
        self.assertEqual(set(target_bases), set(target))
        self.assertEqual(set(draft), {"draft.history", "draft.state", "draft.only"})
        self.assertEqual(set(draft_bases), set(draft))
        for group_id in target:
            self.assertIs(target[group_id], scheduler_union[group_id])
            self.assertIs(target_bases[group_id], scheduler_bases[group_id])

    def test_missing_owner_group_fails_closed(self):
        wrapper = self._wrapper()
        with self.assertRaisesRegex(RuntimeError, "draft.*missing"):
            self.wrapper_type._owner_flat_inputs(
                {
                    "target.history": object(),
                    "target.state": object(),
                    "draft.history": object(),
                },
                {"draft.history": object()},
                wrapper._draft_flat_group_ids,
                owner="draft",
            )


class FlatGraphPlanAccountingTest(_TorchCase):
    """CPU tensors exercise plan binding and actual byte/shape reconciliation."""

    @staticmethod
    def _plan():
        cols = {
            "target": {"target.history": 3, "shared.state": 2},
            "draft": {"draft.history": 4},
        }
        rows = {"target": 2, "draft": 2}
        return SimpleNamespace(
            graph_capture_cols_by_group=lambda owner: dict(cols[owner]),
            graph_batch_rows=lambda owner: rows[owner],
        )

    @staticmethod
    def _pool(plan, *, owner, cols):
        return SimpleNamespace(
            flat_memory_plan=plan,
            cache_owner=owner,
            flat_capture_cols_by_group=dict(cols),
            paged_cache_group_specs=tuple(
                SimpleNamespace(group_id=group_id) for group_id in cols
            ),
        )

    def test_wrapper_passes_owner_local_plan_cols_and_rows_to_backend(self):
        from tokenspeed.runtime.execution.cuda_graph_wrapper import CudaGraphWrapper

        plan = self._plan()
        pool = self._pool(
            plan,
            owner="target",
            cols=plan.graph_capture_cols_by_group("target"),
        )
        kwargs = CudaGraphWrapper._flat_graph_plan_init_kwargs(
            pool,
            owner="target",
            max_bs=2,
        )
        self.assertEqual(
            kwargs["flat_capture_cols_by_group"],
            {"target.history": 3, "shared.state": 2},
        )
        self.assertEqual(kwargs["flat_graph_batch_rows"], 2)

    def test_wrapper_rejects_pool_capture_width_drift(self):
        from tokenspeed.runtime.execution.cuda_graph_wrapper import CudaGraphWrapper

        plan = self._plan()
        pool = self._pool(
            plan,
            owner="target",
            cols={"target.history": 2, "shared.state": 2},
        )
        with self.assertRaisesRegex(RuntimeError, "capture cols disagree"):
            CudaGraphWrapper._flat_graph_plan_init_kwargs(
                pool,
                owner="target",
                max_bs=2,
            )

    def test_actual_table_and_base_tensor_bytes_match_plan(self):
        from tokenspeed.runtime.execution.cuda_graph_wrapper import CudaGraphWrapper

        torch = self.torch
        plan = self._plan()
        cols = plan.graph_capture_cols_by_group("target")
        backend = SimpleNamespace(
            _cuda_graph_paged_cache_block_tables={
                group_id: torch.zeros((2, width), dtype=torch.int32)
                for group_id, width in cols.items()
            },
            _cuda_graph_paged_cache_base_offsets={
                group_id: torch.zeros((2,), dtype=torch.int32) for group_id in cols
            },
        )
        self.assertEqual(
            CudaGraphWrapper._actual_flat_graph_metadata_bytes(
                backend,
                plan,
                owner="target",
            ),
            4 * 2 * ((3 + 1) + (2 + 1)),
        )
        backend._cuda_graph_paged_cache_block_tables["target.history"] = torch.zeros(
            (2, 2), dtype=torch.int32
        )
        with self.assertRaisesRegex(RuntimeError, "shape disagrees"):
            CudaGraphWrapper._actual_flat_graph_metadata_bytes(
                backend,
                plan,
                owner="target",
            )


class WrapperReplayFlatTest(_TorchCase):
    """Call-site wiring: the real _init_replay_metadata must row-pad flat
    tables with 0 (not the -1 default) before handing them to the backend."""

    def _run_replay(
        self,
        flat_block_tables,
        flat_block_table_base_offsets,
        padded_bs,
        actual_bs,
    ):
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
                uses_flat_cache_groups=True,
                uses_paged_cache_groups=False,
                uses_padded_decode_token_mask=False,
                init_forward_metadata_replay_cuda_graph=record,
            ),
            draft_attn_backend=None,
            # Production helper, so the pinned pad_value is the real one.
            _pad_block_tables_to_padded_bs=(
                CudaGraphWrapper._pad_block_tables_to_padded_bs
            ),
            _pad_offsets_to_padded_bs=CudaGraphWrapper._pad_offsets_to_padded_bs,
            _validate_flat_table_base_offsets=(
                CudaGraphWrapper._validate_flat_table_base_offsets
            ),
            _target_flat_group_ids=tuple(flat_block_tables),
            _draft_flat_group_ids=(),
        )
        CudaGraphWrapper._init_replay_metadata(
            mock,
            padded_bs,
            actual_bs,
            torch.arange(padded_bs, dtype=torch.int64),
            torch.ones(padded_bs, dtype=torch.int32),
            torch.zeros((MAX_BS, MAX_NUM_PAGES), dtype=torch.int32),
            _decode_forward_mode(),
            flat_block_tables=flat_block_tables,
            flat_block_table_base_offsets=flat_block_table_base_offsets,
        )
        return recorded

    def test_flat_replay_path_pads_with_zero(self):
        torch = self.torch
        src = {
            "sliding_attention": torch.tensor([[3, 4], [5, 6]], dtype=torch.int32),
            "full_attention": torch.tensor([[7, 8], [9, 1]], dtype=torch.int32),
        }
        bases = {gid: torch.tensor([3, 4], dtype=torch.int32) for gid in src}
        recorded = self._run_replay(src, bases, padded_bs=4, actual_bs=2)
        self.assertEqual(recorded["bs"], 4)
        out = recorded["flat_block_tables"]
        self.assertEqual(set(out), set(src))
        for gid, table in out.items():
            self.assertEqual(tuple(table.shape), (4, 2))
            self.assertTrue((table[:2] == src[gid]).all())
            # Dummy rows must land on the zero-init dummy page 0, never -1:
            # they replay with seq_lens=1 and their col-0 IS dereferenced.
            self.assertTrue((table[2:] == 0).all())
        for gid, offsets in recorded["flat_block_table_base_offsets"].items():
            self.assertEqual(offsets[:2].tolist(), bases[gid].tolist())
            self.assertEqual(offsets[2:].tolist(), [0, 0])

    def test_flat_replay_path_noop_without_padding(self):
        torch = self.torch
        src = {"full_attention": torch.ones((2, 2), dtype=torch.int32)}
        bases = {"full_attention": torch.zeros(2, dtype=torch.int32)}
        recorded = self._run_replay(src, bases, padded_bs=2, actual_bs=2)
        self.assertIs(
            recorded["flat_block_tables"]["full_attention"],
            src["full_attention"],
        )
        self.assertIs(
            recorded["flat_block_table_base_offsets"]["full_attention"],
            bases["full_attention"],
        )


class WrapperCaptureFlatGroupIdsTest(_TorchCase):
    """Call-site wiring: the real _init_capture_metadata must derive
    flat_cache_group_ids from the pool's published specs and pass them to
    the backend capture hook."""

    def _run_capture(self, bs, group_ids, uses_flat=True):
        torch = self.torch

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
                uses_paged_cache_groups=False,
                uses_flat_cache_groups=uses_flat,
                init_forward_metadata_capture_cuda_graph=record,
            ),
            token_to_kv_pool=SimpleNamespace(
                paged_cache_group_specs=tuple(
                    SimpleNamespace(group_id=gid) for gid in group_ids
                )
            ),
            drafter=None,
            use_target_verify_forward_mode=False,
            draft_attn_backend=None,
            _target_flat_group_ids=(tuple(group_ids) if uses_flat else ()),
            _draft_flat_group_ids=(),
        )
        CudaGraphWrapper._init_capture_metadata(mock, bs)
        return recorded

    def test_capture_passes_group_ids_from_pool_specs(self):
        recorded = self._run_capture(2, ["sliding_attention", "full_attention"])
        self.assertEqual(recorded["bs"], 2)
        self.assertEqual(
            recorded["kwargs"]["flat_cache_group_ids"],
            ("sliding_attention", "full_attention"),
        )

    def test_capture_omits_group_ids_when_backend_not_flat(self):
        recorded = self._run_capture(
            2, ["sliding_attention", "full_attention"], uses_flat=False
        )
        self.assertNotIn("flat_cache_group_ids", recorded["kwargs"])

    def test_capture_omits_group_ids_without_specs(self):
        recorded = self._run_capture(2, [])
        self.assertNotIn("flat_cache_group_ids", recorded["kwargs"])


class PackedSpecMetadataRoutingTest(_TorchCase):
    """Packed decode shape must not depend on radix-vs-flat table routing."""

    @staticmethod
    def _flat_pool(*group_ids):
        return SimpleNamespace(
            flat_memory_plan=object(),
            paged_cache_group_specs=tuple(
                SimpleNamespace(group_id=group_id) for group_id in group_ids
            ),
        )

    @staticmethod
    def _flat_backend(record):
        def capture(bs, req_pool_indices, seq_lens, forward_mode, **kwargs):
            record["capture"] = kwargs

        def replay(bs, req_pool_indices, seq_lens, **kwargs):
            record["replay"] = kwargs

        return SimpleNamespace(
            uses_flat_cache_groups=True,
            uses_paged_cache_groups=True,
            uses_padded_decode_token_mask=False,
            init_forward_metadata_capture_cuda_graph=capture,
            init_forward_metadata_replay_cuda_graph=replay,
        )

    def test_flat_target_and_draft_capture_receive_packed_num_tokens(self):
        torch = self.torch
        from tokenspeed.runtime.execution.cuda_graph_wrapper import CudaGraphWrapper

        target_calls = {}
        draft_calls = {}
        mock = SimpleNamespace(
            input_buffers=SimpleNamespace(
                has_mamba=False,
                req_pool_indices_buf=torch.arange(MAX_BS, dtype=torch.int64),
                seq_lens_buf=torch.ones(MAX_BS, dtype=torch.int32),
            ),
            attn_backend=self._flat_backend(target_calls),
            token_to_kv_pool=self._flat_pool("target.history", "shared.state"),
            draft_attn_backend=self._flat_backend(draft_calls),
            draft_token_to_kv_pool=self._flat_pool("draft.history", "shared.state"),
            drafter=object(),
            max_tokens_per_req=3,
            _target_flat_group_ids=("target.history", "shared.state"),
            _draft_flat_group_ids=("draft.history", "shared.state"),
        )

        CudaGraphWrapper._init_capture_metadata(mock, 2)

        self.assertEqual(target_calls["capture"]["num_tokens"], 6)
        self.assertEqual(draft_calls["capture"]["num_tokens"], 6)
        self.assertEqual(
            target_calls["capture"]["flat_cache_group_ids"],
            ("target.history", "shared.state"),
        )
        self.assertEqual(
            draft_calls["capture"]["flat_cache_group_ids"],
            ("draft.history", "shared.state"),
        )

    def test_flat_target_and_draft_replay_receive_packed_num_tokens(self):
        torch = self.torch
        from tokenspeed.runtime.execution.cuda_graph_wrapper import CudaGraphWrapper

        target_calls = {}
        draft_calls = {}
        target_pool = self._flat_pool("target.history", "shared.state")
        draft_pool = self._flat_pool("draft.history", "shared.state")
        draft_req_to_page = torch.zeros((MAX_BS, MAX_NUM_PAGES), dtype=torch.int32)
        mock = SimpleNamespace(
            input_buffers=SimpleNamespace(flat_block_table_staging=None),
            attn_backend=self._flat_backend(target_calls),
            token_to_kv_pool=target_pool,
            draft_attn_backend=self._flat_backend(draft_calls),
            draft_token_to_kv_pool=draft_pool,
            drafter=SimpleNamespace(
                draft_seq_lens_buf=torch.zeros(MAX_BS, dtype=torch.int32),
                req_to_page=draft_req_to_page,
            ),
            max_tokens_per_req=3,
            _pad_block_tables_to_padded_bs=(
                CudaGraphWrapper._pad_block_tables_to_padded_bs
            ),
            _pad_offsets_to_padded_bs=CudaGraphWrapper._pad_offsets_to_padded_bs,
            _validate_flat_table_base_offsets=(
                CudaGraphWrapper._validate_flat_table_base_offsets
            ),
            _target_flat_group_ids=("target.history", "shared.state"),
            _draft_flat_group_ids=("draft.history", "shared.state"),
        )
        flat_tables = {
            group_id: torch.ones((2, 2), dtype=torch.int32)
            for group_id in ("target.history", "draft.history", "shared.state")
        }
        flat_bases = {
            group_id: torch.zeros(2, dtype=torch.int32) for group_id in flat_tables
        }

        CudaGraphWrapper._init_replay_metadata(
            mock,
            2,
            2,
            torch.arange(2, dtype=torch.int64),
            torch.ones(2, dtype=torch.int32),
            torch.zeros((MAX_BS, MAX_NUM_PAGES), dtype=torch.int32),
            _decode_forward_mode(),
            flat_block_tables=flat_tables,
            flat_block_table_base_offsets=flat_bases,
        )

        self.assertEqual(target_calls["replay"]["num_tokens"], 6)
        self.assertEqual(draft_calls["replay"]["num_tokens"], 6)
        self.assertEqual(
            set(target_calls["replay"]["flat_block_tables"]),
            {"target.history", "shared.state"},
        )
        self.assertEqual(
            set(draft_calls["replay"]["flat_block_tables"]),
            {"draft.history", "shared.state"},
        )


class WrapperEagerFlatGuardTest(_TorchCase):
    """Eager parity guard: a multi-group flat pool + flat-consuming backend
    must not reach the backend's single-table fallback without tables."""

    def _call(
        self,
        group_ids,
        flat_block_tables=None,
        flat_block_table_base_offsets=None,
    ):
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
                uses_flat_cache_groups=True,
                uses_paged_cache_groups=False,
            ),
            token_to_kv_pool=SimpleNamespace(
                paged_cache_group_specs=tuple(
                    SimpleNamespace(group_id=gid) for gid in group_ids
                )
            ),
            _target_flat_group_ids=tuple(group_ids),
            _draft_flat_group_ids=(),
            drafter=None,
            _can_use_graph=lambda bs, ctx: False,
            _validate_flat_table_base_offsets=(
                CudaGraphWrapper._validate_flat_table_base_offsets
            ),
            _init_forward_metadata=init_forward_metadata,
            _forward_func=lambda **kwargs: (None, None, None),
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
            req_to_page=torch.zeros((MAX_BS, MAX_NUM_PAGES), dtype=torch.int32),
            flat_block_tables=flat_block_tables,
            flat_block_table_base_offsets=flat_block_table_base_offsets,
        )
        return calls

    def test_multi_group_eager_without_tables_raises(self):
        with self.assertRaisesRegex(RuntimeError, "flat_block_tables"):
            self._call(["sliding_attention", "full_attention"])

    def test_multi_group_eager_with_tables_passes(self):
        torch = self.torch
        tables = {
            "sliding_attention": torch.ones((2, 2), dtype=torch.int32),
            "full_attention": torch.ones((2, 2), dtype=torch.int32),
        }
        calls = self._call(
            ["sliding_attention", "full_attention"],
            flat_block_tables=tables,
            flat_block_table_base_offsets={
                gid: torch.zeros(2, dtype=torch.int32) for gid in tables
            },
        )
        self.assertIs(calls["init_kwargs"]["flat_block_tables"], tables)
        self.assertEqual(
            set(calls["init_kwargs"]["flat_block_table_base_offsets"]),
            set(tables),
        )

    def test_single_group_eager_without_tables_falls_back(self):
        # Documented fallback: with one published group the backend's single
        # table IS that group's table, so no tables are required.
        calls = self._call(["full_attention"])
        self.assertIsNone(calls["init_kwargs"]["flat_block_tables"])


class V4DualSourceRoutingTest(_TorchCase):
    """V4 advertises both ABIs; the pool plan, not the capability flag,
    selects exactly one source for each runtime instance."""

    def _call(self, *, flat_plan, flat_tables=None, flat_bases=None):
        torch = self.torch
        from tokenspeed.runtime.execution.cuda_graph_wrapper import (
            CudaGraphWrapper,
        )
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

        calls = {}

        def init_forward_metadata(*args, **kwargs):
            calls["kwargs"] = kwargs

        paged_tables = {
            "history": torch.ones((2, 2), dtype=torch.int32),
            "state": torch.ones((2, 2), dtype=torch.int32),
        }
        paged_bases = {
            group_id: torch.zeros(2, dtype=torch.int32) for group_id in paged_tables
        }
        pool = SimpleNamespace(
            paged_cache_group_specs=tuple(
                SimpleNamespace(group_id=group_id) for group_id in paged_tables
            ),
            flat_memory_plan=flat_plan,
        )
        mock = SimpleNamespace(
            input_buffers=SimpleNamespace(
                seq_lens_buf=torch.ones(MAX_BS, dtype=torch.int32),
                req_pool_indices_buf=torch.arange(MAX_BS, dtype=torch.int64),
            ),
            config=SimpleNamespace(),
            attn_backend=SimpleNamespace(
                uses_flat_cache_groups=True,
                uses_paged_cache_groups=True,
            ),
            token_to_kv_pool=pool,
            _target_flat_group_ids=(
                tuple(paged_tables) if flat_plan is not None else ()
            ),
            _draft_flat_group_ids=(),
            drafter=None,
            _can_use_graph=lambda bs, ctx: False,
            _validate_flat_table_base_offsets=(
                CudaGraphWrapper._validate_flat_table_base_offsets
            ),
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
        CudaGraphWrapper.__call__(
            mock,
            bs=2,
            ctx=ctx,
            sampling_info=None,
            req_to_page=torch.zeros((MAX_BS, MAX_NUM_PAGES), dtype=torch.int32),
            paged_cache_block_tables=paged_tables,
            paged_cache_block_table_base_offsets=paged_bases,
            flat_block_tables=flat_tables,
            flat_block_table_base_offsets=flat_bases,
        )
        return calls["kwargs"], paged_tables, paged_bases

    def test_legacy_pool_keeps_radix_source_despite_flat_capability(self):
        kwargs, paged_tables, paged_bases = self._call(flat_plan=None)

        self.assertIs(kwargs["paged_cache_block_tables"], paged_tables)
        self.assertIs(kwargs["paged_cache_block_table_base_offsets"], paged_bases)
        self.assertIsNone(kwargs["flat_block_tables"])
        self.assertIsNone(kwargs["flat_block_table_base_offsets"])

    def test_flat_arena_pool_requires_and_selects_flat_source(self):
        torch = self.torch
        with self.assertRaisesRegex(RuntimeError, "flat_block_tables"):
            self._call(flat_plan=object())

        flat_tables = {
            "history": torch.ones((2, 2), dtype=torch.int32),
            "state": torch.ones((2, 2), dtype=torch.int32),
        }
        flat_bases = {
            group_id: torch.zeros(2, dtype=torch.int32) for group_id in flat_tables
        }
        kwargs, _, _ = self._call(
            flat_plan=object(), flat_tables=flat_tables, flat_bases=flat_bases
        )
        self.assertIs(kwargs["flat_block_tables"], flat_tables)
        self.assertIs(kwargs["flat_block_table_base_offsets"], flat_bases)
        self.assertIsNone(kwargs["paged_cache_block_tables"])
        self.assertIsNone(kwargs["paged_cache_block_table_base_offsets"])


class IdleFlatBlockTablesTest(_TorchCase):
    """bs==0 idle replay tables: one col-0 page-0 entry per dummy row."""

    def setUp(self):
        super().setUp()
        from tokenspeed.runtime.execution.cuda_graph_wrapper import (
            CudaGraphWrapper,
        )

        self.idle = CudaGraphWrapper._idle_flat_block_tables
        self.idle_bases = CudaGraphWrapper._idle_flat_block_table_base_offsets

    def _wrapper(self, group_ids):
        return SimpleNamespace(
            token_to_kv_pool=SimpleNamespace(
                paged_cache_group_specs=tuple(
                    SimpleNamespace(group_id=gid) for gid in group_ids
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

    def test_idle_tables_have_explicit_zero_bases(self):
        wrapper = self._wrapper(["sliding_attention", "full_attention"])
        tables = self.idle(wrapper, 3)
        bases = self.idle_bases(wrapper, 3, tables)
        self.assertEqual(set(bases), set(tables))
        for offsets in bases.values():
            self.assertEqual(offsets.dtype, self.torch.int32)
            self.assertEqual(offsets.tolist(), [0, 0, 0])


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
        backend.flat_state_group_ids = frozenset()
        backend.max_num_pages = MAX_NUM_PAGES
        backend.page_size = 2
        backend.device = "cpu"
        backend.cuda_graph_decode_metadata = {}
        backend.cuda_graph_page_table = torch.zeros(
            (MAX_BS, MAX_NUM_PAGES), dtype=torch.int32
        )
        # seq_lens 1 (never 0): flat replay recomputes write locs from these
        # (M11), and seq_len 0 would gather at position -1.
        backend.cuda_graph_seq_lens = torch.ones(MAX_BS, dtype=torch.int32)
        backend.cuda_graph_flat_page_tables = {}
        backend.cuda_graph_flat_block_table_base_offsets = {}
        backend.cuda_graph_flat_out_cache_locs = {}
        backend._cuda_graph_max_bs = MAX_BS
        self.backend = backend

    def _capture(self, bs, flat_cache_group_ids=()):
        torch = self.torch
        self.backend.init_forward_metadata_capture_cuda_graph(
            bs,
            torch.arange(bs, dtype=torch.int64),
            torch.ones(bs, dtype=torch.int32),
            _decode_forward_mode(),
            flat_cache_group_ids=flat_cache_group_ids,
        )
        return self.backend.cuda_graph_decode_metadata[bs]

    def _replay(
        self,
        bs,
        flat_block_tables=None,
        flat_block_table_base_offsets=None,
    ):
        torch = self.torch
        kwargs = {}
        if flat_block_tables is not None:
            kwargs["flat_block_tables"] = flat_block_tables
        if flat_block_table_base_offsets is not None:
            kwargs["flat_block_table_base_offsets"] = flat_block_table_base_offsets
        self.backend.init_forward_metadata_replay_cuda_graph(
            bs,
            torch.arange(MAX_BS, dtype=torch.int64),
            torch.ones(MAX_BS, dtype=torch.int32),
            torch.zeros((MAX_BS, MAX_NUM_PAGES), dtype=torch.int32),
            _decode_forward_mode(),
            **kwargs,
        )


_GROUP_IDS = ("sliding_attention", "full_attention")


class BackendCaptureFlatTest(_BackendCase):
    def test_page_tables_none_without_group_ids(self):
        metadata = self._capture(2)
        self.assertIsNone(metadata.page_tables)
        self.assertIsNone(metadata.block_table_base_offsets)
        self.assertEqual(self.backend.cuda_graph_flat_page_tables, {})

    def test_radix_capture_keeps_single_page_table(self):
        # Radix/single-table capture: page_table stays a live slice of the
        # persistent buffer (replay fills it via the gather path).
        metadata = self._capture(2)
        self.assertIsNotNone(metadata.page_table)
        self.assertEqual(tuple(metadata.page_table.shape), (2, MAX_NUM_PAGES))
        self.assertEqual(
            metadata.page_table.data_ptr(),
            self.backend.cuda_graph_page_table.data_ptr(),
        )

    def test_flat_capture_sheds_single_page_table(self):
        # Flat captures route reads through per-group tables and replay never
        # fills the radix single table: page_table must be None, never a
        # slice of the never-filled zero buffer.
        metadata = self._capture(2, _GROUP_IDS)
        self.assertIsNone(metadata.page_table)

    def test_allocates_persistent_buffers_and_views(self):
        bs = 2
        metadata = self._capture(bs, _GROUP_IDS)
        bufs = self.backend.cuda_graph_flat_page_tables
        self.assertEqual(set(bufs), set(_GROUP_IDS))
        for gid, buf in bufs.items():
            self.assertEqual(tuple(buf.shape), (MAX_BS, MAX_NUM_PAGES))
            self.assertEqual(buf.dtype, self.torch.int32)
            view = metadata.page_tables[gid]
            self.assertEqual(tuple(view.shape), (bs, MAX_NUM_PAGES))
            # Pointer-fixing: metadata views alias the persistent buffer.
            self.assertEqual(view.data_ptr(), buf.data_ptr())
            base_buf = self.backend.cuda_graph_flat_block_table_base_offsets[gid]
            base_view = metadata.block_table_base_offsets[gid]
            self.assertEqual(tuple(base_buf.shape), (MAX_BS,))
            self.assertEqual(base_buf.dtype, self.torch.int32)
            self.assertEqual(tuple(base_view.shape), (bs,))
            self.assertEqual(base_view.data_ptr(), base_buf.data_ptr())

    def test_second_capture_reuses_buffers(self):
        first = self._capture(2, _GROUP_IDS)
        bufs = dict(self.backend.cuda_graph_flat_page_tables)
        base_bufs = dict(self.backend.cuda_graph_flat_block_table_base_offsets)
        second = self._capture(4, _GROUP_IDS)
        self.assertEqual(
            {g: b.data_ptr() for g, b in bufs.items()},
            {
                g: b.data_ptr()
                for g, b in self.backend.cuda_graph_flat_page_tables.items()
            },
        )
        self.assertEqual(
            {g: b.data_ptr() for g, b in base_bufs.items()},
            {
                g: b.data_ptr()
                for g, b in (
                    self.backend.cuda_graph_flat_block_table_base_offsets.items()
                )
            },
        )
        self.assertIsNot(first, second)

    def test_flat_with_spec_verify_records_expanded_loc_views(self):
        # Verify (spec target) keeps [bs]-row tables but records [bs*N]
        # write-loc views (token-major), sized off the persistent buffers.
        self.backend.spec_num_tokens = 2
        torch = self.torch
        self.backend.cuda_graph_page_table = torch.zeros(
            (MAX_BS * 2, MAX_NUM_PAGES), dtype=torch.int32
        )
        self.backend.cuda_graph_seq_lens = torch.zeros(MAX_BS * 2, dtype=torch.int32)
        self._capture(2, _GROUP_IDS)
        meta = self.backend.forward_decode_metadata
        for gid in _GROUP_IDS:
            self.assertEqual(meta.page_tables[gid].shape[0], 2)
            self.assertEqual(meta.out_cache_locs[gid].shape[0], 2 * 2)

    def test_flat_with_dflash_block_decode_asserts(self):
        self.backend.spec_num_tokens = 2
        self.backend.draft_block_decode = True
        torch = self.torch
        self.backend.cuda_graph_page_table = torch.zeros(
            (MAX_BS * 2, MAX_NUM_PAGES), dtype=torch.int32
        )
        self.backend.cuda_graph_seq_lens = torch.zeros(MAX_BS * 2, dtype=torch.int32)
        with self.assertRaisesRegex(AssertionError, "DFLASH"):
            self._capture(2, _GROUP_IDS)


class BackendReplayFlatTest(_BackendCase):
    def setUp(self):
        super().setUp()
        # Capture first so persistent buffers exist (replay indexes them).
        self._capture(2, _GROUP_IDS)

    def test_copies_prefix_and_fills_tail_minus_one(self):
        torch = self.torch
        src = {
            # 0 = null hole (slid-out SWA page); cols narrower than buffer.
            "sliding_attention": torch.tensor([[0, 3], [4, 5]], dtype=torch.int32),
            "full_attention": torch.tensor([[1, 2], [6, 7]], dtype=torch.int32),
        }
        bases = {gid: torch.zeros(2, dtype=torch.int32) for gid in src}
        self._replay(2, src, bases)
        for gid, expected in src.items():
            buf = self.backend.cuda_graph_flat_page_tables[gid]
            self.assertTrue((buf[:2, :2] == expected).all())
            self.assertTrue((buf[:2, 2:] == -1).all())
            # Rows beyond bs untouched (still capture-time zeros).
            self.assertTrue((buf[2:] == 0).all())

    def test_padded_replay_dummy_rows_land_on_page_zero(self):
        # After the wrapper's 0-row-pad and the backend's column fill_(-1),
        # a dummy row reads only col 0 (seq_lens=1) -> dummy page 0.
        torch = self.torch
        from tokenspeed.runtime.execution.cuda_graph_wrapper import (
            CudaGraphWrapper,
        )

        src = {
            "sliding_attention": torch.tensor([[3, 4]], dtype=torch.int32),
            "full_attention": torch.tensor([[5, 6]], dtype=torch.int32),
        }
        padded = CudaGraphWrapper._pad_block_tables_to_padded_bs(
            src, actual_bs=1, padded_bs=2, pad_value=0
        )
        bases = {gid: torch.zeros(2, dtype=torch.int32) for gid in padded}
        self._replay(2, padded, bases)
        for gid, expected in src.items():
            buf = self.backend.cuda_graph_flat_page_tables[gid]
            self.assertTrue((buf[:1, :2] == expected).all())
            # Dummy row: col 0 must be a dereferenceable page (0), never -1.
            self.assertEqual(int(buf[1, 0]), 0)
            self.assertTrue((buf[1, :2] == 0).all())
            self.assertTrue((buf[:2, 2:] == -1).all())

    def test_full_width_src_leaves_no_tail(self):
        torch = self.torch
        src = {
            gid: torch.full((2, MAX_NUM_PAGES), 9, dtype=torch.int32)
            for gid in ("sliding_attention", "full_attention")
        }
        bases = {gid: torch.zeros(2, dtype=torch.int32) for gid in src}
        self._replay(2, src, bases)
        for gid in src:
            buf = self.backend.cuda_graph_flat_page_tables[gid]
            self.assertTrue((buf[:2] == 9).all())

    def test_overwide_src_asserts(self):
        # Both captured groups delivered (the missing-group guard runs
        # first); the overwide one trips the width assert.
        torch = self.torch
        src = {
            "sliding_attention": torch.ones((2, MAX_NUM_PAGES + 1), dtype=torch.int32),
            "full_attention": torch.ones((2, 2), dtype=torch.int32),
        }
        with self.assertRaisesRegex(AssertionError, "cols"):
            self._replay(
                2,
                src,
                {gid: torch.zeros(2, dtype=torch.int32) for gid in src},
            )

    def test_underpadded_rows_assert(self):
        torch = self.torch
        src = {
            "sliding_attention": torch.ones((1, 2), dtype=torch.int32),
            "full_attention": torch.ones((2, 2), dtype=torch.int32),
        }
        with self.assertRaisesRegex(RuntimeError, "rows"):
            self._replay(
                2,
                src,
                {
                    "sliding_attention": torch.zeros(1, dtype=torch.int32),
                    "full_attention": torch.zeros(2, dtype=torch.int32),
                },
            )

    def test_missing_tables_with_flat_buffers_raises(self):
        # A flat-captured graph replayed without tables must be loud, never
        # silently compute over stale/zero page tables.
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self._replay(2)

    def test_missing_tables_empty_dict_raises(self):
        with self.assertRaisesRegex(RuntimeError, "flat_block_tables"):
            self._replay(2, {})

    def test_missing_captured_group_raises(self):
        # Per-group hole: a non-empty dict lacking one captured group would
        # leave that group's buffer stale — must raise naming the group.
        torch = self.torch
        src = {"sliding_attention": torch.ones((2, 2), dtype=torch.int32)}
        bases = {"sliding_attention": torch.zeros(2, dtype=torch.int32)}
        with self.assertRaisesRegex(RuntimeError, "captured.*full_attention"):
            self._replay(2, src, bases)

    def test_table_base_key_mismatch_raises(self):
        torch = self.torch
        tables = {gid: torch.ones((2, 2), dtype=torch.int32) for gid in _GROUP_IDS}
        bases = {"sliding_attention": torch.zeros(2, dtype=torch.int32)}
        with self.assertRaisesRegex(RuntimeError, "table/base group mismatch"):
            self._replay(2, tables, bases)

    def test_base_row_mismatch_raises(self):
        torch = self.torch
        tables = {gid: torch.ones((2, 2), dtype=torch.int32) for gid in _GROUP_IDS}
        bases = {
            "sliding_attention": torch.zeros(2, dtype=torch.int32),
            "full_attention": torch.zeros(1, dtype=torch.int32),
        }
        with self.assertRaisesRegex(RuntimeError, "table rows.*base rows"):
            self._replay(2, tables, bases)

    def test_replay_refreshes_base_when_table_is_unchanged(self):
        torch = self.torch
        # Compact two-column rows. First replay's last logical page is 1;
        # second replay advances the base to 1 and the last logical page to 2.
        # Both resolve to local column 1 while proving the base buffer refreshes.
        tables = {
            "sliding_attention": torch.tensor([[7, 8], [9, 10]], dtype=torch.int32),
            "full_attention": torch.tensor([[11, 12], [13, 14]], dtype=torch.int32),
        }
        zero_bases = {gid: torch.zeros(2, dtype=torch.int32) for gid in tables}
        self.backend.cuda_graph_seq_lens[:2].copy_(
            torch.tensor([4, 4], dtype=torch.int32)
        )
        self._replay(2, tables, zero_bases)

        advanced_bases = {gid: torch.ones(2, dtype=torch.int32) for gid in tables}
        self.backend.cuda_graph_seq_lens[:2].copy_(
            torch.tensor([6, 6], dtype=torch.int32)
        )
        self._replay(2, tables, advanced_bases)

        for gid in tables:
            self.assertEqual(
                self.backend.cuda_graph_flat_block_table_base_offsets[gid][:2].tolist(),
                [1, 1],
            )
        self.assertEqual(
            self.backend.cuda_graph_flat_out_cache_locs["full_attention"][:2].tolist(),
            [25, 29],
        )

    def test_bs_zero_missing_tables_skips(self):
        # Documented bs==0 skip: buffers keep valid page-0/previous entries;
        # outputs are discarded.
        before = {
            gid: buf.clone()
            for gid, buf in self.backend.cuda_graph_flat_page_tables.items()
        }
        self._replay(0)
        for gid, buf in self.backend.cuda_graph_flat_page_tables.items():
            self.assertTrue((buf == before[gid]).all())


class BackendStateGroupShedTest(_BackendCase):
    """family="state" groups (GDN/mamba pages) must never reach MHA's flat
    buffers, table copies, or write-loc math; the hybrid router still hands
    the FULL dict to the mamba backend (see test_gdn_flat_state_paging)."""

    _HYBRID_IDS = ("full_attention", "linear_attention")

    def setUp(self):
        super().setUp()
        self.backend.flat_state_group_ids = frozenset({"linear_attention"})

    def test_init_cuda_graph_state_learns_state_ids_from_specs(self):
        torch = self.torch
        self.backend.init_cuda_graph_state(
            MAX_BS,
            torch.ones(MAX_BS, dtype=torch.int32),
            paged_cache_group_specs=(
                SimpleNamespace(group_id="full_attention", family="history"),
                SimpleNamespace(group_id="linear_attention", family="state"),
            ),
        )
        self.assertEqual(
            self.backend.flat_state_group_ids, frozenset({"linear_attention"})
        )

    def test_capture_buffers_exclude_state_group(self):
        metadata = self._capture(2, self._HYBRID_IDS)
        self.assertEqual(
            set(self.backend.cuda_graph_flat_page_tables), {"full_attention"}
        )
        self.assertEqual(
            set(self.backend.cuda_graph_flat_out_cache_locs), {"full_attention"}
        )
        self.assertEqual(
            set(self.backend.cuda_graph_flat_block_table_base_offsets),
            {"full_attention"},
        )
        self.assertEqual(set(metadata.page_tables), {"full_attention"})
        self.assertEqual(set(metadata.block_table_base_offsets), {"full_attention"})
        self.assertEqual(set(metadata.out_cache_locs), {"full_attention"})

    def test_capture_state_only_yields_no_flat_metadata(self):
        metadata = self._capture(2, ("linear_attention",))
        self.assertIsNone(metadata.page_tables)
        self.assertIsNone(metadata.block_table_base_offsets)
        self.assertIsNone(metadata.out_cache_locs)
        self.assertEqual(self.backend.cuda_graph_flat_page_tables, {})

    def test_replay_skips_state_group_delivery(self):
        torch = self.torch
        self._capture(2, self._HYBRID_IDS)
        src = {
            "full_attention": torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
            # Hole-heavy state table: MHA must not copy or derive locs
            # from it (and has no buffer for it).
            "linear_attention": torch.tensor([[0, 5], [0, 6]], dtype=torch.int32),
        }
        bases = {gid: torch.zeros(2, dtype=torch.int32) for gid in src}
        self._replay(2, src, bases)
        self.assertNotIn("linear_attention", self.backend.cuda_graph_flat_page_tables)
        buf = self.backend.cuda_graph_flat_page_tables["full_attention"]
        self.assertTrue((buf[:2, :2] == src["full_attention"]).all())

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
            req_to_page=torch.zeros((MAX_BS, MAX_NUM_PAGES), dtype=torch.int32),
            forward_mode=forward_mode,
            flat_block_tables={
                "full_attention": torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
                "linear_attention": torch.tensor([[0, 5], [0, 6]], dtype=torch.int32),
            },
            flat_block_table_base_offsets={
                "full_attention": torch.zeros(2, dtype=torch.int32),
                "linear_attention": torch.zeros(2, dtype=torch.int32),
            },
        )
        metadata = self.backend.forward_decode_metadata
        self.assertEqual(set(metadata.page_tables), {"full_attention"})
        self.assertEqual(set(metadata.block_table_base_offsets), {"full_attention"})
        self.assertEqual(set(metadata.out_cache_locs), {"full_attention"})
        # seq_lens [3, 4], page_size 2 -> last pos 2, 3 -> page col 1 ->
        # pages 2, 4 -> locs 2*2+0=4, 4*2+1=9.
        self.assertEqual(metadata.out_cache_locs["full_attention"].tolist(), [4, 9])


class BackendReplayNoFlatBuffersTest(_BackendCase):
    def _replay_with_recorded_gather(
        self,
        bs,
        flat_block_tables=None,
        flat_block_table_base_offsets=None,
    ):
        # The radix single-table fill is a GPU Triton kernel; record the call
        # instead of launching it on this test's CPU tensors.
        from unittest import mock

        import tokenspeed.runtime.layers.attention.backends.mha as mha_mod

        with mock.patch.object(mha_mod, "gather_page_table_with_padding") as gather:
            self._replay(bs, flat_block_tables, flat_block_table_base_offsets)
        return gather

    def test_replay_without_flat_capture_needs_no_tables(self):
        # No flat buffers captured (radix/single-table path): replay without
        # tables stays valid and fills the radix single table.
        self._capture(2)
        gather = self._replay_with_recorded_gather(2)
        gather.assert_called_once()
        self.assertEqual(self.backend.cuda_graph_flat_page_tables, {})

    def test_flat_replay_skips_radix_single_table_fill(self):
        # Flat captures read only the per-group buffers: filling the radix
        # single table would be dead work (see init_forward_metadata_replay).
        torch = self.torch
        self._capture(2, _GROUP_IDS)
        tables = {gid: torch.ones((2, 2), dtype=torch.int32) for gid in _GROUP_IDS}
        gather = self._replay_with_recorded_gather(
            2,
            tables,
            {gid: torch.zeros(2, dtype=torch.int32) for gid in tables},
        )
        gather.assert_not_called()


if __name__ == "__main__":
    unittest.main()
