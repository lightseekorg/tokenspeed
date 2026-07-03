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


class CaptureFlatBlockTablesTest(_TorchCase):
    """Wrapper-side capture placeholders (keys from pool group specs)."""

    def setUp(self):
        super().setUp()
        from tokenspeed.runtime.execution.cuda_graph_wrapper import (
            CudaGraphWrapper,
        )

        self.capture = CudaGraphWrapper._capture_flat_block_tables

    def _wrapper(self, uses_flat=True):
        return SimpleNamespace(
            attn_backend=SimpleNamespace(
                uses_flat_cache_groups=uses_flat,
                max_num_pages=MAX_NUM_PAGES,
            ),
            device="cpu",
        )

    def _pool(self, group_ids):
        return SimpleNamespace(
            paged_cache_group_specs=tuple(
                SimpleNamespace(group_id=gid) for gid in group_ids
            )
        )

    def test_keys_shape_dtype(self):
        out = self.capture(
            self._wrapper(),
            3,
            self._pool(["sliding_attention", "full_attention"]),
        )
        self.assertEqual(set(out), {"sliding_attention", "full_attention"})
        for table in out.values():
            self.assertEqual(tuple(table.shape), (3, MAX_NUM_PAGES))
            self.assertEqual(table.dtype, self.torch.int32)

    def test_none_without_specs(self):
        self.assertIsNone(self.capture(self._wrapper(), 3, self._pool([])))

    def test_none_when_backend_not_flat(self):
        out = self.capture(
            self._wrapper(uses_flat=False), 3, self._pool(["full_attention"])
        )
        self.assertIsNone(out)


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
        backend.max_num_pages = MAX_NUM_PAGES
        backend.device = "cpu"
        backend.cuda_graph_decode_metadata = {}
        backend.cuda_graph_page_table = torch.zeros(
            (MAX_BS, MAX_NUM_PAGES), dtype=torch.int32
        )
        backend.cuda_graph_seq_lens = torch.zeros(MAX_BS, dtype=torch.int32)
        backend.cuda_graph_flat_page_tables = {}
        backend._cuda_graph_max_bs = MAX_BS
        self.backend = backend

    def _capture(self, bs, flat_block_tables=None):
        torch = self.torch
        kwargs = {}
        if flat_block_tables is not None:
            kwargs["flat_block_tables"] = flat_block_tables
        self.backend.init_forward_metadata_capture_cuda_graph(
            bs,
            torch.arange(bs, dtype=torch.int64),
            torch.ones(bs, dtype=torch.int32),
            _decode_forward_mode(),
            **kwargs,
        )
        return self.backend.cuda_graph_decode_metadata[bs]

    def _replay(self, bs, flat_block_tables=None):
        torch = self.torch
        kwargs = {}
        if flat_block_tables is not None:
            kwargs["flat_block_tables"] = flat_block_tables
        self.backend.init_forward_metadata_replay_cuda_graph(
            bs,
            torch.arange(MAX_BS, dtype=torch.int64),
            torch.ones(MAX_BS, dtype=torch.int32),
            torch.zeros((MAX_BS, MAX_NUM_PAGES), dtype=torch.int32),
            _decode_forward_mode(),
            **kwargs,
        )


class BackendCaptureFlatTest(_BackendCase):
    def _placeholders(self, bs):
        torch = self.torch
        return {
            gid: torch.zeros((bs, MAX_NUM_PAGES), dtype=torch.int32)
            for gid in ("sliding_attention", "full_attention")
        }

    def test_page_tables_none_without_flat_kwarg(self):
        metadata = self._capture(2)
        self.assertIsNone(metadata.page_tables)
        self.assertEqual(self.backend.cuda_graph_flat_page_tables, {})

    def test_allocates_persistent_buffers_and_views(self):
        bs = 2
        metadata = self._capture(bs, self._placeholders(bs))
        bufs = self.backend.cuda_graph_flat_page_tables
        self.assertEqual(
            set(bufs), {"sliding_attention", "full_attention"}
        )
        for gid, buf in bufs.items():
            self.assertEqual(tuple(buf.shape), (MAX_BS, MAX_NUM_PAGES))
            self.assertEqual(buf.dtype, self.torch.int32)
            view = metadata.page_tables[gid]
            self.assertEqual(tuple(view.shape), (bs, MAX_NUM_PAGES))
            # Pointer-fixing: metadata views alias the persistent buffer.
            self.assertEqual(view.data_ptr(), buf.data_ptr())

    def test_second_capture_reuses_buffers(self):
        first = self._capture(2, self._placeholders(2))
        bufs = dict(self.backend.cuda_graph_flat_page_tables)
        second = self._capture(4, self._placeholders(4))
        self.assertEqual(
            {g: b.data_ptr() for g, b in bufs.items()},
            {
                g: b.data_ptr()
                for g, b in self.backend.cuda_graph_flat_page_tables.items()
            },
        )
        self.assertIsNot(first, second)

    def test_flat_with_spec_decode_asserts(self):
        self.backend.spec_num_tokens = 2
        torch = self.torch
        self.backend.cuda_graph_page_table = torch.zeros(
            (MAX_BS * 2, MAX_NUM_PAGES), dtype=torch.int32
        )
        self.backend.cuda_graph_seq_lens = torch.zeros(
            MAX_BS * 2, dtype=torch.int32
        )
        with self.assertRaisesRegex(AssertionError, "spec_num_tokens"):
            self._capture(2, self._placeholders(2))


class BackendReplayFlatTest(_BackendCase):
    def setUp(self):
        super().setUp()
        # Capture first so persistent buffers exist (replay indexes them).
        torch = self.torch
        self._capture(
            2,
            {
                gid: torch.zeros((2, MAX_NUM_PAGES), dtype=torch.int32)
                for gid in ("sliding_attention", "full_attention")
            },
        )

    def test_copies_prefix_and_fills_tail_minus_one(self):
        torch = self.torch
        src = {
            # 0 = null hole (slid-out SWA page); cols narrower than buffer.
            "sliding_attention": torch.tensor(
                [[0, 3], [4, 5]], dtype=torch.int32
            ),
            "full_attention": torch.tensor(
                [[1, 2], [6, 7]], dtype=torch.int32
            ),
        }
        self._replay(2, src)
        for gid, expected in src.items():
            buf = self.backend.cuda_graph_flat_page_tables[gid]
            self.assertTrue((buf[:2, :2] == expected).all())
            self.assertTrue((buf[:2, 2:] == -1).all())
            # Rows beyond bs untouched (still capture-time zeros).
            self.assertTrue((buf[2:] == 0).all())

    def test_padded_replay_dummy_rows_land_on_page_zero(self):
        # Wrapper row-pads flat tables with 0 before a padded replay; after
        # the backend's column-tail fill_(-1), a dummy row is [0, ..., -1s].
        # Only col 0 is read for dummy rows (seq_lens=1) -> dummy page 0.
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
        self._replay(2, padded)
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
        self._replay(2, src)
        for gid in src:
            buf = self.backend.cuda_graph_flat_page_tables[gid]
            self.assertTrue((buf[:2] == 9).all())

    def test_overwide_src_asserts(self):
        torch = self.torch
        src = {
            "sliding_attention": torch.ones(
                (2, MAX_NUM_PAGES + 1), dtype=torch.int32
            )
        }
        with self.assertRaisesRegex(AssertionError, "cols"):
            self._replay(2, src)

    def test_underpadded_rows_assert(self):
        torch = self.torch
        src = {
            "sliding_attention": torch.ones((1, 2), dtype=torch.int32)
        }
        with self.assertRaisesRegex(AssertionError, "rows"):
            self._replay(2, src)

    def test_no_flat_kwarg_is_noop_on_buffers(self):
        before = {
            gid: buf.clone()
            for gid, buf in self.backend.cuda_graph_flat_page_tables.items()
        }
        self._replay(2)
        for gid, buf in self.backend.cuda_graph_flat_page_tables.items():
            self.assertTrue((buf == before[gid]).all())


if __name__ == "__main__":
    unittest.main()
