"""M11: flat per-group KV write-location derivation tests.

Plain CPU tensors cover the pure loc-computation helpers, the
TOKENSPEED_FLAT_DEBUG write-invariant checker, eager metadata assembly, and
the out-cache-loc routing seams. CUDA covers persistent graph-buffer views and
the fused replay recomputation.
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

PAGE = 2
MAX_NUM_PAGES = 4
MAX_BS = 4


def _flat_specs(group_sizes):
    from tokenspeed.runtime.configs.paged_cache_spec import PagedCacheGroupSpec

    return tuple(
        PagedCacheGroupSpec(
            group_id=group_id,
            retention="full_history",
            rows_per_page=block_size,
            entry_stride_tokens=1,
            sliding_window_tokens=None,
        )
        for group_id, block_size in group_sizes.items()
    )


def _decode_forward_mode():
    return SimpleNamespace(
        is_mixed=lambda: False,
        is_extend_or_mixed=lambda: False,
    )


def _extend_forward_mode():
    return SimpleNamespace(
        is_mixed=lambda: False,
        is_extend_or_mixed=lambda: True,
    )


class _TorchCase(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch: {exc}")
        self.torch = torch


class _MHACase(_TorchCase):
    """Cases against MHAAttnBackend helpers on a minimally initialized instance."""

    def setUp(self):
        super().setUp()
        try:
            from tokenspeed.runtime.layers.attention.backends.mha import (
                MHAAttnBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs tokenspeed_kernel: {exc}")
        self.MHAAttnBackend = MHAAttnBackend
        self.backend = MHAAttnBackend.__new__(MHAAttnBackend)
        self.backend.page_size = PAGE
        self.backend._learn_flat_state_groups(
            _flat_specs(
                {
                    "sliding_attention": PAGE,
                    "full_attention": PAGE,
                    "bounded": PAGE,
                }
            )
        )


class ComputeFlatOutCacheLocsTest(_MHACase):
    def test_group_page_size_falls_back_only_without_flat_specs(self):
        backend = self.MHAAttnBackend.__new__(self.MHAAttnBackend)
        backend.page_size = PAGE

        cases = (
            ("empty group", "", PAGE),
            ("radix layer label", "sliding_attention", PAGE),
        )
        for name, group_id, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(backend._group_page_size(group_id), expected)

        backend._learn_flat_state_groups(_flat_specs({"full_attention": 4}))
        self.assertEqual(backend._group_page_size("full_attention"), 4)
        for group_id in ("", "sliding_attention"):
            with self.subTest(unknown_group_id=group_id), self.assertRaisesRegex(
                RuntimeError, "absent from.*cache specs"
            ):
                backend._group_page_size(group_id)

        from tokenspeed.runtime.configs.paged_cache_spec import PagedCacheGroupSpec

        state_only_backend = self.MHAAttnBackend.__new__(self.MHAAttnBackend)
        state_only_backend.page_size = PAGE
        state_only_backend._learn_flat_state_groups(
            (
                PagedCacheGroupSpec(
                    group_id="state",
                    retention="sliding_window",
                    rows_per_page=4,
                    entry_stride_tokens=1,
                    sliding_window_tokens=8,
                    family="state",
                    prefix_role="continuation_state",
                    table_layout="bounded_window",
                ),
            )
        )
        for group_id in ("", "unknown"):
            with self.subTest(state_only_group_id=group_id), self.assertRaisesRegex(
                RuntimeError, "absent from.*cache specs"
            ):
                state_only_backend._group_page_size(group_id)

    def test_decode_locs_formula(self):
        torch = self.torch
        # 2 reqs, page_size=2. r0: seq_len 5 -> pos 4 -> page_idx 2, off 0.
        # r1: seq_len 4 -> pos 3 -> page_idx 1, off 1.
        # sliding table has a front hole (page 0 slot) -- never touched.
        tables = {
            "sliding_attention": torch.tensor(
                [[0, 5, 7, -1], [0, 6, -1, -1]], dtype=torch.int32
            ),
            "full_attention": torch.tensor(
                [[1, 2, 3, -1], [4, 8, -1, -1]], dtype=torch.int32
            ),
        }
        seq_lens = torch.tensor([5, 4], dtype=torch.int32)
        bases = {gid: torch.zeros(2, dtype=torch.int32) for gid in tables}
        locs = self.backend._compute_flat_decode_out_cache_locs(
            tables, bases, seq_lens, PAGE
        )
        # sliding: r0 page 7*2+0=14; r1 page 6*2+1=13.
        assert locs["sliding_attention"].tolist() == [14, 13]
        # full: r0 3*2+0=6; r1 8*2+1=17.
        assert locs["full_attention"].tolist() == [6, 17]
        assert locs["full_attention"].dtype == torch.int32
        assert locs["sliding_attention"].dtype == torch.int32

    def test_decode_locs_bounded_negative_pages_route_to_dummy(self):
        torch = self.torch
        tables = {
            # Logical pages 3 and 4 are present in the bounded table, but both
            # map to the negative padding/hole sentinel.
            "bounded": torch.tensor([[-1, -1]], dtype=torch.int32)
        }
        bases = {"bounded": torch.tensor([3], dtype=torch.int32)}
        cases = (
            # pos 7 -> logical page 3, local col 0, offset 1.
            (1, 8, [1]),
            # pos 6..9 -> logical pages 3,3,4,4, local cols 0,0,1,1.
            (4, 10, [0, 1, 0, 1]),
        )

        for num_tokens_per_req, seq_len, expected in cases:
            with self.subTest(num_tokens_per_req=num_tokens_per_req):
                locs = self.backend._compute_flat_decode_out_cache_locs(
                    tables,
                    bases,
                    torch.tensor([seq_len], dtype=torch.int32),
                    PAGE,
                    num_tokens_per_req=num_tokens_per_req,
                )
                # The non-zero logical base selects an in-range local column;
                # clamping physical page -1 to dummy page 0 leaves only the
                # original offset within that page.
                self.assertEqual(locs["bounded"].tolist(), expected)

    def test_extend_locs_formula(self):
        torch = self.torch
        # r0: prefix 2, extend 3 -> positions 2,3,4; r1: prefix 0, extend 2
        # -> 0,1. Bounds come from the CPU mirrors (no GPU sync).
        tables = {
            "full_attention": torch.tensor(
                [[1, 2, 3, -1], [4, 8, -1, -1]], dtype=torch.int32
            )
        }
        prefix_cpu = torch.tensor([2, 0], dtype=torch.int32)
        extend_cpu = torch.tensor([3, 2], dtype=torch.int32)
        bases = {"full_attention": torch.zeros(2, dtype=torch.int32)}
        locs = self.backend._compute_flat_extend_out_cache_locs(
            tables, bases, prefix_cpu, extend_cpu, PAGE
        )
        # r0: pos 2,3,4 -> page_idx 1,1,2 -> pages 2,2,3 ->
        #     locs 2*2+0=4, 2*2+1=5, 3*2+0=6; r1: pages 4,4 -> locs 8, 9.
        assert locs["full_attention"].tolist() == [4, 5, 6, 8, 9]
        assert locs["full_attention"].dtype == torch.int32

    def test_decode_locs_honor_group_page_sizes(self):
        torch = self.torch
        self.backend._learn_flat_state_groups(_flat_specs({"small": 2, "large": 4}))
        tables = {
            "small": torch.tensor([[1, 2, 3]], dtype=torch.int32),
            "large": torch.tensor([[6, 7]], dtype=torch.int32),
        }
        bases = {gid: torch.zeros(1, dtype=torch.int32) for gid in tables}
        locs = self.backend._compute_flat_decode_out_cache_locs(
            tables, bases, torch.tensor([5], dtype=torch.int32), PAGE
        )
        self.assertEqual(locs["small"].tolist(), [6])
        self.assertEqual(locs["large"].tolist(), [28])

    def test_extend_locs_honor_group_page_sizes(self):
        torch = self.torch
        self.backend._learn_flat_state_groups(_flat_specs({"small": 2, "large": 4}))
        tables = {
            "small": torch.tensor([[10, 11, 12]], dtype=torch.int32),
            "large": torch.tensor([[20, 21]], dtype=torch.int32),
        }
        bases = {gid: torch.zeros(1, dtype=torch.int32) for gid in tables}
        locs = self.backend._compute_flat_extend_out_cache_locs(
            tables,
            bases,
            torch.tensor([3], dtype=torch.int32),
            torch.tensor([3], dtype=torch.int32),
            PAGE,
        )
        self.assertEqual(locs["small"].tolist(), [23, 24, 25])
        self.assertEqual(locs["large"].tolist(), [83, 84, 85])

    def test_logical_base_bounds_asserts_are_debug_only(self):
        torch = self.torch
        tables = {"full_attention": torch.tensor([[1, 2]], dtype=torch.int32)}
        bases = {"full_attention": torch.zeros(1, dtype=torch.int32)}
        seq_lens = torch.tensor([2], dtype=torch.int32)
        prefix_lens = torch.tensor([0], dtype=torch.int32)
        extend_lens = torch.tensor([2], dtype=torch.int32)

        with mock.patch.dict(os.environ):
            os.environ.pop("TOKENSPEED_FLAT_DEBUG", None)
            with mock.patch.object(
                torch,
                "_assert_async",
                side_effect=AssertionError("production path ran a bounds reduction"),
            ) as assert_async:
                self.backend._compute_flat_decode_out_cache_locs(
                    tables, bases, seq_lens, PAGE
                )
                self.backend._compute_flat_decode_out_cache_locs(
                    tables, bases, seq_lens, PAGE, num_tokens_per_req=2
                )
                self.backend._compute_flat_extend_out_cache_locs(
                    tables, bases, prefix_lens, extend_lens, PAGE
                )
            assert_async.assert_not_called()

        with mock.patch.dict(
            os.environ, {"TOKENSPEED_FLAT_DEBUG": "1"}
        ), mock.patch.object(torch, "_assert_async") as assert_async:
            self.backend._compute_flat_decode_out_cache_locs(
                tables, bases, seq_lens, PAGE
            )
            self.backend._compute_flat_decode_out_cache_locs(
                tables, bases, seq_lens, PAGE, num_tokens_per_req=2
            )
            self.backend._compute_flat_extend_out_cache_locs(
                tables, bases, prefix_lens, extend_lens, PAGE
            )
        self.assertEqual(assert_async.call_count, 3)


class MaybeCheckFlatWriteLocsTest(_MHACase):
    """TOKENSPEED_FLAT_DEBUG gate: off by default, loud when on."""

    def _table_with_front_hole(self):
        # SWA-style table: front hole (page 0) then real pages 2, 3.
        return {
            "sliding_attention": self.torch.tensor(
                [[0, 2, 3, -1]], dtype=self.torch.int32
            )
        }

    def test_off_by_default_ignores_bad_locs(self):
        torch = self.torch
        bad = {"sliding_attention": torch.tensor([0], dtype=torch.int32)}
        with mock.patch.dict(os.environ):
            os.environ.pop("TOKENSPEED_FLAT_DEBUG", None)
            self.backend._maybe_check_flat_write_locs(
                self._table_with_front_hole(), bad, PAGE
            )

    def test_debug_rejects_write_into_hole_page(self):
        torch = self.torch
        # loc 1 -> page 0 = the slid-out hole; a write there is corruption.
        bad = {"sliding_attention": torch.tensor([1], dtype=torch.int32)}
        with mock.patch.dict(
            os.environ, {"TOKENSPEED_FLAT_DEBUG": "1"}
        ), self.assertRaisesRegex(AssertionError, "null page.*sliding_attention"):
            self.backend._maybe_check_flat_write_locs(
                self._table_with_front_hole(), bad, PAGE
            )

    def test_debug_rejects_page_outside_table(self):
        torch = self.torch
        # loc 18 -> page 9, not in the group's table.
        bad = {"sliding_attention": torch.tensor([18], dtype=torch.int32)}
        with mock.patch.dict(
            os.environ, {"TOKENSPEED_FLAT_DEBUG": "1"}
        ), self.assertRaisesRegex(AssertionError, "escape.*sliding_attention"):
            self.backend._maybe_check_flat_write_locs(
                self._table_with_front_hole(), bad, PAGE
            )

    def test_debug_passes_valid_locs(self):
        torch = self.torch
        # Pages 2 and 3 are real table entries; hole 0 / pad -1 excluded.
        good = {"sliding_attention": torch.tensor([4, 5, 6], dtype=torch.int32)}
        with mock.patch.dict(os.environ, {"TOKENSPEED_FLAT_DEBUG": "1"}):
            self.backend._maybe_check_flat_write_locs(
                self._table_with_front_hole(), good, PAGE
            )

    def test_debug_honors_group_page_size(self):
        torch = self.torch
        self.backend._learn_flat_state_groups(_flat_specs({"wide": 4}))
        tables = {"wide": torch.tensor([[2, 3]], dtype=torch.int32)}
        locs = {"wide": torch.tensor([11, 15], dtype=torch.int32)}
        with mock.patch.dict(os.environ, {"TOKENSPEED_FLAT_DEBUG": "1"}):
            self.backend._maybe_check_flat_write_locs(tables, locs, PAGE)


class InitForwardMetadataAssemblyTest(_MHACase):
    """Real init_forward_metadata on a __init__-bypassed backend (CPU-only:
    plain tensors, no kernels)."""

    def setUp(self):
        super().setUp()
        torch = self.torch
        backend = self.MHAAttnBackend.__new__(self.MHAAttnBackend)
        backend.page_size = PAGE
        backend.max_context_len = MAX_NUM_PAGES * PAGE
        backend.max_num_pages = MAX_NUM_PAGES
        backend.spec_num_tokens = 1
        backend.is_draft = False
        backend.draft_block_decode = False
        backend._learn_flat_state_groups(
            _flat_specs(
                {
                    "sliding_attention": PAGE,
                    "full_attention": PAGE,
                }
            )
        )
        backend.forward_decode_metadata = None
        backend.forward_extend_metadata = None
        self.backend = backend
        self.req_to_page = torch.zeros(
            (MAX_NUM_PAGES, MAX_NUM_PAGES), dtype=torch.int32
        )

    def _init(
        self,
        forward_mode,
        seq_lens,
        flat_block_tables,
        flat_block_table_base_offsets,
        extend_prefix_lens=None,
        extend_seq_lens=None,
    ):
        torch = self.torch
        bs = int(seq_lens.shape[0])
        if extend_prefix_lens is None:
            extend_prefix_lens = torch.zeros(bs, dtype=torch.int32)
        if extend_seq_lens is None:
            extend_seq_lens = seq_lens - extend_prefix_lens
        self.backend.init_forward_metadata(
            bs,
            bs,
            torch.arange(bs, dtype=torch.int64),
            seq_lens,
            self.req_to_page,
            forward_mode,
            extend_seq_lens,
            extend_seq_lens,  # *_cpu twin: same values, sliced + tolist'ed
            extend_prefix_lens,
            extend_prefix_lens,
            flat_block_tables=flat_block_tables,
            flat_block_table_base_offsets=flat_block_table_base_offsets,
        )

    def test_decode_assembly_populates_out_cache_locs(self):
        torch = self.torch
        tables = {
            "sliding_attention": torch.tensor(
                [[0, 5, 7, -1], [0, 6, -1, -1]], dtype=torch.int32
            ),
            "full_attention": torch.tensor(
                [[1, 2, 3, -1], [4, 8, -1, -1]], dtype=torch.int32
            ),
        }
        bases = {gid: torch.zeros(2, dtype=torch.int32) for gid in tables}
        seq_lens = torch.tensor([5, 4], dtype=torch.int32)
        self._init(_decode_forward_mode(), seq_lens, tables, bases)
        md = self.backend.forward_decode_metadata
        self.assertIs(md.page_tables, tables)
        self.assertIs(md.block_table_base_offsets, bases)
        self.assertEqual(md.out_cache_locs["sliding_attention"].tolist(), [14, 13])
        self.assertEqual(md.out_cache_locs["full_attention"].tolist(), [6, 17])
        self.assertEqual(md.out_cache_locs["full_attention"].dtype, self.torch.int32)

    def test_decode_assembly_none_without_flat_tables(self):
        torch = self.torch
        self._init(
            _decode_forward_mode(),
            torch.tensor([5, 4], dtype=torch.int32),
            None,
            None,
        )
        md = self.backend.forward_decode_metadata
        self.assertIsNone(md.page_tables)
        self.assertIsNone(md.out_cache_locs)

    def test_extend_assembly_populates_out_cache_locs(self):
        torch = self.torch
        tables = {
            "full_attention": torch.tensor(
                [[1, 2, 3, -1], [4, 8, -1, -1]], dtype=torch.int32
            )
        }
        bases = {"full_attention": torch.zeros(2, dtype=torch.int32)}
        seq_lens = torch.tensor([5, 2], dtype=torch.int32)
        prefix = torch.tensor([2, 0], dtype=torch.int32)
        self._init(
            _extend_forward_mode(),
            seq_lens,
            tables,
            bases,
            extend_prefix_lens=prefix,
        )
        md = self.backend.forward_extend_metadata
        self.assertIs(md.page_tables, tables)
        # Same hand-derived layout as the formula test: request-order flatten.
        self.assertEqual(md.out_cache_locs["full_attention"].tolist(), [4, 5, 6, 8, 9])
        self.assertEqual(md.out_cache_locs["full_attention"].dtype, self.torch.int32)
        # Non-draft extend never fills decode metadata.
        self.assertIsNone(self.backend.forward_decode_metadata)

    def test_extend_assembly_none_without_flat_tables(self):
        torch = self.torch
        self._init(
            _extend_forward_mode(),
            torch.tensor([3, 2], dtype=torch.int32),
            None,
            None,
        )
        md = self.backend.forward_extend_metadata
        self.assertIsNone(md.page_tables)
        self.assertIsNone(md.out_cache_locs)


class SelectOutCacheLocTest(_MHACase):
    """_select_out_cache_loc mirrors _select_page_table's fallback ladder;
    the public select_out_cache_loc serves prewrite."""

    def setUp(self):
        super().setUp()
        backend = self.MHAAttnBackend.__new__(self.MHAAttnBackend)
        backend.forward_decode_metadata = None
        backend.forward_extend_metadata = None
        self.backend = backend

    def test_select_out_cache_loc_routes_by_group(self):
        torch = self.torch
        md = SimpleNamespace(
            out_cache_locs={
                "sliding_attention": torch.tensor([14], dtype=torch.int32),
                "full_attention": torch.tensor([6], dtype=torch.int32),
            }
        )
        layer = SimpleNamespace(group_id="full_attention")
        fallback = torch.tensor([99], dtype=torch.int32)
        got = self.backend._select_out_cache_loc(layer, md, fallback)
        assert got.tolist() == [6]

    def test_select_out_cache_loc_none_falls_back(self):
        md = SimpleNamespace(out_cache_locs=None)
        fallback = self.torch.tensor([99], dtype=self.torch.int32)
        got = self.backend._select_out_cache_loc(
            SimpleNamespace(group_id="full_attention"), md, fallback
        )
        assert got is fallback

    def test_select_out_cache_loc_single_group_fallback(self):
        # Empty group_id + one-entry dict -> that entry, mirroring
        # _select_page_table's ladder (non-group-aware layer, single group).
        torch = self.torch
        only = torch.tensor([6], dtype=torch.int32)
        md = SimpleNamespace(out_cache_locs={"full_attention": only})
        fallback = torch.tensor([99], dtype=torch.int32)
        got = self.backend._select_out_cache_loc(
            SimpleNamespace(group_id=""), md, fallback
        )
        assert got is only
        # Unknown group_id + single group also resolves to that entry.
        got = self.backend._select_out_cache_loc(
            SimpleNamespace(group_id="zz"), md, fallback
        )
        assert got is only

    def test_select_out_cache_loc_unknown_group_raises(self):
        md = SimpleNamespace(out_cache_locs={"a": None, "b": None})
        with self.assertRaises(KeyError):
            self.backend._select_out_cache_loc(
                SimpleNamespace(group_id="zz"),
                md,
                self.torch.empty(0, dtype=self.torch.int32),
            )

    def test_public_select_uses_decode_metadata(self):
        torch = self.torch
        self.backend.forward_decode_metadata = SimpleNamespace(
            out_cache_locs={"full_attention": torch.tensor([6], dtype=torch.int32)}
        )
        got = self.backend.select_out_cache_loc(
            SimpleNamespace(group_id="full_attention"),
            torch.tensor([99], dtype=torch.int32),
        )
        assert got.tolist() == [6]
        self.backend.forward_decode_metadata = None
        fb = torch.tensor([99], dtype=torch.int32)
        assert (
            self.backend.select_out_cache_loc(
                SimpleNamespace(group_id="full_attention"), fb
            )
            is fb
        )


_GROUP_IDS = ("sliding_attention", "full_attention")


class GraphLocBuffersTest(_MHACase):
    """Persistent per-group write-loc buffers for CUDA graphs: capture hands
    metadata VIEWS of persistent buffers; replay only copy_'s recomputed
    locs into them."""

    def setUp(self):
        super().setUp()
        torch = self.torch
        if not torch.cuda.is_available():
            self.skipTest("stacked replay write locations require CUDA")
        backend = self.MHAAttnBackend.__new__(self.MHAAttnBackend)
        backend.spec_num_tokens = 1
        backend.is_draft = False
        backend.draft_block_decode = False
        backend.flat_state_group_ids = frozenset()
        backend.flat_engine_owned_group_ids = frozenset()
        backend._learn_flat_state_groups(_flat_specs({gid: PAGE for gid in _GROUP_IDS}))
        backend.max_num_pages = MAX_NUM_PAGES
        backend.page_size = PAGE
        backend.device = "cuda"
        backend.cuda_graph_decode_metadata = {}
        backend.cuda_graph_page_table = torch.zeros(
            (MAX_BS, MAX_NUM_PAGES), dtype=torch.int32, device=backend.device
        )
        # Stand-in for the controller's seq_lens_buf; tests pre-write it
        # before replay exactly like the wrapper's input prep does.
        backend.cuda_graph_seq_lens = torch.ones(
            MAX_BS, dtype=torch.int32, device=backend.device
        )
        backend._init_flat_graph_buffers(MAX_BS)
        self.backend = backend

    def _capture(self, bs, flat_cache_group_ids=()):
        torch = self.torch
        self.backend.init_forward_metadata_capture_cuda_graph(
            bs,
            torch.arange(bs, dtype=torch.int64, device=self.backend.device),
            torch.ones(bs, dtype=torch.int32, device=self.backend.device),
            _decode_forward_mode(),
            flat_cache_group_ids=flat_cache_group_ids,
        )
        return self.backend.cuda_graph_decode_metadata[bs]

    def _replay(
        self,
        bs,
        flat_block_tables=None,
        flat_block_table_base_offsets=None,
        seq_lens=None,
    ):
        torch = self.torch
        if seq_lens is not None:
            # Wrapper contract: input prep writes the step's lens (dummy
            # tail = 1) into seq_lens_buf BEFORE replay runs.
            self.backend.cuda_graph_seq_lens[: len(seq_lens)] = torch.tensor(
                seq_lens, dtype=torch.int32, device=self.backend.device
            )
        kwargs = {}
        if flat_block_tables is not None:
            kwargs["flat_block_tables"] = flat_block_tables
        if flat_block_table_base_offsets is not None:
            kwargs["flat_block_table_base_offsets"] = flat_block_table_base_offsets
        self.backend.init_forward_metadata_replay_cuda_graph(
            bs,
            torch.arange(MAX_BS, dtype=torch.int64, device=self.backend.device),
            torch.ones(MAX_BS, dtype=torch.int32, device=self.backend.device),
            torch.zeros(
                (MAX_BS, MAX_NUM_PAGES),
                dtype=torch.int32,
                device=self.backend.device,
            ),
            _decode_forward_mode(),
            **kwargs,
        )

    def test_capture_builds_persistent_loc_buffers(self):
        bs = 2
        metadata = self._capture(bs, _GROUP_IDS)
        bufs = self.backend.cuda_graph_flat_out_cache_locs
        self.assertEqual(set(bufs), set(_GROUP_IDS))
        for gid, buf in bufs.items():
            self.assertEqual(tuple(buf.shape), (MAX_BS,))
            self.assertEqual(buf.dtype, self.torch.int32)
            view = metadata.out_cache_locs[gid]
            self.assertEqual(tuple(view.shape), (bs,))
            # Pointer-fixing: metadata holds a view of the persistent
            # buffer, so the captured graph records a stable address.
            self.assertEqual(view.data_ptr(), buf.data_ptr())
        # Second capture at a different bs reuses the same buffers.
        ptrs = {gid: buf.data_ptr() for gid, buf in bufs.items()}
        second = self._capture(3, _GROUP_IDS)
        self.assertEqual(
            ptrs,
            {
                gid: buf.data_ptr()
                for gid, buf in (self.backend.cuda_graph_flat_out_cache_locs.items())
            },
        )
        for gid in _GROUP_IDS:
            self.assertEqual(tuple(second.out_cache_locs[gid].shape), (3,))
            self.assertEqual(second.out_cache_locs[gid].data_ptr(), ptrs[gid])

    def test_replay_computes_locs_from_persistent_tables(self):
        torch = self.torch
        self._capture(3, _GROUP_IDS)
        # Sentinel-fill the loc buffers: replay must overwrite exactly the
        # first bs rows and leave the tail untouched.
        for buf in self.backend.cuda_graph_flat_out_cache_locs.values():
            buf.fill_(99)
        # page_size=2. Row 2 is a padded dummy row (0-filled, seq_len 1).
        # sliding rows keep a front hole (slid-out page 0 slot).
        tables = {
            "sliding_attention": torch.tensor(
                [[0, 5, 7, -1], [0, 6, -1, -1], [0, 0, 0, 0]],
                dtype=torch.int32,
                device=self.backend.device,
            ),
            "full_attention": torch.tensor(
                [[1, 2, 3, -1], [4, 8, -1, -1], [0, 0, 0, 0]],
                dtype=torch.int32,
                device=self.backend.device,
            ),
        }
        bases = {
            gid: torch.zeros(3, dtype=torch.int32, device=self.backend.device)
            for gid in tables
        }
        self._replay(3, tables, bases, seq_lens=[5, 4, 1])
        locs = self.backend.cuda_graph_flat_out_cache_locs
        # sliding: r0 seq 5 -> pos 4 -> page_idx 2 -> page 7 -> 7*2+0=14;
        #          r1 seq 4 -> pos 3 -> page_idx 1 -> page 6 -> 6*2+1=13;
        #          r2 dummy seq 1 -> pos 0 -> page 0 -> loc 0 (dummy page).
        self.assertEqual(locs["sliding_attention"][:3].tolist(), [14, 13, 0])
        # full: r0 page 3 -> 3*2+0=6; r1 page 8 -> 8*2+1=17; r2 dummy -> 0.
        self.assertEqual(locs["full_attention"][:3].tolist(), [6, 17, 0])
        for buf in locs.values():
            self.assertEqual(buf.dtype, torch.int32)
            # Rows beyond bs untouched (still the sentinel).
            self.assertTrue((buf[3:] == 99).all())

    def test_capture_without_flat_leaves_metadata_locs_none(self):
        metadata = self._capture(2)
        self.assertIsNone(metadata.out_cache_locs)
        self.assertIsNone(metadata.block_table_base_offsets)
        self.assertEqual(
            set(self.backend.cuda_graph_flat_out_cache_locs), set(_GROUP_IDS)
        )


class BaseSelectOutCacheLocTest(_TorchCase):
    """The AttentionBackend default hook is the identity: backends without
    flat cache groups keep the caller's single-stream out_cache_loc."""

    def test_default_hook_returns_fallback_as_is(self):
        from tokenspeed.runtime.layers.attention.backends.base import (
            AttentionBackend,
        )

        class _MinimalBackend(AttentionBackend):
            def init_forward_metadata(self, *args, **kwargs):
                raise NotImplementedError

        backend = _MinimalBackend.__new__(_MinimalBackend)  # skip __init__
        fb = self.torch.tensor([99], dtype=self.torch.int32)
        got = backend.select_out_cache_loc(
            SimpleNamespace(group_id="full_attention"), fb
        )
        assert got is fb


if __name__ == "__main__":
    unittest.main()
