from __future__ import annotations

import os
import sys
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="runtime-1gpu")


def _import_backend():
    from tokenspeed.runtime.layers.attention.backends.trtllm import (
        TRTLLMMHAAttnBackend,
        TRTLLMMHAMetadata,
    )

    return TRTLLMMHAAttnBackend, TRTLLMMHAMetadata


class TRTLLMFlatGroupsTest(unittest.TestCase):
    """The trtllm backend consumes flat per-group tables through the shared
    FlatCacheGroupsMixin: table/write-loc selection routes by layer.group_id,
    metadata drops the radix single table on the flat path, and the CUDA-graph
    buffers follow the capture/replay discipline."""

    def setUp(self):
        try:
            self.Backend, self.Metadata = _import_backend()
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        import torch

        self.torch = torch

    def _bare_backend(self, *, page_size=64, max_num_pages=8, spec_num_tokens=1):
        # Bypass __init__: the paths under test read only these attributes.
        b = self.Backend.__new__(self.Backend)
        b.page_size = page_size
        b.max_num_pages = max_num_pages
        b.max_context_len = page_size * max_num_pages
        b.device = "cpu"
        b.spec_num_tokens = spec_num_tokens
        b.is_draft = False
        b.draft_block_decode = False
        b.forward_decode_metadata = None
        b.forward_prefill_metadata = None
        b.cuda_graph_prefill_metadata = {}
        b.cuda_graph_decode_metadata = {}
        return b

    def _layer(self, group_id):
        from types import SimpleNamespace

        return SimpleNamespace(group_id=group_id)

    def test_flag_declared(self):
        self.assertTrue(self.Backend.uses_flat_cache_groups)

    def test_select_page_table_routes_by_group(self):
        b = self._bare_backend()
        full = self.torch.tensor([[1, 2]], dtype=self.torch.int32)
        swa = self.torch.tensor([[3, 0]], dtype=self.torch.int32)
        meta = self.Metadata(
            page_tables={"full_attention": full, "sliding_attention": swa}
        )
        self.assertIs(b._select_page_table(self._layer("full_attention"), meta), full)
        self.assertIs(b._select_page_table(self._layer("sliding_attention"), meta), swa)

    def test_select_out_cache_loc_routes_by_group(self):
        b = self._bare_backend()
        radix_loc = self.torch.tensor([7], dtype=self.torch.int32)
        full_loc = self.torch.tensor([64], dtype=self.torch.int32)
        meta_none = self.Metadata(out_cache_locs=None)
        self.assertIs(
            b._select_out_cache_loc(
                self._layer("full_attention"), meta_none, radix_loc
            ),
            radix_loc,
        )
        meta = self.Metadata(out_cache_locs={"full_attention": full_loc})
        self.assertIs(
            b._select_out_cache_loc(self._layer("full_attention"), meta, radix_loc),
            full_loc,
        )

    def test_decode_metadata_flat_drops_single_table(self):
        b = self._bare_backend()
        bs = 2
        seq_lens = self.torch.tensor([65, 3], dtype=self.torch.int32)
        tables = {
            "full_attention": self.torch.tensor(
                [[11, 12], [13, -1]], dtype=self.torch.int32
            ),
            "sliding_attention": self.torch.tensor(
                [[21, 22], [23, -1]], dtype=self.torch.int32
            ),
        }
        locs = b._compute_flat_decode_out_cache_locs(tables, seq_lens, b.page_size)
        b._init_decode_metadata(
            bs,
            req_pool_indices=self.torch.tensor([0, 1], dtype=self.torch.int32),
            seq_lens=seq_lens,
            req_to_page=None,
            flat_page_tables=tables,
            flat_out_cache_locs=locs,
        )
        meta = b.forward_decode_metadata
        self.assertIsNone(meta.page_table)
        self.assertIs(meta.page_tables, tables)
        # seq_len 65 -> page index 1, offset 0; seq_len 3 -> page 0, offset 2.
        self.assertEqual(
            meta.out_cache_locs["full_attention"].tolist(),
            [12 * 64 + 0, 13 * 64 + 2],
        )
        self.assertEqual(
            meta.out_cache_locs["sliding_attention"].tolist(),
            [22 * 64 + 0, 23 * 64 + 2],
        )

    def test_extend_metadata_flat_drops_single_table(self):
        b = self._bare_backend()
        bs = 1
        seq_lens = self.torch.tensor([66], dtype=self.torch.int32)
        tables = {"full_attention": self.torch.tensor([[5, 6]], dtype=self.torch.int32)}
        locs = b._compute_flat_extend_out_cache_locs(
            tables,
            self.torch.tensor([64], dtype=self.torch.int32),
            self.torch.tensor([2], dtype=self.torch.int32),
            b.page_size,
        )
        b._init_extend_metadata(
            bs,
            req_pool_indices=self.torch.tensor([0], dtype=self.torch.int32),
            seq_lens=seq_lens,
            req_to_page=None,
            extend_seq_lens_cpu=self.torch.tensor([2], dtype=self.torch.int32),
            flat_page_tables=tables,
            flat_out_cache_locs=locs,
        )
        meta = b.forward_prefill_metadata
        self.assertIsNone(meta.page_table)
        self.assertIs(meta.page_tables, tables)
        # New tokens at positions 64, 65 -> page 6, offsets 0 and 1.
        self.assertEqual(
            meta.out_cache_locs["full_attention"].tolist(), [6 * 64, 6 * 64 + 1]
        )

    def test_graph_capture_and_replay_discipline(self):
        b = self._bare_backend()
        max_bs, bs = 4, 2
        b._init_flat_graph_buffers(max_bs)
        gids = ("full_attention", "sliding_attention")
        page_tables, out_cache_locs = b._flat_capture_group_views(bs, gids)
        self.assertEqual(set(page_tables), set(gids))
        self.assertEqual(page_tables["full_attention"].shape, (bs, b.max_num_pages))

        # Replay without tables must fail loudly (stale-table guard).
        with self.assertRaisesRegex(RuntimeError, "stale page tables"):
            b._flat_replay_stale_guard(bs, None)
        with self.assertRaisesRegex(RuntimeError, "missing captured groups"):
            b._flat_replay_stale_guard(
                bs, {"full_attention": self.torch.zeros((bs, 1))}
            )

        # Replay fill copies rows, pads column tails with the trtllm dummy
        # page 0 (flat_table_tail_pad), recomputes locs.
        seq_lens = self.torch.tensor([65, 1, 1, 1], dtype=self.torch.int32)
        src = {
            "full_attention": self.torch.tensor(
                [[11, 12], [0, -1]], dtype=self.torch.int32
            ),
            "sliding_attention": self.torch.tensor(
                [[21, 22], [0, -1]], dtype=self.torch.int32
            ),
        }
        b._flat_replay_fill(bs, src, seq_lens)
        buf = b.cuda_graph_flat_page_tables["full_attention"]
        self.assertEqual(buf[0, :2].tolist(), [11, 12])
        self.assertEqual(self.Backend.flat_table_tail_pad, 0)
        self.assertEqual(buf[0, 2:].tolist(), [0] * (b.max_num_pages - 2))
        self.assertEqual(
            b.cuda_graph_flat_out_cache_locs["full_attention"][:bs].tolist(),
            [12 * 64 + 0, 0 * 64 + 0],
        )

    def test_flat_with_spec_asserts(self):
        b = self._bare_backend(spec_num_tokens=4)
        tables = {"full_attention": self.torch.zeros((1, 1), dtype=self.torch.int32)}
        with self.assertRaisesRegex(AssertionError, "spec_num_tokens"):
            b.init_forward_metadata(
                bs=1,
                req_pool_indices=self.torch.tensor([0], dtype=self.torch.int32),
                seq_lens=self.torch.tensor([1], dtype=self.torch.int32),
                forward_mode=_DecodeMode(),
                req_to_page=None,
                flat_block_tables=tables,
            )


class _DecodeMode:
    """Minimal ForwardMode stand-in for the decode dispatch path."""

    def is_extend_or_mixed(self):
        return False

    def is_mixed(self):
        return False


if __name__ == "__main__":
    unittest.main()
