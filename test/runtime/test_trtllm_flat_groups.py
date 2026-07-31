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

    def _bare_backend(
        self,
        *,
        page_size=64,
        max_num_pages=8,
        spec_num_tokens=1,
        device="cpu",
        groups=None,
    ):
        # Bypass __init__: the paths under test read only these attributes.
        # Capture/replay tests pass device="cuda" and declare their groups —
        # replay write locs are triton-only (no python fallback).
        b = self.Backend.__new__(self.Backend)
        b.page_size = page_size
        b.max_num_pages = max_num_pages
        b.max_context_len = page_size * max_num_pages
        b.device = device
        if groups is not None:
            b.flat_group_page_sizes = groups
        b.spec_num_tokens = spec_num_tokens
        b.is_draft = False
        b.draft_block_decode = False
        b.forward_decode_metadata = None
        b.forward_prefill_metadata = None
        b.cuda_graph_prefill_metadata = {}
        b.cuda_graph_decode_metadata = {}
        b.spec_cache_seqlens_buf = self.torch.zeros(
            8, dtype=self.torch.int32, device=device
        )
        return b

    def _layer(self, group_id):
        from types import SimpleNamespace

        return SimpleNamespace(group_id=group_id)

    def test_select_page_table_routes_by_group(self):
        b = self._bare_backend()
        full = self.torch.tensor([[1, 2]], dtype=self.torch.int32)
        swa = self.torch.tensor([[3, 0]], dtype=self.torch.int32)
        meta = self.Metadata(
            page_tables={"full_attention": full, "sliding_attention": swa}
        )
        self.assertIs(b._select_page_table(self._layer("full_attention"), meta), full)
        self.assertIs(b._select_page_table(self._layer("sliding_attention"), meta), swa)

    def test_expands_logical_group_pages_for_kernel_reads(self):
        b = self._bare_backend(
            groups={"full_attention": 128},
            max_num_pages=6,
        )
        logical = {
            "full_attention": self.torch.tensor([[3, 5, -1]], dtype=self.torch.int32)
        }

        kernel = b._flat_kernel_page_tables(logical)

        self.assertEqual(
            kernel["full_attention"].tolist(),
            [[6, 7, 10, 11, 0, 1]],
        )

    def test_build_page_table_keeps_radix_direct_copy_path(self):
        b = self._bare_backend(page_size=64, max_num_pages=4)
        req_to_page = self.torch.tensor(
            [[0, 0, 0, 0], [3, 5, 7, 9]], dtype=self.torch.int32
        )
        out = self.torch.empty((1, 4), dtype=self.torch.int32)

        page_table = b._build_page_table(
            self.torch.tensor([1], dtype=self.torch.int32),
            self.torch.tensor([256], dtype=self.torch.int32),
            1,
            req_to_page,
            out,
        )

        self.assertEqual(page_table.data_ptr(), out.data_ptr())
        self.assertEqual(page_table.tolist(), [[3, 5, 7, 9]])

    def test_eager_flat_expands_per_group_table(self):
        b = self._bare_backend(
            page_size=64,
            max_num_pages=4,
            groups={"full_attention": 128},
        )

        b.init_forward_metadata(
            bs=1,
            req_pool_indices=self.torch.tensor([0], dtype=self.torch.int32),
            seq_lens=self.torch.tensor([129], dtype=self.torch.int32),
            forward_mode=_DecodeMode(),
            req_to_page=None,
            flat_block_tables={
                "full_attention": self.torch.tensor([[3, 5]], dtype=self.torch.int32)
            },
        )

        metadata = b.forward_decode_metadata
        self.assertIsNone(metadata.page_table)
        self.assertEqual(
            metadata.page_tables["full_attention"].tolist(),
            [[6, 7, 10, 11]],
        )
        self.assertEqual(metadata.out_cache_locs["full_attention"].tolist(), [10 * 64])

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
        if not self.torch.cuda.is_available():
            self.skipTest("replay write locs are triton-only (needs CUDA)")
        gids = ("full_attention", "sliding_attention")
        b = self._bare_backend(device="cuda", groups={g: 64 for g in gids})
        max_bs, bs = 4, 2
        b._init_flat_graph_buffers(max_bs)
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
        # page 0 (flat_table_tail_pad), recomputes locs (fused triton).
        seq_lens = self.torch.tensor([65, 1, 1, 1], dtype=self.torch.int32).cuda()
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

    def test_graph_replay_expands_scheduler_pages_before_kernel_reads(self):
        if not self.torch.cuda.is_available():
            self.skipTest("replay write locs are triton-only (needs CUDA)")
        gid = "full_attention"
        b = self._bare_backend(device="cuda", groups={gid: 128})
        b._init_flat_graph_buffers(max_bs=2)
        seq_lens = self.torch.tensor([129, 1], dtype=self.torch.int32).cuda()
        src = {
            gid: self.torch.tensor(
                [[3, 5], [0, -1]], dtype=self.torch.int32, device="cuda"
            )
        }

        b._flat_replay_fill(2, src, seq_lens)

        table = b.cuda_graph_flat_page_tables[gid]
        self.assertEqual(table[0, :4].tolist(), [6, 7, 10, 11])
        self.assertEqual(
            b.cuda_graph_flat_out_cache_locs[gid][:2].tolist(),
            [10 * 64, 0],
        )

    def test_verify_metadata_expanded_write_locs(self):
        # Target verify (spec N, not draft): [bs]-row per-group tables in the
        # prefill slot + [bs*N] token-major write locs (radix verify layout).
        b = self._bare_backend(spec_num_tokens=4)
        seq_lens = self.torch.tensor([65, 3], dtype=self.torch.int32)
        tables = {
            "full_attention": self.torch.tensor(
                [[11, 12], [13, -1]], dtype=self.torch.int32
            ),
            "sliding_attention": self.torch.tensor(
                [[21, 22], [23, -1]], dtype=self.torch.int32
            ),
        }
        b.init_forward_metadata(
            bs=2,
            req_pool_indices=self.torch.tensor([0, 1], dtype=self.torch.int32),
            seq_lens=seq_lens,
            forward_mode=_DecodeMode(),
            req_to_page=None,
            flat_block_tables=tables,
        )
        meta = b.forward_prefill_metadata
        self.assertIsNone(meta.page_table)
        self.assertIs(meta.page_tables, tables)
        # req0 positions 61..64 (pages 11,11,11,12); req1 clamps 0,0,1,2 (page 13).
        self.assertEqual(
            meta.out_cache_locs["full_attention"].tolist(),
            [11 * 64 + 61, 11 * 64 + 62, 11 * 64 + 63, 12 * 64 + 0]
            + [13 * 64 + 0, 13 * 64 + 0, 13 * 64 + 1, 13 * 64 + 2],
        )
        self.assertEqual(
            meta.out_cache_locs["sliding_attention"].tolist(),
            [21 * 64 + 61, 21 * 64 + 62, 21 * 64 + 63, 22 * 64 + 0]
            + [23 * 64 + 0, 23 * 64 + 0, 23 * 64 + 1, 23 * 64 + 2],
        )
        # KV seqlens clamped >= N so padded rows avoid empty causal spans.
        self.assertEqual(meta.cache_seqlens_int32.tolist(), [65, 4])

    def test_verify_capture_replay_expanded_loc_views(self):
        if not self.torch.cuda.is_available():
            self.skipTest("replay write locs are triton-only (needs CUDA)")
        b = self._bare_backend(
            spec_num_tokens=4,
            device="cuda",
            groups={"full_attention": 64},
        )
        max_bs, bs = 4, 2
        b._init_flat_graph_buffers(max_bs)
        b.cuda_graph_cache_seqlens = self.torch.ones(
            max_bs, dtype=self.torch.int32, device="cuda"
        )
        b.init_forward_metadata_capture_cuda_graph(
            bs,
            req_pool_indices=self.torch.tensor(
                [0, 1], dtype=self.torch.int32, device="cuda"
            ),
            seq_lens=b.cuda_graph_cache_seqlens[:bs],
            forward_mode=_DecodeMode(),
            flat_cache_group_ids=("full_attention",),
        )
        meta = b.cuda_graph_prefill_metadata[bs]
        self.assertIsNone(meta.page_table)
        self.assertEqual(meta.out_cache_locs["full_attention"].shape[0], bs * 4)
        # Replay refreshes tables and recomputes [bs*N] locs from live lens.
        b.cuda_graph_cache_seqlens[:bs] = self.torch.tensor(
            [65, 1], dtype=self.torch.int32
        )
        src = {
            "full_attention": self.torch.tensor(
                [[11, 12], [0, -1]], dtype=self.torch.int32, device="cuda"
            )
        }
        b.init_forward_metadata_replay_cuda_graph(
            bs,
            req_pool_indices=self.torch.tensor([0, 1], dtype=self.torch.int32),
            seq_lens=b.cuda_graph_cache_seqlens,
            forward_mode=_DecodeMode(),
            flat_block_tables=src,
        )
        locs = b.cuda_graph_flat_out_cache_locs["full_attention"][: bs * 4]
        self.assertEqual(
            locs.tolist(),
            [11 * 64 + 61, 11 * 64 + 62, 11 * 64 + 63, 12 * 64 + 0] + [0, 0, 0, 0],
        )

    def test_prewrite_metadata_routes_verify_to_prefill_slot(self):
        b = self._bare_backend(spec_num_tokens=4)
        prefill, decode = self.Metadata(), self.Metadata()
        b.forward_prefill_metadata, b.forward_decode_metadata = prefill, decode
        # Target verify is DECODE mode; its metadata lives in the prefill slot.
        self.assertIs(b._prewrite_metadata(_DecodeMode()), prefill)
        b.is_draft = True
        self.assertIs(b._prewrite_metadata(_DecodeMode()), decode)

    def test_flat_with_dflash_asserts(self):
        b = self._bare_backend(spec_num_tokens=4)
        b.is_draft = True
        b.draft_block_decode = True
        tables = {"full_attention": self.torch.zeros((1, 1), dtype=self.torch.int32)}
        with self.assertRaisesRegex(AssertionError, "DFLASH"):
            b.init_forward_metadata(
                bs=1,
                req_pool_indices=self.torch.tensor([0], dtype=self.torch.int32),
                seq_lens=self.torch.tensor([1], dtype=self.torch.int32),
                forward_mode=_DecodeMode(),
                req_to_page=None,
                flat_block_tables=tables,
            )


class TRTLLMMixedExtendRowsTest(unittest.TestCase):
    """MIXED packs extend rows first, so the prefill slot covers only those.

    Passing bs straight through made ``seq_lens - extend_prefix_lens`` a shape
    error once a mixed batch also carried a prefix-cache hit.
    """

    def setUp(self):
        try:
            from tokenspeed.runtime.layers.attention.backends.trtllm import (
                TRTLLMMHAAttnBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        import torch

        self.torch = torch
        self.Backend = TRTLLMMHAAttnBackend

    def _backend(self, *, max_num_pages=8, page_size=64):
        b = self.Backend.__new__(self.Backend)
        b.page_size = page_size
        b.max_num_pages = max_num_pages
        b.max_context_len = page_size * max_num_pages
        b.device = "cpu"
        b.spec_num_tokens = 1
        b.is_draft = True
        b.draft_block_decode = False
        b.uses_flat_cache_groups = False
        b.flat_group_page_sizes = {}
        b.flat_state_group_ids = frozenset()
        b.forward_decode_metadata = None
        b.forward_prefill_metadata = None
        b.cuda_graph_prefill_metadata = {}
        b.cuda_graph_decode_metadata = {}
        b.page_table_buf = self.torch.zeros((8, max_num_pages), dtype=self.torch.int32)
        return b

    def test_mixed_batch_prefill_metadata_covers_extend_rows_only(self):
        t = self.torch
        b = self._backend()
        bs, num_extends = 3, 2  # 2 extends + 1 decode
        # Extends carry a prefix-cache hit (the trigger for the subtraction).
        b._init_extend_metadata(
            num_extends,
            req_pool_indices=t.tensor([0, 1, 2], dtype=t.int32),
            seq_lens=t.tensor([40, 50, 900], dtype=t.int32),
            req_to_page=t.zeros((4, 8), dtype=t.int32),
            extend_with_prefix=True,
            extend_prefix_lens=t.tensor([10, 20], dtype=t.int32),
            extend_prefix_lens_cpu=t.tensor([10, 20], dtype=t.int32),
            extend_seq_lens_cpu=t.tensor([30, 30], dtype=t.int32),
        )
        md = b.forward_prefill_metadata
        # The decode row's length (900) must not leak into the prefill slot.
        self.assertEqual(md.cache_seqlens_int32.tolist(), [40, 50])
        # cu_seqlens_q = cumsum(seq_lens - prefix) over extend rows only.
        self.assertEqual(md.cu_seqlens_q.tolist(), [0, 30, 60])
        self.assertEqual(md.cu_seqlens_k.tolist(), [0, 40, 90])
        _ = bs  # documented above; the point is bs is NOT what gets used

    def test_pure_extend_is_unchanged(self):
        t = self.torch
        b = self._backend()
        b._init_extend_metadata(
            2,
            req_pool_indices=t.tensor([0, 1], dtype=t.int32),
            seq_lens=t.tensor([40, 50], dtype=t.int32),
            req_to_page=t.zeros((4, 8), dtype=t.int32),
            extend_with_prefix=True,
            extend_prefix_lens=t.tensor([10, 20], dtype=t.int32),
            extend_prefix_lens_cpu=t.tensor([10, 20], dtype=t.int32),
            extend_seq_lens_cpu=t.tensor([30, 30], dtype=t.int32),
        )
        md = b.forward_prefill_metadata
        self.assertEqual(md.cache_seqlens_int32.tolist(), [40, 50])
        self.assertEqual(md.cu_seqlens_q.tolist(), [0, 30, 60])


class DFlashFlatOptOutTest(unittest.TestCase):
    """DFLASH reads page tables from req_to_page, so its draft backend clears
    uses_flat_cache_groups and records no flat groups -- otherwise capture
    hands it group ids and replay's stale guard demands tables it never gets."""

    def setUp(self):
        try:
            from tokenspeed.runtime.configs.paged_cache_spec import (
                PagedCacheGroupSpec,
            )
            from tokenspeed.runtime.layers.attention.backends.flat_groups import (
                FlatCacheGroupsMixin,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs the tokenspeed runtime deps: {exc}")
        self.Mixin = FlatCacheGroupsMixin
        self.Spec = PagedCacheGroupSpec

    def _specs(self):
        return (
            self.Spec(
                group_id="full_attention",
                retention="full_history",
                rows_per_page=64,
                entry_stride_tokens=1,
                sliding_window_tokens=None,
            ),
        )

    def test_opted_out_backend_records_no_groups(self):
        class Host(self.Mixin):
            uses_flat_cache_groups = False

        h = Host()
        h._learn_flat_state_groups(self._specs())
        self.assertEqual(h.flat_group_page_sizes, {})
        self.assertEqual(h.flat_state_group_ids, frozenset())

    def test_flat_capable_backend_still_records_groups(self):
        class Host(self.Mixin):
            uses_flat_cache_groups = True

        h = Host()
        h._learn_flat_state_groups(self._specs())
        self.assertEqual(h.flat_group_page_sizes, {"full_attention": 64})

    def _mha_config(self, **overrides):
        import torch

        from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig

        kwargs = dict(
            device="cpu",
            backend_name="mha",
            num_attention_heads=4,
            num_kv_heads=4,
            head_dim=8,
            attn_tp_size=1,
            dtype=torch.bfloat16,
            kv_cache_dtype=torch.bfloat16,
            page_size=64,
            context_len=256,
            max_bs=2,
            max_graph_bs=2,
            kv_cache_quant_method="none",
            speculative_num_draft_tokens=1,
            draft_block_decode=False,
            is_draft=False,
        )
        kwargs.update(overrides)
        return MHAConfig(**kwargs)

    def test_real_backends_clear_the_flag_under_dflash(self):
        # Instantiate for real: the opt-out lives in __init__, so a class-level
        # assertion would prove nothing.
        try:
            from tokenspeed.runtime.layers.attention.backends.mha import (
                MHAAttnBackend,
            )
            from tokenspeed.runtime.layers.attention.backends.trtllm import (
                TRTLLMMHAAttnBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        dflash = self._mha_config(
            draft_block_decode=True, is_draft=True, speculative_num_draft_tokens=8
        )
        plain = self._mha_config()
        for cls in (MHAAttnBackend, TRTLLMMHAAttnBackend):
            self.assertFalse(
                cls(dflash).uses_flat_cache_groups,
                f"{cls.__name__} must opt out of flat groups under DFLASH",
            )
            self.assertTrue(
                cls(plain).uses_flat_cache_groups,
                f"{cls.__name__} must stay flat-capable otherwise",
            )

    def test_msa_wrapper_follows_its_halves(self):
        # MSAHybridAttnBackend fans the same kwargs (and flat_cache_group_ids)
        # out to BOTH halves, so it must not advertise flat capability while a
        # half has opted out -- constructing the real wrapper needs a MiniMax
        # config, so drive the delegation directly.
        try:
            from tokenspeed.runtime.layers.attention.backends.msa import (
                MSAHybridAttnBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")

        class _Half:
            def __init__(self, flat):
                self.uses_flat_cache_groups = flat

        w = MSAHybridAttnBackend.__new__(MSAHybridAttnBackend)
        for dense, sparse, expected in (
            (True, True, True),
            (False, True, False),  # DFLASH cleared the dense half
            (True, False, False),  # ... or the sparse half
            (False, False, False),
        ):
            w.full_attn_backend = _Half(dense)
            w.sparse_attn_backend = _Half(sparse)
            self.assertEqual(w.uses_flat_cache_groups, expected, (dense, sparse))

    def test_gdn_kda_hybrid_wrapper_keeps_its_union_declaration(self):
        # Deliberately NOT the same rule: that wrapper's halves cover different
        # families (full attention -> history, mamba -> state) and its flag is
        # a union declared on their behalf -- MambaAttnBackend never sets one.
        # An `and` there would read False and break Kimi-K3 / Qwen3.5 on flat.
        try:
            from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (  # noqa: E501
                HybridLinearAttnBackend,
                MambaAttnBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.assertTrue(HybridLinearAttnBackend.uses_flat_cache_groups)
        self.assertFalse(
            getattr(MambaAttnBackend, "uses_flat_cache_groups", False),
            "mamba half declares nothing; the union lives on the wrapper",
        )


class _DecodeMode:
    """Minimal ForwardMode stand-in for the decode dispatch path."""

    def is_extend_or_mixed(self):
        return False

    def is_mixed(self):
        return False


if __name__ == "__main__":
    unittest.main()
