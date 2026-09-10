"""The cache-group router and its stacked tables: the one block -> page point.

CPU tensors pin the reference slot math and the router's group bookkeeping
(dispatch, padding, delivery guard, write-location publication); CUDA cases
check the fused triton fills against the same torch reference.
"""

from __future__ import annotations

import ast
import os
import sys
import textwrap
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci
from runtime.cache_pool_test_utils import (
    assert_no_alias,
    binding_state,
    reachable_tensors,
    storages_of,
)

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.paged.base import (
    PagedAttentionBackend,
)
from tokenspeed.runtime.layers.attention.backends.paged.cache_group_geometry import (
    CacheGroupGeometry,
)
from tokenspeed.runtime.layers.attention.backends.paged.group_tables import (
    GroupTableSpec,
    GroupTableStacks,
)
from tokenspeed.runtime.layers.attention.backends.paged.router import CacheGroupRouter
from tokenspeed.runtime.layers.attention.backends.paged.write_locations import (
    decode_write_locations,
    extend_write_locations,
)

FULL = "full_attention"
SWA = "sliding_attention"


class WriteLocationMathTest(unittest.TestCase):
    """The slot rule over stacked tables, on CPU (the reference)."""

    def _tables(self, device="cpu"):
        # Two groups, kernel pages 4 and 2, three requests, max_num_pages 6.
        tables = torch.tensor(
            [
                [[7, 8, 9, 0, 0, 0], [3, 0, 5, 6, 0, 0], [0, 0, 0, 0, 0, 0]],
                [[2, 4, 6, 8, 10, 12], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
            ],
            dtype=torch.int32,
            device=device,
        )
        page_sizes = torch.tensor([4, 2], dtype=torch.int32, device=device)
        return tables, page_sizes

    def test_plain_decode_writes_the_last_position(self):
        tables, page_sizes = self._tables()
        seq_lens = torch.tensor([9, 4, 1], dtype=torch.int32)
        out = torch.zeros((2, 8), dtype=torch.int32)
        decode_write_locations(
            tables, page_sizes, seq_lens, out, bs=3, tokens_per_req=1
        )
        # group 0 (P=4): pos 8 -> page 9 slot 0 = 36; pos 3 -> page 3 slot 3 = 15;
        # padded request pos 0 -> page 0 -> dummy slot 0.
        self.assertEqual(out[0, :3].tolist(), [36, 15, 0])
        # group 1 (P=2): pos 8 -> page 10 slot 0 = 20; pos 3 -> page 0 (hole) -> 0.
        self.assertEqual(out[1, :3].tolist(), [20, 0, 0])

    def test_verify_window_is_token_major_and_clamps_padded_requests(self):
        tables, page_sizes = self._tables()
        seq_lens = torch.tensor([9, 1, 1], dtype=torch.int32)
        out = torch.zeros((2, 12), dtype=torch.int32)
        decode_write_locations(
            tables, page_sizes, seq_lens, out, bs=3, tokens_per_req=3
        )
        # request 0 writes positions 6,7,8 -> group 0 page 8 slots 2,3 then
        # page 9 slot 0 -> 34, 35, 36.
        self.assertEqual(out[0, :3].tolist(), [34, 35, 36])
        # A short live request (seq_len 1 < 3) clamps every position at 0.
        self.assertEqual(out[0, 3:6].tolist(), [12, 12, 12])
        # A padded request is all null pages by the fill contract -> slot 0.
        self.assertEqual(out[0, 6:9].tolist(), [0, 0, 0])

    def test_extend_span_is_request_major(self):
        tables, page_sizes = self._tables()
        prefix = torch.tensor([4, 0], dtype=torch.int32)
        new = torch.tensor([5, 2], dtype=torch.int32)
        locs = extend_write_locations(tables, page_sizes, prefix, new, total_tokens=7)
        self.assertEqual(tuple(locs.shape), (2, 7))
        # request 0, group 0: positions 4..8 -> page 8 slots 0..3, page 9 slot 0.
        self.assertEqual(locs[0, :5].tolist(), [32, 33, 34, 35, 36])
        # request 1, group 1 (P=2): positions 0,1 -> page 1 -> slots 2,3.
        self.assertEqual(locs[1, 5:].tolist(), [2, 3])

    def test_positions_past_the_table_route_to_the_dummy_slot(self):
        tables, page_sizes = self._tables()
        seq_lens = torch.tensor([40], dtype=torch.int32)  # page index 9 >= 6 pages
        out = torch.zeros((2, 4), dtype=torch.int32)
        decode_write_locations(
            tables, page_sizes, seq_lens, out, bs=1, tokens_per_req=1
        )
        self.assertEqual(out[:, 0].tolist(), [0, 0])

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_fused_kernels_match_the_reference(self):
        torch.manual_seed(0)
        g, bs, max_num_pages = 3, 17, 40
        tables = torch.randint(0, 50, (g, bs, max_num_pages), dtype=torch.int32)
        tables[:, :, max_num_pages - 5 :] = 0  # trailing nulls
        tables[1, 3, 2] = -1  # a ragged hole
        page_sizes = torch.tensor([1, 4, 16], dtype=torch.int32)
        seq_lens = torch.randint(1, max_num_pages * 4, (bs,), dtype=torch.int32)
        for n in (1, 4):
            ref = torch.zeros((g, bs * n), dtype=torch.int32)
            decode_write_locations(tables, page_sizes, seq_lens, ref, bs, n)
            out = torch.zeros((g, bs * n), dtype=torch.int32, device="cuda")
            decode_write_locations(
                tables.cuda(), page_sizes.cuda(), seq_lens.cuda(), out, bs, n
            )
            torch.testing.assert_close(out.cpu(), ref)
        prefix = torch.randint(0, 30, (bs,), dtype=torch.int32)
        new = torch.randint(1, 200, (bs,), dtype=torch.int32)
        total = int(new.sum())
        ref = extend_write_locations(tables, page_sizes, prefix, new, total)
        out = extend_write_locations(
            tables.cuda(), page_sizes.cuda(), prefix.cuda(), new.cuda(), total
        )
        torch.testing.assert_close(out.cpu(), ref)


class GroupTableStacksTest(unittest.TestCase):
    def _stacks(self, device="cpu", max_bs=4):
        specs = [
            GroupTableSpec(
                FULL, block_granularity=4, kernel_page_size=4, max_num_pages=3
            ),
            GroupTableSpec(
                SWA, block_granularity=4, kernel_page_size=2, max_num_pages=6
            ),
        ]
        return GroupTableStacks(
            specs, max_bs=max_bs, max_tokens_per_req=2, device=device
        )

    def test_fill_expands_pads_and_nulls_holes(self):
        stacks = self._stacks()
        raw = {
            FULL: torch.tensor([[5, 6, -1], [9, 0, 2]], dtype=torch.int32),
            SWA: torch.tensor([[5, 6, -1], [9, 0, 2]], dtype=torch.int32),
        }
        stacks.fill(bs=3, actual_bs=2, block_tables=raw)
        self.assertEqual(
            stacks.table(FULL, 3).tolist(), [[5, 6, 0], [9, 0, 2], [0, 0, 0]]
        )
        # ratio 2: page p -> 2p, 2p+1; holes / ragged pads stay on page 0.
        self.assertEqual(
            stacks.table(SWA, 3).tolist(),
            [[10, 11, 12, 13, 0, 0], [18, 19, 0, 0, 4, 5], [0, 0, 0, 0, 0, 0]],
        )

    def test_fill_rejects_short_tables(self):
        stacks = self._stacks()
        full = torch.ones((2, 3), dtype=torch.int32)
        with self.assertRaisesRegex(
            RuntimeError, "has 2 request rows for a live batch of 3"
        ):
            stacks.fill(3, 3, {FULL: full, SWA: full})

    def test_idle_fill_zeroes_every_request(self):
        stacks = self._stacks()
        stacks.tables.fill_(7)
        placeholder = torch.ones((4, 3), dtype=torch.int32)
        stacks.fill(
            bs=4, actual_bs=0, block_tables={FULL: placeholder, SWA: placeholder}
        )
        self.assertEqual(int(stacks.tables.abs().sum()), 0)

    def test_decode_and_extend_locations_follow_the_stack(self):
        stacks = self._stacks()
        raw = torch.tensor([[5, 6, 2]], dtype=torch.int32)
        stacks.fill(1, 1, {FULL: raw, SWA: raw})
        stacks.compute_decode_locations(1, torch.tensor([5], dtype=torch.int32), 1)
        # pos 4: full (P=4) page 6 slot 0 = 24; swa (P=2) page 12 slot 0 = 24.
        self.assertEqual(int(stacks.decode_locations(FULL, 1, 1)[0]), 24)
        self.assertEqual(int(stacks.decode_locations(SWA, 1, 1)[0]), 24)
        locs = stacks.extend_locations(
            torch.tensor([2], dtype=torch.int32),
            torch.tensor([3], dtype=torch.int32),
            3,
        )
        self.assertEqual(locs[FULL].tolist(), [22, 23, 24])
        self.assertEqual(locs[SWA].tolist(), [22, 23, 24])

    def test_decode_locations_view_is_address_stable(self):
        stacks = self._stacks()
        a = stacks.decode_locations(FULL, 2, 2)
        b = stacks.decode_locations(FULL, 2, 2)
        self.assertEqual(a.data_ptr(), b.data_ptr())
        with self.assertRaisesRegex(RuntimeError, "sized for 2 tokens"):
            stacks.decode_locations(FULL, 1, 3)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_fill_matches_torch_reference(self):
        torch.manual_seed(1)
        stacks_cuda = self._stacks("cuda", max_bs=8)
        stacks_ref = self._stacks("cuda", max_bs=8)
        bs, num_blocks = 5, 3
        # Bridge layout: per-group views of one packed storage (holes -1/0).
        packed = torch.randint(
            -1, 20, (2 * bs * num_blocks,), dtype=torch.int32, device="cuda"
        )
        full = packed[: bs * num_blocks].view(bs, num_blocks)
        swa = packed[bs * num_blocks :].view(bs, num_blocks)
        raw = {FULL: full, SWA: swa}
        stacks_cuda.fill(7, 5, raw)
        stacks_ref._fill_torch(7, 5, [full, swa])
        torch.testing.assert_close(stacks_cuda.tables, stacks_ref.tables)
        # Non-shared storage and a strided source take the same kernel path.
        stacks_cuda.tables.zero_()
        stacks_cuda.fill(7, 5, {FULL: full.clone(), SWA: swa.clone()})
        torch.testing.assert_close(stacks_cuda.tables, stacks_ref.tables)


class _StubLeaf(PagedAttentionBackend):
    """Records what the router hands it; no kernels."""

    def __init__(
        self, kernel_page_size: int, *, is_draft=False, spec=1, context_len=24
    ):
        config = SimpleNamespace(
            device="cpu",
            dtype=torch.float16,
            is_draft=is_draft,
            speculative_num_draft_tokens=spec,
            context_len=context_len,
        )
        component = SimpleNamespace(
            num_attention_heads=8, num_kv_heads=8, attn_tp_size=1, head_dim=16
        )
        super().__init__(config, component, kernel_page_size=kernel_page_size)
        self.calls: list[tuple] = []
        self.page_table_buf = None
        self.seq_lens_buf = None
        self.forward_decode_metadata = None
        self.chunked_prefill_metadata = "chunk-meta"

    def init_cuda_graph_state(self, max_bs):
        self.page_table_buf = torch.zeros(
            (max_bs, self.max_num_pages), dtype=torch.int32
        )
        self.seq_lens_buf = torch.zeros((max_bs,), dtype=torch.int32)

    @property
    def decode_seq_lens_buffer(self):
        return self.seq_lens_buf

    def init_forward_metadata(
        self, bs, num_extends, seq_lens, page_table, forward_mode, **kw
    ):
        self.calls.append(("init", bs, num_extends, page_table.clone(), forward_mode))
        self.last_init_kwargs = kw

    def set_request_slots(self, req_pool_indices):
        self.calls.append(("slots", req_pool_indices.clone()))

    def refresh_decode_metadata(
        self,
        bs,
        actual_bs,
        seq_lens,
        page_table,
        *,
        num_extends=0,
        for_graph_replay=False,
    ):
        self.page_table_buf[:bs].copy_(page_table)
        self.seq_lens_buf[:bs].copy_(seq_lens[:bs].clamp_min(self.verify_floor))
        self.forward_decode_metadata = SimpleNamespace(
            page_table=self.page_table_buf[:bs], seq_lens=self.seq_lens_buf[:bs]
        )
        self.calls.append(("refresh", bs, actual_bs, num_extends, for_graph_replay))

    def forward_decode(
        self, q, k, v, layer, out_cache_loc, pool, bs, save_kv_cache=True, **kw
    ):
        self.calls.append(("decode", layer.group_id, out_cache_loc))
        return q

    def forward_extend(
        self, q, k, v, layer, out_cache_loc, pool, bs, save_kv_cache=True, **kw
    ):
        self.calls.append(("extend", layer.group_id, out_cache_loc))
        return q


def _geometry():
    return CacheGroupGeometry(
        granularities={FULL: 4, SWA: 4, "linear_attention_0": 8},
        families={FULL: "history", SWA: "history", "linear_attention_0": "state"},
        full_history_group_id=FULL,
        row_geometry={FULL: (4, 1), SWA: (4, 1)},
        retentions={FULL: ("full_history", None), SWA: ("sliding_window", 4)},
    )


def _layer(group_id, layer_id=0):
    return SimpleNamespace(group_id=group_id, layer_id=layer_id)


class CacheGroupRouterTest(unittest.TestCase):
    def _router(self, *, is_draft=False, spec=1):
        leaves = {
            FULL: _StubLeaf(4, is_draft=is_draft, spec=spec),
            SWA: _StubLeaf(2, is_draft=is_draft, spec=spec),
        }
        router = CacheGroupRouter(
            None, is_draft=is_draft, spec_num_tokens=spec, device="cpu"
        )
        router.bind(_geometry(), leaves)
        router.init_cuda_graph_state(4)
        return router, leaves

    def _tables(self, bs=2):
        base = torch.tensor([[5, 6, -1], [9, 0, 2]], dtype=torch.int32)[:bs]
        return {
            FULL: base,
            SWA: base,
            "linear_attention_0": torch.ones((bs, 2), dtype=torch.int32),
        }

    def test_window_support_is_forwarded_from_the_leaves(self):
        router, leaves = self._router()
        self.assertFalse(router.supports_layer_sliding_window)

        for leaf in leaves.values():
            leaf.supports_layer_sliding_window = True
        self.assertTrue(router.supports_layer_sliding_window)

        leaves[SWA].supports_layer_sliding_window = False
        self.assertFalse(router.supports_layer_sliding_window)

    def test_rejects_non_history_groups(self):
        router = CacheGroupRouter(None, is_draft=False, spec_num_tokens=1, device="cpu")
        with self.assertRaisesRegex(ValueError, "family 'state'"):
            router.bind(_geometry(), {"linear_attention_0": _StubLeaf(8)})

    def test_refresh_hands_each_leaf_its_own_kernel_page_table(self):
        router, leaves = self._router()
        seq_lens = torch.tensor([9, 4, 1, 1], dtype=torch.int32)
        router.refresh_decode_metadata(
            4,
            2,
            torch.arange(4, dtype=torch.int32),
            seq_lens,
            forward_mode=ForwardMode.DECODE,
            block_tables=self._tables(),
        )
        self.assertEqual(
            leaves[FULL].page_table_buf.tolist(),
            [[5, 6, 0, 0, 0, 0], [9, 0, 2, 0, 0, 0], [0] * 6, [0] * 6],
        )
        self.assertEqual(
            leaves[SWA].page_table_buf.tolist(),
            [
                [10, 11, 12, 13, 0, 0, 0, 0, 0, 0, 0, 0],
                [18, 19, 0, 0, 4, 5, 0, 0, 0, 0, 0, 0],
                [0] * 12,
                [0] * 12,
            ],
        )
        self.assertEqual(leaves[FULL].calls[-2], ("refresh", 4, 2, 0, False))

    def test_forward_dispatches_by_group_with_group_write_locations(self):
        router, leaves = self._router()
        seq_lens = torch.tensor([9, 4], dtype=torch.int32)
        raw = torch.tensor([[5, 6, 7], [9, 0, 2]], dtype=torch.int32)
        tables = {
            FULL: raw,
            SWA: raw,
            "linear_attention_0": torch.ones((2, 2), dtype=torch.int32),
        }
        router.refresh_decode_metadata(
            2,
            2,
            torch.arange(2, dtype=torch.int32),
            seq_lens,
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
        )
        q = torch.zeros(2)
        router.forward(q, None, None, _layer(SWA), None, ForwardMode.DECODE, 2)
        kind, gid, loc = leaves[SWA].calls[-1]
        self.assertEqual((kind, gid), ("decode", SWA))
        # Slot math is page-size invariant: pos 8 of request 0 is raw page 7
        # slot 0 (= kernel page 14 slot 0 at P=2) -> 28; pos 3 of request 1 is
        # raw page 9 slot 3 -> 39. Both groups agree because their tables alias.
        self.assertEqual(loc.tolist(), [28, 39])
        self.assertIs(loc, router.write_locations(_layer(SWA), ForwardMode.DECODE))
        self.assertEqual(
            router.write_locations(_layer(FULL), ForwardMode.DECODE).tolist(), [28, 39]
        )
        self.assertFalse(any(c[0] == "decode" for c in leaves[FULL].calls))
        with self.assertRaisesRegex(KeyError, "names cache group 'nope'"):
            router.forward(q, None, None, _layer("nope"), None, ForwardMode.DECODE, 2)

    def test_decode_write_location_views_are_pointer_stable_per_bs(self):
        router, _ = self._router(spec=2)
        seq_lens = torch.tensor([9, 4, 1, 1], dtype=torch.int32)
        tables = self._tables()
        router.refresh_decode_metadata(
            4,
            2,
            torch.arange(4, dtype=torch.int32),
            seq_lens,
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
        )
        first = router.decode_write_locations
        router.refresh_decode_metadata(
            4,
            1,
            torch.arange(4, dtype=torch.int32),
            seq_lens,
            forward_mode=ForwardMode.DECODE,
            block_tables=self._tables(1),
        )
        self.assertIs(router.decode_write_locations, first)
        self.assertEqual(tuple(first.by_group[FULL].shape), (8,))
        # Verify window N=2 for request 0 (seq 9): positions 7, 8 -> 6*4+3, 6*4+... no:
        # kernel page for pos 7 is table[0][1]=6 -> 6*4+3=27; pos 8 -> table[0][2]=0 -> dummy.
        self.assertEqual(first.by_group[FULL][:2].tolist(), [27, 0])

    def test_draft_router_owns_decode_locations_too(self):
        # Draft and target routers share the write-location contract: the
        # refresh publishes the window and the forward fetches it (no
        # caller-supplied vector anywhere).
        router, leaves = self._router(is_draft=True)
        seq_lens = torch.tensor([9, 4], dtype=torch.int32)
        router.refresh_decode_metadata(
            2,
            2,
            torch.arange(2, dtype=torch.int32),
            seq_lens,
            forward_mode=ForwardMode.DECODE,
            block_tables=self._tables(),
        )
        self.assertIsNotNone(router.decode_write_locations)
        router.forward(
            torch.zeros(2),
            None,
            None,
            _layer(FULL),
            None,
            ForwardMode.DECODE,
            2,
        )
        self.assertIs(
            leaves[FULL].calls[-1][2],
            router.write_locations(_layer(FULL), ForwardMode.DECODE),
        )

    def test_live_decode_without_every_group_table_raises(self):
        router, _ = self._router()
        tables = self._tables()
        del tables[SWA]
        with self.assertRaisesRegex(
            RuntimeError, "missing cache groups \\['sliding_attention'\\]"
        ):
            router.refresh_decode_metadata(
                2,
                2,
                torch.arange(2, dtype=torch.int32),
                torch.tensor([9, 4], dtype=torch.int32),
                forward_mode=ForwardMode.DECODE,
                block_tables=tables,
            )
        # The idle replay has no live requests and tolerates any placeholder.
        router.refresh_decode_metadata(
            2,
            0,
            torch.arange(2, dtype=torch.int32),
            torch.tensor([1, 1], dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            block_tables={
                FULL: torch.zeros((2, 1), dtype=torch.int32),
                SWA: torch.zeros((2, 1), dtype=torch.int32),
            },
        )

    def test_extend_init_publishes_extend_spans(self):
        router, leaves = self._router()
        seq_lens = torch.tensor([9, 4], dtype=torch.int32)
        prefix = torch.tensor([4, 0], dtype=torch.int32)
        new = torch.tensor([5, 4], dtype=torch.int32)
        router.init_forward_metadata(
            2,
            2,
            torch.arange(2, dtype=torch.int32),
            seq_lens,
            ForwardMode.EXTEND,
            block_tables=self._tables(),
            extend_seq_lens=new,
            extend_seq_lens_cpu=new.clone(),
            extend_prefix_lens=prefix,
            extend_prefix_lens_cpu=prefix.clone(),
            extend_with_prefix=True,
        )
        kind, bs, num_extends, page_table, mode = leaves[FULL].calls[-2]
        self.assertEqual(
            (kind, bs, num_extends, mode), ("init", 2, 2, ForwardMode.EXTEND)
        )
        self.assertEqual(page_table.tolist(), [[5, 6, 0, 0, 0, 0], [9, 0, 2, 0, 0, 0]])
        # The prefix flag rides along with the extend lengths to every leaf:
        # FlashMLA sizes its paged-prefix plan by it, and a dropped flag
        # overran that plan in-kernel (chunked prefill IMA).
        for leaf in leaves.values():
            self.assertIs(leaf.last_init_kwargs["extend_with_prefix"], True)
            self.assertIs(leaf.last_init_kwargs["extend_prefix_lens"], prefix)
        locs = router.write_locations(_layer(FULL), ForwardMode.EXTEND)
        # req 0 positions 4..8 over pages [5,6,0]: 24,25,26,27, then page 0 -> 0.
        # req 1 positions 0..3 over page 9: 36..39.
        self.assertEqual(locs.tolist(), [24, 25, 26, 27, 0, 36, 37, 38, 39])
        router.forward(
            torch.zeros(9),
            torch.zeros(9),
            torch.zeros(9),
            _layer(FULL),
            None,
            ForwardMode.EXTEND,
            2,
        )
        self.assertIs(leaves[FULL].calls[-1][2], locs)
        no_extends = torch.zeros(0, dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "serves extend/mixed/idle"):
            router.init_forward_metadata(
                2,
                0,
                torch.arange(2, dtype=torch.int32),
                seq_lens,
                ForwardMode.DECODE,
                block_tables=self._tables(),
                extend_seq_lens=no_extends,
                extend_seq_lens_cpu=no_extends,
                extend_prefix_lens=no_extends,
                extend_prefix_lens_cpu=no_extends,
                extend_with_prefix=False,
            )

    def test_mixed_round_slices_decode_requests_after_the_extend_requests(self):
        router, leaves = self._router(spec=1)
        # Request 0 extends (prefix 4, 5 new tokens), request 1 decodes at
        # seq 4.
        seq_lens = torch.tensor([9, 4], dtype=torch.int32)
        router.init_forward_metadata(
            2,
            1,
            torch.arange(2, dtype=torch.int32),
            seq_lens,
            ForwardMode.MIXED,
            block_tables=self._tables(),
            extend_seq_lens=torch.tensor([5], dtype=torch.int32),
            extend_seq_lens_cpu=torch.tensor([5], dtype=torch.int32),
            extend_prefix_lens=torch.tensor([4], dtype=torch.int32),
            extend_prefix_lens_cpu=torch.tensor([4], dtype=torch.int32),
            extend_with_prefix=True,
        )
        self.assertEqual(
            router.write_locations(_layer(FULL), ForwardMode.EXTEND).tolist(),
            [24, 25, 26, 27, 0],
        )
        # Decode request 1: pos 3 on page 9 -> 39.
        self.assertEqual(
            router.write_locations(_layer(FULL), ForwardMode.DECODE).tolist(), [39]
        )
        router.forward(
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            _layer(FULL),
            None,
            ForwardMode.DECODE,
            1,
        )
        self.assertEqual(leaves[FULL].calls[-1][2].tolist(), [39])

    def test_draft_step_zero_local_decode_writes_the_full_extend_span(self):
        router, leaves = self._router(is_draft=True, spec=1)
        seq_lens = torch.tensor([9], dtype=torch.int32)
        extend_seq_lens = torch.tensor([5], dtype=torch.int32)
        extend_prefix_lens = torch.tensor([4], dtype=torch.int32)
        router.init_forward_metadata(
            1,
            1,
            torch.arange(1, dtype=torch.int32),
            seq_lens,
            ForwardMode.EXTEND,
            block_tables=self._tables(1),
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens.clone(),
            extend_prefix_lens=extend_prefix_lens,
            extend_prefix_lens_cpu=extend_prefix_lens.clone(),
            extend_with_prefix=True,
        )
        router.refresh_decode_metadata(
            1,
            1,
            torch.arange(1, dtype=torch.int32),
            seq_lens,
            forward_mode=ForwardMode.DECODE,
            block_tables=self._tables(1),
            num_extends=1,
        )
        router.forward(
            torch.zeros(1),
            torch.zeros(5),
            torch.zeros(5),
            _layer(FULL),
            None,
            ForwardMode.DECODE,
            1,
        )
        self.assertEqual(leaves[FULL].calls[-1][2].tolist(), [24, 25, 26, 27, 0])

    def test_draft_step_zero_one_token_extend_ignores_q_k_shape_equality(self):
        router, leaves = self._router(is_draft=True, spec=1)
        seq_lens = torch.tensor([5], dtype=torch.int32)
        one = torch.ones(1, dtype=torch.int32)
        prefix = torch.tensor([4], dtype=torch.int32)
        router.init_forward_metadata(
            1,
            1,
            torch.arange(1, dtype=torch.int32),
            seq_lens,
            ForwardMode.EXTEND,
            block_tables=self._tables(1),
            extend_seq_lens=one,
            extend_seq_lens_cpu=one.clone(),
            extend_prefix_lens=prefix,
            extend_prefix_lens_cpu=prefix.clone(),
            extend_with_prefix=True,
        )
        router.refresh_decode_metadata(
            1,
            1,
            torch.arange(1, dtype=torch.int32),
            seq_lens,
            forward_mode=ForwardMode.DECODE,
            block_tables=self._tables(1),
            num_extends=1,
        )
        q = torch.zeros(1)
        router.forward_decode(q, q, q, _layer(FULL), None, 1)
        self.assertEqual(leaves[FULL].calls[-1][2].tolist(), [24])

    def test_draft_step_zero_mixed_decode_composes_both_windows(self):
        router, leaves = self._router(is_draft=True, spec=1)
        seq_lens = torch.tensor([9, 4], dtype=torch.int32)
        extend_seq_lens = torch.tensor([5], dtype=torch.int32)
        extend_prefix_lens = torch.tensor([4], dtype=torch.int32)
        router.init_forward_metadata(
            2,
            1,
            torch.arange(2, dtype=torch.int32),
            seq_lens,
            ForwardMode.MIXED,
            block_tables=self._tables(),
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens.clone(),
            extend_prefix_lens=extend_prefix_lens,
            extend_prefix_lens_cpu=extend_prefix_lens.clone(),
            extend_with_prefix=True,
        )
        router.refresh_decode_metadata(
            2,
            2,
            torch.arange(2, dtype=torch.int32),
            seq_lens,
            forward_mode=ForwardMode.DECODE,
            block_tables=self._tables(),
            num_extends=1,
        )
        router.forward(
            torch.zeros(2),
            torch.zeros(6),
            torch.zeros(6),
            _layer(FULL),
            None,
            ForwardMode.DECODE,
            2,
        )
        self.assertEqual(leaves[FULL].calls[-1][2].tolist(), [24, 25, 26, 27, 0, 39])

        later = router.publish_draft_step_locations(
            cache_start=torch.tensor([4, 0], dtype=torch.int32), num_tokens=1
        )
        router.forward(
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2),
            _layer(FULL),
            None,
            ForwardMode.DECODE,
            2,
        )
        self.assertEqual(leaves[FULL].calls[-1][2].data_ptr(), later.data_ptr())
        self.assertEqual(later.tolist(), [24, 36])

    def test_capture_seeds_with_idle_rows_then_calls_leaf_capture_hooks(self):
        router, leaves = self._router()
        placeholder = {
            FULL: torch.ones((4, 3), dtype=torch.int32),
            SWA: torch.ones((4, 3), dtype=torch.int32),
        }
        router.init_forward_metadata_capture_cuda_graph(
            4,
            torch.arange(4, dtype=torch.int32),
            torch.ones(4, dtype=torch.int32),
            ForwardMode.DECODE,
            block_tables=placeholder,
        )
        self.assertEqual(leaves[FULL].calls[-2], ("refresh", 4, 0, 0, True))
        self.assertEqual(int(leaves[FULL].page_table_buf.abs().sum()), 0)
        self.assertEqual(
            router.decode_write_locations.by_group[SWA].tolist(), [0, 0, 0, 0]
        )

    def test_request_slots_reach_every_leaf_after_each_metadata_build(self):
        # The one side channel beyond page_table/seq_lens: leaves owning
        # per-request side state (DSA's KPool tails) learn this forward's
        # pool slots after the build that could have reset them.
        router, leaves = self._router()
        req = torch.tensor([7, 3], dtype=torch.int32)
        seq = torch.tensor([5, 9], dtype=torch.int32)
        router.refresh_decode_metadata(
            2,
            2,
            req,
            seq,
            forward_mode=ForwardMode.DECODE,
            block_tables=self._tables(),
            num_extends=0,
            for_graph_replay=False,
        )
        for leaf in leaves.values():
            kinds = [c[0] for c in leaf.calls]
            self.assertEqual(kinds[-2:], ["refresh", "slots"])
            self.assertEqual(leaf.calls[-1][1].tolist(), [7, 3])

    def test_single_group_model_surface_proxies_to_the_sole_leaf(self):
        leaf = _StubLeaf(4)
        router = CacheGroupRouter(None, is_draft=False, spec_num_tokens=1, device="cpu")
        router.bind(_geometry(), {FULL: leaf})
        self.assertEqual(router.chunked_prefill_metadata, "chunk-meta")
        multi, _ = self._router()
        with self.assertRaisesRegex(RuntimeError, "single attention cache group"):
            _ = multi.chunked_prefill_metadata

    def test_draft_write_locations_ride_the_history_stack(self):
        """The drafters' in-graph slot math reads the router's address-stable
        full-history table in the stack: page-size invariant vs the raw table,
        and the view survives refreshes at the same address (the capture
        contract the deleted DraftPageStaging carried)."""
        router, _ = self._router(is_draft=True)
        raw = torch.tensor([[5, 6, 2], [9, 3, 7]], dtype=torch.int32)
        tables = {
            FULL: raw,
            SWA: raw,
            "linear_attention_0": torch.ones((2, 2), dtype=torch.int32),
        }
        router.refresh_decode_metadata(
            2,
            2,
            torch.arange(2, dtype=torch.int32),
            torch.tensor([9, 4], dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
        )
        view = router.draft_history_view()
        self.assertEqual(view.page_size, 4)  # FULL leaf kernel page
        # FULL leaf: max_num_pages = ceil(context_len 24 / kernel page 4) = 6.
        self.assertEqual(view.max_tokens, 6 * 4)
        out = torch.zeros(4, dtype=torch.int32)
        router.draft_write_locations_uniform(
            out, cache_start=torch.tensor([4, 7], dtype=torch.int32), num_tokens=2
        )
        # Raw-table math (P=4): req0 pos 4,5 -> raw page 6 slots 0,1 = 24,25;
        # req1 pos 7 -> page 3 slot 3 = 15, pos 8 -> page 7 slot 0 = 28.
        self.assertEqual(out.tolist(), [24, 25, 15, 28])
        # Address stability across refreshes: the same storage is rewritten.
        ptr = view.table.data_ptr()
        router.refresh_decode_metadata(
            2,
            2,
            torch.arange(2, dtype=torch.int32),
            torch.tensor([9, 4], dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
        )
        self.assertEqual(router.draft_history_view().table.data_ptr(), ptr)

    def test_draft_hooks_fan_out_to_every_leaf(self):
        router, leaves = self._router()
        router.advance_draft_forward_metadata(torch.tensor([3, 5], dtype=torch.int32))
        for leaf in leaves.values():
            self.assertEqual(leaf.seq_lens_buf[:2].tolist(), [3, 5])
        router.update_draft_forward_metadata(torch.tensor([7, 7], dtype=torch.int32))
        self.assertEqual(leaves[SWA].seq_lens_buf[:2].tolist(), [7, 7])
        self.assertEqual(router.child_backends(), (leaves[FULL], leaves[SWA]))

    def test_every_metadata_build_clears_the_sparse_topk_share(self):
        """The sparse layers' shared selection is per forward: extend init,
        decode refresh and capture seeding each start it empty, so a "shared"
        layer can never consume the previous forward's top-k."""
        router, _ = self._router()
        share = router.sparse_topk
        self.assertIs(share, router.sparse_topk, "one share per node")

        def publish():
            share.prefill, share.decode = object(), object()

        publish()
        seq_lens = torch.tensor([9, 4], dtype=torch.int32)
        new = torch.tensor([3, 1], dtype=torch.int32)
        prefix = torch.zeros(2, dtype=torch.int32)
        router.init_forward_metadata(
            2,
            2,
            torch.arange(2, dtype=torch.int32),
            seq_lens,
            ForwardMode.EXTEND,
            block_tables=self._tables(),
            extend_seq_lens=new,
            extend_seq_lens_cpu=new.clone(),
            extend_prefix_lens=prefix,
            extend_prefix_lens_cpu=prefix.clone(),
            extend_with_prefix=False,
        )
        self.assertEqual((share.prefill, share.decode), (None, None))

        publish()
        router.refresh_decode_metadata(
            2,
            2,
            torch.arange(2, dtype=torch.int32),
            seq_lens,
            forward_mode=ForwardMode.DECODE,
            block_tables=self._tables(),
        )
        self.assertEqual((share.prefill, share.decode), (None, None))

        publish()
        router.init_forward_metadata_capture_cuda_graph(
            2,
            torch.arange(2, dtype=torch.int32),
            torch.ones(2, dtype=torch.int32),
            ForwardMode.DECODE,
            block_tables=self._tables(),
        )
        self.assertEqual((share.prefill, share.decode), (None, None))

        # The drafter's in-loop seq_lens edit is not a new forward: it leaves
        # the share it just attached alone.
        publish()
        kept = (share.prefill, share.decode)
        router.advance_draft_forward_metadata(torch.ones(2, dtype=torch.int32))
        self.assertEqual((share.prefill, share.decode), kept)


def _spec(
    group_id,
    family="history",
    granularity=4,
    retention="full_history",
    rows_per_page=None,
    entry_stride_tokens=None,
    sliding_window_tokens=None,
):
    return SimpleNamespace(
        group_id=group_id,
        family=family,
        retention=retention,
        block_granularity=granularity,
        rows_per_page=rows_per_page,
        entry_stride_tokens=entry_stride_tokens,
        sliding_window_tokens=sliding_window_tokens,
    )


def _pool(*specs, paged=(FULL, SWA)):
    return SimpleNamespace(
        arena=SimpleNamespace(cache_group_specs=tuple(specs)),
        paged_group_ids=tuple(paged),
    )


class CacheGroupRouterRebindTest(unittest.TestCase):
    """A same-geometry rebind must keep the leaves it already initialised."""

    def _router(self):
        built = []

        def factory(group_id, granularity):
            leaf = _StubLeaf(granularity)
            built.append(group_id)
            return leaf

        router = CacheGroupRouter(
            factory, is_draft=False, spec_num_tokens=1, device="cpu"
        )
        return router, built

    def test_same_geometry_rebind_keeps_leaves_and_their_state(self):
        router, built = self._router()
        specs = (_spec(FULL), _spec(SWA), _spec("linear_attention_0", "state", 8))
        router.set_cache_pool(_pool(*specs))
        self.assertEqual(sorted(built), sorted([FULL, SWA]))
        first = dict(router.leaves)
        counter = object()
        router.register_step_counter(counter)
        router._stacks = object()
        router.decode_write_locations = object()

        bigger = _pool(*specs)  # a different pool object, the same published geometry
        router.set_cache_pool(bigger)

        self.assertEqual(len(built), 2, "leaves must not be rebuilt")
        self.assertIsNone(router._stacks)
        self.assertIsNone(router.decode_write_locations)
        self.assertIsNone(router._extend_write_locations)
        self.assertEqual(router._decode_views, {})
        self.assertEqual(router._decode_request_offset, 0)
        for gid, leaf in router.leaves.items():
            self.assertIs(leaf, first[gid])
            self.assertIs(leaf.step_counter, counter)
            self.assertIs(leaf.cache_pool, bigger)
        self.assertIs(router.cache_pool, bigger)

    def test_changed_geometry_rebind_is_rejected(self):
        router, built = self._router()
        original = _pool(_spec(FULL), _spec(SWA))
        router.set_cache_pool(original)
        first = dict(router.leaves)

        with self.assertRaises(RuntimeError):
            router.set_cache_pool(_pool(_spec(FULL, granularity=8), _spec(SWA)))

        self.assertEqual(len(built), 2)
        self.assertEqual(router.leaves, first)
        self.assertIs(router.cache_pool, original)
        router.set_cache_pool(original)
        self.assertEqual(router.leaves, first)

    def test_same_span_but_different_row_geometry_is_rejected(self):
        """Equal products do not make two published row geometries identical."""
        router, _ = self._router()
        original = _pool(
            _spec(FULL, granularity=4, rows_per_page=4, entry_stride_tokens=1),
            paged=(FULL,),
        )
        changed = _pool(
            _spec(FULL, granularity=4, rows_per_page=2, entry_stride_tokens=2),
            paged=(FULL,),
        )
        router.set_cache_pool(original)

        with self.assertRaisesRegex(RuntimeError, "different geometry"):
            router.set_cache_pool(changed)

        self.assertIs(router.cache_pool, original)

    def test_reordered_full_history_groups_match_a_fresh_bind(self):
        """Spec declaration order is not published geometry."""
        probe_specs = (_spec(FULL), _spec("full_history_1"))
        real_specs = tuple(reversed(probe_specs))
        rebound, _ = self._router()
        rebound.set_cache_pool(_pool(*probe_specs, paged=(FULL, "full_history_1")))

        rebound.set_cache_pool(_pool(*real_specs, paged=("full_history_1", FULL)))

        fresh, _ = self._router()
        fresh.set_cache_pool(_pool(*real_specs, paged=("full_history_1", FULL)))
        self.assertEqual(rebound.geometry, fresh.geometry)
        self.assertEqual(
            rebound.geometry.full_history_group_id,
            fresh.geometry.full_history_group_id,
        )

    def test_rebind_rejects_a_changed_sliding_window(self):
        """The window length is part of the scheduler's retention contract."""
        router, _ = self._router()
        original = _pool(
            _spec(FULL, retention="full_history"),
            _spec(SWA, retention="sliding_window", sliding_window_tokens=128),
        )
        changed = _pool(
            _spec(FULL, retention="full_history"),
            _spec(SWA, retention="sliding_window", sliding_window_tokens=256),
        )
        router.set_cache_pool(original)

        with self.assertRaisesRegex(RuntimeError, "different geometry"):
            router.set_cache_pool(changed)

        self.assertIs(router.cache_pool, original)

    def test_rebind_rejects_a_changed_group_retention_contract(self):
        """A group's allocation/retention meaning is part of its published contract."""
        router, _ = self._router()
        original = _pool(
            _spec(FULL, retention="full_history"),
            _spec(SWA, retention="sliding_window", sliding_window_tokens=128),
        )
        changed = _pool(
            _spec(FULL, retention="full_history"),
            _spec(SWA, retention="full_history"),
        )
        router.set_cache_pool(original)

        with self.assertRaisesRegex(RuntimeError, "different geometry"):
            router.set_cache_pool(changed)

        self.assertIs(router.cache_pool, original)

    def test_the_alias_gate_sees_through_modules(self):
        """Registered side backends are nn.Modules; a view they keep must be found."""
        slab = torch.zeros(4, 4)

        class _Side(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._verify_scratch = slab[1:3]

        node = SimpleNamespace(_speculative_state_backends=[_Side()])
        self.assertEqual(len(reachable_tensors(node)), 1)
        with self.assertRaisesRegex(AssertionError, "_speculative_state_backends"):
            assert_no_alias(node, storages_of(slab))

    def test_a_rebound_router_matches_a_fresh_one_field_by_field(self):
        specs = (_spec(FULL), _spec(SWA))
        rebound, _ = self._router()
        rebound.set_cache_pool(_pool(*specs))
        rebound.init_cuda_graph_state(2)
        old_tensors = [
            t
            for node in (rebound, *rebound.leaves.values())
            for t in reachable_tensors(node)
        ]
        old = storages_of(*old_tensors)
        # A served round's per-forward state, which the rebind must forget.
        rebound._decode_views[(1, 1)] = object()
        rebound.decode_write_locations = object()
        rebound._extend_write_locations = {FULL: torch.zeros(1)}
        rebound._decode_request_offset = 3
        rebound.set_cache_pool(_pool(*specs))
        rebound.init_cuda_graph_state(2)
        fresh, _ = self._router()
        fresh.set_cache_pool(_pool(*specs))
        fresh.init_cuda_graph_state(2)

        self.assertEqual(binding_state(rebound), binding_state(fresh))
        for node in (rebound, *rebound.leaves.values()):
            assert_no_alias(node, old)
        self.assertGreaterEqual(len(old_tensors), 2 * len(rebound.leaves))

    def test_metadata_before_the_re_init_fails_loudly(self):
        router, _ = self._router()
        specs = (_spec(FULL), _spec(SWA))
        router.set_cache_pool(_pool(*specs))
        router.init_cuda_graph_state(2)

        router.set_cache_pool(_pool(*specs))

        with self.assertRaisesRegex(RuntimeError, "init_cuda_graph_state must run"):
            router.stacks

    def test_rebound_leaves_are_as_unbound_as_fresh_ones(self):
        """Graph buffers and views only come back with init_cuda_graph_state."""
        router, _ = self._router()
        specs = (_spec(FULL), _spec(SWA))
        router.set_cache_pool(_pool(*specs))
        router.init_cuda_graph_state(2)
        for leaf in router.leaves.values():
            leaf._decode_views_by_bs[1] = object()

        router.set_cache_pool(_pool(*specs))

        for leaf in router.leaves.values():
            self.assertIsNone(leaf.page_table_buf)
            self.assertIsNone(leaf.seq_lens_buf)
            self.assertEqual(leaf._decode_views_by_bs, {})


class PagedLeafRebindTest(unittest.TestCase):
    """Every leaf forgets the per-forward and per-graph state it annotates as optional.

    The gate reads ``self.<name>: <Type> | None = None`` annotations in ``__init__``
    whose name mentions ``metadata`` or ends in ``_buf``; state declared another way
    (DSA's page-table alias, FlashMLA's keepalive list) gets its own test below.
    """

    def _optional_slots(self, backend_cls):
        import inspect

        init = ast.parse(textwrap.dedent(inspect.getsource(backend_cls.__init__)))
        slots = []
        for node in ast.walk(init):
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Attribute)
                and isinstance(node.target.value, ast.Name)
                and node.target.value.id == "self"
                and ast.unparse(node.annotation).endswith("| None")
            ):
                slots.append(node.target.attr)
        return slots

    def _check(self, backend_cls, expected, **publish_reads):
        slots = self._optional_slots(backend_cls)
        self.assertTrue(slots)
        self.assertEqual(sorted(slots), sorted(expected))
        leaf = backend_cls.__new__(backend_cls)
        leaf._init_pool_binding()
        leaf.cache_pool = object()
        for name, value in publish_reads.items():
            setattr(leaf, name, value)
        stale = object()
        for name in slots:
            setattr(leaf, name, stale)
        pool = object()

        leaf.set_cache_pool(pool)

        self.assertIs(leaf.cache_pool, pool)
        for name in slots:
            self.assertIsNone(getattr(leaf, name), name)

    def test_mha_leaf_forgets_the_previous_forward(self):
        from tokenspeed.runtime.layers.attention.backends.paged.mha import (
            MHAAttnBackend,
        )

        self._check(
            MHAAttnBackend, ["forward_decode_metadata", "forward_extend_metadata"]
        )

    def test_msa_leaf_forgets_the_previous_forward(self):
        from tokenspeed.runtime.layers.attention.backends.paged.msa import (
            MSAAttnBackend,
        )

        self._check(
            MSAAttnBackend, ["forward_decode_metadata", "forward_extend_metadata"]
        )

    def test_mla_leaf_forgets_the_previous_forward(self):
        from tokenspeed.runtime.layers.attention.backends.paged.mla import (
            MLAAttnBackend,
        )

        self._check(
            MLAAttnBackend,
            [
                "forward_decode_metadata",
                "forward_prefill_metadata",
                "chunked_prefill_metadata",
                "_block_page_table_buf",
                "_block_seq_lens_buf",
            ],
        )

    def test_trtllm_mha_leaf_forgets_the_previous_forward(self):
        from tokenspeed.runtime.layers.attention.backends.paged.trtllm import (
            TRTLLMMHAAttnBackend,
        )

        self._check(
            TRTLLMMHAAttnBackend,
            [
                "forward_prefill_metadata",
                "forward_decode_metadata",
                "spec_cache_seqlens_buf",
            ],
        )

    def test_trtllm_mla_leaf_forgets_the_previous_forward(self):
        from tokenspeed.runtime.layers.attention.backends.paged.trtllm_mla import (
            TRTLLMMLABackend,
        )

        self._check(
            TRTLLMMLABackend,
            [
                "forward_decode_metadata",
                "forward_prefill_metadata",
                "chunked_prefill_metadata",
            ],
        )

    @unittest.skipUnless(
        torch.version.hip is None, "FlashInfer wrappers are NVIDIA-only"
    )
    def test_flashmla_leaf_forgets_the_previous_forward(self):
        from tokenspeed.runtime.layers.attention.backends.paged.flashmla import (
            FlashMLABackend,
        )

        self._check(
            FlashMLABackend,
            [
                "forward_decode_metadata",
                "forward_prefill_metadata",
                "chunked_prefill_metadata",
                "_decode_tile_metadata",
            ],
            workspace_buffer=torch.empty(1 << 20, dtype=torch.uint8, device="cuda"),
            indices_updater_prefill=SimpleNamespace(prefill_wrapper_ragged=None),
        )

    def test_trtllm_leaf_forgets_the_verify_views(self):
        from tokenspeed.runtime.layers.attention.backends.paged.trtllm import (
            TRTLLMMHAAttnBackend,
        )

        leaf = TRTLLMMHAAttnBackend.__new__(TRTLLMMHAAttnBackend)
        leaf._init_pool_binding()
        leaf._verify_views_by_bs = {1: object()}
        leaf.set_cache_pool(object())
        self.assertEqual(leaf._verify_views_by_bs, {})

    @unittest.skipUnless(
        torch.version.hip is None, "FlashInfer wrappers are NVIDIA-only"
    )
    def test_flashmla_leaf_forgets_the_tile_keepalives(self):
        from tokenspeed.runtime.layers.attention.backends.paged.flashmla import (
            FlashMLABackend,
        )

        leaf = FlashMLABackend.__new__(FlashMLABackend)
        leaf._init_pool_binding()
        leaf.cache_pool = object()
        leaf._decode_tile_metadata_keepalive = [object()]
        leaf.workspace_buffer = torch.empty(1 << 20, dtype=torch.uint8, device="cuda")
        leaf.prefill_wrapper_ragged = leaf.prefill_wrapper_paged = object()
        leaf.indices_updater_prefill = SimpleNamespace(prefill_wrapper_ragged=object())
        leaf.set_cache_pool(object())
        self.assertEqual(leaf._decode_tile_metadata_keepalive, [])
        self.assertIsNot(leaf.prefill_wrapper_paged, leaf.prefill_wrapper_ragged)
        self.assertIs(
            leaf.indices_updater_prefill.prefill_wrapper_ragged,
            leaf.prefill_wrapper_ragged,
        )

    def test_cutedsl_mla_leaf_forgets_the_previous_forward(self):
        from tokenspeed.runtime.layers.attention.backends.paged.tokenspeed_mla import (
            CuteDSLMLABackend,
        )

        self._check(
            CuteDSLMLABackend,
            [
                "forward_decode_metadata",
                "forward_prefill_metadata",
                "chunked_prefill_metadata",
                "_block_page_table_buf",
                "_block_seq_lens_buf",
            ],
        )

    def test_dsa_forgets_the_prefill_page_table_alias(self):
        from tokenspeed.runtime.layers.attention.backends.paged.dsa import DSABackend

        backend = DSABackend.__new__(DSABackend)
        backend._init_pool_binding()
        backend._dense_backend = SimpleNamespace(
            validate_cache_pool=lambda pool: None,
            set_cache_pool=lambda pool: None,
            _bind=lambda pool: None,
        )
        backend._prefill_page_table = object()
        backend.kpool_runtime = SimpleNamespace(prefill_plan=object())
        backend.kpool_runtime.reset_forward = lambda req_pool_indices: setattr(
            backend.kpool_runtime, "prefill_plan", None
        )
        pool = object()

        backend.set_cache_pool(pool)

        self.assertIs(backend.cache_pool, pool)
        self.assertIsNone(backend._prefill_page_table)
        self.assertIsNone(backend.kpool_runtime.prefill_plan)


if __name__ == "__main__":
    unittest.main()
