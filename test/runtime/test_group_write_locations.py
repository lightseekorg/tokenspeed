"""Write-location seams the router suite does not pin.

test_cache_group_router.py owns the core slot math and the router's
publication discipline; this file keeps the residue that still adds value:
the TOKENSPEED_CACHE_DEBUG checker (``check_write_locations``), hole /
overflow routing edge cases of the pure functions, the MTP re-anchor
(``update_draft_forward_metadata`` + recompute over the same stacks), and
the base backend's refusal to serve write locations it does not own.
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

import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.cache_group_geometry import (
    CacheGroupGeometry,
)
from tokenspeed.runtime.layers.attention.backends.paged import (
    PagedAttentionBackend,
)
from tokenspeed.runtime.layers.attention.backends.router import CacheGroupRouter
from tokenspeed.runtime.layers.attention.backends.write_locations import (
    check_write_locations,
    decode_write_locations,
    extend_write_locations,
)

FULL = "full_attention"
SWA = "sliding_attention"


class CheckWriteLocationsTest(unittest.TestCase):
    """The TOKENSPEED_CACHE_DEBUG assertion: every slot must land in a real
    page of the group's table (the caller gates on the env var; the checker
    itself is unconditional)."""

    def _front_hole_table(self):
        # SWA-style table: front hole (slid-out page 0) then real pages 2, 3.
        return torch.tensor([[0, 2, 3, 0]], dtype=torch.int32)

    def test_valid_locs_pass(self):
        # Pages 2 and 3 are real entries; the hole 0 is excluded.
        locs = torch.tensor([4, 5, 6], dtype=torch.int32)
        check_write_locations(self._front_hole_table(), locs, 2, what="swa decode")

    def test_rejects_write_into_null_page(self):
        # loc 1 -> page 0 = the slid-out hole; a write there is corruption.
        bad = torch.tensor([1], dtype=torch.int32)
        with self.assertRaisesRegex(AssertionError, "null page"):
            check_write_locations(self._front_hole_table(), bad, 2, what="swa decode")

    def test_rejects_page_outside_table(self):
        # loc 18 -> page 9, not in the table.
        bad = torch.tensor([18], dtype=torch.int32)
        with self.assertRaisesRegex(AssertionError, "escape"):
            check_write_locations(self._front_hole_table(), bad, 2, what="swa decode")

    def test_honors_kernel_page_size(self):
        # Same locs read differently at P=4: 11 -> page 2, 15 -> page 3.
        table = torch.tensor([[2, 3]], dtype=torch.int32)
        locs = torch.tensor([11, 15], dtype=torch.int32)
        check_write_locations(table, locs, 4, what="wide")
        with self.assertRaisesRegex(AssertionError, "escape"):
            check_write_locations(table, locs, 2, what="wide")


class HoleOverflowRoutingTest(unittest.TestCase):
    """Holes (page id <= 0) and positions past the table route to slot 0,
    the zero-initialized dummy page that never aliases a live request."""

    def test_decode_position_in_a_hole_routes_to_slot_zero(self):
        # Front hole at column 0 (P=2): seq_len 2 -> pos 1 -> page 0 -> slot 0;
        # seq_len 5 -> pos 4 -> real page 7 -> 14.
        tables = torch.tensor([[[0, 5, 7, 0]]], dtype=torch.int32)
        page_sizes = torch.tensor([2], dtype=torch.int32)
        out = torch.zeros((1, 2), dtype=torch.int32)
        decode_write_locations(
            tables, page_sizes, torch.tensor([2], dtype=torch.int32), out, 1, 1
        )
        self.assertEqual(int(out[0, 0]), 0)
        decode_write_locations(
            tables, page_sizes, torch.tensor([5], dtype=torch.int32), out, 1, 1
        )
        self.assertEqual(int(out[0, 0]), 14)

    def test_ragged_minus_one_entry_routes_to_slot_zero(self):
        # A raw -1 (ragged pad that never went through the fill) is a hole too.
        tables = torch.tensor([[[3, -1]]], dtype=torch.int32)
        page_sizes = torch.tensor([2], dtype=torch.int32)
        out = torch.zeros((1, 1), dtype=torch.int32)
        decode_write_locations(
            tables, page_sizes, torch.tensor([4], dtype=torch.int32), out, 1, 1
        )
        self.assertEqual(int(out[0, 0]), 0)

    def test_extend_span_crossing_hole_and_table_end(self):
        # P=2, width 2, pages [3, 0]: positions 2,3 hit the hole, 4.. overflow.
        tables = torch.tensor([[[3, 0]]], dtype=torch.int32)
        page_sizes = torch.tensor([2], dtype=torch.int32)
        prefix = torch.tensor([1], dtype=torch.int32)
        new = torch.tensor([4], dtype=torch.int32)
        locs = extend_write_locations(tables, page_sizes, prefix, new, 4)
        # pos 1 -> page 3 slot 1 = 7; pos 2, 3 -> hole page 0 -> 0; pos 4 ->
        # page index 2 >= width -> 0.
        self.assertEqual(locs[0].tolist(), [7, 0, 0, 0])


class _StubLeaf(PagedAttentionBackend):
    """Buffer-only leaf (no kernels) for the router re-anchor test."""

    def __init__(self, kernel_page_size: int, *, is_draft: bool):
        config = SimpleNamespace(
            device="cpu",
            dtype=torch.float16,
            is_draft=is_draft,
            speculative_num_draft_tokens=1,
            context_len=24,
        )
        component = SimpleNamespace(
            num_attention_heads=8, num_kv_heads=8, attn_tp_size=1, head_dim=16
        )
        super().__init__(config, component, kernel_page_size=kernel_page_size)

    def init_cuda_graph_state(self, max_bs):
        self.seq_lens_buf = torch.zeros((max_bs,), dtype=torch.int32)
        self.page_table_buf = torch.zeros(
            (max_bs, self.max_num_pages), dtype=torch.int32
        )

    @property
    def decode_seq_lens_buffer(self):
        return self.seq_lens_buf

    def init_forward_metadata(self, *args, **kwargs):
        pass

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

    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError


class MtpReanchorTest(unittest.TestCase):
    """Vanilla MTP re-anchors the draft rows to the committed frontier:
    ``update_draft_forward_metadata`` republishes the frontier into every
    leaf's seq-lens buffer, and the next recompute over the SAME stacks
    derives the decode window from it."""

    def _router(self):
        leaves = {
            FULL: _StubLeaf(4, is_draft=True),
            SWA: _StubLeaf(2, is_draft=True),
        }
        router = CacheGroupRouter(None, is_draft=True, spec_num_tokens=1, device="cpu")
        geometry = CacheGroupGeometry(
            granularities={FULL: 4, SWA: 4},
            families={FULL: "history", SWA: "history"},
            full_history_group_id=FULL,
            history_block_granularity=4,
        )
        router.bind(geometry, leaves)
        router.init_cuda_graph_state(4)
        return router, leaves

    def test_reanchor_updates_leaf_seq_lens_and_recompute_follows(self):
        router, leaves = self._router()
        raw = torch.tensor([[5, 6, 7], [9, 8, 3]], dtype=torch.int32)
        router.refresh_decode_metadata(
            2,
            2,
            None,
            torch.tensor([9, 7], dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            block_tables={FULL: raw, SWA: raw},
        )
        frontier = torch.tensor([6, 9], dtype=torch.int32)
        router.update_draft_forward_metadata(frontier)
        for leaf in leaves.values():
            self.assertEqual(leaf.seq_lens_buf[:2].tolist(), [6, 9])
        # Recompute over the unchanged stacks: the decode window is now the
        # frontier's last position, not the pre-verify one.
        router.stacks.compute_decode_locations(2, frontier, 1)
        # FULL (P=4): pos 5 -> page 6 slot 1 = 25; pos 8 -> page 3 slot 0 = 12.
        self.assertEqual(router.stacks.decode_locations(FULL, 2, 1).tolist(), [25, 12])
        # SWA (P=2, raw grain 4 -> ratio 2): pos 5 -> kernel page 12 slot 1 =
        # 25; pos 8 -> kernel page 6 slot 0 = 12 (same slots: page-size
        # invariant math over aliasing tables).
        self.assertEqual(router.stacks.decode_locations(SWA, 2, 1).tolist(), [25, 12])


class BaseWriteLocationsDefaultTest(unittest.TestCase):
    """A backend that owns no paged write locations must refuse the accessor
    loudly — there is no caller-supplied fallback vector anymore."""

    def test_default_write_locations_raises(self):
        from tokenspeed.runtime.layers.attention.backends.base import (
            AttentionBackend,
        )

        class _MinimalBackend(AttentionBackend):
            pass

        backend = _MinimalBackend.__new__(_MinimalBackend)  # skip __init__
        with self.assertRaisesRegex(NotImplementedError, "write locations"):
            backend.write_locations(
                SimpleNamespace(group_id="full_attention"), ForwardMode.DECODE
            )


if __name__ == "__main__":
    unittest.main()
