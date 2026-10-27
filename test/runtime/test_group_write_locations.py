"""Write-location seams the router suite does not pin.

test_cache_group_router.py owns the core slot math and the router's
publication discipline; this file keeps the residue that still adds value:
hole / overflow routing edge cases of the pure functions, the MTP re-anchor
(``update_draft_forward_metadata`` + recompute over the same stacks), and
the base backend's refusal to serve write locations it does not own.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.paged.base import (
    PagedAttentionBackend,
)
from tokenspeed.runtime.layers.attention.backends.paged.cache_group_geometry import (
    CacheGroupGeometry,
)
from tokenspeed.runtime.layers.attention.backends.paged.router import CacheGroupRouter
from tokenspeed.runtime.layers.attention.backends.paged.write_locations import (
    decode_write_locations,
    extend_write_locations,
    window_write_locations,
)

FULL = "full_attention"
SWA = "sliding_attention"


class TestWindowWriteLocationsTest:
    def test_window_writes_its_slots_from_its_start(self):
        # Pages [3, 5, 0] (P=4): windows inside, across, over the hole, past the table and at the int32 limit.
        tables = torch.tensor([3, 5, 0], dtype=torch.int32).expand(2, 5, 3).contiguous()
        page_sizes = torch.tensor([4, 4], dtype=torch.int32)
        starts = torch.tensor([0, 2, 6, 10, 2**31 - 2], dtype=torch.int32)
        expected = {
            1: [12, 14, 22, 0, 0],
            4: [12, 13, 14, 15, 14, 15, 20, 21, 22, 23] + [0] * 10,
        }
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for n, slots in expected.items():
                out = torch.full((2, 24), -7, dtype=torch.int32, device=device)
                window_write_locations(
                    tables.to(device),
                    page_sizes.to(device),
                    starts.to(device),
                    out,
                    5,
                    n,
                )
                assert out[:, : 5 * n].tolist() == [slots, slots], (device, n)
                assert bool((out[:, 5 * n :] == -7).all()), (device, n)


class TestHoleOverflowRoutingTest:
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
        assert (int(out[0, 0])) == (0)
        decode_write_locations(
            tables, page_sizes, torch.tensor([5], dtype=torch.int32), out, 1, 1
        )
        assert (int(out[0, 0])) == (14)

    def test_ragged_minus_one_entry_routes_to_slot_zero(self):
        # A raw -1 (ragged pad that never went through the fill) is a hole too.
        tables = torch.tensor([[[3, -1]]], dtype=torch.int32)
        page_sizes = torch.tensor([2], dtype=torch.int32)
        out = torch.zeros((1, 1), dtype=torch.int32)
        decode_write_locations(
            tables, page_sizes, torch.tensor([4], dtype=torch.int32), out, 1, 1
        )
        assert (int(out[0, 0])) == (0)

    def test_extend_span_crossing_hole_and_table_end(self):
        # P=2, width 2, pages [3, 0]: positions 2,3 hit the hole, 4.. overflow.
        tables = torch.tensor([[[3, 0]]], dtype=torch.int32)
        page_sizes = torch.tensor([2], dtype=torch.int32)
        prefix = torch.tensor([1], dtype=torch.int32)
        new = torch.tensor([4], dtype=torch.int32)
        locs = extend_write_locations(
            tables, page_sizes, prefix, new, 4, torch.empty((1, 4), dtype=torch.int32)
        )
        # pos 1 -> page 3 slot 1 = 7; pos 2, 3 -> hole page 0 -> 0; pos 4 ->
        # page index 2 >= width -> 0.
        assert (locs[0].tolist()) == ([7, 0, 0, 0])


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


class TestMtpReanchorTest:
    """Vanilla MTP re-anchors the draft rows to the committed frontier:
    ``update_draft_forward_metadata`` republishes the frontier into every
    leaf's seq-lens buffer, and the next recompute over the SAME stacks
    derives the decode window from it."""

    def _router(self):
        leaves = {
            FULL: _StubLeaf(4, is_draft=True),
            SWA: _StubLeaf(2, is_draft=True),
        }
        router = CacheGroupRouter(
            None,
            is_draft=True,
            spec_num_tokens=1,
            device="cpu",
            consumed_group_ids=None,
        )
        geometry = CacheGroupGeometry(
            granularities={FULL: 4, SWA: 4},
            attn_kinds={FULL: "full_history", SWA: "sliding_window"},
            full_history_group_id=FULL,
            row_geometry={FULL: (4, 1), SWA: (4, 1)},
            windows={FULL: None, SWA: 4},
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
            torch.arange(2, dtype=torch.int32),
            torch.tensor([9, 7], dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            block_tables={FULL: raw, SWA: raw},
        )
        frontier = torch.tensor([6, 9], dtype=torch.int32)
        router.update_draft_forward_metadata(frontier)
        for leaf in leaves.values():
            assert (leaf.seq_lens_buf[:2].tolist()) == ([6, 9])
        # Recompute over the unchanged stacks: the decode window is now the
        # frontier's last position, not the pre-verify one.
        router.stacks.compute_decode_locations(2, frontier, 1)
        # FULL (P=4): pos 5 -> page 6 slot 1 = 25; pos 8 -> page 3 slot 0 = 12.
        assert (router.stacks.decode_locations(FULL, 2, 1).tolist()) == ([25, 12])
        # SWA (P=2, raw grain 4 -> ratio 2): pos 5 -> kernel page 12 slot 1 =
        # 25; pos 8 -> kernel page 6 slot 0 = 12 (same slots: page-size
        # invariant math over aliasing tables).
        assert (router.stacks.decode_locations(SWA, 2, 1).tolist()) == ([25, 12])


class TestBaseWriteLocationsDefaultTest:
    """A backend that owns no paged write locations must refuse the accessor
    loudly — there is no caller-supplied fallback vector anymore."""

    def test_default_write_locations_raises(self):
        from tokenspeed.runtime.layers.attention.backends.base import (
            AttentionBackend,
        )

        class _MinimalBackend(AttentionBackend):
            pass

        backend = _MinimalBackend.__new__(_MinimalBackend)  # skip __init__
        with pytest.raises((NotImplementedError), match=("write locations")):
            backend.write_locations(
                SimpleNamespace(group_id="full_attention"), ForwardMode.DECODE
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))


class TestHostPageSizeMirror:
    @pytest.mark.parametrize("device", ["cpu", "npu"])
    @pytest.mark.parametrize("width", [1, 4])
    def test_stacks_avoid_scalar_reads_and_replay_changed_lengths(self, device, width):
        from torch.utils._python_dispatch import TorchDispatchMode

        from tokenspeed.runtime.layers.attention.backends.paged.group_tables import (
            GroupTableSpec,
            GroupTableStacks,
        )

        if device == "npu":
            pytest.importorskip("torch_npu")
            if not torch.npu.is_available():
                pytest.skip("NPU unavailable")
            torch.npu.set_device(0)

        class NoScalarReads(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func == torch.ops.aten._local_scalar_dense.default:
                    raise AssertionError("device page-size scalar read")
                return func(*args, **(kwargs or {}))

        specs = [GroupTableSpec("a", 2, 2, 3), GroupTableSpec("b", 4, 4, 3)]
        stacks = GroupTableStacks(
            specs, max_bs=3, max_tokens_per_req=4, max_extend_tokens=8, device=device
        )
        assert stacks.page_sizes_cpu == (2, 4)
        tables = torch.tensor([3, 0, -1], dtype=torch.int32).expand(2, 3, 3).clone()
        stacks.tables.copy_(tables.to(device))
        lengths = torch.tensor([3, 7, 2**31 - 2], dtype=torch.int32, device=device)
        # Prove the guard detects the old device-scalar fallback.
        with pytest.raises(AssertionError, match="page-size scalar"):
            with NoScalarReads():
                decode_write_locations(
                    stacks.tables,
                    stacks.page_sizes,
                    lengths,
                    stacks.decode_locs,
                    3,
                    width,
                )

        def compute():
            with NoScalarReads():
                stacks.compute_decode_locations(3, lengths, width)

        def check():
            expected = torch.empty((2, 3 * width), dtype=torch.int32)
            decode_write_locations(
                tables, torch.tensor([2, 4]), lengths.cpu(), expected, 3, width
            )
            torch.testing.assert_close(
                stacks.decode_locs[:, : 3 * width].cpu(), expected
            )

        compute()
        check()
        if device == "npu":
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                compute()
            lengths.copy_(torch.tensor([1, 3, 8], dtype=torch.int32, device=device))
            graph.replay()
            torch.npu.synchronize()
            check()
        with NoScalarReads():
            stacks.compute_window_locations(3, lengths, width)
        expected = torch.empty((2, 3 * width), dtype=torch.int32)
        window_write_locations(
            tables, torch.tensor([2, 4]), lengths.cpu(), expected, 3, width
        )
        torch.testing.assert_close(stacks.decode_locs[:, : 3 * width].cpu(), expected)

    @pytest.mark.parametrize("mirror", [(2,), (2, 0), (2, True), [2, 4]])
    def test_invalid_host_page_sizes_are_rejected(self, mirror):
        with pytest.raises(ValueError, match="host page sizes"):
            decode_write_locations(
                torch.ones((2, 1, 2), dtype=torch.int32),
                torch.tensor([2, 4]),
                torch.tensor([2]),
                torch.empty((2, 1), dtype=torch.int32),
                1,
                1,
                page_sizes_cpu=mirror,
            )
