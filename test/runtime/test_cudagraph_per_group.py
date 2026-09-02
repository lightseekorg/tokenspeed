"""Per-group table stacks under CUDA-graph decode.

The router's ``GroupTableStacks`` is the persistent scratch a captured decode
graph reads through the leaves' buffers, refilled every step. This file pins
the two fill paths against each other at odd shapes (mixed expansion ratios,
per-group widths, truncation, row padding) and the padding contract the
graph relies on: column tails and dummy rows are ALWAYS the null page 0 —
never ``-1`` and never a previous step's residue. Everything else the old
DecodeBuffers/wrapper machinery covered lives in test_cache_group_router.py.
"""

from __future__ import annotations

import os
import sys
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import torch

from tokenspeed.runtime.layers.attention.backends.group_tables import (
    GroupTableSpec,
    GroupTableStacks,
)

G0, G1, G2 = "grain_equal", "grain_half", "grain_quarter"


def _odd_specs():
    """Three groups sharing block granularity 4 at different kernel pages:
    ratio 1 (exact width), ratio 2 (width overshoots the raw span -> zero
    tail), ratio 4 (width undershoots -> truncation)."""
    return [
        GroupTableSpec(G0, block_granularity=4, kernel_page_size=4, width=7),
        GroupTableSpec(G1, block_granularity=4, kernel_page_size=2, width=13),
        GroupTableSpec(G2, block_granularity=4, kernel_page_size=1, width=5),
    ]


def _stacks(device, max_bs=6):
    return GroupTableStacks(
        _odd_specs(), max_bs=max_bs, max_tokens_per_req=2, device=device
    )


class PackedUnpackParityTest(unittest.TestCase):
    """The one-launch packed unpack must agree with the per-group reference
    at shapes the bridge really produces: shared storage, per-group column
    counts, holes and ragged -1 pads, padded rows."""

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_packed_matches_per_group_at_odd_shapes(self):
        torch.manual_seed(7)
        rows = 5
        for bs, actual_bs, cols in ((5, 3, (7, 5, 3)), (4, 4, (2, 9, 1))):
            with self.subTest(bs=bs, actual_bs=actual_bs, cols=cols):
                total = rows * sum(cols)
                packed = torch.randint(
                    -1, 20, (total,), dtype=torch.int32, device="cuda"
                )
                srcs, off = [], 0
                for c in cols:
                    srcs.append(packed[off : off + rows * c].view(rows, c))
                    off += rows * c
                stacks_packed = _stacks("cuda")
                stacks_plain = _stacks("cuda")
                # Sentinel residue: both paths must overwrite/zero the same cells.
                stacks_packed.tables.fill_(99)
                stacks_plain.tables.fill_(99)
                self.assertTrue(stacks_packed._fill_packed(bs, actual_bs, srcs))
                stacks_plain._fill_per_group(bs, actual_bs, srcs)
                torch.testing.assert_close(
                    stacks_packed.tables[:, :bs], stacks_plain.tables[:, :bs]
                )
                # And the public entry point picks the packed path here.
                stacks_packed.tables.fill_(99)
                stacks_packed.fill(bs, actual_bs, dict(zip((G0, G1, G2), srcs)))
                torch.testing.assert_close(
                    stacks_packed.tables[:, :bs], stacks_plain.tables[:, :bs]
                )

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_separate_storages_fall_back_to_the_per_group_path(self):
        stacks = _stacks("cuda")
        srcs = [torch.ones((2, 3), dtype=torch.int32, device="cuda") for _ in range(3)]
        self.assertFalse(stacks._fill_packed(2, 2, srcs))
        # The public fill still lands the same expansion through the fallback.
        stacks.fill(2, 2, dict(zip((G0, G1, G2), srcs)))
        # Ratio 4 group: raw page 1 -> kernel pages 4..7, truncated to width 5.
        self.assertEqual(stacks.table(G2, 2)[0].tolist(), [4, 5, 6, 7, 4])


class PaddingContractTest(unittest.TestCase):
    """Column tails and dummy rows are ALWAYS the null page 0 (safe to
    dereference, never a live request) — the old -1 tails plus wrapper
    masking are gone, and no previous step's residue may survive a refill."""

    def test_column_tails_are_zero_not_minus_one(self):
        stacks = _stacks("cpu")
        stacks.tables.fill_(99)  # previous step's residue
        raw = {
            G0: torch.tensor([[5, 6]], dtype=torch.int32),
            G1: torch.tensor([[5, 6]], dtype=torch.int32),
            G2: torch.tensor([[5, 6]], dtype=torch.int32),
        }
        stacks.fill(bs=1, actual_bs=1, block_tables=raw)
        # ratio 1, width 7: 2 live columns then 5 zeros.
        self.assertEqual(stacks.table(G0, 1)[0].tolist(), [5, 6, 0, 0, 0, 0, 0])
        # ratio 2, width 13: 4 live kernel pages then 9 zeros.
        self.assertEqual(
            stacks.table(G1, 1)[0].tolist(),
            [10, 11, 12, 13] + [0] * 9,
        )
        # ratio 4, width 5: truncated inside page 5's expansion — no tail.
        self.assertEqual(stacks.table(G2, 1)[0].tolist(), [20, 21, 22, 23, 24])

    def test_dummy_rows_are_zero_and_residue_is_cleared(self):
        stacks = _stacks("cpu")
        wide = {
            G0: torch.full((4, 7), 3, dtype=torch.int32),
            G1: torch.full((4, 7), 3, dtype=torch.int32),
            G2: torch.full((4, 7), 3, dtype=torch.int32),
        }
        stacks.fill(bs=4, actual_bs=4, block_tables=wide)
        # Next step: narrower live table, padded batch. Rows [1, 3) must be
        # zero and the previous step's wider columns must not leak through.
        narrow = {
            G0: torch.tensor([[5]], dtype=torch.int32),
            G1: torch.tensor([[5]], dtype=torch.int32),
            G2: torch.tensor([[5]], dtype=torch.int32),
        }
        stacks.fill(bs=3, actual_bs=1, block_tables=narrow)
        self.assertEqual(
            stacks.table(G0, 3).tolist(), [[5, 0, 0, 0, 0, 0, 0]] + [[0] * 7] * 2
        )
        self.assertEqual(stacks.table(G1, 3)[0, :2].tolist(), [10, 11])
        self.assertEqual(int(stacks.table(G1, 3)[0, 2:].abs().sum()), 0)
        self.assertEqual(int(stacks.table(G1, 3)[1:].abs().sum()), 0)

    def test_idle_fill_zeroes_the_whole_padded_batch(self):
        stacks = _stacks("cpu")
        stacks.tables.fill_(99)
        placeholder = {
            gid: torch.ones((4, 2), dtype=torch.int32) for gid in (G0, G1, G2)
        }
        stacks.fill(bs=4, actual_bs=0, block_tables=placeholder)
        self.assertEqual(int(stacks.tables[:, :4].abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
