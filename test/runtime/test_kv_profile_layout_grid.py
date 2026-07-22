"""KV profile -> layout-grid flooring.

The memory-dependent profile can land anywhere on the base page grid, but
the hetero slot layout is only viewable on a coarser congruence. The MTP
draft pool shares the target's page-id space at the largest group page's
stride (one 256-row slot per id plus the 256-row dummy for the Inkling
geometry — the drafter consumes the flat full-attention table at its
native stride), so the profiled size floors to ``size ≡ 0 (mod 256)``.
Historical: before the drafter consumed the table at that stride, the
draft pool held one 128-row slot per id and its 256-row page view forced
``size ≡ 128 (mod 256)`` (an ODD id count) — whether a boot survived used
to be the parity of the profiled id count.
"""

# CI Registration (parsed via AST, runtime no-op)
import os
import sys
import unittest
from types import SimpleNamespace

from tokenspeed.runtime.layers.attention.registry import (
    _floor_tokens_to_layout_grid,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")


def _hetero_config():
    return SimpleNamespace(
        layer_kv_head_counts=(16,) * 55 + (8,) * 11,
        slot_tokens=256,
        group_page_sizes={"full_attention": 256},
        page_size=128,
    )


class TestKvProfileLayoutGrid(unittest.TestCase):
    def test_bad_boot_value_lands_on_grid(self):
        # The 2026-07-14 crash value (even id count, 5758) now floors onto
        # the 256 grid instead of the odd-parity grid.
        floored = _floor_tokens_to_layout_grid(737024, _hetero_config())
        self.assertEqual(floored, 737024)
        self.assertEqual(floored % 256, 0)

    def test_off_grid_value_floors(self):
        floored = _floor_tokens_to_layout_grid(737024 + 128, _hetero_config())
        self.assertEqual(floored, 737024)
        self.assertEqual(floored % 256, 0)

    def test_every_residue_lands_on_grid_and_costs_under_one_page(self):
        cfg = _hetero_config()
        for raw in range(736896, 736896 + 512, 128):
            floored = _floor_tokens_to_layout_grid(raw, cfg)
            self.assertLessEqual(floored, raw)
            self.assertLess(raw - floored, 256)
            self.assertEqual(floored % 256, 0)

    def test_noop_without_hetero_head_counts(self):
        cfg = SimpleNamespace(
            layer_kv_head_counts=None,
            slot_tokens=None,
            group_page_sizes=None,
            page_size=128,
        )
        self.assertEqual(_floor_tokens_to_layout_grid(737024, cfg), 737024)


if __name__ == "__main__":
    unittest.main()
