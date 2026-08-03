from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import unittest

import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python"
    / "tokenspeed"
    / "runtime"
    / "layers"
    / "attention"
    / "page_table.py"
)


def _load_page_table_module():
    spec = importlib.util.spec_from_file_location("page_table_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PageTableConversionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_page_table_module()

    def test_expands_logical_pages_into_kernel_pages(self):
        logical = torch.tensor([[3, 5, -1], [0, -1, -1]], dtype=torch.int32)

        actual = self.module.expand_page_table(
            logical,
            logical_page_size=128,
            kernel_page_size=64,
            max_kernel_pages=6,
        )

        expected = torch.tensor(
            [[6, 7, 10, 11, 0, 1], [0, 1, 0, 1, 0, 1]],
            dtype=torch.int32,
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_first_and_last_logical_pages_stay_in_expanded_lcm_range(self):
        # Four logical pages consist of null page 0 and usable pages 1..3.
        logical = torch.tensor([[0, 1, 3, -1]], dtype=torch.int32)

        actual = self.module.expand_page_table(
            logical,
            logical_page_size=128,
            kernel_page_size=64,
        )

        expected = torch.tensor(
            [[0, 1, 2, 3, 6, 7, 0, 1]],
            dtype=torch.int32,
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertLessEqual(int(actual.max()), 7)

    def test_rejects_incompatible_page_sizes(self):
        with self.assertRaisesRegex(ValueError, "positive multiple"):
            self.module.expand_page_table(
                torch.tensor([[1]], dtype=torch.int32),
                logical_page_size=96,
                kernel_page_size=64,
            )

    def test_equal_page_sizes_keep_the_existing_table_view(self):
        table = torch.tensor([[3, 5, -1]], dtype=torch.int32)

        actual = self.module.expand_page_table(
            table,
            logical_page_size=64,
            kernel_page_size=64,
        )

        self.assertEqual(actual.data_ptr(), table.data_ptr())
        self.assertEqual(actual.tolist(), [[3, 5, -1]])

    def test_repeats_sliding_ring_page_across_logical_block(self):
        logical = torch.tensor([[0, 3, -1]], dtype=torch.int32)

        actual = self.module.repeat_page_table(
            logical,
            logical_page_tokens=256,
            retained_span_tokens=128,
        )

        self.assertEqual(
            actual.tolist(),
            [[0, 0, 3, 3, 0, 0]],
        )

    def test_repeats_one_state_checkpoint_across_a_logical_block(self):
        logical = torch.tensor([[2, 5]], dtype=torch.int32)

        actual = self.module.repeat_page_table(
            logical,
            logical_page_tokens=256,
            retained_span_tokens=8,
        )

        self.assertEqual(tuple(actual.shape), (1, 64))
        self.assertEqual(actual[0, :32].tolist(), [2] * 32)
        self.assertEqual(actual[0, 32:].tolist(), [5] * 32)

    def test_compressed_history_keeps_one_kernel_page_per_logical_block(self):
        logical = torch.tensor([[2, 5]], dtype=torch.int32)

        actual = self.module.repeat_page_table(
            logical,
            logical_page_tokens=256,
            retained_span_tokens=256,
        )

        self.assertEqual(actual.data_ptr(), logical.data_ptr())
        self.assertEqual(actual.tolist(), [[2, 5]])

    def test_rejects_ring_geometry_that_does_not_tile_the_logical_page(self):
        with self.assertRaisesRegex(ValueError, "must divide"):
            self.module.repeat_page_table(
                torch.tensor([[1]], dtype=torch.int32),
                logical_page_tokens=256,
                retained_span_tokens=96,
            )


if __name__ == "__main__":
    unittest.main()
