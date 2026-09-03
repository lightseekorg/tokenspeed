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
            block_granularity=128,
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
            block_granularity=128,
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
                block_granularity=96,
                kernel_page_size=64,
            )

    def test_equal_page_sizes_keep_the_existing_table_view(self):
        table = torch.tensor([[3, 5, -1]], dtype=torch.int32)

        actual = self.module.expand_page_table(
            table,
            block_granularity=64,
            kernel_page_size=64,
        )

        self.assertEqual(actual.data_ptr(), table.data_ptr())
        self.assertEqual(actual.tolist(), [[3, 5, -1]])

    def test_build_prefill_slots_with_known_output_size_matches_masked_path(self):
        page_table = torch.tensor([[5, 6], [8, 9]], dtype=torch.int32)
        seq_lens = torch.tensor([3, 2], dtype=torch.int32)
        kwargs = {
            "page_table": page_table,
            "seq_lens": seq_lens,
            "max_seq_len": 3,
            "page_size": 2,
            "device": torch.device("cpu"),
        }

        expected = self.module.build_prefill_kv_workspace_slots(**kwargs)
        actual = self.module.build_prefill_kv_workspace_slots(
            **kwargs,
            num_tokens=5,
        )

        self.assertEqual(actual.tolist(), [10, 11, 12, 16, 17])
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
