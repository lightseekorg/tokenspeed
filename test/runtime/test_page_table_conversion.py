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
