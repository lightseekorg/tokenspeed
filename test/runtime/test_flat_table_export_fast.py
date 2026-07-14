"""Flat page-table export fast paths (per-step O(pages) fix).

flat_block_tables_from_forward_op consumes the ext's contiguous ndarray
export (flat_block_tables_arrays) -- the only supported path; -1 pad /
hole=0 semantics and the num_reqs row check are preserved, and an ext
without the method fails loudly (rebuild required).
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")


class FlatExportFastPathTest(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
            import torch

            from tokenspeed.runtime.engine.scheduler_utils import (
                flat_block_tables_from_forward_op,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + runtime deps: {exc}")
        self.np = np
        self.torch = torch
        self.bridge = flat_block_tables_from_forward_op

    def _ndarray_op(self, tables):
        arrays = {k: self.np.asarray(v, dtype=self.np.int32) for k, v in tables.items()}
        return SimpleNamespace(flat_block_tables_arrays=lambda: arrays)

    def _legacy_op(self, tables):
        return SimpleNamespace(flat_block_tables=tables)

    def test_ndarray_path_preserves_values(self):
        op = self._ndarray_op({"full": [[1, 2, -1], [3, 0, 4]]})
        out = self.bridge(op, "cpu", num_reqs=2)
        self.assertEqual(out["full"].dtype, self.torch.int32)
        self.assertEqual(out["full"].tolist(), [[1, 2, -1], [3, 0, 4]])

    def test_ndarray_path_num_reqs_check(self):
        op = self._ndarray_op({"full": [[1], [2], [3]]})
        with self.assertRaisesRegex(ValueError, "num_reqs=2"):
            self.bridge(op, "cpu", num_reqs=2)

    def test_missing_arrays_method_raises(self):
        op = SimpleNamespace(flat_block_tables={"full": [[1]]})
        with self.assertRaisesRegex(RuntimeError, "flat_block_tables_arrays"):
            self.bridge(op, "cpu", num_reqs=1)


class PagedCacheBridgeStillWorksTest(unittest.TestCase):
    """The paged-cache (radix hetero) bridge keeps the ragged nested-list
    path -- regression pin for the flat-only cleanup."""

    def test_ragged_paged_cache_tables(self):
        try:
            from tokenspeed.runtime.engine.scheduler_utils import (
                paged_cache_block_tables_from_forward_op,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs runtime deps: {exc}")
        op = SimpleNamespace(paged_cache_block_tables={"full": [[1, 2, 3], [4]]})
        out = paged_cache_block_tables_from_forward_op(op, "cpu", num_reqs=2)
        self.assertEqual(out["full"].tolist(), [[1, 2, 3], [4, -1, -1]])


if __name__ == "__main__":
    unittest.main()
