"""Page-table export fast paths.

block_tables_from_forward_op consumes the ext's contiguous ndarray
export (block_tables_arrays) -- the only supported path; -1 pad /
hole=0 semantics and the num_reqs row check are preserved, and an ext
without the method fails loudly (rebuild required).
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")


class TestExportFastPathTest:
    def setup_method(self):
        try:
            import numpy as np
            import torch

            from tokenspeed.runtime.engine.scheduler_utils import (
                block_tables_from_forward_op,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"needs torch + runtime deps: {exc}")
        self.np = np
        self.torch = torch
        self.bridge = block_tables_from_forward_op

    def _ndarray_op(self, tables):
        arrays = {k: self.np.asarray(v, dtype=self.np.int32) for k, v in tables.items()}
        return SimpleNamespace(block_tables_arrays=lambda: arrays)

    def test_ndarray_path_preserves_values(self):
        op = self._ndarray_op({"full": [[1, 2, -1], [3, 0, 4]]})
        out = self.bridge(op, "cpu", num_reqs=2)
        assert (out["full"].dtype) == (self.torch.int32)
        assert (out["full"].tolist()) == ([[1, 2, -1], [3, 0, 4]])

    def test_ndarray_path_num_reqs_check(self):
        op = self._ndarray_op({"full": [[1], [2], [3]]})
        with pytest.raises((ValueError), match=("num_reqs=2")):
            self.bridge(op, "cpu", num_reqs=2)

    def test_missing_arrays_method_raises(self):
        op = SimpleNamespace(block_tables={"full": [[1]]})
        with pytest.raises((AttributeError)):
            self.bridge(op, "cpu", num_reqs=1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
