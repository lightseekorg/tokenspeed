from __future__ import annotations

import os
import sys
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")


def _import_bridge():
    """Import the bridge; skip if torch / tokenspeed_scheduler ext absent."""
    from tokenspeed.runtime.engine.scheduler_utils import (
        flat_block_tables_from_forward_op,
    )

    return flat_block_tables_from_forward_op


class FlatBlockTablesBridgeTest(unittest.TestCase):
    def setUp(self):
        try:
            self.bridge = _import_bridge()
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(
                f"flat bridge unavailable (needs torch + tokenspeed_scheduler "
                f"ext): {exc}"
            )
        import torch

        self.torch = torch

    def _make_op(self, flat_block_tables):
        from types import SimpleNamespace

        import numpy as np

        def rect(v):
            a = np.asarray(v, dtype=np.int32)
            return a if a.ndim == 2 else a.reshape(len(v), 0)

        arrays = {k: rect(v) for k, v in flat_block_tables.items()}
        return SimpleNamespace(
            flat_block_tables=flat_block_tables,
            flat_block_tables_arrays=lambda: arrays,
        )

    def test_two_groups_shape_and_null_hole_preserved(self):
        op = self._make_op(
            {
                "full": [[11, 12], [13, 0]],
                "swa": [[21], [0]],
            }
        )
        out = self.bridge(op, device="cpu", num_reqs=2)
        self.assertEqual(set(out.keys()), {"full", "swa"})

        full = out["full"]
        self.assertEqual(tuple(full.shape), (2, 2))
        self.assertEqual(full.dtype, self.torch.int32)
        self.assertEqual(full.tolist(), [[11, 12], [13, 0]])

        swa = out["swa"]
        self.assertEqual(tuple(swa.shape), (2, 1))
        self.assertEqual(swa.tolist(), [[21], [0]])

    def test_radix_op_without_flat_attrs_returns_empty(self):
        from types import SimpleNamespace

        op = SimpleNamespace()
        self.assertEqual(self.bridge(op, device="cpu"), {})

    def test_attribute_name_is_pinned(self):
        # The bridge reads exactly `flat_block_tables`; a renamed payload
        # yields {} like a radix op, so a rename must be caught here.
        from types import SimpleNamespace

        renamed = SimpleNamespace(flat_page_tables={"full": [[1]]})
        self.assertEqual(self.bridge(renamed, device="cpu", num_reqs=1), {})
        op = self._make_op({"full": [[1]]})
        out = self.bridge(op, device="cpu", num_reqs=1)
        self.assertEqual(out["full"].tolist(), [[1]])

    def test_row_count_mismatch_raises(self):
        op = self._make_op({"full": [[1, 2]]})
        with self.assertRaises(ValueError):
            self.bridge(op, device="cpu", num_reqs=2)

    def test_empty_rows_group_on_live_batch_raises(self):
        # An empty row list may not silently vanish on a live op: downstream
        # replay would see a per-group hole over stale pages.
        op = self._make_op({"full": [[1, 2], [3, 4]], "swa": []})
        with self.assertRaisesRegex(ValueError, r"swa.*0 rows"):
            self.bridge(op, device="cpu", num_reqs=2)

    def test_empty_rows_group_on_zero_req_op_dropped(self):
        # bs==0 replay/idle paths treat the resulting {} as "no tables".
        op = self._make_op({"full": [], "swa": []})
        self.assertEqual(self.bridge(op, device="cpu", num_reqs=0), {})
        self.assertEqual(self.bridge(op, device="cpu"), {})


class FlatFlagGatingTest(unittest.TestCase):
    """uses_flat_cache_groups must default to False so every existing
    backend stays on today's path; needs torch, skips otherwise."""

    def setUp(self):
        try:
            from tokenspeed.runtime.layers.attention.backends.base import (
                AttentionBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + backend base: {exc}")
        self.AttentionBackend = AttentionBackend

    def test_default_backend_does_not_use_flat_groups(self):
        self.assertFalse(self.AttentionBackend.uses_flat_cache_groups)


if __name__ == "__main__":
    unittest.main()
