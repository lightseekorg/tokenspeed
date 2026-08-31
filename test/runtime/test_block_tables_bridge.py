from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")


def _import_bridge():
    """Import the bridge; skip if torch / tokenspeed_scheduler ext absent."""
    from tokenspeed.runtime.engine.scheduler_utils import (
        block_tables_from_forward_op,
    )

    return block_tables_from_forward_op


class BlockTablesBridgeTest(unittest.TestCase):
    def setUp(self):
        try:
            self.bridge = _import_bridge()
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(
                f"cache bridge unavailable (needs torch + tokenspeed_scheduler "
                f"extension): {exc}"
            )
        import torch

        self.torch = torch

    def _make_op(self, block_tables):
        from types import SimpleNamespace

        import numpy as np

        def rect(v):
            a = np.asarray(v, dtype=np.int32)
            return a if a.ndim == 2 else a.reshape(len(v), 0)

        arrays = {k: rect(v) for k, v in block_tables.items()}
        return SimpleNamespace(
            block_tables=block_tables,
            block_tables_arrays=lambda: arrays,
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

    def test_array_export_is_consumed(self):
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

    def test_strict_contract_rejects_missing_extra_and_duplicate_normalized_ids(self):
        op = self._make_op({"full": [[1]]})
        with self.assertRaisesRegex(ValueError, "missing=.*swa"):
            self.bridge(
                op,
                device="cpu",
                num_reqs=1,
                expected_group_ids=("full", "swa"),
            )

        op = self._make_op({"full": [[1]], "swa": [[2]], "extra": [[3]]})
        with self.assertRaisesRegex(ValueError, "extra=.*extra"):
            self.bridge(
                op,
                device="cpu",
                num_reqs=1,
                expected_group_ids=("full", "swa"),
            )

        import numpy as np

        collision = SimpleNamespace(
            block_tables_arrays=lambda: {
                1: np.array([[1]], dtype=np.int32),
                "1": np.array([[2]], dtype=np.int32),
            }
        )
        with self.assertRaisesRegex(ValueError, "collide"):
            self.bridge(collision, device="cpu", num_reqs=1, max_page_id=8)

    def test_strict_contract_rejects_malformed_and_out_of_range_tables(self):
        import numpy as np

        wrong_dtype = SimpleNamespace(
            block_tables_arrays=lambda: {"full": np.array([[1]], dtype=np.int64)}
        )
        with self.assertRaisesRegex(ValueError, "int32"):
            self.bridge(wrong_dtype, device="cpu", num_reqs=1, max_page_id=8)

        out_of_range = self._make_op({"full": [[1, 99]]})
        with self.assertRaisesRegex(ValueError, "outside -1..8"):
            self.bridge(
                out_of_range,
                device="cpu",
                num_reqs=1,
                expected_group_ids=("full",),
                max_page_ids={"full": 8},
            )


if __name__ == "__main__":
    unittest.main()
