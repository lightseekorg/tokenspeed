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
        flat_block_table_base_offsets_from_forward_op,
        flat_block_tables_from_forward_op,
    )

    return (
        flat_block_tables_from_forward_op,
        flat_block_table_base_offsets_from_forward_op,
    )


class FlatBlockTablesBridgeTest(unittest.TestCase):
    def setUp(self):
        try:
            self.bridge, self.base_bridge = _import_bridge()
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(
                f"flat bridge unavailable (needs torch + tokenspeed_scheduler "
                f"ext): {exc}"
            )
        import torch

        self.torch = torch

    def _make_op(self, flat_block_tables, flat_block_table_base_offsets=None):
        from types import SimpleNamespace

        import numpy as np

        def rect(v):
            a = np.asarray(v, dtype=np.int32)
            return a if a.ndim == 2 else a.reshape(len(v), 0)

        arrays = {k: rect(v) for k, v in flat_block_tables.items()}
        return SimpleNamespace(
            flat_block_tables=flat_block_tables,
            flat_block_tables_arrays=lambda: arrays,
            flat_block_table_base_offsets=flat_block_table_base_offsets or {},
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

    def test_flat_base_offsets_preserve_nonzero_compact_origins(self):
        op = self._make_op(
            {"full": [[11], [12]], "state": [[21, 22], [31]]},
            {"full": [0, 0], "state": [7, 19]},
        )
        out, maxima = self.base_bridge(op, device="cpu", num_reqs=2)
        self.assertEqual(out["full"].tolist(), [0, 0])
        self.assertEqual(out["state"].tolist(), [7, 19])
        self.assertEqual(maxima, {"full": 0, "state": 19})

    def test_flat_base_row_mismatch_and_negative_base_fail_closed(self):
        cases = (
            (
                "base row mismatch",
                self._make_op({"state": [[1], [2]]}, {"state": [3]}),
                2,
                r"state.*1 rows",
            ),
            (
                "negative logical base",
                self._make_op({"state": [[1]]}, {"state": [-1]}),
                1,
                "negative logical base",
            ),
        )
        for name, op, num_reqs, error in cases:
            with self.subTest(name), self.assertRaisesRegex(ValueError, error):
                self.base_bridge(op, device="cpu", num_reqs=num_reqs)


class PersistentFlatTableStagingTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.engine.scheduler_utils import (
                FlatBlockTableStagingBuffers,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"persistent flat staging needs torch: {exc}")
        self.torch = torch
        self.Staging = FlatBlockTableStagingBuffers

    def _plan(self, *, depth=2, max_rows=4, byte_delta=0):
        groups = (
            SimpleNamespace(group_id="full", max_export_cols=3),
            SimpleNamespace(group_id="state", max_export_cols=2),
        )
        nbytes = 4 * depth * max_rows * ((3 + 1) + (2 + 1))
        return SimpleNamespace(
            group_table_plans=groups,
            pools=(
                SimpleNamespace(pool_id="pool.full", total_blocks=16),
                SimpleNamespace(pool_id="pool.state", total_blocks=32),
            ),
            scheduler_group_specs=(
                SimpleNamespace(group_id="full", pool_id="pool.full"),
                SimpleNamespace(group_id="state", pool_id="pool.state"),
            ),
            forward_buffer_depth=depth,
            max_scheduled_batch_rows=max_rows,
            forward_input_bytes=nbytes + byte_delta,
            cpu_forward_staging_bytes=nbytes,
        )

    def _op(self, tables, bases, *, copied_cols_override=None):
        torch = self.torch

        class CopyingForwardOp:
            def flat_block_table_group_ids(self):
                return sorted(tables)

            def copy_flat_block_table_to(
                self,
                group_id,
                table_destination,
                base_destination,
                page_id_upper_bound,
            ):
                rows = tables[group_id]
                if any(
                    page_id < -1 or page_id >= page_id_upper_bound
                    for row in rows
                    for page_id in row
                ):
                    raise ValueError("page id outside planned pool")
                table_destination.fill_(-1)
                cols = max((len(row) for row in rows), default=0)
                for row_idx, row in enumerate(rows):
                    if row:
                        begin = row_idx * cols
                        table_destination[begin : begin + len(row)].copy_(
                            torch.tensor(row, dtype=torch.int32)
                        )
                base_values = bases[group_id]
                base_destination[: len(base_values)].copy_(
                    torch.tensor(base_values, dtype=torch.int32)
                )
                if copied_cols_override is not None:
                    cols = copied_cols_override
                return len(rows), cols

        return CopyingForwardOp()

    def test_ring_reuses_planned_storage_without_reallocating(self):
        staging = self.Staging(self._plan(), device="cpu")
        op = self._op(
            {"full": [[1, 2], [3]], "state": [[4], [5, 6]]},
            {"full": [0, 0], "state": [7, 8]},
        )

        first_tables, first_bases = staging.stage(op, num_reqs=2)
        second_tables, _ = staging.stage(op, num_reqs=2)
        third_tables, _ = staging.stage(op, num_reqs=2)

        self.assertNotEqual(
            first_tables["full"].data_ptr(), second_tables["full"].data_ptr()
        )
        self.assertEqual(
            first_tables["full"].data_ptr(), third_tables["full"].data_ptr()
        )
        self.assertEqual(first_tables["full"].tolist(), [[1, 2], [3, -1]])
        self.assertEqual(first_bases["state"].tolist(), [7, 8])

    def test_graph_padding_uses_current_slot_and_page_zero_rows(self):
        staging = self.Staging(self._plan(), device="cpu")
        op = self._op(
            {"full": [[11, 12]], "state": [[21]]},
            {"full": [0], "state": [9]},
        )
        tables, bases = staging.stage(op, num_reqs=1)

        padded_tables, padded_bases = staging.pad_current_for_graph(
            tables,
            bases,
            actual_rows=1,
            padded_rows=3,
        )

        self.assertEqual(padded_tables["full"].tolist(), [[11, 12], [0, 0], [0, 0]])
        self.assertEqual(padded_tables["state"].tolist(), [[21], [0], [0]])
        self.assertEqual(padded_bases["state"].tolist(), [9, 0, 0])
        self.assertEqual(tables["full"].data_ptr(), padded_tables["full"].data_ptr())

    def test_idle_slot_is_cold_safe_and_rotates_without_allocating(self):
        staging = self.Staging(self._plan(depth=2), device="cpu")

        cold_tables, cold_bases = staging.stage_idle(padded_rows=3)
        self.assertEqual(cold_tables["full"].tolist(), [[0], [0], [0]])
        self.assertEqual(cold_tables["state"].tolist(), [[0], [0], [0]])
        self.assertEqual(cold_bases["full"].tolist(), [0, 0, 0])
        self.assertEqual(cold_bases["state"].tolist(), [0, 0, 0])

        op = self._op(
            {"full": [[11, 12]], "state": [[21]]},
            {"full": [0], "state": [9]},
        )
        active_tables, _ = staging.stage(op, num_reqs=1)
        self.assertNotEqual(
            cold_tables["full"].data_ptr(), active_tables["full"].data_ptr()
        )

        # The third use returns to the cold slot. Its allocation identity must
        # remain stable while stale table/base contents are reset to page 0.
        idle_tables, idle_bases = staging.stage_idle(padded_rows=3)
        self.assertEqual(cold_tables["full"].data_ptr(), idle_tables["full"].data_ptr())
        self.assertEqual(idle_tables["full"].tolist(), [[0], [0], [0]])
        self.assertEqual(idle_tables["state"].tolist(), [[0], [0], [0]])
        self.assertEqual(idle_bases["state"].tolist(), [0, 0, 0])

    def test_schema_width_rows_and_accounting_fail_closed(self):
        extra_group = self._op(
            {"full": [[1]], "state": [[2]], "extra": [[3]]},
            {"full": [0], "state": [0], "extra": [0]},
        )
        cases = (
            (
                "accounting drift",
                lambda: self.Staging(self._plan(byte_delta=4), device="cpu"),
                RuntimeError,
                "forward_input_bytes",
            ),
            (
                "unknown group",
                lambda: self.Staging(self._plan(), device="cpu").stage(
                    extra_group, num_reqs=1
                ),
                RuntimeError,
                "extra=.*extra",
            ),
            (
                "row capacity",
                lambda: self.Staging(self._plan(), device="cpu").stage(
                    extra_group, num_reqs=5
                ),
                ValueError,
                "row capacity",
            ),
            (
                "column capacity",
                lambda: self.Staging(self._plan(), device="cpu").stage(
                    self._op(
                        {"full": [[1]], "state": [[2]]},
                        {"full": [0], "state": [0]},
                        copied_cols_override=4,
                    ),
                    num_reqs=1,
                ),
                RuntimeError,
                "columns outside",
            ),
            (
                "pool page range",
                lambda: self.Staging(self._plan(), device="cpu").stage(
                    self._op(
                        {"full": [[16]], "state": [[2]]},
                        {"full": [0], "state": [0]},
                    ),
                    num_reqs=1,
                ),
                ValueError,
                "outside planned pool",
            ),
        )
        for name, operation, error_type, error in cases:
            with self.subTest(name), self.assertRaisesRegex(error_type, error):
                operation()


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
