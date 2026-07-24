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
        FlatBlockTableStagingBuffers,
        flat_block_table_base_offsets_from_forward_op,
        flat_block_tables_from_forward_op,
    )

    return (
        flat_block_tables_from_forward_op,
        flat_block_table_base_offsets_from_forward_op,
        FlatBlockTableStagingBuffers,
    )


class FlatBlockTablesBridgeTest(unittest.TestCase):
    def setUp(self):
        try:
            self.bridge, self.base_bridge, self.staging_cls = _import_bridge()
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
            {"full": [[11], [12]], "state": [[21, 22], [31, 0]]},
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

    def test_staging_generation_publishes_only_after_successful_copy(self):
        from types import SimpleNamespace

        runtime_metadata = SimpleNamespace(
            forward_buffer_depth=1,
            graph_batch_rows=2,
            max_scheduled_batch_rows=2,
            group_table_plans=(
                SimpleNamespace(
                    group_id="history",
                    target_capture_cols=2,
                    draft_capture_cols=2,
                    max_export_cols=2,
                ),
                SimpleNamespace(
                    group_id="state",
                    target_capture_cols=1,
                    draft_capture_cols=0,
                    max_export_cols=1,
                ),
            ),
            # Ten payload int32s plus target/draft owner unpack headers.
            forward_input_bytes=112,
        )
        plan = SimpleNamespace(
            runtime_metadata=runtime_metadata,
            pools=(
                SimpleNamespace(pool_id="history_pool", total_blocks=8),
                SimpleNamespace(pool_id="state_pool", total_blocks=4),
            ),
            scheduler_group_specs=(
                SimpleNamespace(group_id="history", pool_id="history_pool"),
                SimpleNamespace(group_id="state", pool_id="state_pool"),
            ),
        )
        staging = self.staging_cls(plan, device="cpu")

        class Op:
            cache_generation = 0
            fail = False
            copy_calls = 0

            @staticmethod
            def flat_block_table_group_ids():
                return ("history", "state")

            def copy_flat_block_tables_to(
                self,
                destination,
                page_id_upper_bounds,
                copy_metadata,
                payload_offset,
            ):
                self.copy_calls += 1
                self_outer.assertEqual(page_id_upper_bounds.tolist(), [8, 4])
                self_outer.assertEqual(payload_offset, 18)
                if self.fail:
                    raise RuntimeError("copy failed")
                destination[18:23].copy_(
                    self_outer.torch.tensor(
                        [1, 2, 0, 3, 0],
                        dtype=self_outer.torch.int32,
                    )
                )
                copy_metadata.copy_(
                    self_outer.torch.tensor(
                        (
                            (1, 2, 18, 20),
                            (1, 1, 21, 22),
                        ),
                        dtype=self_outer.torch.int64,
                    )
                )
                return 23

        self_outer = self
        op = Op()
        original = staging.stage(op, num_reqs=1)
        self.assertEqual(op.copy_calls, 1)
        self.assertFalse(hasattr(op, "flat_block_tables_arrays"))
        original_ptr = original.tables["history"].data_ptr()
        self.assertIsNotNone(original.packed)
        self.assertEqual(
            original.tables["history"].untyped_storage().data_ptr(),
            original.base_offsets["history"].untyped_storage().data_ptr(),
        )
        self.assertEqual(
            original.tables["history"].untyped_storage().data_ptr(),
            original.packed.buffer.untyped_storage().data_ptr(),
        )
        packed_owner = original.packed.bind("target", ("history", "state"))
        self.assertEqual(
            packed_owner.unpack_meta.tolist(),
            [
                [18, 2, 0, 2, 20, 4],
                [21, 1, 6, 1, 22, 8],
            ],
        )
        self.assertEqual(
            original.packed.bind("draft", ("history",)).unpack_meta.tolist(),
            [[18, 2, 0, 2, 20, 4]],
        )
        self.assertEqual(original.tables["history"].tolist(), [[1, 2]])
        self.assertEqual(original.base_offsets["history"].tolist(), [0])
        self.assertEqual(original.tables["state"].tolist(), [[3]])
        self.assertEqual(original.base_offsets["state"].tolist(), [0])
        original_table = original.tables["history"]
        original_base = original.base_offsets["history"]
        self.assertIs(staging.stage(op, num_reqs=1), original)
        self.assertIs(original.tables["history"], original_table)
        self.assertIs(original.base_offsets["history"], original_base)

        op.cache_generation = 1
        refreshed = staging.stage(op, num_reqs=1)
        self.assertIsNot(refreshed, original)
        self.assertEqual(original.generation, 0)
        self.assertEqual(refreshed.generation, 1)
        self.assertIs(refreshed.tables["history"], original_table)
        self.assertIs(refreshed.base_offsets["history"], original_base)
        self.assertEqual(refreshed.tables["history"].data_ptr(), original_ptr)
        self.assertEqual(op.copy_calls, 3)

        op.cache_generation = 2
        op.fail = True
        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            staging.stage(op, num_reqs=1)
        self.assertEqual(op.copy_calls, 4)
        self.assertIs(staging._slots[0].source, refreshed)
        self.assertEqual(staging._slots[0].source.generation, 1)


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
