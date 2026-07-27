from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).parents[2]
    / "python/tokenspeed/runtime/layers/attention/backends/shortconv_checkpoint.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "shortconv_checkpoint_under_test", _MODULE_PATH
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
plan_shortconv_checkpoint_writes = _MODULE.plan_shortconv_checkpoint_writes


class ShortConvCheckpointPlanTest(unittest.TestCase):
    def test_completed_boundaries_map_to_state_group_page_slots(self):
        writes = plan_shortconv_checkpoint_writes(
            page_size=128,
            state_rows=3,
            seq_lens_before=(0,),
            seq_lens_after=(256,),
            query_start_loc=(0, 256),
        )

        self.assertEqual(
            [(write.request_index, write.page_slot) for write in writes],
            [(0, 0), (0, 1)],
        )
        self.assertEqual(writes[0].packed_rows, (125, 126, 127))
        self.assertEqual(writes[1].packed_rows, (253, 254, 255))
        self.assertEqual(writes[0].prior_state_rows, (None, None, None))

    def test_boundary_near_chunk_start_combines_old_state_and_new_rows(self):
        writes = plan_shortconv_checkpoint_writes(
            page_size=128,
            state_rows=3,
            seq_lens_before=(126,),
            seq_lens_after=(130,),
            query_start_loc=(0, 4),
        )

        self.assertEqual(len(writes), 1)
        write = writes[0]
        self.assertEqual(write.page_slot, 0)
        self.assertEqual(write.packed_rows, (None, 0, 1))
        self.assertEqual(write.prior_state_rows, (2, None, None))

    def test_varlen_batch_uses_request_local_chunk_offsets(self):
        writes = plan_shortconv_checkpoint_writes(
            page_size=128,
            state_rows=3,
            seq_lens_before=(0, 250),
            seq_lens_after=(256, 260),
            query_start_loc=(0, 256, 266),
        )

        self.assertEqual(
            [
                (write.request_index, write.page_slot, write.packed_rows)
                for write in writes
            ],
            [
                (0, 0, (125, 126, 127)),
                (0, 1, (253, 254, 255)),
                (1, 1, (259, 260, 261)),
            ],
        )

    def test_no_completed_boundary_produces_no_writes(self):
        self.assertEqual(
            plan_shortconv_checkpoint_writes(
                page_size=128,
                state_rows=3,
                seq_lens_before=(17,),
                seq_lens_after=(127,),
                query_start_loc=(0, 110),
            ),
            (),
        )

    def test_invalid_geometry_fails_loudly(self):
        cases = [
            (
                "page_size",
                dict(
                    page_size=0,
                    state_rows=3,
                    seq_lens_before=(0,),
                    seq_lens_after=(1,),
                    query_start_loc=(0, 1),
                ),
            ),
            (
                "state_rows",
                dict(
                    page_size=128,
                    state_rows=0,
                    seq_lens_before=(0,),
                    seq_lens_after=(1,),
                    query_start_loc=(0, 1),
                ),
            ),
            (
                "length",
                dict(
                    page_size=128,
                    state_rows=3,
                    seq_lens_before=(0,),
                    seq_lens_after=(1, 2),
                    query_start_loc=(0, 1),
                ),
            ),
            (
                "query_start_loc",
                dict(
                    page_size=128,
                    state_rows=3,
                    seq_lens_before=(0,),
                    seq_lens_after=(1,),
                    query_start_loc=(0,),
                ),
            ),
        ]
        for name, kwargs in cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                plan_shortconv_checkpoint_writes(**kwargs)


if __name__ == "__main__":
    unittest.main()
