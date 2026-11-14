# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""CPU schema/ownership-checker negatives, not simulated NPU qualification."""

import unittest
from copy import deepcopy

from test_persistent_model_ownership import (
    covered,
    validate_trace_ownership,
    validate_witness,
)


def witness(rank=0, operation="validate_startup", when="after", tag=None):
    return {
        "event": "persistent-ownership",
        "pid": rank + 1,
        "rank": rank,
        "operation": operation,
        "when": when,
        "tag": tag,
        "parameters": [
            {"name": "target.weight", "storage_ptr": 4112, "storage_bytes": 64}
        ],
        "pools": [
            {"id": [1, 1], "tag": "weights", "policy": "discard", "scope": "weights"},
            {
                "id": [2, 2],
                "tag": "weights",
                "policy": "cpu_backup",
                "scope": "persistent_buffers",
            },
        ],
        "PTA_segments": [
            {"address": 4096, "total_size": 1024, "segment_pool_id": [1, 1]},
            {"address": 8192, "total_size": 1024, "segment_pool_id": [2, 2]},
        ],
        "native": {
            "callback_fault": False,
            "blocks": [
                {
                    "ptr": "0x1000",
                    "aligned_bytes": 1024,
                    "tag": "weights",
                    "policy": "discard",
                },
                {
                    "ptr": "0x2000",
                    "aligned_bytes": 1024,
                    "tag": "weights",
                    "policy": "backup",
                },
            ],
        },
    }


def trace():
    events = []
    for rank in (0, 1):
        start = witness(rank)
        before = witness(rank, "pause", "before", "weights")
        after = witness(rank, "pause", "after", "weights")
        transition = {
            "event": "pause",
            "tag": "weights",
            "pid": rank + 1,
            "before": deepcopy(before["native"]),
            "after": deepcopy(after["native"]),
        }
        events.extend((start, before, transition, after))
    return events


class OwnershipCheckerTests(unittest.TestCase):
    def test_coverage_is_positive_and_half_open(self):
        self.assertTrue(covered(4096, 1024, 4096, 1024))
        for values in (
            (5120, 0, 4096, 1024),
            (5120, 1, 4096, 1024),
            (4096, -1, 4096, 1024),
            (-1, 1, 0, 1024),
            (4096, 1, 4096, 0),
            (4096, 1, 4096, -1),
            (4096, True, 4096, 1024),
        ):
            with self.subTest(values=values):
                self.assertFalse(covered(*values))

    def test_malformed_parameter_extents_refuse(self):
        for size in (0, -1):
            row = witness()
            row["parameters"][0]["storage_bytes"] = size
            with self.subTest(size=size), self.assertRaisesRegex(
                AssertionError, "Invalid parameter range"
            ):
                validate_witness(row)

    def test_malformed_segment_extents_refuse(self):
        for size in (0, -1):
            row = witness()
            row["PTA_segments"][1]["total_size"] = size
            with self.subTest(size=size), self.assertRaisesRegex(
                AssertionError, "Invalid PTA segment range"
            ):
                validate_witness(row)

    def test_malformed_native_extents_refuse(self):
        for size in (0, -1):
            row = witness()
            row["native"]["blocks"][1]["aligned_bytes"] = size
            with self.subTest(size=size), self.assertRaisesRegex(
                AssertionError, "Invalid native block range"
            ):
                validate_witness(row)

    def test_negative_parameter_start_refuses(self):
        row = witness()
        row["parameters"][0]["storage_ptr"] = -1
        with self.assertRaisesRegex(AssertionError, "Invalid parameter range"):
            validate_witness(row)

    def test_parameter_segment_must_fit_the_same_native_block(self):
        row = witness()
        row["PTA_segments"][0]["total_size"] = 2048
        row["PTA_segments"].append(
            {"address": 4800, "total_size": 128, "segment_pool_id": [1, 1]}
        )
        with self.assertRaisesRegex(AssertionError, "full selected PTA segment"):
            validate_witness(row)

    def test_classifies_parameter_discard_and_separate_persistent_backup(self):
        self.assertEqual(validate_witness(witness())["persistent_PTA_segments"], 1)
        self.assertEqual(validate_trace_ownership(trace(), tp_size=2)["ranks"], [0, 1])

    def test_parameter_cannot_move_to_persistent_backup(self):
        row = witness()
        row["parameters"][0]["storage_ptr"] = 8208
        with self.assertRaisesRegex(AssertionError, "Actual model parameter"):
            validate_witness(row)

    def test_native_parameter_policy_cannot_be_backup(self):
        row = witness()
        row["native"]["blocks"][0]["policy"] = "backup"
        with self.assertRaisesRegex(AssertionError, "Actual model parameter"):
            validate_witness(row)

    def test_persistent_native_policy_cannot_be_discard(self):
        row = witness()
        row["native"]["blocks"][1]["policy"] = "discard"
        with self.assertRaisesRegex(AssertionError, "Persistent PTA allocation"):
            validate_witness(row)

    def test_unrelated_backup_scope_cannot_substitute_for_persistent(self):
        row = witness()
        row["pools"][1]["scope"] = "ordinary_backup_weights"
        with self.assertRaisesRegex(AssertionError, "Canonical persistent"):
            validate_witness(row)

    def test_logical_pools_cannot_alias_one_PTA_pool(self):
        row = witness()
        row["pools"][1]["id"] = [1, 1]
        with self.assertRaisesRegex(AssertionError, "share a PTA pool ID"):
            validate_witness(row)

    def test_missing_real_parameter_storage_refuses(self):
        row = witness()
        row["parameters"] = []
        with self.assertRaisesRegex(AssertionError, "No real model parameter"):
            validate_witness(row)

    def test_uncovered_parameter_storage_refuses(self):
        row = witness()
        row["parameters"][0]["storage_bytes"] = 4096
        with self.assertRaisesRegex(AssertionError, "actual PTA segment"):
            validate_witness(row)

    def test_unowned_extra_backup_block_refuses(self):
        row = witness()
        row["native"]["blocks"].append(
            {
                "ptr": "0x3000",
                "aligned_bytes": 1024,
                "tag": "weights",
                "policy": "backup",
            }
        )
        with self.assertRaisesRegex(AssertionError, "no matching actual PTA ownership"):
            validate_witness(row)

    def test_owned_but_noncanonical_extra_backup_block_refuses(self):
        row = witness()
        row["pools"].append(
            {
                "id": [3, 3],
                "tag": "weights",
                "policy": "cpu_backup",
                "scope": "ordinary_backup_weights",
            }
        )
        row["PTA_segments"].append(
            {"address": 12288, "total_size": 1024, "segment_pool_id": [3, 3]}
        )
        row["native"]["blocks"].append(
            {
                "ptr": "0x3000",
                "aligned_bytes": 1024,
                "tag": "weights",
                "policy": "backup",
            }
        )
        with self.assertRaisesRegex(
            AssertionError, "not the canonical persistent pool"
        ):
            validate_witness(row)

    def test_persistent_pool_object_without_real_allocation_refuses(self):
        row = witness()
        row["PTA_segments"].pop()
        with self.assertRaisesRegex(AssertionError, "no actual PTA allocation"):
            validate_witness(row)

    def test_missing_rank_witness_refuses(self):
        rows = trace()
        rows = [
            row
            for row in rows
            if not (
                row["event"] == "persistent-ownership"
                and row["operation"] == "validate_startup"
                and row["rank"] == 1
            )
        ]
        with self.assertRaisesRegex(AssertionError, "startup ownership witness"):
            validate_trace_ownership(rows, tp_size=2)

    def test_missing_transition_side_refuses(self):
        rows = trace()
        del rows[3]
        with self.assertRaisesRegex(AssertionError, "exactly one before/after"):
            validate_trace_ownership(rows, tp_size=2)

    def test_transition_native_snapshot_mismatch_refuses(self):
        rows = trace()
        rows[2]["after"]["callback_fault"] = True
        with self.assertRaisesRegex(
            AssertionError, "not bound to the actual transition"
        ):
            validate_trace_ownership(rows, tp_size=2)

    def test_rank_PID_association_change_refuses(self):
        rows = trace()
        rows[3]["rank"] = 1
        with self.assertRaisesRegex(AssertionError, "PID/rank association changed"):
            validate_trace_ownership(rows, tp_size=2)


if __name__ == "__main__":
    unittest.main()
