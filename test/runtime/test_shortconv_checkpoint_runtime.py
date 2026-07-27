from __future__ import annotations

import unittest

import torch

from tokenspeed.runtime.layers.attention.backends.inkling import (
    InklingAttnBackend,
    InklingConvMetadata,
    ShortConvCheckpointMetadata,
)


class ShortConvCheckpointRuntimeTest(unittest.TestCase):
    def _metadata(self, checkpoints, cache_indices=(1, 2)):
        return InklingConvMetadata(
            query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
            cache_indices=torch.tensor(cache_indices, dtype=torch.int32),
            has_initial_state=torch.ones(2, dtype=torch.bool),
            is_decode=False,
            checkpoints=checkpoints,
        )

    def test_restore_joins_split_checkpoint_fields(self):
        state = torch.zeros(4, 3, 4)
        first = torch.arange(6 * 3 * 2, dtype=torch.float32).view(6, 3, 2)
        second = first + 100
        metadata = self._metadata(
            ShortConvCheckpointMetadata(
                restore_pages={"kvconv": torch.tensor([2, 0], dtype=torch.int32)},
                write_pages={"kvconv": torch.empty(0, dtype=torch.int32)},
                write_requests=torch.empty(0, dtype=torch.int64),
            )
        )

        InklingAttnBackend.restore_shortconv_checkpoint(
            state, (first, second), metadata, "kvconv"
        )

        self.assertTrue(torch.equal(state[1], torch.cat((first[2], second[2]), -1)))
        self.assertTrue(torch.equal(state[2], torch.zeros_like(state[2])))

    def test_prefill_checkpoint_borrows_old_window_rows(self):
        state = torch.zeros(4, 3, 4)
        state[1] = torch.tensor(
            [[10, 20, 30, 40], [11, 21, 31, 41], [12, 22, 32, 42]],
            dtype=torch.float32,
        )
        x = torch.tensor(
            [[100, 200, 300, 400], [101, 201, 301, 401]],
            dtype=torch.float32,
        )
        first = torch.zeros(6, 3, 2)
        second = torch.zeros(6, 3, 2)
        metadata = self._metadata(
            ShortConvCheckpointMetadata(
                restore_pages={"kvconv": torch.zeros(2, dtype=torch.int32)},
                write_pages={"kvconv": torch.tensor([4], dtype=torch.int32)},
                write_requests=torch.tensor([0], dtype=torch.int64),
                packed_rows=torch.tensor([[0, 0, 1]], dtype=torch.int64),
                prior_state_rows=torch.tensor([[2, 0, 0]], dtype=torch.int64),
                packed_row_mask=torch.tensor([[False, True, True]]),
            )
        )

        InklingAttnBackend.publish_shortconv_checkpoints(
            x, state, (first, second), metadata, "kvconv"
        )

        expected = torch.stack((state[1, 2], x[0], x[1]))
        self.assertTrue(torch.equal(torch.cat((first[4], second[4]), -1), expected))

    def test_decode_writes_only_completed_boundaries(self):
        state = torch.arange(4 * 3 * 4, dtype=torch.float32).view(4, 3, 4)
        first = torch.full((6, 3, 2), -7.0)
        second = torch.full((6, 3, 2), -9.0)
        page_zero_before = (first[0].clone(), second[0].clone())
        metadata = self._metadata(
            ShortConvCheckpointMetadata(
                restore_pages={"kvconv": torch.zeros(2, dtype=torch.int32)},
                write_pages={"kvconv": torch.tensor([0, 5], dtype=torch.int32)},
                write_requests=torch.arange(2, dtype=torch.int64),
            )
        )

        InklingAttnBackend.publish_shortconv_checkpoints(
            torch.empty(2, 4), state, (first, second), metadata, "kvconv"
        )

        self.assertTrue(torch.equal(first[0], page_zero_before[0]))
        self.assertTrue(torch.equal(second[0], page_zero_before[1]))
        self.assertTrue(
            torch.equal(torch.cat((first[5], second[5]), -1), state[2])
        )


if __name__ == "__main__":
    unittest.main()
