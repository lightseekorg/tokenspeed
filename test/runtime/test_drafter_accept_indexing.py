import unittest
from types import SimpleNamespace

import torch

from tokenspeed.runtime.execution.drafter.dflash import DFlash
from tokenspeed.runtime.execution.drafter.eagle import Eagle, EagleDraftInput
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode


def _make_eagle(spec_num_tokens: int = 4, max_bs: int = 8) -> Eagle:
    drafter = Eagle.__new__(Eagle)
    drafter.spec_num_tokens = spec_num_tokens
    drafter.padded_gather_ids_offsets_buf = (
        torch.arange(max_bs, dtype=torch.int64) * spec_num_tokens - 1
    )
    return drafter


class TestDrafterAcceptIndexing(unittest.TestCase):
    def test_eagle_accept_output_indices_stay_inside_each_decode_row(self):
        drafter = _make_eagle(spec_num_tokens=4)

        indices = drafter._accepted_output_indices(
            torch.tensor([0, 1, 4, 99], dtype=torch.int32),
            row_count=4,
        )

        self.assertEqual(indices.tolist(), [0, 4, 11, 15])

    def test_eagle_decode_first_step_uses_safe_gather_ids_for_zero_accept(self):
        drafter = _make_eagle(spec_num_tokens=4)
        output_tokens = torch.arange(12, dtype=torch.int32)
        draft_input = EagleDraftInput(
            input_num_tokens=12,
            num_extends=0,
            forward_mode=ForwardMode.DECODE,
            base_model_output=output_tokens,
            accept_lengths=torch.tensor([0, 1, 4], dtype=torch.int32),
            base_out_hidden_states=torch.empty(0),
        )

        input_ids, gather_ids = drafter._get_first_step_input(
            draft_input,
            bs=3,
            input_num_tokens=12,
        )

        self.assertIs(input_ids, output_tokens)
        self.assertEqual(gather_ids.tolist(), [0, 4, 11])

    def test_eagle_mixed_first_step_keeps_decode_gather_ids_in_range(self):
        drafter = _make_eagle(spec_num_tokens=4)
        drafter.input_buffers = SimpleNamespace(
            shifted_prefill_ids_buf=torch.arange(10, dtype=torch.int32),
            input_lengths_buf=torch.tensor([2, 4, 4], dtype=torch.int32),
        )
        output_tokens = torch.arange(9, dtype=torch.int32) + 100
        draft_input = EagleDraftInput(
            input_num_tokens=10,
            num_extends=1,
            forward_mode=ForwardMode.MIXED,
            base_model_output=output_tokens,
            accept_lengths=torch.tensor([1, 0, 4], dtype=torch.int32),
            base_out_hidden_states=torch.empty(0),
        )

        input_ids, gather_ids = drafter._get_first_step_input(
            draft_input,
            bs=3,
            input_num_tokens=10,
        )

        self.assertEqual(gather_ids.tolist(), [1, 2, 9])
        self.assertEqual(input_ids[2:].tolist(), output_tokens[1:].tolist())

    def test_dflash_current_tokens_use_safe_in_row_dummy_for_zero_accept(self):
        output_tokens = torch.arange(12, dtype=torch.int32)

        current = DFlash._current_tokens_from_output(
            output_tokens=output_tokens,
            accept_lengths=torch.tensor([0, 1, 4], dtype=torch.int32),
            num_extends=0,
            spec_num_tokens=4,
        )

        self.assertEqual(current.tolist(), [0, 4, 11])

    def test_dflash_mixed_current_tokens_do_not_cross_decode_rows(self):
        output_tokens = torch.tensor(
            [100, 10, 11, 12, 13, 20, 21, 22, 23],
            dtype=torch.int32,
        )

        current = DFlash._current_tokens_from_output(
            output_tokens=output_tokens,
            accept_lengths=torch.tensor([1, 0, 4], dtype=torch.int32),
            num_extends=1,
            spec_num_tokens=4,
        )

        self.assertEqual(current.tolist(), [100, 10, 23])


if __name__ == "__main__":
    unittest.main()
