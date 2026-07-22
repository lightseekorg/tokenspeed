import unittest
from types import SimpleNamespace

import torch

from tokenspeed.runtime.execution.drafter.dflash import DFlash
from tokenspeed.runtime.execution.drafter.eagle import Eagle, EagleDraftInput
from tokenspeed.runtime.execution.drafter.mtp import (
    _committed_tail_update,
    _extend_depth_precompute,
    _extend_depth_shifted_ids_from,
    _lookback_shifted_ids,
    _ragged_tail_rows,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem


def _make_eagle(spec_num_tokens: int = 4, max_bs: int = 8) -> Eagle:
    drafter = Eagle.__new__(Eagle)
    drafter.spec_num_tokens = spec_num_tokens
    drafter.padded_gather_ids_offsets_buf = (
        torch.arange(max_bs, dtype=torch.int64) * spec_num_tokens - 1
    )
    return drafter


class TestDrafterAcceptIndexing(unittest.TestCase):
    def test_prepare_draft_input_ids_maps_media_then_clamps_to_vocab(self):
        drafter = _make_eagle()
        drafter.vocab_size = 100
        drafter.set_mm_pad_substitute_ids({Modality.IMAGE: 10, Modality.AUDIO: 20})
        image = MultimodalDataItem(modality=Modality.IMAGE, hash=123)
        audio = MultimodalDataItem(modality=Modality.AUDIO, hash=456)
        image.set_pad_value()
        audio.set_pad_value()
        input_ids = torch.tensor(
            [-4, image.pad_value, audio.pad_value, 42, 999], dtype=torch.int64
        )

        prepared = drafter._prepare_draft_input_ids(input_ids)

        self.assertEqual(prepared.tolist(), [0, 10, 20, 42, 99])
        self.assertEqual(
            input_ids.tolist(),
            [-4, image.pad_value, audio.pad_value, 42, 999],
        )

    def test_eagle_validates_modality_specific_mm_substitutes(self):
        drafter = _make_eagle()
        drafter.vocab_size = 256
        drafter.set_mm_pad_substitute_ids({Modality.IMAGE: 10, Modality.AUDIO: 20})
        self.assertEqual(
            drafter.mm_pad_substitute_ids,
            {Modality.IMAGE: 10, Modality.AUDIO: 20},
        )

        with self.assertRaisesRegex(ValueError, "inside the target vocabulary"):
            drafter.set_mm_pad_substitute_ids({Modality.IMAGE: 256})

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

    def test_extend_depth_shifted_ids_shifts_within_each_request(self):
        # Request A: 5 prefill rows, shift-1 ids [t1..t4, S_A] (S_A = the
        # round's sampled token on the final chunk). Request B: 3 rows,
        # [u1, u2, S_B]. Drafts: A -> a1, a2; B -> b1, b2.
        shift1_ids = torch.tensor([11, 12, 13, 14, 500, 21, 22, 600])
        input_lengths = torch.tensor([5, 3], dtype=torch.int32)
        next_tokens = torch.tensor(
            [[500, 501, 502, 502], [600, 601, 602, 602]], dtype=torch.int32
        )

        pre = _extend_depth_precompute(shift1_ids, input_lengths)
        depth1 = _extend_depth_shifted_ids_from(pre, next_tokens, 1)
        depth2 = _extend_depth_shifted_ids_from(pre, next_tokens, 2)

        self.assertEqual(depth1.tolist(), [12, 13, 14, 500, 501, 22, 600, 601])
        self.assertEqual(depth2.tolist(), [13, 14, 500, 501, 502, 600, 601, 602])

    def test_extend_depth_shifted_ids_single_request_tail_uses_drafts(self):
        shift1_ids = torch.tensor([11, 12, 700])
        input_lengths = torch.tensor([3], dtype=torch.int32)
        next_tokens = torch.tensor([[700, 701, 702, 703]], dtype=torch.int32)

        pre = _extend_depth_precompute(shift1_ids, input_lengths)
        depth3 = _extend_depth_shifted_ids_from(pre, next_tokens, 3)

        # With P=3 and depth 3 every row overshoots the shift-1 ids: local
        # row i consumes t_{i+4}, i.e. drafts d_1..d_3.
        self.assertEqual(depth3.tolist(), [701, 702, 703])

    def test_lookback_shifted_ids_reads_stash_verify_and_drafts(self):
        # k=4, D=2. Request A accepts 2 of [v0..v3]; request B accepts all 4.
        # Stash entry i holds the committed token at position vc-D+1+i, so
        # entry 1 (= token at vc) is the only one any depth >= 1 consumes.
        v = torch.tensor([[500, 501, 502, 503], [600, 601, 602, 603]])
        accept = torch.tensor([[2], [4]])
        next_tokens = torch.tensor(
            [[0, 51, 52, 53], [0, 61, 62, 63]], dtype=torch.int32
        )
        stash = torch.tensor([[41, 42], [71, 72]])

        depth1 = _lookback_shifted_ids(v, accept, next_tokens, stash, 1, 2)
        depth2 = _lookback_shifted_ids(v, accept, next_tokens, stash, 2, 2)

        # depth 1: src = [-1, 0, 1, 2, 3, 4] per request.
        self.assertEqual(
            depth1.view(2, 6).tolist(),
            [[42, 500, 501, 51, 51, 51], [72, 600, 601, 602, 603, 61]],
        )
        # depth 2: src = [0, 1, 2, 3, 4, 5]; past the accept the drafts
        # continue from the accept point (m = src - accept -> d_{m+1}),
        # clamped to this depth's filled columns.
        self.assertEqual(
            depth2.view(2, 6).tolist(),
            [[500, 501, 51, 52, 52, 52], [600, 601, 602, 603, 61, 62]],
        )

    def test_committed_tail_update_blends_across_rounds(self):
        stash = torch.tensor([[10, 11], [20, 21]])
        fresh = torch.tensor([[30, 31, 32, 33], [40, 41, 42, 43]])
        valid = torch.tensor([1, 3])

        updated = _committed_tail_update(stash, fresh, valid, 2)

        # valid=1: rows [-1, 0] -> [old tail, fresh[0]]; valid=3: rows [1, 2].
        self.assertEqual(updated.tolist(), [[11, 30], [41, 42]])

    def test_committed_tail_update_keeps_feature_dims(self):
        stash = torch.arange(4, dtype=torch.float32).view(1, 2, 2)
        fresh = (torch.arange(8, dtype=torch.float32) + 10).view(1, 4, 2)
        valid = torch.tensor([2])

        updated = _committed_tail_update(stash, fresh, valid, 2)

        self.assertEqual(updated.tolist(), [[[10.0, 11.0], [12.0, 13.0]]])

    def test_ragged_tail_rows_borrows_old_tail_on_short_chunks(self):
        flat = torch.arange(6) + 100  # request A rows 0..4, request B row 5
        lengths = torch.tensor([5, 1], dtype=torch.int32)
        old_tail = torch.tensor([[1, 2], [3, 4]])

        updated = _ragged_tail_rows(flat, lengths, old_tail, 2)

        self.assertEqual(updated.tolist(), [[103, 104], [4, 105]])

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
