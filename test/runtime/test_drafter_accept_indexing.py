import unittest
from types import SimpleNamespace

import torch

from tokenspeed.runtime.execution.drafter.dflash import DFlash
from tokenspeed.runtime.execution.drafter.eagle import Eagle, EagleDraftInput
from tokenspeed.runtime.execution.drafter.mtp import (
    Mtp,
    _extend_depth_precompute,
    _extend_depth_shifted_ids_from,
    _frontier_hidden_splice,
    _frontier_shifted_ids,
    _ragged_tail_rows,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    substitute_mm_pad_,
)


def _make_eagle(spec_num_tokens: int = 4, max_bs: int = 8) -> Eagle:
    drafter = Eagle.__new__(Eagle)
    drafter.spec_num_tokens = spec_num_tokens
    drafter.padded_gather_ids_offsets_buf = (
        torch.arange(max_bs, dtype=torch.int64) * spec_num_tokens - 1
    )
    return drafter


def test_mtp_index_sharing_rides_the_draft_backend_share() -> None:
    from tokenspeed.runtime.layers.attention.backends.base import SparseTopKShare

    model = SimpleNamespace(index_share_for_mtp_iteration=True)
    drafter = Eagle.__new__(Eagle)
    drafter.draft_model_runner = SimpleNamespace(model=model)
    drafter.attn_backend = SimpleNamespace(sparse_topk=SparseTopKShare())
    share = drafter.attn_backend.sparse_topk
    first_step_topk = (object(), object())
    share.qsa_metadata = object()

    drafter._attach_dsa_topk(first_step_topk)

    assert share.prefill is first_step_topk[0]
    assert share.decode is first_step_topk[1]
    assert share.qsa_metadata is None
    assert drafter._extract_dsa_topk((None, None)) == first_step_topk

    # The target's last indexer layer leaves its selection on the target
    # backend; the drafter starts the MTP head from there.
    target_share = SparseTopKShare(prefill=object(), decode=object())
    base_ctx = SimpleNamespace(attn_backend=SimpleNamespace(sparse_topk=target_share))
    assert drafter._target_dsa_topk(base_ctx) == (
        target_share.prefill,
        target_share.decode,
    )

    # A head that does not share gets a cleared share every step (a stale
    # selection would otherwise be reused as "already computed"), and the
    # drafter passes its own state through untouched.
    model.index_share_for_mtp_iteration = False
    fallback = (object(), object())
    share.qsa_metadata = object()
    drafter._attach_dsa_topk(fallback)
    assert share.prefill is None and share.decode is None
    assert share.qsa_metadata is None
    assert drafter._extract_dsa_topk(fallback) == fallback
    assert drafter._target_dsa_topk(base_ctx) == (None, None)


class TestDrafterAcceptIndexing(unittest.TestCase):
    def test_mtp_stash_uses_request_pool_capacity(self):
        max_bs = 16
        request_pool_rows = 18
        spec_num_tokens = 4
        input_buffers = SimpleNamespace(
            max_bs=max_bs,
            seq_lens_buf=torch.zeros(max_bs, dtype=torch.int32),
        )
        model_runner = SimpleNamespace(
            device="cpu",
            mapping=SimpleNamespace(attn=SimpleNamespace(dp_size=1)),
            model=SimpleNamespace(),
            model_config=SimpleNamespace(hidden_size=8, dtype=torch.float32),
        )
        runtime_states = SimpleNamespace(
            valid_cache_lengths=torch.zeros(request_pool_rows, dtype=torch.int32)
        )
        backend = SimpleNamespace()

        drafter = Mtp(
            spec_num_tokens=spec_num_tokens,
            spec_num_steps=3,
            draft_model_runner=model_runner,
            attn_backend=backend,
            runtime_states=runtime_states,
            input_buffers=input_buffers,
        )

        self.assertEqual(
            list(drafter._stash_tokens_buf.shape),
            [request_pool_rows, spec_num_tokens - 1],
        )
        self.assertEqual(
            list(drafter._stash_hidden_buf.shape),
            [request_pool_rows, spec_num_tokens - 1, 8],
        )

    def test_substitute_mm_pad_rewrites_media_ids_in_place(self):
        image = MultimodalDataItem(modality=Modality.IMAGE, hash=123)
        audio = MultimodalDataItem(modality=Modality.AUDIO, hash=456)
        image.set_pad_value()
        audio.set_pad_value()
        input_ids = torch.tensor(
            [7, image.pad_value, audio.pad_value, 42], dtype=torch.int64
        )

        out = substitute_mm_pad_(input_ids, {Modality.IMAGE: 10, Modality.AUDIO: 20})

        self.assertIs(out, input_ids)
        self.assertEqual(input_ids.tolist(), [7, 10, 20, 42])

    def test_input_buffers_validate_modality_specific_mm_substitutes(self):
        buffers = InputBuffers.__new__(InputBuffers)
        buffers.set_mm_pad_substitute_ids(
            {Modality.IMAGE: 10, Modality.AUDIO: 20}, vocab_size=256
        )
        self.assertEqual(
            buffers.mm_pad_substitute_ids,
            {Modality.IMAGE: 10, Modality.AUDIO: 20},
        )

        with self.assertRaisesRegex(ValueError, "inside the target"):
            buffers.set_mm_pad_substitute_ids({Modality.IMAGE: 256}, vocab_size=256)

    def test_eagle_decode_first_step_gathers_last_accepted_output(self):
        drafter = _make_eagle(spec_num_tokens=4)
        output_tokens = torch.arange(12, dtype=torch.int32)
        draft_input = EagleDraftInput(
            input_num_tokens=12,
            num_extends=0,
            forward_mode=ForwardMode.DECODE,
            base_model_output=output_tokens,
            accept_lengths=torch.tensor([1, 2, 4], dtype=torch.int32),
            base_out_hidden_states=torch.empty(0),
        )

        input_ids, gather_ids = drafter._get_first_step_input(
            draft_input,
            bs=3,
            input_num_tokens=12,
        )

        self.assertIs(input_ids, output_tokens)
        self.assertEqual(gather_ids.tolist(), [0, 5, 11])

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
            accept_lengths=torch.tensor([1, 2, 4], dtype=torch.int32),
            base_out_hidden_states=torch.empty(0),
        )

        input_ids, gather_ids = drafter._get_first_step_input(
            draft_input,
            bs=3,
            input_num_tokens=10,
        )

        self.assertEqual(gather_ids.tolist(), [1, 3, 9])
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

    def test_frontier_shifted_ids_reads_stash_and_verify(self):
        # k=4. Request A accepts 2 of [v0..v3]; request B accepts all 4.
        # Stash entry i holds the committed token at position vc-k+2+i.
        v = torch.tensor([[500, 501, 502, 503], [600, 601, 602, 603]])
        accept = torch.tensor([2, 4])
        stash = torch.tensor([[41, 42, 43], [71, 72, 73]])

        depth0 = _frontier_shifted_ids(v, accept, stash)

        # src = accept - 4 + j, all rows committed (stash/verify).
        self.assertEqual(
            depth0.view(2, 4).tolist(),
            [[42, 43, 500, 501], [600, 601, 602, 603]],
        )

    def test_frontier_window_rolls_left_into_drafts(self):
        # Depth d+1 ids roll depth d's window one left, appending its draft:
        # the trailing d rows of depth d take this round's drafts d_1..d_d.
        window0 = torch.tensor([[42, 43, 500, 501], [600, 601, 602, 603]])
        drafts = torch.tensor([[51, 52], [61, 62]])

        depth1 = torch.cat([window0[:, 1:], drafts[:, 0:1]], 1)
        depth2 = torch.cat([depth1[:, 1:], drafts[:, 1:2]], 1)

        self.assertEqual(
            depth1.tolist(),
            [[43, 500, 501, 51], [601, 602, 603, 61]],
        )
        self.assertEqual(
            depth2.tolist(),
            [[500, 501, 51, 52], [602, 603, 61, 62]],
        )

    def test_frontier_hidden_splice_gathers_at_accept_boundary(self):
        # H=1; stash rows are the hiddens at positions vc-3..vc-1, fresh
        # rows this round's verify hiddens at vc..vc+3.
        stash = torch.tensor([[[-3.0], [-2.0], [-1.0]], [[-13.0], [-12.0], [-11.0]]])
        fresh = torch.tensor(
            [[[0.0], [1.0], [2.0], [3.0]], [[10.0], [11.0], [12.0], [13.0]]]
        )
        accept = torch.tensor([2, 4])

        spliced = _frontier_hidden_splice(stash, fresh, accept)

        # accept=2: window rows at vc-2..vc+1; accept=4: rows at vc..vc+3.
        self.assertEqual(
            spliced.view(2, 4).tolist(),
            [[-2.0, -1.0, 0.0, 1.0], [10.0, 11.0, 12.0, 13.0]],
        )

    def test_frontier_ids_and_stash_track_positions_over_rounds(self):
        # Positional oracle: the committed token at position p has id
        # 1000+p, the draft candidate for position frontier+m has id
        # 7000+m, and rejected verify entries (junk, id 9000+) must never
        # be read. The stash rolls the way the drafter does: the depth-0
        # window's tail [:, 1:]; row j of depth d must always compose the
        # token at (frontier - k + j) + d + 1.
        torch.manual_seed(0)
        k, steps = 4, 3
        vc = 10
        stash = torch.tensor([[1000 + vc - 2, 1000 + vc - 1, 1000 + vc]])
        for accept in [1, 4, 2, 1, 3, 4, 1, 2]:
            a = torch.tensor([accept])
            v = torch.tensor(
                [[1000 + vc + 1 + i if i < accept else 9000 + i for i in range(k)]]
            )
            drafts = torch.tensor([[7000 + m for m in range(1, steps)]])
            frontier = vc + accept
            depth0 = _frontier_shifted_ids(v, a, stash).view(1, k)
            ids = depth0
            for d in range(steps):
                if d > 0:
                    ids = torch.cat([ids[:, 1:], drafts[:, d - 1 : d]], 1)
                for j in range(k):
                    consumed = (frontier - k + j) + d + 1
                    if consumed <= frontier:
                        self.assertEqual(ids[0, j].item(), 1000 + consumed)
                    else:
                        self.assertEqual(ids[0, j].item(), 7000 + consumed - frontier)
            stash = depth0[:, 1:]
            vc = frontier
            self.assertEqual(stash.tolist(), [[1000 + vc - 2 + i for i in range(3)]])

    def test_frontier_hidden_splice_tracks_positions_over_rounds(self):
        # Same oracle for the hidden side: the target hidden of position p
        # is encoded as float(p); the stash must always hold positions
        # vc-3..vc-1 and the splice must yield the window rows
        # frontier-4..frontier-1.
        k = 4
        vc = 10
        stash = torch.tensor([float(vc - 3 + i) for i in range(3)]).view(1, 3, 1)
        for accept in [1, 4, 2, 1, 3]:
            a = torch.tensor([accept])
            fresh = torch.tensor(
                [float(vc + i) if i < accept else 9000.0 for i in range(k)]
            ).view(1, k, 1)
            frontier = vc + accept

            spliced = _frontier_hidden_splice(stash, fresh, a)

            self.assertEqual(
                spliced.view(k).tolist(),
                [float(frontier - k + j) for j in range(k)],
            )
            stash = spliced.view(1, k, 1)[:, 1:]
            vc = frontier
            self.assertEqual(
                stash.view(3).tolist(), [float(vc - 3 + i) for i in range(3)]
            )

    def test_ragged_tail_rows_borrows_old_tail_on_short_chunks(self):
        flat = torch.arange(6) + 100  # request A rows 0..4, request B row 5
        lengths = torch.tensor([5, 1], dtype=torch.int32)
        old_tail = torch.tensor([[1, 2], [3, 4]])

        updated = _ragged_tail_rows(flat, lengths, old_tail, 2)

        self.assertEqual(updated.tolist(), [[103, 104], [4, 105]])

    def test_dflash_current_tokens_gather_last_accepted_per_row(self):
        output_tokens = torch.arange(12, dtype=torch.int32)

        current = DFlash._current_tokens_from_output(
            output_tokens=output_tokens,
            accept_lengths=torch.tensor([1, 2, 4], dtype=torch.int32),
            num_extends=0,
            spec_num_tokens=4,
        )

        self.assertEqual(current.tolist(), [0, 5, 11])

    def test_dflash_mixed_current_tokens_do_not_cross_decode_rows(self):
        output_tokens = torch.tensor(
            [100, 10, 11, 12, 13, 20, 21, 22, 23],
            dtype=torch.int32,
        )

        current = DFlash._current_tokens_from_output(
            output_tokens=output_tokens,
            accept_lengths=torch.tensor([1, 2, 4], dtype=torch.int32),
            num_extends=1,
            spec_num_tokens=4,
        )

        self.assertEqual(current.tolist(), [100, 11, 23])


if __name__ == "__main__":
    unittest.main()
