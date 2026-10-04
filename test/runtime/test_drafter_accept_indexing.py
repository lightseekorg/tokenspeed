import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from tokenspeed.runtime.execution.drafter.base import BaseDrafter
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


def _multi_depth_forward(
    ctx, input_ids, positions, captured_hidden_states=None, spec_step_idx=0, **kwargs
):
    """The draft-model forward shape the multi-depth drafter requires."""


def _make_mtp(
    *,
    spec_num_tokens: int = 4,
    spec_num_steps: int = 3,
    num_mtp_layers: int | None = None,
    max_bs: int = 16,
    request_pool_rows: int = 18,
    dp_size: int = 1,
    draft_forward=None,
    model_forward=_multi_depth_forward,
) -> Mtp:
    """An ``Mtp`` over stubbed runner/buffers (no weights, no kernels); the
    draft model builds ``num_mtp_layers`` depths (default: one per step)."""
    if num_mtp_layers is None:
        num_mtp_layers = spec_num_steps
    input_buffers = SimpleNamespace(
        max_bs=max_bs,
        seq_lens_buf=torch.zeros(max_bs, dtype=torch.int32),
    )
    model_runner = SimpleNamespace(
        device="cpu",
        mapping=SimpleNamespace(attn=SimpleNamespace(dp_size=dp_size)),
        model=SimpleNamespace(forward=model_forward, num_mtp_layers=num_mtp_layers),
        # What ModelRunner.load_model derives from the forward signature.
        forward_accepts_spec_step_idx=(
            "spec_step_idx" in inspect.signature(model_forward).parameters
        ),
        model_config=SimpleNamespace(
            hidden_size=8,
            dtype=torch.float32,
            requires_request_token_history=False,
        ),
        forward=draft_forward,
    )
    runtime_states = SimpleNamespace(
        valid_cache_lengths=torch.zeros(request_pool_rows, dtype=torch.int32)
    )
    return Mtp(
        spec_num_tokens=spec_num_tokens,
        spec_num_steps=spec_num_steps,
        draft_model_runner=model_runner,
        attn_backend=SimpleNamespace(),
        runtime_states=runtime_states,
        input_buffers=input_buffers,
    )


def _recording_forward(calls: list[dict]):
    """A draft ``forward`` recording each IDLE call's step and sizing."""

    def draft_forward(ctx, input_ids, positions, spec_step_idx, **kwargs):
        calls.append(
            {
                "mode": ctx.forward_mode,
                "step": spec_step_idx,
                "rows": input_ids.numel(),
                "global_num_tokens": ctx.global_num_tokens,
                "global_bs": ctx.global_bs,
                "bs": ctx.bs,
                "input_num_tokens": ctx.input_num_tokens,
                "kwargs": kwargs,
            }
        )

    return draft_forward


def _run_idle_round(drafter) -> list[ForwardMode]:
    """Run one idle-rank round (this rank idle, the peer decoding 2 requests
    over 8 rows) through a stubbed executor; returns the target's forward
    modes."""
    from tokenspeed.runtime.execution.model_executor import ModelExecutor
    from tokenspeed.runtime.execution.types import DpForwardMetadata

    target_calls: list[ForwardMode] = []
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.device = "cpu"
    executor.input_buffers = SimpleNamespace(
        req_pool_indices_buf=torch.zeros(4, dtype=torch.int64)
    )
    executor.runtime_states = SimpleNamespace(
        valid_cache_lengths=torch.zeros(4, dtype=torch.int32), vocab_size=32
    )
    executor.attn_backend = SimpleNamespace()
    executor.token_to_kv_pool = SimpleNamespace()
    executor.model_runner = SimpleNamespace(
        forward=lambda ctx, **kwargs: target_calls.append(ctx.forward_mode)
    )
    executor._model_input_kwargs = lambda bs, num_tokens: {}
    executor.forward_step = SimpleNamespace(can_run=lambda bs, ctx: False)
    executor.drafter = drafter

    executor.execute_idle_forward(
        DpForwardMetadata(
            global_num_tokens=[0, 8],
            global_batch_size=[0, 2],
            global_forward_mode=[ForwardMode.IDLE, ForwardMode.DECODE],
            all_decode_or_idle=True,
            all_extend=False,
            need_idle_forward=True,
        )
    )
    return target_calls


class TestDrafterAcceptIndexing(unittest.TestCase):
    def test_mtp_stash_uses_request_pool_capacity(self):
        request_pool_rows = 18
        spec_num_tokens = 4

        drafter = _make_mtp(
            spec_num_tokens=spec_num_tokens, request_pool_rows=request_pool_rows
        )

        self.assertEqual(
            list(drafter._stash_tokens_buf.shape),
            [request_pool_rows, spec_num_tokens - 1],
        )
        self.assertEqual(
            list(drafter._stash_hidden_buf.shape),
            [request_pool_rows, spec_num_tokens - 1, 8],
        )

    def test_mtp_refuses_a_draft_forward_without_spec_step_idx(self):
        # ModelRunner forwards spec_step_idx only to a forward that declares
        # it (**kwargs does not count); without it every depth would silently
        # run depth 0, so construction fails instead.
        def eagle_shaped_forward(ctx, input_ids, positions, **kwargs):
            pass

        with self.assertRaisesRegex(TypeError, "spec_step_idx"):
            _make_mtp(model_forward=eagle_shaped_forward)

    def test_mtp_refuses_more_steps_than_the_draft_has_depths(self):
        # Step d runs layers[d % num_mtp_layers] onto cache plane d % N: a
        # step count past the depth count would wrap onto plane 0 and
        # overwrite it, so construction refuses it; equal or more depths
        # are fine.
        with self.assertRaisesRegex(ValueError, "2 MTP depth layer"):
            _make_mtp(spec_num_steps=3, num_mtp_layers=2)

        _make_mtp(spec_num_steps=3, num_mtp_layers=3)
        _make_mtp(spec_num_steps=3, num_mtp_layers=8)

    def test_mtp_runs_under_attention_dp_and_sizes_every_depth_like_the_target(
        self,
    ):
        # Attention DP is a parameter of the one drafting path: construction
        # no longer refuses dp_size > 1, and every depth (not just depth 0)
        # mirrors the target's per-rank row counts — the k-window per decode
        # request — never the Eagle chain's one row per request.
        drafter = _make_mtp(spec_num_steps=3, dp_size=4)
        global_num_tokens = [8, 0, 12, 4]
        global_bs = [2, 0, 3, 1]

        steps = drafter.idle_forward_global_num_tokens(global_num_tokens, global_bs)

        self.assertEqual(len(steps), 3)
        for step in steps:
            self.assertIs(step, global_num_tokens)

    def test_mtp_idle_rank_runs_one_empty_idle_forward_per_depth(self):
        # An idle DP rank's round: the executor runs the drafter's depth loop
        # as IDLE forwards over an empty window, one per depth with its own
        # spec_step_idx, each sized by the round's target token counts so the
        # rank enters the same collectives as the ranks with work.
        calls: list[dict] = []
        drafter = _make_mtp(
            spec_num_steps=3, dp_size=2, draft_forward=_recording_forward(calls)
        )

        target_calls = _run_idle_round(drafter)

        self.assertEqual(target_calls, [ForwardMode.IDLE])
        self.assertEqual([c["step"] for c in calls], [0, 1, 2])
        for call in calls:
            self.assertEqual(call["mode"], ForwardMode.IDLE)
            self.assertEqual(
                (call["bs"], call["input_num_tokens"], call["rows"]), (0, 0, 0)
            )
            self.assertEqual(call["global_num_tokens"], [0, 8])
            self.assertEqual(call["global_bs"], [0, 2])
            # No request-token-history view: Mtp drafts do not read one.
            self.assertEqual(call["kwargs"], {})

    def test_idle_round_follows_a_drafter_subclass_idle_hook(self):
        # The executor iterates whatever the drafter's hook lists — one IDLE
        # forward per entry, sized by that entry — so a subclass's override
        # (here a block drafter's single step) shapes the round, not the
        # base class's Eagle default.
        class _BlockShaped(BaseDrafter):
            def idle_forward_global_num_tokens(self, global_num_tokens, global_bs):
                return [global_bs]

            def run(self, *args, **kwargs):
                raise AssertionError("idle rounds never draft")

            def draft(self, *args, **kwargs):
                raise AssertionError("idle rounds never draft")

        calls: list[dict] = []
        drafter = _BlockShaped(
            spec_num_tokens=4,
            spec_num_steps=3,
            draft_model_runner=SimpleNamespace(
                model_config=SimpleNamespace(requires_request_token_history=False),
                forward=_recording_forward(calls),
            ),
            attn_backend=SimpleNamespace(),
        )

        target_calls = _run_idle_round(drafter)

        self.assertEqual(target_calls, [ForwardMode.IDLE])
        self.assertEqual(
            [(c["step"], c["global_num_tokens"]) for c in calls], [(0, [0, 2])]
        )

    def test_dsa_leaf_mtp_frontier_re_expands_the_k_row_indexer_metadata(self):
        # The DSA leaf's k-row top-k reads one context length per query row
        # (``_dsa_seq_lens_2d``, [bs * k, 1]) and a plan over them. The MTP
        # re-anchor keeps that shape and rewrites it in place to the frontier,
        # whereas the Eagle chain's advance re-plans one row per request.
        from tokenspeed.runtime.layers.attention.backends.paged import dsa as dsa_mod

        k, bs = 4, 2
        backend = dsa_mod.DSABackend.__new__(dsa_mod.DSABackend)
        backend.spec_num_tokens = k
        backend.kernel_page_size = 64
        seq_lens_k = torch.tensor([9, 5], dtype=torch.int32)
        metadata = SimpleNamespace(
            seq_lens_k=seq_lens_k,
            _dsa_seq_lens_2d=seq_lens_k.unsqueeze(1)
            .expand(-1, k)
            .reshape(-1, 1)
            .contiguous(),
            _dsa_plan=object(),
        )
        backend._dense_backend = SimpleNamespace(forward_decode_metadata=metadata)
        rows_before = metadata._dsa_seq_lens_2d
        plans: list[dict] = []

        with mock.patch.object(
            dsa_mod, "dsa_plan", side_effect=lambda **kw: plans.append(kw)
        ):
            backend.update_draft_forward_metadata(
                torch.tensor([7, 3, 99], dtype=torch.int32)
            )
            backend.advance_draft_forward_metadata(
                torch.tensor([8, 4, 99], dtype=torch.int32)
            )

        # The re-anchor: seq_lens and every per-token row carry the frontier,
        # same storage (in-graph), and the plan is refreshed in place over
        # the k-row view at the leaf's kernel page size.
        self.assertEqual(seq_lens_k.tolist(), [8, 4])  # last edit = advance
        self.assertIs(metadata._dsa_seq_lens_2d, rows_before)
        self.assertEqual(
            metadata._dsa_seq_lens_2d.view(bs, k).tolist(), [[7] * k, [3] * k]
        )
        self.assertIs(plans[0]["seq_lens_2d"], metadata._dsa_seq_lens_2d)
        self.assertIs(plans[0]["out"], metadata._dsa_plan)
        self.assertEqual(plans[0]["page_size"], 64)
        # The Eagle advance plans [bs, 1] rows and leaves the per-token rows
        # as the round's refresh published them.
        self.assertEqual(tuple(plans[1]["seq_lens_2d"].shape), (bs, 1))
        self.assertEqual(plans[1]["seq_lens_2d"].view(-1).tolist(), [8, 4])
        self.assertEqual(
            metadata._dsa_seq_lens_2d.view(bs, k).tolist(), [[7] * k, [3] * k]
        )

    def test_dsa_leaf_mtp_frontier_rejects_a_stale_k_row_layout(self):
        from tokenspeed.runtime.layers.attention.backends.paged import dsa as dsa_mod

        backend = dsa_mod.DSABackend.__new__(dsa_mod.DSABackend)
        backend.spec_num_tokens = 4
        backend.kernel_page_size = 64
        metadata = SimpleNamespace(
            seq_lens_k=torch.tensor([9, 5], dtype=torch.int32),
            _dsa_seq_lens_2d=torch.zeros((3, 1), dtype=torch.int32),
            _dsa_plan=None,
        )
        backend._dense_backend = SimpleNamespace(forward_decode_metadata=metadata)

        with self.assertRaisesRegex(RuntimeError, "per-token rows"):
            backend.update_draft_forward_metadata(
                torch.tensor([7, 3], dtype=torch.int32)
            )

        # Rows never published (no refresh ran at this bs): the re-anchor
        # must not allocate them itself.
        metadata._dsa_seq_lens_2d = None
        with self.assertRaisesRegex(RuntimeError, "not published"):
            backend.update_draft_forward_metadata(
                torch.tensor([7, 3], dtype=torch.int32)
            )

        backend._dense_backend = SimpleNamespace(forward_decode_metadata=None)
        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            backend.update_draft_forward_metadata(
                torch.tensor([7, 3], dtype=torch.int32)
            )

    def test_dsa_leaf_refresh_allocates_the_k_rows_once_then_rewrites_in_place(
        self,
    ):
        # The round's refresh and the MTP re-anchor share one publish: the
        # first refresh at a bs fills the dense leaf's declared metadata
        # fields (rows + plan), every later refresh or re-anchor rewrites
        # the same storage and refreshes the plan with out=.
        from tokenspeed.runtime.layers.attention.backends.paged import dsa as dsa_mod
        from tokenspeed.runtime.layers.attention.backends.paged.trtllm_mla import (
            TRTLLMMLADecodeMetadata,
        )

        k, bs = 4, 2
        backend = dsa_mod.DSABackend.__new__(dsa_mod.DSABackend)
        backend.spec_num_tokens = k
        backend.kernel_page_size = 64
        metadata = TRTLLMMLADecodeMetadata(
            seq_lens_k=torch.zeros(bs, dtype=torch.int32)
        )
        self.assertIsNone(metadata._dsa_seq_lens_2d)
        self.assertIsNone(metadata._dsa_plan)
        backend._dense_backend = SimpleNamespace(
            forward_decode_metadata=metadata,
            refresh_decode_metadata=lambda *args, **kwargs: None,
        )
        plan = object()
        plans: list[dict] = []

        def fake_plan(**kw):
            plans.append(kw)
            return plan

        page_table = torch.zeros((bs, 1), dtype=torch.int32)
        with mock.patch.object(dsa_mod, "dsa_plan", side_effect=fake_plan):
            backend.refresh_decode_metadata(
                bs, bs, torch.tensor([9, 5, 99], dtype=torch.int32), page_table
            )
            rows = metadata._dsa_seq_lens_2d
            backend.refresh_decode_metadata(
                bs, bs, torch.tensor([10, 6, 99], dtype=torch.int32), page_table
            )
            backend.update_draft_forward_metadata(
                torch.tensor([7, 3, 99], dtype=torch.int32)
            )

        self.assertIs(metadata._dsa_seq_lens_2d, rows)
        self.assertIs(metadata._dsa_plan, plan)
        self.assertEqual(rows.view(bs, k).tolist(), [[7] * k, [3] * k])
        self.assertEqual(metadata.seq_lens_k.tolist(), [7, 3])
        self.assertEqual([p.get("out") for p in plans], [None, plan, plan])
        for p in plans:
            self.assertIs(p["seq_lens_2d"], rows)
            self.assertEqual(p["page_size"], 64)

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
