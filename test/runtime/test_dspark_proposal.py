"""DSpark's semi-autoregressive proposal and the verify-side state commit.

The draft's backbone pass is non-autoregressive: all block positions are
computed at once from one anchor plus mask tokens. The only thing carrying
token-level dependence inside the block is the Markov head, which adds a
learned bigram bias conditioned on the *previously proposed* token. These tests
pin that chaining, the vocab-shard handling of the bias, and the fact that K3's
recurrent KDA state still gets committed after a DSpark verify.
"""

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.execution.cuda_graph_wrapper import (
    _should_update_mamba_state_after_mtp_verify,
)
from tokenspeed.runtime.execution.drafter.dspark import DSpark
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.models.dspark import VanillaMarkov
from tokenspeed.runtime.models.kimi_k3_dspark import K3DSparkConfidenceHead
from tokenspeed.runtime.sampling.backends.greedy import _verify_chain_greedy_torch

VOCAB = 32
RANK = 4
HIDDEN = 6


def _drafter(spec_num_tokens: int = 8, vocab: int = VOCAB) -> DSpark:
    """A DSpark drafter shell carrying only the proposal state."""
    drafter = DSpark.__new__(DSpark)
    drafter.spec_num_tokens = spec_num_tokens
    torch.manual_seed(0)
    head = VanillaMarkov(vocab_size=vocab, markov_rank=RANK)
    drafter.markov_head = head
    return drafter


# --------------------------------------------------------------------------
# The Markov bias
# --------------------------------------------------------------------------


def test_bias_matches_the_heads_own_step_bias() -> None:
    drafter = _drafter()
    prev = torch.tensor([3, 17])
    bias_fn = drafter._make_step_bias_fn(prev)

    produced = bias_fn(0, VOCAB)
    expected = drafter.markov_head.compute_step_bias(prev)
    torch.testing.assert_close(produced, expected)


def test_bias_slices_match_the_full_bias() -> None:
    """A vocab-parallel shard must see exactly its own slice of the bias."""
    drafter = _drafter()
    prev = torch.tensor([5, 5, 9])
    bias_fn = drafter._make_step_bias_fn(prev)
    full = bias_fn(0, VOCAB)

    for start, count in ((0, 8), (8, 8), (24, 8)):
        torch.testing.assert_close(
            bias_fn(start, count), full[:, start : start + count]
        )


def test_out_of_range_previous_tokens_do_not_index_past_the_embedding() -> None:
    """The anchor is the target's last output and each step's input is a
    vocab-parallel argmax; both can be out of range before the loop's own
    clamp runs, and the Markov head embeds them.
    """
    drafter = _drafter()
    for prev in (
        torch.tensor([-1, 0]),
        torch.tensor([VOCAB, VOCAB + 500]),
        torch.tensor([-7, VOCAB * 3]),
    ):
        bias = drafter._make_step_bias_fn(prev)(0, VOCAB)
        assert bias.shape == (2, VOCAB)
        assert torch.isfinite(bias).all()


def test_clamping_maps_to_the_boundary_rows() -> None:
    drafter = _drafter()
    clamped = drafter._make_step_bias_fn(torch.tensor([-4, VOCAB + 9]))(0, VOCAB)
    expected = drafter._make_step_bias_fn(torch.tensor([0, VOCAB - 1]))(0, VOCAB)
    torch.testing.assert_close(clamped, expected)


def test_bias_is_zero_beyond_the_markov_vocabulary() -> None:
    """The target lm_head's added-vocab shard has no Markov row to read.

    Biasing it with whatever memory follows markov_w2 would let padding tokens
    win the argmax, so the added shard must compete unbiased.
    """
    drafter = _drafter()
    bias_fn = drafter._make_step_bias_fn(torch.tensor([1, 2]))

    beyond = bias_fn(VOCAB, 4)
    assert beyond.shape == (2, 4)
    assert torch.all(beyond == 0)


def test_bias_straddling_the_vocab_edge_is_partly_real_partly_zero() -> None:
    drafter = _drafter()
    prev = torch.tensor([7])
    bias_fn = drafter._make_step_bias_fn(prev)
    full = bias_fn(0, VOCAB)

    straddling = bias_fn(VOCAB - 3, 6)
    torch.testing.assert_close(straddling[:, :3], full[:, VOCAB - 3 :])
    assert torch.all(straddling[:, 3:] == 0)


# --------------------------------------------------------------------------
# The proposal chain
# --------------------------------------------------------------------------


def _install_recording_argmax(drafter: DSpark, lm_head_weight: torch.Tensor) -> list:
    """Replace the vocab-parallel argmax with a recording torch reference."""
    seen: list[tuple[int, torch.Tensor]] = []

    def fake_argmax(hidden, out=None, bias_fn=None):
        logits = hidden.float() @ lm_head_weight.float().T
        if bias_fn is not None:
            logits = logits + bias_fn(0, logits.shape[-1]).float()
        argmax = torch.argmax(logits, dim=-1)
        seen.append((len(seen), argmax.clone()))
        if out is not None:
            out.copy_(argmax.view_as(out))
            return out
        return argmax

    drafter._greedy_argmax_vocab_parallel = fake_argmax
    return seen


def test_anchor_is_copied_and_drafts_fill_the_rest_of_the_block() -> None:
    spec = 8
    drafter = _drafter(spec_num_tokens=spec)
    torch.manual_seed(1)
    lm_head = torch.randn(VOCAB, HIDDEN)
    _install_recording_argmax(drafter, lm_head)

    bs = 2
    draft_hidden = torch.randn(bs, spec, HIDDEN)
    block_ids = torch.full((bs, spec), 11, dtype=torch.int32)
    block_ids[:, 0] = torch.tensor([4, 9], dtype=torch.int32)
    next_tokens = torch.zeros((bs, spec), dtype=torch.int32)

    out = drafter._sample_block(draft_hidden, block_ids, next_tokens)

    # Column 0 is the anchor verbatim; the remaining 7 are proposals.
    assert out[:, 0].tolist() == [4, 9]
    assert out.shape == (bs, spec)


@pytest.mark.parametrize("verify_width", [2, 3, 4, 6, 8])
def test_proposal_shape_generalizes_across_verify_widths(verify_width: int) -> None:
    drafter = _drafter(spec_num_tokens=verify_width)
    _install_recording_argmax(drafter, torch.randn(VOCAB, HIDDEN))
    block_ids = torch.zeros((2, verify_width), dtype=torch.int32)
    block_ids[:, 0] = torch.tensor([3, 7], dtype=torch.int32)
    output = drafter._sample_block(
        torch.randn(2, verify_width, HIDDEN),
        block_ids,
        torch.zeros_like(block_ids),
    )
    assert output.shape == (2, verify_width)
    assert torch.equal(output[:, 0], block_ids[:, 0])


@pytest.mark.parametrize("verify_width", [2, 3, 4, 6, 8])
def test_greedy_verify_accepts_a_matching_variable_width_chain(
    verify_width: int,
) -> None:
    candidates = torch.arange(verify_width, dtype=torch.int32).view(1, -1)
    target_predict = torch.zeros((1, verify_width), dtype=torch.int64)
    target_predict[:, :-1] = candidates[:, 1:]
    predicts = torch.empty(verify_width, dtype=torch.int32)
    accept_index = torch.full((1, verify_width), -1, dtype=torch.int32)
    accepted_drafts = torch.zeros(1, dtype=torch.int32)

    _verify_chain_greedy_torch(
        predicts,
        accept_index,
        accepted_drafts,
        candidates,
        target_predict,
        batch_size=1,
        num_draft_tokens=verify_width,
    )
    assert accepted_drafts.item() == verify_width - 1


def test_each_step_reads_the_hidden_one_position_back() -> None:
    """Block position k is proposed from draft_hidden[k-1], not [k].

    This is the off-by-one PR #829's last commit fixed; getting it wrong shifts
    every draft by one position and quietly halves acceptance.
    """
    spec = 4
    drafter = _drafter(spec_num_tokens=spec)
    lm_head = torch.zeros(VOCAB, HIDDEN)
    # Make position p's hidden select token p deterministically.
    for p in range(spec):
        lm_head[p, p % HIDDEN] = 1.0
    drafter.markov_head.markov_w2.weight.data.zero_()

    draft_hidden = torch.zeros(1, spec, HIDDEN)
    for p in range(spec):
        draft_hidden[0, p, p % HIDDEN] = 10.0

    _install_recording_argmax(drafter, lm_head)
    next_tokens = torch.zeros((1, spec), dtype=torch.int32)
    block_ids = torch.zeros((1, spec), dtype=torch.int32)

    out = drafter._sample_block(draft_hidden, block_ids, next_tokens)
    # Token at column k came from hidden row k-1, which selects token k-1.
    assert out[0, 1:].tolist() == [0, 1, 2]


def test_the_chain_conditions_on_the_previous_proposal() -> None:
    """Changing only the anchor must change the downstream proposals.

    If the Markov bias were dropped, the block would be a pure function of the
    hidden states and the anchor would not propagate at all.
    """
    spec = 5
    torch.manual_seed(3)
    lm_head = torch.randn(VOCAB, HIDDEN)
    draft_hidden = torch.randn(1, spec, HIDDEN)

    proposals = []
    for anchor in (2, 21):
        drafter = _drafter(spec_num_tokens=spec)
        # A strong, token-dependent bias so the chain is visible.
        drafter.markov_head.markov_w1.weight.data.normal_(0.0, 3.0)
        drafter.markov_head.markov_w2.weight.data.normal_(0.0, 3.0)
        _install_recording_argmax(drafter, lm_head)
        block_ids = torch.zeros((1, spec), dtype=torch.int32)
        block_ids[0, 0] = anchor
        next_tokens = torch.zeros((1, spec), dtype=torch.int32)
        proposals.append(
            drafter._sample_block(draft_hidden, block_ids, next_tokens).clone()
        )

    assert proposals[0][0, 1:].tolist() != proposals[1][0, 1:].tolist()


def test_proposals_are_valid_token_ids() -> None:
    """clamp_(min=0) guards the vocab-parallel all-shards-lost case."""
    spec = 6
    drafter = _drafter(spec_num_tokens=spec)
    _install_recording_argmax(drafter, torch.randn(VOCAB, HIDDEN))
    out = drafter._sample_block(
        torch.randn(1, spec, HIDDEN),
        torch.zeros((1, spec), dtype=torch.int32),
        torch.zeros((1, spec), dtype=torch.int32),
    )
    assert int(out.min()) >= 0


def test_confidence_scores_are_recorded_per_proposal_position() -> None:
    spec = 4
    drafter = _drafter(spec_num_tokens=spec)
    drafter.confidence_head = K3DSparkConfidenceHead(HIDDEN, RANK, with_markov=True)
    drafter.confidence_scores_buf = torch.empty((1, spec - 1))
    _install_recording_argmax(drafter, torch.randn(VOCAB, HIDDEN))
    drafter._sample_block(
        torch.randn(1, spec, HIDDEN),
        torch.zeros((1, spec), dtype=torch.int32),
        torch.zeros((1, spec), dtype=torch.int32),
    )
    scores = drafter.confidence_scores_buf
    assert scores.shape == (1, spec - 1)
    assert bool(((scores >= 0) & (scores <= 1)).all())


# --------------------------------------------------------------------------
# Recurrent (KDA) state commit after verify
# --------------------------------------------------------------------------


class _BackendWithCommit:
    def update_mamba_state_after_mtp_verify(self, accept_lengths, model):
        return None


class _BackendWithoutCommit:
    pass


def test_kda_commit_fires_for_a_dspark_drafter() -> None:
    """K3's recurrent state must be committed after a DSpark verify too.

    The hook keys on the backend, not the algorithm, so DSpark inherits it --
    this test is what keeps that true if the predicate ever grows an
    algorithm check.
    """
    assert _should_update_mamba_state_after_mtp_verify(
        drafter=DSpark.__new__(DSpark),
        attn_backend=_BackendWithCommit(),
        forward_mode=ForwardMode.DECODE,
    )


def test_kda_commit_is_skipped_without_a_drafter() -> None:
    assert not _should_update_mamba_state_after_mtp_verify(
        drafter=None,
        attn_backend=_BackendWithCommit(),
        forward_mode=ForwardMode.DECODE,
    )


def test_kda_commit_is_skipped_on_a_stateless_backend() -> None:
    assert not _should_update_mamba_state_after_mtp_verify(
        drafter=DSpark.__new__(DSpark),
        attn_backend=_BackendWithoutCommit(),
        forward_mode=ForwardMode.DECODE,
    )


def test_kda_commit_is_skipped_outside_decode() -> None:
    """Prefill writes state inline; there is no verify to commit."""
    assert not _should_update_mamba_state_after_mtp_verify(
        drafter=DSpark.__new__(DSpark),
        attn_backend=_BackendWithCommit(),
        forward_mode=ForwardMode.EXTEND,
    )
