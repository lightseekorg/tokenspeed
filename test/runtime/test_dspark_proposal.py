"""DSpark's semi-autoregressive proposal and the verify-side state commit.

The draft's backbone pass is non-autoregressive: all block positions are
computed at once from one anchor plus mask tokens. The only thing carrying
token-level dependence inside the block is the Markov head, which adds a
learned bigram bias conditioned on the *previously proposed* token. These tests
pin that chaining, the vocab-shard handling of the bias, and the fact that K3's
recurrent KDA state still gets committed after a DSpark verify.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.drafter.deepseek_v4_dspark import DeepseekV4DSpark
from tokenspeed.runtime.execution.drafter.dspark import DSpark
from tokenspeed.runtime.models.dspark import VanillaMarkov

VOCAB = 32
RANK = 4
HIDDEN = 6


class _DeviceLengthReadBomb:
    def __getitem__(self, key):
        raise AssertionError(f"device length buffer was read with {key!r}")


class _RecordingV4DSparkModel:
    window_size = 2

    def __init__(self) -> None:
        self.writes = []

    def write_context_windows_batched(
        self,
        hidden,
        positions,
        slots,
        valid,
        kv_windows,
        first_padding_slot,
    ) -> None:
        self.writes.append(
            (
                hidden.clone(),
                positions.clone(),
                slots.clone(),
                valid.clone(),
                kv_windows,
                first_padding_slot,
            )
        )


def _v4_dspark_window_shell(lengths: torch.Tensor) -> DeepseekV4DSpark:
    drafter = DeepseekV4DSpark.__new__(DeepseekV4DSpark)
    drafter.input_buffers = SimpleNamespace(
        extend_seq_lens_cpu=lengths,
        input_lengths_buf=_DeviceLengthReadBomb(),
        positions_buf=torch.arange(32, dtype=torch.int64),
    )
    drafter.model = _RecordingV4DSparkModel()
    drafter.slot_indices_buf = torch.tensor([2, 4, 6, 8], dtype=torch.int64)
    drafter.kv_windows = object()
    drafter.first_padding_slot = 10
    drafter.context_lengths = torch.zeros(12, dtype=torch.int64)
    return drafter


def _drafter(spec_num_tokens: int = 8, vocab: int = VOCAB) -> DSpark:
    """A DSpark drafter shell carrying only the proposal state."""
    drafter = DSpark.__new__(DSpark)
    drafter.spec_num_tokens = spec_num_tokens
    torch.manual_seed(0)
    head = VanillaMarkov(vocab_size=vocab, markov_rank=RANK)
    drafter.markov_head = head
    return drafter


# --------------------------------------------------------------------------
# DeepSeek V4 prefill window seeding
# --------------------------------------------------------------------------


def test_v4_prefill_window_seeding_uses_the_cpu_length_mirror() -> None:
    drafter = _v4_dspark_window_shell(torch.tensor([2, 3, 0], dtype=torch.int32))
    hidden = torch.arange(10, dtype=torch.float32).reshape(5, 2)

    consumed = drafter._seed_prefill_windows(hidden, num_extends=3)

    assert consumed == 5
    assert len(drafter.model.writes) == 2
    first, second = drafter.model.writes
    torch.testing.assert_close(first[0], hidden[:2].unsqueeze(0))
    torch.testing.assert_close(second[0], hidden[3:5].unsqueeze(0))
    assert first[1].tolist() == [[0, 1]]
    assert second[1].tolist() == [[3, 4]]
    assert first[2].tolist() == [2]
    assert second[2].tolist() == [4]
    assert drafter.context_lengths[[2, 4, 6]].tolist() == [2, 5, 0]


def test_v4_mixed_window_seeding_reads_only_prefill_rows() -> None:
    drafter = _v4_dspark_window_shell(torch.tensor([2, 3, 99], dtype=torch.int32))
    hidden = torch.arange(22, dtype=torch.float32).reshape(11, 2)

    consumed = drafter._seed_prefill_windows(hidden, num_extends=2)

    assert consumed == 5
    assert len(drafter.model.writes) == 2


def test_v4_decode_window_seeding_does_not_read_any_length_buffer() -> None:
    drafter = _v4_dspark_window_shell(torch.empty(0, device="meta"))
    drafter.input_buffers.extend_seq_lens_cpu = _DeviceLengthReadBomb()

    assert drafter._seed_prefill_windows(torch.empty(0, 2), num_extends=0) == 0
    assert drafter.model.writes == []


@pytest.mark.parametrize(
    ("lengths", "num_extends", "hidden_rows", "message"),
    (
        (torch.tensor([1], dtype=torch.int64), 1, 1, "int32 CPU"),
        (torch.empty(1, dtype=torch.int32, device="meta"), 1, 1, "int32 CPU"),
        (torch.tensor([1], dtype=torch.int32), 2, 2, "int32 CPU"),
        (torch.tensor([-1], dtype=torch.int32), 1, 1, "non-negative"),
        (torch.tensor([3], dtype=torch.int32), 1, 2, "exceed"),
    ),
)
def test_v4_prefill_window_seeding_rejects_invalid_cpu_mirrors(
    lengths: torch.Tensor,
    num_extends: int,
    hidden_rows: int,
    message: str,
) -> None:
    drafter = _v4_dspark_window_shell(lengths)

    with pytest.raises(RuntimeError, match=message):
        drafter._seed_prefill_windows(
            torch.empty(hidden_rows, 2), num_extends=num_extends
        )


def test_v4_prefill_window_seeding_rejects_negative_num_extends() -> None:
    drafter = _v4_dspark_window_shell(torch.empty(0, dtype=torch.int32))

    with pytest.raises(ValueError, match="non-negative"):
        drafter._seed_prefill_windows(torch.empty(0, 2), num_extends=-1)


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
    draft_hidden = torch.randn(bs, spec - 1, HIDDEN)
    block_ids = torch.full((bs, spec - 1), 11, dtype=torch.int32)
    block_ids[:, 0] = torch.tensor([4, 9], dtype=torch.int32)
    next_tokens = torch.zeros((bs, spec), dtype=torch.int32)

    out = drafter._sample_block(draft_hidden, block_ids, next_tokens)

    # Column 0 is the anchor verbatim; the remaining 7 are proposals.
    assert out[:, 0].tolist() == [4, 9]
    assert out.shape == (bs, spec)


def test_each_step_reads_the_hidden_one_position_back() -> None:
    """Block position k is proposed from draft_hidden[k-1], not [k].

    This is the off-by-one PR #829's last commit fixed; getting it wrong shifts
    every draft by one position and quietly halves acceptance.
    """
    spec = 4
    drafter = _drafter(spec_num_tokens=spec)
    lm_head = torch.zeros(VOCAB, HIDDEN)
    # Make position p's hidden select token p deterministically.
    for p in range(spec - 1):
        lm_head[p, p % HIDDEN] = 1.0
    drafter.markov_head.markov_w2.weight.data.zero_()

    draft_hidden = torch.zeros(1, spec - 1, HIDDEN)
    for p in range(spec - 1):
        draft_hidden[0, p, p % HIDDEN] = 10.0

    _install_recording_argmax(drafter, lm_head)
    next_tokens = torch.zeros((1, spec), dtype=torch.int32)
    block_ids = torch.zeros((1, spec - 1), dtype=torch.int32)

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
    draft_hidden = torch.randn(1, spec - 1, HIDDEN)

    proposals = []
    for anchor in (2, 21):
        drafter = _drafter(spec_num_tokens=spec)
        # A strong, token-dependent bias so the chain is visible.
        drafter.markov_head.markov_w1.weight.data.normal_(0.0, 3.0)
        drafter.markov_head.markov_w2.weight.data.normal_(0.0, 3.0)
        _install_recording_argmax(drafter, lm_head)
        block_ids = torch.zeros((1, spec - 1), dtype=torch.int32)
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
        torch.randn(1, spec - 1, HIDDEN),
        torch.zeros((1, spec - 1), dtype=torch.int32),
        torch.zeros((1, spec), dtype=torch.int32),
    )
    assert int(out.min()) >= 0


# --------------------------------------------------------------------------
# Recurrent (KDA) state commit after verify
# --------------------------------------------------------------------------


def test_kda_commit_is_a_base_no_op_hook() -> None:
    """K3's recurrent state must be committed after a DSpark verify too.

    The runner calls update_mamba_state_after_mtp_verify unconditionally
    after every drafted decode round (no hasattr probe, no algorithm check):
    the hook keys on the backend override, stateless backends inherit the
    base no-op. This keeps DSpark covered if the call site ever grows an
    algorithm check, and keeps stateless backends safe without a guard.
    """
    from tokenspeed.runtime.layers.attention.backends.base import (
        AttentionBackend,
    )

    class _StatelessBackend(AttentionBackend):
        def init_forward_metadata(self, *args, **kwargs):
            pass

    backend = _StatelessBackend.__new__(_StatelessBackend)
    assert backend.update_mamba_state_after_mtp_verify(None) is None


class _ShardIndices:
    def __init__(self, num_org: int) -> None:
        self.num_org_elements = num_org


class _ShardedHead:
    """Enough of a vocab-parallel head for the block projection to run."""

    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight
        self.shard_indices = _ShardIndices(weight.shape[0])


def test_the_walk_feeds_each_round_its_own_positions_hoisted_slice() -> None:
    """The whole proposal walk must hand round k the hoisted projection of
    position k-1, bit-for-bit what projecting that position alone gives."""
    torch.manual_seed(11)
    drafter = _drafter()
    weight = torch.randn(VOCAB, HIDDEN)
    drafter.lm_head = _ShardedHead(weight)
    received: list[torch.Tensor] = []

    def fake_argmax(hidden, out=None, bias_fn=None, base_logits=None):
        assert base_logits is not None, "the walk must use the hoisted slice"
        assert base_logits.is_contiguous()
        received.append(base_logits.clone())
        logits = base_logits.float()
        if bias_fn is not None:
            logits = logits + bias_fn(0, logits.shape[-1]).float()
        argmax = torch.argmax(logits, dim=-1)
        if out is not None:
            out.copy_(argmax.view_as(out))
            return out
        return argmax

    drafter._greedy_argmax_vocab_parallel = fake_argmax
    hidden = torch.randn(3, drafter.spec_num_tokens, HIDDEN)
    ids = torch.zeros(3, drafter.spec_num_tokens, dtype=torch.long)
    block = torch.randint(0, VOCAB, (3, drafter.spec_num_tokens))
    drafter._sample_block(hidden, block, ids)

    assert len(received) == drafter.spec_num_tokens - 1
    for k, got in enumerate(received, start=1):
        want = hidden[:, k - 1, :].to(weight.dtype) @ weight.T
        assert torch.equal(got, want)


def test_the_block_projection_matches_projecting_each_position() -> None:
    """The walk is semi-autoregressive, so the projection is hoisted out of it
    while the bias and the argmax stay behind. A layout slip here would feed
    each position another position's logits and go unnoticed: every proposal
    would still be a valid token id."""
    rows, positions = 3, 8
    drafter = _drafter(spec_num_tokens=positions)
    torch.manual_seed(1)
    weight = torch.randn(VOCAB, HIDDEN)
    drafter.lm_head = _ShardedHead(weight)
    draft_hidden = torch.randn(rows, positions, HIDDEN)

    block = drafter._block_base_logits(draft_hidden)

    assert block is not None
    assert block.shape == (positions, rows, VOCAB)
    for k in range(positions):
        expected = torch.matmul(draft_hidden[:, k, :], weight.T)
        assert torch.allclose(block[k], expected, atol=1e-5), f"position {k}"


def test_the_block_projection_stands_down_without_a_sharded_head() -> None:
    """Drafters whose head carries no shard metadata keep the per-step path."""
    drafter = _drafter(spec_num_tokens=4)
    assert drafter._block_base_logits(torch.randn(2, 4, HIDDEN)) is None

    drafter.lm_head = _ShardedHead(torch.randn(0, HIDDEN))
    assert drafter._block_base_logits(torch.randn(2, 4, HIDDEN)) is None
