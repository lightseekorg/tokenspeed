"""DSpark's greedy walk as one launch and one exchange per round.

The fused round must reproduce the generic walk token for token: the bias is
the BF16 Markov GEMM, the logit is the BF16 sum, and the winner is the first
maximum over the global vocabulary, which across shards means the lowest
token id among equal logits. Ranks are simulated on one device by running
each shard's round and concatenating the words as the all-gather would, or
taking their maximum as an all-reduce MAX would.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tokenspeed.runtime.execution.drafter import dspark as dspark_module
from tokenspeed.runtime.execution.drafter._dspark_walk import (
    dspark_walk_finish,
    dspark_walk_round,
)
from tokenspeed.runtime.execution.drafter.dspark import DSpark
from tokenspeed.runtime.models.dspark import VanillaMarkov

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused walk is a GPU kernel"
)

RANK = 512
HIDDEN = 64


def _reference_walk(base, latent_table, bias_weight, anchor, shards):
    """The generic walk: BF16 bias GEMM, BF16 sum, per-shard first max, then
    the first rank holding the largest maximum."""
    positions, rows, vocab = base.shape
    width = vocab // shards
    tokens = torch.empty((rows, positions + 1), dtype=torch.int32, device=base.device)
    tokens[:, 0] = anchor
    for k in range(1, positions + 1):
        prev = tokens[:, k - 1].long().clamp(0, latent_table.shape[0] - 1)
        bias = torch.matmul(latent_table[prev], bias_weight.T)
        logits = base[k - 1] + bias
        best, ids = torch.max(logits.view(rows, shards, width), dim=-1)
        ids = ids + torch.arange(shards, device=base.device) * width
        rank = torch.argmax(best, dim=-1, keepdim=True)
        tokens[:, k] = torch.gather(ids, 1, rank).view(-1).to(torch.int32)
    return tokens


def _fused_walk(base, latent_table, bias_weight, anchor, shards, reduce_max=False):
    """Every simulated rank runs its shard; all must agree on the block.

    The words are exchanged gathered, or reduced with MAX to one per row.
    """
    positions, rows, vocab = base.shape
    width = vocab // shards
    device = base.device
    decoded = 1 if reduce_max else shards
    tokens = [
        torch.zeros((rows, positions + 1), dtype=torch.int32, device=device)
        for _ in range(shards)
    ]
    for t in range(shards):
        tokens[t][:, 0] = anchor
    gathered = None
    for k in range(1, positions + 1):
        words = []
        for t in range(shards):
            word = torch.zeros(rows, dtype=torch.int64, device=device)
            dspark_walk_round(
                base[k - 1, :, t * width : (t + 1) * width].contiguous(),
                latent_table,
                bias_weight[t * width : (t + 1) * width],
                tokens[t],
                word,
                gathered,
                column=k,
                vocab_start=t * width,
                tp=decoded,
            )
            words.append(word)
        gathered = torch.stack(words).amax(0) if reduce_max else torch.cat(words)
    for t in range(shards):
        dspark_walk_finish(tokens[t], gathered, column=positions, tp=decoded)
    for t in range(1, shards):
        torch.testing.assert_close(tokens[t], tokens[0], rtol=0, atol=0)
    return tokens[0]


def _weights(vocab, seed, scale=0.05):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    latent_table = torch.randn(
        (vocab, RANK), generator=gen, device="cuda", dtype=torch.bfloat16
    )
    bias_weight = (
        torch.randn((vocab, RANK), generator=gen, device="cuda", dtype=torch.bfloat16)
        * scale
    )
    return latent_table, bias_weight, gen


@pytest.mark.parametrize("reduce_max", [False, True])
@pytest.mark.parametrize("rows", [3, 33])
@pytest.mark.parametrize("shards", [1, 4])
def test_the_fused_walk_matches_the_generic_walk(rows, shards, reduce_max) -> None:
    vocab, positions = 4096, 7
    latent_table, bias_weight, gen = _weights(vocab, seed=rows * 10 + shards)
    base = (
        torch.randn(
            (positions, rows, vocab), generator=gen, device="cuda", dtype=torch.bfloat16
        )
        * 4
    )
    anchor = torch.randint(
        0, vocab, (rows,), generator=gen, device="cuda", dtype=torch.int32
    )

    expected = _reference_walk(base, latent_table, bias_weight, anchor, shards)
    produced = _fused_walk(base, latent_table, bias_weight, anchor, shards, reduce_max)
    assert produced.tolist() == expected.tolist()


def test_a_strong_bias_carries_the_chain() -> None:
    """With the base logits flat, every proposal is the previous token's
    bigram winner, so a wrong previous token derails the whole block."""
    vocab, positions, rows = 1024, 7, 4
    latent_table, bias_weight, gen = _weights(vocab, seed=7, scale=1.0)
    base = torch.zeros((positions, rows, vocab), device="cuda", dtype=torch.bfloat16)
    anchor = torch.tensor([0, 5, 77, 1023], device="cuda", dtype=torch.int32)

    expected = _reference_walk(base, latent_table, bias_weight, anchor, 4)
    produced = _fused_walk(base, latent_table, bias_weight, anchor, 4)
    assert produced.tolist() == expected.tolist()
    assert len({tuple(row) for row in produced[:, 1:].tolist()}) > 1


def test_ties_go_to_the_lowest_token_id_across_shards() -> None:
    vocab, positions, rows = 1024, 2, 2
    latent_table, bias_weight, _ = _weights(vocab, seed=3)
    bias_weight.zero_()
    base = torch.full(
        (positions, rows, vocab), -3.0, device="cuda", dtype=torch.bfloat16
    )
    # Row 0: equal maxima in shard 3 and shard 1; row 1: twice inside shard 2.
    base[:, 0, 900] = 2.0
    base[:, 0, 300] = 2.0
    base[:, 1, 700] = 1.5
    base[:, 1, 520] = 1.5
    anchor = torch.zeros(rows, device="cuda", dtype=torch.int32)

    for reduce_max in (False, True):
        produced = _fused_walk(base, latent_table, bias_weight, anchor, 4, reduce_max)
        assert produced[:, 1:].tolist() == [[300, 300], [520, 520]]


def test_negative_logits_order_by_value_not_by_bits() -> None:
    """Negative BF16 values grow in bit pattern as they fall; the packed key
    must still rank -0.5 above -2.0."""
    vocab, positions, rows = 1024, 1, 1
    latent_table, bias_weight, _ = _weights(vocab, seed=5)
    bias_weight.zero_()
    base = torch.full(
        (positions, rows, vocab), -2.0, device="cuda", dtype=torch.bfloat16
    )
    base[0, 0, 811] = -0.5
    base[0, 0, 12] = -7.0
    anchor = torch.zeros(rows, device="cuda", dtype=torch.int32)

    produced = _fused_walk(base, latent_table, bias_weight, anchor, 4)
    assert produced[0, 1].item() == 811


def test_out_of_range_anchors_read_the_boundary_rows() -> None:
    vocab, positions, rows = 1024, 3, 3
    latent_table, bias_weight, gen = _weights(vocab, seed=9, scale=1.0)
    base = torch.zeros((positions, rows, vocab), device="cuda", dtype=torch.bfloat16)
    wild = torch.tensor([-5, vocab + 40, 2**30], device="cuda", dtype=torch.int32)
    clamped = wild.clamp(0, vocab - 1)

    produced = _fused_walk(base, latent_table, bias_weight, wild, 4)
    expected = _fused_walk(base, latent_table, bias_weight, clamped, 4)
    assert produced[:, 1:].tolist() == expected[:, 1:].tolist()


# --------------------------------------------------------------------------
# The drafter takes the fused walk exactly where it fits
# --------------------------------------------------------------------------


class _Head:
    def __init__(self, weight: torch.Tensor, num_added: int = 0) -> None:
        self.weight = weight
        rows = int(weight.shape[0])
        self.num_embeddings = rows
        self.org_vocab_size = rows - num_added
        self.shard_indices = SimpleNamespace(
            num_org_elements=rows - num_added,
            num_org_elements_padded=rows - num_added,
            num_added_elements=num_added,
            org_vocab_start_index=0,
            added_vocab_start_index=rows - num_added,
        )


def _drafter(vocab: int = 2048, spec: int = 8, num_added: int = 0) -> DSpark:
    torch.manual_seed(0)
    drafter = DSpark.__new__(DSpark)
    drafter.spec_num_tokens = spec
    drafter.draft_sampler = None
    drafter.markov_head = VanillaMarkov(vocab_size=vocab, markov_rank=RANK).to(
        device="cuda", dtype=torch.bfloat16
    )
    drafter.lm_head = _Head(
        torch.randn((vocab, HIDDEN), device="cuda", dtype=torch.bfloat16),
        num_added=num_added,
    )
    drafter.logits_processor = Mock(tp_size=1)
    drafter.input_buffers = SimpleNamespace(max_bs=16)
    drafter._ensure_dist_argmax_state = lambda dtype, device: None
    drafter._walk_words = None
    drafter._walk_gathered = None
    drafter._markov_argmax_resolved = False
    drafter._markov_argmax_workspace = None
    return drafter


@pytest.mark.parametrize("rows", [5])
def test_the_drafter_proposes_the_same_block_either_way(rows) -> None:
    drafter = _drafter()
    gen = torch.Generator(device="cuda").manual_seed(rows)
    hidden = torch.randn(
        (rows, drafter.spec_num_tokens - 1, HIDDEN),
        generator=gen,
        device="cuda",
        dtype=torch.bfloat16,
    )
    block = torch.randint(
        0, 2048, (rows, drafter.spec_num_tokens - 1), generator=gen, device="cuda"
    ).to(torch.int32)
    block_logits = drafter._block_base_logits(hidden, full_vocab=False)
    assert drafter._fused_walk_fits(block_logits)

    blocks = {}
    for fused in (False, True):
        tokens = torch.full(
            (rows, drafter.spec_num_tokens), -1, dtype=torch.int32, device="cuda"
        )
        with patch.object(drafter, "_fused_walk_fits", return_value=fused):
            blocks[fused] = drafter._sample_block(hidden, block, tokens).tolist()
    assert blocks[True] == blocks[False]
    assert [row[0] for row in blocks[True]] == block[:, 0].tolist()


@pytest.mark.parametrize("reduced", [True, False])
def test_each_round_reduces_its_words_or_gathers_them(reduced) -> None:
    """A group with a prepared word MAX reduces each round's words in place
    and the next launch decodes one word per row; otherwise the rounds
    all-gather and decode every rank's word."""
    drafter = _drafter()
    group = (0, 1, 2, 3)
    drafter.logits_processor = Mock(tp_size=4, tp_group=group)
    rows, rounds = 3, drafter.spec_num_tokens - 1
    block_logits = torch.zeros(
        (rounds, rows, 2048 // 4), device="cuda", dtype=torch.bfloat16
    )
    tokens = torch.zeros(
        (rows, drafter.spec_num_tokens), dtype=torch.int32, device="cuda"
    )
    backend = Mock()
    backend.try_word_max.return_value = reduced

    with (
        patch.object(dspark_module, "get_global_backend", return_value=backend),
        patch.object(dspark_module, "dspark_walk_round"),
        patch.object(dspark_module, "dspark_walk_finish") as walk_finish,
        patch.object(dspark_module, "all_gather_single") as gather,
    ):
        drafter._fused_greedy_walk(block_logits, tokens)

    decoded = 1 if reduced else 4
    assert walk_finish.call_args.kwargs["tp"] == decoded
    assert gather.call_count == (0 if reduced else rounds)


def test_the_fused_walk_stands_down_where_it_does_not_fit() -> None:
    drafter = _drafter()
    block_logits = torch.zeros((7, 2, 2048), device="cuda", dtype=torch.bfloat16)
    assert drafter._fused_walk_fits(block_logits)
    assert not drafter._fused_walk_fits(block_logits.float())

    # The added-vocab shard competes unbiased; the fused round has no such lane.
    assert not _drafter(num_added=64)._fused_walk_fits(block_logits[..., :1984])

    # A shard short of an equal split leaves some rank with padding or nothing.
    drafter.logits_processor = Mock(tp_size=3)
    assert not drafter._fused_walk_fits(block_logits)
