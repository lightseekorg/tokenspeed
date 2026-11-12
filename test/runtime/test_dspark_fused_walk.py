"""DSpark's greedy walk as one launch and one all-gather per round.

The fused round must reproduce the generic walk token for token: the bias is
the BF16 Markov GEMM, the logit is the BF16 sum, and the winner is the first
maximum over the global vocabulary, which across shards means the lowest
token id among equal logits. Ranks are simulated on one device by running
each shard's round and concatenating the words as the all-gather would.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tokenspeed.runtime.execution.drafter._dspark_walk import (
    dspark_walk_finish,
    dspark_walk_round,
)
from tokenspeed.runtime.execution.drafter.dspark import DSpark
from tokenspeed.runtime.models.dspark import VanillaMarkov
from tokenspeed.runtime.utils.env import envs

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


def _fused_walk(base, latent_table, bias_weight, anchor, shards):
    """Every simulated rank runs its shard; all must agree on the block."""
    positions, rows, vocab = base.shape
    width = vocab // shards
    device = base.device
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
                tp=shards,
            )
            words.append(word)
        gathered = torch.cat(words)
    for t in range(shards):
        dspark_walk_finish(tokens[t], gathered, column=positions, tp=shards)
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


@pytest.mark.parametrize("rows", [1, 3, 16, 33])
@pytest.mark.parametrize("shards", [1, 4])
def test_the_fused_walk_matches_the_generic_walk(rows, shards) -> None:
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
    produced = _fused_walk(base, latent_table, bias_weight, anchor, shards)
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

    produced = _fused_walk(base, latent_table, bias_weight, anchor, 4)
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
    return drafter


@pytest.mark.parametrize("rows", [1, 5, 16])
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
    block_logits = drafter._block_base_logits(hidden)
    assert drafter._fused_walk_fits(block_logits)

    blocks = {}
    for fused in (False, True):
        tokens = torch.full(
            (rows, drafter.spec_num_tokens), -1, dtype=torch.int32, device="cuda"
        )
        with envs.TOKENSPEED_DSPARK_FUSED_WALK.override(fused):
            blocks[fused] = drafter._sample_block(hidden, block, tokens).tolist()
    assert blocks[True] == blocks[False]
    assert [row[0] for row in blocks[True]] == block[:, 0].tolist()


def test_the_fused_walk_stands_down_where_it_does_not_fit() -> None:
    drafter = _drafter()
    block_logits = torch.zeros((7, 2, 2048), device="cuda", dtype=torch.bfloat16)
    assert drafter._fused_walk_fits(block_logits)

    with envs.TOKENSPEED_DSPARK_FUSED_WALK.override(False):
        assert not drafter._fused_walk_fits(block_logits)
    assert not drafter._fused_walk_fits(block_logits.float())

    # The added-vocab shard competes unbiased; the fused round has no such lane.
    assert not _drafter(num_added=64)._fused_walk_fits(block_logits[..., :1984])

    # A shard short of an equal split leaves some rank with padding or nothing.
    drafter.logits_processor = Mock(tp_size=3)
    assert not drafter._fused_walk_fits(block_logits)
    drafter.logits_processor = Mock(tp_size=1)

    # A distributed argmax state keeps the platform's own fused walk.
    drafter._ensure_dist_argmax_state = lambda dtype, device: object()
    assert not drafter._fused_walk_fits(block_logits)
