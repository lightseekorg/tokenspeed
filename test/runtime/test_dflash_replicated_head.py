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

"""``DFlash._replicate_head_over_attention_dp``: the draft's own LM head.

Under ``dp_lm_head_tp`` the target's head is vocab-sharded over attention-DP
ranks, so every route from it to a token is a collective a drafter cannot keep
its peers in (only the rank holding a request drafts). The shards are gathered
once at setup into a rank-local full-vocabulary head the draft reads with no
collective.

Only the process group is missing here, so the gather is substituted by
concatenating the shards a real ``ParallelLMHead`` built; the heads, processor
and geometry are the real classes.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
from transformers import PretrainedConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

import tokenspeed.runtime.execution.drafter.dflash as dflash_drafter  # noqa: E402
from tokenspeed.runtime.execution.drafter.dflash import DFlash  # noqa: E402
from tokenspeed.runtime.layers.logits_processor import LogitsProcessor  # noqa: E402
from tokenspeed.runtime.layers.vocab_parallel_embedding import (  # noqa: E402
    ParallelLMHead,
)

_VOCAB = 256
_HIDDEN = 8
_TP = 4


def _target_vocabulary(vocab_size: int = _VOCAB) -> torch.Tensor:
    """A vocabulary whose every row is distinguishable from every other, so a
    misplaced shard shows up as a wrong row rather than a wrong norm."""
    rows = torch.arange(vocab_size, dtype=torch.float32).unsqueeze(1)
    return (rows + torch.arange(_HIDDEN, dtype=torch.float32)) / vocab_size


def _target_head(
    rank: int,
    tp_size: int,
    *,
    vocab_size: int = _VOCAB,
    padding_size: int = 64,
    added_vocab: int = 0,
    dtype: torch.dtype = torch.float32,
) -> tuple[ParallelLMHead, torch.Tensor]:
    """One rank's shard of the target's head, with the vocabulary loaded."""
    org_vocab_size = vocab_size - added_vocab
    head = ParallelLMHead(
        vocab_size,
        _HIDDEN,
        bias=False,
        params_dtype=dtype,
        org_num_embeddings=org_vocab_size,
        padding_size=padding_size,
        tp_rank=rank,
        tp_size=tp_size,
        tp_group=tuple(range(tp_size)),
        prefix="lm_head",
    )
    vocabulary = _target_vocabulary(vocab_size).to(dtype)
    # The loader takes the base (org) block; an added-vocabulary block is
    # loaded separately, as a LoRA's would be.
    head.weight_loader(head.weight, vocabulary[:org_vocab_size])
    return head, vocabulary


def _processor(
    tp_size: int, *, dp_lm_head_tp: bool, vocab_size: int = _VOCAB
) -> LogitsProcessor:
    """The target's processor for that head's group."""
    if tp_size == 1:
        return LogitsProcessor(
            PretrainedConfig(vocab_size=vocab_size),
            skip_all_gather=True,
            dp_lm_head_tp=False,
        )
    return LogitsProcessor(
        PretrainedConfig(vocab_size=vocab_size),
        skip_all_gather=True,
        tp_rank=0,
        tp_size=tp_size,
        tp_group=tuple(range(tp_size)),
        dp_lm_head_tp=dp_lm_head_tp,
    )


class _DraftModel(torch.nn.Module):
    """Stands in for the loaded draft network: the method reads its config
    only, and a draft checkpoint's vocabulary may differ from the target's."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.config = PretrainedConfig(vocab_size=vocab_size)


def _drafter(
    head: ParallelLMHead,
    processor: LogitsProcessor,
    *,
    draft_vocab_size: int = _VOCAB,
) -> DFlash:
    """A drafter holding just the state the method under test reads.

    ``DFlash.__init__`` needs a model runner with loaded weights, an attention
    backend and a KV pool; the head replication reads none of that, so the
    instance is built without running it.
    """
    drafter = DFlash.__new__(DFlash)
    drafter.model = _DraftModel(draft_vocab_size)
    drafter.lm_head = head
    drafter.logits_processor = processor
    return drafter


@pytest.fixture
def gather_shards(monkeypatch):
    """Substitute the one missing piece: the group collective.

    Returns a recorder the test fills with the other ranks' shards; the
    patched gather concatenates them in rank order around the calling rank's
    own, which is what ``all_gather(..., dim=0)`` produces.
    """
    state: dict[str, object] = {"shards": None, "calls": 0}

    def fake_all_gather(tensor, group, dim=-1):
        state["calls"] = int(state["calls"]) + 1
        assert group == tuple(group), "the gather takes the head's TP group"
        assert dim == 0, "vocabulary shards concatenate along the row axis"
        shards = state["shards"]
        if shards is None:
            raise AssertionError("the test must register the group's shards")
        return torch.cat(list(shards), dim=dim)

    monkeypatch.setattr(dflash_drafter, "all_gather", fake_all_gather)
    return state


def test_a_replicated_head_is_left_alone(gather_shards) -> None:
    """``dp_lm_head_tp`` false is every other layout: the draft keeps the
    target's head and processor, and no collective is attempted."""
    head, _ = _target_head(0, 1)
    processor = _processor(1, dp_lm_head_tp=False)
    drafter = _drafter(head, processor)

    drafter._replicate_head_over_attention_dp()

    assert drafter.lm_head is head
    assert drafter.logits_processor is processor
    assert gather_shards["calls"] == 0


def test_a_vocab_sharded_head_without_attention_dp_is_left_alone(
    gather_shards,
) -> None:
    """A head sharded over ranks that hold the SAME rows needs no replication:
    the target's forward and the drafter enter that group alike, so the
    drafter's own all-gather argmax completes."""
    head, _ = _target_head(0, _TP)
    processor = _processor(_TP, dp_lm_head_tp=False)
    drafter = _drafter(head, processor)

    drafter._replicate_head_over_attention_dp()

    assert drafter.lm_head is head
    assert drafter.logits_processor is processor
    assert gather_shards["calls"] == 0


def test_the_draft_head_holds_the_whole_vocabulary(gather_shards) -> None:
    """The gathered shards reconstruct the target's vocabulary exactly, in
    order: the draft's token ids must mean what the target's do."""
    heads = [_target_head(rank, _TP) for rank in range(_TP)]
    vocabulary = heads[0][1]
    gather_shards["shards"] = [head.weight.detach() for head, _ in heads]
    drafter = _drafter(heads[0][0], _processor(_TP, dp_lm_head_tp=True))

    drafter._replicate_head_over_attention_dp()

    assert drafter.lm_head is not heads[0][0]
    torch.testing.assert_close(
        drafter.lm_head.weight.detach(), vocabulary, rtol=0, atol=0
    )


def test_the_draft_processor_needs_no_collective(gather_shards) -> None:
    """One rank: ``tp_size == 1`` is what makes the draft's argmax local --
    it takes the early return in ``_greedy_argmax_vocab_parallel`` and
    disables the sharded argmax probe (``not 2 <= 1``)."""
    heads = [_target_head(rank, _TP) for rank in range(_TP)]
    gather_shards["shards"] = [head.weight.detach() for head, _ in heads]
    target_processor = _processor(_TP, dp_lm_head_tp=True)
    drafter = _drafter(heads[0][0], target_processor)

    drafter._replicate_head_over_attention_dp()

    processor = drafter.logits_processor
    assert processor is not target_processor
    assert processor.tp_size == 1
    assert not processor.dp_lm_head_tp
    assert not processor.do_argmax
    assert drafter.lm_head.tp_size == 1


def test_the_draft_head_keeps_the_target_vocabulary_width(gather_shards) -> None:
    """The head is the TARGET's, so the rebuilt processor must carry the
    target's vocabulary: it slices its logits to ``config.vocab_size``, and a
    draft checkpoint's own count would cut the wrong width."""
    draft_vocab_size = _VOCAB // 2
    heads = [_target_head(rank, _TP) for rank in range(_TP)]
    gather_shards["shards"] = [head.weight.detach() for head, _ in heads]
    drafter = _drafter(
        heads[0][0],
        _processor(_TP, dp_lm_head_tp=True),
        draft_vocab_size=draft_vocab_size,
    )

    drafter._replicate_head_over_attention_dp()

    assert drafter.model.config.vocab_size == draft_vocab_size
    assert drafter.logits_processor.config.vocab_size == _VOCAB
    assert drafter.lm_head.num_embeddings == _VOCAB


def test_the_draft_head_matches_the_target_row_for_row(gather_shards) -> None:
    """The replicated head produces the logits the sharded one would, which is
    the equivalence the replacement rests on."""
    heads = [_target_head(rank, _TP) for rank in range(_TP)]
    shards = [head.weight.detach() for head, _ in heads]
    gather_shards["shards"] = shards
    drafter = _drafter(heads[0][0], _processor(_TP, dp_lm_head_tp=True))

    drafter._replicate_head_over_attention_dp()

    hidden = torch.randn(3, _HIDDEN)
    sharded = torch.cat([hidden @ shard.T for shard in shards], dim=-1)
    replicated = hidden @ drafter.lm_head.weight.detach().T
    torch.testing.assert_close(replicated, sharded, rtol=0, atol=0)
    assert replicated.argmax(dim=-1).equal(sharded.argmax(dim=-1))


def test_a_padded_vocabulary_keeps_the_shards_row_geometry(gather_shards) -> None:
    """A vocabulary that is not a multiple of the head's padding: the shards
    are padded blocks, so the gather returns ``num_embeddings_padded`` rows
    rather than ``vocab_size``, and the replicated head must hold the same
    count or the vocabulary lands at shifted indices."""
    vocab_size = _VOCAB - 8  # 248: padded to 256, so 62 rows per shard
    heads = [_target_head(rank, _TP, vocab_size=vocab_size) for rank in range(_TP)]
    gather_shards["shards"] = [head.weight.detach() for head, _ in heads]
    source = heads[0][0]
    assert source.num_embeddings_padded > vocab_size
    drafter = _drafter(
        source,
        _processor(_TP, dp_lm_head_tp=True, vocab_size=vocab_size),
        draft_vocab_size=vocab_size,
    )

    drafter._replicate_head_over_attention_dp()

    replicated = drafter.lm_head
    assert replicated.weight.shape[0] == source.num_embeddings_padded
    assert replicated.num_embeddings_padded == source.num_embeddings_padded
    # The vocabulary's own rows survive; the padding tail is not part of it.
    torch.testing.assert_close(
        replicated.weight.detach()[:vocab_size],
        _target_vocabulary(vocab_size),
        rtol=0,
        atol=0,
    )


def test_an_added_vocabulary_head_is_refused(gather_shards) -> None:
    """An added-vocabulary block sits after each rank's PADDED org block, so
    concatenating the shards does not yield one contiguous vocabulary. Refused
    at setup rather than silently drafting shifted token ids."""
    head, _ = _target_head(0, _TP, added_vocab=16)
    assert head.shard_indices.num_added_elements
    drafter = _drafter(head, _processor(_TP, dp_lm_head_tp=True))

    with pytest.raises(ValueError, match="added-vocabulary"):
        drafter._replicate_head_over_attention_dp()


def test_a_quantized_head_is_refused(gather_shards) -> None:
    """The draft matmuls the gathered rows directly, so packed bytes are not
    the vocabulary's values."""
    head, _ = _target_head(0, _TP)
    packed = torch.nn.Parameter(
        torch.zeros_like(head.weight, dtype=torch.uint8), requires_grad=False
    )
    head.weight = packed
    drafter = _drafter(head, _processor(_TP, dp_lm_head_tp=True))

    with pytest.raises(ValueError, match="unquantized"):
        drafter._replicate_head_over_attention_dp()


def test_a_short_gather_is_refused(gather_shards) -> None:
    """A gather that returns fewer rows than the vocabulary means the group's
    shards did not all arrive; drafting on a truncated head would emit token
    ids the target never holds."""
    heads = [_target_head(rank, _TP) for rank in range(_TP)]
    gather_shards["shards"] = [head.weight.detach() for head, _ in heads[:-1]]
    drafter = _drafter(heads[0][0], _processor(_TP, dp_lm_head_tp=True))

    with pytest.raises(ValueError, match="gathered LM head holds"):
        drafter._replicate_head_over_attention_dp()
