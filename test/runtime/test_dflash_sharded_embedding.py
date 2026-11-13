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

"""A block draft embeds its block ids through the TARGET's embedding, which may
be vocab-sharded over the attention TP group.

``VocabParallelEmbedding._partial_lookup`` answers a sharded lookup with this
rank's rows and ZERO rows for every id outside its shard: the sum over the
group is the embedding, one rank's view is not. A draft that reads an
unreduced lookup therefore runs its whole forward on zeros wherever the id
belongs to a peer -- no error, just a draft that predicts nothing. Serving saw
acceptance 1.29 instead of 3.9 from exactly this.

Only the process group is missing here, so the reduce is substituted by summing
the shards real ``VocabParallelEmbedding`` instances return; the embedding and
its sharding are the real classes.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")

from tokenspeed.runtime.layers.vocab_parallel_embedding import (  # noqa: E402
    VocabParallelEmbedding,
)

_VOCAB = 256
_HIDDEN = 8
_TP = 8


def _vocabulary() -> torch.Tensor:
    """Every row distinguishable from every other, so a missing shard shows up
    as a wrong row rather than a wrong norm."""
    rows = torch.arange(_VOCAB, dtype=torch.float32).unsqueeze(1)
    return (rows + 1.0) * (torch.arange(_HIDDEN, dtype=torch.float32) + 1.0)


def _shard(rank: int, tp_size: int) -> VocabParallelEmbedding:
    """One rank's shard of the target's embedding, with the vocabulary loaded."""
    embedding = VocabParallelEmbedding(
        _VOCAB,
        _HIDDEN,
        params_dtype=torch.float32,
        tp_rank=rank,
        tp_size=tp_size,
        tp_group=tuple(range(tp_size)),
        prefix="embed_tokens",
    )
    embedding.weight_loader(embedding.weight, _vocabulary())
    return embedding


def test_one_rank_of_a_sharded_embedding_is_not_the_embedding():
    """The premise of the reduce: a single shard zeroes other ranks' ids.

    Without this, a test that only checks the summed rows would pass even if
    the lookup were replicated, and would not pin why the reduce is needed.
    """
    ids = torch.arange(_VOCAB, dtype=torch.long)
    shards = [_shard(rank, _TP) for rank in range(_TP)]
    per_rank = [s(ids, reduce_results=False) for s in shards]

    zero_rows = [int((rows.abs().sum(dim=-1) == 0).sum()) for rows in per_rank]
    assert all(count > 0 for count in zero_rows), (
        "every rank of an 8-way shard must zero the ids it does not hold; "
        f"got zero-row counts {zero_rows}"
    )
    # Each id is held by exactly one rank, so the zeroed rows must account for
    # the whole vocabulary on every rank but its owner.
    assert sum(zero_rows) == _VOCAB * (_TP - 1), (
        f"ids are owned by one rank each: expected {_VOCAB * (_TP - 1)} zeroed "
        f"rows across the group, got {sum(zero_rows)}"
    )


def test_summing_the_shards_reproduces_the_embedding():
    """What ``reduce_results=True`` computes: the group sum is the vocabulary.

    The drafter reads this sum (``DFlash._draft_native`` embeds its block ids
    through the target's table). Comparing against the loaded vocabulary, not
    against another lookup, is what makes a silently dropped reduce visible.
    """
    ids = torch.arange(_VOCAB, dtype=torch.long)
    shards = [_shard(rank, _TP) for rank in range(_TP)]

    reduced = sum(s(ids, reduce_results=False) for s in shards)
    torch.testing.assert_close(reduced, _vocabulary(), rtol=0, atol=0)


def test_an_unsharded_embedding_needs_no_reduce():
    """Attention DP holds the whole table on every rank, so an unreduced
    lookup is already the embedding -- the reduce is a no-op there, which is
    why a sharded target is the only one that breaks."""
    ids = torch.arange(_VOCAB, dtype=torch.long)
    sole = _shard(0, 1)

    torch.testing.assert_close(
        sole(ids, reduce_results=False), _vocabulary(), rtol=0, atol=0
    )
