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

"""A drafter sampling from its head's vocabulary shard draws and records what whole rows do.

Under ``--enable-speculative-sampling`` with the target's rows vocab-sharded
and a verifier reading q per shard, each rank samples its shard of the draft
logits and records q over its own columns
(``DraftProposalSampler.propose_vocab_shard``). One GPU plays every rank: the
ranks' gather of the draft partials is answered by reducing the other ranks'
shards in-process, as ``test_sonic_vocab_shard.py`` answers the candidate
gather. A hot-token head samples its gathered logits and each rank records
its own columns of q (``DraftProposalSampler.propose``).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(
    est_time=60,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="The whole-row sampler compared against uses flashinfer's softmax.",
)

from tokenspeed_kernel.ops.sampling import vocab_shard_draft_stats  # noqa: E402
from tokenspeed_kernel.platform import current_platform  # noqa: E402

from tokenspeed.runtime.execution.drafter.speculative_sampling import (  # noqa: E402
    DRAFT_SAMPLE_OFFSET_SALT,
    DraftProposalSampler,
)
from tokenspeed.runtime.execution.input_buffer import InputBuffers  # noqa: E402
from tokenspeed.runtime.execution.runtime_states import RuntimeStates  # noqa: E402
from tokenspeed.runtime.layers import sampled_vocab_shard as shard_module  # noqa: E402
from tokenspeed.runtime.layers.sampled_vocab_shard import (  # noqa: E402
    SampledVocabShard,
)
from tokenspeed.runtime.sampling.backends.base import (  # noqa: E402
    SpeculativeSamplingPools,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and current_platform().is_nvidia),
    reason="the whole-row sampler uses flashinfer's softmax",
)

RANKS = 4
COLUMNS = 6000
VOCAB = RANKS * COLUMNS
POOL = 8
N = 3
MAX_BS = 8
GREEDY_SLOT = 3
STEP = 1


def _pools() -> SpeculativeSamplingPools:
    """Per-slot temperatures, slot 3 greedy, distinct seeds."""
    top_k = torch.full((POOL + 1,), 128, dtype=torch.int32, device="cuda")
    top_k[GREEDY_SLOT] = 1
    return SpeculativeSamplingPools(
        temperature=torch.tensor(
            [1.0, 0.7, 1.0, 1.0, 1.4, 0.3, 1.0, 2.0, 1.0], device="cuda"
        ),
        top_k=top_k,
        top_p=torch.ones(POOL + 1, device="cuda"),
        seed=torch.arange(POOL + 1, dtype=torch.int64, device="cuda") * 1009 + 7,
    )


def _sampler(
    pools: SpeculativeSamplingPools,
    slots: torch.Tensor,
    columns: int | None,
    vocab_map: torch.Tensor | None = None,
    vocab_start: int = 0,
) -> tuple[DraftProposalSampler, RuntimeStates]:
    """A drafter's sampler over a table of whole rows, or of ``columns`` per rank from ``vocab_start``, serving ``slots``."""
    states = RuntimeStates(
        req_pool_size=POOL,
        input_vocab_size=VOCAB,
        vocab_size=VOCAB,
        output_length=N,
        device="cuda",
    )
    states.init_draft_probs(
        spec_num_tokens=N,
        reject_threshold=2.0,
        columns=columns,
        vocab_start=vocab_start,
    )
    states.valid_cache_lengths.copy_(torch.arange(POOL + 1, dtype=torch.int32) * 37)
    buffers = InputBuffers(
        max_bs=MAX_BS,
        max_num_tokens=MAX_BS * N,
        state_write_padding_pool_index=POOL,
        device="cuda",
    )
    buffers.req_pool_indices_buf[: slots.numel()] = slots
    buffers.state_write_req_pool_indices_buf[: slots.numel()] = slots
    sampler = DraftProposalSampler(
        pools=pools,
        runtime_states=states,
        input_buffers=buffers,
        spec_num_tokens=N,
        vocab_map=vocab_map,
        device="cuda",
    )
    return sampler, states


def _answer_gather(monkeypatch, logits, rank, slots, pools, states) -> None:
    """The ranks' gather of the draft partials: this rank's staged block, the other ranks' reduced in-process."""
    offsets = states.valid_cache_lengths.long() * N + DRAFT_SAMPLE_OFFSET_SALT + STEP

    def all_gather_single(landing, staged, group):
        rows = staged.shape[0]
        for peer in range(RANKS):
            block = landing[peer * rows : (peer + 1) * rows]
            if peer == rank:
                block.copy_(staged)
                continue
            vocab_shard_draft_stats(
                logits[:, peer * COLUMNS : (peer + 1) * COLUMNS].contiguous(),
                peer * COLUMNS,
                slots,
                pools.temperature,
                pools.seed,
                offsets,
                block,
                logit_scale=None,
                softcap=None,
            )

    monkeypatch.setattr(shard_module, "all_gather_single", all_gather_single)


def _shard(logits: torch.Tensor, rank: int) -> SampledVocabShard:
    return SampledVocabShard(
        logits=logits[:, rank * COLUMNS : (rank + 1) * COLUMNS].contiguous(),
        vocab_start=rank * COLUMNS,
        logit_scale=None,
        softcap=None,
        tp_size=RANKS,
        tp_group=tuple(range(RANKS)),
        gather_full=None,
    )


def _untouched(states: RuntimeStates, slots: torch.Tensor) -> bool:
    """Whether every step and slot but the recorded ones keeps the sentinel."""
    untouched = torch.ones(POOL + 1, N, dtype=torch.bool, device="cuda")
    untouched[slots, STEP] = False
    untouched[:, -1] = False
    return bool((states.draft_probs[untouched] == states.draft_probs_sentinel).all())


def test_each_rank_draws_the_whole_row_token_and_records_its_columns(monkeypatch):
    torch.manual_seed(11)
    slots = torch.tensor([1, 2, 3, 4, 5, 7], dtype=torch.int64, device="cuda")
    pools = _pools()
    logits = (torch.randn(slots.numel(), VOCAB, device="cuda") * 3).bfloat16()
    # The greedy request's one clear maximum, in shard 2.
    logits[2, 2 * COLUMNS + 99] = 40.0
    whole, whole_states = _sampler(pools, slots, None)
    expected = whole.propose(logits, step=STEP).clone()
    expected_q = whole_states.draft_probs[:, STEP]

    for rank in range(RANKS):
        sampler, states = _sampler(pools, slots, COLUMNS)
        _answer_gather(monkeypatch, logits, rank, slots, pools, states)

        tokens = sampler.propose_vocab_shard(_shard(logits, rank), step=STEP)

        assert torch.equal(tokens, expected), rank
        columns = slice(rank * COLUMNS, (rank + 1) * COLUMNS)
        torch.testing.assert_close(
            states.draft_probs[slots, STEP],
            expected_q[slots, columns],
            rtol=2e-4,
            atol=1e-7,
        )
        assert _untouched(states, slots), rank
    assert int(expected[2]) == 2 * COLUMNS + 99


def test_a_hot_token_head_draws_what_whole_rows_draw_and_records_its_columns():
    torch.manual_seed(13)
    slots = torch.tensor([1, 2, 3, 4, 5, 7], dtype=torch.int64, device="cuda")
    pools = _pools()
    # Hot tokens in the first three shards only, so the last rank records zeros.
    hot = torch.sort(torch.randperm(3 * COLUMNS, device="cuda")[:5000]).values
    logits = (torch.randn(slots.numel(), hot.numel(), device="cuda") * 3).bfloat16()
    # The greedy request's one clear maximum.
    logits[2, 77] = 40.0
    whole, whole_states = _sampler(pools, slots, None, vocab_map=hot)
    expected = whole.propose(logits, step=STEP).clone()
    expected_q = whole_states.draft_probs[:, STEP]

    for rank in range(RANKS):
        sampler, states = _sampler(
            pools, slots, COLUMNS, vocab_map=hot, vocab_start=rank * COLUMNS
        )

        tokens = sampler.propose(logits, step=STEP)

        assert torch.equal(tokens, expected), rank
        columns = slice(rank * COLUMNS, (rank + 1) * COLUMNS)
        assert torch.equal(
            states.draft_probs[slots, STEP], expected_q[slots, columns]
        ), rank
        assert _untouched(states, slots), rank
    assert int(expected[2]) == 77
    assert not bool(expected_q[slots, 3 * COLUMNS :].any())


def test_a_shard_table_takes_proposals_from_shards_only():
    slots = torch.tensor([1], dtype=torch.int64, device="cuda")
    pools = _pools()
    logits = torch.randn(1, VOCAB, device="cuda").bfloat16()
    sharded, _ = _sampler(pools, slots, COLUMNS)
    whole, _ = _sampler(pools, slots, None)
    with pytest.raises(RuntimeError, match="propose_vocab_shard"):
        sharded.propose(logits, step=STEP)
    with pytest.raises(RuntimeError, match="gathered logits"):
        whole.propose_vocab_shard(_shard(logits, 0), step=STEP)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
