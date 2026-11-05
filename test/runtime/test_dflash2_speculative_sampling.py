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

"""DFLASH + DFlash2 under ``--enable-speculative-sampling``: the sparse draft q
the walk records in ``RuntimeStates.future_draft_q``, its reset on admission,
and the drafter's binding to the triton verifier."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

# Mostly CPU; the walk test needs a GPU.
register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed_kernel.ops.sampling.triton import dflash2_sample_path  # noqa: E402
from tokenspeed_kernel.ops.sampling.triton.common import (  # noqa: E402
    TOP_K_DISABLED,
)

from tokenspeed.runtime.execution.drafter.dflash import DFlash  # noqa: E402
from tokenspeed.runtime.execution.drafter.dflash2 import DFlash2  # noqa: E402
from tokenspeed.runtime.execution.input_buffer import InputBuffers  # noqa: E402
from tokenspeed.runtime.execution.runtime_states import RuntimeStates  # noqa: E402
from tokenspeed.runtime.sampling.backends.base import (  # noqa: E402
    SamplingBackendConfig,
    SpeculativeSamplingPools,
)
from tokenspeed.runtime.sampling.backends.triton import (  # noqa: E402
    TritonSamplingBackend,
)
from tokenspeed.runtime.sampling.tree_verify import DraftTreeQ  # noqa: E402

N = 4  # verify width: anchor + 3 drafts
STEPS = N - 1
K = 3  # selector candidates per step
POOL = 6
VOCAB = 32
MAX_BS = 4


def _states() -> RuntimeStates:
    states = RuntimeStates(
        req_pool_size=POOL, vocab_size=VOCAB, output_length=N, device="cpu"
    )
    states.init_draft_tree_q(num_nodes=N, num_slots=STEPS, num_candidates=K)
    return states


def _triton_backend(
    *, enable_speculative_sampling: bool, device: str = "cpu"
) -> TritonSamplingBackend:
    config = SamplingBackendConfig(
        synthetic_acceptance_length=None,
        enable_speculative_sampling=enable_speculative_sampling,
        sampling_stream="batch",
        logprob_order="torch",
        max_bs=MAX_BS,
        max_draft_tokens_per_req=N,
        max_req_pool_size=POOL,
        vocab_size=VOCAB,
        device=device,
    )
    if device != "cpu":
        return TritonSamplingBackend(config)
    # The backend sizes a scratch buffer by the SM count; there is no GPU here.
    with patch.object(
        torch.cuda,
        "get_device_properties",
        return_value=SimpleNamespace(multi_processor_count=4),
    ):
        return TritonSamplingBackend(config)


def _publish(states: RuntimeStates, slots: list[int], has_proposal: list[int]):
    """Record chains for ``slots`` as the walk kernel writes them; returns
    ``(q, q_ids, lane)``."""
    bs = len(slots)
    q = torch.softmax(torch.randn(bs, STEPS, K), dim=-1)
    q_ids = torch.randint(VOCAB, (bs, STEPS, K), dtype=torch.int32)
    lane = torch.randint(K, (bs, STEPS), dtype=torch.int32)
    pos = torch.nn.functional.pad(lane, (1, 0))
    chain = torch.arange(-1, STEPS, dtype=torch.int32)
    proposes = torch.tensor(has_proposal, dtype=torch.bool)[:, None]
    slot = torch.where(proposes, chain[None, :], -1).to(torch.int32)
    rows = torch.tensor(slots, dtype=torch.int64)
    draft_q = states.future_draft_q
    draft_q.q[rows], draft_q.q_ids[rows] = q, q_ids
    draft_q.pos[rows], draft_q.slot[rows] = pos, slot
    return q, q_ids, lane


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_the_sampled_walk_records_each_chain_at_its_state_write_slot():
    """With the flag on, the walk reads each request's sampling parameters and
    cache length at its pool slot and records its chain at its state-write
    slot (the reserved slot for a graph padding row); no other slot changes."""
    torch.manual_seed(0)
    device = "cuda"
    states = RuntimeStates(
        req_pool_size=POOL, vocab_size=VOCAB, output_length=N, device=device
    )
    drafter = _dflash2(states)
    backend = _triton_backend(enable_speculative_sampling=True, device=device)
    drafter.bind_sampling_backend(backend)
    drafter.draft_vocab_size = VOCAB
    drafter.input_buffers = InputBuffers(
        max_bs=MAX_BS,
        max_num_tokens=MAX_BS * N,
        state_write_padding_pool_index=POOL,
        device=device,
    )
    # Request 0 samples (top-k off); request 1 is greedy; row 2 is padding.
    pool_slots, write_slots = [2, 5, 0], [2, 5, POOL]
    pools = drafter._walk_pools
    pools.temperature[2], pools.top_k[2] = 0.7, TOP_K_DISABLED
    pools.temperature[5], pools.top_k[5] = 1.0, 1
    pools.seed[2], pools.seed[5] = 123, 456
    states.valid_cache_lengths[2], states.valid_cache_lengths[5] = 40, 90
    bs = len(pool_slots)
    drafter.input_buffers.req_pool_indices_buf[:bs] = torch.tensor(pool_slots)
    drafter.input_buffers.state_write_req_pool_indices_buf[:bs] = torch.tensor(
        write_slots
    )
    candidate_ids = torch.randint(VOCAB, (bs, STEPS, K), device=device)
    scores = torch.randn(bs, STEPS, K, K, device=device)
    anchors = torch.randint(VOCAB, (bs,), device=device)
    draft_q = states.future_draft_q
    before = DraftTreeQ(
        q=draft_q.q.clone(),
        q_ids=draft_q.q_ids.clone(),
        slot=draft_q.slot.clone(),
        pos=draft_q.pos.clone(),
    )

    out = torch.empty(bs, N, dtype=torch.int32, device=device)
    drafter._walk(candidate_ids, scores, anchors, out)

    # The same walk on explicit arguments, into tables of its own.
    expected = DraftTreeQ.zeros(bs, STEPS, K, N, device)
    expected_out = torch.empty_like(out)
    dflash2_sample_path(
        candidate_ids,
        scores,
        anchors,
        torch.tensor(pool_slots, device=device),
        pools.temperature,
        pools.top_k,
        pools.top_p,
        pools.seed,
        states.valid_cache_lengths,
        N,
        VOCAB,
        expected_out,
        torch.arange(bs, device=device),
        expected.q,
        expected.q_ids,
        expected.pos,
        expected.slot,
    )
    assert torch.equal(out, expected_out)
    for name in ("q", "q_ids", "pos", "slot"):
        table, want, old = (
            getattr(draft_q, name),
            getattr(expected, name),
            getattr(before, name),
        )
        assert torch.equal(table[write_slots], want), name
        untouched = [row for row in range(POOL + 1) if row not in write_slots]
        assert torch.equal(table[untouched], old[untouched]), name
    # The sampled request proposes; the greedy one does not.
    assert draft_q.slot[2].tolist() == [-1, *range(STEPS)]
    assert draft_q.slot[5].tolist() == [-1] * N


def test_a_reset_slot_verifies_by_the_target_only_rule():
    """After reset_states a slot reads exactly like one no drafter ever wrote:
    every node at slot -1, which verify checks against the target's own draw
    instead of a previous occupant's q."""
    torch.manual_seed(1)
    states = _states()
    _publish(states, [1, 3], [1, 1])
    never_drafted = DraftTreeQ.zeros(POOL + 1, STEPS, K, N, "cpu")
    kept = states.future_draft_q.slot[3].clone()

    states.reset_states(
        torch.tensor([1], dtype=torch.int64), torch.tensor([5], dtype=torch.int32)
    )

    assert torch.equal(states.future_draft_q.slot[1], never_drafted.slot[1])
    # The reset is per slot: a live neighbour keeps its proposal.
    assert torch.equal(states.future_draft_q.slot[3], kept)
    assert states.future_draft_q.slot[3].tolist() == [-1, 0, 1, 2]
    # The sparse path never needs the dense per-vocab rows.
    assert states.draft_probs is None


def test_a_landed_or_readmitted_slot_takes_a_new_chain_after_the_reset():
    torch.manual_seed(2)
    states = _states()
    _publish(states, [1], [1])
    states.reset_states(
        torch.tensor([1], dtype=torch.int64), torch.tensor([0], dtype=torch.int32)
    )
    _, _, lane = _publish(states, [1], [1])
    assert states.future_draft_q.slot[1].tolist() == [-1, 0, 1, 2]
    assert states.future_draft_q.pos[1].tolist() == [0, *lane[0].tolist()]


def test_base_dflash_refuses_the_flag_at_bind():
    # Base DFLASH (no candidate selector) proposes the argmax of every block
    # row; only the loaded checkpoint tells it apart from DFlash2.
    drafter = DFlash.__new__(DFlash)
    with pytest.raises(ValueError, match="needs a DFlash2 draft"):
        drafter.bind_sampling_backend(_triton_backend(enable_speculative_sampling=True))
    drafter.bind_sampling_backend(_triton_backend(enable_speculative_sampling=False))


def _dflash2(states: RuntimeStates) -> DFlash2:
    """A DFlash2 with the fields bind_sampling_backend reads, as its
    constructor leaves them (no draft model here)."""
    drafter = DFlash2.__new__(DFlash2)
    drafter.runtime_states = states
    drafter.spec_num_tokens = N
    drafter.selector_top_k = K
    drafter._walk_pools = SpeculativeSamplingPools(
        temperature=torch.ones(1),
        top_k=torch.ones(1, dtype=torch.int32),
        top_p=torch.ones(1),
        seed=torch.zeros(1, dtype=torch.int64),
    )
    drafter._draft_q = None
    return drafter


def test_dflash2_binds_a_sparse_q_and_the_verifiers_pools():
    states = RuntimeStates(
        req_pool_size=POOL, vocab_size=VOCAB, output_length=N, device="cpu"
    )
    drafter = _dflash2(states)
    backend = _triton_backend(enable_speculative_sampling=True)

    drafter.bind_sampling_backend(backend)

    draft_q = states.future_draft_q
    assert drafter._draft_q is draft_q
    assert draft_q.q.shape == (POOL + 1, STEPS, K)
    assert draft_q.slot.shape == (POOL + 1, N)
    assert (draft_q.slot == -1).all()
    # No dense per-vocab rows for this drafter.
    assert states.draft_probs is None
    # The walk samples with the request's own parameters: the verifier's
    # pool rows, scattered on admission.
    pools = backend.speculative_sampling_pools()
    assert drafter._walk_pools.temperature is pools.temperature
    assert drafter._walk_pools.top_k is pools.top_k
    assert drafter._walk_pools.top_p is pools.top_p
    assert drafter._walk_pools.seed is pools.seed


def test_dflash2_without_the_flag_keeps_the_greedy_walk():
    states = RuntimeStates(
        req_pool_size=POOL, vocab_size=VOCAB, output_length=N, device="cpu"
    )
    drafter = _dflash2(states)

    drafter.bind_sampling_backend(_triton_backend(enable_speculative_sampling=False))

    assert states.future_draft_q is None
    assert drafter._draft_q is None
    assert drafter._walk_pools.top_k.tolist() == [1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
