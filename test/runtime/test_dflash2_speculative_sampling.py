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

# CPU-only; scheduled in runtime-1gpu because it imports the full runtime.
register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.execution.drafter.dflash import DFlash  # noqa: E402
from tokenspeed.runtime.execution.drafter.dflash2 import DFlash2  # noqa: E402
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


def _triton_backend(*, enable_speculative_sampling: bool) -> TritonSamplingBackend:
    config = SamplingBackendConfig(
        synthetic_acceptance_length=None,
        enable_speculative_sampling=enable_speculative_sampling,
        sampling_stream="batch",
        logprob_order="torch",
        max_bs=MAX_BS,
        max_draft_tokens_per_req=N,
        max_req_pool_size=POOL,
        vocab_size=VOCAB,
        device="cpu",
    )
    # The backend sizes a scratch buffer by the SM count; there is no GPU here.
    with patch.object(
        torch.cuda,
        "get_device_properties",
        return_value=SimpleNamespace(multi_processor_count=4),
    ):
        return TritonSamplingBackend(config)


def _publish(states: RuntimeStates, slots: list[int], has_proposal: list[int]):
    """Record one walk's chains for ``slots`` in the layout the walk kernel
    writes; returns ``(q, q_ids, lane)``."""
    bs = len(slots)
    q = torch.softmax(torch.randn(bs, STEPS, K), dim=-1)
    q_ids = torch.randint(VOCAB, (bs, STEPS, K), dtype=torch.int32)
    lane = torch.randint(K, (bs, STEPS), dtype=torch.int32)
    pos = torch.nn.functional.pad(lane, (1, 0))
    chain = torch.arange(-1, STEPS, dtype=torch.int32)
    proposes = torch.tensor(has_proposal, dtype=torch.bool)[:, None]
    slot = torch.where(proposes, chain[None, :], -1).to(torch.int32)
    states.future_draft_q.publish_chain(
        torch.tensor(slots, dtype=torch.int64), q, q_ids, pos, slot
    )
    return q, q_ids, lane


def test_a_published_chain_follows_the_verify_layout():
    torch.manual_seed(0)
    states = _states()
    q, q_ids, lane = _publish(states, [2, 4], [1, 0])
    draft_q = states.future_draft_q

    # Node i (draft token i) reads slot i - 1 at its drawn lane; the anchor
    # has no q.
    assert draft_q.slot[2].tolist() == [-1, 0, 1, 2]
    assert draft_q.pos[2].tolist() == [0, *lane[0].tolist()]
    torch.testing.assert_close(draft_q.q[2], q[0], rtol=0, atol=0)
    assert torch.equal(draft_q.q_ids[2], q_ids[0])
    # A chain the walk did not draw from q (greedy, always-fallback or
    # non-finite rows) carries no proposal on any node.
    assert draft_q.slot[4].tolist() == [-1] * N
    # Untouched slots keep the never-drafted state.
    assert draft_q.slot[3].tolist() == [-1] * N


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
