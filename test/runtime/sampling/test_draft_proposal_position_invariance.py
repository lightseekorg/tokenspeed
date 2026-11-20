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

"""Committed positions key the real proposal sampler and its recorded q."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="FlashInfer chain verification requires NVIDIA CUDA.",
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="FlashInfer chain verification requires NVIDIA CUDA",
)


def test_recorded_draft_chain_follows_the_committed_frontier(monkeypatch):
    from tokenspeed.runtime.execution.drafter.speculative_sampling import (
        DraftProposalSampler,
    )
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates
    from tokenspeed.runtime.layers.logits_processor_output import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
    from tokenspeed.runtime.sampling.backends.flashinfer import (
        FlashInferSamplingBackend,
    )
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "numerics", "rl-bitwise")
    monkeypatch.setitem(
        global_server_args_dict, "speculative_rejection_sample_method", "standard"
    )
    pool_size, vocab_size, width, max_bs = 8, 32768, 3, 4
    frontier = 1041
    device = "cuda"
    config = SamplingBackendConfig(
        synthetic_acceptance_length=None,
        enable_speculative_sampling=True,
        sampling_stream="per-request",
        logprob_order="blockwise",
        enable_output_logprobs=True,
        max_bs=max_bs,
        max_draft_tokens_per_req=width,
        max_req_pool_size=pool_size,
        vocab_size=vocab_size,
        device=device,
    )
    backend = FlashInferSamplingBackend(config)
    states = RuntimeStates(pool_size, vocab_size, vocab_size, width, device)
    states.init_draft_probs(spec_num_tokens=width, reject_threshold=2.0)
    buffers = InputBuffers(max_bs, max_bs * width, pool_size, device)
    sampler = DraftProposalSampler(
        pools=backend.speculative_sampling_pools(),
        runtime_states=states,
        input_buffers=buffers,
        spec_num_tokens=width,
        sampling_stream=config.sampling_stream,
        vocab_map=None,
        device=device,
    )
    params = SamplingParams(temperature=1.0, top_k=-1, top_p=1.0, seed=42)
    params.normalize(None)

    def draft_and_verify(slots, *, old_frontier, num_extends, committed, active_row):
        bs = len(slots)
        slot = slots[active_row]
        live_slots = [s for s in slots if s]
        backend.prepare_step(live_slots, live_slots, [params] * len(live_slots), width)
        buffers.req_pool_indices_buf[:bs] = torch.tensor(slots, device=device)
        buffers.state_write_req_pool_indices_buf[:bs] = torch.tensor(
            [s or pool_size for s in slots], device=device
        )
        states.valid_cache_lengths[slot] = old_frontier
        buffers.seq_lens_buf[active_row] = (
            committed if active_row < num_extends else old_frontier + width
        )
        # Decode accepts two tokens in the previous verify. For extend, use the
        # entire chunk's end regardless of how much of it came from cache.
        lengths = torch.full((bs,), 2, dtype=torch.int32, device=device)
        sampler.prepare_round(lengths, num_extends=num_extends)
        logits = torch.zeros(bs, vocab_size, device=device)
        candidates = torch.zeros(1, width, dtype=torch.int32, device=device)
        for step in range(width - 1):
            candidates[0, step + 1] = sampler.propose(logits, step=step)[active_row]
        q = states.draft_probs[slot].clone()

        # p=q accepts the first candidate. A disjoint point mass rejects the
        # second candidate and forces a uniquely determined replacement.
        target_logits = torch.zeros(width, vocab_size, device=device)
        replacement = (int(candidates[0, 2]) + 1) % vocab_size
        target_logits[1].fill_(-torch.inf)
        target_logits[1, replacement] = 0
        states.valid_cache_lengths[slot] = committed
        output = LogitsProcessorOutput(next_token_logits=target_logits)
        info = SamplingBatchInfo(
            req_pool_indices=buffers.req_pool_indices_buf[active_row : active_row + 1],
            valid_cache_lengths=states.valid_cache_lengths,
            draft_probs=states.draft_probs,
        )
        tokens, lengths = backend.verify(output, info, candidates, tree=None)
        assert lengths.tolist() == [2]
        assert tokens[:2].tolist() == [int(candidates[0, 1]), replacement]
        return (
            candidates,
            q,
            tokens[:2].clone(),
            output.next_token_logprobs[:2].view(torch.int32),
        )

    whole = draft_and_verify(
        [1], old_frontier=0, num_extends=1, committed=frontier, active_row=0
    )
    chunked = draft_and_verify(
        [3], old_frontier=701, num_extends=1, committed=frontier, active_row=0
    )
    mixed = draft_and_verify(
        [2, 4, 0, 0],
        old_frontier=frontier - 2,
        num_extends=1,
        committed=frontier,
        active_row=1,
    )
    for result in (chunked, mixed):
        for expected, actual in zip(whole, result, strict=True):
            assert torch.equal(expected, actual)
    next_round = draft_and_verify(
        [3], old_frontier=701, num_extends=1, committed=frontier + 1, active_row=0
    )
    assert not torch.equal(whole[0], next_round[0])
