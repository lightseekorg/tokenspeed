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

"""Admission identity across slot reuse, stop binding and retraction images."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.execution.runtime_states import RuntimeStates  # noqa: E402
from tokenspeed.runtime.execution.types import StopCriteria  # noqa: E402
from tokenspeed.runtime.layers.logits_processor_output import (  # noqa: E402
    LogitsProcessorOutput,
)
from tokenspeed.runtime.sampling.backends.base import (  # noqa: E402
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.backends.triton import (  # noqa: E402
    TritonSamplingBackend,
)
from tokenspeed.runtime.sampling.sampling_batch_info import (  # noqa: E402
    SamplingBatchInfo,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams  # noqa: E402

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    ),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cause", ["old_limit", "old_stop", "old_large_limit"])
def test_new_admission_obeys_its_stop_criteria(device, cause):
    states = RuntimeStates(
        req_pool_size=2,
        input_vocab_size=512,
        vocab_size=512,
        output_length=4,
        device=device,
    )
    slots = torch.tensor([1], dtype=torch.int64, device=device)
    old = StopCriteria(
        (7,) if cause == "old_stop" else (), 6 if cause == "old_limit" else 100
    )
    current = StopCriteria((), 12 if cause == "old_large_limit" else 30)
    states.bind_stop_criteria([41], [1], [old])
    # The same external ID has a fresh admission, even in the same pool row.
    states.bind_stop_criteria([42], [1], [current])
    states.reset_states(slots, torch.tensor([0], dtype=torch.int32, device=device))
    accepted = torch.tensor([1], dtype=torch.int32, device=device)
    states.apply_stop_conditions(
        slots,
        torch.tensor([7], dtype=torch.int32, device=device),
        accepted,
        torch.tensor([10], dtype=torch.int32, device=device),
        torch.tensor([10], dtype=torch.int32, device=device),
        num_extends=1,
        num_prefill_outputs=1,
        decode_width=4,
        padding_index=0,
    )
    assert not bool(states.stopped[1])
    states.valid_cache_lengths[1] = 10
    width = 4 if cause == "old_large_limit" else 1
    accepted.fill_(width)
    states.apply_stop_conditions(
        slots,
        torch.tensor([8, 9, 10, 11], dtype=torch.int32, device=device),
        accepted,
        torch.tensor([width], dtype=torch.int32, device=device),
        torch.empty(0, dtype=torch.int32, device=device),
        num_extends=0,
        num_prefill_outputs=0,
        decode_width=4,
        padding_index=0,
    )
    assert accepted.tolist() == [1]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("destination", [1, 2])
@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("stopped", [False, True])
def test_restore_rebinds_stops_without_resetting_running_state(
    device, destination, prepared, stopped
):
    states = RuntimeStates(
        req_pool_size=2,
        input_vocab_size=512,
        vocab_size=512,
        output_length=4,
        device=device,
    )
    current = StopCriteria((9,), 30)
    states.bind_stop_criteria([41], [1], [StopCriteria((7,), 6)])
    if prepared:
        states.bind_stop_criteria([42], [1], [current])
    # A remote landing can be imaged before its first stop binding.
    states.valid_cache_lengths[1] = 10
    states.future_input_map[1].fill_(23)
    states.spec_candidates_ready[1] = True
    states.stopped[1] = stopped
    image = torch.empty(
        states.slot_state_bytes(), dtype=torch.uint8, pin_memory=device == "cuda"
    )
    stream = torch.cuda.current_stream() if device == "cuda" else None
    states.export_slot_state(1, image, stream, admission_id=42)
    if stream is not None:
        stream.synchronize()
    # Leave a matching owner at the destination: the import must invalidate it.
    states.bind_stop_criteria([43], [destination], [StopCriteria((), 100)])
    states.bind_stop_criteria([42], [destination], [current])
    states.valid_cache_lengths[destination] = 0
    states.future_input_map[destination].zero_()
    states.spec_candidates_ready[destination] = False
    states.stopped[destination] = not stopped
    states.import_slot_state(destination, image, stream, admission_id=42)
    # Another retraction before a forward must not claim stale stop criteria.
    states.export_slot_state(destination, image, stream, admission_id=42)
    if stream is not None:
        stream.synchronize()
    states.import_slot_state(destination, image, stream, admission_id=42)
    states.bind_stop_criteria([42], [destination], [current])
    assert int(states.token_limits[destination]) == 30
    assert states.stop_token_ids[destination, :2].tolist() == [9, -1]
    assert int(states.valid_cache_lengths[destination]) == 10
    assert states.future_input_map[destination].tolist() == [23] * 4
    assert bool(states.spec_candidates_ready[destination])
    assert bool(states.stopped[destination]) is stopped


def _gpu_sampler():
    return TritonSamplingBackend(
        SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=False,
            sampling_stream="per-request",
            logprob_order="torch",
            max_bs=1,
            max_draft_tokens_per_req=4,
            max_req_pool_size=2,
            vocab_size=512,
            device="cuda",
            enable_tp_sync=False,
        )
    )


def _draw_positions(backend):
    info = SamplingBatchInfo(
        req_pool_indices=torch.tensor([1], dtype=torch.int64, device="cuda"),
        valid_cache_lengths=torch.zeros(3, dtype=torch.int32, device="cuda"),
        device="cuda",
    )
    logits = torch.full((1, 512), -100.0, device="cuda")
    logits[0, :8] = torch.arange(8, 0, -1, device="cuda") * 0.01
    outputs = []
    for position in range(16):
        info.valid_cache_lengths[1] = position
        tokens, _ = backend.sample(
            LogitsProcessorOutput(next_token_logits=logits.clone()), info
        )
        outputs.append(int(tokens[0]))
    return outputs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_new_admission_samples_its_greedy_distribution():
    backend = _gpu_sampler()
    old = SamplingParams(temperature=0.75, top_k=8, top_p=0.9, seed=17)
    current = SamplingParams(temperature=0, seed=123)
    for params in (old, current):
        params.resolve_seed("reused")
        params.normalize(None)
    backend.prepare_step([41], [1], [old])
    backend.prepare_step([42], [1], [current])
    assert _draw_positions(backend) == [0] * 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_admission_identity_does_not_change_an_explicit_sampling_seed():
    backend = _gpu_sampler()
    params = SamplingParams(temperature=0.75, top_k=8, top_p=0.9, seed=17)
    params.resolve_seed("reused")
    params.normalize(None)
    backend.prepare_step([41], [1], [params])
    first = _draw_positions(backend)
    backend.prepare_step([42], [1], [params])
    assert _draw_positions(backend) == first
