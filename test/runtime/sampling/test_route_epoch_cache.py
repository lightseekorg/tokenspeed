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

"""The cached sample route/pad follows the batch that actually samples.

``_select_sample_route`` / ``_select_top_k_top_p_pad`` aggregate only the
batch's sampling-params mix, so ``prepare_step`` caches them against the
batch-composition epoch. The risk is a missed invalidation: the assertions
here cover slot reuse, size changes and switches between equally sized
resident batches, including a prepared restore that keeps its slot owner.
Capture variants set the route directly; the next real step must win over
what capture left behind.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

from tokenspeed.runtime.layers.logits_processor_output import LogitsProcessorOutput
from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
from tokenspeed.runtime.sampling.backends.triton import (
    _SAMPLE_ROUTE_GUMBEL_GENERIC,
    _SAMPLE_ROUTE_GUMBEL_NO_FILTER,
    _SAMPLE_ROUTE_GUMBEL_TOP_K,
    _TOP_K_TOP_P_PAD,
    TritonSamplingBackend,
)
from tokenspeed.runtime.sampling.backends.triton_full import TritonFullSamplingBackend
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
from tokenspeed.runtime.sampling.sampling_params import SamplingParams

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _backend(backend_cls: type[TritonSamplingBackend]) -> TritonSamplingBackend:
    config = SamplingBackendConfig(
        synthetic_acceptance_length=None,
        enable_speculative_sampling=False,
        sampling_stream="batch",
        logprob_order="torch",
        max_bs=8,
        max_req_pool_size=4,
        vocab_size=512,
        device=torch.device("cuda"),
    )
    return backend_cls(config)


def test_route_tracks_membership_changes():
    backend = _backend(TritonSamplingBackend)
    no_filter = SamplingParams(seed=0)  # top_k disabled, top_p 1.0
    top_k_10 = SamplingParams(top_k=10, seed=0)

    # Steady batch: nothing filtering -> no-filter route, whole-vocab pad.
    backend.prepare_step([1], [1], [no_filter])
    assert backend._sample_route == _SAMPLE_ROUTE_GUMBEL_NO_FILTER
    assert backend._top_k_top_p_pad == _TOP_K_TOP_P_PAD

    # Same membership again: the cache still serves the same route.
    backend.prepare_step([1], [1], [no_filter])
    assert backend._sample_route == _SAMPLE_ROUTE_GUMBEL_NO_FILTER
    assert backend._route_cache_epoch == backend._batch_composition_epoch

    # A finite top_k joins (a flip): the mix is no longer uniform, and the
    # pad tightens to the 64 bucket.
    backend.prepare_step([1, 2], [1, 2], [no_filter, top_k_10])
    assert backend._sample_route == _SAMPLE_ROUTE_GUMBEL_GENERIC
    assert backend._top_k_top_p_pad == 64

    # The top_k request retracts: no slot flips, only the size changed, and
    # the route must not keep what the departed request forced.
    backend.prepare_step([1], [1], [no_filter])
    assert backend._sample_route == _SAMPLE_ROUTE_GUMBEL_NO_FILTER
    assert backend._top_k_top_p_pad == _TOP_K_TOP_P_PAD

    # An all-finite batch of small top_ks routes to top_k.
    backend.prepare_step([2, 3], [2, 3], [top_k_10, SamplingParams(top_k=20, seed=0)])
    assert backend._sample_route == _SAMPLE_ROUTE_GUMBEL_TOP_K
    assert backend._top_k_top_p_pad == 64


def test_prepare_step_wins_over_the_capture_route():
    backend = _backend(TritonSamplingBackend)
    backend.prepare_step([1], [1], [SamplingParams(seed=0)])
    assert backend._sample_route == _SAMPLE_ROUTE_GUMBEL_NO_FILTER

    # Capture resets the route directly (outside prepare_step).
    backend.prepare_capture(bs=1)
    assert backend._sample_route == _SAMPLE_ROUTE_GUMBEL_GENERIC

    # The next real step restores the batch's route even though membership
    # did not move: capture must not leak past the cache restore.
    backend.prepare_step([1], [1], [SamplingParams(seed=0)])
    assert backend._sample_route == _SAMPLE_ROUTE_GUMBEL_NO_FILTER


def _draw_positions(backend, slots, *, start, steps):
    info = SamplingBatchInfo(
        req_pool_indices=torch.tensor(slots, dtype=torch.int64, device="cuda"),
        valid_cache_lengths=torch.zeros(5, dtype=torch.int32, device="cuda"),
        device="cuda",
    )
    logits = torch.full((len(slots), 512), -100.0, device="cuda")
    logits[:, :256] = torch.arange(256, 0, -1, device="cuda") * 0.001
    outputs = []
    for position in range(start, start + steps):
        info.valid_cache_lengths[info.req_pool_indices] = position
        tokens, _ = backend.sample(
            LogitsProcessorOutput(next_token_logits=logits.clone()), info
        )
        outputs.append(tokens.clone())
    return torch.stack(outputs)


@pytest.mark.parametrize(
    "backend_cls", [TritonSamplingBackend, TritonFullSamplingBackend]
)
@pytest.mark.parametrize("destination", [1, 2])
@pytest.mark.parametrize("delayed", [False, True])
@pytest.mark.parametrize("top_k", [1, 128])
def test_prepared_restore_samples_like_unretracted(
    backend_cls, destination, delayed, top_k
):
    backend = _backend(backend_cls)
    requested = SamplingParams(
        temperature=1.0,
        top_k=top_k,
        seed=17,
        frequency_penalty=0.25 if backend_cls is TritonFullSamplingBackend else 0.0,
    )
    # Exercise both route changes (no-filter -> top-k) and pad expansion
    # (small top-k -> top-k 128) when the requested batch returns.
    other = SamplingParams(top_k=-1 if top_k == 1 else 1, seed=99)
    for params in (requested, other):
        params.normalize(None)
    backend.prepare_step([41], [1], [requested])
    _draw_positions(backend, [1], start=0, steps=4)
    stream = torch.cuda.current_stream()
    image = torch.empty(backend.slot_state_bytes(), dtype=torch.uint8, pin_memory=True)
    backend.export_slot_state(1, image, stream, admission_id=41)
    stream.synchronize()
    # The unretracted continuation includes the full backend's accumulated
    # penalty counts, so clearing the restored owner is not a valid fix.
    expected = _draw_positions(backend, [1], start=4, steps=16)

    backend.prepare_step([42], [destination], [other])
    if delayed:
        backend.prepare_step([43], [3], [other])
    backend.import_slot_state(destination, image, stream, admission_id=41)
    if delayed:
        # A different resident batch can consume an import-time epoch bump
        # before the restored request is scheduled again.
        backend.prepare_step([43], [3], [other])
    backend.prepare_step([41], [destination], [requested])
    actual = _draw_positions(backend, [destination], start=4, steps=16)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "backend_cls", [TritonSamplingBackend, TritonFullSamplingBackend]
)
def test_equal_size_resident_batches_sample_their_own_distribution(backend_cls):
    backend = _backend(backend_cls)
    filtered = SamplingParams(top_k=1, seed=17)
    unfiltered = SamplingParams(seed=99)
    for params in (filtered, unfiltered):
        params.normalize(None)
    batches = [([11, 12], [1, 2], filtered), ([13, 14], [3, 4], unfiltered)]
    expected = []
    for admissions, slots, params in batches:
        backend.prepare_step(admissions, slots, [params, params])
        expected.append(_draw_positions(backend, slots, start=0, steps=16))
    assert not torch.equal(*expected)
    # All four slots remain prepared. Switching batches changes neither the
    # batch size nor any slot owner, but it changes the required route.
    for (admissions, slots, params), reference in zip(batches, expected, strict=True):
        backend.prepare_step(admissions, slots, [params, params])
        assert torch.equal(
            _draw_positions(backend, slots, start=0, steps=16), reference
        )
