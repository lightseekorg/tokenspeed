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

"""The cached sample route/pad must recompute exactly when the batch mix moves.

``_select_sample_route`` / ``_select_top_k_top_p_pad`` aggregate only the
batch's sampling-params mix, so ``prepare_step`` caches them against the
batch-composition epoch. The risk is a missed invalidation: the assertions
here drive both bump sources — a slot flip (a new request takes a slot) and
a size change (a retraction departs without flipping anything) — plus the
capture interplay: capture variants set the route directly, and the next
real step must win over what capture left behind.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
from tokenspeed.runtime.sampling.backends.triton import (
    _SAMPLE_ROUTE_GUMBEL_GENERIC,
    _SAMPLE_ROUTE_GUMBEL_NO_FILTER,
    _SAMPLE_ROUTE_GUMBEL_TOP_K,
    _TOP_K_TOP_P_PAD,
    TritonSamplingBackend,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _backend() -> TritonSamplingBackend:
    config = SamplingBackendConfig(
        synthetic_acceptance_length=None,
        enable_speculative_sampling=False,
        sampling_stream="batch",
        logprob_order="torch",
        max_bs=8,
        max_req_pool_size=4,
        vocab_size=128,
        device=torch.device("cuda"),
    )
    return TritonSamplingBackend(config)


def test_route_tracks_membership_changes():
    backend = _backend()
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
    backend = _backend()
    backend.prepare_step([1], [1], [SamplingParams(seed=0)])
    assert backend._sample_route == _SAMPLE_ROUTE_GUMBEL_NO_FILTER

    # Capture resets the route directly (outside prepare_step).
    backend.prepare_capture(bs=1)
    assert backend._sample_route == _SAMPLE_ROUTE_GUMBEL_GENERIC

    # The next real step restores the batch's route even though membership
    # did not move: capture must not leak past the cache restore.
    backend.prepare_step([1], [1], [SamplingParams(seed=0)])
    assert backend._sample_route == _SAMPLE_ROUTE_GUMBEL_NO_FILTER
