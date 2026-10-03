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

"""CPU tests for TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN: the simulated widths
and where ModelExecutor applies them after verify."""

from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.execution.accept_simulation import (  # noqa: E402
    parse_simulated_accept_length,
    simulated_accept_lengths,
)
from tokenspeed.runtime.execution.model_executor import ModelExecutor  # noqa: E402
from tokenspeed.runtime.execution.output_layout import (  # noqa: E402
    ForwardOutputLayout,
)
from tokenspeed.runtime.layers.logits_processor import (  # noqa: E402
    LogitsProcessorOutput,
)
from tokenspeed.runtime.sampling.sampling_batch_info import (  # noqa: E402
    SamplingBatchInfo,
)

VERIFY_WIDTH = 4


def _scaled(length: str) -> int:
    return parse_simulated_accept_length(
        length, spec_algorithm="MTP", verify_width=VERIFY_WIDTH
    )


@pytest.mark.parametrize("length", ["1", "2", "2.7", "3.25", "4"])
def test_widths_alternate_around_the_average(length):
    average = float(length)
    cache = torch.tensor([0, 17, 1027, 262143], dtype=torch.int32)
    steps = []
    for _ in range(401):
        widths = simulated_accept_lengths(cache, _scaled(length))
        steps.append(widths)
        cache += widths.to(torch.int32)
    widths = torch.stack(steps, dim=1)

    assert widths[:, 0].min() >= 1
    assert widths[:, 0].max() <= math.ceil(average)
    steady = widths[:, 1:]
    assert set(steady.unique().tolist()) <= {math.floor(average), math.ceil(average)}
    for total in steady.sum(dim=1).tolist():
        assert abs(total - average * steady.shape[1]) <= 1


class _NoAcceptSampler:
    """Verify accepts no draft, writing width 1 into one shared buffer."""

    def __init__(self):
        self.lengths = torch.zeros(8, dtype=torch.int32)

    def sample(self, logits_output, sampling_info):
        rows = logits_output.next_token_logits.shape[0]
        return torch.zeros(rows, dtype=torch.int32), torch.ones(rows, dtype=torch.int32)

    def verify(self, logits_output, sampling_info, candidates):
        rows = candidates.shape[0]
        self.lengths[:rows].fill_(1)
        return torch.zeros(candidates.numel(), dtype=torch.int32), self.lengths[:rows]


def _executor(pool_indices: list[int], cache_lengths: torch.Tensor) -> ModelExecutor:
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.sampling_backend = _NoAcceptSampler()
    executor._simulated_accept_length = _scaled("2.7")
    req_pool_indices = torch.zeros(8, dtype=torch.int64)
    req_pool_indices[: len(pool_indices)] = torch.tensor(pool_indices)
    executor.input_buffers = SimpleNamespace(
        req_pool_indices_buf=req_pool_indices,
        force_single_token_verify_buf=torch.zeros(8, dtype=torch.bool),
    )
    executor.runtime_states = SimpleNamespace(valid_cache_lengths=cache_lengths)
    return executor


def test_decode_verify_keeps_simulated_widths_in_the_sampler_buffer():
    cache_lengths = torch.arange(100, 108, dtype=torch.int32)
    pool = [5, 2, 7]
    executor = _executor(pool, cache_lengths)
    ctx = SimpleNamespace(
        bs=3,
        num_extends=0,
        decode_input_ids=None,
        output_layout=ForwardOutputLayout(0, 0, 3, VERIFY_WIDTH),
    )
    candidates = torch.zeros(3, VERIFY_WIDTH, dtype=torch.int32)

    _, lengths = executor._run_sampling(object(), object(), ctx, candidates)
    expected = simulated_accept_lengths(cache_lengths[pool], _scaled("2.7")).tolist()
    assert lengths.tolist() == expected
    assert lengths.data_ptr() == executor.sampling_backend.lengths.data_ptr()

    # Rows the scheduler forces to one token keep one token.
    executor.input_buffers.force_single_token_verify_buf[1] = True
    ctx.decode_input_ids = [-1, 9, -1]
    _, lengths = executor._run_sampling(object(), object(), ctx, candidates)
    assert lengths.tolist() == [expected[0], 1, expected[2]]


def test_mixed_round_simulates_only_its_decode_rows():
    cache_lengths = torch.arange(200, 208, dtype=torch.int32)
    pool = [3, 4, 6, 1]
    executor = _executor(pool, cache_lengths)
    ctx = SimpleNamespace(
        bs=4,
        num_extends=2,
        decode_input_ids=None,
        output_layout=ForwardOutputLayout(2, 2, 2, VERIFY_WIDTH),
    )
    logits = LogitsProcessorOutput(next_token_logits=torch.zeros(2 + 2 * 4, 16))
    info = SamplingBatchInfo(req_pool_indices=torch.tensor(pool), device="cpu")
    candidates = torch.zeros(2, VERIFY_WIDTH, dtype=torch.int32)

    _, lengths = executor._run_sampling(logits, info, ctx, candidates)
    decode = simulated_accept_lengths(cache_lengths[pool[2:]], _scaled("2.7"))
    assert lengths.tolist() == [1, 1, *decode.tolist()]
