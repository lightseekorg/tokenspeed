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

"""CPU tests for TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN: the simulated widths,
the tokens that match them, and where ModelExecutor applies both after
verify."""

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
    simulated_output_tokens,
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
        length, spec_algorithm="MTP", verify_width=VERIFY_WIDTH, draft_tree=False
    )


def test_draft_trees_are_refused():
    with pytest.raises(ValueError, match="chain drafts only"):
        parse_simulated_accept_length(
            "2", spec_algorithm="EAGLE3", verify_width=VERIFY_WIDTH, draft_tree=True
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


def test_simulated_tokens_are_the_drafts_each_width_accepts():
    candidates = torch.tensor([[10, 11, 12, 13]] * 4, dtype=torch.int32)
    # Verify accepted draft 11 and wrote bonus 50; 99 is a stale entry.
    tokens = torch.tensor([[11, 50, 99, 99]] * 4, dtype=torch.int32)
    kept = torch.tensor([1, 2, 3, 4])

    out = simulated_output_tokens(tokens, candidates, torch.full((4,), 2), kept)
    emitted = [out[row, :width].tolist() for row, width in enumerate(kept.tolist())]
    assert emitted == [[11], [11, 50], [11, 12, 13], [11, 12, 13, 13]]


class _NoAcceptSampler:
    """Verify accepts no draft: width 1 and each row's bonus token go into
    shared buffers, whose entries past the bonus stay stale."""

    BONUS = 7
    STALE = 99

    def __init__(self):
        self.tokens = torch.full((8 * VERIFY_WIDTH,), self.STALE, dtype=torch.int32)
        self.lengths = torch.zeros(8, dtype=torch.int32)

    def sample(self, logits_output, sampling_info):
        rows = logits_output.next_token_logits.shape[0]
        return torch.zeros(rows, dtype=torch.int32), torch.ones(rows, dtype=torch.int32)

    def verify(self, logits_output, sampling_info, candidates, *, tree):
        rows = candidates.shape[0]
        tokens = self.tokens[: candidates.numel()]
        tokens.view(rows, -1)[:, 0] = self.BONUS
        self.lengths[:rows].fill_(1)
        return tokens, self.lengths[:rows]


def _candidates(rows: int) -> torch.Tensor:
    return torch.arange(10, 10 + rows * VERIFY_WIDTH, dtype=torch.int32).view(rows, -1)


def _emitted(tokens: torch.Tensor, widths: list[int]) -> list[list[int]]:
    rows = tokens.view(len(widths), -1)
    return [rows[row, :width].tolist() for row, width in enumerate(widths)]


def _accepted_drafts(candidates: torch.Tensor, widths: list[int]) -> list[list[int]]:
    """What rows whose verify accepted nothing emit when they keep ``widths``."""
    return [
        (
            [_NoAcceptSampler.BONUS]
            if width == 1
            else candidates[row, 1 : width + 1].tolist()
        )
        for row, width in enumerate(widths)
    ]


def _executor(pool_indices: list[int], cache_lengths: torch.Tensor) -> ModelExecutor:
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.sampling_backend = _NoAcceptSampler()
    executor.tree_spec = None
    executor._simulated_accept_length = _scaled("2.7")
    req_pool_indices = torch.zeros(8, dtype=torch.int64)
    req_pool_indices[: len(pool_indices)] = torch.tensor(pool_indices)
    executor.input_buffers = SimpleNamespace(
        req_pool_indices_buf=req_pool_indices,
        force_single_token_verify_buf=torch.zeros(8, dtype=torch.bool),
    )
    executor.runtime_states = SimpleNamespace(valid_cache_lengths=cache_lengths)
    return executor


def test_decode_verify_keeps_simulated_widths_and_drafts_in_the_sampler_buffers():
    cache_lengths = torch.arange(100, 108, dtype=torch.int32)
    pool = [5, 2, 7]
    executor = _executor(pool, cache_lengths)
    sampler = executor.sampling_backend
    ctx = SimpleNamespace(
        bs=3,
        num_extends=0,
        decode_input_ids=None,
        output_layout=ForwardOutputLayout(0, 0, 3, VERIFY_WIDTH),
    )
    candidates = _candidates(3)

    tokens, lengths = executor._run_sampling(object(), object(), ctx, candidates)
    expected = simulated_accept_lengths(cache_lengths[pool], _scaled("2.7")).tolist()
    assert expected == [3, 3, 1]
    assert lengths.tolist() == expected
    assert _emitted(tokens, expected) == _accepted_drafts(candidates, expected)
    assert lengths.data_ptr() == sampler.lengths.data_ptr()
    assert tokens.data_ptr() == sampler.tokens.data_ptr()

    # Rows the scheduler forces to one token keep verify's token.
    executor.input_buffers.force_single_token_verify_buf[1] = True
    ctx.decode_input_ids = [-1, 9, -1]
    tokens, lengths = executor._run_sampling(object(), object(), ctx, candidates)
    forced = [expected[0], 1, expected[2]]
    assert lengths.tolist() == forced
    assert _emitted(tokens, forced) == _accepted_drafts(candidates, forced)


def test_mixed_round_simulates_only_its_decode_rows():
    cache_lengths = torch.arange(200, 208, dtype=torch.int32)
    pool = [3, 4, 2, 0]
    executor = _executor(pool, cache_lengths)
    ctx = SimpleNamespace(
        bs=4,
        num_extends=2,
        decode_input_ids=None,
        output_layout=ForwardOutputLayout(2, 2, 2, VERIFY_WIDTH),
    )
    logits = LogitsProcessorOutput(next_token_logits=torch.zeros(2 + 2 * 4, 16))
    info = SamplingBatchInfo(req_pool_indices=torch.tensor(pool), device="cpu")
    candidates = _candidates(2)

    tokens, lengths = executor._run_sampling(logits, info, ctx, candidates)
    decode = simulated_accept_lengths(cache_lengths[pool[2:]], _scaled("2.7")).tolist()
    assert decode == [3, 2]
    assert lengths.tolist() == [1, 1, *decode]
    assert _emitted(tokens[2:], decode) == _accepted_drafts(candidates, decode)
