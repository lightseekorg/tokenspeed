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

"""Real scheduler token views reach input buffers without mutating the plan.

CPU checks substitute only GPU metadata kernels and pinned allocation; CUDA
checks exercise the real fill path, including its asynchronous token uploads.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
from tokenspeed_scheduler import (
    AttnKind,
    CacheGroupConfig,
    RequestSpec,
    Scheduler,
    SchedulerConfig,
)

from tokenspeed.runtime.execution import input_buffer
from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.execution.runtime_states import RuntimeStates

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

VOCAB_SIZE = 128
MAX_BS, MAX_TOKENS, PAD_POOL = 8, 64, 5

pytestmark = pytest.mark.filterwarnings(
    "error:The given NumPy array is not writable:UserWarning"
)


@pytest.fixture(params=["cpu", "cuda"])
def buffers(request, monkeypatch):
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if device == "cpu":
        # The kernels the fill path launches are GPU-only; the CPU runs here
        # check the host token copies around them, so they stand in with
        # the same tensor contract.
        def bulk(self, *specs):
            return [torch.empty(n, dtype=dtype) for n, dtype in specs]

        def positions(extend_prefix_lens, extend_seq_lens, extend_seq_lens_sum, out):
            values = [
                torch.arange(int(p), int(p + n))
                for p, n in zip(extend_prefix_lens, extend_seq_lens)
            ]
            out.copy_(torch.cat(values) if values else torch.empty(0))
            return out, None

        def decode_positions(
            positions_ptr,
            seq_lens_out_ptr,
            req_pool_indices,
            valid_cache_lengths,
            uniform_input_length,
        ):
            starts = valid_cache_lengths[req_pool_indices]
            positions_ptr.copy_(
                (starts[:, None] + torch.arange(uniform_input_length)).flatten()
            )
            seq_lens_out_ptr.copy_(starts + uniform_input_length)

        monkeypatch.setattr(InputBuffers, "_bulk_pinned", bulk)
        monkeypatch.setattr(input_buffer, "compute_position_triton", positions)
        monkeypatch.setattr(input_buffer, "fused_decode_input_prep", decode_positions)
    ib = InputBuffers(
        max_bs=MAX_BS,
        max_num_tokens=MAX_TOKENS,
        state_write_padding_pool_index=PAD_POOL,
        device=device,
        reads_state_checkpoints=False,
    )
    runtime = RuntimeStates(
        req_pool_size=8,
        input_vocab_size=VOCAB_SIZE,
        vocab_size=VOCAB_SIZE,
        output_length=1,
        device=device,
    )
    return ib, runtime


def _scheduler(*, token_budget: int, mixed: bool) -> Scheduler:
    config = SchedulerConfig()
    config.max_batch_size = PAD_POOL - 1
    config.max_scheduled_tokens = token_budget
    config.prefix_granularity = 4
    config.num_device_pages = 128
    config.num_snapshot_pages = 1
    config.enable_mixed_prefill_decode = mixed
    config.cache_groups = [
        CacheGroupConfig(
            group_id="full_attention",
            block_granularity=4,
            total_pages=128,
            attn_kind=AttnKind.Full,
        )
    ]
    return Scheduler(config)


def _submit(scheduler: Scheduler, rid: str, tokens: list[int]) -> None:
    spec = RequestSpec()
    spec.request_id = rid
    spec.tokens = tokens
    scheduler.submit_requests([spec])


def _fill(ib, runtime, op):
    n = op.num_extends()
    runtime.reset_states(
        torch.tensor(op.request_pool_indices[:n], dtype=torch.int64, device=ib.device),
        torch.tensor(op.extend_prefix_lens, dtype=torch.int32, device=ib.device),
    )
    total = sum(op.input_lengths)
    ib.fill_input_buffers(op, runtime, total)
    return total


def test_mid_chunk_predicate_tracks_prefill_completion(buffers):
    ib, runtime = buffers
    scheduler = _scheduler(token_budget=4, mixed=False)
    _submit(scheduler, "a", list(range(12)))
    for expected in (True, True, False):
        op = scheduler.next_execution_plan().forward[0]
        _fill(ib, runtime, op)
        assert ib.all_extends_mid_chunk is expected

    op = scheduler.next_execution_plan().forward[0]
    assert op.num_extends() == 0
    _fill(ib, runtime, op)
    assert ib.all_extends_mid_chunk is False


def test_mid_chunk_predicate_excludes_mixed_batches(buffers):
    ib, runtime = buffers
    scheduler = _scheduler(token_budget=8, mixed=True)
    _submit(scheduler, "a", [10, 11, 12, 13])
    _fill(ib, runtime, scheduler.next_execution_plan().forward[0])
    _submit(scheduler, "b", list(range(32, 48)))
    op = scheduler.next_execution_plan().forward[0]
    assert op.num_extends() == 1
    assert len(op.request_ids) == 2
    _fill(ib, runtime, op)
    assert ib.all_extends_mid_chunk is False


def test_tokens_and_extend_facts_land_intact(buffers):
    ib, runtime = buffers
    scheduler = _scheduler(token_budget=32, mixed=False)
    tokens = {"a": [300, 301, 302, 303], "b": [128000, 128001, 128002]}
    for rid, ids in tokens.items():
        _submit(scheduler, rid, ids)
    op = scheduler.next_execution_plan().forward[0]
    assert set(op.request_ids) == set(tokens)
    expected_ids = [token for rid in op.request_ids for token in tokens[rid]]
    expected_shifted = [
        token for rid in op.request_ids for token in tokens[rid][1:] + [-1]
    ]
    total = _fill(ib, runtime, op)

    assert ib.input_ids_buf[:total].tolist() == expected_ids
    assert ib.shifted_prefill_ids_buf[:total].tolist() == expected_shifted
    assert op.input_ids.tolist() == expected_ids
    assert op.shifted_input_ids.tolist() == expected_shifted
    assert not op.input_ids.flags.writeable
    assert not op.shifted_input_ids.flags.writeable
    n = op.num_extends()
    lengths = [len(tokens[rid]) for rid in op.request_ids]
    assert ib.input_lengths_buf[:n].tolist() == lengths
    assert ib.req_pool_indices_buf[:n].tolist() == op.request_pool_indices
    assert ib.extend_prefix_lens_cpu[:n].tolist() == [0] * n
    assert ib.extend_seq_lens_cpu[:n].tolist() == lengths
    assert ib.extend_prompt_lens_cpu[:n].tolist() == lengths
    assert ib.input_ids_buf[total:].tolist() == [1] * (ib.max_num_tokens - total)


def test_shorter_chunk_clears_previous_token_tail(buffers):
    ib, runtime = buffers
    scheduler = _scheduler(token_budget=8, mixed=False)
    tokens = list(range(300, 312))
    _submit(scheduler, "a", tokens)
    first = scheduler.next_execution_plan().forward[0]
    assert _fill(ib, runtime, first) == 8
    assert ib.input_ids_buf[:8].tolist() == tokens[:8]

    last = scheduler.next_execution_plan().forward[0]
    total = _fill(ib, runtime, last)
    assert total == 4
    assert ib.input_ids_buf[:total].tolist() == tokens[8:]
    assert ib.shifted_prefill_ids_buf[:total].tolist() == tokens[9:] + [-1]
    assert ib.extend_prefix_lens_cpu[0].item() == 8
    assert ib.extend_prompt_lens_cpu[0].item() == len(tokens)
    assert ib.input_ids_buf[total:].tolist() == [1] * (ib.max_num_tokens - total)


def test_state_writes_take_the_batch_rows_then_the_sink_row(buffers):
    """A graph replays padded rows too; their state writes land on the sink
    row, whatever a previous, larger batch left in the buffer."""
    ib, runtime = buffers
    scheduler = _scheduler(token_budget=32, mixed=False)
    for rid in ("a", "b"):
        _submit(scheduler, rid, [300, 301, 302, 303])
    _fill(ib, runtime, scheduler.next_execution_plan().forward[0])
    op = scheduler.next_execution_plan().forward[0]
    assert op.num_extends() == 0 and len(op.request_ids) == 2
    ib.state_write_req_pool_indices_buf.fill_(-1)
    _fill(ib, runtime, op)

    rows = ib.state_write_req_pool_indices_buf.tolist()
    assert rows[:2] == list(op.request_pool_indices)
    assert rows[2:] == [PAD_POOL] * (MAX_BS - 2)
