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

"""The KV pages a finish publishes hold the tokens their hashes name.

The real scheduler, output processor, input buffers, stop conditions and
accepted-frontier update run every round, in the model executor's order; the
model is a fixed greedy token stream and the drafter proposes its
continuation. Each forward "writes KV" by recording the input token at the
slot of the position the input buffers computed, so where a round writes is
decided by the runtime's device-side cache length, exactly as in serving.
"""

from __future__ import annotations

import os
import sys

_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))

import pytest
import torch
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

ts = pytest.importorskip("tokenspeed_scheduler")

from tokenspeed_kernel.ops.metadata import advance_accepted_frontier  # noqa: E402

from tokenspeed.runtime.engine.generation_output_processor import (  # noqa: E402
    OutputProcesser,
    RequestState,
)
from tokenspeed.runtime.engine.scheduler_utils import (  # noqa: E402
    advance_scheduler,
    make_config,
)
from tokenspeed.runtime.execution.input_buffer import InputBuffers  # noqa: E402
from tokenspeed.runtime.execution.runtime_states import RuntimeStates  # noqa: E402
from tokenspeed.runtime.execution.types import ModelExecutionResult  # noqa: E402
from tokenspeed.runtime.sampling.sampling_params import SamplingParams  # noqa: E402

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the input buffers run CUDA kernels"
)

_PAGE = 4
_POOL = 8
_PAGES = 64
_VOCAB = 4096
_STOP = 9
_PROMPT = list(range(10, 16))


class _Sender:
    """Stands in for the detokenizer socket."""

    def __init__(self):
        self.items = []

    def send_pyobj(self, obj):
        self.items.append(obj)


class _Tokenizer:
    eos_token_id = None
    additional_stop_token_ids = None

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


class _Metrics:
    enabled = False


class _Engine:
    """One fused engine: scheduler, runtime state and a greedy token stream.

    ``truth[i]`` is the model's token at sequence index ``i``: the prompt,
    then a continuation whose token at ``stop_at`` is the stop token.
    """

    def __init__(self, *, width: int, overlap: bool, stop_at: int):
        self.width = width
        self.truth = _PROMPT + [
            _STOP if i == stop_at else 100 + i for i in range(len(_PROMPT), 64)
        ]
        self.scheduler = ts.Scheduler(
            make_config(
                num_device_pages=_PAGES + 1,
                max_scheduled_tokens=256,
                max_batch_size=4,
                prefix_granularity=_PAGE,
                num_host_pages=0,
                disable_l2_cache=True,
                enable_l3_storage=False,
                role="fused",
                num_snapshot_pages=1,
                max_retracted_requests=0,
                l3_prefetch_min_pages=0,
                decode_input_tokens=width,
                overlap_schedule_depth=int(overlap),
                cache_groups=[
                    ts.CacheGroupConfig(
                        group_id="history",
                        block_granularity=_PAGE,
                        total_pages=_PAGES + 1,
                        attn_kind=ts.AttnKind.Full,
                    )
                ],
            )
        )
        self.states = RuntimeStates(
            req_pool_size=_POOL, vocab_size=_VOCAB, output_length=width
        )
        self.buffers = InputBuffers(
            max_bs=_POOL, max_num_tokens=256, state_write_padding_pool_index=_POOL
        )
        self.processor = OutputProcesser(
            _Sender(),
            attn_tp_rank=0,
            metrics=_Metrics(),
            spec_algorithm="EAGLE" if width > 1 else None,
            spec_num_tokens=width if width > 1 else None,
        )
        # The token each KV slot holds; -1 where nothing was written.
        self.kv = [-1] * ((_PAGES + 1) * _PAGE)
        # Gathered at dispatch, as the event loop does.
        self.stop_criteria = {}

    def submit(self, request_id: str, tokens: list[int]) -> None:
        self.processor.rid_to_state[request_id] = RequestState(
            prompt_input_ids=list(tokens),
            sampling_params=SamplingParams(
                max_new_tokens=40, stop=[], stop_token_ids={_STOP}
            ),
            stream=False,
            tokenizer=_Tokenizer(),
            computes_prompt_logprobs=True,
        )
        self.stop_criteria[request_id] = self.processor.rid_to_state[
            request_id
        ].stop_criteria
        spec = ts.RequestSpec()
        spec.request_id = request_id
        spec.tokens = list(tokens)
        spec.max_new_tokens = 40
        self.scheduler.submit_requests([spec])

    def plan(self):
        plan = self.scheduler.next_execution_plan()
        (batch,) = [op for op in plan.forward if op.request_ids]
        for src, dst in plan.pages_to_copy_arrays()["history"].tolist():
            for offset in range(_PAGE):
                self.kv[dst * _PAGE + offset] = self.kv[src * _PAGE + offset]
        return batch

    def run(self, batch, accepted: int) -> tuple[list[int], int]:
        """Execute one single-request forward the way the model executor
        does; return its sampled rows and the tokens it keeps."""
        num_extends = batch.num_extends()
        total = sum(batch.input_lengths)
        rows = torch.tensor(batch.request_pool_indices, dtype=torch.int64).cuda()
        if num_extends:
            self.states.reset_states(
                rows, torch.tensor(batch.extend_prefix_lens, dtype=torch.int32).cuda()
            )
        self.states.bind_stop_criteria(
            batch.request_ids,
            batch.request_pool_indices,
            [self.stop_criteria[rid] for rid in batch.request_ids],
        )
        self.buffers.fill_input_buffers(batch, self.states, total, ngram_inputs=None)
        input_ids = self.buffers.input_ids_buf[:total].tolist()
        positions = self.buffers.positions_buf[:total].tolist()
        pages = batch.block_tables_arrays()["history"][0]
        for token, position in zip(input_ids, positions, strict=True):
            self.kv[int(pages[position // _PAGE]) * _PAGE + position % _PAGE] = token
        if num_extends:
            sampled = [self.truth[positions[-1] + 1]]
        else:
            sampled = [self.truth[position + 1] for position in positions]
        accept_lengths = torch.tensor(
            [1 if num_extends else accepted], dtype=torch.int32
        ).cuda()
        self.states.apply_stop_conditions(
            self.buffers.state_write_req_pool_indices_buf[:1],
            torch.tensor(sampled, dtype=torch.int32).cuda(),
            accept_lengths,
            self.buffers.input_lengths_buf[:1],
            self.buffers.extend_prompt_lens_buf[:num_extends],
            num_extends=num_extends,
            num_prefill_outputs=num_extends,
            decode_width=self.width,
            padding_index=self.buffers.state_write_padding_pool_index,
        )
        kept = int(accept_lengths[0])
        # The drafter (or, without one, the sampled token) lays out the next
        # round's inputs: the last accepted token and its continuation.
        bonus = positions[-1] + 1 if num_extends else positions[max(kept, 1) - 1] + 1
        self.states.future_input_map[rows] = torch.tensor(
            [self.truth[bonus : bonus + self.width]], dtype=torch.int32
        ).cuda()
        if self.width > 1:
            self.states.mark_spec_candidates_drafted(rows)
        advance_accepted_frontier(
            self.buffers.state_write_req_pool_indices_buf[:1],
            self.buffers.input_lengths_buf[:1],
            accept_lengths,
            self.states.valid_cache_lengths,
            num_extends,
            self.buffers.state_write_padding_pool_index,
        )
        torch.cuda.synchronize()
        return sampled, kept

    def land(self, batch, sampled: list[int], accepted: int) -> list[int]:
        """Report a landed forward through the output processor; returns the
        tokens the scheduler was told it produced."""
        result = ModelExecutionResult(
            output_tokens=torch.tensor(sampled, dtype=torch.int32),
            output_lengths=torch.tensor([accepted], dtype=torch.int32),
        )
        events = self.processor.post_process_forward_op(
            batch, result, is_prefill_instance=False
        )
        advance_scheduler(self.scheduler, events)
        return [
            token
            for event in events
            if type(event).__name__ == "ExtendResult"
            for token in event.tokens
        ]


@requires_cuda
@pytest.mark.parametrize(
    "width,overlap",
    [(3, False), (3, True), (1, True)],
    ids=["spec", "spec_overlap", "overlap"],
)
def test_a_prefix_hit_after_a_finish_reads_the_kv_of_its_own_tokens(width, overlap):
    """The round that samples the stop token as its first output accepts its
    whole window; the device keeps it through the stop token, and the forward
    the overlap schedule dispatched behind it keeps nothing. The finish
    publishes the frontier before the stop token, and every slot below a hit
    there holds the KV of the token the prompt has at that position."""
    accepts = [2, 3] if width > 1 else [1, 1]
    stop_at = len(_PROMPT) + sum(accepts) + 1
    engine = _Engine(width=width, overlap=overlap, stop_at=stop_at)
    engine.submit("source", _PROMPT)

    batch = engine.plan()
    reported = _PROMPT + engine.land(batch, *engine.run(batch, 1))
    for accepted in accepts:
        batch = engine.plan()
        sampled, kept = engine.run(batch, accepted)
        assert kept == accepted
        reported += engine.land(batch, sampled, kept)

    batch = engine.plan()
    sampled, kept = engine.run(batch, width)
    assert sampled[0] == _STOP and kept == 1
    if overlap:
        # Dispatched before the finishing round lands.
        next_batch = engine.plan()
        next_sampled, next_kept = engine.run(next_batch, min(2, width))
        assert next_kept == 0
    reported += engine.land(batch, sampled, kept)
    assert "source" not in engine.processor.rid_to_state
    if overlap:
        reported += engine.land(next_batch, next_sampled, next_kept)

    # The frontier stops before the stop token: the token every next turn's
    # prompt carries, whether or not its chat template keeps the stop token.
    frontier = len(reported) - 1
    assert reported[-1] == _STOP and frontier == stop_at
    probe = reported + [999]
    engine.submit("probe", probe)
    batch = engine.plan()
    assert list(batch.extend_prefix_lens) == [frontier]
    pages = batch.block_tables_arrays()["history"][0]
    held = [
        engine.kv[int(pages[position // _PAGE]) * _PAGE + position % _PAGE]
        for position in range(frontier)
    ]
    mismatches = [
        (position, probe[position], token)
        for position, token in enumerate(held)
        if token != probe[position]
    ]
    assert not mismatches, (
        f"(position, token the hit's key names, token whose KV the slot holds): "
        f"{mismatches}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
