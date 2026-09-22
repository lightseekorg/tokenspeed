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

"""Real scheduler allocations drive GPU KDA as working state blocks are recycled.

The small three-group pool and KDA kernels come from the existing numerical
harness. Every working block id, prefix hit and sparse hole comes from the
real C++ scheduler through the runtime contract bridge. This is not a model
or MLA numerical test: history blocks participate in scheduling and matching,
while the GPU computations exercise convolution and recurrent state.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

# The CI runner executes each registered test file as a standalone script.
_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))
import pytest
import torch
from ci_system.ci_register import register_cuda_ci

ts = pytest.importorskip("tokenspeed_scheduler")

from test.runtime.conftest import (  # noqa: E402
    KIMI_STATE_GROUPS,
    requires_cuda,
)
from test.runtime.test_kimi_k3_kda import (  # noqa: E402
    _LOWER_BOUND,
    _assert_close,
    _KDAHarness,
    _naive_kda_scan,
    _stub_contract,
    _StubContractPool,
    _to_slab_layout,
    requires_fla,
)

from tokenspeed.runtime.engine.scheduler_utils import (  # noqa: E402
    make_extend_result_event,
    pool_to_cache_groups,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402

register_cuda_ci(
    est_time=180,
    suite="runtime-1gpu",
    nightly=False,
    disabled=None,
    disabled_on_runners=None,
    disabled_on_runners_reason=None,
)

_P = 128
_USABLE_BLOCKS = 32
_PROMPT_TOKENS = 16 * _P - 2
_LAYERS = (0, 1, 2)
_COMPONENTS = ("conv_state", "recurrent_state")


@dataclass
class _Step:
    begin: int
    end: int
    outputs: dict[int, torch.Tensor]
    states: dict[tuple[int, str], torch.Tensor]
    tables: dict[str, list[list[int]]]


class _NumericalReference:
    """FP32 convolution and recurrence, without cache pages or backend metadata."""

    def __init__(
        self, kernel: _KDAHarness, streams: dict[int, dict[str, torch.Tensor]]
    ):
        self.kernel = kernel
        self.streams = streams
        self.computed = 0
        self.convolved = {}
        self.recurrent = {}
        for layer in _LAYERS:
            mixed = streams[layer]["mixed"]
            params = kernel.params[layer]
            total = len(mixed)
            conv = torch.zeros_like(mixed, dtype=torch.float32)
            for column in range(kernel.WIDTH):
                lag = kernel.WIDTH - 1 - column
                conv[lag:] += (
                    mixed[: total - lag].float()
                    * params["conv_weights"][:, column].float()
                )
            conv += params["bias"].float()
            self.convolved[layer] = torch.nn.functional.silu(conv).to(mixed.dtype)
            self.recurrent[layer] = torch.zeros(
                kernel.H, kernel.D, kernel.D, dtype=torch.float32, device=mixed.device
            )

    def step(self, begin: int, end: int) -> _Step:
        assert begin == self.computed and begin < end
        outputs, states = {}, {}
        kernel = self.kernel
        for layer in _LAYERS:
            stream = self.streams[layer]
            params = kernel.params[layer]
            q, k, v = self.convolved[layer][begin:end].split(kernel.key_dim, dim=-1)
            output, recurrent = _naive_kda_scan(
                q.reshape(-1, kernel.H, kernel.D),
                k.reshape(-1, kernel.H, kernel.D),
                v.reshape(-1, kernel.H, kernel.D),
                stream["g_raw"][begin:end].reshape(-1, kernel.H, kernel.D),
                stream["beta_raw"][begin:end],
                params["A_log"],
                params["dt_bias"],
                _LOWER_BOUND,
                self.recurrent[layer],
            )
            self.recurrent[layer] = recurrent
            outputs[layer] = output.flatten(0, 1)
            states[layer, "recurrent_state"] = _to_slab_layout(recurrent).clone()
            states[layer, "conv_state"] = (
                stream["mixed"][end - kernel.WIDTH + 1 : end].transpose(0, 1).clone()
            )
        self.computed = end
        return _Step(begin, end, outputs, states, {})


class _ScheduledKDA:
    def __init__(self, *, seed: int):
        self.contract = _stub_contract(
            prefix_granularity=_P, usable_pages=_USABLE_BLOCKS
        )
        self.pool = _StubContractPool(
            self.contract,
            "cuda",
            conv_dim=3 * _KDAHarness.H * _KDAHarness.D,
            width=_KDAHarness.WIDTH,
            num_heads=_KDAHarness.H,
            head_dim=_KDAHarness.D,
        )
        self.kernel = _KDAHarness(
            self.pool, self.contract, layer_ids=_LAYERS, device="cuda", seed=seed
        )
        config = ts.SchedulerConfig()
        config.prefix_granularity = _P
        config.num_device_pages = _USABLE_BLOCKS + 1
        config.num_host_pages = 0
        config.max_scheduled_tokens = _P
        config.max_batch_size = 1
        config.decode_input_tokens = 1
        config.overlap_schedule_depth = 0
        config.enable_mixed_prefill_decode = False
        config.disable_l2_cache = True
        config.cache_groups = pool_to_cache_groups(self.pool)
        self.scheduler = ts.Scheduler(config)
        self.tokens: dict[str, list[int]] = {}
        self.computed: dict[str, int] = {}
        self.previous_tables: dict[str, list[int]] = {}
        self.released = {group: set() for group in KIMI_STATE_GROUPS}
        self.reused_outputs = {group: set() for group in KIMI_STATE_GROUPS}

    def submit(self, request_id: str, tokens: list[int], prompt_tokens: int):
        self.tokens[request_id] = tokens
        self.computed[request_id] = 0
        request = ts.RequestSpec()
        request.request_id = request_id
        request.tokens = tokens[:prompt_tokens]
        request.max_new_tokens = 16
        self.scheduler.submit_requests([request])

    def finish(self, request_id: str):
        event = ts.ForwardEvent.Finish()
        event.request_id = request_id
        self.scheduler.advance(ts.ExecutionEvent().add_event(event))

    def step(self, request_id: str, streams: dict[int, dict[str, torch.Tensor]]):
        plan = self.scheduler.next_execution_plan()
        batches = [batch for batch in plan.forward if batch.request_ids]
        assert len(batches) == 1
        batch = batches[0]
        assert list(batch.request_ids) == [request_id]
        assert batch.num_extends() in (0, 1)
        tables = {
            group: [list(row) for row in rows]
            for group, rows in batch.block_tables.items()
        }
        is_prefill = batch.num_extends() == 1
        begin = (
            int(batch.extend_prefix_lens[0])
            if is_prefill
            else self.computed[request_id]
        )
        end = begin + int(batch.input_lengths[0])
        assert 0 <= begin < end < len(self.tokens[request_id])
        delivered = self.kernel._delivered(tables)
        request_indices = torch.tensor(
            list(batch.request_pool_indices), dtype=torch.int32, device="cuda"
        )
        seq_lens = torch.tensor([end], dtype=torch.int32, device="cuda")
        if is_prefill:
            lengths_cpu = torch.tensor([end - begin], dtype=torch.int32)
            prefixes_cpu = torch.tensor([begin], dtype=torch.int32)
            self.kernel.backend.init_forward_metadata(
                bs=1,
                num_extends=1,
                req_pool_indices=request_indices,
                seq_lens=seq_lens,
                forward_mode=ForwardMode.EXTEND,
                block_tables=delivered,
                extend_seq_lens=lengths_cpu.to("cuda"),
                extend_seq_lens_cpu=lengths_cpu,
                extend_prefix_lens=prefixes_cpu.to("cuda"),
                extend_prefix_lens_cpu=prefixes_cpu,
                extend_replay_lens_cpu=torch.as_tensor(
                    batch.extend_replay_lens, dtype=torch.int32
                ),
                extend_prompt_lens_cpu=torch.as_tensor(
                    batch.prefill_lengths, dtype=torch.int32
                ),
                extend_with_prefix=begin > 0,
            )
        else:
            self.kernel.backend.refresh_decode_metadata(
                1,
                1,
                request_indices,
                seq_lens,
                forward_mode=ForwardMode.DECODE,
                block_tables=delivered,
                for_graph_replay=False,
            )

        outputs = {}
        states = {}
        for layer, group in zip(_LAYERS, KIMI_STATE_GROUPS, strict=True):
            stream = streams[layer]
            inputs = (
                layer,
                stream["mixed"][begin:end],
                stream["g_raw"][begin:end],
                stream["beta_raw"][begin:end],
            )
            output = (
                self.kernel.extend(*inputs, bs=1)
                if is_prefill
                else self.kernel.decode(*inputs, bs=1)
            )
            outputs[layer] = output.clone()
            state_block = tables[group][0][(end - 1) // _P]
            for component in _COMPONENTS:
                states[layer, component] = self.pool.get_component(layer, component)[
                    state_block
                ].clone()
            # Convolution stores raw projections, not the post-convolution
            # values. This independent check also detects a wrong read/write id.
            expected_conv = stream["mixed"][
                end - self.kernel.WIDTH + 1 : end
            ].transpose(0, 1)
            torch.testing.assert_close(
                states[layer, "conv_state"], expected_conv, atol=0, rtol=0
            )

            if request_id == "source" and is_prefill:
                row = tables[group][0]
                previous = self.previous_tables.get(group, [])
                for slot, old_block in enumerate(previous):
                    if old_block > 0 and row[slot] == 0:
                        self.released[group].add(old_block)
                if state_block in self.released[group]:
                    self.reused_outputs[group].add(state_block)
                self.previous_tables[group] = row

        # Feedback follows actual GPU completion, as the execution-result
        # event does in serving. No synthetic completion before state writes.
        torch.cuda.synchronize()
        final_prefill = is_prefill and end == int(batch.prefill_lengths[0])
        tokens = (
            [self.tokens[request_id][end]] if not is_prefill or final_prefill else []
        )
        event = make_extend_result_event(
            request_id,
            tokens,
            None,
        )
        self.scheduler.advance(ts.ExecutionEvent().add_event(event))
        self.computed[request_id] = end
        return _Step(begin, end, outputs, states, tables)

    def snapshot(self, tables: dict[str, list[list[int]]], boundary: int):
        return {
            (layer, component): self.pool.get_component(layer, component)[
                tables[group][0][boundary // _P - 1]
            ].clone()
            for layer, group in zip(_LAYERS, KIMI_STATE_GROUPS, strict=True)
            for component in _COMPONENTS
        }


def _assert_steps_match(actual: _Step, reference: _Step):
    assert (actual.begin, actual.end) == (reference.begin, reference.end)
    for layer in _LAYERS:
        _assert_close(
            actual.outputs[layer], reference.outputs[layer], f"layer {layer} output"
        )
        torch.testing.assert_close(
            actual.states[layer, "conv_state"],
            reference.states[layer, "conv_state"],
            atol=0,
            rtol=0,
        )
        _assert_close(
            actual.states[layer, "recurrent_state"],
            reference.states[layer, "recurrent_state"],
            f"layer {layer} recurrent state",
        )


@requires_cuda
@requires_fla
def test_scheduler_recycles_unpublished_state_without_changing_kda_or_resume():
    """Recycled scheduler blocks cannot change continuation or cached-prefix state."""
    active = _ScheduledKDA(seed=91)
    source_tokens = list(range(_PROMPT_TOKENS + 5))
    source_streams = {
        layer: active.kernel.token_stream(len(source_tokens)) for layer in _LAYERS
    }
    reference = _NumericalReference(active.kernel, source_streams)
    active.submit("source", source_tokens, _PROMPT_TOKENS)

    steps = 0
    while active.computed["source"] < _PROMPT_TOKENS:
        actual = active.step("source", source_streams)
        _assert_steps_match(actual, reference.step(actual.begin, actual.end))
        steps += 1
        assert active.scheduler.empty_lcm_blocks() > 0
        if steps == 3:
            # Packing is one block per parent. Ordinary prefill checkpoints
            # are never published, so the first checkpoint returns to the
            # pool when its last working reference leaves the table.
            materialized = sum(
                block > 0
                for rows in actual.tables.values()
                for row in rows
                for block in row
            )
            assert active.scheduler.empty_lcm_blocks() == (
                _USABLE_BLOCKS - materialized
            )
            assert all(len(active.released[group]) == 1 for group in KIMI_STATE_GROUPS)

    # The released parents cycle through the allocator and become real GPU
    # output destinations again. Unpublished working state returns directly
    # to the pool, keeping the entire prefill below capacity.
    assert all(active.reused_outputs[group] for group in KIMI_STATE_GROUPS)

    boundary = 15 * _P
    snapshot_tables = actual.tables
    snapshots = active.snapshot(snapshot_tables, boundary)
    for _ in range(4):
        actual = active.step("source", source_streams)
        _assert_steps_match(actual, reference.step(actual.begin, actual.end))
    for key, state in active.snapshot(snapshot_tables, boundary).items():
        assert torch.equal(
            state, snapshots[key]
        ), "decode modified the prefill checkpoint"
    active.finish("source")

    # Decode crossed token 16P, but reuse still starts at the prefill checkpoint.
    # Recompute the remaining prompt and decode history with a different suffix.
    shared_tokens = 16 * _P
    resume_tokens = source_tokens[:shared_tokens] + list(range(30_000, 30_009))
    resume_streams = {}
    for layer in _LAYERS:
        suffix = active.kernel.token_stream(9)
        resume_streams[layer] = {
            name: torch.cat(
                [source_streams[layer][name][:shared_tokens], values], dim=0
            )
            for name, values in suffix.items()
        }
    # Recompute the prefix independently from zero; never seed the oracle
    # from the very cache snapshot whose correctness it must establish.
    reference = _NumericalReference(active.kernel, resume_streams)
    reference.step(0, boundary)
    active.submit("resume", resume_tokens, shared_tokens + 7)
    actual = active.step("resume", resume_streams)
    assert actual.begin == boundary
    _assert_steps_match(actual, reference.step(actual.begin, actual.end))
    while active.computed["resume"] < shared_tokens + 7:
        actual = active.step("resume", resume_streams)
        _assert_steps_match(actual, reference.step(actual.begin, actual.end))
    actual = active.step("resume", resume_streams)
    _assert_steps_match(actual, reference.step(actual.begin, actual.end))
    for key, state in active.snapshot(snapshot_tables, boundary).items():
        assert torch.equal(
            state, snapshots[key]
        ), "prefix resume modified the shared snapshot"
    active.finish("resume")
    for layer in _LAYERS:
        for component in _COMPONENTS:
            assert (
                torch.count_nonzero(active.pool.get_component(layer, component)[0]) == 0
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
