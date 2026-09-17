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

"""Scheduler-owned speculative KDA state versus independent tokenwise decode.

The scheduler supplies all tested cache locations. Only the numerical oracle
uses a fixed private working block: it runs each accepted token independently,
without verification, snapshots, publication or cache eviction.
"""

from __future__ import annotations

import os
import sys

# The CI runner executes each registered test file as a standalone script.
_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))
from test.runtime.conftest import KIMI_STATE_GROUPS, requires_cuda
from test.runtime.test_kimi_k3_kda import (
    _assert_close,
    _backend,
    _KDAHarness,
    _stub_contract,
    _StubContractPool,
    requires_fla,
)

import pytest
import torch
from ci_system.ci_register import register_cuda_ci

ts = pytest.importorskip("tokenspeed_scheduler")

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

_P = 4
_WIDTH = 3
_USABLE = 32
_H = 4
_D = 128
_FA = 128
_LAYERS = (0, 1, 2)
_COMPONENTS = ("conv_state", "recurrent_state")


class _KDA:
    def __init__(self, *, width: int, replay: bool):
        self.contract = _stub_contract(prefix_granularity=_P, usable_pages=_USABLE)
        self.pool = _StubContractPool(
            self.contract,
            "cuda",
            conv_dim=3 * _H * _D,
            width=4,
            num_heads=_H,
            head_dim=_D,
        )
        self.backend = _backend("cuda", contract_pool=self.pool, spec_tokens=width)
        self.backend.init_cuda_graph_state(max_bs=8)
        if width > 1:
            if replay and not self.backend._replay_active:
                pytest.skip("KDA replay commit kernel unavailable")
            if not replay:
                self.backend._replay_active = False
                self.backend._verify_scratch = None
            self.backend.preallocate_verify_workspace(8, width)
        generator = torch.Generator(device="cpu").manual_seed(107)

        def random_tensor(shape, dtype, scale):
            return torch.randn(shape, generator=generator).to("cuda", dtype) * scale

        self.params = {
            layer: {
                "conv_weights": random_tensor((3 * _H * _D, 4), torch.bfloat16, 0.1),
                "f_b_weight": random_tensor((_H * _D, _FA), torch.bfloat16, 0.05),
                "A_log": random_tensor((_H,), torch.float32, 0.1),
                "dt_bias": random_tensor((_H * _D,), torch.float32, 0.1),
            }
            for layer in _LAYERS
        }

    def metadata(
        self,
        tables,
        *,
        begin: int,
        end: int,
        prefill: bool,
        request_slot: int,
        prompt_tokens: int,
    ):
        # Use the same contract-to-runtime table bridge as the numerical KDA
        # harness, with the complete unmodified scheduler tables supplied here.
        delivered = _KDAHarness._delivered(self, tables)
        request_indices = torch.tensor([request_slot], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([end], dtype=torch.int32, device="cuda")
        if prefill:
            prefixes = torch.tensor([begin], dtype=torch.int32)
            lengths = torch.tensor([end - begin], dtype=torch.int32)
            self.backend.init_forward_metadata(
                bs=1,
                num_extends=1,
                req_pool_indices=request_indices,
                seq_lens=seq_lens,
                forward_mode=ForwardMode.EXTEND,
                block_tables=delivered,
                extend_seq_lens=lengths.to("cuda"),
                extend_seq_lens_cpu=lengths,
                extend_prefix_lens=prefixes.to("cuda"),
                extend_prefix_lens_cpu=prefixes,
                extend_replay_lens_cpu=torch.zeros_like(prefixes),
                extend_prompt_lens_cpu=torch.tensor([prompt_tokens], dtype=torch.int32),
                extend_with_prefix=begin > 0,
            )
        else:
            self.backend.refresh_decode_metadata(
                1,
                1,
                request_indices,
                seq_lens,
                forward_mode=ForwardMode.DECODE,
                block_tables=delivered,
                for_graph_replay=False,
            )

    @property
    def device(self):
        return "cuda"

    def forward(self, inputs, *, begin: int, end: int, prefill: bool):
        outputs = {}
        for layer in _LAYERS:
            kwargs = dict(
                layer=None,
                token_to_kv_pool=self.pool,
                bs=1,
                mixed_qkv=inputs["mixed_qkv"][begin:end].clone(),
                f_a_out=inputs["f_a_out"][begin:end],
                beta_raw=inputs["beta_raw"][begin:end],
                g_raw=None,
                bias=None,
                activation="silu",
                key_dim=_H * _D,
                value_dim=_H * _D,
                attention_tp_size=1,
                head_k_dim=_D,
                head_v_dim=_D,
                lower_bound=-5.0,
                layer_id=layer,
                seq_len=end - begin,
                a=None,
                b=None,
                **self.params[layer],
            )
            if prefill:
                output = self.backend.forward_extend(
                    None,
                    None,
                    None,
                    forward_mode=ForwardMode.EXTEND,
                    save_kv_cache=True,
                    **kwargs,
                )
            else:
                output = self.backend.forward_decode(None, None, None, **kwargs)
            outputs[layer] = output.reshape(end - begin, _H, _D).clone()
        return outputs

    def states(self, tables, *, end: int):
        return {
            (layer, component): self.pool.get_component(layer, component)[
                tables[group][0][(end - 1) // _P]
            ].clone()
            for layer, group in zip(_LAYERS, KIMI_STATE_GROUPS, strict=True)
            for component in _COMPONENTS
        }


def _batch(scheduler, request_id: str):
    plan = scheduler.next_execution_plan()
    batches = [batch for batch in plan.forward if batch.request_ids]
    assert len(batches) == 1
    batch = batches[0]
    assert list(batch.request_ids) == [request_id]
    tables = {
        group: [list(row) for row in rows] for group, rows in batch.block_tables.items()
    }
    return batch, tables


def _feedback(scheduler, *, request_id: str, tokens: list[int], decode: bool):
    # Accepted-state commit and completion precede this feedback in every caller.
    torch.cuda.synchronize()
    result = make_extend_result_event(request_id, tokens, None)
    event = ts.ExecutionEvent().add_event(result)
    if decode:
        reserve = ts.ForwardEvent.UpdateReserveNumTokens()
        reserve.request_id = request_id
        reserve.reserve_num_tokens_in_next_schedule_event = len(tokens)
        event.add_event(reserve)
    scheduler.advance(event)


def _assert_states(actual, expected, label: str):
    for layer in _LAYERS:
        torch.testing.assert_close(
            actual[layer, "conv_state"], expected[layer, "conv_state"], atol=0, rtol=0
        )
        _assert_close(
            actual[layer, "recurrent_state"],
            expected[layer, "recurrent_state"],
            f"{label}: layer {layer} recurrent state",
        )


@requires_cuda
@requires_fla
@pytest.mark.parametrize("replay", [False, True], ids=["scratch", "replay"])
def test_speculative_decode_recycles_working_state_and_preserves_prefill_checkpoint(
    replay,
):
    actual = _KDA(width=_WIDTH, replay=replay)
    reference = _KDA(width=1, replay=False)
    config = ts.SchedulerConfig()
    config.prefix_granularity = _P
    config.num_device_pages = _USABLE + 1
    config.num_host_pages = 0
    config.max_scheduled_tokens = 8
    config.max_batch_size = 1
    config.decode_input_tokens = _WIDTH
    config.overlap_schedule_depth = 0
    config.enable_mixed_prefill_decode = False
    config.disable_l2_cache = True
    config.cache_groups = pool_to_cache_groups(actual.pool)
    scheduler = ts.Scheduler(config)

    generator = torch.Generator(device="cpu").manual_seed(113)
    inputs = {
        name: torch.randn((48, columns), generator=generator).to("cuda", torch.bfloat16)
        for name, columns in (
            ("mixed_qkv", 3 * _H * _D),
            ("f_a_out", _FA),
            ("beta_raw", _H),
        )
    }
    # The oracle deliberately never allocates or publishes snapshots. Every
    # accepted token updates the same working block through q_len=1 decode.
    reference_tables = {
        spec.group_id: [[1] * 12] for spec in reference.contract.group_specs
    }
    reference_outputs = {layer: [] for layer in _LAYERS}
    reference_states = {}
    for position in range(41):
        reference.metadata(
            reference_tables,
            begin=position,
            end=position + 1,
            prefill=False,
            request_slot=0,
            prompt_tokens=0,
        )
        outputs = reference.forward(
            inputs, begin=position, end=position + 1, prefill=False
        )
        for layer in _LAYERS:
            reference_outputs[layer].append(outputs[layer])
        reference_states[position + 1] = reference.states(
            reference_tables, end=position + 1
        )
    reference_outputs = {
        layer: torch.cat(outputs, dim=0) for layer, outputs in reference_outputs.items()
    }

    request = ts.RequestSpec()
    request.request_id = "source"
    request.tokens = list(range(4))
    request.max_new_tokens = 40
    scheduler.submit_requests([request])
    batch, tables = _batch(scheduler, "source")
    assert list(batch.input_lengths) == [4]
    actual.metadata(
        tables,
        begin=0,
        end=4,
        prefill=True,
        request_slot=batch.request_pool_indices[0],
        prompt_tokens=batch.prefill_lengths[0],
    )
    outputs = actual.forward(inputs, begin=0, end=4, prefill=True)
    for layer in _LAYERS:
        _assert_close(outputs[layer], reference_outputs[layer][:4], "initial prefill")
    _assert_states(actual.states(tables, end=4), reference_states[4], "initial prefill")
    _feedback(scheduler, request_id="source", tokens=[4], decode=False)

    computed = 4
    previous_tables = tables
    released = {group: set() for group in KIMI_STATE_GROUPS}
    reused_outputs = {group: set() for group in KIMI_STATE_GROUPS}
    # 7 -> 10 crosses 8 without committing there; 10 -> 12 writes an aligned
    # partial acceptance. Both remain working state only; later GPU outputs
    # reuse retired blocks without retaining a Decode snapshot.
    for accepted in [1, 2, 3, 2, 1, 3] + [3, 1] * 6:
        batch, tables = _batch(scheduler, "source")
        assert batch.num_extends() == 0
        assert list(batch.input_lengths) == [_WIDTH]
        expired_slots = max(0, (computed - _WIDTH) // _P)
        for group in KIMI_STATE_GROUPS:
            row = tables[group][0]
            assert all(block == 0 for block in row[:expired_slots])
            for slot, block in enumerate(previous_tables[group][0]):
                if block > 0 and row[slot] == 0:
                    released[group].add(block)
        actual.metadata(
            tables,
            begin=computed,
            end=computed + _WIDTH,
            prefill=False,
            request_slot=batch.request_pool_indices[0],
            prompt_tokens=0,
        )
        outputs = actual.forward(
            inputs, begin=computed, end=computed + _WIDTH, prefill=False
        )
        actual.backend.commit_verified_state(
            torch.tensor([accepted], dtype=torch.int32, device="cuda")
        )
        end = computed + accepted
        for layer in _LAYERS:
            _assert_close(
                outputs[layer][:accepted],
                reference_outputs[layer][computed:end],
                f"accepted verify outputs at {end}",
            )
        _assert_states(
            actual.states(tables, end=end),
            reference_states[end],
            f"accepted endpoint {end}",
        )
        for group in KIMI_STATE_GROUPS:
            output_block = tables[group][0][(end - 1) // _P]
            if output_block in released[group]:
                reused_outputs[group].add(output_block)
        empty_before_feedback = scheduler.empty_lcm_blocks()
        assert empty_before_feedback > 0
        _feedback(
            scheduler,
            request_id="source",
            tokens=list(range(computed + 1, end + 1)),
            decode=True,
        )
        assert scheduler.empty_lcm_blocks() == empty_before_feedback
        previous_tables = tables
        computed = end
    assert computed == 40
    assert all(released.values())
    assert all(
        reused_outputs.values()
    ), "retired working blocks must become GPU outputs again"
    finish = ts.ForwardEvent.Finish()
    finish.request_id = "source"
    scheduler.advance(ts.ExecutionEvent().add_event(finish))

    for boundary in (8, 12, 40):
        request = ts.RequestSpec()
        request.request_id = f"resume_{boundary}"
        request.tokens = list(range(boundary)) + [999]
        request.max_new_tokens = 4
        scheduler.submit_requests([request])
        batch, tables = _batch(scheduler, request.request_id)
        begin = int(batch.extend_prefix_lens[0])
        assert begin == 4
        end = begin + int(batch.input_lengths[0])
        actual.metadata(
            tables,
            begin=begin,
            end=end,
            prefill=True,
            request_slot=batch.request_pool_indices[0],
            prompt_tokens=batch.prefill_lengths[0],
        )
        outputs = actual.forward(inputs, begin=begin, end=end, prefill=True)
        for layer in _LAYERS:
            _assert_close(
                outputs[layer],
                reference_outputs[layer][begin:end],
                f"resume {boundary}",
            )
        _assert_states(
            actual.states(tables, end=end), reference_states[end], f"resume {boundary}"
        )
        _feedback(
            scheduler,
            request_id=request.request_id,
            tokens=[1000] if end == len(request.tokens) else [],
            decode=False,
        )
        # Do not publish the probe's freshly recomputed checkpoints: later
        # probes must inspect only the original source's cache contents.
        abort = ts.ForwardEvent.Abort()
        abort.request_id = request.request_id
        scheduler.advance(ts.ExecutionEvent().add_event(abort))
    for layer in _LAYERS:
        for component in _COMPONENTS:
            assert (
                torch.count_nonzero(actual.pool.get_component(layer, component)[0]) == 0
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
