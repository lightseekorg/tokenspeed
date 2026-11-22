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
    _logical_checkpoint,
    _stub_contract,
    _StubContractPool,
    requires_fla,
)

import pytest
import torch
from ci_system.ci_register import register_cuda_ci
from tokenspeed_kernel.ops.metadata import REQUEST_ENDED

ts = pytest.importorskip("tokenspeed_engine.scheduler")

from test.runtime.scheduler_driver import (  # noqa: E402
    advance_scheduler,
    make_extend_result_event,
    make_finish_event,
    make_update_reserve_tokens_event,
)

from tokenspeed.runtime.engine.scheduler_utils import pool_to_cache_groups  # noqa: E402
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.execution.runtime_states import RuntimeStates  # noqa: E402
from tokenspeed.runtime.execution.types import StopCriteria  # noqa: E402
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (  # noqa: E402
    STATE_INPUT_SLOT,
    STATE_LIVE_SLOT,
)

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


def _checkpoint_components(pool, pages_by_layer):
    return {
        (layer, component): pool.get_component(layer, component)[pages].clone()
        for layer, pages in pages_by_layer.items()
        for component in _COMPONENTS
    }


class _KDA:
    def __init__(self, *, width: int, pool=None):
        if pool is None:
            self.contract = _stub_contract(prefix_granularity=_P, usable_pages=_USABLE)
            self.pool = _StubContractPool(
                self.contract,
                "cuda",
                conv_dim=3 * _H * _D,
                width=4,
                num_heads=_H,
                head_dim=_D,
            )
        else:
            self.pool = pool
            self.contract = pool.arena.runtime_contract
        self.backend = _backend("cuda", contract_pool=self.pool, spec_tokens=width)
        # RuntimeStates.stopped, as the executor binds it: a test marks the
        # slot of a request whose round ends it, before that round's commit.
        self.stopped = torch.zeros(16, dtype=torch.bool, device="cuda")
        # The executor's per-row request ends (sized like its max_bs).
        self.request_ends = torch.zeros(8, dtype=torch.int32, device="cuda")
        self.backend.bind_request_stop_flags(self.stopped, self.request_ends)
        self.backend.init_cuda_graph_state(max_bs=8)
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
                query_shard=None,
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

    def states(self, tables, request_slot: int | None):
        return {
            (layer, component): (
                _logical_checkpoint(
                    self.backend, layer, tables[group][0][STATE_LIVE_SLOT], request_slot
                )
                if component == "recurrent_state" and request_slot is not None
                else self.pool.get_component(layer, component)[
                    tables[group][0][STATE_LIVE_SLOT]
                ].clone()
            )
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


def _scheduler(pool, *, width: int, overlap_schedule_depth: int, usable_pages: int):
    config = ts.SchedulerConfig()
    config.prefix_granularity = _P
    config.num_device_pages = usable_pages + 1
    config.num_host_pages = 0
    config.num_snapshot_pages = 1  # the null page alone: no pool
    config.max_retracted_requests = 0
    config.max_scheduled_tokens = 8
    config.max_batch_size = 1
    config.max_resident_requests = 1
    config.decode_input_tokens = width
    config.overlap_schedule_depth = overlap_schedule_depth
    config.enable_mixed_prefill_decode = False
    config.disable_l2_cache = True
    config.cache_groups = pool_to_cache_groups(pool)
    return ts.Scheduler(config)


def _inputs():
    generator = torch.Generator(device="cpu").manual_seed(113)
    return {
        name: torch.randn((48, columns), generator=generator).to("cuda", torch.bfloat16)
        for name, columns in (
            ("mixed_qkv", 3 * _H * _D),
            ("f_a_out", _FA),
            ("beta_raw", _H),
        )
    }


def _reference(
    inputs, num_positions: int, params, *, checkpoints=(), recurrent_dtype=torch.float32
):
    """Compute independent tokenwise states, rounding once at each dense checkpoint."""
    from test.runtime.kda_torch_reference import kda_torch_reference

    outputs, states = {}, {count: {} for count in range(1, num_positions + 1)}
    for layer in _LAYERS:
        weights = params[layer]
        gate = torch.nn.functional.linear(
            inputs["f_a_out"][:num_positions], weights["f_b_weight"]
        )
        conv = torch.zeros(1, 3 * _H * _D, 3, device="cuda", dtype=torch.bfloat16)
        base = torch.zeros(1, _H, _D, _D, device="cuda")
        parts = []
        begin = 0
        for end in (*checkpoints, num_positions):
            output, recurrent, windows = kda_torch_reference(
                inputs["mixed_qkv"][begin:end].unsqueeze(0),
                conv,
                base,
                inputs["beta_raw"][begin:end],
                conv_weights=weights["conv_weights"],
                bias=None,
                gate=gate[begin:end],
                A_log=weights["A_log"],
                dt_bias=weights["dt_bias"],
                lower_bound=-5.0,
            )
            parts.append(output[0])
            for i in range(end - begin):
                states[begin + i + 1][layer, "recurrent_state"] = recurrent[0, i]
                states[begin + i + 1][layer, "conv_state"] = windows[0, i]
            if end != num_positions:
                base = recurrent[:, -1].to(recurrent_dtype).float()
                conv = windows[:, -1]
                states[end][layer, "recurrent_state"] = base[0]
            begin = end
        outputs[layer] = torch.cat(parts)
    return outputs, states


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
def test_speculative_decode_evolves_the_live_block_and_preserves_the_checkpoints():
    actual = _KDA(width=_WIDTH)
    # The source's history and state plus one probe fit without eviction;
    # the small FIFO makes later probes recycle aborted probe blocks.
    scheduler = _scheduler(
        actual.pool, width=_WIDTH, overlap_schedule_depth=0, usable_pages=24
    )
    inputs = _inputs()
    reference_outputs, reference_states = _reference(inputs, 41, actual.params)

    request = ts.RequestSpec()
    request.request_id = "source"
    request.tokens = list(range(4))
    request.max_new_tokens = 40
    scheduler.submit_requests([request])
    batch, tables = _batch(scheduler, "source")
    assert list(batch.input_lengths) == [4]
    source_slot = batch.request_pool_indices[0]
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
    _assert_states(
        actual.states(tables, source_slot), reference_states[4], "initial prefill"
    )
    _feedback(scheduler, request_id="source", tokens=[4], decode=False)
    prompt_end_tables = tables
    prompt_end = actual.states(tables, source_slot)

    computed = 4
    live_blocks = None
    # Every verify round reads the committed state and commits the accepted
    # prefix into the live block; the first round reads the prompt-end
    # checkpoint from slot 0, every later one evolves the live block in place.
    rounds = [1, 2, 3, 2, 1, 3] + [3, 1] * 6
    for index, accepted in enumerate(rounds):
        batch, tables = _batch(scheduler, "source")
        assert batch.num_extends() == 0
        assert list(batch.input_lengths) == [_WIDTH]
        for group in KIMI_STATE_GROUPS:
            row = tables[group][0]
            if index == 0:
                assert (
                    row[STATE_INPUT_SLOT]
                    == prompt_end_tables[group][0][STATE_LIVE_SLOT]
                )
                assert row[STATE_LIVE_SLOT] not in prompt_end_tables[group][0]
            else:
                assert row[STATE_INPUT_SLOT] == 0
                assert row[STATE_LIVE_SLOT] == live_blocks[group]
        live_blocks = {
            group: tables[group][0][STATE_LIVE_SLOT] for group in KIMI_STATE_GROUPS
        }
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
        if index == len(rounds) - 1:
            # The round reaches the request's token limit: the sampling
            # epilogue marks its slot before the commit runs.
            actual.stopped[source_slot] = True
        actual.backend.commit_verified_state(
            torch.tensor([accepted], dtype=torch.int32, device="cuda"),
            accepted_path=None,
        )
        end = computed + accepted
        for layer in _LAYERS:
            _assert_close(
                outputs[layer][:accepted],
                reference_outputs[layer][computed:end],
                f"accepted verify outputs at {end}",
            )
        _assert_states(
            actual.states(tables, source_slot),
            reference_states[end],
            f"accepted endpoint {end}",
        )
        empty_before_feedback = scheduler.empty_lcm_blocks()
        assert empty_before_feedback > 0
        _feedback(
            scheduler,
            request_id="source",
            tokens=list(range(computed + 1, end + 1)),
            decode=True,
        )
        assert scheduler.empty_lcm_blocks() == empty_before_feedback
        computed = end
    assert computed == 40
    # The ending round folded the slot's records: the live block alone holds
    # the endpoint the finish publishes.
    finish_endpoint = actual.states(tables, None)
    for layer in _LAYERS:
        assert actual.backend._retained_lengths[layer][source_slot].item() == 0
    _assert_states(finish_endpoint, reference_states[40], "ended live block")
    _assert_states(
        actual.states(prompt_end_tables, None),
        prompt_end,
        "prompt-end checkpoint",
    )
    finish = ts.ForwardEvent.Finish()
    finish.request_id = "source"
    scheduler.advance(ts.ExecutionEvent().add_event(finish))
    # The executor clears the flag when the slot's next request starts.
    actual.stopped[source_slot] = False
    finish_tables = tables
    source_pages = {
        layer: [
            prompt_end_tables[group][0][STATE_LIVE_SLOT],
            finish_tables[group][0][STATE_LIVE_SLOT],
        ]
        for layer, group in zip(_LAYERS, KIMI_STATE_GROUPS, strict=True)
    }
    published_components = None
    aborted_live_pages = set()
    reused_live_pages = set()

    # The published checkpoints are the prompt end (4 tokens) and the finish
    # (40 computed tokens). A probe sharing 8 or 12 tokens resumes from the
    # prompt-end state; one sharing all 40 resumes from the finish state.
    for boundary, begin in ((8, 4), (12, 4), (40, 40)):
        request = ts.RequestSpec()
        request.request_id = f"resume_{boundary}"
        request.tokens = list(range(boundary)) + [999]
        request.max_new_tokens = 4
        scheduler.submit_requests([request])
        batch, tables = _batch(scheduler, request.request_id)
        if published_components is None:
            published_components = _checkpoint_components(actual.pool, source_pages)
            for layer in _LAYERS:
                for component in _COMPONENTS:
                    assert torch.equal(
                        published_components[layer, component][1],
                        finish_endpoint[layer, component],
                    ), f"publication changed the ended layer {layer} {component}"
            # The next resident can overwrite the source's former slot.
            actual.backend._retained_records[source_slot].fill_(float("nan"))
        assert int(batch.extend_prefix_lens[0]) == begin
        checkpoint_tables = prompt_end_tables if begin == 4 else finish_tables
        for group in KIMI_STATE_GROUPS:
            assert (
                tables[group][0][STATE_INPUT_SLOT]
                == checkpoint_tables[group][0][STATE_LIVE_SLOT]
            )
        probe_live_pages = {
            group: tables[group][0][STATE_LIVE_SLOT] for group in KIMI_STATE_GROUPS
        }
        for layer, group in zip(_LAYERS, KIMI_STATE_GROUPS, strict=True):
            assert probe_live_pages[group] not in source_pages[layer]
        # Packing is one block per parent. A freed parent may be assigned to
        # another group, so follow its physical id across group assignments.
        reused_live_pages.update(set(probe_live_pages.values()) & aborted_live_pages)
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
            actual.states(tables, batch.request_pool_indices[0]),
            reference_states[end],
            f"resume {boundary}",
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
        aborted_live_pages.update(probe_live_pages.values())
        for key, value in _checkpoint_components(actual.pool, source_pages).items():
            assert torch.equal(
                value.view(torch.uint8), published_components[key].view(torch.uint8)
            ), f"probe execution and abort changed published {key}"
    assert (
        reused_live_pages
    ), f"later probes did not reuse aborted private pages: {reused_live_pages}"
    _assert_states(
        actual.states(prompt_end_tables, None),
        prompt_end,
        "prompt-end checkpoint",
    )
    for layer in _LAYERS:
        for component in _COMPONENTS:
            assert (
                torch.count_nonzero(actual.pool.get_component(layer, component)[0]) == 0
            )


_STOP = 99


class _Commit:
    """What the control plane reports to the scheduler for the one request
    (``ts-control``'s commit): every token a forward kept; then a finish on
    the stop token, or the decode row's reserve; and once the request
    finished, only the kept tokens of a forward still in flight."""

    def __init__(self) -> None:
        self.finished = False

    def land(self, scheduler, batch, sampled: list[int], accepted: int) -> list[int]:
        """Report a landed forward; returns the tokens the scheduler was told
        it produced."""
        torch.cuda.synchronize()
        rid = batch.request_ids[0]
        kept = sampled[:accepted]
        events = [make_extend_result_event(rid, kept)]
        if not self.finished:
            if _STOP in kept:
                self.finished = True
                events.append(make_finish_event(rid))
            elif batch.num_extends() == 0:
                events.append(make_update_reserve_tokens_event(rid, accepted))
        advance_scheduler(scheduler, events)
        return kept


def _masked_tables(runtime, tables, slot):
    """The executor's padding of a stopped request's state rows."""
    masked = runtime.masked_state_tables(
        {
            group: torch.tensor(rows, dtype=torch.int32, device="cuda")
            for group, rows in tables.items()
        },
        torch.tensor([slot], dtype=torch.int64, device="cuda"),
        KIMI_STATE_GROUPS,
    )
    return {group: table.tolist() for group, table in masked.items()}


@requires_cuda
@requires_fla
@pytest.mark.parametrize(
    "width,overlap",
    [(_WIDTH, False), (_WIDTH, True), (1, True)],
    ids=["spec", "spec_overlap", "overlap"],
)
def test_the_checkpoint_published_at_finish_holds_the_state_its_key_names(
    width, overlap
):
    """A finish publishes the live slot under the key of the scheduler's
    frontier (every reported token but the last). The round that samples the
    stop token as its first output accepts its whole window; the device keeps
    it through the stop token, and the forward the overlap schedule dispatched
    behind it runs the request as a padding row. The published state is the
    state before the stop token, bit for bit the oracle's."""
    actual = _KDA(width=width)
    scheduler = _scheduler(
        actual.pool,
        width=width,
        overlap_schedule_depth=int(overlap),
        usable_pages=_USABLE,
    )
    inputs = _inputs()
    _, reference_states = _reference(inputs, 24, actual.params)
    runtime = RuntimeStates(
        req_pool_size=8,
        input_vocab_size=1024,
        vocab_size=1024,
        output_length=width,
        device="cuda",
    )
    # As the executor does: the decode commit reads the epilogue's stop flags
    # and reports a row whose deferred state it could not complete.
    request_ends = torch.zeros(8, dtype=torch.int32, device="cuda")
    actual.backend.bind_request_stop_flags(runtime.stopped, request_ends)
    commit = _Commit()
    prompt = list(range(4))
    criteria = StopCriteria(stop_token_ids=(_STOP,), token_limit=len(prompt) + 40)
    request = ts.RequestSpec()
    request.request_id = "source"
    request.tokens = prompt
    request.max_new_tokens = 40
    scheduler.submit_requests([request])

    batch, tables = _batch(scheduler, "source")
    actual.metadata(
        tables,
        begin=0,
        end=4,
        prefill=True,
        request_slot=batch.request_pool_indices[0],
        prompt_tokens=batch.prefill_lengths[0],
    )
    actual.forward(inputs, begin=0, end=4, prefill=True)
    reported = prompt + commit.land(scheduler, batch, [104], accepted=1)
    # The device-side length: the tokens the live slot has consumed. A round
    # starting there samples the token at sequence index consumed + j as its
    # j-th output.
    consumed = 4

    def run_round(batch, tables, accepted, *, samples_stop=False):
        """One decode in the model executor's order; returns the sampled rows
        and the tokens the round keeps."""
        nonlocal consumed
        slot = batch.request_pool_indices[0]
        runtime.bind_stop_criteria([1], batch.request_pool_indices, [criteria])
        if overlap:
            tables = _masked_tables(runtime, tables, slot)
        actual.metadata(
            tables,
            begin=consumed,
            end=consumed + width,
            prefill=False,
            request_slot=slot,
            prompt_tokens=0,
        )
        actual.forward(inputs, begin=consumed, end=consumed + width, prefill=False)
        sampled = [100 + consumed + j for j in range(1, width + 1)]
        if samples_stop:
            sampled[0] = _STOP
        runtime.valid_cache_lengths[slot] = consumed
        accept_lengths = torch.tensor([accepted], dtype=torch.int32, device="cuda")
        runtime.apply_stop_conditions(
            torch.tensor([slot], dtype=torch.int64, device="cuda"),
            torch.tensor(sampled, dtype=torch.int32, device="cuda"),
            accept_lengths,
            torch.tensor([width], dtype=torch.int32, device="cuda"),
            torch.empty(0, dtype=torch.int32, device="cuda"),
            request_ends[:1],
            num_extends=0,
            num_prefill_outputs=0,
            decode_width=width,
            padding_index=runtime.valid_cache_lengths.shape[0] - 1,
        )
        actual.backend.commit_state_after_acceptance(
            accept_lengths, num_extends=0, accepted_path=None
        )
        kept = int(accept_lengths[0])
        consumed += kept
        return sampled, kept

    for accepted in [2, 3] if width > 1 else [1, 1]:
        batch, tables = _batch(scheduler, "source")
        sampled, kept = run_round(batch, tables, accepted)
        assert kept == accepted
        reported += commit.land(scheduler, batch, sampled, kept)

    batch, tables = _batch(scheduler, "source")
    sampled, kept = run_round(batch, tables, width, samples_stop=True)
    assert kept == 1
    # The device ended the request with its decode state complete.
    assert request_ends[0].item() == REQUEST_ENDED
    if overlap:
        # Planned while the finishing round is still in flight.
        next_batch, next_tables = _batch(scheduler, "source")
    reported += commit.land(scheduler, batch, sampled, kept)
    assert commit.finished
    if overlap:
        # A queued stopped row must leave the entire checkpoint pool untouched,
        # including rejected records outside the published accepted prefix.
        all_pages = {layer: slice(None) for layer in _LAYERS}
        finished_components = _checkpoint_components(actual.pool, all_pages)
        next_sampled, next_kept = run_round(next_batch, next_tables, min(2, width))
        assert next_kept == 0
        reported += commit.land(scheduler, next_batch, next_sampled, next_kept)
        for key, value in _checkpoint_components(actual.pool, all_pages).items():
            assert torch.equal(
                value.view(torch.uint8), finished_components[key].view(torch.uint8)
            ), f"queued stopped row changed checkpoint component {key}"

    # A prompt extending the reported tokens resumes at the frontier, the
    # position of the stop token, from the live slot the finish published.
    frontier = len(reported) - 1
    assert reported[-1] == _STOP
    probe = ts.RequestSpec()
    probe.request_id = "probe"
    probe.tokens = reported + [999]
    probe.max_new_tokens = 4
    scheduler.submit_requests([probe])
    batch, probe_tables = _batch(scheduler, "probe")
    assert int(batch.extend_prefix_lens[0]) == frontier
    published = actual.states(tables, None)
    for group in KIMI_STATE_GROUPS:
        assert (
            probe_tables[group][0][STATE_INPUT_SLOT]
            == tables[group][0][STATE_LIVE_SLOT]
        )
    # The conv state holds the last inputs bit for bit, so it names the token
    # count the checkpoint actually reached.
    reached = [
        count
        for count, expected in reference_states.items()
        if all(
            torch.equal(published[layer, "conv_state"], expected[layer, "conv_state"])
            for layer in _LAYERS
        )
    ]
    assert reached == [frontier], (
        f"the checkpoint published under the key of {frontier} tokens holds "
        f"the state after {reached} tokens"
    )
    _assert_states(published, reference_states[frontier], "finish checkpoint")


def _transfer_kda_pool(usable_pages: int, recurrent_dtype: torch.dtype):
    """The production arena and pool, with one parent per cache block."""
    from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
    from tokenspeed.runtime.layers.attention.kv_cache.hybrid_kda import (
        HybridKDATokenToKVPool,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.kda import (
        kda_state_fields,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
        CacheFieldSpec,
        pack,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
        FULL_ATTENTION,
        LINEAR_ATTENTION,
        CacheGroupSpec,
    )

    groups = (
        (
            CacheGroupSpec(
                group_id=FULL_ATTENTION,
                attn_kind="full_history",
                rows_per_page=_P,
                entry_stride_tokens=1,
                replayable=False,
            ),
            (
                CacheFieldSpec(
                    "layer.3.latent_kv", "latent.0", (_P, 1, 576), "bfloat16"
                ),
            ),
        ),
        *(
            (
                CacheGroupSpec(group_id=group, attn_kind="state", replayable=False),
                kda_state_fields(
                    layer,
                    "slot.0",
                    conv_shape=(3 * _H * _D, 3),
                    recurrent_shape=(_H, _D, _D),
                    recurrent_state_dtype=recurrent_dtype,
                ),
            )
            for layer, group in zip(_LAYERS, KIMI_STATE_GROUPS, strict=True)
        ),
    )
    plan = pack(
        groups,
        prefix_granularity=_P,
        cache_blocks_per_lcm_block={spec.group_id: 1 for spec, _ in groups},
        max_padding_fraction=float("inf"),
    ).bind(usable_pages)
    arena = CacheArena(
        plan, "cuda", cache_group_specs=tuple(spec for spec, _ in groups)
    )
    return HybridKDATokenToKVPool(
        arena=arena,
        model_dtype=torch.bfloat16,
        dtype=torch.bfloat16,
        quant_method=None,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_num=4,
        rank=0,
        layer_types=(
            LINEAR_ATTENTION,
            LINEAR_ATTENTION,
            LINEAR_ATTENTION,
            FULL_ATTENTION,
        ),
    )


@requires_cuda
@requires_fla
@pytest.mark.parametrize("io_backend", ["direct", "kernel"])
@pytest.mark.parametrize("recurrent_dtype", [torch.float32, torch.bfloat16])
def test_cache_pressure_restores_retained_kda_state_and_continues_decode(
    recurrent_dtype, io_backend
):
    """Pressure folds accepted updates into the dense page before the store.

    Restore into another slot, leaving its GPU workspace poisoned, and
    compare resumed decode to a PyTorch recurrence with the same checkpoints.
    """
    import numpy as np

    from tokenspeed.runtime.cache.l2.executor import HostCacheExecutor
    from tokenspeed.runtime.cache.l2.sizing import RetractionPoolRequest
    from tokenspeed.runtime.cache.transfer.ops import RestoreOp, SnapshotOp
    from tokenspeed.runtime.engine.scheduler_utils import cache_ops_from_plan

    usable_pages = 16
    pool = _transfer_kda_pool(usable_pages, recurrent_dtype)
    actual = _KDA(width=_WIDTH, pool=pool)
    assert actual.backend.slot_state_bytes() == 0
    inputs = _inputs()
    pressure_inputs = {name: -value for name, value in inputs.items()}
    reference_outputs, reference_states = _reference(
        inputs, 47, actual.params, checkpoints=(8,), recurrent_dtype=recurrent_dtype
    )
    executor = HostCacheExecutor(
        pool,
        draft_pool=None,
        l2_tier=False,
        host_ratio=0,
        host_size_gb=0,
        snapshot_pool=RetractionPoolRequest(
            host_gb=0, ratio=1, max_retracted_requests=1, tail_lcm_blocks_per_request=0
        ),
        slot_state_exporters=(actual.backend,),
        io_backend=io_backend,
        attn_tp_rank=0,
        kvp_rank=0,
        nprocs_per_node=1,
        world_cpu_group=None,
    )
    config = ts.SchedulerConfig()
    config.prefix_granularity = _P
    config.num_device_pages = usable_pages + 1
    config.num_host_pages = 0
    config.num_snapshot_pages = executor.num_snapshot_pages
    config.max_retracted_requests = executor.max_retracted_requests
    config.max_scheduled_tokens = 16
    config.max_batch_size = 2
    config.max_resident_requests = config.max_batch_size
    config.decode_input_tokens = _WIDTH
    config.overlap_schedule_depth = 0
    config.enable_mixed_prefill_decode = False
    config.disable_l2_cache = True
    # As the data plane configures them: the scheduler lists a retraction
    # victim's live blocks in these groups for the fold.
    cache_groups = pool_to_cache_groups(pool)
    deferred = actual.backend.deferred_decode_state_groups()
    for group in cache_groups:
        group.deferred_decode_state = group.group_id in deferred
    config.cache_groups = cache_groups
    config.host_group_packing = list(executor.host_group_packing)
    scheduler = ts.Scheduler(config)
    group_ids = tuple(spec.group_id for spec in pool.arena.cache_group_specs)
    computed = {"source": 0, "pressure": 0, "reuse": 0}
    admission_ids = {"source": 1, "pressure": 2, "reuse": 3}
    source_tables = None
    saved_pages = {}
    logical_before_store = {}
    snapshot_rows = {}
    stored_fields = {}
    pending_snapshot_acks = []
    resumed = False

    def run_forward(batch, row: int):
        nonlocal source_tables
        rid = batch.request_ids[row]
        tables = {
            group: [list(rows[row])] for group, rows in batch.block_tables.items()
        }
        begin = computed[rid]
        end = begin + int(batch.input_lengths[row])
        prefill = row < batch.num_extends()
        actual.metadata(
            tables,
            begin=begin,
            end=end,
            prefill=prefill,
            request_slot=batch.request_pool_indices[row],
            prompt_tokens=batch.prefill_lengths[row] if prefill else 0,
        )
        outputs = actual.forward(
            inputs if rid == "source" else pressure_inputs,
            begin=begin,
            end=end,
            prefill=prefill,
        )
        accepted = end - begin
        if not prefill:
            # Partial acceptance leaves a nonempty retained prefix, including
            # when the backend consolidates a full history before verify.
            accepted = 2
            actual.backend.commit_verified_state(
                torch.tensor([accepted], dtype=torch.int32, device="cuda"),
                accepted_path=None,
            )
            end = begin + accepted
        computed[rid] = end
        if rid == "source":
            source_tables = tables
            for layer in _LAYERS:
                _assert_close(
                    outputs[layer][:accepted],
                    reference_outputs[layer][begin:end],
                    f"pressure integration outputs {begin}:{end}",
                )
            _assert_states(
                actual.states(tables, batch.request_pool_indices[row]),
                reference_states[end],
                f"endpoint {end}",
            )
        _feedback(
            scheduler,
            request_id=rid,
            tokens=[1000 + end] if prefill else list(range(1000 + begin, 1000 + end)),
            decode=not prefill,
        )
        return tables

    def execute_cache(plan):
        operations = cache_ops_from_plan(
            plan, request_admission=admission_ids.__getitem__
        )
        assert not operations.storage
        stream = torch.cuda.current_stream()
        # As the data plane does first (DeviceHandle.execute): fold the
        # retraction victims' records into the live pages the plan lists.
        folds = {
            group_id: torch.from_numpy(pairs.copy())
            for group_id, pairs in plan.state_folds_arrays().items()
            if len(pairs)
        }
        if any(isinstance(op, SnapshotOp) for op in operations.device):
            assert set(folds) == set(KIMI_STATE_GROUPS), folds
        if folds:
            actual.backend.fold_deferred_state(folds, stream=stream)
        executor.submit_write_backs(
            operations.device, prerequisite_stream=stream, fence_stream=stream
        )
        for op in operations.device:
            if isinstance(op, SnapshotOp):
                for layer, group in zip(_LAYERS, KIMI_STATE_GROUPS, strict=True):
                    live = source_tables[group][0][STATE_LIVE_SLOT]
                    dense = pool.get_component(layer, "recurrent_state")[live]
                    torch.testing.assert_close(
                        dense.float(),
                        logical_before_store[layer],
                        atol=2e-7,
                        rtol=max(2e-5, torch.finfo(recurrent_dtype).eps / 2),
                    )
                    pages = [page for page in source_tables[group][0] if page]
                    saved_pages.update(_checkpoint_components(pool, {layer: pages}))
                for row in op.transfers:
                    for field in pool.arena.plan.fields:
                        if field.group_id == group_ids[row.group_id]:
                            begin = pool.arena.field_block_byte_offset(
                                field.field_id, row.source_page
                            )
                            stored_fields[row.destination_page, field.field_id] = (
                                pool.arena.buffer[
                                    begin : begin + field.payload_bytes
                                ].clone()
                            )
        copies = {
            group: pairs
            for group, pairs in plan.pages_to_copy_arrays().items()
            if len(pairs)
        }
        if copies:
            pool.arena.copy_blocks(copies)
        zeros = {
            group: np.asarray(pages, dtype=np.int64)
            for group, pages in plan.pages_to_zero.items()
            if len(pages)
        }
        if zeros:
            pool.zero_new_blocks(zeros)
        expected_arena = None
        for operation in operations.device:
            if isinstance(operation, RestoreOp):
                expected_arena = pool.arena.buffer.clone()
                for row in operation.transfers:
                    group = group_ids[row.group_id]
                    for field in pool.arena.plan.fields:
                        if field.group_id == group:
                            begin = pool.arena.field_block_byte_offset(
                                field.field_id, row.destination_page
                            )
                            expected_arena[begin : begin + field.payload_bytes].copy_(
                                stored_fields[row.source_page, field.field_id]
                            )
        executor.submit_load_backs(operations.device, prerequisite_stream=stream)
        # All streams, including the Host executor's transfer streams, have
        # landed before their genuine completion events reach the scheduler.
        torch.cuda.synchronize()
        if expected_arena is not None:
            assert torch.equal(
                pool.arena.buffer, expected_arena
            ), "restore changed bytes outside its destinations or lost image bytes"
        acknowledgments = executor.poll_results()
        ready = []
        for acknowledgment in acknowledgments:
            if isinstance(acknowledgment, ts.Cache.SnapshotDoneEvent):
                # Model delayed completion polling: retain the genuine event
                # after its GPU transfer landed. New work can reuse the pages
                # before the scheduler sees the image as ready to restore.
                pending_snapshot_acks.append(acknowledgment)
            else:
                ready.append(acknowledgment)
        if ready:
            advance_scheduler(scheduler, ready)
        return operations.device

    try:
        source = ts.RequestSpec()
        source.request_id = "source"
        source.tokens = list(range(8))
        source.max_new_tokens = 0  # No declared budget: remain eligible for retraction.
        pressure = ts.RequestSpec()
        pressure.request_id = "pressure"
        pressure.tokens = [900, 901, 902, 903]
        pressure.max_new_tokens = 0
        scheduler.submit_requests([source, pressure])
        # Both real residents grow. A victim can give its pages to another
        # decoder before either request fills the whole arena by itself.
        for _ in range(3):
            plan = scheduler.next_execution_plan()
            assert not any(isinstance(op, ts.Cache.SnapshotOp) for op in plan.cache)
            execute_cache(plan)
            for batch in plan.forward:
                for row in range(len(batch.request_ids)):
                    run_forward(batch, row)

        for _ in range(24):
            plan = scheduler.next_execution_plan()
            operations = cache_ops_from_plan(
                plan, request_admission=admission_ids.__getitem__
            ).device
            snapshots = [op for op in operations if isinstance(op, SnapshotOp)]
            if snapshots:
                [snapshot] = snapshots
                assert snapshot.request_id == "source"
                assert scheduler.retracted_size() == 1
                assert source_tables is not None
                source_slot = snapshot.request_pool_index
                for layer, group in zip(_LAYERS, KIMI_STATE_GROUPS, strict=True):
                    live = source_tables[group][0][STATE_LIVE_SLOT]
                    length = actual.backend._retained_lengths[layer][source_slot].item()
                    assert 0 < length < actual.backend._retained_capacity
                    logical_before_store[layer] = _logical_checkpoint(
                        actual.backend, layer, live, source_slot
                    )
                reference_outputs, reference_states = _reference(
                    inputs,
                    47,
                    actual.params,
                    checkpoints=(8, computed["source"]),
                    recurrent_dtype=recurrent_dtype,
                )
                snapshot_rows = {
                    (group_ids[row.group_id], row.source_page): row.destination_page
                    for row in snapshot.transfers
                }
                assert set(snapshot_rows) == {
                    (group, page)
                    for group, rows in source_tables.items()
                    for page in rows[0]
                    if page
                }, "snapshot-only execution must store the entire scheduler image"
                execute_cache(plan)
                actual.backend._retained_records[source_slot].fill_(float("nan"))
                for batch in plan.forward:
                    for row, rid in enumerate(batch.request_ids):
                        assert rid == "pressure"
                        run_forward(batch, row)
                assert [int(event.op_id) for event in pending_snapshot_acks] == [
                    snapshot.op_id
                ], "only the completed snapshot's real ACK may be delayed"
                advance_scheduler(scheduler, [make_finish_event("pressure")])
                reuse = ts.RequestSpec()
                reuse.request_id = "reuse"
                reuse.tokens = [950, 951, 952, 953]
                reuse.max_new_tokens = 1
                scheduler.submit_requests([reuse])
                # The source image is not restore-ready until its held ACK
                # reaches the scheduler. A new real owner can now overwrite
                # the released storage, without a restore-priority seal.
                pressure_tables = None
                for _ in range(8):
                    reuse_plan = scheduler.next_execution_plan()
                    assert not any(
                        isinstance(op, RestoreOp)
                        for op in cache_ops_from_plan(
                            reuse_plan, request_admission=admission_ids.__getitem__
                        ).device
                    ), "source restored before its old storage was reused"
                    execute_cache(reuse_plan)
                    for batch in reuse_plan.forward:
                        for row, rid in enumerate(batch.request_ids):
                            assert rid == "reuse"
                            tables = run_forward(batch, row)
                            if rid == "reuse":
                                pressure_tables = tables
                    if pressure_tables is not None:
                        break
                else:
                    pytest.fail(
                        "retraction never granted the new state owner a forward"
                    )
                old_parents = {page for _, page in snapshot_rows}
                new_parents = {
                    page
                    for rows in pressure_tables.values()
                    for page in rows[0]
                    if page
                }
                assert old_parents & new_parents, "pressure never reused source storage"
                assert any(
                    not torch.equal(
                        pool.get_component(layer, component)[
                            [page for page in source_tables[group][0] if page]
                        ].view(torch.uint8),
                        saved_pages[layer, component].view(torch.uint8),
                    )
                    for layer, group in zip(_LAYERS, KIMI_STATE_GROUPS, strict=True)
                    for component in _COMPONENTS
                ), (
                    f"old checkpoint bytes were never overwritten: "
                    f"source={source_tables}, contender={pressure_tables}, "
                    f"snapshot={snapshot_rows}"
                )
                advance_scheduler(scheduler, [make_finish_event("reuse")])
                advance_scheduler(scheduler, pending_snapshot_acks)
                pending_snapshot_acks.clear()
                break
            execute_cache(plan)
            for batch in plan.forward:
                for row in range(len(batch.request_ids)):
                    run_forward(batch, row)
        else:
            pytest.fail("real cache pressure did not retract the KDA source")

        for _ in range(8):
            plan = scheduler.next_execution_plan()
            operations = cache_ops_from_plan(
                plan, request_admission=admission_ids.__getitem__
            ).device
            restores = [op for op in operations if isinstance(op, RestoreOp)]
            if restores:
                [restore] = restores
                assert restore.request_id == "source"
                assert not any(batch.request_ids for batch in plan.forward)
                restored_pages = {
                    (group_ids[row.group_id], row.source_page): row.destination_page
                    for row in restore.transfers
                }
                assert set(restored_pages) == {
                    (group, host_page)
                    for (group, _), host_page in snapshot_rows.items()
                }
                assert any(
                    restored_pages[group, snapshot_rows[group, page]] != page
                    for group in KIMI_STATE_GROUPS
                    for page in source_tables[group][0]
                    if page
                ), "every state page restored to its original device address"
                assert restore.request_pool_index != source_slot
                records = actual.backend._retained_records[restore.request_pool_index]
                records.fill_(float("nan"))
                execute_cache(plan)
                assert torch.isnan(records).all(), "restore wrote into GPU-only records"
                for layer, group in zip(_LAYERS, KIMI_STATE_GROUPS, strict=True):
                    destinations = [
                        restored_pages[group, snapshot_rows[group, page]]
                        for page in source_tables[group][0]
                        if page
                    ]
                    for component in _COMPONENTS:
                        assert torch.equal(
                            pool.get_component(layer, component)[destinations].view(
                                torch.uint8
                            ),
                            saved_pages[layer, component].view(torch.uint8),
                        ), f"restore changed layer {layer} {component} bytes"
                restored_tables = {
                    group: [
                        [
                            (
                                restored_pages[group, snapshot_rows[group, page]]
                                if page
                                else 0
                            )
                            for page in rows[0]
                        ]
                    ]
                    for group, rows in source_tables.items()
                }
                _assert_states(
                    actual.states(restored_tables, None),
                    reference_states[computed["source"]],
                    "restored endpoint before decode",
                )
                assert scheduler.retracted_size() == 0
                resumed = True
                continue
            execute_cache(plan)
            batches = [batch for batch in plan.forward if batch.request_ids]
            if batches:
                assert resumed, "a forward preceded the restore completion"
                [batch] = batches
                assert list(batch.request_ids) == ["source"]
                assert (
                    batch.num_extends() == 0
                ), "restore must continue decode, not replay"
                for _ in range(2):
                    run_forward(batch, 0)
                    if _ == 0:
                        plan = scheduler.next_execution_plan()
                        execute_cache(plan)
                        [batch] = [batch for batch in plan.forward if batch.request_ids]
                        assert batch.num_extends() == 0
                break
        else:
            pytest.fail("the stored image did not restore and resume decode")
    finally:
        executor.shutdown()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
