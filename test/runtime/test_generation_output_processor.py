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

from __future__ import annotations

import os
import sys
from http import HTTPStatus

import pytest
import torch

from tokenspeed.runtime.engine.generation_output_processor import (
    OutputProcesser,
    RequestState,
)
from tokenspeed.runtime.engine.request_handler import RequestHandler
from tokenspeed.runtime.engine.request_stats import (
    NOOP_STATS,
    RequestStats,
    RequestStatsTracker,
)
from tokenspeed.runtime.execution.types import (
    MAX_STOP_TOKEN_IDS,
    ModelExecutionResult,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=15, suite="runtime-1gpu")


class _Sender:
    def __init__(self):
        self.items = []

    def send_pyobj(self, obj):
        self.items.append(obj)


class _Tokenizer:
    eos_token_id = None
    additional_stop_token_ids = None

    def decode(self, ids):
        return "".join(str(i) for i in ids)


class _Metrics:
    enabled = False

    def __init__(self):
        self.nan_aborts = 0
        self.capacity_aborts = 0

    def record_nan_abort(self):
        self.nan_aborts += 1

    def record_capacity_abort(self):
        self.capacity_aborts += 1


class _ForwardOp:
    request_ids = ["prefill", "decode"]
    request_pool_indices = [0, 1]
    input_lengths = [4, 1]
    # Total prefill size per extend slot, as C++ Request::PrefillSize() reports
    # it: for a fresh request this equals prefix + input.
    prefill_lengths = [4]
    extend_prefix_lens = [0]
    extend_replay_lens = [0]

    def num_extends(self):
        return 1


class _ExecutionResult:
    output_tokens = torch.tensor([11, 22], dtype=torch.int32)
    output_lengths = torch.tensor([1, 1], dtype=torch.int32)
    output_logprobs = None
    output_nan_flags = None
    grammar_completion = None
    next_input_ids = None
    input_token_logprobs = None
    input_logprob_plan = None
    input_top_logprobs_val = None
    input_top_logprobs_idx = None
    output_top_logprobs_val = None
    output_top_logprobs_idx = None


def _state(input_ids: list[int], *, computed_length: int = 0) -> RequestState:
    state = RequestState(
        prompt_input_ids=input_ids,
        sampling_params=SamplingParams(max_new_tokens=8, stop=[], ignore_eos=True),
        stream=False,
        tokenizer=_Tokenizer(),
        computes_prompt_logprobs=True,
    )
    state.computed_length = computed_length
    return state


def test_mixed_forward_updates_reserve_for_decode_slots_only():
    sender = _Sender()
    processor = OutputProcesser(
        sender,
        attn_tp_rank=0,
        metrics=_Metrics(),
    )
    processor.rid_to_state["prefill"] = _state([1, 2, 3, 4])
    processor.rid_to_state["decode"] = _state([5, 6, 7], computed_length=3)

    events = processor.post_process_forward_op(
        _ForwardOp(), _ExecutionResult(), is_prefill_instance=False
    )

    reserve_events = [
        event for event in events if type(event).__name__ == "UpdateReserveNumTokens"
    ]
    assert len(reserve_events) == 1
    assert reserve_events[0].request_id == "decode"
    assert reserve_events[0].reserve_num_tokens_in_next_schedule_event == 1


def test_a_landed_forward_of_a_finished_request_still_reports_to_the_scheduler():
    """A request finished on an earlier result while this forward was already
    dispatched (the overlap schedule): its row is untracked here, but the
    scheduler counts the landing and publishes the frontier it consumed, so
    the row's ExtendResult goes out with its tokens -- empty for a chunk that
    owes none."""
    processor = OutputProcesser(_Sender(), attn_tp_rank=0, metrics=_Metrics())
    processor.rid_to_state["prefill"] = _state([1, 2, 3, 4])

    events = processor.post_process_forward_op(
        _ForwardOp(), _ExecutionResult(), is_prefill_instance=False
    )

    extends = {
        event.request_id: list(event.tokens)
        for event in events
        if type(event).__name__ == "ExtendResult"
    }
    assert extends == {"prefill": [11], "decode": [22]}
    assert "decode" not in processor.rid_to_state

    class _MidChunkOp(_ForwardOp):
        request_ids = ["gone"]
        request_pool_indices = [0]
        input_lengths = [4]
        prefill_lengths = [8]

        def num_extends(self):
            return 1

    class _OneRow(_ExecutionResult):
        output_tokens = torch.tensor([11], dtype=torch.int32)
        output_lengths = torch.tensor([1], dtype=torch.int32)

    events = processor.post_process_forward_op(
        _MidChunkOp(), _OneRow(), is_prefill_instance=False
    )
    (extend,) = [event for event in events if type(event).__name__ == "ExtendResult"]
    assert extend.request_id == "gone"
    assert list(extend.tokens) == []


def test_an_extend_sets_its_computed_length_and_a_decode_adds_to_it():
    """An extend computed its prompt up to prefix plus input, whatever the
    length was before (a prefix hit starts past zero), so a prefix hit that
    completes the prompt commits its first token; a decode adds its input."""

    class _PrefixHitOp(_ForwardOp):
        input_lengths = [4, 1]
        prefill_lengths = [8]
        extend_prefix_lens = [4]

    processor = OutputProcesser(_Sender(), attn_tp_rank=0, metrics=_Metrics())
    hit = _state([1, 2, 3, 4, 5, 6, 7, 8])
    decode = _state([5, 6, 7], computed_length=3)
    processor.rid_to_state["prefill"] = hit
    processor.rid_to_state["decode"] = decode

    processor.post_process_forward_op(
        _PrefixHitOp(), _ExecutionResult(), is_prefill_instance=False
    )

    assert (hit.computed_length, hit.output_ids) == (8, [11])
    assert (decode.computed_length, decode.output_ids) == (4, [22])


def test_cached_tokens_count_the_replayed_window_as_a_hit():
    """A bounded-replay chunk starts its model input below the prefix hit;
    the request still hit through the end of the replayed window, and a
    later chunk of the same request adds nothing."""
    processor = OutputProcesser(_Sender(), attn_tp_rank=0, metrics=_Metrics())
    processor.rid_to_state["hit"] = _state(list(range(20)))
    processor.rid_to_state["fresh"] = _state(list(range(5)))
    # "hit": hit at 8, window 4 -> input starts at 4 with 4 replayed rows.
    processor.add_cached_tokens(["hit", "fresh"], [4, 0], [4, 0])
    assert processor.rid_to_state["hit"].cached_tokens == 8
    assert processor.rid_to_state["fresh"].cached_tokens == 0
    processor.rid_to_state["hit"].computed_length = 12
    processor.add_cached_tokens(["hit"], [12], [0])
    assert processor.rid_to_state["hit"].cached_tokens == 8


def test_mark_abort_notify_client_flag():
    """Pause-initiated aborts must flag the request to stream a terminating
    finish to the (passive) client; client-initiated aborts must not."""
    sender = _Sender()
    processor = OutputProcesser(sender, attn_tp_rank=0, metrics=_Metrics())

    pause_state = _state([1, 2, 3])
    processor.rid_to_state["pause"] = pause_state
    processor.mark_abort("pause", notify_client=True)
    assert pause_state.to_abort
    assert pause_state.abort_notify_client
    assert pause_state.finished  # finished_reason materialized

    client_state = _state([1, 2, 3])
    processor.rid_to_state["client"] = client_state
    processor.mark_abort("client")  # default: client tore down its own state
    assert client_state.to_abort
    assert not client_state.abort_notify_client


def test_nan_flag_finishes_request_with_numerical_error():
    """A request flagged by the NaN guard is finished with
    ABORT_CODE.NumericalError while the rest of the batch continues."""
    from tokenspeed.runtime.engine.request_types import ABORT_CODE, FINISH_ABORT

    sender = _Sender()
    metrics = _Metrics()
    processor = OutputProcesser(sender, attn_tp_rank=0, metrics=metrics)
    prefill_state = _state([1, 2, 3, 4])
    decode_state = _state([5, 6, 7], computed_length=3)
    processor.rid_to_state["prefill"] = prefill_state
    processor.rid_to_state["decode"] = decode_state

    result = _ExecutionResult()
    # Flag only the decode slot.
    result.output_nan_flags = torch.tensor([0, 1], dtype=torch.int32)

    events = processor.post_process_forward_op(
        _ForwardOp(), result, is_prefill_instance=False
    )

    # Flagged request: aborted with NumericalError, removed from tracking.
    # The scheduler gets an Abort (NOT Finish) event — AbortEvent skips the
    # single-table-tree insert and host-KV writeback, so corrupted KV is not reused.
    assert isinstance(decode_state.finished_reason, FINISH_ABORT)
    assert decode_state.finished_reason.err_type == ABORT_CODE.NumericalError
    assert "decode" not in processor.rid_to_state
    abort_events = [e for e in events if type(e).__name__ == "Abort"]
    assert [e.request_id for e in abort_events] == ["decode"]
    assert not [e for e in events if type(e).__name__ == "Finish"]
    assert metrics.nan_aborts == 1

    # Unflagged request keeps running untouched.
    assert not prefill_state.finished
    assert "prefill" in processor.rid_to_state
    assert prefill_state.output_ids == [11]

    # The abort finish reason is streamed to the client.
    assert len(sender.items) == 1
    out = sender.items[0]
    idx = out.rids.index("decode")
    assert out.finished_reasons[idx]["type"] == "abort"
    assert out.finished_reasons[idx]["err_type"] == ABORT_CODE.NumericalError.value


def test_nan_flag_keeps_single_sanitized_token():
    """A NaN-flagged spec-decode slot keeps exactly one (sanitized) token so
    extend-result accounting matches a normal mid-step finish."""
    sender = _Sender()
    metrics = _Metrics()
    processor = OutputProcesser(
        sender,
        attn_tp_rank=0,
        spec_algorithm="eagle",
        spec_num_tokens=4,
        metrics=metrics,
    )
    decode_state = _state([5, 6, 7], computed_length=3)
    processor.rid_to_state["decode"] = decode_state

    class _SpecForwardOp:
        request_ids = ["decode"]
        request_pool_indices = [0]
        input_lengths = [1]
        prefill_lengths = []
        extend_prefix_lens = []
        extend_replay_lens = []

        def num_extends(self):
            return 0

    result = _ExecutionResult()
    result.output_tokens = torch.tensor([11, 22, 33, 44], dtype=torch.int32)
    result.output_lengths = torch.tensor([3], dtype=torch.int32)
    result.output_nan_flags = torch.tensor([1], dtype=torch.int32)

    events = processor.post_process_forward_op(
        _SpecForwardOp(), result, is_prefill_instance=False
    )

    assert decode_state.finished
    # Only the first of the 3 accepted tokens is kept.
    assert decode_state.output_ids == [11]
    extend_events = [e for e in events if type(e).__name__ == "ExtendResult"]
    assert len(extend_events) == 1
    assert list(extend_events[0].tokens) == [11]
    assert metrics.nan_aborts == 1


def test_nan_flag_skips_first_token_pd_handoff():
    """A NaN-terminated P-side request must never transfer its KV: it aborts
    at the scheduler (no remote decode is emitted), while healthy requests in
    the same batch proceed."""
    sender = _Sender()
    processor = OutputProcesser(sender, attn_tp_rank=0, metrics=_Metrics())
    processor.rid_to_state["prefill"] = _state([1, 2, 3, 4])
    processor.rid_to_state["decode"] = _state([5, 6, 7], computed_length=3)

    result = _ExecutionResult()
    result.next_input_ids = None
    result.output_nan_flags = torch.tensor([1, 0], dtype=torch.int32)

    events = processor.post_process_forward_op(
        _ForwardOp(),
        result,
        is_prefill_instance=True,
    )

    by_request = [(type(e).__name__, e.request_id) for e in events]
    assert ("Abort", "prefill") in by_request
    assert ("Abort", "decode") not in by_request
    assert "prefill" not in processor.rid_to_state


def test_nan_outranks_a_client_abort_on_a_local_engine():
    """A request that is both client-aborted and NaN-flagged aborts at the
    scheduler rather than finishing: an ending the client asked for does not
    make the numbers trustworthy, and Finish would cache the corrupt pages."""
    sender = _Sender()
    processor = OutputProcesser(sender, attn_tp_rank=0, metrics=_Metrics())
    processor.rid_to_state["decode"] = _state([5, 6, 7], computed_length=3)
    processor.mark_abort("decode", notify_client=False)

    result = _ExecutionResult()
    result.next_input_ids = None
    result.output_nan_flags = torch.tensor([0, 1], dtype=torch.int32)

    events = processor.post_process_forward_op(
        _ForwardOp(),
        result,
        is_prefill_instance=False,
    )

    by_request = [(type(e).__name__, e.request_id) for e in events]
    assert ("Abort", "decode") in by_request
    assert ("Finish", "decode") not in by_request


def test_nan_outranks_a_client_abort_on_the_prefill_node():
    """Numerical corruption must win over every other ending: a request that
    is both client-aborted and NaN-flagged still aborts at the scheduler, so
    its suspect KV is never shipped to the peer."""
    sender = _Sender()
    processor = OutputProcesser(sender, attn_tp_rank=0, metrics=_Metrics())
    processor.rid_to_state["prefill"] = _state([1, 2, 3, 4])
    processor.mark_abort("prefill", notify_client=False)

    result = _PrefillExecutionResult()
    result.output_nan_flags = torch.tensor([1], dtype=torch.int32)

    events = processor.post_process_forward_op(
        _PrefillForwardOp(),
        result,
        is_prefill_instance=True,
    )

    assert [type(event).__name__ for event in events] == ["ExtendResult", "Abort"]
    assert "prefill" not in processor.rid_to_state


class _RecordingLogger:
    """Capture logger.info(fmt, *args) calls as formatted strings."""

    def __init__(self):
        self.lines: list[str] = []

    def info(self, fmt, *args):
        self.lines.append(fmt % args if args else fmt)

    def warning(self, *a, **k):
        pass


def test_log_request_stats_disabled_by_default():
    """Without --enable-log-request-stats, no ReqStats line is emitted and no
    timestamps are recorded (zero overhead path)."""
    import tokenspeed.runtime.engine.generation_output_processor as gop

    rec = _RecordingLogger()
    gop_logger, gop.logger = gop.logger, rec
    try:
        processor = OutputProcesser(_Sender(), attn_tp_rank=0, metrics=_Metrics())
        assert processor.enable_log_request_stats is False
        state = _state([5, 6, 7], computed_length=3)
        state.sampling_params.max_new_tokens = 1
        processor.rid_to_state["d"] = state

        class _DecodeOp:
            request_ids = ["d"]
            request_pool_indices = [0]
            input_lengths = [1]
            prefill_lengths = []
            extend_prefix_lens = []
            extend_replay_lens = []

            def num_extends(self):
                return 0

        processor.post_process_forward_op(
            _DecodeOp(),
            _ExecutionResult(),
            is_prefill_instance=False,
        )
    finally:
        gop.logger = gop_logger

    assert state.finished
    assert not any("RequestStats(" in line for line in rec.lines)
    # disabled: request still carries the shared no-op tracker (never registered)
    assert state.stats is NOOP_STATS


def test_log_request_stats_line_fields():
    """The per-request stats line reports the right host-side derived values:
    queue/prefill/ttft/total ms, cache-hit, decode throughput, preemption."""
    import tokenspeed.runtime.engine.generation_output_processor as gop
    from tokenspeed.runtime.engine.request_types import FINISH_LENGTH

    rec = _RecordingLogger()
    gop_logger, gop.logger = gop.logger, rec
    try:
        processor = OutputProcesser(
            _Sender(), attn_tp_rank=0, enable_log_request_stats=True, metrics=_Metrics()
        )
        # prompt=4, cache=2 -> cache_hit 0.5; queue 10ms, prefill 20ms, ttft 30ms,
        # total 130ms; output=5 over a 100ms decode window -> decode_tps 40.
        rs = _state([1, 2, 3, 4])
        rs.created_time = 1000.000
        rs.cached_tokens = 2
        rs.output_ids = [11, 12, 13, 14, 15]
        rs.finished_reason = FINISH_LENGTH(length=5)
        rs.stats = RequestStatsTracker()
        rs.stats.scheduled_time = 1000.010
        rs.stats.prefill_done_time = 1000.030
        rs.stats.first_token_time = 1000.030
        rs.stats.preempt_count = 2
        rs.stats.preempt_time = 0.005

        processor._log_request_stats("rid-x", rs, finish_time=1000.130)
    finally:
        gop.logger = gop_logger

    assert len(rec.lines) == 1
    line = rec.lines[0]
    assert line.startswith(
        "Req: rid-x Finish! RequestStats(status='finished', reason='length'"
    )
    assert (
        "prompt_tokens=4, cache_tokens=2, output_tokens=5, cache_hit_rate=0.5" in line
    )
    assert "queue_ms=10.0, prefill_ms=20.0, ttft_ms=30.0, total_ms=130.0" in line
    assert "preempt_ms=5.0, preempt_count=2" in line
    assert "decode_tps=40.0" in line
    assert "acc_len=None, acc_rate=None" in line


def test_log_request_stats_aborted_with_spec_acceptance():
    """Aborted requests log status=aborted; with spec decode on, acc_len and
    acc_rate are populated."""
    import tokenspeed.runtime.engine.generation_output_processor as gop
    from tokenspeed.runtime.engine.request_types import FINISH_ABORT

    rec = _RecordingLogger()
    gop_logger, gop.logger = gop.logger, rec
    try:
        processor = OutputProcesser(
            _Sender(),
            attn_tp_rank=0,
            spec_algorithm="eagle",
            spec_num_tokens=4,
            enable_log_request_stats=True,
            metrics=_Metrics(),
        )
        rs = _state([1, 2, 3, 4])
        rs.created_time = 1000.0
        rs.spec_verify_ct = 10
        rs.accept_draft_tokens = 3.0
        rs.finished_reason = FINISH_ABORT("client abort")
        rs.stats = RequestStatsTracker()
        processor._log_request_stats("rid-a", rs, finish_time=1000.05)
    finally:
        gop.logger = gop_logger

    line = rec.lines[0]
    assert "status='aborted', reason='abort'" in line
    # acc_rate = (acc_len - 1) / proposed drafts = (3 - 1) / (4 - 1)
    assert "acc_len=3.0, acc_rate=0.6667" in line


def test_log_request_stats_noop_without_tracker():
    """A request carrying the no-op tracker (flag off / finished-at-admission)
    is skipped by _log_request_stats's single guard, without raising."""
    import tokenspeed.runtime.engine.generation_output_processor as gop
    from tokenspeed.runtime.engine.request_types import FINISH_LENGTH

    rec = _RecordingLogger()
    gop_logger, gop.logger = gop.logger, rec
    try:
        processor = OutputProcesser(
            _Sender(), attn_tp_rank=0, enable_log_request_stats=True, metrics=_Metrics()
        )
        rs = _state([1, 2, 3])
        rs.finished_reason = FINISH_LENGTH(length=1)
        assert rs.stats is NOOP_STATS  # never registered -> no-op tracker
        processor._log_request_stats("no-tracker", rs, finish_time=123.0)
    finally:
        gop.logger = gop_logger
    assert rec.lines == []


def test_request_stats_from_state_total_on_degenerate_input():
    """from_state never divides by zero / reads a missing stage: a request with
    no output and unset timestamps yields zeros and None, not an exception."""
    from tokenspeed.runtime.engine.request_types import FINISH_ABORT

    rs = _state([1, 2, 3, 4])
    rs.finished_reason = FINISH_ABORT("aborted before any output")
    rs.stats = RequestStatsTracker()  # all timestamps still 0.0
    # output_ids empty, no spec decode, no timestamps set.
    stats = RequestStats.from_state(rs, spec_algorithm=None, spec_num_tokens=None)

    assert stats.status == "aborted" and stats.reason == "abort"
    assert stats.output_tokens == 0
    assert stats.cache_hit_rate == 0.0
    assert stats.queue_ms == stats.prefill_ms == stats.ttft_ms == stats.total_ms == 0.0
    assert stats.decode_tps == 0.0
    assert stats.acc_len is None and stats.acc_rate is None


def test_noop_stats_singleton_is_frozen():
    """NOOP_STATS is shared, so its methods are no-ops and writes raise -- a
    future tracker mutator without a no-op override fails loudly, not silently."""
    import pytest

    NOOP_STATS.mark_scheduled(5.0)  # no-op, does not raise or record
    NOOP_STATS.record_decode_step(1.0, True)
    with pytest.raises(AttributeError):
        NOOP_STATS.scheduled_time = 1.0


def test_log_request_stats_records_timestamps_through_forward():
    """End-to-end: with the flag on, a finishing request gets its post-forward
    timestamps recorded host-side and emits one ReqStats line. (scheduled_time
    is stamped pre-forward in the event loop; simulated here.)"""
    import time

    import tokenspeed.runtime.engine.generation_output_processor as gop

    rec = _RecordingLogger()
    gop_logger, gop.logger = gop.logger, rec
    try:
        processor = OutputProcesser(
            _Sender(), attn_tp_rank=0, enable_log_request_stats=True, metrics=_Metrics()
        )
        # prefill already done; max_new_tokens=1 so it finishes after one token
        state = _state([5, 6, 7], computed_length=3)
        state.sampling_params.max_new_tokens = 1
        processor.register("d", state)  # attaches the stats tracker
        state.stats.mark_scheduled(time.time())  # event loop does this pre-forward

        class _DecodeOp:
            request_ids = ["d"]
            request_pool_indices = [0]
            input_lengths = [1]
            prefill_lengths = []
            extend_prefix_lens = []
            extend_replay_lens = []

            def num_extends(self):
                return 0

        processor.post_process_forward_op(
            _DecodeOp(),
            _ExecutionResult(),
            is_prefill_instance=False,
        )
    finally:
        gop.logger = gop_logger

    assert state.finished
    # Lifecycle timestamps were stamped on the host, in order.
    assert state.stats.scheduled_time > 0.0
    assert state.stats.prefill_done_time >= state.stats.scheduled_time
    assert state.stats.first_token_time > 0.0
    assert state.stats.finish_time > 0.0
    stats_lines = [line for line in rec.lines if "Req: d Finish! RequestStats(" in line]
    assert len(stats_lines) == 1
    assert "status='finished', reason='length'" in stats_lines[0]


def test_log_request_stats_logs_on_each_dp_replica_leader():
    """Per-request logging is gated on attn_tp_rank == 0 (each DP replica's TP
    leader), not the global rank. So a request on a DP replica > 0 (whose leader
    has global_rank != 0) is still logged -- not missed -- while non-leader TP
    shards stay silent so the line isn't duplicated."""
    import tokenspeed.runtime.engine.generation_output_processor as gop
    from tokenspeed.runtime.engine.request_types import FINISH_LENGTH

    def emit(attn_tp_rank):
        rec = _RecordingLogger()
        gop_logger, gop.logger = gop.logger, rec
        try:
            p = OutputProcesser(
                _Sender(),
                attn_tp_rank=attn_tp_rank,
                enable_log_request_stats=True,
                metrics=_Metrics(),
            )
            rs = _state([1, 2, 3, 4])
            rs.finished_reason = FINISH_LENGTH(length=1)
            rs.stats = RequestStatsTracker()
            p._log_request_stats("rid", rs, finish_time=1.0)
        finally:
            gop.logger = gop_logger
        return rec.lines

    # TP leader of ANY DP replica logs (attn_tp_rank == 0 even when global_rank != 0).
    assert any("Req: rid Finish! RequestStats(" in line for line in emit(0))
    # Non-leader TP shards within a replica stay silent (no duplicate line).
    assert emit(1) == []


class _PrefillForwardOp:
    request_ids = ["prefill"]
    request_pool_indices = [3]
    input_lengths = [4]
    prefill_lengths = [4]
    extend_prefix_lens = [0]
    extend_replay_lens = [0]

    def num_extends(self):
        return 1


class _PrefillExecutionResult:
    output_tokens = torch.tensor([101], dtype=torch.int32)
    output_lengths = torch.tensor([1], dtype=torch.int32)
    output_logprobs = None
    output_nan_flags = None
    grammar_completion = None
    next_input_ids = torch.tensor([[101, 102, 103]], dtype=torch.int32)
    input_token_logprobs = None
    input_logprob_plan = None
    input_top_logprobs_val = None
    input_top_logprobs_idx = None
    output_top_logprobs_val = None
    output_top_logprobs_idx = None


class _EmptyPrefillExecutionResult(_PrefillExecutionResult):
    output_tokens = torch.tensor([], dtype=torch.int32)
    output_lengths = torch.tensor([0], dtype=torch.int32)


class _MismatchedPrefillExecutionResult(_PrefillExecutionResult):
    next_input_ids = torch.tensor([[201, 202, 203]], dtype=torch.int32)


def test_prefill_final_chunk_folds_spec_candidates_into_the_extend_result():
    """The scheduler's remote decode is self-contained: the drafter candidates
    ride the final chunk's ExtendResult (the bootstrap token is tokens[0],
    which the scheduler reads back as LastToken)."""
    sender = _Sender()
    processor = OutputProcesser(sender, attn_tp_rank=0, metrics=_Metrics())
    processor.rid_to_state["prefill"] = _state([1, 2, 3, 4])

    events = processor.post_process_forward_op(
        _PrefillForwardOp(),
        _PrefillExecutionResult(),
        is_prefill_instance=True,
    )

    extend = next(e for e in events if type(e).__name__ == "ExtendResult")
    assert list(extend.tokens)[0] == 101
    assert list(extend.spec_candidate_ids) == [101, 102, 103]


def test_prefill_extend_result_does_not_guess_from_next_input_ids():
    sender = _Sender()
    processor = OutputProcesser(sender, attn_tp_rank=0, metrics=_Metrics())
    processor.rid_to_state["prefill"] = _state([1, 2, 3, 4])

    events = processor.post_process_forward_op(
        _PrefillForwardOp(),
        _EmptyPrefillExecutionResult(),
        is_prefill_instance=True,
    )

    for event in events:
        if type(event).__name__ == "ExtendResult":
            assert list(event.spec_candidate_ids) == []


def test_prefill_first_token_checks_spec_candidate_bootstrap():
    sender = _Sender()
    processor = OutputProcesser(sender, attn_tp_rank=0, metrics=_Metrics())
    processor.rid_to_state["prefill"] = _state([1, 2, 3, 4])

    with pytest.raises(RuntimeError, match="Prefill bootstrap token mismatch"):
        processor.post_process_forward_op(
            _PrefillForwardOp(),
            _MismatchedPrefillExecutionResult(),
            is_prefill_instance=True,
        )


@pytest.mark.parametrize("notify_client", [False, True])
def test_aborted_prefill_waits_for_pd_transfer_completion(notify_client):
    sender = _Sender()
    processor = OutputProcesser(sender, attn_tp_rank=0, metrics=_Metrics())
    state = _state([1, 2, 3, 4])
    processor.rid_to_state["prefill"] = state
    processor.mark_abort("prefill", notify_client=notify_client)

    events = processor.post_process_forward_op(
        _PrefillForwardOp(),
        _PrefillExecutionResult(),
        is_prefill_instance=True,
    )

    assert [type(event).__name__ for event in events] == ["ExtendResult"]
    assert processor.rid_to_state["prefill"] is state
    assert sender.items == []

    assert processor.finish_prefill_request("prefill") == []
    assert "prefill" not in processor.rid_to_state
    assert len(sender.items) == int(notify_client)


def test_pd_one_token_request_finishes_at_remote_prefill_done():
    sender = _Sender()
    processor = OutputProcesser(sender, attn_tp_rank=0, metrics=_Metrics())
    state = _state([1, 2, 3], computed_length=3)
    state.sampling_params.max_new_tokens = 1
    processor.rid_to_state["decode"] = state

    processor.on_remote_prefill_done("decode", 101, 2, None)
    events = processor.finish_remote_prefill_only_request("decode")

    assert state.output_ids == [101]
    assert state.finished
    assert "decode" not in processor.rid_to_state
    assert [type(event).__name__ for event in events] == ["Finish"]
    assert events[0].request_id == "decode"
    assert len(sender.items) == 1
    output = sender.items[0]
    assert output.rids == ["decode"]
    assert output.output_ids == [[101]]
    assert output.completion_tokens == [1]
    assert output.cached_tokens == [2]
    assert output.finished_reasons[0] == {"type": "length", "length": 1}


class _Matcher:
    """A matcher that records the tokens it was asked to accept."""

    def __init__(self) -> None:
        self.accepted = []

    def accept_token(self, token_id: int) -> None:
        self.accepted.append(token_id)

    def is_terminated(self) -> bool:
        return False


def test_pd_decode_matcher_accepts_the_prefill_nodes_token():
    """The decode node compiled its own matcher; the prefill node's token is
    the one token it never samples, so it must be accepted here or every
    decode step would mask from a state one token behind."""
    processor = OutputProcesser(_Sender(), attn_tp_rank=0, metrics=_Metrics())
    state = _state([1, 2, 3], computed_length=3)
    state.grammar = _Matcher()
    processor.rid_to_state["decode"] = state

    processor.on_remote_prefill_done("decode", 101, 2, None)

    assert state.output_ids == [101]
    assert state.grammar.accepted == [101]


def test_pd_decode_drops_the_grammar_when_the_bootstrap_token_is_lost():
    """Nothing can bring the matcher in sync without that token, and masking
    from a stale state corrupts the output more quietly than not masking."""
    processor = OutputProcesser(_Sender(), attn_tp_rank=0, metrics=_Metrics())
    state = _state([1, 2, 3], computed_length=3)
    state.grammar = _Matcher()
    processor.rid_to_state["decode"] = state

    processor.on_remote_prefill_done("decode", -1, 2, None)

    assert state.output_ids == []
    assert state.grammar is None


def test_pd_multi_token_request_continues_after_remote_prefill_done():
    sender = _Sender()
    processor = OutputProcesser(sender, attn_tp_rank=0, metrics=_Metrics())
    state = _state([1, 2, 3], computed_length=3)
    state.sampling_params.max_new_tokens = 2
    processor.rid_to_state["decode"] = state

    processor.on_remote_prefill_done("decode", 101, 2, None)
    events = processor.finish_remote_prefill_only_request("decode")

    assert state.output_ids == [101]
    assert not state.finished
    assert processor.rid_to_state["decode"] is state
    assert events == []
    assert sender.items == []


class _SpecMetrics:
    enabled = True

    def __init__(self):
        self.steps = []

    def record_spec_decode_step(
        self, *, num_decode_slots, accepted_draft_tokens, draft_width
    ):
        self.steps.append((num_decode_slots, accepted_draft_tokens, draft_width))


def test_spec_decode_metrics_count_mixed_rounds_and_proposed_drafts():
    """A mixed prefill/decode round still records its decode slot, and 3/4
    proposes 3 drafts per verify step because slot 0 is the accepted root."""
    metrics = _SpecMetrics()
    processor = OutputProcesser(
        _Sender(),
        attn_tp_rank=0,
        spec_algorithm="eagle",
        spec_num_tokens=4,
        metrics=metrics,
    )
    processor.rid_to_state["prefill"] = _state([1, 2, 3, 4])
    processor.rid_to_state["decode"] = _state([5, 6, 7], computed_length=3)

    class _SpecResult(_ExecutionResult):
        output_lengths = torch.tensor([1, 3], dtype=torch.int32)

    processor._emit_spec_decode_metrics(
        forward_op=_ForwardOp(), model_execution_results=_SpecResult()
    )
    assert metrics.steps == [(1, 2, 3)]


@pytest.mark.parametrize(
    "local_hits,prefill_hits",
    [(0, 1280), (256, 1280), (1280, 256), (1280, 0)],
)
def test_remote_prefill_usage_reports_prefill_hits(local_hits, prefill_hits):
    sender = _Sender()
    processor = OutputProcesser(sender, attn_tp_rank=0, metrics=_Metrics())
    state = _state(list(range(2048)), computed_length=2048)
    state.sampling_params.max_new_tokens = 1
    state.cached_tokens = local_hits
    processor.rid_to_state["decode"] = state
    processor.on_remote_prefill_done("decode", 101, prefill_hits, None)
    processor.finish_remote_prefill_only_request("decode")
    assert sender.items[0].cached_tokens == [prefill_hits]


def test_non_pd_cached_tokens_reach_output():
    from tokenspeed.runtime.engine.request_types import FINISH_LENGTH

    sender = _Sender()
    processor = OutputProcesser(sender, attn_tp_rank=0, metrics=_Metrics())
    state = _state(list(range(2048)))
    processor.rid_to_state["local"] = state
    processor.add_cached_tokens(["local"], [1024], [256])
    state.output_ids = [101]
    state.finished_reason = FINISH_LENGTH(length=1)
    processor.stream_output(["local"], [state])
    assert sender.items[0].cached_tokens == [1280]


def test_scheduler_aborts_finish_toward_the_client_with_a_capacity_code():
    """A capacity retraction whose victim could not be imaged aborted that
    request inside the plan build: the processor finishes it to the client
    with ``ABORT_CODE.CapacityAbort`` and the scheduler's detail, counts it,
    returns no event (the scheduler already dropped it) and leaves the other
    requests alone; an id the client aborted in the same round is skipped."""
    from types import SimpleNamespace

    from tokenspeed.runtime.engine.request_types import ABORT_CODE

    sender = _Sender()
    metrics = _Metrics()
    processor = OutputProcesser(sender, attn_tp_rank=0, metrics=metrics)
    victim = _state([1, 2, 3], computed_length=3)
    victim.output_ids = [41, 42]
    processor.rid_to_state["victim"] = victim
    processor.rid_to_state["other"] = _state([4, 5, 6], computed_length=3)
    aborts = [
        SimpleNamespace(
            request_id="victim",
            reason="ImageDoesNotFit",
            detail="no blob slot free (max_retracted_requests=2); raise "
            "--retraction-snapshot-max-requests",
        ),
        SimpleNamespace(request_id="gone", reason="ImageDoesNotFit", detail="x"),
    ]

    assert processor.finish_scheduler_aborted_requests(aborts) is None

    assert set(processor.rid_to_state) == {"other"}
    assert victim.finished and victim.finished_reason.is_error
    reason = victim.finished_reason.to_json()
    assert reason["type"] == "abort"
    assert reason["err_type"] == ABORT_CODE.CapacityAbort.value == 524
    assert "max_retracted_requests=2" in reason["message"]
    assert metrics.capacity_aborts == 1
    (out,) = sender.items
    assert out.rids == ["victim"] and out.finished_reasons[0] == reason


def test_a_client_abort_crossing_a_capacity_abort_keeps_the_clients_finish():
    """The scheduler's abort of a request the client already cancelled in the
    same round: the client's reason stands, no capacity abort is counted and
    nothing is streamed (the tokenizer manager tore down) -- unless a
    pause-initiated abort left a passive client that still needs its finish.
    Either way the state is released: the scheduler emits nothing further."""
    from types import SimpleNamespace

    sender = _Sender()
    metrics = _Metrics()
    processor = OutputProcesser(sender, attn_tp_rank=0, metrics=metrics)
    cancelled = _state([1, 2, 3], computed_length=3)
    paused = _state([4, 5, 6], computed_length=3)
    processor.rid_to_state["cancelled"] = cancelled
    processor.rid_to_state["paused"] = paused
    processor.mark_abort("cancelled")
    processor.mark_abort("paused", notify_client=True)
    aborts = [
        SimpleNamespace(request_id=rid, reason="ImageDoesNotFit", detail="x")
        for rid in ("cancelled", "paused")
    ]

    processor.finish_scheduler_aborted_requests(aborts)

    assert processor.rid_to_state == {}
    assert metrics.capacity_aborts == 0
    assert cancelled.finished_reason.message == "AbortReq from client"
    assert paused.finished_reason.message == "Aborted by pause"
    (out,) = sender.items
    assert out.rids == ["paused"]
    assert out.finished_reasons[0] == paused.finished_reason.to_json()


class _EosTokenizer(_Tokenizer):
    eos_token_id = 3
    additional_stop_token_ids = {4}


def test_stop_token_ids_merge_every_source_unless_eos_is_ignored():
    """The finish check and the device's verify cut read one stop set: the
    request's own ids, the model's EOS ids and the tokenizer's."""
    state = RequestState(
        prompt_input_ids=[1, 2, 3, 4, 5],
        sampling_params=SamplingParams(max_new_tokens=8, stop=[], stop_token_ids={5}),
        stream=False,
        tokenizer=_EosTokenizer(),
        computes_prompt_logprobs=True,
        eos_token_ids=[2],
    )
    assert state.stop_token_ids == {2, 3, 4, 5}
    assert state.stop_criteria.stop_token_ids == (2, 3, 4, 5)
    assert state.stop_criteria.token_limit == 5 + 8
    state.output_ids = [4]
    state.check_finished()
    assert state.finished_reason.to_json()["matched"] == 4

    ignoring = RequestState(
        prompt_input_ids=[1],
        sampling_params=SamplingParams(
            max_new_tokens=8, stop=[], stop_token_ids={5}, ignore_eos=True
        ),
        stream=False,
        tokenizer=_EosTokenizer(),
        computes_prompt_logprobs=True,
        eos_token_ids=[2],
    )
    assert ignoring.stop_criteria.stop_token_ids == ()


class _DecodeOp:
    request_ids = ["decode"]
    request_pool_indices = [1]
    input_lengths = [3]
    prefill_lengths = []
    extend_prefix_lens = []
    extend_replay_lens = []

    def num_extends(self):
        return 0


class _DecodeResult(_ExecutionResult):
    output_tokens = torch.tensor([11, 12, 13], dtype=torch.int32)
    output_lengths = torch.tensor([3], dtype=torch.int32)


def test_a_finish_before_the_last_kept_token_reports_the_whole_round():
    """A stop string ends the request on the round's first token, but the
    device kept all three (it cuts only at stop tokens and the token limit):
    the client sees the first, the scheduler gets every kept token so the
    frontier it publishes matches the state and KV the device holds."""
    processor = OutputProcesser(
        _Sender(),
        attn_tp_rank=0,
        metrics=_Metrics(),
        spec_algorithm="EAGLE",
        spec_num_tokens=3,
    )
    state = RequestState(
        prompt_input_ids=[5, 6, 7],
        sampling_params=SamplingParams(max_new_tokens=8, stop=["11"]),
        stream=False,
        tokenizer=_Tokenizer(),
        computes_prompt_logprobs=True,
    )
    state.computed_length = 3
    processor.rid_to_state["decode"] = state

    events = processor.post_process_forward_op(
        _DecodeOp(), _DecodeResult(), is_prefill_instance=False
    )

    (extend,) = [e for e in events if type(e).__name__ == "ExtendResult"]
    assert list(extend.tokens) == [11, 12, 13]
    assert state.output_ids == [11]
    assert any(type(e).__name__ == "Finish" for e in events)


def test_a_live_decode_that_keeps_no_token_is_an_error():
    """Only a request the device already stopped keeps nothing, and the
    processor no longer tracks a finished request."""

    class _Empty(_DecodeResult):
        output_lengths = torch.tensor([0], dtype=torch.int32)

    processor = OutputProcesser(
        _Sender(),
        attn_tp_rank=0,
        metrics=_Metrics(),
        spec_algorithm="EAGLE",
        spec_num_tokens=3,
    )
    processor.rid_to_state["decode"] = _state([5, 6, 7], computed_length=3)
    with pytest.raises(RuntimeError, match="kept no token"):
        processor.post_process_forward_op(
            _DecodeOp(), _Empty(), is_prefill_instance=False
        )


def test_a_request_with_more_stop_tokens_than_the_device_table_is_refused():
    def state(num_ids):
        return RequestState(
            prompt_input_ids=[1],
            sampling_params=SamplingParams(
                max_new_tokens=8, stop=[], stop_token_ids=set(range(100, 100 + num_ids))
            ),
            stream=False,
            tokenizer=_EosTokenizer(),
            computes_prompt_logprobs=True,
        )

    handler = object.__new__(RequestHandler)
    # The tokenizer's two stop ids count against the limit.
    fits = state(MAX_STOP_TOKEN_IDS - 2)
    handler._refuse_too_many_stop_token_ids(fits)
    assert not fits.finished
    refused = state(MAX_STOP_TOKEN_IDS - 1)
    handler._refuse_too_many_stop_token_ids(refused)
    reason = refused.finished_reason.to_json()
    assert reason["status_code"] == HTTPStatus.BAD_REQUEST
    assert f"at most {MAX_STOP_TOKEN_IDS - 2} other" in reason["message"]


def test_a_row_behind_a_stopped_request_reads_its_own_token():
    """A request the device stopped keeps no token, but its decode row still
    holds its stored token: the next row's token is at the next position, so
    a live request never takes its neighbour's token (and so never misses
    its own stop token)."""

    class _TwoDecodes:
        request_ids = ["stopped", "live"]
        request_pool_indices = [1, 2]
        input_lengths = [1, 1]
        prefill_lengths = []
        extend_prefix_lens = []
        extend_replay_lens = []

        def num_extends(self):
            return 0

    class _Result(_ExecutionResult):
        output_tokens = torch.tensor([11, 22], dtype=torch.int32)
        output_lengths = torch.tensor([0, 1], dtype=torch.int32)

    processor = OutputProcesser(_Sender(), attn_tp_rank=0, metrics=_Metrics())
    live = _state([5, 6, 7], computed_length=3)
    processor.rid_to_state["live"] = live

    events = processor.post_process_forward_op(
        _TwoDecodes(), _Result(), is_prefill_instance=False
    )

    extends = {
        event.request_id: list(event.tokens)
        for event in events
        if type(event).__name__ == "ExtendResult"
    }
    assert extends == {"stopped": [], "live": [22]}
    assert live.output_ids == [22]


class _LetterTokenizer:
    """A tokenizer with an EOS id and an extra stop id; decodes to letters."""

    eos_token_id = 900
    additional_stop_token_ids = [901]

    def decode(self, ids):
        return "".join(chr(ord("a") + i % 26) for i in ids)


_MODEL_EOS = 902


def _stop_state(prompt: list[int], **params) -> RequestState:
    stream = params.pop("stream", False)
    return_logprob = params.pop("return_logprob", False)
    sampling = SamplingParams(**{"max_new_tokens": 30, "stop": [], **params})
    sampling.normalize(None)
    state = RequestState(
        prompt_input_ids=prompt,
        sampling_params=sampling,
        stream=stream,
        tokenizer=_LetterTokenizer(),
        computes_prompt_logprobs=True,
        eos_token_ids=[_MODEL_EOS],
        return_logprob=return_logprob,
    )
    state.computed_length = len(prompt)
    return state


class _DecodeRound:
    """One decode round over ``request_ids`` (the scheduler op's surface)."""

    def __init__(self, request_ids: list[str], width: int) -> None:
        self.request_ids = list(request_ids)
        self.request_pool_indices = list(range(len(request_ids)))
        self.input_lengths = [width] * len(request_ids)
        self.prefill_lengths = []
        self.extend_prefix_lens = []
        self.extend_replay_lens = []

    def num_extends(self) -> int:
        return 0


def test_streamed_frames_carry_the_detokenizer_ids_once():
    """Across a streaming request's frames the decode ids are the prompt's
    surrounding tail and then each output token exactly once, against a
    fixed read offset."""
    processor = OutputProcesser(
        _Sender(),
        attn_tp_rank=0,
        spec_algorithm="EAGLE",
        spec_num_tokens=3,
        metrics=_Metrics(),
    )
    prompt = [10, 11, 12, 13, 14, 15, 16, 17]
    state = _stop_state(prompt, stream=True)
    processor.rid_to_state["s"] = state

    for rows, accepted in (
        [[101, 102, 103], 2],
        [[104, 105, 106], 3],
        [[107, 0, 0], 1],
    ):
        processor.post_process_forward_op(
            _DecodeRound(["s"], 3),
            ModelExecutionResult(
                output_tokens=torch.tensor(rows, dtype=torch.int32),
                output_lengths=torch.tensor([accepted], dtype=torch.int32),
            ),
            is_prefill_instance=False,
        )

    frames = processor.send_to_tokenizer.items
    assert len(frames) == 3
    sent = [i for frame in frames for i in frame.decode_ids[0]]
    assert sent == prompt[3:] + [101, 102, 104, 105, 106, 107]
    assert [frame.read_offsets[0] for frame in frames] == [5, 5, 5]
    assert [frame.output_ids[0] for frame in frames] == [
        [101, 102],
        [104, 105, 106],
        [107],
    ]
