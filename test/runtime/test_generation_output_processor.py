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

import pytest
import torch

from tokenspeed.runtime.engine.generation_output_processor import (
    OutputProcesser,
    RequestState,
)
from tokenspeed.runtime.engine.request_stats import (
    NOOP_STATS,
    RequestStats,
    RequestStatsTracker,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams


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

    def record_nan_abort(self):
        self.nan_aborts += 1


class _ForwardOp:
    request_ids = ["prefill", "decode"]
    request_pool_indices = [0, 1]
    input_lengths = [4, 1]
    extend_prefix_lens = [0]

    def num_extends(self):
        return 1


class _ExecutionResult:
    output_tokens = torch.tensor([11, 22], dtype=torch.int32)
    output_lengths = torch.tensor([1, 1], dtype=torch.int32)
    output_logprobs = None
    output_nan_flags = None
    grammar_completion = None
    next_input_ids = None

    def sync(self):
        return None


def _state(input_ids: list[int], *, computed_length: int = 0) -> RequestState:
    state = RequestState(
        prompt_input_ids=input_ids,
        sampling_params=SamplingParams(max_new_tokens=8, stop=[], ignore_eos=True),
        stream=False,
        tokenizer=_Tokenizer(),
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

    events = processor.post_process_forward_op(_ForwardOp(), _ExecutionResult())

    reserve_events = [
        event for event in events if type(event).__name__ == "UpdateReserveNumTokens"
    ]
    assert len(reserve_events) == 1
    assert reserve_events[0].request_id == "decode"
    assert reserve_events[0].reserve_num_tokens_in_next_schedule_event == 1


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


def test_scheduler_abort_publishes_once_and_releases_request_state():
    from tokenspeed.runtime.engine.request_types import FINISH_ABORT

    sender = _Sender()
    processor = OutputProcesser(sender, attn_tp_rank=0, metrics=_Metrics())
    state = _state([1, 2, 3])
    processor.rid_to_state["r"] = state
    processor.pending_aborts["r"] = 0.0

    processor.publish_scheduler_abort("r", "paged cache exhausted")
    processor.publish_scheduler_abort("r", "duplicate")

    assert isinstance(state.finished_reason, FINISH_ABORT)
    assert state.finished_reason.message == "paged cache exhausted"
    assert "r" not in processor.rid_to_state
    assert "r" not in processor.pending_aborts
    assert len(sender.items) == 1
    assert sender.items[0].rids == ["r"]


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

    events = processor.post_process_forward_op(_ForwardOp(), result)

    # Flagged request: aborted with NumericalError, removed from tracking.
    # The scheduler gets an Abort (NOT Finish) event — AbortEvent skips the
    # radix-tree insert and host-KV writeback, so corrupted KV is not reused.
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
        extend_prefix_lens = []

        def num_extends(self):
            return 0

    result = _ExecutionResult()
    result.output_tokens = torch.tensor([11, 22, 33, 44], dtype=torch.int32)
    result.output_lengths = torch.tensor([3], dtype=torch.int32)
    result.output_nan_flags = torch.tensor([1], dtype=torch.int32)

    events = processor.post_process_forward_op(_SpecForwardOp(), result)

    assert decode_state.finished
    # Only the first of the 3 accepted tokens is kept.
    assert decode_state.output_ids == [11]
    extend_events = [e for e in events if type(e).__name__ == "ExtendResult"]
    assert len(extend_events) == 1
    assert list(extend_events[0].tokens) == [11]
    assert metrics.nan_aborts == 1


def test_nan_flag_skips_first_token_pd_handoff():
    """NaN-terminated requests must not hand their bootstrap token to the PD
    transfer layer — their KV is suspect."""
    sender = _Sender()
    processor = OutputProcesser(sender, attn_tp_rank=0, metrics=_Metrics())
    processor.rid_to_state["prefill"] = _state([1, 2, 3, 4])
    processor.rid_to_state["decode"] = _state([5, 6, 7], computed_length=3)

    result = _ExecutionResult()
    result.next_input_ids = None
    result.output_nan_flags = torch.tensor([1, 0], dtype=torch.int32)

    handoffs = []
    processor.post_process_forward_op(
        _ForwardOp(),
        result,
        on_first_token=lambda rid, *a: handoffs.append(rid),
    )

    # Flagged prefill slot is skipped; the healthy decode slot still hands off.
    assert handoffs == ["decode"]


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
            extend_prefix_lens = []

            def num_extends(self):
                return 0

        processor.post_process_forward_op(_DecodeOp(), _ExecutionResult())
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
    # acc_rate = (acc_len - 1) / draft = (3 - 1) / 4 = 0.5
    assert "acc_len=3.0, acc_rate=0.5" in line


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
            extend_prefix_lens = []

            def num_extends(self):
                return 0

        processor.post_process_forward_op(_DecodeOp(), _ExecutionResult())
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
    extend_prefix_lens = [0]

    def num_extends(self):
        return 1


class _PrefillExecutionResult:
    output_tokens = torch.tensor([101], dtype=torch.int32)
    output_lengths = torch.tensor([1], dtype=torch.int32)
    output_logprobs = None
    output_nan_flags = None
    grammar_completion = None
    next_input_ids = torch.tensor([[101, 102, 103]], dtype=torch.int32)

    def sync(self):
        return None


class _EmptyPrefillExecutionResult(_PrefillExecutionResult):
    output_tokens = torch.tensor([], dtype=torch.int32)
    output_lengths = torch.tensor([0], dtype=torch.int32)


class _MismatchedPrefillExecutionResult(_PrefillExecutionResult):
    next_input_ids = torch.tensor([[201, 202, 203]], dtype=torch.int32)


def test_prefill_first_token_passes_spec_candidates():
    sender = _Sender()
    processor = OutputProcesser(sender, attn_tp_rank=0, metrics=_Metrics())
    processor.rid_to_state["prefill"] = _state([1, 2, 3, 4])
    calls = []

    processor.post_process_forward_op(
        _PrefillForwardOp(),
        _PrefillExecutionResult(),
        is_prefill_instance=True,
        on_first_token=lambda *args: calls.append(args),
    )

    assert calls == [("prefill", 3, 101, [101, 102, 103])]


def test_prefill_first_token_does_not_guess_from_next_input_ids():
    sender = _Sender()
    processor = OutputProcesser(sender, attn_tp_rank=0, metrics=_Metrics())
    processor.rid_to_state["prefill"] = _state([1, 2, 3, 4])
    calls = []

    processor.post_process_forward_op(
        _PrefillForwardOp(),
        _EmptyPrefillExecutionResult(),
        is_prefill_instance=True,
        on_first_token=lambda *args: calls.append(args),
    )

    assert calls == []


def test_prefill_first_token_checks_spec_candidate_bootstrap():
    sender = _Sender()
    processor = OutputProcesser(sender, attn_tp_rank=0, metrics=_Metrics())
    processor.rid_to_state["prefill"] = _state([1, 2, 3, 4])

    with pytest.raises(RuntimeError, match="Prefill bootstrap token mismatch"):
        processor.post_process_forward_op(
            _PrefillForwardOp(),
            _MismatchedPrefillExecutionResult(),
            is_prefill_instance=True,
            on_first_token=lambda *args: None,
        )
