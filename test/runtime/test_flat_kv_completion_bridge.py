# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY.

"""Orthogonal Python contracts for executor-fenced flat KV completion."""

from __future__ import annotations

from dataclasses import fields

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenspeed_scheduler")
scheduler_utils = pytest.importorskip("tokenspeed.runtime.engine.scheduler_utils")
_output = pytest.importorskip("tokenspeed.runtime.engine.generation_output_processor")
_types = pytest.importorskip("tokenspeed.runtime.execution.types")

OutputProcesser = _output.OutputProcesser
RequestState = _output.RequestState
ModelExecutionResult = _types.ModelExecutionResult
SamplingParams = pytest.importorskip(
    "tokenspeed.runtime.sampling.sampling_params"
).SamplingParams


class _Fence:
    def __init__(self) -> None:
        self.calls = 0

    def synchronize(self) -> None:
        self.calls += 1


def _dispatch(
    *,
    generation: int,
    sequence: int,
    start: int,
    end: int,
) -> dict[str, int]:
    return {
        "table_generation": generation,
        "dispatch_seq": sequence,
        "dispatch_raw_start": start,
        "dispatch_raw_end": end,
        "protected_raw_end": end + 4,
    }


class _MixedForwardOp:
    request_ids = ["prefill", "decode"]
    request_pool_indices = [0, 1]
    input_lengths = [4, 4]
    extend_prefix_lens = [4]
    flat_kv_completion_inputs = [
        _dispatch(generation=7, sequence=11, start=4, end=8),
        _dispatch(generation=9, sequence=20, start=10, end=14),
    ]

    @staticmethod
    def num_extends() -> int:
        return 1


def _result(output_lengths=(1, 3)) -> tuple[ModelExecutionResult, _Fence]:
    fence = _Fence()
    return (
        ModelExecutionResult(
            output_tokens=torch.tensor([101, 201, 202, 203], dtype=torch.int32),
            output_lengths=torch.tensor(output_lengths, dtype=torch.int32),
            copy_event=fence,
        ),
        fence,
    )


def test_completion_is_unavailable_before_result_sync() -> None:
    result, fence = _result()

    with pytest.raises(RuntimeError, match="ready only after.*sync"):
        result.materialize_flat_kv_completions(_MixedForwardOp())

    assert result.flat_kv_completions is None
    assert fence.calls == 0


def test_synced_completion_is_exactly_three_host_int_fields() -> None:
    result, fence = _result()
    result.sync()

    completions = result.materialize_flat_kv_completions(_MixedForwardOp())

    assert fence.calls == 1
    assert [item.accepted_raw_end for item in completions] == [8, 13]
    assert tuple(field.name for field in fields(type(completions[0]))) == (
        "table_generation",
        "dispatch_seq",
        "accepted_raw_end",
    )
    assert all(
        type(getattr(item, field.name)) is int
        for item in completions
        for field in fields(item)
    )


class _BoundCompletion:
    pass


class _BoundExtendResult:
    def __init__(self) -> None:
        self.flat_kv_completion = None


class _BoundForwardEvent:
    FlatKVCompletion = _BoundCompletion
    ExtendResult = _BoundExtendResult


def test_extend_event_binds_ready_completion(monkeypatch) -> None:
    result, _ = _result()
    result.sync()
    completion = result.materialize_flat_kv_completions(_MixedForwardOp())[1]
    monkeypatch.setattr(scheduler_utils, "ForwardEvent", _BoundForwardEvent)

    event = scheduler_utils.make_extend_result_event(
        "decode", (), flat_kv_completion=completion
    )

    assert isinstance(event.flat_kv_completion, _BoundCompletion)
    assert (
        event.flat_kv_completion.table_generation,
        event.flat_kv_completion.dispatch_seq,
        event.flat_kv_completion.accepted_raw_end,
    ) == (9, 20, 13)


class _DecodeForwardOp:
    request_ids = ["decode"]
    request_pool_indices = [0]
    input_lengths = [4]
    extend_prefix_lens = []
    flat_kv_completion_inputs = [_dispatch(generation=13, sequence=21, start=4, end=8)]

    @staticmethod
    def num_extends() -> int:
        return 0


class _Tokenizer:
    eos_token_id = None
    additional_stop_token_ids = None

    @staticmethod
    def decode(ids) -> str:
        return "".join(str(value) for value in ids)


class _Sender:
    def send_pyobj(self, obj) -> None:
        del obj


class _Metrics:
    enabled = False

    @staticmethod
    def record_nan_abort() -> None:
        return None


def test_nan_abort_precedes_ready_completion(monkeypatch) -> None:
    events = []

    def extend(request_id, tokens=(), *, flat_kv_completion=None):
        event = ("extend", request_id, tuple(tokens), flat_kv_completion)
        events.append(event)
        return event

    def abort(request_id):
        event = ("abort", request_id)
        events.append(event)
        return event

    monkeypatch.setattr(_output, "make_extend_result_event", extend)
    monkeypatch.setattr(_output, "make_abort_event", abort)

    processor = OutputProcesser(
        _Sender(),
        attn_tp_rank=0,
        spec_algorithm="eagle",
        spec_num_tokens=4,
        metrics=_Metrics(),
    )
    state = RequestState(
        prompt_input_ids=[1, 2, 3, 4, 5],
        sampling_params=SamplingParams(
            max_new_tokens=8,
            stop=[],
            ignore_eos=True,
        ),
        stream=False,
        tokenizer=_Tokenizer(),
    )
    state.computed_length = 5
    processor.rid_to_state["decode"] = state
    result, fence = _result(output_lengths=(4,))
    result.output_nan_flags = torch.tensor([1], dtype=torch.int32)

    changes = processor.post_process_forward_op(_DecodeForwardOp(), result)

    assert fence.calls == 1
    assert [event[0] for event in changes[:2]] == ["abort", "extend"]
    completion = changes[1][3]
    assert (completion.table_generation, completion.dispatch_seq) == (13, 21)
    assert completion.accepted_raw_end == 5
