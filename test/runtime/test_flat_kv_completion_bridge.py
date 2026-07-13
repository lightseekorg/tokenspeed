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

"""Ready-only Python bridge tests for flat KV producer completions."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
from dataclasses import fields, is_dataclass

import pytest


class _FakeScalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _FakeTensor:
    def __init__(self, values, *, dtype=None):
        if isinstance(values, (tuple, list)):
            self._values = list(values)
        else:
            self._values = [values]
        self.dtype = dtype

    def __getitem__(self, index):
        value = self._values[index]
        if isinstance(index, slice):
            return _FakeTensor(value, dtype=self.dtype)
        return _FakeScalar(value)

    def __len__(self):
        return len(self._values)

    def tolist(self):
        return list(self._values)


def _install_fake_torch():
    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = _FakeTensor
    fake_torch.int32 = object()
    fake_torch.tensor = lambda values, dtype=None: _FakeTensor(values, dtype=dtype)
    fake_torch.is_tensor = lambda value: isinstance(value, _FakeTensor)
    sys.modules["torch"] = fake_torch
    return fake_torch


try:
    import torch
except ModuleNotFoundError:
    torch = _install_fake_torch()
    _HAS_RUNTIME_BRIDGE = False
else:
    _HAS_RUNTIME_BRIDGE = True

if _HAS_RUNTIME_BRIDGE:
    from tokenspeed.runtime.engine import scheduler_utils
    from tokenspeed.runtime.engine.generation_output_processor import (
        OutputProcesser,
        RequestState,
    )
    from tokenspeed.runtime.execution.types import (
        FlatKVCompletionInput,
        FlatKVGroupCompletion,
        ModelExecutionResult,
    )
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams
else:
    scheduler_utils = None
    OutputProcesser = None
    RequestState = None
    SamplingParams = None
    execution_types_path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "python"
        / "tokenspeed"
        / "runtime"
        / "execution"
        / "types.py"
    )
    execution_types_spec = importlib.util.spec_from_file_location(
        "flat_kv_completion_execution_types_test",
        execution_types_path,
    )
    assert execution_types_spec is not None and execution_types_spec.loader is not None
    execution_types = importlib.util.module_from_spec(execution_types_spec)
    sys.modules[execution_types_spec.name] = execution_types
    execution_types_spec.loader.exec_module(execution_types)
    FlatKVCompletionInput = execution_types.FlatKVCompletionInput
    FlatKVGroupCompletion = execution_types.FlatKVGroupCompletion
    ModelExecutionResult = execution_types.ModelExecutionResult
    # Keep the fallback local to this module even under whole-suite collection.
    sys.modules.pop("torch", None)

requires_runtime_bridge = pytest.mark.skipif(
    not _HAS_RUNTIME_BRIDGE,
    reason="full scheduler/output bridge requires torch",
)


class _Fence:
    def __init__(self) -> None:
        self.calls = 0

    def synchronize(self) -> None:
        self.calls += 1


class _FailingFence:
    @staticmethod
    def synchronize() -> None:
        raise RuntimeError("fence failed")


def _completion_input(
    *,
    request_id: str,
    generation: int,
    seq: int,
    dispatch_end: int,
    protected_end: int,
    dispatch_start: int | None = None,
):
    if dispatch_start is None:
        dispatch_start = dispatch_end - 4
    return {
        "request_id": request_id,
        "table_generation": generation,
        "dispatch_seq": seq,
        "dispatch_raw_start": dispatch_start,
        "dispatch_raw_end": dispatch_end,
        "protected_raw_end": protected_end,
    }


class _MixedForwardOp:
    request_ids = ["mid", "decode"]
    request_pool_indices = [0, 1]
    input_lengths = [4, 4]
    extend_prefix_lens = [4]
    flat_kv_completion_inputs = [
        _completion_input(
            request_id="mid",
            generation=7,
            seq=11,
            dispatch_end=8,
            protected_end=12,
        ),
        _completion_input(
            request_id="decode",
            generation=9,
            seq=20,
            dispatch_end=14,
            protected_end=20,
        ),
    ]

    @staticmethod
    def num_extends() -> int:
        return 1


class _ExecutionEvidence:
    def __init__(self, completion_inputs=None):
        raw_inputs = (
            _MixedForwardOp.flat_kv_completion_inputs
            if completion_inputs is None
            else completion_inputs
        )
        self.inputs = tuple(FlatKVCompletionInput.from_raw(item) for item in raw_inputs)

    def materialize_group_completions(self, accepted_raw_ends):
        assert len(self.inputs) == len(accepted_raw_ends)
        return tuple(
            (
                FlatKVGroupCompletion(
                    group_id="v4.c4a.compressed_kv",
                    completed_domain_mask=0b11,
                    domain_valid_ends=(item.dispatch_raw_end,) * 2,
                ),
                FlatKVGroupCompletion(
                    group_id="v4.swa_kv",
                    completed_domain_mask=0b11,
                    domain_valid_ends=(item.dispatch_raw_end,) * 2,
                ),
            )
            for item in self.inputs
        )


def _result(output_lengths=(1, 3)) -> tuple[ModelExecutionResult, _Fence]:
    fence = _Fence()
    result = ModelExecutionResult(
        output_tokens=torch.tensor([101, 201, 202, 203], dtype=torch.int32),
        output_lengths=torch.tensor(output_lengths, dtype=torch.int32),
        copy_event=fence,
        _flat_kv_execution_evidence=_ExecutionEvidence(),
    )
    return result, fence


def _assert_host_pod(value) -> None:
    """Recursively reject tensors, CUDA events, pointers, and arbitrary objects."""
    assert not torch.is_tensor(value)
    if value is None or type(value) in (str, int):
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _assert_host_pod(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_host_pod(key)
            _assert_host_pod(item)
        return
    if is_dataclass(value):
        for data_field in fields(value):
            _assert_host_pod(getattr(value, data_field.name))
        return
    pytest.fail(f"non-POD completion member: {type(value).__name__}")


def test_completion_cannot_materialize_before_result_sync():
    result, fence = _result()

    with pytest.raises(RuntimeError, match="ready only after.*sync"):
        result.materialize_flat_kv_completions(_MixedForwardOp())

    assert result.flat_kv_completions is None
    assert fence.calls == 0


def test_failed_fence_does_not_mark_completion_ready():
    result, _ = _result()
    result.copy_event = _FailingFence()

    with pytest.raises(RuntimeError, match="fence failed"):
        result.sync()
    with pytest.raises(RuntimeError, match="ready only after.*sync"):
        result.materialize_flat_kv_completions(_MixedForwardOp())

    assert not result.is_synchronized
    assert result.flat_kv_completions is None


def test_synced_result_materializes_prefill_and_decode_accept_ends_as_pod():
    result, fence = _result()
    result.sync()

    completions = result.materialize_flat_kv_completions(_MixedForwardOp())

    assert fence.calls == 1
    assert result.is_synchronized
    assert [completion.request_id for completion in completions] == ["mid", "decode"]
    # Extend rows accept all KV-writing input tokens, not their garbage/sample
    # output length. Decode/spec rows use the synchronized accepted count.
    assert [completion.accepted_raw_end for completion in completions] == [8, 13]
    assert [completion.dispatch_seq for completion in completions] == [11, 20]
    # Producer progress remains the true dispatch end. C++ owns ordered
    # accepted-end clamping and successor invalidation. The input carries no
    # group schema or producer end to echo.
    assert completions[1].groups[0].domain_valid_ends == (14, 14)
    for completion in completions:
        _assert_host_pod(completion)
        _assert_host_pod(completion.to_pod())
    assert set(completions[0].to_pod()) == {
        "request_id",
        "table_generation",
        "dispatch_seq",
        "accepted_raw_end",
        "protected_raw_end",
        "groups",
    }


def test_completion_rewinds_device_acceptance_to_host_terminal_prefix():
    result, _ = _result(output_lengths=(1, 4))
    result.sync()
    completion = result.materialize_flat_kv_completions(_MixedForwardOp())[1]

    rewound = completion.rewind_to_host_accepted_tokens(
        device_accepted_tokens=4,
        host_accepted_tokens=1,
    )

    assert completion.accepted_raw_end == 14
    assert rewound.accepted_raw_end == 11
    assert rewound.protected_raw_end == completion.protected_raw_end
    assert rewound.groups == completion.groups
    with pytest.raises(ValueError, match="host_accepted_tokens must be <= 4"):
        completion.rewind_to_host_accepted_tokens(
            device_accepted_tokens=4,
            host_accepted_tokens=5,
        )


class _BoundFlatKVGroupCompletion:
    pass


class _BoundFlatKVCompletion:
    pass


class _BoundExtendResult:
    def __init__(self) -> None:
        self.flat_kv_completion = None


class _BoundForwardEvent:
    FlatKVGroupCompletion = _BoundFlatKVGroupCompletion
    FlatKVCompletion = _BoundFlatKVCompletion
    ExtendResult = _BoundExtendResult


@requires_runtime_bridge
def test_make_extend_result_event_explicitly_binds_ready_completion(monkeypatch):
    result, _ = _result()
    result.sync()
    completion = result.materialize_flat_kv_completions(_MixedForwardOp())[1]
    monkeypatch.setattr(scheduler_utils, "ForwardEvent", _BoundForwardEvent)

    event = scheduler_utils.make_extend_result_event(
        "decode", (), flat_kv_completion=completion
    )

    assert event.request_id == "decode"
    assert event.tokens == []
    assert isinstance(event.flat_kv_completion, _BoundFlatKVCompletion)
    bound = event.flat_kv_completion
    assert bound.request_id == "decode"
    assert bound.table_generation == 9
    assert bound.dispatch_seq == 20
    assert bound.accepted_raw_end == 13
    assert bound.protected_raw_end == 20
    assert len(bound.groups) == 2
    assert all(isinstance(group, _BoundFlatKVGroupCompletion) for group in bound.groups)
    assert bound.groups[0].group_id == "v4.c4a.compressed_kv"
    assert bound.groups[0].completed_domain_mask == 0b11
    assert bound.groups[0].domain_valid_ends == [14, 14]


@requires_runtime_bridge
def test_make_extend_result_event_rejects_mismatched_completion_request(monkeypatch):
    result, _ = _result()
    result.sync()
    completion = result.materialize_flat_kv_completions(_MixedForwardOp())[1]
    monkeypatch.setattr(scheduler_utils, "ForwardEvent", _BoundForwardEvent)

    with pytest.raises(ValueError, match="request_id differs"):
        scheduler_utils.make_extend_result_event(
            "other", (), flat_kv_completion=completion
        )


@requires_runtime_bridge
def test_make_extend_result_event_legacy_path_does_not_require_new_binding(
    monkeypatch,
):
    class _LegacyExtendResult:
        __slots__ = ("request_id", "tokens")

    class _LegacyForwardEvent:
        ExtendResult = _LegacyExtendResult

    monkeypatch.setattr(scheduler_utils, "ForwardEvent", _LegacyForwardEvent)

    event = scheduler_utils.make_extend_result_event("legacy", [7])

    assert event.request_id == "legacy"
    assert event.tokens == [7]
    assert not hasattr(event, "flat_kv_completion")


def test_completion_input_rejects_tensor_backed_scalar():
    with pytest.raises(TypeError, match="host Python int"):
        FlatKVCompletionInput(
            request_id="tensor-backed",
            table_generation=torch.tensor(1),
            dispatch_seq=0,
            dispatch_raw_start=0,
            dispatch_raw_end=1,
            protected_raw_end=1,
        )


def test_completion_input_rejects_dispatch_producer_schema():
    raw = _completion_input(
        request_id="echo",
        generation=1,
        seq=2,
        dispatch_end=4,
        protected_end=8,
    )
    raw["groups"] = [{"group_id": "history", "valid_raw_end": 4}]

    with pytest.raises(ValueError, match="must not contain group schema"):
        FlatKVCompletionInput.from_raw(raw)


def test_completion_input_rows_must_match_forward_rows():
    result, _ = _result()
    result.sync()
    op = _MixedForwardOp()
    op.request_ids = op.request_ids[:1]

    with pytest.raises(ValueError, match="row count differs"):
        result.materialize_flat_kv_completions(op)


def test_completion_input_request_id_must_match_its_forward_row():
    result, _ = _result()
    result.sync()
    op = _MixedForwardOp()
    op.request_ids = ["wrong", "decode"]

    with pytest.raises(ValueError, match="row/request mismatch"):
        result.materialize_flat_kv_completions(op)


def test_decode_acceptance_must_stay_inside_dispatched_interval():
    result, _ = _result(output_lengths=(1, 5))
    result.sync()

    with pytest.raises(ValueError, match="outside dispatch interval"):
        result.materialize_flat_kv_completions(_MixedForwardOp())


class _Sender:
    def __init__(self) -> None:
        self.items = []

    def send_pyobj(self, obj) -> None:
        self.items.append(obj)


class _Tokenizer:
    eos_token_id = None
    additional_stop_token_ids = None

    @staticmethod
    def decode(ids) -> str:
        return "".join(str(value) for value in ids)


class _Metrics:
    enabled = False

    @staticmethod
    def record_nan_abort() -> None:
        return None


def _request_state() -> RequestState:
    return RequestState(
        prompt_input_ids=[1, 2, 3],
        sampling_params=SamplingParams(
            max_new_tokens=8,
            stop=[],
            ignore_eos=True,
        ),
        stream=False,
        tokenizer=_Tokenizer(),
    )


class _MidChunkForwardOp:
    request_ids = ["victim"]
    request_pool_indices = [0]
    extend_prefix_lens = [4]
    input_lengths = [4]
    prefill_lengths = [9]
    flat_kv_completion_inputs = [
        _completion_input(
            request_id="victim",
            generation=3,
            seq=5,
            dispatch_end=8,
            protected_end=12,
        )
    ]

    @staticmethod
    def num_extends() -> int:
        return 1


class _DecodeForwardOp:
    request_ids = ["decode"]
    request_pool_indices = [0]
    extend_prefix_lens = []
    input_lengths = [4]
    flat_kv_completion_inputs = [
        _completion_input(
            request_id="decode",
            generation=13,
            seq=21,
            dispatch_end=8,
            protected_end=12,
        )
    ]

    @staticmethod
    def num_extends() -> int:
        return 0


class _DecodeSuccessorForwardOp:
    request_ids = ["decode"]
    request_pool_indices = [0]
    extend_prefix_lens = []
    input_lengths = [4]
    flat_kv_completion_inputs = [
        _completion_input(
            request_id="decode",
            generation=13,
            seq=22,
            dispatch_end=12,
            protected_end=12,
        )
    ]

    @staticmethod
    def num_extends() -> int:
        return 0


def _terminal_decode_state() -> RequestState:
    state = RequestState(
        prompt_input_ids=[1, 2, 3, 4, 5],
        sampling_params=SamplingParams(
            max_new_tokens=1,
            stop=[],
            ignore_eos=True,
        ),
        stream=False,
        tokenizer=_Tokenizer(),
    )
    state.computed_length = 5
    return state


def _patch_forward_events(monkeypatch):
    def _extend(request_id, tokens=(), *, flat_kv_completion=None):
        return {
            "kind": "extend",
            "request_id": request_id,
            "tokens": list(tokens),
            "flat_kv_completion": flat_kv_completion,
        }

    monkeypatch.setattr(
        "tokenspeed.runtime.engine.generation_output_processor."
        "make_extend_result_event",
        _extend,
    )
    monkeypatch.setattr(
        "tokenspeed.runtime.engine.generation_output_processor.make_abort_event",
        lambda request_id: {"kind": "abort", "request_id": request_id},
    )
    monkeypatch.setattr(
        "tokenspeed.runtime.engine.generation_output_processor.make_finish_event",
        lambda request_id: {"kind": "finish", "request_id": request_id},
    )


@requires_runtime_bridge
def test_host_terminal_rewinds_decode_completion_to_emitted_tokens(monkeypatch):
    _patch_forward_events(monkeypatch)
    sender = _Sender()
    processor = OutputProcesser(
        sender,
        attn_tp_rank=0,
        spec_algorithm="eagle",
        spec_num_tokens=4,
        metrics=_Metrics(),
    )
    processor.rid_to_state["decode"] = _terminal_decode_state()
    result, _ = _result(output_lengths=(4,))

    changes = processor.post_process_forward_op(_DecodeForwardOp(), result)

    assert [change["kind"] for change in changes] == ["extend", "finish"]
    extend = changes[0]
    assert extend["tokens"] == [101]
    # The dispatch starts at raw token 4. Host max-length handling accepted
    # only one of the four GPU-accepted speculative tokens, so the scheduler
    # may publish/rewind only through raw end 5, not the materialized end 8.
    assert extend["flat_kv_completion"].accepted_raw_end == 5


@requires_runtime_bridge
def test_nan_abort_precedes_structured_completion_fence(monkeypatch):
    _patch_forward_events(monkeypatch)
    sender = _Sender()
    processor = OutputProcesser(
        sender,
        attn_tp_rank=0,
        spec_algorithm="eagle",
        spec_num_tokens=4,
        metrics=_Metrics(),
    )
    processor.rid_to_state["decode"] = _terminal_decode_state()
    result, _ = _result(output_lengths=(4,))
    result.output_nan_flags = torch.tensor([1], dtype=torch.int32)

    changes = processor.post_process_forward_op(_DecodeForwardOp(), result)

    # Abort must cancel the outstanding dispatch before its execution fence is
    # retired. The following structured completion then drains debt without
    # applying tokens or publishing suspect KV pages.
    assert [change["kind"] for change in changes] == ["abort", "extend"]
    assert changes[1]["tokens"] == [101]

    # An overlapped successor can finish after the NaN request was removed from
    # Python state. It still owes its completion-only fence so the scheduler can
    # drain the canceled generation and finally apply the pending abort.
    successor_result, _ = _result(output_lengths=(4,))
    successor_changes = processor.post_process_forward_op(
        _DecodeSuccessorForwardOp(), successor_result
    )
    assert [change["kind"] for change in successor_changes] == ["extend"]
    assert successor_changes[0]["tokens"] == []
    assert successor_changes[0]["flat_kv_completion"].dispatch_seq == 22


@requires_runtime_bridge
def test_mid_chunk_kv_write_emits_completion_only_result(monkeypatch):
    def _fake_extend_result(request_id, tokens=(), *, flat_kv_completion=None):
        return {
            "request_id": request_id,
            "tokens": list(tokens),
            "flat_kv_completion": flat_kv_completion,
        }

    monkeypatch.setattr(
        "tokenspeed.runtime.engine.generation_output_processor."
        "make_extend_result_event",
        _fake_extend_result,
    )
    sender = _Sender()
    processor = OutputProcesser(sender, attn_tp_rank=0, metrics=_Metrics())
    state = _request_state()
    processor.rid_to_state["victim"] = state
    result, fence = _result(output_lengths=(1,))
    result.output_tokens = torch.tensor([777], dtype=torch.int32)

    changes = processor.post_process_forward_op(_MidChunkForwardOp(), result)

    assert fence.calls == 1
    assert len(changes) == 1
    assert changes[0]["request_id"] == "victim"
    assert changes[0]["tokens"] == []
    assert changes[0]["flat_kv_completion"].request_id == "victim"
    assert changes[0]["flat_kv_completion"].accepted_raw_end == 8
    assert state.output_ids == []
    assert sender.items == []
    # Even though the sampled output is suppressed, this dispatch wrote input
    # KV through raw end 8 and therefore owes a ready producer completion.
    assert len(result.flat_kv_completions) == 1
    completion = result.flat_kv_completions[0]
    assert completion.request_id == "victim"
    assert completion.accepted_raw_end == 8
    assert completion.table_generation == 3
    assert completion.dispatch_seq == 5


def test_legacy_forward_materializes_empty_side_channel_only():
    class _LegacyOp:
        request_ids = ["legacy"]
        input_lengths = [1]
        # The bound FlatForwardOp exposes this as an empty vector on existing
        # paths that do not participate in the completion protocol.
        flat_kv_completion_inputs = []

        @staticmethod
        def num_extends() -> int:
            return 0

    result, _ = _result(output_lengths=(1,))
    result._flat_kv_execution_evidence = None
    result.sync()

    assert result.materialize_flat_kv_completions(_LegacyOp()) == ()
    assert result.flat_kv_completions == ()
