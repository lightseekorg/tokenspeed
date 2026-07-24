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

"""Unit tests for OutputProcesser.post_process_forward_op after a flat retract.

Flat retract rebases the victim's generated tokens into its prefill window
(C++ RebasePrefill), so on re-admission the op's ``prefill_lengths`` exceeds
the original prompt length that RequestState.prefill_finished compares
against. A mid-chunk re-prefill slot must emit NO ExtendResultEvent (the C++
FSM is still Prefilling and would throw) and stream NO token (the sampled
token is garbage). The gate is the op's own chunking criterion:
``extend_prefix_lens[i] + input_lengths[i] < prefill_lengths[i]``.
"""

from __future__ import annotations

from tokenspeed.runtime.engine.generation_output_processor import (
    OutputProcesser,
    RequestState,
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

    def record_nan_abort(self):
        pass


class _Scalar:
    def __init__(self, v):
        self._v = v

    def item(self):
        return self._v


class _Vec:
    """Minimal tensor stand-in: indexable scalars plus tolist()."""

    def __init__(self, vals):
        self._vals = list(vals)

    def __getitem__(self, i):
        return _Scalar(self._vals[i])

    def tolist(self):
        return list(self._vals)


class _Results:
    """ModelExecutionResult stub: one sampled token per slot."""

    def __init__(self, output_tokens, output_lengths):
        self.output_tokens = _Vec(output_tokens)
        self.output_lengths = _Vec(output_lengths)
        self.output_logprobs = None
        self.output_nan_flags = None
        self.grammar_completion = None
        self.next_input_ids = None

    def sync(self):
        pass

    @staticmethod
    def materialize_flat_kv_completions(_forward_op, *, accepted_lengths=None):
        del accepted_lengths
        return ()


class _ForwardOp:
    """Forward-op stub exposing the flat binding's per-slot prefill_lengths."""

    def __init__(
        self,
        request_ids,
        extend_prefix_lens,
        input_lengths,
        num_extends,
        prefill_lengths=None,
    ):
        self.request_ids = request_ids
        self.extend_prefix_lens = extend_prefix_lens
        self.input_lengths = input_lengths
        self.request_pool_indices = list(range(len(request_ids)))
        self._num_extends = num_extends
        if prefill_lengths is not None:
            self.prefill_lengths = prefill_lengths

    def num_extends(self):
        return self._num_extends


def _state(input_ids: list[int]) -> RequestState:
    return RequestState(
        prompt_input_ids=input_ids,
        sampling_params=SamplingParams(max_new_tokens=8, stop=[], ignore_eos=True),
        stream=False,
        tokenizer=_Tokenizer(),
    )


def _proc(sender: _Sender) -> OutputProcesser:
    return OutputProcesser(sender, attn_tp_rank=0, metrics=_Metrics())


def _kinds(events) -> list[str]:
    return [type(e).__name__ for e in events]


def test_mid_chunk_readmit_slot_emits_nothing():
    # Retract-rebase shape: prompt is 3 tokens but the op's prefill length is 9
    # (prompt + 6 generated tokens rebased into the prefill window). First
    # re-admission chunk covers 4+4=8 < 9 — mid-chunk, C++ owes no result.
    sender = _Sender()
    proc = _proc(sender)
    state = _state([1, 2, 3])
    proc.rid_to_state["victim"] = state

    op = _ForwardOp(
        request_ids=["victim"],
        extend_prefix_lens=[4],
        input_lengths=[4],
        num_extends=1,
        prefill_lengths=[9],
    )
    changes = proc.post_process_forward_op(op, _Results([777], [1]))

    # The old prompt-length gate saw computed(8) >= prompt(3) and emitted an
    # ExtendResultEvent (C++ Prefilling FSM throws) plus one garbage token.
    assert "ExtendResult" not in _kinds(changes)
    assert changes == []
    assert state.output_ids == []
    assert sender.items == []
    assert "victim" in proc.rid_to_state


def test_final_chunk_readmit_slot_emits_result():
    # Same rebased request, last chunk: 8+1 == 9 — the op's criterion matches
    # the C++ PrefillDone decision, byte-identical to a fresh request whose
    # final chunk reaches its prompt length.
    sender = _Sender()
    proc = _proc(sender)
    state = _state([1, 2, 3])
    proc.rid_to_state["victim"] = state

    op = _ForwardOp(
        request_ids=["victim"],
        extend_prefix_lens=[8],
        input_lengths=[1],
        num_extends=1,
        prefill_lengths=[9],
    )
    changes = proc.post_process_forward_op(op, _Results([777], [1]))

    assert "ExtendResult" in _kinds(changes)
    assert state.output_ids == [777]
    assert len(sender.items) == 1


def test_decode_slot_unaffected_by_prefill_lengths():
    # Decode slots never consult the extend gate; result flows as before.
    sender = _Sender()
    proc = _proc(sender)
    state = _state([1, 2, 3])
    state.computed_length = 3  # prefill already done
    proc.rid_to_state["dec"] = state

    op = _ForwardOp(
        request_ids=["dec"],
        extend_prefix_lens=[],
        input_lengths=[1],
        num_extends=0,
        prefill_lengths=[],
    )
    changes = proc.post_process_forward_op(op, _Results([555], [1]))

    kinds = _kinds(changes)
    assert "ExtendResult" in kinds
    assert "UpdateReserveNumTokens" in kinds
    assert state.output_ids == [555]


def test_op_without_prefill_lengths_falls_back_to_prompt_gate():
    # Older/radix binding without the field: the prefill_finished gate still
    # suppresses mid-chunk output for fresh requests.
    sender = _Sender()
    proc = _proc(sender)
    state = _state([1, 2, 3, 4, 5, 6, 7, 8])
    proc.rid_to_state["fresh"] = state

    op = _ForwardOp(
        request_ids=["fresh"],
        extend_prefix_lens=[0],
        input_lengths=[4],
        num_extends=1,
        prefill_lengths=None,
    )
    changes = proc.post_process_forward_op(op, _Results([888], [1]))

    assert changes == []
    assert state.output_ids == []
    assert sender.items == []
