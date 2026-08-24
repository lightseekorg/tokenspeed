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

"""Per-role forward dispatch: what each engine role does with one round."""

from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace

from tokenspeed.runtime.engine.forward_dispatch import (
    DecodeDispatcher,
    ForwardDispatcher,
    PlannedForward,
    PrefillDispatcher,
)


class _ForwardThread:
    """Records what the control plane hands to the data plane, in order."""

    def __init__(self, trace) -> None:
        self._trace = trace

    def submit(self, fn) -> Future:
        self._trace.append("submit")
        future: Future = Future()
        future.set_result(fn())
        return future

    def run(self, fn):
        self._trace.append("run")
        return fn()


def _executor(trace):
    def execute_forward_op(forward_op, sampling_params_list, **kwargs):
        trace.append(("forward", kwargs["grammar_inputs"], kwargs))
        return SimpleNamespace(sync=lambda: trace.append("sync"))

    return SimpleNamespace(
        forward_thread=_ForwardThread(trace),
        execute_forward_op=execute_forward_op,
        prepare_remote_cache_slots=lambda rows: trace.append(("slots", rows)),
        runtime_states=object(),
        execution_stream=object(),
        device="cpu",
    )


def _planned(*, num_extends=0, is_local_prefill=True, cache_zero_future=None):
    return PlannedForward(
        forward_op=SimpleNamespace(
            request_ids=["a"],
            request_pool_indices=[3, 4],
            num_extends=lambda: num_extends,
            is_local_prefill=lambda: is_local_prefill,
        ),
        sampling_params_list=[object()],
        dp_metadata=None,
        grammar_inputs="GRAMMAR",
        multimodal_context="MM",
        cache_zero_future=cache_zero_future,
    )


def test_plain_engine_forwards_with_the_batch_grammar():
    trace = []
    pending, on_first_token = ForwardDispatcher(_executor(trace)).dispatch(_planned())

    assert on_first_token is None
    assert trace[0] == "submit"
    assert trace[1][1] == "GRAMMAR"
    assert trace[1][2]["capture_next_input_ids"] is False
    # The handle is not resolved by dispatch; only commit joins it.
    assert "sync" not in trace
    pending.result()
    assert trace[-1] == "sync"


def test_decode_node_triggers_the_receive_for_a_remote_prefill_batch():
    trace = []
    zero_event = SimpleNamespace(synchronize=lambda: trace.append("zero-sync"))
    cache_zero_future: Future = Future()
    cache_zero_future.set_result(zero_event)
    kv_transfer = SimpleNamespace(
        reset_valid_cache_length=lambda *a: trace.append("reset"),
        execute=lambda op: trace.append("rdma"),
    )

    pending, on_first_token = DecodeDispatcher(
        _executor(trace), kv_transfer, pd_cache_enabled=True
    ).dispatch(
        _planned(
            num_extends=1, is_local_prefill=False, cache_zero_future=cache_zero_future
        )
    )

    # No model output this round, and the whole path ran as one unit on the
    # data plane, with the zeroing barrier before the manifest is published.
    assert (pending, on_first_token) == (None, None)
    assert trace == ["run", ("slots", [3]), "reset", "zero-sync", "rdma"]


def test_decode_node_runs_local_batches_without_grammar():
    trace = []
    pending, on_first_token = DecodeDispatcher(
        _executor(trace), SimpleNamespace(), pd_cache_enabled=False
    ).dispatch(_planned(num_extends=0))

    assert pending is not None and on_first_token is None
    assert trace[1][1] is None


def test_prefill_node_hands_the_kv_off_when_no_chunk_is_left():
    trace = []
    kv_transfer = SimpleNamespace(
        execute=lambda op: trace.append("send-kv"),
        store_prefill_token=lambda *a: None,
    )

    result = PrefillDispatcher(
        _executor(trace), kv_transfer, epd_hooks=SimpleNamespace()
    ).dispatch(_planned(num_extends=0))

    assert result == (None, None)
    assert trace == ["send-kv"]


def test_prefill_node_captures_next_input_ids_and_returns_the_token_hook():
    trace = []
    store_prefill_token = lambda *a: None  # noqa: E731 - identity asserted below
    kv_transfer = SimpleNamespace(
        prepare_prefill=lambda op: trace.append("prepare"),
        store_prefill_token=store_prefill_token,
    )
    epd_hooks = SimpleNamespace(
        assert_embeddings_received=lambda ctx: trace.append(("epd", ctx))
    )

    pending, on_first_token = PrefillDispatcher(
        _executor(trace), kv_transfer, epd_hooks=epd_hooks
    ).dispatch(_planned(num_extends=1))

    assert pending is not None
    assert on_first_token is store_prefill_token
    # Control-plane bookkeeping first, GPU work after.
    assert trace[:3] == ["prepare", ("epd", "MM"), "submit"]
    assert trace[3][2]["capture_next_input_ids"] is True
