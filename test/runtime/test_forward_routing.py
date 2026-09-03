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

"""Per-role forward routing: what each engine role does with one round.

The role is a value on ``DeviceHandle``, so these drive the real entry
point (``execute``) rather than a hand-built dispatcher.
"""

from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from tokenspeed.runtime.execution.device import DeviceHandle, DeviceRole
from tokenspeed.runtime.execution.types import PlannedForward
from tokenspeed.runtime.pd.decode_executor import DisaggDecodeExecutor
from tokenspeed.runtime.pd.prefill_executor import DisaggPrefillExecutor


class _ForwardThread:
    """Records what the control plane hands to the data plane, in order."""

    def __init__(self, trace) -> None:
        self._trace = trace

    def submit(self, fn) -> Future:
        self._trace.append("submit")
        future: Future = Future()
        try:
            future.set_result(fn())
        except BaseException as exc:  # noqa: BLE001 — mirrors ForwardThread
            future.set_exception(exc)
        return future

    def run(self, fn):
        self._trace.append("run")
        return fn()


class _PrefillPeer(DisaggPrefillExecutor):
    """A P-role transfer peer the handle will recognize; the real __init__
    (sockets, KV manager) is bypassed on purpose."""

    def __init__(self, trace) -> None:
        self._trace = trace

    def execute(self, op) -> None:
        self._trace.append(("send-kv", op))

    def prepare_prefill(self, op) -> None:
        self._trace.append("prepare")


class _DecodePeer(DisaggDecodeExecutor):
    """A D-role transfer peer; see _PrefillPeer."""

    def __init__(self, trace) -> None:
        self._trace = trace

    def execute(self, op) -> None:
        self._trace.append("rdma")


def _handle(trace, kv_transfer=None):
    """A real DeviceHandle over a fake executor, built the way startup builds
    it: the role comes from the peer, so the trace is the complete crossing."""

    def execute_forward_op(forward_op, sampling_params_list, **kwargs):
        trace.append(("forward", kwargs["grammar_inputs"], kwargs))
        return SimpleNamespace(sync=lambda: trace.append("sync"))

    return DeviceHandle(
        SimpleNamespace(
            forward_thread=_ForwardThread(trace),
            execute_forward_op=execute_forward_op,
            prepare_remote_cache_slots=lambda rows: trace.append(("slots", rows)),
            reset_remote_prefill_cache_lengths=lambda op: trace.append("seed-lengths"),
        ),
        kv_transfer=kv_transfer,
    )


def _plan(*, pages_to_zero=(), cache=(), remote_decode=None, remote_prefill=None):
    """A round's ExecutionPlan as execute() reads it."""
    return SimpleNamespace(
        pages_to_zero=list(pages_to_zero),
        cache=list(cache),
        remote_decode=remote_decode,
        remote_prefill=remote_prefill,
    )


def _planned(*, num_extends=0):
    return PlannedForward(
        forward_op=SimpleNamespace(
            request_ids=["a"],
            request_pool_indices=[3, 4],
            num_extends=lambda: num_extends,
        ),
        sampling_params_list=[object()],
        dp_metadata=None,
        grammar_inputs="GRAMMAR",
        multimodal_context="MM",
    )


def test_plain_engine_forwards_with_the_batch_grammar():
    trace = []
    pending = _handle(trace).execute(_plan(), _planned())

    assert trace[0] == "submit"
    assert trace[1][1] == "GRAMMAR"
    assert trace[1][2]["capture_next_input_ids"] is False
    # The handle is not resolved by dispatch; only commit joins it.
    assert "sync" not in trace
    pending.result()
    assert trace[-1] == "sync"


def test_decode_node_triggers_the_receive_from_the_plan_stream():
    """D role: the remote prefill rides plan.remote_prefill, submitted
    asynchronously like any forward, with the zeroing fence inside."""
    trace = []
    handle = _handle(trace, _DecodePeer(trace))
    # The plan's own page zeroing: the RDMA fence waits on it because
    # Mooncake writes are not ordered by the zeroing stream.
    handle._executor.zero_cache_pages = lambda pages: SimpleNamespace(
        synchronize=lambda: trace.append("zero-sync")
    )
    remote_prefill = SimpleNamespace(request_pool_indices=[3, 4], num_extends=lambda: 1)

    pending = handle.execute(
        _plan(pages_to_zero=[7], remote_prefill=remote_prefill), None
    )
    trace.remove("submit")  # the zeroing submission itself
    trace.remove("submit")  # the receive submission itself

    # No model output this round, and the whole path ran as one unit on the
    # data plane, with the zeroing barrier before the manifest is published.
    assert pending is None
    assert trace == [("slots", [3]), "seed-lengths", "zero-sync", "rdma"]


def test_decode_node_masks_local_batches_with_the_batch_grammar():
    """The matcher was advanced past the prefill node's token when the
    RemotePrefillDoneEvent landed, so decode masks from the right state."""
    trace = []
    pending = _handle(trace, _DecodePeer(trace)).execute(
        _plan(), _planned(num_extends=0)
    )

    assert pending is not None
    assert trace[1][1] == "GRAMMAR"


def test_prefill_node_sends_remote_decodes_from_the_plan_stream():
    """P role: a completed prompt's decode happens on the peer, so its KV
    goes out — on the plan's own stream, beside whatever forward runs."""
    trace = []
    remote_decode = SimpleNamespace(request_ids=["done"])

    pending = _handle(trace, _PrefillPeer(trace)).execute(
        _plan(remote_decode=remote_decode), _planned(num_extends=1)
    )

    # The send reads KV that earlier forwards wrote, so it rides the FIFO
    # rather than racing them from the control plane — ahead of this round's
    # batch, whose KV it does not read (the scheduler held it until the
    # final chunk's result landed). Submitted asynchronously, exactly like
    # the forward itself.
    assert pending is not None
    assert trace[:2] == ["submit", ("send-kv", remote_decode)]


def test_remote_decodes_go_out_even_on_an_idle_round():
    """A DP-idle or frozen round has no batch (planned=None), but the plan's
    remote decodes are the plan's own work: a rank that reports idle over
    one must still send it, or the request strands."""
    trace = []
    remote_decode = SimpleNamespace(request_ids=["done"])

    result = _handle(trace, _PrefillPeer(trace)).execute(
        _plan(remote_decode=remote_decode), None
    )

    assert result is None
    assert trace == ["submit", ("send-kv", remote_decode)]


def test_prefill_node_captures_next_input_ids_for_the_bootstrap_payload():
    trace = []
    peer = _PrefillPeer(trace)

    pending = _handle(trace, peer).execute(_plan(), _planned(num_extends=1))

    assert pending is not None
    # The layerwise arming is enqueued BEFORE the forward it arms — and
    # submitted, not waited on: the per-round path never blocks.
    assert trace[:3] == ["submit", "prepare", "submit"]
    assert trace[3][2]["capture_next_input_ids"] is True


def test_a_failed_transfer_submission_surfaces_at_the_next_round():
    """The peer's work is submitted fire-and-forget like a forward, and its
    semantic completion arrives through the transfer events — so a
    submission that RAISED produces no event and must surface from the
    settle at the next round's execute, not be swallowed."""
    trace = []
    peer = _PrefillPeer(trace)

    def _boom(op):
        raise RuntimeError("no bootstrap sender")

    peer.execute = _boom
    handle = _handle(trace, peer)
    handle.execute(_plan(remote_decode=SimpleNamespace(request_ids=["x"])), None)

    with pytest.raises(RuntimeError, match="transfer submission failed"):
        handle.execute(_plan(), None)


def test_role_values_are_the_disaggregation_modes():
    """The enum's values ARE ``server_args.disaggregation_mode``, so the two
    vocabularies cannot drift into a silent mismatch."""
    assert {role.value for role in DeviceRole} == {"null", "prefill", "decode"}


def test_the_attached_peer_fixes_the_role():
    assert _handle([]).role is DeviceRole.PLAIN
    assert _handle([], _PrefillPeer([])).role is DeviceRole.PD_PREFILL
    assert _handle([], _DecodePeer([])).role is DeviceRole.PD_DECODE


def test_an_unrecognized_transfer_peer_is_refused():
    """The role is read off the peer at construction, so a peer of the wrong
    type fails at startup rather than at the first transfer."""
    with pytest.raises(TypeError, match="Disagg"):
        _handle([], SimpleNamespace())
