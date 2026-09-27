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

"""CPU-only tests for EpdPrefillHooks, the event loop side of EPD prefill
admission. The receive/reassembly controller itself (EpdPrefillAdmission) is
exercised elsewhere; these drive the hooks against fakes, so no Mooncake, ZMQ,
or CUDA context is created.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.epd.prefill_hooks import EpdPrefillHooks  # noqa: E402


class _State:
    def __init__(self, mm_items=None, *, finished: bool = False) -> None:
        if mm_items is not None:
            self.multimodal_inputs = SimpleNamespace(mm_items=mm_items)
        self.finished = finished
        self.finish_calls: list[str] = []

    def set_finish_with_abort(self, message: str) -> None:
        self.finish_calls.append(message)


def _handshaked_item():
    return SimpleNamespace(encode_handshake=object(), encoded=None)


def _plain_item():
    return SimpleNamespace(encode_handshake=None, encoded=None)


class _Admission:
    def __init__(self) -> None:
        self.staged: list[str] = []
        self.drain_result: tuple[list, list] = ([], [])
        self.pending = False

    def stage(self, request_id, mm_items) -> None:
        self.staged.append(request_id)

    def drain(self):
        return self.drain_result

    def has_pending(self) -> bool:
        return self.pending


class _OutputProcessor:
    def __init__(self) -> None:
        self.published: list[str] = []

    def publish_finished_at_admission(self, request_id, _state) -> None:
        self.published.append(request_id)


class _FakeLoop:
    """Only the loop state EpdPrefillHooks reads."""

    def __init__(self) -> None:
        self._pause = SimpleNamespace(admit_blocked=False)
        self.kv_transfer = None
        self.output_processor = _OutputProcessor()
        self.submitted: list[list] = []
        self.scheduler = SimpleNamespace(
            submit_requests=lambda specs: self.submitted.append(list(specs))
        )

    def _submit_scheduler_requests(self, specs) -> None:
        self.scheduler.submit_requests(specs)


def _spec(rid: str):
    return SimpleNamespace(request_id=rid)


# --- try_stage ------------------------------------------------------------------


def test_try_stage_is_a_no_op_on_non_epd_nodes() -> None:
    hooks = EpdPrefillHooks(_FakeLoop(), None)
    state = _State(mm_items=[_handshaked_item()])

    assert not hooks.try_stage(_spec("r0"), state, None)
    assert hooks._staged == {}


def test_try_stage_passes_through_non_encode_routed_requests() -> None:
    admission = _Admission()
    hooks = EpdPrefillHooks(_FakeLoop(), admission)

    # Text-only (no multimodal_inputs at all) and plain multimodal (no
    # per-image encode handshake) both admit immediately.
    assert not hooks.try_stage(_spec("text"), _State(), None)
    assert not hooks.try_stage(_spec("mm"), _State(mm_items=[_plain_item()]), None)
    assert admission.staged == []


def test_try_stage_stages_encode_routed_requests() -> None:
    admission = _Admission()
    hooks = EpdPrefillHooks(_FakeLoop(), admission)
    spec, state = _spec("r0"), _State(mm_items=[_handshaked_item()])

    assert hooks.try_stage(spec, state, "bootstrap")

    assert admission.staged == ["r0"]
    assert hooks._staged == {"r0": (spec, state, "bootstrap")}


# --- drain_ready_embeddings -----------------------------------------------------


def test_drain_submits_admitted_and_registers_deferred_sender() -> None:
    admission = _Admission()
    loop = _FakeLoop()
    registered: list[tuple[str, object]] = []
    loop.kv_transfer = SimpleNamespace(
        register=lambda rid, bootstrap: registered.append((rid, bootstrap))
    )
    hooks = EpdPrefillHooks(loop, admission)
    spec, state = _spec("r0"), _State(mm_items=[_handshaked_item()])
    hooks.try_stage(spec, state, "bootstrap")
    admission.drain_result = (["r0"], [])

    hooks.drain_ready_embeddings()

    assert registered == [("r0", "bootstrap")]
    assert loop.submitted == [[spec]]
    assert hooks._staged == {}


def test_drain_finishes_failed_requests_without_submitting() -> None:
    admission = _Admission()
    loop = _FakeLoop()
    hooks = EpdPrefillHooks(loop, admission)
    spec, state = _spec("r0"), _State(mm_items=[_handshaked_item()])
    hooks.try_stage(spec, state, None)
    admission.drain_result = ([], ["r0"])

    hooks.drain_ready_embeddings()

    assert state.finish_calls == ["EPD embedding receive failed or timed out"]
    assert loop.output_processor.published == ["r0"]
    assert loop.submitted == []
    assert hooks._staged == {}


def test_drain_reaps_requests_aborted_mid_receive() -> None:
    admission = _Admission()
    loop = _FakeLoop()
    loop.kv_transfer = SimpleNamespace(
        register=lambda rid, bootstrap: pytest.fail("must not register the sender")
    )
    hooks = EpdPrefillHooks(loop, admission)
    spec = _spec("r0")
    state = _State(mm_items=[_handshaked_item()], finished=True)
    hooks.try_stage(spec, state, "bootstrap")
    admission.drain_result = (["r0"], [])

    hooks.drain_ready_embeddings()

    assert loop.output_processor.published == ["r0"]
    assert loop.submitted == []


def test_drain_withholds_admission_while_paused() -> None:
    admission = _Admission()
    loop = _FakeLoop()
    loop._pause.admit_blocked = True
    hooks = EpdPrefillHooks(loop, admission)
    hooks.try_stage(_spec("r0"), _State(mm_items=[_handshaked_item()]), None)
    admission.drain_result = (["r0"], [])

    hooks.drain_ready_embeddings()

    assert loop.submitted == []
    assert "r0" in hooks._staged


# --- assert_embeddings_received --------------------------------------------------


def test_assert_embeddings_received_raises_on_unreceived_handshaked_item() -> None:
    hooks = EpdPrefillHooks(_FakeLoop(), _Admission())
    context = SimpleNamespace(
        has_extend_inputs=lambda: True,
        mm_inputs=[SimpleNamespace(mm_items=[_handshaked_item()])],
    )

    with pytest.raises(RuntimeError, match="un-received"):
        hooks.assert_embeddings_received(context)


def test_assert_embeddings_received_accepts_filled_items_and_non_epd() -> None:
    filled = SimpleNamespace(encode_handshake=object(), encoded=object())
    context = SimpleNamespace(
        has_extend_inputs=lambda: True,
        mm_inputs=[SimpleNamespace(mm_items=[filled, _plain_item()])],
    )
    EpdPrefillHooks(_FakeLoop(), _Admission()).assert_embeddings_received(context)
    # Non-EPD node: no-op even with an un-received item.
    bad = SimpleNamespace(
        has_extend_inputs=lambda: True,
        mm_inputs=[SimpleNamespace(mm_items=[_handshaked_item()])],
    )
    EpdPrefillHooks(_FakeLoop(), None).assert_embeddings_received(bad)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
