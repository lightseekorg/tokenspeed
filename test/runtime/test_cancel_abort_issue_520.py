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

"""Regression tests for cancel / abort / retry on the OpenAI HTTP frontend.

Two independent bugs make non-streaming cancellation a no-op; both are
covered here.

* :class:`MarkAbortReleasesSlotTests` — ``mark_abort`` and
  ``register()``'s rid-pending-abort branch must drive the state all
  the way to ``finished_reason``, so the
  ``request_state.to_abort and request_state.finished`` gate in
  ``post_process_forward_op`` actually fires and ``make_finish_event``
  releases the scheduler slot.

* :class:`AwaitWithDisconnectWatchdogTests` — non-streaming OpenAI
  handlers must cancel the inflight engine call when the client
  disconnects, so the engine's ``finally:`` block sends ``AbortReq``
  and ``async_llm``'s "cancellation contract" actually holds.
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-cpu")

from tokenspeed.runtime.engine.generation_output_processor import (  # noqa: E402
    OutputProcesser,
    RequestState,
)
from tokenspeed.runtime.engine.request_types import FINISH_ABORT  # noqa: E402
from tokenspeed.runtime.entrypoints.openai.serving_base import (  # noqa: E402
    OpenAIServingBase,
    await_with_disconnect_watchdog,
)
from tokenspeed.runtime.metrics.collector import EngineMetrics  # noqa: E402
from tokenspeed.runtime.sampling.sampling_params import SamplingParams  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request_state() -> RequestState:
    """Build a minimal RequestState for unit testing the abort path.

    The constructor only really needs ``sampling_params`` to be a
    ``SamplingParams`` instance (so its ``__post_init__``-style
    invariants hold). Everything else can stay default.
    """
    return RequestState(
        prompt_input_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=4),
        stream=False,
        tokenizer=None,
    )


class _NoopSender:
    """Stand-in for the tokenizer-IPC socket that OutputProcesser pushes
    BatchTokenIDOut into. None of the abort-path tests actually send
    anything; we just need a non-``None`` placeholder."""

    def send_pyobj(self, _obj) -> None:  # noqa: D401
        return None


def _make_processor() -> OutputProcesser:
    return OutputProcesser(
        send_to_tokenizer=_NoopSender(),
        metrics=EngineMetrics(labels={}, enabled=False),
    )


class _FakeRequest:
    """Mimics the ``fastapi.Request`` surface that
    ``await_with_disconnect_watchdog`` uses — just ``is_disconnected()``."""

    def __init__(self) -> None:
        self._disconnected = False

    def disconnect(self) -> None:
        self._disconnected = True

    async def is_disconnected(self) -> bool:
        return self._disconnected


# ---------------------------------------------------------------------------
# Bug 1 — mark_abort / register() must materialize finished_reason.
# ---------------------------------------------------------------------------


class MarkAbortReleasesSlotTests(unittest.TestCase):
    """Pin the contract that scheduler-slot release relies on:

    ``post_process_forward_op`` only emits ``make_finish_event`` when
    ``request_state.to_abort and request_state.finished``. Since
    ``finished`` is ``finished_reason is not None``, abort handlers MUST
    set ``finished_reason``, not just flip ``to_abort``.

    Pre-fix, this check failed silently — cancelled requests kept
    running forward steps until natural ``max_tokens`` / EOS.
    """

    def test_mark_abort_after_register_sets_finished_reason(self) -> None:
        proc = _make_processor()
        state = _make_request_state()
        proc.register("rid-1", state)

        proc.mark_abort("rid-1")

        # All three abort-path fields populated, not just to_abort.
        self.assertTrue(state.to_abort)
        self.assertEqual(state.to_abort_message, "AbortReq from client")
        self.assertIsNotNone(state.finished_reason)
        self.assertIsInstance(state.finished_reason, FINISH_ABORT)

        # ``finished`` is a property of finished_reason — this is the
        # exact gate post_process_forward_op uses.
        self.assertTrue(state.finished)

    def test_register_after_mark_abort_sets_finished_reason(self) -> None:
        """Pre-arrival reorder: AbortReq lands before its TokenizedGenerateReqInput.
        The pending-abort branch in register() must apply the same
        full-strength abort, not the half-flip the bug shipped with."""
        proc = _make_processor()
        proc.mark_abort("rid-2")  # before register
        state = _make_request_state()

        proc.register("rid-2", state)

        self.assertTrue(state.to_abort)
        self.assertIsNotNone(state.finished_reason)
        self.assertIsInstance(state.finished_reason, FINISH_ABORT)
        self.assertTrue(state.finished)

    def test_mark_abort_unknown_rid_buffers(self) -> None:
        """An abort for an unknown rid still goes into ``pending_aborts``
        (TTL-bounded). Nothing should be set on a ghost state since
        there isn't one — just verify no crash and the rid is queued."""
        proc = _make_processor()
        proc.mark_abort("rid-ghost")
        self.assertIn("rid-ghost", proc.pending_aborts)

    def test_mark_abort_idempotent(self) -> None:
        """Aborting the same rid twice is fine: second call sees the
        already-aborted state and either re-applies the same fields
        (idempotent) or short-circuits via ``state.finished``. Either
        way, no crash and finished_reason stays set."""
        proc = _make_processor()
        state = _make_request_state()
        proc.register("rid-3", state)

        proc.mark_abort("rid-3")
        first_reason = state.finished_reason
        proc.mark_abort("rid-3")  # second abort

        self.assertIsNotNone(state.finished_reason)
        # Reason stays the same shape (FINISH_ABORT). We don't pin
        # identity here in case future refactors mint a fresh instance.
        self.assertIsInstance(state.finished_reason, FINISH_ABORT)
        del first_reason  # silence "assigned but never used" lint


# ---------------------------------------------------------------------------
# Bug 2 — await_with_disconnect_watchdog must propagate cancellation.
# ---------------------------------------------------------------------------


class AwaitWithDisconnectWatchdogTests(unittest.TestCase):
    """Pin the watchdog's three behaviors:

    1. ``raw_request=None`` is a no-watchdog passthrough (in-process
       callers without a FastAPI request handle).
    2. Awaitable completes normally: result is returned.
    3. Client disconnects: the inflight task is cancelled and
       ``CancelledError`` propagates so the engine's ``finally:`` block
       runs ``abort_request``.
    """

    def test_passthrough_when_no_request(self) -> None:
        async def _producer() -> str:
            return "hello"

        result = asyncio.run(
            await_with_disconnect_watchdog(_producer(), raw_request=None)
        )
        self.assertEqual(result, "hello")

    def test_normal_completion_with_request(self) -> None:
        async def _producer() -> str:
            await asyncio.sleep(0)  # one event-loop hop
            return "world"

        async def _go() -> str:
            return await await_with_disconnect_watchdog(
                _producer(),
                raw_request=_FakeRequest(),
                poll_interval=0.05,
            )

        self.assertEqual(asyncio.run(_go()), "world")

    def test_cancels_on_disconnect(self) -> None:
        """Client disconnects mid-await — the wrapped task must be
        cancelled and ``CancelledError`` propagates. We track whether
        the engine's ``finally:`` block ran by setting a flag from the
        cancelled coroutine."""
        finally_ran = []

        async def _engine_call() -> str:
            try:
                # Simulate a long-running engine call.
                await asyncio.sleep(60)
                return "should-not-reach"
            finally:
                # async_llm.generate_request's finally: block is what
                # actually calls abort_request. Marking a flag here is
                # the unit-test analog.
                finally_ran.append(True)

        async def _go() -> None:
            req = _FakeRequest()

            async def _disconnect_after(delay: float) -> None:
                await asyncio.sleep(delay)
                req.disconnect()

            asyncio.ensure_future(_disconnect_after(0.1))
            with self.assertRaises(asyncio.CancelledError):
                await await_with_disconnect_watchdog(
                    _engine_call(),
                    raw_request=req,
                    poll_interval=0.05,
                )

        asyncio.run(_go())
        self.assertEqual(
            finally_ran,
            [True],
            "engine's finally: block must run before the watchdog returns "
            "— that's where abort_request is called",
        )

    def test_propagates_awaitable_exception(self) -> None:
        """If the awaitable itself raises, the watchdog forwards that
        exception verbatim (not wrapped in CancelledError). The
        non-streaming handler relies on this to surface `ValueError`
        from ``generate_request``."""

        class _CustomError(Exception):
            pass

        async def _raises() -> None:
            await asyncio.sleep(0)
            raise _CustomError("boom")

        async def _go() -> None:
            with self.assertRaises(_CustomError):
                await await_with_disconnect_watchdog(
                    _raises(),
                    raw_request=_FakeRequest(),
                    poll_interval=0.05,
                )

        asyncio.run(_go())


# ---------------------------------------------------------------------------
# Bug 3 — handle_request must absorb CancelledError.
# ---------------------------------------------------------------------------


class HandleRequestAbsorbsCancelledErrorTests(unittest.TestCase):
    """Regression test for client disconnect cancellation handling.

    When the client disconnects during a non-streaming request,
    ``await_with_disconnect_watchdog`` raises ``asyncio.CancelledError``.
    Because ``CancelledError`` inherits from ``BaseException`` (not
    ``Exception``) in Python 3.8+, it was slipping past the
    ``except Exception`` handler in ``handle_request`` and propagating to
    uvicorn, which logged it as "Exception in ASGI application".

    After the fix, ``handle_request`` catches ``CancelledError`` and returns
    a 499 (Client Closed Request) response instead of re-raising.
    """

    def _make_handler_and_request(self, *, raises: BaseException):
        """Return ``(handler, request)`` where the handler's non-streaming
        path raises *raises*."""

        class _FakeRequest:
            stream = False

        class _MinimalHandler(OpenAIServingBase):
            def _request_id_prefix(self):
                return "test"

            def _convert_to_internal_request(self, request):
                return (object(), request)

            async def _handle_non_streaming_request(self, adapted, request, raw):
                raise raises

        return _MinimalHandler(engine_client=None), _FakeRequest()

    def test_cancelled_error_returns_499_not_raises(self) -> None:
        """CancelledError from the engine must not propagate to the caller."""
        from fastapi.responses import ORJSONResponse

        handler, req = self._make_handler_and_request(
            raises=asyncio.CancelledError("client disconnected")
        )

        response = asyncio.run(handler.handle_request(request=req, raw_request=None))

        self.assertIsInstance(response, ORJSONResponse)
        self.assertEqual(response.status_code, 499)

    def test_cancelled_error_does_not_suppress_real_exceptions(self) -> None:
        """A genuine ValueError from the engine still surfaces as 500."""
        from fastapi.responses import ORJSONResponse

        handler, req = self._make_handler_and_request(raises=ValueError("boom"))

        response = asyncio.run(handler.handle_request(request=req, raw_request=None))

        self.assertIsInstance(response, ORJSONResponse)
        self.assertEqual(response.status_code, 500)


if __name__ == "__main__":
    unittest.main()
