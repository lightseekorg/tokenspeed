"""Regression guard for ``AsyncLLM._wait_one_response`` cancellation.

Locks the contract Phase G.1 established: when the task driving a
streaming generator is cancelled (e.g. FastAPI cancelling its route
coroutine because the client disconnected), the generator's
``finally`` drops the rid from ``rid_to_state`` and asks the engine to
abort it exactly once. No dangling per-request state. No
reliance on ``fastapi.Request.is_disconnected()``.

This test uses a stub that bypasses ZMQ / ModelConfig / HF tokenizer
bring-up — the same pattern as
``test/runtime/test_inline_detokenizer_receiver.py``.
"""

import os
import sys

import pytest

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")

import asyncio  # noqa: E402
import types  # noqa: E402

from tokenspeed.runtime.engine.async_llm import AsyncLLM  # noqa: E402
from tokenspeed.runtime.engine.collector import RequestOutputCollector  # noqa: E402
from tokenspeed.runtime.engine.output_processor import ReqState  # noqa: E402


class _FakeFrontend:
    """Captures the aborts the stub AsyncLLM sends the engines; the real one
    is the ``tokenspeed_engine`` frontend, which needs running engines."""

    def __init__(self) -> None:
        self.aborts: list[str] = []

    async def abort(self, rid: str) -> None:
        self.aborts.append(rid)


class _StubAsyncLLM(AsyncLLM):
    """Bypass ZMQ + ModelConfig + HF bring-up.

    Populates only the attributes ``_wait_one_response`` + ``abort_request``
    read: ``rid_to_state``, ``log_requests``, ``enable_metrics``,
    ``tokenizer``, ``model_config``, ``_abort_tasks``, and a capture fake
    ``frontend`` so the test can assert the abort was sent.
    """

    def __init__(self) -> None:
        self.rid_to_state: dict[str, ReqState] = {}
        self.log_requests = False
        self.enable_metrics = False
        self.tokenizer = None
        self.model_config = types.SimpleNamespace(is_multimodal_gen=False)
        self._abort_tasks = set()
        self.frontend = _FakeFrontend()


class _StubReqObj:
    """Minimal stand-in for ``GenerateReqInput`` used by
    ``_wait_one_response``. Only the attributes actually read by the
    generator body are provided.
    """

    def __init__(self, *, rid: str = "r1", stream: bool = True, input_ids=None) -> None:
        self.rid = rid
        self.stream = stream
        self.input_ids = input_ids
        self.text = None
        self.sampling_params = {"skip_special_tokens": False}


def _fresh_state(obj: _StubReqObj) -> ReqState:
    return ReqState(
        RequestOutputCollector(),
        False,
        asyncio.Event(),
        obj,
        created_time=0.0,
    )


class TestWaitOneResponseCancellation:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_cancelled_before_first_output_cleans_up(self) -> None:
        """Cancel while ``_wait_one_response`` is blocked on the first
        event. Expect: rid dropped from ``rid_to_state``; one abort sent to
        the engine.
        """
        mgr = _StubAsyncLLM()
        obj = _StubReqObj(rid="r-cancel-1", stream=True)
        state = _fresh_state(obj)
        mgr.rid_to_state[obj.rid] = state

        # Drive the generator via a task so we can cancel it.
        gen = mgr._wait_one_response(obj)

        async def drain() -> None:
            async for _ in gen:
                pass

        task = asyncio.create_task(drain())

        # Give the generator one event-loop tick to enter
        # ``await state.event.wait()``.
        await asyncio.sleep(0)

        task.cancel()
        with pytest.raises((asyncio.CancelledError)):
            await task

        # Finally block cleanup; the abort is sent from its own task.
        assert (obj.rid) not in (mgr.rid_to_state)
        await asyncio.gather(*mgr._abort_tasks)
        assert mgr.frontend.aborts == [obj.rid]
        # A finished abort is not kept.
        await asyncio.sleep(0)
        assert not mgr._abort_tasks

    @pytest.mark.asyncio(loop_scope="function")
    async def test_normal_finish_does_not_fire_abort(self) -> None:
        """When the generator exits on ``state.finished`` (the engine's
        terminal frame), no abort should fire — the engine already knows the
        request is done.
        """
        mgr = _StubAsyncLLM()
        obj = _StubReqObj(rid="r-finish-1", stream=True)
        state = _fresh_state(obj)
        mgr.rid_to_state[obj.rid] = state

        gen = mgr._wait_one_response(obj)

        async def drive() -> list:
            out = []
            async for chunk in gen:
                out.append(chunk)
            return out

        task = asyncio.create_task(drive())
        await asyncio.sleep(0)

        # Simulate a terminal frame from the scheduler: mark finished,
        # put the final out_dict into the collector, wake the generator.
        state.finished = True
        state.collector.put(
            {
                "text": "hello",
                "output_ids": [1, 2],
                "meta_info": {"id": obj.rid, "finish_reason": None},
            },
            stream=True,
            cumulative=False,
        )
        state.event.set()

        results = await task

        assert (len(results)) == (1)
        assert (results[0]["text"]) == ("hello")
        assert (obj.rid) not in (mgr.rid_to_state)
        # Normal finish ⇒ no abort.
        assert not mgr._abort_tasks
        assert mgr.frontend.aborts == []


class TestAbortTasks:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_finished_aborts_are_not_kept(self) -> None:
        """Every abort runs on a task of its own; once it has reached the
        engine nothing of it stays behind, however many requests ended."""
        mgr = _StubAsyncLLM()
        for i in range(1000):
            mgr._abort_on_engine(f"r{i}")
        await asyncio.gather(*mgr._abort_tasks)
        await asyncio.sleep(0)
        assert len(mgr.frontend.aborts) == 1000
        assert not mgr._abort_tasks


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
