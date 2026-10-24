"""Regression guard for ``AsyncLLM._wait_one_response`` cancellation.

Locks the contract Phase G.1 established: when the task driving a
streaming generator is cancelled (e.g. FastAPI cancelling its route
coroutine because the client disconnected), the generator's
``finally`` drops the rid from ``rid_to_state`` and fires exactly one
``AbortReq`` at the scheduler. No dangling per-request state. No
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
from typing import Any, Dict  # noqa: E402

from tokenspeed.runtime.engine.async_llm import AsyncLLM  # noqa: E402
from tokenspeed.runtime.engine.collector import RequestOutputCollector  # noqa: E402
from tokenspeed.runtime.engine.output_processor import ReqState  # noqa: E402


class _FakeScheduler:
    """Captures ``AbortReq``s the stub AsyncLLM pushes on cancellation."""

    def __init__(self) -> None:
        self.aborts: list = []

    def send_pyobj(self, obj: Any) -> None:
        self.aborts.append(obj)


class _StubAsyncLLM(AsyncLLM):
    """Bypass ZMQ + ModelConfig + HF bring-up.

    Populates only the attributes ``_wait_one_response`` + ``abort_request``
    read: ``rid_to_state``, ``log_requests``, ``enable_metrics``,
    ``tokenizer``, ``model_config``, and an ``engine_core_client`` whose
    ``send_to_scheduler`` is a capture fake so the test can assert the
    ``AbortReq`` was sent.
    """

    def __init__(self) -> None:
        self.rid_to_state: Dict[str, ReqState] = {}
        self.log_requests = False
        self.enable_metrics = False
        self.tokenizer = None
        self.model_config = types.SimpleNamespace(is_multimodal_gen=False)
        self.engine_core_client = types.SimpleNamespace(
            send_to_scheduler=_FakeScheduler()
        )


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
        event. Expect: rid dropped from ``rid_to_state``; one ``AbortReq``
        sent to the scheduler.
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

        # Finally block cleanup.
        assert (obj.rid) not in (mgr.rid_to_state)
        aborts = mgr.engine_core_client.send_to_scheduler.aborts
        _pytest_first_1 = len(aborts)
        _pytest_second_2 = 1
        _pytest_msg_3 = f"expected exactly one AbortReq on cancel, got {len(aborts)}"
        assert _pytest_first_1 == _pytest_second_2, _pytest_msg_3
        # The AbortReq should carry the right rid.
        assert (getattr(aborts[0], "rid", None)) == (obj.rid)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_normal_finish_does_not_fire_abort(self) -> None:
        """When the generator exits on ``state.finished`` (scheduler's
        terminal frame), no ``AbortReq`` should fire — the scheduler
        already knows the request is done.
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
        assert (len(mgr.engine_core_client.send_to_scheduler.aborts)) == (
            0
        ), "normal finish should not schedule an AbortReq"


class _StubBatchAsyncLLM(_StubAsyncLLM):
    """Stub whose per-request send registers a live ``ReqState`` instead of
    tokenizing and talking to the scheduler."""

    async def _tokenize_one_request(self, obj: Any) -> None:
        return None

    def _send_one_request(self, obj: Any, tokenized_obj: Any, created_time) -> None:
        self.rid_to_state[obj.rid] = _fresh_state(obj)


class _StubBatchReqObj:
    """A batch of ``batch_size`` streaming requests, indexable like
    ``GenerateReqInput``."""

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.parallel_sample_num = 1
        self.stream = True

    def __getitem__(self, index: int) -> _StubReqObj:
        return _StubReqObj(rid=f"r{index}", stream=True)


class TestBatchStreamCancellation:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_cancelled_batch_stream_aborts_every_request(self) -> None:
        """Cancel a streaming batch while it waits for its first outputs.
        Expect: every sub-request is dropped from ``rid_to_state`` and gets
        an ``AbortReq``, as for a single request."""
        mgr = _StubBatchAsyncLLM()
        gen = mgr._handle_batch_request(_StubBatchReqObj(batch_size=2))

        task = asyncio.create_task(gen.__anext__())
        # Let the batch send its requests and block waiting for outputs.
        for _ in range(5):
            await asyncio.sleep(0)
        assert set(mgr.rid_to_state) == {"r0", "r1"}

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert mgr.rid_to_state == {}
        aborts = mgr.engine_core_client.send_to_scheduler.aborts
        assert sorted(abort.rid for abort in aborts) == ["r0", "r1"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_closed_batch_stream_aborts_every_request(self) -> None:
        """The consumer stops iterating after the first chunk (for example a
        client disconnect closes the generator): the other sub-requests must
        be aborted too."""
        mgr = _StubBatchAsyncLLM()
        gen = mgr._handle_batch_request(_StubBatchReqObj(batch_size=2))

        task = asyncio.create_task(gen.__anext__())
        for _ in range(5):
            await asyncio.sleep(0)

        # Request r0 produces its first (non-final) chunk.
        state = mgr.rid_to_state["r0"]
        state.collector.put(
            {"text": "a", "output_ids": [1], "meta_info": {"id": "r0"}},
            stream=True,
            cumulative=False,
        )
        state.event.set()
        chunk = await task
        assert chunk["index"] == 0

        await gen.aclose()

        assert mgr.rid_to_state == {}
        aborts = mgr.engine_core_client.send_to_scheduler.aborts
        assert sorted(abort.rid for abort in aborts) == ["r0", "r1"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
