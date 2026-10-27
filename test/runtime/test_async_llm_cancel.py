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
