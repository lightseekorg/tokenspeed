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
import unittest

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")

import asyncio  # noqa: E402
import socket  # noqa: E402
import types  # noqa: E402
from typing import Any, Dict  # noqa: E402
from unittest.mock import patch  # noqa: E402

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


class TestWaitOneResponseCancellation(unittest.IsolatedAsyncioTestCase):
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
        with self.assertRaises(asyncio.CancelledError):
            await task

        # Finally block cleanup.
        self.assertNotIn(obj.rid, mgr.rid_to_state)
        aborts = mgr.engine_core_client.send_to_scheduler.aborts
        self.assertEqual(
            len(aborts),
            1,
            f"expected exactly one AbortReq on cancel, got {len(aborts)}",
        )
        # The AbortReq should carry the right rid.
        self.assertEqual(getattr(aborts[0], "rid", None), obj.rid)

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
        )
        state.event.set()

        results = await task

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "hello")
        self.assertNotIn(obj.rid, mgr.rid_to_state)
        # Normal finish ⇒ no abort.
        self.assertEqual(
            len(mgr.engine_core_client.send_to_scheduler.aborts),
            0,
            "normal finish should not schedule an AbortReq",
        )


class TestAsyncLLMClose(unittest.IsolatedAsyncioTestCase):
    async def test_close_terminates_owned_scheduler_processes(self) -> None:
        mgr = AsyncLLM.__new__(AsyncLLM)
        mgr._closed = False
        mgr._gracefully_exit = False
        mgr._rl_control_server = None
        mgr._rl_control_task = None
        mgr._close_task = None
        mgr.asyncio_tasks = set()

        with patch(
            "tokenspeed.runtime.engine.async_llm.kill_process_tree"
        ) as kill_tree:
            await mgr.close(timeout=1.0)

        kill_tree.assert_called_once_with(os.getpid(), include_parent=False)

    async def test_close_stops_rl_server_and_cancels_owned_tasks(self) -> None:
        mgr = AsyncLLM.__new__(AsyncLLM)
        mgr._closed = False
        mgr._gracefully_exit = False
        server = types.SimpleNamespace(should_exit=False)
        mgr._rl_control_server = server
        mgr._close_task = None

        async def run_rl_server() -> None:
            while not server.should_exit:
                await asyncio.sleep(0)

        async def run_background() -> None:
            await asyncio.Event().wait()

        rl_task = asyncio.create_task(run_rl_server())
        background_task = asyncio.create_task(run_background())
        mgr._rl_control_task = rl_task
        mgr.asyncio_tasks = {background_task}

        await mgr.close(timeout=1.0, terminate_processes=False)

        self.assertTrue(server.should_exit)
        self.assertTrue(rl_task.done())
        self.assertTrue(background_task.cancelled())
        self.assertEqual(mgr.asyncio_tasks, set())
        self.assertIsNone(mgr._rl_control_task)

        # Idempotent: a second close is a no-op rather than re-cancelling tasks.
        await mgr.close(timeout=1.0, terminate_processes=False)

    async def test_close_absorbs_already_cancelled_rl_task(self) -> None:
        mgr = AsyncLLM.__new__(AsyncLLM)
        mgr._closed = False
        mgr._gracefully_exit = False
        mgr._rl_control_server = None
        mgr._close_task = None

        async def wait_forever() -> None:
            await asyncio.Event().wait()

        rl_task = asyncio.create_task(wait_forever())
        rl_task.cancel()
        await asyncio.gather(rl_task, return_exceptions=True)
        background_task = asyncio.create_task(wait_forever())
        mgr._rl_control_task = rl_task
        mgr.asyncio_tasks = {background_task}

        await mgr.close(timeout=1.0, terminate_processes=False)

        self.assertTrue(mgr._closed)
        self.assertIsNone(mgr._rl_control_task)
        self.assertTrue(background_task.cancelled())
        self.assertEqual(mgr.asyncio_tasks, set())

    async def test_close_finishes_cleanup_before_propagating_caller_cancel(
        self,
    ) -> None:
        mgr = AsyncLLM.__new__(AsyncLLM)
        mgr._closed = False
        mgr._gracefully_exit = False
        mgr._rl_control_server = None
        mgr._close_task = None

        async def wait_forever() -> None:
            await asyncio.Event().wait()

        rl_task = asyncio.create_task(wait_forever())
        background_task = asyncio.create_task(wait_forever())
        mgr._rl_control_task = rl_task
        mgr.asyncio_tasks = {background_task}

        close_task = asyncio.create_task(
            mgr.close(timeout=0.01, terminate_processes=False)
        )
        await asyncio.sleep(0)
        close_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await close_task

        self.assertTrue(mgr._closed)
        self.assertTrue(rl_task.cancelled())
        self.assertTrue(background_task.cancelled())
        self.assertIsNone(mgr._rl_control_task)
        self.assertEqual(mgr.asyncio_tasks, set())

    async def test_close_gracefully_stops_real_rl_uvicorn(self) -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]

        mgr = AsyncLLM.__new__(AsyncLLM)
        mgr._closed = False
        mgr._gracefully_exit = False
        mgr._rl_control_server = None
        mgr._close_task = None
        mgr.asyncio_tasks = set()
        mgr.server_args = types.SimpleNamespace(host="127.0.0.1")

        task = asyncio.create_task(mgr._serve_rl_control_plane(object(), port))
        mgr._rl_control_task = task
        deadline = asyncio.get_running_loop().time() + 5.0
        while mgr._rl_control_server is None or not mgr._rl_control_server.started:
            self.assertLess(asyncio.get_running_loop().time(), deadline)
            await asyncio.sleep(0.01)

        await mgr.close(timeout=5.0, terminate_processes=False)

        self.assertTrue(task.done())
        self.assertIsNone(mgr._rl_control_server)
        with socket.socket() as probe:
            self.assertNotEqual(probe.connect_ex(("127.0.0.1", port)), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
