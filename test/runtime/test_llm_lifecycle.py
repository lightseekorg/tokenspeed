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

"""CPU regressions for the real LLM owner thread and private asyncio loop.

Each scenario runs in a bounded subprocess, so a broken shutdown cannot hang
the test runner. Only the two annotation imports are stubbed; the production
LLM class, event loop, tasks, async generators, and threads are all real.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import pathlib
import subprocess
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")


def _load_llm():
    async_module = types.ModuleType("tokenspeed.runtime.engine.async_llm")
    async_module.AsyncLLM = object
    io_module = types.ModuleType("tokenspeed.runtime.engine.io_struct")
    io_module.GenerateReqInput = object
    source = (
        pathlib.Path(__file__).resolve().parents[2]
        / "python"
        / "tokenspeed"
        / "runtime"
        / "engine"
        / "llm.py"
    )
    spec = importlib.util.spec_from_file_location("_llm_lifecycle_under_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            async_module.__name__: async_module,
            io_module.__name__: io_module,
        },
    ):
        spec.loader.exec_module(module)
    return module.LLM


def _shutdown_finalizes_pending_work() -> None:
    finalized = []
    nested_finalized = []

    class NestedBatchAsyncLLM:
        async def generate_request(self, request):
            async for output in self.handle_batch(request):
                yield output

        async def handle_batch(self, request):
            streams = [self.wait_one_response(index) for index in range(3)]
            yield await asyncio.gather(*(stream.__anext__() for stream in streams))

        async def wait_one_response(self, index):
            try:
                yield {"index": index}
            finally:
                await asyncio.sleep(0)
                nested_finalized.append(index)

    llm = _load_llm()(NestedBatchAsyncLLM())

    async def background() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            # A cancel-only shutdown must not pass by merely entering finally.
            await asyncio.sleep(0)
            finalized.append("task")

    async def partial_stream():
        try:
            yield {"text": "first"}
        finally:
            await asyncio.sleep(0)
            finalized.append("generator")

    async def start():
        task = asyncio.create_task(background())
        stream = partial_stream()
        await stream.__anext__()
        await asyncio.sleep(0)
        return task, stream

    # Keep both references alive: garbage collection must not close the stream
    # for us, and no unittest/asyncio runner performs implicit cleanup.
    task, stream = llm.run(start())
    # Match AsyncLLM's nested batch shape: generate() takes one outer result,
    # leaving three inner generators to be finalized through the outer chain.
    assert llm.generate(object()) == [{"index": index} for index in range(3)]
    llm.shutdown()
    state = {
        "finalized": finalized,
        "task_done": task.done(),
        "task_cancelled": task.cancelled(),
        "generator_closed": stream.ag_frame is None,
        "loop_closed": llm._loop.is_closed(),
        "thread_alive": llm._thread.is_alive(),
        "nested_finalized": sorted(nested_finalized),
    }
    print(json.dumps(state), flush=True)
    assert finalized == ["task", "generator"], state
    assert task.done() and task.cancelled(), state
    assert stream.ag_frame is None, state
    assert llm._loop.is_closed(), state
    assert not llm._thread.is_alive(), state
    assert sorted(nested_finalized) == [0, 1, 2], state
    llm.shutdown()


def _concurrent_shutdown_waits_for_cleanup() -> None:
    llm = _load_llm()(types.SimpleNamespace())
    cleanup_entered = threading.Event()
    failures = []
    finished = []

    async def start():
        finish_cleanup = asyncio.Event()

        async def background() -> None:
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_entered.set()
                await finish_cleanup.wait()

        task = asyncio.create_task(background())
        await asyncio.sleep(0)
        return task, finish_cleanup

    task, finish_cleanup = llm.run(start())

    def shutdown() -> None:
        try:
            llm.shutdown()
        except BaseException as exc:
            failures.append(exc)
        else:
            finished.append(True)

    first = threading.Thread(target=shutdown)
    first.start()
    assert cleanup_entered.wait(timeout=2), "shutdown never cancelled pending work"
    ready = threading.Barrier(4)

    def simultaneous_shutdown() -> None:
        ready.wait(timeout=2)
        shutdown()

    others = [threading.Thread(target=simultaneous_shutdown) for _ in range(3)]
    for caller in others:
        caller.start()
    ready.wait(timeout=2)
    assert not finished, "shutdown returned before asynchronous cleanup finished"
    llm._loop.call_soon_threadsafe(finish_cleanup.set)
    for caller in [first, *others]:
        caller.join(timeout=2)
        assert not caller.is_alive(), "concurrent shutdown caller did not return"
    assert not failures, failures
    assert len(finished) == 4, finished
    assert task.done() and task.cancelled()
    assert llm._loop.is_closed()
    assert not llm._thread.is_alive()
    # The post-close case also exercises shutdown's check/close race boundary.
    llm.shutdown()


def _run_and_stream_preserve_results_and_exceptions() -> None:
    failure = ValueError("test generation failed")

    class AsyncGenerator:
        async def generate_request(self, request):
            yield {"text": "first"}
            if request == "fail":
                raise failure
            yield {"text": "last"}

    llm = _load_llm()(AsyncGenerator())
    try:

        async def fail():
            raise failure

        try:
            llm.run(fail())
        except ValueError as exc:
            assert exc is failure
        else:
            raise AssertionError("run swallowed the coroutine exception")

        # Exhaustion needs the terminal sentinel; a missing sentinel hangs the
        # child and is reported by the parent's subprocess timeout.
        assert list(llm.generate_stream("normal")) == [
            {"text": "first"},
            {"text": "last"},
        ]
        stream = llm.generate_stream("fail")
        assert next(stream) == {"text": "first"}
        try:
            next(stream)
        except ValueError as exc:
            assert exc is failure
        else:
            raise AssertionError("stream swallowed the generation exception")
    finally:
        llm.shutdown()


def _shutdown_rejects_submissions_without_leaking_coroutines() -> None:
    llm = _load_llm()(types.SimpleNamespace())
    llm.shutdown()

    async def work():
        raise AssertionError("work started after shutdown")

    coro = work()
    try:
        llm.run(coro)
    except RuntimeError:
        pass
    else:
        raise AssertionError("run accepted work after shutdown")
    assert coro.cr_frame is None, "rejected coroutine was not closed"

    try:
        next(llm.generate_stream(object()))
    except RuntimeError:
        pass
    else:
        raise AssertionError("generate_stream accepted work after shutdown")


def _cleanup_failure_is_reported_after_closing_the_loop() -> None:
    llm = _load_llm()(types.SimpleNamespace())
    failure = ValueError("test background task cleanup failure")
    finalized = []

    async def background() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            raise failure

    async def partial_stream():
        try:
            yield {"text": "first"}
        finally:
            await asyncio.sleep(0)
            finalized.append("generator")

    async def start():
        task = asyncio.create_task(background())
        stream = partial_stream()
        await stream.__anext__()
        await asyncio.sleep(0)
        return task, stream

    task, stream = llm.run(start())
    try:
        llm.shutdown()
    except RuntimeError as exc:
        assert exc.__cause__ is failure, exc
    else:
        raise AssertionError("shutdown hid owner-thread cleanup failure")
    assert task.done() and task.exception() is failure
    assert stream.ag_frame is None
    assert finalized == ["generator"]
    assert llm._loop.is_closed()
    assert not llm._thread.is_alive()


def _async_generator_cleanup_failure_preserves_exception_handler() -> None:
    llm = _load_llm()(types.SimpleNamespace())
    failure = ValueError("test retained async generator cleanup failure")
    finalized = []
    handled = []

    def previous_handler(loop, context) -> None:
        handled.append((loop, context))

    async def partial_stream():
        try:
            yield {"text": "first"}
        finally:
            await asyncio.sleep(0)
            finalized.append("generator")
            raise failure

    async def start():
        llm._loop.set_exception_handler(previous_handler)
        stream = partial_stream()
        await stream.__anext__()
        return stream

    # Retaining the generator ensures shutdown_asyncgens owns its finalization,
    # rather than a garbage-collection aclose task being part of all_tasks().
    stream = llm.run(start())
    assert llm._loop.get_exception_handler() is previous_handler
    shutdown_error = None
    try:
        llm.shutdown()
    except RuntimeError as exc:
        shutdown_error = exc
    state = {
        "shutdown_raised": shutdown_error is not None,
        "cause_preserved": (
            shutdown_error is not None and shutdown_error.__cause__ is failure
        ),
        "finalized": finalized,
        "generator_closed": stream.ag_frame is None,
        "handler_calls": len(handled),
        "handler_restored": llm._loop.get_exception_handler() is previous_handler,
        "loop_closed": llm._loop.is_closed(),
        "thread_alive": llm._thread.is_alive(),
    }
    print(json.dumps(state), flush=True)
    assert shutdown_error is not None, state
    assert shutdown_error.__cause__ is failure, state
    assert finalized == ["generator"], state
    assert stream.ag_frame is None, state
    assert len(handled) == 1, state
    assert handled[0][0] is llm._loop
    assert handled[0][1]["exception"] is failure
    assert handled[0][1]["asyncgen"] is stream
    assert llm._loop.get_exception_handler() is previous_handler, state
    assert llm._loop.is_closed(), state
    assert not llm._thread.is_alive(), state


def _noncooperative_task_reports_timeout_and_can_finish_later() -> None:
    llm = _load_llm()(types.SimpleNamespace())

    async def start():
        release = asyncio.Event()

        async def background() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # Model a coroutine that delays cancellation until its own
                # cleanup can finish, without blocking the asyncio loop.
                await release.wait()

        task = asyncio.create_task(background())
        await asyncio.sleep(0)
        return task, release

    task, release = llm.run(start())
    started = time.monotonic()
    try:
        try:
            llm.shutdown()
        except RuntimeError:
            pass
        else:
            raise AssertionError("shutdown reported success with its thread alive")
        assert time.monotonic() - started >= 4.0, "shutdown skipped its grace period"
        assert llm._thread.is_alive()
        assert not llm._loop.is_closed()
    finally:
        llm._loop.call_soon_threadsafe(release.set)
        llm._thread.join(timeout=2)
    llm.shutdown()
    assert task.done()
    assert llm._loop.is_closed()
    assert not llm._thread.is_alive()


def _stream_cancelled_before_first_step_wakes_consumer() -> None:
    started = []

    class AsyncGenerator:
        async def generate_request(self, request):
            started.append(True)
            yield {"text": "unexpected"}

    llm = _load_llm()(AsyncGenerator())
    blocked = threading.Event()
    release_loop = threading.Event()
    submitted = threading.Event()
    stop_queued = threading.Event()
    consumer_errors = []
    shutdown_errors = []
    results = []

    def block_loop() -> None:
        blocked.set()
        assert release_loop.wait(timeout=5), "test did not release the owner loop"

    llm._loop.call_soon_threadsafe(block_loop)
    assert blocked.wait(timeout=2), "owner loop did not reach the test barrier"
    submit = llm._submit
    schedule = llm._loop.call_soon_threadsafe

    def observed_submit(coro):
        future = submit(coro)
        submitted.set()
        return future

    def observed_schedule(callback, *args, **kwargs):
        handle = schedule(callback, *args, **kwargs)
        if callback == llm._loop.stop:
            stop_queued.set()
        return handle

    def consume() -> None:
        try:
            results.append(next(llm.generate_stream(object())))
        except BaseException as exc:
            consumer_errors.append(exc)

    def shutdown() -> None:
        try:
            llm.shutdown()
        except BaseException as exc:
            shutdown_errors.append(exc)

    with (
        patch.object(llm, "_submit", observed_submit),
        patch.object(llm._loop, "call_soon_threadsafe", observed_schedule),
    ):
        consumer = threading.Thread(target=consume, daemon=True)
        closer = threading.Thread(target=shutdown, daemon=True)
        try:
            consumer.start()
            assert submitted.wait(timeout=2), "stream did not submit its drain"
            closer.start()
            assert stop_queued.wait(timeout=2), "shutdown did not enqueue loop.stop"
        finally:
            release_loop.set()
        closer.join(timeout=2)
        consumer.join(timeout=2)
    assert not closer.is_alive(), "shutdown did not complete"
    assert not consumer.is_alive(), "cancelled stream left its consumer blocked"
    assert not shutdown_errors, shutdown_errors
    assert not started, "test did not cancel before the drain's first step"
    assert not results, results
    assert len(consumer_errors) == 1, consumer_errors
    assert isinstance(consumer_errors[0], asyncio.CancelledError), consumer_errors
    assert llm._loop.is_closed()
    assert not llm._thread.is_alive()


_SCENARIOS = {
    "pending_work": _shutdown_finalizes_pending_work,
    "concurrent_shutdown": _concurrent_shutdown_waits_for_cleanup,
    "results_and_exceptions": _run_and_stream_preserve_results_and_exceptions,
    "rejected_submissions": _shutdown_rejects_submissions_without_leaking_coroutines,
    "cleanup_failure": _cleanup_failure_is_reported_after_closing_the_loop,
    "asyncgen_cleanup_failure": (
        _async_generator_cleanup_failure_preserves_exception_handler
    ),
    "shutdown_timeout": _noncooperative_task_reports_timeout_and_can_finish_later,
    "stream_cancelled_before_start": _stream_cancelled_before_first_step_wakes_consumer,
}


class TestLLMLifecycle(unittest.TestCase):
    def _run_scenario(self, name: str) -> None:
        result = subprocess.run(
            [sys.executable, __file__, "--scenario", name],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        output = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        self.assertEqual(result.returncode, 0, output)
        self.assertNotIn("Task was destroyed but it is pending", result.stderr, output)
        self.assertNotIn("was never awaited", result.stderr, output)
        self.assertNotIn("Exception in thread", result.stderr, output)

    def test_shutdown_awaits_tasks_and_finalizes_async_generators(self) -> None:
        self._run_scenario("pending_work")

    def test_concurrent_shutdown_waits_for_cleanup_and_is_idempotent(self) -> None:
        self._run_scenario("concurrent_shutdown")

    def test_run_and_stream_preserve_results_exceptions_and_terminal(self) -> None:
        self._run_scenario("results_and_exceptions")

    def test_shutdown_rejects_new_work_without_leaking_coroutines(self) -> None:
        self._run_scenario("rejected_submissions")

    def test_shutdown_reports_cleanup_failure_and_closes_loop(self) -> None:
        self._run_scenario("cleanup_failure")

    def test_asyncgen_cleanup_failure_preserves_handler_and_closes_loop(self) -> None:
        self._run_scenario("asyncgen_cleanup_failure")

    def test_shutdown_times_out_for_noncooperative_task(self) -> None:
        self._run_scenario("shutdown_timeout")

    def test_stream_cancelled_before_start_wakes_consumer(self) -> None:
        self._run_scenario("stream_cancelled_before_start")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--scenario":
        _SCENARIOS[sys.argv[2]]()
    else:
        unittest.main(verbosity=2)
