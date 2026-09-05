"""Request-state lifecycle tests for cancellation and abort."""

import os
import sys
import unittest
from unittest.mock import patch

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")

import asyncio  # noqa: E402
import contextlib  # noqa: E402
import types  # noqa: E402
from typing import Any, Dict  # noqa: E402

from tokenspeed.runtime.engine.async_llm import AsyncLLM  # noqa: E402
from tokenspeed.runtime.engine.collector import RequestOutputCollector  # noqa: E402
from tokenspeed.runtime.engine.io_struct import (  # noqa: E402
    AbortReq,
    BatchEmbeddingOut,
)
from tokenspeed.runtime.engine.output_processor import (  # noqa: E402
    OutputProcessor,
    ReqState,
)


class _FakeScheduler:
    def __init__(self) -> None:
        self.aborts: list = []

    def send_pyobj(self, obj: Any) -> None:
        self.aborts.append(obj)


class _StubAsyncLLM(AsyncLLM):
    def __init__(self) -> None:
        self.rid_to_state: Dict[str, ReqState] = {}
        self.log_requests = False
        self.enable_metrics = False
        self.dump_requests_folder = False
        self.tokenizer = None
        self.model_config = types.SimpleNamespace(is_multimodal_gen=False)
        self.server_args = types.SimpleNamespace(
            speculative_algorithm=None,
            weight_version="test",
        )
        self.engine_core_client = types.SimpleNamespace(
            send_to_scheduler=_FakeScheduler()
        )
        self.output_processor = OutputProcessor(self)


class _StubReqObj:
    def __init__(self, *, rid: str = "r1", stream: bool = True, input_ids=None) -> None:
        self.rid = rid
        self.stream = stream
        self.input_ids = input_ids
        self.text = None
        self.sampling_params = {"skip_special_tokens": False}


class _StubBatchObj:
    parallel_sample_num = 1
    stream = True

    def __init__(self) -> None:
        self.items = [_StubReqObj(rid="batch-0"), _StubReqObj(rid="batch-1")]
        self.batch_size = len(self.items)
        self.rid = [item.rid for item in self.items]

    def __getitem__(self, index: int) -> _StubReqObj:
        return self.items[index]


def _fresh_state(obj: _StubReqObj) -> ReqState:
    return ReqState(
        RequestOutputCollector(),
        False,
        asyncio.Event(),
        obj,
        created_time=0.0,
        dispatched=True,
    )


def _embedding_frame(rid: str, finish_reason=None) -> BatchEmbeddingOut:
    return BatchEmbeddingOut(
        rids=[rid],
        finished_reasons=[finish_reason],
        embeddings=[[1.0]],
        prompt_tokens=[1],
    )


class TestWaitOneResponseCancellation(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_waits_for_scheduler_terminal_ack(self) -> None:
        mgr = _StubAsyncLLM()
        obj = _StubReqObj(rid="r-cancel-1", stream=True)
        state = _fresh_state(obj)
        mgr.rid_to_state[obj.rid] = state

        gen = mgr._wait_one_response(obj)

        async def drain() -> None:
            async for _ in gen:
                pass

        task = asyncio.create_task(drain())

        await asyncio.sleep(0)

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertIn(obj.rid, mgr.rid_to_state)
        self.assertTrue(state.abandoned)
        self.assertTrue(state.abort_sent)
        aborts = mgr.engine_core_client.send_to_scheduler.aborts
        self.assertEqual(len(aborts), 1)
        self.assertEqual(getattr(aborts[0], "rid", None), obj.rid)

        mgr.output_processor.handle_batch_output(_embedding_frame(obj.rid))
        self.assertIn(obj.rid, mgr.rid_to_state)
        self.assertFalse(state.collector.has_pending())

        mgr.output_processor.handle_batch_output(
            _embedding_frame(obj.rid, {"type": "abort", "message": "aborted"})
        )
        self.assertNotIn(obj.rid, mgr.rid_to_state)

    async def test_abort_is_idempotent(self) -> None:
        mgr = _StubAsyncLLM()
        obj = _StubReqObj(rid="r-abort")
        mgr.rid_to_state[obj.rid] = _fresh_state(obj)

        mgr.abort_request(obj.rid)
        mgr.abort_request(obj.rid)

        self.assertIn(obj.rid, mgr.rid_to_state)
        self.assertEqual(len(mgr.engine_core_client.send_to_scheduler.aborts), 1)

    async def test_generate_cancel_after_dispatch_keeps_state(self) -> None:
        mgr = _StubAsyncLLM()
        mgr.auto_create_handle_loop = lambda: None
        mgr._generation_admit = asyncio.Event()
        mgr._generation_admit.set()
        mgr.input_processor = types.SimpleNamespace(validate_request=lambda obj: None)
        mgr.model_update_lock = types.SimpleNamespace(
            reader_lock=contextlib.nullcontext()
        )
        obj = _StubReqObj(rid="r-dispatch-race")
        obj.is_single = True
        obj.normalize_batch_and_arguments = lambda: None

        async def tokenize(_obj):
            return types.SimpleNamespace(created_time=0.0, multimodal_inputs=None)

        mgr._tokenize_one_request = tokenize
        task = asyncio.create_task(mgr.generate_request(obj).__anext__())
        for _ in range(10):
            if obj.rid in mgr.rid_to_state:
                break
            await asyncio.sleep(0)

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        state = mgr.rid_to_state[obj.rid]
        self.assertTrue(state.dispatched)
        self.assertTrue(state.abandoned)
        aborts = [
            item
            for item in mgr.engine_core_client.send_to_scheduler.aborts
            if isinstance(item, AbortReq)
        ]
        self.assertEqual(len(aborts), 1)

    async def test_normal_finish_does_not_fire_abort(self) -> None:
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
        self.assertEqual(len(mgr.engine_core_client.send_to_scheduler.aborts), 0)

    async def test_failure_cleanup_drops_pending_and_aborts_dispatched(self) -> None:
        mgr = _StubAsyncLLM()
        pending_obj = _StubReqObj(rid="pending")
        live_obj = _StubReqObj(rid="live")
        pending = _fresh_state(pending_obj)
        pending.dispatched = False
        live = _fresh_state(live_obj)
        mgr.rid_to_state = {"pending": pending, "live": live}

        mgr._release_req_states_on_failure({"pending", "live"})
        mgr._release_req_states_on_failure({"live"})

        self.assertNotIn("pending", mgr.rid_to_state)
        self.assertIn("live", mgr.rid_to_state)
        self.assertTrue(live.abandoned)
        self.assertEqual(len(mgr.engine_core_client.send_to_scheduler.aborts), 1)

    async def test_dispatch_failure_releases_local_state(self) -> None:
        mgr = _StubAsyncLLM()
        obj = _StubReqObj(rid="dispatch-failed")
        calls = []
        mm_inputs = types.SimpleNamespace(
            publish_shm_features=lambda: calls.append("publish"),
            release_shm_features=lambda: calls.append("release"),
        )
        tokenized = types.SimpleNamespace(
            created_time=0.0,
            multimodal_inputs=mm_inputs,
        )

        def fail_send(_obj):
            raise RuntimeError("send failed")

        mgr.engine_core_client.send_to_scheduler.send_pyobj = fail_send
        with self.assertRaisesRegex(RuntimeError, "send failed"):
            mgr._send_one_request(obj, tokenized)

        self.assertNotIn(obj.rid, mgr.rid_to_state)
        self.assertEqual(calls, ["publish", "release"])

    async def test_batch_cancel_closes_all_waiters(self) -> None:
        mgr = _StubAsyncLLM()
        batch = _StubBatchObj()

        async def tokenize(obj):
            return types.SimpleNamespace(
                rid=obj.rid,
                created_time=0.0,
                multimodal_inputs=None,
            )

        mgr._tokenize_one_request = tokenize
        gen = mgr._handle_batch_request(batch, request_rids=set(batch.rid))
        task = asyncio.create_task(gen.__anext__())
        for _ in range(10):
            if len(mgr.rid_to_state) == 2:
                break
            await asyncio.sleep(0)

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(set(mgr.rid_to_state), set(batch.rid))
        self.assertTrue(all(state.abandoned for state in mgr.rid_to_state.values()))
        aborts = [
            item
            for item in mgr.engine_core_client.send_to_scheduler.aborts
            if isinstance(item, AbortReq)
        ]
        self.assertEqual({item.rid for item in aborts}, set(batch.rid))

    async def test_mass_cancel_drains_without_deleted_state_errors(self) -> None:
        mgr = _StubAsyncLLM()
        tasks = []
        rids = [f"mass-{index}" for index in range(64)]
        for rid in rids:
            obj = _StubReqObj(rid=rid)
            mgr.rid_to_state[rid] = _fresh_state(obj)
            tasks.append(asyncio.create_task(mgr._wait_one_response(obj).__anext__()))

        await asyncio.sleep(0)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.assertEqual(set(mgr.rid_to_state), set(rids))

        terminal = BatchEmbeddingOut(
            rids=rids,
            finished_reasons=[{"type": "abort"}] * len(rids),
            embeddings=[[1.0]] * len(rids),
            prompt_tokens=[1] * len(rids),
        )
        with patch("tokenspeed.runtime.engine.output_processor.logger.error") as error:
            mgr.output_processor.handle_batch_output(terminal)

        error.assert_not_called()
        self.assertFalse(mgr.rid_to_state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
