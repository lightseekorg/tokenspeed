"""Regression coverage for node-local multimodal SHM transport."""

import os
import sys
import types
import unittest

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")

from tokenspeed.runtime.engine.async_llm import AsyncLLM  # noqa: E402


class _FakeMultimodalInputs:
    def __init__(self) -> None:
        self.publish_calls = 0

    def publish_shm_features(self) -> None:
        self.publish_calls += 1


class _CaptureSender:
    def __init__(self) -> None:
        self.sent = []

    def send_pyobj(self, obj) -> None:
        self.sent.append(obj)


def _stub_engine(nnodes: int) -> tuple[AsyncLLM, _CaptureSender]:
    engine = AsyncLLM.__new__(AsyncLLM)
    engine.rid_to_state = {}
    engine.server_args = types.SimpleNamespace(
        mapping=types.SimpleNamespace(nnodes=nnodes)
    )
    sender = _CaptureSender()
    engine.engine_core_client = types.SimpleNamespace(send_to_scheduler=sender)
    return engine, sender


class TestAsyncLLMMultimodalTransport(unittest.TestCase):
    def _send(self, nnodes: int):
        engine, sender = _stub_engine(nnodes)
        mm_inputs = _FakeMultimodalInputs()
        request = types.SimpleNamespace(rid=f"image-{nnodes}")
        tokenized = types.SimpleNamespace(
            created_time=1.0,
            multimodal_inputs=mm_inputs,
        )

        engine._send_one_request(request, tokenized, created_time=0.5)
        self.assertEqual(sender.sent, [tokenized])
        self.assertIn(request.rid, engine.rid_to_state)
        return mm_inputs

    def test_single_node_publishes_features_to_shm(self) -> None:
        mm_inputs = self._send(nnodes=1)
        self.assertEqual(mm_inputs.publish_calls, 1)

    def test_multi_node_keeps_features_inline(self) -> None:
        mm_inputs = self._send(nnodes=2)
        self.assertEqual(mm_inputs.publish_calls, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
