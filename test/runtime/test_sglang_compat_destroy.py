"""Tests for the SGLang-compat ``/destroy_weights_update_group`` handler.

Regression guard: this endpoint used to be a hardcoded no-op
(``{"success": True, "message": "noop"}``). It now drives a real teardown:
HTTP -> ``AsyncLLM.destroy_weights_update_group`` -> worker
``ModelRunner.destroy_weights_update_group`` -> torch ``destroy_process_group``.

These run CPU-only against a stub AsyncLLM -- no engine, NCCL, or GPU needed.
"""

import unittest

from fastapi.testclient import TestClient

from tokenspeed.runtime.engine.io_struct import DestroyWeightsUpdateGroupReqInput
from tokenspeed.runtime.entrypoints.sglang_compat_http import build_sglang_compat_app


class _StubLLM:
    """Minimal stand-in for AsyncLLM that records the destroy call."""

    def __init__(self, result):
        self._result = result
        self.calls = []

    async def destroy_weights_update_group(self, obj):
        self.calls.append(obj)
        return self._result


class DestroyWeightsUpdateGroupHandlerTest(unittest.TestCase):
    def _client(self, result):
        llm = _StubLLM(result)
        return TestClient(build_sglang_compat_app(llm)), llm

    def test_invokes_real_teardown_with_group_name(self):
        client, llm = self._client((True, "weight update group destroyed"))
        resp = client.post("/destroy_weights_update_group", json={"group_name": "g1"})

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            resp.json(),
            {"success": True, "message": "weight update group destroyed"},
        )
        # No longer a no-op: the request reached AsyncLLM with the asked group.
        self.assertEqual(len(llm.calls), 1)
        self.assertIsInstance(llm.calls[0], DestroyWeightsUpdateGroupReqInput)
        self.assertEqual(llm.calls[0].group_name, "g1")

    def test_empty_body_defaults_group_name(self):
        # slime/verl may POST destroy with no body; it must not 4xx.
        client, llm = self._client((True, "weight update group not initialized"))
        resp = client.post("/destroy_weights_update_group")

        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])
        self.assertEqual(llm.calls[0].group_name, "weight_update_group")

    def test_failure_maps_to_400(self):
        client, _ = self._client((False, "boom"))
        resp = client.post("/destroy_weights_update_group", json={})

        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json(), {"success": False, "message": "boom"})


class ModelRunnerDestroyIdempotentTest(unittest.TestCase):
    def test_destroy_without_live_group_is_success(self):
        # The path slime hits when destroy is called with no prior init (or
        # twice): no live group -> success, no torch/NCCL work required.
        from tokenspeed.runtime.execution.model_runner import ModelRunner

        runner = object.__new__(ModelRunner)  # bypass __init__/model load
        ok, msg = runner.destroy_weights_update_group(None)

        self.assertTrue(ok)
        self.assertIn("not initialized", msg)


if __name__ == "__main__":
    unittest.main()
