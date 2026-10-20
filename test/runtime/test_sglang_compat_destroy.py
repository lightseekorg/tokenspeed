"""Tests for the SGLang-compat ``/destroy_weights_update_group`` handler.

Regression guard: this endpoint used to be a hardcoded no-op
(``{"success": True, "message": "noop"}``). It now drives a real teardown:
HTTP -> ``AsyncLLM.destroy_weights_update_group`` -> worker
``ModelRunner.destroy_weights_update_group`` -> torch ``destroy_process_group``.

These run CPU-only against a stub AsyncLLM -- no engine, NCCL, or GPU needed.
"""

import os
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.engine.io_struct import (  # noqa: E402
    DestroyWeightsUpdateGroupReqInput,
)
from tokenspeed.runtime.entrypoints.sglang_compat_http import (  # noqa: E402
    build_sglang_compat_app,
)


class _StubLLM:
    """Minimal stand-in for AsyncLLM that records the destroy call."""

    def __init__(self, result):
        self.server_args = SimpleNamespace(rl_control_api_key=None)
        self._result = result
        self.calls = []

    async def destroy_weights_update_group(self, obj):
        self.calls.append(obj)
        return self._result


class TestDestroyWeightsUpdateGroupHandlerTest:
    def _client(self, result):
        llm = _StubLLM(result)
        return TestClient(build_sglang_compat_app(llm)), llm

    def test_invokes_real_teardown_with_group_name(self):
        client, llm = self._client((True, "weight update group destroyed"))
        resp = client.post("/destroy_weights_update_group", json={"group_name": "g1"})

        assert (resp.status_code) == (200)
        assert (resp.json()) == (
            {"success": True, "message": "weight update group destroyed"}
        )
        # No longer a no-op: the request reached AsyncLLM with the asked group.
        assert (len(llm.calls)) == (1)
        assert isinstance((llm.calls[0]), (DestroyWeightsUpdateGroupReqInput))
        assert (llm.calls[0].group_name) == ("g1")

    def test_empty_body_defaults_group_name(self):
        # slime/verl may POST destroy with no body; it must not 4xx.
        client, llm = self._client((True, "weight update group not initialized"))
        resp = client.post("/destroy_weights_update_group")

        assert (resp.status_code) == (200)
        assert resp.json()["success"]
        assert (llm.calls[0].group_name) == ("weight_update_group")

    def test_failure_maps_to_400(self):
        client, _ = self._client((False, "boom"))
        resp = client.post("/destroy_weights_update_group", json={})

        assert (resp.status_code) == (400)
        assert (resp.json()) == ({"success": False, "message": "boom"})


class TestModelRunnerDestroyIdempotentTest:
    def test_destroy_without_live_group_is_success(self):
        # The path slime hits when destroy is called with no prior init (or
        # twice): no live group -> success, no torch/NCCL work required.
        from tokenspeed.runtime.execution.model_runner import ModelRunner

        runner = object.__new__(ModelRunner)  # bypass __init__/model load
        runner._weight_update_pg = None
        ok, msg = runner.destroy_weights_update_group(None)

        assert ok
        assert ("not initialized") in (msg)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
