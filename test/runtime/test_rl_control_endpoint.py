"""The in-engine RL control app as an external gateway sees it: where it
listens, what it advertises, and how it authenticates.

Run with: pytest test/runtime/test_rl_control_endpoint.py
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from fastapi.testclient import TestClient  # noqa: E402
from runtime.rl_fakes import FakeLLM  # noqa: E402

from tokenspeed.runtime.entrypoints import rl_control, sglang_compat_http  # noqa: E402
from tokenspeed.runtime.entrypoints.sglang_compat_http import (  # noqa: E402
    build_sglang_compat_app,
)
from tokenspeed.runtime.utils.server_args import ServerArgs  # noqa: E402


def _args(**overrides):
    base = {"host": "127.0.0.1", "rl_control_port": 40100, "rl_control_host": None}
    base.update(overrides)
    return SimpleNamespace(**base)


class TestControlEndpointHelpers(unittest.TestCase):
    def test_bind_host_defaults_to_engine_host(self):
        self.assertEqual(rl_control.control_bind_host(_args()), "127.0.0.1")
        self.assertEqual(
            rl_control.control_bind_host(_args(rl_control_host="0.0.0.0")), "0.0.0.0"
        )

    def test_control_url_uses_bound_host_and_port(self):
        self.assertEqual(rl_control.control_url(_args()), "http://127.0.0.1:40100")
        self.assertEqual(
            rl_control.control_url(_args(rl_control_host="10.0.0.5")),
            "http://10.0.0.5:40100",
        )
        self.assertEqual(
            rl_control.control_url(_args(host="::1")), "http://[::1]:40100"
        )

    def test_control_url_is_none_without_a_port(self):
        self.assertIsNone(rl_control.control_url(_args(rl_control_port=None)))
        self.assertIsNone(rl_control.control_url(_args(rl_control_port=0)))

    def test_capabilities_match_the_smg_label_contract(self):
        caps = rl_control.capabilities()
        self.assertEqual(caps["rl.pause_modes"], "wait,abort,keep")
        # Derived from the scheduler's dispatcher, which implements only the
        # distributed update path; the other two routes answer 501.
        self.assertEqual(caps["rl.update_from"], "distributed")
        for key in (
            "rl.abort",
            "rl.flush_cache",
            "rl.sleep_wake",
            "rl.reports_weight_version",
        ):
            self.assertEqual(caps[key], "true")
        self.assertNotIn("disk", caps["rl.update_from"])
        self.assertNotIn("tensor", caps["rl.update_from"])

    def test_advertisement_carries_url_and_capabilities(self):
        adv = rl_control.advertisement(_args())
        self.assertEqual(adv["rl.control_url"], "http://127.0.0.1:40100")
        self.assertEqual(adv["rl.abort"], "true")
        self.assertNotIn(
            "rl.control_url", rl_control.advertisement(_args(rl_control_port=None))
        )


class TestServerArgsFlags(unittest.TestCase):
    def test_flags_parse_and_default_to_none(self):
        import argparse

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        ns = parser.parse_args(["--model", "m"])
        self.assertIsNone(ns.rl_control_host)
        self.assertIsNone(ns.rl_control_api_key)
        ns = parser.parse_args(
            [
                "--model",
                "m",
                "--rl-control-host",
                "0.0.0.0",
                "--rl-control-api-key",
                "k",
            ]
        )
        self.assertEqual(ns.rl_control_host, "0.0.0.0")
        self.assertEqual(ns.rl_control_api_key, "k")


class _AuthLLM:
    def __init__(self, api_key):
        self.server_args = SimpleNamespace(
            weight_version="default",
            model="m",
            kvstore_storage_backend=None,
            rl_control_api_key=api_key,
        )


class TestBearerAuth(unittest.TestCase):
    def test_open_when_no_key_is_configured(self):
        client = TestClient(build_sglang_compat_app(_AuthLLM(None)))
        self.assertEqual(client.get("/get_weight_version").status_code, 200)

    def test_open_when_engine_has_no_server_args(self):
        # Some engine stand-ins (tests, minimal stubs) carry no server_args at
        # all; the app must still build and stay open rather than raising.
        client = TestClient(build_sglang_compat_app(object()))
        self.assertEqual(client.get("/health_generate").status_code, 200)

    def test_rejects_missing_or_wrong_bearer(self):
        client = TestClient(build_sglang_compat_app(_AuthLLM("s3cret")))
        resp = client.get("/get_weight_version")
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.json(), {"success": False, "message": "unauthorized"})
        resp = client.get(
            "/get_weight_version", headers={"Authorization": "Bearer nope"}
        )
        self.assertEqual(resp.status_code, 401)

    def test_accepts_the_configured_bearer(self):
        client = TestClient(build_sglang_compat_app(_AuthLLM("s3cret")))
        resp = client.get(
            "/get_weight_version", headers={"Authorization": "Bearer s3cret"}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"weight_version": "default"})


class TestRouteSemantics(unittest.TestCase):
    def _client(self):
        llm = FakeLLM()
        return llm, TestClient(build_sglang_compat_app(llm))

    def test_empty_body_is_400_on_body_routes(self):
        _llm, client = self._client()
        for route in (
            "/init_weights_update_group",
            "/update_weights_from_distributed",
            # /update_weights_from_{disk,tensor} answer 501 before the body is
            # read; see TestUnsupportedWeightUpdateSources.
            "/abort_request",
            "/update_weight_version",
        ):
            resp = client.post(route)
            self.assertEqual(resp.status_code, 400, route)
            self.assertFalse(resp.json()["success"], route)
            resp = client.post(
                route,
                content=b"{not json",
                headers={"content-type": "application/json"},
            )
            self.assertEqual(resp.status_code, 400, route)

    def test_pause_mode_is_honored_and_echoed(self):
        llm, client = self._client()
        self.assertEqual(client.post("/pause_generation").json()["mode"], "wait")
        self.assertEqual(
            client.post("/pause_generation", json={"mode": "keep"}).json()["mode"],
            "keep",
        )
        self.assertEqual(
            client.post("/pause_generation", json={"mode": "abort"}).status_code, 200
        )
        self.assertEqual(
            [m for kind, m in llm.scheduler_calls if kind == "pause"],
            ["wait", "keep", "abort"],
        )

    def test_invalid_pause_mode_is_400_and_reopens_admission(self):
        llm, client = self._client()
        resp = client.post("/pause_generation", json={"mode": "retract"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("invalid pause mode", resp.json()["message"])
        self.assertEqual(llm.scheduler_calls, [])
        self.assertEqual(llm.admission_calls, [])

    def test_flush_cache_accepts_get_and_post(self):
        _llm, client = self._client()
        self.assertEqual(client.get("/flush_cache").status_code, 200)
        self.assertEqual(client.post("/flush_cache").status_code, 200)


class TestUnsupportedWeightUpdateSources(unittest.TestCase):
    """Sources the scheduler has no branch for never reach it.

    Forwarding one raises ``NotImplementedError`` inside the scheduler process,
    which dies and takes the control app and the engine with it.
    """

    def _client(self):
        llm = FakeLLM()
        return llm, TestClient(build_sglang_compat_app(llm))

    def test_disk_is_501_and_forwards_nothing(self):
        llm, client = self._client()
        resp = client.post(
            "/update_weights_from_disk",
            json={"model_path": "/tmp/model", "weight_version": "v9"},
        )
        self.assertEqual(resp.status_code, 501)
        body = resp.json()
        self.assertFalse(body["success"])
        self.assertIn("update_weights_from_disk", body["message"])
        self.assertIn("distributed", body["message"])
        self.assertEqual(llm.updates, [])
        self.assertEqual(llm.server_args.weight_version, "default")

    def test_tensor_is_501_and_forwards_nothing(self):
        llm, client = self._client()
        resp = client.post(
            "/update_weights_from_tensor",
            json={
                "serialized_named_tensors": [],
                "load_format": None,
                "flush_cache": False,
                "weight_version": "v9",
            },
        )
        self.assertEqual(resp.status_code, 501)
        body = resp.json()
        self.assertFalse(body["success"])
        self.assertIn("update_weights_from_tensor", body["message"])
        self.assertIn("distributed", body["message"])
        self.assertEqual(llm.updates, [])
        self.assertEqual(llm.server_args.weight_version, "default")

    def test_distributed_is_untouched(self):
        llm, client = self._client()
        resp = client.post(
            "/update_weights_from_distributed",
            json={
                "names": ["weight"],
                "dtypes": ["float32"],
                "shapes": [[1]],
                "weight_version": "v9",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(llm.updates), 1)

    def test_guard_follows_the_scheduler_constant(self):
        # Data-driven, not hard-coded: widen the supported set and the disk
        # route reaches the engine again.
        llm, client = self._client()
        with mock.patch.object(
            sglang_compat_http,
            "SUPPORTED_WEIGHT_UPDATE_SOURCES",
            frozenset({"disk", "distributed"}),
        ):
            resp = client.post(
                "/update_weights_from_disk", json={"model_path": "/tmp/model"}
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            [type(obj).__name__ for obj in llm.updates],
            ["UpdateWeightFromDiskReqInput"],
        )


if __name__ == "__main__":
    unittest.main()
