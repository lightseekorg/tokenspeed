"""The in-engine RL control app as an external gateway sees it: where it
listens, what it advertises, and how it authenticates.

Run with: pytest test/runtime/test_rl_control_endpoint.py
"""

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from test.runtime.rl_fakes import FakeLLM  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from tokenspeed.runtime.entrypoints import rl_control, sglang_compat_http  # noqa: E402
from tokenspeed.runtime.entrypoints.sglang_compat_http import (  # noqa: E402
    build_sglang_compat_app,
)
from tokenspeed.runtime.utils.server_args import ServerArgs  # noqa: E402


def _args(**overrides):
    base = {"host": "127.0.0.1", "rl_control_port": 40100, "rl_control_host": None}
    base.update(overrides)
    return SimpleNamespace(**base)


class TestControlEndpointHelpers:
    def test_bind_host_defaults_to_engine_host(self):
        assert (rl_control.control_bind_host(_args())) == ("127.0.0.1")
        assert (rl_control.control_bind_host(_args(rl_control_host="0.0.0.0"))) == (
            "0.0.0.0"
        )

    def test_control_url_uses_bound_host_and_port(self):
        assert (rl_control.control_url(_args())) == ("http://127.0.0.1:40100")
        assert (rl_control.control_url(_args(rl_control_host="10.0.0.5"))) == (
            "http://10.0.0.5:40100"
        )
        assert (rl_control.control_url(_args(host="::1"))) == ("http://[::1]:40100")

    def test_control_url_does_not_advertise_wildcard_addresses(self, subtests):
        for host in ("0.0.0.0", "::", "[::]"):
            with subtests.test(host=host):
                args = _args(rl_control_host=host)
                assert (rl_control.control_bind_host(args)) == (host)
                assert (rl_control.control_url(args)) is None
                assert ("rl.control_url") not in (rl_control.advertisement(args))

    def test_control_url_is_none_without_a_port(self):
        assert (rl_control.control_url(_args(rl_control_port=None))) is None
        assert (rl_control.control_url(_args(rl_control_port=0))) is None

    def test_capabilities_match_the_smg_label_contract(self):
        caps = rl_control.capabilities()
        assert (caps["rl.pause_modes"]) == ("wait,abort,keep")
        # Derived from the scheduler's dispatcher, which implements the
        # disk, distributed and Mooncake load paths; tensor answers 501.
        assert (caps["rl.update_from"]) == ("disk,distributed,mooncake")
        for key in (
            "rl.abort",
            "rl.flush_cache",
            "rl.sleep_wake",
            "rl.reports_weight_version",
        ):
            assert (caps[key]) == ("true")
        assert ("tensor") not in (caps["rl.update_from"])

    def test_advertisement_carries_url_and_capabilities(self):
        adv = rl_control.advertisement(_args())
        assert (adv["rl.control_url"]) == ("http://127.0.0.1:40100")
        assert (adv["rl.abort"]) == ("true")
        assert ("rl.control_url") not in (
            rl_control.advertisement(_args(rl_control_port=None))
        )


class TestServerArgsFlags:
    def test_flags_parse_and_default_to_none(self):
        import argparse

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        ns = parser.parse_args(["--model", "m"])
        assert (ns.rl_control_host) is None
        assert (ns.rl_control_api_key) is None
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
        assert (ns.rl_control_host) == ("0.0.0.0")
        assert (ns.rl_control_api_key) == ("k")


class TestServerInfoSecrets:
    def test_engine_server_info_omits_control_api_key(self):
        from dataclasses import dataclass

        from tokenspeed.runtime.entrypoints.engine import Engine

        @dataclass
        class Args:
            host: str = "localhost"
            rl_control_api_key: str = "test-secret"

        engine = SimpleNamespace(
            tokenizer_manager=SimpleNamespace(
                server_args=Args(), get_internal_state=lambda: []
            ),
            llm=SimpleNamespace(run=lambda result: result),
            scheduler_info={},
        )
        info = Engine.get_server_info(engine)
        assert ("rl_control_api_key") not in (info)
        assert (info["host"]) == ("localhost")


class _AuthLLM:
    def __init__(self, api_key):
        self.server_args = SimpleNamespace(
            weight_version="default",
            model="m",
            kvstore_storage_backend=None,
            rl_control_api_key=api_key,
        )


class TestBearerAuth:
    def test_open_when_no_key_is_configured(self):
        client = TestClient(build_sglang_compat_app(_AuthLLM(None)))
        assert (client.get("/get_weight_version").status_code) == (200)

    def test_rejects_missing_or_wrong_bearer(self):
        client = TestClient(build_sglang_compat_app(_AuthLLM("s3cret")))
        resp = client.get("/get_weight_version")
        assert (resp.status_code) == (401)
        assert (resp.json()) == ({"success": False, "message": "unauthorized"})
        resp = client.get(
            "/get_weight_version", headers={"Authorization": "Bearer nope"}
        )
        assert (resp.status_code) == (401)

    def test_non_ascii_bearer_is_unauthorized(self):
        client = TestClient(build_sglang_compat_app(_AuthLLM("s3cret")))
        response = client.get(
            "/get_weight_version", headers={b"Authorization": b"Bearer \xff"}
        )
        assert (response.status_code) == (401)

    def test_accepts_the_configured_bearer(self):
        client = TestClient(build_sglang_compat_app(_AuthLLM("s3cret")))
        resp = client.get(
            "/get_weight_version", headers={"Authorization": "Bearer s3cret"}
        )
        assert (resp.status_code) == (200)
        assert (resp.json()) == ({"weight_version": "default"})


class TestRouteSemantics:
    def _client(self):
        llm = FakeLLM()
        return llm, TestClient(build_sglang_compat_app(llm))

    def test_empty_body_is_400_on_body_routes(self):
        _llm, client = self._client()
        for route in (
            "/init_weights_update_group",
            "/update_weights_from_distributed",
            "/update_weights_from_mooncake",
            "/update_weights_from_disk",
            # /update_weights_from_tensor answers 501 before the body is read;
            # see TestUnsupportedWeightUpdateSources.
            "/abort_request",
            "/update_weight_version",
        ):
            resp = client.post(route)
            _pytest_first_1 = resp.status_code
            _pytest_second_2 = 400
            _pytest_msg_3 = route
            assert _pytest_first_1 == _pytest_second_2, _pytest_msg_3
            _pytest_expr_4 = resp.json()["success"]
            _pytest_msg_5 = route
            assert not _pytest_expr_4, _pytest_msg_5
            resp = client.post(
                route,
                content=b"{not json",
                headers={"content-type": "application/json"},
            )
            _pytest_first_6 = resp.status_code
            _pytest_second_7 = 400
            _pytest_msg_8 = route
            assert _pytest_first_6 == _pytest_second_7, _pytest_msg_8

    def test_malformed_optional_body_has_no_side_effects(self, subtests):
        llm, client = self._client()
        for route in (
            "/pause_generation",
            "/release_memory_occupation",
            "/resume_memory_occupation",
            "/destroy_weights_update_group",
        ):
            for body in (b"{invalid", b"[]", b"null", b'"keep"'):
                with subtests.test(route=route, body=body):
                    response = client.post(route, content=body)
                    assert (response.status_code) == (400)
        assert (llm.scheduler_calls) == ([])
        assert (llm.admission_calls) == ([])
        assert (llm.memory_calls) == ([])

    def test_pause_mode_is_honored_and_echoed(self):
        llm, client = self._client()
        assert (client.post("/pause_generation").json()["mode"]) == ("wait")
        assert (
            client.post("/pause_generation", json={"mode": "keep"}).json()["mode"]
        ) == ("keep")
        assert (
            client.post("/pause_generation", json={"mode": "abort"}).status_code
        ) == (200)
        assert ([m for kind, m in llm.scheduler_calls if kind == "pause"]) == (
            ["wait", "keep", "abort"]
        )

    def test_invalid_pause_mode_is_400_and_reopens_admission(self):
        llm, client = self._client()
        resp = client.post("/pause_generation", json={"mode": "retract"})
        assert (resp.status_code) == (400)
        assert ("invalid pause mode") in (resp.json()["message"])
        assert (llm.scheduler_calls) == ([])
        assert (llm.admission_calls) == ([])

    def test_flush_cache_accepts_get_and_post(self):
        _llm, client = self._client()
        assert (client.get("/flush_cache").status_code) == (200)
        assert (client.post("/flush_cache").status_code) == (200)

    def test_flush_cache_preserves_scheduler_refusal_and_allows_retry(self):
        llm = FakeLLM()
        llm.succeed = False
        client = TestClient(build_sglang_compat_app(llm))
        for method in (client.get, client.post):
            response = method("/flush_cache")
            assert response.status_code == 400
            assert response.json()["success"] is False
        llm.succeed = True
        response = client.post("/flush_cache")
        assert response.status_code == 200
        assert response.json()["success"] is True


class TestUnsupportedWeightUpdateSources:
    """Sources the scheduler has no branch for never reach it.

    Forwarding one raises ``NotImplementedError`` inside the scheduler process,
    which dies and takes the control app and the engine with it.
    """

    def _client(self):
        llm = FakeLLM()
        return llm, TestClient(build_sglang_compat_app(llm))

    def test_disk_reaches_the_engine(self):
        llm, client = self._client()
        resp = client.post(
            "/update_weights_from_disk",
            json={"model_path": "/tmp/model", "weight_version": "v9"},
        )
        assert (resp.status_code) == (200)
        assert ([type(obj).__name__ for obj in llm.updates]) == (
            ["UpdateWeightFromDiskReqInput"]
        )
        assert (llm.server_args.weight_version) == ("v9")

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
        assert (resp.status_code) == (501)
        body = resp.json()
        assert not (body["success"])
        assert ("update_weights_from_tensor") in (body["message"])
        assert ("distributed") in (body["message"])
        assert (llm.updates) == ([])
        assert (llm.server_args.weight_version) == ("default")

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
        assert (resp.status_code) == (200)
        assert (len(llm.updates)) == (1)

    def test_guard_follows_the_scheduler_constant(self):
        # Data-driven, not hard-coded: narrow the supported set and the disk
        # route is refused; widen it and the tensor route reaches the engine.
        llm, client = self._client()
        with mock.patch.object(
            sglang_compat_http,
            "SUPPORTED_WEIGHT_UPDATE_SOURCES",
            frozenset({"distributed"}),
        ):
            resp = client.post(
                "/update_weights_from_disk", json={"model_path": "/tmp/model"}
            )
        assert (resp.status_code) == (501)
        assert (llm.updates) == ([])
        with mock.patch.object(
            sglang_compat_http,
            "SUPPORTED_WEIGHT_UPDATE_SOURCES",
            frozenset({"tensor", "distributed"}),
        ):
            resp = client.post(
                "/update_weights_from_tensor",
                json={"serialized_named_tensors": [], "flush_cache": False},
            )
        assert (resp.status_code) == (200)
        assert ([type(obj).__name__ for obj in llm.updates]) == (
            ["UpdateWeightsFromTensorReqInput"]
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
