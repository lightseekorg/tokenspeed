"""Phase 0: vLLM HTTP compatibility contract.

This file is the source of truth for "make sure vLLM compatibility". It encodes
the exact surface from vLLM's ``vllm/entrypoints/serve/rlhf/api_router.py`` --
paths, methods, request/response JSON, defaults, and status strings -- and
asserts the tokenspeed app matches byte-for-byte. If vLLM changes its surface,
update CONTRACT here and the implementation together.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from tokenspeed.runtime.entrypoints.weight_transfer_http import (
    build_weight_transfer_app,
    router,
)

# --------------------------------------------------------------------------- #
# The contract (mirrors vLLM api_router.py + weight_transfer dataclasses).
# --------------------------------------------------------------------------- #

CONTRACT_ROUTES = {
    ("/init_weight_transfer_engine", "POST"),
    ("/start_weight_update", "POST"),
    ("/update_weights", "POST"),
    ("/finish_weight_update", "POST"),
    ("/pause", "POST"),
    ("/resume", "POST"),
    ("/get_world_size", "GET"),
    ("/is_paused", "GET"),
}


class RecordingManager:
    """Manager double that records calls and returns controllable values."""

    def __init__(self):
        self.calls = []
        self.paused = False
        self.world_size_value = {True: 8, False: 4}

    async def init_engine(self, init_info):
        self.calls.append(("init_engine", init_info))

    async def start_update(self, is_checkpoint_format=True):
        self.calls.append(("start_update", is_checkpoint_format))

    async def update(self, update_info):
        self.calls.append(("update", update_info))

    async def finish_update(self):
        self.calls.append(("finish_update",))

    async def pause(self, mode="abort", clear_cache=True):
        self.calls.append(("pause", mode, clear_cache))
        self.paused = True

    async def resume(self):
        self.calls.append(("resume",))
        self.paused = False

    def is_paused(self):
        return self.paused

    def get_world_size(self, include_dp=True):
        self.calls.append(("get_world_size", include_dp))
        return self.world_size_value[include_dp]


@pytest.fixture
def manager():
    return RecordingManager()


@pytest.fixture
def client(manager):
    return TestClient(build_weight_transfer_app(manager))


# --------------------------------------------------------------------------- #
# Route surface
# --------------------------------------------------------------------------- #


def test_router_exposes_exactly_the_contract_routes():
    found = set()
    for route in router.routes:
        for method in route.methods:
            if method in ("GET", "POST"):
                found.add((route.path, method))
    assert found == CONTRACT_ROUTES


# --------------------------------------------------------------------------- #
# Request/response payloads (string-exact)
# --------------------------------------------------------------------------- #


def test_init_weight_transfer_engine(client, manager):
    info = {"master_address": "h", "master_port": 1, "rank_offset": 1, "world_size": 2}
    r = client.post("/init_weight_transfer_engine", json={"init_info": info})
    assert r.status_code == 200
    assert r.json() == {"message": "Weight transfer initialized"}
    assert manager.calls == [("init_engine", info)]


def test_start_weight_update_default_is_checkpoint_format(client, manager):
    r = client.post("/start_weight_update", json={})
    assert r.status_code == 200
    assert r.json() == {"message": "Weight update started"}
    # vLLM default: is_checkpoint_format=True
    assert manager.calls == [("start_update", True)]


def test_start_weight_update_explicit_flag(client, manager):
    client.post("/start_weight_update", json={"is_checkpoint_format": False})
    assert manager.calls == [("start_update", False)]


def test_update_weights(client, manager):
    info = {"names": ["w"], "dtype_names": ["bfloat16"], "shapes": [[2, 2]]}
    r = client.post("/update_weights", json={"update_info": info})
    assert r.status_code == 200
    assert r.json() == {"message": "Weights updated"}
    assert manager.calls == [("update", info)]


def test_finish_weight_update_empty_body(client, manager):
    r = client.post("/finish_weight_update")
    assert r.status_code == 200
    assert r.json() == {"message": "Weight update finished"}
    assert manager.calls == [("finish_update",)]


def test_pause_defaults(client, manager):
    r = client.post("/pause")
    assert r.status_code == 200
    assert r.json() == {"status": "paused"}
    # vLLM defaults: mode="abort", clear_cache=True
    assert manager.calls == [("pause", "abort", True)]


@pytest.mark.parametrize("mode", ["abort", "wait", "keep"])
def test_pause_modes(client, manager, mode):
    r = client.post(f"/pause?mode={mode}&clear_cache=false")
    assert r.status_code == 200
    assert r.json() == {"status": "paused"}
    assert manager.calls == [("pause", mode, False)]


def test_resume(client, manager):
    r = client.post("/resume")
    assert r.status_code == 200
    assert r.json() == {"status": "resumed"}
    assert manager.calls == [("resume",)]


def test_is_paused(client, manager):
    assert client.get("/is_paused").json() == {"is_paused": False}
    manager.paused = True
    assert client.get("/is_paused").json() == {"is_paused": True}


def test_get_world_size_default_includes_dp(client, manager):
    r = client.get("/get_world_size")
    assert r.status_code == 200
    assert r.json() == {"world_size": 8}
    assert manager.calls == [("get_world_size", True)]


def test_get_world_size_exclude_dp(client, manager):
    r = client.get("/get_world_size?include_dp=false")
    assert r.json() == {"world_size": 4}
    assert manager.calls == [("get_world_size", False)]


# --------------------------------------------------------------------------- #
# Error contract
# --------------------------------------------------------------------------- #


def test_missing_init_info_is_400(client):
    assert client.post("/init_weight_transfer_engine", json={}).status_code == 400


def test_missing_update_info_is_400(client):
    assert client.post("/update_weights", json={}).status_code == 400


def test_invalid_json_is_400(client):
    r = client.post(
        "/update_weights",
        content="not json",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 400


def test_invalid_pause_mode_is_4xx(client):
    assert client.post("/pause?mode=bogus").status_code == 400


# --------------------------------------------------------------------------- #
# Error-code mapping for lifecycle/backend errors
# --------------------------------------------------------------------------- #


class RaisingManager(RecordingManager):
    def __init__(self, exc):
        super().__init__()
        self._exc = exc

    async def update(self, update_info):
        raise self._exc


@pytest.mark.parametrize(
    "exc,code",
    [
        (
            __import__(
                "tokenspeed.runtime.engine.weight_transfer.manager",
                fromlist=["WeightTransferStateError"],
            ).WeightTransferStateError("nope"),
            409,
        ),
        (ValueError("bad"), 400),
        (NotImplementedError("later"), 501),
        (RuntimeError("kaboom"), 500),
    ],
)
def test_update_error_mapping(exc, code):
    app = build_weight_transfer_app(RaisingManager(exc))
    client = TestClient(app, raise_server_exceptions=False)
    r = client.post("/update_weights", json={"update_info": {}})
    assert r.status_code == code
    assert "error" in r.json()


# --------------------------------------------------------------------------- #
# Optional: live diff against the installed vLLM router, if importable.
# --------------------------------------------------------------------------- #


def test_parity_with_installed_vllm_if_available():
    vllm_router = pytest.importorskip("vllm.entrypoints.serve.rlhf.api_router").router
    vllm_routes = set()
    for route in vllm_router.routes:
        for method in getattr(route, "methods", set()):
            if method in ("GET", "POST"):
                vllm_routes.add((route.path, method))
    # Every vLLM weight-transfer route must exist in tokenspeed with same method.
    assert vllm_routes <= CONTRACT_ROUTES
