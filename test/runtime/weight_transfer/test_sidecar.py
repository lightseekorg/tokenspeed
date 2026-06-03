"""Phase 7: sidecar proxy wiring for the weight-transfer endpoints."""

import asyncio
import threading
import time

import pytest
import uvicorn
from fastapi.testclient import TestClient

from tokenspeed.cli.serve_smg import _maybe_add_weight_transfer_port
from tokenspeed.runtime.entrypoints import http_server
from tokenspeed.runtime.entrypoints.weight_transfer_http import (
    build_weight_transfer_app,
)
from tokenspeed.runtime.utils.network import get_free_port

WEIGHT_ROUTES = {
    ("/init_weight_transfer_engine", "POST"),
    ("/start_weight_update", "POST"),
    ("/update_weights", "POST"),
    ("/finish_weight_update", "POST"),
    ("/pause", "POST"),
    ("/resume", "POST"),
    ("/get_world_size", "GET"),
    ("/is_paused", "GET"),
}


# --------------------------------------------------------------------------- #
# Orchestrator port handshake
# --------------------------------------------------------------------------- #


class TestMaybeAddWeightTransferPort:
    def test_on_by_default_allocates_port(self):
        args, url = _maybe_add_weight_transfer_port(["--model", "m"])
        assert "--weight-transfer-port" in args
        assert url.startswith("http://127.0.0.1:")
        port = args[args.index("--weight-transfer-port") + 1]
        assert url.endswith(port)

    def test_opt_out_no_change(self):
        argv = ["--model", "m", "--no-enable-weight-transfer"]
        args, url = _maybe_add_weight_transfer_port(argv)
        assert args == argv
        assert url == ""

    def test_respects_pinned_port(self):
        args, url = _maybe_add_weight_transfer_port(["--weight-transfer-port", "9999"])
        assert args.count("--weight-transfer-port") == 1
        assert url == "http://127.0.0.1:9999"


# --------------------------------------------------------------------------- #
# Sidecar routes
# --------------------------------------------------------------------------- #


def test_sidecar_exposes_weight_routes():
    found = set()
    for route in http_server.app.routes:
        for method in getattr(route, "methods", set()):
            if method in ("GET", "POST"):
                found.add((route.path, method))
    assert WEIGHT_ROUTES <= found


def test_disabled_weight_routes_return_503():
    http_server.build_server(
        gateway_url="http://127.0.0.1:1",
        engine_grpc_addr="127.0.0.1:1",
        engine_http_url="",
        port=0,
    )
    client = TestClient(http_server.app)
    assert client.get("/get_world_size").status_code == 503
    assert client.post("/resume").status_code == 503


# --------------------------------------------------------------------------- #
# End-to-end: sidecar proxies to a live in-engine weight-transfer app
# --------------------------------------------------------------------------- #


class _FakeManager:
    def __init__(self):
        self.paused = False

    async def init_engine(self, init_info):
        pass

    async def start_update(self, is_checkpoint_format=True):
        pass

    async def update(self, update_info):
        pass

    async def finish_update(self):
        pass

    async def pause(self, mode="abort", clear_cache=True):
        self.paused = True

    async def resume(self):
        self.paused = False

    def is_paused(self):
        return self.paused

    def get_world_size(self, include_dp=True):
        return 8 if include_dp else 4


@pytest.fixture
def upstream_engine_app():
    """Run the real weight-transfer app in a background uvicorn thread."""
    port = get_free_port()
    app = build_weight_transfer_app(_FakeManager())
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started, "upstream weight-transfer app failed to start"
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_sidecar_proxies_full_lifecycle(upstream_engine_app):
    http_server.build_server(
        gateway_url="http://127.0.0.1:1",
        engine_grpc_addr="127.0.0.1:1",
        engine_http_url=upstream_engine_app,
        port=0,
    )
    client = TestClient(http_server.app)

    r = client.get("/get_world_size")
    assert r.status_code == 200 and r.json() == {"world_size": 8}

    r = client.post(
        "/init_weight_transfer_engine",
        json={
            "init_info": {
                "master_address": "h",
                "master_port": 1,
                "rank_offset": 1,
                "world_size": 2,
            }
        },
    )
    assert r.status_code == 200 and r.json() == {
        "message": "Weight transfer initialized"
    }

    assert client.post("/start_weight_update", json={}).json() == {
        "message": "Weight update started"
    }
    assert client.post(
        "/update_weights", json={"update_info": {"names": ["w"]}}
    ).json() == {"message": "Weights updated"}
    assert client.post("/finish_weight_update").json() == {
        "message": "Weight update finished"
    }

    assert client.post("/pause?mode=wait").json() == {"status": "paused"}
    assert client.get("/is_paused").json() == {"is_paused": True}
    assert client.post("/resume").json() == {"status": "resumed"}
    assert client.get("/is_paused").json() == {"is_paused": False}
