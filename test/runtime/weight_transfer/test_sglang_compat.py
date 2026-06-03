"""SGLang-compatible weight-sync shim: routes, payload translation, gate.

The shim adapts SGLang's HTTP weight-sync dialect (what slime/miles and verl's
SGLang rollout POST) onto AsyncLLM's existing methods. These tests use a fake
AsyncLLM; the worker-side load is the deferred backend.
"""

import asyncio
import threading
import time
import types

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.engine.io_struct import (
    InitWeightsUpdateGroupReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from tokenspeed.runtime.entrypoints.sglang_compat_http import (
    build_sglang_compat_app,
    router,
)
from tokenspeed.runtime.utils.network import get_free_port

SGLANG_ROUTES = {
    ("/init_weights_update_group", "POST"),
    ("/destroy_weights_update_group", "POST"),
    ("/update_weights_from_distributed", "POST"),
    ("/update_weights_from_tensor", "POST"),
    ("/update_weights_from_disk", "POST"),
    ("/pause_generation", "POST"),
    ("/continue_generation", "POST"),
    ("/flush_cache", "GET"),
    ("/release_memory_occupation", "POST"),
    ("/resume_memory_occupation", "POST"),
    ("/abort_request", "POST"),
    ("/health_generate", "GET"),
}


class FakeAsyncLLM:
    """AsyncLLM stand-in implementing the surface the SGLang shim calls."""

    def __init__(self):
        self.server_args = types.SimpleNamespace(
            mapping=Mapping(rank=0, world_size=1, attn_tp_size=1, attn_dp_size=1)
        )
        self.calls = []
        self.rid_to_state = {"r1": 1, "r2": 2}
        self.paused = False

    async def init_weights_update_group(self, obj):
        assert isinstance(obj, InitWeightsUpdateGroupReqInput)
        self.calls.append(("init", obj))
        return True, "ok"

    async def update_weights_from_distributed(self, obj):
        assert isinstance(obj, UpdateWeightsFromDistributedReqInput)
        self.calls.append(("dist", obj))
        return True, "ok"

    async def update_weights_from_tensor(self, obj):
        assert isinstance(obj, UpdateWeightsFromTensorReqInput)
        self.calls.append(("tensor", obj))
        return True, "ok"

    async def update_weights_from_disk(self, obj):
        self.calls.append(("disk", obj))
        return True, "loaded", 0  # (success, message, num_paused)

    async def flush_cache(self):
        self.calls.append(("flush",))

    async def release_memory_occupation(self, obj):
        self.calls.append(("release",))

    async def resume_memory_occupation(self, obj):
        self.calls.append(("resume",))

    def weight_transfer_block_admission(self):
        self.paused = True

    def weight_transfer_allow_admission(self):
        self.paused = False

    def weight_transfer_abort_inflight(self):
        self.calls.append(("abort_all", list(self.rid_to_state)))
        self.rid_to_state.clear()

    def abort_request(self, rid):
        self.calls.append(("abort", rid))
        self.rid_to_state.pop(rid, None)


@pytest.fixture
def llm():
    return FakeAsyncLLM()


@pytest.fixture
def client(llm):
    return TestClient(build_sglang_compat_app(llm))


# --------------------------------------------------------------------------- #
# Route surface
# --------------------------------------------------------------------------- #


def test_router_exposes_sglang_routes():
    found = set()
    for route in router.routes:
        for method in getattr(route, "methods", set()):
            if method in ("GET", "POST"):
                found.add((route.path, method))
    assert found == SGLANG_ROUTES


# --------------------------------------------------------------------------- #
# Payload translation + delegation
# --------------------------------------------------------------------------- #


def test_init_weights_update_group(client, llm):
    r = client.post(
        "/init_weights_update_group",
        json={
            "master_address": "127.0.0.1",
            "master_port": 1234,
            "rank_offset": 1,
            "world_size": 2,
            "group_name": "g",
            "backend": "nccl",
        },
    )
    assert r.status_code == 200 and r.json() == {"success": True, "message": "ok"}
    obj = llm.calls[0][1]
    assert obj.master_address == "127.0.0.1" and obj.group_name == "g"


def test_update_weights_from_distributed_translates_dtypes(client, llm):
    r = client.post(
        "/update_weights_from_distributed",
        json={
            "names": ["w"],
            "dtypes": ["bfloat16"],  # SGLang field name
            "shapes": [[2, 2]],
            "group_name": "g",
            "flush_cache": False,
        },
    )
    assert r.status_code == 200 and r.json()["success"]
    obj = [c for c in llm.calls if c[0] == "dist"][0][1]
    # dtypes -> dtype_names
    assert obj.dtype_names == ["bfloat16"] and obj.names == ["w"]


def test_update_weights_from_distributed_length_mismatch_400(client):
    r = client.post(
        "/update_weights_from_distributed",
        json={"names": ["a"], "dtypes": ["x"], "shapes": [[1], [2]]},
    )
    assert r.status_code == 400 and r.json()["success"] is False


def test_update_weights_from_tensor(client, llm):
    r = client.post(
        "/update_weights_from_tensor",
        json={"serialized_named_tensors": ["blob"], "load_format": None},
    )
    assert r.status_code == 200 and r.json()["success"]
    assert any(c[0] == "tensor" for c in llm.calls)


def test_update_weights_from_disk(client, llm):
    r = client.post("/update_weights_from_disk", json={"model_path": "/m"})
    assert r.json() == {"success": True, "message": "loaded"}


def test_destroy_weights_update_group_is_noop(client):
    assert client.post(
        "/destroy_weights_update_group", json={"group_name": "g"}
    ).json()["success"]


# --------------------------------------------------------------------------- #
# Pause / resume gate + memory + abort
# --------------------------------------------------------------------------- #


def test_pause_continue_gate(client, llm):
    assert client.post("/pause_generation", json={}).json()["success"]
    assert llm.paused is True
    assert client.post("/continue_generation", json={}).json()["success"]
    assert llm.paused is False


def test_flush_cache_get(client, llm):
    assert client.get("/flush_cache").json()["success"]
    assert ("flush",) in llm.calls


def test_memory_occupation(client, llm):
    assert client.post("/release_memory_occupation", json={}).json()["success"]
    assert client.post("/resume_memory_occupation", json={"tags": ["weights"]}).json()[
        "success"
    ]
    assert ("release",) in llm.calls and ("resume",) in llm.calls


def test_abort_request_all_and_single(client, llm):
    assert client.post("/abort_request", json={"abort_all": True}).json()["success"]
    assert llm.rid_to_state == {}
    llm.rid_to_state = {"x": 1}
    assert client.post("/abort_request", json={"rid": "x"}).json()["success"]
    assert llm.rid_to_state == {}


def test_health_generate(client):
    assert client.get("/health_generate").json() == {"status": "ok"}


# --------------------------------------------------------------------------- #
# Both dialects on one app/port (the "same URL" consolidation)
# --------------------------------------------------------------------------- #


def test_vllm_and_sglang_routes_coexist_on_one_app(llm):
    """The engine mounts the vLLM router + SGLang router on a single app."""
    from tokenspeed.runtime.entrypoints.weight_transfer_http import (
        build_weight_transfer_app,
    )

    class FakeManager:
        def get_world_size(self, include_dp=True):
            return 8

        def is_paused(self):
            return False

    app = build_weight_transfer_app(FakeManager())  # vLLM router + manager state
    app.state.async_llm = llm
    app.include_router(router)  # SGLang router on the SAME app
    c = TestClient(app)
    # vLLM-native endpoint
    assert c.get("/get_world_size").json() == {"world_size": 8}
    # SGLang endpoint on the same app/port
    assert c.get("/health_generate").json() == {"status": "ok"}
    assert c.post(
        "/update_weights_from_distributed",
        json={"names": ["w"], "dtypes": ["bfloat16"], "shapes": [[2, 2]]},
    ).json()["success"]


# --------------------------------------------------------------------------- #
# End-to-end over a real socket
# --------------------------------------------------------------------------- #


@pytest.fixture
def live_server(llm):
    port = get_free_port()
    app = build_sglang_compat_app(llm)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started
    try:
        yield f"http://127.0.0.1:{port}", llm
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_sglang_lifecycle_over_http(live_server):
    base, llm = live_server
    with httpx.Client(base_url=base, timeout=10.0) as c:
        assert c.post(
            "/init_weights_update_group",
            json={
                "master_address": "127.0.0.1",
                "master_port": 1,
                "rank_offset": 1,
                "world_size": 2,
            },
        ).json()["success"]
        assert c.post("/pause_generation", json={}).json()["success"]
        assert llm.paused is True
        assert c.get("/flush_cache").json()["success"]
        assert c.post(
            "/update_weights_from_distributed",
            json={"names": ["w"], "dtypes": ["bfloat16"], "shapes": [[2, 2]]},
        ).json()["success"]
        assert c.post("/continue_generation", json={}).json()["success"]
        assert llm.paused is False
