"""End-to-end tests for the weight-transfer control plane.

These drive the *real* WeightTransferManager through the *real* FastAPI app over
a *real* socket (uvicorn in a background thread), with an HTTP client. They
exercise the full lifecycle, the vLLM-compatible JSON, and -- most importantly --
the pause/resume admission gate actually blocking a concurrent generation on the
same event loop (the headline RL-safety behavior).

A full ``ts serve`` boot is not exercised here: it needs a GPU + a matching
``tokenspeed_kernel`` wheel + the external smg servicer. ``AsyncLLM``'s heavy
imports are stubbed by ``GateBackedStub`` below, which reproduces AsyncLLM's
admission-gate methods 1:1 (see ``runtime/engine/async_llm.py``) using the real
``RWLock``. To run against a live server instead, set ``TOKENSPEED_E2E_URL`` (see
``test_live_server_lifecycle``).
"""

import os
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
import uvicorn

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.engine.aio_rwlock import RWLock
from tokenspeed.runtime.engine.weight_transfer.config import WeightTransferConfig
from tokenspeed.runtime.engine.weight_transfer.manager import WeightTransferManager
from tokenspeed.runtime.entrypoints.weight_transfer_http import (
    build_weight_transfer_app,
)
from tokenspeed.runtime.utils.network import get_free_port


class GateBackedStub:
    """AsyncLLM stand-in reproducing the weight-transfer admission gate.

    Mirrors the ``weight_transfer_*`` methods and ``_wt_admit`` event from
    ``AsyncLLM`` (async_llm.py) so this e2e exercises the real same-loop pause
    semantics without importing the full GPU engine stack. The scheduler-control
    methods record calls instead of doing ZMQ round-trips.
    """

    def __init__(self, *, world_size=8, tp=2, dp=2):
        import asyncio

        self.server_args = SimpleNamespace(
            mapping=Mapping(
                rank=0, world_size=world_size, attn_tp_size=tp, attn_dp_size=dp
            ),
            host="127.0.0.1",
        )
        self._wt_admit = asyncio.Event()
        self._wt_admit.set()
        self.model_update_lock = RWLock()
        self.rid_to_state = {}
        self.events = []

    async def init_weights_update_group(self, obj):
        self.events.append(("init", obj))
        return True, "ok"

    async def update_weights_from_distributed(self, obj):
        self.events.append(("update", obj))
        return True, "ok"

    async def flush_cache(self):
        self.events.append(("flush",))

    # --- admission gate (copied from AsyncLLM) --------------------------------
    def weight_transfer_admission_paused(self):
        return not self._wt_admit.is_set()

    def weight_transfer_block_admission(self):
        self._wt_admit.clear()

    def weight_transfer_allow_admission(self):
        self._wt_admit.set()

    def weight_transfer_abort_inflight(self):
        for rid in list(self.rid_to_state.keys()):
            self.rid_to_state.pop(rid, None)

    async def weight_transfer_drain_inflight(self):
        async with self.model_update_lock.writer_lock:
            pass

    async def generate(self):
        """Mirrors AsyncLLM.generate_request's admission sequence."""
        if not self._wt_admit.is_set():
            await self._wt_admit.wait()
        async with self.model_update_lock.reader_lock:
            return {"ok": True}


def _build_app(backend="nccl"):
    stub = GateBackedStub()
    manager = WeightTransferManager(stub, WeightTransferConfig(backend=backend))
    app = build_weight_transfer_app(manager)

    # A stand-in generation endpoint that goes through the same admission gate
    # the real engine's generate_request uses, so we can observe pause blocking
    # generation over real HTTP.
    @app.post("/test_generate")
    async def _test_generate():
        return await stub.generate()

    return app, stub


def _serve(app):
    port = get_free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started, "weight-transfer server failed to start"
    return server, thread, f"http://127.0.0.1:{port}"


@pytest.fixture
def server_nccl():
    app, stub = _build_app("nccl")
    server, thread, base = _serve(app)
    try:
        yield base, stub
    finally:
        server.should_exit = True
        thread.join(timeout=5)


@pytest.fixture
def server_ipc():
    app, stub = _build_app("ipc")
    server, thread, base = _serve(app)
    try:
        yield base, stub
    finally:
        server.should_exit = True
        thread.join(timeout=5)


# --------------------------------------------------------------------------- #
# Full lifecycle over real HTTP
# --------------------------------------------------------------------------- #


def test_full_lifecycle_over_http(server_nccl):
    base, stub = server_nccl
    with httpx.Client(base_url=base, timeout=10.0) as c:
        assert c.get("/get_world_size").json() == {"world_size": 8}
        assert c.get("/get_world_size", params={"include_dp": "false"}).json() == {
            "world_size": 4
        }

        r = c.post(
            "/init_weight_transfer_engine",
            json={
                "init_info": {
                    "master_address": "127.0.0.1",
                    "master_port": 29500,
                    "rank_offset": 1,
                    "world_size": 9,
                }
            },
        )
        assert r.status_code == 200
        assert r.json() == {"message": "Weight transfer initialized"}

        assert c.post(
            "/start_weight_update", json={"is_checkpoint_format": True}
        ).json() == {"message": "Weight update started"}
        assert c.post(
            "/update_weights",
            json={
                "update_info": {
                    "names": ["model.embed_tokens.weight"],
                    "dtype_names": ["bfloat16"],
                    "shapes": [[32000, 4096]],
                }
            },
        ).json() == {"message": "Weights updated"}
        assert c.post("/finish_weight_update").json() == {
            "message": "Weight update finished"
        }
        assert c.get("/is_paused").json() == {"is_paused": False}

    # The manager really parsed the metadata into the io_struct and handed it to
    # the (stubbed) scheduler-control layer.
    kinds = [e[0] for e in stub.events]
    assert kinds == ["init", "update"]
    update_obj = [e for e in stub.events if e[0] == "update"][0][1]
    assert update_obj.names == ["model.embed_tokens.weight"]
    assert update_obj.dtype_names == ["bfloat16"]
    assert update_obj.shapes == [[32000, 4096]]


# --------------------------------------------------------------------------- #
# Pause/resume actually gates generation (same event loop, real HTTP)
# --------------------------------------------------------------------------- #


def test_pause_keep_blocks_then_resume_unblocks(server_nccl):
    base, stub = server_nccl
    with httpx.Client(base_url=base) as c:
        # Gate open: generation returns promptly.
        assert c.post("/test_generate", timeout=5.0).json() == {"ok": True}

        # Pause (keep): new generation must block.
        assert c.post("/pause", params={"mode": "keep"}).json() == {"status": "paused"}
        assert c.get("/is_paused").json() == {"is_paused": True}

        with pytest.raises(httpx.TimeoutException):
            # Reaches the server and parks on the admission gate -> read timeout.
            c.post("/test_generate", timeout=1.0)

        # Resume: generation flows again.
        assert c.post("/resume").json() == {"status": "resumed"}
        assert c.get("/is_paused").json() == {"is_paused": False}
        assert c.post("/test_generate", timeout=5.0).json() == {"ok": True}


def test_pause_abort_clears_inflight_and_flushes(server_nccl):
    base, stub = server_nccl
    stub.rid_to_state = {"r1": 1, "r2": 2}
    with httpx.Client(base_url=base, timeout=10.0) as c:
        assert c.post("/pause", params={"mode": "abort"}).json() == {"status": "paused"}
        assert c.get("/is_paused").json() == {"is_paused": True}
        assert c.post("/resume").json() == {"status": "resumed"}
    assert stub.rid_to_state == {}
    assert ("flush",) in stub.events  # clear_cache defaults true for abort


def test_pause_wait_drains_concurrent_generation(server_nccl):
    base, stub = server_nccl
    with httpx.Client(base_url=base, timeout=10.0) as c:
        # pause(mode=wait) acquires the writer lock, which waits for in-flight
        # readers. With none in flight it returns promptly and flushes.
        assert c.post("/pause", params={"mode": "wait"}).json() == {"status": "paused"}
        assert ("flush",) in stub.events
        assert c.post("/resume").json() == {"status": "resumed"}


# --------------------------------------------------------------------------- #
# Error contract over real HTTP
# --------------------------------------------------------------------------- #


def test_lifecycle_ordering_errors_over_http(server_nccl):
    base, _ = server_nccl
    with httpx.Client(base_url=base, timeout=10.0) as c:
        # update before start -> 409 Conflict
        r = c.post("/update_weights", json={"update_info": {"names": []}})
        assert r.status_code == 409
        assert "error" in r.json()
        # missing init_info -> 400
        assert c.post("/init_weight_transfer_engine", json={}).status_code == 400
        # invalid pause mode -> 400
        assert c.post("/pause", params={"mode": "bogus"}).status_code == 400


def test_ipc_update_not_implemented_over_http(server_ipc):
    base, _ = server_ipc
    with httpx.Client(base_url=base, timeout=10.0) as c:
        c.post("/init_weight_transfer_engine", json={"init_info": {}})
        c.post("/start_weight_update", json={})
        r = c.post(
            "/update_weights",
            json={
                "update_info": {
                    "names": ["w"],
                    "dtype_names": ["bfloat16"],
                    "shapes": [[2, 2]],
                    "ipc_handles": [{"uuid": []}],
                }
            },
        )
        # Metadata validated, but worker-side IPC load is deferred -> 501.
        assert r.status_code == 501


# --------------------------------------------------------------------------- #
# Optional: drive a live `ts serve --enable-weight-transfer` if configured.
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not os.environ.get("TOKENSPEED_E2E_URL"),
    reason="set TOKENSPEED_E2E_URL to a running ts-serve with --enable-weight-transfer",
)
def test_live_server_lifecycle():
    """Drive a real server. Exercises only the backend-independent surface;
    init/update need a real trainer + the worker-side receive path."""
    base = os.environ["TOKENSPEED_E2E_URL"].rstrip("/")
    with httpx.Client(base_url=base, timeout=30.0) as c:
        ws = c.get("/get_world_size").json()["world_size"]
        assert isinstance(ws, int) and ws >= 1

        assert c.get("/is_paused").json() == {"is_paused": False}
        assert c.post("/pause", params={"mode": "wait"}).json() == {"status": "paused"}
        assert c.get("/is_paused").json() == {"is_paused": True}
        assert c.post("/resume").json() == {"status": "resumed"}
        assert c.get("/is_paused").json() == {"is_paused": False}

        # Ordering error contract holds on the live server too.
        assert (
            c.post("/update_weights", json={"update_info": {"names": []}}).status_code
            == 409
        )
