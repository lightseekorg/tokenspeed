# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""The PD control plane on IPv6 hosts: endpoint formatting, when a ZMQ socket
enables IPv6, the control socket's port, the Mooncake engine's MC_USE_IPV6 and
session id, and one request through real Prefill and Decode executors over IPv4
and IPv6 loopback."""

from __future__ import annotations

import functools
import os
import socket
import sys
import threading
import time
import types

import pytest
import zmq

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from test.runtime import cache_pd_test_utils as cache_pd  # noqa: E402

from tokenspeed.runtime.engine.cache_hooks import CacheOpHooks  # noqa: E402
from tokenspeed.runtime.engine.event_loop import EventLoop  # noqa: E402
from tokenspeed.runtime.engine.generation_output_processor import (  # noqa: E402
    OutputProcesser,
)
from tokenspeed.runtime.engine.l3_cache_hooks import L3CacheHooks  # noqa: E402
from tokenspeed.runtime.engine.pause import (  # noqa: E402
    PauseController,
    PauseHooks,
    PauseState,
)
from tokenspeed.runtime.epd.prefill_hooks import EpdPrefillHooks  # noqa: E402
from tokenspeed.runtime.metrics.collector import EngineMetrics  # noqa: E402
from tokenspeed.runtime.pd.base.bootstrap import BootstrapInfo  # noqa: E402
from tokenspeed.runtime.pd.base.manager import (  # noqa: E402
    DisaggManagerBase,
    enable_ipv6_for_endpoint,
)
from tokenspeed.runtime.pd.base.mooncake_engine import (  # noqa: E402
    MooncakeTransferEngine,
    enable_mooncake_ipv6_for_host,
)
from tokenspeed.runtime.pd.base.status import TransferPoll  # noqa: E402
from tokenspeed.runtime.pd.decode_executor import DisaggDecodeExecutor  # noqa: E402
from tokenspeed.runtime.pd.mooncake.conn import MooncakeKVBootstrapServer  # noqa: E402
from tokenspeed.runtime.pd.mooncake.entities import KVArgs, KVManagerArgs  # noqa: E402
from tokenspeed.runtime.pd.prefill_executor import DisaggPrefillExecutor  # noqa: E402
from tokenspeed.runtime.pd.topology import PDParallelTopology  # noqa: E402
from tokenspeed.runtime.pd.transfer_hooks import PdTransferHooks  # noqa: E402
from tokenspeed.runtime.utils.network import get_free_port, join_host_port  # noqa: E402


def _ipv6_loopback_available() -> bool:
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
            sock.bind(("::1", 0))
    except OSError:
        return False
    return True


requires_ipv6_loopback = pytest.mark.skipif(
    not _ipv6_loopback_available(), reason="IPv6 loopback is unavailable"
)


def _free_port_for_both_families() -> int:
    """A port free on IPv4 and IPv6 alike, for a server that binds both, as
    the bootstrap server does; get_free_port() only probes IPv4."""
    if not _ipv6_loopback_available():
        return get_free_port()
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        sock.bind(("::", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("10.0.0.1", "10.0.0.1:8998"),
        ("fd00::1", "[fd00::1]:8998"),
        ("[fd00::1]", "[fd00::1]:8998"),
        ("fe80::1%eth0", "[fe80::1%eth0]:8998"),
        ("prefill-0.prefill", "prefill-0.prefill:8998"),
    ],
)
def test_join_host_port_brackets_only_ipv6_literals(host, expected):
    assert join_host_port(host, 8998) == expected


@pytest.mark.parametrize(
    ("host", "ipv6"),
    [
        ("[fd00::1]", 1),
        ("10.0.0.1", 0),
        ("v6-only.example", 1),
        # With IPv6 on, libzmq resolves a name to IPv6 first and would miss an
        # IPv4-only listener, so a dual-stack name keeps the IPv4 default.
        ("dual-stack.example", 0),
        ("unresolvable.example", 0),
    ],
)
def test_zmq_ipv6_follows_endpoint_host_address_family(monkeypatch, host, ipv6):
    v6 = (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("fd00::1", 0, 0, 0))
    v4 = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 0))
    addresses = {"v6-only.example": [v6], "dual-stack.example": [v6, v4]}

    def getaddrinfo(name, *args, **kwargs):
        if name not in addresses:
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
        return addresses[name]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    with zmq.Context() as context, context.socket(zmq.PUSH) as sock:
        enable_ipv6_for_endpoint(sock, f"tcp://{host}:8998")
        assert sock.getsockopt(zmq.IPV6) == ipv6


def test_zmq_ipv6_stays_off_for_malformed_name():
    # socket.getaddrinfo raises UnicodeError, not OSError, for "a..b".
    with zmq.Context() as context, context.socket(zmq.PUSH) as sock:
        enable_ipv6_for_endpoint(sock, "tcp://a..b:8998")
        assert sock.getsockopt(zmq.IPV6) == 0


class _ControlSocketManager(DisaggManagerBase):
    """A manager without transfer buffers to register."""

    def register_buffer_to_engine(self):
        pass


@pytest.mark.parametrize(
    ("host", "ipv6"),
    [("127.0.0.1", 0), pytest.param("::1", 1, marks=requires_ipv6_loopback)],
)
def test_control_socket_is_reachable_on_the_port_it_reports(host, ipv6):
    """bind_server_socket stores the port its own bind took in rank_port, and
    a peer reaches the control socket there."""
    manager = _ControlSocketManager(engine=None)
    try:
        manager.bind_server_socket(host)
        assert isinstance(manager.rank_port, int)
        # IPv4 keeps libzmq's default socket options.
        assert manager.server_socket.getsockopt(zmq.IPV6) == ipv6
        endpoint = f"tcp://{join_host_port(host, manager.rank_port)}"
        with zmq.Context() as context, context.socket(zmq.PUSH) as peer:
            peer.setsockopt(zmq.LINGER, 0)
            enable_ipv6_for_endpoint(peer, endpoint)
            peer.connect(endpoint)
            peer.send(b"ping")
            assert manager.server_socket.poll(5000) == zmq.POLLIN
            assert manager.server_socket.recv() == b"ping"
    finally:
        # Garbage-collecting the context of a socket left open would block.
        manager.server_socket.close()


def test_mooncake_ipv6_keeps_explicit_setting(monkeypatch):
    monkeypatch.setenv("MC_USE_IPV6", "on")
    enable_mooncake_ipv6_for_host("fd00::1")
    assert os.environ["MC_USE_IPV6"] == "on"


@pytest.fixture
def fake_mooncake(monkeypatch):
    """Fake Mooncake's native engine module, which needs RDMA, and start with
    MC_USE_IPV6 unset. Returns the (hostname, MC_USE_IPV6) each engine starts
    with and the session ids that transfers write to."""
    initialized, written_sessions = [], []
    rpc_ports = iter((15123, 15432))

    class TransferEngine:
        def __init__(self) -> None:
            self.rpc_port: int | None = None

        def initialize(self, hostname, metadata_server, protocol, device_name):
            initialized.append((hostname, os.environ.get("MC_USE_IPV6")))
            self.rpc_port = next(rpc_ports)
            return 0

        def get_rpc_port(self):
            return self.rpc_port

        def register_memory(self, ptr, length):
            return 0

        def batch_transfer_sync_write_pages(self, session_id, *pages):
            written_sessions.append(session_id)
            return 0

    engine_module = types.ModuleType("mooncake.engine")
    engine_module.TransferEngine = TransferEngine
    monkeypatch.setitem(sys.modules, "mooncake", types.ModuleType("mooncake"))
    monkeypatch.setitem(sys.modules, "mooncake.engine", engine_module)
    # setenv first, so that monkeypatch also removes the value an engine sets.
    monkeypatch.setenv("MC_USE_IPV6", "")
    monkeypatch.delenv("MC_USE_IPV6")
    return initialized, written_sessions


@pytest.mark.parametrize(
    ("hostname", "mc_use_ipv6", "session_id"),
    [
        ("10.0.0.1", None, "10.0.0.1:15123"),
        ("fd00::1", "1", "[fd00::1]:15123"),
    ],
)
def test_mooncake_engine_follows_local_address_family(
    fake_mooncake, hostname, mc_use_ipv6, session_id
):
    initialized, _ = fake_mooncake
    engine = MooncakeTransferEngine(hostname, gpu_id=0, ib_device=None)
    assert initialized == [(hostname, mc_use_ipv6)]
    assert engine.session_id == session_id


def _wait_for(condition) -> None:
    deadline = time.monotonic() + 10
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting for the PD peer"
        time.sleep(0.01)


def _loopback_scheduler(role, allocation_policy):
    import tokenspeed_scheduler as native

    from tokenspeed.runtime.engine.scheduler_utils import make_config

    config = make_config(
        num_device_pages=16,
        max_scheduled_tokens=8,
        max_batch_size=4,
        prefix_granularity=2,
        num_host_pages=0,
        disable_l2_cache=True,
        enable_l3_storage=False,
        role=role,
        num_snapshot_pages=1,
        max_retracted_requests=0,
        storage_config=None,
        decode_allocation_policy=allocation_policy if role == "prefill" else None,
        cache_groups=[
            native.CacheGroupConfig(
                "history",
                2,
                16,
                native.AttnKind.Full,
                transfer_policy=native.CacheTransferPolicy.FullSuffix,
            )
        ],
    )
    core = native.Scheduler(config)
    spec = native.RequestSpec()
    spec.request_id, spec.tokens, spec.max_new_tokens = "req", [1, 2, 3, 4], 8
    core.submit_requests([spec])
    return core


class _PlanCaptured(Exception):
    def __init__(self, plan) -> None:
        super().__init__("Model execution is replaced by the loopback test")
        self.plan = plan


class _PrefillPlanningLoop(EventLoop):
    """Use real admission hooks and planning without constructing GPU models."""

    def __init__(self, scheduler, transfer) -> None:
        self.scheduler = scheduler
        self.kv_transfer = transfer
        self.in_flight_depth = 0
        self._pause = PauseController(send_func=lambda _message: None)
        self._l3 = None
        self._epd_hooks = EpdPrefillHooks(self, admission=None)
        self._pd_hooks = PdTransferHooks(self, device=None)
        self._cache_hooks = CacheOpHooks(
            None,
            speculative_algorithm=None,
            attn_tp_rank=0,
            attn_tp_size=1,
            attn_tp_cpu_group=None,
            pp_size=1,
            pp_cpu_group=None,
            global_rank=0,
        )
        self._l3_hooks = L3CacheHooks(
            scheduler,
            None,
            config=None,
            attn_tp_size=1,
            attn_tp_cpu_group=None,
            pp_size=1,
            pp_cpu_group=None,
            metrics=None,
        )
        self.output_processor = OutputProcesser(
            send_to_tokenizer=lambda _message: None,
            metrics=EngineMetrics(labels={}, enabled=False, reporters=[]),
        )

    def _shutdown_complete(self) -> bool:
        return False

    def _process_new_requests(self) -> None:
        # The test supplies the request directly instead of a tokenizer socket.
        pass

    def _get_forward_op(self, execution_plan):
        raise _PlanCaptured(execution_plan)


@pytest.mark.parametrize(
    ("host", "mc_use_ipv6", "decode_session"),
    [
        ("127.0.0.1", None, "127.0.0.1:15432"),
        pytest.param("::1", "1", "[::1]:15432", marks=requires_ipv6_loopback),
    ],
)
@pytest.mark.parametrize(
    ("allocation_policy", "readiness_order"),
    [
        ("early", "decode_first"),
        ("independent", "decode_first"),
        ("independent", "frozen_before_grant"),
        ("independent", "prefill_first"),
        ("independent", "during_forward"),
        ("independent", "cancel_before_grant"),
        ("independent", "cancel_during_forward"),
    ],
)
def test_request_crosses_pd_control_plane_over_loopback(
    monkeypatch,
    fake_mooncake,
    host,
    mc_use_ipv6,
    decode_session,
    allocation_policy,
    readiness_order,
):
    """Bootstrap, Decode registration, the request manifest, the transfer and
    Prefill's final status between real executors on ``host``, which both roles
    take as their local address."""
    initialized, written_sessions = fake_mooncake
    # Both roles take this host's address from get_local_ip_by_remote(), and
    # Prefill its bootstrap server's from HOST_IP.
    for module in ("conn", "decode", "prefill", "receiver"):
        monkeypatch.setattr(
            f"tokenspeed.runtime.pd.mooncake.{module}.get_local_ip_by_remote",
            lambda: host,
        )
    monkeypatch.setenv("HOST_IP", host)
    monkeypatch.delenv("TOKENSPEED_HOST_IP", raising=False)
    # The managers start endless worker threads and have no shutdown; as
    # daemons they cannot keep the test process alive.
    monkeypatch.setattr(
        threading, "Thread", functools.partial(threading.Thread, daemon=True)
    )

    cache_layout = cache_pd.layout(
        cache_pd.group("history", cache_pd.segment("layer.0.kv", shape=(8,)))
    )
    bootstrap_port = _free_port_for_both_families()
    bootstrap = MooncakeKVBootstrapServer(bootstrap_port)

    def start(executor_cls, kv_data_ptr):
        args = KVManagerArgs(
            bootstrap_port=bootstrap_port,
            dist_init_addr=None,
            topology=PDParallelTopology(
                tp_size=1, tp_rank=0, dp_size=1, dp_rank=0, world_size=1, global_rank=0
            ),
            enable_metrics=False,
            enable_dp_attention=False,
            served_model_name="model",
            app_key=None,
            # What resolve_metrics_options() leaves for a server started without
            # --metrics-reporters.
            metrics_reporters=["prometheus"],
        )
        kv_args = KVArgs(
            engine_rank=0,
            kv_data_ptr=kv_data_ptr,
            ib_device=None,
            gpu_id=0,
            cache_layout=cache_layout,
            cache_fields_by_stage=(("layer.0.kv",),),
        )
        if executor_cls is DisaggPrefillExecutor:
            return executor_cls(
                args, kv_args, gloo_group=None, admission_policy=allocation_policy
            )
        return executor_cls(args, kv_args, gloo_group=None)

    try:
        prefill = start(DisaggPrefillExecutor, 0x10000)
        decode = start(DisaggDecodeExecutor, 0x20000)
        # Mooncake reads MC_USE_IPV6 when its first engine starts.
        assert initialized == [(host, mc_use_ipv6), (host, mc_use_ipv6)]
        assert decode.kv_manager.get_session_id() == decode_session

        info = BootstrapInfo(
            bootstrap_host=host,
            bootstrap_port=bootstrap_port,
            bootstrap_room=7,
        )
        import tokenspeed_scheduler as native

        from tokenspeed.runtime.engine.scheduler_utils import make_config

        # One-rank process-group consensus is identity; only GPU computation and
        # the missing RDMA/process-group environment are replaced in this test.
        monkeypatch.setattr(
            "tokenspeed.runtime.pd.utils.dist.all_reduce", lambda *_a, **_k: None
        )

        p_core, d_core = _loopback_scheduler(
            "prefill", allocation_policy
        ), _loopback_scheduler("decode", None)
        planning_loop = _PrefillPlanningLoop(p_core, prefill)
        source_computed = False

        def plan_prefill():
            with pytest.raises(_PlanCaptured) as captured:
                planning_loop.event_loop()
            return captured.value.plan

        def complete_prefill(plan):
            nonlocal source_computed
            if plan.remote_decode is not None:
                assert source_computed
                prefill.execute(plan.remote_decode)
            for batch in plan.forward:
                for rid in batch.request_ids:
                    result = native.ForwardEvent.ExtendResult()
                    result.request_id, result.tokens = rid, [42]
                    p_core.advance(native.ExecutionEvent().add_event(result))
                    source_computed = True
            events = native.ExecutionEvent()
            for event in prefill.generate_events():
                events.add_event(event)
            p_core.advance(events)

        def step_decode():
            events = native.ExecutionEvent()
            for event in decode.generate_events():
                events.add_event(event)
            d_core.advance(events)
            plan = d_core.next_execution_plan()
            if plan.remote_prefill is not None:
                decode.execute(plan.remote_prefill)

        if readiness_order in ("decode_first", "frozen_before_grant"):
            decode.register("req", info)
            step_decode()
            assert d_core.active_lcm_blocks() > 0
            assert not source_computed
            _wait_for(
                lambda: prefill.kv_manager.room_status(7) == TransferPoll.Bootstrapped
            )
            # The manifest may arrive before the local request is registered.
            prefill.register("req", info)
            assert prefill.kv_manager.room_status(7) == TransferPoll.Bootstrapped
            if readiness_order == "frozen_before_grant":
                frozen = _FrozenPlanningLoop(p_core, prefill)
                with pytest.raises(_FrozenRoundComplete):
                    frozen.event_loop()
                assert p_core.bootstrapping_size() == 1
                assert p_core.active_lcm_blocks() == 0
            if allocation_policy == "early":
                assert prefill.generate_admission_events() == []
                complete_prefill(plan_prefill())
            compute = plan_prefill()
            assert any(batch.request_ids for batch in compute.forward)
            assert compute.remote_decode is None
            assert not written_sessions
            complete_prefill(compute)
            assert p_core.awaiting_destination_size() == 0
            send = plan_prefill()
            assert send.remote_decode is not None
            assert not any(batch.request_ids for batch in send.forward)
            complete_prefill(send)
        else:
            prefill.register("req", info)
            compute = plan_prefill()
            assert any(batch.request_ids for batch in compute.forward)
            assert d_core.active_lcm_blocks() == 0
            assert not written_sessions
            if readiness_order == "cancel_during_forward":
                assert not planning_loop._pd_hooks.cancel_request("req", "cancelled")
                assert p_core.active_lcm_blocks() > 0
            if readiness_order in (
                "prefill_first",
                "cancel_before_grant",
                "cancel_during_forward",
            ):
                complete_prefill(compute)
                assert p_core.awaiting_destination_size() == 1
                assert plan_prefill().remote_decode is None
                if readiness_order in ("cancel_before_grant", "cancel_during_forward"):
                    if readiness_order == "cancel_before_grant":
                        assert planning_loop._pd_hooks.cancel_request(
                            "req", "cancelled"
                        )
                    events = native.ExecutionEvent()
                    for event in planning_loop._pd_hooks.poll_transfer_events():
                        events.add_event(event)
                    p_core.advance(events)
                    assert p_core.active_lcm_blocks() == 0
                    assert not prefill.has_inflight_transfers()
                    assert not written_sessions
                    return
            decode.register("req", info)
            step_decode()
            assert d_core.active_lcm_blocks() > 0
            _wait_for(
                lambda: prefill.kv_manager.room_status(7) == TransferPoll.Bootstrapped
            )
            if readiness_order == "during_forward":
                assert not source_computed
                assert not written_sessions
                complete_prefill(compute)
                assert p_core.awaiting_destination_size() == 0
            send = plan_prefill()
            assert send.remote_decode is not None
            assert not any(batch.request_ids for batch in send.forward)
            complete_prefill(send)

        def advance_pair():
            complete_prefill(plan_prefill())
            step_decode()
            return d_core.decoding_size() == 1 and not prefill.has_inflight_transfers()

        _wait_for(advance_pair)
        assert source_computed
        assert p_core.active_lcm_blocks() == 0
        assert d_core.active_lcm_blocks() > 0
        assert written_sessions == [decode_session]
    finally:
        bootstrap.close()


class _FrozenRoundComplete(Exception):
    pass


class _StopAfterPause(PauseHooks):
    def paused_idle_step(self):
        super().paused_idle_step()
        raise _FrozenRoundComplete


class _FrozenPlanningLoop(_PrefillPlanningLoop):
    def __init__(self, scheduler, transfer):
        super().__init__(scheduler, transfer)
        self.requires_dp_sync = False
        self._pause.state = PauseState.PAUSED_ALL
        self._pause_hooks = _StopAfterPause(self, self._pause, device=None)


def test_frozen_loop_preserves_queued_request_without_l3():
    core = _loopback_scheduler("fused", None)
    loop = _FrozenPlanningLoop(core, None)
    with pytest.raises(_FrozenRoundComplete):
        loop.event_loop()
    assert core.waiting_size() == 1
    assert core.prefilling_size() == 0
    assert core.active_lcm_blocks() == 0
    plan = core.next_execution_plan()
    assert any(batch.request_ids == ["req"] for batch in plan.forward)
