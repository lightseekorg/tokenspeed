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

"""CPU-only tests for the PD transfer side of a sleep release (PD hybrid RL).

A release deregisters the cache arena from the Mooncake engine before the
memory saver unmaps it; a wake registers the re-mapped arena again. The decode
node, whose memory is the target of the prefill node's RDMA writes, takes a
fresh transfer session so no peer keeps writing with the freed pages' keys.
The Mooncake engine is the only fake: it stands in for RDMA registration.
"""

from __future__ import annotations

import os
import sys
import threading

import pytest
import torch

from tokenspeed.runtime.distributed import nccl_suspend

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.pd import (  # noqa: E402
    decode_executor as decode_executor_module,
)
from tokenspeed.runtime.pd import (  # noqa: E402
    prefill_executor as prefill_executor_module,
)
from tokenspeed.runtime.pd.mooncake import decode as decode_module  # noqa: E402
from tokenspeed.runtime.pd.mooncake.entities import KVArgs  # noqa: E402
from tokenspeed.runtime.pd.mooncake.prefill import (  # noqa: E402
    MooncakeKVManagerPrefill,
)

_REGIONS = ((0x1000, 4096), (0x9000, 1024))


@pytest.mark.parametrize(
    "version, bindings, supported",
    [
        ((2, 29, 6), ("suspend", "resume"), False),
        ((2, 29, 7), (), False),
        ((2, 30, 4), ("suspend",), False),
        ((2, 30, 4), ("resume",), False),
        ((2, 29, 7), ("suspend", "resume"), True),
    ],
)
def test_nccl_region_requires_both_bindings_and_a_supported_version(
    monkeypatch, version, bindings, supported
):
    # Stand in for a torch build's bindings and CUDA runtime, without GPUs.
    backend = type("NcclBackend", (), {name: lambda self: None for name in bindings})
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_nccl_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "ProcessGroupNCCL", backend)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda.nccl, "version", lambda: version)

    assert nccl_suspend.nccl_suspend_supported() is supported


class _Engine:
    """Stands in for RDMA registration: tracks which ranges are registered."""

    _sessions = 0

    def __init__(self, hostname: str, gpu_id: int, ib_device: str | None) -> None:
        self.hostname = hostname
        self.gpu_id = gpu_id
        self.ib_device = ib_device
        _Engine._sessions += 1
        self.session_id = f"{hostname}:{17000 + _Engine._sessions}"
        self.registered: dict[int, int] = {}
        self.failing: set[int] = set()

    def register(self, ptr: int, length: int) -> int:
        if ptr in self.failing:
            return -1
        self.registered[ptr] = length
        return 0

    def deregister(self, ptr: int) -> int:
        if ptr in self.failing or ptr not in self.registered:
            return -1
        del self.registered[ptr]
        return 0

    def get_session_id(self) -> str:
        return self.session_id


def _kv_args() -> KVArgs:
    return KVArgs(
        engine_rank=0,
        kv_data_ptr=_REGIONS[0][0],
        ib_device="mlx5_0",
        gpu_id=3,
        cache_layout=None,
        cache_fields_by_stage=(),
        registration_regions=_REGIONS,
    )


def _registered_engine() -> _Engine:
    engine = _Engine("10.0.0.2", 3, "mlx5_0")
    for ptr, length in _REGIONS:
        engine.register(ptr, length)
    return engine


def _prefill_manager() -> MooncakeKVManagerPrefill:
    # The real constructor binds sockets and starts transfer threads; only
    # the registration state is exercised here.
    manager = object.__new__(MooncakeKVManagerPrefill)
    manager.kv_args = _kv_args()
    manager.engine = _registered_engine()
    return manager


def _decode_manager() -> decode_module.MooncakeKVManagerDecode:
    manager = object.__new__(decode_module.MooncakeKVManagerDecode)
    manager.kv_args = _kv_args()
    manager.engine = _registered_engine()
    manager.connection_lock = threading.Lock()
    manager.connection_pool = {"10.0.0.1:9112_0_0": [{"rank_ip": "10.0.0.1"}]}
    return manager


def test_prefill_release_and_wake_reregister_the_arena_on_the_same_session():
    manager = _prefill_manager()
    engine = manager.engine

    manager.deregister_buffer_from_engine()
    assert engine.registered == {}

    manager.restore_buffer_registration()
    # The prefill node initiates every write from its own registration, so
    # no peer holds its keys and the session stays.
    assert manager.engine is engine
    assert engine.registered == dict(_REGIONS)


def test_a_failed_deregistration_fails_the_release_instead_of_unmapping_pinned_pages():
    manager = _prefill_manager()
    manager.engine.failing.add(_REGIONS[1][0])

    with pytest.raises(RuntimeError, match="failed to deregister 1 cache region"):
        manager.deregister_buffer_from_engine()


def test_a_failed_registration_fails_the_wake():
    manager = _prefill_manager()
    manager.deregister_buffer_from_engine()
    manager.engine.failing.add(_REGIONS[0][0])

    with pytest.raises(RuntimeError, match="failed to register 1 cache region"):
        manager.restore_buffer_registration()


def test_decode_wake_registers_under_a_fresh_session_and_forgets_bootstraps(
    monkeypatch,
):
    monkeypatch.setattr(decode_module, "MooncakeTransferEngine", _Engine)
    manager = _decode_manager()
    old_engine = manager.engine

    manager.deregister_buffer_from_engine()
    manager.restore_buffer_registration()

    new_engine = manager.engine
    assert new_engine is not old_engine
    assert new_engine.get_session_id() != old_engine.get_session_id()
    assert (new_engine.hostname, new_engine.gpu_id, new_engine.ib_device) == (
        old_engine.hostname,
        old_engine.gpu_id,
        old_engine.ib_device,
    )
    assert new_engine.registered == dict(_REGIONS)
    assert old_engine.registered == {}
    # The next request bootstraps again and registers the new session with
    # the prefill node.
    assert manager.connection_pool == {}
    assert manager.get_session_id() == new_engine.get_session_id()


class _PrefillExecutor(prefill_executor_module.DisaggPrefillExecutor):
    def __init__(self, manager) -> None:
        self.kv_manager = manager
        self.senders = {}


class _DecodeExecutor(decode_executor_module.DisaggDecodeExecutor):
    def __init__(self, manager) -> None:
        self.kv_manager = manager
        self.receivers = {}


def test_executors_report_and_refuse_releases_with_transfers_in_flight(monkeypatch):
    monkeypatch.setattr(decode_module, "MooncakeTransferEngine", _Engine)
    prefill = _PrefillExecutor(_prefill_manager())
    decode = _DecodeExecutor(_decode_manager())
    assert not prefill.has_inflight_transfers()
    assert not decode.has_inflight_transfers()

    prefill.senders["r0"] = object()
    decode.receivers["r0"] = object()
    assert prefill.has_inflight_transfers() and decode.has_inflight_transfers()
    with pytest.raises(RuntimeError, match="KV sends are in flight"):
        prefill.release_cache_registration()
    with pytest.raises(RuntimeError, match="KV receives are in flight"):
        decode.release_cache_registration()
    # Nothing was deregistered by the refused releases.
    assert prefill.kv_manager.engine.registered == dict(_REGIONS)
    assert decode.kv_manager.engine.registered == dict(_REGIONS)

    prefill.senders.clear()
    decode.receivers.clear()
    prefill.release_cache_registration()
    decode.release_cache_registration()
    prefill.restore_cache_registration()
    decode.restore_cache_registration()
    assert prefill.kv_manager.engine.registered == dict(_REGIONS)
    assert decode.kv_manager.engine.registered == dict(_REGIONS)


def test_nccl_suspend_without_a_process_group_is_a_no_op():
    import torch.distributed as dist

    from tokenspeed.runtime.distributed.nccl_suspend import (
        resume_nccl_communicators,
        suspend_nccl_communicators,
    )

    if dist.is_initialized():
        pytest.skip("another test initialized torch.distributed in this process")
    assert suspend_nccl_communicators() == 0
    assert resume_nccl_communicators() == 0
