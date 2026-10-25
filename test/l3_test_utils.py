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

"""Shared pieces of the L3 tests: real schedulers with Host storage, the
engine host's L3 executor (``tokenspeed_engine.L3Executor``) over a test's
Host pool, and a local Mooncake master for the live store tests."""

from __future__ import annotations

import importlib.util
import os
import shutil
import socket
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class StorageOp:
    """A scheduler storage operation (``Cache.Storage*Op``) as a test writes
    one: ``kind`` is a ``tokenspeed_scheduler.StorageKind`` and ``keys`` are
    ``(group, content_hash, page_offset)``."""

    op_id: int
    kind: object
    keys: list[tuple[int, str, int]]
    host_blocks: list[int]


def start_l3_executor(
    host_buffer,
    *,
    groups: tuple[tuple[int, int, int], ...],
    lcm_block_bytes: int,
    owner_rank: int,
    tp_rank: int,
    deployment_id: str,
    mooncake,
):
    """The engine host's L3 executor over ``host_buffer``, a CPU uint8
    tensor of whole LCM blocks the caller keeps until the executor closes.

    ``groups`` are ``(packing, block_bytes, shard_count)`` per cache group;
    ``mooncake`` is a ``MooncakeStoreSettings``, or None for the process's
    in-memory store. Objects are named under ``deployment_id`` at weight
    version ``v1``.
    """
    from tokenspeed_engine import L3Executor

    from tokenspeed.runtime.cache.l3.config import L3Settings, StorageRuntimeConfig

    num_lcm_blocks = host_buffer.numel() // lcm_block_bytes
    return L3Executor(
        L3Settings(
            runtime=StorageRuntimeConfig.from_extra_config(
                None, host_blocks=num_lcm_blocks, prefix_granularity=1
            ),
            deployment_id=deployment_id,
            has_draft=False,
            weight_version="v1",
            host_buffer=host_buffer.data_ptr(),
            host_buffer_bytes=num_lcm_blocks * lcm_block_bytes,
            groups=groups,
            lcm_block_bytes=lcm_block_bytes,
            num_lcm_blocks=num_lcm_blocks,
            owner_rank=owner_rank,
            tp_rank=tp_rank,
            backend="memory" if mooncake is None else "mooncake",
            mooncake=mooncake,
        )
    )


def host_storage_executor(host_storage, *, shard_counts, rank, deployment_id, mooncake):
    """The L3 executor over a ``HostCacheStorage``'s pool, as the engine host
    starts it for attention-TP rank ``rank`` (also its KV-parallel rank)."""
    return start_l3_executor(
        host_storage.host_buffer,
        groups=tuple(
            zip(
                host_storage.group_packing,
                host_storage.host_cache_block_bytes,
                shard_counts,
                strict=True,
            )
        ),
        lcm_block_bytes=host_storage.host_lcm_block_bytes,
        owner_rank=rank,
        tp_rank=rank,
        deployment_id=deployment_id,
        mooncake=mooncake,
    )


def run_storage(executor, operation: StorageOp, *, timeout_s: float = 20.0):
    """Submit ``operation`` and wait for its completion: ``(op_id, success,
    retired, object_failures)``."""
    executor.submit([operation])
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        _, completions, _ = executor.poll()
        if completions:
            (completion,) = completions
            assert completion[0] == operation.op_id, completion
            return completion
        time.sleep(0.001)
    raise AssertionError(f"storage operation {operation.op_id} did not complete")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("0.0.0.0", 0))
        return probe.getsockname()[1]


def store_library() -> str | None:
    """The store's C ABI the installed tokenspeed-mooncake ships, if any."""
    spec = importlib.util.find_spec("mooncake")
    if spec is None or not spec.submodule_search_locations:
        return None
    path = os.path.join(spec.submodule_search_locations[0], "libmooncake_store.so")
    return path if os.path.exists(path) else None


@contextmanager
def mooncake_master():
    """A local ``mooncake_master``'s address for the live store tests. Skips
    without the tokenspeed-mooncake store library, unless
    ``MOONCAKE_REQUIRE_MASTER=1`` makes that a failure."""
    import pytest

    library = store_library()
    if library is None:
        if os.environ.get("MOONCAKE_REQUIRE_MASTER", "").lower() in ("1", "true"):
            raise RuntimeError("MOONCAKE_REQUIRE_MASTER needs libmooncake_store.so")
        pytest.skip("needs the tokenspeed-mooncake store library")
    binary = os.path.join(os.path.dirname(library), "mooncake_master")
    if not os.path.isfile(binary):
        binary = shutil.which("mooncake_master") or binary
    port = _free_port()
    with tempfile.TemporaryDirectory() as log_dir:
        master = subprocess.Popen(
            [
                binary,
                f"--port={port}",
                f"--metrics_port={_free_port()}",
                "--enable_http_metadata_server=false",
                "--enable_metric_reporting=false",
                f"--log_dir={log_dir}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 20
            while True:
                if master.poll() is not None:
                    raise RuntimeError(f"mooncake_master exited: {master.returncode}")
                try:
                    socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.1)
            yield f"127.0.0.1:{port}"
        finally:
            master.kill()
            master.wait()


def mooncake_client(master: str, *, global_segment_size: int):
    """The settings of a TCP store client of ``master`` mounting
    ``global_segment_size`` bytes (0: none)."""
    from tokenspeed.runtime.cache.l3.config import MooncakeStoreSettings

    return MooncakeStoreSettings.resolve(
        {
            "master_server_address": master,
            "local_hostname": "localhost",
            "metadata_server": "P2PHANDSHAKE",
            "protocol": "tcp",
            "device_name": "",
            "global_segment_size": global_segment_size,
        },
        library=store_library(),
        tp_size=1,
        pp_size=1,
    )


def storage_ops(plan):
    """The Host-only commands in a scheduler's unified cache plan."""
    from tokenspeed_scheduler import Cache

    types = (
        Cache.StorageQueryOp,
        Cache.StorageGetOp,
        Cache.StoragePutOp,
        Cache.StorageCancelOp,
    )
    return [operation for operation in plan.cache if isinstance(operation, types)]


def storage_scheduler():
    """A small real scheduler with explicit Host-storage credits."""
    import tokenspeed_scheduler as ts

    cfg = ts.SchedulerConfig()
    cfg.role = ts.SchedulerConfig.Role.Fused
    cfg.prefix_granularity = 2
    cfg.num_device_pages = 32
    cfg.num_host_pages = 8
    cfg.num_snapshot_pages = 1
    cfg.max_scheduled_tokens = 8
    cfg.max_batch_size = 4
    cfg.disable_l2_cache = False
    cfg.enable_l3_storage = True
    cfg.storage_config = ts.StorageConfig(
        8,
        8,
        4,
        4,
        32,
        1,
        1000,
        1000,
        ts.StoragePressure(8, 65536, 1000),
        ts.StoragePressure(4, 32768, 500),
    )
    cfg.cache_groups = [ts.CacheGroupConfig("full", 2, 32, ts.AttnKind.Full)]
    scheduler = ts.Scheduler(cfg)
    tick = ts.ExecutionEvent()
    tick.add_event(ts.Cache.StorageTick(1, True, True, ts.StoragePressure(0, 0, 0)))
    scheduler.advance(tick)
    request = ts.RequestSpec()
    request.request_id = "waiting"
    request.tokens = [1, 2, 3, 4]
    scheduler.submit_requests([request])
    return scheduler
