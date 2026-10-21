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

"""Live Mooncake Store L3 tests against a real ``mooncake_master``.

The test boots ``mooncake_master`` (or reuses ``MOONCAKE_MASTER``) and drives
``MooncakeKvStore`` / ``L3Executor`` over TCP with ``P2PHANDSHAKE``. Skip when
the Mooncake package is not installed, unless ``MOONCAKE_REQUIRE_MASTER=1``.
"""

from __future__ import annotations

import ctypes
import errno
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import closing, contextmanager
from pathlib import Path

import pytest

from tokenspeed.runtime.cache.l3.factory import create_storage_backend
from tokenspeed.runtime.cache.l3.mooncake import MooncakeKvStore


def _require_or_skip(reason: str) -> None:
    flag = os.environ.get("MOONCAKE_REQUIRE_MASTER", "").strip().lower()
    if flag in {"1", "true", "yes"}:
        raise RuntimeError(reason)
    pytest.skip(reason)


def _libcudart_dir() -> str | None:
    try:
        import nvidia.cu13 as cuda_runtime
    except ImportError:
        return None
    for location in getattr(cuda_runtime, "__path__", []):
        candidate = os.path.join(location, "lib", "libcudart.so.13")
        if os.path.isfile(candidate):
            return os.path.dirname(candidate)
    return None


def _ensure_libcudart() -> str | None:
    """Load ``libcudart`` into this process and export it for child binaries.

    ``tokenspeed-mooncake`` links against CUDA 13 even for Host-only TCP
    transfers. Changing
    ``LD_LIBRARY_PATH`` after Python starts does not affect this process's
    ``dlopen``, so preload via ``ctypes`` as well.
    """

    lib_dir = _libcudart_dir()
    if lib_dir is None:
        return None
    ctypes.CDLL(
        os.path.join(lib_dir, "libcudart.so.13"),
        mode=ctypes.RTLD_GLOBAL,
    )
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [part for part in current.split(os.pathsep) if part]
    if lib_dir not in parts:
        os.environ["LD_LIBRARY_PATH"] = lib_dir + (
            os.pathsep + current if current else ""
        )
    return lib_dir


@contextmanager
def _reserve_master_ports():
    # The master binds all interfaces. A loopback-only probe can miss a
    # listener on another address, and both ports must be held while selecting.
    with (
        closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as rpc,
        closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as admin,
    ):
        rpc.bind(("0.0.0.0", 0))
        admin.bind(("0.0.0.0", 0))
        yield int(rpc.getsockname()[1]), int(admin.getsockname()[1])


def _wait_tcp(
    host: str,
    port: int,
    *,
    timeout_s: float,
    process: subprocess.Popen | None,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"Mooncake master {host}:{port} exited before readiness: "
                f"pid={process.pid}, exit_code={process.returncode}"
            )
        try:
            with closing(socket.create_connection((host, port), timeout=0.2)):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.1)
    raise TimeoutError(f"Mooncake master {host}:{port} did not come up: {last_error}")


def _master_logs(log_path: str, pid: int) -> str:
    paths = [Path(log_path)]
    paths.extend(sorted(Path(log_path).parent.glob(f"mooncake_master.*.{pid}")))
    return "\n".join(
        f"{path.name}: {path.read_bytes()[-3000:]!r}"
        for path in paths
        if path.is_file()
    )


def test_master_port_reservations_cover_all_interfaces():
    with _reserve_master_ports() as ports:
        assert ports[0] != ports[1]
        for port in ports:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as peer:
                with pytest.raises(OSError) as error:
                    peer.bind(("127.0.0.2", port))
                assert error.value.errno == errno.EADDRINUSE


def test_owned_master_early_exit_is_reported_before_readiness():
    with _reserve_master_ports() as (port, _):
        process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(23)"])
        try:
            with pytest.raises(RuntimeError, match="exit_code=23"):
                _wait_tcp("127.0.0.1", port, timeout_s=5, process=process)
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)


class _PtrBuffer:
    """CPU byte buffer with a ``data_ptr()`` for Mooncake ``register_buffer``."""

    def __init__(self, nbytes: int):
        self._nbytes = int(nbytes)
        self._raw = bytearray(self._nbytes)
        self._ctype = (ctypes.c_char * self._nbytes).from_buffer(self._raw)

    def data_ptr(self) -> int:
        return ctypes.addressof(self._ctype)

    def numel(self) -> int:
        return self._nbytes

    def element_size(self) -> int:
        return 1

    def __setitem__(self, key, value) -> None:
        self._raw[key] = bytes(value)

    def __getitem__(self, key):
        return self._raw[key]


class _Host:
    def __init__(self, *, nbytes: int, page: int):
        self.host_buffer = _PtrBuffer(nbytes)
        self._page = int(page)

    def host_block_range(self, group_index: int, block_id: int) -> tuple[int, int]:
        del group_index
        return (int(block_id) - 1) * self._page, self._page


class TestMooncakeMasterLiveTest:
    master_proc: subprocess.Popen[bytes] | None = None
    master_log = None
    master_addr = ""
    started_master = False
    _lib_dir: str | None = None

    @classmethod
    def setup_class(cls) -> None:
        cls._lib_dir = _ensure_libcudart()
        try:
            import mooncake
            from mooncake.store import MooncakeDistributedStore  # noqa: F401
        except ImportError as exc:
            _require_or_skip(f"mooncake package is not installed: {exc}")

        existing = os.environ.get("MOONCAKE_MASTER", "").strip()
        if existing:
            host, _, port_text = existing.rpartition(":")
            _wait_tcp(host or "127.0.0.1", int(port_text), timeout_s=20.0, process=None)
            cls.master_addr = existing
            cls.started_master = False
            return

        package_bin = os.path.join(
            os.path.dirname(mooncake.__file__), "mooncake_master"
        )
        exe = (
            package_bin
            if os.path.isfile(package_bin)
            else shutil.which("mooncake_master")
        )
        if not exe or not os.path.isfile(exe):
            _require_or_skip("mooncake_master is not installed")
        log_dir = os.environ.get("TEST_MOONCAKE_LOG_DIR", "/tmp/mooncake-master-test")
        os.makedirs(log_dir, exist_ok=True)
        env = os.environ.copy()
        if cls._lib_dir:
            env["LD_LIBRARY_PATH"] = (
                cls._lib_dir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
            )
        with _reserve_master_ports() as (port, metrics_port):
            log_path = os.path.join(log_dir, f"master-{port}.log")
            cls.master_log = open(log_path, "wb")
            command = [
                exe,
                f"--port={port}",
                f"--metrics_port={metrics_port}",
                "--enable_http_metadata_server=false",
                "--enable_metric_reporting=false",
                f"--log_dir={log_dir}",
            ]
        # Release the reservations only when the child is ready to inherit
        # responsibility for binding both listeners.
        cls.master_proc = subprocess.Popen(
            command,
            stdout=cls.master_log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        cls.started_master = True
        cls.master_addr = f"127.0.0.1:{port}"
        try:
            _wait_tcp("127.0.0.1", port, timeout_s=20.0, process=cls.master_proc)
        except (RuntimeError, TimeoutError) as exc:
            pid = cls.master_proc.pid
            exit_code = cls.master_proc.poll()
            cls._stop_master()
            _require_or_skip(
                f"mooncake_master failed readiness: rpc={cls.master_addr}, "
                f"admin_port={metrics_port}, pid={pid}, "
                f"exit_code_before_cleanup={exit_code}, error={exc}; "
                f"logs={_master_logs(log_path, pid)}"
            )

    @classmethod
    def _stop_master(cls) -> None:
        proc = cls.master_proc
        cls.master_proc = None
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    proc.kill()
                proc.wait(timeout=5)
        log_file = cls.master_log
        cls.master_log = None
        if log_file is not None:
            log_file.close()

    @classmethod
    def teardown_class(cls) -> None:
        if cls.started_master:
            cls._stop_master()

    def _extra_config(self) -> dict:
        return {
            "master_server_address": self.master_addr,
            "local_hostname": "localhost",
            "metadata_server": "P2PHANDSHAKE",
            "protocol": "tcp",
            "global_segment_size": 64 * 1024 * 1024,
            "device_name": "",
        }

    def test_global_objects_survive_async_writer_shutdown(self, request) -> None:
        from tokenspeed.runtime.cache.l3.config import StorageRuntimeConfig
        from tokenspeed.runtime.cache.l3.executor import (
            L3Executor,
            StorageJob,
        )

        owner_host = _Host(nbytes=4096, page=64)
        owner = MooncakeKvStore(
            self._extra_config(),
            host_buffer=owner_host.host_buffer,
            tp_size=1,
            pp_size=1,
        )
        request.addfinalizer(owner.close)
        clients = dict(self._extra_config(), global_segment_size=0)
        config = StorageRuntimeConfig.from_extra_config(
            None, host_blocks=8, prefix_granularity=64
        )
        key = f"global-async-{time.monotonic_ns()}"
        payload = bytes(range(64))

        def client():
            host = _Host(nbytes=4096, page=64)
            backend = MooncakeKvStore(
                clients, host_buffer=host.host_buffer, tp_size=1, pp_size=1
            )
            return L3Executor(
                backend,
                host,
                key_prefix="global-async",
                rank=0,
                ownership=None,
                config=config,
                prefix_for_weight_version=None,
            )

        def complete(executor):
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                results = executor.poll().completions
                if results:
                    assert len(results) == 1
                    assert results[0].retired
                    assert results[0].success == (True,)
                    return
                time.sleep(0.01)
            pytest.fail("native asynchronous storage did not complete")

        writer = client()
        try:
            writer.host_storage.host_buffer[:64] = payload
            writer.submit(StorageJob(1, "put", (key,), (0,), (64,)))
            complete(writer)
        finally:
            writer.close(timeout_s=10)
        reader = client()
        try:
            reader.submit(StorageJob(1, "get", (key,), (128,), (64,)))
            complete(reader)
            assert bytes(reader.host_storage.host_buffer[128:192]) == payload
            assert reader.backend.remove_by_prefix(key)
        finally:
            reader.close(timeout_s=10)

    def test_l3_executor_round_trips_through_mooncake_master(self) -> None:
        from tokenspeed.runtime.cache.l3.config import StorageRuntimeConfig
        from tokenspeed.runtime.cache.l3.executor import L3Executor, StorageJob

        host = _Host(nbytes=4096, page=64)
        payload = b"mooncake-l3-bytes"
        host.host_buffer[0 : len(payload)] = payload
        backend = create_storage_backend(
            "mooncake",
            json.dumps(self._extra_config()),
            host_buffer=host.host_buffer,
            tp_size=1,
            pp_size=1,
        )
        assert isinstance(backend, MooncakeKvStore)
        executor = L3Executor(
            backend,
            host,
            key_prefix="live",
            rank=0,
            ownership=None,
            config=StorageRuntimeConfig.from_extra_config(
                None, host_blocks=8, prefix_granularity=64
            ),
            prefix_for_weight_version=None,
        )

        def run(op_id, kind):
            keys, offsets, sizes = executor._page_ranges([(0, 1, "h0", 0)])
            executor.submit(
                StorageJob(
                    op_id,
                    kind,
                    tuple(keys),
                    () if kind == "query" else tuple(offsets),
                    () if kind == "query" else tuple(sizes),
                )
            )
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                results = executor.poll().completions
                if results:
                    assert len(results) == 1
                    assert results[0].op_id == op_id and results[0].retired
                    return results[0]
                time.sleep(0.01)
            pytest.fail("native asynchronous storage did not complete")

        try:
            assert run(1, "put").success == (True,)
            assert run(2, "query").success == (True,)
            host.host_buffer[0 : len(payload)] = b"\x00" * len(payload)
            assert run(3, "get").success == (True,)
            assert bytes(host.host_buffer[0 : len(payload)]) == payload
            assert executor.delete_namespace()
            assert run(4, "query").success == (False,)
            # Eviction after EXISTS must be a safely retired native miss.
            result = run(5, "get")
            assert result.success == (False,)
            assert result.object_failures == (True,)
        finally:
            executor.close(timeout_s=10)

    def test_monitor_recovers_after_owned_master_restart(self) -> None:
        """An actual metadata outage opens the circuit; a new owner can serve it.

        Only the master created by this fixture may be stopped. This checks
        recovery of cache service, not persistence of the old in-memory objects.
        """
        from dataclasses import replace

        from tokenspeed.runtime.cache.l3.config import StorageRuntimeConfig
        from tokenspeed.runtime.cache.l3.executor import (
            L3Executor,
            StorageJob,
        )

        cls = type(self)
        if not cls.started_master:
            pytest.skip("restart testing requires a fixture-owned master")
        command = cls.master_proc.args
        restarted = False

        def restart():
            nonlocal restarted
            log_dir = os.environ.get(
                "TEST_MOONCAKE_LOG_DIR", "/tmp/mooncake-master-test"
            )
            cls.master_log = open(
                os.path.join(log_dir, f"master-restart-{os.getpid()}.log"), "wb"
            )
            cls.master_proc = subprocess.Popen(
                command,
                stdout=cls.master_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            host, _, port = cls.master_addr.rpartition(":")
            _wait_tcp(host, int(port), timeout_s=20, process=cls.master_proc)
            restarted = True

        def complete(executor):
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                results = executor.poll().completions
                if results:
                    assert len(results) == 1
                    assert results[0].retired
                    return results[0]
                time.sleep(0.01)
            pytest.fail("native storage operation timed out")

        host = _Host(nbytes=4096, page=64)
        config = dict(self._extra_config(), global_segment_size=0)
        backend = MooncakeKvStore(
            config, host_buffer=host.host_buffer, tp_size=1, pp_size=1
        )
        executor = L3Executor(
            backend,
            host,
            key_prefix="master-restart",
            rank=0,
            ownership=None,
            config=replace(
                StorageRuntimeConfig.from_extra_config(
                    None, host_blocks=8, prefix_granularity=64
                ),
                recovery_interval_ms=50,
            ),
            prefix_for_weight_version=None,
        )
        owner = None
        key = f"recovered-{time.monotonic_ns()}"
        try:
            cls._stop_master()
            executor.submit(StorageJob(1, "query", (key,), (), ()))
            assert complete(executor).success == (False,)
            assert executor.snapshot().health.state == "open"
            restart()
            owner_host = _Host(nbytes=4096, page=64)
            owner = MooncakeKvStore(
                self._extra_config(),
                host_buffer=owner_host.host_buffer,
                tp_size=1,
                pp_size=1,
            )
            time.sleep(0.06)
            executor.submit(StorageJob(2, "query", (key,), (), ()))
            assert complete(executor).success == (False,)
            assert executor.snapshot().health.state == "healthy"
            payload = bytes(range(64))
            host.host_buffer[:64] = payload
            executor.submit(StorageJob(3, "put", (key,), (0,), (64,)))
            assert complete(executor).success == (True,)
            host.host_buffer[:64] = bytes(64)
            executor.submit(StorageJob(4, "get", (key,), (128,), (64,)))
            assert complete(executor).success == (True,)
            assert bytes(host.host_buffer[128:192]) == payload
            assert backend.remove_by_prefix(key)
        finally:
            if not restarted:
                restart()
            executor.close(timeout_s=10)
            if owner is not None:
                owner.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
