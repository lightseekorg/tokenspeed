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
``MooncakeKvStore`` / ``L3HostStore`` over TCP with ``P2PHANDSHAKE``. Skip when
the Mooncake package is not installed, unless ``MOONCAKE_REQUIRE_MASTER=1``.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import unittest
from contextlib import closing

from tokenspeed.runtime.cache.l3.executor import L3HostStore
from tokenspeed.runtime.cache.l3.factory import create_kvstore_storage_backend
from tokenspeed.runtime.cache.l3.mooncake import MooncakeKvStore


def _require_or_skip(reason: str) -> None:
    flag = os.environ.get("MOONCAKE_REQUIRE_MASTER", "").strip().lower()
    if flag in {"1", "true", "yes"}:
        raise RuntimeError(reason)
    raise unittest.SkipTest(reason)


def _libcudart_dir() -> str | None:
    try:
        import nvidia.cuda_runtime as cuda_runtime
    except ImportError:
        return None
    for location in getattr(cuda_runtime, "__path__", []):
        candidate = os.path.join(location, "lib", "libcudart.so.12")
        if os.path.isfile(candidate):
            return os.path.dirname(candidate)
    return None


def _ensure_libcudart() -> str | None:
    """Load ``libcudart`` into this process and export it for child binaries.

    ``mooncake_master`` and ``MooncakeDistributedStore`` are linked against
    CUDA 12 even in the non-CUDA transfer-engine wheel. Changing
    ``LD_LIBRARY_PATH`` after Python starts does not affect this process's
    ``dlopen``, so preload via ``ctypes`` as well.
    """

    lib_dir = _libcudart_dir()
    if lib_dir is None:
        return None
    ctypes.CDLL(
        os.path.join(lib_dir, "libcudart.so.12"),
        mode=ctypes.RTLD_GLOBAL,
    )
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [part for part in current.split(os.pathsep) if part]
    if lib_dir not in parts:
        os.environ["LD_LIBRARY_PATH"] = lib_dir + (
            os.pathsep + current if current else ""
        )
    return lib_dir


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_tcp(host: str, port: int, timeout_s: float = 20.0) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            with closing(socket.create_connection((host, port), timeout=0.2)):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.1)
    raise TimeoutError(f"Mooncake master {host}:{port} did not come up: {last_error}")


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
    def __init__(self, *, nbytes: int = 4096, page: int = 64):
        self.host_buffer = _PtrBuffer(nbytes)
        self._page = int(page)

    def host_block_range(self, group_index: int, block_id: int) -> tuple[int, int]:
        del group_index
        return (int(block_id) - 1) * self._page, self._page


class MooncakeMasterLiveTest(unittest.TestCase):
    master_proc: subprocess.Popen[bytes] | None = None
    master_log = None
    master_addr = ""
    started_master = False
    _lib_dir: str | None = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._lib_dir = _ensure_libcudart()
        try:
            import mooncake
            from mooncake.store import MooncakeDistributedStore  # noqa: F401
        except ImportError as exc:
            _require_or_skip(f"mooncake package is not installed: {exc}")

        existing = os.environ.get("MOONCAKE_MASTER", "").strip()
        if existing:
            host, _, port_text = existing.rpartition(":")
            _wait_tcp(host or "127.0.0.1", int(port_text))
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
        port = _free_port()
        metrics_port = _free_port()
        log_dir = os.environ.get("TEST_MOONCAKE_LOG_DIR", "/tmp/mooncake-master-test")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"master-{port}.log")
        env = os.environ.copy()
        if cls._lib_dir:
            env["LD_LIBRARY_PATH"] = (
                cls._lib_dir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
            )
        cls.master_log = open(log_path, "wb")
        cls.master_proc = subprocess.Popen(
            [
                exe,
                f"--port={port}",
                f"--metrics_port={metrics_port}",
                "--enable_http_metadata_server=false",
                "--enable_metric_reporting=false",
                f"--log_dir={log_dir}",
            ],
            stdout=cls.master_log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        cls.started_master = True
        cls.master_addr = f"127.0.0.1:{port}"
        try:
            _wait_tcp("127.0.0.1", port)
        except TimeoutError:
            cls._stop_master()
            output = b""
            if os.path.isfile(log_path):
                with open(log_path, "rb") as log_file:
                    output = log_file.read(2000)
            _require_or_skip(
                f"mooncake_master failed to bind {cls.master_addr}: {output!r}"
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
    def tearDownClass(cls) -> None:
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

    def test_l3_host_store_round_trips_through_mooncake_master(self) -> None:
        host = _Host()
        payload = b"mooncake-l3-bytes"
        host.host_buffer[0 : len(payload)] = payload
        backend = create_kvstore_storage_backend(
            "mooncake",
            json.dumps(self._extra_config()),
            host_buffer=host.host_buffer,
            tp_size=1,
            pp_size=1,
        )
        self.assertIsInstance(backend, MooncakeKvStore)
        l3 = L3HostStore(backend, host, key_prefix="live", rank=0)
        pages = [(0, 1, "h0", 0)]
        self.assertEqual(l3.backup(pages), [True])
        self.assertEqual(l3.exists(pages), [True])
        host.host_buffer[0 : len(payload)] = b"\x00" * len(payload)
        self.assertEqual(
            bytes(host.host_buffer[0 : len(payload)]), b"\x00" * len(payload)
        )
        self.assertEqual(l3.prefetch(pages), [True])
        self.assertEqual(bytes(host.host_buffer[0 : len(payload)]), payload)
        l3.rotate_namespace()
        self.assertEqual(l3.exists(pages), [False])
        l3.close()


if __name__ == "__main__":
    unittest.main()
