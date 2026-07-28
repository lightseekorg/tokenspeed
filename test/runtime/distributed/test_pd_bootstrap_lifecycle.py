"""CPU-only lifecycle tests for the shared PD/EPD bootstrap server."""

from __future__ import annotations

import http.client
import os
import socket
import sys

import pytest
from aiohttp import web

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=35, suite="runtime-1gpu")

from tokenspeed.runtime.pd.base.bootstrap import DisaggBootstrapServerBase


def _unused_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _get_health(port: int) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def test_constructor_returns_only_after_site_is_bound_and_fields_exist() -> None:
    observed_fields: dict[str, bool] = {}

    class _InspectingServer(DisaggBootstrapServerBase):
        def _run_server(self) -> None:
            for name in (
                "_loop",
                "_runner",
                "_site",
                "_startup_error",
                "_runtime_error",
                "_startup_complete",
                "_stop_requested",
                "_lifecycle_lock",
                "_thread_started",
                "thread",
            ):
                observed_fields[name] = hasattr(self, name)
            super()._run_server()

    port = _unused_tcp_port()
    server = _InspectingServer(port)
    try:
        assert all(observed_fields.values())
        assert server._startup_complete.is_set()
        assert server._startup_error is None
        assert server.thread.is_alive()
        assert _get_health(port) == (200, b"OK")
    finally:
        server.close()


def test_tcp_start_failure_is_raised_by_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fail_start(_site: web.TCPSite) -> None:
        raise OSError("synthetic bind failure")

    monkeypatch.setattr(web.TCPSite, "start", _fail_start)

    with pytest.raises(
        RuntimeError, match="Bootstrap server failed to bind"
    ) as exc_info:
        DisaggBootstrapServerBase(_unused_tcp_port())

    assert isinstance(exc_info.value.__cause__, OSError)
    assert "synthetic bind failure" in str(exc_info.value.__cause__)


def test_close_is_idempotent_and_releases_listening_port() -> None:
    port = _unused_tcp_port()
    first = DisaggBootstrapServerBase(port)

    first.close()
    first.close()

    assert not first.thread.is_alive()

    # Successful immediate reuse proves AppRunner cleanup closed the listener,
    # rather than merely returning after asking the loop to stop.
    second = DisaggBootstrapServerBase(port)
    try:
        assert _get_health(port) == (200, b"OK")
    finally:
        second.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
