"""Tests for the HTTP server sidecar (``tokenspeed.runtime.entrypoints.http_server``).

These run entirely against a mock smg backend — no engine, smg, or GPU needed.
They are written as regression guards for the bugs found while building the
sidecar (PR #305):

  1. Streaming passthrough closed the upstream aiohttp session before FastAPI
     consumed the body iterator, breaking SSE mid-stream. The original mock was
     too fast to catch it (it buffered the whole body before the session
     closed), so the streaming mock here deliberately sleeps between chunks.
  2. Non-streaming passthrough double-encoded the body (JSONResponse(str) wraps
     already-serialized JSON in quotes). The fix returns a raw Response; we
     assert the body is byte-faithful JSON.
  3. gRPC channel was recreated per request (fd/socket leak). The stub must be
     a reused singleton.
  4. gRPC errors surfaced as unhandled 500s. They must map to a clean 503.
  5. --control-port must be parsed as an orchestrator flag, not forwarded to
     the engine or gateway.
"""

import asyncio
import threading
import time
import unittest

import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

# Token chunks the streaming mock emits, one SSE event each, with a delay
# between them so a prematurely-closed upstream session would truncate.
_STREAM_TOKENS = ["1", "2", "3", "4", "5"]
_STREAM_DELAY = 0.1


def _build_mock_smg() -> FastAPI:
    mock = FastAPI()

    @mock.get("/health")
    async def health():
        # smg returns plain-text "OK", not JSON — exercises non-JSON passthrough.
        return PlainTextResponse("OK")

    @mock.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        if body.get("stream"):

            async def _gen():
                for tok in _STREAM_TOKENS:
                    await asyncio.sleep(_STREAM_DELAY)
                    yield f'data: {{"choices":[{{"text":"{tok}"}}]}}\n\n'
                yield "data: [DONE]\n\n"

            return StreamingResponse(_gen(), media_type="text/event-stream")
        # Non-streaming: a JSON object the sidecar must relay byte-faithfully.
        return JSONResponse(
            {
                "choices": [{"text": "paris", "index": 0}],
                "usage": {"completion_tokens": 1},
            }
        )

    @mock.post("/flush_cache")
    async def flush_cache():
        return JSONResponse({"status": "flushed"})

    @mock.api_route("/start_profile", methods=["GET", "POST"])
    async def start_profile():
        return JSONResponse({"status": "profiling"})

    return mock


def _wait(port, path, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            requests.get(f"http://127.0.0.1:{port}{path}", timeout=1)
            return True
        except Exception:
            time.sleep(0.2)
    return False


class TestProxyPassthrough(unittest.TestCase):
    """Integration tests of the smg-passthrough path against a mock backend."""

    MOCK_PORT = 28310
    SIDECAR_PORT = 28311

    @classmethod
    def setUpClass(cls):
        from tokenspeed.runtime.entrypoints import http_server as hs

        cls.hs = hs
        hs._gateway_url = f"http://127.0.0.1:{cls.MOCK_PORT}"
        hs._engine_grpc_addr = "127.0.0.1:1"  # dead — only gRPC tests touch it

        cls._mock_server = uvicorn.Server(
            uvicorn.Config(
                _build_mock_smg(),
                host="127.0.0.1",
                port=cls.MOCK_PORT,
                log_level="error",
            )
        )
        cls._sidecar_server = uvicorn.Server(
            uvicorn.Config(
                hs.app, host="127.0.0.1", port=cls.SIDECAR_PORT, log_level="error"
            )
        )
        cls._t_mock = threading.Thread(target=cls._mock_server.run, daemon=True)
        cls._t_side = threading.Thread(target=cls._sidecar_server.run, daemon=True)
        cls._t_mock.start()
        cls._t_side.start()
        assert _wait(cls.MOCK_PORT, "/health"), "mock smg failed to start"
        assert _wait(cls.SIDECAR_PORT, "/health"), "sidecar failed to start"

    @classmethod
    def tearDownClass(cls):
        cls._mock_server.should_exit = True
        cls._sidecar_server.should_exit = True

    def _url(self, path):
        return f"http://127.0.0.1:{self.SIDECAR_PORT}{path}"

    # -- bug 1: streaming session lifetime ----------------------------------

    def test_streaming_completes_all_chunks(self):
        """Regression: SSE must stream all delayed chunks without the upstream
        session closing mid-stream."""
        chunks = []
        with requests.post(
            self._url("/v1/completions"),
            json={"model": "m", "prompt": "hi", "max_tokens": 5, "stream": True},
            stream=True,
            timeout=15,
        ) as r:
            self.assertEqual(r.status_code, 200)
            self.assertIn("text/event-stream", r.headers.get("content-type", ""))
            for raw in r.iter_lines():
                if raw:
                    chunks.append(raw.decode())

        # Every token must arrive, in order, followed by [DONE].
        for tok in _STREAM_TOKENS:
            self.assertTrue(
                any(f'"text":"{tok}"' in c for c in chunks),
                f"missing streamed token {tok!r}; got {chunks}",
            )
        self.assertTrue(any("[DONE]" in c for c in chunks), f"no [DONE]: {chunks}")

    def test_streaming_is_incremental(self):
        """The response must arrive incrementally, not buffered all-at-once —
        proves the streaming path isn't accidentally reading the full body."""
        first_byte_t = None
        last_byte_t = None
        with requests.post(
            self._url("/v1/completions"),
            json={"model": "m", "prompt": "hi", "max_tokens": 5, "stream": True},
            stream=True,
            timeout=15,
        ) as r:
            for raw in r.iter_lines():
                if raw:
                    now = time.monotonic()
                    if first_byte_t is None:
                        first_byte_t = now
                    last_byte_t = now
        # With 5 chunks * 0.1s delay, spread between first and last must be
        # clearly non-zero if streaming is truly incremental.
        self.assertIsNotNone(first_byte_t)
        self.assertGreater(
            last_byte_t - first_byte_t,
            _STREAM_DELAY,
            "stream did not arrive incrementally",
        )

    # -- bug 2: non-streaming double-encoding -------------------------------

    def test_non_streaming_body_is_faithful_json(self):
        """Regression: body must be a JSON object, not a double-encoded JSON
        string (the JSONResponse(str) bug wrapped it in quotes)."""
        r = requests.post(
            self._url("/v1/completions"),
            json={"model": "m", "prompt": "hi", "max_tokens": 4},
            timeout=10,
        )
        self.assertEqual(r.status_code, 200)
        # Raw body must be a JSON object literal.
        self.assertTrue(
            r.text.lstrip().startswith("{"), f"body looks double-encoded: {r.text[:80]}"
        )
        data = r.json()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["choices"][0]["text"], "paris")

    # -- health proxy (plain-text passthrough) ------------------------------

    def test_health_proxies_plaintext(self):
        r = requests.get(self._url("/health"), timeout=10)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.text, "OK")

    # -- other passthrough routes -------------------------------------------

    def test_flush_cache_passthrough(self):
        r = requests.post(self._url("/flush_cache"), timeout=10)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "flushed")

    def test_start_profile_passthrough(self):
        r = requests.post(self._url("/start_profile"), timeout=10)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "profiling")

    def test_status_code_relayed(self):
        """A non-200 from smg must be relayed, not masked as 200."""
        r = requests.get(self._url("/v1/models"), timeout=10)  # mock has no such route
        self.assertEqual(r.status_code, 404)


class TestGrpcDirect(unittest.TestCase):
    """Unit tests for the gRPC-direct path (no live engine)."""

    def setUp(self):
        from tokenspeed.runtime.entrypoints import http_server as hs

        self.hs = hs
        # Reset the cached channel/stub between tests.
        hs._grpc_channel = None
        hs._grpc_stub = None
        hs._engine_grpc_addr = "127.0.0.1:1"

    # -- bug 3: gRPC channel reuse ------------------------------------------

    def test_stub_is_reused(self):
        """Regression: _stub() must return the same instance across calls
        (creating a new channel per request leaks fds/sockets)."""
        s1 = self.hs._stub()
        s2 = self.hs._stub()
        self.assertIs(s1, s2)
        self.assertIsNotNone(self.hs._grpc_channel)

    # -- bug 4: gRPC error -> 503 -------------------------------------------

    def test_grpc_error_maps_to_503(self):
        """Regression: a failed engine RPC must yield a clean 503, not a 500."""
        port = 28321
        server = uvicorn.Server(
            uvicorn.Config(self.hs.app, host="127.0.0.1", port=port, log_level="error")
        )
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        try:
            assert _wait(port, "/get_server_info")
            r = requests.get(f"http://127.0.0.1:{port}/get_server_info", timeout=10)
            self.assertEqual(r.status_code, 503)
            body = r.json()
            self.assertEqual(body["error"], "engine unavailable")
            self.assertIn("detail", body)
        finally:
            server.should_exit = True


class TestProxyTimeout(unittest.TestCase):
    """The proxy must not impose a wall-clock `total` timeout, which would cut
    off long but healthy streaming generations (P2 review fix)."""

    def test_no_total_cap_but_inactivity_bounded(self):
        from tokenspeed.runtime.entrypoints import http_server as hs

        self.assertIsNone(
            hs._PROXY_TIMEOUT.total,
            "a total timeout would abort long streams mid-flight",
        )
        # Inactivity / connect bounds must still be set so a hung upstream is
        # detected.
        self.assertIsNotNone(hs._PROXY_TIMEOUT.sock_read)
        self.assertIsNotNone(hs._PROXY_TIMEOUT.sock_connect)


class TestControlServerReadiness(unittest.TestCase):
    """`_start_control_server` must wait for an actual bind before reporting
    ready, and report failure when the port is unavailable (P2 review fix)."""

    def test_reports_ready_after_bind(self):
        from tokenspeed.cli.serve_smg import _start_control_server

        port = 28330
        ok = asyncio.run(
            _start_control_server(
                gateway_url="http://127.0.0.1:1",
                engine_grpc_addr="127.0.0.1:1",
                host="127.0.0.1",
                port=port,
                timeout=15,
            )
        )
        self.assertTrue(ok)
        # The socket must really be accepting connections now.
        self.assertTrue(_wait(port, "/health", timeout=2))

    def test_reports_failure_when_port_in_use(self):
        from tokenspeed.cli.serve_smg import _start_control_server

        port = 28331
        # Occupy the port first.
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
        sock.listen(1)
        try:
            ok = asyncio.run(
                _start_control_server(
                    gateway_url="http://127.0.0.1:1",
                    engine_grpc_addr="127.0.0.1:1",
                    host="127.0.0.1",
                    port=port,
                    timeout=5,
                )
            )
            self.assertFalse(ok, "should report failure when port is occupied")
        finally:
            sock.close()


class TestControlPortArg(unittest.TestCase):
    """--control-port orchestrator-flag parsing (bug 5)."""

    def test_control_port_parsed(self):
        from tokenspeed.cli._argsplit import split_argv

        result = split_argv(["--model", "m", "--control-port", "8081"])
        self.assertEqual(result.opts.control_port, 8081)

    def test_control_port_not_forwarded(self):
        from tokenspeed.cli._argsplit import split_argv

        result = split_argv(["--model", "m", "--control-port", "8081"])
        self.assertNotIn("--control-port", result.engine)
        self.assertNotIn("--control-port", result.gateway)

    def test_control_port_defaults_none(self):
        from tokenspeed.cli._argsplit import split_argv

        result = split_argv(["--model", "m"])
        self.assertIsNone(result.opts.control_port)


if __name__ == "__main__":
    unittest.main()
