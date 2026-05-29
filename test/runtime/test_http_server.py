"""Tests for the control-plane HTTP server sidecar."""

import os
import sys
import threading
import time
import unittest

import requests
import uvicorn

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

CONTROL_HOST = "127.0.0.1"


class TestControlHttpServerEndpoints(unittest.TestCase):
    """Test /health and /readiness without a real engine or smg."""

    @classmethod
    def setUpClass(cls):
        from tokenspeed.runtime.entrypoints import http_server as hs

        cls.port = 21101
        config = uvicorn.Config(
            hs.app, host=CONTROL_HOST, port=cls.port, log_level="error"
        )
        cls.server = uvicorn.Server(config)
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        # Wait for the server to be ready
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                requests.get(f"http://{CONTROL_HOST}:{cls.port}/health", timeout=1)
                break
            except Exception:
                time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True

    def test_health(self):
        r = requests.get(f"http://{CONTROL_HOST}:{self.port}/health", timeout=5)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_readiness(self):
        r = requests.get(f"http://{CONTROL_HOST}:{self.port}/readiness", timeout=5)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ready")


class TestOrchestratorControlPort(unittest.TestCase):
    """Test that --control-port is parsed by split_argv."""

    def test_control_port_parsed(self):
        from tokenspeed.cli._argsplit import split_argv

        result = split_argv(["--model", "m", "--control-port", "8081"])
        self.assertEqual(result.opts.control_port, 8081)

    def test_control_port_not_in_engine_or_gateway_args(self):
        from tokenspeed.cli._argsplit import split_argv

        result = split_argv(["--model", "m", "--control-port", "8081"])
        self.assertNotIn("--control-port", result.engine)
        self.assertNotIn("--control-port", result.gateway)

    def test_no_control_port_defaults_none(self):
        from tokenspeed.cli._argsplit import split_argv

        result = split_argv(["--model", "m"])
        self.assertIsNone(result.opts.control_port)


if __name__ == "__main__":
    unittest.main()
