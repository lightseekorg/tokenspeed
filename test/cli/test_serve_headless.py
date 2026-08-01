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

"""``ts serve --headless`` CLI routing (engine-only, no gateway spawn)."""

from __future__ import annotations

import os
import sys
import types
from argparse import Namespace

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from tokenspeed.cli._argsplit import split_headless_argv

HANDSHAKE = [
    "--data-parallel-address",
    "127.0.0.1",
    "--data-parallel-rpc-port",
    "25152",
]


class TestSplitHeadlessArgv:
    def test_absent_returns_none(self):
        assert split_headless_argv(["--model", "/tmp/m"]) is None

    def test_strips_flag_and_passes_server_args_verbatim(self):
        argv = [
            "--headless",
            "--model",
            "/tmp/m",
            *HANDSHAKE,
            "--zmq-engine-index",
            "0",
        ]
        assert split_headless_argv(argv) == [
            "--model",
            "/tmp/m",
            *HANDSHAKE,
            "--zmq-engine-index",
            "0",
        ]

    def test_flag_position_does_not_matter(self):
        assert split_headless_argv(["--model", "/tmp/m", "--headless"]) == [
            "--model",
            "/tmp/m",
        ]

    def test_positional_model_left_for_server_args_parser(self):
        # ServerArgs' own parser accepts the positional model; no rewrite here.
        assert split_headless_argv(["/tmp/m", "--headless"]) == ["/tmp/m"]

    @pytest.mark.parametrize(
        "flag",
        [
            "--engine-startup-timeout",
            "--gateway-startup-timeout",
            "--drain-timeout",
            "--control-port",
        ],
    )
    def test_orchestrator_flags_rejected(self, flag):
        with pytest.raises(ValueError, match="not valid with --headless"):
            split_headless_argv(["--headless", "--model", "/tmp/m", flag, "30"])

    def test_orchestrator_flag_equals_form_rejected(self):
        with pytest.raises(ValueError, match="not valid with --headless"):
            split_headless_argv(["--headless", "--drain-timeout=30"])


class TestHeadlessDispatch:
    def test_run_smg_from_args_routes_to_headless(self, monkeypatch):
        """--headless bypasses the orchestrator (no install check, no run_smg)."""
        from tokenspeed.cli import serve_smg

        called = {}
        monkeypatch.setattr(
            serve_smg, "_run_headless", lambda argv: called.setdefault("argv", argv)
        )

        def fail(*a, **kw):
            raise AssertionError("orchestrator path must not run in headless mode")

        monkeypatch.setattr(serve_smg, "_check_serve_extra_installed", fail)
        monkeypatch.setattr(serve_smg, "split_argv", fail)

        serve_smg.run_smg_from_args(
            Namespace(), ["--headless", "--model", "/tmp/m", *HANDSHAKE]
        )
        assert called["argv"] == ["--model", "/tmp/m", *HANDSHAKE]

    def test_run_headless_calls_engine_entrypoint(self, monkeypatch):
        """_run_headless reuses runtime.entrypoints.engine and sets the proc title."""
        from tokenspeed.cli import serve_smg

        calls = {}
        stub_engine = types.ModuleType("tokenspeed.runtime.entrypoints.engine")
        stub_engine.run_scheduler_headless_from_cli = lambda argv: calls.setdefault(
            "argv", list(argv)
        )
        monkeypatch.setitem(
            sys.modules, "tokenspeed.runtime.entrypoints.engine", stub_engine
        )
        fake_setproctitle = types.ModuleType("setproctitle")
        fake_setproctitle.setproctitle = lambda title: calls.setdefault("title", title)
        monkeypatch.setitem(sys.modules, "setproctitle", fake_setproctitle)

        serve_smg._run_headless(["--model", "/tmp/m", *HANDSHAKE])
        assert calls["argv"] == ["--model", "/tmp/m", *HANDSHAKE]
        assert calls["title"] == "ts-serve-headless"

    def test_default_mode_untouched(self, monkeypatch):
        """Without --headless the orchestrator path runs as before."""
        from tokenspeed.cli import _argsplit, serve_smg

        called = {}
        # Keep the test pure-python: the real snapshot imports the full
        # runtime stack. --model routes via _FANOUT_FLAGS regardless.
        monkeypatch.setattr(_argsplit, "_engine_recognized_flags", lambda: set())
        monkeypatch.setattr(serve_smg, "print_logo", lambda: None)
        monkeypatch.setattr(serve_smg, "_check_serve_extra_installed", lambda: None)
        monkeypatch.setattr(serve_smg, "_prewarm_hf_tokenizer", lambda _: None)

        def fail(argv):
            raise AssertionError("headless launcher must not run in default mode")

        monkeypatch.setattr(serve_smg, "_run_headless", fail)

        async def fake_run_smg(**kwargs):
            called.update(kwargs)
            return 0

        monkeypatch.setattr(serve_smg, "run_smg", fake_run_smg)

        with pytest.raises(SystemExit) as exc:
            serve_smg.run_smg_from_args(Namespace(), ["--model", "/tmp/m"])
        assert exc.value.code == 0
        assert called["engine_args"] == ["--model", "/tmp/m"]

    def test_headless_works_bare_without_handshake_flags(self, monkeypatch):
        """`ts serve --headless --model X` alone dials the default endpoint."""
        from tokenspeed.cli import serve_smg

        called = {}
        monkeypatch.setattr(
            serve_smg, "_run_headless", lambda argv: called.setdefault("argv", argv)
        )
        serve_smg.run_smg_from_args(Namespace(), ["--headless", "--model", "/tmp/m"])
        assert called["argv"] == ["--model", "/tmp/m"]

    def test_module_invocation_keeps_headless_in_raw_argv(self, monkeypatch):
        """`python -m tokenspeed.cli serve --headless ...` forwards the flag.

        --headless is registered on the serve subparser for --help, but the
        orchestrator receives the raw argv, so the flag must survive parsing.
        """
        called = {}

        def fake_smg(args, raw_argv):
            called["raw"] = list(raw_argv)
            called["headless_ns"] = args.headless

        monkeypatch.setattr("tokenspeed.cli.serve_smg.run_smg_from_args", fake_smg)
        monkeypatch.setattr(
            sys, "argv", ["ts", "serve", "--headless", "--model", "/tmp/m", *HANDSHAKE]
        )
        from tokenspeed.cli import main

        main()
        assert called["raw"] == ["--headless", "--model", "/tmp/m", *HANDSHAKE]
        assert called["headless_ns"] is True


class TestHandshakeEndpointDefaults:
    """The handshake host/port pair is defaulted, never required."""

    def test_server_args_defaults_and_composition(self):
        # importorskip(exc_type=ImportError): pulls the full runtime stack,
        # unimportable in an env with drifted kernel wheels.
        server_args = pytest.importorskip(
            "tokenspeed.runtime.utils.server_args", exc_type=ImportError
        )

        fields = server_args.ServerArgs.__dataclass_fields__
        assert fields["data_parallel_address"].default == "127.0.0.1"
        port = fields["data_parallel_rpc_port"].default
        assert port == 30500
        # Outside the 20000-29999 range a frontend may use for derived
        # per-worker handshake ports.
        assert not 20000 <= port <= 29999

        composed = server_args.ServerArgs.zmq_handshake_endpoint(
            types.SimpleNamespace(
                data_parallel_address="127.0.0.1", data_parallel_rpc_port=30500
            )
        )
        assert composed == "tcp://127.0.0.1:30500"

    @pytest.mark.parametrize(
        "flag, value",
        [
            ("--data-parallel-rpc-port", "65536"),  # above u16
            ("--data-parallel-rpc-port", "-1"),
            ("--data-parallel-address", ""),
        ],
    )
    def test_parse_rejects_invalid_handshake_values(self, flag, value, capsys):
        import argparse

        server_args = pytest.importorskip(
            "tokenspeed.runtime.utils.server_args", exc_type=ImportError
        )

        parser = argparse.ArgumentParser(allow_abbrev=False)
        server_args.ServerArgs.add_cli_args(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(["--model", "/tmp/m", flag, value])
        capsys.readouterr()  # swallow argparse usage noise
