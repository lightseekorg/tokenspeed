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

"""Verify the cli/ package's top-level dispatch."""

import sys
from unittest.mock import patch


def test_main_re_exported_from_package():
    from tokenspeed.cli import main as pkg_main
    from tokenspeed.cli.__main__ import main as module_main

    assert pkg_main is module_main


def test_version_subcommand_runs(capsys):
    from tokenspeed.cli import main

    with patch.object(sys, "argv", ["ts", "version"]):
        main()
    out = capsys.readouterr().out
    assert out.startswith("TokenSpeed v")


def test_serve_dispatches_to_smg_orchestrator(monkeypatch):
    """``ts serve`` always routes to the smg orchestrator."""
    called = {}

    def fake_smg(args, raw_argv):
        called["smg"] = list(raw_argv)

    monkeypatch.setattr("tokenspeed.cli.serve_smg.run_smg_from_args", fake_smg)
    monkeypatch.setattr(sys, "argv", ["ts", "serve", "--model", "/tmp/fake"])
    from tokenspeed.cli import main

    main()
    assert called == {"smg": ["--model", "/tmp/fake"]}


def test_serve_does_not_validate_engine_choices(monkeypatch):
    """Gateway-valid values must pass through argparse without engine choices=
    validation. Regression: --tool-call-parser qwen3 was being rejected
    because the engine's choices list doesn't include qwen3, even though
    smg's clap accepts it.
    """
    captured = {}

    def fake_smg(args, raw_argv):
        captured["raw"] = list(raw_argv)

    monkeypatch.setattr("tokenspeed.cli.serve_smg.run_smg_from_args", fake_smg)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ts",
            "serve",
            "--tool-call-parser",
            "qwen3",  # only valid for smg, not engine
            "--reasoning-parser",
            "qwen3",  # same
            "--model",
            "/tmp/fake",
        ],
    )
    from tokenspeed.cli import main

    main()
    assert "--tool-call-parser" in captured["raw"]
    assert "qwen3" in captured["raw"]
    assert "--reasoning-parser" in captured["raw"]
