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

"""Verify the cli/ package keeps the same surface as the old cli.py."""

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


def test_serve_dispatches_to_native(monkeypatch):
    """``ts serve`` routes to ``serve_native.run_native``.

    We stub ``ServerArgs`` so the test doesn't pull in torch / kernel libs.
    """
    import types

    fake_module = types.ModuleType("tokenspeed.runtime.utils.server_args")

    class FakeServerArgs:
        @staticmethod
        def add_cli_args(parser):
            parser.add_argument("--model", type=str, default=None)

        @staticmethod
        def from_cli_args(args):
            return args

    fake_module.ServerArgs = FakeServerArgs
    fake_module.prepare_server_args = lambda argv: argv
    monkeypatch.setitem(
        sys.modules, "tokenspeed.runtime.utils.server_args", fake_module
    )

    called = {}

    def fake_native(args):
        called["native"] = True

    monkeypatch.setattr("tokenspeed.cli.serve_native.run_native", fake_native)
    monkeypatch.setattr(sys, "argv", ["ts", "serve", "--model", "/tmp/fake"])
    from tokenspeed.cli import main

    main()
    assert called == {"native": True}
