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

"""Exercise the server argument declarations through a lightweight launcher."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest
from tokenspeed_cli import add_server_cli_args


def test_server_cli_parses_native_flags_with_only_the_standard_library() -> None:
    python_source = Path(__file__).resolve().parents[2] / "python"
    probe = dedent("""
        import argparse
        import sys

        from tokenspeed_cli import add_server_cli_args

        parser = argparse.ArgumentParser()
        add_server_cli_args(parser)
        args = parser.parse_args([
            "--model-path", "target", "--disaggregation-mode", "decode",
            "--speculative-draft-model-path", "draft",
            "--disable-prefix-caching", "--cudagraph-capture-sizes", "2", "4",
            "--metrics-reporters", "prometheus", "--metrics-reporters", "pycat",
            "--numerics", "site-profile",
        ])
        assert args.model == "target"
        assert args.disaggregation_mode == "decode"
        assert args.speculative_draft_model_path == "draft"
        assert args.enable_prefix_caching is False
        assert args.cudagraph_capture_sizes == [2, 4]
        assert args.metrics_reporters == ["prometheus", "pycat"]
        assert args.numerics == "site-profile"
        assert "tokenspeed" not in sys.modules
        """)
    subprocess.run(
        [sys.executable, "-S", "-c", probe],
        env={**os.environ, "PYTHONPATH": str(python_source)},
        check=True,
    )


def test_server_cli_checks_numerics_against_a_supplied_profile_list() -> None:
    parser = argparse.ArgumentParser()
    add_server_cli_args(parser, numerics_envelopes=["auto", "site-profile"])

    args = parser.parse_args(["--model", "target", "--numerics", "site-profile"])
    assert args.numerics == "site-profile"

    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--model", "target", "--numerics", "unregistered"])
    assert error.value.code == 2
