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

"""Unit tests for the ``ts serve`` startup banner."""

from __future__ import annotations

import io
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from tokenspeed.cli._logo import (
    _RESET,
    _SPEED_STYLE,
    _TOKEN_STYLE,
    print_logo,
    render_logo,
)


class _TTYBuffer(io.StringIO):
    def isatty(self) -> bool:  # noqa: D401 - trivial
        return True


def test_render_logo_colored_wraps_token_in_grey_and_speed_in_blue():
    out = render_logo("9.9.9", color=True)

    body_rows = out.splitlines()[:-1]
    assert len(body_rows) == 5
    for row in body_rows:
        assert row.startswith(_TOKEN_STYLE)
        assert _SPEED_STYLE in row
        assert row.endswith(_RESET)
        # Token segment must close before the speed one opens.
        assert row.index(_RESET) < row.index(_SPEED_STYLE)

    assert "Tokens at the speed of light · v9.9.9" in out
    assert out.endswith("\n")


def test_render_logo_monochrome_omits_escape_codes():
    out = render_logo("0.0.0", color=False)

    assert "\033[" not in out
    assert "Tokens at the speed of light · v0.0.0" in out


def test_print_logo_writes_to_stream_with_color_on_tty(monkeypatch):
    monkeypatch.delenv("TOKENSPEED_DISABLE_LOGO", raising=False)
    buf = _TTYBuffer()

    print_logo("1.2.3", stream=buf)

    assert _TOKEN_STYLE in buf.getvalue()
    assert _SPEED_STYLE in buf.getvalue()
    assert "Tokens at the speed of light · v1.2.3" in buf.getvalue()


def test_print_logo_omits_color_on_non_tty(monkeypatch):
    monkeypatch.delenv("TOKENSPEED_DISABLE_LOGO", raising=False)
    buf = io.StringIO()  # isatty() -> False by default

    print_logo("1.2.3", stream=buf)

    assert "\033[" not in buf.getvalue()
    assert "Tokens at the speed of light · v1.2.3" in buf.getvalue()


def test_print_logo_disabled_via_env(monkeypatch):
    buf = _TTYBuffer()
    monkeypatch.setenv("TOKENSPEED_DISABLE_LOGO", "1")

    print_logo("1.2.3", stream=buf)

    assert buf.getvalue() == ""


def test_print_logo_zero_value_is_not_disabled(monkeypatch):
    # "0" / "false" / "" must NOT count as "disabled" — only truthy strings.
    buf = _TTYBuffer()
    monkeypatch.setenv("TOKENSPEED_DISABLE_LOGO", "0")

    print_logo("1.2.3", stream=buf)

    assert "Tokens at the speed of light · v1.2.3" in buf.getvalue()
