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

"""Startup banner for ``ts serve``: ``Token`` in bright black, ``Speed`` in blue."""

from __future__ import annotations

import os
import sys
from typing import IO

from tokenspeed.version import __version__

_TOKEN_LINES = (
    "████████  ██████  ██   ██ ███████ ███    ██ ",
    "   ██    ██    ██ ██  ██  ██      ████   ██ ",
    "   ██    ██    ██ █████   █████   ██ ██  ██ ",
    "   ██    ██    ██ ██  ██  ██      ██  ██ ██ ",
    "   ██     ██████  ██   ██ ███████ ██   ████ ",
)

_SPEED_LINES = (
    "███████ ██████  ███████ ███████ ██████  ",
    "██      ██   ██ ██      ██      ██   ██ ",
    "███████ ██████  █████   █████   ██   ██ ",
    "     ██ ██      ██      ██      ██   ██ ",
    "███████ ██      ███████ ███████ ██████  ",
)

_TOKEN_STYLE = "\033[90m"  # bright black (grey).
_SPEED_STYLE = "\033[94m"  # bright blue.
_RESET = "\033[0m"

_TAGLINE = "Tokens at the speed of light"
_DISABLE_ENV = "TOKENSPEED_DISABLE_LOGO"
# Visible width of one banner row: len(token) + " " + len(speed).
_BANNER_WIDTH = len(_TOKEN_LINES[0]) + 1 + len(_SPEED_LINES[0])


def _is_disabled() -> bool:
    value = os.environ.get(_DISABLE_ENV, "").strip().lower()
    return value not in ("", "0", "false", "no")


def _stream_supports_color(stream: IO[str]) -> bool:
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except ValueError:
        return False


def render_logo(version: str = __version__, *, color: bool = True) -> str:
    """Return the multi-line banner. Trailing newline included."""
    if color:
        token_open, speed_open, close = _TOKEN_STYLE, _SPEED_STYLE, _RESET
    else:
        token_open = speed_open = close = ""

    rows = [
        f"{token_open}{token}{close} {speed_open}{speed}{close}"
        for token, speed in zip(_TOKEN_LINES, _SPEED_LINES)
    ]
    footer = f"{_TAGLINE} · v{version}"
    pad = max(0, (_BANNER_WIDTH - len(footer)) // 2)
    rows.append(" " * pad + footer)
    return "\n".join(rows) + "\n"


def print_logo(version: str = __version__, *, stream: IO[str] | None = None) -> None:
    """Write the banner to ``stream`` (default stderr).

    No-op when ``TOKENSPEED_DISABLE_LOGO`` is truthy. ANSI colors are emitted
    only when the target stream is a TTY.
    """
    if _is_disabled():
        return
    if stream is None:
        stream = sys.stderr
    stream.write(render_logo(version, color=_stream_supports_color(stream)))
    stream.flush()
