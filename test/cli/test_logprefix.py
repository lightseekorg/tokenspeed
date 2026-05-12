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

"""Tests for the line prefixer."""

from __future__ import annotations

import asyncio
import io

import pytest

from tokenspeed.cli._logprefix import tag_stream


def _make_reader(payload: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(payload)
    reader.feed_eof()
    return reader


@pytest.mark.asyncio
async def test_single_line_is_prefixed():
    reader = _make_reader(b"hello world\n")
    sink = io.StringIO()
    await tag_stream(reader, "ts", sink)
    assert sink.getvalue() == "[ts] hello world\n"


@pytest.mark.asyncio
async def test_multi_line_each_prefixed_independently():
    reader = _make_reader(b"first line\nsecond line\nthird line\n")
    sink = io.StringIO()
    await tag_stream(reader, "smg", sink)
    assert sink.getvalue() == (
        "[smg] first line\n" "[smg] second line\n" "[smg] third line\n"
    )


@pytest.mark.asyncio
async def test_traceback_lines_prefixed_per_line():
    """Per-line prefixing for tracebacks (each frame ends with \\n)."""
    payload = (
        b"Traceback (most recent call last):\n"
        b'  File "x.py", line 1, in <module>\n'
        b"    raise ValueError(42)\n"
        b"ValueError: 42\n"
    )
    reader = _make_reader(payload)
    sink = io.StringIO()
    await tag_stream(reader, "ts", sink)
    assert sink.getvalue() == (
        "[ts] Traceback (most recent call last):\n"
        '[ts]   File "x.py", line 1, in <module>\n'
        "[ts]     raise ValueError(42)\n"
        "[ts] ValueError: 42\n"
    )


@pytest.mark.asyncio
async def test_partial_last_line_without_newline_still_emitted():
    """Crash messages without a trailing newline must still reach the sink."""
    reader = _make_reader(b"died mid-write")
    sink = io.StringIO()
    await tag_stream(reader, "ts", sink)
    assert sink.getvalue() == "[ts] died mid-write\n"


@pytest.mark.asyncio
async def test_eof_with_no_data_emits_nothing():
    reader = _make_reader(b"")
    sink = io.StringIO()
    await tag_stream(reader, "ts", sink)
    assert sink.getvalue() == ""
