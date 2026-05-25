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

from __future__ import annotations

import logging
import threading

_LOG = logging.getLogger("tokenspeed.eplb")
_rank = 0


class _RateLimiter:
    def __init__(self):
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def should_emit(self, key: str, every: int) -> bool:
        if every <= 1:
            return True
        with self._lock:
            count = self._counts.get(key, 0) + 1
            self._counts[key] = count
            return count % every == 0


_rl = _RateLimiter()


def set_rank(rank: int) -> None:
    global _rank
    _rank = int(rank)


def _prefix(fmt: str) -> str:
    return f"[EPLB rank={_rank}] {fmt}"


def debug(fmt, *args):
    _LOG.debug(_prefix(fmt), *args)


def info(fmt, *args):
    _LOG.info(_prefix(fmt), *args)


def warning(fmt, *args):
    _LOG.warning(_prefix(fmt), *args)


def error(fmt, *args):
    _LOG.error(_prefix(fmt), *args)


def debug_rate(fmt, *args, every: int = 200):
    if _rl.should_emit(f"debug:{fmt}", every):
        _LOG.debug(_prefix(fmt + " (every %d)"), *args, every)


def info_rate(fmt, *args, every: int = 20):
    if _rl.should_emit(f"info:{fmt}", every):
        _LOG.info(_prefix(fmt + " (every %d)"), *args, every)
