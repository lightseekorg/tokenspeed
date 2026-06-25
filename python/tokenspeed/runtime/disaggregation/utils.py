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

"""Disaggregation-wide neutral utilities shared by all roles (kv + embedding).

These carry no KV/embedding-specific or backend-specific state, so they live at
the disaggregation root rather than inside any role package: the role/mode enum
and a small blocking queue used by the transfer workers.
"""

import threading
from collections import deque
from enum import Enum


class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"
    # Encode-only server for EPD: runs the vision tower and ships image
    # embeddings to a prefill server. Orthogonal to the prefill/decode KV
    # split; an encode server owns no KV pool and runs no LM forward.
    ENCODE = "encode"


class FastQueue:
    class Empty(Exception):
        """Exception raised when the queue is empty."""

        pass

    def __init__(self):
        self._buf = deque()
        self._cond = threading.Condition()

    def put(self, item):
        with self._cond:
            self._buf.append(item)
            self._cond.notify()

    def get(self):
        with self._cond:
            while not self._buf:
                self._cond.wait()
            return self._buf.popleft()

    def get_nowait(self):
        with self._cond:
            if not self._buf:
                raise FastQueue.Empty()
            return self._buf.popleft()
