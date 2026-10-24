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

"""Shared helpers for PD transfer runtime components."""

from __future__ import annotations

import threading
from collections import deque
from enum import Enum

import torch

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"
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


class TransferBackend(Enum):
    MOONCAKE = "mooncake"


class StepCounter:
    """The device half of layerwise PD streaming's producer-step counter.

    The attention backends tick it as each producer step's KV lands; the
    pinned host copy is what the host's transfer workers poll. The host
    counts the steps it armed itself (``ts-pd``'s ``PrefillOutbox``), modulo
    ``COUNT_NUM_MAX``.
    """

    COUNT_NUM_MAX: int = 2**62

    def __init__(self, device: str, gpu_id: int):
        self.d_ready_cache_step = torch.tensor(0, dtype=torch.int64).cuda(gpu_id)
        self.h_ready_cache_step = torch.tensor(0, dtype=torch.int64, pin_memory=True)

    def record_cache(self):
        self.d_ready_cache_step = (self.d_ready_cache_step + 1) % self.COUNT_NUM_MAX
        self.h_ready_cache_step.copy_(self.d_ready_cache_step, non_blocking=True)
