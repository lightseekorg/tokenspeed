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


def bootstrap_server_class(transfer_backend: TransferBackend):
    """The KV bootstrap server a prefill deployment runs for ``transfer_backend``:
    decode ranks look up the prefill ranks holding a request's KV through it."""
    if TransferBackend(transfer_backend) == TransferBackend.MOONCAKE:
        from tokenspeed.runtime.pd.mooncake.conn import MooncakeKVBootstrapServer

        return MooncakeKVBootstrapServer
    raise ValueError(f"Unsupported transfer backend: {transfer_backend}")


class StepCounter:
    COUNT_NUM_MAX: int = 2**62

    def __init__(self, device: str, gpu_id: int):
        # utilities for cache step
        self.d_ready_cache_step = torch.tensor(0, dtype=torch.int64).cuda(gpu_id)
        self.h_ready_cache_step = torch.tensor(0, dtype=torch.int64, pin_memory=True)
        self.cache_step: int = 0

        # utilities for aux step
        self.d_ready_aux_step = torch.tensor(0, dtype=torch.int64).cuda(gpu_id)
        self.h_ready_aux_step = torch.tensor(0, dtype=torch.int64, pin_memory=True)
        self.aux_step: int = 0

    def current_step(self) -> tuple[int, int]:
        return self.cache_step, self.aux_step

    def advance_step(self, delta_cache_step: int, delta_aux_step: int):
        self.cache_step = (self.cache_step + delta_cache_step) % self.COUNT_NUM_MAX
        self.aux_step = (self.aux_step + delta_aux_step) % self.COUNT_NUM_MAX

    def record_cache(self):
        self.d_ready_cache_step = (self.d_ready_cache_step + 1) % self.COUNT_NUM_MAX
        self.h_ready_cache_step.copy_(self.d_ready_cache_step, non_blocking=True)
