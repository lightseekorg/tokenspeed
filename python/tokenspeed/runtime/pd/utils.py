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

import ctypes
import os
import random
import threading
from collections import deque
from enum import Enum

import torch
import torch.distributed as dist

from tokenspeed.runtime.utils import get_colorful_logger

# env var for testing failure, convert to float explicitly
FAILURE_PROB = float(os.getenv("DISAGGREGATION_TEST_FAILURE_PROB", 0))
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


def poll_and_all_reduce(pollers, gloo_group):
    """Poll transfer state and all-reduce the result across the gloo group."""
    # At a certain probability, mark the poll as failed to simulate failure.
    if FAILURE_PROB > 0:
        from tokenspeed.runtime.pd.base.status import TransferPoll

        polls = [
            (
                int(TransferPoll.Failed)
                if random.random() < FAILURE_PROB
                else int(poller.poll())
            )
            for poller in pollers
        ]
    else:
        polls = [int(poller.poll()) for poller in pollers]
    tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.MIN, group=gloo_group)
    return tensor_to_reduce.tolist()


class TransferBackend(Enum):
    MOONCAKE = "mooncake"


class KVClassType(Enum):
    MANAGER_PREFILL = "manager_prefill"
    MANAGER_DECODE = "manager_decode"
    SENDER = "sender"
    RECEIVER = "receiver"
    BOOTSTRAP_SERVER = "bootstrap_server"


def get_kv_class(transfer_backend: TransferBackend, class_type: KVClassType):
    if transfer_backend == TransferBackend.MOONCAKE:
        from tokenspeed.runtime.pd.mooncake.conn import (
            MooncakeKVBootstrapServer,
        )
        from tokenspeed.runtime.pd.mooncake.decode import (
            MooncakeKVManagerDecode,
        )
        from tokenspeed.runtime.pd.mooncake.prefill import (
            MooncakeKVManagerPrefill,
        )
        from tokenspeed.runtime.pd.mooncake.receiver import (
            MooncakeKVReceiver,
        )
        from tokenspeed.runtime.pd.mooncake.sender import (
            MooncakeKVSender,
        )

        class_mapping = {
            KVClassType.MANAGER_PREFILL: MooncakeKVManagerPrefill,
            KVClassType.MANAGER_DECODE: MooncakeKVManagerDecode,
            KVClassType.SENDER: MooncakeKVSender,
            KVClassType.RECEIVER: (MooncakeKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: MooncakeKVBootstrapServer,
        }
        return class_mapping.get(class_type)

    raise ValueError(f"Unsupported transfer backend: {transfer_backend}")


class StepCounter:
    COUNT_NUM_MAX: int = 2**62

    @classmethod
    def is_step_ready(cls, current_step: int, target_step: int) -> bool:
        # because COUNT_NUM_MAX is very large, we can make sure that if diff is > COUNT_NUM_MAX / 2 means the flush is finished
        # and if the current_sent_count == task_stop_count also means the flush is not finished
        # so if current_sent_count != task_stop_count and diff < COUNT_NUM_MAX / 2, the flush is not finished
        return (
            target_step != current_step
            and (target_step + cls.COUNT_NUM_MAX - current_step) % cls.COUNT_NUM_MAX
            > cls.COUNT_NUM_MAX / 2
        )

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

    def query_ready_cache_step(self) -> int:
        return ctypes.c_int64.from_address(self.h_ready_cache_step.data_ptr()).value
