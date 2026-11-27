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
"""Eight rank-local device forwards for the opt-in peer protocol regression."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import torch
import torch_npu

torch.set_num_threads(1)
assert torch.npu.device_count() == 8
pool = ThreadPoolExecutor(8)
started = Barrier(9, timeout=120)
release = Event()


def prepare(rank):
    torch.npu.set_device(rank)
    x = torch.ones((2048, 4096), device=f"npu:{rank}", dtype=torch.float16)
    w = torch.eye(4096, device=x.device, dtype=x.dtype)
    y = torch.empty_like(x)
    torch.mm(x, w, out=y)
    torch.npu.synchronize()
    return x, w, y


tensors = list(pool.map(prepare, range(8)))


def forward(rank, iterations):
    torch.npu.set_device(rank)
    x, w, y = tensors[rank]
    first = torch.npu.Event(enable_timing=True)
    last = torch.npu.Event(enable_timing=True)
    started.wait()
    assert release.wait(60)
    first.record()
    torch.mm(x, w, out=y)
    torch.npu.synchronize()
    started.wait()
    for _ in range(iterations):
        torch.mm(x, w, out=y)
        x, y = y, x
    last.record()
    last.synchronize()
    return first.elapsed_time(last)


def arm(iterations=512):
    global futures
    release.clear()
    futures = [pool.submit(forward, rank, iterations) for rank in range(8)]
    started.wait()


def begin():
    release.set()
    started.wait()


def finish():
    return [future.result(timeout=120) for future in futures]
