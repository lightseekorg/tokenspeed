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

"""Monokernel tests: cap the host threads of their reference math."""

from __future__ import annotations

import os

import pytest
import torch
from cpu_budget import CGROUP_ROOT, cgroup_cpu_quota, cpu_budget

# The Muse Spark references (attention over long contexts, the experts) run
# FP32 math on the host. Torch and OpenMP start a thread per visible core,
# which on a CI pod with a small CPU quota, or a busy host, oversubscribes the
# CPUs: test_muse_spark_attention_gfx950.py took ~53 min on CI against ~31 s
# on 8 pinned cores. 8 threads keep the references fast and leave room.
MAX_HOST_THREADS = 8


@pytest.fixture(autouse=True)
def _cap_host_threads(monkeypatch):
    """Run each test with torch's intra-op threads (and its child processes'
    OMP_NUM_THREADS, unless set) at most the process's CPU budget (cores,
    cgroup quota) and MAX_HOST_THREADS; restore torch's setting after."""
    budget = cpu_budget(os.sched_getaffinity(0), cgroup_cpu_quota(CGROUP_ROOT))
    threads = min(budget, MAX_HOST_THREADS)
    before = torch.get_num_threads()
    torch.set_num_threads(min(before, threads))
    if "OMP_NUM_THREADS" not in os.environ:
        monkeypatch.setenv("OMP_NUM_THREADS", str(threads))
    yield
    torch.set_num_threads(before)
