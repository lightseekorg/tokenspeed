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

"""The CPU budget of a test process: the cores it may run on and its cgroup's
CPU quota.

Torch and OpenMP start one thread per visible core. On a CI host with
hundreds of cores, or in a container whose quota is a few cores, that many
threads oversubscribe the CPUs and the host-side reference math of a kernel
test slows down by orders of magnitude.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

CGROUP_ROOT = Path("/sys/fs/cgroup")


def cgroup_cpu_quota(root: Path) -> int | None:
    """Whole CPUs of the cgroup quota at ``root`` (rounded up), or None
    without one: cgroup v2 ``cpu.max`` ("<quota> <period>", quota "max"
    for none), else cgroup v1 ``cpu/cpu.cfs_quota_us`` and
    ``cpu.cfs_period_us`` (quota -1 for none)."""
    v2 = root / "cpu.max"
    if v2.is_file():
        quota, period = v2.read_text().split()
        if quota == "max":
            return None
        return max(1, math.ceil(int(quota) / int(period)))
    v1_quota = root / "cpu" / "cpu.cfs_quota_us"
    v1_period = root / "cpu" / "cpu.cfs_period_us"
    if v1_quota.is_file() and v1_period.is_file():
        quota = int(v1_quota.read_text())
        if quota <= 0:
            return None
        return max(1, math.ceil(quota / int(v1_period.read_text())))
    return None


def cpu_budget(affinity: set[int], quota: int | None) -> int:
    """Threads a process should run: its allowed cores, capped by its quota."""
    cores = len(affinity)
    return max(1, min(cores, quota) if quota is not None else cores)


def apply_cpu_budget(torch_module, environ, root: Path = CGROUP_ROOT) -> int:
    """Cap torch's intra-op threads at the budget, and give child processes
    the same cap through ``OMP_NUM_THREADS`` unless the caller set it.

    Returns:
        The budget.
    """
    budget = cpu_budget(os.sched_getaffinity(0), cgroup_cpu_quota(root))
    if torch_module.get_num_threads() > budget:
        torch_module.set_num_threads(budget)
    environ.setdefault("OMP_NUM_THREADS", str(budget))
    return budget
