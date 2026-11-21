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

"""CPU checks for existing split communicators before their first collective."""

from test.ci_system.ci_register import register_cuda_ci
from types import SimpleNamespace

import torch

from tokenspeed.runtime.distributed import nccl_suspend

register_cuda_ci(est_time=5, suite="runtime-1gpu")


def test_sleep_includes_cached_split_communicators_without_creating_unused_groups(
    monkeypatch,
):
    class Backend:
        def __init__(self, initialized, cached):
            self.initialized = initialized
            self.cached = cached
            self.suspended = False
            self.suspensions = 0

        def _is_initialized(self):
            return self.initialized

        def _comm_ptr(self):
            return self.cached

        def suspend(self):
            self.suspended = True
            self.suspensions += 1

        def resume(self):
            self.suspended = False

    live, split, unused = Backend(True, 8), Backend(False, 4), Backend(False, 0)
    groups = [
        SimpleNamespace(_get_backend=lambda device, backend=backend: backend)
        for backend in (live, split, split, unused)
    ]
    monkeypatch.setattr(
        nccl_suspend.distributed_c10d, "_world", SimpleNamespace(pg_map=groups)
    )
    monkeypatch.setattr(nccl_suspend.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(nccl_suspend.dist, "is_nccl_available", lambda: True)
    monkeypatch.setattr(nccl_suspend.dist, "ProcessGroupNCCL", Backend, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    assert nccl_suspend.suspend_nccl_communicators() == 2
    assert live.suspended and split.suspended
    assert live.suspensions == split.suspensions == 1
    assert unused.suspensions == 0
    assert nccl_suspend.resume_nccl_communicators() == 2
    assert not live.suspended and not split.suspended
