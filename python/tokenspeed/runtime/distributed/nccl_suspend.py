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

"""Suspend and resume this process's NCCL communicators (the ``nccl`` memory tag).

``ProcessGroupNCCL.suspend`` asks NCCL to free a communicator's device buffers
(``ncclCommSuspend``) while keeping the communicator; ``resume`` restores them,
so collectives -- including the ones recorded in CUDA graphs -- work again
afterwards. Every process group of the world is visited, because a collective
may run on any of them. Groups whose communicator was never created have no
buffers to free and are skipped.

Only NCCL's own memory is covered: symmetric-memory and multicast workspaces
of the custom collectives stay resident.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

# torch's ProcessGroupNCCL.suspend/resume refuse an older NCCL.
_MIN_NCCL_VERSION = (2, 29, 7)


def nccl_suspend_supported() -> bool:
    """Whether this process's NCCL can suspend and resume communicators."""
    return (
        dist.is_available()
        and dist.is_nccl_available()
        and torch.cuda.is_available()
        and callable(getattr(dist.ProcessGroupNCCL, "suspend", None))
        and callable(getattr(dist.ProcessGroupNCCL, "resume", None))
        and tuple(torch.cuda.nccl.version()) >= _MIN_NCCL_VERSION
    )


def _nccl_backends() -> list:
    """The initialized NCCL backends of every process group, once each."""
    if not (
        dist.is_available()
        and dist.is_initialized()
        and dist.is_nccl_available()
        and torch.cuda.is_available()
    ):
        return []
    device = torch.device("cuda", torch.cuda.current_device())
    backends = []
    seen: set[int] = set()
    for group in list(distributed_c10d._world.pg_map):
        try:
            backend = group._get_backend(device)
        except RuntimeError:
            # A CPU-only (gloo) group has no backend for the device.
            continue
        if not isinstance(backend, dist.ProcessGroupNCCL):
            continue
        if not backend._is_initialized() or id(backend) in seen:
            continue
        seen.add(id(backend))
        backends.append(backend)
    return backends


def suspend_nccl_communicators() -> int:
    """Suspend every initialized NCCL communicator; return how many."""
    backends = _nccl_backends()
    for backend in backends:
        backend.suspend()
    logger.info(f"Suspended {len(backends)} NCCL process group(s)")
    return len(backends)


def resume_nccl_communicators() -> int:
    """Resume every initialized NCCL communicator; return how many."""
    backends = _nccl_backends()
    for backend in backends:
        backend.resume()
    logger.info(f"Resumed {len(backends)} NCCL process group(s)")
    return len(backends)
