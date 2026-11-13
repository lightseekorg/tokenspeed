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
afterwards. Protocol 2 visits the initialized groups in their existing order.
Protocol 3 checks the serving groups and their members before suspending them
in a shared order. Uninitialized groups have no buffers to free.

Only NCCL's own memory is covered: symmetric-memory and multicast workspaces
of the custom collectives stay resident.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d

from tokenspeed.runtime.distributed.process_group_manager import process_group_manager
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


def _nccl_groups() -> list[tuple[dist.ProcessGroup, dist.ProcessGroupNCCL]]:
    """The initialized NCCL groups and their backends, once per backend."""
    if not (
        dist.is_available()
        and dist.is_initialized()
        and dist.is_nccl_available()
        and torch.cuda.is_available()
    ):
        return []
    device = torch.device("cuda", torch.cuda.current_device())
    groups = []
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
        groups.append((group, backend))
    return groups


@dataclass(frozen=True)
class NcclMemoryGroup:
    """One local communicator and its serving-world identity."""

    name: str
    ranks: tuple[int, ...]
    backend: dist.ProcessGroupNCCL


def prepare_nccl_suspend(
    memory_ranks: tuple[int, ...], *, p2p_ranks: tuple[tuple[int, ...], ...]
) -> tuple[NcclMemoryGroup, ...]:
    """Describe the local communicator sequence without entering NCCL."""
    allowed = set(memory_ranks)
    try:
        p2p_groups = {
            process_group_manager.get_device_process_group(ranks) for ranks in p2p_ranks
        }
    except KeyError as error:
        raise RuntimeError("cannot verify the NCCL point-to-point groups") from error
    plan = []
    for group, backend in _nccl_groups():
        name = group.group_name
        # A trainer group can have the same numerical rank IDs in a different
        # rendezvous. Only the default world and managed serving groups share
        # the scheduler's rank namespace.
        if (
            group is not dist.group.WORLD
            and not process_group_manager.owns_device_process_group(group)
        ):
            raise RuntimeError(f"NCCL group {name!r} is outside the serving world")
        ranks = tuple(sorted(dist.get_process_group_ranks(group)))
        if not allowed.issuperset(ranks):
            raise RuntimeError(
                f"NCCL group {name!r} includes ranks outside the memory coordination domain"
            )
        if group in p2p_groups and len(ranks) > 1:
            # c10d can cache per-peer NCCL communicators behind one backend.
            # Its suspend binding does not let us order those communicators.
            raise RuntimeError(
                f"NCCL group {name!r} may use point-to-point communicators with unknown suspend order"
            )
        plan.append(NcclMemoryGroup(name, ranks, backend))
    # c10d gives a serving group the same unique name on all its members.
    # A common order prevents cycles between overlapping communicators.
    return tuple(sorted(plan, key=lambda group: group.name))


def check_nccl_resume(plan: tuple[NcclMemoryGroup, ...]) -> None:
    """Verify the suspended communicators still exist before resuming any."""
    initialized = {id(backend) for _, backend in _nccl_groups()}
    if any(id(group.backend) not in initialized for group in plan):
        raise RuntimeError("a suspended NCCL group is no longer initialized")


def apply_nccl_memory_plan(plan: tuple[NcclMemoryGroup, ...], *, release: bool) -> None:
    """Use the same ordered communicators the serving ranks agreed to."""
    for group in plan:
        (group.backend.suspend if release else group.backend.resume)()
    operation = "Suspended" if release else "Resumed"
    logger.info(f"{operation} {len(plan)} NCCL process group(s)")


def suspend_nccl_communicators() -> int:
    """Suspend every initialized NCCL communicator; return how many."""
    groups = _nccl_groups()
    for _, backend in groups:
        backend.suspend()
    logger.info(f"Suspended {len(groups)} NCCL process group(s)")
    return len(groups)


def resume_nccl_communicators() -> int:
    """Resume every initialized NCCL communicator; return how many."""
    groups = _nccl_groups()
    for _, backend in groups:
        backend.resume()
    logger.info(f"Resumed {len(groups)} NCCL process group(s)")
    return len(groups)
