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

"""Startup agreement on memory limits, before model capacity selection."""

from __future__ import annotations


def local_cache_capacity(local_limit: int, error: str | None) -> int:
    """Validate one standalone capacity calculation without communication."""
    if error is not None:
        raise ValueError(error)
    if local_limit < 1:
        raise ValueError("cache budget must hold a null parent and one usable parent")
    return local_limit


def agree_cache_capacity(
    local_limit: int, error: str | None, *, cpu_group, probe: bool
) -> int:
    """Agree on parent capacity, or fail on every participating rank.

    Serving pools take the smallest memory limit. Probe pools take the largest
    required working set; they must not be grown to fill the memory budget.
    ``cpu_group=None`` explicitly selects a standalone calculation.
    """
    if cpu_group is None:
        return local_cache_capacity(local_limit, error)
    import torch
    import torch.distributed as dist

    failures = [None] * dist.get_world_size(cpu_group)
    dist.all_gather_object(failures, error, group=cpu_group)
    if any(failure is not None for failure in failures):
        details = "; ".join(
            f"rank {rank}: {failure}"
            for rank, failure in enumerate(failures)
            if failure is not None
        )
        raise ValueError(f"cache capacity planning failed: {details}")
    count = torch.tensor(local_limit, dtype=torch.int64)
    dist.all_reduce(
        count, op=dist.ReduceOp.MAX if probe else dist.ReduceOp.MIN, group=cpu_group
    )
    return local_cache_capacity(int(count.item()), None)
