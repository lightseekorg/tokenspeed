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

"""Disjoint cache-page planning for startup decode profiling."""

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter

import torch

from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    CacheGroupSpec,
    compute_max_logical_pages_for_capture,
)


def profile_decode_context(
    context_length: int, *, max_positions: int, verify_width: int, draft_width: int
) -> int:
    """Leave room for the draft following a fully accepted target block."""
    if min(context_length, max_positions, verify_width, draft_width) < 1:
        raise ValueError("profiling dimensions must be positive")
    context = min(context_length, max_positions - draft_width)
    if context < verify_width:
        raise ValueError("position capacity cannot fit target verification and draft")
    return context


@dataclass(frozen=True)
class ProfileCacheGroup:
    group_id: str
    first_page: int
    pages_per_request: int
    batch_size: int

    @property
    def page_stop(self) -> int:
        return self.first_page + self.pages_per_request * self.batch_size

    def block_tables(self) -> tuple[tuple[int, ...], ...]:
        rows = []
        for row in range(self.batch_size):
            first = self.first_page + row * self.pages_per_request
            rows.append(tuple(range(first, first + self.pages_per_request)))
        return tuple(rows)


def plan_profile_cache(
    specs: Sequence[CacheGroupSpec],
    *,
    virtual_packing: Mapping[str, int],
    page_counts: Mapping[str, int],
    batch_size: int,
    context_length: int,
    verify_width: int,
    overlap_schedule_depth: int,
) -> tuple[ProfileCacheGroup, ...] | None:
    """Reserve disjoint parents without mutating an arena or scheduler.

    Call only before serving starts. None means the synthetic working set
    cannot fit, or its sharding is unsupported; callers must not substitute
    aliased placeholder pages and publish that timing as a complete profile.
    """
    if batch_size < 1 or context_length < 1 or verify_width < 1:
        raise ValueError("profiling dimensions must be positive")
    if len({spec.group_id for spec in specs}) != len(specs):
        raise ValueError("duplicate profiling cache group")
    if not specs or any(spec.shard_count != 1 for spec in specs):
        return None
    parent_offset = 0
    groups = []
    for spec in specs:
        packing = virtual_packing[spec.group_id]
        if packing < 1 or page_counts[spec.group_id] < 1:
            raise ValueError("invalid profiling cache capacity")
        width = compute_max_logical_pages_for_capture(
            spec,
            max_context_len=context_length,
            max_tokens_per_req=verify_width,
            overlap_schedule_depth=overlap_schedule_depth,
        )
        pages = width
        group = ProfileCacheGroup(
            spec.group_id,
            1 + parent_offset * packing,
            pages,
            batch_size,
        )
        if group.page_stop > page_counts[spec.group_id]:
            return None
        groups.append(group)
        parent_offset += (batch_size * pages + packing - 1) // packing
    return tuple(groups)


@contextmanager
def preserve_profile_tensors(tensors: Sequence[torch.Tensor]):
    """Restore startup scratch contents in place, including on profiling failure.

    The caller orders profiling and restoration on one stream and joins any
    auxiliary work before exiting. Tensor addresses must not change because
    the already captured graphs retain them.
    """
    snapshots = [tensor.clone() for tensor in tensors]
    try:
        yield
    finally:
        for tensor, snapshot in zip(tensors, snapshots, strict=True):
            tensor.copy_(snapshot)


def measure_startup_graph(
    replay: Callable[[], None],
    prepare: Callable[[], None],
    *,
    device: torch.device,
    host_group: torch.distributed.ProcessGroup | None,
    warmup: int,
    samples: int,
) -> tuple[float, ...]:
    """Measure complete replay wall time before serving, aligned across ranks.

    Preparation and rank barriers are outside the interval. Replay includes
    host policy, agreement and GPU submissions; synchronization includes their
    GPU completion. This is a step cost, not an end-to-end request measurement.
    """
    if warmup < 0 or samples < 1:
        raise ValueError("invalid startup profiling sample counts")
    if device.type != "cuda":
        raise ValueError("startup graph profiling requires CUDA")
    measurements = []
    with torch.cuda.device(device):
        for index in range(warmup + samples):
            prepare()
            torch.cuda.synchronize(device)
            if host_group is not None:
                torch.distributed.barrier(group=host_group)
            start = perf_counter()
            replay()
            torch.cuda.synchronize(device)
            elapsed_ms = (perf_counter() - start) * 1000
            if index >= warmup:
                measurements.append(elapsed_ms)
        elapsed = torch.tensor(
            measurements,
            dtype=torch.float64,
            device="cpu",
        )
        valid = torch.tensor(
            int(bool(torch.isfinite(elapsed).all()) and bool((elapsed > 0).all())),
            dtype=torch.int32,
        )
        if host_group is not None:
            torch.distributed.all_reduce(
                valid, op=torch.distributed.ReduceOp.MIN, group=host_group
            )
            torch.distributed.all_reduce(
                elapsed, op=torch.distributed.ReduceOp.MAX, group=host_group
            )
        if not valid.item():
            raise ValueError("non-finite or non-positive startup graph timing")
        return tuple(elapsed.tolist())


def agree_startup_profile(
    *,
    available: bool,
    batch_size: int,
    context_length: int,
    capacities: tuple[int, ...],
    host_group: torch.distributed.ProcessGroup | None,
) -> bool:
    """All ranks either profile the same graph ladder or skip it together."""
    signature = (batch_size, context_length, capacities) if available else None
    if host_group is None:
        return signature is not None
    peers = [None] * torch.distributed.get_world_size(host_group)
    torch.distributed.all_gather_object(peers, signature, group=host_group)
    return signature is not None and all(peer == signature for peer in peers)
