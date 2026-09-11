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

"""Project the CUDA-graph pool reserve from a throwaway capture measured at startup."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.execution.memory_delta import DriverMemoryDeltaObserver
from tokenspeed.runtime.utils import get_colorful_logger

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.model_executor import ModelExecutor
    from tokenspeed.runtime.utils.server_args import ServerArgs

logger = get_colorful_logger(__name__)


PROBE_ENTRIES_PER_LADDER = 4


def probe_arena_parent_blocks(*, max_forward_tokens: int, context_len: int) -> int:
    """The parent-block floor a probe arena has to clear.

    Every fabricated extend the probe runs -- the autotune dummy prefill and
    each captured prefill bucket -- hands its rows distinct page ids that the
    block tables bound-check, so the floor is the widest of them. Both are
    bounded by the per-forward token budget: the tuning batch is clamped to it
    and so is the bucket ladder, and neither is bounded by the configured
    concurrency, which is why this does not read it. Decode capture adds
    nothing -- its placeholder tables are all null page. The recipe raises
    this further for a family whose page demand is concurrency-driven (the KDA
    raw-gate verify scratch, for one, is the bound pool's own conv slab).
    """
    from tokenspeed.runtime.execution.prefill_graph import dummy_batch_size

    return dummy_batch_size(max_forward_tokens, context_len)


@dataclass(frozen=True)
class CudagraphSeriesEstimate:
    """Projected bytes for one captured ladder."""

    first_capture: int
    extrapolated_rate: int
    total: int


@dataclass(frozen=True)
class CudagraphMemoryEstimate:
    """Projected bytes for every ladder the boot captures."""

    series: Mapping[str, CudagraphSeriesEstimate]
    total: int


def _estimate_series(
    series: str, samples: Sequence[int], entry_count: int
) -> CudagraphSeriesEstimate:
    """Project one ladder from the captures actually sampled out of it.

    The samples are driver-free-memory deltas, so they carry the granularity
    of a driver allocation: most per-entry marginals read zero even though the
    entries cost something, and a negative one means the region handed memory
    back rather than that the pool shrank. The rate is therefore the mean of
    the floored marginals -- a median of mostly-zero samples would extrapolate
    the whole unsampled tail as free, and under-reserving is the failure this
    projection exists to prevent. That same granularity is why the tail is
    bounded: one allocation landing inside a three-sample window would
    otherwise be multiplied by the entries the probe never captured.
    """
    if entry_count == 0:
        if samples:
            raise ValueError(f"{series} projection got samples for no entries")
        return CudagraphSeriesEstimate(0, 0, 0)

    required = min(2, entry_count)
    if not required <= len(samples) <= entry_count:
        raise ValueError(
            f"{series} projection got {len(samples)} samples for "
            f"{entry_count} entries, expected between {required} and {entry_count}"
        )

    first, *marginals = (max(sample, 0) for sample in samples)
    rate = -(-sum(marginals) // len(marginals)) if marginals else 0

    # One shared pool: the tail cannot cost more than what was measured.
    measured = first + sum(marginals)
    remaining = entry_count - len(samples)
    total = measured + min(rate * remaining, measured)

    return CudagraphSeriesEstimate(first, rate, total)


def estimate_cudagraph_memory(
    samples: Mapping[str, Sequence[int]], entry_counts: Mapping[str, int]
) -> CudagraphMemoryEstimate:
    """Project every captured ladder; their pools are disjoint and so add up."""
    unmeasured = sorted(set(samples) - set(entry_counts))
    if unmeasured:
        raise ValueError(f"projection got samples for unknown ladders: {unmeasured}")

    series = {
        name: _estimate_series(name, samples.get(name, ()), entry_count)
        for name, entry_count in entry_counts.items()
    }

    return CudagraphMemoryEstimate(
        series=series, total=sum(estimate.total for estimate in series.values())
    )


def _entry_counts(executor: ModelExecutor) -> dict[str, int]:
    """How many entries each captured ladder records at serving size."""
    return {
        **executor.prefill_graph.capture_entries,
        **executor.forward_step.capture_entries,
    }


def _hungriest_rank(server_args: ServerArgs, total: int) -> int:
    """Size every rank's KV cache for the hungriest rank's projection."""
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )

    if server_args.mapping.world_size == 1:
        return total

    reduced = torch.tensor(total, dtype=torch.float64)
    torch.distributed.all_reduce(
        reduced,
        op=torch.distributed.ReduceOp.MAX,
        group=pg_manager.get_process_group("gloo", server_args.mapping.world_group),
    )
    return int(reduced.item())


def reserve_and_rebind(
    executor: ModelExecutor,
    build_components: Callable[..., tuple],
    server_args: ServerArgs,
    gpu_id: int,
) -> tuple:
    """Measure a capture on the probe pool, then rebuild the real one under it.

    The one ordering the reserve depends on: measure, release (a live graph
    pool is memory ``empty_cache`` cannot return), profile and rebuild with
    the projection deducted, then publish. The backends are handed back to
    the rebuild rather than rebuilt, so the pool the probe measured with is
    the pool that serves, and the factory stays the only thing that binds.
    """
    graph_reserve_bytes = probe_cudagraph_memory(executor, server_args, gpu_id)
    executor.release_graphs()
    components = build_components(
        graph_reserve_bytes=graph_reserve_bytes,
        num_lcm_blocks_override=None,
        reuse_backends=(executor.attn_backend, executor.draft_attn_backend),
    )
    _, token_to_kv_pool, _, draft_token_to_kv_pool, _ = components
    executor.set_cache_pool(token_to_kv_pool, draft_token_to_kv_pool)
    return components


def probe_cudagraph_memory(
    executor: ModelExecutor, server_args: ServerArgs, gpu_id: int
) -> int:
    """Capture a few graphs per family, measure them, and project the pools.

    The executor is bound to a probe-sized arena here: it captures the largest
    few entries of each ladder, the observer records one driver-memory delta
    per capture, and the projection extrapolates the rest of the ladder. The
    caller releases these graphs when it rebinds the real pool.
    """
    device_module = torch.get_device_module(server_args.device)
    observer = DriverMemoryDeltaObserver(device_module, gpu_id)

    executor.capture_graphs(entries=PROBE_ENTRIES_PER_LADDER, observer=observer)

    entry_counts = _entry_counts(executor)
    estimate = estimate_cudagraph_memory(observer.samples, entry_counts)
    reserve = _hungriest_rank(server_args, estimate.total)
    per_series = ", ".join(
        f"{name} {estimate.series[name].first_capture} measured + "
        f"{estimate.series[name].total - estimate.series[name].first_capture} "
        f"projected over {count} entries"
        for name, count in sorted(entry_counts.items())
    )
    logger.info(
        "CUDA-graph memory reserve: %d bytes (this rank: %s)", reserve, per_series
    )
    if entry_counts and not estimate.total:
        # Every capture read as free: a contaminated probe, not a free capture.
        logger.warning(
            "CUDA-graph memory reserve projected 0 bytes over %d captured "
            "entries; the cache is sized as if the graphs were free",
            sum(entry_counts.values()),
        )
    return reserve
