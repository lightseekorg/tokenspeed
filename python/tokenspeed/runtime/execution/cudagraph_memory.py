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
    from tokenspeed.runtime.layers.attention.registry import AttentionBuild
    from tokenspeed.runtime.utils.server_args import ServerArgs

logger = get_colorful_logger(__name__)


# Ladder positions, not graphs; measured, both narrower and wider over-reserve more.
PROBE_ENTRIES_PER_LADDER = 5


def probe_arena_parent_blocks(
    *,
    max_forward_tokens: int,
    context_len: int,
    capture_batch_sizes: Sequence[int] | None,
) -> int:
    """The parent-block floor a probe arena has to clear.

    Each fabricated extend row -- the autotune dummy prefill and every captured
    prefill bucket, including configured capture batch sizes -- takes a
    distinct page; decode capture uses only the null page.
    """
    from tokenspeed.runtime.execution.prefill_graph import dummy_batch_size

    return max(
        dummy_batch_size(max_forward_tokens, context_len),
        max(capture_batch_sizes or (0,)),
    )


@dataclass(frozen=True)
class CudagraphSeriesEstimate:
    """Projected bytes for one captured ladder."""

    measured: int
    extrapolated_rate: int
    unsampled: int


@dataclass(frozen=True)
class CudagraphMemoryEstimate:
    """Bytes the unsampled entries of every ladder are projected to add."""

    series: Mapping[str, CudagraphSeriesEstimate]
    measured_total: int
    unsampled_total: int


def _estimate_series(
    series: str, samples: Sequence[int], entry_count: int
) -> CudagraphSeriesEstimate:
    """Price the entries of one ladder that the probe did not capture.

    An entry's cost tracks its graph's kernel-node count, not its tensor
    sizes, so the tail is priced at the window's mean marginal: the positive
    marginals summed over every marginal, since driver segments make single
    readings lumpy and a region that handed memory back is not a credit.
    Sampling each ladder from its widest entry keeps the projection on the
    safe side (docs/design/unified_path.md).
    """
    if entry_count == 0:
        if samples:
            raise ValueError(f"{series} projection got samples for no entries")
        return CudagraphSeriesEstimate(0, 0, 0)

    required = min(2, entry_count)
    if not required <= len(samples) <= entry_count:
        raise ValueError(
            f"{series} projection got {len(samples)} samples for "
            f"{entry_count} entries, expected between {required} and "
            f"{entry_count}; re-run with --disable-cudagraph-memory-reserve "
            "to size the cache without a probe"
        )

    first, *marginals = samples
    observed = [marginal for marginal in marginals if marginal > 0]
    rate = -(-sum(observed) // len(marginals)) if marginals else 0

    return CudagraphSeriesEstimate(
        max(first, 0) + sum(observed), rate, rate * (entry_count - len(samples))
    )


def estimate_cudagraph_memory(
    samples: Mapping[str, Sequence[int]],
    entry_counts: Mapping[str, int],
) -> CudagraphMemoryEstimate:
    """Project every captured ladder; their pools are disjoint and so add up."""
    unmeasured = sorted(set(samples) - set(entry_counts))
    if unmeasured:
        raise ValueError(
            f"projection got samples for unknown ladders: {unmeasured}; re-run "
            "with --disable-cudagraph-memory-reserve to size the cache without "
            "a probe"
        )

    series = {
        name: _estimate_series(name, samples.get(name, ()), count)
        for name, count in entry_counts.items()
    }

    return CudagraphMemoryEstimate(
        series=series,
        measured_total=sum(estimate.measured for estimate in series.values()),
        unsampled_total=sum(estimate.unsampled for estimate in series.values()),
    )


def _entry_counts(executor: ModelExecutor) -> dict[str, int]:
    """How many entries each captured ladder records at serving size."""
    drafter_entries = 1 if executor.captures_drafter_prefill_graph else 0
    return {
        **executor.prefill_graph.capture_entries,
        **executor.forward_step.capture_entries,
        # Its own private pool, captured after both ladders and released with them.
        **({"prefill:drafter": drafter_entries} if drafter_entries else {}),
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
    build_components: Callable[..., AttentionBuild],
    server_args: ServerArgs,
    gpu_id: int,
    *,
    profiled_cache_bytes: int,
) -> AttentionBuild:
    """Measure a capture on the probe pool, then rebuild the real one under it.

    Order: measure, release (``empty_cache`` cannot return a live graph pool),
    rebuild on the probe build's memory profile less the reserve, publish. The
    backends are handed back rather than rebuilt, so what serves is what was
    measured.
    """
    graph_reserve_bytes = probe_cudagraph_memory(executor, server_args, gpu_id)
    executor.release_graphs()
    attention = build_components(
        graph_reserve_bytes=graph_reserve_bytes,
        probe_batch_rows=None,
        profiled_cache_bytes=profiled_cache_bytes,
        reuse_target_backend=executor.attn_backend,
        reuse_draft_backend=executor.draft_attn_backend,
    )
    executor.set_cache_pool(
        attention.token_to_kv_pool, attention.draft_token_to_kv_pool
    )
    return attention


def probe_cudagraph_memory(
    executor: ModelExecutor, server_args: ServerArgs, gpu_id: int
) -> int:
    """Capture the widest few entries of each ladder, measure, and project.

    The reserve is what the whole probe spent plus the projected cost of the
    entries it skipped: a capture also takes one-time bytes outside every
    per-entry window (warmups, per-shape metadata) that the probe has paid.
    """
    device_module = torch.get_device_module(server_args.device)
    observer = DriverMemoryDeltaObserver(device_module, gpu_id)
    whole = DriverMemoryDeltaObserver(device_module, gpu_id)

    with whole.measure("probe"):
        executor.capture_graphs(entries=PROBE_ENTRIES_PER_LADDER, observer=observer)

    entry_counts = _entry_counts(executor)
    estimate = estimate_cudagraph_memory(observer.samples, entry_counts)
    # Nested but not telescoping: memory freed between windows outlives the bracket.
    spent = max(whole.samples["probe"][0], estimate.measured_total)
    reserve = _hungriest_rank(server_args, spent + estimate.unsampled_total)
    per_series = ", ".join(
        f"{name} {estimate.series[name].measured} measured + "
        f"{estimate.series[name].unsampled} projected over {count} entries"
        for name, count in sorted(entry_counts.items())
    )
    logger.info(
        f"CUDA-graph memory reserve: {reserve} bytes (this rank: probe spent "
        f"{spent}, unsampled entries {estimate.unsampled_total}; {per_series})"
    )
    # Per series: one ladder reading free is invisible in a non-zero total.
    for name, count in sorted(entry_counts.items()):
        series = estimate.series[name]
        samples = observer.samples.get(name, ())
        unsampled = count - len(samples)
        positives = sum(1 for marginal in samples[1:] if marginal > 0)
        # Two, not one: a rate from a single reading prices every skipped entry.
        if unsampled and positives < 2:
            logger.warning(
                f"CUDA-graph memory reserve: {name} measured "
                f"{series.measured} bytes and priced its {unsampled} unsampled "
                f"entries from {positives} non-zero marginals -- the rest were "
                "served from allocator slack; re-run with "
                "--disable-cudagraph-memory-reserve if the boot then OOMs"
            )
    return reserve
