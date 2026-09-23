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

import math
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


# Ladder positions, not graphs: the widest three and two down the ladder.
PROBE_ENTRIES_PER_LADDER = 5
# Readings move in 2 MiB: driver graph memory and allocator segments both round to it.
READING_GRANULE_BYTES = 2 << 20


def probe_positions(count: int, entries: int | None) -> list[int]:
    """Which positions of a ``count``-long ladder a probe of ``entries`` captures.

    The widest few, then one a third and one two thirds of the way down: a
    graph's cost can fall with its entry's width, so the tail is priced
    between samples rather than at the widest ones. ``None`` captures all.
    """
    if entries is None or count <= entries:
        return list(range(count))
    return sorted({*range(entries - 2), count // 3, 2 * count // 3})


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
class CapturedLadder:
    """One ladder as a full capture records it, and the positions a probe sampled."""

    widths: Sequence[int]
    sampled: Sequence[int]


@dataclass(frozen=True)
class CudagraphSeriesEstimate:
    """Bytes one captured ladder took in the probe, and those its skipped entries add."""

    measured: int
    unsampled: int


@dataclass(frozen=True)
class CudagraphMemoryEstimate:
    """Bytes every ladder took in the probe and its skipped entries are projected to add."""

    series: Mapping[str, CudagraphSeriesEstimate]
    measured_total: int
    unsampled_total: int


def _estimate_series(
    series: str, samples: Sequence[int], ladder: CapturedLadder
) -> CudagraphSeriesEstimate:
    """Price the entries of one ladder that the probe did not capture.

    The widest samples form a window priced at its mean marginal: the
    positive marginals summed over every marginal, since driver segments make
    single readings lumpy and a region that handed memory back is not a
    credit, plus one granule for the slack the window started in: the
    readings move in whole granules, whether the driver's graph memory or
    the caching allocator's segments moved them, so a window can start
    inside one. Each sample after the window anchors its own width at its
    reading plus that granule -- capped at the window's rate when the reading
    is within three granules of it, since one reading is lumpy and a flat
    ladder's anchors would otherwise price a lump across every entry below
    them, and at the reading less those three granules further above, so a
    dearer entry stays dearer and a granule more in any reading never lowers
    the reserve. A skipped entry is priced on the line between the anchors
    around its width; narrower than every anchor, at the narrowest one. A
    cost that drops between two anchors is priced short there, and the
    utilization headroom absorbs the difference either way
    (docs/design/unified_path.md).
    """
    widths, sampled = ladder.widths, list(ladder.sampled)
    if sampled != sorted(set(sampled)) or (
        sampled and not 0 <= sampled[-1] < len(widths)
    ):
        raise ValueError(f"{series} probe positions {sampled} are not ladder positions")
    if not widths:
        if samples:
            raise ValueError(f"{series} projection got samples for no entries")
        return CudagraphSeriesEstimate(0, 0)

    required = min(2, len(widths))
    if len(samples) != len(sampled) or not required <= len(samples) or sampled[0] != 0:
        raise ValueError(
            f"{series} projection got {len(samples)} samples at positions "
            f"{sampled} of {len(widths)} entries, expected the first and at "
            f"least {required} in all; re-run with "
            "--disable-cudagraph-memory-reserve to size the cache without a probe"
        )

    first, *marginals = samples
    observed = [marginal for marginal in marginals if marginal > 0]
    # A ladder read entirely from slack is priced at nothing; the probe logs it.
    granule = READING_GRANULE_BYTES if observed else 0
    # The window: the run of consecutive positions the samples open with.
    window = next(
        (i for i, position in enumerate(sampled) if position != i), len(sampled)
    )
    rate = (
        -(-(sum(m for m in marginals[: window - 1] if m > 0) + granule) // (window - 1))
        if window > 1
        else 0
    )
    anchors: dict[float, int] = {}
    if window > 1:
        anchors[sum(widths[1:window]) / (window - 1)] = rate
    for reading, position in zip(marginals[window - 1 :], sampled[window:]):
        # Near the window a reading is a lump at its rate; above, its excess is priced.
        anchor = max(reading, 0) + granule
        if window > 1:
            anchor = min(anchor, max(rate, max(reading, 0) - 3 * granule))
        # Two samples at one width (a bucket's inline variants): the dearer one.
        anchors[widths[position]] = max(anchors.get(widths[position], 0), anchor)
    knots = sorted(anchors.items(), reverse=True)

    def price(width: int) -> float:
        for (wide, at_wide), (narrow, at_narrow) in zip(knots, knots[1:]):
            if narrow <= width <= wide:
                return at_narrow + (at_wide - at_narrow) * (width - narrow) / (
                    wide - narrow
                )
        return knots[-1][1] if width < knots[-1][0] else knots[0][1]

    skipped = [widths[i] for i in range(len(widths)) if i not in set(sampled)]
    unsampled = math.ceil(sum(price(width) for width in skipped)) if knots else 0
    return CudagraphSeriesEstimate(max(first, 0) + sum(observed), unsampled)


def estimate_cudagraph_memory(
    samples: Mapping[str, Sequence[int]],
    ladders: Mapping[str, CapturedLadder],
) -> CudagraphMemoryEstimate:
    """Project every captured ladder; their pools are disjoint and so add up."""
    unmeasured = sorted(set(samples) - set(ladders))
    if unmeasured:
        raise ValueError(
            f"projection got samples for unknown ladders: {unmeasured}; re-run "
            "with --disable-cudagraph-memory-reserve to size the cache without "
            "a probe"
        )

    series = {
        name: _estimate_series(name, samples.get(name, ()), ladder)
        for name, ladder in ladders.items()
    }

    return CudagraphMemoryEstimate(
        series=series,
        measured_total=sum(estimate.measured for estimate in series.values()),
        unsampled_total=sum(estimate.unsampled for estimate in series.values()),
    )


def _ladders(executor: ModelExecutor, entries: int) -> dict[str, CapturedLadder]:
    """Every ladder a full capture records, and the positions the probe sampled."""
    drafter = CapturedLadder((1,), (0,))
    return {
        **executor.prefill_graph.capture_ladders(entries),
        **executor.forward_step.capture_ladders(entries),
        # Its own private pool, captured after both ladders and released with them.
        **(
            {"prefill:drafter": drafter}
            if executor.captures_drafter_prefill_graph
            else {}
        ),
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
    """Capture a few entries of each ladder, measure them, and project the rest.

    The reserve is what the captures themselves took plus the projected cost
    of the entries skipped -- what a boot without a probe pays inside
    its capture windows, the one-time bytes the first captures allocate
    there included. One-time bytes outside every capture (warmups,
    workspaces) are left to the utilization headroom, which funds them on a
    boot without a reserve too.
    """
    device_module = torch.get_device_module(server_args.device)
    observer = DriverMemoryDeltaObserver(device_module, gpu_id)
    executor.capture_graphs(entries=PROBE_ENTRIES_PER_LADDER, observer=observer)

    ladders = _ladders(executor, PROBE_ENTRIES_PER_LADDER)
    estimate = estimate_cudagraph_memory(observer.samples, ladders)
    reserve = _hungriest_rank(
        server_args, estimate.measured_total + estimate.unsampled_total
    )
    per_series = ", ".join(
        f"{name} {estimate.series[name].measured} measured + "
        f"{estimate.series[name].unsampled} projected over {len(ladder.widths)} entries"
        for name, ladder in sorted(ladders.items())
    )
    logger.info(
        f"CUDA-graph memory reserve: {reserve} bytes (this rank: captured "
        f"{estimate.measured_total}, unsampled entries {estimate.unsampled_total}; {per_series})"
    )
    # Per series: one ladder reading free is invisible in a non-zero total.
    for name, ladder in sorted(ladders.items()):
        series = estimate.series[name]
        samples = observer.samples.get(name, ())
        unsampled = len(ladder.widths) - len(samples)
        if unsampled and not series.unsampled:
            logger.warning(
                f"CUDA-graph memory reserve: {name} measured "
                f"{series.measured} bytes and priced its {unsampled} unsampled "
                "entries at nothing -- every sampled marginal was served from "
                "slack; re-run with --disable-cudagraph-memory-reserve if the "
                "boot then OOMs"
            )
    return reserve
