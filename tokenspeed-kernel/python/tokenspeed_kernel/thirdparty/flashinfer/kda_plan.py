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

"""Device schedules for FlashInfer's CuTe KDA prefill launch ABIs."""

from dataclasses import dataclass

import torch
from tokenspeed_kernel._triton import tl, triton

# Token chunk of the ordinary CuTe prefill (``kda_chunked_bt16``).
_ORDINARY_CHUNK = tl.constexpr(16)


@triton.jit
def _persistent_kda_plan(
    boundaries,
    cu32,
    chunk_prefix,
    offsets,
    chains,
    starts,
    lengths,
    sources,
    destinations,
    flags,
    num_sequences,
    num_workers,
    num_flags,
    HEADS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lanes = tl.arange(0, BLOCK)
    total = tl.full((), 0, tl.int64)
    longest = tl.full((), 0, tl.int64)
    tl.store(chunk_prefix, 0)
    tl.store(cu32, 0)
    for seq_begin in range(0, num_sequences, BLOCK):
        rows = seq_begin + lanes
        lo = tl.load(boundaries + rows, rows < num_sequences, 0)
        hi = tl.load(boundaries + rows + 1, rows < num_sequences, 0)
        chunks = (hi - lo + 31) // 32
        tl.store(cu32 + rows + 1, hi, rows < num_sequences)
        tl.store(
            chunk_prefix + rows + 1, total + tl.cumsum(chunks), rows < num_sequences
        )
        total += tl.sum(chunks)
        longest = tl.maximum(longest, tl.max(chunks))
    for flag_begin in range(0, num_flags, BLOCK):
        entries = flag_begin + lanes
        tl.store(flags + entries, 0, entries < num_flags)
    tl.debug_barrier()

    num_chains = num_sequences.to(tl.int64) * HEADS
    if num_chains <= num_workers:
        for chain_begin in range(0, num_chains, BLOCK):
            chain = chain_begin + lanes
            live = chain < num_chains
            row = chain // HEADS
            lo = tl.load(boundaries + row, live, 0)
            hi = tl.load(boundaries + row + 1, live, 0)
            tl.store(offsets + chain, chain, live)
            tl.store(chains + chain, chain, live)
            tl.store(starts + chain, 0, live)
            tl.store(lengths + chain, hi - lo, live)
            tl.store(sources + chain, -1, live)
            tl.store(destinations + chain, -1, live)
        tl.store(offsets + num_workers, num_chains)
    else:
        # A span is at least one whole chain, so a chain needs at most one
        # FP32 handoff; every handoff has a unique worker boundary.
        total *= HEADS
        span = tl.maximum(longest, (total + num_workers - 1) // num_workers)
        split_count = tl.full((), 0, tl.int64)
        for chain_begin in range(0, num_chains, BLOCK):
            chain = chain_begin + lanes
            live = chain < num_chains
            row = chain // HEADS
            lo = tl.load(boundaries + row, live, 0)
            hi = tl.load(boundaries + row + 1, live, 0)
            chunks = (hi - lo + 31) // 32
            prefix = tl.load(chunk_prefix + row, live, 0)
            first = prefix * HEADS + (chain % HEADS) * chunks
            worker = first // span
            last_worker = (first + chunks - 1) // span
            split = live & (last_worker != worker)
            cumulative = tl.cumsum(split.to(tl.int64))
            task = chain + split_count + cumulative - split.to(tl.int64)
            first_tokens = tl.minimum((span - first % span) * 32, hi - lo)
            tl.store(chains + task, chain, live)
            tl.store(starts + task, 0, live)
            tl.store(lengths + task, first_tokens, live)
            tl.store(sources + task, -1, live)
            tl.store(destinations + task, tl.where(split, last_worker - 1, -1), live)
            tl.store(chains + task + 1, chain, split)
            tl.store(starts + task + 1, first_tokens, split)
            tl.store(lengths + task + 1, hi - lo - first_tokens, split)
            tl.store(sources + task + 1, last_worker - 1, split)
            tl.store(destinations + task + 1, -1, split)
            tl.store(offsets + worker, task, live & (first % span == 0))
            tl.store(offsets + last_worker, task + 1, split)
            split_count += tl.sum(split.to(tl.int64))
        used_workers = (total + span - 1) // span
        for worker_begin in range(0, num_workers + 1, BLOCK):
            worker = worker_begin.to(tl.int64) + lanes
            tl.store(
                offsets + worker,
                num_chains + split_count,
                (worker >= used_workers) & (worker <= num_workers),
            )
        tl.debug_barrier()
        # Producers run first, independent chains next, consumers last, so
        # neighboring workers overlap instead of serializing their handoffs.
        for worker_begin in range(0, num_workers, BLOCK):
            worker = worker_begin.to(tl.int64) + lanes
            first_task = tl.load(offsets + worker, worker < num_workers, 0)
            end_task = tl.load(offsets + worker + 1, worker < num_workers, 0)
            nonempty = (worker < num_workers) & (first_task < end_task)
            first_source = tl.load(sources + first_task, nonempty, -1)
            last_destination = tl.load(destinations + end_task - 1, nonempty, -1)
            incoming = nonempty & (first_source >= 0)
            outgoing = nonempty & (last_destination >= 0)
            shift_by = outgoing.to(tl.int32) - incoming.to(tl.int32)
            count = (
                end_task - first_task - incoming.to(tl.int32) - outgoing.to(tl.int32)
            )
            moves = tl.max(tl.where(nonempty & (shift_by != 0), count, 0))
            for field in tl.static_range(5):
                pointer = (chains, starts, lengths, sources, destinations)[field]
                first_value = tl.load(pointer + first_task, incoming, 0)
                last_value = tl.load(pointer + end_task - 1, outgoing, 0)
                for step in range(moves):
                    source_index = tl.where(
                        outgoing, end_task - 2 - step, first_task + 1 + step
                    )
                    move = nonempty & (shift_by != 0) & (step < count)
                    value = tl.load(pointer + source_index, move, 0)
                    tl.store(pointer + source_index + shift_by, value, move)
                tl.store(pointer + first_task, last_value, outgoing)
                tl.store(pointer + end_task - 1, first_value, incoming)


@triton.jit(do_not_specialize=["num_sequences", "num_workers", "num_flags"])
def _kda_prefill_plan(
    boundaries,
    sequence_order,
    cu_chunks,
    cu32,
    chunk_prefix,
    offsets,
    chains,
    starts,
    lengths,
    sources,
    destinations,
    flags,
    num_sequences,
    num_workers,
    num_flags,
    HEADS: tl.constexpr,
    PERSISTENT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lanes = tl.arange(0, BLOCK)
    # Ordinary CuTe: stable longest-first sequence order and 16-token chunk
    # prefix, as FlashInfer's packed metadata kernel would build per launch.
    # Each sequence's rank counts the sequences ahead of it: quadratic in the
    # batch's sequences, which one CTA covers in a few tiles per 128.
    ordinary_total = tl.full((), 0, tl.int32)
    tl.store(cu_chunks, 0)
    for seq_begin in range(0, num_sequences, BLOCK):
        rows = seq_begin + lanes
        live = rows < num_sequences
        lo = tl.load(boundaries + rows, live, 0)
        length = tl.load(boundaries + rows + 1, live, 0) - lo
        chunks = ((length + _ORDINARY_CHUNK - 1) // _ORDINARY_CHUNK).to(tl.int32)
        tl.store(cu_chunks + rows + 1, ordinary_total + tl.cumsum(chunks), live)
        ordinary_total += tl.sum(chunks)
        rank = tl.zeros((BLOCK,), tl.int32)
        for other_begin in range(0, num_sequences, BLOCK):
            others = other_begin + lanes
            other_live = others < num_sequences
            other_lo = tl.load(boundaries + others, other_live, 0)
            other_length = tl.load(boundaries + others + 1, other_live, 0) - other_lo
            before = other_live[None, :] & (
                (other_length[None, :] > length[:, None])
                | (
                    (other_length[None, :] == length[:, None])
                    & (others[None, :] < rows[:, None])
                )
            )
            rank += tl.sum(before.to(tl.int32), axis=1)
        tl.store(sequence_order + rank, rows.to(tl.int32), live)
    if PERSISTENT:
        _persistent_kda_plan(
            boundaries,
            cu32,
            chunk_prefix,
            offsets,
            chains,
            starts,
            lengths,
            sources,
            destinations,
            flags,
            num_sequences,
            num_workers,
            num_flags,
            HEADS,
            BLOCK,
        )


@dataclass(frozen=True)
class KdaPrefillPlan:
    """Launch schedules of one packed batch, shared by every layer that scans it.

    Built once per forward from device boundaries (no D2H), then read by each
    layer's launch. ``flags`` holds one zeroed handoff row per layer, so the
    persistent launches need no per-layer reset; a row serves one launch, so
    refresh the plan before another forward reuses it. Capacity plans are
    refreshed in place before each replay, like the other prefill-graph
    metadata.

    Attributes:
        heads: Local KDA heads of every launch that reads the plan.
        sequence_order/cu_chunks: Ordinary-route int32 sequence order and
            16-token chunk prefix [sequences + 1].
        persistent: Whether the persistent schedule below is built; set when
            the heads divide by 8 and two CTAs per (sequence, head) exceed the
            SM count.
        workers: Persistent CTA count, ``min(sequences * heads, SMs)``.
        cu32, chunk_prefix, offsets, tasks: Persistent int32 boundaries, int64
            32-token chunk prefix, worker task offsets [workers + 1] and the
            five task arrays (chain, start, length, source, destination).
        flags: Int32 handoff flags [layers, workers].
        handoffs: FP32 handoff states shared by the plan's sequential launches.
    """

    heads: int
    sequence_order: torch.Tensor
    cu_chunks: torch.Tensor
    persistent: bool
    workers: int
    cu32: torch.Tensor
    chunk_prefix: torch.Tensor
    offsets: torch.Tensor
    tasks: torch.Tensor
    flags: torch.Tensor
    handoffs: torch.Tensor


def warm_kda_prefill_plan(heads: int) -> None:
    """Compile both plan variants before serving."""
    for persistent in (False, True):
        _kda_prefill_plan.warmup(
            torch.int64,
            torch.int32,
            torch.int32,
            torch.int32,
            torch.int64,
            torch.int32,
            torch.int32,
            torch.int32,
            torch.int32,
            torch.int32,
            torch.int32,
            torch.int32,
            1,
            1,
            1,
            HEADS=heads,
            PERSISTENT=persistent,
            BLOCK=128,
            grid=(1,),
        )


def _sm_count(device: torch.device) -> int:
    # FlashInfer's count sizes the persistent launch; planning with the same
    # count keeps every worker a resident CTA.
    from flashinfer.kda_kernels import kda_chunked_bt16 as kda

    return kda._device_sm_count(device)


def build_kda_prefill_plan(
    boundaries: torch.Tensor, *, heads: int, layers: int
) -> KdaPrefillPlan:
    """Allocate and fill the CuTe prefill schedules of one packed batch.

    Args:
        boundaries: Device int64 packed sequence offsets [sequences + 1].
        heads: Local KDA heads of every layer that reads the plan.
        layers: Launches that read the plan; each takes its own
            ``plan_layer`` row of handoff flags.

    Returns:
        A plan whose storage belongs to this forward or its CUDA graph pool.
    """
    sequences = boundaries.numel() - 1
    device = boundaries.device
    sm_count = _sm_count(device)
    persistent = heads % 8 == 0 and 2 * sequences * heads > sm_count
    workers = min(sequences * heads, sm_count) if persistent else 1
    capacity = (sequences * heads + workers + 15) // 16 * 16 if persistent else 1
    plan = KdaPrefillPlan(
        heads=heads,
        sequence_order=torch.empty(sequences, dtype=torch.int32, device=device),
        cu_chunks=torch.empty(sequences + 1, dtype=torch.int32, device=device),
        persistent=persistent,
        workers=workers,
        cu32=torch.empty(sequences + 1, dtype=torch.int32, device=device),
        chunk_prefix=torch.empty(sequences + 1, dtype=torch.int64, device=device),
        offsets=torch.empty(workers + 1, dtype=torch.int32, device=device),
        tasks=torch.empty((5, capacity), dtype=torch.int32, device=device),
        flags=torch.empty((layers, workers), dtype=torch.int32, device=device),
        # A handoff exists only when sequences * heads exceed the workers.
        handoffs=torch.empty(
            (workers if persistent and sequences * heads > sm_count else 1, 128, 128),
            dtype=torch.float32,
            device=device,
        ),
    )
    refresh_kda_prefill_plan(boundaries, plan)
    return plan


def refresh_kda_prefill_plan(boundaries: torch.Tensor, plan: KdaPrefillPlan) -> None:
    """Rebuild ``plan`` in place from live ``boundaries`` and zero its flags.

    Run once before the layers that read the plan, in their stream order.
    """
    if boundaries.numel() != plan.cu_chunks.numel():
        raise ValueError("KDA prefill plan sequence count changed")
    _kda_prefill_plan[(1,)](
        boundaries,
        plan.sequence_order,
        plan.cu_chunks,
        plan.cu32,
        plan.chunk_prefix,
        plan.offsets,
        *plan.tasks,
        plan.flags,
        boundaries.numel() - 1,
        plan.workers,
        plan.flags.numel(),
        HEADS=plan.heads,
        PERSISTENT=plan.persistent,
        BLOCK=128,
    )
