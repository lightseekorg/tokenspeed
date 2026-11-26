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


@triton.jit(do_not_specialize=["num_sequences", "num_workers"])
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
    for flag_begin in range(0, num_workers, BLOCK):
        workers = flag_begin + lanes
        tl.store(flags + workers, 0, workers < num_workers)
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


def warm_persistent_kda_plan(heads: int) -> None:
    """Compile the metadata kernel even when the startup batch uses ordinary CuTe."""
    _persistent_kda_plan.warmup(
        torch.int64,
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
        HEADS=heads,
        BLOCK=128,
        grid=(1,),
    )


def persistent_kda_plan(boundaries: torch.Tensor, heads: int, workers: int):
    """Build live, graph-replayable task lists and reset their handoff flags.

    Boundaries are device int64 packed sequence offsets. Returns the native
    int32 boundaries, worker offsets, five task arrays and handoff flags.
    All storage belongs to this invocation or its CUDA graph pool.
    """
    sequences = boundaries.numel() - 1
    capacity = (sequences * heads + workers + 15) // 16 * 16
    cu32 = torch.empty_like(boundaries, dtype=torch.int32)
    prefix = torch.empty_like(boundaries)
    offsets = torch.empty(workers + 1, dtype=torch.int32, device=boundaries.device)
    tasks = torch.empty((5, capacity), dtype=torch.int32, device=boundaries.device)
    flags = torch.empty(workers, dtype=torch.int32, device=boundaries.device)
    _persistent_kda_plan[(1,)](
        boundaries,
        cu32,
        prefix,
        offsets,
        *tasks,
        flags,
        sequences,
        workers,
        HEADS=heads,
        BLOCK=128,
    )
    return cu32, offsets, *tasks, flags


@triton.jit(do_not_specialize=["num_sequences"])
def _kda_prefill_plan(
    boundaries, sequence_order, cu_chunks, num_sequences, BLOCK: tl.constexpr
):
    # FlashInfer's packed metadata for the ordinary scan: stable longest-first
    # sequence order and 16-token chunk prefix. A sequence's rank counts the
    # sequences ahead of it, quadratic in the batch's sequences on one CTA.
    lanes = tl.arange(0, BLOCK)
    total = tl.full((), 0, tl.int32)
    tl.store(cu_chunks, 0)
    for seq_begin in range(0, num_sequences, BLOCK):
        rows = seq_begin + lanes
        live = rows < num_sequences
        lo = tl.load(boundaries + rows, live, 0)
        length = tl.load(boundaries + rows + 1, live, 0) - lo
        chunks = ((length + 15) // 16).to(tl.int32)
        tl.store(cu_chunks + rows + 1, total + tl.cumsum(chunks), live)
        total += tl.sum(chunks)
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


@dataclass(frozen=True)
class KdaPrefillPlan:
    """Ordinary-scan metadata of one packed batch, shared by every layer.

    Attributes:
        sequence_order: Int32 stable longest-first sequence order [sequences].
        cu_chunks: Int32 16-token chunk prefix [sequences + 1].
    """

    sequence_order: torch.Tensor
    cu_chunks: torch.Tensor


def warm_kda_prefill_plan() -> None:
    """Compile the plan kernel before serving."""
    _kda_prefill_plan.warmup(
        torch.int64, torch.int32, torch.int32, 1, BLOCK=128, grid=(1,)
    )


def build_kda_prefill_plan(boundaries: torch.Tensor) -> KdaPrefillPlan:
    """Plan the ordinary CuTe prefill of one packed batch.

    Args:
        boundaries: Device int64 packed sequence offsets [sequences + 1].

    Returns:
        A plan whose storage belongs to this forward or its CUDA graph pool.
    """
    sequences = boundaries.numel() - 1
    plan = KdaPrefillPlan(
        sequence_order=torch.empty(
            sequences, dtype=torch.int32, device=boundaries.device
        ),
        cu_chunks=torch.empty(
            sequences + 1, dtype=torch.int32, device=boundaries.device
        ),
    )
    refresh_kda_prefill_plan(boundaries, plan)
    return plan


def refresh_kda_prefill_plan(boundaries: torch.Tensor, plan: KdaPrefillPlan) -> None:
    """Rebuild ``plan`` in place from live ``boundaries``, before its layers run."""
    if boundaries.numel() != plan.cu_chunks.numel():
        raise ValueError("KDA prefill plan sequence count changed")
    _kda_prefill_plan[(1,)](
        boundaries,
        plan.sequence_order,
        plan.cu_chunks,
        boundaries.numel() - 1,
        BLOCK=128,
    )
