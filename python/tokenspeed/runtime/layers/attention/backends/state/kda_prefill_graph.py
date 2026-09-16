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

"""KDA capacity metadata owned by startup-captured outer graphs."""

from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass, replace

import torch
from tokenspeed_kernel.ops.attention.gdn.triton import (
    CAUSAL_CONV1D_BLOCK_M,
    build_causal_conv1d_capacity_metadata,
    refresh_causal_conv1d_capacity_metadata,
)
from tokenspeed_kernel.ops.attention.kda import KdaPrefillCapacity

from tokenspeed.runtime.layers.attention.backends.state.mamba import (
    MambaForwardMetadata,
    _PrefillCheckpointBatch,
)
from tokenspeed.runtime.utils.tensor import upload_packed


@dataclass(kw_only=True)
class KdaPrefillGraphMetadata(MambaForwardMetadata):
    capacity: KdaPrefillCapacity

    @property
    def prefill_token_extent(self) -> int:
        return self.capacity.token_capacity


@dataclass(frozen=True)
class _CheckpointCapacityBatch(_PrefillCheckpointBatch):
    """Fixed execution slots, not additional scheduler requests or state blocks.

    ``rows`` and ``body_rows`` map request rows, never cache block IDs.
    Negative ``tail_state_rows`` keep a dummy tail from replacing body state.
    ``output_token_sources`` maps original output rows into concatenated
    body/tail storage; the tail base is the body capacity, not its live length.
    ``packed_capacity`` is the restored outer output capacity, not the sum of
    the two scan allocations.
    """

    packed_capacity: int
    tail_state_rows: torch.Tensor
    output_token_sources: torch.Tensor

    @property
    def output_sources(self) -> torch.Tensor:
        return self.output_token_sources

    @property
    def state_update_rows(self) -> torch.Tensor:
        return self.tail_state_rows

    @property
    def use_token_views(self) -> bool:
        # Live body/tail offsets change between replays; Python slices would
        # freeze their capture-time values. Both scans use the shared packer.
        return False

    @property
    def token_extent(self) -> int:
        return self.packed_capacity


def _checkpoint_slot_batch(source, bucket, tail_capacity):
    """Pad checkpoint execution to one tail slot per real request.

    Inactive slots get one zero-input token, never a zero-length native scan.
    Their token map and state destination are negative, so dummy results cannot
    escape into request outputs or persistent state. Real cache ownership and
    checkpoint selection continue to come exclusively from source metadata.
    """
    lengths = source.extend_seq_lens_cpu
    live = source.prefill_checkpoint_batch
    body_lengths = lengths if live is None else live.body_seq_lens_cpu
    tail_lengths = lengths - body_lengths
    active = tail_lengths > 0
    slot_lengths = tail_lengths.clamp_min(1)
    rows = torch.arange(lengths.numel(), dtype=torch.int64)
    starts = source.cu_extend_seq_lens_cpu[:-1]

    def bounds(values):
        return torch.cat((values.new_zeros(1), values.cumsum(0))).to(torch.int64)

    body_bounds, tail_bounds = bounds(body_lengths), bounds(slot_lengths)
    KdaPrefillCapacity(bucket, rows.numel()).validate(body_bounds, bucket)
    KdaPrefillCapacity(tail_capacity, rows.numel()).validate(tail_bounds, tail_capacity)

    def indices(sequence_starts, sequence_lengths, capacity):
        offsets = torch.arange(int(sequence_lengths.sum()), dtype=torch.int64)
        offsets -= torch.repeat_interleave(
            bounds(sequence_lengths)[:-1], sequence_lengths
        )
        packed = torch.repeat_interleave(sequence_starts, sequence_lengths) + offsets
        result = torch.full((capacity,), -1, dtype=torch.int64)
        result[: packed.numel()] = packed
        return result

    body_indices = indices(starts, body_lengths, bucket)
    tail_indices = indices(starts + body_lengths, slot_lengths, tail_capacity)
    tail_indices[: int(slot_lengths.sum())].masked_fill_(
        ~torch.repeat_interleave(active, slot_lengths), -1
    )
    # Build the inverse once with the other host metadata, not once per layer.
    # Negative sources also make the gather write zero to all bucket padding.
    output_sources = torch.full((bucket,), -1, dtype=torch.int64)
    for indices_, offset in ((body_indices, 0), (tail_indices, bucket)):
        valid = indices_ >= 0
        output_sources[indices_[valid]] = torch.arange(indices_.numel())[valid] + offset
    parts = (
        rows,
        starts,
        body_lengths,
        torch.zeros_like(rows),
        body_indices,
        body_bounds,
        tail_indices,
        tail_bounds,
        rows.masked_fill(~active, -1),
        output_sources,
    )
    (
        device_rows,
        device_starts,
        checkpoint_lengths,
        positions,
        body_indices,
        body_boundaries,
        tail_indices,
        tail_boundaries,
        state_rows,
        output_sources,
    ) = upload_packed(parts, source.query_start_loc.device)
    if live is not None:
        positions.index_copy_(0, live.rows, live.checkpoint_positions)
    return _CheckpointCapacityBatch(
        rows=device_rows,
        sequence_starts=device_starts,
        checkpoint_seq_lens=checkpoint_lengths,
        checkpoint_positions=positions,
        body_rows=device_rows,
        body_token_indices=body_indices,
        body_query_start_loc=body_boundaries,
        body_seq_lens_cpu=body_lengths.clone(),
        body_cu_seqlens_cpu=body_bounds,
        tail_token_indices=tail_indices,
        tail_query_start_loc=tail_boundaries,
        tail_seq_lens_cpu=slot_lengths,
        tail_cu_seqlens_cpu=tail_bounds,
        packed_capacity=bucket,
        tail_state_rows=state_rows,
        output_token_sources=output_sources,
    )


def _refresh_checkpoint_destinations(target, source):
    """Refresh existing destination buffers and mask inactive checkpoint slots.

    No live checkpoints clears all destinations, preventing prior page IDs
    from leaking into this replay. State-group keys, shapes and dtypes must
    match; incompatible storage raises rather than rebinding captured tensors.
    """
    old = target.state_checkpoint_blocks_by_group
    new = source.state_checkpoint_blocks_by_group
    if new is not None and old.keys() != new.keys():
        raise RuntimeError("KDA graph state groups changed without pool rebind")
    inactive = target.prefill_checkpoint_batch.state_update_rows < 0
    for group, indices in old.items():
        if new is None:
            indices.fill_(-1)
        else:
            if indices.shape != new[group].shape or indices.dtype != new[group].dtype:
                raise RuntimeError("KDA graph state index geometry changed")
            indices.copy_(new[group])
            indices.masked_fill_(inactive, -1)


def _capacity_metadata(source, bucket, tail_capacity):
    """Create an isolated execution snapshot without allocating request state.

    Reserve fixed checkpoint/tail slots for inline capture.
    The graph owner retains the snapshot and its stable device buffers.
    """
    capacity = KdaPrefillCapacity(bucket, source.extend_seq_lens_cpu.numel())
    capacity.validate(source.cu_extend_seq_lens_cpu, bucket)
    cloned = _clone_metadata(source)
    result = KdaPrefillGraphMetadata(
        **{
            field.name: getattr(cloned, field.name)
            for field in fields(MambaForwardMetadata)
        },
        capacity=capacity,
    )
    result.prefill_checkpoint_batch = _checkpoint_slot_batch(
        source, bucket, tail_capacity
    )
    result.state_checkpoint_blocks_by_group = {
        group: torch.full_like(indices, -1)
        for group, indices in source.state_out_blocks_by_group.items()
    }
    _refresh_checkpoint_destinations(result, source)
    # Bound total packed work plus one partial block per request. Stable maps
    # are shared across layers and refreshed before replay on its consumer stream.
    result.conv_prefill_metadata = build_causal_conv1d_capacity_metadata(
        result.query_start_loc,
        bucket,
        CAUSAL_CONV1D_BLOCK_M,
    )
    return result


def _clone_metadata(value):
    """Clone tensors, dictionaries and dataclass fields recursively.

    Other values are retained as-is, not deep-copied. New mutable field types
    need an isolation review before they can be shared by these snapshots.
    """
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_metadata(item) for key, item in value.items()}
    if is_dataclass(value):
        return replace(
            value,
            **{
                field.name: _clone_metadata(getattr(value, field.name))
                for field in fields(value)
            },
        )
    return value


class KdaOuterGraphBinding:
    """Own a bucket's stable KDA metadata for capture in the outer graph.

    Args:
        backend: KDA leaf whose metadata is temporarily bound.
        bucket: Packed token capacity selected by the outer graph.
        source: Live or startup placeholder metadata defining the scan topology.
    """

    def __init__(self, backend, bucket, source):
        self.backend = backend
        self.pool = backend.cache_pool
        tail_capacity = min(
            bucket,
            source.extend_seq_lens_cpu.numel()
            * max(1, backend._prefix_granularity - 1),
        )
        self.metadata = _capacity_metadata(source, bucket, tail_capacity)

    def compatible(self, ctx):
        """Check pool, transfer, forward mode and exact request-count matching.

        Length, tail-capacity and state-group checks still run during refresh
        and may raise. Passing this predicate is not full geometry validation.
        """
        source = self.backend.forward_metadata
        return (
            self.backend.cache_pool is self.pool
            and self.backend.step_counter is None
            and ctx.forward_mode.is_extend()
            and ctx.num_extends == ctx.bs == self.metadata.capacity.num_sequences
            and source.extend_seq_lens_cpu is not None
            and source.extend_seq_lens_cpu.numel()
            == self.metadata.capacity.num_sequences
        )

    @contextmanager
    def bind(self, refresh: bool):
        """Temporarily bind the outer owner's stable metadata and inline flag.

        Use ``refresh=False`` for warmup/capture and ``True`` for live replay.
        Variants sharing the outer pool run serially. Exiting restores backend
        references, but does not undo GPU work already queued on the stream.
        """
        backend = self.backend
        source = backend.forward_metadata
        previous_inline = backend.prefill_graph_inline
        if refresh:
            _refresh_capacity_metadata(self.metadata, source)
        backend.forward_metadata = self.metadata
        backend.prefill_graph_inline = True
        try:
            yield
        finally:
            backend.forward_metadata = source
            backend.prefill_graph_inline = previous_inline


def _refresh_capacity_metadata(target, source):
    """Refresh target contents at stable addresses without modifying source.

    Run once before all consuming layers, in consumer-stream order; do not
    overlap a refresh with replay using the same buffers. Checkpoint maps are
    rebuilt on CPU and uploaded through temporary packed storage before being
    copied into the target. In-place refresh is not allocation-free or H2D-free.
    State-group geometry changes raise instead of replacing bound storage.
    """
    target.capacity.validate(
        source.cu_extend_seq_lens_cpu, target.capacity.token_capacity
    )
    for name in (
        "query_start_loc",
        "scan_query_start_loc",
        "query_start_loc_int64",
        "extend_seq_lens_cpu",
        "cu_extend_seq_lens_cpu",
    ):
        getattr(target, name).copy_(getattr(source, name))
    refresh_causal_conv1d_capacity_metadata(
        target.query_start_loc,
        target.conv_prefill_metadata,
        target.capacity.token_capacity,
    )
    checkpoint = target.prefill_checkpoint_batch
    if checkpoint is not None:
        live = _checkpoint_slot_batch(
            source,
            target.capacity.token_capacity,
            checkpoint.tail_token_indices.numel(),
        )
        for field in fields(_CheckpointCapacityBatch):
            old, new = getattr(checkpoint, field.name), getattr(live, field.name)
            if isinstance(old, torch.Tensor):
                old.copy_(new)
        _refresh_checkpoint_destinations(target, source)
    for name in (
        "state_in_blocks_by_group",
        "state_out_blocks_by_group",
    ):
        old, new = getattr(target, name), getattr(source, name)
        if old is None and new is None:
            continue
        if old.keys() != new.keys():
            raise RuntimeError("KDA graph state groups changed without pool rebind")
        for group, indices in old.items():
            if indices.shape != new[group].shape or indices.dtype != new[group].dtype:
                raise RuntimeError("KDA graph state index geometry changed")
            indices.copy_(new[group])
