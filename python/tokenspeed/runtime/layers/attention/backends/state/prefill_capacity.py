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

"""Capacity-shaped recurrent prefill metadata shared by eager and CUDA graphs."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.attention.gdn.triton import (
    CAUSAL_CONV1D_BLOCK_M,
    build_causal_conv1d_capacity_metadata,
    refresh_causal_conv1d_capacity_metadata,
)

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.state.mamba import (
    MambaAttnBackend,
    MambaForwardMetadata,
    _PrefillCheckpointBatch,
)
from tokenspeed.runtime.utils.tensor import upload_packed

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.configs.base import (
        AttnConfig,
        SoftmaxAttnConfig,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool


@dataclass(frozen=True)
class PrefillCapacity:
    """Token rows and request slots one prefill shape reserves.

    Each request may use up to ``token_capacity`` rows, but their combined
    live lengths must fit it too.
    """

    token_capacity: int
    num_sequences: int

    def __post_init__(self):
        if self.token_capacity <= 0 or self.num_sequences <= 0:
            raise ValueError("prefill capacities must be positive")

    def validate(self, live_boundaries: torch.Tensor, tokens: int) -> None:
        """Admit host boundaries of positive packed sequences that fit, without a D2H."""
        if tokens != self.token_capacity:
            raise ValueError("input extent differs from the token capacity")
        if live_boundaries.numel() != self.num_sequences + 1:
            raise ValueError("sequence count differs from the live boundaries")
        lengths = live_boundaries[1:] - live_boundaries[:-1]
        if live_boundaries[0].item() != 0 or (lengths <= 0).any().item():
            raise ValueError("capacity prefill requires positive packed sequences")
        if live_boundaries[-1].item() > self.token_capacity:
            raise ValueError("live tokens exceed the token capacity")


@dataclass(kw_only=True)
class CapacityPrefillMetadata(MambaForwardMetadata):
    capacity: PrefillCapacity

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


def _checkpoint_slot_batch(source, bucket, tail_capacity, num_sequences):
    """Pad checkpoint execution to the captured request capacity.

    Inactive slots get one zero-input token, never a zero-length native scan.
    Their token map and state destination are negative, so dummy results cannot
    escape into request outputs or persistent state. Real cache ownership and
    checkpoint selection continue to come exclusively from source metadata.
    """
    lengths = source.extend_seq_lens_cpu
    padding = num_sequences - lengths.numel()
    if padding < 0:
        raise ValueError("live request count exceeds the captured capacity")
    live = source.prefill_checkpoint_batch
    body_lengths = lengths if live is None else live.body_seq_lens_cpu
    tail_lengths = lengths - body_lengths
    active = tail_lengths > 0
    slot_lengths = tail_lengths.clamp_min(1)
    rows = torch.arange(num_sequences, dtype=torch.int64)
    starts = source.cu_extend_seq_lens_cpu[:-1]

    def bounds(values):
        return torch.cat((values.new_zeros(1), values.cumsum(0))).to(torch.int64)

    # Native scans require positive lengths even for unused request slots.
    # These slots have no source tokens, output rows or cache destinations.
    body_slots = torch.cat((body_lengths, body_lengths.new_ones(padding)))
    tail_slots = torch.cat((slot_lengths, slot_lengths.new_ones(padding)))
    body_bounds, tail_bounds = bounds(body_slots), bounds(tail_slots)
    PrefillCapacity(bucket, rows.numel()).validate(body_bounds, bucket)
    PrefillCapacity(tail_capacity, rows.numel()).validate(tail_bounds, tail_capacity)

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
    active = torch.cat((active, active.new_zeros(padding)))
    starts = torch.cat(
        (starts, starts.new_full((padding,), int(source.cu_extend_seq_lens_cpu[-1])))
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
        body_slots,
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
        body_seq_lens_cpu=body_slots,
        body_cu_seqlens_cpu=body_bounds,
        tail_token_indices=tail_indices,
        tail_query_start_loc=tail_boundaries,
        tail_seq_lens_cpu=tail_slots,
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
        raise RuntimeError("graph state groups changed without pool rebind")
    inactive = target.prefill_checkpoint_batch.state_update_rows < 0
    for group, indices in old.items():
        if new is None:
            indices.fill_(-1)
        else:
            if (
                indices.numel() < new[group].numel()
                or indices.dtype != new[group].dtype
            ):
                raise RuntimeError("graph state index geometry changed")
            indices[: new[group].numel()].copy_(new[group])
            indices[new[group].numel() :].fill_(-1)
            indices.masked_fill_(inactive, -1)


def _capacity_metadata(source, bucket, tail_capacity):
    """Create an isolated execution snapshot without allocating request state.

    Eager and captured forwards reserve the same checkpoint/tail slots.
    Startup capture retains the snapshot; uncaptured shapes do not retain it.
    """
    capacity = PrefillCapacity(bucket, source.extend_seq_lens_cpu.numel())
    capacity.validate(source.cu_extend_seq_lens_cpu, bucket)
    cloned = _clone_metadata(source)
    result = CapacityPrefillMetadata(
        **{
            field.name: getattr(cloned, field.name)
            for field in fields(MambaForwardMetadata)
        },
        capacity=capacity,
    )
    result.prefill_checkpoint_batch = _checkpoint_slot_batch(
        source, bucket, tail_capacity, capacity.num_sequences
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


def prepare_capacity_prefill_metadata(
    source, token_capacity, prefix_granularity, target
):
    """Prepare one execution shape, optionally refreshing retained storage.

    ``source`` is the scheduler-derived metadata for this forward. Fresh
    execution metadata reserves its positive-length request count. A retained
    target may have extra request slots: convolution gives them zero length,
    while each native scan packs one masked dummy token per slot. ``target`` is
    either a startup-retained view for that shape or None for fresh per-forward storage.
    No request state is allocated here. Returns the metadata consumed by every
    recurrent layer, without a temporary backend binding or an execution-mode flag.
    """
    if target is None:
        tail_capacity = min(
            token_capacity,
            source.extend_seq_lens_cpu.numel() * max(1, prefix_granularity - 1),
        )
        return _capacity_metadata(source, token_capacity, tail_capacity)
    if target.capacity.token_capacity != token_capacity:
        raise ValueError("prefill metadata token capacity changed")
    _refresh_capacity_metadata(target, source)
    return target


def _refresh_capacity_metadata(target, source):
    """Refresh target contents at stable addresses without modifying source.

    Run once before all consuming layers, in consumer-stream order; do not
    overlap a refresh with replay using the same buffers. Checkpoint maps are
    rebuilt on CPU and uploaded through temporary packed storage before being
    copied into the target. In-place refresh is not allocation-free or H2D-free.
    State-group geometry changes raise instead of replacing bound storage.
    """
    actual_bs = source.extend_seq_lens_cpu.numel()
    capacity = target.capacity
    PrefillCapacity(capacity.token_capacity, actual_bs).validate(
        source.cu_extend_seq_lens_cpu, capacity.token_capacity
    )
    padding = capacity.num_sequences - actual_bs
    if (
        padding < 0
        or int(source.cu_extend_seq_lens_cpu[-1]) + padding > capacity.token_capacity
    ):
        raise ValueError("padded requests exceed the captured capacity")
    for name in (
        "query_start_loc",
        "scan_query_start_loc",
        "query_start_loc_int64",
        "extend_seq_lens_cpu",
        "cu_extend_seq_lens_cpu",
    ):
        old, new = getattr(target, name), getattr(source, name)
        old[: new.numel()].copy_(new)
        # Repeated boundaries give convolution no work for unused requests.
        # Body/tail scans use their separate positive-length packed boundaries.
        pad_value = (
            0
            if name == "extend_seq_lens_cpu"
            else int(source.cu_extend_seq_lens_cpu[-1])
        )
        old[new.numel() :].fill_(pad_value)
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
            capacity.num_sequences,
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
            raise RuntimeError("graph state groups changed without pool rebind")
        for group, indices in old.items():
            if (
                indices.numel() < new[group].numel()
                or indices.dtype != new[group].dtype
            ):
                raise RuntimeError("graph state index geometry changed")
            indices[: new[group].numel()].copy_(new[group])
            indices[new[group].numel() :].fill_(-1)


class CapacityPrefillBackend(MambaAttnBackend):
    """A recurrent backend whose layers a prefill graph can capture inline.

    Startup capture retains one capacity metadata per (tokens, requests) shape;
    each later forward of that shape refreshes it in place before replay. A
    subclass decides which forwards it admits and whether uncaptured shapes
    also run on the capacity layout.
    """

    # Whether eager forwards of uncaptured shapes also run the capacity layout.
    _capacity_layout_when_uncaptured: bool

    def __init__(self, config: AttnConfig, spec: SoftmaxAttnConfig) -> None:
        super().__init__(config, spec)
        self._prefill_metadata: dict[tuple[int, int], CapacityPrefillMetadata] = {}
        self._prefill_metadata_pool: CachePool | None = None

    def init_prefill_graph_state(self, max_num_tokens: int, max_bs: int) -> None:
        # The orchestrator releases old graphs first; these are execution buffers, not state pages.
        self._reset_prefill_metadata()

    def _reset_prefill_metadata(self) -> None:
        self._prefill_metadata.clear()
        self._prefill_metadata_pool = None

    def _publish_cache_pool(self, cache_pool: CachePool) -> None:
        super()._publish_cache_pool(cache_pool)
        self._reset_prefill_metadata()

    @property
    def prefill_metadata_is_capture_ready(self) -> bool:
        metadata = self.forward_metadata
        return (
            isinstance(metadata, CapacityPrefillMetadata)
            and self._prefill_metadata.get(
                (metadata.capacity.token_capacity, metadata.capacity.num_sequences)
            )
            is metadata
        )

    @abstractmethod
    def _admits_capacity_prefill(self) -> bool:
        """Whether this backend's configuration can run capacity prefills."""

    def admits_prefill_graph(
        self, token_capacity: int, bs: int, forward_mode: ForwardMode
    ) -> bool:
        return (
            self._admits_capacity_prefill()
            and self.step_counter is None
            and forward_mode.is_extend()
        )

    def prepare_prefill_metadata(
        self, token_capacity: int, bs: int, forward_mode: ForwardMode, *, capture: bool
    ) -> bool:
        if not self.admits_prefill_graph(token_capacity, bs, forward_mode):
            return False
        if (
            self._prefill_metadata_pool is not None
            and self._prefill_metadata_pool is not self.cache_pool
        ):
            raise RuntimeError("cache pool changed without graph release/recapture")
        source = self.forward_metadata
        key = (token_capacity, bs)
        retained = self._prefill_metadata.get(key)
        if (
            retained is None
            and not capture
            and not self._capacity_layout_when_uncaptured
        ):
            return True
        actual_bs = source.extend_seq_lens_cpu.numel()
        if actual_bs > bs or (actual_bs != bs and (capture or retained is None)):
            raise ValueError("request padding requires an existing captured capacity")
        target = prepare_capacity_prefill_metadata(
            source, token_capacity, self._prefix_granularity, retained
        )
        if capture:
            # Only startup grows retained storage; serving never captures.
            self._prefill_metadata[key] = target
            self._prefill_metadata_pool = self.cache_pool
        if self._prefill_metadata.get(key) is target:
            self._refresh_captured_prefill(target)
        self.forward_metadata = target
        return True

    def _refresh_captured_prefill(self, metadata: CapacityPrefillMetadata) -> None:
        """Refresh backend-owned buffers a retained shape's graphs read, before each use."""
