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
)

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

    def scrub_padding(self, *tensors: torch.Tensor | None) -> None:
        """Zero the rows past the live tokens of bucket-shaped inputs.

        The live total changes between replays of one captured shape, so the
        bound is read from the device boundaries (the last convolution
        boundary, which the refresh pads with the live total), never from a
        Python int the capture would freeze.
        """
        live_total = self.query_start_loc[-1]
        for tensor in tensors:
            if tensor is None:
                continue
            rows = torch.arange(tensor.shape[0], device=tensor.device)
            padding = (rows >= live_total).view(-1, *([1] * (tensor.ndim - 1)))
            tensor.masked_fill_(padding, 0)


def _capacity_metadata(source, bucket):
    """Create an isolated execution snapshot without allocating request state.

    Eager and captured forwards share the shape. Startup capture retains the
    snapshot; uncaptured shapes do not retain it.
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


def prepare_capacity_prefill_metadata(source, token_capacity, target):
    """Prepare one execution shape, optionally refreshing retained storage.

    ``source`` is the scheduler-derived metadata for this forward. Fresh
    execution metadata reserves its positive-length request count. A retained
    target may have extra request slots: convolution gives them zero length
    and the scan one masked dummy token per slot. ``target`` is either a
    startup-retained view for that shape or None for fresh per-forward storage.
    No request state is allocated here. Returns the metadata consumed by every
    recurrent layer, without a temporary backend binding or an execution-mode flag.
    """
    if target is None:
        return _capacity_metadata(source, token_capacity)
    if target.capacity.token_capacity != token_capacity:
        raise ValueError("prefill metadata token capacity changed")
    _refresh_capacity_metadata(target, source)
    return target


def _refresh_capacity_metadata(target, source):
    """Refresh target contents at stable addresses without modifying source.

    Run once before all consuming layers, in consumer-stream order; do not
    overlap a refresh with replay using the same buffers. In-place refresh is
    not allocation-free or H2D-free. State-group geometry changes raise
    instead of replacing bound storage.
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
    live_total = int(source.cu_extend_seq_lens_cpu[-1])
    # Convolution gives unused request slots no work: repeated boundaries.
    target.query_start_loc[: actual_bs + 1].copy_(source.query_start_loc)
    target.query_start_loc[actual_bs + 1 :].fill_(live_total)
    # The scan needs a positive length per slot: each unused slot scans one
    # padding row past the live tokens (zeroed by ``scrub_padding`` before
    # and after the convolution; its output row and final state are dropped).
    for name in (
        "scan_query_start_loc",
        "query_start_loc_int64",
        "cu_extend_seq_lens_cpu",
    ):
        old, new = getattr(target, name), getattr(source, name)
        old[: actual_bs + 1].copy_(new)
        # Built where the target lives: no pageable H2D copy on the forward thread.
        torch.arange(
            live_total + 1,
            live_total + padding + 1,
            dtype=old.dtype,
            device=old.device,
            out=old[actual_bs + 1 :],
        )
    target.extend_seq_lens_cpu[:actual_bs].copy_(source.extend_seq_lens_cpu)
    target.extend_seq_lens_cpu[actual_bs:].fill_(1)
    refresh_causal_conv1d_capacity_metadata(
        target.query_start_loc,
        target.conv_prefill_metadata,
        target.capacity.token_capacity,
    )
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
        # The capacity layout carries no in-forward State checkpoint, and a
        # captured scan stores none: such a forward keeps its live metadata
        # and runs this layer at its eager break.
        if source.state_checkpoints is not None:
            return False
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
        target = prepare_capacity_prefill_metadata(source, token_capacity, retained)
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
