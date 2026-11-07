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
from tokenspeed_kernel.ops.attention.gdn import GdnStateCheckpoints
from tokenspeed_kernel.ops.attention.gdn.triton import (
    CAUSAL_CONV1D_BLOCK_M,
    build_causal_conv1d_capacity_metadata,
    refresh_causal_conv1d_capacity_metadata,
)

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.state.checkpoint import (
    InForwardStateCheckpoints,
)
from tokenspeed.runtime.layers.attention.backends.state.mamba import (
    MambaAttnBackend,
    MambaForwardMetadata,
)
from tokenspeed.runtime.utils.tensor import upload_packed
from tokenspeed.runtime.utils.triton import tl, triton

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


@dataclass(frozen=True)
class CapacityCheckpointLayout:
    """How capacity shapes store in-forward State checkpoints from a fixed scan.

    A captured scan bakes its checkpoint interval and buffer. The interval is
    the bound pool's prefix granularity, a divisor of every checkpoint offset
    the scheduler places, so one launch stores the state at any live
    checkpoint; the per-forward slots only say which stored state goes to
    which block.

    Args:
        every_n_tokens: The scan's checkpoint interval (the bound pool's
            prefix granularity).
        states: Float32 scan scratch ``[rows, *state_shape]`` that every shape
            shares (forwards run in stream order), with at least
            ``max(1, token_capacity // every_n_tokens)`` rows.
        group_ids: The State groups a checkpoint stores.
    """

    every_n_tokens: int
    states: torch.Tensor
    group_ids: tuple[str, ...]


@dataclass(kw_only=True)
class CapacityPrefillMetadata(MambaForwardMetadata):
    """Bucket-shaped metadata; see ``CapacityPrefillBackend``.

    With a ``CapacityCheckpointLayout`` the inherited checkpoint fields hold
    fixed slots, one per request slot (a row writes at most one checkpoint):
    ``state_checkpoints`` padded as its docstring describes,
    ``scan_state_checkpoints`` at the layout's interval with per-sequence
    counts refreshed in place, and ``scan_checkpoint_rows`` (scratch row 0 for
    an unused slot). Without one, the capacity layout stores no checkpoint and
    they stay ``None``.
    """

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
        live_total = self.query_start_loc[-1:]
        for tensor in tensors:
            if tensor is not None:
                _zero_rows_from(tensor, live_total)


_ZERO_ROWS_BLOCK = 1024


# The grid is the row count and the live total is read on device, so neither
# compiles a variant. The bound is the last of a shape's boundaries: its
# address alignment follows the request count, so it stays out of the compile
# key. row_stride and num_cols are model widths, fixed per scrubbed input.
@triton.jit(do_not_specialize_on_alignment=["live_ptr"])
def _zero_rows_from_kernel(ptr, live_ptr, row_stride, num_cols, BLOCK: tl.constexpr):
    # One program per row: live rows only read the bound, so an unpadded
    # forward writes nothing. The row offset grows with the token count, so it
    # is int64; offsets inside a row stay int32.
    row = tl.program_id(0)
    if row >= tl.load(live_ptr):
        base = ptr + row.to(tl.int64) * row_stride
        for start in range(0, num_cols, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            tl.store(
                base + cols,
                tl.zeros([BLOCK], dtype=ptr.dtype.element_ty),
                mask=cols < num_cols,
            )


def _zero_rows_from(tensor: torch.Tensor, live_total: torch.Tensor) -> None:
    """Zero ``tensor``'s rows from the device bound ``live_total[0]`` onward.

    A masked fill over the whole tensor would rewrite every live row too, for
    each scrubbed input of every layer; this touches the padding rows only.
    Host tensors and rows whose trailing dims are not contiguous keep the
    masked fill.
    """
    rows = tensor.shape[0]
    if rows == 0:
        return
    if not tensor.is_cuda or (tensor.dim() > 1 and not tensor[0].is_contiguous()):
        index = torch.arange(rows, device=tensor.device)
        padding = (index >= live_total).view(-1, *([1] * (tensor.dim() - 1)))
        tensor.masked_fill_(padding, 0)
        return
    _zero_rows_from_kernel[(rows,)](
        tensor,
        live_total,
        tensor.stride(0),
        tensor.numel() // rows,
        BLOCK=_ZERO_ROWS_BLOCK,
    )


def _capacity_metadata(source, bucket, checkpoints):
    """Create an isolated execution snapshot without allocating request state.

    Eager and captured forwards share the shape. Startup capture retains the
    snapshot; uncaptured shapes do not retain it.
    """
    capacity = PrefillCapacity(bucket, source.extend_seq_lens_cpu.numel())
    capacity.validate(source.cu_extend_seq_lens_cpu, bucket)
    if checkpoints is None and source.state_checkpoints is not None:
        raise ValueError("this capacity layout stores no in-forward State checkpoint")
    # The live checkpoint plan fits this forward only (its interval and scan
    # buffer); fixed slots replace it.
    cloned = _clone_metadata(
        replace(
            source,
            state_checkpoints=None,
            scan_state_checkpoints=None,
            scan_checkpoint_rows=None,
        )
    )
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
    if checkpoints is not None:
        (
            result.state_checkpoints,
            result.scan_state_checkpoints,
            result.scan_checkpoint_rows,
        ) = _checkpoint_slots(capacity, checkpoints, result.query_start_loc.device)
        _refresh_checkpoint_slots(result, source.state_checkpoints)
    return result


def _checkpoint_slots(
    capacity: PrefillCapacity,
    layout: CapacityCheckpointLayout,
    device: torch.device | str,
) -> tuple[InForwardStateCheckpoints, GdnStateCheckpoints, torch.Tensor]:
    """Allocate a shape's fixed checkpoint slots, all unused.

    Args:
        capacity: The shape; one slot per request slot.
        layout: The backend's checkpoint interval, scan scratch and groups.
        device: Where the slots live.

    Returns:
        ``(state checkpoints, scan checkpoints, scan row of each slot)`` for
        the ``CapacityPrefillMetadata`` fields of those names.
    """
    states = max(1, capacity.token_capacity // layout.every_n_tokens)
    if layout.states.shape[0] < states:
        raise ValueError("the checkpoint scratch holds fewer states than the shape")
    slots = capacity.num_sequences
    empty = torch.empty(0, dtype=torch.int64)
    checkpoints = InForwardStateCheckpoints(
        rows=empty,
        offsets=empty,
        token_index=torch.full(
            (slots,), capacity.token_capacity, dtype=torch.int64, device=device
        ),
        blocks_by_group={
            group: torch.full((slots,), -1, dtype=torch.int32, device=device)
            for group in layout.group_ids
        },
    )
    scan = GdnStateCheckpoints(
        every_n_tokens=layout.every_n_tokens,
        states=layout.states,
        cu_starts=torch.zeros(slots + 1, dtype=torch.int32, device=device),
    )
    return checkpoints, scan, torch.zeros(slots, dtype=torch.int64, device=device)


def _refresh_checkpoint_slots(
    target: CapacityPrefillMetadata, live: InForwardStateCheckpoints | None
) -> None:
    """Write this forward's in-forward State checkpoints into the fixed slots.

    Run once the target's lengths are refreshed: the scan stores
    ``length // interval`` states per sequence, none for an unused request
    slot (which scans one row). A live checkpoint takes the scan row of its
    offset; the slots past the live ones keep an in-range token index (the
    bucket end) and block ``-1``, which the masked writers skip. Device
    tensors keep their addresses; one pinned upload carries the scan counts
    and rows.

    Args:
        target: Capacity metadata holding slots from ``_checkpoint_slots``.
        live: The forward's checkpoints from its scheduler metadata, or None.

    Raises:
        ValueError: More live checkpoints than request slots, or an offset
            the fixed interval does not divide.
        RuntimeError: The live checkpoints name other State groups.
    """
    slots = target.state_checkpoints
    scan = target.scan_state_checkpoints
    interval = scan.every_n_tokens
    counts = target.extend_seq_lens_cpu.to(torch.int64) // interval
    cu_starts = torch.zeros(counts.numel() + 1, dtype=torch.int64)
    torch.cumsum(counts, 0, out=cu_starts[1:])
    if int(cu_starts[-1]) > scan.states.shape[0]:
        raise ValueError("live lengths store more states than the checkpoint scratch")
    rows = torch.zeros(counts.numel(), dtype=torch.int64)
    count = 0 if live is None else live.rows.numel()
    if count:
        if count > rows.numel():
            raise ValueError("more in-forward checkpoints than request slots")
        if live.blocks_by_group.keys() != slots.blocks_by_group.keys():
            raise RuntimeError("graph state groups changed without pool rebind")
        if bool((live.offsets % interval != 0).any()):
            raise ValueError(
                "an in-forward checkpoint is not a multiple of the captured "
                f"{interval}-token scan interval"
            )
        rows[:count] = cu_starts[live.rows] + live.offsets // interval - 1
    device_cu_starts, device_rows = upload_packed(
        (cu_starts.to(torch.int32), rows), scan.cu_starts.device
    )
    scan.cu_starts.copy_(device_cu_starts)
    target.scan_checkpoint_rows.copy_(device_rows)
    slots.token_index[count:].fill_(target.capacity.token_capacity)
    for group, blocks in slots.blocks_by_group.items():
        blocks[count:].fill_(-1)
        if count:
            blocks[:count].copy_(live.blocks_by_group[group])
    if count:
        slots.token_index[:count].copy_(live.token_index)
    empty = torch.empty(0, dtype=torch.int64)
    target.state_checkpoints = replace(
        slots,
        rows=empty if live is None else live.rows,
        offsets=empty if live is None else live.offsets,
    )


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


def prepare_capacity_prefill_metadata(source, token_capacity, target, *, checkpoints):
    """Prepare one execution shape, optionally refreshing retained storage.

    ``source`` is the scheduler-derived metadata for this forward. Fresh
    execution metadata reserves its positive-length request count. A retained
    target may have extra request slots: convolution gives them zero length
    and the scan one masked dummy token per slot. ``target`` is either a
    startup-retained view for that shape or None for fresh per-forward storage.
    ``checkpoints`` is the backend's ``CapacityCheckpointLayout``, or None when
    its capacity layout stores no in-forward State checkpoint (a source with
    checkpoints is then refused); a retained target was built with the same.
    No request state is allocated here. Returns the metadata consumed by every
    recurrent layer, without a temporary backend binding or an execution-mode flag.
    """
    if target is None:
        return _capacity_metadata(source, token_capacity, checkpoints)
    if target.capacity.token_capacity != token_capacity:
        raise ValueError("prefill metadata token capacity changed")
    retained = target.scan_state_checkpoints
    if (checkpoints is None) != (retained is None) or (
        checkpoints is not None
        and checkpoints.every_n_tokens != retained.every_n_tokens
    ):
        raise RuntimeError("checkpoint layout changed without graph recapture")
    _refresh_capacity_metadata(target, source)
    return target


def _refresh_capacity_metadata(target, source):
    """Refresh target contents at stable addresses without modifying source.

    Run once before all consuming layers, in consumer-stream order; do not
    overlap a refresh with replay using the same buffers. In-place refresh is
    not allocation-free or H2D-free. State-group geometry changes raise
    instead of replacing bound storage.
    """
    if target.scan_state_checkpoints is None and source.state_checkpoints is not None:
        raise ValueError("this capacity layout stores no in-forward State checkpoint")
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
    if target.scan_state_checkpoints is not None:
        _refresh_checkpoint_slots(target, source.state_checkpoints)


class CapacityPrefillBackend(MambaAttnBackend):
    """A recurrent backend whose layers a prefill graph can capture inline.

    Startup capture retains one capacity metadata per (tokens, requests) shape;
    each later forward of that shape refreshes it in place before replay. A
    subclass decides which forwards it admits, whether uncaptured shapes also
    run on the capacity layout, and whether that layout stores in-forward
    State checkpoints (``_capacity_checkpoint_interval``).
    """

    # Whether eager forwards of uncaptured shapes also run the capacity layout.
    _capacity_layout_when_uncaptured: bool

    def __init__(self, config: AttnConfig, spec: SoftmaxAttnConfig) -> None:
        super().__init__(config, spec)
        self._prefill_metadata: dict[tuple[int, int], CapacityPrefillMetadata] = {}
        self._prefill_metadata_pool: CachePool | None = None
        # The widest token capacity the prefill graph captures, which sizes the
        # checkpoint scan scratch every shape shares; None before a capture.
        self._prefill_graph_max_tokens: int | None = None
        self._checkpoint_states: torch.Tensor | None = None

    def init_prefill_graph_state(self, max_num_tokens: int, max_bs: int) -> None:
        # The orchestrator releases old graphs first; these are execution buffers, not state pages.
        self._reset_prefill_metadata()
        self._prefill_graph_max_tokens = max_num_tokens

    def _reset_prefill_metadata(self) -> None:
        self._prefill_metadata.clear()
        self._prefill_metadata_pool = None
        self._checkpoint_states = None

    def _capacity_checkpoint_interval(self) -> int | None:
        """The scan interval at which the capacity layout stores in-forward
        State checkpoints, or None when it stores none (a forward writing
        some then keeps its live metadata and eager break)."""
        return None

    def _checkpoint_layout(
        self, token_capacity: int
    ) -> CapacityCheckpointLayout | None:
        """This backend's checkpoint layout for a shape, allocating the scratch once."""
        interval = self._capacity_checkpoint_interval()
        if interval is None:
            return None
        rows = max(1, token_capacity // interval)
        if self._checkpoint_states is None or self._checkpoint_states.shape[0] < rows:
            # Sized for the widest captured shape when the capture announced it;
            # a shape that outgrows the scratch gets a new one, and shapes built
            # on the old one keep it.
            widest = max(token_capacity, self._prefill_graph_max_tokens or 0)
            ssm_states = self._state_components(self._state_layer_ids()[0])[1]
            self._checkpoint_states = torch.empty(
                (max(1, widest // interval), *ssm_states.shape[1:]),
                dtype=torch.float32,
                device=self.device,
            )
        return CapacityCheckpointLayout(
            every_n_tokens=interval,
            states=self._checkpoint_states,
            group_ids=self._state_group_ids,
        )

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
        # Without checkpoint slots the capacity layout carries no in-forward
        # State checkpoint and a captured scan stores none: such a forward
        # keeps its live metadata and runs this layer at its eager break.
        if (
            source.state_checkpoints is not None
            and self._capacity_checkpoint_interval() is None
        ):
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
        target = prepare_capacity_prefill_metadata(
            source,
            token_capacity,
            retained,
            checkpoints=self._checkpoint_layout(token_capacity),
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
