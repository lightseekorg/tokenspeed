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

"""KDA (Kimi Delta Attention) backends: the scan seams KDA overrides on the
shared linear-attention machinery, and the composite wrapper KDA hybrids use.
See ``KdaAttnBackend`` for what separates the family from GDN."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.attention.gdn import GdnStateCheckpoints
from tokenspeed_kernel.ops.attention.kda import (
    KDA_RETAINED_COMMIT_DESCRIPTOR_COLUMNS,
    KdaOutputNorm,
    KdaPrefillCapacity,
    KdaRetainedHistory,
    KdaRetainedRecords,
    kda_fused_paged_verify,
    kda_fused_paged_verify_fuses_producers,
    kda_fused_paged_verify_uses_split_producers,
    kda_materialize_retained_pages,
    kda_materialize_retained_state,
    kda_paged_prefill,
)
from tokenspeed_kernel.ops.attention.kda import (
    kda_recurrent_layout as kda_recurrent_layout_default,
)
from tokenspeed_kernel.ops.attention.kda import (
    kda_reset_retained_length,
    kda_retained_supported,
    kda_verify_conv_update,
    resolve_kda_batched_retained_commit,
    resolve_kda_batched_retained_prepare,
)
from tokenspeed_kernel.platform import current_platform
from typing_extensions import override

from tokenspeed.runtime.execution.request_slots import RequestSlotLayout
from tokenspeed.runtime.execution.slot_state import stream_scope
from tokenspeed.runtime.layers.attention.backends.state.mamba import (
    _reject_skip_term,
    logger,
)
from tokenspeed.runtime.layers.attention.backends.state.prefill_capacity import (
    CapacityPrefillBackend,
    CapacityPrefillMetadata,
)
from tokenspeed.runtime.layers.attention.backends.support import TreeSupport
from tokenspeed.runtime.layers.attention.kv_cache.recipes.kda import (
    kda_history_capacity,
)
from tokenspeed.runtime.utils.cuda_stream import StreamFork

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.configs.base import (
        AttnConfig,
        SoftmaxAttnConfig,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool


KDA_PREFILL_BACKENDS = ("auto", "fla", "flashkda", "cutedsl_kda", "cudnn")


def resolve_kda_backend(kda_backend: str) -> str:
    """Resolve the KDA prefill backend policy for runtime construction.

    On AMD, the backend policy is ignored and compatible kernels are selected
    using registry priority. On NVIDIA, ``auto`` picks ``cutedsl_kda`` when its
    device-specific implementation is available, ``flashkda`` on SM90+, and
    ``fla`` otherwise. Explicit CuteDSL selection is validated against device
    support. Decode is unaffected.

    Args:
        kda_backend: Normalized prefill backend name from the launch arguments.

    Returns:
        The resolved prefill backend name passed to ``KdaAttnBackend``.
    """
    platform = current_platform()
    if platform.is_amd:
        # Named backend policies are NVIDIA-specific; let the registry decide.
        return "auto"

    from tokenspeed_kernel.ops.attention.kda.cute_dsl import cutedsl_kda_supported

    if kda_backend == "auto":
        if cutedsl_kda_supported():
            resolved = "cutedsl_kda"
        elif platform.is_hopper_plus:
            resolved = "flashkda"
        else:
            resolved = "fla"
        logger.info(f"KDA prefill backend auto-resolved to {resolved!s}")
        return resolved
    if kda_backend == "cutedsl_kda" and not cutedsl_kda_supported():
        raise ValueError(
            "--kda-backend cutedsl_kda requires an NVIDIA sm_100 or sm_103 device"
        )
    if kda_backend == "cudnn":
        from tokenspeed_kernel.ops.attention.kda.cudnn import cudnn_kda_supported

        if not cudnn_kda_supported():
            raise ValueError(
                "--kda-backend cudnn requires an NVIDIA sm_100 or sm_103 device"
            )
    return kda_backend


def _slice_kda_prefill_inputs(
    num_real_tokens: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slice packed KDA inputs to the real-token prefix along their token axis."""
    return (
        query[:, :num_real_tokens],
        key[:, :num_real_tokens],
        value[:, :num_real_tokens],
        gate[:, :num_real_tokens],
        beta[:, :num_real_tokens],
    )


def _pack_token_spans(
    tensor: torch.Tensor, spans: Sequence[tuple[int, int]]
) -> torch.Tensor:
    """Pack each ``[start, end)`` span of a packed input's token axis in order."""
    if len(spans) == 1:
        return tensor[:, spans[0][0] : spans[0][1]]
    return torch.cat([tensor[:, start:end] for start, end in spans], dim=1)


class KdaAttnBackend(CapacityPrefillBackend):
    """Attention backend for KDA linear attention layers (Kimi-K3).

    Everything generic to linear attention -- state paging and cache groups --
    is inherited. Decode and speculative verify share deferred commitment.
    Active requests keep FP32 replay records and their accepted count in
    their own request slots, outside the cache arena; cache pages hold only
    dense checkpoints. A decode commit only advances the count, except in the
    forward that ends a request, where it folds the records into the live
    State block; a retraction folds them before imaging. Before each verify
    round, one launch pair over every layer folds the records of requests
    that move to a private live block or would overflow the history capacity
    (``_prepare_retained_history``). Standard decode is a one-token window
    with one accepted token.
    KDA also replaces the scan seams: its decay gate is
    per-channel (a low-rank ``f_a``/``f_b`` projection plus raw beta logits)
    where GDN's is scalar per head, and its decode/verify kernels can fuse
    the conv, the gate GEMV and the recurrence into a single launch.
    """

    _prefill_needs_request_slots = True
    # Retraction folds accepted records into the dense page before slot reuse
    # (fold_deferred_state). A recycled or restored slot starts with an
    # accepted count of zero.
    workspace_slot_state = (
        "_retained_records",
        "_retained_length_storage",
    )
    _verify_reads_committed_recurrent_state = True
    _verify_packed_qkv_views = True
    # A second scan stores each requested interior checkpoint.
    _stores_in_forward_state_checkpoints = True
    # Eager forwards keep the same metadata contract as replay.
    _capacity_layout_when_uncaptured = True

    def __init__(
        self,
        config: AttnConfig,
        spec: SoftmaxAttnConfig,
        *,
        enable_prefill_graph: bool,
        kda_backend: str,
    ) -> None:
        if config.is_draft:
            raise ValueError(
                "KDA draft execution is unsupported; use a full-attention draft"
            )
        super().__init__(config, spec)
        self._prefill_graph_enabled = enable_prefill_graph
        self.max_bs = config.max_bs
        # Capability validation uses the platform layout when binding the pool.
        self.kda_recurrent_layout = kda_recurrent_layout_default()
        if self.kda_recurrent_layout not in ("k_major", "v_major"):
            raise ValueError(
                f"unsupported KDA recurrent layout {self.kda_recurrent_layout!r}"
            )
        self._record_slot_count = RequestSlotLayout(
            config.max_resident_requests
        ).capacity
        self._retained_capacity = kda_history_capacity(
            self.speculative_num_draft_tokens
        )
        self._batched_replay_kernel = None
        self._batched_prepare_kernel = None
        self._verify_split_producers = False
        # RuntimeStates.stopped and the executor's per-row request_ends, bound
        # by the executor (bind_request_stop_flags).
        self._request_stopped: torch.Tensor | None = None
        self._request_ends: torch.Tensor | None = None
        self._reset_replay_state()
        self.kda_backend = kda_backend.strip().lower()
        if self.kda_backend not in KDA_PREFILL_BACKENDS:
            raise ValueError(
                f"--kda-backend must be one of {', '.join(KDA_PREFILL_BACKENDS)}; "
                f"got {self.kda_backend!r}"
            )
        logger.info(
            f"KDA prefill routes through {self.kda_backend!s}; decode remains on the "
            "platform-selected kernels",
        )

    @override
    def _prefill_state_dtype(self, cache_dtype: torch.dtype) -> torch.dtype:
        return torch.float32

    @override
    def _prefill_supports_strided_qkv(self) -> bool:
        return self.kda_backend in ("cutedsl_kda", "cudnn")

    def _admits_capacity_prefill(self) -> bool:
        return self._prefill_graph_enabled and self.kda_backend in (
            "cutedsl_kda",
            "cudnn",
        )

    def _reset_replay_state(self) -> None:
        self._verify_producer_stream: torch.cuda.Stream | None = None
        self._verify_producer_forks: dict[int, StreamFork] = {}
        self._replay_workspace_ready = False
        self._retained_qkv: torch.Tensor | None = None
        self._retained_records: torch.Tensor | None = None
        # INT32 [layers, request slots, 1]: accepted records since the dense
        # state of each request's page. _retained_lengths views it per layer.
        self._retained_length_storage: torch.Tensor | None = None
        self._record_slots: torch.Tensor | None = None
        # The recurrent state dtype every layer shares (validate_cache_pool).
        self._retained_state_dtype: torch.dtype | None = None
        # Per-layer INT32 [request slots, 1] views of _retained_length_storage.
        self._retained_lengths: dict[int, torch.Tensor] = {}
        self._retained_histories: dict[tuple[int, int], KdaRetainedHistory] = {}
        self._replay_descriptors: torch.Tensor | None = None
        self._replay_group_indices: torch.Tensor | None = None
        self._descriptor_row_by_layer: dict[int, int] = {}
        self._replay_group_ids: tuple[str, ...] = ()
        self._replay_group_rows: dict[str, int] = {}
        # Host cache of each group's (table offset in elements, row stride,
        # column count, source-page output pointer). Upload only when these
        # change; a new packed table base alone does not invalidate the cache.
        self._replay_commit_table_descriptors_host: tuple[
            tuple[int, int, int, int], ...
        ] = ()
        # Device INT64 [groups, 4] copy of the host descriptors for the commit planner.
        self._replay_commit_tables: torch.Tensor | None = None
        # Fixed-capacity INT32 page buffers, viewed as [groups, batch] at use.
        # Reads preserve source checkpoint IDs before page outputs are updated.
        self._replay_commit_reads: torch.Tensor | None = None
        # Destination state-page IDs selected for the accepted endpoints.
        self._replay_commit_pages: torch.Tensor | None = None
        # INT32 [max_bs] accepted counts, clamped by the shared commit planner.
        self._replay_commit_steps: torch.Tensor | None = None
        # Split-producer INT64 [max_bs, 3]: (slot, generation, checkpoint
        # position). Refreshed before verify; PR1 uses the batch row as slot.
        self._replay_record_metadata: torch.Tensor | None = None
        # INT64 [layers, max_bs, draft_tokens, 2]: (absolute position,
        # generation). Producers stamp records; recovery checks these stamps
        # before reading payloads to reject stale or overwritten records.
        self._replay_record_stamps: torch.Tensor | None = None
        # Incremented for each metadata refresh to identify this round's
        # workspace ownership, even when a batch slot is reused.
        self._replay_record_generation = 0

    @property
    @override
    def deferred_state_commit(self) -> bool:
        return True

    @staticmethod
    def _replay_shape(kv_pool: CachePool) -> dict[str, int]:
        first_layer = min(kv_pool.state_group_by_layer)
        num_heads, head_dim = kv_pool.get_component(
            first_layer, "recurrent_state"
        ).shape[1:3]
        return {"num_heads": num_heads, "head_dim": head_dim}

    @override
    def validate_cache_pool(self, cache_pool: CachePool) -> None:
        super().validate_cache_pool(cache_pool)
        geometries = set()
        for layer in cache_pool.state_group_by_layer:
            conv = cache_pool.get_component(layer, "conv_state")
            recurrent = cache_pool.get_component(layer, "recurrent_state")
            pages, heads, dim, value_dim = recurrent.shape
            if (
                dim <= 0
                or value_dim != dim
                or recurrent.dtype not in (torch.float32, torch.bfloat16)
                or conv.shape != (pages, 3 * heads * dim, 3)
                or conv.dtype not in (self.dtype, torch.bfloat16)
                or conv.stride()[1:] != (3, 1)
                or conv.device != recurrent.device
                or any(
                    stride <= 0
                    for field in (conv, recurrent)
                    for stride in field.stride()
                )
            ):
                raise ValueError(
                    "retained KDA cache has incompatible state geometry or strides"
                )
            if not kda_retained_supported(
                self.dtype,
                recurrent_layout=self.kda_recurrent_layout,
                num_heads=heads,
                head_dim=dim,
                conv_kernel_size=conv.shape[-1] + 1,
            ):
                raise ValueError(
                    "retained KDA cache requires supported target verify geometry"
                )
            # One commit and fold launch serves every layer.
            geometries.add((heads, dim, conv.dtype, recurrent.dtype, recurrent.device))
        if len(geometries) != 1:
            raise ValueError("retained KDA layers require common state geometry")

    @override
    def _publish_cache_pool(self, cache_pool: CachePool) -> None:
        # validate_cache_pool already checked every field before publication.
        super()._publish_cache_pool(cache_pool)
        self._reset_replay_state()
        shape = self._replay_shape(cache_pool)
        self._batched_replay_kernel = resolve_kda_batched_retained_commit(
            self.dtype, recurrent_layout=self.kda_recurrent_layout, **shape
        )
        self._batched_prepare_kernel = resolve_kda_batched_retained_prepare(
            self.dtype, recurrent_layout=self.kda_recurrent_layout, **shape
        )
        self._verify_split_producers = kda_fused_paged_verify_uses_split_producers(
            self.dtype,
            recurrent_layout=self.kda_recurrent_layout,
            **shape,
        )

    def _reset_graph_state(self) -> None:
        super()._reset_graph_state()
        self._retained_histories: dict[tuple[int, int], KdaRetainedHistory] = {}

    def _history_for_rows(self, layer_id, live_indices, record_slots):
        row = self._descriptor_row_by_layer[layer_id]
        correction, key, decay = self._retained_records[:, row].unbind(1)
        return KdaRetainedHistory(
            correction,
            key,
            decay,
            self._retained_lengths[layer_id],
            live_indices,
            record_slots,
        )

    def _retained_history(self, layer_id, live_indices):
        key = (layer_id, live_indices.numel())
        history = self._retained_histories.get(key)
        if history is None or history.live_state_indices is not live_indices:
            history = self._history_for_rows(
                layer_id, live_indices, self._record_slots[: live_indices.numel()]
            )
            self._retained_histories[key] = history
        return history

    @override
    def deferred_decode_state_groups(self) -> frozenset[str]:
        # Accepted records stay in request slots until a request ends or is
        # retracted (fold_deferred_state).
        return frozenset(self._state_groups())

    @override
    def bind_request_stop_flags(
        self, stopped: torch.Tensor, request_ends: torch.Tensor
    ) -> None:
        if stopped.ndim != 1 or stopped.dtype != torch.bool:
            raise ValueError("request stop flags must be a bool vector")
        if request_ends.ndim != 1 or request_ends.dtype != torch.int32:
            raise ValueError("request ends must be an INT32 vector")
        self._request_stopped = stopped
        self._request_ends = request_ends

    @override
    def fold_deferred_state(
        self, state_folds: Mapping[str, torch.Tensor], *, stream: torch.Stream
    ) -> None:
        """Fold each listed slot's accepted records into its listed State block.

        The scheduler lists a retraction victim's live block per group, the
        block its decode commits wrote in place; a slot without records --
        restored and not yet decoding -- leaves its block unchanged.
        """
        if self._retained_records is None:
            return
        with stream_scope(stream):
            # Per group, the unit-stride page and slot vectors the kernels take.
            folds = {}
            for group_id, pairs in state_folds.items():
                pairs = pairs.pin_memory().to(self.device, non_blocking=True)
                folds[group_id] = (
                    pairs[:, 1].contiguous(),
                    pairs[:, 0].to(torch.int64),
                )
            for layer_id in self._state_layer_ids():
                fold = folds.get(self._state_group_for(layer_id))
                if fold is None:
                    continue
                pages, slots = fold
                _, recurrent = self._state_components(layer_id)
                kda_materialize_retained_pages(
                    recurrent,
                    history=self._history_for_rows(layer_id, pages, slots),
                    source_indices=pages,
                    recurrent_layout=self.kda_recurrent_layout,
                )

    def _prepare_prefill_state_inputs(
        self,
        layer_id,
        conv_states,
        recurrent_states,
        source_indices,
        live_indices,
        *,
        dtype: torch.dtype,
    ):
        recurrent, initial = super()._prepare_prefill_state_inputs(
            layer_id,
            conv_states,
            recurrent_states,
            source_indices,
            live_indices,
            dtype=dtype,
        )
        if self._retained_records is not None:
            slots = self.forward_metadata.req_pool_indices
            history = self._history_for_rows(layer_id, live_indices, slots)
            kda_materialize_retained_state(
                recurrent,
                history=history,
                source_indices=source_indices,
                recurrent_layout=self.kda_recurrent_layout,
            )
            # The scan starts from the gathered logical state and stores a
            # dense one into the live page, so these rows' slots hold no
            # accepted records any more. A separate ordered launch clears
            # their counts after every gathered tile has read them.
            kda_reset_retained_length(
                self._retained_lengths[layer_id],
                write_indices=live_indices,
                record_slots=slots,
                page_count=recurrent_states.shape[0],
            )
        return recurrent, initial

    @override
    def reset_request_workspace(
        self, request_slots: torch.Tensor, *, stream: torch.Stream
    ) -> None:
        """Clear the accepted counts a reused slot keeps for its previous owner.

        An owner whose end the device did not see (an abort, a finish decided
        on the host) leaves its records counted. The reset is ordered on
        ``stream`` ahead of the new owner's first forward, or of a fold of
        the slot that a retraction before it would run.
        """
        super().reset_request_workspace(request_slots, stream=stream)
        if self._retained_length_storage is not None and request_slots.numel():
            with stream_scope(stream):
                storage = self._retained_length_storage
                storage.index_fill_(
                    1, request_slots.to(device=storage.device, dtype=torch.int64), 0
                )

    @override
    def import_slot_state(
        self, slot: int, src: torch.Tensor, stream, *, admission_id: int
    ) -> None:
        """Start a restored request with no accepted records.

        Retraction folded the request's records into the page it imaged; the
        restored slot may still count records of another owner.
        """
        super().import_slot_state(slot, src, stream, admission_id=admission_id)
        if self._retained_length_storage is not None:
            with stream_scope(stream):
                self._retained_length_storage[:, slot].zero_()

    def _allocate_replay_workspace(self) -> None:
        """Allocate max_bs * T original QKV rows before eager or graph execution.

        Batch refreshes reuse these buffers; binding a pool resets them.
        """
        if self._replay_workspace_ready:
            return
        rows = self.max_bs * self.speculative_num_draft_tokens
        layer_ids = tuple(self._state_layer_ids())
        self._descriptor_row_by_layer = {
            layer_id: row for row, layer_id in enumerate(layer_ids)
        }
        self._replay_group_ids = tuple(self._state_groups())
        self._replay_group_rows = {
            group_id: row for row, group_id in enumerate(self._replay_group_ids)
        }
        with torch.inference_mode(False):
            addresses = torch.zeros(
                (
                    len(layer_ids),
                    KDA_RETAINED_COMMIT_DESCRIPTOR_COLUMNS,
                ),
                dtype=torch.uint64,
                device=self.device,
            )
            group_indices = torch.tensor(
                [
                    self._replay_group_rows[self._state_group_for(layer_id)]
                    for layer_id in layer_ids
                ],
                dtype=torch.int32,
                device=self.device,
            )
            first_conv, first_state = self._state_components(layer_ids[0])
            heads, dim = first_state.shape[1:3]
            self._retained_state_dtype = first_state.dtype
            # Each request keeps its record row when the batch is reordered.
            self._retained_records = torch.zeros(
                (
                    self._record_slot_count,
                    len(layer_ids),
                    3,
                    heads,
                    self._retained_capacity,
                    dim,
                ),
                dtype=torch.float32,
                device=self.device,
            )
            self._retained_length_storage = torch.zeros(
                (len(layer_ids), self._record_slot_count, 1),
                dtype=torch.int32,
                device=self.device,
            )
            self._retained_lengths = {
                layer_id: self._retained_length_storage[row]
                for row, layer_id in enumerate(layer_ids)
            }
            self._record_slots = torch.zeros(
                self.max_bs, dtype=torch.int64, device=self.device
            )
            if self._verify_split_producers and first_conv.is_cuda:
                self._verify_producer_stream = torch.cuda.Stream(
                    device=first_conv.device, priority=-1
                )
                self._verify_producer_forks = {
                    layer_id: StreamFork(self._verify_producer_stream)
                    for layer_id in layer_ids
                }
            payload_shape = (len(layer_ids), rows)
            self._retained_qkv = torch.empty(
                (*payload_shape, first_conv.shape[1]),
                dtype=self.dtype,
                device=self.device,
            )
            descriptors = []
            for row, layer in enumerate(layer_ids):
                conv, recurrent = self._state_components(layer)
                length = self._retained_lengths[layer]
                qkv = self._retained_qkv[row]
                descriptors.append(
                    [
                        qkv.data_ptr(),
                        conv.data_ptr(),
                        length.data_ptr(),
                        conv.shape[0],
                        qkv.stride(0),
                        conv.stride(0),
                        length.stride(0),
                        conv.stride(1),
                        conv.stride(2),
                        recurrent.data_ptr(),
                        *recurrent.stride(),
                    ]
                )
            addresses.copy_(
                torch.tensor(descriptors, dtype=torch.uint64, device=self.device)
            )
            self._replay_descriptors = addresses
            self._replay_group_indices = group_indices
            self._replay_commit_tables = torch.empty(
                (len(self._replay_group_ids), 4),
                dtype=torch.int64,
                device=self.device,
            )
            self._replay_commit_pages = torch.empty(
                len(self._replay_group_ids) * self.max_bs,
                dtype=torch.int32,
                device=self.device,
            )
            self._replay_commit_reads = torch.empty_like(self._replay_commit_pages)
            self._replay_commit_steps = torch.empty(
                self.max_bs, dtype=torch.int32, device=self.device
            )
            self._replay_record_metadata = torch.full(
                (self.max_bs, 3), -1, dtype=torch.int64, device=self.device
            )
            self._replay_record_stamps = torch.full(
                (
                    len(layer_ids),
                    self.max_bs,
                    self.speculative_num_draft_tokens,
                    2,
                ),
                -1,
                dtype=torch.int64,
                device=self.device,
            )
            if first_state.is_cuda and not torch.cuda.is_current_stream_capturing():
                # Compile the prepare launches before serving: one padding row
                # takes the serving signature and changes nothing.
                padding = torch.full(
                    (len(self._replay_group_ids), 1),
                    self.pad_slot_id,
                    dtype=torch.int32,
                    device=self.device,
                )
                self._prepare_retained_rows(
                    padding,
                    padding,
                    torch.zeros(2, dtype=torch.int32, device=self.device),
                    self._record_slots[:1],
                )

        self._replay_workspace_ready = True

    @override
    def _verify_state_blocks(
        self,
        bs: int,
        padded_bs: int,
        block_tables: Mapping[str, torch.Tensor],
        *,
        pages_out: Mapping[str, torch.Tensor],
        live_pages_out: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        # The scheduler names input and live pages in fixed table slots.
        tables = super()._verify_state_blocks(
            bs,
            padded_bs,
            block_tables,
            pages_out=pages_out,
            live_pages_out=live_pages_out,
        )
        if (
            self._replay_commit_tables is not None
            and self._replay_commit_tables.is_cuda
        ):
            rows = [tables[group] for group in self._replay_group_ids]
            if any(
                table.dtype != torch.int32 or table.stride(1) != 1 for table in rows
            ):
                raise ValueError(
                    "KDA commit requires INT32 tables with contiguous columns"
                )
            # Packed tables can move without changing their relative geometry.
            # Masking stopped rows allocates each table separately, so those
            # offsets can change even when the batch shape stays the same.
            table_base = rows[0]
            table_descriptors = tuple(
                (
                    (table.data_ptr() - table_base.data_ptr())
                    // table_base.element_size(),
                    table.stride(0),
                    table.shape[1],
                    pages_out[group].data_ptr(),
                )
                for group, table in zip(self._replay_group_ids, rows, strict=True)
            )
            if table_descriptors != self._replay_commit_table_descriptors_host:
                # Keep the forward thread ahead of the GPU. The pinned allocator
                # retains this staging allocation until the async copy completes.
                self._replay_commit_tables.copy_(
                    torch.tensor(
                        table_descriptors,
                        dtype=torch.int64,
                        device="cpu",
                        pin_memory=True,
                    ),
                    non_blocking=True,
                )
                self._replay_commit_table_descriptors_host = table_descriptors
        return tables

    @override
    def _prepare_verify_records(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        draft_token_num: int,
        req_pool_indices: torch.Tensor,
    ) -> None:
        self._record_slots[:bs].copy_(req_pool_indices[:bs])
        if self._replay_record_metadata is not None and seq_lens.is_cuda:
            from tokenspeed_kernel.ops.attention.kda.vllm_recoverssm import (
                prepare_recoverssm_metadata,
            )

            # Refresh the live rows before verify writes candidate records.
            # A new generation prevents recovery from consuming an older
            # round's records when the same workspace slots are reused.
            self._replay_record_generation += 1
            prepare_recoverssm_metadata(
                seq_lens,
                self._replay_record_metadata,
                batch=bs,
                generation=self._replay_record_generation,
                draft_token_num=draft_token_num,
            )

    @override
    def refresh_decode_metadata(
        self,
        bs: int,
        actual_bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        forward_mode: ForwardMode,
        block_tables: Mapping[str, torch.Tensor],
        num_extends: int = 0,
        for_graph_replay: bool = False,
        **kwargs,
    ) -> None:
        super().refresh_decode_metadata(
            bs,
            actual_bs,
            req_pool_indices,
            seq_lens,
            forward_mode=forward_mode,
            block_tables=block_tables,
            num_extends=num_extends,
            for_graph_replay=for_graph_replay,
            **kwargs,
        )
        # The refresh arms the commit exactly when it prepared a verify round
        # over live rows: an eager or graph decode, or a mixed forward's
        # decode rows. Each reaches its verify only after this point.
        if self._verify_commit_ctx is not None:
            self._prepare_retained_history(bs, actual_bs)

    def _prepare_retained_history(self, bs: int, actual_bs: int) -> None:
        """Ready every layer's retained history for this round's verify.

        Two launches for all layers, ordered after the refresh wrote the
        round's pages, boundaries and record slots and before the forward. A
        request that moves to a private live block, or whose accepted count
        plus the round's window exceeds the history capacity (a shorter row
        folds like a full one: verify reads at most capacity - window
        records), gets its source state with its records applied in its live
        block, and its count clears.
        Verify then reads only live blocks, and never writes a block or count.
        """
        if not self._replay_descriptors.is_cuda:
            # CPU-only metadata tests run no kernels, as in the record refresh.
            return
        self._prepare_retained_rows(
            self.state_in_tables[bs - 1][:, :actual_bs],
            self.state_out_tables[bs - 1][:, :actual_bs],
            self.forward_metadata.query_start_loc[: actual_bs + 1],
            self._record_slots[:actual_bs],
        )

    def _prepare_retained_rows(
        self,
        source_pages: torch.Tensor,
        live_pages: torch.Tensor,
        query_start_loc: torch.Tensor,
        record_slots: torch.Tensor,
    ) -> None:
        """Launch the batched prepare over [groups, rows] source and live pages."""
        correction, key, decay = self._retained_records.unbind(2)
        _, _, heads, capacity, dim = correction.shape
        self._batched_prepare_kernel(
            descriptors=self._replay_descriptors,
            group_indices=self._replay_group_indices,
            source_indices=source_pages,
            live_indices=live_pages,
            query_start_loc=query_start_loc,
            record_slots=record_slots,
            record_slot_count=self._record_slot_count,
            retained_correction=correction,
            retained_key=key,
            retained_decay=decay,
            history_capacity=capacity,
            draft_token_num=self.speculative_num_draft_tokens,
            num_heads=heads,
            head_dim=dim,
            recurrent_state_dtype=self._retained_state_dtype,
        )

    @override
    def _resolve_verify_commit_pages(
        self, accepted_length: torch.Tensor, group_ids: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Use the shared resolver when no GPU replay workspace is available.
        # Otherwise, plan all state groups in one launch after acceptance.
        if self._replay_commit_tables is None or not accepted_length.is_cuda:
            return super()._resolve_verify_commit_pages(accepted_length, group_ids)
        from tokenspeed_kernel.ops.attention.kda.triton import (
            commit_state_groups,
        )

        # Descriptor rows and output rows must refer to the same state groups.
        if tuple(group_ids) != self._replay_group_ids:
            raise ValueError(
                "commit group order must match the prepared table descriptors"
            )
        assert self._replay_commit_reads is not None
        assert self._replay_commit_pages is not None
        assert self._replay_commit_steps is not None
        tables, width, _ = self._verify_commit_ctx
        batch = accepted_length.shape[0]
        # Reuse fixed-capacity storage, packed for the current batch size.
        # Returned views remain valid until the next planner call overwrites them.
        pages = self._replay_commit_pages[: len(group_ids) * batch].view(
            len(group_ids), batch
        )
        steps = self._replay_commit_steps[:batch]
        # Preserve source page IDs for recovery and select destination pages
        # from each group's live slot. Counts include the target token and are
        # clamped to [1, width]. This plans the commit; it does not write states.
        commit_state_groups(
            accepted_length,
            tables[self._replay_group_ids[0]],
            self._replay_commit_tables,
            batch_size=batch,
            draft_tokens=width,
            pages_out=pages,
            reads_out=self._replay_commit_reads[: len(group_ids) * batch].view(
                len(group_ids), batch
            ),
            steps_out=steps,
        )
        return pages, steps

    @override
    def _ensure_verify_scratch(self, bs: int, draft_token_num: int) -> None:
        self._allocate_replay_workspace()

    @override
    def preallocate_verify_workspace(self, max_bs: int, draft_token_num: int) -> int:
        self._ensure_verify_scratch(max_bs, draft_token_num)
        payload_bytes = (
            self._retained_qkv.nbytes
            + self._retained_records.nbytes
            + self._retained_length_storage.nbytes
        )
        metadata_bytes = sum(
            tensor.nbytes
            for tensor in (
                self._replay_descriptors,
                self._replay_group_indices,
                self._replay_commit_tables,
                self._replay_commit_pages,
                self._replay_commit_reads,
                self._replay_commit_steps,
                self._replay_record_metadata,
                self._replay_record_stamps,
                self._record_slots,
            )
            if tensor is not None
        )
        return payload_bytes + metadata_bytes

    def _kda_gate(
        self,
        g_raw: torch.Tensor | None,
        f_a_out: torch.Tensor | None,
        f_b_weight: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Resolve the raw per-channel gate for scans and record verification.

        Use the model's precomputed gate when supplied; otherwise apply the
        second projection to the low-rank activation.

        Args:
            g_raw: Gate the model already materialized, when it did.
            f_a_out: Low-rank gate activation feeding the GEMV.
            f_b_weight: Second gate projection.

        Returns:
            The gate, or None when the model supplied neither form.
        """
        if g_raw is None and f_a_out is not None:
            return torch.nn.functional.linear(f_a_out, f_b_weight)
        else:
            return g_raw

    @override
    def tree_support(self) -> TreeSupport:
        blocker = "the fused KDA verify kernel follows a chain; no draft trees yet"
        return TreeSupport(verify_blocker=blocker, draft_blocker=blocker)

    @override
    def _verify(
        self,
        mixed_qkv: torch.Tensor,
        conv_weights: torch.Tensor,
        conv_comp: torch.Tensor,
        ssm_comp: torch.Tensor,
        state_in_blocks: torch.Tensor,
        *,
        layer_id: int,
        bias: torch.Tensor | None,
        g_raw: torch.Tensor | None,
        f_a_out: torch.Tensor | None,
        f_b_weight: torch.Tensor | None,
        beta_raw: torch.Tensor | None,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        batch_size: int,
        draft_token_num: int,
        value_dim: int,
        attn_tp_size: int,
        head_v_dim: int,
        lower_bound: float | None,
        output_norm: KdaOutputNorm | None,
    ) -> torch.Tensor:
        """Write private candidates; acceptance publishes the selected prefix."""
        qkv = self._retained_qkv[self._descriptor_row_by_layer[layer_id]]
        rows = batch_size * draft_token_num
        heads = value_dim // attn_tp_size // head_v_dim
        # Preserve main's small-batch fused producers within the retained path.
        split = self._verify_split_producers and not (
            g_raw is None
            and bias is None
            and kda_fused_paged_verify_fuses_producers(
                self.dtype,
                recurrent_layout=self.kda_recurrent_layout,
                num_heads=heads,
                head_dim=head_v_dim,
                draft_token_num=draft_token_num,
                batch_size=batch_size,
            )
        )
        split_producers = {}
        if split:
            fork = self._verify_producer_forks[layer_id]
            with fork.scope(enable=True) as producer_fork:
                with producer_fork.branch():
                    g_raw = self._kda_gate(g_raw, f_a_out, f_b_weight)[:rows]
                conv_qkv = kda_verify_conv_update(
                    mixed_qkv[:rows],
                    conv_weights,
                    conv_comp,
                    state_in_blocks[:batch_size],
                    num_heads=heads,
                    head_dim=head_v_dim,
                    draft_token_num=draft_token_num,
                    recurrent_layout=self.kda_recurrent_layout,
                    bias=bias,
                    original_qkv=qkv[:rows, : mixed_qkv.shape[-1]],
                    query_start_loc=self.forward_metadata.query_start_loc[
                        : batch_size + 1
                    ],
                )
            # Eager allocations on the producer stream stay live through
            # the consumer's verify. Graph capture owns its allocations.
            if not torch.cuda.is_current_stream_capturing():
                consumer_stream = torch.cuda.current_stream()
                g_raw.record_stream(consumer_stream)
            split_producers = {"g_raw": g_raw, "conv_qkv": conv_qkv}
        else:
            if bias is not None:
                raise ValueError("fused KDA convolution does not support a bias")
            # The registered kernel owns an absent gate's projection.
            split_producers = {"g_raw": None if g_raw is None else g_raw[:rows]}
        # FP32 records belong to request slots. Generation stamps index batch rows.
        assert self._replay_record_metadata is not None
        assert self._replay_record_stamps is not None
        live_indices = self.forward_metadata.state_out_blocks_by_group[
            self._state_group_for(layer_id)
        ]
        retained_records = KdaRetainedRecords(
            history=self._retained_history(layer_id, live_indices),
            original_qkv=None if split else qkv[:rows, : mixed_qkv.shape[-1]],
            metadata=self._replay_record_metadata[:batch_size],
            stamps=self._replay_record_stamps[self._descriptor_row_by_layer[layer_id]],
            query_start_loc=self.forward_metadata.query_start_loc[: batch_size + 1],
        )
        num_value_heads = value_dim // attn_tp_size // head_v_dim
        fused_out = kda_fused_paged_verify(
            mixed_qkv,
            conv_weights,
            conv_comp,
            f_a_out,
            f_b_weight,
            beta_raw,
            A_log,
            dt_bias,
            state_pool=ssm_comp,
            read_indices=state_in_blocks[:batch_size],
            num_heads=num_value_heads,
            head_dim=head_v_dim,
            draft_token_num=draft_token_num,
            lower_bound=lower_bound,
            recurrent_layout=self.kda_recurrent_layout,
            retained_records=retained_records,
            output_norm=output_norm,
            override=None,
            solution=None,
            **split_producers,
        )
        return fused_out

    @override
    def commit_verified_state(
        self, accepted_length: torch.Tensor, *, accepted_path: torch.Tensor | None
    ) -> None:
        """Replay and eagerly commit this round's accepted KDA prefix (a chain:
        KDA refuses draft trees)."""
        ctx = self._verify_commit_ctx
        if ctx is None:
            return

        _, draft_token_num, _ = ctx
        bs = accepted_length.shape[0]
        group_ids = list(self._replay_group_ids or self._state_groups())
        write_stack, steps = self._resolve_verify_commit_pages(
            accepted_length, group_ids
        )
        metadata = (
            self._mixed_forward_metadata.decode
            if self._mixed_forward_metadata is not None
            else self.forward_metadata
        )
        # The decode rows follow a mixed forward's extend rows in request_ends.
        first_row = (
            self._mixed_forward_metadata.num_extends
            if self._mixed_forward_metadata is not None
            else 0
        )
        first_layer = self._state_layer_ids()[0]
        conv, state = self._state_components(first_layer)
        if self._request_stopped is None or self._request_ends is None:
            raise RuntimeError(
                "the KDA decode commit needs the request stop flags and ends "
                "(bind_request_stop_flags)"
            )
        correction, key, decay = self._retained_records.unbind(2)
        # The planner clamps each active interval to at least one accepted
        # token. Untouched layers therefore fail the generation-stamp check
        # and leave their cache fields unchanged, including subset forwards.
        # A request that ended in this forward has its records folded into
        # its live block, which its finish publishes; if a layer rejects its
        # commit, its row in request_ends says the block is incomplete.
        self._batched_replay_kernel(
            descriptors=self._replay_descriptors,
            group_indices=self._replay_group_indices,
            read_indices=self._replay_commit_reads[: len(group_ids) * bs].view(
                len(group_ids), bs
            ),
            write_indices=write_stack,
            accepted_length=steps,
            query_start_loc=metadata.query_start_loc[: bs + 1],
            record_slots=self._record_slots[:bs],
            record_slot_count=self._record_slot_count,
            record_metadata=self._replay_record_metadata[:bs],
            record_stamps=self._replay_record_stamps,
            retained_correction=correction,
            retained_key=key,
            retained_decay=decay,
            request_stopped=self._request_stopped,
            request_ends=self._request_ends[first_row : first_row + bs],
            draft_token_num=draft_token_num,
            history_capacity=self._retained_capacity,
            num_heads=state.shape[1],
            head_dim=state.shape[2],
            conv_width=conv.shape[-1] + 1,
            activation_dtype=self.dtype,
            conv_state_dtype=conv.dtype,
            recurrent_state_dtype=state.dtype,
        )
        self._verify_commit_ctx = None
        return

    @override
    def _prefill_scan(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        recurrent_state: torch.Tensor,
        query_start_loc: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        D: torch.Tensor | None,
        a: torch.Tensor | None,
        b: torch.Tensor | None,
        g_raw: torch.Tensor | None,
        f_a_out: torch.Tensor | None,
        f_b_weight: torch.Tensor | None,
        beta_raw: torch.Tensor | None,
        seq_len: int,
        num_real_tokens: int,
        lower_bound: float | None,
        cu_seqlens_cpu: torch.Tensor | None = None,
        state_checkpoints: GdnStateCheckpoints | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run an exact-length or fixed-capacity KDA scan with live boundaries.

        Ordinary metadata slices inputs to the real-token prefix. Capacity
        metadata retains the physical packed extent; GPU boundaries determine
        live work, so that extent is not a live-token count. The scan masks
        loads past the live boundaries; inline output padding is cleared by the
        in-graph scrub. Only an ordinary break uses the handoff copy. A
        prefill graph runs a checkpointing forward on ordinary metadata at its
        eager breaks, so the output is zero-padded to ``seq_len`` rows there.

        ``cu_seqlens_cpu`` is the required metadata-built int64 mirror of
        ``query_start_loc``. It supplies exact-length host planning and
        capacity admission without a per-layer synchronizing D2H. The prepared
        capacity path builds its chunk plan on device instead of on the host.

        ``state_checkpoints`` is the forward's checkpoint plan. The KDA scan
        kernels store no interior state, so ``_checkpoint_prefix_states``
        computes the plan rows the forward publishes; its other rows stay
        unwritten.
        """
        _reject_skip_term(D)
        head_k_dim = query.shape[3]
        num_value_heads = value.shape[2]

        g_kda = self._kda_gate(g_raw, f_a_out, f_b_weight).view(
            1, seq_len, num_value_heads, head_k_dim
        )

        beta_kda = beta_raw.view(1, seq_len, num_value_heads)

        query, key, value, g_kda, beta_kda = _slice_kda_prefill_inputs(
            num_real_tokens, query, key, value, g_kda, beta_kda
        )

        if cu_seqlens_cpu is None:
            raise RuntimeError(
                "KDA prefill scan requires the metadata-built cu_seqlens_cpu "
                "hint; init_forward_metadata constructs it for every extend "
                "batch"
            )

        if query_start_loc.dtype != torch.int64:
            raise RuntimeError("KDA prefill requires metadata-built int64 boundaries")
        if state_checkpoints is not None:
            state_checkpoints.states.index_copy_(
                0,
                self.forward_metadata.scan_checkpoint_rows,
                self._checkpoint_prefix_states(
                    (query, key, value, g_kda, beta_kda),
                    recurrent_state,
                    cu_seqlens_cpu,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    lower_bound=lower_bound,
                ),
            )
        kda_result = kda_paged_prefill(
            query,
            key,
            value,
            g_kda,
            beta_kda,
            A_log,
            dt_bias,
            initial_state=recurrent_state,
            cu_seqlens=query_start_loc,
            cu_seqlens_cpu=cu_seqlens_cpu,
            capacity=(
                KdaPrefillCapacity(seq_len, query_start_loc.numel() - 1)
                if isinstance(self.forward_metadata, CapacityPrefillMetadata)
                else None
            ),
            lower_bound=lower_bound,
            solution=None if self.kda_backend == "auto" else self.kda_backend,
            recurrent_layout=self.kda_recurrent_layout,
        )

        out = kda_result.out.squeeze(0)
        if out.shape[0] < seq_len:
            # A live scan at a padded graph bucket's eager break; zero the rest.
            out = torch.nn.functional.pad(out, (0, 0, 0, 0, 0, seq_len - out.shape[0]))
        return out, kda_result.final_state

    def _checkpoint_prefix_states(
        self,
        inputs: tuple[torch.Tensor, ...],
        recurrent_state: torch.Tensor,
        cu_seqlens_cpu: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        lower_bound: float | None,
    ) -> torch.Tensor:
        """Return the FP32 state at each of this forward's in-forward checkpoints.

        Each is the final state of an exact-length scan over its row's tokens
        before the checkpoint, from the row's input state: what a forward
        ending at the checkpoint would leave in the live slot.

        Args:
            inputs: Packed ``(query, key, value, gate, beta)`` of the forward.
            recurrent_state: The forward's per-row input states.
            cu_seqlens_cpu: Host int64 boundaries of the forward's rows.
            A_log: FP32 gate parameter.
            dt_bias: FP32 gate bias.
            lower_bound: KDA decay clamp.

        Returns:
            One state per checkpoint, in checkpoint order.
        """
        checkpoints = self.forward_metadata.state_checkpoints
        starts = cu_seqlens_cpu[checkpoints.rows]
        spans = list(zip(starts.tolist(), (starts + checkpoints.offsets).tolist()))
        prefix_cu_seqlens_cpu = torch.cat(
            (torch.zeros(1, dtype=torch.int64), checkpoints.offsets.cumsum(0))
        )
        # Keep the forward thread ahead of the GPU: pageable copies would sync.
        rows, prefix_cu_seqlens = (
            tensor.pin_memory().to(recurrent_state.device, non_blocking=True)
            for tensor in (checkpoints.rows, prefix_cu_seqlens_cpu)
        )
        result = kda_paged_prefill(
            *(_pack_token_spans(tensor, spans) for tensor in inputs),
            A_log,
            dt_bias,
            initial_state=recurrent_state[rows],
            cu_seqlens=prefix_cu_seqlens,
            cu_seqlens_cpu=prefix_cu_seqlens_cpu,
            capacity=None,
            lower_bound=lower_bound,
            solution=None if self.kda_backend == "auto" else self.kda_backend,
            recurrent_layout=self.kda_recurrent_layout,
        )
        return result.final_state

    @override
    def _prepare_prefill_scan_query_start_loc(
        self, query_start_loc: torch.Tensor
    ) -> torch.Tensor:
        """Prepare the reusable int64 boundaries of the prefill scan."""
        return query_start_loc.to(torch.int64)
