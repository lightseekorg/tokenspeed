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

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.attention.gdn import GdnStateCheckpoints
from tokenspeed_kernel.ops.attention.kda import (
    KdaPrefillCapacity,
    KdaReplayRecords,
    kda_fused_paged_verify_uses_split_producers,
    kda_paged_decode,
    kda_paged_prefill,
)
from tokenspeed_kernel.ops.attention.kda import (
    kda_recurrent_layout as kda_recurrent_layout_default,
)
from tokenspeed_kernel.ops.attention.kda import (
    kda_replay_commit_supported,
    kda_replay_payload_specs,
    kda_verify_conv_update,
    resolve_kda_batched_replay_commit,
    try_kda_fused_paged_decode,
    try_kda_fused_paged_verify,
)
from tokenspeed_kernel.platform import current_platform
from typing_extensions import override

from tokenspeed.runtime.layers.attention.backends.state.mamba import (
    _reject_skip_term,
    logger,
)
from tokenspeed.runtime.layers.attention.backends.state.prefill_capacity import (
    CapacityPrefillBackend,
    CapacityPrefillMetadata,
)
from tokenspeed.runtime.layers.attention.backends.support import TreeSupport
from tokenspeed.runtime.utils.cuda_stream import StreamFork

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.configs.base import (
        AttnConfig,
        SoftmaxAttnConfig,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool


KDA_PREFILL_BACKENDS = ("auto", "fla", "flashkda", "cutedsl_kda")


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


class KdaAttnBackend(CapacityPrefillBackend):
    """Attention backend for KDA linear attention layers (Kimi-K3).

    Everything generic to linear attention -- state paging and cache groups --
    is inherited. When replay commit is available, decode and speculative
    verify capture the same recovery payload without a per-position state
    tape. Commit recovers the accepted prefix from the committed page;
    standard decode is a one-token window with one accepted token.
    KDA also replaces the scan seams: its decay gate is
    per-channel (a low-rank ``f_a``/``f_b`` projection plus raw beta logits)
    where GDN's is scalar per head, and its decode/verify kernels can fuse
    the conv, the gate GEMV and the recurrence into a single launch.
    """

    _verify_reads_committed_recurrent_state = True
    _verify_packed_qkv_views = True
    # Eager forwards keep the same metadata contract as replay.
    _capacity_layout_when_uncaptured = True

    def __init__(
        self,
        config: AttnConfig,
        spec: SoftmaxAttnConfig,
        *,
        enable_prefill_graph: bool,
        kda_backend: str = "auto",
    ) -> None:
        super().__init__(config, spec)
        self._prefill_graph_enabled = enable_prefill_graph
        self.max_bs = config.max_bs
        # The platform layout; the workspace planner probes the same one.
        self.kda_recurrent_layout = kda_recurrent_layout_default()
        if self.kda_recurrent_layout not in ("k_major", "v_major"):
            raise ValueError(
                f"unsupported KDA recurrent layout {self.kda_recurrent_layout!r}"
            )
        self._replay_active = False
        self._batched_replay_kernel = None
        self._verify_split_producers = False
        self._reset_replay_state()
        self.kda_backend = (kda_backend or "auto").strip().lower()
        if self.kda_backend not in KDA_PREFILL_BACKENDS:
            raise ValueError(
                f"--kda-backend must be one of {', '.join(KDA_PREFILL_BACKENDS)}; "
                f"got {self.kda_backend!r}"
            )
        logger.info(
            f"KDA prefill routes through {self.kda_backend!s}; decode remains on the "
            "platform-selected kernels",
        )

    def _admits_capacity_prefill(self) -> bool:
        return self._prefill_graph_enabled and self.kda_backend == "cutedsl_kda"

    def _reset_replay_state(self) -> None:
        self._verify_producer_stream: torch.cuda.Stream | None = None
        self._verify_producer_forks: dict[int, StreamFork] = {}
        self._replay_payloads: tuple[torch.Tensor, ...] | None = None
        self._replay_weights: dict[
            int,
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                int,
                int,
                float | None,
            ],
        ] = {}
        self._replay_descriptors: torch.Tensor | None = None
        self._replay_group_indices: torch.Tensor | None = None
        self._batched_replay_launch: (
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None] | None
        ) = None
        self._batched_replay_ready = False
        self._replay_descriptor_bound: set[int] = set()
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
        return not self.is_draft and (self._replay_active or self.spec_num_tokens > 1)

    @staticmethod
    def _replay_shape(kv_pool: CachePool) -> dict[str, int]:
        first_layer = min(kv_pool.state_group_by_layer)
        num_heads, head_dim = kv_pool.get_component(
            first_layer, "recurrent_state"
        ).shape[1:3]
        return {"num_heads": num_heads, "head_dim": head_dim}

    @staticmethod
    def _layer_pools_are_uniform(kv_pool: CachePool) -> bool:
        geometries = {
            tuple(
                (tuple(component.shape[1:]), component.stride(0))
                for component in (
                    kv_pool.get_component(layer_id, "conv_state"),
                    kv_pool.get_component(layer_id, "recurrent_state"),
                )
            )
            for layer_id in kv_pool.state_group_by_layer
        }
        return len(geometries) == 1

    @override
    def validate_cache_pool(self, cache_pool: CachePool) -> None:
        super().validate_cache_pool(cache_pool)
        replay_active = kda_replay_commit_supported(
            self.dtype,
            recurrent_layout=self.kda_recurrent_layout,
            **self._replay_shape(cache_pool),
        )
        if (
            replay_active
            and not self.is_draft
            and not self._layer_pools_are_uniform(cache_pool)
        ):
            raise RuntimeError("batched KDA replay requires uniform layer pools")

    @override
    def _publish_cache_pool(self, cache_pool: CachePool) -> None:
        super()._publish_cache_pool(cache_pool)
        self._reset_replay_state()
        shape = self._replay_shape(cache_pool)
        self._replay_active = kda_replay_commit_supported(
            self.dtype,
            recurrent_layout=self.kda_recurrent_layout,
            **shape,
        )
        self._batched_replay_kernel = (
            resolve_kda_batched_replay_commit(self.dtype, **shape)
            if self._replay_active
            else None
        )
        self._verify_split_producers = self._replay_active and (
            kda_fused_paged_verify_uses_split_producers(
                self.dtype,
                store_states=False,
                recurrent_layout=self.kda_recurrent_layout,
                **shape,
            )
        )

    def _allocate_replay_workspace(self) -> None:
        """Allocate max_bs * T payload rows before eager or graph execution.

        Batch refreshes reuse these buffers; binding a pool resets them.
        """
        if self._replay_payloads is not None:
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
                (len(layer_ids), 10), dtype=torch.uint64, device=self.device
            )
            group_indices = torch.tensor(
                [
                    self._replay_group_rows[self._state_group_for(layer_id)]
                    for layer_id in layer_ids
                ],
                dtype=torch.int32,
                device=self.device,
            )
            first_conv, first_ssm = self._state_components(layer_ids[0])
            if self._verify_split_producers and first_conv.is_cuda:
                self._verify_producer_stream = torch.cuda.Stream(
                    device=first_conv.device, priority=-1
                )
                self._verify_producer_forks = {
                    layer_id: StreamFork(self._verify_producer_stream)
                    for layer_id in layer_ids
                }
            hv, head_dim = first_ssm.shape[1:3]
            payload_shape = (len(layer_ids), rows)
            self._replay_payloads = tuple(
                torch.empty(
                    (*payload_shape, channels),
                    dtype=payload_dtype,
                    device=self.device,
                )
                for channels, payload_dtype in kda_replay_payload_specs(
                    self.dtype,
                    conv_channels=first_conv.shape[1],
                    num_heads=hv,
                    head_dim=head_dim,
                )
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

    @override
    def _verify_state_blocks(
        self,
        bs: int,
        block_tables: Mapping[str, torch.Tensor],
        *,
        pages_out: Mapping[str, torch.Tensor],
        live_pages_out: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        # The scheduler names input and live pages in fixed table slots.
        tables = super()._verify_state_blocks(
            bs, block_tables, pages_out=pages_out, live_pages_out=live_pages_out
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
        self, bs: int, seq_lens: torch.Tensor, draft_token_num: int
    ) -> None:
        if self._replay_record_metadata is not None and seq_lens.is_cuda:
            from tokenspeed_kernel.ops.attention.kda.vllm_recoverssm import (
                prepare_recoverssm_metadata,
            )

            # Refresh the live rows before verify writes candidate records.
            # A new generation prevents recovery from consuming an older
            # round's records when the same workspace slots are reused.
            self._replay_record_generation += 1
            prepare_recoverssm_metadata(
                seq_lens[:bs].to(torch.int64) - draft_token_num,
                self._replay_record_metadata,
                batch=bs,
                generation=self._replay_record_generation,
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

    def _replay_payload(self, layer_id: int) -> tuple[torch.Tensor, ...]:
        """Return one layer's views in the shared descriptor order.

        Records hold original QKV, FP32 U, activation-dtype K/raw gate and
        an empty reserved buffer. All record consumers use this tuple order.
        """
        assert self._replay_payloads is not None
        row = self._descriptor_row_by_layer[layer_id]
        return tuple(payload[row] for payload in self._replay_payloads)

    def _bind_replay_descriptor(self, layer_id: int, weights: tuple) -> None:
        """Publish one layer's stable pointers after its weights are observed."""
        if layer_id in self._replay_descriptor_bound:
            return
        assert self._replay_descriptors is not None
        conv_w, A_log, dt_bias, _num_heads, _head_dim, lower_bound = weights
        if dt_bias is None:
            return
        conv, state = self._state_components(layer_id)
        qkv, f_a, beta, gate = self._replay_payload(layer_id)
        row = self._descriptor_row_by_layer[layer_id]
        addresses = self._replay_descriptors
        addresses[row].copy_(
            torch.tensor(
                [
                    qkv.data_ptr(),
                    conv_w.data_ptr(),
                    conv.data_ptr(),
                    f_a.data_ptr(),
                    0,
                    beta.data_ptr(),
                    A_log.data_ptr(),
                    dt_bias.data_ptr(),
                    state.data_ptr(),
                    gate.data_ptr(),
                ],
                dtype=torch.uint64,
                device=self.device,
            )
        )
        self._replay_descriptor_bound.add(layer_id)
        if len(self._replay_descriptor_bound) == len(self._descriptor_row_by_layer):
            first = next(iter(self._replay_weights.values()))
            first_conv_w, _, _, first_heads, first_dim, _ = first
            geometry = (
                first_heads,
                first_dim,
                first_conv_w.shape[1],
            )
            first_layer = next(iter(self._replay_weights))
            first_conv, first_state = self._state_components(first_layer)
            first_qkv, first_fa, first_beta, first_gate = self._replay_payload(
                first_layer
            )
            strides = (
                first_qkv.stride(0),
                first_conv.stride(0),
                first_fa.stride(0),
                first_beta.stride(0),
                first_state.stride(0),
                first_gate.stride(0),
            )
            lower_bounds = set()
            for current_layer, layer_weights in self._replay_weights.items():
                layer_conv_w, _, _, layer_heads, layer_dim, _ = layer_weights
                if (
                    layer_heads,
                    layer_dim,
                    layer_conv_w.shape[1],
                ) != geometry:
                    raise RuntimeError("batched KDA replay requires uniform geometry")
                layer_conv, layer_state = self._state_components(current_layer)
                layer_qkv, layer_fa, layer_beta, layer_gate = self._replay_payload(
                    current_layer
                )
                if (
                    layer_qkv.stride(0),
                    layer_conv.stride(0),
                    layer_fa.stride(0),
                    layer_beta.stride(0),
                    layer_state.stride(0),
                    layer_gate.stride(0),
                ) != strides:
                    raise RuntimeError("batched KDA replay requires uniform strides")
                lower_bounds.add(layer_weights[-1])
            if len(lower_bounds) != 1:
                raise RuntimeError("batched KDA replay requires one lower bound")
            conv_width = geometry[2]
            if self._batched_replay_kernel is not None:
                descriptors = self._replay_descriptors
                group_indices = self._replay_group_indices
                assert group_indices is not None
                draft_tokens = self.speculative_num_draft_tokens

                def launch(read_indices, write_indices, accepted_length):
                    assert self._replay_record_metadata is not None
                    assert self._replay_record_stamps is not None
                    record_args = {
                        "correction_stride": strides[2],
                        "key_gate_stride": strides[3],
                        "activation_dtype": first_beta.dtype,
                        "conv_state_dtype": first_conv.dtype,
                        "record_metadata": self._replay_record_metadata[
                            : accepted_length.shape[0]
                        ],
                        "record_stamps": self._replay_record_stamps,
                    }
                    self._batched_replay_kernel(
                        descriptors=descriptors,
                        group_indices=group_indices,
                        read_indices=read_indices,
                        write_indices=write_indices,
                        accepted_length=accepted_length,
                        draft_token_num=draft_tokens,
                        num_heads=geometry[0],
                        head_dim=geometry[1],
                        qkv_stride=strides[0],
                        conv_stride=strides[1],
                        state_stride=strides[4],
                        conv_width=conv_width,
                        lower_bound=next(iter(lower_bounds)),
                        **record_args,
                    )

                self._batched_replay_launch = launch
                self._batched_replay_ready = True

    @override
    def _ensure_verify_scratch(self, bs: int, draft_token_num: int) -> None:
        if not self._replay_active:
            return super()._ensure_verify_scratch(bs, draft_token_num)
        self._allocate_replay_workspace()
        rows = max(len(self.query_start_loc_list), bs)
        scratch = self._verify_scratch
        if scratch is not None:
            # Allocated once at its maximum; graphs may hold its addresses, so
            # an overrun is an invariant violation, never a resize.
            capacity = next(iter(scratch.values()))[0].shape[0]
            if capacity < rows:
                raise RuntimeError(
                    f"KDA verify needs {rows} transient conv rows but the "
                    f"preallocated scratch holds {capacity}"
                )
            return
        scratch = {}
        for layer_id in self._state_layer_ids():
            conv, _ = self._state_components(layer_id)
            scratch[layer_id] = (
                torch.empty(
                    (rows, *conv.shape[1:]), dtype=conv.dtype, device=conv.device
                ),
                None,
            )
        self._verify_scratch = scratch

    @override
    def preallocate_verify_workspace(self, max_bs: int, draft_token_num: int) -> int:
        if not self._replay_active:
            return super().preallocate_verify_workspace(max_bs, draft_token_num)
        self._ensure_verify_scratch(max_bs, draft_token_num)
        conv_bytes = sum(pair[0].nbytes for pair in self._verify_scratch.values())
        payload_bytes = sum(payload.nbytes for payload in (self._replay_payloads or ()))
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
            )
            if tensor is not None
        )
        return conv_bytes + payload_bytes + metadata_bytes

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
    def _decode(
        self,
        mixed_qkv: torch.Tensor,
        conv_weights: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        read_indices: torch.Tensor,
        write_indices: torch.Tensor,
        *,
        f_a_out: torch.Tensor | None,
        f_b_weight: torch.Tensor | None,
        beta_raw: torch.Tensor | None,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        value_dim: int,
        attn_tp_size: int,
        head_v_dim: int,
        lower_bound: float | None,
    ) -> torch.Tensor | None:
        if f_a_out is None:
            return None

        num_value_heads = value_dim // attn_tp_size // head_v_dim
        return try_kda_fused_paged_decode(
            mixed_qkv,
            conv_weights,
            conv_states,
            f_a_out,
            f_b_weight,
            beta_raw,
            A_log,
            dt_bias,
            state_pool=ssm_states,
            read_indices=read_indices,
            write_indices=write_indices,
            num_heads=num_value_heads,
            head_dim=head_v_dim,
            cu_seqlens=self.forward_metadata.query_start_loc,
            lower_bound=lower_bound,
            recurrent_layout=self.kda_recurrent_layout,
        )

    @override
    def _decode_scan(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        ssm_states: torch.Tensor,
        read_indices: torch.Tensor,
        write_indices: torch.Tensor,
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
        lower_bound: float | None,
    ) -> torch.Tensor:
        _reject_skip_term(D)
        seq_len = query.shape[0]
        num_heads = query.shape[2]
        head_k_dim = query.shape[3]
        num_value_heads = value.shape[2]
        head_v_dim = value.shape[3]

        query_start_loc = self.forward_metadata.query_start_loc
        g_raw = self._kda_gate(g_raw, f_a_out, f_b_weight)
        query = query.view(1, seq_len, num_heads, head_k_dim)
        key = key.view(1, seq_len, num_heads, head_k_dim)
        value = value.view(1, seq_len, num_value_heads, head_v_dim)
        g_kda = g_raw.view(1, seq_len, num_value_heads, head_k_dim)
        beta_kda = beta_raw.view(1, seq_len, num_value_heads)

        core_attn_out = kda_paged_decode(
            query,
            key,
            value,
            g_kda,
            beta_kda,
            A_log,
            dt_bias,
            state_pool=ssm_states,
            read_indices=read_indices,
            write_indices=write_indices,
            cu_seqlens=query_start_loc,
            lower_bound=lower_bound,
            recurrent_layout=self.kda_recurrent_layout,
        )
        return core_attn_out.squeeze(0)

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
        conv_scratch: torch.Tensor,
        ssm_comp: torch.Tensor,
        ssm_scratch: torch.Tensor,
        state_in_blocks: torch.Tensor,
        output_indices: torch.Tensor,
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
    ) -> torch.Tensor | None:
        """Keep live state read-only; acceptance commits the selected prefix."""
        if self._replay_active:
            qkv, f_a, beta, gate = self._replay_payload(layer_id)
            rows = batch_size * draft_token_num
            heads = value_dim // attn_tp_size // head_v_dim
            split_producers = {}
            if self._verify_split_producers:
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
            # Records hold FP32 corrections and activation-dtype K/raw gates.
            # Request rows, not pool block IDs, size the per-round cache.
            assert self._replay_record_metadata is not None
            assert self._replay_record_stamps is not None
            replay_records = KdaReplayRecords(
                original_qkv=(
                    None
                    if self._verify_split_producers
                    else qkv[:rows, : mixed_qkv.shape[-1]]
                ),
                metadata=self._replay_record_metadata[:batch_size],
                stamps=self._replay_record_stamps[
                    self._descriptor_row_by_layer[layer_id], :batch_size
                ],
                correction=f_a[:rows, : heads * head_v_dim]
                .view(batch_size, draft_token_num, heads, head_v_dim)
                .transpose(1, 2),
                key_gate=beta[:rows, : 2 * heads * head_v_dim]
                .view(batch_size, draft_token_num, heads, 2 * head_v_dim)
                .transpose(1, 2),
                query_start_loc=self.forward_metadata.query_start_loc[: batch_size + 1],
            )
            num_value_heads = value_dim // attn_tp_size // head_v_dim
            if layer_id not in self._replay_weights:
                # Parameters are stable objects; model weight updates copy into
                # their storage, so binding their pointers once cannot stale.
                self._replay_weights[layer_id] = (
                    conv_weights,
                    A_log,
                    dt_bias,
                    num_value_heads,
                    head_v_dim,
                    lower_bound,
                )
                self._bind_replay_descriptor(layer_id, self._replay_weights[layer_id])
            fused_out = try_kda_fused_paged_verify(
                mixed_qkv,
                conv_weights,
                conv_comp,
                conv_scratch,
                f_a_out,
                f_b_weight,
                beta_raw,
                A_log,
                dt_bias,
                state_pool=ssm_comp,
                state_scratch=ssm_scratch,
                read_indices=state_in_blocks[:batch_size],
                write_indices=output_indices[:batch_size],
                num_heads=num_value_heads,
                head_dim=head_v_dim,
                draft_token_num=draft_token_num,
                lower_bound=lower_bound,
                store_states=False,
                recurrent_layout=self.kda_recurrent_layout,
                replay_records=replay_records,
                **split_producers,
            )
            if fused_out is None:
                raise RuntimeError(
                    "KDA fused paged verify kernel vanished after the replay "
                    "capability probe reported it available"
                )
            return fused_out
        if f_a_out is None or bias is not None:
            return None
        else:
            num_value_heads = value_dim // attn_tp_size // head_v_dim
            return try_kda_fused_paged_verify(
                mixed_qkv,
                conv_weights,
                conv_comp,
                conv_scratch,
                f_a_out,
                f_b_weight,
                beta_raw,
                A_log,
                dt_bias,
                state_pool=ssm_comp,
                state_scratch=ssm_scratch,
                read_indices=state_in_blocks[:batch_size],
                write_indices=output_indices[:batch_size],
                num_heads=num_value_heads,
                head_dim=head_v_dim,
                draft_token_num=draft_token_num,
                lower_bound=lower_bound,
                recurrent_layout=self.kda_recurrent_layout,
                replay_records=None,
            )

    @override
    def _verify_scan(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        ssm_comp: torch.Tensor,
        ssm_scratch: torch.Tensor,
        state_in_blocks: torch.Tensor,
        output_indices: torch.Tensor,
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
        batch_size: int,
        draft_token_num: int,
        seq_len: int,
        lower_bound: float | None,
        replay_payload: torch.Tensor | None,
    ) -> torch.Tensor:

        _reject_skip_term(D)
        if replay_payload is not None:
            raise ValueError("the KDA verify scan fills no replay payload")
        from tokenspeed_kernel.ops.attention.kda.triton import (
            kda_recurrent_decode_mtp,
        )

        num_heads = query.shape[2]
        head_k_dim = query.shape[3]
        num_value_heads = value.shape[2]
        head_v_dim = value.shape[3]

        query_b = query.view(batch_size, draft_token_num, num_heads, head_k_dim)
        key_b = key.view(batch_size, draft_token_num, num_heads, head_k_dim)
        value_b = value.view(batch_size, draft_token_num, num_value_heads, head_v_dim)

        g_b = self._kda_gate(g_raw, f_a_out, f_b_weight).view(
            batch_size, draft_token_num, num_value_heads, head_k_dim
        )

        beta_b = beta_raw.view(batch_size, draft_token_num, num_value_heads)
        # The recurrence has independent read and write pools: committed pages
        # provide the initial state while every speculative checkpoint lands in scratch.
        initial_pool = ssm_comp
        initial_rows = state_in_blocks[:batch_size]
        write_rows = output_indices[:batch_size]
        state_out = ssm_scratch

        return kda_recurrent_decode_mtp(
            query_b,
            key_b,
            value_b,
            g_b,
            beta_b,
            A_log,
            dt_bias,
            initial_pool,
            initial_rows,
            write_rows,
            h_pool_out=state_out,
            lower_bound=lower_bound,
            recurrent_layout=self.kda_recurrent_layout,
        ).reshape(1, seq_len, num_value_heads, head_v_dim)

    @override
    def commit_verified_state(
        self, accepted_length: torch.Tensor, *, accepted_path: torch.Tensor | None
    ) -> None:
        """Replay and eagerly commit this round's accepted KDA prefix (a chain:
        KDA refuses draft trees)."""
        if not self._replay_active:
            return super().commit_verified_state(
                accepted_length, accepted_path=accepted_path
            )
        ctx = self._verify_commit_ctx
        if ctx is None:
            return

        _, draft_token_num, read_pages_by_group = ctx
        bs = accepted_length.shape[0]
        group_ids = list(self._replay_group_ids or self._state_groups())
        write_stack, steps = self._resolve_verify_commit_pages(
            accepted_length, group_ids
        )
        if self._batched_replay_ready:
            self._batched_replay_launch(
                (
                    self._replay_commit_reads[: len(group_ids) * bs].view(
                        len(group_ids), bs
                    )
                    if accepted_length.is_cuda and self._replay_commit_reads is not None
                    else torch.stack(
                        [
                            read_pages_by_group[group_id][:bs]
                            for group_id in self._replay_group_ids
                        ]
                    ).to(torch.int32)
                ),
                write_stack,
                steps,
            )
            self._verify_commit_ctx = None
            return
        # A layer subset uses the same record recovery kernel. Full model
        # forwards bind all layers and use the single batched launch above.
        for layer_id, weights in self._replay_weights.items():
            conv_w, _, _, heads, dim, lower_bound = weights
            row = self._descriptor_row_by_layer[layer_id]
            conv, state = self._state_components(layer_id)
            qkv, correction, key_gate, _ = self._replay_payload(layer_id)
            self._batched_replay_kernel(
                descriptors=self._replay_descriptors[row : row + 1],
                group_indices=self._replay_group_indices[row : row + 1],
                read_indices=self._replay_commit_reads[: len(group_ids) * bs].view(
                    len(group_ids), bs
                ),
                write_indices=write_stack,
                accepted_length=steps,
                draft_token_num=draft_token_num,
                num_heads=heads,
                head_dim=dim,
                qkv_stride=qkv.stride(0),
                conv_stride=conv.stride(0),
                correction_stride=correction.stride(0),
                key_gate_stride=key_gate.stride(0),
                state_stride=state.stride(0),
                conv_width=conv_w.shape[1],
                lower_bound=lower_bound,
                activation_dtype=key_gate.dtype,
                conv_state_dtype=conv.dtype,
                record_metadata=self._replay_record_metadata[:bs],
                record_stamps=self._replay_record_stamps[row : row + 1],
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
        live work, so that extent is not a live-token count. The adapter clears
        inputs before full-tile loads; inline output padding is cleared by the
        in-graph scrub. Only an ordinary break uses the handoff copy.

        ``cu_seqlens_cpu`` is the required metadata-built int64 mirror of
        ``query_start_loc``. It supplies exact-length host planning and
        capacity admission without a per-layer synchronizing D2H. The prepared
        capacity path builds its chunk plan on device instead of on the host.
        """
        if state_checkpoints is not None:
            raise NotImplementedError(
                f"{type(self).__name__} stores no in-forward State checkpoints"
            )
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

        return kda_result.out.squeeze(0), kda_result.final_state

    @override
    def _prepare_prefill_scan_query_start_loc(
        self, query_start_loc: torch.Tensor
    ) -> torch.Tensor:
        """Prepare the reusable int64 boundaries of the prefill scan."""
        return query_start_loc.to(torch.int64)
