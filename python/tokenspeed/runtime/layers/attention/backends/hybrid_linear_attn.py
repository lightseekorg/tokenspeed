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

"""Hybrid linear attention backend for Qwen3.5 GDN models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.attention import (
    gdn_chunk_prefill,
    gdn_decode_mtp,
    gdn_decode_step,
    gdn_replay_commit,
)
from tokenspeed_kernel.ops.attention.triton.gdn_qkv_split import (
    fused_qkv_split_gdn_prefill,
)
from tokenspeed_kernel.ops.attention.triton.linear.index import (
    set_total_chunks_hint,
    set_total_chunks_hint_uniform,
)

from tokenspeed.runtime.execution.breakable_cuda_graph import (
    break_point,
    current_forward_ctx,
    scrub_padding_tail,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import (
    AttentionBackend,
    init_backend_cuda_graph_state,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    cache_debug_enabled,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    LINEAR_ATTENTION,
)
from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from tokenspeed.runtime.layers.attention.linear.gdn import fused_gdn_gating

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.configs.base import BaseAttnConfig
    from tokenspeed.runtime.layers.paged_attention import PagedAttention

# Default cache group id carrying GDN/Mamba state pages.
_STATE_GROUP_ID = LINEAR_ATTENTION


def _mask_fresh_initial_state(
    recurrent_state: torch.Tensor,
    has_initial_states: torch.Tensor | None,
) -> torch.Tensor:
    """Zero the initial recurrent state of sequences that have no history.

    On the paged cache path a fresh sequence's ``state_in`` page is freshly
    allocated from the shared BlockPool and may carry a previous tenant's
    bytes (the slabs alias every cache group, so those bytes can be fp8 MLA
    latents that reinterpret as huge/NaN fp32) — the recurrent kernels have
    no per-sequence has_initial_state gate, so mask here.

    Args:
        recurrent_state: ``[B, ...]`` gathered per-sequence initial states.
        has_initial_states: ``[B]`` bool, True where the sequence resumes
            real history (prefix hit / later chunk). None means every
            sequence is fresh.

    Returns:
        The masked initial state (zeros for fresh sequences).
    """
    if has_initial_states is None:
        return torch.zeros_like(recurrent_state)
    mask = has_initial_states.to(recurrent_state.device, torch.bool)
    mask = mask.view(-1, *([1] * (recurrent_state.dim() - 1)))
    return torch.where(mask, recurrent_state, torch.zeros_like(recurrent_state))


@dataclass(frozen=True)
class _StateBlockIndexPlan:
    checkpoint_granularity: int
    before: torch.Tensor
    after: torch.Tensor
    has_history: torch.Tensor
    in_slots: torch.Tensor
    out_slots: torch.Tensor


def _compute_state_block_index_plan(
    checkpoint_granularity: int,
    seq_lens_before: torch.Tensor,
    seq_lens_after: torch.Tensor,
) -> _StateBlockIndexPlan:
    before = seq_lens_before.to(torch.int64)
    after = seq_lens_after.to(torch.int64)
    in_slots = torch.div(
        before - 1, checkpoint_granularity, rounding_mode="floor"
    ).clamp_(min=0)
    out_slots = torch.div(after - 1, checkpoint_granularity, rounding_mode="floor")
    return _StateBlockIndexPlan(
        checkpoint_granularity=checkpoint_granularity,
        before=before,
        after=after,
        has_history=before > 0,
        in_slots=in_slots,
        out_slots=out_slots,
    )


def _gather_state_block_indices(
    rows: torch.Tensor,
    plan: _StateBlockIndexPlan,
    *,
    out_slots_safe: torch.Tensor | None = None,
    validate: bool = True,
    group_id: str = _STATE_GROUP_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    bs = plan.before.shape[0]
    rows = rows[:bs]
    max_slots = rows.shape[1]
    if out_slots_safe is None:
        out_slots_safe = plan.out_slots.clamp(min=0, max=max_slots - 1)

    state_in = rows.gather(1, plan.in_slots.unsqueeze(1)).squeeze(1)
    state_in = torch.where(plan.has_history, state_in, torch.zeros_like(state_in))
    state_out = rows.gather(1, out_slots_safe.unsqueeze(1)).squeeze(1)

    if validate:
        if bool((plan.after <= 0).any()):
            raise ValueError(
                "state paging: seq_lens_after must be >= 1 for every request"
            )
        if bool((plan.out_slots >= max_slots).any()):
            raise ValueError(
                "state paging: out page slot exceeds table width "
                f"{max_slots} (checkpoint_granularity="
                f"{plan.checkpoint_granularity})"
            )
        if bool((state_in[plan.has_history] <= 0).any()):
            raise ValueError(
                "state paging: in page is a pad (-1) or hole (0) for a "
                "request with history; reading it would silently resume "
                f"from the zero state ({group_id!r} table)"
            )
        if bool((state_out <= 0).any()):
            raise ValueError(
                "state paging: out page is a pad (-1) or hole (0); the "
                "request's working state page must be present in the "
                f"{group_id!r} table"
            )
        # A step that crosses a page boundary or resumes from a prefix hit
        # reads a page that is, or becomes, a read-only snapshot. It must write
        # a different page.
        # in == out is legal only for in-place evolution inside one page.
        crossing = plan.has_history & (plan.in_slots != out_slots_safe)
        if bool((state_in[crossing] == state_out[crossing]).any()):
            raise ValueError(
                "state paging: a boundary-crossing or prefix-resuming step "
                "resolves the same page for input and output; the input "
                "page is a read-only prefix snapshot and writing it would "
                f"corrupt every branch sharing it ({group_id!r} table)"
            )
        # The <= 0 raise above guarantees every state_out entry is positive.
        if torch.unique(state_out).numel() != state_out.numel():
            raise ValueError(
                f"state out pages must be unique per batch ({group_id!r} "
                "table): two requests writing one working state page would "
                "silently clobber each other"
            )
    return state_in.to(torch.int32), state_out.to(torch.int32)


def compute_state_block_indices(
    rows: torch.Tensor,
    checkpoint_granularity: int,
    seq_lens_before: torch.Tensor,
    seq_lens_after: torch.Tensor,
    *,
    validate: bool = True,
    group_id: str = _STATE_GROUP_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dual-index state pages: in = slot of position n-1 (0/null when no
    history), out = slot of the step's last position. rows: [bs, max_slots]
    int32 page ids (-1 pad, 0 hole). Within a slot in == out (in-place
    evolution); crossing a checkpoint boundary reads the old slot and writes
    the new one; resuming from a prefix hit reads the claimed snapshot slot
    and writes the fresh working slot.

    Args:
        rows: ``[bs, max_slots]`` int32 page-id table of one state group.
        checkpoint_granularity: Tokens between two state checkpoints (``P``).
        seq_lens_before: Per-request token count before this forward.
        seq_lens_after: Per-request token count after this forward.
        validate: Run the host-synchronizing write-side checks.
        group_id: State group the table belongs to; only used to attribute
            validation errors (multi-group KDA runs this once per group).

    Returns:
        ``(state_in, state_out)`` int32 page ids per request.
    """
    plan = _compute_state_block_index_plan(
        checkpoint_granularity, seq_lens_before, seq_lens_after
    )
    return _gather_state_block_indices(rows, plan, validate=validate, group_id=group_id)


def _prepare_cache_prefill_state_inputs(
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    state_in_blocks: torch.Tensor,
    state_out_blocks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize logical-zero fresh state without reading physical page 0."""
    has_initial_state = state_in_blocks > 0
    safe_input_pages = torch.where(
        has_initial_state, state_in_blocks, state_out_blocks
    ).to(torch.int64)
    state_out_blocks = state_out_blocks.to(torch.int64)

    # A fresh row copies its working page to itself. Only a resumed row reads
    # the checkpoint named by state_in_blocks.
    conv_states[state_out_blocks] = conv_states[safe_input_pages]
    recurrent_state = ssm_states[safe_input_pages]
    broadcast_mask = has_initial_state.reshape(
        (-1,) + (1,) * (recurrent_state.ndim - 1)
    )
    recurrent_state.masked_fill_(~broadcast_mask, 0)
    return recurrent_state, has_initial_state


def _prepare_gdn_decode_state_path(
    ssm_states: torch.Tensor,
    initial_state_indices: torch.Tensor,
    output_state_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, str | None]:
    """Select a safe decode solution while preserving graph padding indices.

    FlashInfer's FP32 kernels skip negative state rows, and the portable Triton
    kernels guard negative reads and writes for both FP32 and BF16 state. The
    FlashInfer BF16 path instead redirects padding to row 0, which is a live,
    scheduler-owned row in the single-table Mamba pool. Until that kernel supports a
    padding mask, route BF16 state through Triton and keep ``-1`` unchanged.
    """
    solution = "triton" if ssm_states.dtype == torch.bfloat16 else None
    return initial_state_indices, output_state_indices, solution


@dataclass
class MambaForwardMetadata:
    query_start_loc: torch.Tensor | None
    mamba_output_indices: torch.Tensor | None = None
    extend_prefix_lens: torch.Tensor | None = None
    extend_seq_lens_cpu: torch.Tensor | None = None
    # Per-state-group metadata is gathered once per group and batch;
    # layers select their entry via ``pool.group_id_for_layer(layer_id)``.
    state_in_blocks_by_group: dict[str, torch.Tensor] | None = None
    state_out_blocks_by_group: dict[str, torch.Tensor] | None = None


@dataclass
class _GDNReplayWorkspace:
    payload: torch.Tensor
    parameters: torch.Tensor
    layer_ids: tuple[int, ...]
    initialized_layers: set[int]
    geometry: tuple[int, int, int, int]
    state_dtype: torch.dtype


class MambaAttnBackend(AttentionBackend):
    """Attention backend for Mamba/GDN linear attention layers."""

    # This backend consumes state-family tables through dual-index state
    # paging; history-family groups belong to the full-attention sub-backend.
    # The hybrid wrapper unions the sub-backends' declarations, so a Kimi-K3
    # contract (history + state) is covered once both consumers exist.
    cache_consumer_families = frozenset({"state"})

    def __init__(self, config: BaseAttnConfig):
        super().__init__(config)
        self.pad_slot_id = -1
        self.forward_metadata: MambaForwardMetadata = None
        self.query_start_loc_list = []
        self.cached_cuda_graph_decode_query_start_loc: torch.Tensor = None
        self.cached_cuda_graph_verify_query_start_loc: torch.Tensor = None
        self.speculative_num_draft_tokens = getattr(
            config, "speculative_num_draft_tokens", 0
        )
        self.kv_pool = None
        self.state_paging_active = False
        self._checkpoint_granularity = 1
        self._state_group_ids: tuple[str, ...] = ()
        # CUDA-graph buffers: one persistent dual-index
        # (state_in/state_out) [bs] buffer per state group and captured batch
        # size. Values are keyed by group ID and indexed by ``bs - 1``.
        self.state_in_by_group: dict[str, list[torch.Tensor]] = {}
        self.state_out_by_group: dict[str, list[torch.Tensor]] = {}
        self.replay_ssm = bool(getattr(config, "replay_ssm", False))
        self._gdn_replay: _GDNReplayWorkspace | None = None

    def set_kv_pool(self, kv_pool) -> None:
        """Bind a unified pool that publishes state groups and component views."""
        self.kv_pool = kv_pool
        contract = kv_pool.arena.runtime_contract
        if contract is None:
            raise RuntimeError(
                "MambaAttnBackend requires a KV pool with a runtime cache contract"
            )
        state_group_ids = tuple(
            spec.group_id for spec in contract.group_specs if spec.family == "state"
        )
        if not state_group_ids:
            raise RuntimeError(
                "MambaAttnBackend requires at least one state-family cache group"
            )
        if not callable(getattr(kv_pool, "group_id_for_layer", None)) or not callable(
            getattr(kv_pool, "get_component", None)
        ):
            raise RuntimeError(
                "MambaAttnBackend requires group_id_for_layer() and get_component()"
            )
        self._state_group_ids = state_group_ids
        self.state_paging_active = True
        checkpoint_granularities = {
            spec.checkpoint_granularity
            for spec in contract.group_specs
            if spec.family == "state"
        }
        if len(checkpoint_granularities) != 1 or None in checkpoint_granularities:
            raise RuntimeError(
                "MambaAttnBackend requires one shared state-group "
                f"checkpoint_granularity, got {sorted(checkpoint_granularities, key=str)}"
            )
        self._checkpoint_granularity = int(checkpoint_granularities.pop())

    def _state_block_bounds(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-request (before, after) token counts for dual-index paging.

        seq_lens counts the tokens computed AFTER this forward (decode:
        q_len 1; extend: prefix + chunk). Computed once per batch and shared
        by every state group.
        """
        after = seq_lens[:bs]
        if forward_mode.is_decode_or_idle():
            before = after - 1
        else:
            num_extends = int(kwargs.get("num_extends", bs))
            if not 0 <= num_extends <= bs:
                raise ValueError("num_extends must be between 0 and bs")
            extend_prefix_lens = kwargs.get("extend_prefix_lens")
            if extend_prefix_lens is not None:
                extend_before = extend_prefix_lens[:num_extends].to(
                    device=after.device, dtype=after.dtype
                )
            else:
                extend_before = torch.zeros_like(after[:num_extends])
            before = torch.cat(
                (extend_before, after[num_extends:] - self.spec_num_tokens)
            )
        return before, after

    def _state_layer_ids(self) -> list[int]:
        """Recurrent layer ids backed by the unified cache pool."""
        state_groups = set(self._state_group_ids)
        return sorted(
            layer_id
            for layer_id, group_id in self.kv_pool._group_ids_by_layer.items()
            if group_id in state_groups
        )

    def _state_groups(self) -> tuple[str, ...]:
        return self._state_group_ids

    def _state_group_for(self, layer_id: int) -> str:
        return self.kv_pool.group_id_for_layer(layer_id)

    def _state_components(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.kv_pool.get_component(layer_id, "conv_state"),
            self.kv_pool.get_component(layer_id, "recurrent_state"),
        )

    def _verify_state_blocks(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        draft_token_num: int,
        kwargs: dict,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        """Target-verify state paging: per-group committed-state pages.

        Verify reads the state at the last COMMITTED position
        (``seq_lens - draft_token_num``); speculative outputs stay out of the
        state slab, and the accepted state is committed back by
        ``update_mamba_state_after_mtp_verify``. Returns the per-group in
        pages, the committed lengths, and the per-group group tables (kept
        for the commit's dynamic page resolve).
        """
        committed = (seq_lens[:bs].to(torch.int64) - draft_token_num).clamp_min(0)
        in_slots = torch.div(
            (committed - 1).clamp_min(0),
            self._checkpoint_granularity,
            rounding_mode="floor",
        )
        has_history = committed > 0
        state_in_blocks: dict[str, torch.Tensor] = {}
        tables: dict[str, torch.Tensor] = {}
        cache_metadata = kwargs.get("cache_metadata")
        if cache_metadata is None:
            raise RuntimeError(
                "target-verify requires cache metadata on the forward batch"
            )
        forward_batch = kwargs.get("forward_batch")
        rows_by_group = {
            group_id: cache_metadata.require_table(
                group_id, active_forward_op=forward_batch
            )
            for group_id in self._state_groups()
        }
        for group_id, rows in rows_by_group.items():
            slots_safe = in_slots.clamp(min=0, max=rows.shape[1] - 1)
            pages = rows[:bs].gather(1, slots_safe.unsqueeze(1)).squeeze(1)
            pages = torch.where(has_history, pages, torch.full_like(pages, -1)).to(
                torch.int32
            )
            state_in_blocks[group_id] = pages
            tables[group_id] = rows
        return state_in_blocks, committed, tables

    def _ensure_verify_scratch(self, bs: int, draft_token_num: int) -> None:
        """Lazily allocate graph-stable verify scratch and replay inputs."""
        max_bs = max(len(self.query_start_loc_list), bs)
        rows_needed = max_bs * (draft_token_num + 1)
        scratch = getattr(self, "_verify_scratch", None)
        if scratch is not None and next(iter(scratch.values()))[0].shape[0] >= (
            rows_needed
        ):
            return
        self._verify_scratch = {}
        self._verify_copy_tables = None
        layer_ids = tuple(self._state_layer_ids())
        for layer_id in layer_ids:
            conv, ssm = self._state_components(layer_id)
            self._verify_scratch[layer_id] = (
                torch.zeros(
                    (rows_needed, *conv.shape[1:]),
                    dtype=conv.dtype,
                    device=conv.device,
                ),
                (
                    None
                    if self.replay_ssm
                    else torch.zeros(
                        (rows_needed, *ssm.shape[1:]),
                        dtype=ssm.dtype,
                        device=ssm.device,
                    )
                ),
            )

        if self.replay_ssm:
            conv, ssm = self._state_components(layer_ids[0])
            num_v_heads, head_v_dim, head_k_dim = ssm.shape[1:]
            key_width = (conv.shape[1] - num_v_heads * head_v_dim) // 2
            with torch.inference_mode(False):
                self._gdn_replay = _GDNReplayWorkspace(
                    payload=torch.empty(
                        (
                            len(layer_ids),
                            max_bs * draft_token_num,
                            key_width + num_v_heads * head_v_dim + 2 * num_v_heads,
                        ),
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    parameters=torch.empty(
                        (len(layer_ids), 2, num_v_heads),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    layer_ids=layer_ids,
                    initialized_layers=set(),
                    geometry=(
                        key_width // head_k_dim,
                        num_v_heads,
                        head_k_dim,
                        head_v_dim,
                    ),
                    state_dtype=ssm.dtype,
                )

    def preallocate_verify_workspace(self, max_bs: int, draft_token_num: int) -> int:
        """Allocate graph-stable verify state and return its byte size."""
        if not self.state_paging_active or self.is_draft:
            return 0
        self._ensure_verify_scratch(max_bs, draft_token_num)
        return sum(
            tensor.nbytes
            for layer_scratch in self._verify_scratch.values()
            for tensor in layer_scratch
            if tensor is not None
        )

    def _verify_copy_tables_get(self) -> dict:
        """Pointer tables for the batched verify state copies and replay:
        per-layer base addresses, row strides, and state-group selectors.
        Rebuilt only when the scratch is reallocated so CUDA graph capture
        can record stable tensors."""
        tables = getattr(self, "_verify_copy_tables", None)
        if tables is not None:
            return tables
        layer_ids = list(self._state_layer_ids())
        group_ids = self._state_groups()
        group_index = {group_id: i for i, group_id in enumerate(group_ids)}
        conv_src, conv_dst, ssm_src, ssm_dst, group_sel = [], [], [], [], []
        conv_src_st, conv_dst_st, ssm_src_st, ssm_dst_st = [], [], [], []
        ssm_element_st = []
        conv_bytes: int | None = None
        ssm_bytes: int | None = None

        def _row_stride_i32(t: torch.Tensor) -> int:
            # Slab components are page-interleaved as_strided views: row
            # payload contiguous, row-to-row stride the physical page.
            if t[0].numel() and not t[0].is_contiguous():
                raise RuntimeError(
                    "batched verify state copy requires contiguous row payloads"
                )
            stride_bytes = t.stride(0) * t.element_size()
            if stride_bytes % 4:
                raise RuntimeError("state row stride must be 4-byte aligned")
            return stride_bytes // 4

        for layer_id in layer_ids:
            conv, ssm = self._state_components(layer_id)
            conv_scratch, ssm_scratch = self._verify_scratch[layer_id]
            row_c = conv[0].numel() * conv.element_size()
            row_s = ssm[0].numel() * ssm.element_size()
            conv_bytes = row_c if conv_bytes is None else conv_bytes
            ssm_bytes = row_s if ssm_bytes is None else ssm_bytes
            if row_c != conv_bytes or row_s != ssm_bytes:
                raise RuntimeError("verify state rows must be uniform per kind")
            conv_src.append(conv.data_ptr())
            conv_dst.append(conv_scratch.data_ptr())
            conv_src_st.append(_row_stride_i32(conv))
            conv_dst_st.append(_row_stride_i32(conv_scratch))
            ssm_src.append(ssm.data_ptr())
            ssm_src_st.append(_row_stride_i32(ssm))
            ssm_element_st.append(ssm.stride(0))
            if not self.replay_ssm:
                ssm_dst.append(ssm_scratch.data_ptr())
                ssm_dst_st.append(_row_stride_i32(ssm_scratch))
            group_sel.append(group_index[self._state_group_for(layer_id)])

        def _u64(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.uint64, device=self.device)

        def _i64(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int64, device=self.device)

        tables = {
            "conv_comp": _u64(conv_src),
            "conv_scratch": _u64(conv_dst),
            "conv_comp_stride": _i64(conv_src_st),
            "conv_scratch_stride": _i64(conv_dst_st),
            "conv_bytes": conv_bytes,
            "ssm_comp": _u64(ssm_src),
            "ssm_scratch": None if self.replay_ssm else _u64(ssm_dst),
            "ssm_comp_stride": _i64(ssm_src_st),
            "ssm_element_stride": _i64(ssm_element_st),
            "ssm_scratch_stride": None if self.replay_ssm else _i64(ssm_dst_st),
            "ssm_bytes": ssm_bytes,
            "group_sel": _i64(group_sel),
            "num_layers": len(layer_ids),
        }
        self._verify_copy_tables = tables
        return tables

    def _verify_seed_dst_rows(self, bs: int, draft_token_num: int) -> torch.Tensor:
        """Memoized layer-major ``[L*bs]`` scratch init-row ids (row
        ``req*(T+1)`` per request, tiled per layer). Graph replay must see the
        identical tensor, mirroring ``_verify_scratch_grid``."""
        cache = getattr(self, "_verify_seed_dst_cache", None)
        if cache is None:
            cache = self._verify_seed_dst_cache = {}
        tables = self._verify_copy_tables_get()
        key = (bs, draft_token_num, tables["num_layers"])
        rows = cache.get(key)
        if rows is None:
            init = torch.arange(bs, dtype=torch.int64, device=self.device) * (
                draft_token_num + 1
            )
            rows = init.repeat(tables["num_layers"])
            cache[key] = rows
        return rows

    def _seed_verify_scratch_batched(self, bs: int, draft_token_num: int) -> None:
        """Seed verify scratch from each layer's committed state page."""
        from tokenspeed_kernel.ops.kvcache.triton import copy_state_rows

        tables = self._verify_copy_tables_get()
        state_in_by_group = self.forward_metadata.state_in_blocks_by_group
        sin_stack = torch.stack(
            [state_in_by_group[group_id][:bs] for group_id in self._state_groups()]
        ).to(torch.int64)
        src_rows = sin_stack.index_select(0, tables["group_sel"]).reshape(-1)
        copy_state_rows(
            tables["conv_comp"],
            tables["conv_scratch"],
            src_rows,
            self._verify_seed_dst_rows(bs, draft_token_num),
            row_bytes=tables["conv_bytes"],
            src_row_strides=tables["conv_comp_stride"],
            dst_row_strides=tables["conv_scratch_stride"],
        )
        if not self.replay_ssm:
            copy_state_rows(
                tables["ssm_comp"],
                tables["ssm_scratch"],
                src_rows,
                self._verify_seed_dst_rows(bs, draft_token_num),
                row_bytes=tables["ssm_bytes"],
                src_row_strides=tables["ssm_comp_stride"],
                dst_row_strides=tables["ssm_scratch_stride"],
            )

    def _verify_scratch_grid(self, bs: int, draft_token_num: int) -> torch.Tensor:
        """Scratch row grid ``[bs, draft_token_num]``: row ``req*(T+1)`` is
        the seeded init window, rows ``req*(T+1)+1+t`` the per-position
        outputs. Memoized per (bs, T): CUDA-graph capture records the tensor's
        storage, so replays must present the identical tensor."""
        cache = getattr(self, "_verify_grid_cache", None)
        if cache is None:
            cache = self._verify_grid_cache = {}
        grid = cache.get((bs, draft_token_num))
        if grid is not None:
            return grid
        stride = draft_token_num + 1
        base = (
            torch.arange(bs, dtype=torch.int32, device=self.device) * stride
        ).unsqueeze(1)
        steps = torch.arange(
            1, draft_token_num + 1, dtype=torch.int32, device=self.device
        ).unsqueeze(0)
        grid = base + steps
        cache[(bs, draft_token_num)] = grid
        return grid

    def commit_verified_state(self, accepted_length: torch.Tensor) -> None:
        """Commit the accepted draft prefix into each group's state slab."""
        ctx = getattr(self, "_verify_commit_ctx", None)
        if ctx is None:
            return
        committed, tables, draft_token_num, read_pages_by_group = ctx
        bs = accepted_length.shape[0]
        k = accepted_length.to(torch.int64).clamp(min=1, max=draft_token_num)
        new_last = committed[:bs] + k - 1
        slot = torch.div(new_last, self._checkpoint_granularity, rounding_mode="floor")
        stride = draft_token_num + 1
        src_rows = (
            torch.arange(bs, dtype=torch.int64, device=accepted_length.device) * stride
            + k
        )
        pages_by_group: dict[str, torch.Tensor] = {}
        for group_id in self._state_groups():
            rows_tbl = tables[group_id]
            slot_safe = slot.clamp(min=0, max=rows_tbl.shape[1] - 1)
            pages_by_group[group_id] = (
                rows_tbl[:bs]
                .gather(1, slot_safe.unsqueeze(1))
                .squeeze(1)
                .to(torch.int64)
                .clamp_min(0)
            )
        copy_tables = self._verify_copy_tables_get()
        pages_stack = torch.stack(
            [pages_by_group[group_id] for group_id in self._state_groups()]
        )
        dst_rows = pages_stack.index_select(0, copy_tables["group_sel"]).reshape(-1)
        from tokenspeed_kernel.ops.kvcache.triton import copy_state_rows

        src_tiled = src_rows.repeat(copy_tables["num_layers"])
        copy_state_rows(
            copy_tables["conv_scratch"],
            copy_tables["conv_comp"],
            src_tiled,
            dst_rows,
            row_bytes=copy_tables["conv_bytes"],
            src_row_strides=copy_tables["conv_scratch_stride"],
            dst_row_strides=copy_tables["conv_comp_stride"],
        )
        if self.replay_ssm:
            replay = self._gdn_replay
            gdn_replay_commit(
                replay.payload,
                replay.parameters,
                state_addresses=copy_tables["ssm_comp"],
                state_row_strides=copy_tables["ssm_element_stride"],
                read_indices=torch.stack(
                    [
                        read_pages_by_group[group_id][:bs]
                        for group_id in self._state_groups()
                    ]
                )
                .index_select(0, copy_tables["group_sel"])
                .to(torch.int32),
                write_indices=dst_rows.view(copy_tables["num_layers"], bs).to(
                    torch.int32
                ),
                accepted_length=k.to(torch.int32),
                draft_token_num=draft_token_num,
                geometry=replay.geometry,
                state_dtype=replay.state_dtype,
            )
        else:
            copy_state_rows(
                copy_tables["ssm_scratch"],
                copy_tables["ssm_comp"],
                src_tiled,
                dst_rows,
                row_bytes=copy_tables["ssm_bytes"],
                src_row_strides=copy_tables["ssm_scratch_stride"],
                dst_row_strides=copy_tables["ssm_comp_stride"],
            )
        self._verify_commit_ctx = None

    def _cache_contract_state_blocks(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        kwargs: dict,
        *,
        validate: bool | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Per-state-group (state_in, state_out) page-id mappings for this
        forward from the operation-bound cache metadata.

        The dual-index gather runs ONCE per state group per batch — never per
        layer. State layers select their group's entry via
        ``pool.group_id_for_layer(layer_id)`` at forward time.

        validate: explicit True/False wins; None (the hot-path default)
        validates only under TOKENSPEED_CACHE_DEBUG=1 (the checks host-sync).

        Returns:
            ``(state_in_blocks, state_out_blocks)`` mappings keyed by state
            group id, each value an int32 ``[bs]`` page-id tensor.
        """
        if validate is None:
            validate = cache_debug_enabled()
        cache_metadata = kwargs.get("cache_metadata")
        if cache_metadata is None:
            raise RuntimeError(
                "MambaAttnBackend requires operation-bound cache metadata"
            )
        forward_batch = kwargs.get("forward_batch")
        before, after = self._state_block_bounds(bs, seq_lens, forward_mode, kwargs)
        plan = _compute_state_block_index_plan(
            self._checkpoint_granularity, before, after
        )
        out_slots_by_width: dict[int, torch.Tensor] = {}
        state_in_blocks: dict[str, torch.Tensor] = {}
        state_out_blocks: dict[str, torch.Tensor] = {}
        for group_id in self._state_group_ids:
            rows = cache_metadata.require_table(
                group_id, active_forward_op=forward_batch
            )
            table_width = rows.shape[1]
            out_slots_safe = out_slots_by_width.get(table_width)
            if out_slots_safe is None:
                out_slots_safe = plan.out_slots.clamp(min=0, max=table_width - 1)
                out_slots_by_width[table_width] = out_slots_safe
            state_in, state_out = _gather_state_block_indices(
                rows,
                plan,
                out_slots_safe=out_slots_safe,
                validate=validate,
                group_id=group_id,
            )
            state_in_blocks[group_id] = state_in
            state_out_blocks[group_id] = state_out
        return state_in_blocks, state_out_blocks

    def init_forward_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = ForwardMode.DECODE,
        **kwargs,
    ):
        is_target_verify = (
            forward_mode.is_decode_or_idle()
            and not self.is_draft
            and self.spec_num_tokens > 1
        )
        is_draft_extend = (
            forward_mode.is_decode_or_idle()
            and self.is_draft
            and self.spec_num_tokens > 1
        )

        num_extends = int(
            kwargs.get(
                "num_extends",
                bs if forward_mode.is_extend_or_mixed() else 0,
            )
        )
        if not 0 <= num_extends <= bs:
            raise ValueError("num_extends must be between 0 and bs")
        mamba_output_indices = None
        extend_seq_lens_cpu = kwargs.get("extend_seq_lens_cpu")
        if extend_seq_lens_cpu is not None:
            extend_seq_lens_cpu = extend_seq_lens_cpu[:num_extends]
        if forward_mode.is_decode_or_idle() and self.spec_num_tokens == 1:
            query_start_loc = torch.arange(
                0, bs + 1, dtype=torch.int32, device=self.device
            )
        elif forward_mode.is_extend_or_mixed() or is_target_verify or is_draft_extend:
            if is_target_verify or is_draft_extend:
                tokens_per_req = kwargs.get(
                    "tokens_per_req", self.speculative_num_draft_tokens
                )
                query_start_loc = torch.arange(
                    0,
                    bs * tokens_per_req + 1,
                    step=tokens_per_req,
                    dtype=torch.int32,
                    device=self.device,
                )
                set_total_chunks_hint_uniform(bs, tokens_per_req, query_start_loc)
            else:
                extend_start_loc = kwargs.get("extend_start_loc")
                extend_seq_lens = kwargs.get("extend_seq_lens")
                if forward_mode.is_mixed():
                    if extend_seq_lens is None or extend_seq_lens_cpu is None:
                        raise RuntimeError(
                            "mixed GDN metadata requires extend sequence lengths"
                        )
                    query_lens = torch.full(
                        (bs,),
                        self.spec_num_tokens,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    query_lens[:num_extends] = extend_seq_lens[:num_extends]
                    query_start_loc = torch.zeros(
                        bs + 1, dtype=torch.int32, device=self.device
                    )
                    torch.cumsum(query_lens, dim=0, out=query_start_loc[1:])
                    extend_seq_lens_cpu = torch.cat(
                        (
                            extend_seq_lens_cpu,
                            torch.full(
                                (bs - num_extends,),
                                self.spec_num_tokens,
                                dtype=torch.int32,
                            ),
                        )
                    )
                elif extend_start_loc is not None and extend_seq_lens is not None:
                    query_start_loc = torch.empty(
                        (bs + 1,), dtype=torch.int32, device=self.device
                    )
                    query_start_loc[:bs] = extend_start_loc
                    query_start_loc[bs] = extend_start_loc[-1] + extend_seq_lens[-1]
                    if extend_seq_lens_cpu is None:
                        extend_seq_lens_cpu = extend_seq_lens[:bs].to(
                            device="cpu", dtype=torch.int32
                        )
                else:
                    extend_prefix_lens = kwargs.get("extend_prefix_lens")
                    if extend_prefix_lens is not None:
                        extend_lens = (seq_lens[:bs] - extend_prefix_lens[:bs]).to(
                            torch.int32
                        )
                    else:
                        # No prefix: all tokens are new
                        extend_lens = seq_lens[:bs].to(torch.int32)
                    query_start_loc = torch.zeros(
                        bs + 1, dtype=torch.int32, device=self.device
                    )
                    torch.cumsum(extend_lens, dim=0, out=query_start_loc[1:])
                    if extend_seq_lens_cpu is None:
                        extend_seq_lens_cpu = extend_lens.to(device="cpu")
                set_total_chunks_hint(extend_seq_lens_cpu, query_start_loc)
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

        state_in_blocks_by_group = None
        state_out_blocks_by_group = None
        # Idle/bs==0 forwards carry no requests and never reach the mamba
        # forward (router returns early), so no tables are required.
        if bs > 0 and not forward_mode.is_idle():
            if is_draft_extend:
                raise RuntimeError("state paging on a draft worker is unsupported")
            if is_target_verify:
                draft_token_num = int(
                    kwargs.get("tokens_per_req", self.speculative_num_draft_tokens)
                )
                (
                    state_in_blocks_by_group,
                    verify_committed,
                    verify_tables,
                ) = self._verify_state_blocks(bs, seq_lens, draft_token_num, kwargs)
                # Slab out pages are unused under verify; alias in so shape
                # contracts hold.
                state_out_blocks_by_group = state_in_blocks_by_group
                self._ensure_verify_scratch(bs, draft_token_num)
                mamba_output_indices = self._verify_scratch_grid(bs, draft_token_num)
                self._verify_commit_ctx = (
                    verify_committed,
                    verify_tables,
                    draft_token_num,
                    state_in_blocks_by_group,
                )
            else:
                (
                    state_in_blocks_by_group,
                    state_out_blocks_by_group,
                ) = self._cache_contract_state_blocks(
                    bs, seq_lens, forward_mode, kwargs
                )

        self.forward_metadata = MambaForwardMetadata(
            query_start_loc=query_start_loc,
            mamba_output_indices=mamba_output_indices,
            extend_prefix_lens=kwargs.get("extend_prefix_lens"),
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            state_in_blocks_by_group=state_in_blocks_by_group,
            state_out_blocks_by_group=state_out_blocks_by_group,
        )

    # ---- CUDA graph state ----

    def init_cuda_graph_state(self, max_num_tokens: int):
        for i in range(max_num_tokens):
            self.query_start_loc_list.append(
                torch.empty((i + 2,), dtype=torch.int32, device=self.device)
            )
            # Keep one graph-stable dual-index buffer pair per state group.
            for gid in self._state_group_ids:
                self.state_in_by_group.setdefault(gid, []).append(
                    torch.full(
                        (i + 1,),
                        self.pad_slot_id,
                        dtype=torch.int32,
                        device=self.device,
                    )
                )
                self.state_out_by_group.setdefault(gid, []).append(
                    torch.full(
                        (i + 1,),
                        self.pad_slot_id,
                        dtype=torch.int32,
                        device=self.device,
                    )
                )
        self.cached_cuda_graph_decode_query_start_loc = torch.arange(
            0, max_num_tokens + 1, dtype=torch.int32, device=self.device
        )
        if self.speculative_num_draft_tokens > 0:
            # Need max_num_tokens+1 entries (one per request + sentinel).
            # Each entry is request_index * spec_num_draft_tokens.
            self.cached_cuda_graph_verify_query_start_loc = torch.arange(
                0,
                (max_num_tokens + 1) * self.speculative_num_draft_tokens,
                step=self.speculative_num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        self._qsl_dirty = [False] * max_num_tokens
        self._qsl_last_mode = [None] * max_num_tokens

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        **kwargs,
    ):
        is_target_verify = (
            forward_mode.is_decode_or_idle()
            and not self.is_draft
            and self.spec_num_tokens > 1
        )
        is_draft_extend = (
            forward_mode.is_decode_or_idle()
            and self.is_draft
            and self.spec_num_tokens > 1
        )

        if forward_mode.is_decode_or_idle() and self.spec_num_tokens == 1:
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
            )
        elif is_target_verify or is_draft_extend:
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
            )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

        mamba_output_indices = None
        state_in_blocks_by_group = None
        state_out_blocks_by_group = None
        if self.state_paging_active:
            # Real tables only arrive at replay; capture binds the persistent
            # buffers (all pad_slot_id: kernels skip reads/writes at capture,
            # so state slab rows are never dirtied by the capture pass).
            if is_draft_extend:
                raise RuntimeError("state paging on a draft worker is unsupported")
            if is_target_verify:
                # Verify capture/prewarm: pad state_in (reads zeros / writes
                # skip) + the real scratch grid; commit stays disarmed.
                draft_token_num = int(self.speculative_num_draft_tokens)
                self._ensure_verify_scratch(bs, draft_token_num)
                mamba_output_indices = self._verify_scratch_grid(bs, draft_token_num)
                self._verify_commit_ctx = None
            state_in_blocks_by_group = {}
            state_out_blocks_by_group = {}
            for gid in self._state_group_ids:
                state_in = self.state_in_by_group[gid][bs - 1]
                state_out = self.state_out_by_group[gid][bs - 1]
                state_in.fill_(self.pad_slot_id)
                state_out.fill_(self.pad_slot_id)
                state_in_blocks_by_group[gid] = state_in
                state_out_blocks_by_group[gid] = state_out
        self._qsl_dirty[bs - 1] = False
        self._qsl_last_mode[bs - 1] = (forward_mode, self.spec_num_tokens > 1)
        self.forward_metadata = MambaForwardMetadata(
            query_start_loc=self.query_start_loc_list[bs - 1],
            mamba_output_indices=mamba_output_indices,
            state_in_blocks_by_group=state_in_blocks_by_group,
            state_out_blocks_by_group=state_out_blocks_by_group,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = None,
        **kwargs,
    ):
        # State attention has no page table; the shared replay call's
        # page_table keyword is absorbed by **kwargs unused.
        num_padding = kwargs.get("num_padding", 0)

        real_bs = bs - num_padding
        req_pool_indices = req_pool_indices[:bs]

        is_target_verify = (
            forward_mode is not None
            and forward_mode.is_decode_or_idle()
            and not self.is_draft
            and self.spec_num_tokens > 1
        )
        is_draft_extend = (
            forward_mode is not None
            and forward_mode.is_decode_or_idle()
            and self.is_draft
            and self.spec_num_tokens > 1
        )

        mamba_output_indices = None

        if num_padding == 0:
            need_copy = self._qsl_dirty[bs - 1] or self._qsl_last_mode[bs - 1] != (
                forward_mode,
                self.spec_num_tokens > 1,
            )
            if need_copy:
                if forward_mode.is_decode_or_idle() and self.spec_num_tokens == 1:
                    self.query_start_loc_list[bs - 1].copy_(
                        self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
                    )
                elif is_target_verify or is_draft_extend:
                    self.query_start_loc_list[bs - 1].copy_(
                        self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
                    )
                self._qsl_dirty[bs - 1] = False
                self._qsl_last_mode[bs - 1] = (forward_mode, self.spec_num_tokens > 1)
        else:
            if forward_mode.is_decode_or_idle() and self.spec_num_tokens == 1:
                self.query_start_loc_list[bs - 1][:real_bs].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[:real_bs]
                )
                self.query_start_loc_list[bs - 1][real_bs:].fill_(real_bs)
            elif is_target_verify or is_draft_extend:
                self.query_start_loc_list[bs - 1][:real_bs].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[:real_bs]
                )
                self.query_start_loc_list[bs - 1][real_bs:].fill_(
                    real_bs * self.speculative_num_draft_tokens
                )
            else:
                raise ValueError(f"Invalid forward mode: {forward_mode=}")
            self._qsl_dirty[bs - 1] = True
            self._qsl_last_mode[bs - 1] = (forward_mode, self.spec_num_tokens > 1)

        state_in_blocks_by_group = None
        state_out_blocks_by_group = None
        if self.state_paging_active and is_target_verify:
            # Target-verify replay: refresh the captured state_in buffers,
            # re-arm the post-round commit, and keep the recorded scratch grid.
            draft_token_num = int(self.speculative_num_draft_tokens)
            self._ensure_verify_scratch(bs, draft_token_num)
            mamba_output_indices = self._verify_scratch_grid(bs, draft_token_num)
            pages_by_group = None
            if real_bs > 0:
                (
                    pages_by_group,
                    verify_committed,
                    verify_tables,
                ) = self._verify_state_blocks(
                    real_bs, seq_lens, draft_token_num, kwargs
                )
                self._verify_commit_ctx = (
                    verify_committed,
                    verify_tables,
                    draft_token_num,
                    pages_by_group,
                )
            else:
                self._verify_commit_ctx = None
            captured_in = {
                group_id: self.state_in_by_group[group_id][bs - 1]
                for group_id in self._state_groups()
            }
            captured_out = {
                group_id: self.state_out_by_group[group_id][bs - 1]
                for group_id in self._state_groups()
            }
            state_in_blocks_by_group = {}
            state_out_blocks_by_group = {}
            for group_id in self._state_groups():
                state_in = captured_in[group_id]
                state_out = captured_out[group_id]
                if pages_by_group is not None:
                    state_in[:real_bs].copy_(pages_by_group[group_id][:real_bs])
                if real_bs < bs:
                    state_in[real_bs:].fill_(self.pad_slot_id)
                # Slab out pages are unused under verify; keep the captured
                # buffer inert.
                state_out.fill_(self.pad_slot_id)
                state_in_blocks_by_group[group_id] = state_in
                state_out_blocks_by_group[group_id] = state_out
        elif self.state_paging_active:
            # For multi-group state paging, dual indexing runs once per
            # state group over the real rows. Padded rows get pad_slot_id (-1),
            # which state kernels skip, so they never touch a live page.
            # Decode-only
            # (q_len == 1): before = seq_lens - 1. Validation defaults off on
            # the replay hot path (host sync); TOKENSPEED_CACHE_DEBUG=1 arms it.
            # bs==0 idle replay carries no operation-bound metadata; every row
            # is a dummy padded row, so skip the dual-index gather entirely.
            state_in_blocks_by_group, state_out_blocks_by_group = (
                self._replay_contract_state_pages(
                    bs, real_bs, seq_lens, forward_mode, kwargs
                )
            )

        self.forward_metadata = MambaForwardMetadata(
            query_start_loc=self.query_start_loc_list[bs - 1],
            mamba_output_indices=mamba_output_indices,
            state_in_blocks_by_group=state_in_blocks_by_group,
            state_out_blocks_by_group=state_out_blocks_by_group,
        )

    def _replay_contract_state_pages(
        self,
        bs: int,
        real_bs: int,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        kwargs: dict,
    ) -> tuple[dict, dict]:
        """Fill the per-bs persistent state-page buffers for a decode replay.

        Fast path: one prep-tape launch computes every group's dual-index
        pages straight into the persistent buffers and pads the tail (the
        eager chain is ~10 launches per group plus copies/fills). Falls back
        to the eager chain when the tape preconditions do not hold (seq_lens
        dtype, group count, debug validation).
        """
        cache_metadata = kwargs.get("cache_metadata")
        forward_batch = kwargs.get("forward_batch")
        gids = self._state_group_ids
        use_tape = (
            not cache_debug_enabled()
            and seq_lens.is_cuda
            and seq_lens.dtype == torch.int32
            and len(gids) <= 4
            and cache_metadata is not None
        )
        if use_tape:
            from tokenspeed_kernel.ops.metadata import PrepTape, Reg

            tapes = getattr(self, "_replay_state_tapes", None)
            if tapes is None:
                tapes = self._replay_state_tapes = {}
            tape = tapes.get(bs)
            if tape is None:
                tape = PrepTape(self.device)
                for i, gid in enumerate(gids):
                    sin = self.state_in_by_group[gid][bs - 1]
                    sout = self.state_out_by_group[gid][bs - 1]
                    tape.state_pages(
                        sin,
                        sout,
                        rows_ptr=Reg(Reg.PTR0 + i),
                        seq_lens_ptr=Reg.PTR4,
                        bs=Reg.REAL_BS,
                        max_slots=Reg(Reg.USER0 + i),
                        page_size=self._checkpoint_granularity,
                    )
                    tape.filltail(
                        sin, live=Reg.REAL_BS, total=bs, value=self.pad_slot_id
                    )
                    tape.filltail(
                        sout, live=Reg.REAL_BS, total=bs, value=self.pad_slot_id
                    )
                tape.finalize()
                tapes[bs] = tape
            regs = {Reg.REAL_BS: real_bs, Reg.PTR4: seq_lens}
            for i, gid in enumerate(gids):
                rows = cache_metadata.require_table(
                    gid, active_forward_op=forward_batch
                )
                regs[Reg(Reg.PTR0 + i)] = rows
                regs[Reg(Reg.USER0 + i)] = rows.shape[1]
            tape.run(regs)
            return (
                {g: self.state_in_by_group[g][bs - 1] for g in gids},
                {g: self.state_out_by_group[g][bs - 1] for g in gids},
            )

        state_in_by = state_out_by = None
        if real_bs > 0:
            state_in_by, state_out_by = self._cache_contract_state_blocks(
                real_bs,
                seq_lens,
                forward_mode,
                kwargs,
                validate=None,
            )
        in_by_group: dict[str, torch.Tensor] = {}
        out_by_group: dict[str, torch.Tensor] = {}
        for gid in gids:
            state_in_blocks = self.state_in_by_group[gid][bs - 1]
            state_out_blocks = self.state_out_by_group[gid][bs - 1]
            if real_bs > 0:
                state_in_blocks[:real_bs].copy_(state_in_by[gid][:real_bs])
                state_out_blocks[:real_bs].copy_(state_out_by[gid][:real_bs])
            if real_bs < bs:
                state_in_blocks[real_bs:].fill_(self.pad_slot_id)
                state_out_blocks[real_bs:].fill_(self.pad_slot_id)
            in_by_group[gid] = state_in_blocks
            out_by_group[gid] = state_out_blocks
        return in_by_group, out_by_group

    # ---- Forward ----

    def _layer_state(
        self, layer_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Resolve one recurrent layer's page indices and component views."""
        metadata = self.forward_metadata
        state_in_by_group = metadata.state_in_blocks_by_group
        group_id = self._state_group_for(layer_id)
        if state_in_by_group is None or group_id not in state_in_by_group:
            raise RuntimeError(
                f"state paging: layer {layer_id} resolves to group {group_id!r}, "
                "but the forward batch has no page indices for that group"
            )
        conv_states, ssm_states = self._state_components(layer_id)
        return (
            state_in_by_group[group_id],
            metadata.state_out_blocks_by_group[group_id],
            conv_states,
            ssm_states,
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        # Multi-token decode (target verify or drafter compound) reuses
        # the multi-token kernel path in forward_extend. `q` is None for
        # hybrid linear-attn layers; the token count comes from mixed_qkv.
        q_len_per_req = kwargs["mixed_qkv"].shape[0] // bs if bs > 0 else 1
        if q_len_per_req > 1:
            return self.forward_extend(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                bs,
                forward_mode=ForwardMode.DECODE,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )

        mixed_qkv = kwargs["mixed_qkv"]
        conv_weights = kwargs["conv_weights"]
        bias = kwargs["bias"]
        activation = kwargs["activation"]
        key_dim = kwargs["key_dim"]
        value_dim = kwargs["value_dim"]
        attn_tp_size = kwargs["attention_tp_size"]
        head_k_dim = kwargs["head_k_dim"]
        head_v_dim = kwargs["head_v_dim"]
        a = kwargs.get("a")
        b = kwargs.get("b")
        f_a_out = kwargs.get("f_a_out")
        f_b_weight = kwargs.get("f_b_weight")
        g_raw = kwargs.get("g_raw")
        beta_raw = kwargs.get("beta_raw")
        output_gate = kwargs.get("output_gate")
        norm_weight = kwargs.get("norm_weight")
        norm_eps = kwargs.get("norm_eps")
        gate_lower_bound = kwargs.get("lower_bound")
        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        layer_id = kwargs["layer_id"]

        # Read the page holding position n-1 and write the page holding
        # position n. Padding rows use -1 and are skipped by both kernels.
        state_in_blocks, state_out_blocks, conv_states, ssm_states = self._layer_state(
            layer_id
        )
        read_indices = state_in_blocks

        fused_out = self._decode(
            mixed_qkv,
            conv_weights,
            conv_states,
            ssm_states,
            read_indices,
            state_out_blocks,
            f_a_out=f_a_out,
            f_b_weight=f_b_weight,
            beta_raw=beta_raw,
            A_log=A_log,
            dt_bias=dt_bias,
            value_dim=value_dim,
            attn_tp_size=attn_tp_size,
            head_v_dim=head_v_dim,
            lower_bound=gate_lower_bound,
            output_gate=output_gate,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
        if fused_out is not None:
            return fused_out

        # Stride-aware fused decoders consume packed projection views directly.
        # Preserve the shared fallback's established compact input layout.
        mixed_qkv = mixed_qkv.contiguous()
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            conv_weights,
            bias,
            activation,
            conv_state_indices=read_indices,
            output_state_indices=state_out_blocks.view(-1, 1),
        )

        query, key, value = torch.split(
            mixed_qkv,
            [
                key_dim // attn_tp_size,
                key_dim // attn_tp_size,
                value_dim // attn_tp_size,
            ],
            dim=-1,
        )
        seq_len = query.shape[0]
        num_heads = query.shape[1] // head_k_dim
        # [B, 1, H, K] / [B, 1, HV, V]: B=this decode step's request count,
        # T=1. gdn_decode_step's K-last state pool means no transpose is
        # needed between this call and the pool/state-slab storage.
        query = query.view(seq_len, 1, num_heads, head_k_dim)
        key = key.view(seq_len, 1, num_heads, head_k_dim)
        value = value.view(seq_len, 1, value.shape[1] // head_v_dim, head_v_dim)

        return self._decode_scan(
            query,
            key,
            value,
            ssm_states,
            read_indices,
            state_out_blocks,
            A_log=A_log,
            dt_bias=dt_bias,
            a=a,
            b=b,
            g_raw=g_raw,
            f_a_out=f_a_out,
            f_b_weight=f_b_weight,
            beta_raw=beta_raw,
            lower_bound=gate_lower_bound,
            output_gate=output_gate,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )

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
        output_gate: torch.Tensor | None,
        norm_weight: torch.Tensor | None,
        norm_eps: float | None,
    ) -> torch.Tensor | None:
        """Whole-step decode attempt; ``None`` falls through to the shared flow.

        Sits before the conv update because a family's kernel may absorb it.

        GDN has no fused conv+gate+scan kernel, so the base returns None and
        the caller runs the shared conv update / qkv split / scan flow. KDA
        overrides this with a kernel that absorbs all three stages and may
        itself decline (unsupported shape or platform), which is why the
        sentinel is "not handled, continue" rather than a family switch.

        Args:
            mixed_qkv: Packed ``[T, key+key+value]`` conv input.
            conv_weights: Depthwise conv filters.
            conv_states: Conv-window component of this layer's state slab.
            ssm_states: Recurrent component of this layer's state slab.
            read_indices: Per-request state page holding position n-1.
            write_indices: Per-request state page receiving position n.
            f_a_out: Low-rank gate activation (KDA); None on GDN.
            f_b_weight: Second gate projection consumed inside the fusion.
            beta_raw: Raw per-head beta logits (KDA).
            A_log: Per-channel decay parameter.
            dt_bias: Per-channel timestep bias.
            value_dim: Pre-TP value width, used to derive the head count.
            attn_tp_size: Attention tensor-parallel size.
            head_v_dim: Value head dimension.
            lower_bound: KDA decay clamp.
            output_gate: Optional KDA gated-norm logits.
            norm_weight: Optional KDA output RMSNorm weight.
            norm_eps: Optional KDA output RMSNorm epsilon.

        Returns:
            The layer output when a fused kernel ran, else None.
        """
        return None

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
        a: torch.Tensor | None,
        b: torch.Tensor | None,
        g_raw: torch.Tensor | None,
        f_a_out: torch.Tensor | None,
        f_b_weight: torch.Tensor | None,
        beta_raw: torch.Tensor | None,
        lower_bound: float | None,
        output_gate: torch.Tensor | None,
        norm_weight: torch.Tensor | None,
        norm_eps: float | None,
    ) -> torch.Tensor:
        """Single-token recurrent scan over the split, conv'd projections.

        The boundary sits right after the shared split/reshape because that is
        where the families stop agreeing: GDN consumes scalar-per-head decay
        ``a``/``b`` through ``gdn_decode_step``, KDA a per-channel gate
        ``g_raw`` plus raw beta logits through ``kda_paged_decode``. Both
        return this backend's ``[1, B, Hv, V]`` decode-output convention.

        Args:
            query: ``[B, 1, H, K]`` conv'd, split query.
            key: ``[B, 1, H, K]`` conv'd, split key.
            value: ``[B, 1, Hv, V]`` conv'd, split value.
            ssm_states: Recurrent component of this layer's state slab.
            read_indices: Per-request state page holding position n-1.
            write_indices: Per-request state page receiving position n.
            A_log: Per-channel decay parameter.
            dt_bias: Per-channel timestep bias.
            a: GDN scalar-per-head decay input.
            b: GDN scalar-per-head beta input.
            g_raw: KDA per-channel gate, when the model precomputed it.
            f_a_out: KDA low-rank gate activation (gate GEMV source).
            f_b_weight: KDA second gate projection.
            beta_raw: KDA raw per-head beta logits.
            lower_bound: KDA decay clamp.
            output_gate: Optional KDA gated-norm logits.
            norm_weight: Optional KDA output RMSNorm weight.
            norm_eps: Optional KDA output RMSNorm epsilon.

        Returns:
            ``[1, B, Hv, V]`` layer output.
        """
        (
            decode_initial_indices,
            decode_output_indices,
            decode_solution,
        ) = _prepare_gdn_decode_state_path(
            ssm_states,
            read_indices,
            write_indices,
        )
        core_attn_out = gdn_decode_step(
            q=query,
            k=key,
            v=value,
            A_log=A_log,
            a=a.unsqueeze(1),
            dt_bias=dt_bias,
            b=b.unsqueeze(1),
            initial_state=ssm_states,
            initial_state_indices=decode_initial_indices,
            # Write to the out page, not the possibly shared input page.
            output_state_indices=decode_output_indices,
            use_qk_l2norm=True,
            solution=decode_solution,
        )
        # [B, 1, Hv, V] (pool/indices-major) -> [1, B, Hv, V], this backend's
        # decode-output convention (matches gdn_chunk_prefill's B=1-leading out).
        return core_attn_out.transpose(0, 1)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        forward_mode: ForwardMode,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        mixed_qkv = kwargs["mixed_qkv"]
        conv_weights = kwargs["conv_weights"]
        bias = kwargs["bias"]
        activation = kwargs["activation"]
        key_dim = kwargs["key_dim"]
        value_dim = kwargs["value_dim"]
        attn_tp_size = kwargs["attention_tp_size"]
        head_k_dim = kwargs["head_k_dim"]
        head_v_dim = kwargs["head_v_dim"]
        # Gating inputs are family-specific and are consumed by the scan seams:
        # scalar-per-head a/b here, a per-channel gate plus raw beta logits in
        # the subclass (A_log / dt_bias are per-channel for both).
        a = kwargs.get("a")
        b = kwargs.get("b")
        g_raw = kwargs.get("g_raw")
        f_a_out = kwargs.get("f_a_out")
        f_b_weight = kwargs.get("f_b_weight")
        beta_raw = kwargs.get("beta_raw")
        gate_lower_bound = kwargs.get("lower_bound")
        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        layer_id = kwargs["layer_id"]
        seq_len = kwargs["seq_len"]

        # `q` is None for hybrid linear-attn layers; the token count comes
        # from seq_len carried in kwargs.
        q_len_per_req = seq_len // bs if bs > 0 else 1
        is_target_verify = (
            forward_mode is not None
            and forward_mode.is_decode_or_idle()
            and not self.is_draft
            and q_len_per_req > 1
        )

        query_start_loc = self.forward_metadata.query_start_loc

        if is_target_verify:
            draft_token_num = kwargs.get(
                "draft_token_num", self.speculative_num_draft_tokens
            )
            batch_size = seq_len // draft_token_num
            output_indices = self.forward_metadata.mamba_output_indices
            state_in_blocks, _, conv_comp, ssm_comp = self._layer_state(layer_id)
            conv_scratch, ssm_scratch = self._verify_scratch[layer_id]
            fused_out = self._verify(
                mixed_qkv,
                conv_weights,
                conv_comp,
                conv_scratch,
                ssm_comp,
                ssm_scratch,
                state_in_blocks,
                output_indices,
                bias=bias,
                f_a_out=f_a_out,
                f_b_weight=f_b_weight,
                beta_raw=beta_raw,
                A_log=A_log,
                dt_bias=dt_bias,
                batch_size=batch_size,
                draft_token_num=draft_token_num,
                value_dim=value_dim,
                attn_tp_size=attn_tp_size,
                head_v_dim=head_v_dim,
                lower_bound=gate_lower_bound,
            )
            if fused_out is not None:
                return fused_out
            # Read the committed window and write per-position states into the
            # verify scratch. The accepted position is committed afterward.
            if layer_id == self._state_layer_ids()[0]:
                self._seed_verify_scratch_batched(batch_size, draft_token_num)
            init_rows = output_indices[:batch_size, 0] - 1
            conv_states = conv_scratch
            conv_read = init_rows
            conv_out = output_indices[:batch_size]
            # shouldn't use contiguous here, because causal_conv1d_update
            # support input non-contiguous
            mixed_qkv_reshaped = mixed_qkv.view(
                batch_size, draft_token_num, -1
            ).transpose(1, 2)
            mixed_qkv_processed = causal_conv1d_update(
                mixed_qkv_reshaped,
                conv_states,
                conv_weights,
                bias,
                activation,
                conv_state_indices=conv_read,
                output_state_indices=conv_out,
            )
            # needn't contiguous here.
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
        else:
            state_in_blocks, state_out_blocks, conv_states, ssm_states = (
                self._layer_state(layer_id)
            )
            state_out_long = state_out_blocks.to(torch.int64)
            recurrent_state, has_initial_states = _prepare_cache_prefill_state_inputs(
                conv_states,
                ssm_states,
                state_in_blocks,
                state_out_long,
            )
            conv_cache_indices = state_out_blocks
            extend_prefix_lens = kwargs.get("extend_prefix_lens")
            if extend_prefix_lens is None:
                extend_prefix_lens = self.forward_metadata.extend_prefix_lens
            extend_seq_lens_cpu = kwargs.get("extend_seq_lens_cpu")
            if extend_seq_lens_cpu is None:
                extend_seq_lens_cpu = self.forward_metadata.extend_seq_lens_cpu

            # Zero padded rows so garbage can't reach recurrent state (see scrub_padding_tail).
            num_real_tokens = seq_len
            if extend_seq_lens_cpu is not None:
                num_real_tokens = int(sum(int(x) for x in extend_seq_lens_cpu))
                scrub_padding_tail(num_real_tokens, mixed_qkv, a, b)

            mixed_qkv_t = mixed_qkv.transpose(0, 1)
            mixed_qkv = causal_conv1d_fn(
                mixed_qkv_t,
                conv_weights,
                bias,
                activation=activation,
                conv_states=conv_states,
                has_initial_state=has_initial_states,
                cache_indices=conv_cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=extend_seq_lens_cpu,
            ).transpose(0, 1)[:seq_len]

        key_split_dim = key_dim // attn_tp_size
        value_split_dim = value_dim // attn_tp_size
        num_heads = key_split_dim // head_k_dim
        num_value_heads = value_split_dim // head_v_dim

        replay_inputs = None
        if is_target_verify and self.replay_ssm:
            replay = self._gdn_replay
            layer_slot = replay.layer_ids.index(layer_id)
            # A_log and dt_bias are model-static; copy them once per layer
            # during warmup so CUDA graph capture records no copy nodes.
            if layer_id not in replay.initialized_layers:
                replay.parameters[layer_slot, 0].copy_(A_log)
                replay.parameters[layer_slot, 1].copy_(dt_bias)
                replay.initialized_layers.add(layer_id)
            replay_inputs = (
                replay.payload[layer_slot, :seq_len],
                a.view(seq_len, -1),
                b.view(seq_len, -1),
            )

        query, key, value = fused_qkv_split_gdn_prefill(
            mixed_qkv,
            num_q_heads=num_heads,
            num_k_heads=num_heads,
            num_v_heads=num_value_heads,
            head_q=head_k_dim,
            head_k=head_k_dim,
            head_v=head_v_dim,
            replay=replay_inputs,
        )

        if is_target_verify:
            core_attn_out = self._verify_scan(
                query,
                key,
                value,
                ssm_comp,
                ssm_scratch,
                state_in_blocks,
                output_indices,
                A_log=A_log,
                dt_bias=dt_bias,
                a=a,
                b=b,
                g_raw=g_raw,
                f_a_out=f_a_out,
                f_b_weight=f_b_weight,
                beta_raw=beta_raw,
                batch_size=batch_size,
                draft_token_num=draft_token_num,
                seq_len=seq_len,
                lower_bound=gate_lower_bound,
            )
        else:
            core_attn_out, last_recurrent_state = self._prefill_scan(
                query,
                key,
                value,
                recurrent_state,
                query_start_loc,
                A_log=A_log,
                dt_bias=dt_bias,
                a=a,
                b=b,
                g_raw=g_raw,
                f_a_out=f_a_out,
                f_b_weight=f_b_weight,
                beta_raw=beta_raw,
                seq_len=seq_len,
                num_real_tokens=num_real_tokens,
                lower_bound=gate_lower_bound,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
            )
            last_recurrent_state = last_recurrent_state.to(ssm_states.dtype, copy=False)
            # Extend indices never carry pad(-1), so this write is unguarded.
            ssm_states[state_out_long] = last_recurrent_state

        return core_attn_out

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
        bias: torch.Tensor | None,
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
        """Whole-round verify attempt; ``None`` falls through to the shared flow.

        Sits before the scratch seeding because a family's kernel may seed
        (or skip) the scratch itself.

        GDN has no fused verify kernel, so the base returns None and the caller
        seeds the verify scratch and runs the shared conv update. KDA overrides
        this with a kernel that fuses conv(+silu), the gate GEMV and the
        per-position recurrence — it also seeds itself, which is why the seam
        sits above the seeding rather than only around the scan.

        Args:
            mixed_qkv: Packed ``[T, key+key+value]`` conv input.
            conv_weights: Depthwise conv filters.
            conv_comp: Conv-window component of this layer's state slab.
            conv_scratch: Per-position conv-window verify scratch.
            ssm_comp: Recurrent component of this layer's state slab.
            ssm_scratch: Per-position recurrent verify scratch.
            state_in_blocks: Per-request committed-state page ids.
            output_indices: ``[bs, T]`` verify scratch row grid.
            bias: Conv bias; a fused path requires the bias-free conv.
            f_a_out: Low-rank gate activation (KDA); None on GDN.
            f_b_weight: Second gate projection consumed inside the fusion.
            beta_raw: Raw per-head beta logits (KDA).
            A_log: Per-channel decay parameter.
            dt_bias: Per-channel timestep bias.
            batch_size: Requests in this verify round.
            draft_token_num: Verified positions per request.
            value_dim: Pre-TP value width, used to derive the head count.
            attn_tp_size: Attention tensor-parallel size.
            head_v_dim: Value head dimension.
            lower_bound: KDA decay clamp.

        Returns:
            The layer output when a fused kernel ran, else None.
        """
        return None

    def _verify_scan(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        ssm_comp: torch.Tensor,
        ssm_scratch: torch.Tensor | None,
        state_in_blocks: torch.Tensor,
        output_indices: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
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
    ) -> torch.Tensor:
        """Per-position recurrent scan of a target-verify round.

        The scratch fallback writes one recurrent state per draft position so
        ``commit_verified_state`` can publish the accepted one. ReplaySSM reads
        the committed slab directly, suppresses speculative state writes, and
        reconstructs the accepted prefix during commit.

        Args:
            query: ``[1, seq_len, H, K]`` conv'd, split query.
            key: ``[1, seq_len, H, K]`` conv'd, split key.
            value: ``[1, seq_len, Hv, V]`` conv'd, split value.
            ssm_comp: Recurrent component of this layer's state slab.
            ssm_scratch: Per-position recurrent verify scratch, or None on
                ReplaySSM.
            state_in_blocks: Per-request committed-state page ids.
            output_indices: ``[bs, T]`` verify scratch row grid.
            A_log: Per-channel decay parameter.
            dt_bias: Per-channel timestep bias.
            a: GDN scalar-per-head decay input.
            b: GDN scalar-per-head beta input.
            g_raw: KDA per-channel gate, when the model precomputed it.
            f_a_out: KDA low-rank gate activation (gate GEMV source).
            f_b_weight: KDA second gate projection.
            beta_raw: KDA raw per-head beta logits.
            batch_size: Requests in this verify round.
            draft_token_num: Verified positions per request.
            seq_len: Total tokens in the round (``batch_size * T``).
            lower_bound: KDA decay clamp.

        Returns:
            ``[1, seq_len, Hv, V]`` layer output.
        """
        num_heads = query.shape[2]
        head_k_dim = query.shape[3]
        num_value_heads = value.shape[2]
        head_v_dim = value.shape[3]
        # Request-major varlen layout: [B, T, H, D] is a plain view, no movement.
        query_b = query.view(batch_size, draft_token_num, num_heads, head_k_dim)
        key_b = key.view(batch_size, draft_token_num, num_heads, head_k_dim)
        value_b = value.view(batch_size, draft_token_num, num_value_heads, head_v_dim)
        a_b = a.view(batch_size, draft_token_num, -1)
        b_b = b.view(batch_size, draft_token_num, -1)

        if self.replay_ssm:
            initial_state = ssm_comp
            initial_indices = state_in_blocks[:batch_size]
            output_state_indices = None
        else:
            initial_state = ssm_scratch
            initial_indices = output_indices[:batch_size, 0] - 1
            output_state_indices = output_indices
        (
            mtp_initial_indices,
            mtp_output_indices,
            mtp_solution,
        ) = _prepare_gdn_decode_state_path(
            initial_state,
            initial_indices,
            output_state_indices,
        )
        return gdn_decode_mtp(
            query_b,
            key_b,
            value_b,
            A_log=A_log,
            a=a_b,
            dt_bias=dt_bias,
            b=b_b,
            initial_state=initial_state,
            initial_state_indices=mtp_initial_indices,
            use_qk_l2norm=True,
            output_state_indices=mtp_output_indices,
            disable_state_update=self.replay_ssm,
            solution=mtp_solution,
        ).reshape(1, seq_len, num_value_heads, head_v_dim)

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
        a: torch.Tensor | None,
        b: torch.Tensor | None,
        g_raw: torch.Tensor | None,
        f_a_out: torch.Tensor | None,
        f_b_weight: torch.Tensor | None,
        beta_raw: torch.Tensor | None,
        seq_len: int,
        num_real_tokens: int,
        lower_bound: float | None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunked scan of an extend/prefill batch, from the gathered state.

        The caller owns the state plumbing on both sides (it gathers
        ``recurrent_state`` before and writes the returned final state to the
        out page after), so the seam is exactly the kernel call: GDN runs
        ``gdn_chunk_prefill`` over gates built from scalar a/b, KDA runs
        ``kda_paged_prefill`` over a per-channel gate under a selectable
        solution.

        Args:
            query: ``[1, seq_len, H, K]`` conv'd, split query.
            key: ``[1, seq_len, H, K]`` conv'd, split key.
            value: ``[1, seq_len, Hv, V]`` conv'd, split value.
            recurrent_state: Per-request initial recurrent state.
            query_start_loc: Varlen cumulative token offsets.
            A_log: Per-channel decay parameter.
            dt_bias: Per-channel timestep bias.
            a: GDN scalar-per-head decay input.
            b: GDN scalar-per-head beta input.
            g_raw: KDA per-channel gate, when the model precomputed it.
            f_a_out: KDA low-rank gate activation (gate GEMV source).
            f_b_weight: KDA second gate projection.
            beta_raw: KDA raw per-head beta logits.
            seq_len: Padded token extent of the batch.
            num_real_tokens: Token extent excluding the graph padding tail.
            lower_bound: KDA decay clamp.
            extend_seq_lens_cpu: Host-side per-sequence extend lengths whose
                prefix sum equals ``query_start_loc``. The KDA override
                forwards it so the CuteDSL wrapper can plan without a D2H
                read; the GDN scan plans on device and ignores it.

        Returns:
            ``(core_attn_out, last_recurrent_state)``.
        """
        head_k_dim = query.shape[3]
        beta = b.sigmoid()
        g = fused_gdn_gating(A_log, a, dt_bias)
        g = g.unsqueeze(0)
        beta = beta.unsqueeze(0)

        gdn_result = gdn_chunk_prefill(
            query,
            key,
            value,
            g,
            beta,
            scale=head_k_dim**-0.5,
            initial_state=recurrent_state,
            cu_seqlens=query_start_loc,
            qk_l2norm=True,
            output_final_state=True,
            output_h=False,
        )
        return gdn_result.out, gdn_result.final_state


class HybridLinearAttnBackend(AttentionBackend):
    """Hybrid backend that routes between full attention and linear attention by layer ID."""

    # Both sub-backends consume per-group tables (MHA: KV pages; Mamba:
    # dual-index state pages). Target verify publishes only the accepted
    # position.
    uses_cache_groups: bool = True

    def __init__(
        self,
        full_attn_backend: AttentionBackend,
        linear_attn_backend: MambaAttnBackend,
        full_attn_layers: list[int],
    ):
        self.device = full_attn_backend.device
        self.full_attn_layers = set(full_attn_layers)
        self.full_attn_backend = full_attn_backend
        self.linear_attn_backend = linear_attn_backend

    # The MLA full-attention sub-backend owns the spec-decode token width and
    # the chunked-prefill machinery. The DeepseekV3-style MLA layer forward
    # (reused by Kimi-K3) reads these off ``ctx.attn_backend`` -- which is this
    # hybrid wrapper -- so route them to the full-attention backend.
    @property
    def spec_num_tokens(self) -> int:
        return self.full_attn_backend.spec_num_tokens

    @property
    def chunked_prefill_metadata(self):
        return self.full_attn_backend.chunked_prefill_metadata

    @property
    def data_type(self):
        return self.full_attn_backend.data_type

    @property
    def supports_mla_projected_value_decode(self) -> bool:
        return self.full_attn_backend.supports_mla_projected_value_decode

    def override_num_extends(self, num_extends: int):
        return self.full_attn_backend.override_num_extends(num_extends)

    def forward_extend_chunked(self, *args, **kwargs):
        return self.full_attn_backend.forward_extend_chunked(*args, **kwargs)

    def advance_draft_forward_metadata(self, seq_lens: torch.Tensor) -> None:
        # Composite: the full-attention child owns the seq_lens the draft reads.
        self.full_attn_backend.advance_draft_forward_metadata(seq_lens)

    @property
    def cache_consumer_families(self) -> frozenset[str]:
        """Cache families consumed by the two child backends."""
        families: frozenset[str] = frozenset()
        for backend in (self.full_attn_backend, self.linear_attn_backend):
            families |= frozenset(getattr(backend, "cache_consumer_families", ()))
        return families

    def select_out_cache_loc(self, layer, out_cache_loc, forward_mode=None):
        """Route the write-location hook to the owning sub-backend by layer.

        Model-owned KV writers (MLA latent writes) call this through
        ``ctx.attn_backend`` — this wrapper for hybrid models — so the
        full-attention sub-backend's write locations must be reachable here.
        """
        return self._backend_for_layer(layer.layer_id).select_out_cache_loc(
            layer, out_cache_loc, forward_mode
        )

    def _backends(self):
        return [self.full_attn_backend, self.linear_attn_backend]

    def set_cache_pool(self, cache_pool) -> None:
        self.cache_pool = cache_pool
        for backend in self._backends():
            backend.set_cache_pool(cache_pool)

    def _backend_for_layer(self, layer_id: int) -> AttentionBackend:
        if self.linear_attn_backend is None or layer_id in self.full_attn_layers:
            return self.full_attn_backend
        return self.linear_attn_backend

    # ---- Metadata delegation ----

    def init_forward_metadata(self, *args, **kwargs):
        self.full_attn_backend.init_forward_metadata(*args, **kwargs)
        self.linear_attn_backend.init_forward_metadata(*args, **kwargs)

    def init_cuda_graph_state(self, max_bs: int, **kwargs):
        # kwargs (e.g. cache_group_specs, so the full backend sheds
        # state-family groups) are forwarded through the shared signature
        # filter: the full backend is user-selectable and may have a narrow
        # signature (e.g. TRTLLM MHA takes only (max_bs,)), and the mamba
        # backend keeps its narrow signature today.
        init_backend_cuda_graph_state(self.full_attn_backend, max_bs, **kwargs)
        init_backend_cuda_graph_state(self.linear_attn_backend, max_bs, **kwargs)

    def register_step_counter(self, step_counter):
        # Hybrid layerwise transfer needs one global step per model layer,
        # including both full-attention and mamba layers. Normal attention
        # dispatch records in this wrapper; model-owned chunked prefill bypasses
        # that dispatch, so its full-attention child needs the same counter.
        self.step_counter = step_counter
        self.full_attn_backend.register_step_counter(step_counter)

    def init_forward_metadata_capture_cuda_graph(self, *args, **kwargs):
        self.full_attn_backend.init_forward_metadata_capture_cuda_graph(*args, **kwargs)
        self.linear_attn_backend.init_forward_metadata_capture_cuda_graph(
            *args, **kwargs
        )

    def init_forward_metadata_replay_cuda_graph(self, *args, **kwargs):
        self.full_attn_backend.init_forward_metadata_replay_cuda_graph(*args, **kwargs)
        self.linear_attn_backend.init_forward_metadata_replay_cuda_graph(
            *args, **kwargs
        )

    def support_kv_cache_prewrite(
        self, forward_mode: ForwardMode | None = None
    ) -> bool:
        return self.full_attn_backend.support_kv_cache_prewrite(forward_mode)

    # ---- Forward dispatch ----

    @break_point
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc,
        token_to_kv_pool,
        forward_mode: ForwardMode,
        bs: int,
        save_kv_cache: bool = True,
        record_kv_cache: bool | None = None,
        **kwargs,
    ):
        """Dispatch one layer to its full-attention or GDN backend (the break point).

        Overrides the base forward, so it carries its own ``@break_point``;
        the frozen capture-time scalars (forward_mode/bs) are re-read from the
        ambient ctx (semantics: see breakable_cuda_graph). The GDN scan's
        batched [1, T, Hv, D] output is collapsed to z-shaped [T, Hv, D].
        """
        if forward_mode is None:
            return super().forward(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                forward_mode,
                bs,
                save_kv_cache,
                record_kv_cache=record_kv_cache,
                **kwargs,
            )

        # Frozen capture-time scalars, re-read live (see docstring); no-op in eager.
        amb = current_forward_ctx()
        if amb is not None:
            forward_mode = amb.forward_mode
            bs = amb.bs

        if forward_mode.is_idle():
            if layer is None:
                return torch.empty_like(kwargs["z"])
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)

        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        backend = self._backend_for_layer(layer_id)

        # See AttentionBackend.forward for the record_kv_cache contract; the step
        # is recorded in this wrapper (not the child backends) to keep one step
        # per model layer across full-attn + mamba. Idle already returned above.
        with self.record_pd_cache_step(forward_mode, save_kv_cache, record_kv_cache):
            if forward_mode.is_decode():
                ret = backend.forward_decode(
                    q,
                    k,
                    v,
                    layer,
                    out_cache_loc,
                    token_to_kv_pool,
                    bs,
                    save_kv_cache=save_kv_cache,
                    **kwargs,
                )
            else:
                ret = backend.forward_extend(
                    q,
                    k,
                    v,
                    layer,
                    out_cache_loc,
                    token_to_kv_pool,
                    bs,
                    save_kv_cache=save_kv_cache,
                    forward_mode=forward_mode,
                    **kwargs,
                )
        # Collapse the GDN scan's batched [1, T, Hv, D] to z-shaped (see docstring).
        if ret is not None and ret.dim() == 4:
            # Strictly [1, T, Hv, D]: a genuine B>1 must fail loud, not corrupt the handoff.
            assert (
                ret.shape[0] == 1
            ), f"GDN scan batched rank expected leading 1, got {ret.shape}"
            ret = ret.flatten(0, 1)
        return ret

    def forward_decode(
        self, q, k, v, layer, out_cache_loc, token_to_kv_pool, bs, **kwargs
    ):
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        return self._backend_for_layer(layer_id).forward_decode(
            q, k, v, layer, out_cache_loc, token_to_kv_pool, bs, **kwargs
        )

    def forward_extend(
        self, q, k, v, layer, out_cache_loc, token_to_kv_pool, bs, **kwargs
    ):
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        return self._backend_for_layer(layer_id).forward_extend(
            q, k, v, layer, out_cache_loc, token_to_kv_pool, bs, **kwargs
        )

    def update_mamba_state_after_mtp_verify(self, accepted_length, model):
        self.linear_attn_backend.commit_verified_state(accepted_length)
