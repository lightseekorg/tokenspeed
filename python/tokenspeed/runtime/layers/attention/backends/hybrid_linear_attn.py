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
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.attention import (
    GdnCheckpointLayout,
    gdn_chunk_prefill,
    gdn_decode_mtp,
    gdn_decode_step,
    kda_paged_decode,
    kda_paged_prefill,
    try_kda_fused_paged_decode,
    try_kda_fused_paged_verify,
)
from tokenspeed_kernel.ops.attention.triton.gdn_qkv_split import (
    fused_qkv_split_gdn_prefill,
)
from tokenspeed_kernel.ops.attention.triton.linear.chunk_delta_h import (
    CHUNK_SIZE as FLA_CHUNK_SIZE,
)
from tokenspeed_kernel.ops.attention.triton.linear.index import (
    set_total_chunks_hint,
    set_total_chunks_hint_uniform,
)

from tokenspeed.runtime.configs.flat_cache_runtime import flat_cache_debug_enabled
from tokenspeed.runtime.configs.paged_cache_spec import LINEAR_ATTENTION
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
from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from tokenspeed.runtime.layers.attention.linear.gdn import fused_gdn_gating
from tokenspeed.runtime.layers.attention.linear.mamba_state_scatter_triton import (
    fused_mamba_state_copy,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.configs.base import BaseAttnConfig
    from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
    from tokenspeed.runtime.layers.paged_attention import PagedAttention

# Flat KV-cache group id carrying GDN/mamba2 state pages.
_STATE_GROUP_ID = LINEAR_ATTENTION


def _mask_fresh_initial_state(
    recurrent_state: torch.Tensor,
    has_initial_states: torch.Tensor | None,
) -> torch.Tensor:
    """Zero the initial recurrent state of sequences that have no history.

    On the flat cache path a fresh sequence's ``state_in`` page is freshly
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
class _StatePageIndexPlan:
    page_size: int
    before: torch.Tensor
    after: torch.Tensor
    has_history: torch.Tensor
    in_slots: torch.Tensor
    out_slots: torch.Tensor


def _compute_state_page_index_plan(
    page_size: int,
    seq_lens_before: torch.Tensor,
    seq_lens_after: torch.Tensor,
) -> _StatePageIndexPlan:
    before = seq_lens_before.to(torch.int64)
    after = seq_lens_after.to(torch.int64)
    in_slots = torch.div(before - 1, page_size, rounding_mode="floor").clamp_(min=0)
    out_slots = torch.div(after - 1, page_size, rounding_mode="floor")
    return _StatePageIndexPlan(
        page_size=page_size,
        before=before,
        after=after,
        has_history=before > 0,
        in_slots=in_slots,
        out_slots=out_slots,
    )


def _gather_state_page_indices(
    rows: torch.Tensor,
    plan: _StatePageIndexPlan,
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
                "state paging: out page slot exceeds flat table width "
                f"{max_slots} (page_size={plan.page_size})"
            )
        if bool((state_in[plan.has_history] <= 0).any()):
            raise ValueError(
                "state paging: in page is a pad (-1) or hole (0) for a "
                "request with history; reading it would silently resume "
                f"from the zero state (flat {group_id!r} table)"
            )
        if bool((state_out <= 0).any()):
            raise ValueError(
                "state paging: out page is a pad (-1) or hole (0); the "
                "request's working state page must be present in the flat "
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
                f"corrupt every branch sharing it (flat {group_id!r} table)"
            )
        # The <= 0 raise above guarantees every state_out entry is positive.
        if torch.unique(state_out).numel() != state_out.numel():
            raise ValueError(
                f"state out pages must be unique per batch (flat {group_id!r} "
                "table): two requests writing one working state page would "
                "silently clobber each other"
            )
    return state_in.to(torch.int32), state_out.to(torch.int32)


def compute_state_page_indices(
    rows: torch.Tensor,
    page_size: int,
    seq_lens_before: torch.Tensor,
    seq_lens_after: torch.Tensor,
    *,
    validate: bool = True,
    group_id: str = _STATE_GROUP_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dual-index state pages: in = page of position n-1 (0/null when no
    history), out = page of the step's last position. rows: [bs, max_pages]
    int32 page ids (-1 pad, 0 hole). Within a page in == out (in-place
    evolution); crossing a boundary reads the old page and writes the new
    one; resuming from a prefix hit reads the claimed snapshot page and
    writes the fresh working page.

    Args:
        rows: ``[bs, max_pages]`` int32 page-id table of one state group.
        page_size: Tokens per state page (the flat block size ``P``).
        seq_lens_before: Per-request token count before this forward.
        seq_lens_after: Per-request token count after this forward.
        validate: Run the host-synchronizing write-side checks.
        group_id: State group the table belongs to; only used to attribute
            validation errors (multi-group KDA runs this once per group).

    Returns:
        ``(state_in, state_out)`` int32 page ids per request.
    """
    plan = _compute_state_page_index_plan(page_size, seq_lens_before, seq_lens_after)
    return _gather_state_page_indices(rows, plan, validate=validate, group_id=group_id)


def compute_state_page_indices_by_group(
    rows_by_group: dict[str, torch.Tensor],
    page_size: int,
    seq_lens_before: torch.Tensor,
    seq_lens_after: torch.Tensor,
    *,
    validate: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Resolve state pages for groups sharing one logical page geometry."""
    plan = _compute_state_page_index_plan(page_size, seq_lens_before, seq_lens_after)
    state_in_by_group = {}
    state_out_by_group = {}
    for group_id, rows in rows_by_group.items():
        state_in, state_out = _gather_state_page_indices(
            rows,
            plan,
            validate=validate,
            group_id=group_id,
        )
        state_in_by_group[group_id] = state_in
        state_out_by_group[group_id] = state_out
    return state_in_by_group, state_out_by_group


def _prepare_flat_prefill_state_inputs(
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    state_in_pages: torch.Tensor,
    state_out_pages: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize logical-zero fresh state without reading physical page 0."""
    has_initial_state = state_in_pages > 0
    safe_input_pages = torch.where(
        has_initial_state, state_in_pages, state_out_pages
    ).to(torch.int64)
    state_out_pages = state_out_pages.to(torch.int64)

    # A fresh row copies its working page to itself. Only a resumed row reads
    # the checkpoint named by state_in_pages.
    conv_states[state_out_pages] = conv_states[safe_input_pages]
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
    scheduler-owned row in the radix Mamba pool. Until that kernel supports a
    padding mask, route BF16 state through Triton and keep ``-1`` unchanged.
    """
    solution = "triton" if ssm_states.dtype == torch.bfloat16 else None
    return initial_state_indices, output_state_indices, solution


@dataclass
class MambaForwardMetadata:
    query_start_loc: torch.Tensor | None
    mamba_cache_indices: torch.Tensor
    mamba_output_indices: torch.Tensor | None = None
    mamba_req_pool_indices: torch.Tensor | None = None
    extend_prefix_lens: torch.Tensor | None = None
    extend_seq_lens_cpu: torch.Tensor | None = None
    # Pre-computed src/dst indices for extracting Mamba prefix-cache snapshots.
    track_ssm_h_src: torch.Tensor | None = None
    track_ssm_h_src_fla: torch.Tensor | None = None
    track_ssm_h_dst: torch.Tensor | None = None
    track_conv_indices: torch.Tensor | None = None
    track_ssm_final_src: torch.Tensor | None = None
    track_ssm_final_dst: torch.Tensor | None = None
    # Flat path (state slabs): dual in/out page indices per request.
    state_in_pages: dict[str, torch.Tensor] | None = None
    state_out_pages: dict[str, torch.Tensor] | None = None
    # FlatKV per-state-group KDA metadata is gathered once per group and batch;
    # layers select their entry via ``pool.group_id_for_layer(layer_id)``.
    state_in_pages_by_group: dict[str, torch.Tensor] | None = None
    state_out_pages_by_group: dict[str, torch.Tensor] | None = None


class LayerMappedKVPool:
    """Wraps a KV pool to map global layer IDs to internal pool indices.

    For hybrid models, only full attention layers have KV cache. This wrapper
    translates global layer IDs (e.g., 3, 7, 11) to pool indices (0, 1, 2).
    """

    def __init__(
        self, inner_pool: BaseTokenToKVPool, full_attention_layer_ids: list[int]
    ):
        self.inner = inner_pool
        self.layer_ids = list(full_attention_layer_ids)
        self.layer_map = {
            global_id: pool_idx
            for pool_idx, global_id in enumerate(full_attention_layer_ids)
        }
        # Expose page_size from inner pool for the scheduler
        self.page_size = getattr(inner_pool, "page_size", 1)

    def _map(self, layer_id: int) -> int:
        return self.layer_map.get(layer_id, layer_id)

    @contextmanager
    def _mapped(self, layer):
        """Temporarily remap ``layer.layer_id`` to its inner-pool slot."""
        orig = layer.layer_id
        layer.layer_id = self._map(orig)
        try:
            yield
        finally:
            layer.layer_id = orig

    def set_kv_buffer(
        self,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor | None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
    ):
        with self._mapped(layer):
            self.inner.set_kv_buffer(layer, out_cache_loc, k, v, k_scale, v_scale)

    def get_kv_buffer(self, layer_id: int):
        return self.inner.get_kv_buffer(self._map(layer_id))

    def get_key_buffer(self, layer_id: int):
        return self.inner.get_key_buffer(self._map(layer_id))

    def get_value_buffer(self, layer_id: int):
        return self.inner.get_value_buffer(self._map(layer_id))

    # MLA pools index their per-layer kv_buffer by ``layer.layer_id`` directly.
    # In a hybrid model the inner MLA pool only holds the full-attention layers,
    # so the global id must be mapped to its pool slot first (mirrors
    # ``set_kv_buffer``). Reached via the DeepseekV3-style MLA chunked-prefill
    # path (Kimi-K3).
    def set_mla_kv_buffer(self, layer, loc, cache_k_nope, cache_k_rope):
        # Prefill breakable-graph padding contract: the dummy-batch capture (and
        # bucket-padding rows) whose ``out_cache_loc`` is the reserved
        # ``dummy_kv_slot`` can carry NaN into this fp8 KV write. The paged MLA
        # decode kernel reads that shared dummy slot through the zero-padded
        # block-table entries and computes ``q·k`` BEFORE applying the causal
        # mask, so the NaN survives it (``NaN + -inf = NaN``) and poisons a live
        # row's softmax -> NaN logits -> token 0. Eager prefill leaves the dummy
        # slot finite (``q·0`` masks cleanly), which is why the bug only appears
        # with the prefill graph on. Ask the cache writer to sanitize in-kernel
        # so real rows stay bitwise unchanged without allocating two temporary
        # tensors or launching two separate nan_to_num kernels.
        with self._mapped(layer):
            self.inner.set_mla_kv_buffer(
                layer,
                loc,
                cache_k_nope,
                cache_k_rope,
                sanitize=True,
            )

    def get_mla_kv_buffer(self, layer, loc, dst_dtype=None):
        with self._mapped(layer):
            return self.inner.get_mla_kv_buffer(layer, loc, dst_dtype)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class SimpleMambaPool:
    """Mamba state pool indexed by scheduler-assigned cache slots."""

    def __init__(
        self,
        size: int,
        num_mamba_layers: int,
        conv_state_shape: tuple,
        temporal_state_shape: tuple,
        conv_dtype: torch.dtype,
        ssm_dtype: torch.dtype,
        mamba_layer_ids: list[int],
        device: str,
        page_size: int = 1,
        speculative_num_draft_tokens: int = 0,
        max_req_pool_size: int = 0,
    ):
        self.size = size
        self.device = device
        self.mamba_layer_ids = list(mamba_layer_ids)
        self.page_size = page_size
        self.mamba_map = {layer_id: i for i, layer_id in enumerate(mamba_layer_ids)}
        self.max_req_pool_size = max_req_pool_size

        # Base slots (working + checkpoint) are allocated by C++ scheduler.
        # Python-only draft rows live after the scheduler-owned range and are
        # addressed by normal row indices in the same tensors.
        self.base_size = size
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        self.current_input_size = (
            max_req_pool_size + 1 if max_req_pool_size > 0 else size
        )
        self.draft_slots_per_req = max(0, speculative_num_draft_tokens - 1)
        self.draft_base = size
        self.draft_total_slots = self.current_input_size * self.draft_slots_per_req
        total_size = size + self.draft_total_slots
        self.total_size = total_size

        # Allocate conv state: (num_mamba_layers, total_size, conv_dim, state_len)
        self.conv_state = torch.zeros(
            num_mamba_layers,
            total_size,
            *conv_state_shape,
            dtype=conv_dtype,
            device=device,
        )
        # temporal_state_shape is already K-last [Hv, V, K], matching the
        # native GDN decode/MTP/chunk-prefill state layout.
        self.ssm_state = torch.zeros(
            num_mamba_layers,
            total_size,
            *temporal_state_shape,
            dtype=ssm_dtype,
            device=device,
        )

        self.mamba_cache = (self.conv_state, self.ssm_state)
        self.layer_transfer_counter = None

        self.current_input_indices = torch.full(
            (self.current_input_size,), -1, dtype=torch.int32, device=device
        )

    def get_mamba_indices(self, mamba_pool_indices: torch.Tensor) -> torch.Tensor:
        """Return mamba cache indices directly (allocated by C++ scheduler)."""
        return mamba_pool_indices.to(torch.int32)

    @staticmethod
    @torch.compile(dynamic=True)
    def _build_mtp_output_indices_kernel(
        output_indices: torch.Tensor,
        req_pool_indices: torch.Tensor,
        working_indices: torch.Tensor,
        draft_base: int,
        draft_slots_per_req: int,
        draft_token_num: int,
    ) -> None:
        """Fused fill of MTP target-verify output index table.

        Inductor fuses the working-column write and the draft-grid write into
        as few elementwise kernels as possible.  The host-side early returns
        (draft_token_num<=0, ``out is None``) are kept in the wrapper.
        """
        bs = working_indices.shape[0]
        working = working_indices.to(torch.int32)
        valid = working >= 0
        output_indices[:, 0] = torch.where(valid, working, -1)

        if draft_token_num > 1 and draft_slots_per_req > 0:
            req = req_pool_indices[:bs].to(torch.int32)
            steps = torch.arange(
                draft_token_num - 1, dtype=torch.int32, device=working.device
            )
            draft = draft_base + req[:, None] * draft_slots_per_req + steps[None, :]
            output_indices[:, 1:] = torch.where(
                valid[:, None] & (req >= 0)[:, None],
                draft,
                -1,
            )

    def get_mtp_output_indices(
        self,
        req_pool_indices: torch.Tensor,
        working_indices: torch.Tensor,
        draft_token_num: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build per-request target-verify outputs: [working, draft0, ...]."""
        bs = working_indices.shape[0]
        if out is not None:
            output_indices = out
            output_indices.fill_(-1)
        else:
            output_indices = torch.full(
                (bs, draft_token_num),
                -1,
                dtype=torch.int32,
                device=working_indices.device,
            )
        if draft_token_num <= 0:
            return output_indices

        self._build_mtp_output_indices_kernel(
            output_indices,
            req_pool_indices,
            working_indices,
            self.draft_base,
            self.draft_slots_per_req,
            draft_token_num,
        )
        return output_indices

    @staticmethod
    @torch.compile(dynamic=True)
    def _get_current_input_indices_kernel(
        req_pool_indices: torch.Tensor,
        working_indices: torch.Tensor,
        current_input_indices_buf: torch.Tensor,
        current_input_size: int,
    ) -> torch.Tensor:
        """Fused gather + masked-where for the no-COW path."""
        n = working_indices.shape[0]
        req = req_pool_indices[:n].to(torch.int32)
        working = working_indices.to(torch.int32)
        valid = (working >= 0) & (req >= 0)
        safe = req.clamp(0, current_input_size - 1).to(torch.int64)
        stored = current_input_indices_buf[safe]
        current = torch.where(valid & (stored >= 0), stored, working)
        current = torch.where(valid, current, torch.full_like(current, -1))
        return current

    @staticmethod
    @torch.compile(dynamic=True)
    def _get_current_input_indices_with_cow_kernel(
        req_pool_indices: torch.Tensor,
        working_indices: torch.Tensor,
        cow_src_indices: torch.Tensor,
        current_input_indices_buf: torch.Tensor,
        current_input_size: int,
    ) -> torch.Tensor:
        """Fused gather + masked-where for the COW path."""
        n = working_indices.shape[0]
        req = req_pool_indices[:n].to(torch.int32)
        working = working_indices.to(torch.int32)
        cow = cow_src_indices[:n].to(torch.int32)
        valid = (working >= 0) & (req >= 0)
        safe = req.clamp(0, current_input_size - 1).to(torch.int64)
        stored = current_input_indices_buf[safe]
        current = torch.where(valid & (stored >= 0), stored, working)
        current = torch.where(valid, current, torch.full_like(current, -1))
        current = torch.where(
            (cow >= 0) & valid & (current == working),
            cow,
            current,
        )
        return current

    def get_current_input_indices(
        self,
        req_pool_indices: torch.Tensor,
        working_indices: torch.Tensor,
        cow_src_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the row each request should read at the start of target verify."""
        if cow_src_indices is None:
            return self._get_current_input_indices_kernel(
                req_pool_indices,
                working_indices,
                self.current_input_indices,
                self.current_input_size,
            )
        return self._get_current_input_indices_with_cow_kernel(
            req_pool_indices,
            working_indices,
            cow_src_indices,
            self.current_input_indices,
            self.current_input_size,
        )

    def reset_current_inputs(
        self, req_pool_indices: torch.Tensor, working_indices: torch.Tensor
    ) -> None:
        """Mark freshly allocated/reused scheduler slots as canonical."""
        req_pool_indices = req_pool_indices[: working_indices.shape[0]].to(torch.int32)
        working_indices = working_indices.to(torch.int32)
        self.current_input_indices[req_pool_indices.long()] = working_indices

    @staticmethod
    @torch.compile(dynamic=True)
    def _update_current_inputs_after_verify_kernel(
        req_pool_indices: torch.Tensor,
        output_indices: torch.Tensor,
        accepted_lengths: torch.Tensor,
        current_input_indices: torch.Tensor,
        max_col: int,
    ) -> None:
        """Fused gather-scatter for the after-verify input pointer update.

        Inductor fuses clamp/arange/sub/dtype-convert into a single elementwise
        kernel; the gather and the in-place scatter on ``current_input_indices``
        each remain a single kernel.  All tensors stay on GPU; no host sync.
        """
        n = accepted_lengths.shape[0]
        req = req_pool_indices[:n].to(torch.int64)
        idx = (accepted_lengths.clamp(min=1, max=max_col) - 1).to(torch.int64)
        rows = torch.arange(n, device=accepted_lengths.device, dtype=torch.int64)
        selected = output_indices[rows, idx].to(torch.int32)
        current_input_indices[req] = selected

    def update_current_inputs_after_verify(
        self,
        req_pool_indices: torch.Tensor,
        output_indices: torch.Tensor,
        accepted_lengths: torch.Tensor,
    ) -> None:
        if output_indices is None or output_indices.numel() == 0:
            return
        self._update_current_inputs_after_verify_kernel(
            req_pool_indices,
            output_indices,
            accepted_lengths,
            self.current_input_indices,
            output_indices.shape[1],
        )

    def register_layer_transfer_counter(self, layer_transfer_counter):
        self.layer_transfer_counter = layer_transfer_counter

    def get_mamba_params(self, layer_id: int):
        """Return per-layer cache slices."""
        internal_idx = self.mamba_map[layer_id]
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(internal_idx)
        return [self.mamba_cache[i][internal_idx] for i in range(len(self.mamba_cache))]

    def get_mamba_params_all_layers(self):
        """Return all layers for all cache components."""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(self.conv_state.shape[0] - 1)
        return [self.mamba_cache[i] for i in range(len(self.mamba_cache))]

    def get_contiguous_buf_infos(self):
        """Return per-layer mamba cache buffers for disaggregated transfer."""
        data_ptrs = []
        data_lens = []
        item_lens = []
        for cache in self.mamba_cache:
            for layer_id in range(cache.shape[0]):
                layer_cache = cache[layer_id]
                data_ptrs.append(layer_cache.data_ptr())
                data_lens.append(layer_cache.nbytes)
                item_lens.append(layer_cache[0].nbytes)
        return data_ptrs, data_lens, item_lens

    def get_contiguous_buf_unit_lens(self):
        unit_lens = []
        for cache in self.mamba_cache:
            for layer_id in range(cache.shape[0]):
                layer_cache = cache[layer_id]
                unit_lens.append(layer_cache[0, 0].nbytes)
        return unit_lens

    def get_contiguous_buf_layer_ids(self):
        """Return global layer ids aligned with get_contiguous_buf_infos()."""
        return self.mamba_layer_ids * len(self.mamba_cache)


class MambaAttnBackend(AttentionBackend):
    """Attention backend for Mamba/GDN linear attention layers."""

    # This backend consumes state-family FlatKV tables through dual-index state
    # paging; history-family groups belong to the full-attention sub-backend.
    # The hybrid wrapper unions the sub-backends' declarations, so a Kimi-K3
    # contract (history + state) is covered once both consumers exist.
    flat_cache_consumer_families = frozenset({"state"})

    def __init__(
        self,
        config: BaseAttnConfig,
        is_kda: bool = False,
        kda_backend: str = "auto",
    ):
        super().__init__(config)
        self.pad_slot_id = -1
        # Kernel family: GDN (scalar decay) or KDA (per-channel, Kimi-K3).
        self.is_kda = is_kda
        # KDA prefill kernel policy, resolved by the registry: "auto" picks
        # the fastest available kernel (cutedsl_kda > flashkda > fla), "fla" forces
        # the portable scan (decode is unaffected either way).
        self.kda_backend = "fla"
        if self.is_kda:
            self.kda_backend = (kda_backend or "auto").strip().lower()
            if self.kda_backend == "auto":
                self.kda_backend = "fla"
            if self.kda_backend not in ("fla", "flashkda", "cutedsl_kda"):
                raise ValueError(
                    "--kda-backend must be one of auto, fla, flashkda, cutedsl_kda; "
                    f"got {self.kda_backend!r}"
                )
            logger.info(
                "KDA prefill routes through %s; decode remains on the "
                "platform-selected kernels",
                self.kda_backend,
            )
        self.forward_metadata: MambaForwardMetadata = None
        self.state_indices_list = []
        self.query_start_loc_list = []
        self.cached_cuda_graph_decode_query_start_loc: torch.Tensor = None
        self.cached_cuda_graph_verify_query_start_loc: torch.Tensor = None
        self.output_indices_list = []
        self.speculative_num_draft_tokens = getattr(
            config, "speculative_num_draft_tokens", 0
        )
        self.pool: SimpleMambaPool = None
        # Flat path (state slabs on the KV pool, dual in/out page indexing).
        self.kv_pool = None
        self.flat_state_active = False
        self._flat_state_page_size = 1
        self.flat_state_in_list: list[dict[str, torch.Tensor]] = []
        self.flat_state_out_list: list[dict[str, torch.Tensor]] = []
        self._state_group_ids: tuple[str, ...] = ()
        self._state_group_for_layer: dict[int, str] = {}
        # FlatKV contract path: state group ids in
        # contract order; empty when not contract-bound.
        self._flat_contract_bound = False
        self._flat_state_group_ids: tuple[str, ...] = ()
        # FlatKV CUDA-graph buffers: one persistent dual-index
        # (state_in/state_out) [bs] buffer per state group and captured batch
        # size. Values are keyed by group ID and indexed by ``bs - 1``.
        self.flat_state_in_by_group: dict[str, list[torch.Tensor]] = {}
        self.flat_state_out_by_group: dict[str, list[torch.Tensor]] = {}

    def set_pool(self, pool: SimpleMambaPool):
        self.pool = pool

    def set_kv_pool(self, kv_pool) -> None:
        """Bind the (layer-mapped) KV pool. Two structurally gated flat modes:

        - FlatKV contract pool: the pool publishes a
          ``runtime_contract``. Every ``family="state"``
          group becomes a KDA state group; the pool must resolve layers
          (``group_id_for_layer``) and per-layer component views
          (``get_component``). A contract with no state group is a hard
          error — never a silent SimpleMambaPool fallback.
        - Legacy single-group state slabs (GDN/Qwen): flat state paging turns
          on iff the pool allocated state slabs AND publishes the state cache
          group — publication (paged_cache_spec.publish_paged_cache_groups)
          is the runtime signal that flat block tables will actually be
          delivered (radix ext / spec decode never publish).
        """
        self.kv_pool = kv_pool
        contract = getattr(kv_pool, "runtime_contract", None)
        if contract is not None:
            state_group_ids = tuple(
                spec.group_id
                for spec in contract.group_specs
                if getattr(spec, "family", "history") == "state"
            )
            if not state_group_ids:
                raise RuntimeError(
                    "MambaAttnBackend: the KV pool publishes a FlatKV runtime "
                    "contract without a state-family cache group; recurrent "
                    "layers cannot be served, and a SimpleMambaPool fallback "
                    "is forbidden"
                )
            if not callable(
                getattr(kv_pool, "group_id_for_layer", None)
            ) or not callable(getattr(kv_pool, "get_component", None)):
                raise RuntimeError(
                    "MambaAttnBackend: a contract-bound KV pool must expose "
                    "group_id_for_layer() and get_component()"
                )
            self._flat_contract_bound = True
            self._flat_state_group_ids = state_group_ids
            self.flat_state_active = True
            self._flat_state_page_size = int(contract.block_size)
            return
        self._flat_contract_bound = False
        self._flat_state_group_ids = ()
        specs = getattr(kv_pool, "paged_cache_group_specs", ())
        layer_types = tuple(getattr(kv_pool, "_layer_types", ()))
        layer_group_ids = tuple(getattr(kv_pool, "layer_cache_group_ids", ()))
        if layer_group_ids and len(layer_group_ids) != len(layer_types):
            raise ValueError("layer cache group ids must align with layer types")
        if not layer_group_ids:
            layer_group_ids = layer_types
        self._state_group_for_layer = {
            layer_id: layer_group_ids[layer_id]
            for layer_id, label in enumerate(layer_types)
            if label == LINEAR_ATTENTION
        }
        self._state_group_ids = tuple(
            dict.fromkeys(self._state_group_for_layer.values())
        )
        published_group_ids = {str(spec.group_id) for spec in specs}
        self.flat_state_active = (
            bool(getattr(kv_pool, "state_slabs", None))
            and bool(self._state_group_ids)
            and set(self._state_group_ids).issubset(published_group_ids)
        )
        self._flat_state_page_size = int(getattr(kv_pool, "page_size", 1))

    def _state_page_bounds(
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

    def _flat_state_pages(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        kwargs: dict,
        *,
        validate: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(state_in, state_out) page ids for this forward, from the flat
        state-group block table (legacy single-group state-slab path).

        validate: explicit True/False wins; None (the hot-path default)
        validates only under TOKENSPEED_FLAT_DEBUG=1 (the checks host-sync).
        """
        if validate is None:
            validate = flat_cache_debug_enabled()
        flat_tables = kwargs.get("flat_block_tables")
        if not flat_tables or _STATE_GROUP_ID not in flat_tables:
            raise RuntimeError(
                "MambaAttnBackend: flat state paging is active (pool "
                f"publishes the {_STATE_GROUP_ID!r} group with state slabs) "
                "but flat_block_tables is missing the group "
                f"(got {sorted(flat_tables) if flat_tables else flat_tables!r})"
            )
        rows = flat_tables[_STATE_GROUP_ID]
        before, after = self._state_page_bounds(bs, seq_lens, forward_mode, kwargs)
        return compute_state_page_indices(
            rows,
            self._flat_state_page_size,
            before,
            after,
            validate=validate,
        )

    def _flat_state_pages_by_group(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        kwargs: dict,
        *,
        validate: bool | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Resolve every legacy LCM state group with one slot calculation."""
        flat_tables = kwargs.get("flat_block_tables")
        missing_groups = [
            group_id
            for group_id in self._state_group_ids
            if not flat_tables or group_id not in flat_tables
        ]
        if missing_groups:
            raise RuntimeError(
                "MambaAttnBackend: flat state paging is active (pool "
                f"publishes state groups {self._state_group_ids!r}) but "
                "flat_block_tables is missing groups "
                f"{missing_groups!r} "
                f"(got {sorted(flat_tables) if flat_tables else flat_tables!r})"
            )
        if validate is None:
            validate = flat_cache_debug_enabled()
        before, after = self._state_page_bounds(bs, seq_lens, forward_mode, kwargs)
        return compute_state_page_indices_by_group(
            {group_id: flat_tables[group_id] for group_id in self._state_group_ids},
            self._flat_state_page_size,
            before,
            after,
            validate=validate,
        )

    def _state_pages_for_layer(
        self, layer_id: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.forward_metadata.state_in_pages is None:
            return None, None
        group_id = self._state_group_for_layer[layer_id]
        return (
            self.forward_metadata.state_in_pages[group_id],
            self.forward_metadata.state_out_pages[group_id],
        )

    def _flat_mamba_layer_ids(self) -> list[int]:
        """Recurrent layer ids backed by the active Flat state pool."""
        if not self._flat_contract_bound:
            return sorted(self._state_group_for_layer)
        state_groups = set(self._flat_state_group_ids)
        return sorted(
            layer_id
            for layer_id, group_id in self.kv_pool._group_ids_by_layer.items()
            if group_id in state_groups
        )

    def _flat_state_groups(self) -> tuple[str, ...]:
        if self._flat_contract_bound:
            return self._flat_state_group_ids
        return self._state_group_ids

    def _flat_state_group_for(self, layer_id: int) -> str:
        if self._flat_contract_bound:
            return self.kv_pool.group_id_for_layer(layer_id)
        return self._state_group_for_layer[layer_id]

    def _flat_state_components(
        self, layer_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._flat_contract_bound:
            return (
                self.kv_pool.get_component(layer_id, "conv_state"),
                self.kv_pool.get_component(layer_id, "recurrent_state"),
            )
        return self.kv_pool.get_state_buffers(layer_id)

    def _flat_verify_state_pages(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        draft_token_num: int,
        kwargs: dict,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        """Target-verify flat state paging: per-group committed-state pages.

        Verify reads the state at the last COMMITTED position
        (``seq_lens - draft_token_num``); per-position outputs go to the
        verify scratch, and the accepted state is committed back by
        ``update_mamba_state_after_mtp_verify``. Returns the per-group in
        pages, the committed lengths, and the per-group flat tables (kept
        for the commit's dynamic page resolve).
        """
        committed = (seq_lens[:bs].to(torch.int64) - draft_token_num).clamp_min(0)
        in_slots = torch.div(
            (committed - 1).clamp_min(0),
            self._flat_state_page_size,
            rounding_mode="floor",
        )
        has_history = committed > 0
        state_in_pages: dict[str, torch.Tensor] = {}
        tables: dict[str, torch.Tensor] = {}
        if self._flat_contract_bound:
            flat_cache_metadata = kwargs.get("flat_cache_metadata")
            if flat_cache_metadata is None:
                raise RuntimeError(
                    "flat target-verify requires flat cache metadata on the forward"
                )
            flat_cache_forward_op = kwargs.get("flat_cache_forward_op")
            rows_by_group = {
                group_id: flat_cache_metadata.require_table(
                    group_id, active_forward_op=flat_cache_forward_op
                )
                for group_id in self._flat_state_groups()
            }
        else:
            flat_block_tables = kwargs.get("flat_block_tables")
            missing = [
                group_id
                for group_id in self._flat_state_groups()
                if not flat_block_tables or group_id not in flat_block_tables
            ]
            if missing:
                raise RuntimeError(
                    f"flat target-verify is missing state tables {missing!r}"
                )
            rows_by_group = {
                group_id: flat_block_tables[group_id]
                for group_id in self._flat_state_groups()
            }
        for group_id, rows in rows_by_group.items():
            slots_safe = in_slots.clamp(min=0, max=rows.shape[1] - 1)
            pages = rows[:bs].gather(1, slots_safe.unsqueeze(1)).squeeze(1)
            pages = torch.where(has_history, pages, torch.full_like(pages, -1)).to(
                torch.int32
            )
            state_in_pages[group_id] = pages
            tables[group_id] = rows
        return state_in_pages, committed, tables

    def _ensure_verify_scratch(self, bs: int, draft_token_num: int) -> None:
        """Lazily allocate per-group verify scratch: one init row plus
        ``draft_token_num`` per-position rows per request, for both the conv
        window and the recurrent state (rollback source for partial accepts).
        Sized once at the max the backend can see; graph warmup runs this
        path eagerly before capture."""
        max_bs = max(len(self.state_indices_list), bs)
        rows_needed = max_bs * (draft_token_num + 1)
        scratch = getattr(self, "_verify_scratch", None)
        if scratch is not None and next(iter(scratch.values()))[0].shape[0] >= (
            rows_needed
        ):
            return
        self._verify_scratch = {}
        self._verify_copy_tables = None
        for layer_id in self._flat_mamba_layer_ids():
            conv, ssm = self._flat_state_components(layer_id)
            self._verify_scratch[layer_id] = (
                torch.zeros(
                    (rows_needed, *conv.shape[1:]),
                    dtype=conv.dtype,
                    device=conv.device,
                ),
                torch.zeros(
                    (rows_needed, *ssm.shape[1:]),
                    dtype=ssm.dtype,
                    device=ssm.device,
                ),
            )

    def preallocate_flat_verify_workspace(
        self, max_bs: int, draft_token_num: int
    ) -> int:
        """Allocate graph-stable Flat verify state and return its byte size."""
        if not self.flat_state_active or self.is_draft:
            return 0
        self._ensure_verify_scratch(max_bs, draft_token_num)
        return sum(
            tensor.nbytes
            for layer_scratch in self._verify_scratch.values()
            for tensor in layer_scratch
        )

    def _verify_copy_tables_get(self) -> dict:
        """Pointer tables for the batched verify state copies: per-layer base
        addresses of the state slabs and their verify scratches, plus each
        layer's index into ``_flat_state_group_ids``. Rebuilt only when the
        scratch is (re)allocated; the tensors are stable so CUDA-graph capture
        can record them."""
        tables = getattr(self, "_verify_copy_tables", None)
        if tables is not None:
            return tables
        layer_ids = list(self._flat_mamba_layer_ids())
        group_ids = self._flat_state_groups()
        group_index = {group_id: i for i, group_id in enumerate(group_ids)}
        conv_src, conv_dst, ssm_src, ssm_dst, group_sel = [], [], [], [], []
        conv_src_st, conv_dst_st, ssm_src_st, ssm_dst_st = [], [], [], []
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
            conv, ssm = self._flat_state_components(layer_id)
            conv_scratch, ssm_scratch = self._verify_scratch[layer_id]
            row_c = conv[0].numel() * conv.element_size()
            row_s = ssm[0].numel() * ssm.element_size()
            conv_bytes = row_c if conv_bytes is None else conv_bytes
            ssm_bytes = row_s if ssm_bytes is None else ssm_bytes
            if row_c != conv_bytes or row_s != ssm_bytes:
                raise RuntimeError("verify state rows must be uniform per kind")
            conv_src.append(conv.data_ptr())
            conv_dst.append(conv_scratch.data_ptr())
            ssm_src.append(ssm.data_ptr())
            ssm_dst.append(ssm_scratch.data_ptr())
            conv_src_st.append(_row_stride_i32(conv))
            conv_dst_st.append(_row_stride_i32(conv_scratch))
            ssm_src_st.append(_row_stride_i32(ssm))
            ssm_dst_st.append(_row_stride_i32(ssm_scratch))
            group_sel.append(group_index[self._flat_state_group_for(layer_id)])

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
            "ssm_scratch": _u64(ssm_dst),
            "ssm_comp_stride": _i64(ssm_src_st),
            "ssm_scratch_stride": _i64(ssm_dst_st),
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
        """Seed every KDA layer's verify-scratch init conv and recurrent rows
        from its group's committed state page. Negative page ids zero-fill."""
        from tokenspeed_kernel.ops.kvcache.triton import copy_state_rows

        tables = self._verify_copy_tables_get()
        state_in_by_group = self.forward_metadata.state_in_pages_by_group
        sin_stack = torch.stack(
            [state_in_by_group[group_id][:bs] for group_id in self._flat_state_groups()]
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

    def flat_commit_verified_state(self, accepted_length: torch.Tensor) -> None:
        """Commit the accepted draft position's state (conv window +
        recurrent) from the verify scratch into each group's state slab at
        the new committed page. All device-side; graph-safe."""
        ctx = getattr(self, "_verify_commit_ctx", None)
        if ctx is None:
            return
        committed, tables, draft_token_num = ctx
        bs = accepted_length.shape[0]
        k = accepted_length.to(torch.int64).clamp(min=1, max=draft_token_num)
        new_last = committed[:bs] + k - 1
        slot = torch.div(new_last, self._flat_state_page_size, rounding_mode="floor")
        stride = draft_token_num + 1
        src_rows = (
            torch.arange(bs, dtype=torch.int64, device=accepted_length.device) * stride
            + k
        )
        pages_by_group: dict[str, torch.Tensor] = {}
        for group_id in self._flat_state_groups():
            rows_tbl = tables[group_id]
            slot_safe = slot.clamp(min=0, max=rows_tbl.shape[1] - 1)
            pages_by_group[group_id] = (
                rows_tbl[:bs]
                .gather(1, slot_safe.unsqueeze(1))
                .squeeze(1)
                .to(torch.int64)
                .clamp_min(0)
            )
        # Batched commit: scratch row -> committed page for every KDA layer in
        # one launch per state kind (was a per-layer gather/scatter pair).
        from tokenspeed_kernel.ops.kvcache.triton import copy_state_rows

        copy_tables = self._verify_copy_tables_get()
        pages_stack = torch.stack(
            [pages_by_group[group_id] for group_id in self._flat_state_groups()]
        )
        dst_rows = pages_stack.index_select(0, copy_tables["group_sel"]).reshape(-1)
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

    def _flat_contract_state_pages(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        kwargs: dict,
        *,
        validate: bool | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Per-state-group (state_in, state_out) page-id mappings for this
        forward, from the operation-bound flat cache metadata.

        The dual-index gather runs ONCE per state group per batch — never per
        layer. KDA layers select their group's entry via
        ``pool.group_id_for_layer(layer_id)`` at forward time.

        validate: explicit True/False wins; None (the hot-path default)
        validates only under TOKENSPEED_FLAT_DEBUG=1 (the checks host-sync).

        Returns:
            ``(state_in_pages, state_out_pages)`` mappings keyed by state
            group id, each value an int32 ``[bs]`` page-id tensor.
        """
        if validate is None:
            validate = flat_cache_debug_enabled()
        flat_cache_metadata = kwargs.get("flat_cache_metadata")
        if flat_cache_metadata is None:
            # Missing FlatKV metadata must never select a legacy state path
            # (req_to_page / SimpleMambaPool).
            raise RuntimeError(
                "MambaAttnBackend is bound to the FlatKV contract but "
                "received no flat cache metadata; refusing any legacy "
                "state path"
            )
        flat_cache_forward_op = kwargs.get("flat_cache_forward_op")
        before, after = self._state_page_bounds(bs, seq_lens, forward_mode, kwargs)
        plan = _compute_state_page_index_plan(self._flat_state_page_size, before, after)
        out_slots_by_width: dict[int, torch.Tensor] = {}
        state_in_pages: dict[str, torch.Tensor] = {}
        state_out_pages: dict[str, torch.Tensor] = {}
        for group_id in self._flat_state_group_ids:
            rows = flat_cache_metadata.require_table(
                group_id, active_forward_op=flat_cache_forward_op
            )
            table_width = rows.shape[1]
            out_slots_safe = out_slots_by_width.get(table_width)
            if out_slots_safe is None:
                out_slots_safe = plan.out_slots.clamp(min=0, max=table_width - 1)
                out_slots_by_width[table_width] = out_slots_safe
            state_in, state_out = _gather_state_page_indices(
                rows,
                plan,
                out_slots_safe=out_slots_safe,
                validate=validate,
                group_id=group_id,
            )
            state_in_pages[group_id] = state_in
            state_out_pages[group_id] = state_out
        return state_in_pages, state_out_pages

    def reset_current_inputs(
        self, req_pool_indices: torch.Tensor, working_indices: torch.Tensor
    ):
        if self.pool is not None:
            self.pool.reset_current_inputs(req_pool_indices, working_indices)

    def init_forward_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = ForwardMode.DECODE,
        **kwargs,
    ):
        mamba_pool_indices = kwargs.get("mamba_pool_indices")
        if self.pool is None:
            # Poolless flat mode: states are addressed via state_in/out_pages;
            # every consumer of mamba_cache_indices is pool/radix-gated.
            mamba_cache_indices = None
        elif mamba_pool_indices is not None:
            mamba_cache_indices = self.pool.get_mamba_indices(mamba_pool_indices[:bs])
        else:
            mamba_cache_indices = self.pool.get_mamba_indices(req_pool_indices[:bs])

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
        if is_target_verify and not self.flat_state_active:
            draft_token_num = int(
                kwargs.get("tokens_per_req", self.speculative_num_draft_tokens)
            )
            cow_src_indices = kwargs.get("mamba_cow_src_indices")
            mamba_input_indices = self.pool.get_current_input_indices(
                req_pool_indices[:bs], mamba_cache_indices, cow_src_indices
            )
            mamba_output_indices = self.pool.get_mtp_output_indices(
                req_pool_indices[:bs],
                mamba_cache_indices,
                draft_token_num,
            )
            mamba_cache_indices = mamba_input_indices

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

        state_in_pages = None
        state_out_pages = None
        state_in_pages_by_group = None
        state_out_pages_by_group = None
        # Idle/bs==0 forwards carry no requests and never reach the mamba
        # forward (router returns early), so no tables are required.
        if self.flat_state_active and bs > 0 and not forward_mode.is_idle():
            if is_draft_extend:
                raise RuntimeError(
                    "flat state paging on a draft worker is unsupported; the "
                    "MTP draft runs on a classic pool"
                )
            if is_target_verify:
                draft_token_num = int(
                    kwargs.get("tokens_per_req", self.speculative_num_draft_tokens)
                )
                (
                    state_in_pages_by_group,
                    verify_committed,
                    verify_tables,
                ) = self._flat_verify_state_pages(bs, seq_lens, draft_token_num, kwargs)
                # Slab out pages are unused under verify (per-position states
                # land in the scratch); alias in so shape contracts hold.
                state_out_pages_by_group = state_in_pages_by_group
                self._ensure_verify_scratch(bs, draft_token_num)
                mamba_output_indices = self._verify_scratch_grid(bs, draft_token_num)
                self._verify_commit_ctx = (
                    verify_committed,
                    verify_tables,
                    draft_token_num,
                )
            elif self._flat_contract_bound:
                (
                    state_in_pages_by_group,
                    state_out_pages_by_group,
                ) = self._flat_contract_state_pages(bs, seq_lens, forward_mode, kwargs)
            else:
                state_in_pages, state_out_pages = self._flat_state_pages_by_group(
                    bs, seq_lens, forward_mode, kwargs
                )

        track_ssm_h_src = None
        track_ssm_h_src_fla = None
        track_ssm_h_dst = None
        track_conv_indices = None
        track_ssm_final_src = None
        track_ssm_final_dst = None
        if (
            (forward_mode.is_extend_or_mixed() or is_draft_extend)
            and not is_target_verify
            # Radix-only prefix snapshot tracking: flat mode snapshots states
            # via page claiming (dual-index), and track indices address the
            # SimpleMambaPool row space, not the state slabs.
            and not self.flat_state_active
        ):
            extend_prefix_lens_kw = kwargs.get("extend_prefix_lens")
            mamba_track_pool_indices = kwargs.get("mamba_track_pool_indices")
            if (
                extend_prefix_lens_kw is not None
                and mamba_track_pool_indices is not None
            ):
                prefix = extend_prefix_lens_kw[:bs].to(
                    dtype=torch.int32, device=self.device
                )
                track_indices = mamba_track_pool_indices[:bs].to(
                    dtype=torch.int32, device=self.device
                )
                extend_lens = (seq_lens[:bs] - prefix).to(torch.int32)
                checkpoint_mask = (track_indices >= 0) & (mamba_cache_indices >= 0)

                page_size = getattr(self.pool, "page_size", 1)
                final_lens = prefix + extend_lens
                last_inserted_lens = (final_lens // page_size) * page_size
                track_lens = last_inserted_lens - prefix
                track_inside = (
                    checkpoint_mask & (track_lens > 0) & (track_lens < extend_lens)
                )
                track_mask = track_inside & ((track_lens % FLA_CHUNK_SIZE) == 0)
                # C++ attaches the checkpoint slot to the last KV page inserted
                # for this chunk. When a chunk has an intermediate branch and
                # ends exactly on a page boundary, the final state must win.
                final_mask = (
                    checkpoint_mask
                    & (final_lens >= page_size)
                    & ((final_lens % page_size) == 0)
                )
                if final_mask.any():
                    track_ssm_final_src = mamba_cache_indices[final_mask]
                    track_ssm_final_dst = track_indices[final_mask]

                if track_mask.any():
                    (
                        track_ssm_h_src,
                        track_ssm_h_src_fla,
                        track_ssm_h_dst,
                    ) = self._compute_track_ssm_indices(
                        track_lens,
                        track_mask,
                        track_indices,
                        seq_lens[:bs] - prefix,  # extend_seq_lens
                    )
                    track_conv_indices = self._compute_track_conv_indices(
                        query_start_loc,
                        track_lens,
                        track_mask,
                    )

        self.forward_metadata = MambaForwardMetadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=mamba_cache_indices,
            mamba_output_indices=mamba_output_indices,
            mamba_req_pool_indices=req_pool_indices[:bs],
            extend_prefix_lens=kwargs.get("extend_prefix_lens"),
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            track_ssm_h_src=track_ssm_h_src,
            track_ssm_h_src_fla=track_ssm_h_src_fla,
            track_ssm_h_dst=track_ssm_h_dst,
            track_conv_indices=track_conv_indices,
            track_ssm_final_src=track_ssm_final_src,
            track_ssm_final_dst=track_ssm_final_dst,
            state_in_pages=state_in_pages,
            state_out_pages=state_out_pages,
            state_in_pages_by_group=state_in_pages_by_group,
            state_out_pages_by_group=state_out_pages_by_group,
        )

    def _compute_track_conv_indices(
        self,
        query_start_loc: torch.Tensor,
        track_lens: torch.Tensor,
        track_mask: torch.Tensor,
    ):
        """Compute packed input indices for conv windows at tracked boundaries."""
        conv_state_len = self.pool.conv_state.shape[-1]
        lens_m = track_lens[track_mask]
        start = query_start_loc[:-1][track_mask] + lens_m - conv_state_len
        indices = start.unsqueeze(-1) + torch.arange(
            conv_state_len,
            device=self.device,
            dtype=start.dtype,
        )
        return indices.clamp(0, query_start_loc[-1] - 1)

    def _compute_track_ssm_indices(
        self,
        track_lens: torch.Tensor,
        track_mask: torch.Tensor,
        mamba_track_indices: torch.Tensor,
        extend_seq_lens: torch.Tensor,
    ):
        """Compute src/dst indices for extracting intermediate SSM states.

        Matching conv windows are gathered separately from packed pre-conv inputs.
        """
        # flashinfer ckpts[k] = state after processing chunk k = FLA h[k+1]
        num_fi_ckpts = extend_seq_lens // FLA_CHUNK_SIZE
        offset = torch.zeros_like(num_fi_ckpts)
        offset[1:] = torch.cumsum(num_fi_ckpts[:-1], dim=0)
        num_fla_states = (extend_seq_lens - 1) // FLA_CHUNK_SIZE + 1
        fla_offset = torch.zeros_like(num_fla_states)
        fla_offset[1:] = torch.cumsum(num_fla_states[:-1], dim=0)

        lens_m = track_lens[track_mask]
        offset_m = offset[track_mask]
        fla_offset_m = fla_offset[track_mask]
        dst_m = mamba_track_indices[track_mask]

        # FLA h[lens//C] = flashinfer ckpts[lens//C - 1].
        # track_mask guarantees lens_m >= FLA_CHUNK_SIZE so lens_m // C >= 1.
        track_ssm_h_src = offset_m + (lens_m // FLA_CHUNK_SIZE - 1)
        track_ssm_h_src_fla = fla_offset_m + (lens_m // FLA_CHUNK_SIZE)
        track_ssm_h_dst = dst_m

        return (
            track_ssm_h_src,
            track_ssm_h_src_fla,
            track_ssm_h_dst,
        )

    # ---- CUDA graph state ----

    def init_cuda_graph_state(
        self, max_num_tokens: int, seq_lens_buf: torch.Tensor = None
    ):
        del seq_lens_buf  # mamba doesn't use seq_lens_buf.
        for i in range(max_num_tokens):
            self.state_indices_list.append(
                torch.full(
                    (i + 1,), self.pad_slot_id, dtype=torch.int32, device=self.device
                )
            )
            self.query_start_loc_list.append(
                torch.empty((i + 2,), dtype=torch.int32, device=self.device)
            )
            if self.flat_state_active and not self._flat_contract_bound:
                self.flat_state_in_list.append(
                    {
                        group_id: torch.full(
                            (i + 1,),
                            self.pad_slot_id,
                            dtype=torch.int32,
                            device=self.device,
                        )
                        for group_id in self._state_group_ids
                    }
                )
                self.flat_state_out_list.append(
                    {
                        group_id: torch.full(
                            (i + 1,),
                            self.pad_slot_id,
                            dtype=torch.int32,
                            device=self.device,
                        )
                        for group_id in self._state_group_ids
                    }
                )
            elif self.flat_state_active and self._flat_contract_bound:
                # Multi-group KDA state paging keeps one persistent dual-index
                # [bs] buffer per state group. Replay refreshes each buffer in
                # place while the captured graph keeps its address.
                for gid in self._flat_state_group_ids:
                    self.flat_state_in_by_group.setdefault(gid, []).append(
                        torch.full(
                            (i + 1,),
                            self.pad_slot_id,
                            dtype=torch.int32,
                            device=self.device,
                        )
                    )
                    self.flat_state_out_by_group.setdefault(gid, []).append(
                        torch.full(
                            (i + 1,),
                            self.pad_slot_id,
                            dtype=torch.int32,
                            device=self.device,
                        )
                    )
            if self.speculative_num_draft_tokens > 0:
                self.output_indices_list.append(
                    torch.full(
                        (i + 1, self.speculative_num_draft_tokens),
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

        mamba_pool_indices = kwargs.get("mamba_pool_indices")
        # Reuse the pre-allocated [bs]-length buffer as mamba_indices so the
        # capture path matches the replay path: zero allocation, single write.
        padded_mamba_indices = self.state_indices_list[bs - 1]
        if self.pool is None:
            # Poolless flat mode: the buffer stays all pad_slot_id (consumers
            # are pool/radix-gated; states travel via state_in/out_pages).
            padded_mamba_indices.fill_(self.pad_slot_id)
        elif mamba_pool_indices is not None:
            padded_mamba_indices[:bs].copy_(
                self.pool.get_mamba_indices(mamba_pool_indices[:bs])
            )
        else:
            padded_mamba_indices[:bs].copy_(
                self.pool.get_mamba_indices(req_pool_indices[:bs])
            )
        mamba_output_indices = None
        if is_target_verify and not self.flat_state_active:
            cow_src_indices = kwargs.get("mamba_cow_src_indices")
            mamba_input_indices = self.pool.get_current_input_indices(
                req_pool_indices[:bs], padded_mamba_indices, cow_src_indices
            )
            mamba_output_indices = self.output_indices_list[bs - 1]
            self.pool.get_mtp_output_indices(
                req_pool_indices[:bs],
                padded_mamba_indices,
                self.speculative_num_draft_tokens,
                out=mamba_output_indices,
            )
            padded_mamba_indices.copy_(mamba_input_indices)
        state_in_pages = None
        state_out_pages = None
        state_in_pages_by_group = None
        state_out_pages_by_group = None
        if self.flat_state_active:
            # Real tables only arrive at replay; capture binds the persistent
            # buffers (all pad_slot_id: kernels skip reads/writes at capture,
            # so state slab rows are never dirtied by the capture pass).
            if is_draft_extend:
                raise RuntimeError("flat state paging on a draft worker is unsupported")
            if is_target_verify:
                # Verify capture/prewarm: pad state_in (reads zeros / writes
                # skip) + the real scratch grid; commit stays disarmed.
                draft_token_num = int(self.speculative_num_draft_tokens)
                self._ensure_verify_scratch(bs, draft_token_num)
                mamba_output_indices = self._verify_scratch_grid(bs, draft_token_num)
                self._verify_commit_ctx = None
            if self._flat_contract_bound:
                # Keep one dual-index buffer pair per FlatKV state group. A KDA
                # layer selects its group through pool.group_id_for_layer.
                state_in_pages_by_group = {}
                state_out_pages_by_group = {}
                for gid in self._flat_state_group_ids:
                    state_in = self.flat_state_in_by_group[gid][bs - 1]
                    state_out = self.flat_state_out_by_group[gid][bs - 1]
                    state_in.fill_(self.pad_slot_id)
                    state_out.fill_(self.pad_slot_id)
                    state_in_pages_by_group[gid] = state_in
                    state_out_pages_by_group[gid] = state_out
            else:
                flat_ids = kwargs.get("flat_cache_group_ids", ())
                missing = set(self._state_group_ids) - set(flat_ids)
                if missing:
                    raise RuntimeError(
                        "flat GDN state paging: capture is missing flat cache "
                        f"group ids {sorted(missing)!r} "
                        f"(got {tuple(flat_ids)!r})"
                    )
                state_in_pages = self.flat_state_in_list[bs - 1]
                state_out_pages = self.flat_state_out_list[bs - 1]
                for pages in state_in_pages.values():
                    pages.fill_(self.pad_slot_id)
                for pages in state_out_pages.values():
                    pages.fill_(self.pad_slot_id)
                if is_target_verify:
                    state_in_pages_by_group = state_in_pages
                    state_out_pages_by_group = state_out_pages
        self._qsl_dirty[bs - 1] = False
        self._qsl_last_mode[bs - 1] = (forward_mode, self.spec_num_tokens > 1)
        self.forward_metadata = MambaForwardMetadata(
            query_start_loc=self.query_start_loc_list[bs - 1],
            mamba_cache_indices=self.state_indices_list[bs - 1],
            mamba_output_indices=mamba_output_indices,
            mamba_req_pool_indices=req_pool_indices[:bs],
            state_in_pages=state_in_pages,
            state_out_pages=state_out_pages,
            state_in_pages_by_group=state_in_pages_by_group,
            state_out_pages_by_group=state_out_pages_by_group,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = None,
        req_to_page: torch.Tensor = None,
        **kwargs,
    ):
        num_padding = kwargs.get("num_padding", 0)
        mamba_pool_indices = kwargs.get("mamba_pool_indices")

        real_bs = bs - num_padding
        req_pool_indices = req_pool_indices[:bs]

        # Reuse the pre-allocated [bs]-length buffer as the padded mamba_indices
        # so downstream ops (get_mtp_output_indices, get_current_input_indices)
        # see the full-batch shape with padding rows already set to -1.
        # Zero extra allocations on this hot path.
        padded_mamba_indices = self.state_indices_list[bs - 1]
        if self.pool is None:
            # Poolless flat mode: the buffer stays all pad_slot_id (consumers
            # are pool/radix-gated; states travel via state_in/out_pages).
            padded_mamba_indices.fill_(self.pad_slot_id)
        else:
            if mamba_pool_indices is not None:
                padded_mamba_indices[:real_bs].copy_(
                    self.pool.get_mamba_indices(mamba_pool_indices[:real_bs])
                )
            else:
                padded_mamba_indices[:real_bs].copy_(
                    self.pool.get_mamba_indices(req_pool_indices[:real_bs])
                )
            if num_padding > 0:
                padded_mamba_indices[real_bs:].fill_(-1)

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
        if is_target_verify and not self.flat_state_active:
            cow_src_indices = kwargs.get("mamba_cow_src_indices")
            mamba_input_indices = self.pool.get_current_input_indices(
                req_pool_indices, padded_mamba_indices, cow_src_indices
            )
            mamba_output_indices = self.output_indices_list[bs - 1]
            self.pool.get_mtp_output_indices(
                req_pool_indices,
                padded_mamba_indices,
                self.speculative_num_draft_tokens,
                out=mamba_output_indices,
            )
            # mamba_input_indices already encodes padding via padded_mamba_indices.
            padded_mamba_indices.copy_(mamba_input_indices)

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

        state_in_pages = None
        state_out_pages = None
        state_in_pages_by_group = None
        state_out_pages_by_group = None
        if self.flat_state_active and is_target_verify:
            # Flat target-verify replay: refresh the CAPTURED per-bs state_in
            # buffers with each group's committed-state page (page of
            # ``seq_lens - T - 1``), re-arm the commit context for the
            # post-round scratch->slab commit, and keep the memoized scratch
            # grid the graph recorded. Padded rows stay pad_slot_id — the
            # state kernels skip them to protect the dummy page.
            draft_token_num = int(self.speculative_num_draft_tokens)
            self._ensure_verify_scratch(bs, draft_token_num)
            mamba_output_indices = self._verify_scratch_grid(bs, draft_token_num)
            pages_by_group = None
            if real_bs > 0:
                (
                    pages_by_group,
                    verify_committed,
                    verify_tables,
                ) = self._flat_verify_state_pages(
                    real_bs, seq_lens, draft_token_num, kwargs
                )
                self._verify_commit_ctx = (
                    verify_committed,
                    verify_tables,
                    draft_token_num,
                )
            else:
                self._verify_commit_ctx = None
            if self._flat_contract_bound:
                captured_in = {
                    group_id: self.flat_state_in_by_group[group_id][bs - 1]
                    for group_id in self._flat_state_groups()
                }
                captured_out = {
                    group_id: self.flat_state_out_by_group[group_id][bs - 1]
                    for group_id in self._flat_state_groups()
                }
            else:
                captured_in = self.flat_state_in_list[bs - 1]
                captured_out = self.flat_state_out_list[bs - 1]
            state_in_pages_by_group = {}
            state_out_pages_by_group = {}
            for group_id in self._flat_state_groups():
                state_in = captured_in[group_id]
                state_out = captured_out[group_id]
                if pages_by_group is not None:
                    state_in[:real_bs].copy_(pages_by_group[group_id][:real_bs])
                if real_bs < bs:
                    state_in[real_bs:].fill_(self.pad_slot_id)
                # Slab out pages are unused under verify (per-position states
                # land in the scratch); keep the captured buffer inert.
                state_out.fill_(self.pad_slot_id)
                state_in_pages_by_group[group_id] = state_in
                state_out_pages_by_group[group_id] = state_out
            if not self._flat_contract_bound:
                state_in_pages = state_in_pages_by_group
                state_out_pages = state_out_pages_by_group
        elif self.flat_state_active and self._flat_contract_bound:
            # For multi-group KDA state paging, dual indexing runs once per
            # state group over the real rows. Padded rows get pad_slot_id (-1),
            # which state kernels skip, so they never touch a live page.
            # Decode-only
            # (q_len == 1): before = seq_lens - 1. Validation defaults off on
            # the replay hot path (host sync); TOKENSPEED_FLAT_DEBUG=1 arms it.
            # bs==0 idle replay carries no operation-bound metadata; every row
            # is a dummy padded row, so skip the dual-index gather entirely.
            state_in_pages_by_group, state_out_pages_by_group = (
                self._replay_contract_state_pages(
                    bs, real_bs, seq_lens, forward_mode, kwargs
                )
            )
            state_in_pages = None
            state_out_pages = None
        elif self.flat_state_active:
            # Decode-only (q_len == 1): before = seq_lens - 1. Padded rows
            # (zero table rows, seq_lens 1) are overwritten with pad_slot_id
            # below, so validation is skipped to avoid a host sync.
            state_in, state_out = self._flat_state_pages_by_group(
                bs,
                seq_lens,
                forward_mode,
                kwargs,
                validate=False,
            )
            state_in_pages = self.flat_state_in_list[bs - 1]
            state_out_pages = self.flat_state_out_list[bs - 1]
            for group_id in self._state_group_ids:
                state_in_pages[group_id][:real_bs].copy_(state_in[group_id][:real_bs])
                state_out_pages[group_id][:real_bs].copy_(state_out[group_id][:real_bs])
                if num_padding > 0:
                    state_in_pages[group_id][real_bs:].fill_(self.pad_slot_id)
                    state_out_pages[group_id][real_bs:].fill_(self.pad_slot_id)

        self.forward_metadata = MambaForwardMetadata(
            query_start_loc=self.query_start_loc_list[bs - 1],
            mamba_cache_indices=self.state_indices_list[bs - 1],
            mamba_output_indices=mamba_output_indices,
            mamba_req_pool_indices=req_pool_indices,
            state_in_pages=state_in_pages,
            state_out_pages=state_out_pages,
            state_in_pages_by_group=state_in_pages_by_group,
            state_out_pages_by_group=state_out_pages_by_group,
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
        flat_cache_metadata = kwargs.get("flat_cache_metadata")
        flat_cache_forward_op = kwargs.get("flat_cache_forward_op")
        gids = self._flat_state_group_ids
        use_tape = (
            not flat_cache_debug_enabled()
            and seq_lens.is_cuda
            and seq_lens.dtype == torch.int32
            and len(gids) <= 4
            and flat_cache_metadata is not None
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
                    sin = self.flat_state_in_by_group[gid][bs - 1]
                    sout = self.flat_state_out_by_group[gid][bs - 1]
                    tape.state_pages(
                        sin,
                        sout,
                        rows_ptr=Reg(Reg.PTR0 + i),
                        seq_lens_ptr=Reg.PTR4,
                        bs=Reg.REAL_BS,
                        max_slots=Reg(Reg.USER0 + i),
                        page_size=self._flat_state_page_size,
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
                rows = flat_cache_metadata.require_table(
                    gid, active_forward_op=flat_cache_forward_op
                )
                regs[Reg(Reg.PTR0 + i)] = rows
                regs[Reg(Reg.USER0 + i)] = rows.shape[1]
            tape.run(regs)
            return (
                {g: self.flat_state_in_by_group[g][bs - 1] for g in gids},
                {g: self.flat_state_out_by_group[g][bs - 1] for g in gids},
            )

        state_in_by = state_out_by = None
        if real_bs > 0:
            state_in_by, state_out_by = self._flat_contract_state_pages(
                real_bs,
                seq_lens,
                forward_mode,
                kwargs,
                validate=None,
            )
        in_by_group: dict[str, torch.Tensor] = {}
        out_by_group: dict[str, torch.Tensor] = {}
        for gid in gids:
            state_in_pages = self.flat_state_in_by_group[gid][bs - 1]
            state_out_pages = self.flat_state_out_by_group[gid][bs - 1]
            if real_bs > 0:
                state_in_pages[:real_bs].copy_(state_in_by[gid][:real_bs])
                state_out_pages[:real_bs].copy_(state_out_by[gid][:real_bs])
            if real_bs < bs:
                state_in_pages[real_bs:].fill_(self.pad_slot_id)
                state_out_pages[real_bs:].fill_(self.pad_slot_id)
            in_by_group[gid] = state_in_pages
            out_by_group[gid] = state_out_pages
        return in_by_group, out_by_group

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    # ---- Forward ----

    def _layer_flat_state(
        self, layer_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Resolve one layer's flat state paging for the current forward.

        Args:
            layer_id: Global model layer id of the recurrent layer.

        Returns:
            ``(state_in_pages, state_out_pages, conv_states, ssm_states)``,
            or ``None`` when the current forward carries no flat state
            metadata (radix SimpleMambaPool path).

        Contract path: the layer's group (``pool.group_id_for_layer``)
        selects the batch-level dual indices computed once per group in
        ``init_forward_metadata``, and the conv/recurrent tensors are the
        pool's no-copy, page-id-indexed component views. Legacy path: the
        single-group tensors and the pool's state slab pair, byte-for-byte
        the pre-contract behavior.
        """
        metadata = self.forward_metadata
        state_in_by_group = metadata.state_in_pages_by_group
        if state_in_by_group is not None:
            group_id = self._flat_state_group_for(layer_id)
            state_in = state_in_by_group.get(group_id)
            if state_in is None:
                # A recurrent layer must resolve to a bound state group.
                raise RuntimeError(
                    f"flat state paging: layer {layer_id} resolves to group "
                    f"{group_id!r}, which has no computed state page indices "
                    f"(state groups: {sorted(state_in_by_group)})"
                )
            conv_states, ssm_states = self._flat_state_components(layer_id)
            return (
                state_in,
                metadata.state_out_pages_by_group[group_id],
                conv_states,
                ssm_states,
            )
        if metadata.state_in_pages is not None:
            group_id = self._state_group_for_layer[layer_id]
            conv_states, ssm_states = self.kv_pool.get_state_buffers(layer_id)
            return (
                metadata.state_in_pages[group_id],
                metadata.state_out_pages[group_id],
                conv_states,
                ssm_states,
            )
        return None

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
        kda_lower_bound = kwargs.get("lower_bound")
        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        layer_id = kwargs["layer_id"]

        cache_indices = self.forward_metadata.mamba_cache_indices
        flat_state = self._layer_flat_state(layer_id)
        use_flat = flat_state is not None

        if use_flat:
            # Dual-index: read the page holding position n-1, write the page
            # holding position n (in == out within a page; a boundary crossing
            # reads the old page and writes the new one). pad_slot_id rows
            # (graph padding) are skipped by both kernels.
            state_in_pages, state_out_pages, conv_states, ssm_states = flat_state
            read_indices = state_in_pages
        else:
            state_out_pages = None
            conv_states, ssm_states, *rest = self.pool.get_mamba_params(layer_id)
            read_indices = cache_indices

        if self.is_kda and use_flat and f_a_out is not None:
            num_value_heads = value_dim // attn_tp_size // head_v_dim
            core_attn_out = try_kda_fused_paged_decode(
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
                write_indices=state_out_pages,
                num_heads=num_value_heads,
                head_dim=head_v_dim,
                cu_seqlens=self.forward_metadata.query_start_loc,
                lower_bound=kda_lower_bound,
            )
            if core_attn_out is not None:
                return core_attn_out

        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            conv_weights,
            bias,
            activation,
            conv_state_indices=read_indices,
            output_state_indices=(state_out_pages.view(-1, 1) if use_flat else None),
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

        if self.is_kda:
            num_value_heads = value.shape[2]
            query_start_loc = self.forward_metadata.query_start_loc
            if g_raw is None:
                g_raw = torch.nn.functional.linear(f_a_out, f_b_weight)
            query = query.view(1, seq_len, num_heads, head_k_dim)
            key = key.view(1, seq_len, num_heads, head_k_dim)
            value = value.view(1, seq_len, num_value_heads, head_v_dim)
            g_kda = g_raw.view(1, seq_len, num_value_heads, head_k_dim)
            beta_kda = beta_raw.view(1, seq_len, num_value_heads)
            return kda_paged_decode(
                query,
                key,
                value,
                g_kda,
                beta_kda,
                A_log,
                dt_bias,
                state_pool=ssm_states,
                read_indices=read_indices,
                write_indices=(state_out_pages if use_flat else read_indices),
                cu_seqlens=query_start_loc,
                lower_bound=kda_lower_bound,
            ).squeeze(0)

        (
            decode_initial_indices,
            decode_output_indices,
            decode_solution,
        ) = _prepare_gdn_decode_state_path(
            ssm_states,
            read_indices,
            state_out_pages if use_flat else None,
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
            # Flat: write to the out page, not the (possibly shared) in page;
            # pool: None writes back in place to initial_state_indices.
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
        # GDN passes scalar-per-head a/b; KDA (self.is_kda) passes a per-channel
        # gate g_raw + raw beta logits instead (A_log / dt_bias are per-channel).
        a = kwargs.get("a")
        b = kwargs.get("b")
        g_raw = kwargs.get("g_raw")

        def _kda_g_raw() -> torch.Tensor:
            # KDA: the fused decode/verify kernels absorb f_b into the scan;
            # the remaining paths need the plain GEMV, computed on first use.
            nonlocal g_raw
            if g_raw is None and kwargs.get("f_a_out") is not None:
                g_raw = torch.nn.functional.linear(
                    kwargs["f_a_out"], kwargs["f_b_weight"]
                )
            return g_raw

        beta_raw = kwargs.get("beta_raw")
        kda_lower_bound = kwargs.get("lower_bound")
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
        cache_indices = self.forward_metadata.mamba_cache_indices

        if is_target_verify:
            draft_token_num = kwargs.get(
                "draft_token_num", self.speculative_num_draft_tokens
            )
            batch_size = seq_len // draft_token_num
            output_indices = self.forward_metadata.mamba_output_indices
            flat_state = (
                self._layer_flat_state(layer_id) if self.flat_state_active else None
            )
            if (
                flat_state is not None
                and self.is_kda
                and kwargs.get("f_a_out") is not None
                and kwargs.get("bias") is None
            ):
                # Verify megafusion: conv(+silu), f_b gate GEMV, and the
                # per-position recurrence in one launch, per-position conv
                # windows and states landing in the verify scratches.
                state_in_pages, _, conv_comp, ssm_states = flat_state
                conv_scratch, ssm_scratch = self._verify_scratch[layer_id]
                num_value_heads = value_dim // attn_tp_size // head_v_dim
                fused_out = try_kda_fused_paged_verify(
                    mixed_qkv,
                    conv_weights,
                    conv_comp,
                    conv_scratch,
                    kwargs["f_a_out"],
                    kwargs["f_b_weight"],
                    beta_raw,
                    A_log,
                    dt_bias,
                    state_pool=ssm_states,
                    state_scratch=ssm_scratch,
                    read_indices=state_in_pages[:batch_size],
                    write_indices=output_indices[:batch_size],
                    num_heads=num_value_heads,
                    head_dim=head_v_dim,
                    draft_token_num=draft_token_num,
                    lower_bound=kda_lower_bound,
                )
                if fused_out is not None:
                    return fused_out
            if flat_state is not None:
                # Flat verify: read the committed window, write per-position
                # windows to the verify scratch (the accepted one commits
                # back post-verification). The first KDA layer seeds EVERY
                # layer's init row in one batched launch; the copy runs inside
                # the captured region so replays re-read the refreshed
                # state_in buffers.
                state_in_pages, _, conv_comp, ssm_states = flat_state
                conv_scratch, ssm_scratch = self._verify_scratch[layer_id]
                if layer_id == self._flat_mamba_layer_ids()[0]:
                    self._seed_verify_scratch_batched(batch_size, draft_token_num)
                init_rows = output_indices[:batch_size, 0] - 1
                conv_states = conv_scratch
                conv_read = init_rows
                conv_out = output_indices[:batch_size]
                self._verify_flat_layer_state = (
                    state_in_pages,
                    ssm_states,
                    ssm_scratch,
                )
            else:
                conv_states, ssm_states = self.pool.get_mamba_params(layer_id)
                conv_read = cache_indices[:batch_size]
                conv_out = output_indices[:batch_size]
                self._verify_flat_layer_state = None
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
            flat_state = self._layer_flat_state(layer_id)
            use_flat = flat_state is not None
            if use_flat:
                state_in_pages, state_out_pages, conv_states, ssm_states = flat_state
                state_out_long = state_out_pages.to(torch.int64)
                recurrent_state, has_initial_states = (
                    _prepare_flat_prefill_state_inputs(
                        conv_states,
                        ssm_states,
                        state_in_pages,
                        state_out_long,
                    )
                )
                conv_cache_indices = state_out_pages
            else:
                conv_states, ssm_states = self.pool.get_mamba_params(layer_id)
                conv_cache_indices = cache_indices
            extend_prefix_lens = kwargs.get("extend_prefix_lens")
            if extend_prefix_lens is None:
                extend_prefix_lens = self.forward_metadata.extend_prefix_lens
            extend_seq_lens_cpu = kwargs.get("extend_seq_lens_cpu")
            if extend_seq_lens_cpu is None:
                extend_seq_lens_cpu = self.forward_metadata.extend_seq_lens_cpu
            if not use_flat:
                has_initial_states = (
                    extend_prefix_lens > 0 if extend_prefix_lens is not None else None
                )
            need_radix_track = (
                self.forward_metadata.track_ssm_h_src is not None
                and self.forward_metadata.track_ssm_h_src.numel() > 0
            )
            need_h_track = need_radix_track

            # Zero padded rows so garbage can't reach recurrent state (see scrub_padding_tail).
            if extend_seq_lens_cpu is not None:
                ntok = int(sum(int(x) for x in extend_seq_lens_cpu))
                scrub_padding_tail(ntok, mixed_qkv, a, b)

            mixed_qkv_t = mixed_qkv.transpose(0, 1)
            if need_radix_track:
                if self.forward_metadata.track_conv_indices is None:
                    raise RuntimeError(
                        "Missing conv indices for intermediate mamba track"
                    )
                conv_states[self.forward_metadata.track_ssm_h_dst] = mixed_qkv_t[
                    :, self.forward_metadata.track_conv_indices
                ].transpose(0, 1)

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

        query, key, value = fused_qkv_split_gdn_prefill(
            mixed_qkv,
            num_q_heads=num_heads,
            num_k_heads=num_heads,
            num_v_heads=num_value_heads,
            head_q=head_k_dim,
            head_k=head_k_dim,
            head_v=head_v_dim,
        )

        if is_target_verify:
            if self.is_kda:
                flat_layer_state = getattr(self, "_verify_flat_layer_state", None)
                if flat_layer_state is None:
                    # Non-flat KDA verify has no kernel path (and K3 always
                    # runs the flat contract pool).
                    raise NotImplementedError(
                        "KDA (Kimi-K3) target-verify requires the FlatKV "
                        "contract pool"
                    )
                from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (
                    fused_recurrent_kda_mtp,
                )

                state_in_pages, ssm_comp, ssm_scratch = flat_layer_state
                draft_token_num = kwargs.get(
                    "draft_token_num", self.speculative_num_draft_tokens
                )
                batch_size = seq_len // draft_token_num
                query_b = query.view(batch_size, draft_token_num, num_heads, head_k_dim)
                key_b = key.view(batch_size, draft_token_num, num_heads, head_k_dim)
                value_b = value.view(
                    batch_size, draft_token_num, num_value_heads, head_v_dim
                )
                g_b = _kda_g_raw().view(
                    batch_size, draft_token_num, num_value_heads, head_k_dim
                )
                beta_b = beta_raw.view(batch_size, draft_token_num, num_value_heads)
                grid = self.forward_metadata.mamba_output_indices[:batch_size]
                core_attn_out = fused_recurrent_kda_mtp(
                    query_b,
                    key_b,
                    value_b,
                    g_b,
                    beta_b,
                    A_log,
                    dt_bias,
                    ssm_comp,
                    state_in_pages[:batch_size].to(torch.int64),
                    grid,
                    h_pool_out=ssm_scratch,
                    lower_bound=kda_lower_bound,
                ).reshape(1, seq_len, num_value_heads, head_v_dim)
                return core_attn_out
            draft_token_num = kwargs.get(
                "draft_token_num", self.speculative_num_draft_tokens
            )
            batch_size = seq_len // draft_token_num
            # fused_qkv_split_gdn_prefill's [1, seq_len, H, D] varlen-style
            # layout is request-major (token = req * draft_token_num + step),
            # so reshaping the leading two dims into [B, T, H, D] is a plain
            # view -- gdn_decode_mtp's dense-batch contract, no data movement.
            query_b = query.view(batch_size, draft_token_num, num_heads, head_k_dim)
            key_b = key.view(batch_size, draft_token_num, num_heads, head_k_dim)
            value_b = value.view(
                batch_size, draft_token_num, num_value_heads, head_v_dim
            )
            a_b = a.view(batch_size, draft_token_num, -1)
            b_b = b.view(batch_size, draft_token_num, -1)

            output_indices = self.forward_metadata.mamba_output_indices
            flat_verify_state = getattr(self, "_verify_flat_layer_state", None)
            if flat_verify_state is not None:
                _, _, ssm_states = flat_verify_state
                cache_indices = output_indices[:batch_size, 0] - 1
            (
                mtp_initial_indices,
                mtp_output_indices,
                mtp_solution,
            ) = _prepare_gdn_decode_state_path(
                ssm_states,
                cache_indices,
                output_indices,
            )
            core_attn_out = gdn_decode_mtp(
                query_b,
                key_b,
                value_b,
                A_log=A_log,
                a=a_b,
                dt_bias=dt_bias,
                b=b_b,
                initial_state=ssm_states,
                initial_state_indices=mtp_initial_indices,
                use_qk_l2norm=True,
                output_state_indices=mtp_output_indices,
                disable_state_update=False,
                solution=mtp_solution,
            ).reshape(1, seq_len, num_value_heads, head_v_dim)
        elif self.is_kda:
            g_kda = _kda_g_raw().view(1, seq_len, num_value_heads, head_k_dim)
            beta_kda = beta_raw.view(1, seq_len, num_value_heads)
            if not use_flat:
                recurrent_state = ssm_states[cache_indices].contiguous()
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
                lower_bound=kda_lower_bound,
                solution=None if self.kda_backend == "auto" else self.kda_backend,
            )
            core_attn_out = kda_result.out.squeeze(0)
            last_recurrent_state = kda_result.final_state
            last_recurrent_state = last_recurrent_state.to(ssm_states.dtype, copy=False)
            # Extend indices never carry pad(-1) — unguarded like the GDN write below.
            ssm_states[state_out_long if use_flat else cache_indices] = (
                last_recurrent_state
            )
            need_final_track = (
                self.forward_metadata.track_ssm_final_src is not None
                and self.forward_metadata.track_ssm_final_src.numel() > 0
            )
            if need_final_track:
                fused_mamba_state_copy(
                    conv_states,
                    self.forward_metadata.track_ssm_final_src,
                    self.forward_metadata.track_ssm_final_dst,
                    single_layer=True,
                )
                fused_mamba_state_copy(
                    ssm_states,
                    self.forward_metadata.track_ssm_final_src,
                    self.forward_metadata.track_ssm_final_dst,
                    single_layer=True,
                )
        else:
            beta = b.sigmoid()
            g = fused_gdn_gating(A_log, a, dt_bias)
            g = g.unsqueeze(0)
            beta = beta.unsqueeze(0)

            if not use_flat:
                recurrent_state = ssm_states[cache_indices]
            need_final_track = (
                self.forward_metadata.track_ssm_final_src is not None
                and self.forward_metadata.track_ssm_final_src.numel() > 0
            )

            fi_h_checkpoints = None
            h_src = None
            if need_h_track:
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
                    output_h=True,
                )
                core_attn_out = gdn_result.out
                last_recurrent_state = gdn_result.final_state
                if gdn_result.h is None:
                    raise RuntimeError(
                        "gdn_chunk_prefill(output_h=True) must return checkpoints"
                    )
                if gdn_result.h_layout is GdnCheckpointLayout.FLASHINFER:
                    fi_h_checkpoints = gdn_result.h
                    h_src = self.forward_metadata.track_ssm_h_src
                elif gdn_result.h_layout is GdnCheckpointLayout.FLA:
                    fi_h_checkpoints = gdn_result.h.squeeze(0)
                    h_src = self.forward_metadata.track_ssm_h_src_fla
                else:
                    raise RuntimeError(
                        "gdn_chunk_prefill(output_h=True) returned unsupported "
                        f"checkpoint layout {gdn_result.h_layout}"
                    )
            else:
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
                core_attn_out = gdn_result.out
                last_recurrent_state = gdn_result.final_state
            last_recurrent_state = last_recurrent_state.to(ssm_states.dtype, copy=False)
            ssm_states[state_out_long if use_flat else cache_indices] = (
                last_recurrent_state
            )

            if need_h_track:
                ssm_states[self.forward_metadata.track_ssm_h_dst] = fi_h_checkpoints[
                    h_src
                ].to(ssm_states.dtype, copy=False)

            if need_final_track:
                fused_mamba_state_copy(
                    conv_states,
                    self.forward_metadata.track_ssm_final_src,
                    self.forward_metadata.track_ssm_final_dst,
                    single_layer=True,
                )
                fused_mamba_state_copy(
                    ssm_states,
                    self.forward_metadata.track_ssm_final_src,
                    self.forward_metadata.track_ssm_final_dst,
                    single_layer=True,
                )

        return core_attn_out


class HybridLinearAttnBackend(AttentionBackend):
    """Hybrid backend that routes between full attention and linear attention by layer ID."""

    flat_spec_capable: bool = True
    # Both sub-backends consume flat per-group tables (MHA: KV pages; mamba:
    # dual-index state pages). Target verify keeps speculative state in a
    # fixed scratch and publishes only the accepted position.
    uses_flat_cache_groups: bool = True

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

    def override_num_extends(self, num_extends: int):
        return self.full_attn_backend.override_num_extends(num_extends)

    def forward_extend_chunked(self, *args, **kwargs):
        return self.full_attn_backend.forward_extend_chunked(*args, **kwargs)

    @property
    def flat_cache_consumer_families(self) -> frozenset[str]:
        """FlatKV contract families this composite can consume: the union of
        its sub-backends' declarations. The startup guard
        (validate_flat_scheduler_config) rejects a contract pool whose
        required families are not all covered — e.g. Kimi-K3 until both the
        MLA (history) and KDA (state) flat consumers exist.
        """
        families: frozenset[str] = frozenset()
        for backend in (self.full_attn_backend, self.linear_attn_backend):
            families |= frozenset(getattr(backend, "flat_cache_consumer_families", ()))
        return families

    def select_out_cache_loc(self, layer, out_cache_loc, forward_mode=None):
        """Route the write-location hook to the owning sub-backend by layer.

        Model-owned KV writers (MLA latent writes) call this through
        ``ctx.attn_backend`` — this wrapper for hybrid models — so the
        full-attention sub-backend's flat write locations must be reachable
        here. Identity for backends without flat cache groups.
        """
        return self._backend_for_layer(layer.layer_id).select_out_cache_loc(
            layer, out_cache_loc, forward_mode
        )

    def _backends(self):
        return [self.full_attn_backend, self.linear_attn_backend]

    def _backend_for_layer(self, layer_id: int) -> AttentionBackend:
        if self.linear_attn_backend is None or layer_id in self.full_attn_layers:
            return self.full_attn_backend
        return self.linear_attn_backend

    _MAMBA_KWARGS = frozenset(
        {
            "mamba_pool_indices",
            "mamba_cow_src_indices",
            "mamba_branching_seqlens",
            "mamba_track_pool_indices",
        }
    )

    @staticmethod
    def _split_mamba_kwargs(kwargs: dict) -> tuple[dict, dict]:
        mamba_kw = {}
        common_kw = {}
        for k, v in kwargs.items():
            if k in HybridLinearAttnBackend._MAMBA_KWARGS:
                mamba_kw[k] = v
            else:
                common_kw[k] = v
        return common_kw, mamba_kw

    # ---- Metadata delegation ----

    def init_forward_metadata(self, *args, **kwargs):
        common_kw, mamba_kw = self._split_mamba_kwargs(kwargs)
        self.full_attn_backend.init_forward_metadata(*args, **common_kw)
        self.linear_attn_backend.init_forward_metadata(*args, **common_kw, **mamba_kw)

    def init_cuda_graph_state(self, max_bs: int, seq_lens_buf: torch.Tensor, **kwargs):
        # kwargs (e.g. paged_cache_group_specs, so the full backend sheds
        # state-family groups) are forwarded through the shared signature
        # filter: the full backend is user-selectable and may have a narrow
        # signature (e.g. TRTLLM MHA takes only (max_bs, seq_lens_buf)), and
        # the mamba backend keeps its narrow signature today.
        init_backend_cuda_graph_state(
            self.full_attn_backend, max_bs, seq_lens_buf, **kwargs
        )
        init_backend_cuda_graph_state(
            self.linear_attn_backend, max_bs, seq_lens_buf, **kwargs
        )

    def register_step_counter(self, step_counter):
        # Hybrid layerwise transfer needs one global step per model layer,
        # including both full-attention and mamba layers. Record steps in this
        # wrapper instead of in child backends to avoid double counting.
        self.step_counter = step_counter

    def init_forward_metadata_capture_cuda_graph(self, *args, **kwargs):
        common_kw, mamba_kw = self._split_mamba_kwargs(kwargs)
        self.full_attn_backend.init_forward_metadata_capture_cuda_graph(
            *args, **common_kw
        )
        self.linear_attn_backend.init_forward_metadata_capture_cuda_graph(
            *args, **common_kw, **mamba_kw
        )

    def init_forward_metadata_replay_cuda_graph(self, *args, **kwargs):
        common_kw, mamba_kw = self._split_mamba_kwargs(kwargs)
        self.full_attn_backend.init_forward_metadata_replay_cuda_graph(
            *args, **common_kw
        )
        self.linear_attn_backend.init_forward_metadata_replay_cuda_graph(
            *args, **common_kw, **mamba_kw
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

    def reset_current_inputs(self, *args, **kwargs):
        if self.linear_attn_backend is None:
            return
        if hasattr(self.linear_attn_backend, "reset_current_inputs"):
            self.linear_attn_backend.reset_current_inputs(*args, **kwargs)

    def update_mamba_state_after_mtp_verify(self, accepted_length, model):
        # Flat contract pool: the per-position verify states live in the
        # scratch; commit the accepted position back to the state slab.
        commit = getattr(self.linear_attn_backend, "flat_commit_verified_state", None)
        if commit is not None and getattr(
            self.linear_attn_backend, "_verify_commit_ctx", None
        ) is not None:
            commit(accepted_length)
            return
        # mamba_cache_indices are input rows during target-verify. The first
        # output row is always the scheduler-owned working slot, so use the
        # output index table to update the next-round input pointer.
        output_indices = self.linear_attn_backend.forward_metadata.mamba_output_indices
        if output_indices is None:
            return
        req_pool_indices = (
            self.linear_attn_backend.forward_metadata.mamba_req_pool_indices
        )
        if req_pool_indices is None:
            return
        pool = getattr(self.linear_attn_backend, "pool", None)
        if pool is None:
            if getattr(self.linear_attn_backend, "flat_state_active", False):
                return
            raise RuntimeError(
                "MTP verify produced mamba output indices but the linear "
                "attention backend has no SimpleMambaPool and flat state "
                "paging is inactive"
            )
        request_number = accepted_length.shape[0]
        pool.update_current_inputs_after_verify(
            req_pool_indices[:request_number],
            output_indices[:request_number],
            accepted_length,
        )
