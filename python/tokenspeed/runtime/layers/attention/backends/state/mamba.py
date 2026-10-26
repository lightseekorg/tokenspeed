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

"""Recurrent-state attention backend: Qwen3.5 GDN, the KDA base (Kimi-K3,
GLM-5.3) and Qwen4-exp build on it."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from tokenspeed_kernel.ops.attention.gdn import (
    GdnChainVerifyForm,
    GdnStateCheckpoints,
    gdn_chain_verify,
    gdn_chain_verify_form,
    gdn_chain_verify_supported,
    gdn_chunk_prefill,
    gdn_decode_mtp,
    gdn_decode_step,
    gdn_replay_commit,
    gdn_tree_verify_needs_node_states,
    validate_replay_commit_setup,
)
from tokenspeed_kernel.ops.attention.gdn.triton import (
    CAUSAL_CONV1D_BLOCK_M,
    CausalConv1dPrefillMetadata,
    build_causal_conv1d_prefill_metadata,
    fused_qkv_split_gdn_prefill,
    gdn_chain_verify_warm_batch_sizes,
)
from tokenspeed_kernel.ops.attention.gdn.triton import (
    prepare_prefill_state_inputs as _prepare_cache_prefill_state_inputs,
)
from tokenspeed_kernel.ops.attention.gdn.triton import (
    set_total_chunks_hint,
    set_total_chunks_hint_uniform,
)
from tokenspeed_kernel.ops.attention.kda.triton import (
    commit_state_pages,
    verify_state_blocks,
)
from tokenspeed_kernel.ops.conv import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from tokenspeed_kernel.ops.kvcache.triton import (
    compact_window_rows,
    copy_state_rows,
    state_verify_commit_rows,
    store_state_rows,
)
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.execution.breakable_cuda_graph import (
    scrub_padding_tail,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.query_shard import QueryShardPlan
from tokenspeed.runtime.layers.attention.backends.base import (
    AttentionBackend,
    reject_bounded_replay,
    reject_query_shard,
)
from tokenspeed.runtime.layers.attention.backends.state.checkpoint import (
    STATE_TABLE_SLOTS,
    InForwardStateCheckpoints,
    compute_state_block_indices,
    in_forward_state_checkpoints,
    scan_checkpoint_plan,
)
from tokenspeed.runtime.layers.attention.backends.state.utils import row_stride_i32
from tokenspeed.runtime.layers.attention.backends.support import TreeSupport
from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    cache_debug_enabled,
)
from tokenspeed.runtime.layers.attention.linear.gdn import fused_gdn_gating

logger = logging.getLogger(__name__)

# The replay tape holds one row pointer (PTR0..PTR7) and width (USER0..USER7) per group.
_TAPE_MAX_STATE_GROUPS = 8

if TYPE_CHECKING:
    from tokenspeed_kernel.ops.metadata import PrepTape

    from tokenspeed.runtime.layers.attention.backends.paged.tree_verify import (
        TreeVerifyInputs,
    )
    from tokenspeed.runtime.layers.attention.configs.base import (
        AttnConfig,
        SoftmaxAttnConfig,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


def _packed_qkv_views(
    mixed_qkv: torch.Tensor,
    *,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_q: int,
    head_k: int,
    head_v: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expose packed Q/K/V rows without changing storage or materializing copies."""
    seq_len = mixed_qkv.shape[0]
    widths = (
        num_q_heads * head_q,
        num_k_heads * head_k,
        num_v_heads * head_v,
    )
    query, key, value = mixed_qkv.split(widths, dim=-1)
    return (
        query.view(1, seq_len, num_q_heads, head_q),
        key.view(1, seq_len, num_k_heads, head_k),
        value.view(1, seq_len, num_v_heads, head_v),
    )


def _reject_skip_term(D: torch.Tensor | None) -> None:
    """GDN and KDA recurrences have no ``D * x`` skip term; refuse one."""
    if D is not None:
        raise ValueError("the GDN/KDA recurrence has no D skip term")


def _prepare_gdn_decode_state_path(
    ssm_states: torch.Tensor,
    initial_state_indices: torch.Tensor,
    output_state_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, str | None]:
    """Select a safe decode solution while preserving graph padding indices.

    FlashInfer's FP32 kernels skip negative state rows, and the portable Triton
    kernels guard negative reads and writes for both FP32 and BF16 state. The
    FlashInfer BF16 path instead redirects padding to row 0 — the arena's
    reserved null page, which must stay zero for every fresh-state read.
    Until that kernel supports a padding mask, route BF16 state through
    Triton and keep ``-1`` unchanged.
    """
    return (
        initial_state_indices,
        output_state_indices,
        gdn_decode_solution(ssm_states.dtype),
    )


def gdn_decode_solution(ssm_dtype: torch.dtype) -> str | None:
    """The GDN decode/verify solution forced for a state dtype (see above)."""
    return "triton" if ssm_dtype == torch.bfloat16 else None


def _build_cu_extend_seq_lens_cpu(
    extend_seq_lens_cpu: torch.Tensor, expected_len: int
) -> torch.Tensor:
    """Host prefix sum of the scheduler's extend lengths, as an int64 tensor.

    The contents must equal ``query_start_loc`` — a wrong copy silently
    corrupts the kernels' host chunk plans. Both are built together here in
    ``init_forward_metadata`` (mirroring MHA's ``cu_extend_seq_lens_cpu``),
    so a length misalignment can only mean a caller broke that contract:
    fail loudly instead of silently degrading to a stream-synchronizing
    boundary re-read inside the kernel.

    Args:
        extend_seq_lens_cpu: CPU per-sequence extend lengths.
        expected_len: ``query_start_loc.numel()`` of the batch.

    Returns:
        Host int64 tensor ``[0, lens[0], lens[0]+lens[1], ...]`` with
        ``expected_len`` entries.

    Raises:
        RuntimeError: the lengths disagree with ``query_start_loc`` on the
            sequence count.
    """
    if extend_seq_lens_cpu is None:
        raise RuntimeError("host extend lengths are required for prefill")
    if extend_seq_lens_cpu.numel() + 1 != expected_len:
        raise RuntimeError(
            "host extend lengths disagree with query_start_loc on the "
            f"sequence count: {extend_seq_lens_cpu.numel() + 1} boundaries "
            f"vs {expected_len} entries"
        )
    bounds = torch.zeros(expected_len, dtype=torch.int64)
    torch.cumsum(extend_seq_lens_cpu.to(torch.int64), dim=0, out=bounds[1:])
    return bounds


@dataclass
class MambaForwardMetadata:
    query_start_loc: torch.Tensor | None
    # Boundary tensor used by the recurrent prefill scan. It aliases the
    # int32 convolution boundary for GDN and is a once-per-forward int64 copy
    # for KDA backends whose native wrapper and launch-plan memo use int64.
    scan_query_start_loc: torch.Tensor | None
    mamba_output_indices: torch.Tensor | None = None
    extend_seq_lens_cpu: torch.Tensor | None = None
    # Host int64 prefix sum of extend_seq_lens_cpu, equal to
    # query_start_loc's contents; built once per extend batch (mirroring
    # MHA's field of the same name) and reused by every layer's prefill scan
    # so no kernel re-reads the device boundaries (a stream-synchronizing
    # D2H per layer per chunk). Fresh per batch — never mutated in place.
    cu_extend_seq_lens_cpu: torch.Tensor | None = None
    # Device int64 mirror for scan ABIs; keep the int32 query/conv indices.
    # Owned by this extend/mixed forward, not cast again by every KDA layer.
    query_start_loc_int64: torch.Tensor | None = None
    # One read-only convolution schedule per forward, reused across layers.
    # Never refill a previous forward's storage while its kernels are in flight.
    conv_prefill_metadata: CausalConv1dPrefillMetadata | None = None
    # Per-state-group metadata is gathered once per group and batch;
    # layers select their entry via ``pool.state_group_by_layer[layer_id]``.
    state_in_blocks_by_group: dict[str, torch.Tensor] | None = None
    state_out_blocks_by_group: dict[str, torch.Tensor] | None = None
    # The in-forward State checkpoints this extend forward writes, and the
    # chunked scan's plan for them (its interval, buffer and the buffer row of
    # each checkpoint); None when the forward writes none.
    state_checkpoints: InForwardStateCheckpoints | None = None
    scan_state_checkpoints: GdnStateCheckpoints | None = None
    scan_checkpoint_rows: torch.Tensor | None = None

    @property
    def prefill_token_extent(self) -> int | None:
        """Live packed extent; capacity metadata may override storage geometry."""
        if self.extend_seq_lens_cpu is None:
            return None
        return int(sum(int(x) for x in self.extend_seq_lens_cpu))

    def scrub_padding(self, *tensors: torch.Tensor | None) -> None:
        """Zero the padded tail rows of token-shaped inputs in place."""
        scrub_padding_tail(self.prefill_token_extent, *tensors)


@dataclass
class _MixedStateMetadata:
    num_extends: int
    prefill_tokens: int
    prefill: MambaForwardMetadata
    decode: MambaForwardMetadata


@dataclass
class _GDNReplayWorkspace:
    payload: torch.Tensor
    parameters: torch.Tensor
    layer_ids: tuple[int, ...]
    initialized_layers: set[int]
    geometry: tuple[int, int, int, int]
    state_dtype: torch.dtype

    def __post_init__(self) -> None:
        validate_replay_commit_setup(
            self.payload,
            self.parameters,
            geometry=self.geometry,
            state_dtype=self.state_dtype,
        )


_StateLayerGeometry = tuple[
    tuple[int, tuple[int, ...], torch.dtype, tuple[int, ...], torch.dtype], ...
]


class MambaAttnBackend(AttentionBackend):
    """Attention backend for Mamba/GDN linear attention layers."""

    # This backend consumes state tables (two slots per request, see
    # ``state.checkpoint``); history groups belong to the
    # full-attention sub-backend. The hybrid wrapper unions the sub-backends'
    # declarations, so a Kimi-K3 contract (history + state) is covered once
    # both consumers exist.
    cache_consumer_attn_kinds = frozenset({"state"})
    _verify_reads_committed_recurrent_state: bool = False
    _verify_packed_qkv_views: bool = False
    _decode_packed_qkv_views: bool = False
    # Whether the family's chunked tree verify scan fills the ReplaySSM payload.
    _verify_scan_writes_replay: bool = False
    # Whether the family's prefill scan stores in-forward State checkpoints
    # (SchedulerConfig::state_checkpoint_in_forward).
    _stores_in_forward_state_checkpoints: bool = False
    # Whether the family's verify scan may run GDN's chain form (gdn_chain_verify).
    _verify_scan_runs_chain_form: bool = False

    def __init__(self, config: AttnConfig, spec: SoftmaxAttnConfig):
        super().__init__(config, spec)
        self.pad_slot_id = -1
        self.forward_metadata: MambaForwardMetadata | None = None
        self._mixed_forward_metadata: _MixedStateMetadata | None = None
        self._reset_graph_state()
        self.speculative_num_draft_tokens = self.spec_num_tokens
        self.state_paging_active = False
        self._state_group_ids: tuple[str, ...] = ()
        self._state_layer_geometry: _StateLayerGeometry = ()
        linear_attn = config.component(LinearAttnConfig)
        self.replay_ssm = linear_attn is not None and bool(linear_attn.replay_ssm)
        self._gdn_replay: _GDNReplayWorkspace | None = None
        # ReplaySSM tree verify: node states shared by all layers; payload addresses for the commit.
        self.draft_tree = linear_attn is not None and bool(linear_attn.draft_tree)
        self._tree_node_state_workspace = (
            linear_attn is not None and linear_attn.tree_node_state_workspace
        )
        self._tree_node_states: torch.Tensor | None = None
        self._replay_payload_addresses: torch.Tensor | None = None
        self._replay_payload_rows: torch.Tensor | None = None
        # Draft-tree verify (bind_tree_verify): per-node parents.
        self.tree_verify: TreeVerifyInputs | None = None
        self._verify_scratch = None
        self._verify_commit_ctx = None
        self._verify_copy_tables: dict[str, torch.Tensor | int | None] | None = None

    @property
    def kv_pool(self) -> CachePool | None:
        return self.cache_pool

    @property
    def stores_in_forward_state_checkpoints(self) -> bool:
        return self._stores_in_forward_state_checkpoints

    def _reset_graph_state(self) -> None:
        """Index buffers init_cuda_graph_state rebuilds; a rebind keeps them."""
        self.query_start_loc_list: list[torch.Tensor] = []
        self.cached_cuda_graph_decode_query_start_loc: torch.Tensor | None = None
        self.cached_cuda_graph_verify_query_start_loc: torch.Tensor | None = None
        # CUDA-graph buffers: one persistent (state_in/state_out) [bs] buffer
        # pair per state group for every bs up to max_decode_bs (the runner
        # sizes them, never the capture ladder). Values are keyed by group ID
        # and indexed by ``bs - 1``.
        self.state_in_by_group: dict[str, list[torch.Tensor]] = {}
        self.state_out_by_group: dict[str, list[torch.Tensor]] = {}
        self._verify_seed_dst_cache: dict[tuple[int, int, int], torch.Tensor] = {}
        self._verify_grid_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._chain_ancestors_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._verify_base_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._qsl_dirty: list[bool] = []
        self._qsl_last_mode: list[tuple[ForwardMode, bool] | None] = []
        # Tapes bake in the index buffers' addresses, so they die with them.
        self._replay_state_tapes: dict[int, PrepTape] = {}

    def set_kv_pool(self, kv_pool: CachePool) -> None:
        """Bind a unified pool that publishes state groups and component views."""
        self.set_cache_pool(kv_pool)

    def _state_geometry(
        self, kv_pool: CachePool
    ) -> tuple[tuple[str, ...], _StateLayerGeometry]:
        """State group ids and per-layer state geometry, or raise."""
        contract = kv_pool.arena.runtime_contract
        if contract is None:
            raise RuntimeError(
                "MambaAttnBackend requires a KV pool with a runtime cache contract"
            )
        if getattr(kv_pool, "state_group_by_layer", None) is None or not callable(
            getattr(kv_pool, "get_component", None)
        ):
            raise RuntimeError(
                "MambaAttnBackend requires state_group_by_layer and get_component()"
            )
        claimed_groups = set(kv_pool.state_group_by_layer.values())
        state_specs = tuple(
            spec
            for spec in contract.group_specs
            if spec.group_id in claimed_groups and spec.attn_kind == "state"
        )
        state_group_ids = tuple(spec.group_id for spec in state_specs)
        if not state_group_ids or set(state_group_ids) != claimed_groups:
            raise RuntimeError(
                "MambaAttnBackend requires a state group for every recurrent layer"
            )
        layer_geometry = tuple(
            (
                layer,
                tuple(conv.shape[1:]),
                conv.dtype,
                tuple(recurrent.shape[1:]),
                recurrent.dtype,
            )
            for layer in sorted(kv_pool.state_group_by_layer)
            for conv, recurrent in (
                (
                    kv_pool.get_component(layer, "conv_state"),
                    kv_pool.get_component(layer, "recurrent_state"),
                ),
            )
        )
        geometry = (tuple(sorted(state_group_ids)), layer_geometry)
        if self.kv_pool is not None and geometry != (
            tuple(sorted(self._state_group_ids)),
            self._state_layer_geometry,
        ):
            raise RuntimeError(
                "MambaAttnBackend cannot rebind a pool of a different state geometry"
            )
        return geometry

    def validate_cache_pool(self, cache_pool: CachePool) -> None:
        super().validate_cache_pool(cache_pool)
        self._state_geometry(cache_pool)

    def _publish_cache_pool(self, cache_pool: CachePool) -> None:
        """Latch the state geometry; init and preallocate rebuild the rest."""
        state_group_ids, layer_geometry = self._state_geometry(cache_pool)
        super()._publish_cache_pool(cache_pool)
        self._state_group_ids = state_group_ids
        self.state_paging_active = True
        self._state_layer_geometry = layer_geometry
        # init_cuda_graph_state and preallocate_verify_workspace rebuild these for the new pool.
        self._verify_scratch = None
        self._verify_copy_tables = None
        self._verify_commit_ctx = None
        self._mixed_forward_metadata = None
        self._gdn_replay = None
        self._tree_node_states = None
        self._replay_payload_addresses = None
        self._replay_payload_rows = None
        self._replay_state_tapes = {}
        self.forward_metadata = None

    @staticmethod
    def _decode_state_tokens_before(bs: int, seq_lens: torch.Tensor) -> torch.Tensor:
        """Per-request token count before a q_len-1 decode: ``seq_lens``
        counts the tokens computed AFTER this forward."""
        return seq_lens[:bs] - 1

    def _extend_state_tokens_before(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        num_extends: int,
        extend_prefix_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Per-request token count before an extend / MIXED forward: the
        leading extend rows start at their cached prefix, the trailing decode
        rows ``spec_num_tokens`` before ``seq_lens``. Computed once per batch
        and shared by every state group."""
        after = seq_lens[:bs]
        extend_before = extend_prefix_lens[:num_extends].to(
            device=after.device, dtype=after.dtype
        )
        return torch.cat((extend_before, after[num_extends:] - self.spec_num_tokens))

    def _state_layer_ids(self) -> list[int]:
        """Recurrent layer ids backed by the unified cache pool."""
        return sorted(self.kv_pool.state_group_by_layer)

    def _state_groups(self) -> tuple[str, ...]:
        return self._state_group_ids

    def _state_rows(
        self, block_tables: Mapping[str, torch.Tensor], group_id: str
    ) -> torch.Tensor:
        """This forward's raw block table for one state group.

        State paging reads the delivered per-group dict directly (the same
        complete mapping every node receives); a missing declared group is a
        delivery bug, not a fallback point.
        """
        rows = block_tables.get(group_id)
        if rows is None:
            raise RuntimeError(
                f"state paging: block_tables is missing state group "
                f"{group_id!r} (delivered: {sorted(map(str, block_tables))})"
            )
        return rows

    def _state_group_for(self, layer_id: int) -> str:
        try:
            return self.kv_pool.state_group_by_layer[layer_id]
        except KeyError as exc:
            raise RuntimeError(f"layer {layer_id} has no state cache group") from exc

    def _state_components(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.kv_pool.get_component(layer_id, "conv_state"),
            self.kv_pool.get_component(layer_id, "recurrent_state"),
        )

    @property
    def deferred_state_commit(self) -> bool:
        """Whether target decode publishes state after acceptance."""
        return not self.is_draft and self.spec_num_tokens > 1

    def _prepare_verify_records(
        self, bs: int, seq_lens: torch.Tensor, draft_token_num: int
    ) -> None:
        """Refresh family-specific recovery records before eager or graph verify.

        State-page ownership remains with the scheduler; families may use the
        committed positions only to identify their current-round records.
        """

    def _verify_state_blocks(
        self,
        bs: int,
        block_tables: Mapping[str, torch.Tensor],
        *,
        pages_out: Mapping[str, torch.Tensor],
        live_pages_out: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Target-verify state paging: per-group committed-state pages.

        Verify reads the committed state from the input slot (slot 0 when it
        holds a block, else the live slot 1); speculative outputs stay out of
        the state slab, and the accepted state is committed back to slot 1 by
        ``commit_state_after_acceptance``. Writes each group's in
        pages into ``pages_out[group][:bs]`` and returns the per-group tables
        (kept for the commit's page resolve).
        """
        tables: dict[str, torch.Tensor] = {}
        for group_id in self._state_groups():
            rows = self._state_rows(block_tables, group_id)
            verify_state_blocks(
                rows,
                batch_size=bs,
                pages_out=pages_out[group_id],
                live_pages_out=(
                    None if live_pages_out is None else live_pages_out[group_id]
                ),
            )
            tables[group_id] = rows
        return tables

    def _ensure_verify_scratch(self, bs: int, draft_token_num: int) -> None:
        """Lazily allocate graph-stable verify scratch and replay inputs."""
        max_bs = max(len(self.query_start_loc_list), bs)
        rows_needed = max_bs * (draft_token_num + 1)
        scratch = self._verify_scratch
        if scratch is not None and next(iter(scratch.values()))[0].shape[0] >= (
            rows_needed
        ):
            return
        scratch = {}
        self._verify_copy_tables = None
        layer_ids = tuple(self._state_layer_ids())
        for layer_id in layer_ids:
            conv, ssm = self._state_components(layer_id)
            scratch[layer_id] = (
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
            if (
                self.draft_tree
                and self._tree_node_state_workspace
                and gdn_tree_verify_needs_node_states(draft_token_num)
            ):
                self._tree_node_states = torch.zeros(
                    (max_bs, draft_token_num, *ssm.shape[1:]),
                    dtype=ssm.dtype,
                    device=ssm.device,
                )
            if self.draft_tree:
                payload = self._gdn_replay.payload
                self._replay_payload_addresses = torch.tensor(
                    [payload[i].data_ptr() for i in range(payload.shape[0])],
                    dtype=torch.int64,
                    device=self.device,
                )
                self._replay_payload_rows = torch.arange(
                    max_bs * draft_token_num, dtype=torch.int32, device=self.device
                )
        self._verify_scratch = scratch

    def preallocate_verify_workspace(self, max_bs: int, draft_token_num: int) -> int:
        """Allocate graph-stable verify state and return its byte size."""
        if not self.state_paging_active or self.is_draft:
            return 0
        self._ensure_verify_scratch(max_bs, draft_token_num)
        self._warm_chain_verify(max_bs, draft_token_num)
        total = sum(
            tensor.nbytes
            for layer_scratch in self._verify_scratch.values()
            for tensor in layer_scratch
            if tensor is not None
        )
        if self._gdn_replay is not None:
            total += self._gdn_replay.payload.nbytes
            total += self._gdn_replay.parameters.nbytes
        if self._tree_node_states is not None:
            total += self._tree_node_states.nbytes
        return total

    def _warm_chain_verify(self, max_bs: int, width: int) -> None:
        """Compile the chain verify's launch tiers up to ``max_bs`` (all-padding launches)."""
        if not self.replay_ssm or self.draft_tree:
            return
        num_k_heads, num_v_heads, head_k_dim, head_v_dim = self._gdn_replay.geometry
        sizes = gdn_chain_verify_warm_batch_sizes(
            (
                bs
                for bs in range(1, max_bs + 1)
                if self._chain_verify_form(bs, width, num_v_heads)
                is GdnChainVerifyForm.CHAIN
            ),
            num_v_heads,
            width,
        )
        if not sizes:
            return
        _, ssm = self._state_components(self._state_layer_ids()[0])
        rows = sizes[-1] * width
        # Dense rows, as the split writes q/k/v (the gates' layout is not specialized).
        query = torch.zeros(
            (rows, num_k_heads, head_k_dim), dtype=self.dtype, device=self.device
        )
        value = torch.zeros(
            (rows, num_v_heads, head_v_dim), dtype=self.dtype, device=self.device
        )
        gates = torch.zeros((rows, num_v_heads), dtype=self.dtype, device=self.device)
        parameters = torch.zeros(num_v_heads, dtype=torch.float32, device=self.device)
        padding = torch.full(
            (sizes[-1],), self.pad_slot_id, dtype=torch.int32, device=self.device
        )
        for bs in sizes:
            tokens = bs * width
            q = query[:tokens].view(bs, width, num_k_heads, head_k_dim)
            gdn_chain_verify(
                q,
                q,
                value[:tokens].view(bs, width, num_v_heads, head_v_dim),
                A_log=parameters,
                a=gates[:tokens].view(bs, width, num_v_heads),
                dt_bias=parameters,
                b=gates[:tokens].view(bs, width, num_v_heads),
                initial_state=ssm,
                initial_state_indices=padding[:bs],
                scale=None,
                use_qk_l2norm=True,
            )

    def _verify_copy_tables_get(self) -> dict[str, torch.Tensor | int | None]:
        """Pointer tables for the batched verify state copies and replay:
        per-layer base addresses, row strides, and state-group selectors.
        Rebuilt only when the scratch is reallocated so CUDA graph capture
        can record stable tensors."""
        tables = self._verify_copy_tables
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
            conv_src_st.append(row_stride_i32(conv))
            conv_dst_st.append(row_stride_i32(conv_scratch))
            ssm_src.append(ssm.data_ptr())
            ssm_src_st.append(row_stride_i32(ssm))
            ssm_element_st.append(ssm.stride(0))
            if not self.replay_ssm:
                ssm_dst.append(ssm_scratch.data_ptr())
                ssm_dst_st.append(row_stride_i32(ssm_scratch))
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
        cache = self._verify_seed_dst_cache
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
        tables = self._verify_copy_tables_get()
        state_in_by_group = self.forward_metadata.state_in_blocks_by_group
        sin_stack = torch.stack(
            [state_in_by_group[group_id][:bs] for group_id in self._state_groups()]
        )
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
        if not self.replay_ssm and not self._verify_reads_committed_recurrent_state:
            copy_state_rows(
                tables["ssm_comp"],
                tables["ssm_scratch"],
                src_rows,
                self._verify_seed_dst_rows(bs, draft_token_num),
                row_bytes=tables["ssm_bytes"],
                src_row_strides=tables["ssm_comp_stride"],
                dst_row_strides=tables["ssm_scratch_stride"],
            )

    def bind_tree_verify(self, inputs: TreeVerifyInputs) -> None:
        """Verify draft trees: each node's conv window and recurrent state
        continue from its parent's scratch row; commit reads the accepted path."""
        if self.replay_ssm and not self.draft_tree:
            raise RuntimeError(
                "ReplaySSM draft-tree verify needs the draft-tree workspaces the "
                "cache recipe plans (LinearAttnConfig.draft_tree)"
            )
        self.tree_verify = inputs

    def _tree_parents(self, bs: int) -> torch.Tensor | None:
        return None if self.tree_verify is None else self.tree_verify.parent[:bs]

    def _verify_ancestors(
        self, bs: int, draft_token_num: int, num_value_heads: int
    ) -> torch.Tensor | None:
        """Ancestor masks for the GDN verify: the draft tree's, or a ReplaySSM
        chain's where it verifies in the tree form (a one-path tree).
        Memoized per (bs, T): CUDA-graph capture records the tensor's storage."""
        if self.tree_verify is not None:
            return self.tree_verify.mask.view(-1, self.tree_verify.num_nodes)[:bs]
        if (
            self._chain_verify_form(bs, draft_token_num, num_value_heads)
            is not GdnChainVerifyForm.TREE
        ):
            return None
        key = (bs, draft_token_num)
        ancestors = self._chain_ancestors_cache.get(key)
        if ancestors is None:
            steps = torch.arange(draft_token_num, device=self.device)
            ancestors = ((1 << (steps + 1)) - 1).repeat(bs, 1)
            self._chain_ancestors_cache[key] = ancestors
        return ancestors

    def _chain_verify_form(
        self, bs: int, width: int, num_value_heads: int
    ) -> GdnChainVerifyForm:
        """The verify form of ``bs`` chains of ``width`` rows, from the shape alone."""
        if not self.replay_ssm:
            return GdnChainVerifyForm.STEP
        state_shape, state_dtype = self._state_layer_geometry[0][3:]
        chain_form = self._verify_scan_runs_chain_form and gdn_chain_verify_supported(
            state_shape[-1], state_shape[-2], state_dtype
        )
        return gdn_chain_verify_form(
            width, bs * num_value_heads, state_dtype, chain_form=chain_form
        )

    def _pack_projection(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor | None,
        b: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """The conv input and gates as the dense decode kernels read them;
        families whose models hand over dense tensors return them as is."""
        return mixed_qkv, a, b

    def _verify_scratch_grid(self, bs: int, draft_token_num: int) -> torch.Tensor:
        """Scratch row grid ``[bs, draft_token_num]``: row ``req*(T+1)`` is
        the seeded init window, rows ``req*(T+1)+1+t`` the per-position
        outputs. Memoized per (bs, T): CUDA-graph capture records the tensor's
        storage, so replays must present the identical tensor."""
        cache = self._verify_grid_cache
        grid = cache.get((bs, draft_token_num))
        if grid is not None:
            return grid
        base = self._verify_scratch_base_rows(bs, draft_token_num).unsqueeze(1)
        steps = torch.arange(
            1, draft_token_num + 1, dtype=torch.int32, device=self.device
        ).unsqueeze(0)
        grid = base + steps
        cache[(bs, draft_token_num)] = grid
        return grid

    def _verify_scratch_base_rows(self, bs: int, draft_token_num: int) -> torch.Tensor:
        """Graph-stable scratch initialization row for each request."""
        cache = self._verify_base_cache
        key = (bs, draft_token_num)
        rows = cache.get(key)
        if rows is None:
            rows = torch.arange(bs, dtype=torch.int32, device=self.device) * (
                draft_token_num + 1
            )
            cache[key] = rows
        return rows

    def _resolve_verify_commit_pages(
        self, accepted_length: torch.Tensor, group_ids: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve the pending verify's write pages (every group's live slot)
        in the supplied group order, and the clamped accepted step counts."""
        tables, draft_token_num, _ = self._verify_commit_ctx
        bs = accepted_length.shape[0]
        write_stack = torch.empty(
            (len(group_ids), bs), dtype=torch.int32, device=accepted_length.device
        )
        steps = torch.empty(bs, dtype=torch.int32, device=accepted_length.device)
        for out_row, group_id in enumerate(group_ids):
            commit_state_pages(
                accepted_length,
                tables[group_id],
                batch_size=bs,
                draft_tokens=draft_token_num,
                pages_out=write_stack,
                out_row=out_row,
                steps_out=steps,
            )
        return write_stack, steps

    def _replay_commit(
        self, payload: torch.Tensor, parameters: torch.Tensor, **tables: Any
    ) -> None:
        """Rebuild every layer's accepted state from the replay payload.

        Families with another recurrence replace the kernel; the payload,
        tables and page resolution stay shared.
        """
        gdn_replay_commit(payload, parameters, **tables)

    def tree_support(self) -> TreeSupport:
        return TreeSupport(
            verify_blocker=None,
            draft_blocker="draft-tree lanes have no linear-attention path",
        )

    def commit_verified_state(
        self, accepted_length: torch.Tensor, *, accepted_path: torch.Tensor | None
    ) -> None:
        """Commit the accepted draft prefix with fused per-group page resolves;
        ``accepted_path`` is the accepted draft-tree path, ``None`` for a chain."""
        ctx = self._verify_commit_ctx
        if ctx is None:
            return
        _, draft_token_num, read_pages_by_group = ctx
        bs = accepted_length.shape[0]
        group_ids = self._state_groups()
        write_stack, steps = self._resolve_verify_commit_pages(
            accepted_length, group_ids
        )
        copy_tables = self._verify_copy_tables_get()
        src_tiled, dst_rows = torch.empty(
            (2, copy_tables["num_layers"] * bs),
            dtype=torch.int32,
            device=accepted_length.device,
        ).unbind(0)
        source_steps = steps
        if accepted_path is not None:
            # Scratch row step s holds node s - 1: the last accepted node is path[steps - 1].
            last = (steps - 1).long().unsqueeze(1)
            source_steps = accepted_path.gather(1, last).squeeze(1) + 1
        state_verify_commit_rows(
            source_steps,
            write_stack,
            src_tiled,
            dst_rows,
            verify_width=draft_token_num,
            num_layers=copy_tables["num_layers"],
            group_indices=copy_tables["group_sel"],
        )
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
            if accepted_path is not None:
                # Replay reads the accepted tokens' payload rows in order from the window's front.
                compact_window_rows(
                    self._replay_payload_addresses,
                    self._replay_payload_rows[: bs * draft_token_num],
                    accepted_path,
                    row_bytes=replay.payload.shape[-1] * replay.payload.element_size(),
                )
            self._replay_commit(
                replay.payload,
                replay.parameters,
                state_addresses=copy_tables["ssm_comp"],
                state_row_strides=copy_tables["ssm_element_stride"],
                read_indices=torch.stack(
                    [read_pages_by_group[group_id][:bs] for group_id in group_ids]
                )
                .index_select(0, copy_tables["group_sel"])
                .to(torch.int32),
                write_indices=dst_rows.view(copy_tables["num_layers"], bs),
                accepted_length=steps,
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
        tokens_before: torch.Tensor,
        block_tables: Mapping[str, torch.Tensor],
        *,
        validate: bool | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Per-state-group (state_in, state_out) block-id mappings for this
        forward from the delivered per-group two-slot tables.

        ``tokens_before`` is the per-request token count before this forward
        (see the ``_*_state_tokens_before`` helpers); ``0`` reads no block. The
        resolve runs ONCE per state group per batch -- never per layer. State
        layers select their group's entry via
        ``pool.state_group_by_layer[layer_id]`` at forward time.

        validate: explicit True/False wins; None on the hot path
        validates only under TOKENSPEED_CACHE_DEBUG=1 (the checks host-sync).

        Returns:
            ``(state_in_blocks, state_out_blocks)`` mappings keyed by state
            group id, each value an int32 ``[bs]`` block-id tensor.
        """
        if validate is None:
            validate = cache_debug_enabled()
        state_in_blocks: dict[str, torch.Tensor] = {}
        state_out_blocks: dict[str, torch.Tensor] = {}
        for group_id in self._state_group_ids:
            state_in, state_out = compute_state_block_indices(
                self._state_rows(block_tables, group_id),
                tokens_before,
                validate=validate,
                group_id=group_id,
            )
            state_in_blocks[group_id] = state_in
            state_out_blocks[group_id] = state_out
        return state_in_blocks, state_out_blocks

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        *,
        block_tables: Mapping[str, torch.Tensor],
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_replay_lens_cpu: torch.Tensor,
        extend_prompt_lens_cpu: torch.Tensor,
        extend_with_prefix: bool,
        query_shard: QueryShardPlan | None,
        extend_state_checkpoint_lens_cpu: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        reject_query_shard(query_shard, "MambaStateBackend")
        del extend_prefix_lens_cpu, extend_with_prefix
        self._mixed_forward_metadata = None
        del extend_prompt_lens_cpu, kwargs
        reject_bounded_replay(extend_replay_lens_cpu, "MambaStateBackend")
        if not (forward_mode.is_extend_or_mixed() or forward_mode.is_idle()):
            raise RuntimeError(
                "Mamba decode metadata goes through refresh_decode_metadata; "
                f"init_forward_metadata only serves extend/mixed/idle ({forward_mode})"
            )
        if not 0 <= num_extends <= bs:
            raise ValueError("num_extends must be between 0 and bs")
        if forward_mode.is_idle():
            # Idle warmup carries no requests and never reaches the state
            # kernels (the router returns early); the rows only take the
            # decode query shape — one token, or the verify window under
            # speculation — so the warm-up forward is sized like a decode.
            tokens_per_req = self.spec_num_tokens
            query_start_loc = torch.arange(
                0,
                bs * tokens_per_req + 1,
                step=tokens_per_req,
                dtype=torch.int32,
                device=self.device,
            )
            if tokens_per_req > 1:
                set_total_chunks_hint_uniform(bs, tokens_per_req, query_start_loc)
            self.forward_metadata = MambaForwardMetadata(
                query_start_loc=query_start_loc,
                scan_query_start_loc=query_start_loc,
            )
            return

        # The extend rows lead with their new-token counts; a MIXED round's
        # decode rows each carry spec_num_tokens verify tokens.
        query_lens = torch.full(
            (bs,), self.spec_num_tokens, dtype=torch.int32, device=self.device
        )
        query_lens[:num_extends] = extend_seq_lens[:num_extends]
        query_start_loc = torch.zeros(bs + 1, dtype=torch.int32, device=self.device)
        torch.cumsum(query_lens, dim=0, out=query_start_loc[1:])
        extend_seq_lens_cpu = torch.cat(
            (
                extend_seq_lens_cpu[:num_extends],
                torch.full(
                    (bs - num_extends,), self.spec_num_tokens, dtype=torch.int32
                ),
            )
        )
        set_total_chunks_hint(extend_seq_lens_cpu, query_start_loc)
        cu_extend_seq_lens_cpu = _build_cu_extend_seq_lens_cpu(
            extend_seq_lens_cpu, query_start_loc.numel()
        )

        state_in_blocks_by_group = None
        state_out_blocks_by_group = None
        state_checkpoints = None
        scan_state_checkpoints = None
        scan_checkpoint_rows = None
        scan_query_start_loc = self._prepare_prefill_scan_query_start_loc(
            query_start_loc
        )
        if bs > 0:
            state_in_blocks_by_group, state_out_blocks_by_group = (
                self._cache_contract_state_blocks(
                    self._extend_state_tokens_before(
                        bs, seq_lens, num_extends, extend_prefix_lens
                    ),
                    block_tables,
                    validate=None,
                )
            )
            state_checkpoints = in_forward_state_checkpoints(
                extend_state_checkpoint_lens_cpu,
                extend_seq_lens_cpu,
                num_extends,
                block_tables,
                self._state_group_ids,
            )
        if state_checkpoints is not None:
            if not self._stores_in_forward_state_checkpoints:
                raise RuntimeError(
                    f"{type(self).__name__} cannot store in-forward State checkpoints"
                )
            scan_state_checkpoints, scan_checkpoint_rows = self._scan_state_checkpoints(
                state_checkpoints, extend_seq_lens_cpu
            )

        self.forward_metadata = MambaForwardMetadata(
            query_start_loc=query_start_loc,
            scan_query_start_loc=scan_query_start_loc,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            cu_extend_seq_lens_cpu=cu_extend_seq_lens_cpu,
            query_start_loc_int64=scan_query_start_loc.to(dtype=torch.int64),
            conv_prefill_metadata=build_causal_conv1d_prefill_metadata(
                query_start_loc, extend_seq_lens_cpu, CAUSAL_CONV1D_BLOCK_M
            ),
            state_in_blocks_by_group=state_in_blocks_by_group,
            state_out_blocks_by_group=state_out_blocks_by_group,
            state_checkpoints=state_checkpoints,
            scan_state_checkpoints=scan_state_checkpoints,
            scan_checkpoint_rows=scan_checkpoint_rows,
        )

        if forward_mode.is_mixed() and self.deferred_state_commit and num_extends < bs:
            self._prepare_mixed_verify_metadata(
                bs, num_extends, req_pool_indices, seq_lens, block_tables
            )

    def _scan_state_checkpoints(
        self,
        checkpoints: InForwardStateCheckpoints,
        scan_seq_lens_cpu: torch.Tensor,
    ) -> tuple[GdnStateCheckpoints, torch.Tensor]:
        """The scan checkpoint plan and float32 buffer for this forward's checkpoints.

        The plan is the chunked scan's ``scan_checkpoint_plan`` for this
        forward's in-forward checkpoints; the buffer is what the scan fills.

        Args:
            checkpoints: This forward's checkpoints.
            scan_seq_lens_cpu: Host ``[bs]`` tokens of each scanned sequence.

        Returns:
            ``(scan checkpoints, device int64 buffer row of each checkpoint)``.
        """
        every_n_tokens, cu_starts, rows = scan_checkpoint_plan(
            checkpoints, scan_seq_lens_cpu
        )
        ssm_states = self._state_components(self._state_layer_ids()[0])[1]
        states = torch.empty(
            (int(cu_starts[-1]), *ssm_states.shape[1:]),
            dtype=torch.float32,
            device=self.device,
        )
        return (
            GdnStateCheckpoints(
                every_n_tokens=every_n_tokens,
                states=states,
                cu_starts=cu_starts.to(device=self.device, dtype=torch.int32),
            ),
            rows.to(self.device),
        )

    def _prepare_mixed_verify_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        block_tables: Mapping[str, torch.Tensor],
    ) -> None:
        """Reuse ordinary prefill and decode metadata for their respective rows."""
        full_metadata = self.forward_metadata
        lengths = full_metadata.extend_seq_lens_cpu[:num_extends]
        boundaries = full_metadata.query_start_loc[: num_extends + 1]
        prefill = MambaForwardMetadata(
            query_start_loc=boundaries,
            scan_query_start_loc=full_metadata.scan_query_start_loc[: num_extends + 1],
            extend_seq_lens_cpu=lengths,
            cu_extend_seq_lens_cpu=full_metadata.cu_extend_seq_lens_cpu[
                : num_extends + 1
            ],
            query_start_loc_int64=full_metadata.query_start_loc_int64[
                : num_extends + 1
            ],
            conv_prefill_metadata=(
                build_causal_conv1d_prefill_metadata(
                    boundaries, lengths, CAUSAL_CONV1D_BLOCK_M
                )
                if num_extends
                else None
            ),
            state_in_blocks_by_group={
                group: pages[:num_extends]
                for group, pages in full_metadata.state_in_blocks_by_group.items()
            },
            state_out_blocks_by_group={
                group: pages[:num_extends]
                for group, pages in full_metadata.state_out_blocks_by_group.items()
            },
        )
        decode_bs = bs - num_extends
        self.refresh_decode_metadata(
            decode_bs,
            decode_bs,
            req_pool_indices[num_extends:bs],
            seq_lens[num_extends:bs],
            forward_mode=ForwardMode.DECODE,
            for_graph_replay=False,
            block_tables={
                group: table[num_extends:bs] for group, table in block_tables.items()
            },
        )
        self._mixed_forward_metadata = _MixedStateMetadata(
            num_extends, int(lengths.sum()), prefill, self.forward_metadata
        )
        self.forward_metadata = full_metadata

    def commit_state_after_acceptance(
        self,
        accepted_lengths: torch.Tensor,
        *,
        num_extends: int,
        accepted_path: torch.Tensor | None,
    ) -> None:
        if num_extends:
            # An ordinary mixed scan has already published its final state.
            if self._mixed_forward_metadata is None:
                return
            accepted_lengths = accepted_lengths[num_extends:]
            if accepted_path is not None:
                accepted_path = accepted_path[num_extends:]
        self.commit_verified_state(accepted_lengths, accepted_path=accepted_path)
        self._mixed_forward_metadata = None

    # ---- CUDA graph state ----

    def init_cuda_graph_state(self, max_bs: int, **kwargs):
        """Rebuild the index buffers before any graph that reads them is captured."""
        self._reset_graph_state()
        for i in range(max_bs):
            self.query_start_loc_list.append(
                torch.empty((i + 2,), dtype=torch.int32, device=self.device)
            )
            # Keep one graph-stable (state_in, state_out) buffer pair per state group.
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
            0, max_bs + 1, dtype=torch.int32, device=self.device
        )
        if self.speculative_num_draft_tokens > 0:
            # Need max_bs+1 entries (one per request + sentinel).
            # Each entry is request_index * spec_num_draft_tokens.
            self.cached_cuda_graph_verify_query_start_loc = torch.arange(
                0,
                (max_bs + 1) * self.speculative_num_draft_tokens,
                step=self.speculative_num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        self._qsl_dirty = [False] * max_bs
        self._qsl_last_mode = [None] * max_bs
        self._prepare_replay_state_tapes(max_bs)

    def _prepare_replay_state_tapes(self, max_bs: int) -> None:
        """Upload immutable tape descriptions before any serving refresh."""
        gids = self._state_group_ids
        if (
            not self.state_paging_active
            or torch.device(self.device).type != "cuda"
            or len(gids) > _TAPE_MAX_STATE_GROUPS
        ):
            return
        from tokenspeed_kernel.ops.metadata import PrepTape, Reg

        for bs in range(1, max_bs + 1):
            tape = PrepTape(self.device)
            for i, gid in enumerate(gids):
                sin = self.state_in_by_group[gid][bs - 1]
                sout = self.state_out_by_group[gid][bs - 1]
                tape.state_pages(
                    sin,
                    sout,
                    rows_ptr=Reg(Reg.PTR0 + i),
                    bs=Reg.REAL_BS,
                    row_stride=Reg(Reg.USER0 + i),
                )
                tape.filltail(sin, live=Reg.REAL_BS, total=bs, value=self.pad_slot_id)
                tape.filltail(sout, live=Reg.REAL_BS, total=bs, value=self.pad_slot_id)
            tape.finalize()
            self._replay_state_tapes[bs] = tape

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        **kwargs,
    ):
        is_target_verify = (
            forward_mode.is_decode_or_idle() and self.deferred_state_commit
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
            scan_query_start_loc=self.query_start_loc_list[bs - 1],
            mamba_output_indices=mamba_output_indices,
            state_in_blocks_by_group=state_in_blocks_by_group,
            state_out_blocks_by_group=state_out_blocks_by_group,
        )

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
        del req_pool_indices, kwargs
        self._mixed_forward_metadata = None
        real_bs = actual_bs
        num_padding = bs - actual_bs

        is_target_verify = (
            forward_mode.is_decode_or_idle() and self.deferred_state_commit
        )
        is_draft_extend = (
            forward_mode.is_decode_or_idle()
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
            state_in_blocks_by_group = {
                group_id: self.state_in_by_group[group_id][bs - 1]
                for group_id in self._state_groups()
            }
            state_out_blocks_by_group = {
                group_id: self.state_out_by_group[group_id][bs - 1]
                for group_id in self._state_groups()
            }
            if real_bs > 0:
                # The commit runs before the next refresh, so it may read the captured pages.
                verify_tables = self._verify_state_blocks(
                    real_bs,
                    block_tables,
                    pages_out=state_in_blocks_by_group,
                    live_pages_out=state_out_blocks_by_group,
                )
                self._verify_commit_ctx = (
                    verify_tables,
                    draft_token_num,
                    state_in_blocks_by_group,
                )
                self._prepare_verify_records(real_bs, seq_lens, draft_token_num)
            else:
                self._verify_commit_ctx = None
            for group_id in self._state_groups():
                if real_bs < bs:
                    state_in_blocks_by_group[group_id][real_bs:].fill_(self.pad_slot_id)
                    state_out_blocks_by_group[group_id][real_bs:].fill_(
                        self.pad_slot_id
                    )
        elif self.state_paging_active:
            # The state block resolve runs once per state group over the real
            # rows. Padded rows get pad_slot_id (-1), which state kernels skip,
            # so they never touch a live page. Validation defaults off on the
            # replay hot path (host sync); TOKENSPEED_CACHE_DEBUG=1 arms it.
            # bs==0 idle replay carries no operation-bound metadata; every row
            # is a dummy padded row, so skip the resolve entirely.
            state_in_blocks_by_group, state_out_blocks_by_group = (
                self._replay_contract_state_blocks(bs, real_bs, seq_lens, block_tables)
            )

        self.forward_metadata = MambaForwardMetadata(
            query_start_loc=self.query_start_loc_list[bs - 1],
            scan_query_start_loc=self.query_start_loc_list[bs - 1],
            mamba_output_indices=mamba_output_indices,
            state_in_blocks_by_group=state_in_blocks_by_group,
            state_out_blocks_by_group=state_out_blocks_by_group,
        )

    def _replay_contract_state_blocks(
        self,
        bs: int,
        real_bs: int,
        seq_lens: torch.Tensor,
        block_tables: Mapping[str, torch.Tensor],
    ) -> tuple[dict, dict]:
        """Fill the per-bs persistent state-block buffers for a decode replay.

        Fast path: one prep-tape launch resolves every group's (in, out) blocks
        straight into the persistent buffers and pads the tail (the eager
        chain is several launches per group plus copies/fills). Falls back to
        the eager chain when the tape preconditions do not hold (group count,
        debug validation, a table on the host).
        """
        gids = self._state_group_ids
        use_tape = (
            not cache_debug_enabled()
            and len(gids) <= _TAPE_MAX_STATE_GROUPS
            and all(
                gid in block_tables
                and block_tables[gid].is_cuda
                and block_tables[gid].dtype == torch.int32
                and block_tables[gid].shape[1] >= STATE_TABLE_SLOTS
                for gid in gids
            )
        )
        if use_tape:
            from tokenspeed_kernel.ops.metadata import Reg

            tape = self._replay_state_tapes[bs]
            regs = {Reg.REAL_BS: real_bs}
            for i, gid in enumerate(gids):
                rows = block_tables[gid]
                regs[Reg(Reg.PTR0 + i)] = rows
                regs[Reg(Reg.USER0 + i)] = rows.stride(0)
            tape.run(regs)
            return (
                {g: self.state_in_by_group[g][bs - 1] for g in gids},
                {g: self.state_out_by_group[g][bs - 1] for g in gids},
            )

        state_in_by = state_out_by = None
        if real_bs > 0:
            state_in_by, state_out_by = self._cache_contract_state_blocks(
                self._decode_state_tokens_before(real_bs, seq_lens),
                block_tables,
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
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        # Multi-token decode (target verify or drafter compound) reuses
        # the multi-token kernel path in forward_extend. `q` is None for
        # hybrid linear-attn layers; the token count comes from mixed_qkv.
        q_len_per_req = kwargs["mixed_qkv"].shape[0] // bs if bs > 0 else 1
        if q_len_per_req > 1 or self.deferred_state_commit:
            kwargs.setdefault("seq_len", kwargs["mixed_qkv"].shape[0])
            return self.forward_extend(
                q,
                k,
                v,
                layer,
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
        gate_lower_bound = kwargs.get("lower_bound")
        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        D = kwargs.get("D")
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
        )
        if fused_out is not None:
            return fused_out

        # Stride-aware decoders consume packed projection views directly.
        # Otherwise a family whose model hands over projection views makes them
        # dense here, and the shared fallback keeps its compact input layout.
        if not self._decode_packed_qkv_views:
            mixed_qkv, a, b = self._pack_projection(mixed_qkv, a, b)
            mixed_qkv = mixed_qkv.contiguous()
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            conv_weights,
            bias,
            activation,
            conv_state_indices=read_indices,
            output_state_indices=state_out_blocks.view(-1, 1),
            parent_indices=None,
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
            D=D,
            a=a,
            b=b,
            g_raw=g_raw,
            f_a_out=f_a_out,
            f_b_weight=f_b_weight,
            beta_raw=beta_raw,
            lower_bound=gate_lower_bound,
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
        D: torch.Tensor | None,
        a: torch.Tensor | None,
        b: torch.Tensor | None,
        g_raw: torch.Tensor | None,
        f_a_out: torch.Tensor | None,
        f_b_weight: torch.Tensor | None,
        beta_raw: torch.Tensor | None,
        lower_bound: float | None,
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
            D: Mamba2 per-head skip coefficient; None for GDN and KDA.
            a: GDN scalar-per-head decay input.
            b: GDN scalar-per-head beta input.
            g_raw: KDA per-channel gate, when the model precomputed it.
            f_a_out: KDA low-rank gate activation (gate GEMV source).
            f_b_weight: KDA second gate projection.
            beta_raw: KDA raw per-head beta logits.
            lower_bound: KDA decay clamp.

        Returns:
            ``[1, B, Hv, V]`` layer output.
        """
        _reject_skip_term(D)
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

    def _forward_mixed_state(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool,
        kwargs: dict,
    ) -> torch.Tensor:
        metadata = self._mixed_forward_metadata
        original_metadata = self.forward_metadata
        outputs = []
        token_inputs = ("mixed_qkv", "a", "b", "g_raw", "f_a_out", "beta_raw")
        segments = (
            (
                0,
                metadata.prefill_tokens,
                metadata.num_extends,
                metadata.prefill,
                ForwardMode.EXTEND,
            ),
            (
                metadata.prefill_tokens,
                metadata.prefill_tokens
                + (bs - metadata.num_extends) * self.spec_num_tokens,
                bs - metadata.num_extends,
                metadata.decode,
                ForwardMode.DECODE,
            ),
        )
        try:
            for begin, end, rows, part_metadata, mode in segments:
                if not rows:
                    continue
                inputs = dict(kwargs, seq_len=end - begin)
                for name in token_inputs:
                    value = inputs.get(name)
                    if value is not None:
                        inputs[name] = value[begin:end]
                self.forward_metadata = part_metadata
                output = self.forward_extend(
                    q,
                    k,
                    v,
                    layer,
                    token_to_kv_pool,
                    rows,
                    forward_mode=mode,
                    save_kv_cache=save_kv_cache,
                    **inputs,
                )
                outputs.append(output.squeeze(0) if output.ndim == 4 else output)
        finally:
            self.forward_metadata = original_metadata
        output = torch.cat(outputs, dim=0)
        padding = kwargs["seq_len"] - output.shape[0]
        if padding:
            output = torch.cat((output, output.new_zeros((padding, *output.shape[1:]))))
        return output

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        token_to_kv_pool,
        bs: int,
        forward_mode: ForwardMode,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if forward_mode.is_mixed() and self._mixed_forward_metadata is not None:
            return self._forward_mixed_state(
                q, k, v, layer, token_to_kv_pool, bs, save_kv_cache, kwargs
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
        D = kwargs.get("D")
        layer_id = kwargs["layer_id"]
        seq_len = kwargs["seq_len"]

        is_target_verify = (
            forward_mode.is_decode_or_idle() and self.deferred_state_commit
        )

        query_start_loc = self.forward_metadata.query_start_loc

        conv_replay = None
        conv_rows = None
        if is_target_verify:
            draft_token_num = self.speculative_num_draft_tokens
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
                layer_id=layer_id,
                bias=bias,
                g_raw=g_raw,
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
            if self.tree_verify is None and not self._verify_packed_qkv_views:
                # A chain verifies through the dense MTP kernels (FlashInfer's).
                mixed_qkv, a, b = self._pack_projection(mixed_qkv, a, b)
            if (
                self.tree_verify is None
                and self._chain_verify_form(
                    batch_size, draft_token_num, value_dim // attn_tp_size // head_v_dim
                )
                is GdnChainVerifyForm.CHAIN
            ):
                # The chain form's conv update writes the payload and dense q/k/v rows (no split).
                conv_replay = self._replay_inputs(
                    layer_id, seq_len, a, b, A_log=A_log, dt_bias=dt_bias
                )
                conv_rows = mixed_qkv.new_empty(mixed_qkv.shape).view(
                    batch_size, draft_token_num, -1
                )
            # Read the committed window and write per-position states into the
            # verify scratch. The accepted position is committed afterward.
            if layer_id == self._state_layer_ids()[0]:
                self._seed_verify_scratch_batched(batch_size, draft_token_num)
            conv_states = conv_scratch
            conv_read = self._verify_scratch_base_rows(batch_size, draft_token_num)
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
                parent_indices=self._tree_parents(batch_size),
                out=None if conv_rows is None else conv_rows.transpose(1, 2),
                replay_payload=conv_replay,
            )
            # needn't contiguous here.
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
        else:
            state_in_blocks, state_out_blocks, conv_states, ssm_states = (
                self._layer_state(layer_id)
            )
            recurrent_state, has_initial_states = _prepare_cache_prefill_state_inputs(
                conv_states,
                ssm_states,
                state_in_blocks,
                state_out_blocks,
            )
            conv_cache_indices = state_out_blocks
            extend_seq_lens_cpu = self.forward_metadata.extend_seq_lens_cpu

            # Zero padded rows so garbage can't reach recurrent state (see scrub_padding_tail).
            num_real_tokens = seq_len
            if extend_seq_lens_cpu is not None:
                num_real_tokens = self.forward_metadata.prefill_token_extent
                self.forward_metadata.scrub_padding(
                    mixed_qkv, a, b, g_raw, f_a_out, beta_raw
                )

            checkpoints = self.forward_metadata.state_checkpoints
            if checkpoints is not None:
                # The conv state at a checkpoint is the window of the
                # conv's inputs just before it (oldest first).
                window = conv_states.shape[-1]
                rows = checkpoints.token_index.unsqueeze(1) + torch.arange(
                    -window, 0, device=checkpoints.token_index.device
                )
                conv_states[
                    checkpoints.blocks_by_group[self._state_group_for(layer_id)].long()
                ] = (mixed_qkv[rows].transpose(1, 2).to(conv_states.dtype))

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
                prefill_metadata=self.forward_metadata.conv_prefill_metadata,
            ).transpose(0, 1)[:seq_len]
            if extend_seq_lens_cpu is not None:
                # The convolution allocates its output uninitialized and gives
                # the padded rows no work, so the rows a capacity scan reads
                # past the live tokens are zeroed again here.
                self.forward_metadata.scrub_padding(mixed_qkv)

        key_split_dim = key_dim // attn_tp_size
        value_split_dim = value_dim // attn_tp_size
        num_heads = key_split_dim // head_k_dim
        num_value_heads = value_split_dim // head_v_dim

        replay_inputs = None
        if is_target_verify and self.replay_ssm and conv_replay is None:
            replay_inputs = self._replay_inputs(
                layer_id, seq_len, a, b, A_log=A_log, dt_bias=dt_bias
            )

        # GDN's chunked tree verify (no node states) fills the replay payload as it reads k/v/a/b.
        scan_replay = None
        if (
            replay_inputs is not None
            and self._verify_scan_writes_replay
            and self.tree_verify is not None
            and self._tree_node_states is None
        ):
            scan_replay, replay_inputs = replay_inputs[0], None
        # KDA's verify consumes zero-copy strided views. When recurrent-state
        # replay is enabled, the split kernel also saves the persistent inputs
        # needed to reconstruct accepted state later.
        if (
            (
                self._verify_packed_qkv_views
                or scan_replay is not None
                or conv_replay is not None
            )
            and replay_inputs is None
            and mixed_qkv.stride(-1) == 1
            and is_target_verify
        ):
            query, key, value = _packed_qkv_views(
                mixed_qkv,
                num_q_heads=num_heads,
                num_k_heads=num_heads,
                num_v_heads=num_value_heads,
                head_q=head_k_dim,
                head_k=head_k_dim,
                head_v=head_v_dim,
            )
        else:
            valid_tokens = (
                query_start_loc[-1:]
                if current_platform().is_hopper and not is_target_verify
                else None
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
                valid_tokens=valid_tokens,
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
                D=D,
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
                replay_payload=scan_replay,
            )
        else:
            metadata = self.forward_metadata
            scan_query_start_loc = metadata.scan_query_start_loc
            if extend_seq_lens_cpu is not None:
                set_total_chunks_hint(extend_seq_lens_cpu, scan_query_start_loc)
            core_attn_out, last_recurrent_state = self._prefill_scan(
                query,
                key,
                value,
                recurrent_state,
                scan_query_start_loc,
                A_log=A_log,
                dt_bias=dt_bias,
                D=D,
                a=a,
                b=b,
                g_raw=g_raw,
                f_a_out=f_a_out,
                f_b_weight=f_b_weight,
                beta_raw=beta_raw,
                seq_len=seq_len,
                num_real_tokens=num_real_tokens,
                lower_bound=gate_lower_bound,
                cu_seqlens_cpu=metadata.cu_extend_seq_lens_cpu,
                state_checkpoints=metadata.scan_state_checkpoints,
            )
            # The final state goes to the live slot. Capacity metadata pads the
            # request dimension with -1 blocks whose dummy scan result the
            # masked writer drops.
            store_state_rows(last_recurrent_state, ssm_states, state_out_blocks)
            if metadata.state_checkpoints is not None:
                store_state_rows(
                    metadata.scan_state_checkpoints.states[
                        metadata.scan_checkpoint_rows
                    ],
                    ssm_states,
                    metadata.state_checkpoints.blocks_by_group[
                        self._state_group_for(layer_id)
                    ],
                )
            if extend_seq_lens_cpu is not None:
                # Padded output rows hold the dummy scan's results (or nothing);
                # the outer forward reads the whole bucket.
                self.forward_metadata.scrub_padding(
                    core_attn_out[0] if core_attn_out.ndim == 4 else core_attn_out
                )

        return core_attn_out

    def _replay_inputs(
        self,
        layer_id: int,
        seq_len: int,
        a: torch.Tensor,
        b: torch.Tensor | None,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """This layer's ReplaySSM payload rows and the gate rows they save."""
        replay = self._gdn_replay
        layer_slot = replay.layer_ids.index(layer_id)
        # Model-static A_log and dt_bias copy once per layer in warmup, so capture records no copy.
        if layer_id not in replay.initialized_layers:
            replay.parameters[layer_slot, 0].copy_(A_log)
            replay.parameters[layer_slot, 1].copy_(dt_bias)
            replay.initialized_layers.add(layer_id)
        # A family without a beta gate (Mamba2) repeats a in the unused b slot.
        return (
            replay.payload[layer_slot, :seq_len],
            a.view(seq_len, -1),
            (a if b is None else b).view(seq_len, -1),
        )

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
        """Whole-round verify attempt; ``None`` falls through to the shared flow.

        Sits before the scratch seeding because a family's kernel may seed
        (or skip) the scratch itself.

        GDN has no fused verify kernel, so the base returns None and the caller
        seeds the verify scratch and runs the shared conv update. KDA overrides
        this to prepare its convolution and gate inputs and run verification.
        Its record path reads committed state without seeding a state tape.

        Args:
            mixed_qkv: Packed ``[T, key+key+value]`` conv input.
            conv_weights: Depthwise conv filters.
            conv_comp: Conv-window component of this layer's state slab.
            conv_scratch: Per-position conv-window verify scratch.
            ssm_comp: Recurrent component of this layer's state slab.
            ssm_scratch: Per-position recurrent verify scratch.
            state_in_blocks: Per-request committed-state page ids.
            output_indices: ``[bs, T]`` verify scratch row grid.
            layer_id: Model layer whose verify payload is being processed.
            bias: Optional convolution bias before activation.
            g_raw: Precomputed raw per-channel gate (KDA), when supplied.
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
            D: Mamba2 per-head skip coefficient; None for GDN and KDA.
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
            replay_payload: ``[seq_len, width]`` ReplaySSM payload the scan
                fills with each token's k, v, a and b, or None when the split
                filled it.

        Returns:
            ``[1, seq_len, Hv, V]`` layer output.
        """
        _reject_skip_term(D)
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

        if (
            self.tree_verify is None
            and self._chain_verify_form(batch_size, draft_token_num, num_value_heads)
            is GdnChainVerifyForm.CHAIN
        ):
            # Whole chain windows from the committed pages; the conv update filled the payload.
            return gdn_chain_verify(
                query_b,
                key_b,
                value_b,
                A_log=A_log,
                a=a_b,
                dt_bias=dt_bias,
                b=b_b,
                initial_state=ssm_comp,
                initial_state_indices=state_in_blocks[:batch_size],
                scale=None,
                use_qk_l2norm=True,
            ).reshape(1, seq_len, num_value_heads, head_v_dim)

        intermediate_states = None
        output_state = None
        if self.replay_ssm:
            initial_state = ssm_comp
            initial_indices = state_in_blocks[:batch_size]
            output_state_indices = None
            if self._tree_node_states is not None:
                intermediate_states = self._tree_node_states[:batch_size]
        elif self._verify_reads_committed_recurrent_state:
            # Read the committed page in place and stage each position's state
            # in the scratch: no seed copy of the committed state.
            initial_state = ssm_comp
            initial_indices = state_in_blocks[:batch_size]
            output_state_indices = output_indices
            output_state = ssm_scratch
        else:
            initial_state = ssm_scratch
            initial_indices = self._verify_scratch_base_rows(
                batch_size, draft_token_num
            )
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
            output_state=output_state,
            intermediate_states_buffer=intermediate_states,
            tree_ancestors=self._verify_ancestors(
                batch_size, draft_token_num, num_value_heads
            ),
            replay_payload=replay_payload,
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
            D: Mamba2 per-head skip coefficient; None for GDN and KDA.
            a: GDN scalar-per-head decay input.
            b: GDN scalar-per-head beta input.
            g_raw: KDA per-channel gate, when the model precomputed it.
            f_a_out: KDA low-rank gate activation (gate GEMV source).
            f_b_weight: KDA second gate projection.
            beta_raw: KDA raw per-head beta logits.
            seq_len: Padded token extent of the batch.
            num_real_tokens: Input extent to retain: live tokens for ordinary
                metadata, packed storage capacity for capacity metadata.
            lower_bound: KDA decay clamp.
            cu_seqlens_cpu: Metadata-built host int64 copy of
                ``query_start_loc``'s contents (see
                ``MambaForwardMetadata.cu_extend_seq_lens_cpu``). The KDA
                override uses it for exact-length host planning or capacity
                admission without a per-layer synchronizing D2H. Prepared KDA
                plans are built on device; GDN plans on device and ignores it.
            state_checkpoints: The in-forward State checkpoints the scan stores
                (``MambaForwardMetadata.scan_state_checkpoints``), or ``None``;
                only a family with ``_stores_in_forward_state_checkpoints``
                takes them.

        Returns:
            ``(core_attn_out, last_recurrent_state)``.
        """
        _reject_skip_term(D)
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
            state_checkpoints=state_checkpoints,
        )
        return gdn_result.out, gdn_result.final_state

    def _prepare_prefill_scan_query_start_loc(
        self, query_start_loc: torch.Tensor
    ) -> torch.Tensor:
        """Return the device boundary tensor consumed by prefill scans."""
        return query_start_loc
