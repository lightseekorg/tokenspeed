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

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.attention.dsa import (
    dsa_decode,
    dsa_plan,
    dsa_prefill,
    select_dsa_prefill_topk_for_rows,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import NoKernelFoundError

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.configs.numerics import BITWISE_ENVELOPES
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.query_shard import QueryShardPlan
from tokenspeed.runtime.layers.attention.backends.paged.base import (
    PagedAttentionBackend,
)
from tokenspeed.runtime.layers.attention.backends.paged.mla import MLAAttnBackend
from tokenspeed.runtime.layers.attention.backends.paged.trtllm_mla import (
    TRTLLMMLABackend,
)
from tokenspeed.runtime.layers.attention.backends.support import CudaGraphSupport
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.dsa import (
    DSAConfig,
    dsa_history_gather_workspace_rows,
    index_k_row_bytes,
)
from tokenspeed.runtime.layers.attention.dcp.cache import (
    HistoryGatherPlan,
    HistoryGatherWorkspace,
    gather_history_rows,
    plan_history_gather,
)
from tokenspeed.runtime.layers.attention.dcp.comm import (
    combine_attention_partials,
    gather_query_heads,
)
from tokenspeed.runtime.layers.attention.dcp.placement import (
    CachePlacement,
    owned_history_rows,
    resolve_cache_slots,
)
from tokenspeed.runtime.layers.attention.kernel_page_sizes import (
    DSA_SPARSE_PAGE_SIZE,
)
from tokenspeed.runtime.layers.attention.kpool import KPoolRuntime
from tokenspeed.runtime.layers.attention.kv_cache.dsa import split_index_k_rows
from tokenspeed.runtime.layers.attention.page_table import (
    build_prefill_kv_workspace_slots,
)
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.utils.env import global_server_args_dict

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool


def _k_row_context_lengths(seq_lens: torch.Tensor, k: int) -> torch.Tensor:
    """``[bs]`` request lengths as the indexer's per-token rows: ``k`` query
    rows per request, each carrying its request's context length
    (``[bs * k, 1]``; the top-k applies the per-row causal bound downstream,
    see ``dsa_decode_topk``). Not a view for ``k > 1``: reshaping the
    expanded (stride-0) rows materializes a contiguous copy, so later edits
    of ``seq_lens`` do not reach it and keepers rewrite it in place
    (``_publish_k_row_indexer_rows``). Only ``k == 1`` aliases ``seq_lens``;
    callers that keep the rows call ``.contiguous()`` to cover that case."""
    return seq_lens.unsqueeze(1).expand(-1, k).reshape(-1, 1)


@dataclasses.dataclass(frozen=True)
class QueryShardHistoryGroup:
    """One request group of a sharded extend: its history and this rank's queries.

    Attributes:
        requests: The extend requests in the group (a contiguous range).
        row_base: Workspace row of the group's first history row: the
            request-major concatenation of every extend request's history
            (prefix plus chunk) numbers its rows, and ``dsa_prefill_topk``
            returns rows in that numbering.
        rows: History rows of the group.
        local_query: This rank's query rows that belong to the group, as a
            slice of the rank's shard rows.
        gather: How the group's rows split over the page owners.
    """

    requests: slice
    row_base: int
    rows: int
    local_query: slice
    gather: HistoryGatherPlan


@dataclasses.dataclass(frozen=True)
class DSAQueryShardMetadata:
    """The sharded extend arm's per-forward plan.

    Attributes:
        plan: The forward's query shard.
        groups: Request groups whose history fits the gather workspace, in
            request order; every rank runs every group's gather (a collective)
            and attends only the groups it has query rows in.
    """

    plan: QueryShardPlan
    groups: tuple[QueryShardHistoryGroup, ...]


def _make_dense_leaf(
    config: AttnConfig, spec: DSAConfig, platform, kernel_page_size: int
) -> PagedAttentionBackend:
    # The dense delegate interprets spec.backend_name itself (the MLA leaf's
    # kernel-solution map only knows its own names) — the 'dsa' name that
    # selected THIS wrapper must not leak through.
    dense_spec = dataclasses.replace(spec, backend_name=None)
    if platform.is_nvidia:
        return TRTLLMMLABackend(config, dense_spec, kernel_page_size=kernel_page_size)
    if platform.is_amd:
        return MLAAttnBackend(config, dense_spec, kernel_page_size=kernel_page_size)
    raise RuntimeError(f"DSA backend does not support platform {platform.vendor!r}.")


class DSABackend(PagedAttentionBackend):
    """DSA leaf for sparse MLA attention.

    Dense MLA metadata and dense attention calls are delegated to a platform
    leaf sharing the same kernel page size; the sparse path maps its top-k
    slots through the same page table.
    """

    default_kernel_page_size = DSA_SPARSE_PAGE_SIZE

    # DSA's sparse indexer reads this backend's chunked_prefill_metadata from
    # inside the captured prefill segment, but the prefill graph rebinds only
    # the live ForwardContext at replay — the backend metadata object stays
    # frozen at capture-time (dummy) values. Keep prefills eager.
    cuda_graph_support = CudaGraphSupport(prefill_graph=False)

    def __init__(self, config: AttnConfig, spec: DSAConfig, *, kernel_page_size: int):
        super().__init__(config, spec, kernel_page_size=kernel_page_size)
        platform = current_platform()
        self._dense_backend = _make_dense_leaf(config, spec, platform, kernel_page_size)
        self.kvp_group = tuple(config.kvp_group)
        self.kvp_rank = config.kvp_rank
        self.kvp_block_granularity: int | None = None
        self.kvp_virtual_block_count: int | None = None
        if len(self.kvp_group) > 1 and spec.index_kpool is not None:
            raise ValueError("DSA KVP does not yet support KPool selection")
        # Query context parallelism: the extend rows this rank computes are a
        # shard of the chunk and attend the gathered history of their
        # requests; the gather splits by page owner (the KVP group, or this
        # rank alone) and lands in a workspace sized for one whole history.
        self.qcp_group = tuple(config.qcp_group)
        self.qcp_rank = config.qcp_rank
        if len(self.qcp_group) > 1 and spec.index_kpool is not None:
            raise ValueError("DSA query context parallelism does not support KPool")
        self.index_head_dim = spec.index_head_dim
        # The index-K plane's format (configs/dsa.py INDEX_K_FORMATS): the
        # history gather packs rows in it and hands the indexer that form.
        self.index_k_format = spec.index_k_format
        self.query_shard_metadata: DSAQueryShardMetadata | None = None
        # Allocated by the target leaf (preallocate_history_gather_workspace),
        # shared with the draft leaf (adopt_history_gather_workspace).
        self._history_workspace: HistoryGatherWorkspace | None = None
        self.index_topk = spec.index_topk
        self.kv_lora_rank = spec.kv_lora_rank
        self.qk_nope_head_dim = spec.qk_nope_head_dim
        self.qk_rope_head_dim = spec.qk_rope_head_dim
        self.v_head_dim = spec.v_head_dim
        self.kv_cache_dim = spec.kv_cache_dim
        self.scaling = spec.scaling
        self.data_type = config.kv_cache_dtype
        self.q_data_type = config.dtype
        self.num_attention_heads = spec.num_attention_heads
        self.num_local_heads = spec.num_attention_heads // spec.attn_tp_size
        # rl-bitwise pins the sparse decode onto the batch-invariant no-split
        # leaves; without one registered, selection fails at the first decode
        # instead of silently serving an occupancy-split kernel.
        self.batch_invariant: bool = (
            global_server_args_dict["numerics"] in BITWISE_ENVELOPES
        )
        self.kernel_solution: str | None = "aok" if self.batch_invariant else None
        # --dsa-slot-order: how the cores reduce the selected slots
        # (docs/design/numerics.md, invariance.batch).
        self.slot_order: str = global_server_args_dict["dsa_slot_order"]
        self._prefill_page_table: torch.Tensor | None = None
        self.kpool_runtime = (
            KPoolRuntime(spec.index_kpool, spec.index_topk)
            if spec.index_kpool is not None
            else None
        )
        if self.kpool_runtime is not None:
            # GLM-5.3-Flash (the one KPool consumer) handles padded prefill
            # replay explicitly in its model code, so the class-level DSA
            # restriction does not apply to it.
            self.cuda_graph_support = CudaGraphSupport(prefill_graph=True)
        if len(self.qcp_group) > 1:
            self._probe_history_gather_topk_leaf(spec)

    def _probe_history_gather_topk_leaf(self, spec: DSAConfig) -> None:
        """Select, at construction, the indexer leaf the sharded extend arm
        will hand gathered index-K rows to.

        Under query context parallelism the indexer scores each request
        group's history as rows in workspace-row order of this leaf's
        ``index_k_format`` (``gather_history_index_k``, then
        ``dsa_prefill_topk(index_k_fp8=, index_k_scale=)`` or
        ``(index_k_bf16=)``), which only a leaf declaring the kernel package's
        ``index_k_workspace_rows`` feature for that format serves. Selection
        would otherwise first run in the first sharded prefill; probing it here
        turns a platform without such a leaf into a startup error. The probe
        uses the forward's selection inputs this leaf knows: the model dtype
        for the indexer query, fp32 weights, the spec's indexer geometry, this
        leaf's page size, and the envelope's batch-invariance and solution pin.
        """
        try:
            select_dsa_prefill_topk_for_rows(
                index_k_format=self.index_k_format,
                q_dtype=self.q_data_type,
                weights_dtype=torch.float32,
                index_heads=spec.index_n_heads,
                head_dim=self.index_head_dim,
                topk=self.index_topk,
                page_size=self.kernel_page_size,
                batch_invariant=self.batch_invariant,
                solution=self.kernel_solution,
            )
        except NoKernelFoundError as e:
            raise NoKernelFoundError(
                "DSA query context parallelism hands the indexer gathered "
                f"index-K rows of index_k_format={self.index_k_format!r}; no "
                "dsa_prefill_topk leaf declaring the index_k_workspace_rows "
                f"feature for that format is registered on this platform: {e}"
            ) from e

    def configure_runtime(
        self,
        *,
        block_granularity: int,
        virtual_block_count: int,
        shard_count: int,
        **kwargs,
    ) -> None:
        super().configure_runtime(**kwargs)
        if (
            shard_count != len(self.kvp_group)
            or block_granularity % self.kernel_page_size
        ):
            raise ValueError("DSA cache geometry does not match KVP topology")
        self.kvp_block_granularity = block_granularity
        self.kvp_virtual_block_count = virtual_block_count

    def preallocate_history_gather_workspace(self, max_model_len: int) -> int:
        """Allocate the gathered-history workspace of the sharded extend arm.

        Reserved from the cache budget like a verify workspace: the recipe
        plans the same bytes (``dsa_history_gather_workspace_bytes``) before
        sizing the arena. Returns the bytes allocated so the caller can check
        them against the plan.
        """
        rows = dsa_history_gather_workspace_rows(
            max_model_len, page_size=self.kernel_page_size
        )
        self._history_workspace = HistoryGatherWorkspace(
            rows=rows,
            kv=torch.empty(
                (rows, self.kv_cache_dim), dtype=self.data_type, device=self.device
            ),
            # Index-K rows packed in the plane's format (FP8 bytes then fp32
            # scales, or bf16 keys), one gather per group.
            index_k=torch.empty(
                (rows, index_k_row_bytes(self.index_head_dim, self.index_k_format)),
                dtype=torch.uint8,
                device=self.device,
            ),
            index_k_format=self.index_k_format,
        )
        return self._history_workspace.nbytes

    def history_gather_workspace(self) -> HistoryGatherWorkspace | None:
        return self._history_workspace

    def adopt_history_gather_workspace(self, workspace: HistoryGatherWorkspace) -> None:
        """Gather into another leaf's workspace (the draft into the target's).

        The two leaves must agree on the row geometry the gathers write:
        latent width and dtype, the index-K format and its packed row bytes,
        and whole kernel pages of this leaf's page size.
        """
        row_bytes = index_k_row_bytes(self.index_head_dim, self.index_k_format)
        if (
            workspace.kv.shape[1] != self.kv_cache_dim
            or workspace.kv.dtype != self.data_type
            or workspace.index_k_format != self.index_k_format
            or workspace.index_k.shape[1] != row_bytes
            or workspace.rows % self.kernel_page_size
        ):
            raise ValueError(
                "history gather workspace geometry mismatch: "
                f"kv {tuple(workspace.kv.shape)} {workspace.kv.dtype}, index_k "
                f"{tuple(workspace.index_k.shape)} {workspace.index_k_format}, "
                f"rows {workspace.rows}; this leaf gathers [{self.kv_cache_dim}] "
                f"{self.data_type} latent, {row_bytes}-byte {self.index_k_format} "
                f"index-K rows in pages of {self.kernel_page_size}"
            )
        self._history_workspace = workspace

    def cache_placement(self, layer) -> CachePlacement | None:
        if len(self.kvp_group) == 1:
            return None
        if self.kvp_block_granularity is None or self.kvp_virtual_block_count is None:
            raise RuntimeError("DSA KVP cache geometry is not configured")
        return CachePlacement(
            self.kvp_block_granularity,
            self.kvp_virtual_block_count,
            self.kvp_group,
            self.kvp_rank,
        )

    def set_request_slots(self, req_pool_indices: torch.Tensor) -> None:
        # KPool's tail state is indexed by request-pool slot, and its
        # per-forward plan must not outlive the metadata build that
        # produced it: the router publishes the slots after every build,
        # which is exactly the reset point.
        if self.kpool_runtime is not None:
            self.kpool_runtime.reset_forward(req_pool_indices)

    def require_kpool_runtime(self) -> KPoolRuntime:
        """Return the configured KPool runtime for sparse pooled indexing."""
        if self.kpool_runtime is None:
            raise RuntimeError("DSA backend was created without KPool configuration")
        return self.kpool_runtime

    def kpool_prefill_page_table(self, num_requests: int) -> torch.Tensor:
        """The kernel-page history rows KPool prefill maps its top-k through."""
        table = self._prefill_page_table
        if table is None and self.chunked_prefill_metadata is not None:
            table = self.chunked_prefill_metadata.page_table
        if table is None:
            raise RuntimeError("DSA KPool prefill requires a full-history page table")
        if num_requests < 0 or table.shape[0] < num_requests:
            raise RuntimeError(
                "DSA KPool prefill page-table row mismatch: "
                f"table={table.shape[0]}, requests={num_requests}"
            )
        return table[:num_requests]

    def kpool_decode_page_table(
        self, row_start: int, num_requests: int
    ) -> torch.Tensor:
        """The kernel-page history rows KPool decode maps its top-k through."""
        metadata = self.forward_decode_metadata
        table = None if metadata is None else metadata.page_table
        row_end = row_start + num_requests
        if (
            table is None
            or row_start < 0
            or num_requests < 0
            or table.shape[0] < row_end
        ):
            rows = None if table is None else table.shape[0]
            raise RuntimeError(
                "DSA KPool decode page-table row mismatch: "
                f"table={rows}, rows=[{row_start}, {row_end})"
            )
        return table[row_start:row_end]

    # ------------------------------------------------------------------
    # Delegation surface
    # ------------------------------------------------------------------

    @property
    def forward_decode_metadata(self):
        return self._dense_backend.forward_decode_metadata

    @property
    def forward_prefill_metadata(self):
        return self._dense_backend.forward_prefill_metadata

    @property
    def chunked_prefill_metadata(self):
        return self._dense_backend.chunked_prefill_metadata

    @property
    def max_num_pages(self) -> int:
        # The dense leaf pads its table width to the fused-kernel block
        # constraint; the router sizes this leaf's tables the same way.
        return self._dense_backend.max_num_pages

    @max_num_pages.setter
    def max_num_pages(self, value: int) -> None:
        del value  # derived from the dense leaf

    @property
    def decode_seq_lens_buffer(self) -> torch.Tensor:
        return self._dense_backend.decode_seq_lens_buffer

    def child_backends(self):
        return (self._dense_backend,)

    def slot_state_rows(self, slot: int) -> list[torch.Tensor]:
        # KPool keeps each request's incomplete index pool in a request-local
        # tail outside the index-K pages. The tail workspace is one arena
        # allocation shared by the target and draft views, so the target
        # backend images it once and a draft backend lists nothing.
        if self.kpool_runtime is None or self.is_draft:
            return []
        if self.cache_pool is None:
            raise RuntimeError("DSA KPool slot state needs a bound cache pool")
        return self.cache_pool.kpool_tail_slot_rows(slot)

    def _publish_cache_pool(self, cache_pool: CachePool) -> None:
        super()._publish_cache_pool(cache_pool)
        self._prefill_page_table = None
        self.query_shard_metadata = None
        self.kvp_block_granularity = None
        self.kvp_virtual_block_count = None
        if self.kpool_runtime is not None:
            self.kpool_runtime.reset_forward(None)

    def register_step_counter(self, step_counter):
        self.step_counter = step_counter
        self._dense_backend.step_counter = step_counter

    def override_num_extends(self, num_extends: int):
        return self._dense_backend.override_num_extends(num_extends)

    def init_cuda_graph_state(self, max_bs: int) -> None:
        self._dense_backend.init_cuda_graph_state(max_bs)

    # Capture is inherited: the leaf default routes through this wrapper's
    # refresh, whose lazy arm fills the dense leaf's per-bs cached metadata
    # fields _dsa_seq_lens_2d / _dsa_plan once per bs.

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def refresh_decode_metadata(
        self,
        bs: int,
        actual_bs: int,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        *,
        num_extends: int = 0,
        for_graph_replay: bool = False,
    ) -> None:
        self._dense_backend.refresh_decode_metadata(
            bs,
            actual_bs,
            seq_lens,
            page_table,
            num_extends=num_extends,
            for_graph_replay=for_graph_replay,
        )
        metadata = self.forward_decode_metadata
        if metadata._dsa_seq_lens_2d is None:
            # First refresh at a lazily-built bs (no capture ran): allocate the
            # per-token rows once on the dense leaf's per-bs view; later
            # refreshes (and the drafter's re-anchor) rewrite them in place.
            metadata._dsa_seq_lens_2d = _k_row_context_lengths(
                seq_lens[:bs], self.spec_num_tokens
            ).contiguous()
            metadata._dsa_plan = dsa_plan(
                seq_lens_2d=metadata._dsa_seq_lens_2d,
                page_size=self.kernel_page_size,
            )
            return
        self._publish_k_row_indexer_rows(metadata, seq_lens, bs)

    def _publish_k_row_indexer_rows(
        self, metadata, seq_lens: torch.Tensor, bs: int
    ) -> None:
        """Rewrite the per-token indexer rows to ``seq_lens[:bs]`` (``k`` rows
        per request) and refresh their plan, both in place: fixed shapes and
        storage, so a captured graph replays the edit."""
        k = self.spec_num_tokens
        rows = metadata._dsa_seq_lens_2d
        if rows.shape[0] != bs * k:
            raise RuntimeError(
                "DSA draft per-token rows do not match the decode batch: "
                f"rows={rows.shape[0]}, requests={bs}, width={k}"
            )
        rows.copy_(_k_row_context_lengths(seq_lens[:bs], k))
        dsa_plan(
            seq_lens_2d=rows,
            page_size=self.kernel_page_size,
            out=metadata._dsa_plan,
        )

    def advance_draft_forward_metadata(self, seq_lens: torch.Tensor) -> None:
        """Eagle chain step: one query row per request, so the plan is
        rebuilt from the ``[bs, 1]`` request lengths (the per-token
        ``_dsa_seq_lens_2d`` is left as the round's refresh published it)."""
        metadata = self.forward_decode_metadata
        if metadata is None or metadata.seq_lens_k is None:
            raise RuntimeError("DSA draft decode metadata was not initialized")
        metadata.seq_lens_k.copy_(seq_lens[: metadata.seq_lens_k.numel()])

        dsa_plan(
            seq_lens_2d=metadata.seq_lens_k.unsqueeze(1),
            page_size=self.kernel_page_size,
            out=metadata._dsa_plan,
        )

    def update_draft_forward_metadata(self, frontier: torch.Tensor) -> None:
        """Multi-depth MTP re-anchor: every depth re-runs ``spec_num_tokens``
        query rows per request ending at ``frontier``, the same k-row shape
        the round's :meth:`refresh_decode_metadata` published. The k-row
        top-k reads one context length per query row (``_dsa_seq_lens_2d``,
        ``[bs * k, 1]``) and its plan, so both are rewritten in place to
        the frontier; the kernel derives row ``j``'s causal bound as
        ``frontier - (k - 1) + j``."""
        metadata = self.forward_decode_metadata
        if metadata is None or metadata.seq_lens_k is None:
            raise RuntimeError("DSA draft decode metadata was not initialized")
        if metadata._dsa_seq_lens_2d is None:
            raise RuntimeError(
                "DSA draft per-token rows were not published: the round's "
                "refresh_decode_metadata must run before the re-anchor"
            )
        bs = metadata.seq_lens_k.numel()
        metadata.seq_lens_k.copy_(frontier[:bs])
        self._publish_k_row_indexer_rows(metadata, frontier, bs)

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        forward_mode: ForwardMode,
        *,
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_with_prefix: bool,
        query_shard: QueryShardPlan | None,
        page_table_cpu: torch.Tensor | None,
        **kwargs,
    ):
        if not (forward_mode.is_extend_or_mixed() or forward_mode.is_idle()):
            raise RuntimeError(
                "DSA decode metadata goes through refresh_decode_metadata; "
                f"init_forward_metadata only serves extend/mixed ({forward_mode})"
            )
        # The dense delegate describes the whole extend span (its page table
        # and lengths serve the indexer's workspace slots); the shard is this
        # leaf's: it attends local rows against gathered history below.
        self._dense_backend.init_forward_metadata(
            bs,
            num_extends,
            seq_lens,
            page_table,
            forward_mode,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            extend_prefix_lens=extend_prefix_lens,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_with_prefix=extend_with_prefix,
            query_shard=None,
            page_table_cpu=None,
            **kwargs,
        )
        self.query_shard_metadata = None
        if query_shard is not None and query_shard.size > 1:
            if forward_mode.is_mixed():
                raise RuntimeError(
                    "DSA query context parallelism serves pure extend forwards; a "
                    "MIXED round carries replicated decode rows"
                )
            if page_table_cpu is None:
                raise RuntimeError(
                    "DSA query context parallelism needs the host page table to "
                    "split the history gather by page owner"
                )
            self.query_shard_metadata = self._plan_query_shard(
                query_shard,
                num_extends=num_extends,
                page_table=page_table,
                page_table_cpu=page_table_cpu,
                seq_lens=seq_lens,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
                extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            )
        # Target mixed batches carry decode rows needing the per-token plan.
        # A draft's plan is rebuilt by the wrapper's refresh_decode_metadata
        # after this init (the unified draft contract).
        if forward_mode.is_mixed() and not self.is_draft:
            metadata = self.forward_decode_metadata
            # Per-token context lengths: the paged-MQA-logits kernel only supports
            # next_n == 1, so each verify token is its own row (bs * spec_num_tokens
            # rows).
            metadata._dsa_seq_lens_2d = _k_row_context_lengths(
                seq_lens, self.spec_num_tokens
            ).contiguous()
            if num_extends < bs:
                # Decode rows only: skip the extend requests' per-token block.
                seq_lens_2d = metadata._dsa_seq_lens_2d[
                    num_extends * self.spec_num_tokens :
                ]
            else:
                # The dsa_plan is unused, alias to full-batch seq_lens_2d to
                # generate dsa_plan as a placeholder
                seq_lens_2d = metadata._dsa_seq_lens_2d
            metadata._dsa_plan = dsa_plan(
                seq_lens_2d=seq_lens_2d, page_size=self.kernel_page_size
            )

        self._prefill_page_table = None
        if num_extends > 0 and forward_mode.is_extend_or_mixed():
            cmeta = self._dense_backend.chunked_prefill_metadata
            if cmeta is not None:
                # The sparse indexer's top-k maps through the same kernel page
                # table the extend rows read (DSA's sparse page size equals
                # the leaf's kernel page size by construction).
                self._prefill_page_table = page_table[:num_extends]
                cmeta.page_table = self._prefill_page_table

    # ------------------------------------------------------------------
    # Query context parallelism: the gathered-history extend arm
    # ------------------------------------------------------------------

    def _plan_query_shard(
        self,
        plan: QueryShardPlan,
        *,
        num_extends: int,
        page_table: torch.Tensor,
        page_table_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
    ) -> DSAQueryShardMetadata:
        """Group the extend requests by history and plan each group's gather.

        Host arithmetic over the pinned length mirrors and the host page table
        (owned-row counts per rank); the device work is one slot build per
        group. Greedy grouping by summed history length against the gather
        workspace, a request never splits.
        """
        if self._history_workspace is None:
            raise RuntimeError(
                "DSA query context parallelism needs its history gather workspace; "
                "preallocate_history_gather_workspace (or the draft's adoption of "
                "the target's) did not run"
            )
        extend_lens = [int(x) for x in extend_seq_lens_cpu[:num_extends].tolist()]
        prefix_lens = [int(x) for x in extend_prefix_lens_cpu[:num_extends].tolist()]
        if sum(extend_lens) != plan.total_rows:
            raise RuntimeError(
                f"query shard plans {plan.total_rows} rows but the extend requests "
                f"carry {sum(extend_lens)}"
            )
        history_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        placement = self.cache_placement(None)
        owned = owned_history_rows(
            page_table_cpu[:num_extends],
            torch.tensor(history_lens, dtype=torch.int64),
            page_size=self.kernel_page_size,
            placement=placement,
        )
        cap = self._history_workspace.rows
        groups: list[QueryShardHistoryGroup] = []
        query_start = 0
        row_base = 0
        first = 0
        while first < num_extends:
            if history_lens[first] > cap:
                raise RuntimeError(
                    f"request history of {history_lens[first]} rows exceeds the "
                    f"{cap}-row query-context-parallel gather workspace"
                )
            last = first
            rows = history_lens[first]
            while last + 1 < num_extends and rows + history_lens[last + 1] <= cap:
                last += 1
                rows += history_lens[last]
            requests = slice(first, last + 1)
            query_rows = sum(extend_lens[requests])
            local_lo = min(max(query_start, plan.local_start), plan.local_end)
            local_hi = min(
                max(query_start + query_rows, plan.local_start), plan.local_end
            )
            virtual_slots = build_prefill_kv_workspace_slots(
                page_table=page_table[requests],
                seq_lens=seq_lens[requests],
                max_seq_len=max(history_lens[requests]),
                page_size=self.kernel_page_size,
                device=page_table.device,
                num_tokens=rows,
            )
            gather = plan_history_gather(
                virtual_slots,
                placement=placement,
                owned_rows_per_rank=owned[:, requests].sum(dim=1).tolist(),
            )
            groups.append(
                QueryShardHistoryGroup(
                    requests=requests,
                    row_base=row_base,
                    rows=rows,
                    local_query=slice(
                        local_lo - plan.local_start, local_hi - plan.local_start
                    ),
                    gather=gather,
                )
            )
            query_start += query_rows
            row_base += rows
            first = last + 1
        return DSAQueryShardMetadata(plan=plan, groups=tuple(groups))

    def require_query_shard_metadata(self) -> DSAQueryShardMetadata:
        """The sharded extend arm's plan for the current forward."""
        if self.query_shard_metadata is None:
            raise RuntimeError("this forward is not a sharded DSA extend")
        return self.query_shard_metadata

    def gather_history_kv(
        self, layer, token_to_kv_pool, group: QueryShardHistoryGroup
    ) -> torch.Tensor:
        """Assemble one group's latent rows in position order.

        Returns a ``[pages * kernel_page_size, kv_cache_dim]`` view of the
        gather workspace (valid until the next group's gather) whose leading
        ``group.rows`` rows are the history: a whole number of kernel pages, so
        every ``dsa_prefill`` solution's view of a flat ``[slots, dim]`` cache
        holds, paged or not; the padding rows are never selected. A collective
        over the page owners: every rank calls it for every group.
        """
        if self._history_workspace is None:
            raise RuntimeError("DSA history gather workspace is not allocated")
        kv_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_flat = kv_cache.reshape(-1, kv_cache.shape[-1])
        local = kv_flat.index_select(0, group.gather.local_fetch_slots)
        gather_history_rows(group.gather, local, out=self._history_workspace.kv)
        page = self.kernel_page_size
        return self._history_workspace.kv[: -(-group.rows // page) * page]

    def gather_history_index_k(
        self, layer_id: int, token_to_kv_pool, group: QueryShardHistoryGroup
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Assemble one group's index-K rows in position order, in this leaf's
        ``index_k_format`` (the pool read, the workspace rows and the split
        all follow it), as the rows ``dsa_prefill_topk`` takes in
        workspace-row order: ``fp8_scaled`` gives ``[rows, index_head_dim]``
        FP8 bytes and ``[rows, groups]`` fp32 scales (``index_k_fp8`` /
        ``index_k_scale``); ``bf16`` gives ``[rows, index_head_dim]`` bf16 keys
        and ``None`` (``index_k_bf16``). Views of the gather workspace, valid
        until the next group's gather. One collective over the page owners
        moves the rows packed as bytes."""
        if self._history_workspace is None:
            raise RuntimeError("DSA history gather workspace is not allocated")
        packed = token_to_kv_pool.gather_index_k_rows(
            layer_id,
            group.gather.local_fetch_slots,
            index_k_format=self.index_k_format,
        )
        gathered = gather_history_rows(
            group.gather, packed, out=self._history_workspace.index_k
        )
        return split_index_k_rows(
            gathered,
            index_head_dim=self.index_head_dim,
            index_k_format=self.index_k_format,
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _query_heads(self, q: torch.Tensor, layer) -> int:
        """The heads ``q`` carries: ``[rows, heads, head_dim]``, or
        ``[rows, heads * head_dim]``.

        The sparse cores take their head count from the query, not from
        ``layer.tp_q_head_num``: one model layer serves a forward whose rows
        carry every head (head-replicated weights; head TP, whose exchange
        delivered every head of this rank's own rows) and one whose rows
        carry the attention-TP slice (plain attention TP; the replicated-row
        forwards of a head-TP engine, which exchange nothing). Any other
        count is a layout bug, refused here.
        """
        if q.dim() == 3:
            heads, head_dim = q.shape[1], q.shape[2]
        elif q.dim() == 2:
            heads, head_dim = divmod(q.shape[1], layer.head_dim)
            head_dim = layer.head_dim if head_dim == 0 else -1
        else:
            raise ValueError(f"DSA query must be 2-D or 3-D, got {tuple(q.shape)}")
        if head_dim != layer.head_dim:
            raise ValueError(
                f"DSA query {tuple(q.shape)} does not split into heads of "
                f"{layer.head_dim}"
            )
        if heads not in (self.num_local_heads, self.num_attention_heads):
            raise ValueError(
                f"DSA query carries {heads} heads, neither the attention-TP slice "
                f"({self.num_local_heads}) nor every head ({self.num_attention_heads})"
            )
        return heads

    def _query_holds_every_head(self, heads: int) -> bool:
        """Whether a query of ``heads`` heads carries the model's heads rather
        than the attention-TP slice (``_query_heads``). The DCP combine's form
        follows it, never a mapping assumption: every head attends the owned
        pages and all-reduces the weighted partials; the slice gathers the
        group's query heads in and reduce-scatters its own back out."""
        return heads != self.num_local_heads

    def _validate_logit_cap(self, logits_soft_cap: float) -> None:
        if logits_soft_cap and logits_soft_cap > 0:
            raise NotImplementedError(
                "TokenSpeed DSA fused dense attention does not support "
                f"logits_soft_cap={logits_soft_cap}. Sparse DSA kernels must "
                "preserve the capped-score semantics before enabling this model."
            )

    def _validate_dense_context(self, seq_lens: torch.Tensor, bs: int) -> None:
        if seq_lens is None or bs <= 0:
            return
        active_seq_lens = seq_lens[:bs]
        if active_seq_lens.numel() == 0:
            return
        max_seq_len = int(active_seq_lens.max().item())
        if max_seq_len > self.index_topk:
            raise NotImplementedError(
                "TokenSpeed DSA dense attention is exact only when every "
                f"request has seq_len <= index_topk ({self.index_topk}); got "
                f"max seq_len {max_seq_len}. Sparse DSA top-k indices are "
                "required for longer contexts."
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        **kwargs,
    ) -> torch.Tensor:
        # The model drives DSA prefill through forward_extend_chunked /
        # forward_sparse_prefill directly.
        raise NotImplementedError(
            "DSA prefill runs through forward_extend_chunked / forward_sparse_prefill"
        )

    def forward_extend_chunked(
        self,
        q,
        k,
        v,
        scaling,
        logits_soft_cap,
        *,
        cum_seq_lens_q,
        cum_seq_lens_kv,
        max_q_len,
        max_kv_len,
        seq_lens,
        batch_size,
        causal,
        out: torch.Tensor | None = None,
    ):
        self._validate_logit_cap(logits_soft_cap)
        if self.query_shard_metadata is not None:
            raise RuntimeError(
                "a sharded DSA extend attends through forward_sparse_prefill; the "
                "dense delegate sees whole-span metadata for shard rows"
            )
        self._validate_dense_context(seq_lens, batch_size)
        return self._dense_backend.forward_extend_chunked(
            q,
            k,
            v,
            scaling,
            logits_soft_cap,
            cum_seq_lens_q=cum_seq_lens_q,
            cum_seq_lens_kv=cum_seq_lens_kv,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            seq_lens=seq_lens,
            batch_size=batch_size,
            causal=causal,
            out=out,
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        topk_indices: torch.Tensor | None = None,
        topk_lens: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        self._validate_logit_cap(layer.logit_cap)
        if topk_indices is not None:
            return self.forward_sparse_decode(
                q=q,
                layer=layer,
                token_to_kv_pool=token_to_kv_pool,
                bs=bs,
                topk_indices=topk_indices,
                topk_lens=topk_lens,
            )
        if len(self.kvp_group) > 1:
            raise ValueError("Sharded DSA decode requires global top-k selection")
        metadata = self.forward_decode_metadata
        if metadata is not None and metadata.seq_lens_k is not None:
            num_extends = int(metadata.num_extends or 0)
            self._validate_dense_context(metadata.seq_lens_k[num_extends:], bs)
        return self._dense_backend.forward_decode(
            q=q,
            k=k,
            v=v,
            layer=layer,
            out_cache_loc=out_cache_loc,
            token_to_kv_pool=token_to_kv_pool,
            bs=bs,
            **kwargs,
        )

    def forward_sparse_prefill(
        self,
        *,
        q: torch.Tensor,
        layer,
        token_to_kv_pool,
        kv_seq_lens: torch.Tensor | None,
        topk_slots: torch.Tensor,
        topk_lens: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        """Attend to preselected KV slots and merge DCP partials.

        topk_slots contains one candidate row per query, with -1 for invalid
        entries; topk_lens gives valid counts. kv_seq_lens optionally supplies
        per-query causal lengths, bounded by max_seq_len. KV is already written.
        Returns token-major attention output, flattened over heads and features.

        Under a query shard ``q`` holds this rank's rows and ``topk_slots``
        are history workspace rows (the request-major rows of every extend
        request's prefix and chunk, as ``dsa_prefill_topk`` numbers them, the
        rows ``gather_history_index_k`` scored): per request group the KV
        history is gathered from its page owners into one buffer and the
        local rows attend it with every head -- pure data movement, so a
        row's bytes are those of an unsharded forward, and no LSE merge. The
        query must carry every head (head-replicated weights, or head TP over
        the shard group, whose exchange delivers every head of the local
        rows before the core); the attention-TP slice is refused.
        """
        heads = self._query_heads(q, layer)
        if layer.logit_cap and layer.logit_cap > 0:
            self._validate_logit_cap(layer.logit_cap)
        if getattr(token_to_kv_pool, "quant_method", None) == "per_token_head":
            raise RuntimeError(
                "DSA sparse prefill does not support "
                "kv_cache_quant_method='per_token_head' yet."
            )
        if topk_slots.shape[0] != q.shape[0]:
            raise RuntimeError(
                "DSA sparse prefill metadata token mismatch: "
                f"indices={topk_slots.shape[0]}, q_tokens={q.shape[0]}"
            )
        if topk_lens.shape[0] != q.shape[0]:
            raise RuntimeError(
                "DSA sparse prefill top-k length mismatch: "
                f"lens={topk_lens.shape[0]}, q_tokens={q.shape[0]}"
            )
        if kv_seq_lens is not None and (
            kv_seq_lens.dim() != 1 or kv_seq_lens.numel() != q.shape[0]
        ):
            raise RuntimeError(
                "DSA sparse prefill physical length mismatch: "
                f"lens={tuple(kv_seq_lens.shape)}, q_tokens={q.shape[0]}"
            )
        if self.query_shard_metadata is not None:
            # A rank without rows still joins every group's gather, so the
            # sharded arm runs before the empty-query return.
            if topk_slots.dim() != 2 or topk_slots.shape[1] <= 0:
                raise RuntimeError(
                    "DSA sparse prefill top-k shape mismatch: "
                    f"indices={tuple(topk_slots.shape)}"
                )
            if not self._query_holds_every_head(heads):
                raise RuntimeError(
                    "the sharded DSA extend attends the gathered history with every "
                    f"head; the query carries the attention-TP slice of {heads} "
                    "heads (under head TP the exchange delivers every head of the "
                    "local rows before the core)"
                )
            q_view = q.view(q.shape[0], heads, layer.head_dim)
            if self.data_type == torch.float8_e4m3fn and q_view.dtype != self.data_type:
                q_view = q_view.to(self.data_type)
            out = self._forward_sharded_sparse_prefill(
                q_view=q_view,
                layer=layer,
                token_to_kv_pool=token_to_kv_pool,
                kv_seq_lens=kv_seq_lens,
                topk_slots=topk_slots,
                topk_lens=topk_lens,
                max_seq_len=max_seq_len,
            )
            if self.step_counter is not None:
                self.step_counter.record_cache()
            return out.reshape(-1, heads * layer.v_head_dim)
        if q.shape[0] == 0:
            return q.new_empty((0, heads * layer.v_head_dim))
        # KPool selection can append up to pool_size - 1 visible tail tokens,
        # so its workspace may be wider than the configured pooled top-k.
        if topk_slots.dim() != 2 or topk_slots.shape[1] <= 0:
            raise RuntimeError(
                "DSA sparse prefill top-k shape mismatch: "
                f"indices={tuple(topk_slots.shape)}"
            )
        q_view = q.view(q.shape[0], heads, layer.head_dim)
        if self.data_type == torch.float8_e4m3fn and q_view.dtype != self.data_type:
            q_view = q_view.to(self.data_type)
        kv_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)

        use_dcp = len(self.kvp_group) > 1
        keep_all_heads = use_dcp and self._query_holds_every_head(heads)
        if use_dcp:
            slots, owned = resolve_cache_slots(topk_slots, self.cache_placement(layer))
            topk_slots = torch.where(owned, slots, -1)
            if not keep_all_heads:
                q_view = gather_query_heads(q_view, self.kvp_group)
        out = dsa_prefill(
            q=q_view,
            kv_cache=kv_cache,
            sparse_kv_cache=None,
            topk_slots=topk_slots,
            topk_lens=topk_lens.to(device=q.device, dtype=torch.int32).contiguous(),
            kv_seq_lens=(
                kv_seq_lens.to(device=q.device, dtype=torch.int32).contiguous()
                if kv_seq_lens is not None
                else None
            ),
            max_seqlen_k=max_seq_len,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            softmax_scale=layer.scaling,
            page_size=self.kernel_page_size,
            logit_cap=layer.logit_cap,
            k_scale=1.0,
            return_lse=use_dcp,
            solution=self.kernel_solution,
            slot_order=self.slot_order,
        )
        if use_dcp:
            local_output, local_lse = out
            out = combine_attention_partials(
                local_output,
                local_lse,
                group=self.kvp_group,
                rank=self.kvp_rank,
                sink=None,
                keep_all_heads=keep_all_heads,
            )
        # GLM's sparse-prefill path writes both the latent KV and index_k before
        # entering this method, but bypasses the backend's forward and its
        # normal PD readiness hook. Publish the layer only after the dependent
        # sparse-attention launch has been enqueued, so layerwise transfer cannot
        # observe either cache field before it is ready.
        if self.step_counter is not None:
            self.step_counter.record_cache()
        return out.reshape(-1, heads * layer.v_head_dim)

    def _forward_sharded_sparse_prefill(
        self,
        *,
        q_view: torch.Tensor,
        layer,
        token_to_kv_pool,
        kv_seq_lens: torch.Tensor | None,
        topk_slots: torch.Tensor,
        topk_lens: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        """The gathered-buffer arm: per request group, gather the history and
        attend this rank's rows of the group against it."""
        meta = self.require_query_shard_metadata()
        if q_view.shape[0] != meta.plan.local_rows:
            raise RuntimeError(
                f"query shard rank {meta.plan.rank} attends {meta.plan.local_rows} "
                f"rows, got {q_view.shape[0]}"
            )
        out = q_view.new_empty(
            (q_view.shape[0], q_view.shape[1], layer.v_head_dim),
            dtype=(
                torch.bfloat16 if q_view.dtype == torch.float8_e4m3fn else q_view.dtype
            ),
        )
        topk_lens = topk_lens.to(device=q_view.device, dtype=torch.int32).contiguous()
        if kv_seq_lens is not None:
            kv_seq_lens = kv_seq_lens.to(
                device=q_view.device, dtype=torch.int32
            ).contiguous()
        for group in meta.groups:
            # Every rank joins every group's gather; the collective does not
            # know which ranks have query rows in the group.
            kv_group = self.gather_history_kv(layer, token_to_kv_pool, group)
            rows = group.local_query
            if rows.stop <= rows.start:
                continue
            slots = topk_slots[rows]
            slots = torch.where(slots >= 0, slots - group.row_base, -1)
            # This call mirrors the unsharded arm's: every kernel facade
            # option the unsharded call passes (the slot-order selection
            # included) must be passed here too.
            out[rows] = dsa_prefill(
                q=q_view[rows],
                kv_cache=kv_group,
                sparse_kv_cache=None,
                topk_slots=slots,
                topk_lens=topk_lens[rows],
                kv_seq_lens=None if kv_seq_lens is None else kv_seq_lens[rows],
                max_seqlen_k=max_seq_len,
                qk_nope_head_dim=self.qk_nope_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                softmax_scale=layer.scaling,
                page_size=self.kernel_page_size,
                logit_cap=layer.logit_cap,
                k_scale=1.0,
                return_lse=False,
                solution=self.kernel_solution,
                slot_order=self.slot_order,
            )
        return out

    def forward_sparse_decode(
        self,
        *,
        q: torch.Tensor,
        layer,
        token_to_kv_pool,
        bs: int,
        topk_indices: torch.Tensor,
        topk_lens: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.kernel_page_size != DSA_SPARSE_PAGE_SIZE:
            raise RuntimeError(
                f"DSA sparse decode requires kernel_page_size="
                f"{DSA_SPARSE_PAGE_SIZE} for "
                f"sparse KV layout, got {self.kernel_page_size}."
            )
        if getattr(token_to_kv_pool, "quant_method", None) == "per_token_head":
            raise RuntimeError(
                "DSA sparse decode does not support "
                "kv_cache_quant_method='per_token_head' yet."
            )
        allow_fp8_query = (
            self.data_type == torch.float8_e4m3fn and q.dtype == torch.float8_e4m3fn
        )
        if q.dtype != torch.bfloat16 and not allow_fp8_query:
            raise RuntimeError(
                "DSA sparse decode requires BF16 query tensors, or FP8 query "
                f"tensors on FP8 KV sparse paths, got {q.dtype}."
            )
        if topk_indices.dtype != torch.int32:
            topk_indices = topk_indices.to(torch.int32)
        if topk_indices.shape[-1] != self.index_topk and topk_lens is None:
            raise RuntimeError(
                "DSA sparse decode top-k width mismatch: "
                f"indices={topk_indices.shape[-1]}, expected={self.index_topk}"
            )
        num_tokens = q.shape[0]
        # Spec-verify feeds q_len_per_req query rows per request while plain
        # decode and the draft model's own decode steps feed one; derive the
        # width from the actual batch shape (bs is the decode request count)
        # rather than spec_num_tokens, which the draft backend inherits from the
        # shared config.
        if bs > 0 and num_tokens % bs == 0:
            q_len_per_req = num_tokens // bs
        else:
            q_len_per_req = 1
        num_reqs = num_tokens // q_len_per_req
        metadata = self.forward_decode_metadata
        if metadata is None or metadata.seq_lens_k is None:
            raise RuntimeError("DSA sparse decode requires decode metadata.")
        num_extends = int(metadata.num_extends or 0)
        available_reqs = max(0, int(metadata.seq_lens_k.shape[0]) - num_extends)
        if available_reqs < num_reqs:
            if available_reqs <= 0 or q.shape[0] % available_reqs != 0:
                raise RuntimeError(
                    "DSA sparse decode metadata batch mismatch: "
                    f"seq_lens={available_reqs}, requests={num_reqs}, "
                    f"q_tokens={q.shape[0]}."
                )
            num_reqs = available_reqs
            q_len_per_req = q.shape[0] // available_reqs
        seq_lens = metadata.seq_lens_k[num_extends : num_extends + num_reqs]
        if seq_lens.numel() != num_reqs:
            raise RuntimeError(
                "DSA sparse decode metadata batch mismatch: "
                f"seq_lens={seq_lens.numel()}, requests={num_reqs}."
            )
        num_tokens = q.shape[0]
        expected_tokens = num_reqs * int(q_len_per_req)
        if num_tokens != expected_tokens:
            raise RuntimeError(
                "DSA sparse decode token shape mismatch: "
                f"q_tokens={num_tokens}, requests={num_reqs}, "
                f"q_len_per_req={q_len_per_req}."
            )
        if topk_lens is not None:
            if topk_lens.dim() != 1 or topk_lens.numel() != num_tokens:
                raise RuntimeError(
                    "DSA sparse decode top-k length mismatch: "
                    f"lens={tuple(topk_lens.shape)}, q_tokens={num_tokens}."
                )
            topk_lens = topk_lens.to(device=q.device, dtype=torch.int32).contiguous()

        # Physical KV length per query row: verify row t of a request sees
        # seq_len - (width - 1 - t) tokens (the block's own future is masked
        # by the kernel's per-row length, not by top-k selection).
        seq_lens = seq_lens.to(device=q.device, dtype=torch.int32).contiguous()
        if q_len_per_req == 1:
            kv_seq_lens = seq_lens
        else:
            offsets = torch.arange(
                q_len_per_req, device=q.device, dtype=torch.int32
            ) - (q_len_per_req - 1)
            kv_seq_lens = (
                seq_lens.unsqueeze(1).add(offsets).clamp_min(0).reshape(-1).contiguous()
            )

        heads = self._query_heads(q, layer)
        q_view = q.view(num_tokens, heads, layer.head_dim)
        if self.data_type == torch.float8_e4m3fn:
            q_view = q_view.to(self.data_type)
        kv_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)

        max_seqlen_k = int(
            getattr(metadata, "max_seq_len_k", 0) or self.max_context_len
        )
        use_dcp = len(self.kvp_group) > 1
        # The combine's form follows the query's heads: every head
        # (head-replicated attention weights, as the drafter's steps on a
        # query-sharding engine without head TP carry) keeps all heads -- no
        # query-head gather in, an all-reduce of the weighted partials out;
        # the attention-TP slice (plain attention TP, or the drafter's steps
        # under head TP over the query shards, which exchange nothing)
        # gathers heads in and reduce-scatters them back.
        keep_all_heads = use_dcp and self._query_holds_every_head(heads)
        topk_slots = topk_indices.view(num_tokens, -1)
        if use_dcp:
            slots, owned = resolve_cache_slots(topk_slots, self.cache_placement(layer))
            topk_slots = torch.where(owned, slots, -1)
            if not keep_all_heads:
                q_view = gather_query_heads(q_view, self.kvp_group)
        out = dsa_decode(
            q=q_view,
            kv_cache=kv_cache,
            sparse_kv_cache=None,
            topk_slots=topk_slots,
            topk_lens=topk_lens,
            max_seqlen_k=max_seqlen_k,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            softmax_scale=layer.scaling,
            page_size=self.kernel_page_size,
            q_len_per_req=q_len_per_req,
            kv_seq_lens=kv_seq_lens,
            logit_cap=layer.logit_cap,
            k_scale=1.0,
            return_lse=use_dcp,
            solution=self.kernel_solution,
            slot_order=self.slot_order,
        )
        if use_dcp:
            local_output, local_lse = out
            out = combine_attention_partials(
                local_output,
                local_lse,
                group=self.kvp_group,
                rank=self.kvp_rank,
                sink=None,
                keep_all_heads=keep_all_heads,
            ).to(
                torch.bfloat16 if q_view.dtype == torch.float8_e4m3fn else q_view.dtype
            )
        return out.reshape(-1, heads * layer.v_head_dim)


register_backend("dsa", {AttentionArch.DSA}, DSABackend)
