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
import logging
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.attention.dsa import (
    dsa_decode,
    dsa_decode_dcp_records,
    dsa_decode_dcp_records_supported,
    dsa_decode_dcp_records_workspace_bytes,
    dsa_plan,
    dsa_prefill,
    dsa_prefill_dcp_records,
    dsa_prefill_dcp_records_supported,
    select_dsa_prefill_topk_for_rows,
)
from tokenspeed_kernel.ops.attention.dsa.flashinfer import (
    reserve_flashinfer_trtllm_dsa_buffers,
    trtllm_sparse_mla_query_heads,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import KernelRegistry, priority_band_name
from tokenspeed_kernel.selection import KernelPlan, NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.configs.numerics import BITWISE_ENVELOPES
from tokenspeed.runtime.distributed.comm_ops import (
    acquire_exchange_input,
    all_gather_single,
)
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
    dsa_kvp_history_prefill,
    index_k_row_bytes,
)
from tokenspeed.runtime.layers.attention.dcp.cache import (
    HistoryGatherPlan,
    HistoryGatherWorkspace,
    all_gather_history_rows,
    gather_history_rows,
    plan_history_gather,
)
from tokenspeed.runtime.layers.attention.dcp.comm import (
    combine_attention_partials,
    combine_attention_records,
    gather_query_heads,
    gather_query_heads_padded,
    gather_query_heads_rank_major,
)
from tokenspeed.runtime.layers.attention.dcp.placement import (
    CachePlacement,
    compact_owned_cache_slots,
    cyclic_slot_owner,
    max_owned_index_pages,
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

logger = logging.getLogger(__name__)

# Logged sparse-leaf configurations: one log line per distinct configuration.
_LOGGED_SPARSE_LEAVES: set[tuple] = set()

# Owned-width KVP index tiles span whole groups of this many pages, so tile widths take few values.
_OWNED_INDEX_COLUMN_ALIGN = 16


def _priority_band(name: str) -> str:
    """The registered priority band of kernel ``name``, spelled as
    ``describe_kernel`` does (``PORTABLE``, ``SPECIALIZED+1``), so a portable
    fallback stands out in the startup log."""
    spec = KernelRegistry.get().get_by_name(name)
    if spec is None:
        return "unregistered"
    return priority_band_name(spec.priority)


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


def balanced_group_rows(lengths: list[int], cap: int) -> int:
    """Minimize the largest contiguous group without increasing group count.

    ``lengths`` are nonnegative request history row counts, each at most the
    nonnegative workspace ``cap``. Requests stay whole and in input order.
    Return the smallest row bound preserving greedy packing's group count
    at ``cap``; return ``cap`` for an empty input.
    """
    if cap < 0:
        raise ValueError(f"history workspace capacity must be nonnegative, got {cap}")
    for length in lengths:
        if length < 0:
            raise ValueError(
                f"request history length must be nonnegative, got {length}"
            )
        if length > cap:
            raise RuntimeError(
                f"request history of {length} rows exceeds the "
                f"{cap}-row query-context-parallel gather workspace"
            )
    if not lengths:
        return cap

    def count(bound: int) -> int:
        groups, rows = 1, 0
        for length in lengths:
            if rows + length > bound:
                groups, rows = groups + 1, 0
            rows += length
        return groups

    # For nonnegative lengths, greedy packing uses the fewest contiguous
    # groups at a given bound. That count cannot rise as the bound grows.
    target = count(cap)
    lo, hi = max(lengths), cap
    while lo < hi:
        mid = (lo + hi) // 2
        if count(mid) <= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _k_row_causal_lengths(seq_lens: torch.Tensor, k: int) -> torch.Tensor:
    """``[bs]`` request lengths as the sparse decode core's per-token KV
    lengths (``[bs * k]`` int32): verify row ``t`` of a request sees
    ``seq_len - (k - 1 - t)`` tokens, clamped at zero."""
    offsets = torch.arange(k, device=seq_lens.device, dtype=torch.int32) - (k - 1)
    return seq_lens.to(torch.int32).unsqueeze(1).add(offsets).clamp_min(0).reshape(-1)


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
            slice of the rank's shard rows (a rank's rows are in request
            order, so a request range is one run of them).
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


@dataclasses.dataclass(frozen=True)
class HistoryPrefillGroup:
    """One request group of a KVP prefill over gathered histories.

    Every rank sends ``send_pages`` kernel pages of a cache plane -- the
    group's history pages it owns, in history order, then padding pages no
    row maps to -- and one all-gather lands them rank-major: rank ``k``'s
    pages from gathered page ``k * send_pages``.

    Attributes:
        requests: The extend requests in the group (a contiguous range).
        query: The group's extend rows of the forward.
        fetch_pages: ``[send_pages]`` int64 local kernel pages this rank sends.
    """

    requests: slice
    query: slice
    fetch_pages: torch.Tensor

    @property
    def send_pages(self) -> int:
        return self.fetch_pages.numel()


@dataclasses.dataclass(frozen=True)
class HistoryPrefillPlan:
    """A pure KVP extend's prefill over gathered histories
    (``dsa_kvp_history_prefill``).

    Attributes:
        groups: Request groups in request order. Every rank gathers every
            group, one group's pages at a time.
        slots: ``[rows]`` int64 gathered slot of every history row, the rows
            numbered request-major over every extend request's prefix and
            chunk (``dsa_prefill_topk``'s workspace rows); a group's rows
            map into its own gathered pages.
    """

    groups: tuple[HistoryPrefillGroup, ...]
    slots: torch.Tensor


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
    _make_dense_leaf = staticmethod(_make_dense_leaf)

    # Opt-in only: a padded replay needs all DSA ops in a break narrowing to live rows.
    cuda_graph_support = CudaGraphSupport(prefill_graph=False)
    # Owner-masked writes and KVP-merged cores (dcp/) let hybrid DSA layers shard KV.
    supports_mla_dcp = True
    # Off until __init__ reads dsa_kvp_history_prefill; the plan is made per
    # pure extend.
    _history_prefill: bool = False
    history_prefill: HistoryPrefillPlan | None = None

    def __init__(self, config: AttnConfig, spec: DSAConfig, *, kernel_page_size: int):
        if len(config.kvp_group) > 1 and (
            spec.index_topk <= 0 or spec.index_topk & (spec.index_topk - 1)
        ):
            raise ValueError("DSA KVP candidate topk must be a positive power of two")
        super().__init__(config, spec, kernel_page_size=kernel_page_size)
        platform = current_platform()
        self._dense_backend = self._make_dense_leaf(
            config, spec, platform, kernel_page_size
        )
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
        self.index_n_heads = spec.index_n_heads
        # The index-K plane's format (configs/dsa.py INDEX_K_FORMATS): the
        # history gather packs rows in it and hands the indexer that form.
        self.index_k_format = spec.index_k_format
        self.query_shard_metadata: DSAQueryShardMetadata | None = None
        # The current pure KVP extend's prefill over gathered histories, or None.
        self.history_prefill: HistoryPrefillPlan | None = None
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
        # Plan from static geometry; sparse request sizes never select a new core.
        self.batch_invariant: bool = (
            global_server_args_dict["numerics"] in BITWISE_ENVELOPES
        )
        # --dsa-slot-order: how the cores reduce the selected slots
        # (docs/design/numerics.md, invariance.batch).
        self.slot_order: str = global_server_args_dict["dsa_slot_order"]
        self.kernel_solution: str | None = None
        self.prefill_kernel_plan = None
        self.decode_kernel_plan = None
        if self.batch_invariant:
            signature = format_signature(q=dense_tensor_format(self.q_data_type))
            traits = {
                "qk_nope_head_dim": self.qk_nope_head_dim,
                "kv_lora_rank": self.kv_lora_rank,
                "qk_rope_head_dim": self.qk_rope_head_dim,
                "page_size": self.kernel_page_size,
                "has_kv_cache": True,
                "has_sparse_kv_cache": False,
                "logit_cap": False,
                "topk_layout": "global_slots",
                "slot_order": self.slot_order,
                "return_lse": frozenset({False, len(self.kvp_group) > 1}),
            }
            variable = frozenset({"batch_size", "tokens", "topk", "max_seqlen_k"})
            self.prefill_kernel_plan = KernelPlan.select(
                "attention",
                "dsa_prefill",
                signature,
                features=frozenset({"batch_invariant"}),
                static_traits={**traits, "q_len": 1},
                variable_traits=variable,
                solution=None,
                declared_traits=(
                    frozenset({"slot_order"})
                    if self.slot_order == "sorted"
                    else frozenset()
                ),
                ignore_layout=False,
            )
            self.decode_kernel_plan = KernelPlan.select(
                "attention",
                "dsa_decode",
                signature,
                features=frozenset({"batch_invariant"}),
                static_traits={
                    **traits,
                    "q_len": frozenset(range(1, self.spec_num_tokens + 1)),
                },
                variable_traits=variable,
                solution=None,
                declared_traits=(
                    frozenset({"slot_order"})
                    if self.slot_order == "sorted"
                    else frozenset()
                ),
                ignore_layout=False,
            )
        # Rank-major query gather with partials in the exchange layout; a pinned prefill plan keeps its own core.
        self.kvp_prefill_query_layout: bool = self.prefill_kernel_plan is None
        # KVP prefill's Triton FP8 core, probed once a router's solution pin has landed.
        self._dcp_records_core: bool = False
        # Whether that selection runs the trtllm-gen core, which takes a padded query.
        self._dcp_padded_query: bool = False
        # Whether every KVP decode selection runs gfx950's dense core on an
        # FP8 cache, whose DCP records form the head-sharded decode runs.
        self._dcp_decode_records_core: bool = False
        self._prefill_page_table: torch.Tensor | None = None
        # Owned KVP index-tile width in kernel pages; None means the page table's width.
        self._prefill_index_columns: int | None = None
        self.kpool_runtime = (
            KPoolRuntime(spec.index_kpool, spec.index_topk)
            if spec.index_kpool is not None
            else None
        )
        padded_replay_opt_in = False
        if self.kpool_runtime is not None:
            # GLM-5.3-Flash (the one KPool consumer) handles padded prefill
            # replay explicitly in its model code, so the class-level DSA
            # restriction does not apply to it.
            self.cuda_graph_support = CudaGraphSupport(prefill_graph=True)
        elif spec.padded_prefill_replay and platform.is_nvidia:
            # K3.1 narrows padded replays to live rows; validated on NVIDIA only.
            self.cuda_graph_support = CudaGraphSupport(prefill_graph=True)
            padded_replay_opt_in = True
        if len(self.qcp_group) > 1:
            # The sharded extend arm gathers each request's history in the
            # eager attention break; a captured prefill would replay the
            # dummy's gather. Overrides the KPool consumer's declaration too.
            self.cuda_graph_support = CudaGraphSupport(prefill_graph=False)
            self._probe_history_gather_topk_leaf(spec)
        # A pure KVP extend over gathered histories attends each request
        # group's history in the eager attention break, as the sharded extend
        # arm does, so prefill capture is off here too.
        self._history_prefill: bool = dsa_kvp_history_prefill(config)
        if self._history_prefill:
            self.cuda_graph_support = CudaGraphSupport(prefill_graph=False)
        # Planned sparse cores are captured only if they declare cuda_graph.
        self.cuda_graph_support = self.cuda_graph_support & CudaGraphSupport.from_plans(
            prefill=self.prefill_kernel_plan, decode=self.decode_kernel_plan
        )
        if padded_replay_opt_in and self.cuda_graph_support.prefill_graph:
            # Logged after the vetoes on every rank, so a dissenting rank shows.
            logger.info("DSA prefill graph admitted (padded_prefill_replay)")

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
        # Probed here: a router's solution pin lands after construction.
        self._dcp_records_core = self._probe_dcp_records_core()
        self._dcp_padded_query = self._probe_dcp_padded_query()
        self._dcp_decode_records_core = self._probe_dcp_decode_records_core()
        self._log_sparse_leaves()
        self._reserve_trtllm_sparse_buffers()

    def _reserve_trtllm_sparse_buffers(self) -> None:
        """Allocate the persistent buffers of every sparse call the forward
        makes that selects a trtllm-gen core, now rather than in its first
        forward (``reserve_flashinfer_trtllm_dsa_buffers``), whose first one
        is the startup tuning forward: allocated between its activations, they
        may keep the rest of a freed activation block's segment resident. A
        call carries the KVP group's gathered heads, or every head where the
        query holds them.
        """
        if torch.device(self.device).type != "cuda":
            return
        use_dcp = len(self.kvp_group) > 1
        heads = (
            {self.num_local_heads * len(self.kvp_group), self.num_attention_heads}
            if use_dcp
            else {self.num_local_heads}
        )
        for _, mode, signature, traits, solution, plan in self._sparse_leaf_probes():
            if mode not in ("dsa_prefill", "dsa_decode"):
                continue
            try:
                name, _ = self._resolve_sparse_leaf(
                    mode, signature, traits, solution, plan
                )
            except (NoKernelFoundError, TypeError, ValueError):
                # The forward's own selection raises it.
                continue
            if name.startswith("flashinfer_trtllm_"):
                for num_heads in sorted(heads):
                    reserve_flashinfer_trtllm_dsa_buffers(
                        self.device,
                        num_heads=num_heads,
                        return_lse=traits["return_lse"],
                    )

    def _probe_dcp_records_core(self) -> bool:
        """Whether the head-sharded KVP prefill may run the DCP records core.

        ``dsa_prefill_dcp_records`` is the Triton ``dsa_prefill`` core reading
        the rank-major gather and writing the partials exchange's records, so
        it stands in only where the forward's own ``dsa_prefill`` selection
        (the selection inputs ``_sparse_leaf_probes`` mirrors, solution pin
        and kernel override included) picks ``triton_dsa_prefill`` without a
        plan, and where the records op selects for the same inputs. Boot
        configuration only, so every rank of the group agrees.

        Only an FP8 cache's query qualifies: its bytes take the NCCL
        inner-dimension gather (``gather_query_heads``), whose relayout copy
        the rank-major gather skips. A BF16 query's inner-dimension gather
        takes the multicast kernels where multicast reaches, which land it
        token-major with no copy, while the rank-major gather is NCCL only;
        that trade is unmeasured, so a BF16 query keeps its gather.
        """
        if (
            len(self.kvp_group) <= 1
            or self.prefill_kernel_plan is not None
            or self.data_type != torch.float8_e4m3fn
        ):
            return False
        if current_platform().is_npu:
            # The Ascend leaf launches its own operators, not these facades.
            return False
        for _, mode, signature, traits, solution, _ in self._sparse_leaf_probes():
            if mode != "dsa_prefill" or not traits["return_lse"]:
                continue
            try:
                core = select_kernel(
                    "attention", mode, signature, traits=traits, solution=solution
                )
                select_kernel(
                    "attention",
                    "dsa_prefill_dcp_records",
                    signature,
                    traits=traits,
                    solution=solution,
                )
            except NoKernelFoundError:
                return False
            return core.name == "triton_dsa_prefill"
        return False

    def _probe_dcp_padded_query(self) -> bool:
        """Whether the head-sharded KVP prefill may gather its query heads
        straight into the trtllm-gen core's padded query.

        That core runs 65..127 heads (K3.1's 96 gathered at KVP8) as 128 and
        would pad the token-major gather with a second copy of every row;
        ``gather_query_heads_padded`` lands the rank-major gather in the
        padded query with one. Only where the forward's own LSE
        ``dsa_prefill`` selection picks ``flashinfer_trtllm_dsa_prefill``
        without a plan, on an FP8 cache, whose query takes the NCCL gather
        either way (as :meth:`_probe_dcp_records_core`). Boot configuration
        only, so every rank of the group agrees.
        """
        if (
            len(self.kvp_group) <= 1
            or self.prefill_kernel_plan is not None
            or self.data_type != torch.float8_e4m3fn
            or current_platform().is_npu
        ):
            return False
        for _, mode, signature, traits, solution, _ in self._sparse_leaf_probes():
            if mode != "dsa_prefill" or not traits["return_lse"]:
                continue
            try:
                core = select_kernel(
                    "attention", mode, signature, traits=traits, solution=solution
                )
            except NoKernelFoundError:
                return False
            return core.name == "flashinfer_trtllm_dsa_prefill"
        return False

    def _probe_dcp_decode_records_core(self) -> bool:
        """Whether the head-sharded KVP decode may run the DCP records core.

        ``dsa_decode_dcp_records`` is gfx950's dense ``dsa_decode`` core
        reading the rank-major gather, its split reduction writing the
        partials exchange's records, so it stands in only where every decode
        selection the forward makes (``_sparse_leaf_probes``: one row per
        request and the verify rows) picks ``gluon_dsa_decode_gfx950``
        without a plan, and where the records op selects for the same inputs.
        An FP8 cache only, as :meth:`_probe_dcp_records_core`. Boot
        configuration only, so every rank of the group agrees.
        """
        if (
            len(self.kvp_group) <= 1
            or self.decode_kernel_plan is not None
            or self.data_type != torch.float8_e4m3fn
            or current_platform().is_npu
        ):
            return False
        decodes = [
            probe for probe in self._sparse_leaf_probes() if probe[1] == "dsa_decode"
        ]
        for _, mode, signature, traits, solution, _ in decodes:
            try:
                core = select_kernel(
                    "attention", mode, signature, traits=traits, solution=solution
                )
                select_kernel(
                    "attention",
                    "dsa_decode_dcp_records",
                    signature,
                    traits=traits,
                    solution=solution,
                )
            except NoKernelFoundError:
                return False
            if core.name != "gluon_dsa_decode_gfx950":
                return False
        return bool(decodes)

    def _sparse_leaf_probes(self) -> list[tuple]:
        """The sparse selections this leaf's forward makes, as ``(label, mode,
        signature, traits, solution, plan)``.

        Each mirrors its facade call: the core query is FP8 on an FP8 KV cache
        (the cast in ``forward_sparse_prefill`` / ``forward_sparse_decode``),
        else the model dtype; ``return_lse`` follows DCP, and a query-sharded
        extend adds its LSE-free arm; decode runs one row per request and
        ``spec_num_tokens`` verify rows; the cores' width is ``index_topk``
        plus a KPool's ``pool_size - 1`` tail slots (GLM-5.3-Flash's 2051).
        Under DCP the indexer scores the model-dtype index query and head
        weights against packed index-K pages of this leaf's page size,
        without a solution pin (``select_dsa_topk``).
        """
        use_dcp = len(self.kvp_group) > 1
        q_dtype = (
            torch.float8_e4m3fn
            if self.data_type == torch.float8_e4m3fn
            else self.q_data_type
        )
        signature = format_signature(q=dense_tensor_format(q_dtype))
        kpool = self.kpool_runtime
        traits = {
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "page_size": self.kernel_page_size,
            "topk": self.index_topk + (0 if kpool is None else kpool.pool_size - 1),
            "has_kv_cache": True,
            "has_sparse_kv_cache": False,
            "logit_cap": False,
            "topk_layout": "global_slots",
            "slot_order": self.slot_order,
        }
        probes = []
        prefill_lse = {use_dcp} | ({False} if len(self.qcp_group) > 1 else set())
        for return_lse in sorted(prefill_lse):
            probes.append(
                (
                    f"dsa_prefill(q_len=1, return_lse={return_lse})",
                    "dsa_prefill",
                    signature,
                    {**traits, "q_len": 1, "return_lse": return_lse},
                    self.kernel_solution,
                    self.prefill_kernel_plan,
                )
            )
        for q_len in sorted({1, self.spec_num_tokens}):
            probes.append(
                (
                    f"dsa_decode(q_len={q_len}, return_lse={use_dcp})",
                    "dsa_decode",
                    signature,
                    {**traits, "q_len": q_len, "return_lse": use_dcp},
                    self.kernel_solution,
                    self.decode_kernel_plan,
                )
            )
        if use_dcp:
            probes.append(
                (
                    "dsa_index_candidates",
                    "dsa_index_candidates",
                    format_signature(
                        q=dense_tensor_format(self.q_data_type),
                        weights=dense_tensor_format(self.q_data_type),
                    ),
                    {
                        "index_heads": self.index_n_heads,
                        "head_dim": self.index_head_dim,
                        "page_size": self.kernel_page_size,
                        "index_k_layout": "packed",
                    },
                    None,
                    None,
                )
            )
        return probes

    def _resolve_sparse_leaf(
        self, mode: str, signature, traits: dict, solution: str | None, plan
    ) -> tuple[str, bool]:
        """The kernel the forward's ``mode`` call runs, and whether a plan fixed it.

        A batch-invariant plan names its kernel without reselection, after the
        validation the facades run before every launch, so a plan the first
        forward would reject (an FP8-cast query, another solution pin) raises
        here; otherwise ``select_kernel`` makes the facade's selection
        (``TOKENSPEED_KERNEL_OVERRIDE_*`` included) and caches it for the
        forward.
        """
        if plan is not None:
            plan.validate("attention", mode, signature, traits, None, solution, None)
            return plan.kernel.name, True
        if mode == "dsa_index_candidates" and self.index_k_format != "fp8_scaled":
            # The facade refuses any other plane before it selects.
            raise TypeError(
                "DSA index candidates require packed FP8 Index-K storage, "
                f"index_k_format is {self.index_k_format!r}"
            )
        kernel = select_kernel(
            "attention", mode, signature, traits=traits, solution=solution
        )
        return kernel.name, False

    def _log_sparse_leaves(self) -> None:
        """Log the sparse leaves this leaf's forward will run, once per
        distinct configuration in the process.

        Every leaf is printed with its priority band, so a portable fallback
        (the Triton core under DCP, say) shows at startup rather than in a
        profile. A probe that raises reads ``none`` (no leaf matches) or
        ``unresolved`` and is repeated at WARNING with its error; logging
        never fails startup. Ranks of a KVP group share the configuration,
        so only its first rank logs.
        """
        if current_platform().is_npu:
            # The Ascend leaf launches its own operators, not these facades.
            return
        try:
            if self.kvp_rank != 0:
                return
            probes = self._sparse_leaf_probes()
            key = (
                self.data_type,
                len(self.kvp_group),
                tuple(
                    (
                        label,
                        mode,
                        signature,
                        tuple(sorted(traits.items())),
                        solution,
                        None if plan is None else plan.kernel.name,
                    )
                    for label, mode, signature, traits, solution, plan in probes
                ),
            )
        except Exception as e:
            logger.warning(f"DSA sparse leaves unresolved: {type(e).__name__}: {e!s}")
            return
        if key in _LOGGED_SPARSE_LEAVES:
            return
        _LOGGED_SPARSE_LEAVES.add(key)
        entries = []
        for label, mode, signature, traits, solution, plan in probes:
            try:
                name, planned = self._resolve_sparse_leaf(
                    mode, signature, traits, solution, plan
                )
                band = _priority_band(name)
            except Exception as e:
                missing = isinstance(e, NoKernelFoundError)
                entries.append(f"{label}={'none' if missing else 'unresolved'}")
                logger.warning(
                    f"DSA sparse leaf {label} unresolved: {type(e).__name__}: {e!s}"
                )
                continue
            entries.append(f"{label}={name} [{band}{', planned' if planned else ''}]")
        logger.info(
            f"DSA sparse leaves (kv_cache={self.data_type!s}, "
            f"kvp={len(self.kvp_group):d}, solution={self.kernel_solution!s}): "
            + "; ".join(entries)
        )

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
        kv_bytes = rows * self.kv_cache_dim * self.data_type.itemsize
        # Index-K rows packed in the plane's format (FP8 bytes then fp32
        # scales, or bf16 keys), one gather per group.
        row_bytes = index_k_row_bytes(self.index_head_dim, self.index_k_format)
        buffer = torch.empty(
            max(kv_bytes, rows * row_bytes), dtype=torch.uint8, device=self.device
        )
        self._history_workspace = HistoryGatherWorkspace(
            rows=rows,
            kv=buffer[:kv_bytes].view(self.data_type).view(rows, self.kv_cache_dim),
            index_k=buffer[: rows * row_bytes].view(rows, row_bytes),
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

    def takes_owned_decode_topk(self, layer) -> bool:
        return len(self.kvp_group) > 1

    def dcp_decode_query_rider(self, layer, q: torch.Tensor) -> torch.Tensor | None:
        """This rank's decode query heads as the bytes an owned top-k's
        score gather can carry (``select_owned_dsa_topk_with_rider``), when
        the sparse decode of ``q`` over that top-k takes the DCP records
        core: ``forward_sparse_decode`` then reads every rank's heads where
        that gather lands them (``gathered_query``) instead of gathering
        them itself. None otherwise. Decided from host sizes and boot
        configuration, the same on every rank."""
        if not self.takes_owned_decode_topk(layer):
            return None
        heads = self._query_heads(q, layer)
        q_view = self._decode_query_view(q, layer, heads)
        if not self._takes_dcp_records_decode(
            tokens=q.shape[0],
            heads=heads,
            topk=self.index_topk,
            q_dtype=q_view.dtype,
            kv_dtype=self.data_type,
        ):
            return None
        return q_view.reshape(q.shape[0], -1).contiguous().view(torch.uint8)

    def _decode_query_view(self, q: torch.Tensor, layer, heads: int) -> torch.Tensor:
        """``q`` as the sparse decode cores read it: ``[rows, heads,
        head_dim]``, FP8 over an FP8 cache."""
        q_view = q.view(q.shape[0], heads, layer.head_dim)
        if self.data_type == torch.float8_e4m3fn:
            q_view = q_view.to(self.data_type)
        return q_view

    def _takes_dcp_records_decode(
        self,
        *,
        tokens: int,
        heads: int,
        topk: int,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
    ) -> bool:
        """Whether a head-sharded decode of ``tokens`` rows of ``heads``
        query heads over ``topk`` compacted slots takes the records core
        (``_forward_dcp_records_decode``)."""
        return (
            len(self.kvp_group) > 1
            and not self._query_holds_every_head(heads)
            and self._dcp_decode_records_core
            and dsa_decode_dcp_records_supported(
                tokens=tokens,
                heads=heads * len(self.kvp_group),
                topk=topk,
                q_dtype=q_dtype,
                kv_dtype=kv_dtype,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
            )
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
        else:
            self._publish_k_row_indexer_rows(metadata, seq_lens, bs)
        self._publish_kv_seq_lens(metadata, bs)

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

    def _publish_kv_seq_lens(self, metadata, bs: int) -> None:
        """Publish the sparse decode core's per-token causal KV lengths for a
        target verify. They derive from the dense leaf's ``seq_lens_k`` (its
        verify-clamped copy, padded graph rows included), so they equal what
        ``forward_sparse_decode`` would derive; written in place once
        allocated, so a captured graph replays the edit."""
        if self.is_draft or self.spec_num_tokens <= 1:
            return
        causal = _k_row_causal_lengths(metadata.seq_lens_k[:bs], self.spec_num_tokens)
        if metadata._dsa_kv_seq_lens is None:
            metadata._dsa_kv_seq_lens = causal
        else:
            metadata._dsa_kv_seq_lens.copy_(causal)

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
        # Decided from host values every rank of the KVP group shares, so
        # the ranks run the same gathers.
        self.history_prefill = None
        if (
            self._history_prefill
            and forward_mode.is_extend()
            and num_extends > 0
            and page_table_cpu is not None
        ):
            self.history_prefill = self._plan_history_prefill(
                num_extends=num_extends,
                page_table_cpu=page_table_cpu,
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
            self._publish_kv_seq_lens(metadata, bs)
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
        self._prefill_index_columns = None
        if (
            len(self.kvp_group) > 1
            and forward_mode.is_extend()
            and num_extends > 0
            and self.query_shard_metadata is None
            and self.history_prefill is None
            and page_table_cpu is not None
        ):
            self._prefill_index_columns = self._plan_prefill_index_columns(
                page_table_cpu[:num_extends],
                extend_prefix_lens_cpu[:num_extends].to(torch.int64)
                + extend_seq_lens_cpu[:num_extends].to(torch.int64),
            )

    # KVP prefill: owned-width index tiles

    def _plan_prefill_index_columns(
        self, page_table_cpu: torch.Tensor, seq_lens_cpu: torch.Tensor
    ) -> int:
        """The owned width of a pure extend's KVP index tiles, in kernel pages.

        Host arithmetic over the host page table and lengths, identical on
        every rank, so every rank tiles and exchanges the same rows: the most
        pages any rank owns below any request's length bounds every query
        row's owned pages. The width is at least ``ceil(index_topk /
        page_size)`` pages, so the logits stay wide enough for the
        length-aware top-k, in whole groups of ``_OWNED_INDEX_COLUMN_ALIGN``
        pages; the indexer clamps it to the forward's page table.
        """
        bound = max_owned_index_pages(
            page_table_cpu,
            seq_lens_cpu,
            page_size=self.kernel_page_size,
            placement=self.cache_placement(None),
        )
        floor = -(-self.index_topk // self.kernel_page_size)
        align = _OWNED_INDEX_COLUMN_ALIGN
        return -(-max(bound, floor) // align) * align

    def prefill_index_columns(self) -> int | None:
        """The owned width, in kernel pages, of this forward's KVP index tiles.

        Set by :meth:`init_forward_metadata` for a pure extend over KVP pages
        (and not query-sharded): the
        model's KVP top-k plans each query row's owned index pages that wide
        instead of the page table's width, so a logits budget fits more rows
        per tile. ``None`` keeps the page table's width.
        """
        return self._prefill_index_columns

    # ------------------------------------------------------------------
    # KVP prefill over gathered histories (dsa_kvp_history_prefill)
    # ------------------------------------------------------------------

    def takes_history_prefill_topk(self) -> bool:
        """Whether this forward's indexer selects over gathered history pages
        (:meth:`gather_history_index_k_pages`) with the plan's slot map: a
        pure KVP extend over gathered histories."""
        return self.history_prefill is not None

    def require_history_prefill(self) -> HistoryPrefillPlan:
        """This forward's prefill over gathered histories."""
        if self.history_prefill is None:
            raise RuntimeError(
                "this forward is not a KVP prefill over gathered histories"
            )
        return self.history_prefill

    def _plan_history_prefill(
        self,
        *,
        num_extends: int,
        page_table_cpu: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
    ) -> HistoryPrefillPlan:
        """Group the extend requests and plan each group's page gather.

        Host arithmetic over the host page table and lengths, identical on
        every rank; the device work is the plan's copies and one slot build.
        Greedy grouping by gathered pages against the history workspace, a
        request never splits; a request whose pages alone exceed it gathers
        into a transient buffer (:meth:`_gather_history_pages`).
        """
        if self._history_workspace is None:
            raise RuntimeError(
                "the DSA history gather needs its workspace; "
                "preallocate_history_gather_workspace did not run"
            )
        page = self.kernel_page_size
        placement = self.cache_placement(None)
        degree = 1 if placement is None else len(placement.group)
        rank = 0 if placement is None else placement.rank
        extend_lens = [int(x) for x in extend_seq_lens_cpu[:num_extends].tolist()]
        history_lens = [
            int(p) + e
            for p, e in zip(extend_prefix_lens_cpu[:num_extends].tolist(), extend_lens)
        ]
        pages = [
            page_table_cpu[r, : -(-rows // page)].to(torch.int64)
            for r, rows in enumerate(history_lens)
        ]
        owners = [cyclic_slot_owner(p * page, placement) for p in pages]
        if any(bool((owner < 0).any()) for owner in owners):
            raise RuntimeError("a history page of the KVP extend has no owner")
        counts = [torch.bincount(owner, minlength=degree) for owner in owners]
        capacity = self._history_workspace.rows // page
        table = torch.zeros((num_extends, max(map(len, pages))), dtype=torch.int32)
        groups: list[HistoryPrefillGroup] = []
        first = query_start = 0
        while first < num_extends:
            last, owned = first, counts[first].clone()
            while (
                last + 1 < num_extends
                and degree * int((owned + counts[last + 1]).max()) <= capacity
            ):
                last += 1
                owned += counts[last]
            send = int(owned.max())
            requests = slice(first, last + 1)
            owner = torch.cat(owners[requests])
            # Rank k's pages in history order: the stable sort's run of k.
            order = torch.argsort(owner, stable=True)
            starts = torch.cumsum(owned, 0) - owned
            ordinal = torch.empty_like(order)
            ordinal[order] = torch.arange(order.numel())
            gathered = owner * send + ordinal - starts[owner]
            at = 0
            for r in range(first, last + 1):
                table[r, : len(pages[r])] = gathered[at : at + len(pages[r])]
                at += len(pages[r])
            lo = int(starts[rank])
            mine = torch.cat(pages[requests])[order[lo : lo + int(owned[rank])]]
            local, _ = resolve_cache_slots(mine * page, placement)
            fetch = torch.zeros(send, dtype=torch.int64)
            fetch[: mine.numel()] = local // page
            query_rows = sum(extend_lens[requests])
            groups.append(
                HistoryPrefillGroup(
                    requests=requests,
                    query=slice(query_start, query_start + query_rows),
                    fetch_pages=self._upload_history_plan(fetch),
                )
            )
            query_start += query_rows
            first = last + 1
        slots = build_prefill_kv_workspace_slots(
            page_table=self._upload_history_plan(table),
            seq_lens=self._upload_history_plan(
                torch.tensor(history_lens, dtype=torch.int64)
            ),
            max_seq_len=max(history_lens),
            page_size=page,
            device=torch.device(self.device),
            num_tokens=sum(history_lens),
        )
        return HistoryPrefillPlan(groups=tuple(groups), slots=slots)

    def _upload_history_plan(self, host: torch.Tensor) -> torch.Tensor:
        """A pageable upload blocks the host until the stream drains the
        previous forward; pinned staging keeps it stream-ordered and
        asynchronous. CPU backends (tests) take the plain path."""
        if torch.device(self.device).type != "cuda":
            return host.to(self.device)
        return host.pin_memory().to(self.device, non_blocking=True)

    def _gather_history_pages(
        self,
        cache: torch.Tensor,
        row_bytes: int,
        group: HistoryPrefillGroup,
        workspace: torch.Tensor,
    ) -> torch.Tensor:
        """All-gather ``group``'s pages of one cache plane into the leading
        bytes of ``workspace`` (a transient buffer when they exceed it).

        ``cache`` is the plane as the pool holds it, whole kernel pages of
        ``row_bytes``-byte slots in slot order; the result has its dtype and
        trailing shape, its slots being the gathered slots the plan's
        ``slots`` name. A collective over the page owners: every rank calls
        it for every group.
        """
        if not cache.is_contiguous():
            raise RuntimeError("the history gather reads a contiguous cache plane")
        page_bytes = self.kernel_page_size * row_bytes
        raw = cache.reshape(-1).view(torch.uint8)
        pages = raw[: raw.numel() // page_bytes * page_bytes].view(-1, page_bytes)
        send = pages.index_select(0, group.fetch_pages)
        size = len(self.kvp_group) * send.numel()
        landing = workspace.reshape(-1).view(torch.uint8)
        out = (
            landing[:size]
            if size <= landing.numel()
            else torch.empty(size, dtype=torch.uint8, device=send.device)
        )
        all_gather_single(out, send.reshape(-1), self.kvp_group)
        return out.view(cache.dtype).view(-1, *cache.shape[1:])

    def gather_history_index_k_pages(
        self, layer_id: int, token_to_kv_pool, group: HistoryPrefillGroup
    ) -> torch.Tensor:
        """One group's history pages of the layer's index-K plane, in the
        pool's own page layout and shape, so the prefill top-k scores them
        through the plan's slot map as it scores the pool."""
        if self._history_workspace is None:
            raise RuntimeError("DSA history gather workspace is not allocated")
        if int(token_to_kv_pool.index_k_page_size) != self.kernel_page_size:
            raise RuntimeError(
                f"the history gather moves {self.kernel_page_size}-row kernel "
                f"pages; the index-K plane pages {token_to_kv_pool.index_k_page_size}"
            )
        return self._gather_history_pages(
            token_to_kv_pool.get_index_k_buffer(layer_id),
            index_k_row_bytes(self.index_head_dim, self.index_k_format),
            group,
            self._history_workspace.index_k,
        )

    def gather_history_kv_pages(
        self, layer, token_to_kv_pool, group: HistoryPrefillGroup
    ) -> torch.Tensor:
        """One group's history pages of the layer's latent cache, in the
        cache's own shape: the slots the plan's slot map names."""
        if self._history_workspace is None:
            raise RuntimeError("DSA history gather workspace is not allocated")
        kv_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)
        return self._gather_history_pages(
            kv_cache,
            kv_cache[0].numel() * kv_cache.element_size(),
            group,
            self._history_workspace.kv,
        )

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
        group. Keep the minimum number of contiguous groups that fit the gather
        workspace, then minimize their largest summed history length. A
        request never splits.
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
        bound = balanced_group_rows(history_lens, self._history_workspace.rows)
        placement = self.cache_placement(None)
        owned = owned_history_rows(
            page_table_cpu[:num_extends],
            torch.tensor(history_lens, dtype=torch.int64),
            page_size=self.kernel_page_size,
            placement=placement,
        )
        groups: list[QueryShardHistoryGroup] = []
        local_start = 0
        row_base = 0
        first = 0
        while first < num_extends:
            last = first
            rows = history_lens[first]
            while last + 1 < num_extends and rows + history_lens[last + 1] <= bound:
                last += 1
                rows += history_lens[last]
            requests = slice(first, last + 1)
            local_rows = sum(plan.local_rows_per_request[requests])
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
                    local_query=slice(local_start, local_start + local_rows),
                    gather=gather,
                )
            )
            local_start += local_rows
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather one group's latent rows, rank-major, without reordering.

        The sparse core reads rows by index, so the rows stay where the
        all-gather put them and the caller maps the selected positions
        instead, avoiding a full history copy. The tradeoff is mapping
        ``[local_query_rows, topk]`` indices for the selected rows.

        Returns ``(kv, rows)``: a ``[pages * kernel_page_size, kv_cache_dim]``
        view of the gather workspace (valid until the next group's gather) --
        a whole number of kernel pages, so every ``dsa_prefill`` solution's
        view of a flat ``[slots, dim]`` cache holds, paged or not; the padding
        rows are never selected -- and the ``[group.rows]`` int32 row of
        ``kv`` holding each history position. A collective over the page
        owners: every rank calls it for every group.
        """
        if self._history_workspace is None:
            raise RuntimeError("DSA history gather workspace is not allocated")
        kv_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_flat = kv_cache.reshape(-1, kv_cache.shape[-1])
        local = kv_flat.index_select(0, group.gather.send_slots)
        page = self.kernel_page_size
        rows = len(group.gather.group) * group.gather.send_rows
        paged_rows = -(-rows // page) * page
        out = self._history_workspace.kv
        if paged_rows > out.shape[0]:
            # The owners' send padding pushed a full workspace's group past
            # its capacity.
            out = out.new_empty((paged_rows, out.shape[1]))
        all_gather_history_rows(group.gather, local, out=out)
        return out[:paged_rows], group.gather.gathered_index

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
            group.gather.send_slots,
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
        topk_owned: bool = False,
        gathered_query: torch.Tensor | None = None,
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
                topk_owned=topk_owned,
                gathered_query=gathered_query,
            )
        if topk_owned or gathered_query is not None:
            raise ValueError("an owned decode top-k needs its slots")
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
        topk_lens = topk_lens.to(device=q.device, dtype=torch.int32).contiguous()
        if self.history_prefill is not None:
            # Over gathered histories: topk_slots are gathered slots
            # (the plan's slot map), attended against each group's gathered
            # pages with the query's heads; nothing to merge.
            out = self._attend_history_pages(
                self.history_prefill.groups,
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
        # Gathered heads of a padded query; its partials end with the padding heads'.
        live_heads = None
        if use_dcp:
            # Owned slots first with their count as length: ~topk/kvp entries per row.
            topk_slots, topk_lens = compact_owned_cache_slots(
                topk_slots, topk_lens, self.cache_placement(layer)
            )
            # Decided before the query gather from inputs every rank shares.
            if (
                not keep_all_heads
                and self.kvp_prefill_query_layout
                and self._dcp_records_core
                and dsa_prefill_dcp_records_supported(
                    tokens=q_view.shape[0],
                    heads=heads * len(self.kvp_group),
                    topk=topk_slots.shape[1],
                )
            ):
                out = self._forward_dcp_records_prefill(
                    q_view=q_view,
                    layer=layer,
                    kv_cache=kv_cache,
                    topk_slots=topk_slots,
                    topk_lens=topk_lens,
                )
                if self.step_counter is not None:
                    self.step_counter.record_cache()
                return out.reshape(-1, heads * layer.v_head_dim)
            gathered_heads = heads * len(self.kvp_group)
            core_heads = trtllm_sparse_mla_query_heads(gathered_heads)
            if (
                not keep_all_heads
                and self.kvp_prefill_query_layout
                and self._dcp_padded_query
                and core_heads != gathered_heads
            ):
                q_view = gather_query_heads_padded(q_view, self.kvp_group, core_heads)
                live_heads = gathered_heads
            elif not keep_all_heads:
                q_view = gather_query_heads(q_view, self.kvp_group)
        out = dsa_prefill(
            q=q_view,
            kv_cache=kv_cache,
            sparse_kv_cache=None,
            topk_slots=topk_slots,
            topk_lens=topk_lens,
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
            kernel_plan=self.prefill_kernel_plan,
            slot_order=self.slot_order,
        )
        if use_dcp:
            local_output, local_lse = out
            if live_heads is not None:
                local_output = local_output[:, :live_heads]
                local_lse = local_lse[:, :live_heads]
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

    def _attend_history_pages(
        self,
        groups: tuple[HistoryPrefillGroup, ...],
        *,
        q_view: torch.Tensor,
        layer,
        token_to_kv_pool,
        kv_seq_lens: torch.Tensor | None,
        topk_slots: torch.Tensor,
        topk_lens: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        """Per request group, gather the latent history pages and attend the
        group's rows against them with the query's heads."""
        if kv_seq_lens is not None:
            kv_seq_lens = kv_seq_lens.to(
                device=q_view.device, dtype=torch.int32
            ).contiguous()

        def attend(rows: slice, kv: torch.Tensor) -> torch.Tensor:
            # This call mirrors the unsharded arm's: every kernel facade
            # option the unsharded call passes (the slot-order selection
            # included) must be passed here too.
            return dsa_prefill(
                q=q_view[rows],
                kv_cache=kv,
                sparse_kv_cache=None,
                topk_slots=topk_slots[rows],
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
                kernel_plan=self.prefill_kernel_plan,
                slot_order=self.slot_order,
            )

        if len(groups) == 1 and groups[0].query == slice(0, q_view.shape[0]):
            return attend(
                groups[0].query,
                self.gather_history_kv_pages(layer, token_to_kv_pool, groups[0]),
            )
        out = q_view.new_empty(
            (q_view.shape[0], q_view.shape[1], layer.v_head_dim),
            dtype=(
                torch.bfloat16 if q_view.dtype == torch.float8_e4m3fn else q_view.dtype
            ),
        )
        for group in groups:
            kv = self.gather_history_kv_pages(layer, token_to_kv_pool, group)
            out[group.query] = attend(group.query, kv)
        return out

    def _forward_dcp_records_prefill(
        self,
        *,
        q_view: torch.Tensor,
        layer,
        kv_cache: torch.Tensor,
        topk_slots: torch.Tensor,
        topk_lens: torch.Tensor,
    ) -> torch.Tensor:
        """The head-sharded KVP arm with the qlayout lever: the same core and
        exchange as ``dsa_prefill`` plus ``combine_attention_partials``, bit
        for bit, without the relayout of the gathered heads or the repack of
        the partials. The query heads stay rank-major as the all-gather lands
        them, the core writes each head's partial straight into the record its
        owner receives, and one all-to-all sends them, as the other arm's."""
        records = dsa_prefill_dcp_records(
            q=gather_query_heads_rank_major(q_view, self.kvp_group),
            kv_cache=kv_cache,
            topk_slots=topk_slots,
            topk_lens=topk_lens,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            softmax_scale=layer.scaling,
            page_size=self.kernel_page_size,
            logit_cap=layer.logit_cap,
            k_scale=1.0,
            solution=self.kernel_solution,
            slot_order=self.slot_order,
        )
        # The Triton core emits BF16 partials, which the combine keeps.
        return combine_attention_records(
            records,
            group=self.kvp_group,
            head_dim=self.kv_lora_rank,
            partial_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            sink=None,
        )

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
            kv_group, kv_rows = self.gather_history_kv(layer, token_to_kv_pool, group)
            rows = group.local_query
            if rows.stop <= rows.start:
                continue
            slots = topk_slots[rows]
            # The selected positions, mapped to the rows of the rank-major
            # gather; the order (the reduction order) is unchanged.
            # Workspace tails can contain stale slots outside this history.
            # Mask them before indexing, not only inside the attention kernel.
            columns = torch.arange(slots.shape[1], device=slots.device)
            valid = (slots >= 0) & (columns < topk_lens[rows, None])
            positions = torch.where(valid, slots - group.row_base, 0).to(torch.int64)
            slots = torch.where(valid, kv_rows[positions], -1)
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
                kernel_plan=self.prefill_kernel_plan,
                slot_order=self.slot_order,
            )
        return out

    def _published_kv_seq_lens(
        self, metadata, num_extends: int, q_len_per_req: int, num_tokens: int
    ) -> torch.Tensor | None:
        """The round's per-token causal KV lengths for the decode rows, or
        ``None`` when this forward did not publish rows of its width.

        A target verify's refresh (or mixed init) publishes them once, so
        every DSA layer of the forward reads one tensor instead of
        re-deriving it; a draft chain step moves ``seq_lens_k`` without a
        refresh, so drafts keep the local form.
        """
        rows = metadata._dsa_kv_seq_lens
        if rows is None or self.is_draft or q_len_per_req != self.spec_num_tokens:
            return None
        start = num_extends * q_len_per_req
        if rows.numel() < start + num_tokens:
            return None
        return rows[start : start + num_tokens]

    def forward_sparse_decode(
        self,
        *,
        q: torch.Tensor,
        layer,
        token_to_kv_pool,
        bs: int,
        topk_indices: torch.Tensor,
        topk_lens: torch.Tensor | None,
        topk_owned: bool = False,
        gathered_query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sparse decode over ``topk_indices``: the whole selection as
        scheduler slots, or with ``topk_owned`` (a sharded cache only) this
        rank's share of it as local cache slots, ``topk_lens`` counting them.

        ``gathered_query`` is every rank's ``q`` as the owned top-k's score
        gather landed it, uint8 ``[ranks, rows, bytes]``: the rider
        :meth:`dcp_decode_query_rider` gave for ``q``.
        """
        if topk_owned and (len(self.kvp_group) == 1 or topk_lens is None):
            raise ValueError(
                "an owned decode top-k is one rank's share of a sharded "
                "cache's selection, with its counts"
            )
        if gathered_query is not None and not topk_owned:
            raise ValueError("a gathered query rides an owned top-k's gather")
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
            kv_seq_lens = self._published_kv_seq_lens(
                metadata, num_extends, q_len_per_req, num_tokens
            )
            if kv_seq_lens is None:
                kv_seq_lens = _k_row_causal_lengths(seq_lens, q_len_per_req)

        heads = self._query_heads(q, layer)
        q_view = self._decode_query_view(q, layer, heads)
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
            # Owned slots first, counted as length (full width without topk_lens).
            # An owned top-k already is.
            if not topk_owned:
                topk_slots, topk_lens = compact_owned_cache_slots(
                    topk_slots, topk_lens, self.cache_placement(layer)
                )
            # Decided before the query gather, from boot configuration and
            # host sizes every rank of the group shares.
            records = self._takes_dcp_records_decode(
                tokens=num_tokens,
                heads=heads,
                topk=topk_slots.shape[1],
                q_dtype=q_view.dtype,
                kv_dtype=kv_cache.dtype,
            )
            if gathered_query is not None and not records:
                raise ValueError(
                    "a gathered query is the records core's; this decode "
                    "does not take it"
                )
            if records:
                out = self._forward_dcp_records_decode(
                    q_view=q_view,
                    layer=layer,
                    kv_cache=kv_cache,
                    topk_slots=topk_slots,
                    topk_lens=topk_lens,
                    q_len_per_req=q_len_per_req,
                    gathered_query=gathered_query,
                )
                return out.reshape(-1, heads * layer.v_head_dim)
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
            kernel_plan=self.decode_kernel_plan,
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

    def _forward_dcp_records_decode(
        self,
        *,
        q_view: torch.Tensor,
        layer,
        kv_cache: torch.Tensor,
        topk_slots: torch.Tensor,
        topk_lens: torch.Tensor,
        q_len_per_req: int,
        gathered_query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The head-sharded KVP decode as :meth:`_forward_dcp_records_prefill`
        runs the prefill: the same core and exchange as ``dsa_decode`` plus
        ``combine_attention_partials``, bit for bit, without the relayout of
        the gathered heads or the repack of the partials. A
        ``gathered_query`` (``forward_sparse_decode``) stands in for the
        query gather."""
        if gathered_query is None:
            q = gather_query_heads_rank_major(q_view, self.kvp_group)
        else:
            tokens, heads, dim = q_view.shape
            row_bytes = heads * dim * q_view.element_size()
            if gathered_query.shape != (len(self.kvp_group), tokens, row_bytes):
                raise ValueError(
                    f"gathered query {tuple(gathered_query.shape)} does not "
                    f"hold {len(self.kvp_group)} ranks' {tuple(q_view.shape)} heads"
                )
            q = gathered_query.view(q_view.dtype).view(-1, tokens, heads, dim)
        # The records lead the workspace, where an exchange that pulls in
        # place reads them.
        workspace_bytes = dsa_decode_dcp_records_workspace_bytes(
            tokens=q.shape[1],
            heads=q.shape[0] * q.shape[2],
            topk=topk_slots.shape[1],
            q_dtype=q.dtype,
            kv_dtype=kv_cache.dtype,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
        )
        records = dsa_decode_dcp_records(
            q=q,
            kv_cache=kv_cache,
            topk_slots=topk_slots,
            topk_lens=topk_lens,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            softmax_scale=layer.scaling,
            page_size=self.kernel_page_size,
            q_len_per_req=q_len_per_req,
            logit_cap=layer.logit_cap,
            k_scale=1.0,
            solution=self.kernel_solution,
            slot_order=self.slot_order,
            workspace=acquire_exchange_input(workspace_bytes, q, self.kvp_group).view(
                torch.int32
            ),
        )
        # The core writes BF16 partials for its FP8 query, and the combine
        # keeps their dtype: the BF16 output the other arm casts to.
        return combine_attention_records(
            records,
            group=self.kvp_group,
            head_dim=self.kv_lora_rank,
            partial_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            sink=None,
        )


if not current_platform().is_npu:
    register_backend("dsa", {AttentionArch.DSA}, DSABackend)
