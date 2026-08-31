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

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.attention.flash_attn import flash_attn_varlen_func
from tokenspeed_kernel.ops.attention.flash_mla import (
    flash_mla_with_kvcache,
    get_mla_metadata,
)
from tokenspeed_kernel.ops.attention.flashinfer import (
    BatchMLAPagedAttentionWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.mla_cache_groups import (
    MlaCacheGroupMixin,
)
from tokenspeed.runtime.layers.attention.chunk import (
    build_chunked_prefill_metadata_arrays,
)
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.kernel_page_sizes import (
    FLASH_MLA_PAGE_SIZE as PAGE_SIZE,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    cache_debug_enabled,
)
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.layers.attention.utils import (
    create_flashinfer_kv_indices_triton,
)
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.flashinfer_config import get_flashinfer_workspace_size

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


@dataclass
class FlashMLADecodeMetadata:
    num_extends: int = 0
    flashmla_metadata: object | None = None
    page_table: torch.Tensor | None = None
    seq_lens_k: torch.Tensor | None = None
    # Cache-group path only: absolute latent write locations, request-major, with
    # ``group_q_len_per_req`` entries per batch row (1 outside target verify).
    # None on the classic page_table path.
    group_out_cache_loc: torch.Tensor | None = None
    group_q_len_per_req: int = 1


@dataclass
class _PrefillMetadata:
    prefill_wrapper: BatchMLAPagedAttentionWrapper
    use_ragged: bool
    # Cache-group path only: packed absolute latent write locations for the extend
    # tokens (query order). None on the classic page_table path.
    group_out_cache_loc: torch.Tensor | None = None


@dataclass
class _ChunkedPrefillMetadata:
    extend_prefix_lens: torch.Tensor
    extend_prefix_lens_cpu: torch.Tensor
    extend_seq_lens: torch.Tensor
    extend_seq_lens_cpu: torch.Tensor
    req_pool_indices: torch.Tensor
    cum_extend_seq_lens: torch.Tensor
    max_extend_seq_len: int
    chunked_loop_num: int
    chunk_kv_indices_list: list
    chunked_seq_len: torch.Tensor
    cu_chunked_seq_len: torch.Tensor
    max_chunk_len_per_loop: list


# Shared across all flashinfer prefill wrappers used by FlashMLABackend.
_global_workspace_buffer = None


class FlashMLABackend(MlaCacheGroupMixin, AttentionBackend):
    """FlashMLA attention backend for TokenSpeed scheduling.

    Uses the FlashMLA kernel for decode (any q_len); uses FlashInfer's MLA
    prefill wrappers for the EXTEND path.

    Decode consumes the LCM full-history table when bound to a cache-group
    contract (see :class:`MlaCacheGroupMixin`); otherwise it reads the classic
    ``page_table`` table. The FlashMLA kernel walks pages at a fixed
    ``PAGE_SIZE`` stride, so that is the backend's kernel page size.
    """

    # Eager refresh swaps in a fresh tile-schedule object every step (the
    # kernel freezes the schedule on first use); the replayed graph re-runs
    # its recorded schedule-build instead and never reads this field through
    # Python, so the pointer guard must not pin it.
    graph_unstable_metadata_fields: frozenset[str] = frozenset({"flashmla_metadata"})

    def __init__(self, config: MLAConfig):
        super().__init__(config)

        # Parse constants
        self.max_context_len = config.context_len
        self.kv_cache_quant_method = config.kv_cache_quant_method
        self.cache_dtype = config.kv_cache_dtype

        # Cache-group (LCM) state. Latched on the first cache metadata;
        # the FlashMLA kernel's page stride is PAGE_SIZE, so that is the kernel
        # page size the group-table expansion targets.
        self._cache_groups_bound = False
        self.kernel_page_size = PAGE_SIZE
        self.max_num_pages = (self.max_context_len + PAGE_SIZE - 1) // PAGE_SIZE

        # MLA-specific dimensions
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_cache_dim = config.kv_lora_rank + config.qk_rope_head_dim
        self.scaling = config.scaling
        self.softmax_scale = config.scaling
        self.data_type = config.kv_cache_dtype
        self.q_data_type = config.dtype
        self.num_local_heads = config.num_attention_heads // config.attn_tp_size
        self.num_q_heads = config.num_attention_heads // config.attn_tp_size

        # FlashMLA-specific
        self.draft_token_num = 0
        # A block drafter (DFLASH/DSpark) drafts a whole block per pass; its
        # captured decode graph seeds block-end lengths via
        # ``fill_block_decode_seq_lens`` (see the drafter's is_capturing path).
        self.draft_block_decode = bool(getattr(config, "draft_block_decode", False))

        if self.kv_cache_quant_method == "per_token_head":
            raise NotImplementedError(
                "FlashMLABackend no longer supports "
                "kv_cache_quant_method='per_token_head'."
            )
        if self.cache_dtype == torch.float8_e4m3fn:
            raise NotImplementedError(
                "FlashMLABackend no longer supports dense FP8 KV cache. "
                "Use a non-FP8 KV cache."
            )

        # Workspace buffer + flashinfer prefill wrappers (EXTEND path only).
        global _global_workspace_buffer
        if _global_workspace_buffer is None:
            _global_workspace_buffer = torch.empty(
                get_flashinfer_workspace_size(),
                dtype=torch.uint8,
                device=config.device,
            )
        self.workspace_buffer = _global_workspace_buffer

        max_bs = config.max_bs
        self.kv_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=config.device
        )
        self.qo_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=config.device
        )

        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )
        self.prefill_wrapper_paged = BatchMLAPagedAttentionWrapper(
            self.workspace_buffer,
            backend="auto",
        )
        self.indices_updater_prefill = _PrefillIndicesUpdater(config, self)

        # Metadata state. Decode and prefill metadata are split so MIXED batches
        # can carry both simultaneously (decode-half + prefill-half sub-contexts
        # dispatch to their respective metadata).
        self.forward_decode_metadata: FlashMLADecodeMetadata | None = None
        self.forward_prefill_metadata: _PrefillMetadata | None = None
        self.chunked_prefill_metadata: _ChunkedPrefillMetadata | None = None
        self.last_seq_lens_sum: int | None = None
        # FlashMLA builds its tile schedule lazily inside the FIRST
        # flash_mla_with_kvcache call (from that call's cache_seqlens) and then
        # freezes it on the FlashMLASchedMeta object. So a sched object must not
        # be reused across calls whose cache_seqlens differ, or the kernel keeps
        # attending the stale (first-seen) sequence length. Eager decode takes a
        # fresh object every step; under CUDA graph a fresh object is created at
        # capture time (see init_forward_metadata_capture_cuda_graph) so the
        # schedule build is recorded into the graph and recomputed from the live
        # cache_seqlens buffer on each replay. Strong refs to every sched captured
        # into a graph, so none is GC'd while its graph is alive (multiple sampling
        # variants may capture the same bs).
        self._decode_tile_metadata_keepalive: list[object] = []

    # ------------------------------------------------------------------
    # Metadata init
    # ------------------------------------------------------------------

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        page_table: torch.Tensor = None,
        extend_with_prefix: bool = False,
        extend_prefix_lens: torch.Tensor | None = None,
        **kwargs,
    ):
        kwargs.pop("cache_metadata", None)
        kwargs.pop("forward_batch", None)
        group_table = self._resolve_full_history_table(
            kwargs.pop("block_tables", None), bs
        )
        if group_table is not None:
            self._cache_groups_bound = True
        elif self.is_draft and bs > 0:
            # The drafter drives this draft backend directly with no group
            # tables; it hands over the batch-ordered draft page table (row i
            # is batch position i, raw scheduler pages). Same expansion as a
            # wrapper-delivered group table.
            self._cache_groups_bound = True
            group_table = self._expand_history_table(page_table[:bs])
        elif self._cache_groups_bound and bs > 0 and not forward_mode.is_idle():
            raise RuntimeError(
                "FlashMLABackend is bound to cache groups but received no "
                "group tables; refusing the legacy page_table path"
            )

        if not (forward_mode.is_extend_or_mixed() or forward_mode.is_idle()):
            raise RuntimeError(
                "FlashMLA decode metadata goes through refresh_decode_metadata; "
                f"init_forward_metadata only serves extend/mixed ({forward_mode})"
            )
        if forward_mode.is_extend_or_mixed():
            self._init_prefill_metadata(
                req_pool_indices=req_pool_indices[:num_extends],
                seq_lens=seq_lens[:num_extends],
                page_table=page_table,
                extend_with_prefix=extend_with_prefix,
                extend_prefix_lens=extend_prefix_lens,
                extend_prefix_lens_cpu=kwargs.pop("extend_prefix_lens_cpu"),
                extend_seq_lens=kwargs.pop("extend_seq_lens"),
                extend_seq_lens_cpu=kwargs.pop("extend_seq_lens_cpu"),
                group_table=(
                    group_table[:num_extends] if group_table is not None else None
                ),
            )
        # Target mixed/idle batches carry decode rows whose metadata this
        # init must cover. A draft's decode metadata instead comes from the
        # wrapper's refresh_decode_metadata after this init (the unified
        # draft contract).
        if forward_mode.is_idle() or (forward_mode.is_mixed() and not self.is_draft):
            self._init_decode_metadata(
                bs,
                num_extends,
                req_pool_indices,
                seq_lens,
                page_table,
                group_table=group_table,
                q_len_per_req=self._verify_q_len(forward_mode),
            )

    @contextmanager
    def override_num_extends(self, num_extends: int):
        assert self.forward_decode_metadata is not None
        prev = self.forward_decode_metadata.num_extends
        self.forward_decode_metadata.num_extends = num_extends
        try:
            yield
        finally:
            self.forward_decode_metadata.num_extends = prev

    def _new_eager_tile_metadata(self):
        """Return a fresh (uninitialized) FlashMLASchedMeta for one eager decode.

        FlashMLA freezes its tile schedule from the FIRST kernel call's
        cache_seqlens, so eager decode must hand the kernel a fresh object every
        step — reusing one would keep attending the first step's sequence length.
        The object itself is cheap; the schedule build happens inside the kernel.
        """
        return get_mla_metadata()[0]

    def _capture_decode_tile_metadata(self, bs: int):
        """Return the FlashMLASchedMeta to record into the CUDA graph for ``bs``.

        Must be called *inside* graph capture: a fresh (uninitialized) object is
        created so the schedule build is captured and re-executed from the live
        cache_seqlens buffer on every replay. The object is kept alive for the
        lifetime of the graph that recorded it (a graph replays the recorded
        schedule-build kernel against this object's tensors).
        """
        tile_metadata = get_mla_metadata()[0]
        self._decode_tile_metadata_keepalive.append(tile_metadata)
        return tile_metadata

    def _init_decode_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        group_table: torch.Tensor | None = None,
        q_len_per_req: int = 1,
    ):
        if group_table is not None:
            # The table is already in FlashMLA's PAGE_SIZE kernel pages; resolve
            # the write locations. Target verify writes the whole spec window
            # (seq-N..seq-1); plain decode writes position seq-1.
            page_table_rows = group_table[:bs]
            group_out_cache_loc = self._cache_decode_out_cache_loc(
                group_table,
                seq_lens,
                batch_size=bs,
                validate_pages=cache_debug_enabled(),
                q_len_per_req=q_len_per_req,
            )
        else:
            # No group table: the idle/warmup forward before the backend binds
            # to the contract (a live LCM batch always resolves a group table
            # from the target's cache metadata or the draft's batch-ordered
            # page table; init_forward_metadata raises if a bound backend gets
            # neither). page_table is only an empty/dummy placeholder here,
            # batch-ordered like the draft table.
            page_table_rows = page_table[:bs]
            group_out_cache_loc = None
        self.forward_decode_metadata = FlashMLADecodeMetadata(
            num_extends=num_extends,
            flashmla_metadata=self._new_eager_tile_metadata(),
            page_table=page_table_rows,
            seq_lens_k=seq_lens,
            group_out_cache_loc=group_out_cache_loc,
            group_q_len_per_req=q_len_per_req,
        )

    def select_out_cache_loc(self, layer, out_cache_loc, forward_mode=None):
        """Group-derived latent write location on the cache-group path.

        Identity when not cache-group bound (classic page_table path) or when
        idle. Decode writes one location per request (position seq-1); extend
        writes the packed [prefix, seq) locations per request in query order.
        """
        if (
            not self._cache_groups_bound
            or forward_mode is None
            or forward_mode.is_idle()
        ):
            return out_cache_loc
        if self.is_draft:
            # A draft (MTP/EAGLE chain) owns its per-step write locations and
            # passes num_extends == bs by its own convention; the group metadata
            # holds no draft locations, so honor the caller's out_cache_loc.
            return out_cache_loc
        if forward_mode.is_decode():
            metadata = self.forward_decode_metadata
            if metadata is None or metadata.group_out_cache_loc is None:
                raise RuntimeError("FlashMLA decode write locations are missing")
            # Locations are request-major with group_q_len_per_req entries per
            # row, so a mixed batch skips whole verify windows, not single rows.
            locs = metadata.group_out_cache_loc[
                metadata.num_extends * metadata.group_q_len_per_req :
            ]
        else:
            metadata = self.forward_prefill_metadata
            if metadata is None or metadata.group_out_cache_loc is None:
                raise RuntimeError("FlashMLA prefill write locations are missing")
            locs = metadata.group_out_cache_loc
        if out_cache_loc is not None and locs.shape[0] != out_cache_loc.shape[0]:
            raise RuntimeError(
                f"FlashMLA write locations cover {locs.shape[0]} tokens but "
                f"the caller provided {out_cache_loc.shape[0]}"
            )
        return locs

    def _init_prefill_metadata(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        extend_with_prefix: bool,
        extend_prefix_lens: torch.Tensor | None,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        group_table: torch.Tensor | None = None,
    ):
        # EXTEND path — flashinfer ragged/paged prefill.
        if extend_prefix_lens is None:
            raise RuntimeError(
                "FlashMLABackend.init_forward_metadata requires "
                "extend_prefix_lens in extend mode."
            )
        seq_lens_cpu = seq_lens.cpu()
        seq_lens_sum = seq_lens_cpu.sum().item()
        self.last_seq_lens_sum = seq_lens_sum

        extend_no_prefix = not extend_with_prefix
        use_ragged = (
            not global_server_args_dict["mla_disable_ragged"] and extend_no_prefix
        )

        # The cache-group path needs two differently-shaped views of the kernel
        # full-history table:
        #   * flashinfer paged prefill (plan page_size=1) walks a PER-TOKEN slot
        #     table, so expand each token to its absolute latent slot.
        #   * chunked prefix replay (create_chunked_cache_kv_indices_paged) walks
        #     the PAGE table directly, deriving slot = page_id*p + pos%p
        #     in-kernel.
        # New-token write locations come from _extend_out_cache_loc either way.
        group_out_cache_loc = None
        if group_table is not None:
            prefill_table = self._group_per_token_slot_table(
                group_table,
                batch_size=seq_lens.shape[0],
                page_size=self.kernel_page_size,
                max_context_len=self.max_context_len,
            ).to(torch.int32)
            prefill_req_pool_indices = torch.arange(
                seq_lens.shape[0], dtype=torch.int64, device=prefill_table.device
            )
            chunk_table = group_table[: seq_lens.shape[0]]
            chunk_page_size = self.kernel_page_size
            group_out_cache_loc = self._extend_out_cache_loc(
                group_table[: seq_lens.shape[0]],
                extend_prefix_lens_cpu,
                extend_seq_lens_cpu,
                validate_pages=cache_debug_enabled(),
            )
        else:
            # No group table: a warmup placeholder forward before any tables
            # are published. It never reads real KV, so page_table is an
            # empty/dummy batch-ordered placeholder here (row i == batch
            # position i). A live LCM batch always resolves a group table.
            prefill_table = page_table
            prefill_req_pool_indices = torch.arange(
                seq_lens.shape[0], dtype=torch.int64, device=page_table.device
            )
            chunk_table = page_table
            chunk_page_size = PAGE_SIZE

        self.indices_updater_prefill.update(
            prefill_req_pool_indices,
            seq_lens,
            seq_lens_sum,
            extend_prefix_lens,
            page_table=prefill_table,
            prefill_wrapper_paged=self.prefill_wrapper_paged,
            use_ragged=use_ragged,
        )
        self.forward_prefill_metadata = _PrefillMetadata(
            self.prefill_wrapper_paged, use_ragged, group_out_cache_loc
        )

        num_extends = extend_seq_lens.shape[0]
        cum_extend_seq_lens = torch.zeros(
            num_extends + 1, device=self.device, dtype=torch.int32
        )
        torch.cumsum(extend_seq_lens, dim=0, out=cum_extend_seq_lens[1:])
        max_extend_seq_len = extend_seq_lens_cpu.max().item()
        (
            chunked_loop_num,
            chunk_kv_indices_list,
            chunked_seq_len,
            cu_chunked_seq_len,
            max_chunk_len_per_loop,
        ) = build_chunked_prefill_metadata_arrays(
            extend_prefix_lens,
            extend_prefix_lens_cpu,
            chunk_table,
            prefill_req_pool_indices,
            chunk_page_size,
        )
        self.chunked_prefill_metadata = _ChunkedPrefillMetadata(
            extend_prefix_lens=extend_prefix_lens,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            req_pool_indices=prefill_req_pool_indices,
            cum_extend_seq_lens=cum_extend_seq_lens,
            max_extend_seq_len=max_extend_seq_len,
            chunked_loop_num=chunked_loop_num,
            chunk_kv_indices_list=chunk_kv_indices_list,
            chunked_seq_len=chunked_seq_len,
            cu_chunked_seq_len=cu_chunked_seq_len,
            max_chunk_len_per_loop=max_chunk_len_per_loop,
        )

    # ------------------------------------------------------------------
    # CUDA graph (decode only, any q_len)
    # ------------------------------------------------------------------

    def init_cuda_graph_state(self, max_bs: int, **kwargs):
        max_context_len = self.max_context_len + PAGE_SIZE - 1
        # 4 PAGES are reserved for speculation
        self.cuda_graph_kv_indices = torch.full(
            (max_bs, (max_context_len + 4 * PAGE_SIZE) // PAGE_SIZE),
            1,
            dtype=torch.int32,
            device="cuda",
        )
        # Own the persistent cache_seqlens buffer the captured decode kernel reads from
        self.cuda_graph_seq_lens = torch.zeros(max_bs, dtype=torch.int32, device="cuda")
        # Persistent write-location buffer whose address the captured graph
        # records; replay refreshes it in place from the live full-history
        # table. Target verify records spec_num_tokens write locations per
        # request; a draft owns its per-step locations and never reads it
        # (select_out_cache_loc's draft guard).
        if not self.is_draft:
            self.cuda_graph_group_out_cache_loc = torch.zeros(
                max_bs * max(1, self.spec_num_tokens),
                dtype=torch.int64,
                device="cuda",
            )
        else:
            self.cuda_graph_group_out_cache_loc = None
        # Buffers were (re)allocated: cached per-bs views must rebuild.
        self.decode_cuda_graph_metadata: dict[int, FlashMLADecodeMetadata] = {}

    def _decode_views(self, bs: int) -> FlashMLADecodeMetadata:
        """Per-bs decode metadata views over the persistent buffers.

        One builder for capture and refresh; cached per bs — pointer-stable,
        no storage allocated. ``flashmla_metadata`` is deliberately per-step
        mutable (exempted via ``graph_unstable_metadata_fields``): eager
        refresh installs a fresh tile schedule, capture installs the recorded
        one, and replay never reads it through Python.
        """
        metadata = self.decode_cuda_graph_metadata.get(bs)
        if metadata is not None:
            return metadata
        q_len = self._graph_verify_q_len()
        group_out_cache_loc = None
        if self.cuda_graph_group_out_cache_loc is not None:
            group_out_cache_loc = self.cuda_graph_group_out_cache_loc[: bs * q_len]
        metadata = FlashMLADecodeMetadata(
            num_extends=0,
            flashmla_metadata=None,
            page_table=self.cuda_graph_kv_indices[:bs],
            seq_lens_k=self.cuda_graph_seq_lens[:bs],
            group_out_cache_loc=group_out_cache_loc,
            group_q_len_per_req=q_len,
        )
        self.decode_cuda_graph_metadata[bs] = metadata
        return metadata

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        cache_group_ids: tuple[str, ...] = (),
        page_table: torch.Tensor | None = None,
        **kwargs,
    ):
        # The one sanctioned capture-only asymmetry: flash_mla freezes its
        # tile schedule on the first kernel call against a sched-meta, so
        # capture installs a dedicated object whose schedule-build the graph
        # records (kept alive for the graph's lifetime); eager refresh swaps
        # in a fresh one per step instead. The refresh super() runs seeds the
        # seq_lens the recorded schedule-build reads.
        super().init_forward_metadata_capture_cuda_graph(
            bs,
            req_pool_indices,
            seq_lens,
            forward_mode,
            cache_group_ids=cache_group_ids,
            page_table=page_table,
            **kwargs,
        )
        self.forward_decode_metadata.flashmla_metadata = (
            self._capture_decode_tile_metadata(bs)
        )

    def refresh_decode_metadata(
        self,
        bs: int,
        actual_bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        forward_mode: ForwardMode,
        page_table: torch.Tensor | None = None,
        num_extends: int = 0,
        for_graph_replay: bool = False,
        **kwargs,
    ) -> None:
        if forward_mode is None or not forward_mode.is_decode_or_idle():
            raise RuntimeError(f"Not supported forward mode: {forward_mode}")

        metadata = self._decode_views(bs)
        # Verify rows span seq-N..seq-1; clamp so a request shorter than the
        # window does not resolve locations before its start. On replay the
        # width must match what capture baked into the recorded buffer views
        # (_graph_verify_q_len is deterministic, so recomputing restores it).
        if for_graph_replay:
            q_len = self._graph_verify_q_len()
        else:
            q_len = self._verify_q_len(forward_mode)
        # clamp_min(1) is the identity, so the verify clamp is unconditional.
        self.cuda_graph_seq_lens[:bs].copy_(seq_lens[:bs].clamp_min(q_len))

        # The wrapper's per-group tables refresh the block table whenever
        # delivered — decode-only PD nodes included (they never run an extend
        # forward, so gating on the extend-latched _cache_groups_bound alone
        # would leave the kernel reading a stale/zero kv-index table instead
        # of the transferred KV). Latch _cache_groups_bound for the write-loc
        # path.
        refreshed = False
        table = self._resolve_full_history_table(kwargs.get("block_tables"), 0)
        if table is not None:
            self._cache_groups_bound = True
            # Refresh the block table and per-request write locations in place
            # from the live full-history table (cache-group path). Live tables
            # carry one row per REAL request; the idle replay's synthesized
            # placeholder rows are all dummies, so actual_bs caps the copy.
            real_bs = min(int(table.shape[0]), bs, actual_bs)
            if real_bs > 0:
                self.cuda_graph_kv_indices[:real_bs, : table.shape[1]].copy_(
                    table[:real_bs]
                )
                self._cache_decode_out_cache_loc(
                    table,
                    self.cuda_graph_seq_lens,
                    batch_size=real_bs,
                    validate_pages=cache_debug_enabled(),
                    out=self.cuda_graph_group_out_cache_loc,
                    q_len_per_req=q_len,
                )
            # Padded rows resolve to the null page 0.
            self.cuda_graph_kv_indices[real_bs:bs].zero_()
            if self.cuda_graph_group_out_cache_loc is not None:
                self.cuda_graph_group_out_cache_loc[
                    real_bs * q_len : bs * q_len
                ].zero_()
            refreshed = True
        elif self.is_draft and bs > 0:
            # Draft: expand the staged batch-ordered draft page table (raw
            # scheduler pages) straight into the persistent kv-indices buffer.
            # Latch the group binding: the draft consumes published pages from
            # here on.
            self._cache_groups_bound = True
            self._expand_history_table(
                page_table[:bs], out=self.cuda_graph_kv_indices[:bs]
            )
        elif page_table is not None:
            # Idle/warmup before the backend binds: page_table is an empty
            # batch-ordered placeholder. A live LCM batch takes one of the
            # branches above.
            block_table = page_table[:bs]
            self.cuda_graph_kv_indices[:bs, : block_table.shape[1]].copy_(block_table)

        # Bind the cached per-bs views on BOTH paths. The replayed kernels
        # read the refreshed buffers directly; eager forwards read them
        # through this metadata. flash_mla freezes its tile schedule on the
        # first kernel call against a sched-meta, so eager installs a FRESH
        # object per step, while replay leaves the field alone — the graph
        # re-runs its recorded schedule-build and never reads it from Python.
        metadata.num_extends = num_extends
        metadata.group_q_len_per_req = q_len
        if for_graph_replay:
            # Restore the capture-baked loc view (an interleaved eager step
            # may have re-pointed it): same buffer, same width, same address.
            metadata.group_out_cache_loc = (
                self.cuda_graph_group_out_cache_loc[: bs * q_len]
                if self.cuda_graph_group_out_cache_loc is not None
                else None
            )
        else:
            metadata.flashmla_metadata = self._new_eager_tile_metadata()
            metadata.group_out_cache_loc = (
                self.cuda_graph_group_out_cache_loc[: bs * q_len]
                if refreshed and self.cuda_graph_group_out_cache_loc is not None
                else None
            )
        self.forward_decode_metadata = metadata

    def fill_block_decode_seq_lens(self, bs: int, block_seq_lens: torch.Tensor) -> None:
        """Publish block-end cache lengths inside a captured draft graph.

        A block drafter runs its multi-token pass inside the captured graph and
        writes the per-request block-end length here (one row per request; the
        block's ``draft_query_width`` queries share it, non-causal). ``forward_decode``
        repeats each row across the block's queries, so the graph keeps ``bs``
        rows -- mirrors ``TRTLLMMLABackend.fill_block_decode_seq_lens``.
        """
        if not self.draft_block_decode:
            raise RuntimeError("Block decode sequence lengths require DFLASH mode.")
        self.cuda_graph_seq_lens[:bs].copy_(
            block_seq_lens[:bs].clamp(self.spec_num_tokens, self.max_context_len)
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool,
        forward_mode: ForwardMode,
        **kwargs,
    ):
        assert forward_mode is not None and forward_mode.is_extend()

        # Prefill: dispatch to ragged (MHA-style) or absorbed (MQA) path.
        if self.forward_prefill_metadata.use_ragged:
            return self._forward_normal_extend(q, k, v, layer, save_kv_cache)
        else:
            return self._forward_absorbed_extend(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                save_kv_cache,
            )

    def forward_extend_chunked(
        self,
        q,
        k,
        v,
        scaling,
        logits_soft_cap=None,
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
        if causal:
            step_counter = self.step_counter
            if step_counter is not None:
                step_counter.record_cache()
        head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # flash_attn_varlen_func has no `out=` parameter; copy into the
        # caller-provided buffer at the end when requested.
        output, lse, *_ = flash_attn_varlen_func(
            q=q.view(-1, self.num_local_heads, head_dim),
            k=k.view(-1, self.num_local_heads, head_dim).to(q.dtype),
            v=v.view(-1, self.num_local_heads, self.v_head_dim).to(q.dtype),
            cu_seqlens_q=cum_seq_lens_q,
            cu_seqlens_k=cum_seq_lens_kv,
            max_seqlen_q=max_q_len,
            max_seqlen_k=max_kv_len,
            softmax_scale=scaling,
            causal=causal,
            return_attn_probs=True,
        )
        if out is not None:
            out.copy_(output.view(out.shape))
            output = out
        # lse must be transposed when using fa3.
        return output, lse.T.contiguous()

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool,
        **kwargs,
    ) -> torch.Tensor:
        # Multi-token decode (target verify or drafter compound) reuses
        # the multi-token kernel path in forward_extend.
        q_len_per_req = q.shape[0] // bs if bs > 0 else 1
        if q_len_per_req > 1:
            metadata = self.forward_decode_metadata
            num_extends = metadata.num_extends
            bs = (
                q.shape[0]
                if self.is_draft
                else metadata.page_table.shape[0] - num_extends
            )

        o, _ = self._run_flash_mla_decode(
            q,
            k,
            v,
            layer,
            out_cache_loc,
            token_to_kv_pool,
            bs,
            save_kv_cache=save_kv_cache,
            cache_seqlens_offset=self.draft_token_num,
        )

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    # ------------------------------------------------------------------
    # EXTEND prefill helpers
    # ------------------------------------------------------------------

    def _forward_normal_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        save_kv_cache: bool = True,
    ):
        assert not save_kv_cache

        o = self.prefill_wrapper_ragged.forward(
            q,
            k.view(-1, layer.tp_k_head_num, layer.head_dim),
            v.view(-1, layer.tp_k_head_num, layer.v_head_dim),
            causal=True,
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
        )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _forward_absorbed_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        save_kv_cache: bool = True,
    ):
        # q is whole Q [T, H, head_dim]; k is whole latent [T, 1, head_dim].
        # flashinfer prefill_wrapper.run() requires q_nope / q_pe split, so
        # slice views here (free) before handing off to the kernel.
        assert k is not None

        if save_kv_cache:
            token_to_kv_pool.set_mla_kv_buffer(
                layer,
                out_cache_loc,
                k[..., : layer.v_head_dim],
                k[..., layer.v_head_dim :],
            )

        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        q_nope = q[..., : layer.v_head_dim]
        q_pe = q[..., layer.v_head_dim :]
        o = q_nope.new_empty(q_nope.shape)

        k_buf = token_to_kv_pool.get_key_buffer(layer.layer_id).to(q_nope.dtype)
        o = self.forward_prefill_metadata.prefill_wrapper.run(
            q_nope,
            q_pe,
            k_buf[:, :, : layer.v_head_dim],
            k_buf[:, :, layer.v_head_dim :],
            out=o,
        )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _run_flash_mla_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        *,
        save_kv_cache: bool,
        cache_seqlens_offset: int,
    ):
        if k is not None:
            assert v is not None
            if save_kv_cache:
                token_to_kv_pool.set_kv_buffer(layer, out_cache_loc, k, v)

        metadata = self.forward_decode_metadata
        num_extends = metadata.num_extends
        k_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)
        assert (
            layer.tp_q_head_num == self.num_q_heads
        ), f"{layer.tp_q_head_num=} != {self.num_q_heads=}"
        reshape_q = q.view(bs, -1, self.num_q_heads, layer.head_dim)

        page_table = metadata.page_table[num_extends : num_extends + bs]
        cache_seqlens = metadata.seq_lens_k.to(torch.int32) + cache_seqlens_offset
        # Draft block-decode: forward_decode flattened q to one kernel row per
        # drafted block position (bs == bs_orig * draft_query_width), but the
        # group table and seq_lens carry one entry per request. Repeat each
        # request's row across its block positions so every block query attends
        # the whole block (block-diffusion), mirroring tokenspeed_mla's
        # _expand_block_decode_metadata; the FlashMLA kernel requires
        # cache_seqlens to be shape (num_kernel_rows).
        src_rows = page_table.shape[0]
        if self.is_draft and 0 < src_rows < bs and bs % src_rows == 0:
            width = bs // src_rows
            page_table = page_table.repeat_interleave(width, dim=0)
            cache_seqlens = cache_seqlens.repeat_interleave(width)

        return flash_mla_with_kvcache(
            q=reshape_q,
            k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
            block_table=page_table,
            cache_seqlens=cache_seqlens,
            head_dim_v=self.kv_lora_rank,
            tile_scheduler_metadata=metadata.flashmla_metadata,
            softmax_scale=layer.scaling,
            causal=True,
        )


class _PrefillIndicesUpdater:
    """Plans FlashInfer MLA prefill wrappers for the EXTEND path."""

    def __init__(self, config: MLAConfig, attn_backend: FlashMLABackend):
        self.num_local_heads = config.num_attention_heads // config.attn_tp_size
        self.kv_cache_quant_method = config.kv_cache_quant_method
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.scaling = config.scaling
        self.data_type = config.kv_cache_dtype
        self.q_data_type = config.dtype
        self.attn_backend = attn_backend

        self.kv_indptr = attn_backend.kv_indptr
        self.qo_indptr = attn_backend.qo_indptr
        self.prefill_wrapper_ragged = attn_backend.prefill_wrapper_ragged

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        page_table: torch.Tensor = None,
        prefill_wrapper_paged: BatchMLAPagedAttentionWrapper = None,
        use_ragged: bool = False,
    ):
        if use_ragged:
            paged_kernel_lens = prefix_lens
            paged_kernel_lens_sum = 0
        else:
            paged_kernel_lens = seq_lens
            paged_kernel_lens_sum = seq_lens_sum

        self._call_begin_forward(
            self.prefill_wrapper_ragged,
            prefill_wrapper_paged,
            req_pool_indices,
            paged_kernel_lens,
            paged_kernel_lens_sum,
            seq_lens,
            prefix_lens,
            self.kv_indptr,
            self.qo_indptr,
            use_ragged,
            page_table=page_table,
        )

    def _call_begin_forward(
        self,
        wrapper_ragged: BatchPrefillWithRaggedKVCacheWrapper,
        wrapper_paged: BatchMLAPagedAttentionWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr: torch.Tensor,
        use_ragged: bool,
        page_table: torch.Tensor = None,
    ):
        bs = len(seq_lens)
        sm_scale = self.scaling

        assert len(seq_lens) == len(req_pool_indices)
        torch.cumsum(paged_kernel_lens, dim=0, out=kv_indptr[1 : bs + 1])
        kv_indptr = kv_indptr[: bs + 1]
        if wrapper_paged._use_cuda_graph:
            kv_indices = wrapper_paged._kv_indices_buf
        else:
            kv_indices = torch.empty(
                paged_kernel_lens_sum,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
        if page_table is not None:
            create_flashinfer_kv_indices_triton[(bs,)](
                page_table,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                None,
                kv_indices,
                page_table.shape[1],
            )
        torch.cumsum(seq_lens - prefix_lens, dim=0, out=qo_indptr[1 : bs + 1])
        qo_indptr = qo_indptr[: bs + 1]

        if use_ragged:
            wrapper_ragged.begin_forward(
                qo_indptr=qo_indptr,
                kv_indptr=qo_indptr,
                num_qo_heads=self.num_local_heads,
                num_kv_heads=self.num_local_heads,
                head_dim_qk=self.qk_nope_head_dim + self.qk_rope_head_dim,
                head_dim_vo=self.v_head_dim,
                q_data_type=self.q_data_type,
            )
        else:
            kv_len_arr = kv_indptr[1:] - kv_indptr[:-1]
            wrapper_paged.plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_len_arr,
                self.num_local_heads,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                1,
                True,
                sm_scale,
                self.q_data_type,
                self.data_type,
            )


register_backend("flashmla", {AttentionArch.MLA}, FlashMLABackend)
