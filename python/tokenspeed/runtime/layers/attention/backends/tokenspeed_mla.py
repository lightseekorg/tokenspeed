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

"""
CuteDSL MLA attention backend for TokenSpeed scheduling.

Uses CuTe DSL JIT-compiled kernels for MLA decode and prefill on Blackwell SM100 GPUs:
- tokenspeed_mla_decode for decode/verify
- tokenspeed_mla_prefill for prefill
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import triton
from tokenspeed_kernel.ops.attention.tokenspeed_mla import (
    get_num_sm,
    tokenspeed_mla_decode,
    tokenspeed_mla_prefill,
    warmup_compile_prefill,
)

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.workspace import workspace_pool
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.trtllm_mla import (
    TRTLLM_BLOCK_CONSTRAINT,
    TRTLLMMLAChunkedPrefillMetadata,
)
from tokenspeed.runtime.layers.attention.chunk import (
    build_chunked_prefill_metadata_arrays,
)
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.kernel_page_sizes import (
    TOKENSPEED_MLA_DEFAULT_PAGE_SIZE,
    TOKENSPEED_MLA_SUPPORTED_PAGE_SIZES,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    cache_debug_enabled,
)
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.pdl import pdl_enabled

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention

logger = logging.getLogger(__name__)

# Fallback q_len capacity for warming the decode workspace when the backend
# runs without speculative decoding (q_len is then 1, but keep the historical
# floor so draft experiments do not immediately hit the frozen-pool error).
_CUTEDSL_WARMUP_Q_LEN_FLOOR = 8


@dataclass
class CuteDSLMLAPrefillMetadata:
    max_seq_len: int
    cum_seq_lens: torch.Tensor
    seq_lens: torch.Tensor
    # Paged cache only: absolute latent write locations for the extend tokens,
    # flattened in q/k/v token order (positions [prefix, seq) per request).
    group_out_cache_loc: torch.Tensor | None = None


@dataclass
class CuteDSLMLADecodeMetadata:
    num_extends: int = 0
    block_kv_indices: torch.Tensor | None = None
    max_seq_len_k: int | None = None
    seq_lens_k: torch.Tensor | None = None
    # Paged cache only: absolute latent write locations, group_q_len_per_req
    # entries per batch row. Mixed-batch decode skips whole prefill windows.
    group_out_cache_loc: torch.Tensor | None = None
    group_q_len_per_req: int = 1


class CuteDSLMLABackend(AttentionBackend):
    """CuteDSL MLA attention backend for Blackwell SM100 GPUs.

    Decode uses CuTe DSL JIT-compiled kernels via tokenspeed_mla_decode().
    Prefill uses CuTe DSL FMHA kernel via tokenspeed_mla_prefill().
    """

    _logged_decode = False
    _logged_prefill = False

    # Cache contract capability: this backend consumes only history-family
    # (full-attention) tables; state groups belong to the linear sub-backend.
    cache_consumer_families = frozenset({"history"})
    draft_seq_lens_attr: str = "cuda_graph_seq_lens_buf"

    def __init__(self, config: MLAConfig):
        super().__init__(config)

        # Latched the first time paged cache metadata arrives. Once bound, a
        # forward without cache metadata is a hard error: the cache contract
        # forbids falling back to legacy page_table metadata.
        self._cache_groups_bound = False
        # Set by the registry before graph capture; see mark_cache_contract.
        self._cache_contract_bound = False

        self.max_context_len = config.context_len
        self.kernel_page_size = (
            config.kernel_page_size
            if config.kernel_page_size is not None
            else TOKENSPEED_MLA_DEFAULT_PAGE_SIZE
        )

        # MLA dimensions
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_cache_dim = config.kv_cache_dim
        self.scaling = config.scaling
        self.data_type = config.kv_cache_dtype
        self.q_data_type = config.dtype

        # Decode scratch comes from the shared WorkspacePool: the kernel's own
        # get_workspace_size formula is B*H*q_len*split_kv*(D+1)*acc_bytes with
        # B*split_kv <= num_SMs, giving the closed-form bound used in
        # _cutedsl_workspace. The content is partial decode accumulators,
        # consumed within each op and never zero-initialized, so sharing the
        # block is safe. Warm to the verify-path peak now: graph capture runs
        # the decode forward with the pool frozen.
        self._num_heads_per_tp = config.num_attention_heads // config.attn_tp_size
        self._workspace_pool = workspace_pool(config.device)
        self.cutedsl_workspace = self._cutedsl_workspace(
            max(_CUTEDSL_WARMUP_Q_LEN_FLOOR, self.spec_num_tokens or 1)
        )

        # Pre-compile prefill kernel variants so JIT doesn't run during serving.
        # The backend may be constructed once per attention layer (60x for
        # Kimi-K2.5), but `warmup_compile_prefill` is idempotent: each config
        # is only JIT'd once and cached in a module-global dict.
        # tokenspeed_mla requires --kv-cache-dtype fp8_e4m3, so tokenspeed's
        # FP8 prefill path (deepseek_v3.py:946 `use_fp8_prefill`) is always
        # on and feeds fp8_e4m3fn q/k/v to the kernel — bf16 is unreachable
        # for this backend.
        d_qk = self.qk_nope_head_dim + self.qk_rope_head_dim
        warmup_compile_prefill(
            q_dtype=torch.float8_e4m3fn,
            d_qk=d_qk,
            d_v=self.v_head_dim,
            enable_pdl=pdl_enabled(),
        )

        # Validate page_size
        if self.kernel_page_size not in TOKENSPEED_MLA_SUPPORTED_PAGE_SIZES:
            raise ValueError(
                f"tokenspeed_mla backend requires page_size 32 or 64, got {self.kernel_page_size}"
            )

        # tokenspeed_mla's CuTe DSL kernel only supports fp8_e4m3 KV cache; check
        # at startup so misconfiguration surfaces here, not in the first forward.
        kv_cache_dtype = global_server_args_dict.get("kv_cache_dtype", "auto")
        if kv_cache_dtype != "fp8_e4m3":
            raise NotImplementedError(
                f"tokenspeed_mla backend requires --kv-cache-dtype fp8_e4m3, "
                f"got {kv_cache_dtype!r}."
            )

        self.num_local_heads = self._num_heads_per_tp

        # Metadata
        self.forward_decode_metadata: CuteDSLMLADecodeMetadata | None = None
        self.forward_prefill_metadata: CuteDSLMLAPrefillMetadata | None = None
        self.decode_cuda_graph_metadata: dict[int, CuteDSLMLADecodeMetadata] = {}
        self.decode_cuda_graph_kv_indices = None
        # Cache contract decode graph: persistent write-location buffer whose
        # address the captured graph records; replay refreshes it in place.
        # Allocated in init_cuda_graph_state only when _cache_contract_bound.
        self.decode_cuda_graph_group_out_cache_loc: torch.Tensor | None = None
        self.chunked_prefill_metadata: TRTLLMMLAChunkedPrefillMetadata | None = None

    def _cutedsl_workspace(self, q_len_capacity: int) -> torch.Tensor:
        """Per-use view of the shared block, sized by the closed-form bound."""
        required = (
            get_num_sm(self.device)
            * self._num_heads_per_tp
            * q_len_capacity
            * (self.kv_lora_rank + 1)
            * 4
        )
        (buf,) = self._workspace_pool.allocate(((required,), torch.int8))
        return buf

    def mark_cache_contract(self) -> None:
        """Mark this MLA backend as a Kimi-K3 Cache contract sub-backend.

        Called by the registry when the backend is constructed for the
        Kimi-K3 LCM contract path. Enables grouped CUDA-graph
        capture/replay with stable full-attention block-table and write-location
        buffers.
        """
        if self.is_draft:
            # The CuteDSL draft keeps its batch-ordered page table. Only target
            # forwards consume scheduler cache-group metadata.
            return
        self._cache_contract_bound = True

    def _calc_padded_blocks(self, max_seq_len: int) -> int:
        """Calculate block count padded to satisfy the fused-kernel constraint."""
        blocks = triton.cdiv(max_seq_len, self.kernel_page_size)
        constraint = TRTLLM_BLOCK_CONSTRAINT // self.kernel_page_size
        if blocks % constraint != 0:
            blocks = triton.cdiv(blocks, constraint) * constraint
        return blocks

    def _create_block_kv_indices(
        self,
        batch_size: int,
        max_blocks: int,
        page_table: torch.Tensor,
        block_kv_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the page table from the batch-ordered placeholder table.

        Only the idle/warmup path before the backend binds to the cache
        contract reaches this; ``page_table`` is batch-ordered (row i == batch
        position i).
        """
        if block_kv_indices is None:
            block_kv_indices = torch.zeros(
                (batch_size, max_blocks), dtype=torch.int32, device=self.device
            )

        copy_len = min(max_blocks, page_table.shape[1])

        block_kv_indices[:batch_size, :copy_len] = page_table[:batch_size, :copy_len]

        return block_kv_indices

    # ---- Paged cache full-attention-table metadata ----

    def _resolve_full_history_table(
        self, cache_metadata, forward_batch, bs: int
    ) -> torch.Tensor:
        """This forward's full-attention table in this backend's kernel pages.

        Host-side structural validation only (no GPU sync): operation
        freshness, row coverage, and kernel-page divisibility all check inside
        ``kernel_table``.
        """
        table = cache_metadata.kernel_table(
            kernel_page_size=self.kernel_page_size,
            max_pages=self._calc_padded_blocks(self.max_context_len),
            active_forward_op=forward_batch,
        )
        if table.shape[0] < bs:
            raise RuntimeError(
                f"full-attention table has {table.shape[0]} rows but the "
                f"batch has {bs} requests"
            )
        return table

    def _maybe_debug_check_write_pages(self, pages: torch.Tensor) -> None:
        """TOKENSPEED_CACHE_DEBUG=1 (GPU sync): latent writes must never land
        in the null page 0 or a -1 hole."""
        if not cache_debug_enabled() or pages.numel() == 0:
            return
        if not bool((pages > 0).all().item()):
            raise RuntimeError(
                "MLA write location resolves to the null page 0 or a " "-1 table hole"
            )

    def _cache_decode_out_cache_loc(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        group_table: torch.Tensor,
        out: torch.Tensor | None = None,
        q_len_per_req: int = 1,
    ) -> torch.Tensor:
        """Absolute latent write locations per batch row.

        Plain decode writes one location (position ``seq-1``); speculative
        target-verify writes ``q_len_per_req`` trailing positions
        (``seq-q_len .. seq-1``), flattened request-major to match the query
        layout. ``out`` (CUDA-graph replay): write the locations in place into
        the persistent buffer the graph recorded — same ``data_ptr`` — instead
        of allocating a fresh tensor. No host sync on either path.
        """
        page_size = self.kernel_page_size
        last = (seq_lens[:bs].to(torch.int64) - 1).clamp_min(0)
        if q_len_per_req == 1:
            pos = last.unsqueeze(1)
        else:
            steps = torch.arange(
                1 - q_len_per_req, 1, device=seq_lens.device, dtype=torch.int64
            )
            pos = (last.unsqueeze(1) + steps).clamp_min(0)  # [bs, q_len]
        page_idx = torch.div(pos, page_size, rounding_mode="floor")
        pages = group_table[:bs].gather(1, page_idx)
        self._maybe_debug_check_write_pages(pages)
        locs = (
            pages.clamp_min(0).to(torch.int64) * page_size + (pos % page_size)
        ).reshape(-1)
        if out is not None:
            out[: bs * q_len_per_req].copy_(locs)
            return out
        return locs

    def _extend_out_cache_loc(
        self,
        group_table: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Absolute latent write locations for extend tokens.

        Positions ``[prefix_len, seq_len)`` per request, flattened in q/k/v
        token order. Bounds come from the CPU length mirrors — no GPU sync.
        """
        page_size = self.kernel_page_size
        device = group_table.device
        chunks: list[torch.Tensor] = []
        pages_for_check: list[torch.Tensor] = []
        for row, (start, num_new) in enumerate(
            zip(extend_prefix_lens_cpu.tolist(), extend_seq_lens_cpu.tolist())
        ):
            start = int(start)
            num_new = int(num_new)
            if num_new <= 0:
                continue
            max_col = (start + num_new - 1) // page_size
            if max_col >= group_table.shape[1]:
                raise RuntimeError(
                    "extend write locations out of table bounds: "
                    f"table {tuple(group_table.shape)} req={row} prefix={start} "
                    f"new={num_new} page_size={page_size} needs "
                    f"column {max_col}"
                )
            pos = torch.arange(start, start + num_new, dtype=torch.int64, device=device)
            pages = group_table[row].gather(0, pos // page_size)
            pages_for_check.append(pages)
            chunks.append(pages.to(torch.int64) * page_size + pos % page_size)
        if not chunks:
            return torch.empty(0, dtype=torch.int64, device=device)
        if cache_debug_enabled():
            self._maybe_debug_check_write_pages(torch.cat(pages_for_check))
        return torch.cat(chunks)

    # ---- Metadata initialization ----

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        page_table: torch.Tensor,
        seq_lens_cpu: torch.Tensor | None = None,
        **kwargs,
    ):
        cache_metadata = kwargs.pop("cache_metadata", None)
        forward_batch = kwargs.pop("forward_batch", None)
        group_table = None
        if cache_metadata is not None and self.is_draft:
            # The MTP draft runs on its own classic paged pool; the target's
            # Cache metadata rides the shared forward kwargs; ignore it here.
            cache_metadata = None
            forward_batch = None
        if cache_metadata is not None:
            self._cache_groups_bound = True
            group_table = self._resolve_full_history_table(
                cache_metadata, forward_batch, bs
            )
            if cache_debug_enabled():
                cache_metadata.validate_live_pages(
                    seq_lens[:bs], active_forward_op=forward_batch
                )
        elif self._cache_groups_bound and bs > 0 and not forward_mode.is_idle():
            # Missing Paged cache metadata must never select the legacy page_table
            # path after the backend is bound to the contract.
            raise RuntimeError(
                "tokenspeed_mla is bound to the Cache contract but received "
                "no paged cache metadata; refusing the legacy page_table path"
            )

        if forward_mode.is_extend_or_mixed():
            self._init_prefill_metadata(
                seq_lens[:num_extends],
                req_pool_indices=req_pool_indices[:num_extends],
                page_table=page_table,
                extend_prefix_lens=kwargs.pop("extend_prefix_lens"),
                extend_prefix_lens_cpu=kwargs.pop("extend_prefix_lens_cpu"),
                extend_seq_lens=kwargs.pop("extend_seq_lens"),
                extend_seq_lens_cpu=kwargs.pop("extend_seq_lens_cpu"),
                group_table=group_table,
            )
        # Drafter steps 1..N are pure DECODE on full bs regardless of target
        # mode, so under is_draft we also fill decode_metadata under EXTEND
        # so the multi-step loop has metadata. The wrapper pre-writes
        # draft_seq_lens before calling here so `seq_lens` aliases the
        # drafter's live buffer.
        if (
            forward_mode.is_decode()
            or forward_mode.is_mixed()
            or (forward_mode.is_extend() and self.is_draft)
        ):
            # Target-verify decodes q_len tokens per request; the cache write
            # locations must cover every one of them.
            verify_q_len = (
                self.spec_num_tokens
                if (
                    self.spec_num_tokens > 1
                    and not self.is_draft
                    and (forward_mode.is_decode() or forward_mode.is_mixed())
                )
                else 1
            )
            self._init_decode_metadata(
                bs,
                num_extends,
                req_pool_indices,
                seq_lens,
                page_table,
                group_table=group_table,
                q_len_per_req=verify_q_len,
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

    def select_out_cache_loc(self, layer, out_cache_loc, forward_mode=None):
        """Paged cache write-location hook for model-owned latent KV writes.

        Forwards without grouped cache metadata return caller locations untouched;
        byte-for-byte the base-class behavior, so DeepSeek-style MLA models
        once cache metadata is bound, the group-derived
        locations stored in the current forward's metadata are authoritative
        and the caller's page_table-derived locations are discarded.
        """
        if not self._cache_groups_bound:
            return out_cache_loc
        if forward_mode is None or forward_mode.is_idle():
            return out_cache_loc
        if forward_mode.is_decode():
            metadata = self.forward_decode_metadata
            if metadata is None or metadata.group_out_cache_loc is None:
                raise RuntimeError(
                    "MLA decode write locations are missing; "
                    "init_forward_metadata must run with paged cache metadata"
                )
            locs = metadata.group_out_cache_loc[
                metadata.num_extends * metadata.group_q_len_per_req :
            ]
        else:
            metadata = self.forward_prefill_metadata
            if metadata is None or metadata.group_out_cache_loc is None:
                raise RuntimeError(
                    "MLA prefill write locations are missing; "
                    "init_forward_metadata must run with paged cache metadata"
                )
            locs = metadata.group_out_cache_loc
        if out_cache_loc is not None and locs.shape[0] != out_cache_loc.shape[0]:
            raise RuntimeError(
                f"MLA write locations cover {locs.shape[0]} tokens but "
                f"the caller provided {out_cache_loc.shape[0]}"
            )
        return locs

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
        max_blocks = self._calc_padded_blocks(self.max_context_len)
        if group_table is not None:
            # Paged cache path: kernel pages and write locations both derive from
            # the full-attention table; page_table is never consulted.
            block_kv_indices = self._create_block_kv_indices(
                bs, max_blocks, group_table
            )
            group_out_cache_loc = self._cache_decode_out_cache_loc(
                bs,
                seq_lens,
                group_table,
                q_len_per_req=q_len_per_req,
            )
        else:
            block_kv_indices = self._create_block_kv_indices(bs, max_blocks, page_table)
            group_out_cache_loc = None

        self.forward_decode_metadata = CuteDSLMLADecodeMetadata(
            block_kv_indices=block_kv_indices,
            max_seq_len_k=self.max_context_len,
            seq_lens_k=seq_lens,
            num_extends=num_extends,
            group_out_cache_loc=group_out_cache_loc,
            group_q_len_per_req=q_len_per_req,
        )

    def _init_prefill_metadata(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor | None = None,
        page_table: torch.Tensor | None = None,
        extend_prefix_lens: torch.Tensor | None = None,
        extend_prefix_lens_cpu: torch.Tensor | None = None,
        extend_seq_lens: torch.Tensor | None = None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
        group_table: torch.Tensor | None = None,
    ):
        # Worst-case bound to avoid GPU->CPU sync from seq_lens.max().item().
        # TODO: track a loose CPU upper bound (advance by chunked_prefill_size /
        # accept_lengths.max(); correct when accurate values land) for tighter
        # kernel-grid sizing without syncing.
        max_seq_len = self.max_context_len
        cum_seq_lens = torch.zeros(
            len(seq_lens) + 1, dtype=torch.int32, device=seq_lens.device
        )
        torch.cumsum(seq_lens, dim=0, out=cum_seq_lens[1:])

        assert (
            seq_lens.dtype == torch.int32
        ), f"seq_lens must be int32, got {seq_lens.dtype}"
        num_extends = extend_seq_lens.shape[0]
        if group_table is not None:
            group_out_cache_loc = self._extend_out_cache_loc(
                group_table[:num_extends],
                extend_prefix_lens_cpu,
                extend_seq_lens_cpu,
            )
        else:
            group_out_cache_loc = None
        self.forward_prefill_metadata = CuteDSLMLAPrefillMetadata(
            max_seq_len=max_seq_len,
            cum_seq_lens=cum_seq_lens,
            seq_lens=seq_lens,
            group_out_cache_loc=group_out_cache_loc,
        )
        cum_extend_seq_lens = torch.zeros(
            num_extends + 1, device=self.device, dtype=torch.int32
        )
        torch.cumsum(extend_seq_lens, dim=0, out=cum_extend_seq_lens[1:])
        max_extend_seq_len = extend_seq_lens_cpu.max().item()
        if group_table is not None:
            # Paged cache path: chunked history reads gather from the
            # full-attention table (rows are batch-ordered, prefill rows
            # first) in kernel pages.
            chunk_page_table = group_table[:num_extends]
            chunk_req_pool_indices = torch.arange(
                num_extends, dtype=torch.int64, device=group_table.device
            )
            chunk_page_size = self.kernel_page_size
        else:
            # Idle/warmup placeholder: page_table is batch-ordered (row i ==
            # batch position i), so identity row indices apply.
            chunk_page_table = page_table
            chunk_req_pool_indices = torch.arange(
                num_extends, dtype=torch.int64, device=page_table.device
            )
            chunk_page_size = self.kernel_page_size
        (
            chunked_loop_num,
            chunk_kv_indices_list,
            chunked_seq_len,
            cu_chunked_seq_len,
            max_chunk_len_per_loop,
        ) = build_chunked_prefill_metadata_arrays(
            extend_prefix_lens,
            extend_prefix_lens_cpu,
            chunk_page_table,
            chunk_req_pool_indices,
            chunk_page_size,
        )
        self.chunked_prefill_metadata = TRTLLMMLAChunkedPrefillMetadata(
            extend_prefix_lens=extend_prefix_lens,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            req_pool_indices=req_pool_indices,
            cum_extend_seq_lens=cum_extend_seq_lens,
            max_extend_seq_len=max_extend_seq_len,
            chunked_loop_num=chunked_loop_num,
            chunk_kv_indices_list=chunk_kv_indices_list,
            chunked_seq_len=chunked_seq_len,
            cu_chunked_seq_len=cu_chunked_seq_len,
            max_chunk_len_per_loop=max_chunk_len_per_loop,
        )

    # ---- CUDA Graph ----

    def init_cuda_graph_state(self, max_bs: int):
        # Own the cache-seqlens buffer; replay copies the live lengths in, so
        # graph state does not depend on the controller mutating a shared tensor.
        self.cuda_graph_seq_lens_buf = torch.zeros(
            max_bs, dtype=torch.int32, device=self.device
        )
        max_blocks = self._calc_padded_blocks(self.max_context_len)
        self.decode_cuda_graph_kv_indices = torch.zeros(
            (max_bs, max_blocks), dtype=torch.int32, device=self.device
        )
        if self._cache_contract_bound:
            # The Paged cache decode graph uses one absolute latent write slot per
            # row, refreshed in place from the current full-attention table.
            # Allocate the stable-address buffer outside the graph pool, like
            # the KV indices.
            self.decode_cuda_graph_group_out_cache_loc = torch.zeros(
                max_bs * max(1, self.spec_num_tokens),
                dtype=torch.int64,
                device=self.device,
            )
        else:
            self.decode_cuda_graph_group_out_cache_loc = None

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        cache_group_ids: tuple[str, ...] = (),
        **kwargs,
    ):
        # Structural gate: the target (contract always marked by the registry)
        # takes the Paged cache capture path; the MTP draft, whose
        # mark_cache_contract deliberately early-returns, keeps the
        # batch-ordered non-cache path.
        uses_cache_groups = bool(cache_group_ids) or self._cache_contract_bound
        if uses_cache_groups and self.is_draft:
            raise NotImplementedError(
                "tokenspeed_mla draft worker does not take the Paged cache path"
            )
        if forward_mode.is_extend_or_mixed():
            raise NotImplementedError(
                f"tokenspeed_mla CUDA graph capture not supported for {forward_mode}"
            )

        max_blocks = self._calc_padded_blocks(self.max_context_len)
        block_kv_indices = self.decode_cuda_graph_kv_indices[:bs, :max_blocks]

        if uses_cache_groups:
            # The captured Paged cache graph reads the kernel page table and latent
            # write locations from persistent buffers; replay refreshes both
            # in place from the current full-attention table. Latch _cache_groups_bound
            # so the recorded forward_decode takes the cache write-location
            # branch (select_out_cache_loc), matching the eager path.
            self._cache_groups_bound = True
            if self.decode_cuda_graph_group_out_cache_loc is None:
                raise RuntimeError(
                    "tokenspeed_mla Paged cache graph capture: the cache write-location "
                    "buffer is not allocated; init_cuda_graph_state ran before "
                    "the backend was marked as the Cache contract sub-backend"
                )
            # Placeholders resolve to the null page 0 until the first replay.
            block_kv_indices.zero_()
            capture_q_len = (
                self.spec_num_tokens
                if (self.spec_num_tokens > 1 and not self.is_draft)
                else 1
            )
            group_out_cache_loc = self.decode_cuda_graph_group_out_cache_loc[
                : bs * capture_q_len
            ]
            group_out_cache_loc.zero_()
            metadata = CuteDSLMLADecodeMetadata(
                block_kv_indices=block_kv_indices,
                max_seq_len_k=self.max_context_len,
                seq_lens_k=self.cuda_graph_seq_lens_buf[:bs],
                num_extends=0,
                group_out_cache_loc=group_out_cache_loc,
                group_q_len_per_req=capture_q_len,
            )
        else:
            metadata = CuteDSLMLADecodeMetadata(
                block_kv_indices=block_kv_indices,
                max_seq_len_k=self.max_context_len,
                seq_lens_k=self.cuda_graph_seq_lens_buf[:bs],
                num_extends=0,
            )
        # Seed the owned buffer: the capture run reads it before replay.
        metadata.seq_lens_k.copy_(seq_lens[:bs])
        self.decode_cuda_graph_metadata[bs] = metadata
        self.forward_decode_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = None,
        page_table: torch.Tensor = None,
        **kwargs,
    ):
        cache_metadata = kwargs.get("cache_metadata")
        if forward_mode is not None and forward_mode.is_extend_or_mixed():
            raise NotImplementedError(
                f"tokenspeed_mla CUDA graph replay not supported for {forward_mode}"
            )

        metadata = self.decode_cuda_graph_metadata[bs]
        # The captured metadata is the source of truth: a cache write-location buffer
        # exists iff this bs was captured on the Paged cache path.
        uses_cache_groups = metadata.group_out_cache_loc is not None
        if uses_cache_groups and self.is_draft:
            raise NotImplementedError(
                "tokenspeed_mla draft worker does not take the Paged cache path"
            )

        # Copy the live cache lengths into our own buffer (metadata.seq_lens_k
        # views it) on both paths; the grouped helper only refreshes tables.
        self.cuda_graph_seq_lens_buf[:bs].copy_(seq_lens[:bs])

        if uses_cache_groups:
            self._replay_refresh_decode(
                bs,
                seq_lens,
                metadata,
                cache_metadata,
                kwargs.get("forward_batch"),
            )
            self.forward_decode_metadata = metadata
            return

        # Block indices are refreshed separately; when the block table is
        # aliased to a peer backend, that peer's replay already populated it.
        if page_table is not None and not self._page_table_aliased:
            self._create_block_kv_indices(
                bs,
                metadata.block_kv_indices.shape[1],
                page_table,
                metadata.block_kv_indices,
            )

        self.forward_decode_metadata = metadata

    def _replay_refresh_decode(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        metadata: "CuteDSLMLADecodeMetadata",
        cache_metadata,
        forward_batch,
    ) -> None:
        """Refresh captured decode buffers from this replay's
        full-attention table.

        ``bs`` is the padded batch size. The operation-bound metadata carries
        one table row per REAL request; the trailing padded rows have no
        request and resolve to the null page 0 so the captured kernels
        dereference the permanently-zero dummy page.
        The refresh writes into the SAME persistent buffers the graph
        recorded (``metadata.block_kv_indices`` / ``metadata.group_out_cache_loc``)
        — stable addresses, no reallocation, no host sync.
        """
        if metadata.group_out_cache_loc is None:
            raise RuntimeError(
                "tokenspeed_mla Paged cache graph replay: captured decode metadata "
                "has no cache write-location buffer"
            )
        # Target verify graphs are captured with q_len write locations per
        # request (request-major flattened); mirror the capture-time width.
        replay_q_len = metadata.group_q_len_per_req
        max_blocks = metadata.block_kv_indices.shape[1]
        real_bs = 0
        if cache_metadata is not None:
            # Row coverage is not required here (bs is the padded batch size
            # and the table carries one row per REAL request), so pass 0.
            table = self._resolve_full_history_table(cache_metadata, forward_batch, 0)
            real_bs = min(int(table.shape[0]), bs)
            if real_bs > 0:
                self._create_block_kv_indices(
                    real_bs,
                    max_blocks,
                    table,
                    metadata.block_kv_indices,
                )
                self._cache_decode_out_cache_loc(
                    real_bs,
                    seq_lens,
                    table,
                    out=metadata.group_out_cache_loc[: real_bs * replay_q_len],
                    q_len_per_req=replay_q_len,
                )
        # Padded (and bs==0 idle) rows: null page 0 for both the kernel page
        # table and the write location, so they never touch a live page. Note
        # these rows DO scribble latent slot 0 of page 0 (bytes [0, 576) of
        # each aliased slab); that range stays inside the aliased KDA
        # conv_state component, which is never consumed from page 0 — the
        # recurrent_state bytes read for fresh requests sit above it and
        # remain zero.
        if real_bs < bs:
            metadata.block_kv_indices[real_bs:bs].zero_()
            metadata.group_out_cache_loc[
                real_bs * replay_q_len : bs * replay_q_len
            ].zero_()

    # ---- Forward: Decode ----

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
    ) -> torch.Tensor:
        # q is whole Q [T, H, head_dim]; k is whole latent [T, 1, head_dim].
        if save_kv_cache:
            assert k is not None
            if self._cache_groups_bound:
                # Paged cache: write locations derive from the full-attention
                # table, never from the caller's page_table-derived locs.
                out_cache_loc = self.select_out_cache_loc(
                    layer, out_cache_loc, ForwardMode.DECODE
                )
            token_to_kv_pool.set_mla_kv_buffer(
                layer,
                out_cache_loc,
                k[..., : self.kv_lora_rank],
                k[..., self.kv_lora_rank :],
            )

        metadata = self.forward_decode_metadata
        num_extends = metadata.num_extends
        q_len_per_req = q.shape[0] // bs
        query = q.view(bs, q_len_per_req, layer.tp_q_head_num, layer.head_dim)

        softmax_scale = layer.scaling
        if self.data_type == torch.float8_e4m3fn:
            query = query.to(self.data_type)
            k_scale = (
                layer.k_scale_float
                if getattr(layer, "k_scale_float", None) is not None
                else 1.0
            )
            softmax_scale = k_scale * layer.scaling

        # Prepare KV cache: [num_pages, page_size, kv_cache_dim] (3D for CuteDSL)
        k_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)
        if self.data_type != k_cache.dtype:
            k_cache = k_cache.to(self.data_type)
        kv_cache = k_cache.view(-1, self.kernel_page_size, self.kv_cache_dim)

        if not CuteDSLMLABackend._logged_decode:
            logger.info(
                "CuteDSL MLA decode kernel invoked (tokenspeed_mla_decode, query_dtype=%s, kv_dtype=%s)",
                query.dtype,
                kv_cache.dtype,
            )
            CuteDSLMLABackend._logged_decode = True

        self.cutedsl_workspace = self._cutedsl_workspace(query.shape[1])

        raw_out = tokenspeed_mla_decode(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=self.cutedsl_workspace,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=metadata.block_kv_indices[num_extends:],
            seq_lens=metadata.seq_lens_k[num_extends:],
            max_seq_len=metadata.max_seq_len_k,
            softmax_scale=softmax_scale,
            enable_pdl=pdl_enabled(),
        )

        return raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    # ---- Forward: Extend/Prefill ----

    def forward_extend_chunked(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
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
        if causal:
            step_counter = getattr(self, "step_counter", None)
            if step_counter is not None:
                step_counter.record_cache()

        head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # The CuteDSL FMHA prefill kernel assumes packed (contiguous) Q/K/V; its
        # TMA descriptors ignore input strides. On the BF16 (NoPE, e.g. Kimi-K3)
        # path V arrives as a non-contiguous slice ``kv[..., qk_nope:]`` of the
        # fused kv_b_proj output (its stride skips the interleaved k_nope block),
        # so without this the kernel reads interleaved garbage and produces an
        # attention output orthogonal to the correct result. Force contiguity on
        # all three; Q/K are already contiguous so ``.contiguous()`` is a no-op.
        q = q.reshape(-1, self.num_local_heads, head_dim).contiguous()
        k = k.reshape(-1, self.num_local_heads, head_dim).contiguous()
        v = v.reshape(-1, self.num_local_heads, self.v_head_dim).contiguous()

        # CuteDSL FMHA MLA: if Q is FP8, ensure K/V match. `.to()` is a no-op
        # when the source dtype already matches.
        if q.dtype == torch.float8_e4m3fn:
            k = k.to(torch.float8_e4m3fn)
            v = v.to(torch.float8_e4m3fn)

        if not CuteDSLMLABackend._logged_prefill:
            logger.info(
                "CuteDSL MLA prefill kernel invoked (tokenspeed_mla_prefill, "
                f"q_dtype={q.dtype})"
            )
            CuteDSLMLABackend._logged_prefill = True

        result = tokenspeed_mla_prefill(
            query=q,
            key=k,
            value=v,
            seq_lens=seq_lens,
            cum_seq_lens=cum_seq_lens_kv,
            max_seq_len=max_kv_len,
            batch_size=batch_size,
            softmax_scale=scaling,
            is_causal=causal,
            return_lse=True,
            cum_seq_lens_q=cum_seq_lens_q,
            max_seq_len_q=max_q_len,
            enable_pdl=pdl_enabled(),
            out=out,
        )

        if isinstance(result, tuple):
            out, lse = result[0], result[1]
        else:
            out, lse = result, None

        if out.dtype != self.q_data_type:
            out = out.to(self.q_data_type)

        return out, lse


register_backend("tokenspeed_mla", {AttentionArch.MLA}, CuteDSLMLABackend)
