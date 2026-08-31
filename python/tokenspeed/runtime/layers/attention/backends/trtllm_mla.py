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
MLA attention backend for TokenSpeed scheduling.

Uses fused kernels optimized for SM100 (Blackwell) GPUs.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import triton
from tokenspeed_kernel.ops.attention.flashinfer import (
    trtllm_batch_decode_with_kv_cache_mla,
    trtllm_ragged_attention_deepseek,
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
    TRTLLM_MLA_DEFAULT_PAGE_SIZE,
    TRTLLM_MLA_SUPPORTED_PAGE_SIZES,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    cache_debug_enabled,
)
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.utils.env import envs

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention

logger = logging.getLogger(__name__)

# Block constraint from flashinfer: block_num % (128 / page_size) == 0
TRTLLM_BLOCK_CONSTRAINT = 128

# Shared workspace buffer for fused kernels, zero-initialized. NOT eligible
# for the WorkspacePool: zero-init is required for the kernel's internal
# semaphore mechanism, i.e. the content carries state between launches, and
# the pool's shared block hands the same bytes to every consumer. Size in MB
# via TOKENSPEED_WORKSPACE_TRTLLM_MLA_MB.
_trtllm_workspace_buffer = None


def get_trtllm_workspace_buffer(device):
    """Get or create the shared fused-kernel workspace buffer."""
    global _trtllm_workspace_buffer
    if _trtllm_workspace_buffer is None:
        _trtllm_workspace_buffer = torch.zeros(
            envs.TOKENSPEED_WORKSPACE_TRTLLM_MLA_MB.get() * (1 << 20),
            dtype=torch.uint8,
            device=device,
        )
    return _trtllm_workspace_buffer


@dataclass
class TRTLLMMLAPrefillMetadata:
    max_seq_len: int
    cum_seq_lens: torch.Tensor
    seq_lens: torch.Tensor


@dataclass
class TRTLLMMLAChunkedPrefillMetadata:
    extend_prefix_lens: torch.Tensor
    extend_prefix_lens_cpu: torch.Tensor
    extend_seq_lens: torch.Tensor
    extend_seq_lens_cpu: torch.Tensor
    req_pool_indices: torch.Tensor
    cum_extend_seq_lens: torch.Tensor  # cumsum prefix-padded, sized num_extends+1
    max_extend_seq_len: int
    # Per-prefix-chunk arrays for non-causal cross-attention (built once per
    # iteration in _init_prefill_metadata, indexed by loop_idx in the model).
    chunked_loop_num: int
    chunk_kv_indices_list: list  # List[torch.Tensor], one per loop_idx
    chunked_seq_len: torch.Tensor  # (chunked_loop_num, num_extends) int32 GPU
    cu_chunked_seq_len: torch.Tensor  # (chunked_loop_num, num_extends+1) int32 GPU
    max_chunk_len_per_loop: list  # List[int], one per loop_idx
    # Per-request batch-ordered kernel page table. Populated only by the DSA
    # backend for sparse-prefill top-k; plain MLA leaves it None.
    page_table: torch.Tensor | None = None


@dataclass
class TRTLLMMLADecodeMetadata:
    num_extends: int = 0
    block_kv_indices: torch.Tensor | None = None
    max_seq_len_k: int | None = None
    seq_lens_k: torch.Tensor | None = None
    # Cache-group path only: absolute latent write locations, request-major, with
    # ``group_q_len_per_req`` entries per row (1 outside target verify).
    group_out_cache_loc: torch.Tensor | None = None
    group_q_len_per_req: int = 1


class TRTLLMMLABackend(MlaCacheGroupMixin, AttentionBackend):
    """trtllm_mla attention backend using fused kernels."""

    draft_seq_lens_attr: str = "cuda_graph_seq_lens_buf"

    def __init__(self, config: MLAConfig):
        super().__init__(config)

        self.max_context_len = config.context_len
        self.kernel_page_size = (
            config.kernel_page_size
            if config.kernel_page_size is not None
            else TRTLLM_MLA_DEFAULT_PAGE_SIZE
        )
        # Cache-group (LCM) state. The trtllm kernel walks pages at page_size,
        # padded to the fused-kernel block constraint (see _calc_padded_blocks).
        self._cache_groups_bound = False
        self._cache_contract_bound = False
        self.max_num_pages = self._calc_padded_blocks(config.context_len)

        # MLA dimensions
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_cache_dim = config.kv_cache_dim
        self.scaling = config.scaling
        self.data_type = config.kv_cache_dtype
        self.q_data_type = config.dtype
        self.draft_block_decode = config.draft_block_decode

        # Workspace zero-initialized for the fused kernel semaphore.
        self.trtllm_workspace = get_trtllm_workspace_buffer(config.device)

        # Validate page_size
        if self.kernel_page_size not in TRTLLM_MLA_SUPPORTED_PAGE_SIZES:
            raise ValueError(
                f"trtllm_mla backend requires page_size 32 or 64, got {self.kernel_page_size}"
            )

        self.num_local_heads = config.num_attention_heads // config.attn_tp_size

        # Metadata
        self.forward_decode_metadata: TRTLLMMLADecodeMetadata | None = None
        self.forward_prefill_metadata: TRTLLMMLAPrefillMetadata | None = None
        self.decode_cuda_graph_metadata: dict[int, TRTLLMMLADecodeMetadata] = {}
        self.decode_cuda_graph_kv_indices = None
        self.chunked_prefill_metadata: TRTLLMMLAChunkedPrefillMetadata | None = None

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
        """Copy a batch-ordered kernel page table into TRTLLM metadata.

        ``page_table`` is batch-ordered (row i == batch position i) and must
        already be in this backend's kernel-page units — callers expand raw
        scheduler tables through ``_expand_history_table`` first (warmup
        placeholders are empty/dummy and copy as-is).
        """
        if block_kv_indices is None:
            block_kv_indices = torch.zeros(
                (batch_size, max_blocks), dtype=torch.int32, device=self.device
            )
        else:
            block_kv_indices[:batch_size].zero_()

        copy_len = min(max_blocks, page_table.shape[1])

        # Pages beyond actual seq_len are 0 (from the table init); the kernel
        # uses seq_lens to bound access so these padding entries are never read.
        block_kv_indices[:batch_size, :copy_len] = page_table[:batch_size, :copy_len]

        return block_kv_indices

    # ---- Metadata initialization ----

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        page_table: torch.Tensor,
        **kwargs,
    ):
        kwargs.pop("cache_metadata", None)
        kwargs.pop("forward_batch", None)
        group_table = self._resolve_full_history_table(
            kwargs.pop("block_tables", None), bs
        )
        if group_table is not None:
            self._cache_groups_bound = True
        elif self.is_draft and self._cache_contract_bound and bs > 0:
            # The drafter drives this backend directly with no group tables;
            # the batch-ordered draft page table (row i is batch position i)
            # carries raw scheduler pages, expanded by the decode refresh
            # like any group table.
            self._cache_groups_bound = True
        if not (forward_mode.is_extend_or_mixed() or forward_mode.is_idle()):
            raise RuntimeError(
                "trtllm_mla decode metadata goes through refresh_decode_metadata; "
                f"init_forward_metadata only serves extend/mixed ({forward_mode})"
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
        # For target_verify, the draft tokens have already been written to the KV
        # cache. The seq_lens passed in should already reflect the full context.
        # Use max_context_len to avoid GPU->CPU sync from seq_lens.max().item()
        max_blocks = self._calc_padded_blocks(self.max_context_len)

        group_out_cache_loc = None
        if group_table is not None:
            # The table is already in kernel pages; copy into a buffer padded
            # to the fused-kernel block constraint. DSA's sparse top-k slots
            # are mapped through this same table, so it must come from the
            # group geometry rather than page_table.
            block_kv_indices = self._create_block_kv_indices(
                bs, max_blocks, group_table
            )
            group_out_cache_loc = self._cache_decode_out_cache_loc(
                group_table,
                seq_lens,
                batch_size=bs,
                validate_pages=cache_debug_enabled(),
                q_len_per_req=q_len_per_req,
            )
        else:
            block_kv_indices = self._create_block_kv_indices(bs, max_blocks, page_table)

        assert (
            seq_lens.dtype == torch.int32
        ), f"seq_lens must be int32, got {seq_lens.dtype}"
        self.forward_decode_metadata = TRTLLMMLADecodeMetadata(
            num_extends=num_extends,
            block_kv_indices=block_kv_indices,
            max_seq_len_k=self.max_context_len,
            seq_lens_k=seq_lens,
            group_out_cache_loc=group_out_cache_loc,
            group_q_len_per_req=q_len_per_req,
        )

    def select_out_cache_loc(self, layer, out_cache_loc, forward_mode=None):
        """Group-derived latent write location on the cache-group path.

        Identity when not cache-group bound or idle. A draft owns its per-step
        locations (it passes ``num_extends == bs`` by its own convention), so it
        keeps the caller's. Target decode writes the whole verify window.
        """
        if (
            not self._cache_groups_bound
            or forward_mode is None
            or forward_mode.is_idle()
            or self.is_draft
        ):
            return out_cache_loc
        if not forward_mode.is_decode():
            return out_cache_loc
        metadata = self.forward_decode_metadata
        if metadata is None or metadata.group_out_cache_loc is None:
            return out_cache_loc
        locs = metadata.group_out_cache_loc[
            metadata.num_extends * metadata.group_q_len_per_req :
        ]
        if out_cache_loc is not None and locs.shape[0] != out_cache_loc.shape[0]:
            raise RuntimeError(
                f"trtllm_mla write locations cover {locs.shape[0]} tokens but "
                f"the caller provided {out_cache_loc.shape[0]}"
            )
        return locs

    def _init_prefill_metadata(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor | None = None,
        page_table: torch.Tensor | None = None,
        extend_prefix_lens: torch.Tensor | None = None,
        extend_prefix_lens_cpu: torch.Tensor | None = None,
        extend_seq_lens: torch.Tensor | None = None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
    ):
        max_seq_len = self.max_context_len
        cum_seq_lens = torch.zeros(
            len(seq_lens) + 1, dtype=torch.int32, device=seq_lens.device
        )
        torch.cumsum(seq_lens, dim=0, out=cum_seq_lens[1:])

        assert (
            seq_lens.dtype == torch.int32
        ), f"seq_lens must be int32, got {seq_lens.dtype}"
        self.forward_prefill_metadata = TRTLLMMLAPrefillMetadata(
            max_seq_len=max_seq_len,
            cum_seq_lens=cum_seq_lens,
            seq_lens=seq_lens,
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
            page_table,
            # page_table is batch-ordered (row i == batch position i).
            torch.arange(
                extend_prefix_lens.shape[0],
                dtype=torch.int64,
                device=page_table.device,
            ),
            self.kernel_page_size,
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
        # Cache contract: persistent write-location buffer whose address
        # the captured graph records; replay refreshes it in place. Target
        # verify records spec_num_tokens locations per request.
        if self._cache_contract_bound:
            self.decode_cuda_graph_group_out_cache_loc = torch.zeros(
                max_bs * max(1, self.spec_num_tokens),
                dtype=torch.int64,
                device=self.device,
            )
        else:
            self.decode_cuda_graph_group_out_cache_loc = None

    def _decode_views(self, bs: int) -> TRTLLMMLADecodeMetadata:
        """Per-bs decode metadata views over the persistent buffers.

        One builder for capture and refresh; cached per bs — pointer-stable,
        no storage allocated.
        """
        metadata = self.decode_cuda_graph_metadata.get(bs)
        if metadata is not None:
            return metadata
        max_blocks = self._calc_padded_blocks(self.max_context_len)
        capture_q_len = self._graph_verify_q_len()
        group_out_cache_loc = None
        if self._cache_contract_bound:
            if self.decode_cuda_graph_group_out_cache_loc is None:
                raise RuntimeError(
                    "trtllm_mla cache-group buffer was not allocated; "
                    "mark_cache_contract must run before init_cuda_graph_state"
                )
            group_out_cache_loc = self.decode_cuda_graph_group_out_cache_loc[
                : bs * capture_q_len
            ]
        metadata = TRTLLMMLADecodeMetadata(
            num_extends=0,
            block_kv_indices=self.decode_cuda_graph_kv_indices[:bs, :max_blocks],
            max_seq_len_k=self.max_context_len,
            seq_lens_k=self.cuda_graph_seq_lens_buf[:bs],
            group_out_cache_loc=group_out_cache_loc,
            group_q_len_per_req=capture_q_len,
        )
        self.decode_cuda_graph_metadata[bs] = metadata
        return metadata

    # Capture is inherited (base default: bind_decode_views + idle refresh).

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
        if forward_mode is not None and forward_mode.is_extend_or_mixed():
            raise NotImplementedError(
                f"trtllm_mla decode refresh not supported for {forward_mode}"
            )

        metadata = self._decode_views(bs)
        # The cached view bakes num_extends=0; a mixed round's decode rows
        # start after the extend rows, so publish this round's split.
        metadata.num_extends = num_extends

        # Copy the live cache lengths into our own buffer (metadata.seq_lens_k
        # views it). Block indices are refreshed separately; when the block
        # table is aliased to a peer backend, that peer's refresh already
        # populated it with identical content.
        q_len = metadata.group_q_len_per_req
        # clamp_min(1) is the identity, so the verify clamp is unconditional.
        self.cuda_graph_seq_lens_buf[:bs].copy_(seq_lens[:bs].clamp_min(q_len))

        # The wrapper's per-group tables refresh the block table whenever
        # delivered — decode-only PD nodes included (they never run an extend
        # forward, so an extend-latched gate would leave the kernel on a
        # stale/zero block table instead of the transferred KV).
        group_table = self._resolve_full_history_table(kwargs.get("block_tables"), 0)
        if group_table is not None:
            self._cache_groups_bound = True
        elif (
            self.is_draft
            and self._cache_contract_bound
            and bs > 0
            and page_table is not None
        ):
            # Draft: the staged batch-ordered table carries raw scheduler
            # pages; expand into this backend's kernel pages.
            group_table = self._expand_history_table(page_table[:bs])
        if group_table is not None:
            # Live tables carry one row per REAL request; the idle replay's
            # synthesized placeholder rows are all dummies (actual_bs caps).
            real_bs = min(int(group_table.shape[0]), bs, actual_bs)
            if real_bs > 0 and not self._page_table_aliased:
                self._create_block_kv_indices(
                    real_bs,
                    metadata.block_kv_indices.shape[1],
                    group_table,
                    metadata.block_kv_indices,
                )
            if metadata.group_out_cache_loc is not None and real_bs > 0:
                self._cache_decode_out_cache_loc(
                    group_table,
                    self.cuda_graph_seq_lens_buf,
                    batch_size=real_bs,
                    validate_pages=cache_debug_enabled(),
                    out=metadata.group_out_cache_loc,
                    q_len_per_req=q_len,
                )
            # Padded rows resolve to the null page 0.
            metadata.block_kv_indices[real_bs:bs].zero_()
            if metadata.group_out_cache_loc is not None:
                metadata.group_out_cache_loc[real_bs * q_len : bs * q_len].zero_()
        elif page_table is not None and not self._page_table_aliased:
            self._create_block_kv_indices(
                bs,
                metadata.block_kv_indices.shape[1],
                page_table,
                metadata.block_kv_indices,
            )

        self.forward_decode_metadata = metadata

    def fill_block_decode_seq_lens(self, bs: int, block_seq_lens: torch.Tensor) -> None:
        """Publish block-end cache lengths inside a captured draft graph.

        Args:
            bs: Number of draft requests.
            block_seq_lens: Per-request lengths after writing the draft block.
        """
        if not self.draft_block_decode:
            raise RuntimeError("Block decode sequence lengths require DFLASH mode.")
        self.cuda_graph_seq_lens_buf[:bs].copy_(
            block_seq_lens[:bs].clamp(self.spec_num_tokens, self.max_context_len)
        )

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
            token_to_kv_pool.set_mla_kv_buffer(
                layer,
                out_cache_loc,
                k[..., : self.kv_lora_rank],
                k[..., self.kv_lora_rank :],
            )

        metadata = self.forward_decode_metadata
        # A block drafter describes only decode rows. Older callers used
        # num_extends=bs as an internal "whole block" convention; honoring it
        # here slices every page-table and sequence-length row away.
        num_extends = 0 if self.draft_block_decode else metadata.num_extends
        q_len_per_req = q.shape[0] // bs if bs > 0 else 1

        if q_len_per_req > 1 and self.is_draft:
            query = q.view(-1, layer.tp_q_head_num, layer.head_dim).unsqueeze(1)
            page_table = metadata.block_kv_indices[num_extends:].repeat_interleave(
                q_len_per_req, dim=0
            )
            base_lens = metadata.seq_lens_k[num_extends:].repeat_interleave(
                q_len_per_req
            )
            if self.draft_block_decode:
                # The whole latent block is written before attention, so every
                # query sees the same block-end length (non-causal block decode).
                seq_lens = base_lens
                max_seq_len = metadata.max_seq_len_k
            else:
                # Eagle/MTP catch-up: each successive token sees one more KV.
                offsets = torch.arange(
                    q_len_per_req, device=base_lens.device, dtype=base_lens.dtype
                ).repeat(bs)
                seq_lens = base_lens + offsets
                max_seq_len = metadata.max_seq_len_k + q_len_per_req
        else:
            # Plain decode (q_len=1) or bs-grouped multi-token decode.
            query = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
            page_table = metadata.block_kv_indices[num_extends:]
            seq_lens = metadata.seq_lens_k[num_extends:]
            max_seq_len = metadata.max_seq_len_k

        if self.data_type == torch.float8_e4m3fn:
            query = query.to(self.data_type)
            k_scale = (
                layer.k_scale_float
                if getattr(layer, "k_scale_float", None) is not None
                else 1.0
            )
            bmm1_scale = k_scale * layer.scaling
        else:
            bmm1_scale = layer.scaling

        k_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)
        if self.data_type != k_cache.dtype:
            k_cache = k_cache.to(self.data_type)
        kv_cache = k_cache.view(-1, self.kernel_page_size, self.kv_cache_dim).unsqueeze(
            1
        )

        raw_out = trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=self.trtllm_workspace,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=page_table,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=bmm1_scale,
        )

        return raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)

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
        if causal:
            step_counter = getattr(self, "step_counter", None)
            if step_counter is not None:
                step_counter.record_cache()

        head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        q = q.reshape(-1, self.num_local_heads, head_dim)
        k = k.reshape(-1, self.num_local_heads, head_dim)
        v = v.reshape(-1, self.num_local_heads, self.v_head_dim)

        # FP8 prefill: if Q is already FP8 (model decided to use FP8 prefill),
        # ensure K/V match. If Q is BF16, respect the model's decision.
        if q.dtype == torch.float8_e4m3fn:
            k = k.to(torch.float8_e4m3fn)
            v = v.to(torch.float8_e4m3fn)

        if out is None:
            # The ragged path does not support FP8 output.
            out_dtype = self.q_data_type
            if out_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                out_dtype = torch.bfloat16

            out = torch.empty(
                q.shape[0],
                q.shape[1],
                v.shape[2],
                device=q.device,
                dtype=out_dtype,
            )

        result = trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=self.trtllm_workspace,
            seq_lens=seq_lens,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            bmm1_scale=scaling,
            bmm2_scale=1.0,
            o_sf_scale=-1.0,
            batch_size=batch_size,
            window_left=-1,
            cum_seq_lens_q=cum_seq_lens_q,
            cum_seq_lens_kv=cum_seq_lens_kv,
            is_causal=causal,
            return_lse=True,
            out=out,
        )

        if isinstance(result, tuple):
            return result[0], result[1]
        return result, None


register_backend("trtllm_mla", {AttentionArch.MLA}, TRTLLMMLABackend)
