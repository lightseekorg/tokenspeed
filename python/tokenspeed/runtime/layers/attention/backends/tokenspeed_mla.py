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
CuteDSL MLA attention leaf for TokenSpeed scheduling.

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
from tokenspeed.runtime.layers.attention.backends.paged import (
    PagedAttentionBackend,
)
from tokenspeed.runtime.layers.attention.backends.trtllm_mla import (
    TRTLLM_BLOCK_CONSTRAINT,
    TRTLLMMLAChunkedPrefillMetadata,
)
from tokenspeed.runtime.layers.attention.chunk import (
    build_chunked_prefill_metadata_arrays,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.kernel_page_sizes import (
    TOKENSPEED_MLA_DEFAULT_PAGE_SIZE,
    TOKENSPEED_MLA_SUPPORTED_PAGE_SIZES,
)
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.utils.env import global_server_args_dict

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


@dataclass
class CuteDSLMLADecodeMetadata:
    num_extends: int = 0
    page_table: torch.Tensor | None = None
    max_seq_len_k: int | None = None
    seq_lens_k: torch.Tensor | None = None
    # Verify window width baked into the graph views (1 outside target verify).
    q_len_per_req: int = 1


class CuteDSLMLABackend(PagedAttentionBackend):
    """CuteDSL MLA leaf for Blackwell SM100 GPUs.

    Decode uses CuTe DSL JIT-compiled kernels via tokenspeed_mla_decode().
    Prefill uses CuTe DSL FMHA kernel via tokenspeed_mla_prefill().
    """

    default_kernel_page_size = TOKENSPEED_MLA_DEFAULT_PAGE_SIZE

    _logged_decode = False
    _logged_prefill = False

    def __init__(self, config: AttnConfig, spec: MLAConfig, *, kernel_page_size: int):
        if kernel_page_size not in TOKENSPEED_MLA_SUPPORTED_PAGE_SIZES:
            raise ValueError(
                f"tokenspeed_mla backend requires page_size 32 or 64, got {kernel_page_size}"
            )
        super().__init__(config, spec, kernel_page_size=kernel_page_size)

        # Block draft: rows expand per block position; see block_decode_active.
        self.draft_block_decode = bool(config.draft_block_decode)

        # MLA dimensions
        self.kv_lora_rank = spec.kv_lora_rank
        self.qk_nope_head_dim = spec.qk_nope_head_dim
        self.qk_rope_head_dim = spec.qk_rope_head_dim
        self.v_head_dim = spec.v_head_dim
        self.kv_cache_dim = spec.kv_cache_dim
        self.scaling = spec.scaling
        self.data_type = config.kv_cache_dtype
        self.q_data_type = config.dtype

        # Decode scratch comes from the shared WorkspacePool: the kernel's own
        # get_workspace_size formula is B*H*q_len*split_kv*(D+1)*acc_bytes with
        # B*split_kv <= num_SMs, giving the closed-form bound used in
        # _cutedsl_workspace. The content is partial decode accumulators,
        # consumed within each op and never zero-initialized, so sharing the
        # block is safe. Warm to the verify-path peak now: graph capture runs
        # the decode forward with the pool frozen.
        self._num_heads_per_tp = spec.num_attention_heads // spec.attn_tp_size
        self._workspace_pool = workspace_pool(config.device)
        self.cutedsl_workspace = self._cutedsl_workspace(
            max(_CUTEDSL_WARMUP_Q_LEN_FLOOR, self.spec_num_tokens or 1)
        )

        # Pre-compile prefill kernel variants so JIT doesn't run during serving.
        # The backend may be constructed once per attention layer (60x for
        # Kimi-K2.5), but `warmup_compile_prefill` is idempotent: each config
        # is only JIT'd once and cached in a module-global dict.
        # tokenspeed_mla requires --kv-cache-dtype fp8_e4m3, so tokenspeed's
        # FP8 prefill path (deepseek_v3.py `use_fp8_prefill`) is always on and
        # feeds fp8_e4m3fn q/k/v to the kernel — bf16 is unreachable here.
        d_qk = self.qk_nope_head_dim + self.qk_rope_head_dim
        warmup_compile_prefill(
            q_dtype=torch.float8_e4m3fn,
            d_qk=d_qk,
            d_v=self.v_head_dim,
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
        self.chunked_prefill_metadata: TRTLLMMLAChunkedPrefillMetadata | None = None
        self.page_table_buf: torch.Tensor | None = None
        self.seq_lens_buf: torch.Tensor | None = None
        self._decode_views_by_bs: dict[int, CuteDSLMLADecodeMetadata] = {}

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

    @property
    def max_num_pages(self) -> int:
        # Kernel page-table width, padded to the fused-kernel block constraint.
        return self._calc_padded_blocks(self.max_context_len)

    @max_num_pages.setter
    def max_num_pages(self, value: int) -> None:
        # The base constructor assigns the plain ceil-div width; this leaf
        # derives the padded width from context instead.
        del value

    def _calc_padded_blocks(self, max_seq_len: int) -> int:
        """Calculate block count padded to satisfy the fused-kernel constraint."""
        blocks = triton.cdiv(max_seq_len, self.kernel_page_size)
        constraint = TRTLLM_BLOCK_CONSTRAINT // self.kernel_page_size
        if blocks % constraint != 0:
            blocks = triton.cdiv(blocks, constraint) * constraint
        return blocks

    # ---- Metadata initialization ----

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        forward_mode: ForwardMode,
        *,
        extend_seq_lens: torch.Tensor | None = None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
        extend_prefix_lens: torch.Tensor | None = None,
        extend_prefix_lens_cpu: torch.Tensor | None = None,
        **kwargs,
    ):
        if not (forward_mode.is_extend_or_mixed() or forward_mode.is_idle()):
            raise RuntimeError(
                "tokenspeed_mla decode metadata goes through "
                "refresh_decode_metadata; init_forward_metadata only serves "
                f"extend/mixed ({forward_mode})"
            )
        if forward_mode.is_extend_or_mixed():
            self._init_prefill_metadata(
                seq_lens[:num_extends],
                page_table=page_table[:num_extends],
                extend_prefix_lens=extend_prefix_lens[:num_extends],
                extend_prefix_lens_cpu=extend_prefix_lens_cpu[:num_extends],
                extend_seq_lens=extend_seq_lens[:num_extends],
                extend_seq_lens_cpu=extend_seq_lens_cpu[:num_extends],
            )
        # Target mixed/idle batches carry decode rows whose metadata this
        # init must cover; the same in-place refresh serves them. A draft's
        # decode metadata instead comes from the wrapper's refresh after this
        # init (the unified draft contract).
        if forward_mode.is_idle() or (forward_mode.is_mixed() and not self.is_draft):
            self.refresh_decode_metadata(
                bs, bs, seq_lens, page_table, num_extends=num_extends
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

    def _init_prefill_metadata(
        self,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
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
        self.forward_prefill_metadata = CuteDSLMLAPrefillMetadata(
            max_seq_len=max_seq_len,
            cum_seq_lens=cum_seq_lens,
            seq_lens=seq_lens,
        )
        cum_extend_seq_lens = torch.zeros(
            num_extends + 1, device=self.device, dtype=torch.int32
        )
        torch.cumsum(extend_seq_lens, dim=0, out=cum_extend_seq_lens[1:])
        max_extend_seq_len = extend_seq_lens_cpu.max().item()
        # The table is batch-ordered kernel pages (row i == batch position i).
        req_pool_indices = torch.arange(
            num_extends, dtype=torch.int64, device=page_table.device
        )
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
            req_pool_indices,
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
            page_table=page_table,
        )

    # ---- CUDA Graph ----

    def init_cuda_graph_state(self, max_bs: int) -> None:
        # Own the cache-seqlens buffer; every refresh copies the live lengths
        # in. Block decode records spec_num_tokens rows per request.
        rows = max_bs * (self.spec_num_tokens if self.block_decode_active else 1)
        self.seq_lens_buf = torch.zeros(rows, dtype=torch.int32, device=self.device)
        self.page_table_buf = torch.zeros(
            (rows, self.max_num_pages), dtype=torch.int32, device=self.device
        )
        self._decode_views_by_bs = {}

    @property
    def decode_seq_lens_buffer(self) -> torch.Tensor:
        return self.seq_lens_buf

    def _decode_views(self, bs: int) -> CuteDSLMLADecodeMetadata:
        """Per-bs decode metadata views over the persistent buffers.

        One builder for capture and refresh; cached per bs — pointer-stable,
        no storage allocated.
        """
        metadata = self._decode_views_by_bs.get(bs)
        if metadata is not None:
            return metadata
        if self.block_decode_active:
            rows = bs * self.spec_num_tokens
            metadata = CuteDSLMLADecodeMetadata(
                page_table=self.page_table_buf[:rows],
                max_seq_len_k=self.max_context_len,
                seq_lens_k=self.seq_lens_buf[:rows],
                num_extends=0,
                q_len_per_req=1,
            )
        else:
            metadata = CuteDSLMLADecodeMetadata(
                page_table=self.page_table_buf[:bs],
                max_seq_len_k=self.max_context_len,
                seq_lens_k=self.seq_lens_buf[:bs],
                num_extends=0,
                q_len_per_req=self.verify_floor,
            )
        self._decode_views_by_bs[bs] = metadata
        return metadata

    # Capture is inherited (the leaf default: idle refresh over the same buffers).

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
        metadata = self._decode_views(bs)
        # The cached view bakes num_extends=0; a mixed round's decode rows
        # start after the extend rows, so publish this round's split.
        metadata.num_extends = num_extends
        if self.block_decode_active:
            # Replicate each request's page table across its block rows. Under
            # replay the lengths come from fill_block_decode_seq_lens, inside
            # the graph; eager has no in-graph writer, so fill here (and the
            # capture seeding needs the same safe baseline).
            spec = self.spec_num_tokens
            width = self.page_table_buf.shape[1]
            rows = self.page_table_buf[: bs * spec].view(bs, spec, width)
            cols = min(page_table.shape[1], width)
            rows[:, :, :cols].copy_(page_table[:bs, None, :cols])
            if cols < width:
                rows[:, :, cols:].zero_()
            if not for_graph_replay or actual_bs == 0:
                self.fill_block_decode_seq_lens(bs, seq_lens)
            self.forward_decode_metadata = metadata
            return
        # clamp_min(1) is the identity, so the verify clamp is unconditional.
        self.seq_lens_buf[:bs].copy_(seq_lens[:bs].clamp_min(metadata.q_len_per_req))
        # The persistent buffer is padded to the fused-kernel block constraint;
        # columns past the router table's width stay 0 (never read: the kernel
        # bounds access by seq_lens). Padded (and idle) rows are already null
        # pages in the router table.
        cols = min(page_table.shape[1], self.page_table_buf.shape[1])
        self.page_table_buf[:bs, :cols].copy_(page_table[:bs, :cols])
        self.forward_decode_metadata = metadata

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
        num_extends = metadata.num_extends
        if self.block_decode_active:
            # Keeping the block on the query axis would re-impose causal order.
            query = q.view(-1, layer.tp_q_head_num, layer.head_dim).unsqueeze(1)
        else:
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
            block_tables=metadata.page_table[num_extends:],
            seq_lens=metadata.seq_lens_k[num_extends:],
            max_seq_len=metadata.max_seq_len_k,
            softmax_scale=softmax_scale,
        )

        return raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    # ---- Forward: Extend/Prefill ----

    def forward_extend(
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
        raise NotImplementedError(
            "tokenspeed_mla has no dense extend kernel; DeepSeek's model path "
            "runs prefill through forward_extend_chunked"
        )

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
            step_counter = self.step_counter
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
