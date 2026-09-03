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
MLA attention leaf using fused kernels optimized for SM100 (Blackwell) GPUs.
"""

from __future__ import annotations

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
from tokenspeed.runtime.layers.attention.backends.paged import (
    PagedAttentionBackend,
)
from tokenspeed.runtime.layers.attention.chunk import (
    build_chunked_prefill_metadata_arrays,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.kernel_page_sizes import (
    TRTLLM_MLA_DEFAULT_PAGE_SIZE,
    TRTLLM_MLA_SUPPORTED_PAGE_SIZES,
)
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.utils.env import envs

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention

# Block constraint from flashinfer: block_num % (128 / page_size) == 0
TRTLLM_BLOCK_CONSTRAINT = 128


def calc_padded_blocks(max_seq_len: int, kernel_page_size: int) -> int:
    """Kernel page-table width for ``max_seq_len`` tokens, padded to the
    fused-kernel block constraint.

    Args:
        max_seq_len: Per-request token extent the table must cover.
        kernel_page_size: Tokens per kernel page.

    Returns:
        The page count, rounded up to a multiple of
        ``TRTLLM_BLOCK_CONSTRAINT // kernel_page_size``.
    """
    blocks = triton.cdiv(max_seq_len, kernel_page_size)
    constraint = TRTLLM_BLOCK_CONSTRAINT // kernel_page_size
    if blocks % constraint != 0:
        blocks = triton.cdiv(blocks, constraint) * constraint
    return blocks


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
    cum_extend_seq_lens: torch.Tensor  # cumsum prefix-padded, sized num_extends+1
    max_extend_seq_len: int
    # Per-prefix-chunk arrays for non-causal cross-attention (built once per
    # iteration in _init_prefill_metadata, indexed by loop_idx in the model).
    chunked_loop_num: int
    chunk_kv_indices_list: list  # List[torch.Tensor], one per loop_idx
    chunked_seq_len: torch.Tensor  # (chunked_loop_num, num_extends) int32 GPU
    cu_chunked_seq_len: torch.Tensor  # (chunked_loop_num, num_extends+1) int32 GPU
    max_chunk_len_per_loop: list  # List[int], one per loop_idx
    # The extend rows' batch-ordered [num_extends, max_num_pages] kernel page
    # table; the DSA wrapper maps its sparse-prefill top-k through it.
    page_table: torch.Tensor | None = None


@dataclass
class TRTLLMMLADecodeMetadata:
    num_extends: int = 0
    page_table: torch.Tensor | None = None
    max_seq_len_k: int | None = None
    seq_lens_k: torch.Tensor | None = None
    # Verify window width baked into the graph views (1 outside target verify).
    q_len_per_req: int = 1


class TRTLLMMLABackend(PagedAttentionBackend):
    """trtllm_mla leaf using fused kernels."""

    default_kernel_page_size = TRTLLM_MLA_DEFAULT_PAGE_SIZE

    def __init__(self, config: AttnConfig, spec: MLAConfig, *, kernel_page_size: int):
        if kernel_page_size not in TRTLLM_MLA_SUPPORTED_PAGE_SIZES:
            raise ValueError(
                f"trtllm_mla backend requires page_size 32 or 64, got {kernel_page_size}"
            )
        super().__init__(config, spec, kernel_page_size=kernel_page_size)
        # The trtllm kernel walks pages at page_size, padded to the
        # fused-kernel block constraint.
        self.max_num_pages = calc_padded_blocks(config.context_len, kernel_page_size)

        # MLA dimensions
        self.kv_lora_rank = spec.kv_lora_rank
        self.qk_nope_head_dim = spec.qk_nope_head_dim
        self.qk_rope_head_dim = spec.qk_rope_head_dim
        self.v_head_dim = spec.v_head_dim
        self.kv_cache_dim = spec.kv_cache_dim
        self.scaling = spec.scaling
        self.data_type = config.kv_cache_dtype
        self.q_data_type = config.dtype
        self.draft_block_decode = bool(config.draft_block_decode)

        # Workspace zero-initialized for the fused kernel semaphore.
        self.trtllm_workspace = get_trtllm_workspace_buffer(config.device)

        self.num_local_heads = spec.num_attention_heads // spec.attn_tp_size

        # Metadata
        self.forward_decode_metadata: TRTLLMMLADecodeMetadata | None = None
        self.forward_prefill_metadata: TRTLLMMLAPrefillMetadata | None = None
        self.chunked_prefill_metadata: TRTLLMMLAChunkedPrefillMetadata | None = None

    # ---- Metadata initialization ----

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
        **kwargs,
    ):
        if not (forward_mode.is_extend_or_mixed() or forward_mode.is_idle()):
            raise RuntimeError(
                "trtllm_mla decode metadata goes through refresh_decode_metadata; "
                f"init_forward_metadata only serves extend/mixed ({forward_mode})"
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
            self.kernel_page_size,
        )
        self.chunked_prefill_metadata = TRTLLMMLAChunkedPrefillMetadata(
            extend_prefix_lens=extend_prefix_lens,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
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

    def _decode_views(self, bs: int) -> TRTLLMMLADecodeMetadata:
        """Per-bs decode metadata views over the persistent buffers.

        One builder for capture and refresh; cached per bs — pointer-stable,
        no storage allocated.
        """
        metadata = self._decode_views_by_bs.get(bs)
        if metadata is not None:
            return metadata
        metadata = TRTLLMMLADecodeMetadata(
            num_extends=0,
            page_table=self.page_table_buf[:bs],
            max_seq_len_k=self.max_context_len,
            seq_lens_k=self.seq_lens_buf[:bs],
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
        del for_graph_replay
        metadata = self._decode_views(bs)
        # The cached view bakes num_extends=0; a mixed round's decode rows
        # start after the extend rows, so publish this round's split.
        metadata.num_extends = num_extends
        # clamp_min(1) is the identity, so the verify clamp is unconditional.
        self.seq_lens_buf[:bs].copy_(seq_lens[:bs].clamp_min(metadata.q_len_per_req))
        # The persistent buffer is padded to the fused-kernel block constraint;
        # columns past the router table's width stay 0 (never read: the kernel
        # bounds access by seq_lens).
        num_pages = min(page_table.shape[1], self.max_num_pages)
        self.page_table_buf[:bs, :num_pages].copy_(page_table[:bs, :num_pages])
        self.forward_decode_metadata = metadata

    @property
    def block_decode_expansion(self) -> int:
        """One entry per request even under block decode: ``forward_decode``
        repeats it across the block's queries at forward time."""
        return 1

    def fill_block_decode_seq_lens(self, bs: int, block_seq_lens: torch.Tensor) -> None:
        """Publish block-end cache lengths inside a captured draft graph (one
        entry per request; see :attr:`block_decode_expansion`).

        Args:
            bs: Number of draft requests.
            block_seq_lens: Per-request lengths after writing the draft block.
        """
        if not self.draft_block_decode:
            raise RuntimeError("Block decode sequence lengths require DFLASH mode.")
        self.seq_lens_buf[:bs].copy_(
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
        # A block drafter's decode metadata describes only decode rows, so
        # there are no leading extend rows to slice away.
        num_extends = 0 if self.draft_block_decode else metadata.num_extends
        q_len_per_req = q.shape[0] // bs if bs > 0 else 1

        if q_len_per_req > 1 and self.is_draft:
            query = q.view(-1, layer.tp_q_head_num, layer.head_dim).unsqueeze(1)
            page_table = metadata.page_table[num_extends:].repeat_interleave(
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
            page_table = metadata.page_table[num_extends:]
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
            "trtllm_mla has no dense extend kernel; DeepSeek's model path runs "
            "prefill through forward_extend_chunked"
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
        if causal:
            step_counter = self.step_counter
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
