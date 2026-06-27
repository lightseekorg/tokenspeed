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

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel import (
    attn_plan,
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_prefill,
)
from tokenspeed_kernel.ops.kvcache.triton import (
    fused_fp8_set_kv_buffer,
    gather_page_table_with_padding,
)

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.layers.attention.utils import build_page_table
from tokenspeed.runtime.utils.common import ceil_div

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


_KERNEL_SOLUTION_BY_BACKEND = {
    "mha": None,
    "fa3": "fa3",
    "fa4": "fa4",
    "triton": "triton",
    "flashinfer": "flashinfer",
}


@dataclass(kw_only=True)
class MHAExtendMetadata:
    # Device-side metadata:
    # - seq_lens: total length after this step
    # - extend_seq_lens: length of new tokens
    #   cu_extend_seq_lens: the cumsum version of extend_seq_lens
    #   cu_seqlens_kv: the cumsum version of seq_lens
    # - extend_prefix_lens: length of the cached prefix tokens
    # seq_lens[i] = extend_prefix_lens[i] + extend_seq_lens[i]
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    extend_seq_lens: torch.Tensor
    cu_extend_seq_lens: torch.Tensor
    cu_seqlens_kv: torch.Tensor
    extend_prefix_lens: torch.Tensor
    # Host-side metadata:
    extend_seq_lens_cpu: list[int]
    cu_extend_seq_lens_cpu: list[int]
    max_extend_seq_len: int
    max_extend_prefix_len: int = 0


@dataclass(kw_only=True)
class MHADecodeMetadata:
    # Device-side metadata.
    page_table: torch.Tensor
    seq_lens: torch.Tensor


class MHAAttnBackend(AttentionBackend):
    """Standard MHA backend that routes through tokenspeed_kernel attention APIs."""

    def support_kv_cache_prewrite(
        self, forward_mode: ForwardMode | None = None
    ) -> bool:
        return forward_mode is not None and forward_mode.is_decode()

    def __init__(self, config: MHAConfig):
        super().__init__(config)
        # Map the selected backend to the corresponding kernel solution string.
        backend_name = config.backend_name or "mha"
        self.kernel_solution = _KERNEL_SOLUTION_BY_BACKEND[backend_name]

        # Static information needed for metadata construction and kernel dispatch
        self.max_context_len = config.context_len
        self.page_size = config.page_size
        self.max_num_pages = ceil_div(self.max_context_len, self.page_size)
        num_q_heads = config.num_attention_heads
        num_kv_heads = config.num_kv_heads
        self.tp_q_head_num = max(num_q_heads // config.attn_tp_size, 1)
        self.tp_kv_head_num = max(num_kv_heads // config.attn_tp_size, 1)
        self.head_dim = config.head_dim
        self.qkv_dtype = config.dtype
        self.kv_cache_dtype = config.kv_cache_dtype
        self.is_fp8 = self.kv_cache_dtype == torch.float8_e4m3fn
        self.plan = partial(
            attn_plan,
            dtype=torch.float8_e4m3fn if self.is_fp8 else self.qkv_dtype,
            head_dim=self.head_dim,
            return_lse=False,
            solution=self.kernel_solution,
        )
        # DFLASH draft: expand decode metadata to spec_num_tokens rows/request
        # (whole block in one decode forward), with uniform non-causal seq_lens.
        self.draft_block_decode = bool(getattr(config, "draft_block_decode", False))

        # Forward metadata is initialized in the runner per forward call
        self.forward_decode_metadata: MHADecodeMetadata | None = None
        self.forward_extend_metadata: MHAExtendMetadata | None = None

    # ------------------------------------------------------------------
    # Metadata initialization
    # ------------------------------------------------------------------

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_page: torch.Tensor,
        forward_mode: ForwardMode,
        # Only consumed on the extend/mixed path; decode callers (e.g. the
        # DFLASH draft and the cuda-graph wrapper's draft decode init) omit
        # them, so they must be optional.
        extend_seq_lens: torch.Tensor | None = None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
        extend_prefix_lens: torch.Tensor | None = None,
        extend_prefix_lens_cpu: torch.Tensor | None = None,
        **kwargs,
    ):
        assert not forward_mode.is_mixed(), "mha backend does not support mixed batch"

        seq_lens = seq_lens[:bs]
        page_table = build_page_table(
            req_pool_indices[:bs],
            req_to_page,
            self.page_size,
            self.max_context_len,
        )

        if forward_mode.is_extend_or_mixed():
            assert extend_seq_lens is not None
            assert extend_seq_lens_cpu is not None
            assert extend_prefix_lens is not None
            assert extend_prefix_lens_cpu is not None

            # Create cumulative sum of the sequence lengths for Q and KV.
            extend_seq_lens = extend_seq_lens[:bs]
            extend_seq_lens_cpu = [int(x) for x in extend_seq_lens_cpu[:bs].tolist()]
            cu_extend_seq_lens = torch.nn.functional.pad(
                torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32),
                (1, 0),
            )
            cu_extend_seq_lens_cpu = [0]
            for length in extend_seq_lens_cpu:
                cu_extend_seq_lens_cpu.append(cu_extend_seq_lens_cpu[-1] + length)
            cu_seqlens_kv = torch.nn.functional.pad(
                torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
                (1, 0),
            )
            extend_prefix_lens = extend_prefix_lens[:bs]
            max_extend_seq_len = max(extend_seq_lens_cpu)
            max_extend_prefix_len = int(extend_prefix_lens_cpu[:bs].max().item())

            self.forward_extend_metadata = MHAExtendMetadata(
                page_table=page_table,
                seq_lens=seq_lens,
                extend_seq_lens=extend_seq_lens,
                cu_extend_seq_lens=cu_extend_seq_lens,
                cu_seqlens_kv=cu_seqlens_kv,
                extend_prefix_lens=extend_prefix_lens,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
                cu_extend_seq_lens_cpu=cu_extend_seq_lens_cpu,
                max_extend_seq_len=max_extend_seq_len,
                max_extend_prefix_len=max_extend_prefix_len,
            )

            # Drafter: also fill decode_metadata so step 1+ multi-step has
            # metadata under EXTEND/MIXED target. seq_lens is the drafter's
            # live alias buffer (wrapper pre-writes it before this call).
            if self.is_draft:
                self.forward_decode_metadata = MHADecodeMetadata(
                    page_table=page_table,
                    seq_lens=seq_lens,
                )
        else:
            if self.draft_block_decode and self.spec_num_tokens > 1:
                # DFLASH drafts a whole block in one decode forward; the decode
                # kernel keys masking off max_seqlen_q, so expand each request
                # into spec_num_tokens rows with the SAME full seq_len. That
                # makes max_seqlen_q == 1 per row, so every block query attends
                # over the entire block (non-causal block-diffusion drafting).
                # Target verify keeps the unexpanded multi-query decode path.
                expanded_page_table, expanded_seq_lens = (
                    self._make_spec_metadata_buffers(
                        bs,
                        page_table.device,
                    )
                )
                self._fill_spec_metadata_uniform(
                    expanded_page_table,
                    expanded_seq_lens,
                    page_table,
                    seq_lens,
                )
                self.forward_decode_metadata = MHADecodeMetadata(
                    page_table=expanded_page_table,
                    seq_lens=expanded_seq_lens,
                )
            else:
                self.forward_decode_metadata = MHADecodeMetadata(
                    page_table=page_table,
                    seq_lens=seq_lens,
                )

    def init_cuda_graph_state(self, max_bs: int, seq_lens_buf: torch.Tensor):
        assert (
            seq_lens_buf.dtype == torch.int32
            and seq_lens_buf.dim() == 1
            and seq_lens_buf.shape[0] >= max_bs
        ), (
            f"seq_lens_buf must be int32 with shape[0] >= {max_bs}, "
            f"got {seq_lens_buf.dtype} {tuple(seq_lens_buf.shape)}"
        )

        self.cuda_graph_decode_metadata = {}
        if self.draft_block_decode and self.spec_num_tokens > 1:
            # DFLASH draft block: expand to spec_num_tokens decode rows per
            # request (one row per block position), so max_seqlen_q == 1 per row
            # and every block query attends over the whole block (non-causal).
            self.cuda_graph_page_table, self.cuda_graph_seq_lens = (
                self._make_spec_metadata_buffers(max_bs, self.device)
            )
            self.cuda_graph_page_table.zero_()
            # seq_lens are filled from the live draft length inside the captured
            # graph; seed a valid baseline so any pre-broadcast read stays in range.
            self.cuda_graph_seq_lens.fill_(self.spec_num_tokens)
            return
        self.cuda_graph_page_table = torch.zeros(
            (max_bs, self.max_num_pages), dtype=torch.int32, device=self.device
        )
        if self.spec_num_tokens > 1 and not self.is_draft:
            self.cuda_graph_seq_lens = torch.empty(
                (max_bs,), dtype=torch.int32, device=self.device
            )
        else:
            # Alias controller's seq_lens_buf — backend never mutates it.
            self.cuda_graph_seq_lens = seq_lens_buf

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
    ):
        assert not forward_mode.is_extend_or_mixed()

        if self.draft_block_decode and self.spec_num_tokens > 1:
            # DFLASH draft block: spec_num_tokens decode rows per request.
            expanded_bs = bs * self.spec_num_tokens
            metadata = MHADecodeMetadata(
                page_table=self.cuda_graph_page_table[:expanded_bs, :],
                seq_lens=self.cuda_graph_seq_lens[:expanded_bs],
            )
            # Uniform non-causal seq_lens are written by the drafter inside the
            # captured graph (see fill_block_decode_seq_lens); seed a safe
            # baseline for the capture run before that op records.
            metadata.seq_lens.fill_(self.spec_num_tokens)
        else:
            metadata = MHADecodeMetadata(
                page_table=self.cuda_graph_page_table[:bs, :],
                seq_lens=self.cuda_graph_seq_lens[:bs],
            )
            if self.spec_num_tokens > 1 and not self.is_draft:
                metadata.seq_lens.copy_(seq_lens[:bs].clamp_min(self.spec_num_tokens))
        self.cuda_graph_decode_metadata[bs] = metadata
        self.forward_decode_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_page: torch.Tensor,
        forward_mode: ForwardMode,
        **kwargs,
    ):
        assert not forward_mode.is_extend_or_mixed()

        gather_page_table_with_padding(
            req_to_page=req_to_page,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out=self.cuda_graph_page_table,
            bs=bs,
            max_num_pages=self.max_num_pages,
            page_size=self.page_size,
            dummy_slot=0,
        )
        if self.spec_num_tokens > 1 and not self.is_draft:
            self.cuda_graph_seq_lens[:bs].copy_(seq_lens[:bs])
        elif self.draft_block_decode:
            # DFLASH draft: replicate each request's page table to its
            # spec_num_tokens block rows. The block-end seq_lens are filled by
            # the drafter inside the captured graph, so they are not touched
            # here (they re-derive from the live draft length on every replay).
            base_page_table = req_to_page[req_pool_indices[:bs], : self.max_num_pages]
            self.cuda_graph_page_table[: bs * self.spec_num_tokens, :].view(
                bs, self.spec_num_tokens, self.max_num_pages
            ).copy_(base_page_table[:, None, :])

        if bs in self.cuda_graph_decode_metadata:
            self.forward_decode_metadata = self.cuda_graph_decode_metadata[bs]

    def fill_block_decode_seq_lens(self, bs: int, block_seq_lens: torch.Tensor) -> None:
        """DFLASH: broadcast each request's block-end length to its
        spec_num_tokens cuda-graph decode rows (uniform, non-causal).

        Called by the drafter inside the captured graph so that on every replay
        the expanded seq_lens re-derive from the live draft length (which is
        recomputed in-graph from the target's accept lengths).

        Args:
            bs: Number of draft requests.
            block_seq_lens: ``[bs]`` per-request block-end lengths
                (prefix + spec_num_tokens).
        """
        spec = self.spec_num_tokens
        self.cuda_graph_seq_lens[: bs * spec].view(bs, spec).copy_(
            block_seq_lens[:bs]
            .clamp(self.spec_num_tokens, self.max_context_len)
            .unsqueeze(1)
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        assert layer.qk_head_dim == layer.v_head_dim
        assert (k is None) == (v is None)
        has_kv = k is not None

        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        if has_kv:
            k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
            v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
        sinks = kwargs.get("sinks")

        return self._forward_decode(
            q,
            k,
            v,
            layer,
            out_cache_loc,
            token_to_kv_pool,
            self.forward_decode_metadata,
            save_kv_cache=save_kv_cache,
            sinks=sinks,
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        assert layer.qk_head_dim == layer.v_head_dim
        assert (k is None) == (v is None)
        assert k is not None

        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

        metadata = self.forward_extend_metadata
        sinks = kwargs.get("sinks")
        plan = self.plan(
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            sinks=sinks,
        )

        extend_mode = plan.get("extend_mode", "prewrite")
        if metadata.max_extend_prefix_len == 0 and extend_mode == "postwrite":
            return self._forward_prefill(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                metadata,
                save_kv_cache,
                sinks,
            )
        else:
            return self._forward_extend(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                metadata,
                save_kv_cache,
                sinks,
            )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        metadata: MHAExtendMetadata,
        save_kv_cache: bool,
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        # TODO: use a custom kernel to do downcast
        if self.is_fp8:
            q = q.to(torch.float8_e4m3fn)
            k = k.to(torch.float8_e4m3fn)
            v = v.to(torch.float8_e4m3fn)

        output = mha_prefill(
            q=q,
            k=k,
            v=v,
            cu_seqlens=metadata.cu_extend_seq_lens,
            cu_seqlens_cpu=metadata.cu_extend_seq_lens_cpu,
            max_seqlen=metadata.max_extend_seq_len,
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            sinks=sinks,
            solution=self.kernel_solution,
        )
        output = output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)
        if save_kv_cache:
            self._save_kv_cache(layer, out_cache_loc, token_to_kv_pool, k, v)
        return output

    def _forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        metadata: MHAExtendMetadata,
        save_kv_cache: bool,
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        if save_kv_cache:
            self._save_kv_cache(layer, out_cache_loc, token_to_kv_pool, k, v)

        if self.is_fp8:
            q = q.to(torch.float8_e4m3fn)

        k_cache, v_cache = self._get_kv_cache(layer, token_to_kv_pool)
        output = mha_extend_with_kvcache(
            q=q,
            cu_seqlens_q=metadata.cu_extend_seq_lens,
            cu_seqlens_kv=metadata.cu_seqlens_kv,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=metadata.page_table,
            cache_seqlens=metadata.seq_lens,
            max_seqlen_q=metadata.max_extend_seq_len,
            max_seqlen_k=self.max_context_len,
            # DFLASH marks its draft attention non-causal so the draft block's
            # query positions attend bidirectionally. Every other layer leaves
            # the attribute unset, so this stays causal by default.
            is_causal=not bool(getattr(layer, "non_causal", False)),
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            sinks=sinks,
            solution=self.kernel_solution,
        )
        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        metadata: MHADecodeMetadata,
        save_kv_cache: bool,
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        if save_kv_cache:
            self._save_kv_cache(layer, out_cache_loc, token_to_kv_pool, k, v)

        if self.is_fp8:
            q = q.to(torch.float8_e4m3fn)

        k_cache, v_cache = self._get_kv_cache(layer, token_to_kv_pool)
        max_seqlen_q = q.shape[0] // metadata.seq_lens.shape[0]
        output = mha_decode_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=metadata.page_table,
            cache_seqlens=metadata.seq_lens,
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            sinks=sinks,
            max_seqlen_k=self.max_context_len,
            max_seqlen_q=max_seqlen_q,
            solution=self.kernel_solution,
        )
        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _save_kv_cache(
        self,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
    ) -> None:
        if k is None:
            return

        if self.is_fp8 and k.dtype != torch.float8_e4m3fn:
            k_cache, v_cache = token_to_kv_pool.get_kv_buffer(layer.layer_id)
            fused_fp8_set_kv_buffer(
                k=k,
                v=v,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_loc=out_cache_loc,
                k_scale=layer.k_scale,
                v_scale=layer.v_scale,
                page_size=self.page_size,
            )
        else:
            token_to_kv_pool.set_kv_buffer(
                layer,
                out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

    def _get_kv_cache(self, layer: PagedAttention, token_to_kv_pool):
        k_cache = token_to_kv_pool.get_key_buffer(layer.layer_id).view(
            -1,
            self.page_size,
            layer.tp_k_head_num,
            layer.qk_head_dim,
        )
        v_cache = token_to_kv_pool.get_value_buffer(layer.layer_id).view(
            -1,
            self.page_size,
            layer.tp_v_head_num,
            layer.v_head_dim,
        )
        return k_cache, v_cache

    def _make_spec_metadata_buffers(
        self,
        bs: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expanded_bs = bs * self.spec_num_tokens
        cuda_graph_page_table = torch.empty(
            (expanded_bs, self.max_num_pages),
            dtype=torch.int32,
            device=device,
        )
        cuda_graph_seq_lens = torch.empty(
            (expanded_bs,),
            dtype=torch.int32,
            device=device,
        )
        return (cuda_graph_page_table, cuda_graph_seq_lens)

    def _fill_spec_metadata_uniform(
        self,
        expanded_page_table: torch.Tensor,
        expanded_seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        """Expand spec metadata with a uniform (non-causal) seq_len per row.

        Replicates the full seq_len to all spec_num_tokens rows of a request so
        each row decodes with max_seqlen_q == 1 over the whole block. Used by the
        DFLASH drafter so every block query attends over the entire block
        (non-causal block-diffusion drafting), as opposed to the target's
        unexpanded causal multi-query verify path.
        """
        bs = seq_lens.shape[0]
        spec_num_tokens = self.spec_num_tokens
        expanded_page_table = expanded_page_table.view(
            bs, spec_num_tokens, self.max_num_pages
        )
        expanded_page_table.copy_(page_table[:, None, :])
        # Clamp to max_context_len so the draft decode never asks the attention
        # kernel for more than max_num_pages worth of page-table columns. The
        # block-end length is prefix + spec_num_tokens, which can exceed
        # max_context_len for a request near the context limit; without the
        # clamp the kernel reads page_table[:, >= max_num_pages] out of bounds
        # (CUDA illegal memory access). Mirrors fill_block_decode_seq_lens on the
        # cuda-graph path (this eager path is taken by mixed prefill+decode
        # batches even when cuda graphs are enabled).
        expanded_seq_lens.view(bs, spec_num_tokens).copy_(
            seq_lens.clamp(spec_num_tokens, self.max_context_len)[:, None]
        )


for _backend_name in _KERNEL_SOLUTION_BY_BACKEND:
    register_backend(_backend_name, {AttentionArch.MHA}, MHAAttnBackend)
