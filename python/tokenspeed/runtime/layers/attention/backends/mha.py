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
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel import (
    mha_decode_with_kvcache,
    mha_prefill,
    mha_prefill_with_kvcache,
)

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.layers.attention.utils import build_page_table

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


_KERNEL_SOLUTION_BY_BACKEND = {
    "mha": None,
    "fa3": "fa3",
    "fa4": "fa4",
    "triton": "triton",
    "flashinfer": "flashinfer",
}


@dataclass
class MHAMetadata:
    cache_seqlens_int32: torch.Tensor
    page_table: torch.Tensor
    cu_seqlens_q: torch.Tensor | None = None
    max_seq_len_q: int | None = None
    max_seq_len_k: int | None = None
    use_direct_prefill: bool = False


class MHAAttnBackend(AttentionBackend):
    """Standard MHA backend that routes through tokenspeed_kernel attention APIs."""

    @property
    def support_kv_cache_prewrite(self) -> bool:
        return False

    def __init__(self, config: MHAConfig):
        super().__init__(config)
        backend_name = config.backend_name or "mha"
        if backend_name not in _KERNEL_SOLUTION_BY_BACKEND:
            raise ValueError(f"Unsupported MHA backend: {backend_name!r}")
        self.kernel_solution = _KERNEL_SOLUTION_BY_BACKEND[backend_name]
        self.max_context_len = config.context_len
        self.page_size = config.page_size
        self.max_num_pages = (
            self.max_context_len + self.page_size - 1
        ) // self.page_size
        self.forward_decode_metadata: MHAMetadata | None = None
        self.forward_prefill_metadata: MHAMetadata | None = None

    def init_forward_metadata(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        req_to_page: torch.Tensor,
        extend_prefix_lens: torch.Tensor | None = None,
        **kwargs,
    ):
        assert (
            seq_lens.dtype == torch.int32
        ), f"seq_lens must be int32, got {seq_lens.dtype}"
        seq_lens = seq_lens[:bs]
        page_table = build_page_table(
            req_pool_indices[:bs],
            req_to_page,
            self.page_size,
            self.max_context_len,
        )

        if forward_mode.is_decode_or_idle():
            self.forward_decode_metadata = MHAMetadata(
                cache_seqlens_int32=seq_lens,
                page_table=page_table,
                max_seq_len_k=self.max_context_len,
            )
            return

        if forward_mode.is_target_verify() or forward_mode.is_draft_extend():
            tokens_per_req = num_tokens // bs if bs > 0 else 1
            self.forward_prefill_metadata = MHAMetadata(
                cache_seqlens_int32=seq_lens,
                cu_seqlens_q=self._make_uniform_cu_seqlens(
                    bs,
                    tokens_per_req,
                    seq_lens.device,
                ),
                page_table=page_table,
                max_seq_len_q=tokens_per_req,
                max_seq_len_k=self.max_context_len,
            )
            self.forward_decode_metadata = MHAMetadata(
                cache_seqlens_int32=seq_lens.clone(),
                page_table=page_table,
                max_seq_len_k=self.max_context_len,
            )
            return

        if extend_prefix_lens is None:
            extend_seq_lens = seq_lens
        else:
            assert (
                extend_prefix_lens.dtype == torch.int32
            ), f"extend_prefix_lens must be int32, got {extend_prefix_lens.dtype}"
            extend_seq_lens = seq_lens - extend_prefix_lens[:bs]

        extend_seq_lens_cpu = kwargs.get("extend_seq_lens_cpu")
        assert (
            extend_seq_lens_cpu is not None
        ), "mha extend requires extend_seq_lens_cpu"
        max_seq_len_q = int(extend_seq_lens_cpu[:bs].max().item())
        extend_prefix_lens_cpu = kwargs.get("extend_prefix_lens_cpu")
        if extend_prefix_lens is None:
            use_direct_prefill = True
        elif extend_prefix_lens_cpu is not None:
            use_direct_prefill = not bool(extend_prefix_lens_cpu[:bs].any().item())
        else:
            use_direct_prefill = False

        self.forward_prefill_metadata = MHAMetadata(
            cache_seqlens_int32=seq_lens,
            cu_seqlens_q=self._make_cu_seqlens(extend_seq_lens),
            page_table=page_table,
            max_seq_len_q=max_seq_len_q,
            max_seq_len_k=self.max_context_len,
            use_direct_prefill=use_direct_prefill,
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
        self.cuda_graph_prefill_metadata = {}
        self.cuda_graph_decode_metadata = {}
        # Alias controller's seq_lens_buf — backend never mutates it.
        self.cuda_graph_page_table = torch.zeros(
            (max_bs, self.max_num_pages), dtype=torch.int32, device=self.device
        )
        self.cuda_graph_cache_seqlens = seq_lens_buf

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
    ):
        if forward_mode.is_decode():
            cache_seqlens = self.cuda_graph_cache_seqlens[:bs]
            metadata = MHAMetadata(
                cache_seqlens_int32=cache_seqlens,
                page_table=self.cuda_graph_page_table[:bs, :],
                max_seq_len_k=self.max_context_len,
            )
            self.cuda_graph_decode_metadata[bs] = metadata
            self.forward_decode_metadata = metadata
        elif forward_mode.is_target_verify():
            spec_num_tokens = num_tokens // bs if bs > 0 else 1
            cache_seqlens = self.cuda_graph_cache_seqlens[:bs]
            metadata = MHAMetadata(
                cache_seqlens_int32=cache_seqlens,
                cu_seqlens_q=self._make_uniform_cu_seqlens(
                    bs,
                    spec_num_tokens,
                    self.device,
                ),
                page_table=self.cuda_graph_page_table[:bs, :],
                max_seq_len_q=spec_num_tokens,
                max_seq_len_k=self.max_context_len,
            )
            self.cuda_graph_prefill_metadata[bs] = metadata
            self.forward_prefill_metadata = metadata
        elif forward_mode.is_draft_extend():
            spec_num_tokens = num_tokens // bs if bs > 0 else 1
            cache_seqlens = self.cuda_graph_cache_seqlens[:bs]
            metadata = MHAMetadata(
                cache_seqlens_int32=cache_seqlens,
                cu_seqlens_q=self._make_uniform_cu_seqlens(
                    bs,
                    spec_num_tokens,
                    self.device,
                ),
                page_table=self.cuda_graph_page_table[:bs, :],
                max_seq_len_q=spec_num_tokens,
                max_seq_len_k=self.max_context_len,
            )
            self.cuda_graph_prefill_metadata[bs] = metadata
            self.forward_prefill_metadata = metadata

            metadata = MHAMetadata(
                cache_seqlens_int32=cache_seqlens,
                page_table=self.cuda_graph_page_table[:bs, :],
                max_seq_len_k=self.max_context_len,
            )
            self.cuda_graph_decode_metadata[bs] = metadata
            self.forward_decode_metadata = metadata
        else:
            raise NotImplementedError(
                f"mha CUDA graph capture not supported for {forward_mode}"
            )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        req_to_page: torch.Tensor = None,
        **kwargs,
    ):
        # cache_seqlens aliases seq_lens_buf; only page_table needs refresh.
        if req_to_page is not None:
            self.cuda_graph_page_table[:bs, : self.max_num_pages].copy_(
                req_to_page[req_pool_indices[:bs], : self.max_num_pages]
            )

        if forward_mode.is_decode():
            self.forward_decode_metadata = self.cuda_graph_decode_metadata[bs]
        elif forward_mode.is_target_verify():
            self.forward_prefill_metadata = self.cuda_graph_prefill_metadata[bs]
        elif forward_mode.is_draft_extend():
            self.forward_prefill_metadata = self.cuda_graph_prefill_metadata[bs]
            self.forward_decode_metadata = self.cuda_graph_decode_metadata[bs]
        else:
            raise NotImplementedError(
                f"mha CUDA graph replay not supported for {forward_mode}"
            )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        save_kv_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if layer.qk_head_dim != layer.v_head_dim:
            raise NotImplementedError("mha backend requires qk_head_dim == v_head_dim")

        if save_kv_cache and k is not None:
            token_to_kv_pool.set_kv_buffer(
                layer,
                out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        metadata = self.forward_decode_metadata
        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k_cache, v_cache = self._get_kv_cache(layer, token_to_kv_pool)

        result = mha_decode_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens_int32,
            softmax_scale=layer.scaling,
            is_causal=True,
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            sinks=kwargs.get("sinks"),
            max_seqlen_k=metadata.max_seq_len_k,
            solution=self.kernel_solution,
        )
        return self._unwrap_output(result).reshape(
            -1, layer.tp_q_head_num * layer.v_head_dim
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        forward_mode: ForwardMode,
        save_kv_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if layer.qk_head_dim != layer.v_head_dim:
            raise NotImplementedError("mha backend requires qk_head_dim == v_head_dim")

        metadata = self.forward_prefill_metadata
        cu_seqlens_q = metadata.cu_seqlens_q
        assert cu_seqlens_q is not None
        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k = None if k is None else k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v = None if v is None else v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

        if k is not None and v is not None and metadata.use_direct_prefill:
            return self._forward_extend_direct(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                metadata,
                cu_seqlens_q,
                save_kv_cache,
                kwargs.get("sinks"),
            )

        return self._forward_extend_with_kvcache(
            q,
            k,
            v,
            layer,
            out_cache_loc,
            token_to_kv_pool,
            metadata,
            cu_seqlens_q,
            save_kv_cache,
            kwargs.get("sinks"),
        )

    def _forward_extend_direct(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        metadata: MHAMetadata,
        cu_seqlens_q: torch.Tensor,
        save_kv_cache: bool,
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        result = mha_prefill(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=metadata.max_seq_len_q,
            max_seqlen_k=metadata.max_seq_len_q,
            softmax_scale=layer.scaling,
            is_causal=True,
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            sinks=sinks,
            solution=self.kernel_solution,
        )
        output = self._unwrap_output(result).reshape(
            -1, layer.tp_q_head_num * layer.v_head_dim
        )
        if save_kv_cache:
            token_to_kv_pool.set_kv_buffer(
                layer,
                out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )
        return output

    def _forward_extend_with_kvcache(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        metadata: MHAMetadata,
        cu_seqlens_q: torch.Tensor,
        save_kv_cache: bool,
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        if save_kv_cache and k is not None:
            token_to_kv_pool.set_kv_buffer(
                layer,
                out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        k_cache, v_cache = self._get_kv_cache(layer, token_to_kv_pool)
        result = mha_prefill_with_kvcache(
            q=q,
            k=None if save_kv_cache else k,
            v=None if save_kv_cache else v,
            cu_seqlens_q=cu_seqlens_q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens_int32,
            softmax_scale=layer.scaling,
            is_causal=True,
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            sinks=sinks,
            max_seqlen_q=metadata.max_seq_len_q,
            max_seqlen_k=metadata.max_seq_len_k,
            solution=self.kernel_solution,
        )
        return self._unwrap_output(result).reshape(
            -1, layer.tp_q_head_num * layer.v_head_dim
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

    @staticmethod
    def _make_cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(
            torch.cumsum(lengths, dim=0, dtype=torch.int32),
            (1, 0),
        )

    @staticmethod
    def _make_uniform_cu_seqlens(
        batch_size: int,
        tokens_per_req: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.arange(
            0,
            batch_size * tokens_per_req + 1,
            tokens_per_req,
            dtype=torch.int32,
            device=device,
        )

    @staticmethod
    def _unwrap_output(result):
        if isinstance(result, tuple):
            return result[0]
        return result


for _backend_name in _KERNEL_SOLUTION_BY_BACKEND:
    register_backend(_backend_name, {AttentionArch.MHA}, MHAAttnBackend)
