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
    mha_decode_scheduler_metadata,
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_merge_state,
    mha_prefill,
)

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.layers.attention.utils import build_page_table
from tokenspeed.runtime.utils.env import global_server_args_dict

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
    prefix_seqlens_int32: torch.Tensor | None = None
    max_prefix_seq_len: int | None = None
    has_prefix: bool = False
    # FA3 scheduler metadata pre-computed once per scheduler step. When set,
    # the FA3 decode kernel skips its internal prepare_varlen_num_blocks
    # launch.
    scheduler_metadata: torch.Tensor | None = None


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
        self.mha_extend_mode = global_server_args_dict.get("mha_extend_mode", "paged")
        self.max_context_len = config.context_len
        self.page_size = config.page_size
        self.max_num_pages = (
            self.max_context_len + self.page_size - 1
        ) // self.page_size
        self.forward_decode_metadata: MHAMetadata | None = None
        self.forward_prefill_metadata: MHAMetadata | None = None

        # Constants for the FA3 scheduler-metadata pre-compute.
        self._tp_q_head_num = max(config.num_attention_heads // config.attn_tp_size, 1)
        self._tp_k_head_num = max(config.num_kv_heads // config.attn_tp_size, 1)
        self._head_dim = config.head_dim
        self._qkv_dtype = config.dtype

    def init_forward_metadata(
        self,
        bs: int,
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

        if forward_mode.is_extend_or_mixed():
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
            if not forward_mode.is_extend():
                has_prefix = False
            elif extend_prefix_lens is None:
                has_prefix = False
            elif extend_prefix_lens_cpu is not None:
                has_prefix = bool(extend_prefix_lens_cpu[:bs].any().item())
            else:
                has_prefix = False

            prefix_seqlens = None
            max_prefix_seq_len = None
            if extend_prefix_lens is not None:
                prefix_seqlens = extend_prefix_lens[:bs]
                if extend_prefix_lens_cpu is not None:
                    max_prefix_seq_len = int(extend_prefix_lens_cpu[:bs].max().item())

            self.forward_prefill_metadata = MHAMetadata(
                cache_seqlens_int32=seq_lens,
                cu_seqlens_q=self._make_cu_seqlens(extend_seq_lens),
                page_table=page_table,
                max_seq_len_q=max_seq_len_q,
                max_seq_len_k=self.max_context_len,
                prefix_seqlens_int32=prefix_seqlens,
                max_prefix_seq_len=max_prefix_seq_len,
                has_prefix=has_prefix,
            )
            if not self.is_draft:
                return
            # Drafter: also fill decode_metadata so step 1+ multi-step has
            # metadata under EXTEND/MIXED target. seq_lens is the drafter's
            # live alias buffer (wrapper pre-writes it before this call).
            self.forward_decode_metadata = MHAMetadata(
                cache_seqlens_int32=seq_lens,
                page_table=page_table,
                max_seq_len_k=self.max_context_len,
            )
            return

        if self.spec_num_tokens > 1:
            self.forward_prefill_metadata = MHAMetadata(
                cache_seqlens_int32=seq_lens,
                cu_seqlens_q=self._make_uniform_cu_seqlens(
                    bs,
                    self.spec_num_tokens,
                    seq_lens.device,
                ),
                page_table=page_table,
                max_seq_len_q=self.spec_num_tokens,
                max_seq_len_k=self.max_context_len,
            )
            if self.is_draft:
                # Drafter follow-up single-token steps after the first.
                # cache_seqlens_int32 aliases seq_lens (drafter's live buffer)
                # so multi-step in-place advances propagate to the kernel.
                self.forward_decode_metadata = MHAMetadata(
                    cache_seqlens_int32=seq_lens,
                    page_table=page_table,
                    max_seq_len_k=self.max_context_len,
                )
        else:
            metadata = MHAMetadata(
                cache_seqlens_int32=seq_lens,
                page_table=page_table,
                max_seq_len_k=self.max_context_len,
            )
            metadata.scheduler_metadata = self._maybe_compute_scheduler_metadata(
                bs, seq_lens
            )
            self.forward_decode_metadata = metadata

    def _maybe_compute_scheduler_metadata(
        self, bs: int, cache_seqlens: torch.Tensor
    ) -> torch.Tensor | None:
        """Pre-compute FA3 decode scheduler metadata once per step.

        Returns ``None`` when the active backend does not consume pre-computed
        scheduler metadata (only FA3 on Hopper does); the kernel then falls
        back to its internal prepare_varlen_num_blocks launch.
        """
        return mha_decode_scheduler_metadata(
            batch_size=bs,
            max_seqlen_q=1,
            max_seqlen_k=self.max_context_len,
            num_heads_q=self._tp_q_head_num,
            num_heads_kv=self._tp_k_head_num,
            headdim=self._head_dim,
            cache_seqlens=cache_seqlens,
            qkv_dtype=self._qkv_dtype,
            page_size=self.page_size,
            causal=True,
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
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
    ):
        if forward_mode.is_extend_or_mixed():
            raise NotImplementedError(
                f"mha CUDA graph capture not supported for {forward_mode}"
            )

        cache_seqlens = self.cuda_graph_cache_seqlens[:bs]
        if self.spec_num_tokens > 1:
            metadata = MHAMetadata(
                cache_seqlens_int32=cache_seqlens,
                cu_seqlens_q=self._make_uniform_cu_seqlens(
                    bs,
                    self.spec_num_tokens,
                    self.device,
                ),
                page_table=self.cuda_graph_page_table[:bs, :],
                max_seq_len_q=self.spec_num_tokens,
                max_seq_len_k=self.max_context_len,
            )
            self.cuda_graph_prefill_metadata[bs] = metadata
            self.forward_prefill_metadata = metadata
            if self.is_draft:
                metadata = MHAMetadata(
                    cache_seqlens_int32=cache_seqlens,
                    page_table=self.cuda_graph_page_table[:bs, :],
                    max_seq_len_k=self.max_context_len,
                )
                self.cuda_graph_decode_metadata[bs] = metadata
                self.forward_decode_metadata = metadata
        else:
            metadata = MHAMetadata(
                cache_seqlens_int32=cache_seqlens,
                page_table=self.cuda_graph_page_table[:bs, :],
                max_seq_len_k=self.max_context_len,
            )
            self.cuda_graph_decode_metadata[bs] = metadata
            self.forward_decode_metadata = metadata

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

        if forward_mode.is_extend_or_mixed():
            raise NotImplementedError(
                f"mha CUDA graph replay not supported for {forward_mode}"
            )

        if bs in self.cuda_graph_prefill_metadata:
            self.forward_prefill_metadata = self.cuda_graph_prefill_metadata[bs]
        if bs in self.cuda_graph_decode_metadata:
            self.forward_decode_metadata = self.cuda_graph_decode_metadata[bs]

    def forward_decode(
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
        if layer.qk_head_dim != layer.v_head_dim:
            raise NotImplementedError("mha backend requires qk_head_dim == v_head_dim")

        # Multi-token decode (q_len > 1) reuses the prefill kernel via the
        # uniform-stride prefill slot; plain decode uses the single-token slot.
        q_len_per_req = q.shape[0] // bs if bs > 0 else 1
        if q_len_per_req > 1:
            return self._forward_extend(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                self.forward_prefill_metadata,
                save_kv_cache=save_kv_cache,
                sinks=kwargs.get("sinks"),
            )

        has_kv = k is not None
        if has_kv != (v is not None):
            raise ValueError("mha decode requires k and v to both be present or absent")

        if save_kv_cache and has_kv:
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

        # Precomputed scheduler metadata bakes in the canonical-decode
        # attention pattern (causal, no sliding window, no softcap, no
        # sinks). For layers that deviate, fall back to the FA3 kernel's
        # internal prepare_varlen path so attention output stays correct.
        sinks = kwargs.get("sinks")
        scheduler_metadata = (
            metadata.scheduler_metadata
            if (layer.sliding_window_size < 0 and not layer.logit_cap and sinks is None)
            else None
        )

        result = mha_decode_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens_int32,
            softmax_scale=layer.scaling,
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            sinks=sinks,
            max_seqlen_k=metadata.max_seq_len_k,
            scheduler_metadata=scheduler_metadata,
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
        bs: int,
        save_kv_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if layer.qk_head_dim != layer.v_head_dim:
            raise NotImplementedError("mha backend requires qk_head_dim == v_head_dim")

        metadata = self.forward_prefill_metadata
        has_kv = k is not None
        if has_kv != (v is not None):
            raise ValueError("mha extend requires k and v to both be present or absent")

        if has_kv:
            if metadata.has_prefix:
                if self.mha_extend_mode == "ragged":
                    return self._forward_split_prefill(
                        q,
                        k,
                        v,
                        layer,
                        out_cache_loc,
                        token_to_kv_pool,
                        metadata,
                        save_kv_cache,
                        kwargs.get("sinks"),
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
                        kwargs.get("sinks"),
                    )
            return self._forward_prefill(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                metadata,
                save_kv_cache,
                kwargs.get("sinks"),
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
                kwargs.get("sinks"),
            )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        metadata: MHAMetadata,
        save_kv_cache: bool,
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        cu_seqlens_q = metadata.cu_seqlens_q
        assert cu_seqlens_q is not None
        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
        result = mha_prefill(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=metadata.max_seq_len_q,
            max_seqlen_k=metadata.max_seq_len_q,
            softmax_scale=layer.scaling,
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

    def _forward_split_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        metadata: MHAMetadata,
        save_kv_cache: bool,
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        assert metadata.prefix_seqlens_int32 is not None
        assert metadata.max_prefix_seq_len is not None
        cu_seqlens_q = metadata.cu_seqlens_q
        assert cu_seqlens_q is not None
        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

        chunk_result = mha_prefill(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=metadata.max_seq_len_q,
            max_seqlen_k=metadata.max_seq_len_q,
            softmax_scale=layer.scaling,
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            sinks=sinks,
            return_lse=True,
            solution=self.kernel_solution,
        )
        chunk_out, chunk_lse = chunk_result

        k_cache, v_cache = self._get_kv_cache(layer, token_to_kv_pool)
        prefix_result = mha_extend_with_kvcache(
            q=q,
            cu_seqlens_q=cu_seqlens_q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=metadata.page_table,
            cache_seqlens=metadata.prefix_seqlens_int32,
            softmax_scale=layer.scaling,
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            return_lse=True,
            max_seqlen_q=metadata.max_seq_len_q,
            max_seqlen_k=metadata.max_prefix_seq_len,
            solution=self.kernel_solution,
        )
        prefix_out, prefix_lse = prefix_result

        output, _ = mha_merge_state(
            chunk_out.contiguous(),
            chunk_lse.contiguous(),
            prefix_out.contiguous(),
            prefix_lse.contiguous(),
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
        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        metadata: MHAMetadata,
        save_kv_cache: bool,
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        cu_seqlens_q = metadata.cu_seqlens_q
        assert cu_seqlens_q is not None
        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k = None if k is None else k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v = None if v is None else v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
        has_kv = k is not None
        if save_kv_cache and has_kv:
            token_to_kv_pool.set_kv_buffer(
                layer,
                out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )
        elif save_kv_cache:
            raise ValueError("mha extend requires KV when save_kv_cache=True")
        elif has_kv:
            raise ValueError("mha_extend_with_kvcache requires KV to be prewritten")

        k_cache, v_cache = self._get_kv_cache(layer, token_to_kv_pool)
        result = mha_extend_with_kvcache(
            q=q,
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
