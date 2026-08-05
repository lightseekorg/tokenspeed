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
from tokenspeed_kernel import (
    mla_decode_with_kvcache,
    mla_extend_with_kvcache,
    mla_prefill,
    mla_use_absorbed_extend,
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
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    cache_debug_enabled,
)
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.layers.attention.utils import build_page_table
from tokenspeed.runtime.utils.common import ceil_div

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


@dataclass(kw_only=True)
class MLAPrefillMetadata:
    # Device-side metadata for explicit Q/K/V MLA prefill and prefix replay.
    seq_lens: torch.Tensor
    req_pool_indices: torch.Tensor
    extend_prefix_lens: torch.Tensor
    extend_seq_lens: torch.Tensor
    cum_extend_seq_lens: torch.Tensor
    cum_seq_lens_kv: torch.Tensor | None
    page_table: torch.Tensor | None
    # Host-side metadata.
    extend_seq_lens_cpu: list[int]
    max_extend_seq_len: int
    max_extend_prefix_len: int
    use_absorbed_cached_extend: bool
    # Per-prefix-chunk arrays consumed by DeepSeek's chunked prefix replay.
    chunked_loop_num: int
    chunk_kv_indices_list: list[torch.Tensor]
    chunked_seq_len: torch.Tensor
    cu_chunked_seq_len: torch.Tensor
    max_chunk_len_per_loop: list[int]
    # Paged cache only: absolute latent locations for model-owned extend writes.
    group_out_cache_loc: torch.Tensor | None = None


@dataclass(kw_only=True)
class MLADecodeMetadata:
    # num_extends lets mixed batches slice decode requests after extend requests.
    num_extends: int
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    # Paged cache only: absolute latent write locations, request-major, with
    # ``group_q_len_per_req`` entries per row (1 outside target verify).
    group_out_cache_loc: torch.Tensor | None = None
    group_q_len_per_req: int = 1

    @property
    def block_kv_indices(self) -> torch.Tensor:
        return self.page_table

    @property
    def seq_lens_k(self) -> torch.Tensor:
        return self.seq_lens


class MLAAttnBackend(MlaCacheGroupMixin, AttentionBackend):
    """Unified MLA backend routed through tokenspeed_kernel MLA APIs."""

    supports_mla_projected_value_decode = True

    def __init__(self, config: MLAConfig):
        super().__init__(config)

        self._cache_groups_bound = False
        self._cache_contract_bound = False
        self.max_context_len = config.context_len
        self.page_size = config.page_size
        self.max_num_pages = ceil_div(self.max_context_len, self.page_size)

        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_cache_dim = config.kv_cache_dim
        self.scaling = config.scaling
        self.data_type = config.kv_cache_dtype
        self.q_data_type = config.dtype
        self.num_local_heads = config.num_attention_heads // config.attn_tp_size

        self.kernel_solution = None

        self.forward_decode_metadata: MLADecodeMetadata | None = None
        self.forward_prefill_metadata: MLAPrefillMetadata | None = None
        self.chunked_prefill_metadata: MLAPrefillMetadata | None = None
        self.decode_cuda_graph_metadata: dict[int, MLADecodeMetadata] = {}
        self.cuda_graph_page_table: torch.Tensor | None = None
        self.cuda_graph_seq_lens: torch.Tensor | None = None
        self.decode_cuda_graph_group_out_cache_loc: torch.Tensor | None = None

    def _should_use_absorbed_cached_extend(
        self, *, max_extend_seq_len: int, max_extend_prefix_len: int
    ) -> bool:
        return max_extend_prefix_len > 0 and mla_use_absorbed_extend(
            q_dtype=self.q_data_type,
            kv_dtype=self.data_type,
            num_q_heads=self.num_local_heads,
            page_size=self.page_size,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            max_seqlen_q=max_extend_seq_len,
            solution=self.kernel_solution,
        )

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        forward_mode: ForwardMode,
        extend_seq_lens: torch.Tensor | None = None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
        extend_prefix_lens: torch.Tensor | None = None,
        extend_prefix_lens_cpu: torch.Tensor | None = None,
        **kwargs,
    ):
        cache_metadata = kwargs.pop("cache_metadata", None)
        forward_batch = kwargs.pop("forward_batch", None)
        group_table = None
        logical_page_size = None
        if cache_metadata is not None:
            if self.is_draft:
                raise NotImplementedError(
                    "MLA draft worker does not take the Paged cache path"
                )
            self._cache_groups_bound = True
            group_table, logical_page_size = self._resolve_full_history_table(
                cache_metadata, forward_batch, bs
            )
            if cache_debug_enabled():
                self._validate_live_pages(group_table, seq_lens[:bs], logical_page_size)
        elif self._cache_groups_bound and bs > 0 and not forward_mode.is_idle():
            raise RuntimeError(
                "MLAAttnBackend is bound to Paged cache but received no paged cache "
                "metadata; refusing the legacy page_table path"
            )

        if forward_mode.is_extend_or_mixed():
            self._init_prefill_metadata(
                seq_lens=seq_lens[:num_extends],
                req_pool_indices=req_pool_indices[:num_extends],
                page_table=page_table,
                extend_prefix_lens=extend_prefix_lens[:num_extends],
                extend_prefix_lens_cpu=extend_prefix_lens_cpu[:num_extends],
                extend_seq_lens=extend_seq_lens[:num_extends],
                extend_seq_lens_cpu=extend_seq_lens_cpu[:num_extends],
                group_table=group_table,
                logical_page_size=logical_page_size,
            )

        if (
            forward_mode.is_decode()
            or forward_mode.is_mixed()
            or (forward_mode.is_extend() and self.is_draft)
        ):
            self._init_decode_metadata(
                bs=bs,
                num_extends=num_extends,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                page_table=page_table,
                group_table=group_table,
                logical_page_size=logical_page_size,
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

    def _init_prefill_metadata(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        page_table: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        group_table: torch.Tensor | None = None,
        logical_page_size: int | None = None,
    ):
        extend_seq_lens_cpu_list = [int(x) for x in extend_seq_lens_cpu.tolist()]
        cum_extend_seq_lens = torch.zeros(
            extend_seq_lens.shape[0] + 1,
            device=self.device,
            dtype=torch.int32,
        )
        torch.cumsum(extend_seq_lens, dim=0, out=cum_extend_seq_lens[1:])

        max_extend_seq_len = max(extend_seq_lens_cpu_list, default=0)
        max_extend_prefix_len = int(extend_prefix_lens_cpu.max().item())
        use_absorbed_cached_extend = self._should_use_absorbed_cached_extend(
            max_extend_seq_len=max_extend_seq_len,
            max_extend_prefix_len=max_extend_prefix_len,
        )

        cum_seq_lens_kv = None
        absorbed_page_table = None
        if use_absorbed_cached_extend:
            cum_seq_lens_kv = torch.zeros_like(cum_extend_seq_lens)
            torch.cumsum(seq_lens, dim=0, out=cum_seq_lens_kv[1:])

        if group_table is not None:
            assert logical_page_size is not None
            group_out_cache_loc = self._extend_out_cache_loc(
                group_table[: seq_lens.shape[0]],
                extend_prefix_lens_cpu,
                extend_seq_lens_cpu,
                logical_page_size=logical_page_size,
                validate_pages=cache_debug_enabled(),
            )
            chunk_page_table = group_table[: seq_lens.shape[0]]
            chunk_req_pool_indices = torch.arange(
                seq_lens.shape[0], dtype=torch.int64, device=group_table.device
            )
            chunk_page_size = logical_page_size
            if use_absorbed_cached_extend:
                absorbed_page_table = self._expand_group_page_table(
                    group_table,
                    batch_size=seq_lens.shape[0],
                    logical_page_size=logical_page_size,
                )
        else:
            # Idle/warmup placeholder: page_table is batch-ordered (row i ==
            # batch position i), so identity row indices apply.
            group_out_cache_loc = None
            chunk_page_table = page_table
            chunk_req_pool_indices = torch.arange(
                seq_lens.shape[0], dtype=torch.int64, device=page_table.device
            )
            chunk_page_size = self.page_size
            if use_absorbed_cached_extend:
                absorbed_page_table = build_page_table(
                    req_pool_indices,
                    page_table,
                    self.page_size,
                    self.max_context_len,
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
            chunk_page_table,
            chunk_req_pool_indices,
            chunk_page_size,
        )

        metadata = MLAPrefillMetadata(
            seq_lens=seq_lens,
            req_pool_indices=req_pool_indices,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
            cum_extend_seq_lens=cum_extend_seq_lens,
            cum_seq_lens_kv=cum_seq_lens_kv,
            page_table=absorbed_page_table,
            extend_seq_lens_cpu=extend_seq_lens_cpu_list,
            max_extend_seq_len=max_extend_seq_len,
            max_extend_prefix_len=max_extend_prefix_len,
            use_absorbed_cached_extend=use_absorbed_cached_extend,
            chunked_loop_num=chunked_loop_num,
            chunk_kv_indices_list=chunk_kv_indices_list,
            chunked_seq_len=chunked_seq_len,
            cu_chunked_seq_len=cu_chunked_seq_len,
            max_chunk_len_per_loop=max_chunk_len_per_loop,
            group_out_cache_loc=group_out_cache_loc,
        )
        self.forward_prefill_metadata = metadata
        self.chunked_prefill_metadata = metadata

    def _init_decode_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        group_table: torch.Tensor | None = None,
        logical_page_size: int | None = None,
        q_len_per_req: int = 1,
    ):
        if group_table is not None:
            assert logical_page_size is not None
            page_table = self._expand_group_page_table(
                group_table,
                batch_size=bs,
                logical_page_size=logical_page_size,
            )
            group_out_cache_loc = self._cache_decode_out_cache_loc(
                group_table,
                seq_lens,
                batch_size=bs,
                logical_page_size=logical_page_size,
                validate_pages=cache_debug_enabled(),
                q_len_per_req=q_len_per_req,
            )
        else:
            page_table = build_page_table(
                req_pool_indices[:bs],
                page_table,
                self.page_size,
                self.max_context_len,
            )
            group_out_cache_loc = None
        self.forward_decode_metadata = MLADecodeMetadata(
            num_extends=num_extends,
            page_table=page_table,
            seq_lens=seq_lens[:bs],
            group_out_cache_loc=group_out_cache_loc,
            group_q_len_per_req=q_len_per_req,
        )

    def select_out_cache_loc(self, layer, out_cache_loc, forward_mode=None):
        if (
            not self._cache_groups_bound
            or forward_mode is None
            or forward_mode.is_idle()
        ):
            return out_cache_loc
        if forward_mode.is_decode():
            metadata = self.forward_decode_metadata
            if metadata is None or metadata.group_out_cache_loc is None:
                raise RuntimeError("MLA decode write locations are missing")
            locs = metadata.group_out_cache_loc[
                metadata.num_extends * metadata.group_q_len_per_req :
            ]
        else:
            metadata = self.forward_prefill_metadata
            if metadata is None or metadata.group_out_cache_loc is None:
                raise RuntimeError("MLA prefill write locations are missing")
            locs = metadata.group_out_cache_loc
        if out_cache_loc is not None and locs.shape[0] != out_cache_loc.shape[0]:
            raise RuntimeError(
                f"MLA write locations cover {locs.shape[0]} tokens but "
                f"the caller provided {out_cache_loc.shape[0]}"
            )
        return locs

    def init_cuda_graph_state(self, max_bs: int):
        self.cuda_graph_page_table = torch.zeros(
            (max_bs, self.max_num_pages), dtype=torch.int32, device=self.device
        )
        # Own the cache-seqlens buffer; replay copies the live lengths in, so
        # graph state does not depend on the controller mutating a shared tensor.
        self.cuda_graph_seq_lens = torch.zeros(
            max_bs, dtype=torch.int32, device=self.device
        )
        self.decode_cuda_graph_metadata = {}
        if self._cache_contract_bound:
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
        if forward_mode.is_extend_or_mixed():
            raise NotImplementedError(
                f"mla CUDA graph capture not supported for {forward_mode}"
            )

        uses_cache_groups = bool(cache_group_ids) or self._cache_contract_bound
        if uses_cache_groups and self.is_draft:
            raise NotImplementedError(
                "MLA draft worker does not take the Paged cache path"
            )
        page_table = self.cuda_graph_page_table[:bs, :]
        capture_q_len = self._graph_verify_q_len()
        if uses_cache_groups:
            self._cache_groups_bound = True
            if self.decode_cuda_graph_group_out_cache_loc is None:
                raise RuntimeError(
                    "MLA Paged cache graph capture buffer was not allocated; "
                    "mark_cache_contract must run before init_cuda_graph_state"
                )
            page_table.zero_()
            group_out_cache_loc = self.decode_cuda_graph_group_out_cache_loc[
                : bs * capture_q_len
            ]
            group_out_cache_loc.zero_()
        else:
            group_out_cache_loc = None
        metadata = MLADecodeMetadata(
            num_extends=0,
            page_table=page_table,
            seq_lens=self.cuda_graph_seq_lens[:bs],
            group_out_cache_loc=group_out_cache_loc,
            group_q_len_per_req=capture_q_len,
        )
        # Seed the owned buffer: the capture run reads it before replay. Verify
        # rows span seq-N..seq-1, so a shorter length would start before zero.
        if capture_q_len > 1:
            metadata.seq_lens.copy_(seq_lens[:bs].clamp_min(capture_q_len))
        else:
            metadata.seq_lens.copy_(seq_lens[:bs])
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
        if forward_mode is not None and forward_mode.is_extend_or_mixed():
            raise NotImplementedError(
                f"mla CUDA graph replay not supported for {forward_mode}"
            )

        metadata = self.decode_cuda_graph_metadata[bs]
        # Copy the live lengths into our own cache-seqlens buffer (metadata.seq_lens
        # views it); both metadata paths read it at replay.
        q_len = metadata.group_q_len_per_req
        if q_len > 1:
            self.cuda_graph_seq_lens[:bs].copy_(seq_lens[:bs].clamp_min(q_len))
        else:
            self.cuda_graph_seq_lens[:bs].copy_(seq_lens[:bs])
        if metadata.group_out_cache_loc is not None:
            self._replay_refresh_decode(
                bs,
                seq_lens,
                metadata,
                kwargs.get("cache_metadata"),
                kwargs.get("forward_batch"),
            )
        else:
            # Idle/warmup replay before the backend binds, or a draft driven with
            # the batch-ordered draft page table (row i == batch position i).
            self.cuda_graph_page_table[:bs, : self.max_num_pages].copy_(
                page_table[:bs, : self.max_num_pages]
            )
        self.forward_decode_metadata = metadata

    def _replay_refresh_decode(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        metadata: MLADecodeMetadata,
        cache_metadata,
        forward_batch,
    ) -> None:
        if metadata.group_out_cache_loc is None:
            raise RuntimeError("MLA graph metadata has no write-location buffer")
        q_len = metadata.group_q_len_per_req
        real_bs = 0
        if cache_metadata is not None:
            table, logical_page_size = self._resolve_full_history_table(
                cache_metadata, forward_batch, 0
            )
            real_bs = min(int(table.shape[0]), bs)
            if real_bs > 0:
                self._expand_group_page_table(
                    table,
                    batch_size=real_bs,
                    logical_page_size=logical_page_size,
                    out=metadata.page_table,
                )
                self._cache_decode_out_cache_loc(
                    table,
                    seq_lens,
                    batch_size=real_bs,
                    logical_page_size=logical_page_size,
                    validate_pages=cache_debug_enabled(),
                    out=metadata.group_out_cache_loc,
                    q_len_per_req=q_len,
                )
        metadata.page_table[real_bs:bs].zero_()
        metadata.group_out_cache_loc[real_bs * q_len : bs * q_len].zero_()

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

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
        # q is absorbed MLA query [T, H, R + D_rope]; k is compressed KV
        # [T, 1, R + D_rope]. DeepSeek normally writes cache before this call.
        if save_kv_cache:
            assert k is not None
            out_cache_loc = self.select_out_cache_loc(
                layer,
                out_cache_loc,
                kwargs.get("forward_mode", ForwardMode.DECODE),
            )
            token_to_kv_pool.set_mla_kv_buffer(
                layer,
                out_cache_loc,
                k[..., : self.kv_lora_rank],
                k[..., self.kv_lora_rank :],
            )

        metadata = self.forward_decode_metadata
        assert metadata is not None
        num_extends = metadata.num_extends
        q_len_per_req = q.shape[0] // bs if bs > 0 else 1

        if q_len_per_req > 1:
            query = q.view(-1, layer.tp_q_head_num, layer.head_dim).unsqueeze(1)
            page_table = metadata.page_table[num_extends:].repeat_interleave(
                q_len_per_req, dim=0
            )
            cache_seqlens = metadata.seq_lens[num_extends:].repeat_interleave(
                q_len_per_req
            )
            # Draft catch-up starts from the current draft KV length; target
            # verify starts from the final target KV length and backs up.
            offset_start = 0 if self.is_draft else 1 - q_len_per_req
            offsets = torch.arange(
                offset_start,
                offset_start + q_len_per_req,
                device=cache_seqlens.device,
                dtype=cache_seqlens.dtype,
            ).repeat(bs)
            cache_seqlens = cache_seqlens + offsets
            max_seqlen_k = self.max_context_len
        else:
            query = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
            page_table = metadata.page_table[num_extends:]
            cache_seqlens = metadata.seq_lens[num_extends:]
            max_seqlen_k = self.max_context_len

        softmax_scale = layer.scaling
        if self.data_type in (torch.float8_e4m3fn, torch.float8_e5m2):
            k_scale = (
                layer.k_scale_float
                if getattr(layer, "k_scale_float", None) is not None
                else 1.0
            )
            softmax_scale = k_scale * softmax_scale

        kv_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)
        if self.data_type != kv_cache.dtype:
            kv_cache = kv_cache.to(self.data_type)
        kv_cache = kv_cache.view(-1, self.page_size, 1, self.kv_cache_dim)

        value_weight = kwargs.get("value_weight")
        gate = kwargs.get("output_gate")
        projected_out = kwargs.get("projected_output")
        if value_weight is not None:
            result = mla_decode_with_kvcache(
                query,
                kv_cache,
                page_table,
                cache_seqlens,
                max_seqlen_k,
                self.qk_nope_head_dim,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                softmax_scale,
                value_weight=value_weight,
                gate=gate,
                out=projected_out,
                logit_cap=layer.logit_cap,
            )
        else:
            result = mla_decode_with_kvcache(
                q=query,
                kv_cache=kv_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                max_seqlen_k=max_seqlen_k,
                qk_nope_head_dim=self.qk_nope_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                softmax_scale=softmax_scale,
                logit_cap=layer.logit_cap,
                solution=self.kernel_solution,
            )
        output = self._unwrap_output(result)
        if value_weight is not None:
            return output.reshape(-1, value_weight.shape[0] * value_weight.shape[2])
        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

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
        if save_kv_cache:
            raise NotImplementedError(
                "MLA forward_extend cannot derive compressed cache rows from "
                "materialized K/V; DeepSeek writes MLA cache in the model path"
            )

        metadata = self.forward_prefill_metadata
        assert metadata is not None
        if metadata.use_absorbed_cached_extend:
            assert metadata.page_table is not None
            assert metadata.cum_seq_lens_kv is not None
            q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
            kv_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)
            if self.data_type != kv_cache.dtype:
                kv_cache = kv_cache.to(self.data_type)
            kv_cache = kv_cache.view(-1, self.page_size, 1, self.kv_cache_dim)
            result = mla_extend_with_kvcache(
                q=q,
                kv_cache=kv_cache,
                page_table=metadata.page_table,
                cache_seqlens=metadata.seq_lens,
                cu_seqlens_q=metadata.cum_extend_seq_lens,
                cu_seqlens_kv=metadata.cum_seq_lens_kv,
                max_seqlen_q=metadata.max_extend_seq_len,
                max_seqlen_k=self.max_context_len,
                qk_nope_head_dim=self.qk_nope_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                softmax_scale=layer.scaling,
                is_causal=True,
                logit_cap=layer.logit_cap,
                solution=self.kernel_solution,
            )
            output = self._unwrap_output(result)
            return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

        if metadata.max_extend_prefix_len > 0:
            raise NotImplementedError(
                "MLA prefix-cache extend is handled by DeepSeek's chunked "
                "prefix replay path via forward_extend_chunked"
            )

        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
        result = mla_prefill(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=metadata.cum_extend_seq_lens,
            cu_seqlens_kv=metadata.cum_extend_seq_lens,
            max_seqlen_q=metadata.max_extend_seq_len,
            max_seqlen_kv=metadata.max_extend_seq_len,
            softmax_scale=layer.scaling,
            seq_lens_kv=metadata.extend_seq_lens,
            is_causal=True,
            logit_cap=layer.logit_cap,
            solution=self.kernel_solution,
        )
        output = self._unwrap_output(result)
        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

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
            step_counter = getattr(self, "step_counter", None)
            if step_counter is not None:
                step_counter.record_cache()

        head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        q = q.reshape(-1, self.num_local_heads, head_dim)
        k = k.reshape(-1, self.num_local_heads, head_dim)
        v = v.reshape(-1, self.num_local_heads, self.v_head_dim)

        if q.dtype == torch.float8_e4m3fn:
            k = k.to(torch.float8_e4m3fn)
            v = v.to(torch.float8_e4m3fn)

        result = mla_prefill(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cum_seq_lens_q,
            cu_seqlens_kv=cum_seq_lens_kv,
            max_seqlen_q=max_q_len,
            max_seqlen_kv=max_kv_len,
            softmax_scale=scaling,
            seq_lens_kv=seq_lens,
            is_causal=causal,
            logit_cap=logits_soft_cap or 0.0,
            return_lse=True,
            out=out,
            solution=self.kernel_solution,
        )

        if isinstance(result, tuple):
            return result[0], result[1]
        return result, None

    def _unwrap_output(self, result):
        if isinstance(result, tuple):
            return result[0]
        return result


register_backend("mla", {AttentionArch.MLA}, MLAAttnBackend)
