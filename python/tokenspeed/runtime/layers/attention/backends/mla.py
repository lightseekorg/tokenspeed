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
from tokenspeed_kernel import mla_decode_with_kvcache, mla_prefill

from tokenspeed.runtime.configs.flat_cache_runtime import flat_cache_debug_enabled
from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.chunk import (
    build_chunked_prefill_metadata_arrays,
)
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
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
    # Host-side metadata.
    extend_seq_lens_cpu: list[int]
    max_extend_seq_len: int
    max_extend_prefix_len: int
    # Per-prefix-chunk arrays consumed by DeepSeek's chunked prefix replay.
    chunked_loop_num: int
    chunk_kv_indices_list: list[torch.Tensor]
    chunked_seq_len: torch.Tensor
    cu_chunked_seq_len: torch.Tensor
    max_chunk_len_per_loop: list[int]
    # FlatKV only: absolute latent locations for model-owned extend writes.
    flat_out_cache_loc: torch.Tensor | None = None


@dataclass(kw_only=True)
class MLADecodeMetadata:
    # num_extends lets mixed batches slice decode requests after extend requests.
    num_extends: int
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    # FlatKV only: one absolute latent write location per batch row.
    flat_out_cache_loc: torch.Tensor | None = None

    @property
    def block_kv_indices(self) -> torch.Tensor:
        return self.page_table

    @property
    def seq_lens_k(self) -> torch.Tensor:
        return self.seq_lens


class MLAAttnBackend(AttentionBackend):
    """Unified MLA backend routed through tokenspeed_kernel MLA APIs."""

    flat_cache_consumer_families = frozenset({"history"})

    def __init__(self, config: MLAConfig):
        super().__init__(config)

        self._flat_bound = False
        self._flat_contract_bound = False
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
        self.decode_cuda_graph_flat_out_cache_loc: torch.Tensor | None = None

    def mark_flat_contract(self) -> None:
        """Enable FlatKV graph state before capture buffers are allocated."""
        self._flat_contract_bound = True

    def _resolve_flat_full_table(
        self, flat_cache_metadata, flat_cache_forward_op, bs: int
    ) -> tuple[torch.Tensor, int]:
        table = flat_cache_metadata.require_full_attention_table(
            active_forward_op=flat_cache_forward_op
        )
        if table.shape[0] < bs:
            raise RuntimeError(
                f"flat full-attention table has {table.shape[0]} rows but the "
                f"batch has {bs} requests"
            )
        flat_page_size = int(flat_cache_metadata.block_size)
        if flat_page_size <= 0 or flat_page_size % self.page_size:
            raise RuntimeError(
                f"flat page size {flat_page_size} is not a positive multiple "
                f"of the MLA kernel page size {self.page_size}"
            )
        if table.stride(0) != table.shape[1] and table.shape[0] > 1:
            table = table.contiguous()
        return table, flat_page_size

    @staticmethod
    def _validate_flat_live_pages(
        table: torch.Tensor, seq_lens: torch.Tensor, flat_page_size: int
    ) -> None:
        """Reject null or missing FlatKV pages inside each request's live range."""
        if table.numel() == 0 or seq_lens.numel() == 0:
            return
        batch_size = seq_lens.shape[0]
        live_pages = (
            (seq_lens.to(torch.int64) + flat_page_size - 1) // flat_page_size
        ).clamp_max_(table.shape[1])
        columns = torch.arange(table.shape[1], device=table.device)
        live_entries = table[:batch_size][
            columns.unsqueeze(0) < live_pages.unsqueeze(1)
        ]
        if not bool((live_entries > 0).all().item()):
            raise RuntimeError(
                "flat full-attention table contains -1 or the null page 0 "
                "inside a live range"
            )

    def _flat_expand_page_table(
        self,
        table: torch.Tensor,
        *,
        batch_size: int,
        flat_page_size: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Expand FlatKV scheduler pages for this backend's MLA kernel pages."""
        ratio = flat_page_size // self.page_size
        if ratio <= 0 or flat_page_size % self.page_size:
            raise ValueError(
                "flat_page_size must be a positive multiple of MLA kernel page size"
            )
        if out is None:
            out = torch.zeros(
                (batch_size, self.max_num_pages),
                dtype=torch.int32,
                device=table.device,
            )
        flat_columns = min(ceil_div(self.max_num_pages, ratio), table.shape[1])
        if flat_columns <= 0:
            out[:batch_size].zero_()
            return out
        expanded = (
            table[:batch_size, :flat_columns].clamp_min(0).to(torch.int32).unsqueeze(-1)
            * ratio
            + torch.arange(ratio, dtype=torch.int32, device=table.device)
        ).reshape(batch_size, flat_columns * ratio)
        copy_len = min(self.max_num_pages, expanded.shape[1])
        out[:batch_size, :copy_len].copy_(expanded[:, :copy_len])
        if copy_len < out.shape[1]:
            out[:batch_size, copy_len:].zero_()
        return out

    @staticmethod
    def _flat_decode_out_cache_loc(
        table: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        batch_size: int,
        flat_page_size: int,
        validate_pages: bool = False,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return absolute cache locations for decoded tokens in FlatKV."""
        positions = (seq_lens[:batch_size].to(torch.int64) - 1).clamp_min(0)
        page_indices = torch.div(positions, flat_page_size, rounding_mode="floor")
        pages = table[:batch_size].gather(1, page_indices.unsqueeze(1)).squeeze(1)
        if validate_pages and pages.numel() and not bool((pages > 0).all().item()):
            raise RuntimeError(
                "flat MLA write location resolves to the null page 0 or a "
                "-1 table hole"
            )
        locations = pages.clamp_min(0).to(torch.int64) * flat_page_size + (
            positions % flat_page_size
        )
        if out is not None:
            out[:batch_size].copy_(locations)
            return out
        return locations

    @staticmethod
    def _flat_extend_out_cache_loc(
        table: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        *,
        flat_page_size: int,
        validate_pages: bool = False,
    ) -> torch.Tensor:
        """Return packed FlatKV extend-write locations in query order."""
        chunks: list[torch.Tensor] = []
        pages_for_validation: list[torch.Tensor] = []
        for row, (start, num_new) in enumerate(
            zip(
                extend_prefix_lens_cpu.tolist(),
                extend_seq_lens_cpu.tolist(),
                strict=True,
            )
        ):
            start, num_new = int(start), int(num_new)
            if num_new <= 0:
                continue
            max_column = (start + num_new - 1) // flat_page_size
            if max_column >= table.shape[1]:
                raise RuntimeError(
                    "flat extend write locations exceed the full-attention "
                    f"table: row={row}, prefix={start}, new={num_new}, "
                    f"flat_page_size={flat_page_size}, columns={table.shape[1]}"
                )
            positions = torch.arange(
                start, start + num_new, dtype=torch.int64, device=table.device
            )
            pages = table[row].gather(0, positions // flat_page_size)
            pages_for_validation.append(pages)
            chunks.append(
                pages.to(torch.int64) * flat_page_size + positions % flat_page_size
            )
        if not chunks:
            return torch.empty(0, dtype=torch.int64, device=table.device)
        if validate_pages and not bool(
            (torch.cat(pages_for_validation) > 0).all().item()
        ):
            raise RuntimeError(
                "flat MLA write location resolves to the null page 0 or a "
                "-1 table hole"
            )
        return torch.cat(chunks)

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_page: torch.Tensor,
        forward_mode: ForwardMode,
        extend_seq_lens: torch.Tensor | None = None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
        extend_prefix_lens: torch.Tensor | None = None,
        extend_prefix_lens_cpu: torch.Tensor | None = None,
        **kwargs,
    ):
        flat_cache_metadata = kwargs.pop("flat_cache_metadata", None)
        flat_cache_forward_op = kwargs.pop("flat_cache_forward_op", None)
        flat_table = None
        flat_page_size = None
        if flat_cache_metadata is not None:
            if self.is_draft or self.spec_num_tokens > 1:
                raise NotImplementedError(
                    "MLA FlatKV does not support speculative decoding"
                )
            self._flat_bound = True
            flat_table, flat_page_size = self._resolve_flat_full_table(
                flat_cache_metadata, flat_cache_forward_op, bs
            )
            if flat_cache_debug_enabled():
                self._validate_flat_live_pages(
                    flat_table, seq_lens[:bs], flat_page_size
                )
        elif self._flat_bound and bs > 0 and not forward_mode.is_idle():
            raise RuntimeError(
                "MLAAttnBackend is bound to FlatKV but received no flat cache "
                "metadata; refusing the legacy req_to_page path"
            )

        if forward_mode.is_extend_or_mixed():
            self._init_prefill_metadata(
                seq_lens=seq_lens[:num_extends],
                req_pool_indices=req_pool_indices[:num_extends],
                req_to_page=req_to_page,
                extend_prefix_lens=extend_prefix_lens[:num_extends],
                extend_prefix_lens_cpu=extend_prefix_lens_cpu[:num_extends],
                extend_seq_lens=extend_seq_lens[:num_extends],
                extend_seq_lens_cpu=extend_seq_lens_cpu[:num_extends],
                flat_table=flat_table,
                flat_page_size=flat_page_size,
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
                req_to_page=req_to_page,
                flat_table=flat_table,
                flat_page_size=flat_page_size,
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
        req_to_page: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        flat_table: torch.Tensor | None = None,
        flat_page_size: int | None = None,
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

        if flat_table is not None:
            assert flat_page_size is not None
            flat_out_cache_loc = self._flat_extend_out_cache_loc(
                flat_table[: seq_lens.shape[0]],
                extend_prefix_lens_cpu,
                extend_seq_lens_cpu,
                flat_page_size=flat_page_size,
                validate_pages=flat_cache_debug_enabled(),
            )
            chunk_req_to_page = flat_table[: seq_lens.shape[0]]
            chunk_req_pool_indices = torch.arange(
                seq_lens.shape[0], dtype=torch.int64, device=flat_table.device
            )
            chunk_page_size = flat_page_size
        else:
            flat_out_cache_loc = None
            chunk_req_to_page = req_to_page
            chunk_req_pool_indices = req_pool_indices
            chunk_page_size = self.page_size

        (
            chunked_loop_num,
            chunk_kv_indices_list,
            chunked_seq_len,
            cu_chunked_seq_len,
            max_chunk_len_per_loop,
        ) = build_chunked_prefill_metadata_arrays(
            extend_prefix_lens,
            extend_prefix_lens_cpu,
            chunk_req_to_page,
            chunk_req_pool_indices,
            chunk_page_size,
        )

        metadata = MLAPrefillMetadata(
            seq_lens=seq_lens,
            req_pool_indices=req_pool_indices,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
            cum_extend_seq_lens=cum_extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu_list,
            max_extend_seq_len=max_extend_seq_len,
            max_extend_prefix_len=max_extend_prefix_len,
            chunked_loop_num=chunked_loop_num,
            chunk_kv_indices_list=chunk_kv_indices_list,
            chunked_seq_len=chunked_seq_len,
            cu_chunked_seq_len=cu_chunked_seq_len,
            max_chunk_len_per_loop=max_chunk_len_per_loop,
            flat_out_cache_loc=flat_out_cache_loc,
        )
        self.forward_prefill_metadata = metadata
        self.chunked_prefill_metadata = metadata

    def _init_decode_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_page: torch.Tensor,
        flat_table: torch.Tensor | None = None,
        flat_page_size: int | None = None,
    ):
        if flat_table is not None:
            assert flat_page_size is not None
            page_table = self._flat_expand_page_table(
                flat_table,
                batch_size=bs,
                flat_page_size=flat_page_size,
            )
            flat_out_cache_loc = self._flat_decode_out_cache_loc(
                flat_table,
                seq_lens,
                batch_size=bs,
                flat_page_size=flat_page_size,
                validate_pages=flat_cache_debug_enabled(),
            )
        else:
            page_table = build_page_table(
                req_pool_indices[:bs],
                req_to_page,
                self.page_size,
                self.max_context_len,
            )
            flat_out_cache_loc = None
        self.forward_decode_metadata = MLADecodeMetadata(
            num_extends=num_extends,
            page_table=page_table,
            seq_lens=seq_lens[:bs],
            flat_out_cache_loc=flat_out_cache_loc,
        )

    def select_out_cache_loc(self, layer, out_cache_loc, forward_mode=None):
        if not self._flat_bound or forward_mode is None or forward_mode.is_idle():
            return out_cache_loc
        if forward_mode.is_decode():
            metadata = self.forward_decode_metadata
            if metadata is None or metadata.flat_out_cache_loc is None:
                raise RuntimeError("flat MLA decode write locations are missing")
            locs = metadata.flat_out_cache_loc[metadata.num_extends :]
        else:
            metadata = self.forward_prefill_metadata
            if metadata is None or metadata.flat_out_cache_loc is None:
                raise RuntimeError("flat MLA prefill write locations are missing")
            locs = metadata.flat_out_cache_loc
        if out_cache_loc is not None and locs.shape[0] != out_cache_loc.shape[0]:
            raise RuntimeError(
                f"flat MLA write locations cover {locs.shape[0]} tokens but "
                f"the caller provided {out_cache_loc.shape[0]}"
            )
        return locs

    def init_cuda_graph_state(self, max_bs: int, seq_lens_buf: torch.Tensor):
        assert (
            seq_lens_buf.dtype == torch.int32
            and seq_lens_buf.dim() == 1
            and seq_lens_buf.shape[0] >= max_bs
        ), (
            f"seq_lens_buf must be int32 with shape[0] >= {max_bs}, "
            f"got {seq_lens_buf.dtype} {tuple(seq_lens_buf.shape)}"
        )
        self.cuda_graph_page_table = torch.zeros(
            (max_bs, self.max_num_pages), dtype=torch.int32, device=self.device
        )
        self.cuda_graph_seq_lens = seq_lens_buf
        self.decode_cuda_graph_metadata = {}
        if self._flat_contract_bound:
            self.decode_cuda_graph_flat_out_cache_loc = torch.zeros(
                max_bs, dtype=torch.int64, device=self.device
            )
        else:
            self.decode_cuda_graph_flat_out_cache_loc = None

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        flat_cache_group_ids: tuple[str, ...] = (),
        **kwargs,
    ):
        if forward_mode.is_extend_or_mixed():
            raise NotImplementedError(
                f"mla CUDA graph capture not supported for {forward_mode}"
            )

        flat = bool(flat_cache_group_ids) or self._flat_contract_bound
        if flat and (self.is_draft or self.spec_num_tokens > 1):
            raise NotImplementedError(
                "MLA FlatKV CUDA graph capture does not support speculative decoding"
            )
        page_table = self.cuda_graph_page_table[:bs, :]
        if flat:
            self._flat_bound = True
            if self.decode_cuda_graph_flat_out_cache_loc is None:
                raise RuntimeError(
                    "MLA FlatKV graph capture buffer was not allocated; "
                    "mark_flat_contract must run before init_cuda_graph_state"
                )
            page_table.zero_()
            flat_out_cache_loc = self.decode_cuda_graph_flat_out_cache_loc[:bs]
            flat_out_cache_loc.zero_()
        else:
            flat_out_cache_loc = None
        metadata = MLADecodeMetadata(
            num_extends=0,
            page_table=page_table,
            seq_lens=self.cuda_graph_seq_lens[:bs],
            flat_out_cache_loc=flat_out_cache_loc,
        )
        self.decode_cuda_graph_metadata[bs] = metadata
        self.forward_decode_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = None,
        req_to_page: torch.Tensor = None,
        **kwargs,
    ):
        if forward_mode is not None and forward_mode.is_extend_or_mixed():
            raise NotImplementedError(
                f"mla CUDA graph replay not supported for {forward_mode}"
            )

        metadata = self.decode_cuda_graph_metadata[bs]
        if metadata.flat_out_cache_loc is not None:
            self._flat_replay_refresh_decode(
                bs,
                seq_lens,
                metadata,
                kwargs.get("flat_cache_metadata"),
                kwargs.get("flat_cache_forward_op"),
            )
        else:
            self.cuda_graph_page_table[:bs, : self.max_num_pages].copy_(
                req_to_page[req_pool_indices[:bs], : self.max_num_pages]
            )
        self.forward_decode_metadata = metadata

    def _flat_replay_refresh_decode(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        metadata: MLADecodeMetadata,
        flat_cache_metadata,
        flat_cache_forward_op,
    ) -> None:
        if metadata.flat_out_cache_loc is None:
            raise RuntimeError("flat MLA graph metadata has no write-location buffer")
        real_bs = 0
        if flat_cache_metadata is not None:
            table, flat_page_size = self._resolve_flat_full_table(
                flat_cache_metadata, flat_cache_forward_op, 0
            )
            real_bs = min(int(table.shape[0]), bs)
            if real_bs > 0:
                self._flat_expand_page_table(
                    table,
                    batch_size=real_bs,
                    flat_page_size=flat_page_size,
                    out=metadata.page_table,
                )
                self._flat_decode_out_cache_loc(
                    table,
                    seq_lens,
                    batch_size=real_bs,
                    flat_page_size=flat_page_size,
                    validate_pages=flat_cache_debug_enabled(),
                    out=metadata.flat_out_cache_loc,
                )
        metadata.page_table[real_bs:bs].zero_()
        metadata.flat_out_cache_loc[real_bs:bs].zero_()

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
