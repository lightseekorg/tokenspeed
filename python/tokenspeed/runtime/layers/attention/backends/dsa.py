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

import torch
from tokenspeed_kernel.ops.attention.flash_mla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    get_mla_metadata,
)
from tokenspeed_kernel.ops.attention.flashinfer import (
    trtllm_batch_decode_with_kv_cache_mla,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import error_fn

try:
    from tokenspeed_kernel.thirdparty import deep_gemm
except Exception:
    deep_gemm = None

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.trtllm_mla import TRTLLMMLABackend
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.layers.attention.registry import register_backend

_DSA_TRTLLM_WORKSPACE_BYTES = 384 * 1024 * 1024
_SPARSE_IMPL_FLASHMLA = "flashmla"
_SPARSE_IMPL_TRTLLM = "trtllm"
_dsa_trtllm_workspace_buffers: dict[torch.device, torch.Tensor] = {}


def _get_dsa_trtllm_workspace_buffer(device: torch.device | str) -> torch.Tensor:
    device = torch.device(device)
    workspace = _dsa_trtllm_workspace_buffers.get(device)
    if workspace is None:
        # Sparse prefill treats every query token as a decode row, so the
        # plain TRTLLM MLA workspace is too small for long prefill chunks.
        workspace = torch.zeros(
            _DSA_TRTLLM_WORKSPACE_BYTES,
            dtype=torch.uint8,
            device=device,
        )
        _dsa_trtllm_workspace_buffers[device] = workspace
    return workspace


def _flashmla_sparse_prefill_head_multiple() -> int:
    platform = current_platform()
    return 128 if platform.is_nvidia and platform.is_blackwell_plus else 64


def _flashmla_sparse_prefill_padded_heads(
    num_heads: int,
    head_multiple: int,
) -> int:
    if num_heads <= 0:
        raise RuntimeError(
            "DSA sparse prefill requires a positive query head count, "
            f"got {num_heads}."
        )
    return ((num_heads + head_multiple - 1) // head_multiple) * head_multiple


def _flashmla_sparse_decode_padded_heads(num_heads: int) -> int:
    if num_heads <= 0:
        raise RuntimeError(
            "DSA sparse decode requires a positive query head count, "
            f"got {num_heads}."
        )
    if num_heads <= 64:
        return 64
    if num_heads <= 128:
        return 128
    return num_heads


def _default_dsa_sparse_prefill_impl(kv_cache_dtype: torch.dtype | None = None) -> str:
    platform = current_platform()
    if (
        platform.is_nvidia
        and platform.is_blackwell
        and kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    ):
        return _SPARSE_IMPL_TRTLLM
    return _SPARSE_IMPL_FLASHMLA


def _workspace_indices_to_kv_slots(
    workspace_indices: torch.Tensor,
    kv_workspace_slots: torch.Tensor | None,
) -> torch.Tensor:
    if kv_workspace_slots is None:
        return workspace_indices.to(torch.int32)
    if workspace_indices.numel() == 0:
        return workspace_indices.to(torch.int32)

    flat_indices = workspace_indices.reshape(-1)
    valid = flat_indices >= 0
    flat_slots = flat_indices.to(torch.int64)
    if valid.any():
        flat_slots[valid] = kv_workspace_slots.to(
            device=workspace_indices.device,
            dtype=torch.int64,
        ).index_select(0, flat_slots[valid])
    return flat_slots.view_as(workspace_indices).to(torch.int32)


class DSABackend(AttentionBackend):
    """DSA backend for sparse MLA attention.

    Dense MLA metadata and dense attention calls are delegated to TRTLLMMLABackend.
    """

    def __init__(self, config: DSAConfig):
        super().__init__(config)
        self._dense_backend = TRTLLMMLABackend(config)
        self.index_topk = config.index_topk
        self.max_context_len = config.context_len
        self.page_size = config.page_size
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_cache_dim = config.kv_cache_dim
        self.scaling = config.scaling
        self.data_type = config.kv_cache_dtype
        self.q_data_type = config.dtype
        self.num_local_heads = config.num_attention_heads // config.attn_tp_size
        self.trtllm_workspace = _get_dsa_trtllm_workspace_buffer(config.device)
        self._prefill_workspace_buffer: torch.Tensor | None = None
        self._prefill_workspace_rows = 0
        self._prefill_workspace_dim = 0
        self._prefill_query_workspace: torch.Tensor | None = None
        self._decode_query_workspace: torch.Tensor | None = None
        self._prefill_query_workspace_num_heads: int | None = None
        self._decode_query_workspace_num_heads: int | None = None
        self._prefill_block_tables: torch.Tensor | None = None
        # Only graph-captured decode workspaces must survive a regrow. Eager
        # buffers are not referenced by replay and can be released normally.
        self._decode_query_workspace_captured = False
        self._sparse_prefill_impl = _default_dsa_sparse_prefill_impl(self.data_type)
        self._sparse_decode_impl = _SPARSE_IMPL_TRTLLM

    def _get_trtllm_workspace(self) -> torch.Tensor:
        workspace = getattr(self, "trtllm_workspace", None)
        if workspace is not None:
            return workspace
        dense_backend = getattr(self, "_dense_backend", None)
        workspace = getattr(dense_backend, "trtllm_workspace", None)
        if workspace is not None:
            self.trtllm_workspace = workspace
            return workspace
        raise RuntimeError("DSA TRTLLM sparse path requires a workspace buffer.")

    def _get_sparse_prefill_forward(self):
        name = getattr(
            self,
            "_sparse_prefill_impl",
            _default_dsa_sparse_prefill_impl(getattr(self, "data_type", None)),
        )
        if name == _SPARSE_IMPL_TRTLLM:
            return self._forward_sparse_prefill_trtllm
        if name == _SPARSE_IMPL_FLASHMLA:
            return self._forward_sparse_prefill_flashmla
        raise ValueError(
            f"Unknown DSA sparse prefill implementation: {name}. "
            f"Expected one of: {_SPARSE_IMPL_FLASHMLA}, {_SPARSE_IMPL_TRTLLM}."
        )

    def _get_sparse_decode_forward(self):
        name = getattr(self, "_sparse_decode_impl", _SPARSE_IMPL_TRTLLM)
        if name == _SPARSE_IMPL_TRTLLM:
            return self._forward_sparse_decode_trtllm
        if name == _SPARSE_IMPL_FLASHMLA:
            return self._forward_sparse_decode_flashmla
        raise ValueError(
            f"Unknown DSA sparse decode implementation: {name}. "
            f"Expected one of: {_SPARSE_IMPL_FLASHMLA}, {_SPARSE_IMPL_TRTLLM}."
        )

    @property
    def forward_decode_metadata(self):
        return self._dense_backend.forward_decode_metadata

    @property
    def forward_prefill_metadata(self):
        return self._dense_backend.forward_prefill_metadata

    @property
    def chunked_prefill_metadata(self):
        return self._dense_backend.chunked_prefill_metadata

    @property
    def decode_cuda_graph_metadata(self):
        return self._dense_backend.decode_cuda_graph_metadata

    @property
    def decode_cuda_graph_kv_indices(self):
        return self._dense_backend.decode_cuda_graph_kv_indices

    @decode_cuda_graph_kv_indices.setter
    def decode_cuda_graph_kv_indices(self, value):
        self._dense_backend.decode_cuda_graph_kv_indices = value

    @property
    def _block_table_aliased(self):
        return getattr(self._dense_backend, "_block_table_aliased", False)

    @_block_table_aliased.setter
    def _block_table_aliased(self, value):
        self._dense_backend._block_table_aliased = value

    def register_step_counter(self, step_counter):
        super().register_step_counter(step_counter)
        self._dense_backend.register_step_counter(step_counter)

    def override_num_extends(self, num_extends: int):
        return self._dense_backend.override_num_extends(num_extends)

    def init_cuda_graph_state(self, max_bs: int, seq_lens_buf: torch.Tensor):
        return self._dense_backend.init_cuda_graph_state(max_bs, seq_lens_buf)

    def _clear_metadata_caches(self) -> None:
        metadata_objects = [
            self.forward_decode_metadata,
            self.forward_prefill_metadata,
        ]
        metadata_objects.extend(self.decode_cuda_graph_metadata.values())
        seen = set()
        for metadata in metadata_objects:
            if metadata is None or id(metadata) in seen:
                continue
            seen.add(id(metadata))
            for attr in tuple(vars(metadata)):
                if attr.startswith("_dsa_") and attr.endswith("_cache"):
                    delattr(metadata, attr)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
    ):
        result = self._dense_backend.init_forward_metadata_capture_cuda_graph(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            forward_mode=forward_mode,
        )
        self._clear_metadata_caches()
        return result

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = None,
        req_to_page: torch.Tensor = None,
        **kwargs,
    ):
        result = self._dense_backend.init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            forward_mode=forward_mode,
            req_to_page=req_to_page,
            **kwargs,
        )
        self._refresh_decode_topk_schedule_metadata(bs)
        return result

    def _refresh_decode_topk_schedule_metadata(self, bs: int) -> None:
        if deep_gemm is None:
            return
        metadata = getattr(self._dense_backend, "forward_decode_metadata", None)
        if metadata is None:
            return
        schedule_metadata = getattr(
            metadata,
            "_dsa_paged_mqa_schedule_metadata",
            None,
        )
        if schedule_metadata is None:
            return

        # Rebuild the per-token visible lengths the capture used: verify
        # graphs schedule [bs, q_len] rows (token j of a request sees
        # seq_len - q_len + j + 1 positions), plain decode [bs, 1].
        q_len = int(getattr(metadata, "_dsa_paged_mqa_schedule_q_len", 1) or 1)
        base_lens = metadata.seq_lens_k[:bs].to(torch.int32)
        if q_len == 1:
            seq_lens = base_lens.view(-1, 1).contiguous()
        else:
            offsets = torch.arange(
                1 - q_len, 1, device=base_lens.device, dtype=torch.int32
            )
            seq_lens = (
                (base_lens.view(-1, 1) + offsets.view(1, -1)).clamp_min_(0).contiguous()
            )
        refreshed = deep_gemm.get_paged_mqa_logits_metadata(
            seq_lens,
            self.page_size,
            deep_gemm.get_num_sms(),
        )
        if (
            schedule_metadata.shape != refreshed.shape
            or schedule_metadata.device != refreshed.device
            or schedule_metadata.dtype != refreshed.dtype
        ):
            raise RuntimeError(
                "DSA CUDA graph paged-MQA schedule metadata changed shape "
                "during replay; recapture or use eager for this batch. "
                f"captured={tuple(schedule_metadata.shape)} {schedule_metadata.dtype} "
                f"{schedule_metadata.device}, refreshed={tuple(refreshed.shape)} "
                f"{refreshed.dtype} {refreshed.device}"
            )
        with torch.inference_mode():
            schedule_metadata.copy_(refreshed)

    def get_cuda_graph_seq_len_fill_value(self):
        return self._dense_backend.get_cuda_graph_seq_len_fill_value()

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        req_to_page: torch.Tensor,
        spec_info=None,
        **kwargs,
    ):
        out = self._dense_backend.init_forward_metadata(
            bs=bs,
            num_extends=num_extends,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            forward_mode=forward_mode,
            req_to_page=req_to_page,
            spec_info=spec_info,
            **kwargs,
        )
        self._prefill_block_tables = None
        if (
            num_extends > 0
            and req_to_page is not None
            and forward_mode.is_extend_or_mixed()
        ):
            cmeta = getattr(self._dense_backend, "chunked_prefill_metadata", None)
            if cmeta is not None and cmeta.req_pool_indices is not None:
                ext_idx = cmeta.req_pool_indices[:num_extends].long()
                self._prefill_block_tables = req_to_page[ext_idx]
        return out

    def _get_sparse_decode_tile_metadata(self, num_reqs: int, q_len: int):
        if get_mla_metadata is error_fn:
            raise RuntimeError(
                "DSA sparse decode requires FlashMLA. "
                "Build/install `tokenspeed-kernel/python` with FlashMLA."
            )
        # FlashMLA lazily generates the tile schedule on the first kernel call
        # with a given sched_meta and only allows reuse while the inputs that
        # shaped it stay constant. We never pass topk_length (see the kernel
        # call site), so the schedule depends only on (batch, q_len, heads,
        # topk) and a per-shape persistent sched_meta is safe to reuse —
        # including under CUDA graph, where the first call for a shape happens
        # during eager warmup and capture/replay then reuse the initialized
        # metadata buffers at fixed addresses.
        cache = getattr(self, "_sparse_decode_sched_meta", None)
        if cache is None:
            cache = {}
            self._sparse_decode_sched_meta = cache
        key = (int(num_reqs), int(q_len))
        meta = cache.get(key)
        if meta is None:
            meta = get_mla_metadata()[0]
            cache[key] = meta
        return meta

    def _get_prefill_workspace(
        self,
        *,
        num_reqs: int,
        max_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        workspace_reqs = max(1, int(num_reqs))
        workspace_seq_len = max(1, int(max_seq_len))
        rows = workspace_reqs * workspace_seq_len
        workspace = self._get_prefill_workspace_rows(rows=rows, device=device)
        return workspace[:rows].view(
            workspace_reqs,
            workspace_seq_len,
            1,
            self.kv_cache_dim,
        )

    def _get_prefill_workspace_rows(
        self,
        *,
        rows: int,
        device: torch.device,
    ) -> torch.Tensor:
        rows = max(1, int(rows))
        if (
            self._prefill_workspace_buffer is None
            or self._prefill_workspace_buffer.device != device
            or self._prefill_workspace_dim != self.kv_cache_dim
            or self._prefill_workspace_rows < rows
        ):
            self._prefill_workspace_buffer = torch.empty(
                (rows, 1, self.kv_cache_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            self._prefill_workspace_rows = rows
            self._prefill_workspace_dim = self.kv_cache_dim
        assert self._prefill_workspace_buffer is not None
        return self._prefill_workspace_buffer[:rows]

    def _get_prefill_query_workspace(
        self,
        *,
        rows: int,
        padded_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if (
            self._prefill_query_workspace is None
            or self._prefill_query_workspace.device != device
            or self._prefill_query_workspace.dtype != dtype
            or self._prefill_query_workspace.shape[0] < rows
            or self._prefill_query_workspace.shape[1] != padded_heads
            or self._prefill_query_workspace.shape[2] != head_dim
        ):
            self._prefill_query_workspace = torch.empty(
                (rows, padded_heads, head_dim),
                dtype=dtype,
                device=device,
            )
            self._prefill_query_workspace.zero_()
            self._prefill_query_workspace_num_heads = 0
        return self._prefill_query_workspace[:rows]

    def _get_decode_query_workspace(
        self,
        *,
        num_tokens: int,
        seq_len: int,
        padded_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        if (
            self._decode_query_workspace is None
            or self._decode_query_workspace.device != device
            or self._decode_query_workspace.dtype != dtype
            or self._decode_query_workspace.shape[0] < num_tokens
            or self._decode_query_workspace.shape[1] != seq_len
            or self._decode_query_workspace.shape[2] != padded_heads
            or self._decode_query_workspace.shape[3] != head_dim
        ):
            if (
                self._decode_query_workspace is not None
                and self._decode_query_workspace_captured
            ):
                # A captured CUDA graph may still replay into the old buffer
                # (q_len flips between decode and spec-verify re-trigger this
                # path); keep it alive instead of returning it to the
                # allocator.
                retired = getattr(self, "_retired_decode_query_workspaces", None)
                if retired is None:
                    retired = []
                    self._retired_decode_query_workspaces = retired
                retired.append(self._decode_query_workspace)
            self._decode_query_workspace = torch.empty(
                (num_tokens, seq_len, padded_heads, head_dim),
                dtype=dtype,
                device=device,
            )
            self._decode_query_workspace.zero_()
            self._decode_query_workspace_num_heads = 0
            self._decode_query_workspace_captured = capturing
        elif capturing:
            self._decode_query_workspace_captured = True
        return self._decode_query_workspace[:num_tokens]

    def _pad_sparse_prefill_query_heads(
        self,
        q: torch.Tensor,
        *,
        num_heads: int,
        head_dim: int,
        head_multiple: int,
    ) -> tuple[torch.Tensor, int]:
        q = q.view(-1, num_heads, head_dim)
        padded_heads = _flashmla_sparse_prefill_padded_heads(
            num_heads,
            head_multiple,
        )
        if padded_heads == num_heads:
            return q.contiguous(), num_heads

        q_padded = self._get_prefill_query_workspace(
            rows=q.shape[0],
            padded_heads=padded_heads,
            head_dim=head_dim,
            device=q.device,
            dtype=q.dtype,
        )
        q_padded[:, :num_heads, :].copy_(q)
        previous_num_heads = getattr(
            self,
            "_prefill_query_workspace_num_heads",
            None,
        )
        if previous_num_heads is None or previous_num_heads > num_heads:
            q_padded[:, num_heads:, :].zero_()
        self._prefill_query_workspace_num_heads = num_heads
        return q_padded, num_heads

    def _pad_sparse_decode_query_heads(
        self,
        q: torch.Tensor,
        *,
        num_heads: int,
    ) -> tuple[torch.Tensor, int]:
        padded_heads = _flashmla_sparse_decode_padded_heads(num_heads)
        if padded_heads == num_heads:
            return q.contiguous(), num_heads

        q_padded = self._get_decode_query_workspace(
            num_tokens=q.shape[0],
            seq_len=q.shape[1],
            padded_heads=padded_heads,
            head_dim=q.shape[3],
            device=q.device,
            dtype=q.dtype,
        )
        q_padded[:, :, :num_heads, :].copy_(q)
        previous_num_heads = getattr(
            self,
            "_decode_query_workspace_num_heads",
            None,
        )
        if previous_num_heads is None or previous_num_heads > num_heads:
            q_padded[:, :, num_heads:, :].zero_()
        self._decode_query_workspace_num_heads = num_heads
        return q_padded, num_heads

    def _validate_logit_cap(self, logits_soft_cap: float) -> None:
        if logits_soft_cap and logits_soft_cap > 0:
            raise NotImplementedError(
                "TokenSpeed DSA fused dense attention does not support "
                f"logits_soft_cap={logits_soft_cap}. Sparse DSA kernels must "
                "preserve the capped-score semantics before enabling this model."
            )

    def _validate_dense_context(self, seq_lens: torch.Tensor, bs: int) -> None:
        if seq_lens is None or bs <= 0:
            return
        active_seq_lens = seq_lens[:bs]
        if active_seq_lens.numel() == 0:
            return
        max_seq_len = int(active_seq_lens.max().item())
        if max_seq_len > self.index_topk:
            raise NotImplementedError(
                "TokenSpeed DSA dense attention is exact only when every "
                f"request has seq_len <= index_topk ({self.index_topk}); got "
                f"max seq_len {max_seq_len}. Sparse DSA top-k indices are "
                "required for longer contexts."
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
        self._validate_logit_cap(logits_soft_cap)
        self._validate_dense_context(seq_lens, batch_size)
        return self._dense_backend.forward_extend_chunked(
            q,
            k,
            v,
            scaling,
            logits_soft_cap,
            cum_seq_lens_q=cum_seq_lens_q,
            cum_seq_lens_kv=cum_seq_lens_kv,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            seq_lens=seq_lens,
            batch_size=batch_size,
            causal=causal,
            out=out,
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = True,
        topk_indices: torch.Tensor | None = None,
        topk_lens: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        self._validate_logit_cap(layer.logit_cap)
        if topk_indices is not None:
            return self._forward_sparse_decode(
                q=q,
                k=k,
                v=v,
                layer=layer,
                out_cache_loc=out_cache_loc,
                token_to_kv_pool=token_to_kv_pool,
                bs=bs,
                save_kv_cache=save_kv_cache,
                topk_indices=topk_indices,
                topk_lens=topk_lens,
            )
        metadata = getattr(self, "forward_decode_metadata", None)
        if metadata is not None and metadata.seq_lens_k is not None:
            num_extends = int(metadata.num_extends or 0)
            self._validate_dense_context(metadata.seq_lens_k[num_extends:], bs)
        return self._dense_backend.forward_decode(
            q=q,
            k=k,
            v=v,
            layer=layer,
            out_cache_loc=out_cache_loc,
            token_to_kv_pool=token_to_kv_pool,
            bs=bs,
            save_kv_cache=save_kv_cache,
            **kwargs,
        )

    def forward_sparse_prefill(
        self,
        *,
        q: torch.Tensor,
        layer,
        token_to_kv_pool,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        workspace_indices: torch.Tensor,
        topk_lens: torch.Tensor,
        kv_workspace_slots: torch.Tensor | None = None,
        max_seq_len: int,
    ) -> torch.Tensor:
        return self._get_sparse_prefill_forward()(
            q=q,
            layer=layer,
            token_to_kv_pool=token_to_kv_pool,
            block_tables=block_tables,
            seq_lens=seq_lens,
            workspace_indices=workspace_indices,
            topk_lens=topk_lens,
            kv_workspace_slots=kv_workspace_slots,
            max_seq_len=max_seq_len,
        )

    def _forward_sparse_prefill_flashmla(
        self,
        *,
        q: torch.Tensor,
        layer,
        token_to_kv_pool,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        workspace_indices: torch.Tensor,
        topk_lens: torch.Tensor,
        kv_workspace_slots: torch.Tensor | None = None,
        max_seq_len: int,
    ) -> torch.Tensor:
        if flash_mla_sparse_fwd is error_fn:
            raise RuntimeError(
                "DSA sparse prefill requires FlashMLA. "
                "Build/install `tokenspeed-kernel/python` with FlashMLA."
            )
        if layer.logit_cap and layer.logit_cap > 0:
            self._validate_logit_cap(layer.logit_cap)
        if self.page_size != 64:
            raise RuntimeError(
                "DSA sparse prefill currently requires page_size=64 for "
                f"FlashMLA sparse KV layout, got {self.page_size}."
            )
        if self.kv_lora_rank != 512:
            raise RuntimeError(
                "DSA sparse prefill requires kv_lora_rank=512 for FlashMLA, "
                f"got {self.kv_lora_rank}."
            )
        if getattr(token_to_kv_pool, "quant_method", None) == "per_token_head":
            raise RuntimeError(
                "DSA sparse prefill does not support "
                "kv_cache_quant_method='per_token_head' yet."
            )
        if self.data_type in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise RuntimeError(
                "DSA sparse prefill does not support FP8 MLA KV cache layout yet."
            )
        if q.dtype != torch.bfloat16:
            raise RuntimeError(
                "DSA sparse prefill requires BF16 query tensors, " f"got {q.dtype}."
            )

        num_reqs = int(seq_lens.numel())
        if workspace_indices.shape[0] != q.shape[0]:
            raise RuntimeError(
                "DSA sparse prefill metadata token mismatch: "
                f"indices={workspace_indices.shape[0]}, q_tokens={q.shape[0]}"
            )
        if topk_lens.shape[0] != q.shape[0]:
            raise RuntimeError(
                "DSA sparse prefill top-k length mismatch: "
                f"lens={topk_lens.shape[0]}, q_tokens={q.shape[0]}"
            )
        if num_reqs == 0 or q.shape[0] == 0:
            return q.new_empty((0, layer.tp_q_head_num * layer.v_head_dim))

        k_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)
        if k_cache.dtype != torch.bfloat16:
            raise RuntimeError(
                "DSA sparse prefill currently requires BF16 MLA KV cache, "
                f"got {k_cache.dtype}."
            )

        if kv_workspace_slots is not None:
            if kv_workspace_slots.dim() != 1:
                raise RuntimeError(
                    "DSA sparse prefill KV slot shape mismatch: "
                    f"expected 1-D packed slots, got "
                    f"{tuple(kv_workspace_slots.shape)}"
                )
            kv_workspace = self._get_prefill_workspace_rows(
                rows=kv_workspace_slots.numel(),
                device=q.device,
            )
            flat_slots = kv_workspace_slots.to(
                device=q.device,
                dtype=torch.int64,
            )
            flat_workspace = kv_workspace.view(-1, 1, self.kv_cache_dim)
            flat_workspace.copy_(
                k_cache.index_select(0, flat_slots).view_as(flat_workspace)
            )
        else:
            kv_workspace = self._get_prefill_workspace(
                num_reqs=num_reqs,
                max_seq_len=int(max_seq_len),
                device=q.device,
            )
            for req_idx in range(num_reqs):
                seq_len = int(seq_lens[req_idx].item())
                if seq_len <= 0:
                    continue
                local = torch.arange(seq_len, dtype=torch.int64, device=q.device)
                page_offsets = torch.div(
                    local,
                    self.page_size,
                    rounding_mode="floor",
                )
                pages = (
                    block_tables[req_idx]
                    .to(torch.int64)
                    .index_select(
                        0,
                        page_offsets,
                    )
                )
                slots = pages * self.page_size + (local % self.page_size)
                kv_workspace[req_idx, :seq_len].copy_(k_cache.index_select(0, slots))

        q_kernel, actual_num_heads = self._pad_sparse_prefill_query_heads(
            q,
            num_heads=layer.tp_q_head_num,
            head_dim=layer.head_dim,
            head_multiple=_flashmla_sparse_prefill_head_multiple(),
        )

        # Invalid sparse slots are encoded as -1; do not pass topk_length here.
        out, _, _ = flash_mla_sparse_fwd(
            q=q_kernel,
            kv=kv_workspace.view(-1, 1, self.kv_cache_dim),
            indices=workspace_indices.unsqueeze(1),
            sm_scale=layer.scaling,
            d_v=layer.v_head_dim,
        )
        if out.shape[1] != actual_num_heads:
            out = out[:, :actual_num_heads, :]
        return out.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _forward_sparse_prefill_trtllm(
        self,
        *,
        q: torch.Tensor,
        layer,
        token_to_kv_pool,
        workspace_indices: torch.Tensor,
        topk_lens: torch.Tensor,
        kv_workspace_slots: torch.Tensor | None,
        max_seq_len: int,
        block_tables: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del block_tables, seq_lens
        if trtllm_batch_decode_with_kv_cache_mla is error_fn:
            raise RuntimeError(
                "DSA sparse prefill requires TRTLLM sparse MLA on Blackwell. "
                "Build/install `tokenspeed-kernel/python` with FlashInfer TRTLLM "
                "decode support."
            )
        allow_fp8_query = (
            self.data_type == torch.float8_e4m3fn and q.dtype == torch.float8_e4m3fn
        )
        if q.dtype != torch.bfloat16 and not allow_fp8_query:
            raise RuntimeError(
                "DSA sparse prefill requires BF16 query tensors, or FP8 query "
                f"tensors on the TRTLLM FP8 KV path, got {q.dtype}."
            )
        if topk_lens.shape[0] != q.shape[0]:
            raise RuntimeError(
                "DSA sparse prefill top-k length mismatch: "
                f"lens={topk_lens.shape[0]}, q_tokens={q.shape[0]}"
            )
        if workspace_indices.shape != (q.shape[0], self.index_topk):
            raise RuntimeError(
                "DSA sparse prefill top-k shape mismatch: "
                f"indices={tuple(workspace_indices.shape)}, "
                f"expected={(q.shape[0], self.index_topk)}"
            )
        if kv_workspace_slots is None:
            raise RuntimeError(
                "DSA TRTLLM sparse prefill requires kv_workspace_slots to "
                "map workspace-local top-k rows back to KV cache slots."
            )
        if q.shape[0] == 0:
            return q.new_empty((0, layer.tp_q_head_num * layer.v_head_dim))

        block_tables = _workspace_indices_to_kv_slots(
            workspace_indices.to(torch.int32),
            kv_workspace_slots,
        ).view(q.shape[0], 1, self.index_topk)
        seq_lens = topk_lens.to(device=q.device, dtype=torch.int32).contiguous()
        q = q.view(q.shape[0], 1, layer.tp_q_head_num, layer.head_dim)
        if self.data_type == torch.float8_e4m3fn:
            q = q.to(self.data_type)

        k_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)
        if self.data_type != k_cache.dtype:
            k_cache = k_cache.to(self.data_type)
        kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)

        k_scale = (
            layer.k_scale_float
            if getattr(layer, "k_scale_float", None) is not None
            else 1.0
        )
        out = trtllm_batch_decode_with_kv_cache_mla(
            query=q,
            kv_cache=kv_cache,
            workspace_buffer=self._get_trtllm_workspace(),
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            sparse_mla_top_k=self.index_topk,
            bmm1_scale=k_scale * layer.scaling,
            backend="trtllm-gen",
        )
        return out.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _forward_sparse_decode(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool,
        topk_indices: torch.Tensor,
        topk_lens: torch.Tensor | None,
    ) -> torch.Tensor:
        del v
        if self.page_size != 64:
            raise RuntimeError(
                "DSA sparse decode currently requires page_size=64 for "
                f"sparse KV layout, got {self.page_size}."
            )
        if getattr(token_to_kv_pool, "quant_method", None) == "per_token_head":
            raise RuntimeError(
                "DSA sparse decode does not support "
                "kv_cache_quant_method='per_token_head' yet."
            )
        allow_trtllm_fp8_query = (
            getattr(self, "_sparse_decode_impl", "trtllm") == "trtllm"
            and getattr(self, "data_type", torch.bfloat16) == torch.float8_e4m3fn
            and q.dtype == torch.float8_e4m3fn
        )
        if q.dtype != torch.bfloat16 and not allow_trtllm_fp8_query:
            raise RuntimeError(
                "DSA sparse decode requires BF16 query tensors, or FP8 query "
                f"tensors on the TRTLLM FP8 KV path, got {q.dtype}."
            )
        if save_kv_cache:
            assert k is not None
            token_to_kv_pool.set_mla_kv_buffer(
                layer,
                out_cache_loc,
                k[..., : self.kv_lora_rank],
                k[..., self.kv_lora_rank :],
            )

        if topk_indices.dtype != torch.int32:
            topk_indices = topk_indices.to(torch.int32)
        if topk_indices.shape[-1] != self.index_topk:
            raise RuntimeError(
                "DSA sparse decode top-k width mismatch: "
                f"indices={topk_indices.shape[-1]}, expected={self.index_topk}"
            )
        num_tokens = q.shape[0]
        # Spec-verify feeds q_len_per_req query rows per request while plain
        # decode and the draft model's own decode steps feed one; derive the
        # width from the actual batch shape (bs is the decode request count)
        # rather than spec_num_tokens, which the draft backend inherits from the
        # shared config.
        if bs > 0 and num_tokens % bs == 0:
            q_len_per_req = num_tokens // bs
        else:
            q_len_per_req = 1
        num_reqs = num_tokens // q_len_per_req

        return self._get_sparse_decode_forward()(
            q=q,
            layer=layer,
            token_to_kv_pool=token_to_kv_pool,
            num_reqs=num_reqs,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
            q_len_per_req=q_len_per_req,
        )

    def _forward_sparse_decode_flashmla(
        self,
        *,
        q: torch.Tensor,
        layer,
        token_to_kv_pool,
        num_reqs: int,
        topk_indices: torch.Tensor,
        topk_lens: torch.Tensor | None = None,
        q_len_per_req: int = 1,
    ) -> torch.Tensor:
        del topk_lens
        if flash_mla_with_kvcache is error_fn:
            raise RuntimeError(
                "DSA multi-token sparse decode requires FlashMLA. "
                "Build/install `tokenspeed-kernel/python` with FlashMLA."
            )
        if not hasattr(token_to_kv_pool, "get_sparse_decode_kv_buffer"):
            raise RuntimeError(
                "DSA multi-token sparse decode requires a DSA KV pool with "
                "sparse decode cache storage."
            )
        q = q.view(num_reqs, q_len_per_req, layer.tp_q_head_num, layer.head_dim)
        q, actual_num_heads = self._pad_sparse_decode_query_heads(
            q,
            num_heads=layer.tp_q_head_num,
        )
        k_cache = token_to_kv_pool.get_sparse_decode_kv_buffer(layer.layer_id)
        kv_cache = k_cache.view(-1, self.page_size, 1, k_cache.shape[-1])

        out, _ = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=self.kv_lora_rank,
            tile_scheduler_metadata=self._get_sparse_decode_tile_metadata(
                num_reqs,
                q_len_per_req,
            ),
            softmax_scale=layer.scaling,
            is_fp8_kvcache=True,
            indices=topk_indices.view(num_reqs, q_len_per_req, -1),
            # Lengths are carried by -1 padding; keep FlashMLA scheduling static.
        )
        if out.dim() == 4:
            if out.shape[2] != actual_num_heads:
                out = out[:, :, :actual_num_heads, :]
            out = out.reshape(-1, actual_num_heads, out.shape[-1])
        elif out.shape[1] != actual_num_heads:
            out = out[:, :actual_num_heads, :]
        return out.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _forward_sparse_decode_trtllm(
        self,
        *,
        q: torch.Tensor,
        layer,
        token_to_kv_pool,
        num_reqs: int,
        topk_indices: torch.Tensor,
        topk_lens: torch.Tensor | None = None,
        q_len_per_req: int = 1,
    ) -> torch.Tensor:
        if trtllm_batch_decode_with_kv_cache_mla is error_fn:
            raise RuntimeError(
                "DSA plain sparse decode requires TRTLLM sparse MLA. "
                "Build/install `tokenspeed-kernel/python` with FlashInfer TRTLLM "
                "decode support."
            )
        metadata = getattr(self, "forward_decode_metadata", None)
        if (
            metadata is None
            or metadata.seq_lens_k is None
            or metadata.max_seq_len_k is None
        ):
            raise RuntimeError("DSA plain sparse decode requires decode metadata.")
        num_extends = int(metadata.num_extends or 0)
        available_reqs = max(0, int(metadata.seq_lens_k.shape[0]) - num_extends)
        if available_reqs < num_reqs:
            if available_reqs <= 0 or q.shape[0] % available_reqs != 0:
                raise RuntimeError(
                    "DSA plain sparse decode metadata batch mismatch: "
                    f"seq_lens={available_reqs}, requests={num_reqs}, "
                    f"q_tokens={q.shape[0]}."
                )
            num_reqs = available_reqs
            q_len_per_req = q.shape[0] // available_reqs
        seq_lens = metadata.seq_lens_k[num_extends : num_extends + num_reqs]
        if seq_lens.numel() != num_reqs:
            raise RuntimeError(
                "DSA plain sparse decode metadata batch mismatch: "
                f"seq_lens={seq_lens.numel()}, requests={num_reqs}."
            )
        num_tokens = q.shape[0]
        expected_tokens = num_reqs * int(q_len_per_req)
        if num_tokens != expected_tokens:
            raise RuntimeError(
                "DSA TRTLLM sparse decode token shape mismatch: "
                f"q_tokens={num_tokens}, requests={num_reqs}, "
                f"q_len_per_req={q_len_per_req}."
            )
        if q_len_per_req > 1:
            if getattr(self, "is_draft", False):
                offsets = torch.arange(
                    q_len_per_req,
                    device=seq_lens.device,
                    dtype=seq_lens.dtype,
                )
            else:
                offsets = torch.arange(
                    1 - q_len_per_req,
                    1,
                    device=seq_lens.device,
                    dtype=seq_lens.dtype,
                )
            seq_lens = (
                (seq_lens.view(-1, 1) + offsets.view(1, -1))
                .clamp_min_(0)
                .reshape(-1)
                .contiguous()
            )
        if topk_lens is not None:
            if topk_lens.dim() != 1 or topk_lens.numel() != num_tokens:
                raise RuntimeError(
                    "DSA TRTLLM sparse decode top-k length mismatch: "
                    f"lens={tuple(topk_lens.shape)}, q_tokens={num_tokens}."
                )
            # seq_lens is relative to the sparse block table, not dense context.
            seq_lens = topk_lens.to(device=q.device, dtype=torch.int32).contiguous()

        q = q.view(num_tokens, 1, layer.tp_q_head_num, layer.head_dim)
        if self.data_type == torch.float8_e4m3fn:
            q = q.to(self.data_type)
        k_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)
        if self.data_type != k_cache.dtype:
            k_cache = k_cache.to(self.data_type)
        kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)

        k_scale = (
            layer.k_scale_float
            if getattr(layer, "k_scale_float", None) is not None
            else 1.0
        )
        out = trtllm_batch_decode_with_kv_cache_mla(
            query=q,
            kv_cache=kv_cache,
            workspace_buffer=self._get_trtllm_workspace(),
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=topk_indices.view(num_tokens, 1, -1),
            seq_lens=seq_lens,
            max_seq_len=metadata.max_seq_len_k,
            sparse_mla_top_k=self.index_topk,
            bmm1_scale=k_scale * layer.scaling,
            backend="trtllm-gen",
        )
        return out.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)


register_backend("dsa", {AttentionArch.DSA}, DSABackend)
