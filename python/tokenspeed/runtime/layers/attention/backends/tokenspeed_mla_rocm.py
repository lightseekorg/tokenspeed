# Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT
"""
AIter MLA attention backend for AMD MI355X (gfx950, CDNA4).

Parallel to tokenspeed_mla.py (NVIDIA Blackwell SM100 CuTe DSL backend).
This backend routes MLA decode and prefill through AIter on ROCm.

Status: SKELETON. Class is registered and supports() is wired, but the
forward_decode / forward_prefill methods raise NotImplementedError until
the AIter call sites are implemented (see TODOs).

Phase 1 implementation plan (tracked in MI355X optimization plan):
  1. Wire forward_decode -> aiter.mla_decode (FP8/BF16)
  2. Wire forward_prefill -> aiter ragged FMHA for MLA
  3. KV layout: reuse mla_kv_pack_quantize_fp8 (already cross-platform)
  4. Re-tune dispatch buckets in runtime/cache/utils.py for MI355X HBM3e BW
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.registry import register_backend

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention

logger = logging.getLogger(__name__)


@dataclass
class AIterMLADecodeMetadata:
    """Per-step metadata for AIter MLA decode. Fill in actual fields when
    we know the exact AIter call signature for MLA decode."""

    seq_lens: torch.Tensor
    page_table: torch.Tensor  # [B, max_pages]
    cu_seq_lens: torch.Tensor | None  # for prefill chunking, None for pure decode
    max_seq_len: int


@dataclass
class AIterMLAPrefillMetadata:
    max_seq_len: int
    cu_seq_lens: torch.Tensor
    seq_lens: torch.Tensor


class TokenspeedMlaRocmBackend(AttentionBackend):
    """AIter-backed MLA attention for AMD MI355X / MI350X."""

    @classmethod
    def supports(cls, model_config, server_args) -> bool:
        platform = current_platform()
        # CDNA4 only for now. CDNA3 (MI300) lacks the FP8/FP4 paths AIter MLA
        # uses; would need a separate backend.
        if not platform.is_cdna4:
            return False
        # Try import. AIter may not have the MLA kernel exposed yet.
        try:
            import aiter  # noqa: F401
        except ImportError:
            return False
        return True

    def __init__(self, model_runner, **kwargs):
        super().__init__(model_runner=model_runner, **kwargs)
        self._workspace = None
        # TODO: warmup AIter MLA kernels here, similar to
        #   warmup_compile_prefill() in tokenspeed_mla.py.

    def init_forward_metadata(self, forward_batch):
        # TODO: build AIterMLADecodeMetadata or AIterMLAPrefillMetadata from
        # forward_batch. Should be cheap; runs every step.
        raise NotImplementedError(
            "TokenspeedMlaRocmBackend.init_forward_metadata: not yet implemented. "
            "Phase 1 work item — see Phase 1 plan in MI355X optimization doc."
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "PagedAttention",
        forward_batch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        # TODO: route through AIter MLA decode. Approximate call:
        #   import aiter
        #   o = aiter.mla_decode(
        #       q, k_buffer, v_buffer,
        #       page_table=metadata.page_table,
        #       cache_seqlens=metadata.seq_lens,
        #       sm_scale=layer.scaling,
        #       window_left=-1,
        #       kv_lora_rank=layer.kv_lora_rank,
        #       qk_rope_head_dim=layer.qk_rope_head_dim,
        #   )
        # The exact API needs to be confirmed against AIter HEAD; the kernel
        # may live under aiter.mha_fwd_kvcache or similar.
        raise NotImplementedError(
            "TokenspeedMlaRocmBackend.forward_decode: pending AIter MLA wiring."
        )

    def forward_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "PagedAttention",
        forward_batch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        # TODO: route through AIter ragged FMHA for MLA prefill. Likely:
        #   import aiter
        #   o = aiter.flash_attn_varlen_func(
        #       q, k, v,
        #       cu_seqlens_q=metadata.cu_seq_lens,
        #       cu_seqlens_k=metadata.cu_seq_lens,
        #       max_seqlen_q=metadata.max_seq_len,
        #       max_seqlen_k=metadata.max_seq_len,
        #       softmax_scale=layer.scaling,
        #       causal=True,
        #   )
        raise NotImplementedError(
            "TokenspeedMlaRocmBackend.forward_prefill: pending AIter FMHA wiring."
        )


# Register for MLA arch. Lower priority than trtllm_mla on NVIDIA; this
# backend will only be selected when current_platform().is_cdna4 is True
# (see registry _get_default_backend_name change in this PR).
register_backend(
    name="tokenspeed_mla_rocm",
    cls=TokenspeedMlaRocmBackend,
    archs={AttentionArch.MLA},
)
