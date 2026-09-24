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

"""Step-by-step attention prologues: the portable path that chains the step kernels."""

from __future__ import annotations

from functools import lru_cache

import torch
from tokenspeed_kernel.ops.attention.prologue.types import (
    BOOLS,
    ROPE_STYLES,
    GQAPrologueOutput,
    HeadKVCache,
    HeadNorm,
    KVCacheFormat,
    LatentKVCache,
    MLAExpandedKV,
    MLAPrologueOutput,
    MRope,
    RopeStyle,
    Rotary,
)
from tokenspeed_kernel.ops.embedding import apply_rope, apply_rope_mla
from tokenspeed_kernel.ops.kvcache.per_token_head import store_latent_per_token_head
from tokenspeed_kernel.ops.kvcache.triton import (
    fused_fp8_set_kv_buffer,
    quantize_store_kv_mxfp8,
    set_mla_kv_buffer_triton,
    store_kv_cache,
)
from tokenspeed_kernel.ops.layernorm import qk_rmsnorm
from tokenspeed_kernel.ops.quantization.triton import fp8_quantize
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@lru_cache
def _mrope_rows(mrope: MRope, device: torch.device) -> torch.Tensor:
    """The T/H/W position row that drives each rotary pair."""
    rows = torch.repeat_interleave(torch.arange(3), torch.tensor(mrope.section))
    if mrope.interleaved:
        pair = torch.arange(sum(mrope.section))
        rows = torch.zeros_like(pair)
        rows[(pair % 3 == 1) & (pair < 3 * mrope.section[1])] = 1
        rows[(pair % 3 == 2) & (pair < 3 * mrope.section[2])] = 2
    return rows.to(device)


def _rope_table(rotary: Rotary) -> tuple[torch.Tensor, torch.Tensor]:
    """Positions and cos/sin table for ``embedding.rope``; multimodal RoPE
    gathers each rotary pair's row, so the rotation itself is plain RoPE."""
    positions = rotary.positions
    if positions.ndim == 1:
        return positions, rotary.cos_sin_cache
    rows = _mrope_rows(rotary.mrope, positions.device).repeat(2)
    pairs = torch.arange(rotary.rotary_dim, device=positions.device)
    table = rotary.cos_sin_cache[positions][rows, :, pairs].transpose(0, 1)
    return (
        torch.arange(positions.shape[-1], device=positions.device),
        table.contiguous(),
    )


def store_kv(
    cache: HeadKVCache, k: torch.Tensor, v: torch.Tensor, enable_pdl: bool
) -> None:
    """Store the leading ``cache.slots.numel()`` prepared K/V rows in the cache's format."""
    rows = cache.slots.numel()
    if rows == 0:
        return
    k = k[:rows].view(rows, *cache.k_cache.shape[1:])
    v = v[:rows].view(rows, *cache.v_cache.shape[1:])
    if cache.format is KVCacheFormat.NATIVE:
        store_kv_cache(
            k, v, cache.k_cache, cache.v_cache, cache.slots, enable_pdl=enable_pdl
        )
    elif cache.format is KVCacheFormat.FP8:
        fused_fp8_set_kv_buffer(
            k,
            v,
            cache.k_cache,
            cache.v_cache,
            cache.slots,
            page_size=1,
            enable_pdl=enable_pdl,
        )
    else:
        quantize_store_kv_mxfp8(
            k,
            v,
            cache.k_cache,
            cache.v_cache,
            cache.scales.k,
            cache.scales.v,
            cache.slots,
            page_tokens=cache.scales.page_tokens,
            enable_pdl=enable_pdl,
        )


@register_kernel(
    "attention",
    "gqa_prologue",
    name="composite_gqa_prologue",
    solution="composite",
    capability=CapabilityRequirement(vendors=frozenset({"amd", "ascend", "nvidia"})),
    signatures=format_signatures(("q",), "dense", {torch.float16, torch.bfloat16}),
    priority=Priority.PORTABLE,
    traits={
        "has_norm": BOOLS,
        "kv_format": frozenset({"native", "fp8", "mxfp8"}),
        "partial_write": BOOLS,
        "kv_convert": BOOLS,
        "mrope": BOOLS,
        "partial_rotary": BOOLS,
        "return_kv": BOOLS,
        "rope_style": ROPE_STYLES,
    },
)
def composite_gqa_prologue(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    norm: HeadNorm | None,
    rotary: Rotary | None,
    cache: HeadKVCache,
    return_kv: bool,
    enable_pdl: bool,
) -> GQAPrologueOutput:
    q = q.flatten(1)
    if norm is not None:
        q, k = qk_rmsnorm(
            q,
            k,
            norm.q_weight,
            norm.k_weight,
            norm.eps,
            weight_offset=norm.weight_offset,
        )
    if rotary is not None:
        positions, table = _rope_table(rotary)
        q, k = apply_rope(
            positions,
            q,
            k,
            cache.k_cache.shape[-1],
            table,
            is_neox=rotary.style is RopeStyle.NEOX,
        )
    store_kv(cache, k, v, enable_pdl)
    return GQAPrologueOutput(
        q=q, k=k if return_kv else None, v=v if return_kv else None
    )


def _rope_quantize(
    rotary: Rotary | None,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return apply_rope_mla(
        positions=None if rotary is None else rotary.positions,
        q_rope=q_pe,
        k_rope=k_pe,
        q_nope=q_nope,
        k_nope=k_nope,
        cos_sin_cache=None if rotary is None else rotary.cos_sin_cache,
        is_neox=rotary is not None and rotary.style is RopeStyle.NEOX,
        quant_scale_q=1.0,
        quant_scale_kv=1.0,
    )


def _write_latent(
    cache: LatentKVCache, k_nope: torch.Tensor, k_rope: torch.Tensor, enable_pdl: bool
) -> None:
    if cache.slots.numel() == 0:
        return
    if cache.format is KVCacheFormat.FP8_PER_TOKEN_HEAD:
        planes = cache.kv_cache
        store_latent_per_token_head(
            planes.latent,
            planes.scale,
            planes.rope,
            cache.slots,
            k_nope,
            k_rope,
            sanitize=cache.sanitize,
        )
        return
    set_mla_kv_buffer_triton(
        cache.kv_cache,
        cache.slots,
        k_nope,
        k_rope,
        enable_pdl=enable_pdl,
        sanitize=cache.sanitize,
    )


@register_kernel(
    "attention",
    "mla_prologue",
    name="composite_mla_prologue",
    solution="composite",
    capability=CapabilityRequirement(vendors=frozenset({"amd", "nvidia"})),
    signatures=format_signatures(("query",), "dense", {torch.float16, torch.bfloat16}),
    priority=Priority.PORTABLE,
    traits={
        "expanded": BOOLS,
        "full_write": BOOLS,
        "kv_format": frozenset({"native", "fp8", "fp8_per_token_head"}),
        "kv_convert": BOOLS,
        "rope_style": ROPE_STYLES,
        "sanitize": BOOLS,
    },
)
def composite_mla_prologue(
    *,
    query: torch.Tensor,
    q_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    expanded: MLAExpandedKV | None,
    rotary: Rotary | None,
    cache: LatentKVCache,
    enable_pdl: bool,
) -> MLAPrologueOutput:
    rows = cache.slots.numel()
    num_heads, rope_dim = q_pe.shape[1:]
    q_nope_dim = query.shape[-1] - rope_dim
    rank = latent_cache.shape[-1] - rope_dim
    latent = latent_cache.unsqueeze(1)
    k_nope, k_pe = latent[..., :rank], latent[..., rank:]
    key = value = None
    if cache.format is KVCacheFormat.FP8:
        if expanded is None:
            query, latent = _rope_quantize(
                rotary, query[..., :q_nope_dim], q_pe, k_nope, k_pe
            )
            k_nope, k_pe = latent[..., :rank], latent[..., rank:]
        else:
            k_rope = k_pe.expand(-1, num_heads, -1)
            query, key = _rope_quantize(
                rotary, query[..., :q_nope_dim], q_pe, expanded.k_nope, k_rope
            )
            value = expanded.value
            value = fp8_quantize(value.reshape(-1, value.shape[-1])).view(value.shape)
            k_pe = key[:, :1, q_nope_dim:]
    else:
        if rotary is not None:
            q_pe, k_pe = apply_rope(
                rotary.positions,
                q_pe,
                k_pe,
                rope_dim,
                rotary.cos_sin_cache,
                is_neox=rotary.style is RopeStyle.NEOX,
            )
        q_rope = query[..., q_nope_dim:]
        if q_pe.data_ptr() != q_rope.data_ptr():
            q_rope.copy_(q_pe)
        if expanded is not None:
            key = torch.cat((expanded.k_nope, k_pe.expand(-1, num_heads, -1)), dim=-1)
            value = expanded.value
    _write_latent(cache, k_nope[:rows], k_pe[:rows], enable_pdl)
    return MLAPrologueOutput(query=query, key=key, value=value)
