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

"""Attention prologue: everything between the QKV projections and core attention.

A prologue takes a layer's projected query and key/value rows and leaves the
query ready for the attention kernel and the key/value rows in the KV cache:
optional per-head QK RMSNorm, optional rotary embedding (plain or multimodal),
KV quantization for the cache format, and the KV cache write. Models describe
the steps once; which kernels run them is a dispatch decision made here, so a
fused kernel covering several steps can replace the step-by-step path without
any model knowing.

Two layouts share one vocabulary:

* :func:`gqa_prologue` -- grouped-query (and multi-head) attention
  with separate K and V cache planes.
* :func:`mla_prologue` -- multi-head latent attention with one
  latent-plus-RoPE cache plane.

docs/design/attention-prologue.md states the numerics contract. Inputs may be
overwritten.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from tokenspeed_kernel.ops.attention.prologue.checks import (
    check_gqa_request,
    check_latent_write,
    check_mla_request,
)
from tokenspeed_kernel.ops.attention.prologue.composite import store_latent
from tokenspeed_kernel.ops.attention.prologue.types import (
    GQAPrologueOutput,
    HeadKVCache,
    HeadNorm,
    KVCacheFormat,
    LatentKVCache,
    MLAExpandedKV,
    MLAPrologueOutput,
    MRope,
    MXFP8Scales,
    PerTokenHeadPlanes,
    RopeStyle,
    Rotary,
)
from tokenspeed_kernel.platform import current_platform, pdl_enabled
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import (
    SelectedKernel,
    select_kernel,
    spec_matches_shape_traits,
    spec_matches_traits,
)
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@lru_cache(maxsize=4096)
def _select(
    mode: str,
    dtype: torch.dtype,
    traits: tuple[tuple[str, object], ...],
    solution: str | None,
    override: str | None,
) -> SelectedKernel:
    """Select the kernel for a request, once per distinct request; an override
    may name only a kernel this platform runs whose traits cover the request,
    since solutions drop inputs their traits exclude."""
    role = "q" if mode == "gqa_prologue" else "query"
    kernel = select_kernel(
        "attention",
        mode,
        format_signature(**{role: dense_tensor_format(dtype)}),
        traits=dict(traits),
        solution=solution,
        override=override,
    )
    spec = KernelRegistry.get().get_by_name(kernel.name)
    if not (
        (spec.family, spec.mode) == ("attention", mode)
        and spec.capability.satisfied_by(current_platform())
        and spec_matches_traits(spec, dict(traits))
        and spec_matches_shape_traits(spec, dict(traits))
    ):
        raise ValueError(
            f"{kernel.name} does not serve attention.{mode} with {dict(traits)}"
        )
    return kernel


def gqa_prologue(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    norm: HeadNorm | None,
    rotary: Rotary | None,
    cache: HeadKVCache,
    return_kv: bool,
    solution: str | None,
    override: str | None,
) -> GQAPrologueOutput:
    """Prepare one GQA layer's query and write its key/value rows to the cache.

    Head counts and widths are read from ``cache``.

    Args:
        q: Query ``[num_tokens, num_q_heads * head_dim]``, or ``[num_tokens,
            num_q_heads, head_dim]``; rows and heads may be strided (a view of
            the packed QKV projection) but each head is contiguous.
        k: Key ``[num_tokens, num_kv_heads * head_dim]``.
        v: Value ``[num_tokens, num_kv_heads * head_dim]``.
        norm: Per-head RMSNorm, or ``None``.
        rotary: Rotary embedding, or ``None`` for NoPE.
        cache: KV cache destination.
        return_kv: Also return the rotated key and the value rows, for
            attention kernels that read them instead of the cache.
        solution: Optional registered solution to select.
        override: Optional exact kernel name; it must serve the request.

    Returns:
        The query and optional key/value rows for core attention.
    """
    check_gqa_request(q, k, v, norm, rotary, cache)
    num_tokens = q.shape[0]
    num_kv_heads, head_dim = cache.k_cache.shape[1:]
    num_q_heads = q.shape[1:].numel() // head_dim

    traits = {
        "head_dim": head_dim,
        "token_heads": num_tokens * num_q_heads,
        "has_norm": norm is not None,
        "kv_format": cache.format.value,
        "kv_convert": cache.format is KVCacheFormat.NATIVE
        and cache.k_cache.dtype is not q.dtype,
        "mrope": rotary is not None and rotary.positions.ndim == 2,
        "partial_rotary": rotary is not None and rotary.rotary_dim != head_dim,
        "return_kv": return_kv,
        "rope_style": "none" if rotary is None else rotary.style.value,
    }
    kernel = _select("gqa_prologue", q.dtype, tuple(traits.items()), solution, override)
    shape_params = {
        "num_tokens": num_tokens,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
    }
    ShapeCapture.get().record(
        "attention", "gqa_prologue", kernel.name, q.dtype, shape_params
    )
    with kernel_scope(
        "attention", "gqa_prologue", q.dtype, kernel_name=kernel.name, **shape_params
    ):
        return kernel(
            q=q,
            k=k,
            v=v,
            norm=norm,
            rotary=rotary,
            cache=cache,
            return_kv=return_kv,
            enable_pdl=pdl_enabled(),
        )


def qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    head_dim: int,
    norm: HeadNorm | None,
    rotary: Rotary | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize and rotate query and key heads without writing a cache.

    For keys that are not attention K/V, such as a sparse-attention indexer's;
    the GQA prologue's kernels and numerics, with no cache rows to write.

    Args:
        q: Query ``[num_tokens, q_heads * head_dim]``.
        k: Key ``[num_tokens, k_heads * head_dim]``.
        head_dim: Width of one head.
        norm: Per-head RMSNorm, or ``None``.
        rotary: Rotary embedding, or ``None`` for NoPE.

    Returns:
        The query and key, each ``[num_tokens, heads * head_dim]``.
    """
    no_rows = k.new_empty(0, k.shape[1:].numel() // head_dim, head_dim)
    out = gqa_prologue(
        q,
        k,
        k,
        norm=norm,
        rotary=rotary,
        cache=HeadKVCache(
            k_cache=no_rows,
            v_cache=no_rows,
            scales=None,
            slots=k.new_empty(0, dtype=torch.int64),
        ),
        return_kv=True,
        solution=None,
        override=None,
    )
    return out.q, out.k


def write_latent(
    latent_cache: torch.Tensor, *, rotary: Rotary | None, cache: LatentKVCache
) -> None:
    """Store an MLA layer's latent rows, their RoPE part rotated: the prologue's
    cache write alone, for the graph segment before an attention break. The
    break's :func:`mla_prologue` then finds its rows written and skips the
    store (a zero-row cache) or rewrites the same rows.

    Args:
        latent_cache: ``[num_tokens, kv_lora_rank + rope_dim]`` normalized
            latent followed by the unrotated key RoPE part, left as given.
        rotary: Rotary embedding, or ``None`` for NoPE.
        cache: Latent cache destination with one slot per row.
    """
    check_latent_write(latent_cache, rotary, cache)
    store_latent(latent_cache, rotary, cache, pdl_enabled())


def mla_prologue(
    query: torch.Tensor,
    q_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    *,
    expanded: MLAExpandedKV | None,
    rotary: Rotary | None,
    cache: LatentKVCache,
    solution: str | None,
    override: str | None,
) -> MLAPrologueOutput:
    """Prepare one MLA layer's attention inputs and write its latent rows.

    Args:
        query: ``[num_tokens, num_heads, q_nope_dim + rope_dim]`` buffer whose
            leading channels already hold the query's non-RoPE part: the
            absorbed latent query (``q_nope_dim == kv_lora_rank``) or a
            per-head query that attends expanded keys. Its RoPE channels
            receive the rotated ``q_pe`` unless the cache is FP8 (not planes).
        q_pe: Unrotated query RoPE part ``[num_tokens, num_heads, rope_dim]``;
            it may alias ``query``'s RoPE channels.
        latent_cache: ``[num_tokens, kv_lora_rank + rope_dim]`` normalized
            latent followed by the unrotated key RoPE part, which may be
            rotated in place.
        expanded: Per-head keys and values when attention does not absorb
            the latent up-projection; ``None`` for absorbed attention, which
            reads the latent cache.
        rotary: Rotary embedding, or ``None`` for NoPE.
        cache: Latent cache destination.
        solution: Optional registered solution to select.
        override: Optional exact kernel name; it must serve the request.

    Returns:
        The query, plus the per-head keys and values of expanded attention:
        FP8 e4m3 for an FP8 cache (not per-token-head planes), else in the query dtype.
    """
    check_mla_request(query, q_pe, latent_cache, expanded, rotary, cache)
    num_tokens, num_heads, width = query.shape
    rope_dim = q_pe.shape[-1]
    kv_lora_rank = latent_cache.shape[-1] - rope_dim

    traits = {
        "token_heads": num_tokens * num_heads,
        "expanded": expanded is not None,
        "full_write": cache.slots.numel() == num_tokens,
        "kv_format": cache.format.value,
        "kv_convert": cache.format is KVCacheFormat.NATIVE
        and cache.kv_cache.dtype is not query.dtype,
        "rope_style": "none" if rotary is None else rotary.style.value,
        "sanitize": cache.sanitize,
    }
    kernel = _select(
        "mla_prologue", query.dtype, tuple(traits.items()), solution, override
    )
    shape_params = {
        "num_tokens": num_tokens,
        "num_heads": num_heads,
        "q_nope_dim": width - rope_dim,
        "kv_lora_rank": kv_lora_rank,
        "qk_rope_head_dim": rope_dim,
    }
    ShapeCapture.get().record(
        "attention", "mla_prologue", kernel.name, query.dtype, shape_params
    )
    with kernel_scope(
        "attention",
        "mla_prologue",
        query.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            query=query,
            q_pe=q_pe,
            latent_cache=latent_cache,
            expanded=expanded,
            rotary=rotary,
            cache=cache,
            enable_pdl=pdl_enabled(),
        )


__all__ = [
    "GQAPrologueOutput",
    "HeadKVCache",
    "HeadNorm",
    "KVCacheFormat",
    "LatentKVCache",
    "MLAExpandedKV",
    "MLAPrologueOutput",
    "MRope",
    "MXFP8Scales",
    "PerTokenHeadPlanes",
    "RopeStyle",
    "Rotary",
    "gqa_prologue",
    "mla_prologue",
    "qk_norm_rope",
    "write_latent",
]


# Backend registration (side-effect import; the composite registers on its import above)
import tokenspeed_kernel.ops.attention.prologue.triton  # noqa: E402,F401
