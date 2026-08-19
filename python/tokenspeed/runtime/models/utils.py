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

"""Helpers shared across runtime model implementations."""

import torch
from tokenspeed_kernel.ops.embedding import FusedMLASetKVBufferArg, FusedSetKVBufferArg

from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.utils import print_warning_once


def validate_attention_partition(
    total_num_heads: int,
    total_num_kv_heads: int,
    tp_size: int,
) -> None:
    if tp_size <= 0:
        raise ValueError(f"tp_size must be positive, got {tp_size}.")
    if total_num_heads % tp_size != 0:
        raise ValueError(
            f"num_attention_heads={total_num_heads} must be divisible by tp_size={tp_size}."
        )
    if total_num_kv_heads <= 0:
        raise ValueError(
            f"num_key_value_heads must be positive, got {total_num_kv_heads}."
        )
    if total_num_kv_heads >= tp_size:
        if total_num_kv_heads % tp_size != 0:
            raise ValueError(
                f"num_key_value_heads={total_num_kv_heads} must be divisible by tp_size={tp_size}."
            )
    elif tp_size % total_num_kv_heads != 0:
        raise ValueError(
            f"tp_size={tp_size} must be divisible by num_key_value_heads={total_num_kv_heads}."
        )


def create_fused_set_kv_buffer_arg(
    value: torch.Tensor,
    layer: PagedAttention,
    out_cache_loc: torch.Tensor,
    token_to_kv_pool,
):
    """Build fused RoPE+KV write arguments when the fused path is supported."""

    from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool

    layer_id = layer.layer_id

    k_buffer = token_to_kv_pool.get_key_buffer(layer_id)
    v_buffer = token_to_kv_pool.get_value_buffer(layer_id)

    is_mla = isinstance(token_to_kv_pool, MLATokenToKVPool)

    if is_mla:
        kv_lora_rank = token_to_kv_pool.kv_lora_rank
        k_buffer = k_buffer[..., kv_lora_rank:].view(k_buffer.shape[0], -1)
        v_buffer = v_buffer[..., :kv_lora_rank].view(v_buffer.shape[0], -1)
    else:
        k_buffer = k_buffer.view(k_buffer.shape[0], -1)
        v_buffer = v_buffer.view(v_buffer.shape[0], -1)

    # Non-trivial scales need 1/scale applied before FP8 cast — the fused kernel
    # doesn't support this yet, so log a warning and skip the fused path.
    k_scale = layer.k_scale
    v_scale = layer.v_scale
    if (k_scale is not None and k_scale != 1.0) or (
        v_scale is not None and v_scale != 1.0
    ):
        print_warning_once(
            f"Fused RoPE+KV write disabled: non-trivial k_scale={k_scale} v_scale={v_scale}"
        )
        return None

    return FusedSetKVBufferArg(
        value=value,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        k_scale=None,
        v_scale=None,
        cache_loc=out_cache_loc,
    )


# Grid CTA count (tokens * (heads + 1), the fused kernel's own launch shape)
# at which the fused write stops being measured no-slower than the split path
# it replaces. 2048 tokens at H=16 (measured, see the perf table in the
# commit that added this) is the calibration point; scaling by heads matters
# because the grid's CTA count -- and so the fused kernel's own wall time --
# grows with heads at a fixed token count, while the split path's does not
# the same way. Confirmed at H=32 (safe to 1024 tokens, 18.9% regression at
# 2048) and H=64 (safe to 512 tokens, 4.9% regression at 768).
_FUSED_MLA_KV_MAX_TOKEN_HEADS = 2048 * 16


def create_fused_mla_set_kv_buffer_arg(
    k_nope: torch.Tensor,
    rope_dim: int,
    rotary_emb: object | None,
    out_cache_loc: torch.Tensor,
    token_to_kv_pool: object,
    layer_id: int,
    num_q_heads: int,
    q_nope: torch.Tensor | None = None,
) -> FusedMLASetKVBufferArg | None:
    """Arguments for the fused MLA RoPE + quantize + KV write, or ``None``.

    ``None`` is the single place this configuration is judged to have no fused
    form, so callers fall back rather than branch. Every reason lives here:
    ``token_to_kv_pool`` is not an ``MLATokenToKVPool``, or it overrides the
    latent write (Kimi-K3's hybrid KDA pool overrides it to force sanitize --
    see below); the token*head count is past the validated range; the pool's
    latent rows are not one dense ``[tokens, 1, nope+rope]`` tensor (the
    per-token-head quantized pool keeps three); or no registered RoPE solution
    advertises the fused traits.

    ``rotary_emb=None`` is the NoPE form and IS fused when every other check
    passes -- the tables are simply absent from the returned argument. Kimi-K3
    is NoPE, but its hybrid KDA pool overrides the latent write and so never
    reaches this form; a K3 draft (a heterogeneous MLA cache view) gets a
    plain pool and does.
    """

    from tokenspeed_kernel.ops.embedding import supports_fused_mla_kv_write

    from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool

    if not isinstance(token_to_kv_pool, MLATokenToKVPool):
        return None
    # A pool that overrides the latent write does so to add something this
    # fused write does not have -- the hybrid KDA pool overrides it to force
    # sanitize=True, whose comment describes a padded-row NaN reaching a live
    # row's softmax through the shared dummy slot. Fusing would bypass the
    # override silently, so decline whenever one exists.
    if type(token_to_kv_pool).set_mla_kv_buffer is not (
        MLATokenToKVPool.set_mla_kv_buffer
    ):
        return None
    if out_cache_loc.numel() * num_q_heads > _FUSED_MLA_KV_MAX_TOKEN_HEADS:
        return None

    kv_buffer = token_to_kv_pool.get_key_buffer(layer_id)
    if not isinstance(kv_buffer, torch.Tensor):
        return None
    if not supports_fused_mla_kv_write(
        q_dtype=k_nope.dtype,
        k_dtype=k_nope.dtype,
        has_rope=rotary_emb is not None,
        is_neox=True if rotary_emb is None else rotary_emb.is_neox_style,
        has_q_nope=q_nope is not None,
    ):
        return None
    # The kernel converts BF16 inputs to the FP8 cache dtype on store.
    if kv_buffer.ndim != 3 or kv_buffer.shape[1] != 1:
        return None
    if kv_buffer.shape[2] != k_nope.shape[-1] + rope_dim:
        return None

    return FusedMLASetKVBufferArg(
        k_nope=k_nope,
        kv_buffer=kv_buffer.view(kv_buffer.shape[0], -1),
        cache_loc=out_cache_loc,
        q_nope=q_nope,
        cos_sin_cache=None if rotary_emb is None else rotary_emb.cos_sin_cache,
        is_neox=True if rotary_emb is None else rotary_emb.is_neox_style,
    )
