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


"""Optional flash_ops Lite NoPE prolog; no TS-owned extension or converter."""

from __future__ import annotations

import importlib
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _flash_mla_prolog():
    try:
        importlib.import_module("flash_ops")
    except ModuleNotFoundError as exc:
        if exc.name != "flash_ops":
            raise
        return None
    packet = getattr(torch.ops.custom, "npu_mla_prolog_v3", None)
    if packet is None:
        return None
    op = packet.default
    arguments = {arg.name for arg in op._schema.arguments}
    if not {"enable_rope", "ckvkr_repo_mode"} <= arguments:
        return None
    return op


def mla_prolog_available() -> bool:
    """Whether flash_ops registers the Lite-capable V3 schema."""
    return _flash_mla_prolog() is not None


def _supports_mla_prolog(
    token_x,
    weight_dq,
    weight_uq_qr,
    weight_uk,
    weight_dkv_kr,
    rmsnorm_gamma_cq,
    rmsnorm_gamma_ckv,
    kv_cache,
    cache_index,
) -> bool:
    # Metadata-only admission: safe at capture time, no device synchronization.
    if token_x.device.type != "npu" or token_x.ndim != 2:
        return False
    tokens, hidden = token_x.shape
    if tokens == 0 or weight_uk.ndim != 3:
        return False
    heads = weight_uk.shape[0]
    if hidden not in (3072, 4096) or heads not in (4, 8, 16, 32, 64):
        return False
    expected = (
        (token_x, (tokens, hidden)),
        (weight_dq, (hidden, 1536)),
        (weight_uq_qr, (1536, heads * 192)),
        (weight_uk, (heads, 128, 512)),
        (weight_dkv_kr, (hidden, 576)),
        (rmsnorm_gamma_cq, (1536,)),
        (rmsnorm_gamma_ckv, (512,)),
    )
    if any(
        t.shape != shape
        or t.dtype != torch.bfloat16
        or t.device != token_x.device
        or not t.is_contiguous()
        for t, shape in expected
    ):
        return False
    if (
        kv_cache.ndim != 4
        or kv_cache.shape[2:] != (1, 576)
        or kv_cache.shape[0] == 0
        or not 16 <= kv_cache.shape[1] <= 1024
        or kv_cache.shape[1] % 16 != 0
        or not kv_cache.is_contiguous()
        or kv_cache.dtype != torch.bfloat16
        or kv_cache.device != token_x.device
        or cache_index.shape != (tokens,)
        or cache_index.dtype not in (torch.int32, torch.int64)
        or cache_index.device != token_x.device
        or not cache_index.is_contiguous()
    ):
        return False
    import torch_npu

    return all(
        torch_npu.get_npu_format(weight) == 29
        for weight in (weight_dq, weight_uq_qr, weight_dkv_kr)
    )


def mla_prolog(
    token_x: torch.Tensor,
    weight_dq: torch.Tensor,
    weight_uq_qr: torch.Tensor,
    weight_uk: torch.Tensor,
    weight_dkv_kr: torch.Tensor,
    rmsnorm_gamma_cq: torch.Tensor,
    rmsnorm_gamma_ckv: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_index: torch.Tensor,
    *,
    rmsnorm_epsilon_cq: float,
    rmsnorm_epsilon_ckv: float,
    return_query_norm: bool = False,
) -> tuple[torch.Tensor, ...] | None:
    """Fuse Lite projections/norm/absorption and write the existing packed cache.

    Projection weights use logical [in, out] NZ layout; weight_uk is
    [heads, 128, 512]. Cache is BF16 [pages, page_size, 1, 576] and cache_index
    supplies backend-owned int32/int64 flattened slots. The adapter widens
    int32 slots on-device for the operator; graph replay refreshes this cast.
    Return absorbed Q and its unrotated
    64-wide auxiliary tail, or None before any write for unsupported inputs.
    Errors from an eligible operator propagate instead of retrying after writes.
    """
    op = _flash_mla_prolog()
    if op is None or not _supports_mla_prolog(
        token_x,
        weight_dq,
        weight_uq_qr,
        weight_uk,
        weight_dkv_kr,
        rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv,
        kv_cache,
        cache_index,
    ):
        return None
    if cache_index.dtype == torch.int32:
        cache_index = cache_index.to(dtype=torch.int64)
    # V3's output-shape inference requires rope tensors even for NoPE.
    # Their values are not read when enable_rope=False. Packed mode ignores
    # kr_cache; use a small separate placeholder, not a second managed cache.
    rope = token_x.new_empty((token_x.shape[0], 64))
    kr_placeholder = token_x.new_empty((1, 1, 1, 64))
    outputs = op(
        token_x,
        weight_dq,
        weight_uq_qr,
        weight_uk,
        weight_dkv_kr,
        rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv,
        rope,
        rope,
        kv_cache,
        kr_placeholder,
        cache_index=cache_index,
        dequant_scale_x=None,
        dequant_scale_w_dq=None,
        dequant_scale_w_uq_qr=None,
        dequant_scale_w_dkv_kr=None,
        quant_scale_ckv=None,
        quant_scale_ckr=None,
        smooth_scales_cq=None,
        actual_seq_len=None,
        k_nope_clip_alpha=None,
        rmsnorm_epsilon_cq=rmsnorm_epsilon_cq,
        rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv,
        cache_mode="PA_BSND",
        query_norm_flag=return_query_norm,
        weight_quant_mode=0,
        kv_cache_quant_mode=0,
        query_quant_mode=0,
        ckvkr_repo_mode=1,
        quant_scale_repo_mode=0,
        tile_size=128,
        qc_qr_scale=1.0,
        kc_scale=1.0,
        enable_rope=False,
    )
    if return_query_norm:
        return outputs[0], outputs[1], outputs[3]
    return outputs[0], outputs[1]
