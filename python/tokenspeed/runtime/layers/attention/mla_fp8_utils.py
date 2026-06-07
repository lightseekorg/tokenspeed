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

"""Shared utilities for MLA FP8 attention backends."""

import torch
from tokenspeed_kernel.ops.embedding.flashinfer import mla_rope_quantize_fp8

from tokenspeed.runtime.utils.pdl import pdl_enabled


def mla_fused_rope_fp8_quantize(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    is_neox: bool = True,
    k_scale_inv: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused RoPE + FP8 quantization for pre-shaped MLA tensors.

    Returns:
        (query_fp8, key_fp8)
    """
    nope_dim = q_nope.size(-1)
    qk_rope_head_dim = q_pe.size(-1)

    query_fp8 = torch.empty(
        q_nope.shape[:-1] + (nope_dim + qk_rope_head_dim,),
        dtype=torch.float8_e4m3fn,
        device=q_nope.device,
    )
    key_fp8 = torch.empty(
        k_nope.shape[:-1] + (nope_dim + qk_rope_head_dim,),
        dtype=torch.float8_e4m3fn,
        device=k_nope.device,
    )

    if q_nope.size(0) == 0:
        return query_fp8, key_fp8

    mla_rope_quantize_fp8(
        q_rope=q_pe,
        k_rope=k_pe,
        q_nope=q_nope,
        k_nope=k_nope,
        cos_sin_cache=cos_sin_cache,
        pos_ids=positions,
        is_neox=is_neox,
        quantize_dtype=torch.float8_e4m3fn,
        q_rope_out=query_fp8[..., nope_dim:],
        k_rope_out=key_fp8[..., nope_dim:],
        q_nope_out=query_fp8[..., :nope_dim],
        k_nope_out=key_fp8[..., :nope_dim],
        quant_scale_q=1.0,
        quant_scale_kv=k_scale_inv,
        enable_pdl=pdl_enabled(),
    )

    return query_fp8, key_fp8
