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

"""Portable fallbacks for optional TokenSpeed MLA kernels."""

from typing import Optional, Tuple

import torch


def mla_kv_pack_quantize_fp8(
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    v: torch.Tensor,
    k_scale_inv: float = 1.0,
    v_scale_inv: float = 1.0,
    k_out: Optional[torch.Tensor] = None,
    v_out: Optional[torch.Tensor] = None,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    enable_pdl: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack MLA keys and quantize K/V when the fused extension is unavailable.

    ``enable_pdl`` is accepted for API compatibility. PyTorch stream ordering
    provides the required producer/consumer dependency for this fallback.
    """
    del enable_pdl

    if k_nope.ndim != 3:
        raise ValueError(f"k_nope must be 3D, got shape {tuple(k_nope.shape)}")
    if v.ndim != 3:
        raise ValueError(f"v must be 3D, got shape {tuple(v.shape)}")
    if k_pe.ndim not in (2, 3):
        raise ValueError(f"k_pe must be 2D or 3D, got shape {tuple(k_pe.shape)}")

    seq_len, num_heads, _ = k_nope.shape
    if v.shape[:2] != (seq_len, num_heads):
        raise ValueError(
            f"v shape {tuple(v.shape)} mismatches k_nope {tuple(k_nope.shape)}"
        )
    if k_pe.shape[0] != seq_len:
        raise ValueError(
            f"k_pe first dim {k_pe.shape[0]} mismatches k_nope first dim {seq_len}"
        )
    if k_pe.ndim == 3:
        if k_pe.shape[1] != 1:
            raise ValueError(f"k_pe head dim must be 1, got {k_pe.shape[1]}")
        k_pe = k_pe.squeeze(1)
    if fp8_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"unsupported FP8 dtype: {fp8_dtype}")

    k_pe = k_pe.unsqueeze(1).expand(-1, num_heads, -1)
    quantized_k = (
        torch.cat((k_nope, k_pe), dim=-1).float().mul_(k_scale_inv).to(fp8_dtype)
    )
    quantized_v = v.float().mul_(v_scale_inv).to(fp8_dtype)

    if k_out is None:
        k_out = quantized_k
    else:
        k_out.copy_(quantized_k)
    if v_out is None:
        v_out = quantized_v
    else:
        v_out.copy_(quantized_v)
    return k_out, v_out
