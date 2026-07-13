# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Correctness tests for fused GDN QKV split kernel (b0/b1/b3)."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.triton.gdn_qkv_split import (
    fused_qkv_split_gdn_prefill,
)
from tokenspeed_kernel.ops.attention.triton.linear.l2norm import l2norm_fwd

# Qwen3.5 production shapes
CONFIGS = [
    # (num_q_heads, num_v_heads, head_q, head_v, T)
    (16, 16, 128, 128, 512),  # GQA equal heads, short seq
    (16, 16, 128, 128, 2048),  # long seq
    (16, 32, 128, 128, 512),  # GVA: num_v > num_q
    (16, 64, 128, 128, 1024),  # wider V
]


def _ref_split(mixed_qkv, nq, nk, nv, hq, hk, hv):
    """Reference: torch.split + view (current baseline)."""
    T = mixed_qkv.shape[0]
    q_ref, k_ref, v_ref = torch.split(mixed_qkv, [nq * hq, nk * hk, nv * hv], dim=-1)
    q_ref = q_ref.view(1, T, nq, hq)
    k_ref = k_ref.view(1, T, nk, hk)
    v_ref = v_ref.view(1, T, nv, hv)
    return q_ref, k_ref, v_ref


@pytest.mark.parametrize("nq,nv,hq,hv,T", CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_split_correctness(nq, nv, hq, hv, T, dtype):
    nk = nq
    hk = hq
    torch.manual_seed(42)
    mixed_qkv = torch.randn(T, nq * hq + nk * hk + nv * hv, dtype=dtype, device="cuda")

    q_ref, k_ref, v_ref = _ref_split(mixed_qkv, nq, nk, nv, hq, hk, hv)
    q, k, v = fused_qkv_split_gdn_prefill(
        mixed_qkv, nq, nk, nv, hq, hk, hv, fuse_l2norm=False
    )

    assert q.shape == (1, T, nq, hq)
    assert k.shape == (1, T, nk, hk)
    assert v.shape == (1, T, nv, hv)
    assert torch.max(torch.abs(q.float() - q_ref.float())) < 1e-5, "Q mismatch"
    assert torch.max(torch.abs(k.float() - k_ref.float())) < 1e-5, "K mismatch"
    assert torch.max(torch.abs(v.float() - v_ref.float())) < 1e-5, "V mismatch"


@pytest.mark.parametrize("nq,nv,hq,hv,T", CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_split_l2norm_correctness(nq, nv, hq, hv, T, dtype):
    """Fused l2norm variant must match torch.split + separate l2norm_fwd."""
    nk = nq
    hk = hq
    torch.manual_seed(7)
    mixed_qkv = torch.randn(T, nq * hq + nk * hk + nv * hv, dtype=dtype, device="cuda")

    q_ref, k_ref, v_ref = _ref_split(mixed_qkv, nq, nk, nv, hq, hk, hv)
    q_ref_norm = l2norm_fwd(q_ref)
    k_ref_norm = l2norm_fwd(k_ref)

    q, k, v = fused_qkv_split_gdn_prefill(
        mixed_qkv, nq, nk, nv, hq, hk, hv, fuse_l2norm=True
    )

    # bf16 accumulation order differs from l2norm_fwd (axis=1 vs sequential per-head);
    # after dividing by norm (~sqrt(128)≈11), max rounding delta ~1e-3.
    assert (
        torch.max(torch.abs(q.float() - q_ref_norm.float())) < 2e-3
    ), "Q l2norm mismatch"
    assert (
        torch.max(torch.abs(k.float() - k_ref_norm.float())) < 2e-3
    ), "K l2norm mismatch"
    assert torch.max(torch.abs(v.float() - v_ref.float())) < 1e-5, "V mismatch"


@pytest.mark.parametrize("nq,nv,hq,hv,T", CONFIGS[:2])
def test_strided_input(nq, nv, hq, hv, T):
    """Strided input must produce same result as contiguous (b3 fallback)."""
    nk = nq
    hk = hq
    dtype = torch.bfloat16
    torch.manual_seed(99)
    # Create strided view: slice every other row then pad back
    big = torch.randn(T * 2, nq * hq + nk * hk + nv * hv, dtype=dtype, device="cuda")
    strided = big[::2]  # stride(0) = 2 * original stride
    assert not strided.is_contiguous()

    q_ref, k_ref, v_ref = _ref_split(strided.contiguous(), nq, nk, nv, hq, hk, hv)
    q, k, v = fused_qkv_split_gdn_prefill(
        strided, nq, nk, nv, hq, hk, hv, fuse_l2norm=False
    )

    assert torch.max(torch.abs(q.float() - q_ref.float())) < 1e-5
    assert torch.max(torch.abs(k.float() - k_ref.float())) < 1e-5
    assert torch.max(torch.abs(v.float() - v_ref.float())) < 1e-5
