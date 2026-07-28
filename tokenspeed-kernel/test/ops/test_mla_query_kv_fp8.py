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

"""Tests for the fused NoPE MLA fp8 query assembly + latent KV commit."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.triton.mla_query_assemble import (
    mla_nope_query_fp8,
)
from tokenspeed_kernel.ops.attention.triton.mla_query_kv_fp8 import (
    mla_nope_query_kv_fp8,
)
from tokenspeed_kernel.platform import current_platform

platform = current_platform()
torch.manual_seed(42)

pytestmark = pytest.mark.skipif(
    not platform.is_nvidia,
    reason="fp8 MLA query/KV tests target NVIDIA GPUs.",
)

NOPE_DIM = 512
PE_DIM = 64
D = NOPE_DIM + PE_DIM
FP8 = torch.float8_e4m3fn


def _make_inputs(device, T=1, H=12, kv_rows=1024, loc_dtype=torch.int64):
    """Decode-shaped inputs with the runtime's real (strided) views."""
    Q = torch.randn(T, H, D, dtype=torch.bfloat16, device=device)
    q_full = torch.randn(T, H, 128 + PE_DIM, dtype=torch.bfloat16, device=device)
    q_pe = q_full[..., 128:]  # strided column slice, like the q_b split
    # latent rows as a column slice of a wider projection (row stride != D)
    packed = torch.randn(T, 1536 + D + 1536, dtype=torch.bfloat16, device=device)
    latent = packed[:, 1536 : 1536 + D]
    kv = torch.zeros(kv_rows, 1, D, dtype=FP8, device=device)
    loc = torch.randperm(kv_rows - 64, device=device)[:T].to(loc_dtype) + 64
    return Q[..., :NOPE_DIM], q_pe, latent, kv, loc


def _reference(q_nope, q_pe, latent, kv, loc):
    """The unfused pair: query quant kernel + torch KV write with sanitize."""
    ref_q = mla_nope_query_fp8(q_nope, q_pe)
    row = latent.float()
    max_finite = float(torch.finfo(FP8).max)
    row = torch.nan_to_num(row, nan=0.0, posinf=max_finite, neginf=-max_finite)
    kv.view(kv.shape[0], -1)[loc.long()] = row.to(FP8)
    return ref_q


@pytest.mark.parametrize("T", [1, 4, 32])
@pytest.mark.parametrize("loc_dtype", [torch.int64, torch.int32])
def test_matches_unfused_pair(device: str, T: int, loc_dtype: torch.dtype) -> None:
    q_nope, q_pe, latent, kv, loc = _make_inputs(device, T=T, loc_dtype=loc_dtype)
    kv_ref = kv.clone()
    ref_q = _reference(q_nope, q_pe, latent, kv_ref, loc)

    out_q = mla_nope_query_kv_fp8(q_nope, q_pe, latent, kv, loc, sanitize=True)

    assert out_q.dtype == FP8
    torch.testing.assert_close(
        out_q.view(torch.uint8), ref_q.view(torch.uint8), atol=0, rtol=0
    )
    torch.testing.assert_close(
        kv.view(torch.uint8), kv_ref.view(torch.uint8), atol=0, rtol=0
    )


def test_sanitize_replaces_nonfinite_latents(device: str) -> None:
    q_nope, q_pe, latent, kv, loc = _make_inputs(device)
    latent = latent.contiguous()
    latent[0, 0] = float("nan")
    latent[0, 7] = float("inf")
    latent[0, NOPE_DIM + 3] = float("-inf")

    mla_nope_query_kv_fp8(q_nope, q_pe, latent, kv, loc, sanitize=True)

    row = kv[loc[0].long(), 0].float()
    max_finite = float(torch.finfo(FP8).max)
    assert row[0].item() == 0.0
    assert row[7].item() == max_finite
    assert row[NOPE_DIM + 3].item() == -max_finite
    assert torch.isfinite(row).all()


def test_latent_with_singleton_head_dim(device: str) -> None:
    """The runtime passes K as ``latent.unsqueeze(1)``; accept [T, 1, D]."""
    q_nope, q_pe, latent, kv, loc = _make_inputs(device)
    kv_ref = kv.clone()
    ref_q = _reference(q_nope, q_pe, latent, kv_ref, loc)

    out_q = mla_nope_query_kv_fp8(
        q_nope, q_pe, latent.unsqueeze(1), kv, loc, sanitize=True
    )

    torch.testing.assert_close(
        out_q.view(torch.uint8), ref_q.view(torch.uint8), atol=0, rtol=0
    )
    torch.testing.assert_close(
        kv.view(torch.uint8), kv_ref.view(torch.uint8), atol=0, rtol=0
    )


def test_out_parameter_reused(device: str) -> None:
    q_nope, q_pe, latent, kv, loc = _make_inputs(device)
    out = torch.empty(1, 12, D, dtype=FP8, device=device)
    ret = mla_nope_query_kv_fp8(q_nope, q_pe, latent, kv, loc, out=out)
    assert ret.data_ptr() == out.data_ptr()


def test_cuda_graph_capture_replay(device: str) -> None:
    """Capture-safe: no alloc/sync inside capture beyond the fp8 out buffer."""
    q_nope, q_pe, latent, kv, loc = _make_inputs(device)
    latent = latent.contiguous()
    out = torch.empty(1, 12, D, dtype=FP8, device=device)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mla_nope_query_kv_fp8(q_nope, q_pe, latent, kv, loc, out=out)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mla_nope_query_kv_fp8(q_nope, q_pe, latent, kv, loc, out=out)

    # Replay with refreshed inputs; results must track the new values.
    q_nope.copy_(torch.randn_like(q_nope))
    latent.copy_(torch.randn_like(latent))
    graph.replay()
    torch.cuda.synchronize()

    kv_ref = torch.zeros_like(kv)
    ref_q = _reference(q_nope, q_pe, latent, kv_ref, loc)
    torch.testing.assert_close(
        out.view(torch.uint8), ref_q.view(torch.uint8), atol=0, rtol=0
    )
    row = loc[0].long()
    torch.testing.assert_close(
        kv[row].view(torch.uint8), kv_ref[row].view(torch.uint8), atol=0, rtol=0
    )
