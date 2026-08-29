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

"""SM80 W8A16 block-scaled FP8 GEMM (compute_bf16 path) tests.

Covers the fp16-bitcast e4m3 widening used by the Triton kernel at small M
and the dequant+cuBLAS fallback at large M. Both must stay within 3e-3 L2
relative error of an fp32 dequant reference (the bf16-dequant cuBLAS
baseline criterion).
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.gemm.triton import w8a8_block_fp8_matmul_triton

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

BLOCK = [128, 128]
TOL = 3e-3


def _dequant(Q: torch.Tensor, S: torch.Tensor, block_m: int, block_k: int):
    m, k = Q.shape
    q = Q.to(torch.float32).view(m // block_m, block_m, k // block_k, block_k)
    return (q * S[:, None, :, None]).view(m, k)


def _make_problem(m: int, n: int, k: int, dev: str):
    torch.manual_seed(0)
    a = (torch.randn(m, k, device=dev) * 0.1).clamp(-448, 448).to(
        torch.float8_e4m3fn
    )
    b = (torch.randn(n, k, device=dev) * 0.1).clamp(-448, 448).to(
        torch.float8_e4m3fn
    )
    a_s = torch.rand(m, k // BLOCK[1], device=dev) * 0.01 + 1e-4
    b_s = torch.rand(n // BLOCK[0], k // BLOCK[1], device=dev) * 0.01 + 1e-4
    return a, b, a_s, b_s


def _rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    return ((out.float() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("m", [64, 257, 1024, 1031, 2048])
def test_w8a16_blockscale_matches_fp32_dequant(device: str, m: int) -> None:
    """L2 rel error vs fp32 dequant reference <= 3e-3 (cuBLAS baseline).

    m >= 1024 exercises the dequant+cuBLAS fallback, including M values
    that are not a multiple of the tile size.
    """
    n, k = 512, 384
    a, b, a_s, b_s = _make_problem(m, n, k, device)
    out = w8a8_block_fp8_matmul_triton(
        a, b, a_s, b_s, BLOCK, output_dtype=torch.bfloat16, compute_bf16=True
    )
    ref = _dequant(a, a_s, 1, BLOCK[1]) @ _dequant(b, b_s, *BLOCK).T
    assert _rel_err(out, ref) <= TOL


def test_w8a16_convert_exact_finite_patterns(device: str) -> None:
    """Every finite e4m3 value (incl. all subnormals) converts exactly.

    ``B`` is one-hot so each output element equals one activation value in
    any reduction order; with unit scales the result must be bit-identical
    to a direct e4m3 -> bf16 cast.
    """
    m, n, k = 256, BLOCK[0], BLOCK[1]
    pats = torch.arange(256, device=device, dtype=torch.uint8)
    pats = pats[(pats & 0x7F) != 0x7F]  # drop the NaN patterns
    a = pats.repeat(m * k // pats.numel() + 1)[: m * k]
    a = a.reshape(m, k).view(torch.float8_e4m3fn)
    b = torch.zeros(n, k, device=device)
    b[torch.arange(n), torch.arange(n)] = 1.0
    b = b.to(torch.float8_e4m3fn)
    a_s = torch.ones(m, k // BLOCK[1], device=device)
    b_s = torch.ones(n // BLOCK[0], k // BLOCK[1], device=device)
    out = w8a8_block_fp8_matmul_triton(
        a, b, a_s, b_s, BLOCK, output_dtype=torch.bfloat16, compute_bf16=True
    )
    assert torch.equal(out, a.to(torch.bfloat16))


def test_w8a16_out_buffer_and_3d_activation(device: str) -> None:
    """3D activations and a caller-provided ``out`` work for both paths."""
    n, k = 256, 256
    for m in (64, 1024):
        a, b, a_s, b_s = _make_problem(m, n, k, device)
        a3 = a.view(4, m // 4, k)
        a3_s = a_s.view(4, m // 4, k // BLOCK[1])
        out = torch.empty(4, m // 4, n, device=device, dtype=torch.bfloat16)
        ret = w8a8_block_fp8_matmul_triton(
            a3, b, a3_s, b_s, BLOCK,
            output_dtype=torch.bfloat16, compute_bf16=True, out=out,
        )
        assert ret is out
        ref = _dequant(a, a_s, 1, BLOCK[1]) @ _dequant(b, b_s, *BLOCK).T
        assert _rel_err(out, ref.view_as(out)) <= TOL


@pytest.mark.parametrize("n,k", [(4096, 5120), (5120, 6784)])
@pytest.mark.parametrize("m", [1, 2, 3, 8])
def test_w8a16_decode_gemv_matches_fp32_dequant(
    device: str, m: int, n: int, k: int
) -> None:
    """Split-K decode GEMV within 3e-3 L2 of the fp32 dequant reference.

    Exercises both split kernels: the scalar reduction at m == 1 and the
    padded fp16 tensor-core dot at m > 1, with k blocks divisible and not
    divisible by 128-block count 8 (6784 -> 53 scale columns).
    """
    from tokenspeed_kernel.ops.gemm.w8a16_gemv import w8a16_decode_gemv

    torch.manual_seed(0)
    a = (torch.randn(m, k, device=device) * 2.0).to(torch.bfloat16)
    w = (torch.randn(n, k, device=device) * 0.1).to(torch.float8_e4m3fn)
    b_s = torch.rand(n // BLOCK[0], k // BLOCK[1], device=device) * 0.02 + 5e-3
    bias = torch.randn(n, device=device) * 0.01
    out = w8a16_decode_gemv(a, w, b_s, bias=bias)
    ref = a.float() @ _dequant(w, b_s, *BLOCK).T + bias.float()
    assert _rel_err(out, ref) <= TOL


@pytest.mark.parametrize("m", [1, 4])
def test_w8a16_decode_gemv_graph_replay_deterministic(device: str, m: int) -> None:
    """Graph capture + replay reproduces the eager result bit-for-bit."""
    from tokenspeed_kernel.ops.gemm.w8a16_gemv import w8a16_decode_gemv

    torch.manual_seed(0)
    n, k = 4096, 5120
    w = (torch.randn(n, k, device=device) * 0.1).to(torch.float8_e4m3fn)
    b_s = torch.rand(n // BLOCK[0], k // BLOCK[1], device=device) * 0.02 + 5e-3
    a = (torch.randn(m, k, device=device) * 2.0).to(torch.bfloat16)

    out = w8a16_decode_gemv(a, w, b_s)  # eager (also JIT-compiles)
    ref = out.clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = w8a16_decode_gemv(a, w, b_s)
    for _ in range(3):
        g.replay()
        assert torch.equal(out, ref)
    a.copy_(torch.randn(m, k, device=device).to(torch.bfloat16))
    eager = w8a16_decode_gemv(a, w, b_s)
    g.replay()
    assert torch.equal(out, eager)
