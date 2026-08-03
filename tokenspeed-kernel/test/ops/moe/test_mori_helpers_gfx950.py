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
"""Single-GPU (gfx950) numeric unit tests for the MORI EP compute helpers.

These exercise only the local compute pieces — no ``mori``/``flydsl`` package and
no multi-rank collective — so they run on one MI350X in the existing kernel CI and
skip elsewhere. Each helper is checked against an independent reference (cosine
similarity), and padding rows are checked to stay exactly zero (MORI ``combine``
reads only valid rows, so padding must never leak).
"""

from __future__ import annotations

import pytest
import torch


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return "gfx950" in arch


if not _is_gfx950():
    pytest.skip(
        "MORI EP compute helpers are gfx950 (CDNA4) only", allow_module_level=True
    )

import tokenspeed_kernel  # noqa: E402
from kimi3_reference import dequantize_mxfp4  # noqa: E402
from tokenspeed_kernel.ops.moe.mori.bf16 import masked_grouped_gemm  # noqa: E402
from tokenspeed_kernel.ops.moe.mori.mxfp4 import _grouped_mxfp4_gemm_3d  # noqa: E402
from tokenspeed_kernel.ops.moe.triton.mxfp4 import (  # noqa: E402
    grouped_mxfp4_ffn,
    triton_mxfp4_moe_weights,
)

with tokenspeed_kernel._triton.redirect_triton_to_tokenspeed_triton():  # noqa: E402
    from triton_kernels.tensor import make_ragged_tensor_metadata


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0
    ).item()


def _quant_mxfp4(w2d: torch.Tensor):
    # linear group-of-32 scale layout (matches the checkpoint layout the kernel expects)
    return tokenspeed_kernel.quantize_mxfp4(
        w2d.contiguous(), scale_size=32, scale_layout="linear"
    )


def _make_mxfp4_expert_weights(E: int, H: int, I: int, dev, gen):
    """Random bf16 experts -> per-expert MXFP4 (w13 [E,2I,H], w2 [E,H,I])."""
    w13 = (
        torch.randn(E, 2 * I, H, generator=gen, device=dev, dtype=torch.bfloat16) * 0.05
    )
    w2 = torch.randn(E, H, I, generator=gen, device=dev, dtype=torch.bfloat16) * 0.05
    w13_p = torch.stack([_quant_mxfp4(w13[e])[0] for e in range(E)])
    w13_s = torch.stack([_quant_mxfp4(w13[e])[1] for e in range(E)])
    w2_p = torch.stack([_quant_mxfp4(w2[e])[0] for e in range(E)])
    w2_s = torch.stack([_quant_mxfp4(w2[e])[1] for e in range(E)])
    return w13_p, w13_s, w2_p, w2_s


def _swizzled_weight_module(w13_p, w13_s, w2_p, w2_s, top_k=8):
    w = torch.nn.Module()
    w.w13_weight, w.w13_weight_scale = w13_p, w13_s
    w.w2_weight, w.w2_weight_scale = w2_p, w2_s
    w.top_k = top_k
    triton_mxfp4_moe_weights(
        {}, w
    )  # swizzle once -> w13_weight_triton_tensor, precision_config
    return w


def test_masked_grouped_gemm_matches_per_expert_loop() -> None:
    dev = torch.device("cuda", 0)
    gen = torch.Generator(device=dev).manual_seed(7)
    E, cap, H, I = 8, 32, 256, 128
    packed_x = (
        torch.randn(E, cap, H, generator=gen, device=dev, dtype=torch.bfloat16) * 0.1
    )
    w13 = (
        torch.randn(E, 2 * I, H, generator=gen, device=dev, dtype=torch.bfloat16) * 0.02
    )
    w2 = torch.randn(E, H, I, generator=gen, device=dev, dtype=torch.bfloat16) * 0.02
    counts = torch.tensor([0, 1, 5, 32, 17, 0, 8, 31], device=dev, dtype=torch.int32)

    got = masked_grouped_gemm(packed_x, counts, w13, w2)

    ref = torch.zeros_like(packed_x)
    for e in range(E):
        n = int(counts[e].item())
        if n == 0:
            continue
        h = packed_x[e, :n].float()
        gu = h @ w13[e].float().t()
        inter = torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]
        ref[e, :n] = (inter @ w2[e].float().t()).to(packed_x.dtype)

    valid = torch.arange(cap, device=dev)[None, :] < counts[:, None].to(torch.long)
    assert _cos(got[valid], ref[valid]) > 0.999
    # padding rows must be exactly zero
    assert got[~valid].abs().max().item() == 0.0


def test_grouped_mxfp4_ffn_matches_dequant_reference() -> None:
    dev = torch.device("cuda", 0)
    gen = torch.Generator(device=dev).manual_seed(3)
    E, H, I = 4, 256, 128
    counts_list = [8, 0, 20, 12]
    counts = torch.tensor(counts_list, device=dev, dtype=torch.int32)
    total = int(counts.sum().item())

    w13_p, w13_s, w2_p, w2_s = _make_mxfp4_expert_weights(E, H, I, dev, gen)
    w13_dq = dequantize_mxfp4(w13_p, w13_s).float()
    w2_dq = dequantize_mxfp4(w2_p, w2_s).float()
    w = _swizzled_weight_module(w13_p, w13_s, w2_p, w2_s)

    x_flat = (
        torch.randn(total, H, generator=gen, device=dev, dtype=torch.bfloat16) * 0.1
    )
    meta = make_ragged_tensor_metadata(counts, total)
    got = grouped_mxfp4_ffn(x_flat, meta, w).float()

    ref = torch.zeros(total, H, device=dev, dtype=torch.float32)
    off = 0
    for e, n in enumerate(counts_list):
        if n == 0:
            continue
        gu = x_flat[off : off + n].float() @ w13_dq[e].t()
        inter = torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]
        ref[off : off + n] = inter @ w2_dq[e].t()
        off += n

    assert _cos(got, ref) > 0.98


def test_grouped_mxfp4_gemm_3d_bridge() -> None:
    """The MORI 3-D [E,cap,H] <-> triton ragged bridge, native mxfp4, no dequant.

    ``_grouped_mxfp4_gemm_3d`` writes the FFN output back into ``packed_x`` IN PLACE
    (returns None) and deliberately leaves padding rows untouched (MORI combine reads
    only valid rows via src_info). So we compute the reference from a pre-call clone,
    then check: valid rows match the reference, and padding rows are left unmodified.
    """
    dev = torch.device("cuda", 0)
    gen = torch.Generator(device=dev).manual_seed(11)
    E, cap, H, I = 6, 32, 256, 128
    counts_list = [5, 0, 32, 1, 17, 8]
    counts = torch.tensor(counts_list, device=dev, dtype=torch.int32)

    w13_p, w13_s, w2_p, w2_s = _make_mxfp4_expert_weights(E, H, I, dev, gen)
    w13_dq = dequantize_mxfp4(w13_p, w13_s).float()
    w2_dq = dequantize_mxfp4(w2_p, w2_s).float()
    w = _swizzled_weight_module(w13_p, w13_s, w2_p, w2_s)

    # padding rows deliberately non-zero — must be left untouched by the gather/scatter
    packed = (
        torch.randn(E, cap, H, generator=gen, device=dev, dtype=torch.bfloat16) * 0.1
    )
    packed_before = packed.clone()

    # reference FFN over the ORIGINAL valid rows (before the in-place overwrite)
    ref = torch.zeros_like(packed, dtype=torch.float32)
    for e, n in enumerate(counts_list):
        if n == 0:
            continue
        gu = packed_before[e, :n].float() @ w13_dq[e].t()
        inter = torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]
        ref[e, :n] = inter @ w2_dq[e].t()

    # tight bound == sum(counts): exercises the m < E*cap ragged-truncation path and checks
    # no valid row is dropped (a too-small bound would silently lose tokens).
    n_recv_bound = int(counts.sum().item())
    out = _grouped_mxfp4_gemm_3d(
        packed, counts, w, n_recv_bound
    )  # in place -> returns None
    assert out is None
    got = packed  # mutated in place

    valid = torch.arange(cap, device=dev)[None, :] < counts[:, None].to(torch.long)
    assert _cos(got[valid], ref[valid]) > 0.98
    # padding rows are only ignored downstream, so they must be left exactly as-is
    assert torch.equal(got[~valid], packed_before[~valid])
