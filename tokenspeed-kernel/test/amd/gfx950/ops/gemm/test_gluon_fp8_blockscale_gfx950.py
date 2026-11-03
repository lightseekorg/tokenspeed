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

"""Public FP32 block-scale dispatch, numerical boundaries and graph safety."""

from __future__ import annotations

import importlib

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.ops.gemm import mm as kernel_mm
from tokenspeed_kernel.ops.gemm.gluon import gluon_mm_fp8_blockscale_gfx950
from tokenspeed_kernel.ops.quantization import quantize_fp8
from tokenspeed_kernel.profiling import ShapeCapture
from tokenspeed_kernel.workspace import capture_session

amd_mm = importlib.import_module("tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8.mm")
amd_decode = importlib.import_module(
    "tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8.decode"
)


def strided_copy(value):
    backing = torch.empty(
        (2 * value.shape[0], 2 * value.shape[1]), device=value.device, dtype=value.dtype
    )
    backing[::2, ::2].copy_(value)
    return backing[::2, ::2]


def values(m, n, k):
    torch.manual_seed(7)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.5
    b_backing = torch.empty((n, k + 32), device="cuda", dtype=torch.float8_e4m3fn)
    b_backing[:, :k] = (torch.randn((n, k), device="cuda") * 0.5).to(
        torch.float8_e4m3fn
    )
    sa = strided_copy(torch.rand((m, (k + 127) // 128), device="cuda") + 0.37)
    sb = strided_copy(
        torch.rand(((n + 127) // 128, (k + 127) // 128), device="cuda") + 0.29
    )
    return a, b_backing[:, :k], sa, sb


def reference(a, b, sa, sb, out_dtype):
    m, k = a.shape
    n = b.shape[0]
    ar = a.float() * sa.repeat_interleave(128, 1)[:, :k]
    br = b.float() * sb.repeat_interleave(128, 0).repeat_interleave(128, 1)[:n, :k]
    return (ar @ br.T).to(out_dtype)


def run(a, b, sa, sb, out):
    return kernel_mm(
        a,
        b,
        A_scales=sa,
        B_scales=sb,
        out_dtype=torch.bfloat16,
        block_size=[128, 128],
        quant="mxfp8",
        out=out,
    )


# Model projection dimensions represent decode and the three matrix tile families.
# Row counts are test samples, not a claim about observed serving traffic.
@pytest.mark.parametrize(
    "shape,online,unaligned",
    [
        pytest.param((1, 4096, 512), False, False, id="shared-down-decode-single"),
        pytest.param((129, 4096, 512), False, False, id="shared-down-decode-batched"),
        pytest.param((1, 1024, 4096), False, False, id="shared-gate-up-small-matrix"),
        pytest.param(
            (1616, 1024, 4096), False, False, id="shared-gate-up-medium-matrix"
        ),
        pytest.param((3536, 4096, 512), False, False, id="shared-down-large-matrix"),
        # Synthetic API boundaries cover partial blocks and unaligned storage.
        pytest.param((17, 128, 3104), False, False, id="matrix-partial-k"),
        pytest.param((65, 1024, 4096), False, True, id="matrix-unaligned"),
        pytest.param((17, 130, 288), False, True, id="decode-partial-blocks-unaligned"),
        pytest.param((1, 4096, 512), True, False, id="decode-online-quantization"),
        pytest.param((1616, 1024, 4096), True, False, id="matrix-online-quantization"),
    ],
)
def test_public_fp32_scales_arbitrary_strides_and_canaries(shape, online, unaligned):
    m, n, k = shape
    a, b, sa, sb = values(m, n, k)
    if online:
        qa, actual_sa = quantize_fp8(
            a,
            granularity="token_group",
            group_size=128,
            scale_encoding="float32",
            solution="triton",
        )
        supplied_sa = None
    else:
        qa, actual_sa = a.to(torch.float8_e4m3fn), sa
        supplied_sa = sa
        a = qa
    if unaligned:
        operands = []
        for value in (a, b):
            backing = torch.empty(
                value.shape[0] * (k + 3) + 1, device=value.device, dtype=value.dtype
            )
            view = backing[1:].as_strided(value.shape, (k + 3, 1))
            view.copy_(value)
            operands.append(view)
        a, b = operands
        qa = a
    expected = reference(qa, b, actual_sa, sb, torch.bfloat16)
    backing = torch.full((m + 2, n + 32), -7, device="cuda", dtype=torch.bfloat16)
    out = backing[1 : m + 1, 16 : n + 16]
    saved = (actual_sa.clone(), sb.clone())
    ShapeCapture.reset()
    capture = ShapeCapture.get()
    capture.enabled = True
    try:
        actual = run(a, b, supplied_sa, sb, out)
    finally:
        capture.enabled = False
    assert capture._records[-1].kernel_name == "gluon_mm_fp8_blockscale_gfx950"
    assert actual is out and expected.abs().max().item() > 0.02 / 0.99
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.01)
    assert torch.all(backing[[0, -1], :] == -7)
    assert torch.all(backing[1 : m + 1, :16] == -7)
    assert torch.all(backing[1 : m + 1, n + 16 :] == -7)
    assert torch.equal(actual_sa, saved[0]) and torch.equal(sb, saved[1])


@pytest.mark.parametrize("shape", [(17, 130, 3072), (300, 384, 4096)])
@pytest.mark.parametrize("scale_name", ["A_scales", "B_scales"])
@pytest.mark.parametrize("invalid", ["dtype", "shape"])
def test_invalid_fp32_scale_reports_expected_and_received(shape, scale_name, invalid):
    m, n, k = shape
    a, b, sa, sb = values(m, n, k)
    expected = (m if scale_name == "A_scales" else (n + 127) // 128, (k + 127) // 128)
    received = (expected[0] + 1, expected[1]) if invalid == "shape" else expected
    dtype = torch.float16 if invalid == "dtype" else torch.float32
    bad_scale = torch.empty(received, device="cuda", dtype=dtype)
    with pytest.raises(ValueError) as raised:
        gluon_mm_fp8_blockscale_gfx950(
            a.to(torch.float8_e4m3fn),
            b,
            bad_scale if scale_name == "A_scales" else sa,
            bad_scale if scale_name == "B_scales" else sb,
            torch.bfloat16,
            alpha=None,
            block_size=[128, 128],
            out=None,
        )
    assert str(raised.value) == (
        f"{scale_name} must be FP32 with shape {expected}, "
        f"got dtype={dtype}, shape={received}"
    )


@pytest.mark.parametrize("scaled", [False, True])
def test_contiguous_single_row_preserves_output_and_multiplier(scaled):
    m, n, k = 1, 4096, 512
    a, b, sa, sb = values(m, n, k)
    a, b = a.to(torch.float8_e4m3fn), b.contiguous()
    alpha = torch.tensor(2.0, device="cuda") if scaled else None
    expected = reference(a, b, sa, sb, torch.bfloat16)
    if scaled:
        expected *= alpha
    backing = torch.full((m + 2, n + 32), -7, device="cuda", dtype=torch.bfloat16)
    out = backing[1 : m + 1, 16 : n + 16]
    actual = kernel_mm(
        a,
        b,
        A_scales=sa,
        B_scales=sb,
        out_dtype=torch.bfloat16,
        alpha=alpha,
        block_size=(128, 128),
        quant="mxfp8",
        out=out,
    )
    assert actual is out
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.01)
    assert torch.all(backing[[0, -1], :] == -7)
    assert torch.all(backing[1 : m + 1, :16] == -7)
    assert torch.all(backing[1 : m + 1, n + 16 :] == -7)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((65, 1024, 3072), id="split-decode"),
        pytest.param((1616, 1024, 4096), id="matrix"),
    ],
)
def test_fp32_scales_are_reread_during_graph_replay(shape):
    m, n, k = shape
    a = torch.full((m, k), 0.5, device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.full((n, k), 0.5, device="cuda", dtype=torch.float8_e4m3fn)
    sa = strided_copy(torch.ones((m, (k + 127) // 128), device="cuda"))
    sb = strided_copy(torch.ones(((n + 127) // 128, (k + 127) // 128), device="cuda"))
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    run(a, b, sa, sb, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with capture_session(graph), torch.cuda.graph(graph):
        run(a, b, sa, sb, out)
    for av, bv in ((1.25, 0.75), (0.5, 2.0)):
        sa.fill_(av)
        sb.fill_(bv)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            out, torch.full_like(out, k * 0.25 * av * bv), atol=0, rtol=0
        )


# Synthetic contract guards: combining scales first can lose a representable
# block result. These exponents do not represent measured model scale ranges.
@pytest.mark.parametrize(
    "shape,kind",
    [
        pytest.param(
            (1, 4096, 512), "overflow_coefficient", id="decode-scale-overflow"
        ),
        pytest.param(
            (1616, 1024, 4096), "underflow_coefficient", id="matrix-scale-underflow"
        ),
    ],
)
def test_fp32_scale_multiplications_keep_their_order(shape, kind):
    m, n, k = shape
    value, active, ae, be = (
        (2.0**-8, 1, 100, 30)
        if kind == "overflow_coefficient"
        else (384.0, 128, -100, -50)
    )
    a = torch.zeros((m, k), device="cuda")
    b = torch.zeros((n, k), device="cuda")
    for group in range(k // 128):
        a[:, group * 128 : group * 128 + active] = value
        b[:, group * 128 : group * 128 + active] = value
    sa = torch.exp2(
        (
            ae
            + torch.arange(m, device="cuda")[:, None] % 2
            + torch.arange(k // 128, device="cuda")[None, :] % 2
        ).float()
    )
    sb = torch.exp2(
        (
            be
            + torch.arange(n // 128, device="cuda")[:, None] % 2
            + torch.arange(k // 128, device="cuda")[None, :] % 2
        ).float()
    )
    expected = (value * value * active) * (
        sa.double()[:, None, :] * sb.double()[None, :, :]
    ).sum(-1)
    expected = expected.repeat_interleave(128, 1).to(torch.bfloat16)
    assert torch.isfinite(expected).all() and (expected != 0).all()
    actual = run(a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn), sa, sb, None)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "rows,n,k",
    [((129, 130), 4096, 512), ((848, 896), 6144, 4096), ((3536, 3540), 4096, 512)],
)
def test_same_tile_row_counts_reuse_compiled_kernels(rows, n, k):
    b = torch.ones((n, k), device="cuda", dtype=torch.float8_e4m3fn)
    sb = torch.ones(((n + 127) // 128, k // 128), device="cuda")

    def execute(m):
        a = torch.ones((m, k), device="cuda", dtype=torch.float8_e4m3fn)
        sa = torch.ones((m, k // 128), device="cuda")
        return run(a, b, sa, sb, None)

    execute(rows[0])
    with assert_no_triton_compile(
        amd_mm.gluon_mm_mxfp8_gfx950,
        amd_mm.gluon_mm_mxfp8_reduce_gfx950,
        amd_decode.gluon_mm_mxfp8_decode_gfx950,
    ):
        execute(rows[1])
