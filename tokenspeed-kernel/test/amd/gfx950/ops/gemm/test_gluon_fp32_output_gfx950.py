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


"""gfx950 FP32 projection dispatch, accumulation and graph replay."""

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("gfx950 required", allow_module_level=True)

from tokenspeed_kernel.ops.gemm import mm
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.fp32_output import (
    gluon_mm_bf16_fp32_narrow_gfx950,
    gluon_mm_bf16_fp32_num_counters_gfx950,
    gluon_mm_bf16_fp32_serial_gfx950,
    launch_gluon_mm_bf16_fp32_gfx950,
    launch_gluon_mm_bf16_fp32_serial_gfx950,
    launch_gluon_mm_bf16_fp32_split_gfx950,
    serial_projection_tile,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import gluon_mm_a16w16_medium_gfx950

# The Inkling2 projections of half the width: local and full QKV, a separate
# K or V source's rows of them, router, wo.
HALF_WIDTH = [
    (2152, 2304),
    (2168, 2304),
    (256, 2304),
    (264, 2304),
    (128, 2304),
    (2304, 768),
]


def split_counters(m, n):
    """Arrival counters of the program-a-split kernel's 16x64 tiles."""
    return torch.zeros(-(-m // 16) * -(-n // 64), device="cuda", dtype=torch.int32)


def project(a, b, out):
    return mm(
        a,
        b,
        A_scales=None,
        B_scales=None,
        bias=None,
        out=out,
        out_dtype=torch.float32,
        alpha=None,
        block_size=None,
        quant=None,
        override=None,
        solution=None,
        prepacked_scales=False,
    )


@pytest.mark.parametrize(
    "n,k",
    [(8, 4608), (384, 4608), (3792, 4608), (3808, 4608), (4608, 1536), *HALF_WIDTH],
)
def test_gfx950_projection_batch_invariance_and_accuracy(n, k):
    torch.manual_seed(8)
    source = torch.randn((1, k), device="cuda", dtype=torch.bfloat16) * 0.3
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.03
    expected = source.double() @ weight.double().T
    first = project(source, weight, None)
    torch.testing.assert_close(first.double(), expected, rtol=2e-6, atol=2e-6)
    # The MFMA kernels specialize runtime scalars by alignment class; the
    # rows choose among them (serial_projection_tile).
    for count in (16, 17, 96, 97, 1024, 1025):
        project(source.expand(count, -1).contiguous(), weight, None)
    with assert_no_triton_compile(
        gluon_mm_bf16_fp32_narrow_gfx950,
        gluon_mm_a16w16_medium_gfx950,
        gluon_mm_bf16_fp32_serial_gfx950,
    ):
        for count in (2, 4, 8, 24, 32, 64, 71, 79, 128, 256, 1287, 1962, 2048):
            inputs = torch.randn((count, k), device="cuda", dtype=torch.bfloat16)
            row = count // 2
            inputs[row].copy_(source[0])
            actual = project(inputs, weight, None)
            assert torch.equal(actual[row : row + 1], first)
    assert not torch.equal(first, first.bfloat16().float())


@pytest.mark.parametrize(
    "n,k",
    [(8, 4608), (384, 4608), (3792, 4608), (3808, 4608), (4608, 1536), *HALF_WIDTH],
)
@pytest.mark.parametrize("padding,offset", [(3, 0), (4, 1)])
def test_gfx950_projection_strided_output_graph(n, k, padding, offset):
    torch.manual_seed(11)
    a = torch.empty((17, k + padding), device="cuda", dtype=torch.bfloat16)[
        :, offset : offset + k
    ]
    b = torch.empty((n, k + padding + 2), device="cuda", dtype=torch.bfloat16)[
        :, offset : offset + k
    ]
    a.normal_(mean=0, std=0.3)
    b.normal_(mean=0, std=0.03)
    backing = torch.full((17, n + 3), 12345.0, device="cuda")
    out = backing[:, :n]
    assert project(a, b, out) is out
    torch.cuda.synchronize()
    with torch.cuda.stream(torch.cuda.Stream()):
        project(a, b, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        project(a, b, out)
    for _ in range(3):
        a.normal_(mean=0, std=0.3)
        b.normal_(mean=0, std=0.03)
        graph.replay()
        captured = out.clone()
        torch.testing.assert_close(
            captured.double(), a.double() @ b.double().T, rtol=2e-6, atol=2e-6
        )
        project(a, b, out)
        assert torch.equal(out, captured)
        assert (backing[:, n:] == 12345.0).all()


@pytest.mark.parametrize(
    "n,segments",
    [
        (2152, [(768, 1024)]),
        (2168, [(768, 1024), (2150, 2158)]),
        (2168, [(1024, 1280), (2158, 2166)]),
    ],
)
@pytest.mark.parametrize("m", [8, 129, 769, 2048])
def test_gfx950_qkv_rows_project_their_columns_of_the_whole_qkv(n, segments, m):
    """A K or V source's rows of the QKV weight give the bits of their
    columns in the whole projection."""
    torch.manual_seed(m + n)
    a = torch.randn((m, 2304), device="cuda", dtype=torch.bfloat16) * 0.3
    b = torch.randn((n, 2304), device="cuda", dtype=torch.bfloat16) * 0.03
    whole = project(a, b, None)
    rows = torch.cat([b[start:end] for start, end in segments])
    expected = torch.cat([whole[:, start:end] for start, end in segments], dim=1)
    assert torch.equal(project(a, rows, None), expected)


@pytest.mark.parametrize("operand", ["a", "b", "out"])
def test_gfx950_projection_rejects_buffer_offset_wrap(operand):
    a = torch.empty((2, 4608), device="cuda", dtype=torch.bfloat16)
    b = torch.empty((384, 4608), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((2, 384), device="cuda", dtype=torch.float32)
    if operand == "a":
        a = torch.empty_strided(a.shape, (2**31, 1), device=a.device, dtype=a.dtype)
    elif operand == "b":
        b = torch.empty_strided(b.shape, (2**23, 1), device=b.device, dtype=b.dtype)
    else:
        out = torch.empty_strided(
            out.shape, (2**30, 1), device=out.device, dtype=out.dtype
        )
    with pytest.raises(ValueError, match="32-bit buffer offsets"):
        project(a, b, out)


def test_gfx950_projection_rejects_split_k_offset_overflow():
    a = torch.empty((1, 4608), device="cuda", dtype=torch.bfloat16).expand(65536, -1)
    b = torch.empty((3792, 4608), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((1, 3792), device="cuda", dtype=torch.float32).expand(65536, -1)
    counters = torch.empty(0, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="Unsupported FP32 projection shape"):
        launch_gluon_mm_bf16_fp32_gfx950(a, b, out, counters=counters)


@pytest.mark.parametrize(
    "n,k", [(384, 4608), (3792, 4608), (3808, 4608), (4608, 1536), *HALF_WIDTH]
)
@pytest.mark.parametrize("tile", [(64, 64), (128, 128)])
@pytest.mark.parametrize("m", [1, 24, 128, 383, 2048, 4096])
def test_gfx950_projection_serial_splits_give_the_split_k_bits(n, k, tile, m):
    """Every K split in one program, added in split order, gives each output
    the bits of a program a split (FP32 words compared, -0.0 included)."""
    torch.manual_seed(m + n)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.3
    # A row of exact -0.0 sums: zero inputs but one tiny negative product.
    a[m // 2].zero_()
    a[m // 2, 5] = -1e-30
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.03
    counters = split_counters(m, n)
    split = launch_gluon_mm_bf16_fp32_split_gfx950(a, b, None, counters=counters)
    serial = launch_gluon_mm_bf16_fp32_serial_gfx950(a, b, None, tile=tile)
    assert torch.equal(serial.view(torch.int32), split.view(torch.int32))
    assert torch.signbit(serial[m // 2]).any()


@pytest.mark.parametrize("n,k", [(384, 4608), (3808, 4608), (4608, 1536)])
def test_gfx950_projection_serial_kernel_writes_strided_rows(n, k):
    torch.manual_seed(n)
    m = 300
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.03
    backing = torch.full((m, n + 5), 12345.0, device="cuda")
    out = backing[:, 2 : n + 2]
    counters = split_counters(m, n)
    expected = launch_gluon_mm_bf16_fp32_split_gfx950(a, b, None, counters=counters)
    assert launch_gluon_mm_bf16_fp32_serial_gfx950(a, b, out, tile=(64, 64)) is out
    assert torch.equal(out, expected)
    assert (backing[:, :2] == 12345.0).all() and (backing[:, n + 2 :] == 12345.0).all()


def test_gfx950_projection_serial_rows_need_no_counters():
    assert serial_projection_tile(24, 3808, 4608) is None
    assert gluon_mm_bf16_fp32_num_counters_gfx950(24, 3808, 4608) > 0
    assert serial_projection_tile(2048, 3808, 4608) == (128, 128)
    assert gluon_mm_bf16_fp32_num_counters_gfx950(2048, 3808, 4608) == 0
