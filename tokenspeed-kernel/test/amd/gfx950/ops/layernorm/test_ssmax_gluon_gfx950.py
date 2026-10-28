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

"""Byte-exact query/key and graph-position qualification, gfx950 only."""

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.ops.layernorm.gluon import NAME
from tokenspeed_kernel.ops.layernorm.qk_rmsnorm_ssmax import qk_rmsnorm_ssmax
from tokenspeed_kernel_amd.ops.gfx950.layernorm.qk_rmsnorm_ssmax import (
    gluon_qk_rmsnorm_ssmax_gfx950,
)


def fixture(rows, strided, dtype):
    generator = torch.Generator(device="cuda").manual_seed(320128)
    packed = torch.randn(
        (rows, 1792), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    q, k = packed[:, :768], packed[:, 768:896]
    if not strided:
        q, k = q.contiguous(), k.contiguous()
    weights = [
        (torch.rand(128, device="cuda", generator=generator) + 0.5).bfloat16()
        for _ in range(2)
    ]
    positions = torch.tensor(
        [0, 1, 2047, 2048, 49999, 50000, 131071, 1048575], device="cuda", dtype=dtype
    )
    positions = positions.repeat((rows + 7) // 8)[:rows].contiguous()
    return q, k, *weights, 1e-6, positions


def run(args, candidate):
    return qk_rmsnorm_ssmax(
        *args, solution=None, override=NAME if candidate else "torch_qk_rmsnorm_ssmax"
    )


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16, 32, 64, 128, 256, 848, 896, 8192])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_queries_and_unchanged_keys_are_byte_exact(rows, strided, dtype):
    args = fixture(rows, strided, dtype)
    actual, expected = run(args, True), run(args, False)
    for a, b in zip(actual, expected, strict=True):
        assert torch.isfinite(a).all()
        assert torch.equal(a, b)


def test_dynamic_positions_and_values_in_same_graph_without_compilation():
    args = fixture(33, True, torch.int64)
    run(args, True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(args, True)
    for value in [0, 2047, 2048, 50000, 131071]:
        args[-1].fill_(value)
        args[0].mul_(0.875)
        args[1].mul_(0.9375)
        with assert_no_triton_compile(gluon_qk_rmsnorm_ssmax_gfx950):
            graph.replay()
            eager = run(args, True)
        expected = run(args, False)
        for a, b, c in zip(captured, eager, expected, strict=True):
            assert torch.equal(a, b)
            assert torch.equal(a, c)


def test_row_counts_and_epsilon_are_runtime_values():
    run(fixture(17, True, torch.int64), True)
    with assert_no_triton_compile(gluon_qk_rmsnorm_ssmax_gfx950):
        for rows in [1, 2, 15, 16, 33, 127, 256, 848]:
            args = list(fixture(rows, True, torch.int64))
            args[-2] = 1e-5
            actual, expected = run(args, True), run(args, False)
            assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


def test_affine_bf16_boundary_cannot_be_elided():
    args = list(fixture(3, True, torch.int64))
    args[0].fill_(1.0)
    args[2].copy_((torch.arange(128, device="cuda").float() / 128 + 0.5).bfloat16())
    args[4] = 0.5
    args[5].fill_(65535)
    actual, expected = run(args, True), run(args, False)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    affine = torch.rsqrt(torch.tensor(1.5, device="cuda")) * args[2].float()
    scale = torch.exp(0.1 * torch.log1p(torch.tensor(65536.0, device="cuda") / 2048))
    skipped_boundary = (affine * scale).bfloat16().repeat(3, 6)
    assert (expected[0] != skipped_boundary).any()
