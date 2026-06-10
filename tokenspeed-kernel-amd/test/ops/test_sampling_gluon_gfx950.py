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

from __future__ import annotations

import pytest
import torch

MODEL_VOCABS = {
    "deepseek_v4": 129280,
    "qwen3_5": 151936,
    "minimax_m2": 200064,
}


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return "gfx950" in arch


if not _is_gfx950():
    pytest.skip(
        "AMD GFX950 is required for Gluon argmax tests", allow_module_level=True
    )


from tokenspeed_kernel_amd.ops.sampling.gluon import (  # noqa: E402
    argmax_gfx950,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_argmax_matches_torch_for_dtypes(dtype):
    torch.manual_seed(0xA950)
    x = torch.randn(8, 4096, device="cuda", dtype=dtype)
    out = argmax_gfx950.argmax(x)
    torch.testing.assert_close(out, torch.argmax(x, dim=-1), atol=0, rtol=0)


@pytest.mark.parametrize(
    "M,N",
    [
        (1, MODEL_VOCABS["deepseek_v4"]),
        (2, MODEL_VOCABS["deepseek_v4"]),
        (3, MODEL_VOCABS["deepseek_v4"]),
        (4, MODEL_VOCABS["deepseek_v4"]),
        (16, MODEL_VOCABS["qwen3_5"]),
        (64, MODEL_VOCABS["minimax_m2"]),
        (128, MODEL_VOCABS["qwen3_5"]),
    ],
)
def test_argmax_matches_torch_for_model_shapes(M, N):
    torch.manual_seed(M ^ N)
    x = 0.1 * torch.randn(M, N, device="cuda", dtype=torch.float32)
    out = argmax_gfx950.argmax(x)
    torch.testing.assert_close(out, torch.argmax(x, dim=-1), atol=0, rtol=0)


@pytest.mark.parametrize(
    "M,N,dtype",
    [
        (1, MODEL_VOCABS["deepseek_v4"], torch.float32),
        (4, MODEL_VOCABS["deepseek_v4"], torch.float32),
        (8, MODEL_VOCABS["deepseek_v4"], torch.float16),
        (128, MODEL_VOCABS["deepseek_v4"], torch.bfloat16),
    ],
)
def test_argmax_all_nan_rows_return_sentinel(M, N, dtype):
    x = torch.full((M, N), float("nan"), device="cuda", dtype=dtype)
    out = argmax_gfx950.argmax(x)
    expected = torch.full((M,), -1, device="cuda", dtype=out.dtype)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.parametrize("M", [4, 128])
def test_argmax_ignores_nan_but_preserves_valid_negative_infinity(M):
    N = MODEL_VOCABS["deepseek_v4"]
    x = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float32)
    x[0, 123] = 0.5
    x[0, 456] = 1.0
    x[1].fill_(-float("inf"))
    x[2, 7] = 3.0
    x[2, 5] = 3.0

    out = argmax_gfx950.argmax(x)
    expected = torch.full((M,), -1, device="cuda", dtype=out.dtype)
    expected[:3] = torch.tensor([456, 0, 5], device="cuda", dtype=out.dtype)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_argmax_returns_first_index_on_ties():
    M, N = 4, 4096
    x = torch.full((M, N), -100.0, device="cuda", dtype=torch.float32)
    plant_positions = [
        [0, 7, 9],
        [3, 4],
        [128, 1024, 2048],
        [N - 1, 17],
    ]
    for row, positions in enumerate(plant_positions):
        for pos in positions:
            x[row, pos] = 0.0
    torch.testing.assert_close(
        argmax_gfx950.argmax(x), torch.argmax(x, dim=-1), atol=0, rtol=0
    )


@pytest.mark.parametrize("out_dtype", [torch.int32, torch.int64])
def test_argmax_writes_into_strided_caller_buffer(out_dtype):
    M, N = 8, 4096
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    storage = torch.empty(M * 2, device="cuda", dtype=out_dtype)
    out = storage[::2]
    returned = argmax_gfx950.argmax(x, out=out)
    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out.long(), torch.argmax(x, dim=-1), atol=0, rtol=0)


def test_argmax_out_buffer_under_cuda_graph():
    M, N = 16, MODEL_VOCABS["deepseek_v4"]
    torch.manual_seed(M ^ N ^ 0xC0DE)
    x = 0.1 * torch.randn(M, N, device="cuda", dtype=torch.float32)
    out = torch.empty(M, dtype=torch.int32, device="cuda")

    argmax_gfx950.argmax(x, out=out)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        argmax_gfx950.argmax(x, out=out)

    new_x = 0.1 * torch.randn_like(x)
    x.copy_(new_x)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.long(), torch.argmax(x, dim=-1), atol=0, rtol=0)
