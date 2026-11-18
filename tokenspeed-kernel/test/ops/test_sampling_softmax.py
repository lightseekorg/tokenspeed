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

"""The family-level ``softmax``: ``softmax(logits / T)`` of each row in fp32."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.sampling import softmax
from tokenspeed_kernel.ops.sampling.triton.softmax import (
    _chunk_stats_kernel,
    _normalize_kernel,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from utils import assert_no_triton_compile

# Kimi-K3's vocabulary, one 4096-column chunk and a column, and a row
# narrower than a chunk.
_VOCABS = (163840, 4097, 1000)
_RTOL = 1e-5
_ATOL = 1e-9


def _inputs(
    rows: int, vocab: int, dtype: torch.dtype, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    logits = (torch.randn(rows, vocab, generator=gen) * 4).to(device, dtype)
    temperature = torch.linspace(0.4, 1.6, rows, device=device)
    return logits, temperature


def _reference(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float() / temperature[:, None], dim=-1)


def test_selects_the_platform_kernel() -> None:
    kernel = select_kernel(
        "sampling",
        "softmax",
        format_signature(logits=dense_tensor_format(torch.bfloat16)),
    )
    expected = (
        "flashinfer_softmax" if current_platform().is_nvidia else "triton_softmax"
    )
    assert kernel.name == expected


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("vocab", _VOCABS)
def test_matches_torch(vocab: int, dtype: torch.dtype, device: str) -> None:
    logits, temperature = _inputs(5, vocab, dtype, device)
    got = softmax(logits, temperature)
    assert got.dtype == torch.float32 and got.is_contiguous()
    torch.testing.assert_close(
        got, _reference(logits, temperature), rtol=_RTOL, atol=_ATOL
    )


def test_reads_rows_with_a_wider_stride(device: str) -> None:
    base, temperature = _inputs(3, 5000, torch.bfloat16, device)
    logits = base[:, :4097]
    torch.testing.assert_close(
        softmax(logits, temperature),
        _reference(logits, temperature),
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_masked_columns_and_rows_without_a_finite_softmax(device: str) -> None:
    logits, temperature = _inputs(6, 10000, torch.float32, device)
    # More than a whole chunk of -inf ahead of the first finite column.
    logits[0, :5000] = float("-inf")
    # A NaN in a chunk with no finite value still makes the row NaN.
    logits[1, 4096:] = float("-inf")
    logits[1, 5000] = float("nan")
    logits[2, 9000] = float("inf")
    logits[3] = float("-inf")
    # An empty chunk must add 0 even when the row's maximum is far below 0.
    logits[5] -= 30000.0
    logits[5, :5000] = float("-inf")
    got = softmax(logits, temperature, solution="triton")
    expected = _reference(logits, temperature)
    for row in (0, 4, 5):
        torch.testing.assert_close(got[row], expected[row], rtol=_RTOL, atol=_ATOL)
    assert torch.equal(got[0, :5000], torch.zeros(5000, device=device))
    assert got[1:4].isnan().all()


def test_row_offsets_past_int32(device: str) -> None:
    # Row 1 starts 2^31 + 64 elements into the storage; only the two rows are
    # written or read.
    vocab = 1000
    stride = 2**31 + 64
    storage = torch.empty(stride + vocab, dtype=torch.bfloat16, device=device)
    logits = storage.as_strided((2, vocab), (stride, 1))
    src, temperature = _inputs(2, vocab, torch.bfloat16, device)
    logits.copy_(src)
    torch.testing.assert_close(
        softmax(logits, temperature, solution="triton"),
        _reference(src, temperature),
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_row_counts_do_not_recompile(device: str) -> None:
    logits, temperature = _inputs(64, _VOCABS[0], torch.bfloat16, device)
    softmax(logits[:3], temperature[:3], solution="triton")
    with assert_no_triton_compile(_chunk_stats_kernel, _normalize_kernel):
        for rows in (1, 8, 64):
            softmax(logits[:rows], temperature[:rows], solution="triton")
