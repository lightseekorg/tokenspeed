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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from tokenspeed_kernel.ops import ple


def test_ple_host_gather_uses_device_visible_table_pointer(monkeypatch) -> None:
    table = torch.empty((4, 8), dtype=torch.bfloat16)
    ids = torch.tensor([[0, 2]], dtype=torch.int64)
    out = torch.empty((2, 8), dtype=torch.bfloat16)
    pointer = MagicMock(return_value=12345)
    kernel = MagicMock()
    monkeypatch.setattr(
        ple,
        "current_platform",
        lambda: SimpleNamespace(device_visible_data_ptr=pointer),
    )
    monkeypatch.setattr(ple, "_ple_host_gather_kernel", kernel)

    assert ple.ple_host_gather(table, ids, out, 0, 4, None, None, table_shards=4) is out
    pointer.assert_called_once_with(table)
    args, kwargs = kernel.__getitem__.return_value.call_args
    assert args[0] == 12345
    assert args[1].shape == (2,)
    assert args[4].shape == (2, 8)
    assert kwargs["IS_FP8"] is False


def test_ple_host_gather_empty_ids_skip_pointer_mapping(monkeypatch) -> None:
    table = torch.empty((4, 8), dtype=torch.bfloat16)
    ids = torch.empty(0, dtype=torch.int64)
    out = torch.empty((0, 8), dtype=torch.bfloat16)
    pointer = MagicMock()
    kernel = MagicMock()
    monkeypatch.setattr(
        ple,
        "current_platform",
        lambda: SimpleNamespace(device_visible_data_ptr=pointer),
    )
    monkeypatch.setattr(ple, "_ple_host_gather_kernel", kernel)

    assert ple.ple_host_gather(table, ids, out, 0, 4, None, None, table_shards=4) is out
    pointer.assert_not_called()
    kernel.__getitem__.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "head_dim, table_shards, scale_mode",
    [(160, 4, "scalar"), (512, 4, "row"), (512, 4, "none"), (160, 1, "scalar")],
    ids=["flash-tp4", "max-online-fp8", "max-bf16", "flash-tp1"],
)
def test_host_gather_decode_and_prefill(head_dim, table_shards, scale_mode):
    from tokenspeed_kernel.ops.ple.triton import (
        _ple_host_gather_kernel,
        _ple_host_gather_persistent_kernel,
    )
    from utils import assert_no_triton_compile

    vocab_start = 20_000_000 if table_shards > 1 else 0
    dtype = torch.bfloat16 if scale_mode == "none" else torch.float8_e4m3fn
    table = torch.randn(128, head_dim).to(dtype).pin_memory()
    scalar = 0.00019931793212890625 if scale_mode == "scalar" else None
    scales = torch.linspace(0.5, 1.5, 128)
    row_scale = scales.cuda() if scale_mode == "row" else None

    def check(tokens):
        local_ids = torch.arange(tokens * 16) % 144 - 8
        ids = (vocab_start + local_ids).reshape(tokens, 16).cuda()
        out = torch.empty((ids.numel(), head_dim), device="cuda", dtype=torch.bfloat16)
        ple.ple_host_gather(
            table,
            ids,
            out,
            vocab_start,
            vocab_start + 128,
            scalar,
            row_scale,
            table_shards=table_shards,
        )
        rows = local_ids.clamp(0, 127)
        expected = table.float()[rows]
        if scalar is not None:
            expected *= scalar
        if row_scale is not None:
            expected *= scales[rows, None]
        expected[(local_ids < 0) | (local_ids >= 128)] = 0
        torch.testing.assert_close(out.cpu(), expected.bfloat16(), atol=0, rtol=0)

    for tokens in (1, 16, 8192):
        check(tokens)
    with assert_no_triton_compile(
        _ple_host_gather_kernel, _ple_host_gather_persistent_kernel
    ):
        for tokens in (4, 32, 16384):
            check(tokens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_host_gather_large_table_offset():
    dim = 160
    table_rows = 2**31 // dim + 2
    table = torch.empty((table_rows, dim), dtype=torch.float8_e4m3fn, pin_memory=True)
    last = (torch.arange(dim).float() % 31 - 15).to(table.dtype)
    table[-1].copy_(last)
    ids = torch.full((4, 16), table_rows - 1, device="cuda", dtype=torch.int64)
    out = torch.empty((ids.numel(), dim), device="cuda", dtype=torch.bfloat16)
    scale = 0.00019931793212890625
    ple.ple_host_gather(table, ids, out, 0, table_rows, scale, None, table_shards=4)
    expected = (last.float() * scale).bfloat16().expand_as(out)
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)
