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

    assert (
        ple.ple_host_gather(
            table, ids, out, 0, 4, None, None, table_shards=4, background=False
        )
        is out
    )
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

    assert (
        ple.ple_host_gather(
            table, ids, out, 0, 4, None, None, table_shards=4, background=False
        )
        is out
    )
    pointer.assert_not_called()
    kernel.__getitem__.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("rows", [1, 255, 256, 257, 1027])
@pytest.mark.parametrize("head_dim", [32, 96, 128])
@pytest.mark.parametrize("scale_mode", ["none", "scalar", "row"])
@pytest.mark.parametrize("background", [False, True])
def test_host_gather_row_tiles(rows, head_dim, scale_mode, background):
    dtype = torch.bfloat16 if scale_mode == "none" else torch.float8_e4m3fn
    table = torch.randn(31, head_dim).to(dtype).pin_memory()
    ids = torch.arange(rows, device="cuda", dtype=torch.int64) % 40
    out = torch.empty((rows, head_dim), device="cuda", dtype=torch.bfloat16)
    scales = torch.linspace(0.5, 1.5, 31, device="cuda")
    ple.ple_host_gather(
        table,
        ids,
        out,
        3,
        34,
        0.5 if scale_mode == "scalar" else None,
        scales if scale_mode == "row" else None,
        table_shards=4,
        background=background,
    )
    valid = (ids >= 3) & (ids < 34)
    local = (ids - 3).clamp(0, 30)
    expected = table.float().cuda()[local]
    if scale_mode == "scalar":
        expected *= 0.5
    elif scale_mode == "row":
        expected *= scales[local, None]
    expected[~valid] = 0
    torch.testing.assert_close(out, expected.bfloat16(), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_host_gather_row_count_does_not_recompile():
    from tokenspeed_kernel.ops.ple.triton import _ple_host_gather_kernel
    from utils import assert_no_triton_compile

    table = torch.randn(31, 128, dtype=torch.bfloat16).pin_memory()
    ids = torch.zeros(2048, dtype=torch.int64, device="cuda")
    out = torch.empty((2048, 128), dtype=torch.bfloat16, device="cuda")

    def run(rows):
        ple.ple_host_gather(
            table,
            ids[:rows],
            out[:rows],
            0,
            31,
            None,
            None,
            table_shards=4,
            background=False,
        )

    for rows in (1, 16, 17, 256, 257):
        run(rows)
    with assert_no_triton_compile(_ple_host_gather_kernel):
        for rows in (32, 33, 255, 272, 273, 1024, 1027):
            run(rows)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("rows", [1, 255, 256, 257, 1027])
def test_host_gather_checkpoint_scale(rows):
    payload = torch.arange(256, dtype=torch.uint8).repeat(20)[: 31 * 160]
    payload[payload % 128 == 127] = 0
    table = payload.view(torch.float8_e4m3fn).reshape(31, 160).pin_memory()
    ids = torch.arange(rows, device="cuda", dtype=torch.int64) % 40
    out = torch.empty((rows, 160), device="cuda", dtype=torch.bfloat16)
    scale = 0.00019931793212890625
    ple.ple_host_gather(
        table, ids, out, 3, 34, scale, None, table_shards=4, background=False
    )
    cpu_ids = ids.cpu()
    valid = (cpu_ids >= 3) & (cpu_ids < 34)
    expected = table.float()[(cpu_ids - 3).clamp(0, 30)] * scale
    expected[~valid] = 0
    torch.testing.assert_close(out.cpu(), expected.bfloat16(), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("rows", [32768, 32769, 131071, 131072, 131073])
@pytest.mark.parametrize("table_shards", [1, 4])
@pytest.mark.parametrize("scale_mode", ["none", "scalar", "row"])
@pytest.mark.parametrize("background", [False, True])
def test_host_gather_persistent_and_sorted(rows, table_shards, scale_mode, background):
    dtype = torch.bfloat16 if scale_mode == "none" else torch.float8_e4m3fn
    table = torch.randn(37, 160).to(dtype).pin_memory()
    cpu_ids = torch.arange(rows, dtype=torch.int64) % 48 - 3
    cpu_ids[-3:] = torch.tensor([-(2**40), 2**31 + 3, 2**32 + 3])
    ids = cpu_ids.cuda()
    out = torch.empty((rows, 160), device="cuda", dtype=torch.bfloat16)
    scales = torch.linspace(0.5, 1.5, 37, device="cuda")
    scale = 0.00019931793212890625 if scale_mode == "scalar" else None

    def run():
        return ple.ple_host_gather(
            table,
            ids,
            out,
            3,
            40,
            scale,
            scales if scale_mode == "row" else None,
            table_shards=table_shards,
            background=background,
        )

    run()
    expected = table.float()[(cpu_ids - 3).clamp(0, 36)]
    if scale_mode == "scalar":
        expected *= scale
    elif scale_mode == "row":
        expected *= scales.cpu()[(cpu_ids - 3).clamp(0, 36), None]
    expected[(cpu_ids < 3) | (cpu_ids >= 40)] = 0
    torch.testing.assert_close(out.cpu(), expected.bfloat16(), atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    out.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(out.cpu(), expected.bfloat16(), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("table_shards", [1, 4])
@pytest.mark.parametrize("background", [False, True])
def test_host_gather_persistent_shapes_do_not_recompile(table_shards, background):
    from tokenspeed_kernel.ops.ple.triton import (
        _ple_host_gather_kernel,
        _ple_host_gather_persistent_kernel,
    )
    from utils import assert_no_triton_compile

    table = torch.ones((31, 160), dtype=torch.bfloat16).pin_memory()
    ids = torch.zeros(131200, device="cuda", dtype=torch.int64)
    out = torch.empty((131200, 160), device="cuda", dtype=torch.bfloat16)

    def run(rows):
        ple.ple_host_gather(
            table,
            ids[:rows],
            out[:rows],
            0,
            31,
            None,
            None,
            table_shards=table_shards,
            background=background,
        )

    for rows in (32768, 32769, 131072, 131073):
        run(rows)
    with assert_no_triton_compile(
        _ple_host_gather_kernel, _ple_host_gather_persistent_kernel
    ):
        for rows in (32784, 32785, 65536, 65539, 131088, 131089):
            run(rows)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_host_gather_sorted_large_table_offset():
    dim = 160
    table_rows = 2**31 // dim + 2
    table = torch.empty((table_rows, dim), dtype=torch.float8_e4m3fn, pin_memory=True)
    last = (torch.arange(dim).float() % 31 - 15).to(torch.float8_e4m3fn)
    table[-1].copy_(last)
    rows = 131073
    ids = torch.full((rows,), table_rows - 1, device="cuda", dtype=torch.int64)
    out = torch.empty((rows, dim), device="cuda", dtype=torch.bfloat16)
    scale = 0.00019931793212890625
    ple.ple_host_gather(
        table, ids, out, 0, table_rows, scale, None, table_shards=4, background=False
    )
    expected = (last.float() * scale).bfloat16().expand(rows, dim)
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("background", [False, True])
def test_sorted_host_gather_on_side_stream_with_graph(background):
    table = torch.randn(31, 160).to(torch.float8_e4m3fn).pin_memory()
    cpu_ids = torch.arange(131073, dtype=torch.int64) % 31
    ids = cpu_ids.cuda()
    out = torch.empty((len(cpu_ids), 160), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    ids.record_stream(stream)
    out.record_stream(stream)
    scale = 0.00019931793212890625

    def run():
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            ple.ple_host_gather(
                table,
                ids,
                out,
                0,
                31,
                scale,
                None,
                table_shards=1,
                background=background,
            )
        torch.cuda.current_stream().wait_stream(stream)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    out.fill_(float("nan"))
    graph.replay()
    expected = (table.float()[cpu_ids] * scale).bfloat16()
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)
