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

import pytest
import torch
from tokenspeed_kernel.ops.embedding import host_gather, triton_host_gather
from utils import assert_no_triton_compile


def test_cpu_uint8_row_gather_masks_invalid_ids():
    table = torch.arange(20 * 8, dtype=torch.uint8).reshape(20, 8)
    indices = torch.tensor([[0, 3, -1, 19, 20]])
    out = host_gather.uint8_row_gather(table, indices, None)
    expected = table[indices.clamp(0, 19)]
    expected[indices < 0] = 0
    expected[indices >= 20] = 0
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_host_uint8_row_gather_matches_index():
    table = torch.arange(32 * 64, dtype=torch.uint8).reshape(32, 64)
    if torch.cuda.is_available():
        table = table.pin_memory()
    indices = torch.tensor([0, 7, 31, -1, 32], device="cuda:0")
    out = host_gather.uint8_row_gather(table, indices, None)
    expected = torch.zeros(5, 64, dtype=torch.uint8, device="cuda:0")
    expected[0] = table[0]
    expected[1] = table[7]
    expected[2] = table[31]
    torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=0, atol=0)


def _mxfp8_table(rows: int, dim: int, generator: torch.Generator) -> torch.Tensor:
    """Packed [codes | scales] rows; scales span blocks under the 1e-4 amax floor."""
    codes = torch.randint(0, 256, (rows, dim), dtype=torch.uint8, generator=generator)
    # 0x7F/0xFF are E4M3 NaN.
    codes[(codes & 0x7F) == 0x7F] = 0
    scales = torch.randint(
        95, 135, (rows, dim // 32), dtype=torch.uint8, generator=generator
    )
    codes[0] = 0
    return torch.cat([codes, scales], dim=1)


def _quantized_rows(table: torch.Tensor, indices: torch.Tensor):
    """Reference: dequantize the rows to BF16, then quantize them 1x32 UE8M0."""
    dim = table.shape[1] // 33 * 32
    ids = indices.cpu().long()
    valid = (ids >= 0) & (ids < table.shape[0])
    rows = table[ids.clamp(0, table.shape[0] - 1)].masked_fill(~valid[..., None], 0)
    codes = rows[..., :dim].view(torch.float8_e4m3fn).float()
    scales = (rows[..., dim:].int() << 23).view(torch.float32)
    values = (codes.unflatten(-1, (-1, 32)) * scales[..., None]).bfloat16().float()
    amax = values.abs().amax(-1).clamp_min(1e-4)
    bits = (amax / 448.0).view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).int()
    quantized = (values / (exponent << 23).view(torch.float32)[..., None]).to(
        torch.float8_e4m3fn
    )
    tokens = ids.shape[0]
    return quantized.view(torch.uint8).reshape(tokens, -1), exponent.to(
        torch.uint8
    ).reshape(tokens, -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_cuda_mxfp8_row_gather_matches_quantized_dequant(index_dtype):
    generator = torch.Generator().manual_seed(0)
    table = _mxfp8_table(40, 256, generator).pin_memory()
    ids = torch.randint(0, 40, (5, 3, 4), generator=generator)
    ids[0, 1, 0], ids[2, 1, 3], ids[4, 1, 1] = -1, 40, 0
    # A layer's slice of [tokens, layers, columns] ids: not contiguous.
    indices = ids.to(device="cuda:0", dtype=index_dtype)[:, 1]
    codes, scales = host_gather.mxfp8_row_gather(table, indices, None, None)
    expected_codes, expected_scales = _quantized_rows(table, indices)
    assert codes.shape == (5, 4 * 256) and codes.dtype == torch.float8_e4m3fn
    assert torch.equal(codes.view(torch.uint8).cpu(), expected_codes)
    assert torch.equal(scales.cpu(), expected_scales)
    # Most rows are re-scaled, not copied: the same values, new codes.
    assert not torch.equal(scales.cpu(), table[ids[:, 1].clamp(0, 39), 256:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_mxfp8_row_gather_reads_rows_past_2gib():
    # Row offsets are bytes: the last rows of a 2.2 GB table lie past 2^31.
    rows = (1 << 31) // 264 + 2
    table = torch.zeros(rows, 264, dtype=torch.uint8, device="cuda:0")
    tail = _mxfp8_table(3, 256, torch.Generator().manual_seed(1))
    table[-2:] = tail[1:].cuda()
    indices = torch.tensor([[rows - 1, rows - 2]], device="cuda:0")
    codes, scales = host_gather.mxfp8_row_gather(table, indices, None, None)
    expected_codes, expected_scales = _quantized_rows(tail, torch.tensor([[2, 1]]))
    assert torch.equal(codes.view(torch.uint8).cpu(), expected_codes)
    assert torch.equal(scales.cpu(), expected_scales)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_mxfp8_row_gather_does_not_recompile_per_batch():
    table = _mxfp8_table(64, 256, torch.Generator().manual_seed(2)).pin_memory()
    ids = torch.randint(0, 64, (64, 2, 24), device="cuda:0")
    codes = torch.empty(64, 24 * 256, dtype=torch.float8_e4m3fn, device="cuda:0")
    scales = torch.empty(64, 24 * 8, dtype=torch.uint8, device="cuda:0")
    host_gather.mxfp8_row_gather(table, ids[:3, 0], codes[:3], scales[:3])
    with assert_no_triton_compile(triton_host_gather._host_mxfp8_row_gather_kernel):
        for tokens in (1, 5, 16, 17, 64):
            host_gather.mxfp8_row_gather(
                table, ids[:tokens, 1], codes[:tokens], scales[:tokens]
            )
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_cuda_mxfp8_row_gather_sorted_path_matches_with_repeated_ids(index_dtype):
    # Enough entries to take the sorted path; few table rows so most ids
    # repeat and each run of equal ids is read once and copied to the rest.
    generator = torch.Generator().manual_seed(3)
    table = _mxfp8_table(97, 256, generator).pin_memory()
    columns = 24
    tokens = triton_host_gather._SORT_MIN_ROWS // columns + 3
    ids = torch.randint(0, 97, (tokens, 2, columns), generator=generator)
    ids[0, 1, 0], ids[5, 1, 7], ids[tokens - 1, 1, columns - 1] = -1, 97, -5
    indices = ids.to(device="cuda:0", dtype=index_dtype)[:, 1]
    assert indices.numel() >= triton_host_gather._SORT_MIN_ROWS
    codes, scales = host_gather.mxfp8_row_gather(table, indices, None, None)
    expected_codes, expected_scales = _quantized_rows(table, indices)
    assert torch.equal(codes.view(torch.uint8).cpu(), expected_codes)
    assert torch.equal(scales.cpu(), expected_scales)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_mxfp8_row_gather_sorted_path_reads_rows_past_2gib():
    # The sorted path widens its own id load, so it needs its own far-offset
    # case: enough entries to take it, all on the last two rows of a 2.2 GB
    # table, which puts the duplicate copies out there too. Only int32 ids
    # exercise the widening; an int64 id tensor loads wide already. A guard
    # region of the same size precedes the table, so an offset that wrapped
    # reads its zeros inside the allocation instead of faulting.
    rows = (1 << 31) // 264 + 2
    pool = torch.zeros(2 * rows, 264, dtype=torch.uint8, device="cuda:0")
    table = pool[rows:]
    tail = _mxfp8_table(3, 256, torch.Generator().manual_seed(5))
    table[-2:] = tail[1:].cuda()
    columns = 2
    tokens = triton_host_gather._SORT_MIN_ROWS // columns + 1
    indices = torch.full(
        (tokens, columns), rows - 1, dtype=torch.int32, device="cuda:0"
    )
    indices[::3, 0] = rows - 2
    assert indices.numel() >= triton_host_gather._SORT_MIN_ROWS
    codes, scales = host_gather.mxfp8_row_gather(table, indices, None, None)
    # Table row rows - 1 holds tail row 2, and rows - 2 holds tail row 1.
    expected_codes, expected_scales = _quantized_rows(tail, indices - (rows - 3))
    assert torch.equal(codes.view(torch.uint8).cpu(), expected_codes)
    assert torch.equal(scales.cpu(), expected_scales)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_mxfp8_row_gather_sorted_path_does_not_recompile_per_batch():
    table = _mxfp8_table(64, 256, torch.Generator().manual_seed(4)).pin_memory()
    columns = 24
    base = triton_host_gather._SORT_MIN_ROWS // columns + 1
    ids = torch.randint(0, 64, (base + 64, 2, columns), device="cuda:0")
    codes = torch.empty(
        base + 64, columns * 256, dtype=torch.float8_e4m3fn, device="cuda:0"
    )
    scales = torch.empty(base + 64, columns * 8, dtype=torch.uint8, device="cuda:0")
    host_gather.mxfp8_row_gather(table, ids[:base, 0], codes[:base], scales[:base])
    with assert_no_triton_compile(
        triton_host_gather._host_mxfp8_row_gather_kernel,
        triton_host_gather._copy_duplicate_rows_kernel,
    ):
        for tokens in (base + 2, base + 17, base + 33, base + 64):
            host_gather.mxfp8_row_gather(
                table, ids[:tokens, 1], codes[:tokens], scales[:tokens]
            )
    torch.cuda.synchronize()
