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
from tokenspeed_kernel.ops.moe.combine_quant import DeferredMoERoutes, combine_quant_fp8


def operands():
    return [
        torch.zeros((10, 512), dtype=torch.bfloat16),
        torch.arange(10, dtype=torch.int32).view(2, 5),
        torch.ones((2, 5)),
        torch.tensor([10], dtype=torch.int64),
    ]


@pytest.mark.parametrize(
    "index,replacement,message",
    [
        (0, torch.zeros((10, 257), dtype=torch.bfloat16), "multiple of 128"),
        (0, torch.zeros((5, 512), dtype=torch.bfloat16), "capacity >="),
        (1, torch.zeros((2, 6), dtype=torch.int32), "Route map"),
        (1, torch.zeros((2, 10), dtype=torch.int32)[:, ::2], "Route map"),
        (2, torch.zeros((2, 33)), "1..32 slots"),
        (2, torch.zeros((0, 5)), "1..4096"),
        (2, torch.zeros((2, 0)), "1..32 slots"),
        (2, torch.zeros((4097, 5)), "1..4096"),
        (3, torch.tensor(10, dtype=torch.int64), "tensor"),
        (0, torch.zeros((10, 512), dtype=torch.float32), "BF16 partial"),
        (1, torch.zeros((2, 5), dtype=torch.int64), "BF16 partial"),
        (2, torch.zeros((2, 5), dtype=torch.bfloat16), "BF16 partial"),
        (3, torch.tensor([10], dtype=torch.int32), "BF16 partial"),
        (0, torch.zeros((512, 10), dtype=torch.bfloat16).T, "contiguous"),
        (2, torch.zeros((5, 2)).T, "contiguous"),
    ],
)
def test_invalid_storage_rejected(index, replacement, message):
    values = operands()
    values[index] = replacement
    with pytest.raises(ValueError, match=message.replace("[", r"\[")):
        combine_quant_fp8(*values, solution=None)


def test_cpu_operands_are_not_dispatched():
    with pytest.raises(ValueError, match="CUDA device"):
        combine_quant_fp8(*operands(), solution=None)


def test_valid_row_view_tracks_in_place_native_refresh():
    partial, inverse, weights, _ = operands()
    offsets = torch.tensor([0, 8, 10])
    output = DeferredMoERoutes(partial, inverse, weights, offsets)
    view = output.valid_rows
    offsets[-1] = 5
    assert view.tolist() == [5]
    del output, offsets
    assert view.tolist() == [5]


@pytest.mark.parametrize("topk", (1, 2, 3, 5, 8, 16, 32))
@pytest.mark.parametrize("column_map", (False, True))
@pytest.mark.parametrize("hidden", (128, 256, 768, 1024))
def test_topk_reaches_device_validation(topk, column_map, hidden):
    inverse = torch.arange(2 * topk, dtype=torch.int32).view(2, topk)
    if column_map:
        inverse = torch.empty_strided((2, topk), (1, 2), dtype=torch.int32).copy_(
            inverse
        )
    with pytest.raises(ValueError, match="CUDA device"):
        combine_quant_fp8(
            torch.zeros((2 * topk, hidden), dtype=torch.bfloat16),
            inverse,
            torch.ones((2, topk)),
            torch.tensor([2 * topk], dtype=torch.int64),
            solution=None,
        )


def block_fp8_operands(experts, hidden, intermediate):
    # Meta tensors exercise the real shape contract without allocating model weights.
    return [
        torch.empty((8, hidden), dtype=torch.bfloat16, device="meta"),
        torch.empty(
            (experts, 2 * intermediate, hidden),
            dtype=torch.float8_e4m3fn,
            device="meta",
        ),
        torch.empty(
            (experts, hidden, intermediate), dtype=torch.float8_e4m3fn, device="meta"
        ),
        torch.empty((experts, 2 * intermediate // 128, hidden // 128), device="meta"),
        torch.empty((experts, hidden // 128, intermediate // 128), device="meta"),
    ]


@pytest.mark.parametrize("experts", (1, 33, 65))
@pytest.mark.parametrize(
    "hidden,intermediate", ((128, 256), (256, 768), (768, 512), (1024, 2048))
)
def test_deferred_native_geometry(experts, hidden, intermediate):
    from tokenspeed_kernel.thirdparty.flashinfer.sm90_moe import (
        _validate_deferred_weight_geometry,
    )

    assert _validate_deferred_weight_geometry(
        *block_fp8_operands(experts, hidden, intermediate)
    ) == (experts, hidden, intermediate)


@pytest.mark.parametrize(
    "experts,hidden,intermediate",
    ((0, 256, 768), (33, 0, 768), (33, 257, 768), (33, 256, 0), (33, 256, 769)),
)
def test_deferred_native_rejects_unaligned_geometry(experts, hidden, intermediate):
    from tokenspeed_kernel.thirdparty.flashinfer.sm90_moe import (
        _validate_deferred_weight_geometry,
    )

    with pytest.raises(ValueError):
        _validate_deferred_weight_geometry(
            *block_fp8_operands(experts, hidden, intermediate)
        )


@pytest.mark.parametrize("index", (1, 2, 3, 4))
def test_deferred_native_rejects_inconsistent_weights(index):
    from tokenspeed_kernel.thirdparty.flashinfer.sm90_moe import (
        _validate_deferred_weight_geometry,
    )

    values = block_fp8_operands(33, 256, 768)
    values[index] = values[index][:1]
    with pytest.raises(ValueError, match="weight or scale geometry"):
        _validate_deferred_weight_geometry(*values)


@pytest.mark.parametrize(
    "index,dtype",
    (
        (0, torch.float16),
        (1, torch.bfloat16),
        (2, torch.float8_e5m2),
        (3, torch.bfloat16),
        (4, torch.float64),
    ),
)
def test_deferred_native_rejects_other_quantization_formats(index, dtype):
    from tokenspeed_kernel.thirdparty.flashinfer.sm90_moe import (
        _validate_deferred_weight_geometry,
    )

    values = block_fp8_operands(33, 256, 768)
    values[index] = values[index].to(dtype)
    with pytest.raises(ValueError):
        _validate_deferred_weight_geometry(*values)


def test_combine_rejects_index_overflow():
    values = [
        torch.empty((131072, 16384), dtype=torch.bfloat16, device="meta"),
        torch.empty((4096, 32), dtype=torch.int32, device="meta"),
        torch.empty((4096, 32), device="meta"),
        torch.empty((1,), dtype=torch.int64, device="meta"),
    ]
    with pytest.raises(ValueError, match="indexing range"):
        combine_quant_fp8(*values, solution=None)
