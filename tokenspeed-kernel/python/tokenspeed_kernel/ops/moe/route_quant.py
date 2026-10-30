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

"""Owned route metadata and quantized input for a native expert runner."""

from dataclasses import dataclass

import torch

TOKEN_BUCKETS = (1, 8, 16, 32, 64, 128, 256, 1024, 4096)


@dataclass(frozen=True)
class PreparedMoEInput:
    """One routing producer's inputs, maps and packed block-FP8 activations.

    ``inverse`` is a [tokens, topk] view of the native slot-major row map.
    ``source_rows`` stores slot*tokens+token for each initialized packed row.
    ``offsets[-1]`` bounds that prefix; unused rows and scale padding are not
    initialized. ``packed`` has a four-row-aligned capacity for native TMA
    descriptors; its padding is not an additional route. Keep all fields alive
    through the native consumer. Refresh this storage only after its earlier
    consumers have completed.
    """

    input: torch.Tensor
    ids: torch.Tensor
    expert_offset: int
    experts: int
    packed: torch.Tensor
    scales: torch.Tensor
    offsets: torch.Tensor
    source_rows: torch.Tensor
    inverse: torch.Tensor
    counts: torch.Tensor
    cursors: torch.Tensor
    scale_rows: torch.Tensor


def route_quant_fp8(
    input: torch.Tensor,
    ids: torch.Tensor,
    *,
    experts: int,
    expert_offset: int,
    solution: str | None,
    out: PreparedMoEInput | None = None,
) -> PreparedMoEInput:
    """Pack local expert routes and quantize BF16 inputs in one launch.

    Args:
        input: Finite contiguous BF16 CUDA input [tokens, hidden], with a
            positive model-static hidden size divisible by 128.
        ids: Contiguous int32 global expert IDs [tokens, topk]. IDs outside
            the selected local interval produce inverse=-1 and no packed row.
        experts: Model-static local expert count, currently 1..1024.
        expert_offset: Model-static start of the contiguous local expert interval.
        solution: Explicit solution, or None to select from the kernel registry.
        out: Optional existing storage bound to these input and ID tensors.

    Returns:
        Owned FP8 rows, padded FP32 scales, offsets and inverse/source maps.
        Current token buckets are 1, 8, 16, 32, 64, 128, 256, 1024 and 4096;
        top-k is a model-static value in 1..32. Warm each used bucket before
        CUDA graph capture. Other shapes must retain their existing producer.
    """
    if input.ndim != 2 or input.shape[1] <= 0 or input.shape[1] % 128:
        raise ValueError("Route input width must be a positive multiple of 128")
    hidden = input.shape[1]
    tokens = input.shape[0]
    if tokens not in TOKEN_BUCKETS:
        raise ValueError("Unsupported route token bucket")
    if ids.ndim != 2 or ids.shape[0] != tokens or not 1 <= ids.shape[1] <= 32:
        raise ValueError("Route IDs must have shape [tokens, topk] with 1..32 slots")
    if (
        type(experts) is not int
        or not 1 <= experts <= 1024
        or type(expert_offset) is not int
        or expert_offset < 0
        or expert_offset + experts > 2**31 - 1
    ):
        raise ValueError("Invalid local expert interval")
    if input.dtype != torch.bfloat16 or ids.dtype != torch.int32:
        raise ValueError("Expected BF16 inputs and int32 route IDs")
    if not input.is_contiguous() or not ids.is_contiguous():
        raise ValueError("Route inputs and IDs must be contiguous")
    if (
        not input.is_cuda
        or ids.device != input.device
        or input.data_ptr() % 16
        or ids.data_ptr() % 4
    ):
        raise ValueError("Route inputs must be aligned tensors on one CUDA device")
    topk = ids.shape[1]
    rows = tokens * topk
    if rows * hidden > 2**31 - 1:
        raise ValueError("Packed inputs exceed the int32 indexing range")
    packed_rows = ((rows + 3) // 4) * 4
    extent = ((rows + experts * 31 + 31) // 32) * 32
    if out is None:

        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=input.device)

        out = PreparedMoEInput(
            input,
            ids,
            expert_offset,
            experts,
            empty((packed_rows, hidden), torch.float8_e4m3fn),
            empty((hidden // 128, extent), torch.float32),
            empty((experts + 1,), torch.int64),
            empty((rows,), torch.int32),
            empty((topk, tokens), torch.int32).T,
            empty((experts,), torch.int32),
            empty((experts,), torch.int32),
            empty((rows,), torch.int32),
        )
    _validate_prepared_input(
        out, input=input, ids=ids, experts=experts, expert_offset=expert_offset
    )
    from tokenspeed_kernel.selection import select_kernel
    from tokenspeed_kernel.signature import dense_format_signature

    kernel = select_kernel(
        "moe",
        "route_quant_fp8",
        dense_format_signature(("input",), (input.dtype,)),
        solution=solution,
    )
    kernel(out)
    return out


def _validate_prepared_input(
    out: PreparedMoEInput,
    *,
    input: torch.Tensor,
    ids: torch.Tensor,
    experts: int,
    expert_offset: int,
) -> None:
    """Validate the same owned allocations at production and native consumption."""
    if type(out) is not PreparedMoEInput:
        raise TypeError("Expected the original PreparedMoEInput result")
    if input.ndim != 2 or input.shape[1] <= 0 or input.shape[1] % 128:
        raise ValueError("Route input width must be a positive multiple of 128")
    tokens, topk = ids.shape
    hidden = input.shape[1]
    rows = tokens * topk
    if rows * hidden > 2**31 - 1:
        raise ValueError("Packed inputs exceed the int32 indexing range")
    packed_rows = ((rows + 3) // 4) * 4
    extent = ((rows + experts * 31 + 31) // 32) * 32
    if (
        out.input is not input
        or out.ids is not ids
        or out.experts != experts
        or out.expert_offset != expert_offset
    ):
        raise ValueError("Prepared storage belongs to different inputs or experts")
    specs = (
        (out.packed, (packed_rows, hidden), (hidden, 1), torch.float8_e4m3fn, 16),
        (out.scales, (hidden // 128, extent), (extent, 1), torch.float32, 16),
        (out.offsets, (experts + 1,), (1,), torch.int64, 8),
        (out.source_rows, (rows,), (1,), torch.int32, 4),
        (out.inverse, (tokens, topk), (1, tokens), torch.int32, 4),
        (out.counts, (experts,), (1,), torch.int32, 4),
        (out.cursors, (experts,), (1,), torch.int32, 4),
        (out.scale_rows, (rows,), (1,), torch.int32, 4),
    )
    ranges = [
        (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
        for t in (input, ids)
    ]
    for tensor, shape, stride, dtype, alignment in specs:
        if (
            tensor.shape != shape
            or tensor.stride() != stride
            or tensor.dtype != dtype
            or tensor.device != input.device
            or tensor.data_ptr() % alignment
        ):
            raise ValueError("Incompatible route destination layout")
        start, size = (
            tensor.untyped_storage().data_ptr(),
            tensor.untyped_storage().nbytes(),
        )
        if any(
            max(start, other) < min(start + size, other + extent)
            for other, extent in ranges
        ):
            raise ValueError("Route destinations must have disjoint storage")
        ranges.append((start, size))
