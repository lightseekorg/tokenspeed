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

"""Quantize an ordered expert-route reduction for its output projection."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeferredMoERoutes:
    """One invocation's owned expert outputs and routing metadata.

    ``partial`` is BF16 [tokens*topk, hidden]; ``inverse`` is an int32
    [tokens, topk] view, ``weights`` is FP32 [tokens, topk], and
    ``expert_offsets`` is int64 [local_experts+1]. The last offset bounds
    the initialized prefix of partial. Keep this object alive through its
    consumers; a subsequent call must not overwrite caller-supplied storage
    while earlier consumers can still read it.
    """

    partial: torch.Tensor
    inverse: torch.Tensor
    weights: torch.Tensor
    expert_offsets: torch.Tensor

    @property
    def valid_rows(self) -> torch.Tensor:
        return self.expert_offsets[-1:]


def combine_quant_fp8(
    partial: torch.Tensor,
    inverse: torch.Tensor,
    weights: torch.Tensor,
    valid_rows: torch.Tensor,
    *,
    solution: str | None,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine routes in slot order, round to BF16, and quantize to FP8.

    Args:
        partial: Contiguous BF16 expert outputs [capacity, hidden], with a
            positive model-static hidden size divisible by 128.
        inverse: Int32 [tokens, topk] indices into partial. Either token-major
            stride (topk, 1) or native slot-major stride (1, tokens) is accepted.
        weights: Contiguous FP32 route weights [tokens, topk].
        valid_rows: Int64 CUDA tensor [1] holding the current local row count.
            Negative indices and indices beyond this count contribute zero.
            All four inputs must come from the same completed routing producer.
        solution: Kernel solution, or None for registry selection.
        out: Optional disjoint destinations: contiguous E4M3 [tokens, hidden]
            and FP32 scales [tokens, hidden/128] with physical stride (1, tokens).

    Returns:
        Quantized values and scales for 128-channel blocks.
        Finite inputs, model-static top-k in 1..32, at most 4096 tokens, and
        stream dependencies are required. This operation performs no projection
        or communication.
    """
    if partial.ndim != 2 or partial.shape[1] <= 0 or partial.shape[1] % 128:
        raise ValueError("Expert output width must be a positive multiple of 128")
    hidden = partial.shape[1]
    quant_blocks = hidden // 128
    if partial.numel() > 2**31 - 1:
        raise ValueError("Expert outputs exceed the int32 indexing range")
    if weights.ndim != 2 or not 1 <= weights.shape[1] <= 32:
        raise ValueError(
            "Route weights must have shape [tokens, topk] with 1..32 slots"
        )
    tokens, topk = weights.shape
    if not 1 <= tokens <= 4096 or partial.shape[0] < tokens * topk:
        raise ValueError("Expected 1..4096 tokens and capacity >= tokens * topk")
    if inverse.shape != weights.shape or inverse.stride() not in (
        (topk, 1),
        (1, tokens),
    ):
        raise ValueError("Route map must be token-major or slot-major [tokens, topk]")
    if valid_rows.shape != (1,) or not valid_rows.is_contiguous():
        raise ValueError("Local row count must be a contiguous tensor [1]")
    inputs = (partial, inverse, weights, valid_rows)
    if tuple(t.dtype for t in inputs) != (
        torch.bfloat16,
        torch.int32,
        torch.float32,
        torch.int64,
    ):
        raise ValueError("Expected BF16 partial, int32 map, FP32 weights, int64 count")
    if not partial.is_contiguous() or not weights.is_contiguous():
        raise ValueError("Expert outputs and route weights must be contiguous")
    if not partial.is_cuda or any(t.device != partial.device for t in inputs):
        raise ValueError("Combine operands must share one CUDA device")
    if (
        partial.data_ptr() % 16
        or inverse.data_ptr() % 4
        or weights.data_ptr() % 4
        or valid_rows.data_ptr() % 8
    ):
        raise ValueError("Combine operands have incompatible pointer alignment")
    if out is None:
        out = (
            torch.empty(
                (tokens, hidden), dtype=torch.float8_e4m3fn, device=partial.device
            ),
            torch.empty_strided(
                (tokens, quant_blocks),
                (1, tokens),
                dtype=torch.float32,
                device=partial.device,
            ),
        )
    q, scales = out
    if (
        q.shape != (tokens, hidden)
        or q.dtype != torch.float8_e4m3fn
        or not q.is_contiguous()
        or scales.shape != (tokens, quant_blocks)
        or scales.dtype != torch.float32
        or scales.stride() != (1, tokens)
        or any(t.device != partial.device for t in out)
        or q.data_ptr() % 16
        or scales.data_ptr() % 4
    ):
        raise ValueError("Invalid block-FP8 quantization destinations")
    # Reject whole-storage overlap, including disjoint views of a shared allocation.
    output_ranges = [
        (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes()) for t in out
    ]
    input_ranges = [
        (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes()) for t in inputs
    ]
    for i, (start, size) in enumerate(output_ranges):
        for other, extent in input_ranges + output_ranges[:i]:
            if max(start, other) < min(start + size, other + extent):
                raise ValueError(
                    "Quantization destinations must not overlap inputs or each other"
                )
    from tokenspeed_kernel.selection import select_kernel
    from tokenspeed_kernel.signature import dense_format_signature

    kernel = select_kernel(
        "moe",
        "combine_quant_fp8",
        dense_format_signature(("partial",), (partial.dtype,)),
        solution=solution,
    )
    return kernel(partial, inverse, weights, valid_rows, out)
