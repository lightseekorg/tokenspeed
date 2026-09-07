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

"""FlashInfer FA2 fallback for direct-slot QSA sparse attention."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement, pdl_enabled
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from tokenspeed_kernel.thirdparty.flashinfer.qsa_sparse import (
    get_flashinfer_qsa_sparse_runner,
)

_SUPPORTED_HEAD_DIMS = frozenset({64, 128, 256})


@triton.jit
def _prepare_flashinfer_qsa_metadata_kernel(
    selected_slots,
    indices,
    packed_mask,
    stride_s_n,
    stride_s_k,
    stride_i_n,
    packed_stride,
    WIDTH: tl.constexpr,
    PACKED_WIDTH: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    row = tl.program_id(0)
    byte_offsets = tl.program_id(1) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    bit_offsets = tl.arange(0, 8)
    columns = byte_offsets[:, None] * 8 + bit_offsets[None, :]
    column_mask = columns < WIDTH
    slots = tl.load(
        selected_slots + row * stride_s_n + columns * stride_s_k,
        mask=column_mask,
        other=-1,
    ).to(tl.int32)
    valid = column_mask & (slots > 0)
    tl.store(
        indices + row * stride_i_n + columns,
        tl.where(valid, slots, 0),
        mask=column_mask,
    )
    packed = tl.sum(valid.to(tl.int32) << bit_offsets[None, :], axis=1)
    tl.store(
        packed_mask + row * packed_stride + byte_offsets,
        packed,
        mask=byte_offsets < PACKED_WIDTH,
    )


def _prepare_flashinfer_qsa_metadata(
    selected_slots: torch.Tensor,
    indices: torch.Tensor,
    packed_mask: torch.Tensor,
    *,
    enable_pdl: bool,
) -> None:
    """Sanitize slots and update a cached FA2 plan's packed validity mask."""

    if selected_slots.ndim != 2 or indices.shape != selected_slots.shape:
        raise ValueError("QSA slots and FlashInfer indices must share [rows, width]")
    if indices.dtype != torch.int32 or not indices.is_contiguous():
        raise TypeError("FlashInfer QSA indices must be contiguous int32")
    if packed_mask.dtype != torch.uint8 or not packed_mask.is_contiguous():
        raise TypeError("FlashInfer QSA packed mask must be contiguous uint8")
    if selected_slots.device != indices.device or indices.device != packed_mask.device:
        raise ValueError("QSA slots, indices, and packed mask must share one device")
    rows, width = selected_slots.shape
    packed_width = (width + 7) // 8
    if packed_mask.numel() != rows * packed_width:
        raise ValueError(
            "FlashInfer QSA packed mask has the wrong size: "
            f"expected {rows * packed_width}, got {packed_mask.numel()}"
        )
    if rows == 0 or width == 0:
        return
    block_bytes = 128
    pdl_kwargs = {"launch_pdl": True} if enable_pdl else {}
    _prepare_flashinfer_qsa_metadata_kernel[
        (rows, triton.cdiv(packed_width, block_bytes))
    ](
        selected_slots,
        indices,
        packed_mask,
        selected_slots.stride(0),
        selected_slots.stride(1),
        indices.stride(0),
        packed_width,
        WIDTH=width,
        PACKED_WIDTH=packed_width,
        BLOCK_BYTES=block_bytes,
        num_warps=4,
        num_stages=1,
        **pdl_kwargs,
    )


_BF16_SIGNATURE = format_signature(
    q=dense_tensor_format(torch.bfloat16),
    k_cache=dense_tensor_format(torch.bfloat16),
    v_cache=dense_tensor_format(torch.bfloat16),
)
_FP8_SIGNATURE = format_signature(
    q=dense_tensor_format(torch.bfloat16),
    k_cache=dense_tensor_format(torch.float8_e4m3fn),
    v_cache=dense_tensor_format(torch.float8_e4m3fn),
)


@register_kernel(
    "attention",
    "qsa_sparse_attention",
    name="flashinfer_fa2_qsa_sparse_attention",
    solution="flashinfer",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(8, 0),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=frozenset({_BF16_SIGNATURE}),
    traits={
        "head_dim": _SUPPORTED_HEAD_DIMS,
        "value_head_dim": _SUPPORTED_HEAD_DIMS,
    },
    priority=Priority.PERFORMANT,
    tags={"fallback", "fa2", "sparse"},
)
@register_kernel(
    "attention",
    "qsa_sparse_attention",
    name="flashinfer_fa2_fp8_qsa_sparse_attention",
    solution="flashinfer",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 0),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=frozenset({_FP8_SIGNATURE}),
    traits={
        "head_dim": _SUPPORTED_HEAD_DIMS,
        "value_head_dim": _SUPPORTED_HEAD_DIMS,
    },
    priority=Priority.PERFORMANT,
    tags={"fallback", "fa2", "fp8", "sparse"},
)
def flashinfer_fa2_qsa_sparse_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    scale: float,
    max_seqlen_q: int,
    metadata_capacity_rows: int | None,
    k_scale: float | torch.Tensor | None,
    v_scale: float | torch.Tensor | None,
) -> torch.Tensor:
    """Run the BF16 or FP8-cache NVIDIA QSA fallback with FlashInfer FA2."""

    del max_seqlen_q  # Each selected-slot row is an independent FA2 query row.

    runner = get_flashinfer_qsa_sparse_runner(q.device)
    plan = runner.plan(
        q,
        k_cache,
        v_cache,
        selected_slots.shape[1],
        softmax_scale=scale,
        metadata_capacity_rows=metadata_capacity_rows,
    )
    use_pdl = pdl_enabled()
    _prepare_flashinfer_qsa_metadata(
        selected_slots,
        plan.indices,
        plan.packed_mask,
        enable_pdl=use_pdl,
    )
    return runner.run(
        plan,
        q,
        k_cache,
        v_cache,
        k_scale=k_scale,
        v_scale=v_scale,
        enable_pdl=use_pdl,
    )


__all__ = ["flashinfer_fa2_qsa_sparse_attention"]
