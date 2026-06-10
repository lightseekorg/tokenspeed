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

import torch
from torch import nn

from tokenspeed.runtime.layers.moe.backends.base import MoEBackend


def swizzle_blockscale(scales: torch.Tensor) -> torch.Tensor:
    scale_ndim = scales.ndim
    if scale_ndim == 2:
        scales = scales.unsqueeze(0)
    assert scales.ndim == 3
    batches, rows, cols = scales.shape

    def round_up(value: int, multiple: int) -> int:
        return (value + multiple - 1) // multiple * multiple

    rows_padded = round_up(rows, 128)
    cols_padded = round_up(cols, 4)
    padded = torch.zeros(
        (batches, rows_padded, cols_padded), dtype=scales.dtype, device=scales.device
    )
    padded[:batches, :rows, :cols] = scales
    padded = padded.reshape(batches, rows_padded // 128, 4, 32, cols_padded // 4, 4)
    padded = padded.permute((0, 1, 4, 3, 2, 5)).contiguous()
    if scale_ndim == 2:
        return padded.reshape(rows_padded, cols_padded)
    return padded.reshape(batches, rows_padded, cols_padded)


def interleave_gate_up_chunks(
    tensor: torch.Tensor, *, chunk_rows: int = 64, up_first: bool = True
) -> torch.Tensor:
    num_rows = tensor.shape[1]
    if num_rows % 2 != 0:
        raise ValueError(f"Expected even gate/up rows, got shape={tuple(tensor.shape)}")

    half_rows = num_rows // 2
    if half_rows % chunk_rows != 0:
        raise ValueError(
            f"Expected intermediate_size divisible by {chunk_rows}, got {half_rows}"
        )

    first_half = tensor[:, :half_rows, :]
    second_half = tensor[:, half_rows:, :]
    up, gate = (second_half, first_half) if up_first else (first_half, second_half)

    num_chunks = half_rows // chunk_rows
    up_chunks = up.reshape(tensor.shape[0], num_chunks, chunk_rows, tensor.shape[2])
    gate_chunks = gate.reshape(tensor.shape[0], num_chunks, chunk_rows, tensor.shape[2])
    return torch.stack((up_chunks, gate_chunks), dim=2).reshape_as(tensor).contiguous()


def create_fp4_weights(
    backend: MoEBackend,
    layer: nn.Module,
    num_local_experts: int,
    hidden_size: int,
    ispp: int,
    group_size: int,
    *,
    is_gated: bool = True,
) -> None:
    from tokenspeed.runtime.utils import set_weight_attrs

    # FP4 packed weights: 2 FP4 values per uint8 byte
    # w13 = gate_up_proj: [num_experts, 2*intermediate, hidden//2]
    # or up_proj for non-gated activations: [num_experts, intermediate, hidden//2]
    w13_rows = 2 * ispp if is_gated else ispp
    w13_weight = torch.nn.Parameter(
        torch.empty(num_local_experts, w13_rows, hidden_size // 2, dtype=torch.uint8),
        requires_grad=False,
    )
    # w2 = down_proj: [num_experts, hidden, intermediate//2]
    w2_weight = torch.nn.Parameter(
        torch.empty(num_local_experts, hidden_size, ispp // 2, dtype=torch.uint8),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight", w13_weight)
    layer.register_parameter("w2_weight", w2_weight)

    # Block scales (FP8-E4M3): one scale per group_size elements
    w13_weight_scale = torch.nn.Parameter(
        torch.empty(
            num_local_experts,
            w13_rows,
            hidden_size // group_size,
            dtype=torch.float8_e4m3fn,
        ),
        requires_grad=False,
    )
    w2_weight_scale = torch.nn.Parameter(
        torch.empty(
            num_local_experts,
            hidden_size,
            ispp // group_size,
            dtype=torch.float8_e4m3fn,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight_scale", w13_weight_scale)
    layer.register_parameter("w2_weight_scale", w2_weight_scale)

    # Per-tensor scales (float32)
    w13_scale_shape = (num_local_experts, 2) if is_gated else (num_local_experts,)
    w13_weight_scale_2 = torch.nn.Parameter(
        torch.empty(*w13_scale_shape, dtype=torch.float32), requires_grad=False
    )
    w2_weight_scale_2 = torch.nn.Parameter(
        torch.empty(num_local_experts, dtype=torch.float32), requires_grad=False
    )
    layer.register_parameter("w13_weight_scale_2", w13_weight_scale_2)
    layer.register_parameter("w2_weight_scale_2", w2_weight_scale_2)

    # Input scales (float32) - per-expert
    w13_input_scale = torch.nn.Parameter(
        torch.empty(*w13_scale_shape, dtype=torch.float32), requires_grad=False
    )
    w2_input_scale = torch.nn.Parameter(
        torch.empty(num_local_experts, dtype=torch.float32), requires_grad=False
    )
    layer.register_parameter("w13_input_scale", w13_input_scale)
    layer.register_parameter("w2_input_scale", w2_input_scale)

    # Set weight loaders
    weight_loader = backend._make_weight_loader()
    set_weight_attrs(w13_weight, {"weight_loader": weight_loader})
    set_weight_attrs(w2_weight, {"weight_loader": weight_loader})

    scale_loader = backend._make_group_scale_loader()
    set_weight_attrs(w13_weight_scale, {"weight_loader": scale_loader})
    set_weight_attrs(w2_weight_scale, {"weight_loader": scale_loader})

    per_tensor_loader = backend._per_tensor_scale_loader()
    set_weight_attrs(w13_weight_scale_2, {"weight_loader": per_tensor_loader})
    set_weight_attrs(w2_weight_scale_2, {"weight_loader": per_tensor_loader})
    set_weight_attrs(w13_input_scale, {"weight_loader": per_tensor_loader})
    set_weight_attrs(w2_input_scale, {"weight_loader": per_tensor_loader})


def finalize_common_flashinfer_weights(layer: nn.Module, *, swap_gate_up: bool) -> None:
    if swap_gate_up:
        # Swap w1 and w3 as the definition of
        # SwiGLU uses the fused-kernel layout.
        half_w = layer.w13_weight.shape[1] // 2
        temp_w = layer.w13_weight.data[:, :half_w, :].clone()
        layer.w13_weight.data[:, :half_w, :] = layer.w13_weight.data[:, half_w:, :]
        layer.w13_weight.data[:, half_w:, :] = temp_w
        del temp_w

        half_s = layer.w13_weight_scale.shape[1] // 2
        temp_s = layer.w13_weight_scale.data[:, :half_s, :].clone()
        layer.w13_weight_scale.data[:, :half_s, :] = layer.w13_weight_scale.data[
            :, half_s:, :
        ]
        layer.w13_weight_scale.data[:, half_s:, :] = temp_s
        del temp_s

    # Reduce w13_weight_scale_2: take per-expert value (w1 and w3 should match)
    w13_ws2 = layer.w13_weight_scale_2[:, 0]
    layer.w13_weight_scale_2 = torch.nn.Parameter(w13_ws2, requires_grad=False)

    # Shared FlashInfer FP4 kernels expect SCALAR input scales (global max)
    w13_input_scale = layer.w13_input_scale.max().to(torch.float32)
    w2_input_scale = layer.w2_input_scale.max().to(torch.float32)

    # Compute alpha = input_scale * weight_scale_2
    layer.g1_alphas = torch.nn.Parameter(
        (w13_input_scale * w13_ws2).to(torch.float32), requires_grad=False
    )
    layer.g2_alphas = torch.nn.Parameter(
        (w2_input_scale * layer.w2_weight_scale_2).to(torch.float32),
        requires_grad=False,
    )

    # Compute quantization inverse scales
    layer.w13_input_scale_quant = torch.nn.Parameter(
        (1.0 / w13_input_scale).to(torch.float32), requires_grad=False
    )
    layer.w2_input_scale_quant = torch.nn.Parameter(
        (1.0 / w2_input_scale).to(torch.float32), requires_grad=False
    )

    # Swizzle block scales for FlashInfer FP4 kernels, then free originals to
    # save memory.
    layer.w13_blockscale_swizzled = torch.nn.Parameter(
        swizzle_blockscale(layer.w13_weight_scale), requires_grad=False
    )
    del layer.w13_weight_scale

    layer.w2_blockscale_swizzled = torch.nn.Parameter(
        swizzle_blockscale(layer.w2_weight_scale), requires_grad=False
    )
    del layer.w2_weight_scale

    # Also free per-shard scales that are no longer needed
    del layer.w13_weight_scale_2
    del layer.w2_weight_scale_2
    del layer.w13_input_scale
    del layer.w2_input_scale


__all__ = [
    "create_fp4_weights",
    "finalize_common_flashinfer_weights",
    "interleave_gate_up_chunks",
]
