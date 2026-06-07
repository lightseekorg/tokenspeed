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

import logging

import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.quantization.flashinfer import fp4_quantize
from torch.nn.parameter import Parameter

from tokenspeed.runtime.layers.quantization.base_config import QuantizeMethodBase

logger = logging.getLogger(__name__)


def _pdl_enabled() -> bool:
    from tokenspeed.runtime.utils.pdl import pdl_enabled

    return pdl_enabled()


class Nvfp4LinearMethod(QuantizeMethodBase):
    """Linear method for NVFP4 quantization.

    Weight structure:
    - weight: uint8 [output_size, input_size // 2] (packed FP4)
    - weight_scale: float8_e4m3fn [output_size, input_size // group_size]
    - weight_scale_2: float32 scalar (per-tensor)
    - input_scale: float32 scalar (per-tensor)
    """

    def __init__(self, quant_config):
        self.quant_config = quant_config
        self.group_size = quant_config.group_size

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # FP4 packed weight: 2 values per byte, input_dim halved
        weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        # Set attributes for TP sharding in weight_loader
        weight.output_dim = 0
        weight.input_dim = 1
        if weight_loader:
            weight.weight_loader = weight_loader
        layer.register_parameter("weight", weight)

        # Block scales: one per group_size elements
        weight_scale = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        weight_scale.output_dim = 0
        weight_scale.input_dim = 1
        if weight_loader:
            weight_scale.weight_loader = weight_loader
        layer.register_parameter("weight_scale", weight_scale)

        # Per-tensor scales: scalar per partition, use needs_scalar_to_array for fused loading
        input_scale = Parameter(
            torch.full(
                (len(output_partition_sizes),),
                torch.finfo(torch.float32).min,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        input_scale.needs_scalar_to_array = True
        if weight_loader:
            input_scale.weight_loader = weight_loader
        layer.register_parameter("input_scale", input_scale)

        weight_scale_2 = Parameter(
            torch.full(
                (len(output_partition_sizes),),
                torch.finfo(torch.float32).min,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        weight_scale_2.needs_scalar_to_array = True
        if weight_loader:
            weight_scale_2.weight_loader = weight_loader
        layer.register_parameter("weight_scale_2", weight_scale_2)

    def process_weights_after_loading(self, layer):
        """Compute alpha and input_scale_inv, swizzle block scales."""
        logger.debug(
            "[FP4_DENSE_POSTLOAD] w=%s(%s) ws=%s is=%s ws2=%s",
            layer.weight.shape,
            layer.weight.dtype,
            layer.weight_scale.shape,
            layer.input_scale,
            layer.weight_scale_2,
        )
        input_scale = layer.input_scale.max().to(torch.float32)
        weight_scale_2 = layer.weight_scale_2.max().to(torch.float32)
        layer.input_scale = Parameter(input_scale, requires_grad=False)
        layer.weight_scale_2 = Parameter(weight_scale_2, requires_grad=False)
        layer.alpha = Parameter(input_scale * weight_scale_2, requires_grad=False)
        layer.input_scale_inv = Parameter(
            (1.0 / input_scale).to(torch.float32), requires_grad=False
        )
        if layer.interleave_linear_and_gate:
            gate_weight, linear_weight = layer.weight.chunk(2, dim=0)
            linear_gate_weight = torch.cat((linear_weight, gate_weight), dim=0)
            layer.weight_swiglu_interleaved = Parameter(
                interleave_linear_and_gate(
                    linear_gate_weight,
                    group_size=64,
                    dim=0,
                ),
                requires_grad=False,
            )
            # layer.weight_scale is the canonical unswizzled [N, K/group]
            # tensor. Reorder gate/linear first, then swizzle for the CUTE
            # kernel; chunking layer.weight_scale_interleaved would be wrong.
            gate_scale, linear_scale = layer.weight_scale.chunk(2, dim=0)
            linear_gate_scale = torch.cat((linear_scale, gate_scale), dim=0)
            layer.weight_scale_swiglu_interleaved = Parameter(
                swizzle_blockscale_2d(
                    interleave_linear_and_gate(
                        linear_gate_scale,
                        group_size=64,
                        dim=0,
                    )
                ),
                requires_grad=False,
            )
            del layer.weight
            del layer.weight_scale
        else:
            # Swizzle block scales for CUTLASS
            layer.weight_scale_interleaved = Parameter(
                swizzle_blockscale_2d(layer.weight_scale),
                requires_grad=False,
            )
            del layer.weight_scale

    def apply(self, layer, x, bias=None):
        """Forward pass: quantize input to FP4, run FP4 GEMM.

        ``x`` may be either a bf16/fp16 activation tensor (normal path) or a
        pre-quantized ``(x_fp4, x_scale)`` tuple.
        """
        w_n = layer.output_size_per_partition

        if isinstance(x, tuple):
            # Pre-quantized path: no fp4_quantize launch. Output dtype is bf16.
            x_fp4, x_scale = x
            output_dtype = torch.bfloat16

        else:
            x_fp4, x_scale = fp4_quantize(
                x, layer.input_scale_inv, enable_pdl=_pdl_enabled()
            )
            output_dtype = x.dtype

        kernel_override = layer.override_kernel_name
        out = tokenspeed_kernel.mm(
            x_fp4,
            layer.weight.T,
            A_scales=x_scale,
            B_scales=layer.weight_scale_interleaved.T,
            bias=bias,
            out_dtype=output_dtype,
            alpha=layer.alpha,
            quant="nvfp4",
            enable_pdl=_pdl_enabled(),
            override=kernel_override,
            expected_kernel_name=kernel_override or "cublaslt_mm_nvfp4",
        )
        return out.view(x_fp4.size(0), w_n)


# -------------------------------------------------------------------------
# Utilities for FP4 linear method
# -------------------------------------------------------------------------


def swizzle_blockscale_2d(scales):
    """Swizzle 2D FP8 block scales for CUTLASS."""
    M, K = scales.shape

    def round_up(x, m):
        return (x + m - 1) // m * m

    M_padded = round_up(M, 128)
    K_padded = round_up(K, 4)
    padded = torch.zeros((M_padded, K_padded), dtype=scales.dtype, device=scales.device)
    padded[:M, :K] = scales
    rows, cols = padded.shape
    padded = padded.reshape(rows // 128, 4, 32, cols // 4, 4)
    padded = padded.permute((0, 3, 2, 1, 4))
    return padded.contiguous().reshape(M_padded, K_padded)


def interleave_linear_and_gate(
    tensor: torch.Tensor,
    group_size: int = 64,
    dim: int = 0,
) -> torch.Tensor:
    """Interleave ``[linear all][gate all]`` as ``[linear chunk][gate chunk]``.

    This matches the FC1 GEMM+SwiGLU preprocessing layout expected by
    ``cute_dsl_nvfp4_dense_gemm_swiglu_blackwell``.
    """
    if tensor.ndim == 0:
        raise ValueError("expected a tensor with at least one dimension")

    dim = dim % tensor.ndim
    sizes = tensor.size()
    dim_size = sizes[dim]
    if dim_size % (group_size * 2) != 0:
        raise ValueError(
            f"dimension {dim} size {dim_size} must be divisible by "
            f"2 * group_size={2 * group_size}"
        )

    prev_sizes = sizes[:dim]
    post_sizes = sizes[dim + 1 :]
    return (
        tensor.reshape(
            *prev_sizes,
            2,
            dim_size // (group_size * 2),
            group_size,
            *post_sizes,
        )
        .transpose(dim, dim + 1)
        .reshape(*sizes)
        .contiguous()
    )
