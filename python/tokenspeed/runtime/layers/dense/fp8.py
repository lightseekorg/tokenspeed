# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
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
from tokenspeed_kernel.ops.gemm.fp8_utils import (
    per_block_quant_fp8,
    per_token_group_quant_fp8,
    per_token_quant_fp8,
    static_quant_fp8,
    swizzle_mxfp8_scale,
)
from torch.nn.parameter import Parameter

logger = logging.getLogger(__name__)

try:
    from tokenspeed_kernel.thirdparty.deep_gemm import ceil_to_ue8m0 as _ceil_to_ue8m0
    from tokenspeed_kernel.thirdparty.deep_gemm import (
        transform_sf_into_required_layout as _transform_sf,
    )
except ImportError:
    _ceil_to_ue8m0 = None
    _transform_sf = None

try:
    from tokenspeed_kernel.ops.gemm.flashinfer import (
        has_flashinfer_fp8_blockscale,
        has_flashinfer_mxfp8,
        prepare_flashinfer_fp8_blockscale_weight_scales,
    )
except ImportError:
    has_flashinfer_fp8_blockscale = None
    has_flashinfer_mxfp8 = None
    prepare_flashinfer_fp8_blockscale_weight_scales = None

from tokenspeed.runtime.layers.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from tokenspeed.runtime.layers.quantization.base_config import LinearMethodBase
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.layers.quantization.utils import (
    block_dequant,
    convert_to_channelwise,
)
from tokenspeed.runtime.utils.pdl import pdl_enabled


class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Limitations:
    1. Only support per-tensor quantization due to torch._scaled_mm support.
    2. Only support float8_e4m3fn data type due to the limitation of
       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.block_quant = self.quant_config.weight_block_size is not None

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

        if self.block_quant and self.quant_config.is_checkpoint_fp8_serialized:
            block_n, block_k = (
                self.quant_config.weight_block_size[0],
                self.quant_config.weight_block_size[1],
            )
            # Required by row parallel
            if input_size > input_size_per_partition:
                if input_size_per_partition % block_k != 0:
                    raise ValueError(
                        f"Weight input_size_per_partition = "
                        f"{input_size_per_partition} is not divisible by "
                        f"weight quantization block_k = {block_k}."
                    )
            # Required by column parallel or enabling merged weights
            if (
                output_size > output_size_per_partition
                or len(output_partition_sizes) > 1
            ):
                for output_partition_size in output_partition_sizes:
                    if output_partition_size % block_n != 0:
                        raise ValueError(
                            f"Weight output_partition_size = "
                            f"{output_partition_size} is not divisible by "
                            f"weight quantization block_n = {block_n}."
                        )

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # WEIGHT
        weight_dtype = (
            torch.float8_e4m3fn
            if self.quant_config.is_checkpoint_fp8_serialized
            else params_dtype
        )

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition, input_size_per_partition, dtype=weight_dtype
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # If checkpoint is serialized fp8, load them.
        # Otherwise, wait until process_weights_after_loading.
        if self.quant_config.is_checkpoint_fp8_serialized:
            # WEIGHT SCALE
            if self.block_quant:
                if hasattr(self.quant_config, "activation_scheme"):
                    if self.quant_config.activation_scheme != "dynamic":
                        raise ValueError(
                            "Block FP8 requires dynamic activation quantization."
                        )
                elif hasattr(self.quant_config, "linear_activation_scheme"):
                    if self.quant_config.linear_activation_scheme != "dynamic":
                        raise ValueError(
                            "Block FP8 requires dynamic linear activation quantization."
                        )
                scale_dtype = self.quant_config.weight_scale_dtype
                scale = BlockQuantScaleParameter(
                    data=torch.empty(
                        (output_size_per_partition + block_n - 1) // block_n,
                        (input_size_per_partition + block_k - 1) // block_k,
                        dtype=scale_dtype,
                    ),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=weight_loader,
                )
                if scale_dtype == torch.uint8:
                    scale.zero_()
                else:
                    scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("weight_scale_inv", scale)
            else:
                scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                    weight_loader=weight_loader,
                )
                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("weight_scale", scale)

            # INPUT ACTIVATION SCALE
            if (
                hasattr(self.quant_config, "activation_scheme")
                and self.quant_config.activation_scheme == "static"
            ) or (
                hasattr(self.quant_config, "linear_activation_scheme")
                and self.quant_config.linear_activation_scheme == "static"
            ):
                scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                    weight_loader=weight_loader,
                )

                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("input_scale", scale)
            else:
                layer.register_parameter("input_scale", None)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.block_quant:
            if not self.quant_config.is_checkpoint_fp8_serialized:
                qweight, weight_scale = per_block_quant_fp8(
                    layer.weight.data, self.quant_config.weight_block_size
                )
                layer.weight = Parameter(qweight, requires_grad=False)
                layer.register_parameter(
                    "weight_scale_inv", Parameter(weight_scale, requires_grad=False)
                )
                layer.input_scale = None
            layer._use_deep_gemm_fp8 = False
            layer._use_flashinfer_fp8_blockscale = False
            is_bmm = getattr(layer, "is_bmm", False)
            is_ue8m0 = getattr(self.quant_config, "scale_fmt", None) == "ue8m0"
            scale_requires_transform = (
                is_ue8m0 and layer.weight_scale_inv.dtype.is_floating_point
            )
            if (
                _transform_sf is not None
                and _ceil_to_ue8m0 is not None
                and scale_requires_transform
            ):
                N, K = layer.weight.shape
                block_n, block_k = self.quant_config.weight_block_size
                if is_bmm:
                    # Grouped (batched) projection (V4 attention wo_a, weight
                    # [groups * n, K], consumed per group as [n, K]). Transform
                    # the block scale into the deep_gemm MN-major layout with the
                    # group axis so deep_gemm.fp8_einsum("bhr,hdr->bhd") runs the
                    # output projection as one native FP8 GEMM (no FP32 dequant).
                    # recipe is (1, block_n, block_k) at load; the runtime einsum
                    # uses (1, 1, block_n) on SM100.
                    g = layer.bmm_batch_size
                    n = N // g
                    if n % block_n == 0 and K % block_k == 0:
                        sf = _ceil_to_ue8m0(layer.weight_scale_inv.data).view(
                            g, n // block_n, K // block_k
                        )
                        layer.weight_scale_inv.data = _transform_sf(
                            sf=sf,
                            mn=n,
                            k=K,
                            recipe=(1, block_n, block_k),
                            num_groups=g,
                            is_sfa=False,
                        )
                        layer._deep_gemm_block_size = [block_n, block_k]
                        layer._use_deep_gemm_fp8 = True
                elif N % 64 == 0 and K % 128 == 0:
                    sf = _ceil_to_ue8m0(layer.weight_scale_inv.data)
                    layer.weight_scale_inv.data = _transform_sf(
                        sf=sf,
                        mn=N,
                        k=K,
                        recipe=(1, block_n, block_k),
                        is_sfa=False,
                    )
                    layer._use_deep_gemm_fp8 = True
            layer._use_bmm_bf16_fallback = False
            if is_bmm and not layer._use_deep_gemm_fp8:
                # The grouped runtime path (DeepSeek-V4 attention wo_a) normally
                # runs as a single native FP8 GEMM through deep_gemm.fp8_einsum.
                # Where that path cannot be prepared -- notably on ROCm, which
                # has no deep_gemm build -- dequantize the block-scaled weight to
                # bfloat16 once at load so the projection can still run, as a
                # plain batched GEMM. The weight doubles in size and the GEMM is
                # slower, so the FP8 path stays preferred wherever it exists.
                # Where deep_gemm *is* available, a preparation failure means
                # the weight itself is wrong (unaligned dims, missing scales).
                # Keep failing fast there rather than quietly degrading.
                if _transform_sf is not None or layer.weight_scale_inv is None:
                    raise RuntimeError(
                        "is_bmm weight requires the deep_gemm FP8 block-scale "
                        "path but it could not be prepared (deep_gemm_available="
                        f"{_transform_sf is not None}, ue8m0={is_ue8m0}, "
                        f"weight={tuple(layer.weight.shape)}); ensure FP8 "
                        "block-quant ue8m0 weights with block-aligned dims and "
                        "deep_gemm installed."
                    )
                layer.weight = Parameter(
                    block_dequant(
                        layer.weight.data,
                        layer.weight_scale_inv.data,
                        list(self.quant_config.weight_block_size),
                    ).to(torch.bfloat16),
                    requires_grad=False,
                )
                layer._use_bmm_bf16_fallback = True
                logger.warning(
                    "Dequantizing is_bmm weight %s to bfloat16: the deep_gemm "
                    "FP8 block-scale path is unavailable (deep_gemm_available=%s, "
                    "ue8m0=%s). Expect higher memory use and lower throughput on "
                    "this projection.",
                    tuple(layer.weight.shape),
                    _transform_sf is not None,
                    is_ue8m0,
                )
            layer._use_flashinfer_mxfp8 = False
            if (
                not layer._use_deep_gemm_fp8
                and not is_bmm
                and has_flashinfer_mxfp8 is not None
                and has_flashinfer_mxfp8()
                and tuple(self.quant_config.weight_block_size) == (1, 32)
                and layer.weight_scale_inv.dtype == torch.uint8
                and layer.weight_scale_inv.dim() == 2
            ):
                N, K = layer.weight.shape
                if N >= 128 and K >= 128 and K % 32 == 0:
                    # Swizzle the e8m0 scales once into the F8_128x4 layout the
                    # flashinfer cute-dsl GEMM consumes; the Triton fallback
                    # cannot read this layout, so apply() pins the kernel.
                    layer.weight_scale_inv.data = swizzle_mxfp8_scale(
                        layer.weight_scale_inv.data, N, K
                    )
                    layer._use_flashinfer_mxfp8 = True
            if (
                not layer._use_deep_gemm_fp8
                and not layer._use_flashinfer_mxfp8
                and not is_bmm
                and has_flashinfer_fp8_blockscale is not None
                and has_flashinfer_fp8_blockscale()
                and prepare_flashinfer_fp8_blockscale_weight_scales is not None
                and tuple(self.quant_config.weight_block_size) == (128, 128)
                and layer.weight_scale_inv.dtype == torch.float32
                and layer.weight_scale_inv.dim() == 2
            ):
                N, K = layer.weight.shape
                if N % 128 == 0 and K % 128 == 0:
                    prepared_scales = prepare_flashinfer_fp8_blockscale_weight_scales(
                        layer.weight_scale_inv.data
                    )
                    buffer_name = "_flashinfer_fp8_weight_scales_mn"
                    if buffer_name in layer._buffers:
                        layer._buffers[buffer_name] = prepared_scales
                    else:
                        layer.register_buffer(
                            buffer_name,
                            prepared_scales,
                            persistent=False,
                        )
                    layer._use_flashinfer_fp8_blockscale = True
        else:
            layer.weight = Parameter(layer.weight.data, requires_grad=False)

            # If checkpoint not serialized fp8, quantize the weights.
            if not self.quant_config.is_checkpoint_fp8_serialized:
                # apply per-channel quantization default as
                qweight, weight_scale = per_token_group_quant_fp8(
                    layer.weight, layer.weight.shape[-1]
                )
                weight_scale = weight_scale.t().contiguous()

                # Update the layer with the new values.
                layer.weight = Parameter(qweight.t(), requires_grad=False)
                layer.weight_scale = Parameter(weight_scale, requires_grad=False)
                layer.input_scale = None

            # If checkpoint is fp8, handle that there are N scales for N
            # shards in a fused module
            else:
                layer.weight_scale = Parameter(
                    layer.weight_scale.data, requires_grad=False
                )
                if (
                    hasattr(self.quant_config, "activation_scheme")
                    and self.quant_config.activation_scheme == "static"
                ) or (
                    hasattr(self.quant_config, "linear_activation_scheme")
                    and self.quant_config.linear_activation_scheme == "static"
                ):
                    layer.input_scale = Parameter(
                        layer.input_scale.data, requires_grad=False
                    )

                weight = layer.weight
                weight_scale = convert_to_channelwise(
                    layer.weight_scale, layer.logical_widths
                )

                # Update layer with new values.
                layer.weight = Parameter(weight.t(), requires_grad=False)
                layer.weight_scale = Parameter(weight_scale, requires_grad=False)
                if (
                    hasattr(self.quant_config, "activation_scheme")
                    and self.quant_config.activation_scheme == "static"
                ) or (
                    hasattr(self.quant_config, "linear_activation_scheme")
                    and self.quant_config.linear_activation_scheme == "static"
                ):
                    layer.input_scale = Parameter(
                        layer.input_scale.max(), requires_grad=False
                    )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        block_scale: torch.Tensor | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:

        if self.block_quant:
            input_2d = x.view(-1, x.shape[-1])
            output_shape = [*x.shape[:-1], layer.weight.shape[0]]
            output_dtype = output_dtype or x.dtype

            if getattr(layer, "_use_deep_gemm_fp8", False):
                override = "deep_gemm_mm_fp8_blockscale"
            elif getattr(layer, "_use_flashinfer_mxfp8", False):
                override = "flashinfer_mm_mxfp8"
            elif (
                getattr(layer, "_use_flashinfer_fp8_blockscale", False)
                and block_scale is None
            ):
                override = "flashinfer_mm_fp8_blockscale"
            else:
                override = None
            use_flashinfer_fp8_blockscale = override == "flashinfer_mm_fp8_blockscale"
            output = tokenspeed_kernel.mm(
                input_2d,
                layer.weight,
                A_scales=block_scale,
                B_scales=(
                    layer._flashinfer_fp8_weight_scales_mn
                    if use_flashinfer_fp8_blockscale
                    else layer.weight_scale_inv
                ),
                bias=bias,
                out_dtype=output_dtype,
                quant="mxfp8",
                block_size=self.quant_config.weight_block_size,
                override=override,
                enable_pdl=pdl_enabled(),
                prepacked_scales=use_flashinfer_fp8_blockscale,
            )
            return output.to(dtype=output_dtype).view(*output_shape)
        else:
            input = x
            weight = layer.weight
            weight_scale = layer.weight_scale
            input_scale = layer.input_scale

            # View input as 2D matrix for fp8 methods
            input_2d = input.view(-1, input.shape[-1])
            output_shape = [*input.shape[:-1], weight.shape[1]]

            if input_scale is not None:
                if input_scale.numel() != 1:
                    raise ValueError(
                        f"input_scale must contain exactly one value, got {input_scale.numel()}."
                    )
                qinput, x_scale = static_quant_fp8(input_2d, input_scale)
            else:
                qinput, x_scale = per_token_quant_fp8(input_2d)

            qinput = qinput.view(-1, qinput.shape[-1])

            output = tokenspeed_kernel.mm(
                qinput,
                weight,
                A_scales=x_scale,
                B_scales=weight_scale,
                out_dtype=input.dtype,
            )
            if bias is not None:
                output = output + bias
            return output.view(*output_shape)
