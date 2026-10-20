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

"""compressed-tensors FP8 W8A8 linears (llm-compressor ``FP8``/``FP8_DYNAMIC``).

The checkpoint stores ``weight`` as FP8 E4M3 ``[N, K]`` and ``weight_scale``
per output channel ``[N, 1]`` (strategy ``channel``) or per tensor ``[1]``
(strategy ``tensor``, one per shard of a fused linear), in the scale dtype the
tool wrote (BF16 by default, FP32, FP16 or FP8 E4M3); every value converts
exactly to the FP32 parameters here. Activations are quantized per token at run
time (``dynamic``) or with the checkpoint's static per-tensor ``input_scale``.

Where tokenspeed_kernel registers a dense FP8 GEMM with per-token activation
and per-channel weight scales (``gemm.mm`` on ``scaled-fp8`` channel scales)
and a per-token FP8 quantizer of its input (NVIDIA arch 10 and AMD gfx950,
today) the layer runs on them. Elsewhere (SM90 and SM120
among others) the weight is dequantized to the model dtype at load and the
layer runs as an unquantized one; a warning names the platform. A dequantized
value is the exact product of code and scale rounded once (BF16, FP16 or
FP32).
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable

import torch
from compressed_tensors.quantization import QuantizationStrategy
from torch.nn.parameter import Parameter

from tokenspeed.runtime.layers.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from tokenspeed.runtime.layers.quantization.compressed_tensors.schemes.compressed_tensors_scheme import (
    CompressedTensorsScheme,
)
from tokenspeed.runtime.layers.quantization.utils import convert_to_channelwise

logger = logging.getLogger(__name__)

__all__ = [
    "CompressedTensorsW8A8Fp8",
    "dequantize_fp8_per_channel",
    "fp8_channel_gemm_available",
    "fp8_dynamic_linear",
    "fp8_unscaled_gemm_available",
    "fp8_unscaled_linear",
]

_FP8 = torch.float8_e4m3fn


@functools.cache
def _warn_once(message: str) -> None:
    logger.warning(message)


def fp8_channel_gemm_available(platform=None) -> bool:
    """Whether a dense FP8 GEMM taking per-token activation and per-output-
    channel weight FP32 scales is registered for ``platform`` (default: this
    process's), and the per-token FP8 quantizer of its input."""
    import tokenspeed_kernel.ops.gemm  # noqa: F401  (registers the GEMMs)
    import tokenspeed_kernel.ops.quantization  # noqa: F401  (and the quantizers)
    from tokenspeed_kernel.platform import current_platform
    from tokenspeed_kernel.registry import KernelRegistry
    from tokenspeed_kernel.signature import (
        ScaleFormat,
        dense_tensor_format,
        format_signature,
        tensor_format,
    )

    platform = platform or current_platform()
    registry = KernelRegistry.get()
    scale = ScaleFormat(storage_dtype=torch.float32, granularity="channel")
    operand = tensor_format("scaled-fp8", _FP8, scale=scale)
    kernels = registry.get_for_operator(
        "gemm",
        "mm",
        platform=platform,
        format_signature=format_signature(a=operand, b=operand),
    )
    # Check the shared activation format rather than requiring a vendor's
    # implementation. Both the portable and vendor quantizers return FP32
    # per-token scales for BF16/FP16 inputs.
    quantizers = [
        spec
        for spec in registry.get_for_operator(
            "quantization",
            "fp8_with_scale",
            platform=platform,
            format_signature=format_signature(x=dense_tensor_format(torch.bfloat16)),
        )
        if "token" in spec.traits.get("granularity", ())
        and "float32" in spec.traits.get("scale_encoding", ())
        and "m_max" not in spec.traits
        and "token_fp8_min_scaling" in spec.features
    ]
    return bool(kernels) and bool(quantizers)


def fp8_unscaled_gemm_available(platform=None) -> bool:
    """Whether a vendor (not Triton, not torch) FP8 GEMM taking one FP32
    scale per operand is registered for ``platform`` (default: this
    process's): the GEMM ``fp8_unscaled_linear`` runs with unit scales,
    leaving the per-token and per-channel scales to the kernel that consumes
    its output."""
    import tokenspeed_kernel.ops.gemm  # noqa: F401  (registers the GEMMs)
    from tokenspeed_kernel.platform import current_platform
    from tokenspeed_kernel.registry import KernelRegistry
    from tokenspeed_kernel.signature import ScaleFormat, format_signature, tensor_format

    scale = ScaleFormat(storage_dtype=torch.float32, granularity="tensor")
    operand = tensor_format("scaled-fp8", _FP8, scale=scale)
    kernels = KernelRegistry.get().get_for_operator(
        "gemm",
        "mm",
        platform=platform or current_platform(),
        format_signature=format_signature(a=operand, b=operand),
    )
    return any(spec.solution not in ("triton", "torch") for spec in kernels)


def fp8_unscaled_linear(
    codes: torch.Tensor, weight: torch.Tensor, unit_scale: torch.Tensor
) -> torch.Tensor:
    """The FP8 codes' product with an FP8 weight, unscaled, in BF16.

    The per-tensor FP8 GEMM the registry selects (a vendor GEMM where
    ``fp8_unscaled_gemm_available``) with both scales 1: the output is the
    exact product of the codes accumulated in FP32 and rounded once to BF16.
    ``|code| <= 448``, so a sum over K stays below ``448^2 * K`` (8.2e8 at
    K = 4,096), far inside BF16's range, and the rounding is relative (2^-9),
    as the scaled GEMM's own output rounding is. The consumer of the output
    applies the per-token and per-channel scales
    (``tokenspeed_kernel.ops.quantization.triton.fp8_dequantize_unscaled``).

    Args:
        codes: ``[M, K]`` FP8 E4M3 activation codes.
        weight: ``[K, N]`` FP8 weight (a transposed ``[N, K]``).
        unit_scale: A one-element FP32 tensor holding 1 on the device,
            allocated outside CUDA-graph capture by the caller.

    Returns:
        ``[M, N]`` BF16 ``codes @ weight``.

    Raises:
        ValueError: ``weight`` is KN-contiguous instead of a transposed
            ``[N, K]``.
    """
    # GEMM selection keys the weight layout on ``B.stride(-1) == 1``, and every
    # vendor per-tensor FP8 GEMM declares ``b_inner_stride_one: False``. A
    # KN-contiguous weight would select the Triton GEMM run with unit scales
    # instead, slower than the scaled GEMM it replaces while the caller still
    # pays the consumer that applies the scales.
    if weight.stride(-1) == 1:
        raise ValueError(
            "weight must be a [K, N] view of a contiguous [N, K] FP8 weight "
            f"(a transposed weight has stride(-1) != 1), got shape "
            f"{tuple(weight.shape)} stride {weight.stride()}: no vendor "
            "per-tensor FP8 GEMM takes a KN-contiguous weight"
        )

    from tokenspeed_kernel.ops.gemm import mm

    return mm(
        codes,
        weight,
        A_scales=unit_scale,
        B_scales=unit_scale,
        out_dtype=torch.bfloat16,
        quant="fp8",
    )


def _round_once(exact: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # FP8 x FP32 products have at most 28 significant bits and are exact in
    # FP64. Round them once, to nearest even: FP64 -> FP32 is one rounding,
    # but Torch casts FP64 to BF16 and FP16 through FP32, which rounds twice.
    if dtype not in (torch.bfloat16, torch.float16):
        return exact.to(dtype)
    bits = 8 if dtype == torch.bfloat16 else 11
    mantissa, exponent = torch.frexp(exact)
    rounded = torch.ldexp(torch.round(mantissa * 2**bits) / 2**bits, exponent)
    if dtype == torch.float16:
        # Below 2**-14 FP16 keeps a fixed step of 2**-24 (subnormals).
        subnormal = torch.round(exact * 2**24) / 2**24
        rounded = torch.where(exact.abs() < 2**-14, subnormal, rounded)
    # The rounded values are exact in FP32 and in ``dtype`` (or overflow).
    return rounded.to(dtype)


def dequantize_fp8_per_channel(
    weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """``dtype`` values of FP8 E4M3 ``weight [N, K]`` times its scale per
    output channel (``[N, 1]``, ``[N]``) or per tensor (one value)."""
    scale = scale.double()
    scale = scale.reshape(()) if scale.numel() == 1 else scale.reshape(-1, 1)
    return _round_once(weight.double() * scale, dtype)


def _check_scales(layer: torch.nn.Module, name: str, scale: torch.Tensor) -> None:
    # A scale the checkpoint did not provide keeps its NaN fill.
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise ValueError(
            f"{layer.prefix}.{name}: "
            "compressed-tensors FP8 scales must be finite and positive "
            "(missing from the checkpoint?)"
        )


class CompressedTensorsW8A8Fp8(CompressedTensorsScheme):
    """FP8 E4M3 weights with static per-channel or per-tensor scales, FP8
    activations quantized per token at run time or with a static per-tensor
    input scale."""

    def __init__(self, strategy: str, is_static_input_scheme: bool) -> None:
        if strategy not in (QuantizationStrategy.CHANNEL, QuantizationStrategy.TENSOR):
            raise ValueError(
                f"compressed-tensors FP8 W8A8 weights take a channel or tensor "
                f"scale, got strategy {strategy!r}"
            )
        self.strategy = strategy
        self.is_static_input_scheme = is_static_input_scheme
        # Decided at load: True when no FP8 GEMM serves the platform and the
        # weight was dequantized.
        self.dequantized = False
        self._unquantized = None

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition, input_size_per_partition, dtype=_FP8
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        if self.strategy == QuantizationStrategy.CHANNEL:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.full(
                    (output_size_per_partition, 1), float("nan"), dtype=torch.float32
                ),
                output_dim=0,
                weight_loader=weight_loader,
            )
        else:
            weight_scale = PerTensorScaleParameter(
                data=torch.full(
                    (len(output_partition_sizes),), float("nan"), dtype=torch.float32
                ),
                weight_loader=weight_loader,
            )
        layer.register_parameter("weight_scale", weight_scale)

        if self.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(
                data=torch.full(
                    (len(output_partition_sizes),), float("nan"), dtype=torch.float32
                ),
                weight_loader=weight_loader,
            )
            layer.register_parameter("input_scale", input_scale)
        else:
            layer.register_parameter("input_scale", None)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_scale = layer.weight_scale.data
        if self.strategy == QuantizationStrategy.TENSOR:
            # One scale per shard of a fused linear, as one per output channel.
            weight_scale = convert_to_channelwise(weight_scale, layer.logical_widths)
        _check_scales(layer, "weight_scale", weight_scale)
        input_scale = None
        if self.is_static_input_scheme:
            _check_scales(layer, "input_scale", layer.input_scale.data)
            # The shards of a fused linear share one input: the largest scale.
            input_scale = layer.input_scale.data.max()

        if fp8_channel_gemm_available():
            # tokenspeed_kernel.ops.gemm.mm takes the weight as [K, N].
            layer.weight = Parameter(layer.weight.data.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            if input_scale is not None:
                layer.input_scale = Parameter(input_scale, requires_grad=False)
            return

        from tokenspeed_kernel.platform import current_platform

        platform = current_platform()
        _warn_once(
            "compressed-tensors FP8 W8A8 linears are dequantized to "
            f"{str(layer.orig_dtype).removeprefix('torch.')} at load: no FP8 "
            "GEMM with per-channel weight scales and per-token FP8 input "
            f"quantization is registered for {platform.vendor} "
            f"{platform.arch_version}"
        )
        dequantized = dequantize_fp8_per_channel(
            layer.weight.data, weight_scale, layer.orig_dtype
        )
        layer.weight = Parameter(dequantized, requires_grad=False)
        layer.register_parameter("weight_scale", None)
        layer.register_parameter("input_scale", None)
        from tokenspeed.runtime.layers.dense import UnquantizedLinearMethod

        self._unquantized = UnquantizedLinearMethod()
        self._unquantized.process_weights_after_loading(layer)
        self.dequantized = True

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.dequantized:
            assert self._unquantized is not None
            return self._unquantized.apply(layer, x, bias)

        input_2d = x.reshape(-1, x.shape[-1])
        if layer.input_scale is not None:
            from tokenspeed_kernel.ops.gemm import mm
            from tokenspeed_kernel.ops.quantization import quantize_fp8

            qinput, x_scale = quantize_fp8(input_2d, scale=layer.input_scale)
            output = mm(
                qinput,
                layer.weight,
                A_scales=x_scale,
                B_scales=layer.weight_scale,
                out_dtype=x.dtype,
                quant="fp8",
            )
        else:
            output = fp8_dynamic_linear(input_2d, layer.weight, layer.weight_scale)
        if bias is not None:
            output = output + bias
        return output.view(*x.shape[:-1], layer.weight.shape[1])


def fp8_dynamic_linear(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor:
    """``x @ weight`` of an FP8 W8A8 linear with dynamic per-token activations.

    Args:
        x: ``[M, K]`` BF16 or FP16 activations.
        weight: ``[K, N]`` FP8 E4M3 weight (a transposed ``[N, K]``), as
            ``process_weights_after_loading`` leaves it.
        weight_scale: FP32 ``[N, 1]`` per-channel scales.

    Returns:
        ``[M, N]`` output in ``x``'s dtype. ``tokenspeed_kernel.ops.gemm.mm``
        quantizes ``x`` per token in one launch before the GEMM (the Triton
        quantizer for decode rows on NVIDIA and every row count on gfx950;
        TRT-LLM's above 64 rows on NVIDIA).
    """
    from tokenspeed_kernel.ops.gemm import mm

    return mm(x, weight, B_scales=weight_scale, out_dtype=x.dtype, quant="fp8")
