# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import annotations

from typing import Literal

import torch
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "quantize_fp8",
    "quantize_fp8_with_scale",
    "quantize_mxfp8",
    "quantize_nvfp4",
    "quantize_mxfp4",
]


def quantize_fp8(
    x: torch.Tensor,
    scale: float | torch.Tensor | None = None,
    # kernel options
    enable_pdl: bool = False,
    # dispatch options
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Quantize x to same-shape FP8 with an optional scalar scale.

    This API covers static activation-scale cases such as GPT-OSS MXFP4 MoE
    W4A8 input quantization and plain FP8 casts. If scale is provided, it is
    the actual quantization scale, so the backend computes x / scale before
    casting. If scale is omitted, the backend performs a pure FP8 cast.

    Args:
        x: Input tensor.
        scale: Optional scalar scale, as a Python value or scalar tensor. If
            provided, the backend computes x / scale before casting. If omitted,
            the backend performs a pure FP8 cast.
        enable_pdl: Whether to request Programmatic Dependent Launch support.
        override: Optional exact kernel name or solution override.
        solution: Optional registered solution to select.

    Returns:
        Quantized FP8 tensor with the same shape as x.

    Non-scalar scales belong in dynamic quantization APIs and should be
    on-device tensors.

    """

    traits = {
        "has_scale": scale is not None,
    }
    signature = format_signature(x=dense_tensor_format(x.dtype))
    kernel = select_kernel(
        "quantization",
        "fp8",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )
    shape_params = {
        "shape": tuple(x.shape),
        "has_scale": scale is not None,
    }
    ShapeCapture.get().record("quantization", "fp8", kernel.name, x.dtype, shape_params)
    with kernel_scope(
        "quantization", "fp8", x.dtype, kernel_name=kernel.name, **shape_params
    ):
        return kernel(
            x,
            scale=scale,
            enable_pdl=enable_pdl,
        )


def quantize_fp8_with_scale(
    x: torch.Tensor,
    # quantization options
    granularity: Literal["tensor", "token", "token_group"] = "tensor",
    group_size: int | None = None,
    scale_encoding: Literal["float32", "ue8m0", "packed_ue8m0"] = "float32",
    # kernel options
    enable_pdl: bool = False,
    # dispatch options
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize x to FP8 while dynamically computing scales.

    Use granularity="tensor" for one scale over the whole tensor,
    granularity="token" for one scale per row/token, and
    granularity="token_group" for one scale per row/token and contiguous group
    along the last dimension.

    Args:
        x: Input tensor.
        granularity: Scale granularity. Supported values are tensor, token, and
            token_group.
        group_size: Number of contiguous values per scale group along the last
            dimension. Required for token_group granularity.
        scale_encoding: Scale encoding for token_group granularity, such as
            float32, ue8m0, or packed_ue8m0.
        enable_pdl: Whether to request Programmatic Dependent Launch support.
        override: Optional exact kernel name or solution override.
        solution: Optional registered solution to select.

    Returns:
        Tuple of quantized FP8 tensor and scale tensor.

    The expected scale shapes are [1] for tensor granularity, [M, 1] for token
    granularity, and [M, ceil(K / group_size)] or a backend-specific layout for
    token_group granularity, where M = x.reshape(-1, x.shape[-1]).shape[0].
    Returned scales use float32 dtype for scale_encoding="float32" and a
    backend-specific encoded integer dtype for non-float encodings such as
    "ue8m0".
    """

    if granularity not in {"tensor", "token", "token_group"}:
        raise ValueError(f"unsupported FP8 dynamic granularity: {granularity!r}")
    if granularity == "token_group":
        if group_size is None or group_size <= 0:
            raise ValueError(
                f"token_group granularity requires positive group_size, got {group_size}"
            )
        granularity_trait = f"token_group_{group_size}"
    else:
        granularity_trait = granularity
    traits = {
        "granularity": granularity_trait,
        "scale_encoding": scale_encoding,
    }
    signature = format_signature(x=dense_tensor_format(x.dtype))
    kernel = select_kernel(
        "quantization",
        "fp8_with_scale",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )
    shape_params = {
        "shape": tuple(x.shape),
        "granularity": granularity_trait,
        "group_size": group_size,
        "scale_encoding": scale_encoding,
    }
    ShapeCapture.get().record(
        "quantization", "fp8_with_scale", kernel.name, x.dtype, shape_params
    )
    with kernel_scope(
        "quantization",
        "fp8_with_scale",
        x.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            x,
            granularity=granularity,
            group_size=group_size,
            scale_encoding=scale_encoding,
            enable_pdl=enable_pdl,
        )


def quantize_mxfp8(
    x: torch.Tensor,
    # kernel options
    enable_pdl: bool = False,
    # dispatch options
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize x to MXFP8 format.

    MXFP8 uses FP8 data plus encoded vector scales, commonly one scale per 32
    values along the last dimension.

    Args:
        x: Input tensor.
        enable_pdl: Whether to request Programmatic Dependent Launch support.
        override: Optional exact kernel name or solution override.
        solution: Optional registered solution to select.

    Returns:
        Tuple of quantized MXFP8 tensor and encoded scale tensor.

    """

    traits = {}
    signature = format_signature(x=dense_tensor_format(x.dtype))
    kernel = select_kernel(
        "quantization",
        "mxfp8",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )
    shape_params = {
        "shape": tuple(x.shape),
    }
    ShapeCapture.get().record(
        "quantization", "mxfp8", kernel.name, x.dtype, shape_params
    )
    with kernel_scope(
        "quantization", "mxfp8", x.dtype, kernel_name=kernel.name, **shape_params
    ):
        return kernel(
            x,
            enable_pdl=enable_pdl,
        )


def quantize_nvfp4(
    x: torch.Tensor,
    scale: float | torch.Tensor | None = None,
    # quantization options
    scale_layout: Literal["linear", "swizzled"] = "swizzled",
    # kernel options
    enable_pdl: bool = False,
    # dispatch options
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize x to packed NVFP4.

    NVFP4 uses packed E2M1x2 data with one E4M3 scale factor per 16 values.
    The quantized output is usually shaped [M, K/2].

    scale is the actual input scale. Backend adapters should handle any
    backend-specific inverse-scale convention internally.

    Args:
        x: Input tensor.
        scale: Optional scalar input scale.
        scale_layout: Scale-factor layout. "linear" returns unswizzled scales;
            "swizzled" requests the backend-specific layout used by FP4 GEMM.
        enable_pdl: Whether to request Programmatic Dependent Launch support.
        override: Optional exact kernel name or solution override.
        solution: Optional registered solution to select.

    Returns:
        Tuple of packed NVFP4 tensor and scale-factor tensor.

    """

    traits = {
        "scale_layout": scale_layout,
        "has_scale": scale is not None,
    }
    signature = format_signature(x=dense_tensor_format(x.dtype))
    kernel = select_kernel(
        "quantization",
        "nvfp4",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )
    shape_params = {
        "shape": tuple(x.shape),
        "scale_layout": scale_layout,
        "has_scale": scale is not None,
    }
    ShapeCapture.get().record(
        "quantization", "nvfp4", kernel.name, x.dtype, shape_params
    )
    with kernel_scope(
        "quantization", "nvfp4", x.dtype, kernel_name=kernel.name, **shape_params
    ):
        return kernel(
            x,
            scale=scale,
            scale_layout=scale_layout,
            enable_pdl=enable_pdl,
        )


def quantize_mxfp4(
    x: torch.Tensor,
    # quantization options
    global_scale: float | None = None,
    scale_size: int = 32,
    scale_layout: Literal["linear", "128x4", "8x4"] = "128x4",
    # kernel options
    enable_pdl: bool = False,
    # dispatch options
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize x to packed MXFP4.

    MXFP4 uses packed E2M1x2 data with one UE8M0 scale factor per scale_size
    values, usually scale_size=32. The quantized output is usually shaped
    [M, K/2].

    global_scale is optional because some backends compute the global scale
    internally from the input. If provided, it is the actual global scale.

    Args:
        x: Input tensor.
        global_scale: Optional scalar global scale.
        scale_size: Number of values per scale-factor vector.
        scale_layout: Scale-factor layout, such as linear, 128x4, or 8x4.
        enable_pdl: Whether to request Programmatic Dependent Launch support.
        override: Optional exact kernel name or solution override.
        solution: Optional registered solution to select.

    Returns:
        Tuple of packed MXFP4 tensor and scale-factor tensor.

    """

    if scale_size <= 0:
        raise ValueError(f"scale_size must be positive, got {scale_size}")

    traits = {
        "scale_size": scale_size,
        "scale_layout": scale_layout,
        "has_global_scale": global_scale is not None,
        "scale_encoding": "ue8m0",
    }
    signature = format_signature(x=dense_tensor_format(x.dtype))
    kernel = select_kernel(
        "quantization",
        "mxfp4",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )
    shape_params = {
        "shape": tuple(x.shape),
        "scale_size": scale_size,
        "scale_layout": scale_layout,
        "has_global_scale": global_scale is not None,
    }
    ShapeCapture.get().record(
        "quantization", "mxfp4", kernel.name, x.dtype, shape_params
    )
    with kernel_scope(
        "quantization", "mxfp4", x.dtype, kernel_name=kernel.name, **shape_params
    ):
        return kernel(
            x,
            global_scale=global_scale,
            scale_size=scale_size,
            scale_layout=scale_layout,
            enable_pdl=enable_pdl,
        )
