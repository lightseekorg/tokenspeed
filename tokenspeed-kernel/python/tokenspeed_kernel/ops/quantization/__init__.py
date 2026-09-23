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
    "quantize_mxfp8",
    "quantize_nvfp4",
    "quantize_mxfp4",
]


def _quantize_fp8_roundtrip(
    x: torch.Tensor,
    group_size: int,
    scale_encoding: Literal["ue8m0"],
    *,
    override: str | None,
    solution: str | None,
) -> torch.Tensor:
    """Simulate grouped FP8 quantization and return the dequantized tensor.

    Each contiguous group on the last dimension receives an independently
    computed scale.  The result has the same shape and dtype as ``x``.  This
    operation is intended for models whose published inference contract
    requires an explicit FP8 round trip before a higher-precision operation.

    Args:
        x: Input tensor with a contiguous last dimension.
        group_size: Number of consecutive values that share a scale.
        scale_encoding: Scale selection contract.  ``"ue8m0"`` chooses the
            next power-of-two scale needed for finite E4M3 values.
        override: Optional exact kernel name override.
        solution: Optional registered solution to select.

    Returns:
        The grouped FP8-quantized and dequantized tensor in ``x.dtype``.
    """

    if group_size <= 0 or x.shape[-1] % group_size != 0:
        raise ValueError(
            "FP8 quantize/dequantize requires the last dimension to be "
            f"divisible by a positive group_size; got shape={tuple(x.shape)}, "
            f"group_size={group_size}."
        )
    traits = {
        "dequantize": True,
        "group_size": group_size,
        "scale_encoding": scale_encoding,
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
        "group_size": group_size,
        "scale_encoding": scale_encoding,
    }
    ShapeCapture.get().record(
        "quantization",
        "fp8",
        kernel.name,
        x.dtype,
        shape_params,
    )
    with kernel_scope(
        "quantization",
        "fp8",
        x.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            x,
            group_size=group_size,
            scale_encoding=scale_encoding,
        )


def quantize_fp8(
    x: torch.Tensor,
    scale: float | torch.Tensor | None = None,
    granularity: Literal["token", "token_group", "block"] | None = None,
    group_size: int | None = None,
    block_size: tuple[int, int] | list[int] | None = None,
    scale_encoding: Literal["float32", "ue8m0", "packed_ue8m0"] = "float32",
    dequantize: bool = False,
    enable_pdl: bool = False,
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Quantize a tensor to FP8 and return its scale when one is used.

    With no granularity this performs a plain or static-scale cast. Dynamic
    granularities compute canonical scales from the input. With dequantize=True,
    the FP8 values are immediately reconstructed in the input dtype and the
    returned scale is None.
    """
    if dequantize:
        if scale is not None:
            raise ValueError("FP8 dequantization does not accept a static scale")
        if granularity != "token_group" or group_size is None:
            raise ValueError(
                "FP8 dequantization requires token_group granularity and group_size"
            )
        if block_size is not None or scale_encoding != "ue8m0":
            raise ValueError("FP8 dequantization requires UE8M0 token-group scales")
        return (
            _quantize_fp8_roundtrip(
                x,
                group_size=group_size,
                scale_encoding=scale_encoding,
                override=override,
                solution=solution,
            ),
            None,
        )

    if granularity is not None:
        if scale is not None:
            raise ValueError("dynamic FP8 quantization does not accept scale")
        return _quantize_fp8_dynamic(
            x,
            granularity=granularity,
            group_size=group_size,
            block_size=block_size,
            scale_encoding=scale_encoding,
            enable_pdl=enable_pdl,
            override=override,
            solution=solution or (None if override is not None else "triton"),
        )

    traits = {"dequantize": False, "has_scale": scale is not None}
    signature = format_signature(x=dense_tensor_format(x.dtype))
    kernel = select_kernel(
        "quantization",
        "fp8",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )
    shape_params = {"shape": tuple(x.shape), "has_scale": scale is not None}
    ShapeCapture.get().record("quantization", "fp8", kernel.name, x.dtype, shape_params)
    with kernel_scope(
        "quantization", "fp8", x.dtype, kernel_name=kernel.name, **shape_params
    ):
        values = kernel(x, scale=scale, enable_pdl=enable_pdl)
    if scale is None:
        return values, None
    if isinstance(scale, torch.Tensor):
        return values, scale
    return values, torch.tensor([scale], dtype=torch.float32, device=x.device)


def _quantize_fp8_dynamic(
    x: torch.Tensor,
    # quantization options
    granularity: Literal["token", "token_group", "block"] = "token",
    group_size: int | None = None,
    block_size: tuple[int, int] | list[int] | None = None,
    scale_encoding: Literal["float32", "ue8m0", "packed_ue8m0"] = "float32",
    # kernel options
    enable_pdl: bool = False,
    # dispatch options
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize x to FP8 while dynamically computing scales.

    Use granularity="token" for one scale per row/token,
    granularity="token_group" for one scale per row/token and contiguous group
    along the last dimension, and granularity="block" for one scale per 2-D
    block.

    Args:
        x: Input tensor.
        granularity: Scale granularity: token, token_group, or block.
        group_size: Number of contiguous values per scale group along the last
            dimension. Required for token_group granularity.
        block_size: Two-dimensional scale block. Required for block granularity.
        scale_encoding: Scale encoding for token_group granularity, such as
            float32, ue8m0, or packed_ue8m0.
        enable_pdl: Whether to request Programmatic Dependent Launch support.
        override: Optional exact kernel name or solution override.
        solution: Optional registered solution to select.

    Returns:
        Tuple of quantized FP8 tensor and scale tensor.

    The expected scale shapes are [M, 1] for token granularity,
    [M, ceil(K / group_size)] for token_group granularity, and
    [ceil(M / block_m), ceil(K / block_k)] for block granularity.
    Returned scales use float32 dtype for scale_encoding="float32" and a
    backend-specific encoded integer dtype for non-float encodings such as
    "ue8m0".
    """

    if granularity not in {"token", "token_group", "block"}:
        raise ValueError(f"unsupported FP8 dynamic granularity: {granularity!r}")
    if granularity == "token_group":
        if group_size is None or group_size <= 0:
            raise ValueError(
                f"token_group granularity requires positive group_size, got {group_size}"
            )
        granularity_trait = f"token_group_{group_size}"
    elif granularity == "block":
        if block_size is None or len(block_size) != 2 or min(block_size) <= 0:
            raise ValueError("block granularity requires a positive 2-D block_size")
        granularity_trait = f"block_{int(block_size[0])}_{int(block_size[1])}"
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
        "block_size": tuple(block_size) if block_size is not None else None,
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
        kernel_args = {
            "granularity": granularity,
            "group_size": group_size,
            "scale_encoding": scale_encoding,
            "enable_pdl": enable_pdl,
        }
        if block_size is not None:
            kernel_args["block_size"] = tuple(block_size)
        return kernel(x, **kernel_args)


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


# Backend registration (side-effect imports).
import tokenspeed_kernel.ops.quantization.flashinfer  # noqa: E402,F401
import tokenspeed_kernel.ops.quantization.triton  # noqa: E402,F401
import tokenspeed_kernel.ops.quantization.trtllm  # noqa: E402,F401
