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
from tokenspeed_kernel.selection import SelectedKernel, select_kernel
from tokenspeed_kernel.signature import (
    dense_format_signature,
    dense_tensor_format,
    format_signature,
)

__all__ = [
    "quantize_fp8",
    "quantize_fp8_with_gate",
    "quantize_fp8_with_scale",
    "rmsnorm_quantize_fp8",
    "silu_and_mul_quantize_fp8",
    "quantize_mxfp8",
    "quantize_nvfp4",
    "quantize_mxfp4",
    "rmsnorm_mxfp4_quant",
    "rmsnorm_nvfp4_quant",
    "rmsnorm_quantize_fp8_group32",
    "silu_and_mul_quantize_fp8_group32",
]


def rmsnorm_mxfp4_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    weight_offset: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse affine RMSNorm with CDNA4-native dynamic MXFP4 quantization.

    Args:
        x: BF16 rows shaped ``[tokens, width]``.
        weight: Learned BF16 RMSNorm weight shaped ``[width]``.
        eps: Epsilon added to the row variance.
        weight_offset: Center added to the checkpoint weight.

    Returns:
        Packed E2M1 rows and CDNA4-swizzled E8M0 scales.
    """
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_cdna4:
        raise RuntimeError("fused RMSNorm MXFP4 quantization requires CDNA4")
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.quantize_gluon import (
        rmsnorm_mxfp4_quantize,
    )

    return rmsnorm_mxfp4_quantize(x, weight, eps, weight_offset)


def rmsnorm_quantize_fp8_group32(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    override: str | None,
    solution: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize BF16 [M,K] rows and quantize them with exact group-32 scales.

    Quantization preserves the 1e-4 amax floor and the BF16 normalization
    boundary. Multiple GEMMs can share the codes and scales.

    Args:
        x: BF16 rows ``[M, K]`` with unit inner stride; K divisible by 32.
        weight: Contiguous BF16 RMSNorm weight ``[K]`` on x's device.
        eps: Positive epsilon added to the row mean square.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.

    Returns:
        BF16 normalized rows ``[M, K]``, E4M3 codes ``[M, K]`` and canonical
        uint8 UE8M0 scales ``[M, K/32]``.
    """
    kernel = select_kernel(
        "quantization",
        "rmsnorm_fp8_group32_ue8m0",
        format_signature(x=dense_tensor_format(x.dtype)),
        traits={},
        override=override,
        solution=solution,
    )
    return kernel(x, weight, eps)


def silu_and_mul_quantize_fp8_group32(
    x: torch.Tensor, limit: float | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """SwiGLU of BF16 gate/up rows, quantized with exact group-32 UE8M0 scales.

    The activation is ``silu(min(gate, limit)) * clamp(up, -limit, limit)``
    rounded to BF16 as ``silu_and_mul`` does, then quantized as
    ``quantize_fp8(granularity="token_group", group_size=32,
    scale_encoding="ue8m0")`` does, keeping the 1e-4 amax floor.

    Args:
        x: BF16 ``[M, 2 * I]`` rows (gate columns, then up columns) with a
            contiguous last axis; I divisible by 32.
        limit: Positive SwiGLU clamp, or None for no clamp.

    Returns:
        E4M3 codes ``[M, I]`` and uint8 UE8M0 scales ``[M, I / 32]``.
    """
    kernel = select_kernel(
        "quantization",
        "silu_and_mul_fp8_group32_ue8m0",
        format_signature(x=dense_tensor_format(x.dtype)),
        traits={},
        override=None,
        solution=None,
    )
    return kernel(x, limit)


def _quantize_fp8_roundtrip(
    x: torch.Tensor,
    group_size: int,
    scale_encoding: Literal["ue8m0"],
    *,
    override: str | None,
    solution: str | None,
    required_features: frozenset[str] | None = None,
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
        features=required_features,
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
    *,
    required_features: frozenset[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Quantize a tensor to FP8 and return its scale when one is used.

    With no granularity this performs a plain or static-scale cast. Dynamic
    granularities compute canonical scales from the input. With dequantize=True,
    the FP8 values are immediately reconstructed in the input dtype and the
    returned scale is None.

    Args:
        x: Input tensor; block granularity requires two dimensions.
        scale: Optional static scalar scale. Omission means a plain FP8 cast
            when granularity is None.
        granularity: Dynamic token, token-group, or two-dimensional block scaling.
        group_size: Values per token group; required for token_group.
        block_size: Positive block dimensions; required for block granularity.
        scale_encoding: Storage encoding of dynamic scales.
        dequantize: Reconstruct the input dtype after UE8M0 group quantization.
        enable_pdl: Request Programmatic Dependent Launch where supported.
            Round trips and block quantization reject this option.
        override: Optional exact registered kernel override. Native-layout
            quantizers may return strided scales; see their documented contract.
        solution: Optional restriction to a registered backend family.
        required_features: Optional arithmetic or API features the selected
            kernel must provide, including when an override is supplied.

    Returns:
        Quantized values and canonical scales, or reconstructed values and
        None for a round trip. Static tensor scales retain their identity;
        Python scalar scales are returned as a one-element FP32 tensor.
    """
    if dequantize:
        if enable_pdl:
            raise ValueError("FP8 dequantization does not support enable_pdl=True")
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
                required_features=required_features,
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
            solution=solution,
            required_features=required_features,
        )

    traits = {"dequantize": False, "has_scale": scale is not None}
    signature = format_signature(x=dense_tensor_format(x.dtype))
    kernel = select_kernel(
        "quantization",
        "fp8",
        signature,
        features=required_features,
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
    return values, torch.full((1,), scale, dtype=torch.float32, device=x.device)


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
    required_features: frozenset[str] | None = None,
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
        granularity_trait = "block"
    else:
        granularity_trait = granularity
    kernel = _select_fp8_dynamic(
        x.dtype,
        x.shape[0] if x.ndim == 2 else None,
        granularity_trait,
        scale_encoding,
        solution=solution,
        override=override,
        required_features=required_features,
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


def _select_fp8_dynamic(
    dtype: torch.dtype,
    rows: int | None,
    granularity_trait: str,
    scale_encoding: str,
    *,
    solution: str | None,
    override: str | None,
    required_features: frozenset[str] | None = None,
) -> SelectedKernel:
    traits = {
        "granularity": granularity_trait,
        "scale_encoding": scale_encoding,
        "scale_layout": "token_major",
    }
    if rows is not None:
        # Rows of a 2-D x, for kernels with a row bound (``m_max``). Kernels
        # without one ignore it; a kernel with one never gets other ranks.
        traits["m"] = rows
    return select_kernel(
        "quantization",
        "fp8_with_scale",
        format_signature(x=dense_tensor_format(dtype)),
        traits=traits,
        features=required_features,
        solution=solution,
        override=override,
    )


def select_fp8_token_quantizer(
    rows: int,
    dtype: torch.dtype,
    *,
    solution: str | None,
    override: str | None,
    required_features: frozenset[str] | None = None,
) -> SelectedKernel:
    """The kernel ``quantize_fp8(x, granularity="token")`` runs for a 2-D ``x``
    of ``rows`` rows of ``dtype``; nothing is launched.

    Args:
        rows: Row count of ``x``.
        dtype: ``x``'s dtype.
        solution: Restrict ranked selection to one solution, or ``None``.
        override: Exact kernel name to force, or ``None`` for ranked selection.
        required_features: Optional features required of the quantizer.

    Raises:
        NoKernelFoundError: No registered kernel serves the call.
    """
    return _select_fp8_dynamic(
        dtype,
        rows,
        "token",
        "float32",
        solution=solution,
        override=override,
        required_features=required_features,
    )


def quantize_fp8_with_scale(
    x: torch.Tensor,
    *,
    granularity: Literal["tensor", "token", "token_group"] = "token",
    group_size: int | None = None,
    solution: str | None = None,
    override: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` to FP8 and return dynamically computed float32 scales.

    Args:
        x: Input tensor.
        granularity: ``"tensor"`` computes a single whole-tensor scale returned
            with shape ``(1,)``; ``"token"`` returns one scale per row with
            shape ``[M, 1]``; ``"token_group"`` returns one scale per contiguous
            group of ``group_size`` values along the last dimension.
        group_size: Values per scale group; required for ``"token_group"``.
        solution: Optional restriction to a registered backend family.
        override: Optional exact registered kernel override.

    Returns:
        Quantized FP8 values and their float32 scales.
    """
    if granularity == "tensor":
        fp8_info = torch.finfo(torch.float8_e4m3fn)
        scale = (x.float().abs().amax() / fp8_info.max).reshape(1).to(torch.float32)
        values, _ = quantize_fp8(x, scale=scale, override=override)
        return values, scale
    return quantize_fp8(
        x,
        granularity=granularity,
        group_size=group_size,
        scale_encoding="float32",
        solution=solution,
        override=override,
    )


def rmsnorm_quantize_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    residual: torch.Tensor | None = None,
    round_residual_sum_bf16: bool,
    x_scale: float | None = None,
    residual_scale: float | None = None,
    # kernel options
    enable_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """RMSNorm that also quantizes its output to FP8 per token, in one kernel.

    The codes and scales are bit for bit those the Triton per-token quantizer
    (``quantize_fp8(granularity="token")`` up to 64 rows on SM100 and newer)
    writes for the stored output, for an FP8 GEMM that reads it. One Triton
    kernel implements it, so it is called directly, without kernel selection.

    Args:
        x: ``[..., hidden_size]`` BF16 or FP16 input.
        weight: ``[hidden_size]`` affine weight.
        eps: Epsilon added to the mean of squares.
        residual: Optional ``x``-shaped tensor added to ``x`` before the norm.
        round_residual_sum_bf16: Round the sum ``x + residual`` to BF16 before
            it is stored and normalized, as an eager BF16 add followed by the
            norm. Needs BF16 ``x`` and ``residual``.
        x_scale, residual_scale: Optional float multipliers of ``x`` and
            ``residual``, each product rounded to BF16 before the add. Need
            ``round_residual_sum_bf16=True``.
        enable_pdl: Launch with Programmatic Dependent Launch.

    Returns:
        ``(out, residual_sum, codes, scales)``: the ``x``-shaped output, the
        residual sum (None without ``residual``), and contiguous FP8 E4M3
        ``[rows, hidden_size]`` codes and FP32 ``[rows, 1]`` scales of ``out``.
    """
    return triton_rmsnorm_quantize_fp8(
        x,
        weight,
        eps,
        residual=residual,
        round_residual_sum_bf16=round_residual_sum_bf16,
        x_scale=x_scale,
        residual_scale=residual_scale,
        enable_pdl=enable_pdl,
    )


def silu_and_mul_quantize_fp8(
    x: torch.Tensor,
    token_scales: torch.Tensor,
    channel_scales: torch.Tensor,
    *,
    # kernel options
    enable_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SwiGLU of an FP8 GEMM output whose scales are not applied yet,
    quantized to FP8 per token, in one kernel.

    For an FP8 W8A8 gate/up projection run with unit scales: the kernel
    applies the per-token and per-channel scales the GEMM left out, then
    ``silu(gate) * up`` as ``silu_and_mul`` rounds it, then quantizes each
    row as the Triton per-token quantizer
    (``quantize_fp8(granularity="token")`` up to 64 rows on SM100 and newer)
    does, for the down projection. One Triton kernel implements it, so it is
    called directly, without kernel selection.

    Args:
        x: ``[M, 2 * I]`` BF16 or FP16 GEMM output (gate columns, then up
            columns) with a contiguous last axis, ``I`` at most 16384.
        token_scales: FP32 ``[M, 1]`` or ``[M]`` per-token scales of the
            GEMM's FP8 input.
        channel_scales: FP32 ``[2 * I, 1]`` or ``[2 * I]`` per-channel weight
            scales of the GEMM.
        enable_pdl: Launch with Programmatic Dependent Launch.

    Returns:
        ``(codes, scales)``: contiguous FP8 E4M3 ``[M, I]`` codes and FP32
        ``[M, 1]`` scales of the activation.

    Raises:
        ValueError: A layout or dtype outside the above, or ``I`` above 16384.
    """
    return triton_silu_and_mul_quantize_fp8(
        x, token_scales, channel_scales, enable_pdl=enable_pdl
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


def quantize_fp8_with_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    *,
    norm_weight: torch.Tensor | None,
    eps: float,
    enable_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply optional RMSNorm, sigmoid gating, and dynamic FP8 quantization.

    Args:
        x: BF16 rows [tokens, channels], with channels divisible by 128.
        gate: BF16 sigmoid logits with the same shape; rows may be strided.
        norm_weight: Per-head [head_dim] RMSNorm weight, or None for gating only.
            Its length must divide channels; FP8 groups remain 128 elements.
        eps: RMSNorm epsilon; unused when norm_weight is None.
        enable_pdl: Enable the producer/consumer programmatic-launch dependency.

    Returns:
        Contiguous E4M3 values and FP32 scales [tokens, channels / 128].
        Normalization and gating round to BF16 before the per-group quantizer.
    """
    kernel = select_kernel(
        "quantization",
        "gated_fp8",
        format_signature(
            x=dense_tensor_format(x.dtype), gate=dense_tensor_format(gate.dtype)
        ),
    )
    return kernel(x, gate, norm_weight=norm_weight, eps=eps, enable_pdl=enable_pdl)


def rmsnorm_nvfp4_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    weight_offset: float,
    global_scale: torch.Tensor,
    mailbox_sentinel: int | None,
    enable_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm a BF16 row and quantize it to NVFP4 in one launch.

    Reproduces ``affine_rmsnorm`` (with its BF16 output rounding) followed by
    ``fp4_quantize(..., is_sf_swizzled_layout=False)`` bit for bit, so the
    result feeds the TRT-LLM FP4 MoE kernels directly.

    With ``mailbox_sentinel`` the rows are this rank's Lamport mailbox, still
    being filled by the sharded projection's multicast stores: each program
    polls its row until no BF16 element still holds the sentinel's half, then
    re-arms the row, so no gather launch stands between the projection and
    the experts.

    Args:
        x: BF16 rows ``[rows, width]`` with ``width`` a multiple of 16; with
            a sentinel, the contiguous mailbox rows and a power-of-two width.
        weight: Learned affine norm weight ``[width]``.
        eps: Epsilon added to the variance.
        weight_offset: Center added to the checkpoint weight.
        global_scale: FP32 ``[1]`` NVFP4 global scale, the inverse of the
            activation's input scale.
        mailbox_sentinel: The 32-bit word ``x`` is armed with, whose 16-bit
            halves agree, or None for an ordinary tensor.
        enable_pdl: Enable NVIDIA programmatic dependent launch.

    Returns:
        ``(packed, scales)``: E2M1 pairs as ``uint8 [rows, width // 2]`` and
        E4M3 block scales as ``uint8 [rows, width // 16]`` in linear layout.
    """
    kernel = select_kernel(
        "quantization",
        "rmsnorm_nvfp4",
        dense_format_signature(("x",), (x.dtype,)),
    )
    return kernel(
        x,
        weight,
        eps,
        weight_offset,
        global_scale,
        mailbox_sentinel,
        enable_pdl,
    )


# Backend registration (side-effect imports).
import tokenspeed_kernel.ops.quantization.flashinfer  # noqa: E402,F401
import tokenspeed_kernel.ops.quantization.gluon  # noqa: E402,F401
import tokenspeed_kernel.ops.quantization.triton  # noqa: E402,F401
import tokenspeed_kernel.ops.quantization.trtllm  # noqa: E402,F401
from tokenspeed_kernel.ops.quantization.triton import (  # noqa: E402
    triton_rmsnorm_quantize_fp8,
    triton_silu_and_mul_quantize_fp8,
)
