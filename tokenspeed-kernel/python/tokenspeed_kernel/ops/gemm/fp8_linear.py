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

"""Prepared block-FP8 linear contracts."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable

import torch
from tokenspeed_kernel.ops.gemm.flashinfer import (
    has_flashinfer_fp8_blockscale,
    has_flashinfer_mxfp8,
    prepare_flashinfer_fp8_blockscale_weight_scales,
)
from tokenspeed_kernel.ops.gemm.fp8_utils import swizzle_mxfp8_scale
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import KernelRegistry

try:
    from tokenspeed_kernel.thirdparty.deep_gemm import (
        ceil_to_ue8m0,
        transform_sf_into_required_layout,
    )
except ImportError:
    ceil_to_ue8m0 = None  # type: ignore[assignment]
    transform_sf_into_required_layout = None  # type: ignore[assignment]


class _PreparedFp8Linear(torch.nn.Module):
    def __init__(
        self,
        *,
        override: str | None,
        block_size: tuple[int, int],
        prepared_weight_scales: torch.Tensor | None = None,
        prepacked_scales: bool = False,
        activation: str | None = None,
        warmup: Callable | None = None,
        warmup_key: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.override = override
        self.block_size = block_size
        self.prepacked_scales = prepacked_scales
        self.activation = activation
        self.warmup = warmup
        self.warmup_key = warmup_key
        self.register_buffer(
            "prepared_weight_scales", prepared_weight_scales, persistent=False
        )


def _warmup_deep_gemm_fp8_linears(
    plans: list[_PreparedFp8Linear], max_tokens: int
) -> None:
    from tokenspeed_kernel.thirdparty.deep_gemm.warmup import warmup_fp8_gemm_nt

    by_device: dict[torch.device, set[tuple[int, int]]] = {}
    for plan in plans:
        assert plan.warmup_key is not None
        assert plan.prepared_weight_scales is not None
        n, k = plan.warmup_key
        device = plan.prepared_weight_scales.device
        by_device.setdefault(device, set()).add((n, k))
    for device, shapes in by_device.items():
        warmup_fp8_gemm_nt(list(shapes), max_tokens, device)


def prepare_fp8_linear(
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    block_size: tuple[int, int] | list[int],
    scale_format: str | None = None,
) -> object:
    """Prepare an opaque block-FP8 linear implementation contract.

    Backend selection, persistent scale layout conversion, fused-activation
    support, and warmup behavior are owned by the returned plan. Callers must
    retain the plan without inspecting it and pass it to the related execution,
    activation, and warmup APIs.

    Args:
        weight: FP8 weight in ``[N, K]`` layout.
        weight_scales: Canonical block scales loaded with the weight.
        block_size: Logical scale block shape ``[block_n, block_k]``.
        scale_format: Logical checkpoint scale encoding, such as ``"ue8m0"``.

    Returns:
        An opaque prepared FP8 linear plan.
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must have shape [N, K], got {tuple(weight.shape)}")
    if len(block_size) != 2 or min(block_size) <= 0:
        raise ValueError("block_size must contain two positive dimensions")

    block_n, block_k = int(block_size[0]), int(block_size[1])
    n, k = weight.shape
    platform = current_platform()
    deep_gemm_spec = KernelRegistry.get().get_by_name("deep_gemm_mm_fp8_blockscale")
    scale_requires_transform = (
        scale_format == "ue8m0" and weight_scales.dtype.is_floating_point
    )
    if (
        ceil_to_ue8m0 is not None
        and transform_sf_into_required_layout is not None
        and deep_gemm_spec is not None
        and scale_requires_transform
        and platform.is_nvidia
        and n % 64 == 0
        and k % 128 == 0
    ):
        prepared_scales = transform_sf_into_required_layout(
            sf=ceil_to_ue8m0(weight_scales),
            mn=n,
            k=k,
            recipe=(1, block_n, block_k),
            is_sfa=False,
        )
        supports_fused_activation = (
            platform.is_blackwell_plus
            and os.environ.get("TOKENSPEED_DISABLE_DEEP_GEMM_UE8M0") != "1"
        )
        return _PreparedFp8Linear(
            override="deep_gemm_mm_fp8_blockscale",
            block_size=(block_n, block_k),
            prepared_weight_scales=prepared_scales,
            activation=("swiglu" if supports_fused_activation else None),
            warmup=(
                _warmup_deep_gemm_fp8_linears if supports_fused_activation else None
            ),
            warmup_key=(n, k) if supports_fused_activation else None,
        )

    if (
        has_flashinfer_mxfp8()
        and (block_n, block_k) == (1, 32)
        and weight_scales.dtype == torch.uint8
        and weight_scales.ndim == 2
        and n >= 128
        and k >= 128
        and k % 32 == 0
    ):
        return _PreparedFp8Linear(
            override="flashinfer_mm_mxfp8",
            block_size=(block_n, block_k),
            prepared_weight_scales=swizzle_mxfp8_scale(weight_scales, n, k),
        )

    if (
        has_flashinfer_fp8_blockscale()
        and (block_n, block_k) == (128, 128)
        and weight_scales.dtype == torch.float32
        and weight_scales.ndim == 2
        and n % 128 == 0
        and k % 128 == 0
    ):
        return _PreparedFp8Linear(
            override="flashinfer_mm_fp8_blockscale",
            block_size=(block_n, block_k),
            prepared_weight_scales=prepare_flashinfer_fp8_blockscale_weight_scales(
                weight_scales
            ),
            prepacked_scales=True,
        )

    return _PreparedFp8Linear(
        override=None,
        block_size=(block_n, block_k),
    )


def _require_fp8_linear_plan(plan: object) -> _PreparedFp8Linear:
    if not isinstance(plan, _PreparedFp8Linear):
        raise TypeError("plan must be returned by prepare_fp8_linear")
    return plan


def fp8_linear(
    plan: object,
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    *,
    input_scales: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Execute a block-FP8 linear operation through a prepared plan.

    Args:
        plan: Opaque plan returned by :func:`prepare_fp8_linear`.
        x: Input matrix ``[M, K]``. It may be floating point for online
            quantization or FP8 when ``input_scales`` is supplied.
        weight: FP8 weight matrix ``[N, K]``.
        weight_scales: Canonical persistent weight block scales.
        input_scales: Optional pre-quantized activation block scales.
        bias: Optional output bias.
        out_dtype: Requested output dtype.
        enable_pdl: Request Programmatic Dependent Launch when supported.

    Returns:
        The linear output matrix ``[M, N]``.
    """
    typed_plan = _require_fp8_linear_plan(plan)
    override = typed_plan.override
    prepacked_scales = typed_plan.prepacked_scales and input_scales is None
    if typed_plan.prepacked_scales and not prepacked_scales:
        override = None
    selected_weight_scales = (
        typed_plan.prepared_weight_scales
        if typed_plan.prepared_weight_scales is not None and override is not None
        else weight_scales
    )

    from tokenspeed_kernel.ops.gemm import mm

    return mm(
        x,
        weight,
        A_scales=input_scales,
        B_scales=selected_weight_scales,
        bias=bias,
        out_dtype=out_dtype,
        quant="mxfp8",
        block_size=list(typed_plan.block_size),
        override=override,
        enable_pdl=enable_pdl,
        prepacked_scales=prepacked_scales,
    )


def _fp8_linear_activation(
    plan: object,
    x: torch.Tensor,
    *,
    activation: str,
    limit: float | None,
    alpha: float,
    beta: float,
    enable_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    typed_plan = _require_fp8_linear_plan(plan)
    if typed_plan.activation != activation or x.ndim != 2:
        return None
    if x.shape[-1] % 2 != 0 or x.shape[-1] // 2 % typed_plan.block_size[1] != 0:
        return None

    from tokenspeed_kernel.ops.activation.triton import fused_swiglu_fp8_ue8m0

    return fused_swiglu_fp8_ue8m0(
        x,
        swiglu_limit=limit or 0.0,
        swiglu_alpha=alpha,
        swiglu_beta=beta,
        enable_pdl=enable_pdl,
    )


def warmup_prepared_fp8_linears(plans: Iterable[object], max_tokens: int) -> None:
    """Warm backend implementations selected by prepared FP8 linear plans.

    Args:
        plans: Opaque plans returned by :func:`prepare_fp8_linear`.
        max_tokens: Largest token count to include in backend warmup sweeps.

    Returns:
        None.
    """
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    grouped: dict[Callable, list[_PreparedFp8Linear]] = {}
    seen: set[tuple[Callable, torch.device, int, int]] = set()
    for plan in plans:
        typed_plan = _require_fp8_linear_plan(plan)
        if typed_plan.warmup is None or typed_plan.warmup_key is None:
            continue
        assert typed_plan.prepared_weight_scales is not None
        n, k = typed_plan.warmup_key
        key = (typed_plan.warmup, typed_plan.prepared_weight_scales.device, n, k)
        if key in seen:
            continue
        seen.add(key)
        grouped.setdefault(typed_plan.warmup, []).append(typed_plan)
    for warmup, prepared_plans in grouped.items():
        warmup(prepared_plans, max_tokens)


__all__ = [
    "fp8_linear",
    "prepare_fp8_linear",
    "warmup_prepared_fp8_linears",
]
