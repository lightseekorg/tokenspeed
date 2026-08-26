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

from functools import wraps
from inspect import signature

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
    pdl_enabled,
)
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()

fp4_quantize = error_fn
flashinfer_quantize_mxfp8 = error_fn
flashinfer_quantize_nvfp4 = error_fn
mxfp8_quantize = error_fn
nvfp4_block_scale_interleave = error_fn
fp8_blockscale_quantize_runner_sm90 = error_fn


def _resolve_enable_pdl(enable_pdl: bool | None) -> bool:
    return pdl_enabled() if enable_pdl is None else enable_pdl


def _with_pdl_default(function):
    enable_pdl_index = tuple(signature(function).parameters).index("enable_pdl")

    @wraps(function)
    def wrapper(*args, **kwargs):
        if len(args) > enable_pdl_index:
            args = (
                *args[:enable_pdl_index],
                _resolve_enable_pdl(args[enable_pdl_index]),
                *args[enable_pdl_index + 1 :],
            )
        else:
            kwargs["enable_pdl"] = _resolve_enable_pdl(kwargs.get("enable_pdl"))
        return function(*args, **kwargs)

    return wrapper


if platform.is_nvidia:
    from flashinfer import mxfp8_quantize as _mxfp8_quantize

    mxfp8_quantize = _with_pdl_default(_mxfp8_quantize)

    if platform.is_hopper:
        from flashinfer.gemm.gemm_base import (
            get_fp8_blockscale_gemm_runner_sm90 as fp8_blockscale_quantize_runner_sm90,
        )

    # FlashInfer's MXFP8 quantizer supports SM100 and newer architectures.
    # Hopper uses the portable Triton implementation registered alongside
    # this kernel.
    @register_kernel(
        "quantization",
        "mxfp8",
        name="flashinfer_quantize_mxfp8",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
        traits={},
        priority=Priority.PERFORMANT,
    )
    def flashinfer_quantize_mxfp8(
        x: torch.Tensor,
        enable_pdl: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return mxfp8_quantize(x, False, enable_pdl=_resolve_enable_pdl(enable_pdl))


if platform.is_nvidia and platform.is_blackwell:
    from flashinfer import fp4_quantize as _fp4_quantize
    from flashinfer import (
        nvfp4_block_scale_interleave,
    )

    fp4_quantize = _with_pdl_default(_fp4_quantize)

    @register_kernel(
        "quantization",
        "nvfp4",
        name="flashinfer_quantize_nvfp4",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
        traits={
            "has_scale": frozenset({True}),
        },
        priority=Priority.PERFORMANT,
    )
    def flashinfer_quantize_nvfp4(
        x: torch.Tensor,
        scale: float | torch.Tensor | None = None,
        scale_layout: str = "swizzled",
        enable_pdl: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The public quantization API uses the actual scale; FlashInfer's FP4
        # helper expects the inverse scale used before packing.
        scale_inv = 1.0 / scale
        return fp4_quantize(
            x,
            global_scale=scale_inv,
            sf_vec_size=16,
            is_sf_swizzled_layout=scale_layout == "swizzled",
            enable_pdl=_resolve_enable_pdl(enable_pdl),
        )


__all__ = [
    "fp4_quantize",
    "mxfp8_quantize",
    "nvfp4_block_scale_interleave",
    "fp8_blockscale_quantize_runner_sm90",
]
