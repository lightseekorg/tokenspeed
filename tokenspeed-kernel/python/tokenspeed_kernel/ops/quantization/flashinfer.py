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

from functools import cache, wraps
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

    _fp4_quantize_allocating = _with_pdl_default(_fp4_quantize)
    _fp4_quantize_signature = signature(_fp4_quantize)

    @cache
    def _fp4_output_module(major: int, minor: int):
        from flashinfer.quantization.fp4_quantization import (
            gen_fp4_quantization_sm100_module,
            gen_fp4_quantization_sm103_module,
            gen_fp4_quantization_sm107_module,
        )

        generators = {
            (10, 0): gen_fp4_quantization_sm100_module,
            (10, 3): gen_fp4_quantization_sm103_module,
            (10, 7): gen_fp4_quantization_sm107_module,
        }
        if (major, minor) not in generators:
            raise ValueError(f"FP4 output buffers do not support SM{major}{minor}")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Initialize FP4 output buffers before graph capture")
        return generators[major, minor]().build_and_load()

    def fp4_quantize(
        *args,
        out: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize FP4, optionally writing into caller-owned output buffers.

        Args:
            args: Positional arguments accepted by FlashInfer fp4_quantize.
            out: Packed uint8 values [M, K/2] and uint8 scale storage
                [ceil(M/16)*16, K/16]. Supplied buffers must be contiguous
                on the input device. This form supports row-major CUDA
                NVFP4 with linear scales and K divisible by 64.
            kwargs: Keyword arguments accepted by FlashInfer fp4_quantize.
                Every argument is forwarded or applied to the CUDA binding.

        Returns:
            Packed values and valid scale rows [M, K/16]. These are views of
            out when supplied. Omission uses FlashInfer's allocated outputs.
        """
        if out is None:
            return _fp4_quantize_allocating(*args, **kwargs)
        from flashinfer.quantization.nvfp4_quantization_utils import (
            nvfp4_4over6_code,
            resolve_nvfp4_4over6,
        )

        bound = _fp4_quantize_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        params = bound.arguments
        x = params["input"]
        if (
            params["backend"] != "cuda"
            or params["sf_vec_size"] != 16
            or params["sf_use_ue8m0"]
            or params["is_sf_swizzled_layout"]
        ):
            raise ValueError("FP4 output buffers require CUDA NVFP4 with linear scales")
        if (
            x.ndim != 2
            or not x.is_cuda
            or not x.is_contiguous()
            or x.dtype not in (torch.float16, torch.bfloat16)
            or x.shape[1] % 64 != 0
        ):
            raise ValueError(
                "FP4 output buffers require contiguous CUDA FP16/BF16 [M, K], K % 64 == 0"
            )
        values, scales = out
        rows, cols = x.shape
        expected = ((rows, cols // 2), ((rows + 15) // 16 * 16, cols // 16))
        for tensor, shape in zip(out, expected, strict=True):
            if (
                tensor.shape != shape
                or tensor.dtype != torch.uint8
                or tensor.device != x.device
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 16 != 0
            ):
                raise ValueError(
                    f"FP4 output buffer must be aligned contiguous uint8 {shape} on {x.device}"
                )
        if rows == 0:
            return values, scales
        global_scale = params["global_scale"]
        if global_scale is not None:
            global_scale = global_scale.to(device=x.device, dtype=torch.float32)
        recipe = resolve_nvfp4_4over6(params["nvfp4_4over6"])
        major, minor = torch.cuda.get_device_capability(x.device)
        _fp4_output_module(major, minor).fp4_quantize(
            x,
            global_scale,
            values,
            scales.reshape(-1),
            params["sf_vec_size"],
            params["sf_use_ue8m0"],
            params["is_sf_swizzled_layout"],
            params["is_sf_8x4_layout"],
            params["is_global_scale_inversed"],
            _resolve_enable_pdl(params["enable_pdl"]),
            nvfp4_4over6_code(recipe),
        )
        return values, scales[:rows]

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
