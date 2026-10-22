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

    _fp4_quantize = _with_pdl_default(_fp4_quantize)

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
        input: torch.Tensor,
        global_scale: torch.Tensor | None = None,
        *args,
        out: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward FP4 quantization, or write linear NVFP4 into ``out``.

        Without ``out``, all arguments follow FlashInfer's API. With ``out``,
        pass FP16/BF16 ``input`` [M, K], ``global_scale``,
        ``is_sf_swizzled_layout=False``, and optional ``enable_pdl`` only.
        Output buffers are contiguous uint8 [M, K/2] and [ceil(M/16)*16, K/16].
        Return the packed values and the first M scale rows.
        """
        if out is None:
            return _fp4_quantize(input, global_scale, *args, **kwargs)
        if args or kwargs.keys() - {"is_sf_swizzled_layout", "enable_pdl"}:
            raise ValueError(
                "FP4 output buffers accept only linear layout and PDL options"
            )
        if kwargs.get("is_sf_swizzled_layout", True):
            raise ValueError("FP4 output buffers require linear scales")
        from flashinfer.quantization.nvfp4_quantization_utils import (
            nvfp4_4over6_code,
            resolve_nvfp4_4over6,
        )

        x = input
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
        if global_scale is not None:
            global_scale = global_scale.to(device=x.device, dtype=torch.float32)
        major, minor = torch.cuda.get_device_capability(x.device)
        _fp4_output_module(major, minor).fp4_quantize(
            x,
            global_scale,
            values,
            scales.reshape(-1),
            16,
            False,
            False,
            False,
            False,
            _resolve_enable_pdl(kwargs.get("enable_pdl")),
            nvfp4_4over6_code(resolve_nvfp4_4over6()),
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
