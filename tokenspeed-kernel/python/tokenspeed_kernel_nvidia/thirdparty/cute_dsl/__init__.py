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

"""Third-party CuTe DSL kernels."""

from __future__ import annotations

# Keep package-level re-exports lazy: the implementation modules import
# CUDA/CuTe-only dependencies at module import time.
_EXPORTS = {
    "ArgmaxKernel": "tokenspeed_kernel_nvidia.thirdparty.cute_dsl.argmax",
    "CUDAGraphCompatibleWrapper": "tokenspeed_kernel_nvidia.thirdparty.cute_dsl.argmax",
    "Sm100BlockScaledPersistentDenseGemmKernel": (
        "tokenspeed_kernel_nvidia.thirdparty.cute_dsl." "nvfp4_gemm_swiglu_nvfp4_quant"
    ),
    "cvt_sf_M32x4xrm_K4xrk_L_to_MKL": (
        "tokenspeed_kernel_nvidia.thirdparty.cute_dsl." "nvfp4_gemm_swiglu_nvfp4_quant"
    ),
    "cvt_sf_MKL_to_M32x4xrm_K4xrk_L": (
        "tokenspeed_kernel_nvidia.thirdparty.cute_dsl." "nvfp4_gemm_swiglu_nvfp4_quant"
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
