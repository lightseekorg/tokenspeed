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

from importlib import import_module

from tokenspeed.runtime.layers.moe.core.registry import register_backend_family

_BACKEND_SPECS = {
    ("unquantized", "triton"): (
        "tokenspeed.runtime.layers.moe.backends.unquantized.triton",
        "Bf16TritonBackend",
    ),
    ("unquantized", "flashinfer_cutlass"): (
        "tokenspeed.runtime.layers.moe.backends.unquantized.flashinfer_cutlass",
        "Bf16FlashinferCutlassBackend",
    ),
    ("unquantized", "flashinfer_trtllm"): (
        "tokenspeed.runtime.layers.moe.backends.unquantized.flashinfer_trtllm",
        "Bf16FlashinferTrtllmBackend",
    ),
    ("fp8", "triton"): (
        "tokenspeed.runtime.layers.moe.backends.fp8.triton",
        "Fp8TritonBackend",
    ),
    ("fp8", "flashinfer_cutlass"): (
        "tokenspeed.runtime.layers.moe.backends.fp8.flashinfer_cutlass",
        "Fp8FlashinferCutlassBackend",
    ),
    ("w8a8_fp8", "triton"): (
        "tokenspeed.runtime.layers.moe.backends.w8a8_fp8.triton",
        "W8A8PerTokenPerChannelFp8TritonBackend",
    ),
    ("nvfp4", "flashinfer_cutlass"): (
        "tokenspeed.runtime.layers.moe.backends.nvfp4.flashinfer_cutlass",
        "Nvfp4FlashinferCutlassBackend",
    ),
    ("nvfp4", "flashinfer_cutedsl"): (
        "tokenspeed.runtime.layers.moe.backends.nvfp4.flashinfer_cutedsl",
        "Nvfp4FlashinferCuteDslBackend",
    ),
    ("nvfp4", "flashinfer_trtllm"): (
        "tokenspeed.runtime.layers.moe.backends.nvfp4.flashinfer_trtllm",
        "Nvfp4FlashinferTrtllmBackend",
    ),
    ("mxfp4", "flashinfer_mxfp4"): (
        "tokenspeed.runtime.layers.moe.backends.mxfp4.flashinfer",
        "Mxfp4FlashinferMxfp4Backend",
    ),
    ("mxfp4", "triton_kernel_fp8"): (
        "tokenspeed.runtime.layers.moe.backends.mxfp4.triton_kernel_fp8",
        "Mxfp4Fp8TritonKernelBackend",
    ),
    ("mxfp4", "triton_kernel"): (
        "tokenspeed.runtime.layers.moe.backends.mxfp4.triton_kernel",
        "Mxfp4TritonKernelBackend",
    ),
    ("wna16", "marlin"): (
        "tokenspeed.runtime.layers.moe.backends.wna16.marlin",
        "Wna16MarlinBackend",
    ),
}
_REGISTERED = set()


def ensure_backend_family_registered(quant: str, impl: str) -> None:
    key = (quant, impl)
    if key in _REGISTERED:
        return

    module_name, class_name = _BACKEND_SPECS[key]
    backend_cls = getattr(import_module(module_name), class_name)
    register_backend_family(quant=quant, impl=impl, cls=backend_cls)
    _REGISTERED.add(key)
