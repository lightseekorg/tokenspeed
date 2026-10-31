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

"""Compile and launch TRT-LLM blockwise FP8 GEMM with unchanged FP32 scales."""

import ctypes
import threading

import torch

_compiled = {}
_compile_lock = threading.Lock()


def _compile_arguments(a, b, sa, sb, out):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack, make_ptr

    m, k = a.shape
    n = b.shape[0]
    pointers = tuple(
        make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        for dtype, tensor in (
            (cutlass.Float8E4M3FN, a),
            (cutlass.Float8E4M3FN, b),
            (cutlass.Float32, sa),
            (cutlass.Float32, sb),
        )
    )
    c = from_dlpack(out.view(1, m, n).permute(1, 2, 0), enable_tvm_ffi=True)
    c = c.mark_layout_dynamic(leading_dim=1)
    return (m, n, k, m, n // 128, k // 128, 1, *pointers, c)


def _arguments(a, b, sa, sb, out):
    """Pass live storage directly through TVM-FFI after the types are compiled."""
    m, k = a.shape
    n = b.shape[0]
    addresses = (a.data_ptr(), b.data_ptr(), sa.data_ptr(), sb.data_ptr())
    if any(address % 16 for address in addresses) or out.data_ptr() % 16:
        raise ValueError("TRT-LLM CuTe-DSL tensor pointers must be 16-byte aligned")
    # Compile-time CuTe descriptors establish dtype/layout/alignment. At runtime
    # TVM-FFI accepts opaque pointers and a Torch tensor without rebuilding them.
    return (
        m,
        n,
        k,
        m,
        n // 128,
        k // 128,
        1,
        *(ctypes.c_void_p(address) for address in addresses),
        out.as_strided((m, n, 1), (n, 1, m * n)),
    )


def _compile(a, b, sa, sb, out, tactic):
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.runtime import make_fake_stream
    from tokenspeed_kernel.thirdparty.trtllm_blockwise._kernel import (
        Sm100BlockwiseGemmKernel,
    )

    two, tile, cluster = tactic
    kernel = Sm100BlockwiseGemmKernel(cutlass.Float32, two, tile, cluster)
    active = utils.HardwareInfo().get_max_active_clusters(cluster[0] * cluster[1])
    return cute.compile(
        kernel.wrapper,
        *_compile_arguments(a, b, sa, sb, out),
        active,
        make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--opt-level 2 --enable-tvm-ffi",
    )


def _run(a, b, a_scales, b_scales, out, tactic):
    """Launch a chosen tactic, or the existing row-count rule for tactic -1."""
    # A scales are logically [M,K/128], but this kernel reads MN-major.
    sa = a_scales.T.contiguous()
    sb = b_scales.contiguous()
    m = a.shape[0]
    if m == 0:
        return out
    if tactic == -1:
        tile = 64 if m <= 64 else (128 if m <= 128 else 256)
        tactic = (tile == 256, (tile, 128), (2 if tile == 256 else 1, 1))
    key = (a.device.index, tactic)
    if key not in _compiled:
        with torch.cuda.device(a.device), _compile_lock:
            if key not in _compiled:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Warm up TRT-LLM CuTe-DSL before CUDA graph capture"
                    )
                _compiled[key] = _compile(a, b, sa, sb, out, tactic)
    from cuda.bindings.driver import CUstream

    # Query the stream at every launch, including capture on an auxiliary
    # stream. Read the raw handle without constructing a temporary Torch Stream.
    # Never bind the stream or pointers used during compilation.
    _compiled[key](
        *_arguments(a, b, sa, sb, out),
        CUstream(torch._C._cuda_getCurrentRawStream(a.device.index)),
    )
    return out
