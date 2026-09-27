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

"""TRT-LLM blockwise FP8 adapter. All scales retain their original FP32 values."""

import threading

import torch

_compiled = {}
_compile_lock = threading.Lock()


def _arguments(a, b, sa, sb, out):
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


def _compile(a, b, sa, sb, out, tile_m):
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.runtime import make_fake_stream
    from tokenspeed_kernel.thirdparty.trtllm_blockwise._kernel import (
        Sm100BlockwiseGemmKernel,
    )

    two = tile_m == 256
    cluster = (2, 1) if two else (1, 1)
    kernel = Sm100BlockwiseGemmKernel(cutlass.Float32, two, (tile_m, 128), cluster)
    active = utils.HardwareInfo().get_max_active_clusters(cluster[0] * cluster[1])
    return cute.compile(
        kernel.wrapper,
        *_arguments(a, b, sa, sb, out),
        active,
        make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--opt-level 2 --enable-tvm-ffi",
    )


def prepare(device):
    """Compile reusable dynamic-shape variants on device before graph capture."""
    with torch.cuda.device(device), _compile_lock:
        if (device.index, 64) in _compiled:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare TRT-LLM CuTe-DSL before CUDA graph capture")
        a = torch.empty((256, 128), device=device, dtype=torch.float8_e4m3fn)
        b = torch.empty((128, 128), device=device, dtype=torch.float8_e4m3fn)
        sa = torch.empty((1, 256), device=device, dtype=torch.float32)
        sb = torch.empty((1, 1), device=device, dtype=torch.float32)
        out = torch.empty((256, 128), device=device, dtype=torch.bfloat16)
        compiled = {
            (device.index, tile): _compile(a, b, sa, sb, out, tile)
            for tile in (64, 128, 256)
        }
        _compiled.update(compiled)


def blockwise_gemm(a, b, a_scales, b_scales, out):
    """Run FP8 [M,K] @ [N,K].T with 1x128/128x128 FP32 scales into BF16 out."""
    # A scales are logically [M,K/128], but this kernel reads MN-major.
    sa = a_scales.T.contiguous()
    sb = b_scales.contiguous()
    m = a.shape[0]
    if m == 0:
        return out
    tile = 64 if m <= 64 else (128 if m <= 128 else 256)
    key = (a.device.index, tile)
    if key not in _compiled:
        prepare(a.device)
    from cuda.bindings.driver import CUstream

    # Query the stream at every launch, including capture on an auxiliary
    # stream. Never bind the stream or pointers used during compilation.
    _compiled[key](
        *_arguments(a, b, sa, sb, out),
        CUstream(torch.cuda.current_stream(a.device).cuda_stream),
    )
    return out
