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

# Keyed by (device index, tile, reserved SM clusters).
_compiled = {}
_compile_lock = threading.Lock()


def _key(device, tile):
    from tokenspeed_kernel.sm_reservation import reserved_sm_clusters

    return device.index, tile, reserved_sm_clusters(device)


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
    from cutlass.runtime import make_fake_stream
    from tokenspeed_kernel.sm_reservation import max_active_clusters
    from tokenspeed_kernel.thirdparty.trtllm_blockwise._kernel import (
        Sm100BlockwiseGemmKernel,
    )

    two = tile_m == 256
    cluster = (2, 1) if two else (1, 1)
    kernel = Sm100BlockwiseGemmKernel(cutlass.Float32, two, (tile_m, 128), cluster)
    active = max_active_clusters(a.device, cluster[0] * cluster[1])
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
        if _key(device, 64) in _compiled:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare TRT-LLM CuTe-DSL before CUDA graph capture")
        a = torch.empty((256, 128), device=device, dtype=torch.float8_e4m3fn)
        b = torch.empty((128, 128), device=device, dtype=torch.float8_e4m3fn)
        sa = torch.empty((1, 256), device=device, dtype=torch.float32)
        sb = torch.empty((1, 1), device=device, dtype=torch.float32)
        out = torch.empty((256, 128), device=device, dtype=torch.bfloat16)
        compiled = {
            _key(device, tile): _compile(a, b, sa, sb, out, tile)
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
    key = _key(a.device, tile)
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


# Per-head variant, keyed like _compiled.
_compiled_heads = {}


def _head_arguments(a, b, sa, sb, out, row_offset, head_rows):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    m, k = a.shape

    def pointer(dtype, tensor, byte_offset=0):
        return make_ptr(
            dtype,
            tensor.data_ptr() + byte_offset,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )

    return (
        m,
        out.shape[1],
        k,
        m,
        k // 128,
        pointer(cutlass.Float8E4M3FN, a),
        pointer(cutlass.Float8E4M3FN, b, row_offset * k),
        pointer(cutlass.Float32, sa),
        pointer(cutlass.Float32, sb, (row_offset // 128) * (k // 128) * 4),
        pointer(cutlass.Float8E4M3FN, out),
        head_rows,
        out.stride(0),
        out.stride(1),
    )


def _compile_heads(a, b, sa, sb, out, tile_m):
    import cutlass
    import cutlass.cute as cute
    from cutlass.runtime import make_fake_stream
    from tokenspeed_kernel.sm_reservation import max_active_clusters
    from tokenspeed_kernel.thirdparty.trtllm_blockwise._kernel import (
        Sm100BlockwiseGemmKernel,
    )

    two = tile_m == 256
    cluster = (2, 1) if two else (1, 1)
    kernel = Sm100BlockwiseGemmKernel(cutlass.Float32, two, (tile_m, 128), cluster)
    return cute.compile(
        kernel.wrapper_heads,
        *_head_arguments(a, b, sa, sb, out, 0, b.shape[0]),
        max_active_clusters(a.device, cluster[0] * cluster[1]),
        make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--opt-level 2",
    )


def prepare_heads(device):
    """Compile the per-head variants on device before serving or graph capture."""
    with torch.cuda.device(device), _compile_lock:
        if _key(device, 64) in _compiled_heads:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare TRT-LLM CuTe-DSL before CUDA graph capture")
        a = torch.empty((256, 128), device=device, dtype=torch.float8_e4m3fn)
        b = torch.empty((128, 128), device=device, dtype=torch.float8_e4m3fn)
        sa = torch.empty((1, 256), device=device, dtype=torch.float32)
        sb = torch.empty((1, 1), device=device, dtype=torch.float32)
        out = torch.empty((256, 1, 128), device=device, dtype=torch.float8_e4m3fn)
        compiled = {
            _key(device, tile): _compile_heads(a, b, sa, sb, out, tile)
            for tile in (64, 128, 256)
        }
        _compiled_heads.update(compiled)


def blockwise_gemm_heads(a, b, a_scales, b_scales, out, *, row_offset, head_rows):
    """Block-FP8 GEMM per head into an FP8 destination with free row and head strides.

    For every head h, out[:, h, :] = a @ b[h * head_rows + row_offset :][:128].T,
    rounded to BF16 and then cast to E4M3, so it equals the BF16 GEMM output cast
    to FP8.

    Args:
        a: Contiguous E4M3 activations [M, K].
        b: Contiguous E4M3 weights [N, K], K a multiple of 128.
        a_scales: FP32 activation scales [M, K/128].
        b_scales: Contiguous FP32 weight scales [N/128, K/128].
        out: E4M3 destination [M, H, 128] with unit inner stride and 16-byte
            aligned row and head strides.
        row_offset: First weight row of head 0, a multiple of 128.
        head_rows: Weight rows between consecutive heads, a multiple of 128.
    Returns:
        out.
    """
    m, k = a.shape
    heads = out.shape[1]
    if (
        a.dtype != torch.float8_e4m3fn
        or b.dtype != torch.float8_e4m3fn
        or out.dtype != torch.float8_e4m3fn
        or out.shape != (m, heads, 128)
        or out.stride(2) != 1
        or out.stride(0) % 16
        or out.stride(1) % 16
        or k % 128
        or b.shape[1] != k
        or row_offset % 128
        or head_rows % 128
        or row_offset + (heads - 1) * head_rows + 128 > b.shape[0]
        or not a.is_contiguous()
        or not b.is_contiguous()
        or not b_scales.is_contiguous()
        or a_scales.shape != (m, k // 128)
        or b_scales.shape != (b.shape[0] // 128, k // 128)
    ):
        raise ValueError("Invalid per-head TRT-LLM CuTe-DSL GEMM arguments")
    if m == 0:
        return out
    sa = a_scales.T.contiguous()
    tile = 64 if m <= 64 else (128 if m <= 128 else 256)
    key = _key(a.device, tile)
    if key not in _compiled_heads:
        prepare_heads(a.device)
    from cuda.bindings.driver import CUstream

    _compiled_heads[key](
        *_head_arguments(a, b, sa, b_scales, out, row_offset, head_rows),
        CUstream(torch.cuda.current_stream(a.device).cuda_stream),
    )
    return out
