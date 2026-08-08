// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Marlin WNA16 MoE GEMM (vendored), pre-compiled with the specializations
// tokenspeed needs: bf16 activations, no expert bias.
// MXFP4 (E8M0 group-32 scales) is only numerically valid on the bf16
// activation path, so no fp16 instantiation is exported.

#include <tvm/ffi/function.h>

#include "moe_align_kernel.cu"
#include "moe_wna16_marlin.cuh"

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    moe_wna16_marlin_gemm_bf16,
    (moe_wna16_marlin_gemm<bf16_t, false, false>));

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    moe_wna16_marlin_gemm_bf16_ep,
    (moe_wna16_marlin_gemm<bf16_t, true, false>));

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    moe_align_block_size,
    (MoeAlignBlockSizeKernel<int32_t>::run));
