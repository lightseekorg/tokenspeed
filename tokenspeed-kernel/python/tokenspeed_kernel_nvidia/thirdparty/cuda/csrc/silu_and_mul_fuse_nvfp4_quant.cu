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

/*
 * Dense silu_and_mul + NVFP4 block-scale quantize (128x4 swizzled scales),
 * with PDL.
 *
 * Fork of flashinfer's ``cvt_fp16_to_fp4_expert`` (which is MoE-shaped with
 * an n_experts / mask interface) — stripped to the dense single-batch case
 * and wired for Programmatic Dependent Launch so the upstream FC1 GEMM and
 * downstream FC2 GEMM can overlap across this fused quantize.
 *
 * Per token: read 2*I bf16/fp16 values, compute gate*silu(up) = y, quantize
 * y into [I/2] NVFP4 bytes + [I/16] per-block fp8_e4m3 scales written in
 * 128x4 swizzled layout. SM100+ required (PTX cvt.rn.satfinite.e2m1x2.f32).
 */

#include <cuda_runtime.h>

#include "tensorrt_llm/kernels/quantization_utils.cuh"
#include "tvm_ffi_utils.h"

using tensorrt_llm::kernels::cvt_quant_to_fp4_get_sf_out_offset;
using tensorrt_llm::kernels::cvt_warp_fp16_to_fp4;
using tensorrt_llm::kernels::PackedVec;
using tensorrt_llm::kernels::silu_and_mul;

namespace tokenspeed {

constexpr int CVT_ELTS_PER_THREAD = 8;
constexpr int CVT_FP4_SF_VEC_SIZE = 16;
constexpr int CVT_FP4_NUM_THREADS_PER_SF =
    CVT_FP4_SF_VEC_SIZE / CVT_ELTS_PER_THREAD;  // = 2

template <typename Type>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(1024, 2) silu_and_mul_fuse_nvfp4_quant_kernel(
#else
silu_and_mul_fuse_nvfp4_quant_kernel(
#endif
    int32_t numCols, Type const* in, float const* SFScale, uint32_t* out,
    uint32_t* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using PackedVecT = PackedVec<Type, CVT_ELTS_PER_THREAD>;
  static_assert(sizeof(PackedVecT) == sizeof(Type) * CVT_ELTS_PER_THREAD,
                "Vec size is not matched.");

  asm volatile("griddepcontrol.wait;");

  // Each block owns one row (token). Within a row, threadIdx.x tiles the
  // column dim. This keeps threads that share a scale (2 consecutive
  // threads per SF_VEC_SIZE=16 elements) within the same warp and
  // contiguous — required for the __shfl_xor_sync cross-thread amax.
  int const rowIdx = blockIdx.x;
  int const colsPerRow = numCols / CVT_ELTS_PER_THREAD;
  int const actualColsPerRow = colsPerRow * 2;  // input is concatenated gate|up
  float const SFScaleVal = (SFScale == nullptr) ? 1.0f : SFScale[0];

  PackedVecT const* inPacked = reinterpret_cast<PackedVecT const*>(in);

  for (int colIdx = threadIdx.x; colIdx < colsPerRow; colIdx += blockDim.x) {
    int64_t const inOffset =
        static_cast<int64_t>(rowIdx) * actualColsPerRow + colIdx;
    PackedVecT gate_vec = inPacked[inOffset];
    PackedVecT up_vec = inPacked[inOffset + colsPerRow];
    silu_and_mul<Type, CVT_ELTS_PER_THREAD>(gate_vec, up_vec);

    int64_t const outOffset =
        static_cast<int64_t>(rowIdx) * colsPerRow + colIdx;
    auto sf_out =
        cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_SF_VEC_SIZE,
                                           CVT_FP4_NUM_THREADS_PER_SF>(
            rowIdx, colIdx, numCols, SFout);
    out[outOffset] =
        cvt_warp_fp16_to_fp4<Type, CVT_FP4_SF_VEC_SIZE, CVT_ELTS_PER_THREAD,
                             /*UE8M0_SF=*/false>(gate_vec, SFScaleVal, sf_out);
  }

  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

}  // namespace tokenspeed

void silu_and_mul_fuse_nvfp4_quant(TensorView out, TensorView scale_out,
                                    TensorView input, TensorView global_scale,
                                    bool enable_pdl) {
  TVM_FFI_ICHECK_EQ(input.ndim(), 2) << "input must be 2-D [M, 2*I]";
  int const numRows = int(input.size(0));
  int const twoI = int(input.size(1));
  TVM_FFI_ICHECK_EQ(twoI % 2, 0) << "input last dim must be even";
  int const numCols = twoI / 2;
  TVM_FFI_ICHECK_EQ(numCols % tokenspeed::CVT_FP4_SF_VEC_SIZE, 0)
      << "numCols must be a multiple of 16";
  TVM_FFI_ICHECK_EQ(numCols % tokenspeed::CVT_ELTS_PER_THREAD, 0)
      << "numCols must be a multiple of 8";

  TVM_FFI_ICHECK_EQ(out.ndim(), 2);
  TVM_FFI_ICHECK_EQ(out.size(0), numRows);
  TVM_FFI_ICHECK_EQ(out.size(1), numCols / 2);  // 2 fp4 values per byte

  TVM_FFI_ICHECK_EQ(global_scale.ndim(), 1);
  TVM_FFI_ICHECK_EQ(global_scale.size(0), 1);

  cudaSetDevice(out.device().device_id);
  const cudaStream_t stream = get_stream(out.device());

  int const colsPerRow = numCols / tokenspeed::CVT_ELTS_PER_THREAD;
  // cap at 1024 threads/block; for typical shared-expert I=512..2048 this is
  // 64..256 threads and one tile per thread, fitting in a single pass.
  int const blockSize = std::min(colsPerRow, 1024);

  cudaLaunchConfig_t config;
  config.gridDim = dim3(numRows);
  config.blockDim = dim3(blockSize);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
    auto kernel = tokenspeed::silu_and_mul_fuse_nvfp4_quant_kernel<c_type>;
    cudaLaunchKernelEx(
        &config, kernel, numCols,
        static_cast<c_type const*>(input.data_ptr()),
        static_cast<float const*>(global_scale.data_ptr()),
        static_cast<uint32_t*>(out.data_ptr()),
        static_cast<uint32_t*>(scale_out.data_ptr()));
    cudaError_t err = cudaGetLastError();
    TVM_FFI_ICHECK(err == cudaSuccess)
        << "silu_and_mul_fuse_nvfp4_quant launch failed: "
        << cudaGetErrorString(err);
    return true;
  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(silu_and_mul_fuse_nvfp4_quant,
                              silu_and_mul_fuse_nvfp4_quant);
