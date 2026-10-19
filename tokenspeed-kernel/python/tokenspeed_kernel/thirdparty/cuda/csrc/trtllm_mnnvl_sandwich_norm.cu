/*
 * Copyright (c) 2026 LightSeek Foundation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * tvm_ffi binding: the pre/post ("sandwich") norm boundary on the MNNVL
 * all-reduce kernels (include/flashinfer/comm/trtllm_mnnvl_sandwich_norm.cuh).
 * The all-reduce is the plain MNNVL all-reduce's: the same vendored kernels,
 * launcher and fp32_acc=false sum, at the strategy the caller resolved from
 * the workspace. Only the fp32_acc=false kernels are instantiated.
 */

#include <cstdint>

#include "flashinfer/comm/trtllm_mnnvl_sandwich_norm.cuh"
#include "tvm_ffi_utils.h"

using namespace flashinfer::trtllm_mnnvl_allreduce_fusion;
namespace sandwich = flashinfer::trtllm_allreduce_fusion::sandwich;

namespace {

constexpr auto kPattern = AllReduceFusionPattern::kARSandwichResidualRMSNorm;

bool aligned16(void const* ptr) { return reinterpret_cast<uintptr_t>(ptr) % 16 == 0; }

void check_bf16(TensorView tensor, int64_t numel, char const* name) {
  TVM_FFI_ICHECK(encode_dlpack_dtype(tensor.dtype()) == bfloat16_code) << name << " must be BF16";
  TVM_FFI_ICHECK(tensor.numel() == numel) << name << " must hold " << numel << " elements";
  TVM_FFI_ICHECK(aligned16(tensor.data_ptr())) << name << " must be 16-byte aligned";
}

cudaError_t launch(AllReduceFusionParams<__nv_bfloat16> const& params, MnnvlCommArgs const& comm,
                   bool launch_with_pdl) {
  switch (params.nranks) {
    case 2:
      return mnnvl_allreduce_fusion_kernel_launcher<kPattern, __nv_bfloat16, 2,
                                                    /*Fp32AccKernels=*/false>(
          params, comm, launch_with_pdl, /*fp32_acc=*/false);
    case 4:
      return mnnvl_allreduce_fusion_kernel_launcher<kPattern, __nv_bfloat16, 4,
                                                    /*Fp32AccKernels=*/false>(
          params, comm, launch_with_pdl, /*fp32_acc=*/false);
    case 8:
      return mnnvl_allreduce_fusion_kernel_launcher<kPattern, __nv_bfloat16, 8,
                                                    /*Fp32AccKernels=*/false>(
          params, comm, launch_with_pdl, /*fp32_acc=*/false);
    case 16:
      return mnnvl_allreduce_fusion_kernel_launcher<kPattern, __nv_bfloat16, 16,
                                                    /*Fp32AccKernels=*/false>(
          params, comm, launch_with_pdl, /*fp32_acc=*/false);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace

void trtllm_mnnvl_sandwich_norm_allreduce(
    TensorView allreduce_in, TensorView residual_in, TensorView post_norm_gamma,
    TensorView rms_gamma, TensorView norm_out, TensorView residual_out, int64_t world_size,
    int64_t world_rank, int64_t token_num, int64_t hidden_size, int64_t multicast_ptr,
    int64_t buffer_ptr_local, TensorView peer_ptrs, TensorView buffer_flags,
    bool launch_with_pdl, bool use_oneshot, bool trigger_completion_at_end, double rms_eps,
    double x_scale, double residual_scale) {
  cudaSetDevice(allreduce_in.device().device_id);
  TVM_FFI_ICHECK(world_size == 2 || world_size == 4 || world_size == 8 || world_size == 16)
      << "sandwich norm all-reduce: world size " << world_size << " is not 2, 4, 8 or 16";
  TVM_FFI_ICHECK(token_num >= 1 && token_num <= details::kMnnvlTwoShotMaxToken)
      << "sandwich norm all-reduce: token_num " << token_num << " is outside [1, "
      << details::kMnnvlTwoShotMaxToken << "]";
  // Whole warps of eight-element groups, at most kMaxWarps of them; the
  // launcher refuses a hidden size its cluster partition cannot split.
  TVM_FFI_ICHECK(hidden_size % 256 == 0 && hidden_size / 256 >= 1 &&
                 hidden_size / 256 <= sandwich::kMaxWarps)
      << "sandwich norm all-reduce: hidden size " << hidden_size
      << " is not a multiple of 256 in [256, " << 256 * sandwich::kMaxWarps << "]";
  int64_t const rows = token_num * hidden_size;
  check_bf16(allreduce_in, rows, "allreduce_in");
  check_bf16(residual_in, rows, "residual_in");
  check_bf16(norm_out, rows, "norm_out");
  check_bf16(residual_out, rows, "residual_out");
  check_bf16(post_norm_gamma, hidden_size, "post_norm_gamma");
  check_bf16(rms_gamma, hidden_size, "rms_gamma");
  TVM_FFI_ICHECK(multicast_ptr != 0 && buffer_ptr_local != 0) << "mnnvl workspace required";
  TVM_FFI_ICHECK(peer_ptrs.numel() == world_size) << "peer_ptrs must hold one base per rank";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(buffer_flags.dtype()), encode_dlpack_dtype(dl_uint32))
      << "buffer_flags must be uint32";
  TVM_FFI_ICHECK_GE(buffer_flags.numel(), 9) << "buffer_flags must hold >= 9 uint32 words";

  AllReduceFusionParams<__nv_bfloat16> params;
  params.nranks = static_cast<int>(world_size);
  params.rank = static_cast<int>(world_rank);
  params.size = static_cast<int>(rows);
  params.hidden_dim = static_cast<int>(hidden_size);
  params.scale_stride = 0;
  params.workspace = nullptr;  // mnnvl comm pointers are passed separately
  params.allreduce_in = allreduce_in.data_ptr();
  params.allreduce_out = nullptr;
  params.residual_in = residual_in.data_ptr();
  params.residual_out = residual_out.data_ptr();
  params.norm_out = norm_out.data_ptr();
  params.partial_normed_out = nullptr;
  params.quant_out = nullptr;
  params.scale_out = nullptr;
  params.rms_gamma = rms_gamma.data_ptr();
  params.rms_eps = static_cast<float>(rms_eps);
  params.scale_factor = nullptr;
  params.use_oneshot = use_oneshot;
  params.stream = get_stream(allreduce_in.device());
  params.pattern = kPattern;
  params.trigger_completion_at_end = trigger_completion_at_end;
  params.residual_reduce_scattered = false;
  params.post_norm_gamma = post_norm_gamma.data_ptr();
  params.x_scale = static_cast<float>(x_scale);
  params.residual_scale = static_cast<float>(residual_scale);

  MnnvlCommArgs comm;
  comm.multicast_ptr = reinterpret_cast<void*>(multicast_ptr);
  comm.buffer_ptr_local = reinterpret_cast<void*>(buffer_ptr_local);
  comm.peer_ptrs = reinterpret_cast<void* const*>(peer_ptrs.data_ptr());
  comm.buffer_flags = reinterpret_cast<uint32_t*>(buffer_flags.data_ptr());

  cudaError_t status = launch(params, comm, launch_with_pdl);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sandwich norm all-reduce failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_mnnvl_sandwich_norm_allreduce,
                              trtllm_mnnvl_sandwich_norm_allreduce);
