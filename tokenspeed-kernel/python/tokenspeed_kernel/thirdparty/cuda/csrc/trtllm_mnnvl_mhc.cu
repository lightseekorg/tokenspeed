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
 */

#include "flashinfer/comm/trtllm_mnnvl_allreduce_fusion.cuh"
#include "tvm_ffi_utils.h"
using namespace flashinfer::trtllm_mnnvl_allreduce_fusion;

void trtllm_mnnvl_mhc(
    TensorView x, TensorView residual, TensorView post, TensorView comb, TensorView pre,
    TensorView weight, TensorView residual_out, TensorView norm_out, double eps,
    int64_t rank, int64_t multicast_ptr, int64_t local_ptr,
    TensorView peers, TensorView flags, int64_t capacity_bytes, bool use_oneshot,
    bool pdl) {
  TVM_FFI_ICHECK(x.ndim() == 2);
  int64_t tokens = x.size(0), hidden = x.size(1), nranks = peers.numel();
  TVM_FFI_ICHECK(hidden == 5120 && tokens > 0 && tokens <= details::kMnnvlTwoShotMaxToken);
  TVM_FFI_ICHECK(rank >= 0 && rank < nranks && multicast_ptr && local_ptr);
  for (auto t : {x, residual, weight, residual_out, norm_out}) {
    TVM_FFI_ICHECK(t.IsContiguous() && t.device().device_type == kDLCUDA);
    TVM_FFI_ICHECK(t.device().device_id == x.device().device_id);
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(t.dtype()), bfloat16_code);
  }
  for (auto t : {post, comb, pre}) {
    TVM_FFI_ICHECK(t.IsContiguous() && t.device().device_type == kDLCUDA);
    TVM_FFI_ICHECK(t.device().device_id == x.device().device_id);
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(t.dtype()), float32_code);
  }
  TVM_FFI_ICHECK(residual.numel() == tokens * 4 * hidden);
  TVM_FFI_ICHECK(residual_out.numel() == residual.numel());
  TVM_FFI_ICHECK(norm_out.numel() == x.numel() && weight.numel() == hidden);
  TVM_FFI_ICHECK(post.numel() == tokens * 4 && pre.numel() == tokens * 4);
  TVM_FFI_ICHECK(comb.numel() == tokens * 16 && eps > 0);
  TVM_FFI_ICHECK(peers.ndim() == 1 && peers.IsContiguous() && flags.numel() >= 9);
  // Communication slots scale with group size, independently of HC's four streams.
  int64_t const lane_tokens = use_oneshot ? tokens * nranks
      : 2 * ((tokens + nranks - 1) / nranks) * nranks;
  int64_t const payload_bytes = lane_tokens * hidden * sizeof(__nv_bfloat16);
  TVM_FFI_ICHECK(capacity_bytes >= payload_bytes);
  cudaSetDevice(x.device().device_id);
  AllReduceFusionParams<__nv_bfloat16> params{};
  params.nranks = nranks;
  params.rank = rank;
  params.size = tokens * hidden;
  params.hidden_dim = hidden;
  params.allreduce_in = x.data_ptr();
  params.residual_in = residual.data_ptr();
  params.residual_out = residual_out.data_ptr();
  params.mhc_post = static_cast<float*>(post.data_ptr());
  params.mhc_pre = static_cast<float*>(pre.data_ptr());
  params.mhc_comb = static_cast<float*>(comb.data_ptr());
  params.rms_gamma = weight.data_ptr();
  params.norm_out = norm_out.data_ptr();
  params.rms_eps = eps;
  params.use_oneshot = use_oneshot;
  params.pattern = AllReduceFusionPattern::kAllReduceMhcNorm;
  params.trigger_completion_at_end = true;
  params.stream = get_stream(x.device());
  MnnvlCommArgs comm{reinterpret_cast<void*>(multicast_ptr),
                    reinterpret_cast<void*>(local_ptr),
                    reinterpret_cast<uint32_t*>(flags.data_ptr()),
                    reinterpret_cast<void* const*>(peers.data_ptr())};
  auto status = mnnvl_dispatch_ranks(params.nranks, [&](auto ranks) {
    return mnnvl_allreduce_fusion_kernel_launcher<AllReduceFusionPattern::kAllReduceMhcNorm,
                                                 __nv_bfloat16, decltype(ranks)::value>(
        params, comm, pdl, /*fp32_acc=*/false);
  });
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_mnnvl_mhc, trtllm_mnnvl_mhc);
