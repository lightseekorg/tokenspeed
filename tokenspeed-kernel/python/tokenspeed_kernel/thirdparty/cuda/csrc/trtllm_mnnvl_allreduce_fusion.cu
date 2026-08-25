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
 * tvm_ffi binding for the MNNVL-structured one-shot allreduce fusion
 * (see include/flashinfer/comm/trtllm_mnnvl_allreduce_fusion.cuh, vendored/
 * adapted from flashinfer's trtllm_mnnvl_allreduce.cuh, Apache-2.0).
 */

#include <string>

#include "flashinfer/comm/trtllm_mnnvl_allreduce_fusion.cuh"
#include "tvm_ffi_utils.h"

using namespace flashinfer::trtllm_mnnvl_allreduce_fusion;

using tvm::ffi::Optional;

#define DISPATCH_HALF_TYPES_FOR_MNNVL_ALLREDUCE(dtype, c_type, ...)                \
  [&] {                                                                            \
    switch (encode_dlpack_dtype(dtype)) {                                          \
      case float16_code: {                                                         \
        using c_type = half;                                                       \
        return __VA_ARGS__();                                                      \
      }                                                                            \
      case bfloat16_code: {                                                        \
        using c_type = __nv_bfloat16;                                              \
        return __VA_ARGS__();                                                      \
      }                                                                            \
      default:                                                                     \
        TVM_FFI_LOG_AND_THROW(NotImplementedError)                                 \
            << "mnnvl allreduce fusion only supports float16/bfloat16 payloads."; \
    }                                                                              \
  }()

void trtllm_mnnvl_allreduce_fusion(
    TensorView allreduce_in, int64_t world_size, int64_t world_rank, int64_t token_num,
    int64_t hidden_size, int64_t multicast_ptr, int64_t buffer_ptr_local,
    TensorView peer_ptrs, TensorView buffer_flags,
    bool launch_with_pdl, bool use_oneshot, bool trigger_completion_at_end, bool fp32_acc,
    int64_t pattern_code,
    Optional<TensorView> allreduce_out, Optional<TensorView> residual_in,
    Optional<TensorView> residual_out, Optional<TensorView> norm_out,
    Optional<TensorView> rms_gamma, Optional<double> rms_eps, Optional<TensorView> attnres_m,
    Optional<TensorView> attnres_s, Optional<TensorView> attnres_acc,
    Optional<TensorView> attnres_res_w, Optional<TensorView> attnres_out_norm_w,
    Optional<int64_t> latent_width) {
  cudaSetDevice(allreduce_in.device().device_id);
  TVM_FFI_ICHECK(multicast_ptr != 0) << "multicast_ptr must be a valid NVLS multicast VA";
  TVM_FFI_ICHECK(buffer_ptr_local != 0) << "buffer_ptr_local must be a valid unicast VA";
  TVM_FFI_ICHECK(peer_ptrs.numel() == world_size)
      << "peer_ptrs must hold one unicast base per rank";

  DISPATCH_HALF_TYPES_FOR_MNNVL_ALLREDUCE(allreduce_in.dtype(), c_type, [&] {
    AllReduceFusionParams<c_type> params;
    params.nranks = world_size;
    params.rank = world_rank;
    params.size = token_num * hidden_size;
    params.hidden_dim = hidden_size;
    params.workspace = nullptr;  // mnnvl comm pointers are passed separately
    params.allreduce_in = reinterpret_cast<void*>(allreduce_in.data_ptr());
    params.allreduce_out = allreduce_out.has_value()
                               ? reinterpret_cast<void*>(allreduce_out.value().data_ptr())
                               : nullptr;
    params.residual_in =
        residual_in.has_value() ? reinterpret_cast<void*>(residual_in.value().data_ptr()) : nullptr;
    params.residual_out = residual_out.has_value()
                              ? reinterpret_cast<void*>(residual_out.value().data_ptr())
                              : nullptr;
    params.norm_out =
        norm_out.has_value() ? reinterpret_cast<void*>(norm_out.value().data_ptr()) : nullptr;
    params.partial_normed_out = nullptr;
    params.quant_out = nullptr;
    params.scale_out = nullptr;
    params.scale_stride = 0;
    params.rms_gamma =
        rms_gamma.has_value() ? reinterpret_cast<void*>(rms_gamma.value().data_ptr()) : nullptr;
    params.rms_eps = rms_eps.has_value() ? static_cast<float>(rms_eps.value()) : 0.0f;
    params.attnres_m =
        attnres_m.has_value() ? reinterpret_cast<void*>(attnres_m.value().data_ptr()) : nullptr;
    params.attnres_s =
        attnres_s.has_value() ? reinterpret_cast<void*>(attnres_s.value().data_ptr()) : nullptr;
    params.attnres_acc =
        attnres_acc.has_value() ? reinterpret_cast<void*>(attnres_acc.value().data_ptr()) : nullptr;
    params.attnres_res_w = attnres_res_w.has_value()
                               ? reinterpret_cast<void*>(attnres_res_w.value().data_ptr())
                               : nullptr;
    params.attnres_out_norm_w =
        attnres_out_norm_w.has_value()
            ? reinterpret_cast<void*>(attnres_out_norm_w.value().data_ptr())
            : nullptr;
    params.latent_width = latent_width.has_value() ? static_cast<int>(latent_width.value()) : 0;
    params.scale_factor = nullptr;
    params.use_oneshot = use_oneshot;
    params.pattern = static_cast<AllReduceFusionPattern>(pattern_code);
    params.trigger_completion_at_end = trigger_completion_at_end;
    params.residual_reduce_scattered = false;
    params.stream = get_stream(allreduce_in.device());

    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(buffer_flags.dtype()), encode_dlpack_dtype(dl_uint32))
        << "buffer_flags must be uint32";
    TVM_FFI_ICHECK_GE(buffer_flags.numel(), 9) << "buffer_flags must hold >= 9 uint32 words";

    MnnvlCommArgs comm;
    comm.multicast_ptr = reinterpret_cast<void*>(multicast_ptr);
    comm.buffer_ptr_local = reinterpret_cast<void*>(buffer_ptr_local);
    comm.peer_ptrs = reinterpret_cast<void* const*>(peer_ptrs.data_ptr());
    comm.buffer_flags = reinterpret_cast<uint32_t*>(buffer_flags.data_ptr());

    auto status = mnnvl_allreduce_fusion_op(params, comm, launch_with_pdl, fp32_acc);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "mnnvl_allreduce_fusion_op failed with error code " << cudaGetErrorString(status);
  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_mnnvl_allreduce_fusion, trtllm_mnnvl_allreduce_fusion);
