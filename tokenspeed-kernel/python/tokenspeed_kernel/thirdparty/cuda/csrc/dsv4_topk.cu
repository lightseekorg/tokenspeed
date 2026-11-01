/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * TokenSpeed modifications:
 *   - Ported the PyTorch custom op boundary to TokenSpeed's TVM FFI CUDA
 *     extension style.
 *   - Exposed only the persistent decode top-k contract needed by the
 *     model runtime.
 */

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "dsv4_persistent_topk.cuh"
#include "tvm_ffi_utils.h"

using tvm::ffi::TensorView;

namespace {

cudaError_t launch_persistent_cluster_topk(
    const vllm::persistent::PersistentTopKParams& params, uint32_t num_rows,
    uint32_t cluster_size, cudaStream_t stream,
    void (*kernel)(vllm::persistent::PersistentTopKParams), bool enable_pdl,
    bool allow_nonportable_cluster_size) {
  namespace P = vllm::persistent;
  if (allow_nonportable_cluster_size) {
    const cudaError_t attribute_error = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (attribute_error != cudaSuccess) return attribute_error;
  }

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(num_rows, cluster_size, 1);
  config.blockDim = dim3(P::kThreadsPerBlock);
  config.dynamicSmemBytes = 0;
  config.stream = stream;

  cudaLaunchAttribute attributes[2]{};
  attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  attributes[1].id = cudaLaunchAttributeClusterDimension;
  attributes[1].val.clusterDim = {1, cluster_size, 1};
  config.attrs = attributes;
  config.numAttrs = 2;

  return cudaLaunchKernelEx(&config, kernel, params);
}

bool device_supports_cluster_launch(int device_id) {
  int supported = 0;
  const cudaError_t error = cudaDeviceGetAttribute(
      &supported, cudaDevAttrClusterLaunch, device_id);
  if (error != cudaSuccess) {
    // Treat an unavailable capability query conservatively. Clear the runtime
    // error before launching the non-cluster fallback below.
    (void)cudaGetLastError();
    return false;
  }
  return supported != 0;
}

bool is_unsupported_cluster_launch_error(cudaError_t error) {
  return error == cudaErrorInvalidValue ||
         error == cudaErrorInvalidConfiguration ||
         error == cudaErrorInvalidClusterSize ||
         error == cudaErrorNotSupported;
}

int device_sm_count(int device_id) {
  // Serving workers use one GPU or homogeneous full GPUs.
  static const int sm_count = [device_id] {
    int count = 0;
    const cudaError_t error = cudaDeviceGetAttribute(
        &count, cudaDevAttrMultiProcessorCount, device_id);
    TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
    return count;
  }();
  return sm_count;
}

template <int TopK, int Level, bool kUsePDL>
void launch_noncluster_topk_level(
    const vllm::persistent::PersistentTopKParams& params,
    bool prefer_single_cta, cudaStream_t stream) {
  namespace P = vllm::persistent;
  auto kernel = P::persistent_topk_kernel<TopK, Level, kUsePDL, 2>;
  if constexpr (kUsePDL) {
    // A one-block register budget spreads small grids of waiting CTAs
    // across SMs as scoring finishes. The compiler may trade occupancy
    // for fewer instructions in register and streaming selectors.
    if (prefer_single_cta) {
      kernel = P::persistent_topk_kernel<TopK, Level, true, 1>;
    }
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(params.num_rows);
    config.blockDim = dim3(P::kThreadsPerBlock);
    config.stream = stream;
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = true;
    config.attrs = &attribute;
    config.numAttrs = 1;
    const cudaError_t error = cudaLaunchKernelEx(&config, kernel, params);
    TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
  } else {
    kernel<<<params.num_rows, P::kThreadsPerBlock, 0, stream>>>(params);
  }
}

template <int TopK, bool kUsePDL>
void launch_noncluster_persistent_topk(
    const vllm::persistent::PersistentTopKParams& params, int64_t num_rows,
    int64_t max_seq_len, int device_id, cudaStream_t stream) {
  namespace P = vllm::persistent;
  bool prefer_single_cta = false;
  if constexpr (kUsePDL) {
    prefer_single_cta = num_rows <= device_sm_count(device_id);
  }
  if (max_seq_len <= P::kRegister2MaxSeqLen) {
    launch_noncluster_topk_level<TopK, 0, kUsePDL>(params, prefer_single_cta,
                                                   stream);
  } else if (max_seq_len <= P::kRegister4MaxSeqLen) {
    launch_noncluster_topk_level<TopK, 1, kUsePDL>(params, prefer_single_cta,
                                                   stream);
  } else {
    launch_noncluster_topk_level<TopK, 2, kUsePDL>(params, prefer_single_cta,
                                                   stream);
  }
}

template <int TopK, bool kUsePDL>
void launch_persistent_topk(const TensorView& logits, const TensorView& lengths,
                            TensorView& output, int64_t max_seq_len,
                            int64_t q_len_per_req, cudaStream_t stream) {
  namespace P = vllm::persistent;

  const int64_t num_rows = logits.size(0);
  const int64_t stride = logits.size(1);

  P::PersistentTopKParams params;
  params.input = static_cast<float*>(logits.data_ptr());
  params.output = static_cast<int32_t*>(output.data_ptr());
  params.lengths = static_cast<int32_t*>(lengths.data_ptr());
  params.num_rows = static_cast<uint32_t>(num_rows);
  params.stride = static_cast<uint32_t>(stride);
  params.top_k = static_cast<uint32_t>(TopK);
  params.max_seq_len = static_cast<uint32_t>(max_seq_len);
  params.q_len_per_req = static_cast<uint32_t>(q_len_per_req);

  cudaError_t launch_error = cudaSuccess;
  const bool use_cluster =
      max_seq_len > P::kClusterFloor && num_rows <= 32 &&
      device_supports_cluster_launch(logits.device().device_id);
  if (use_cluster) {
    const auto rows = static_cast<uint32_t>(num_rows);
    if (max_seq_len > P::kClusterHist10Floor &&
        max_seq_len <= P::kClusterRegister4MaxSeqLen) {
      launch_error = launch_persistent_cluster_topk(
          params, rows, 8, stream,
          P::cluster8_register4_topk_kernel<TopK, 10, kUsePDL>, kUsePDL, false);
    } else if (max_seq_len > P::kClusterRegister4MaxSeqLen &&
               max_seq_len <= P::kCluster16MaxSeqLen) {
      launch_error = launch_persistent_cluster_topk(
          params, rows, 16, stream, P::cluster16_topk_kernel<TopK, 10, kUsePDL>,
          kUsePDL, true);
      if (is_unsupported_cluster_launch_error(launch_error)) {
        // A cluster of 16 CTAs is an architecture-specific optimization. Keep
        // the portable cluster-8 implementation as the correctness fallback.
        (void)cudaGetLastError();
        launch_error = launch_persistent_cluster_topk(
            params, rows, P::kClusterSize, stream,
            P::cluster_topk_kernel<TopK, 10, kUsePDL>, kUsePDL, false);
      }
    } else if (max_seq_len > P::kClusterHist10Floor) {
      launch_error = launch_persistent_cluster_topk(
          params, rows, P::kClusterSize, stream,
          P::cluster_topk_kernel<TopK, 10, kUsePDL>, kUsePDL, false);
    } else {
      launch_error = launch_persistent_cluster_topk(
          params, rows, P::kClusterSize, stream,
          P::cluster_topk_kernel<TopK, 9, kUsePDL>, kUsePDL, false);
    }
    if (is_unsupported_cluster_launch_error(launch_error)) {
      // Some Hopper SKUs reject cluster launches despite sharing sm_90 with
      // cluster-capable devices. Fall back to the original single-CTA kernel.
      (void)cudaGetLastError();
      launch_noncluster_persistent_topk<TopK, kUsePDL>(
          params, num_rows, max_seq_len, logits.device().device_id, stream);
      launch_error = cudaSuccess;
    }
  } else {
    launch_noncluster_persistent_topk<TopK, kUsePDL>(
        params, num_rows, max_seq_len, logits.device().device_id, stream);
  }

  cudaError_t err =
      launch_error == cudaSuccess ? cudaGetLastError() : launch_error;
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "dsv4_persistent_topk failed: " << cudaGetErrorString(err);
}

template <int TopK>
void dispatch_persistent_topk(const TensorView& logits,
                              const TensorView& lengths, TensorView& output,
                              int64_t max_seq_len, int64_t q_len_per_req,
                              bool enable_pdl, cudaStream_t stream) {
  if (enable_pdl) {
    launch_persistent_topk<TopK, true>(logits, lengths, output, max_seq_len,
                                       q_len_per_req, stream);
  } else {
    launch_persistent_topk<TopK, false>(logits, lengths, output, max_seq_len,
                                        q_len_per_req, stream);
  }
}

}  // namespace

void dsv4_persistent_topk(TensorView logits, TensorView lengths,
                          TensorView output, int64_t k, int64_t max_seq_len,
                          int64_t q_len_per_req, bool enable_pdl) {
  CHECK_CUDA(logits);
  CHECK_CUDA(lengths);
  CHECK_CUDA(output);
  CHECK_DIM(2, logits);
  TVM_FFI_ICHECK(lengths.ndim() == 1 || lengths.ndim() == 2)
      << "lengths must be 1D or 2D";
  CHECK_DIM(2, output);
  CHECK_CONTIGUOUS(logits);
  CHECK_CONTIGUOUS(lengths);
  CHECK_CONTIGUOUS(output);
  TVM_FFI_ICHECK(logits.dtype() == dl_float32) << "logits must be float32";
  TVM_FFI_ICHECK(lengths.dtype() == dl_int32) << "lengths must be int32";
  TVM_FFI_ICHECK(output.dtype() == dl_int32) << "output must be int32";
  TVM_FFI_ICHECK(q_len_per_req >= 1) << "q_len_per_req must be >= 1";
  TVM_FFI_ICHECK(logits.size(0) % q_len_per_req == 0)
      << "q_len_per_req must divide num_rows";
  TVM_FFI_ICHECK(lengths.numel() == logits.size(0) / q_len_per_req)
      << "lengths size mismatch (expected num_rows / q_len_per_req)";
  TVM_FFI_ICHECK(output.size(0) == logits.size(0) && output.size(1) == k)
      << "output size mismatch";
  TVM_FFI_ICHECK(k == 512 || k == 1024 || k == 2048)
      << "persistent topk supports k=512, k=1024, or k=2048, got " << k;
  TVM_FFI_ICHECK(max_seq_len > 0) << "max_seq_len must be positive";

  cudaError_t err = cudaSetDevice(logits.device().device_id);
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "cudaSetDevice failed: " << cudaGetErrorString(err);
  cudaStream_t stream = get_stream(logits.device());
  if (k == 512) {
    dispatch_persistent_topk<512>(logits, lengths, output, max_seq_len,
                                  q_len_per_req, enable_pdl, stream);
  } else if (k == 1024) {
    dispatch_persistent_topk<1024>(logits, lengths, output, max_seq_len,
                                   q_len_per_req, enable_pdl, stream);
  } else {
    dispatch_persistent_topk<2048>(logits, lengths, output, max_seq_len,
                                   q_len_per_req, enable_pdl, stream);
  }
}
