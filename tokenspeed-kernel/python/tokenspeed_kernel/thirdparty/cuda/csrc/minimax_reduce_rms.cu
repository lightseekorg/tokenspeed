/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#include <cstdint>

#include "flashinfer/comm/minimax_reduce_rms.cuh"
#include "tvm_ffi_utils.h"

using flashinfer::minimax_ar::MinimaxDType;
using flashinfer::minimax_ar::MiniMaxReduceRMSParams;

namespace {

inline MinimaxDType dlpack_to_minimax_dtype(DLDataType dtype) {
  switch (encode_dlpack_dtype(dtype)) {
    case float16_code:
      return MinimaxDType::kHALF;
    case bfloat16_code:
      return MinimaxDType::kBF16;
    case float32_code:
      return MinimaxDType::kFLOAT;
    default:
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "Unsupported dtype for minimax_reduce_rms (expected fp16/bf16/fp32).";
  }
  return MinimaxDType::kBF16;  // unreachable
}

inline void check_nranks(int64_t nranks) {
  TVM_FFI_ICHECK(nranks == 2 || nranks == 4 || nranks == 8 || nranks == 16)
      << "minimax_reduce_rms: only nranks in {2,4,8,16} supported, got " << nranks;
}

}  // anonymous namespace

// Single-matrix Lamport AR + RMSNorm. `input` is [token_num, hidden_dim_local]
// with hidden_dim_local = global_hidden_dim / nranks. `norm_weight` is
// [hidden_dim_local] in bf16. Writes `rms_norm_out` of the same shape/dtype
// as `input` in-place; caller allocates.
void minimax_allreduce_rms(TensorView input, TensorView norm_weight, TensorView rms_norm_out,
                           TensorView workspace, int64_t rank, int64_t nranks, double eps,
                           bool trigger_completion_at_end, bool launch_with_pdl) {
  cudaSetDevice(input.device().device_id);
  TVM_FFI_ICHECK_EQ(input.ndim(), 2) << "minimax_allreduce_rms: input must be 2D";
  TVM_FFI_ICHECK_EQ(norm_weight.ndim(), 1) << "minimax_allreduce_rms: norm_weight must be 1D";
  TVM_FFI_ICHECK_EQ(input.size(-1), norm_weight.size(0))
      << "minimax_allreduce_rms: input hidden dim must match norm_weight";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(norm_weight.dtype()), bfloat16_code)
      << "minimax_allreduce_rms: norm_weight must be bf16";
  TVM_FFI_ICHECK_EQ(input.dtype(), rms_norm_out.dtype())
      << "minimax_allreduce_rms: input/output dtype mismatch";
  check_nranks(nranks);

  MiniMaxReduceRMSParams params;
  params.nranks = static_cast<int>(nranks);
  params.rank = static_cast<int>(rank);
  params.dtype = dlpack_to_minimax_dtype(input.dtype());
  params.size_q = static_cast<int>(input.numel());
  params.hidden_dim = static_cast<int>(input.size(-1));
  params.size_k = 0;
  params.hidden_dim_k = 0;
  params.workspace = reinterpret_cast<void**>(workspace.data_ptr());
  params.allreduce_in = input.data_ptr();
  params.rms_norm_out = rms_norm_out.data_ptr();
  params.rms_gamma = norm_weight.data_ptr();
  params.allreduce_in_k = nullptr;
  params.rms_norm_out_k = nullptr;
  params.rms_gamma_k = nullptr;
  params.rms_eps = static_cast<float>(eps);
  params.stream = get_stream(input.device());
  params.trigger_completion_at_end = trigger_completion_at_end;
  params.enable_pdl = launch_with_pdl;

  flashinfer::minimax_ar::minimax_reduce_rms_op(params);
}

// Fused Q+K AR + RMSNorm. Requires the globally supported fast-path shape
// (global_head_dim_q == 6144, global_head_dim_k == 1024). Writes
// rms_norm_out_q/k (caller-allocated, same shape/dtype as q/k).
void minimax_allreduce_rms_qk(TensorView q, TensorView k, TensorView norm_weight_q,
                              TensorView norm_weight_k, TensorView rms_norm_out_q,
                              TensorView rms_norm_out_k, TensorView workspace, int64_t rank,
                              int64_t nranks, double eps, bool trigger_completion_at_end,
                              bool launch_with_pdl) {
  cudaSetDevice(q.device().device_id);
  TVM_FFI_ICHECK_EQ(q.dtype(), k.dtype()) << "minimax_allreduce_rms_qk: q/k dtype mismatch";
  TVM_FFI_ICHECK_EQ(q.ndim(), 2) << "q must be 2D";
  TVM_FFI_ICHECK_EQ(k.ndim(), 2) << "k must be 2D";
  TVM_FFI_ICHECK_EQ(q.size(0), k.size(0)) << "q and k must have same num_token";
  TVM_FFI_ICHECK_EQ(norm_weight_q.ndim(), 1);
  TVM_FFI_ICHECK_EQ(norm_weight_k.ndim(), 1);
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(norm_weight_q.dtype()), bfloat16_code)
      << "norm_weight_q must be bf16";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(norm_weight_k.dtype()), bfloat16_code)
      << "norm_weight_k must be bf16";
  int64_t head_dim_q = q.size(-1);
  int64_t head_dim_k = k.size(-1);
  TVM_FFI_ICHECK_GE(head_dim_q, head_dim_k) << "head_dim_q must be >= head_dim_k";
  TVM_FFI_ICHECK_EQ(head_dim_q, norm_weight_q.size(0));
  TVM_FFI_ICHECK_EQ(head_dim_k, norm_weight_k.size(0));
  TVM_FFI_ICHECK_EQ(head_dim_q * nranks, 6144)
      << "minimax_allreduce_rms_qk: only global head_dim_q == 6144 supported";
  TVM_FFI_ICHECK_EQ(head_dim_k * nranks, 1024)
      << "minimax_allreduce_rms_qk: only global head_dim_k == 1024 supported";
  TVM_FFI_ICHECK_EQ(q.dtype(), rms_norm_out_q.dtype());
  TVM_FFI_ICHECK_EQ(k.dtype(), rms_norm_out_k.dtype());
  check_nranks(nranks);

  // Outputs must be tightly packed since the kernel writes them at the
  // tightly-packed stride (ThreadsPerRowQ/K).
  TVM_FFI_ICHECK_EQ(rms_norm_out_q.stride(0), head_dim_q)
      << "rms_norm_out_q must be row-contiguous (stride[0] == head_dim_q)";
  TVM_FFI_ICHECK_EQ(rms_norm_out_q.stride(1), 1) << "rms_norm_out_q stride[1] must be 1";
  TVM_FFI_ICHECK_EQ(rms_norm_out_k.stride(0), head_dim_k)
      << "rms_norm_out_k must be row-contiguous (stride[0] == head_dim_k)";
  TVM_FFI_ICHECK_EQ(rms_norm_out_k.stride(1), 1) << "rms_norm_out_k stride[1] must be 1";

  // Inputs may be a strided slice (e.g. Q from a fused QKV buffer). The
  // float4 fast path indexes rows at `q_row_stride_f4` float4s, so the
  // element-stride must (a) be divisible by elems_per_access, and (b) leave
  // every row 16-byte aligned. Inner stride must be 1 (no transposed views).
  const MinimaxDType dtype = dlpack_to_minimax_dtype(q.dtype());
  const int elems_per_access = (dtype == MinimaxDType::kFLOAT) ? 4 : 8;
  const int64_t q_row_stride_elems = q.stride(0);
  const int64_t k_row_stride_elems = k.stride(0);
  TVM_FFI_ICHECK_EQ(q.stride(1), 1)
      << "minimax_allreduce_rms_qk: q inner stride must be 1, got " << q.stride(1);
  TVM_FFI_ICHECK_EQ(k.stride(1), 1)
      << "minimax_allreduce_rms_qk: k inner stride must be 1, got " << k.stride(1);
  TVM_FFI_ICHECK_EQ(q_row_stride_elems % elems_per_access, 0)
      << "minimax_allreduce_rms_qk: q row stride " << q_row_stride_elems
      << " must be divisible by elems_per_access=" << elems_per_access;
  TVM_FFI_ICHECK_EQ(k_row_stride_elems % elems_per_access, 0)
      << "minimax_allreduce_rms_qk: k row stride " << k_row_stride_elems
      << " must be divisible by elems_per_access=" << elems_per_access;

  MiniMaxReduceRMSParams params;
  params.nranks = static_cast<int>(nranks);
  params.rank = static_cast<int>(rank);
  params.dtype = dtype;
  params.size_q = static_cast<int>(q.numel());
  params.hidden_dim = static_cast<int>(head_dim_q);
  params.size_k = static_cast<int>(k.numel());
  params.hidden_dim_k = static_cast<int>(head_dim_k);
  params.q_row_stride_f4 = static_cast<int>(q_row_stride_elems / elems_per_access);
  params.k_row_stride_f4 = static_cast<int>(k_row_stride_elems / elems_per_access);
  params.workspace = reinterpret_cast<void**>(workspace.data_ptr());
  params.allreduce_in = q.data_ptr();
  params.rms_gamma = norm_weight_q.data_ptr();
  params.allreduce_in_k = k.data_ptr();
  params.rms_gamma_k = norm_weight_k.data_ptr();
  params.rms_norm_out = rms_norm_out_q.data_ptr();
  params.rms_norm_out_k = rms_norm_out_k.data_ptr();
  params.rms_eps = static_cast<float>(eps);
  params.stream = get_stream(q.device());
  params.trigger_completion_at_end = trigger_completion_at_end;
  params.enable_pdl = launch_with_pdl;

  flashinfer::minimax_ar::minimax_reduce_rms_op(params);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_allreduce_rms, minimax_allreduce_rms);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_allreduce_rms_qk, minimax_allreduce_rms_qk);
