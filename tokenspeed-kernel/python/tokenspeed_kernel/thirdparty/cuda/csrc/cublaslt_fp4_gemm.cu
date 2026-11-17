/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// NVFP4 cuBLASLt GEMM, adapted from TensorRT-LLM's CublasLtFP4GemmRunner
// (cpp/tensorrt_llm/thop/cublasFp4ScaledMM.cpp) and the block-scaled path of
// CublasMMWrapper (cpp/tensorrt_llm/common/cublasMMWrapper.cpp). The matmul
// descriptors, scale modes, heuristic preference, and workspace size match
// run_gemm(..., tactic=0), so the same algorithm and bits are produced.

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <tuple>

#include "tvm_ffi_utils.h"

namespace {

// TensorRT-LLM's CUBLAS_WORKSPACE_SIZE.
constexpr size_t kWorkspaceBytes = 33554432;

void check_cublas(cublasStatus_t status, char const* what) {
  TVM_FFI_ICHECK(status == CUBLAS_STATUS_SUCCESS)
      << what << " failed: " << cublasGetStatusString(status) << " (" << int(status) << ")";
}

cublasLtHandle_t lt_handle(int device) {
  static thread_local std::map<int, cublasLtHandle_t> handles;
  auto it = handles.find(device);
  if (it != handles.end()) return it->second;
  cublasLtHandle_t handle = nullptr;
  check_cublas(cublasLtCreate(&handle), "cublasLtCreate");
  handles.emplace(device, handle);
  return handle;
}

// One cuBLASLt problem in TensorRT-LLM's operand order: the weight is the
// first (transposed) operand, so C[m, n] row-major is column-major [n, m].
struct Problem {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatrixLayout_t c = nullptr;
  bool has_algo = false;
  cublasLtMatmulAlgo_t algo{};
};

void set_attr(cublasLtMatmulDesc_t op, cublasLtMatmulDescAttributes_t attr, void const* value,
              size_t size) {
  check_cublas(cublasLtMatmulDescSetAttribute(op, attr, value, size),
               "cublasLtMatmulDescSetAttribute");
}

void set_scale_pointers(cublasLtMatmulDesc_t op, void const* a_scale, void const* b_scale) {
  set_attr(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale));
  set_attr(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale));
}

// createDescriptors(CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, k, k, n, 0) followed by
// setScaleDescriptors for CUDA_R_4F_E2M1 operands.
Problem create_problem(int m, int n, int k, cudaDataType_t out_type) {
  Problem p;
  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;
  check_cublas(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_4F_E2M1, k, n, k), "cublasLtMatrixLayoutCreate");
  check_cublas(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_4F_E2M1, k, m, k), "cublasLtMatrixLayoutCreate");
  check_cublas(cublasLtMatrixLayoutCreate(&p.c, out_type, n, m, n), "cublasLtMatrixLayoutCreate");
  check_cublas(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "cublasLtMatmulDescCreate");
  set_attr(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  set_attr(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
  int8_t fast_accum = 0;
  set_attr(p.op, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum, sizeof(fast_accum));
  cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
  set_attr(p.op, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode));

  cublasLtMatmulMatrixScale_t vec16 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  cublasLtMatmulMatrixScale_t scalar = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
  set_attr(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &vec16, sizeof(vec16));
  set_attr(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &vec16, sizeof(vec16));
  set_attr(p.op, CUBLASLT_MATMUL_DESC_C_SCALE_MODE, &scalar, sizeof(scalar));
  set_attr(p.op, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &scalar, sizeof(scalar));
  set_attr(p.op, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, &scalar, sizeof(scalar));
  void const* null_ptr = nullptr;
  set_attr(p.op, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, &null_ptr, sizeof(null_ptr));
  set_attr(p.op, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &null_ptr, sizeof(null_ptr));
  set_attr(p.op, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, &null_ptr, sizeof(null_ptr));
  return p;
}

// CublasMMWrapper::getTactics: up to 200 heuristics within the workspace and
// any reduction scheme; the runner keeps the successful ones and tactic 0
// takes the first.
void select_algo(cublasLtHandle_t handle, Problem& p) {
  cublasLtMatmulPreference_t preference = nullptr;
  check_cublas(cublasLtMatmulPreferenceCreate(&preference), "cublasLtMatmulPreferenceCreate");
  check_cublas(cublasLtMatmulPreferenceInit(preference), "cublasLtMatmulPreferenceInit");
  uint64_t workspace_bytes = kWorkspaceBytes;
  check_cublas(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                    &workspace_bytes, sizeof(workspace_bytes)),
               "cublasLtMatmulPreferenceSetAttribute");
  uint32_t reduction_mask = CUBLASLT_REDUCTION_SCHEME_MASK;
  check_cublas(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                                    &reduction_mask, sizeof(reduction_mask)),
               "cublasLtMatmulPreferenceSetAttribute");
  constexpr int kMaxHeuristics = 200;
  cublasLtMatmulHeuristicResult_t heuristics[kMaxHeuristics];
  int count = 0;
  cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(handle, p.op, p.a, p.b, p.c, p.c, preference,
                                                         kMaxHeuristics, heuristics, &count);
  cublasLtMatmulPreferenceDestroy(preference);
  check_cublas(status, "cublasLtMatmulAlgoGetHeuristic");
  for (int i = 0; i < count; ++i) {
    if (heuristics[i].state == CUBLAS_STATUS_SUCCESS && heuristics[i].workspaceSize <= kWorkspaceBytes) {
      p.algo = heuristics[i].algo;
      p.has_algo = true;
      return;
    }
  }
}

}  // namespace

// out[m, n] = alpha * (a[m, k] @ b[n, k]^T) with NVFP4 operands and
// 16-element UE4M3 block scales, as run_gemm(a, b, a_sf, b_sf, alpha, 0, 0).
// Only the data pointers of a_sf and b_sf are read, like the TensorRT-LLM op.
void cublaslt_fp4_gemm(TensorView out, TensorView a, TensorView b, TensorView a_sf, TensorView b_sf,
                       TensorView alpha, TensorView beta, TensorView workspace) {
  CHECK_CUDA(a);
  CHECK_CUDA(b);
  CHECK_CUDA(out);
  CHECK_DIM(2, a);
  CHECK_DIM(2, b);
  CHECK_DIM(2, out);
  TVM_FFI_ICHECK(a.dtype() == dl_uint8 && b.dtype() == dl_uint8) << "a and b must be packed FP4 bytes";
  TVM_FFI_ICHECK(alpha.dtype() == dl_float32 && alpha.numel() > 0) << "alpha must be a float32 tensor";
  TVM_FFI_ICHECK(beta.dtype() == dl_float32 && beta.numel() > 0) << "beta must be a float32 tensor";
  TVM_FFI_ICHECK(workspace.dtype() == dl_uint8 && static_cast<size_t>(workspace.numel()) >= kWorkspaceBytes)
      << "workspace must hold " << kWorkspaceBytes << " bytes";
  CHECK_CONTIGUOUS(out);

  int const m = static_cast<int>(a.size(0));
  int const k = static_cast<int>(a.size(1)) * 2;
  int const n = static_cast<int>(b.size(0));
  TVM_FFI_ICHECK_EQ(b.size(1) * 2, k) << "a and b must share K";
  TVM_FFI_ICHECK(out.size(0) == m && out.size(1) == n) << "out must be [m, n]";

  cudaDataType_t out_type = CUDA_R_16BF;
  if (out.dtype() == dl_bfloat16) {
    out_type = CUDA_R_16BF;
  } else if (out.dtype() == dl_float16) {
    out_type = CUDA_R_16F;
  } else if (out.dtype() == dl_float32) {
    out_type = CUDA_R_32F;
  } else {
    TVM_FFI_ICHECK(false) << "out must be float16, bfloat16 or float32";
  }

  int const device = a.device().device_id;
  cublasLtHandle_t handle = lt_handle(device);

  // The runner's per-(m, k, n, device, out dtype) heuristic cache; the
  // descriptors are cached with it and only the scale pointers change per call.
  using Key = std::tuple<int, int, int, int, int>;
  static thread_local std::map<Key, Problem> problems;
  Key key{m, k, n, device, static_cast<int>(out_type)};
  auto it = problems.find(key);
  if (it == problems.end()) {
    Problem p = create_problem(m, n, k, out_type);
    set_scale_pointers(p.op, a_sf.data_ptr(), b_sf.data_ptr());
    select_algo(handle, p);
    it = problems.emplace(key, p).first;
  }
  Problem& p = it->second;
  TVM_FFI_ICHECK(p.has_algo) << "No suitable cuBLASLt algorithm found for block-scaled GEMM (m=" << m
                             << ", n=" << n << ", k=" << k << ")";

  // Weight first: cuBLASLt's A is b with b_sf, its B is a with a_sf.
  set_scale_pointers(p.op, b_sf.data_ptr(), a_sf.data_ptr());
  cudaStream_t stream = get_stream(a.device());
  check_cublas(cublasLtMatmul(handle, p.op, alpha.data_ptr(), b.data_ptr(), p.a, a.data_ptr(), p.b,
                              beta.data_ptr(), out.data_ptr(), p.c, out.data_ptr(), p.c, &p.algo,
                              workspace.data_ptr(), kWorkspaceBytes, stream),
               "cublasLtMatmul");
}
