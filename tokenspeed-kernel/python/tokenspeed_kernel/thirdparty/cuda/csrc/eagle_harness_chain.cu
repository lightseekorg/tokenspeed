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

// EAGLE-4 chained rounds: each depth's graph tail-launches the next or the finalize; no host wait.

#include <cstdint>

#include "tvm_ffi_utils.h"

namespace {

// execs: [d] depth d's exec (0: uncaptured), [last_depth + 1 + sampled] finalize; go[d]: d drafts.
__global__ void chain_kernel(const int64_t* __restrict__ execs, const int32_t* __restrict__ go,
                             const bool* __restrict__ sampled, int depth, int last_depth) {
  const bool deeper = depth < last_depth && go[depth + 1] != 0;
  const int64_t next = execs[deeper ? depth + 1 : last_depth + 1 + (*sampled ? 1 : 0)];
  // An uncaptured graph would leave the round's tree unwritten: trap instead.
  if (next == 0 ||
      cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(next), cudaStreamGraphTailLaunch) !=
          cudaSuccess) {
    __trap();
  }
}

void check_cuda(cudaError_t status, const char* what) {
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "eagle_harness_chain: " << what << ": " << cudaGetErrorString(status);
}

}  // namespace

int64_t eagle_harness_chain_instantiate(int64_t graph, int64_t stream) {
  cudaGraphExec_t exec = nullptr;
  check_cuda(cudaGraphInstantiateWithFlags(&exec, reinterpret_cast<cudaGraph_t>(graph),
                                           cudaGraphInstantiateFlagDeviceLaunch),
             "instantiate for device launch");
  check_cuda(cudaGraphUpload(exec, reinterpret_cast<cudaStream_t>(stream)), "upload");
  return reinterpret_cast<int64_t>(exec);
}

void eagle_harness_chain_destroy(int64_t exec) {
  check_cuda(cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(exec)), "destroy");
}

void eagle_harness_chain_next(TensorView execs, TensorView go, TensorView sampled, int64_t depth,
                              int64_t last_depth) {
  TVM_FFI_ICHECK(execs.dtype() == dl_int64 && execs.ndim() == 1 && execs.stride(0) == 1 &&
                 execs.size(0) == last_depth + 3)
      << "eagle_harness_chain: exec table";
  TVM_FFI_ICHECK(go.dtype() == dl_int32 && go.ndim() == 1 && go.stride(0) == 1 &&
                 go.size(0) > last_depth)
      << "eagle_harness_chain: go flags";
  TVM_FFI_ICHECK(sampled.dtype() == dl_bool && sampled.numel() >= 1)
      << "eagle_harness_chain: sampled flag";
  TVM_FFI_ICHECK(depth >= 1 && depth <= last_depth) << "eagle_harness_chain: depth";
  chain_kernel<<<1, 1, 0, get_stream(execs.device())>>>(
      static_cast<const int64_t*>(execs.data_ptr()), static_cast<const int32_t*>(go.data_ptr()),
      static_cast<const bool*>(sampled.data_ptr()), static_cast<int>(depth),
      static_cast<int>(last_depth));
  check_cuda(cudaGetLastError(), "chain kernel launch");
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_chain_instantiate, eagle_harness_chain_instantiate);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_chain_destroy, eagle_harness_chain_destroy);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_chain_next, eagle_harness_chain_next);
