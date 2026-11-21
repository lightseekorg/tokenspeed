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

// The TRT-LLM-gen block-scaled batched expert GEMM on its own: NVFP4 or MXFP8
// weights and activations, BF16 output, no fused activation. Rows arrive
// grouped by expert and padded to the token tile, described by the
// dynamic-batch metadata the MoE routing writes; NVFP4 rows take an FP32
// per-row output scale. Configs that split K are never chosen, so every
// element sums its K blocks in one order whatever the tile.
#include <map>
#include <memory>
#include <tuple>
#include <vector>

#include <tvm/ffi/extra/cuda/device_guard.h>

#include "flashinfer/trtllm/batched_gemm/KernelRunner.h"
#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/BatchedGemmInterface.h"
#include "flashinfer/trtllm/fused_moe/runner.h"
#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

namespace {

namespace btg = batchedGemm::trtllm::gen;
using tensorrt_llm::kernels::TrtllmGenBatchedGemmRunner;
using tensorrt_llm::kernels::TrtllmGenBatchedGemmRunnerOptions;

TrtllmGenBatchedGemmRunner& runner(bool nvfp4, int64_t tile) {
  static std::map<std::tuple<bool, int64_t>, std::unique_ptr<TrtllmGenBatchedGemmRunner>> cache;
  auto& slot = cache[{nvfp4, tile}];
  if (!slot) {
    btg::Dtype const dtype = nvfp4 ? btg::Dtype::E2m1 : btg::Dtype::MxE4m3;
    // Weights are the MMA's A side (transposed output), as in the MoE GEMMs.
    TrtllmGenBatchedGemmRunnerOptions const options{
        .dtypeA = dtype,
        .dtypeB = dtype,
        .dtypeC = btg::Dtype::Bfloat16,
        .eltwiseActType = tensorrt_llm::kernels::EltwiseActType::None,
        .fusedAct = false,
        .routeAct = false,
        .staticBatch = false,
        .transposeMmaOutput = true,
        .tileSize = static_cast<int32_t>(tile),
        .epilogueTileM = 128,
        .useShuffledMatrix = true,
        .weightLayout = batchedGemm::gemm::MatrixLayout::MajorK,
        .usePerTokenScaling = nvfp4,
        .perTokenSfDtype = nvfp4 ? btg::Dtype::Fp32 : btg::Dtype::Void};
    slot = std::make_unique<TrtllmGenBatchedGemmRunner>(options);
  }
  return *slot;
}

int32_t max_ctas(int64_t tokens, int64_t top_k, int64_t experts, int64_t tile) {
  return tensorrt_llm::kernels::trtllmgen_moe::Routing::getMaxNumCtasInBatchDim(
      static_cast<int32_t>(tokens), static_cast<int32_t>(top_k), static_cast<int32_t>(experts),
      static_cast<int32_t>(tile));
}

}  // namespace

// The runner's preferred valid config for [tokens * top_k routed rows] x
// [experts, n, k] weights that does not split K and reads the scale layouts
// the callers write: activations in 8x4 or 128x4 tiles, weights in 128x4
// tiles (block_scale_interleave).
int64_t block_scaled_bmm_config(bool nvfp4, int64_t tile, int64_t n, int64_t k, int64_t experts,
                                int64_t tokens, int64_t top_k) {
  using btg::SfLayout;
  auto const& configs = batchedGemm::batchedGemm::BatchedGemmInterface().getBatchedGemmConfigs();
  for (int64_t config :
       runner(nvfp4, tile).getValidConfigIndices(tokens, n, k, {}, tokens, experts,
                                                 max_ctas(tokens, top_k, experts, tile))) {
    auto const& options = configs[config].mOptions;
    if (options.mNumSlicesForSplitK == 1 && options.mSfLayoutA == SfLayout::R128c4 &&
        (options.mSfLayoutB == SfLayout::R8c4 || options.mSfLayoutB == SfLayout::R128c4)) {
      return config;
    }
  }
  TVM_FFI_ICHECK(false) << "no TRT-LLM-gen batched GEMM config without split-K and with "
                           "8x4 or 128x4 activation and 128x4 weight scales";
  return -1;
}

// The activation scale-factor layout of a config: 1 8x4, 3 128x4.
int64_t block_scaled_bmm_sf_layout(bool nvfp4, int64_t tile, int64_t config) {
  return static_cast<int64_t>(runner(nvfp4, tile).getSfLayoutB(static_cast<int32_t>(config)));
}

int64_t block_scaled_bmm_workspace(bool nvfp4, int64_t tile, int64_t n, int64_t k,
                                   int64_t experts, int64_t tokens, int64_t top_k,
                                   int64_t config) {
  return static_cast<int64_t>(runner(nvfp4, tile).getWorkspaceSizeInBytes(
      tokens, n, k, {}, tokens, experts, max_ctas(tokens, top_k, experts, tile),
      static_cast<int32_t>(config)));
}

void block_scaled_bmm(TensorView x, TensorView x_scale, TensorView w, TensorView w_scale,
                      Optional<TensorView> row_scale, TensorView ones, TensorView out,
                      TensorView num_ctas, TensorView num_padded, TensorView cta_expert,
                      TensorView cta_limit, TensorView workspace, bool nvfp4, int64_t tile,
                      int64_t tokens, int64_t top_k, int64_t config, bool enable_pdl) {
  int64_t const experts = w.size(0), n = w.size(1);
  int64_t const k = nvfp4 ? 2 * w.size(2) : w.size(2);
  TVM_FFI_ICHECK_EQ(w.ndim(), 3);
  TVM_FFI_ICHECK_EQ(out.ndim(), 2);
  TVM_FFI_ICHECK_EQ(out.size(1), n);
  TVM_FFI_ICHECK_EQ(out.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(x.size(0), out.size(0));
  TVM_FFI_ICHECK_EQ(ones.numel(), experts);
  TVM_FFI_ICHECK_EQ(ones.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(row_scale.has_value(), nvfp4) << "NVFP4 rows need, MXFP8 rows refuse, a row scale";
  for (TensorView metadata : {num_ctas, num_padded, cta_expert, cta_limit})
    TVM_FFI_ICHECK_EQ(metadata.dtype(), dl_int32);
  ffi::CUDADeviceGuard guard(x.device().device_id);
  runner(nvfp4, tile).run(
      tokens, n, k, {}, tokens, experts, max_ctas(tokens, top_k, experts, tile), x.data_ptr(),
      x_scale.data_ptr(), w.data_ptr(), w_scale.data_ptr(),
      /* perTokensSfA */ row_scale.has_value() ? row_scale.value().data_ptr() : nullptr,
      /* perTokensSfB */ nullptr, static_cast<float const*>(ones.data_ptr()),
      /* scaleGateC */ nullptr, /* bias */ nullptr, /* alpha */ nullptr, /* beta */ nullptr,
      /* clampLimit */ nullptr, out.data_ptr(), /* outSfC */ nullptr, /* routeMap */ nullptr,
      static_cast<int32_t const*>(num_padded.data_ptr()),
      static_cast<int32_t const*>(cta_expert.data_ptr()),
      static_cast<int32_t const*>(cta_limit.data_ptr()),
      static_cast<int32_t const*>(num_ctas.data_ptr()), /* permutedIdxToBiasRowIdx */ nullptr,
      workspace.data_ptr(), get_stream(x.device()), x.device().device_id,
      static_cast<int32_t>(config), enable_pdl);
}

// getCubin for the runner; the loader registers its callback (setup_cubin_loader).
namespace flashinfer {
namespace trtllm_cubin_loader {
#include <flashinfer/cubin_loader.h>
}
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(block_scaled_bmm_config, block_scaled_bmm_config);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(block_scaled_bmm_sf_layout, block_scaled_bmm_sf_layout);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(block_scaled_bmm_workspace, block_scaled_bmm_workspace);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(block_scaled_bmm, block_scaled_bmm);
