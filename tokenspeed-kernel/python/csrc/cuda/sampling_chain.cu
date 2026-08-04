/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <flashinfer/sampling.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;
using tvm::ffi::Optional;

void verify_chain_greedy(
    TensorView predicts,
    TensorView accept_index,
    TensorView accept_token_num,
    TensorView candidates,
    TensorView target_predict,
    uint64_t batch_size,
    uint64_t num_draft_tokens,
    bool enable_pdl)
{
  int bs = static_cast<int>(batch_size);
  int draft_tokens = static_cast<int>(num_draft_tokens);
  TVM_FFI_ICHECK_EQ(bs, accept_index.size(0));
  TVM_FFI_ICHECK_EQ(draft_tokens, accept_index.size(1));
  TVM_FFI_ICHECK_EQ(bs, accept_token_num.size(0));
  TVM_FFI_ICHECK_EQ(bs, candidates.size(0));
  TVM_FFI_ICHECK_EQ(draft_tokens, candidates.size(1));
  TVM_FFI_ICHECK_EQ(bs, target_predict.size(0));
  TVM_FFI_ICHECK_EQ(predicts.dtype(), dl_int32);
  TVM_FFI_ICHECK_EQ(accept_index.dtype(), dl_int32);
  TVM_FFI_ICHECK_EQ(accept_token_num.dtype(), dl_int32);
  TVM_FFI_ICHECK_EQ(candidates.dtype(), dl_int32);
  TVM_FFI_ICHECK_EQ(target_predict.dtype(), dl_int64);

  auto stream = get_stream(predicts.device());
  cudaError_t status = sampling::VerifyChainGreedy(
    static_cast<int32_t*>(predicts.data_ptr()),
    static_cast<int32_t*>(accept_index.data_ptr()),
    static_cast<int32_t*>(accept_token_num.data_ptr()),
    static_cast<int32_t*>(candidates.data_ptr()),
    static_cast<int64_t*>(target_predict.data_ptr()),
    bs,
    draft_tokens,
    enable_pdl,
    stream
  );
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "VerifyChainGreedy failed with error code " << cudaGetErrorString(status);
}

void chain_speculative_sampling_target_only(
    TensorView predicts,
    TensorView accept_index,
    TensorView accept_token_num,
    TensorView candidates,
    TensorView uniform_samples,
    TensorView uniform_samples_for_final_sampling,
    TensorView target_probs,
    Optional<TensorView> draft_probs,
    double threshold_single,
    double threshold_acc,
    bool deterministic,
    bool enable_pdl
) {
  CHECK_INPUT(candidates);
  CHECK_INPUT(uniform_samples);
  CHECK_INPUT(uniform_samples_for_final_sampling);
  CHECK_INPUT(target_probs);
  auto device = target_probs.device();
  CHECK_DIM(1, predicts);
  CHECK_DIM(2, accept_index);
  CHECK_DIM(1, accept_token_num);
  CHECK_DIM(2, candidates);
  CHECK_DIM(2, uniform_samples);
  CHECK_DIM(3, target_probs);
  unsigned int batch_size = uniform_samples.size(0);
  unsigned int num_draft_tokens = candidates.size(1);
  unsigned int vocab_size = target_probs.size(2);
  TVM_FFI_ICHECK_EQ(batch_size, candidates.size(0));
  TVM_FFI_ICHECK_EQ(batch_size, target_probs.size(0));
  TVM_FFI_ICHECK_EQ(num_draft_tokens, uniform_samples.size(1));
  TVM_FFI_ICHECK_EQ(num_draft_tokens, target_probs.size(1));
  TVM_FFI_ICHECK_EQ(vocab_size, target_probs.size(2));
  TVM_FFI_ICHECK_EQ(batch_size, accept_index.size(0));
  TVM_FFI_ICHECK_EQ(batch_size, accept_token_num.size(0));
  TVM_FFI_ICHECK_GE(threshold_single, 0);
  TVM_FFI_ICHECK_GE(1, threshold_single);
  TVM_FFI_ICHECK_GE(threshold_acc, 0);
  TVM_FFI_ICHECK_GE(1, threshold_acc);

  float* draft_probs_ptr = nullptr;
  if (draft_probs.has_value()) {
    CHECK_INPUT(draft_probs.value());
    CHECK_DIM(3, draft_probs.value());
    TVM_FFI_ICHECK_EQ(batch_size, draft_probs.value().size(0));
    TVM_FFI_ICHECK_EQ(num_draft_tokens, draft_probs.value().size(1));
    TVM_FFI_ICHECK_EQ(vocab_size, draft_probs.value().size(2));
    draft_probs_ptr = static_cast<float*>(draft_probs.value().data_ptr());
  }

  auto stream = get_stream(target_probs.device());
  cudaError_t status = sampling::ChainSpeculativeSamplingTargetOnly<float, int32_t>(
      static_cast<int32_t*>(predicts.data_ptr()),
      static_cast<int32_t*>(accept_index.data_ptr()),
      static_cast<int32_t*>(accept_token_num.data_ptr()),
      static_cast<int32_t*>(candidates.data_ptr()),
      static_cast<float*>(uniform_samples.data_ptr()),
      static_cast<float*>(uniform_samples_for_final_sampling.data_ptr()),
      static_cast<float*>(target_probs.data_ptr()),
      draft_probs_ptr,
      static_cast<uint32_t>(batch_size),
      static_cast<uint32_t>(num_draft_tokens),
      static_cast<uint32_t>(vocab_size),
      static_cast<float>(threshold_single),
      static_cast<float>(threshold_acc),
      deterministic,
      enable_pdl,
      stream);

    TVM_FFI_ICHECK(status == cudaSuccess)
      << "ChainSpeculativeSamplingTargetOnly failed with error code " << cudaGetErrorString(status);
}
