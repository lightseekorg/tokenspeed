/*
 * Copyright (c) 2023 by FlashInfer team.
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
#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

void verify_chain_greedy(
    TensorView predicts,
    TensorView accept_index,
    TensorView accept_token_num,
    TensorView candidates,
    TensorView target_predict,
    uint64_t batch_size,
    uint64_t num_draft_tokens,
    bool enable_pdl
);

void chain_speculative_sampling_target_only(
    TensorView predicts, TensorView accept_index, TensorView accept_token_num,
    TensorView candidates, TensorView uniform_samples, TensorView uniform_samples_for_final_sampling,
    TensorView target_probs, Optional<TensorView> draft_probs, double threshold_single,
    double threshold_acc, bool deterministic, bool enable_pdl
);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(verify_chain_greedy, verify_chain_greedy);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(chain_speculative_sampling_target_only, chain_speculative_sampling_target_only);
