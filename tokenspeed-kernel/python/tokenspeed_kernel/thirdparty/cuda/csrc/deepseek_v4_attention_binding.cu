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

#include "tvm_ffi_utils.h"

using tvm::ffi::TensorView;

void fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    TensorView q,
    TensorView kv,
    TensorView k_cache,
    TensorView slot_mapping,
    TensorView positions,
    TensorView cos_sin_cache,
    double rms_norm_eps,
    int64_t cache_block_size);

void deepseek_v4_indexer_topk_prefill(TensorView logits,
                                      TensorView row_starts,
                                      TensorView row_ends,
                                      TensorView output,
                                      int64_t k);

void deepseek_v4_gather_paged_indexer_mxfp4_cache(TensorView kv_cache,
                                                  TensorView values_out,
                                                  TensorView scales_out,
                                                  TensorView block_table,
                                                  TensorView cu_seq_lens,
                                                  int64_t cache_block_size);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert,
                              fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(deepseek_v4_indexer_topk_prefill,
                              deepseek_v4_indexer_topk_prefill);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(deepseek_v4_gather_paged_indexer_mxfp4_cache,
                              deepseek_v4_gather_paged_indexer_mxfp4_cache);
