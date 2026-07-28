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
 * Launcher declaration for the warp-specialized online-softmax AttnRes v2
 * forward (kernel vendored in attn_res_fwd_online_v2.cu). Candidates are the
 * ``num_blocks`` block-residual snapshots followed by ``layer_residual``
 * (N = num_blocks + 1); the mix output always carries a fused following
 * RMSNorm. Only H = 7168 is instantiated.
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

using bf16_t = __nv_bfloat16;

// block_residual : bf16, addressed as base + token * block_stride_m +
//                  block * block_stride_r (strides in elements), so both
//                  token-major and block-major layouts work.
// layer_residual : bf16 [T*B, H] contiguous. When ``delta`` is non-null the
//                  kernel first folds it in (layer_residual += delta, written
//                  back in bf16) and the updated stream is the prefix
//                  candidate.
// delta          : bf16 [T*B, H] contiguous or nullptr.
// output         : bf16 [T*B, H]; rmsnorm(mix) * output_norm_weight.
// enable_pdl     : launch with programmatic stream serialization (PDL).
void run_attn_res_fwd_online_v2(const bf16_t* block_residual, bf16_t* layer_residual,
                                const bf16_t* delta, const bf16_t* res_weight,
                                const bf16_t* rms_weight,
                                const bf16_t* output_norm_weight, bf16_t* output,
                                int num_blocks, int T, int B, int block_stride_m,
                                int block_stride_r, float rms_eps,
                                float output_norm_eps, int num_sm, bool enable_pdl,
                                cudaStream_t stream);
