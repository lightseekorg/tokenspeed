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
 * tvm_ffi binding for the warp-specialized online-softmax AttnRes v2 forward
 * (SM100, H = 7168). Wraps ``run_attn_res_fwd_online_v2`` declared in
 * ``attn_res/attn_res_v2.cuh``. Candidate order is ``blocks[0..KB-1]`` followed
 * by ``prefix`` (N = KB + 1); the following RMSNorm is always fused into the
 * epilogue. The ``_delta`` variant additionally folds ``prefix += delta`` in
 * (written back to ``prefix`` in bf16) before scoring, so the updated stream is
 * the prefix candidate.
 */
#include <cstdint>

#include "attn_res/attn_res.cuh"
#include "attn_res/attn_res_v2.cuh"
#include "tvm_ffi_utils.h"

// prefix          : bf16 [T, H] residual stream (updated in place with delta)
// delta           : bf16 [T, H] or nullptr
// blocks          : bf16 [KB, T, H] snapshots; leading strides arbitrary
//                   (16-byte aligned), last dim contiguous
// res/rms weights : bf16 [H] scorer projection / candidate RMSNorm weights
// out_norm_weight : bf16 [H] fused following-RMSNorm weight
// output          : bf16 [T, H]
static void attn_res_fwd_v2_impl(TensorView prefix, const bf16_t* delta_ptr,
                                 TensorView blocks, TensorView res_weight,
                                 TensorView rms_weight, TensorView out_norm_weight,
                                 TensorView output, double rms_eps,
                                 double out_norm_eps, bool enable_pdl) {
  cudaSetDevice(prefix.device().device_id);

  TVM_FFI_ICHECK_EQ(prefix.ndim(), 2) << "attn_res_fwd_v2: prefix must be [T, H]";
  TVM_FFI_ICHECK_EQ(blocks.ndim(), 3) << "attn_res_fwd_v2: blocks must be [KB, T, H]";

  const int T = static_cast<int>(prefix.size(0));
  const int H = static_cast<int>(prefix.size(1));
  const int KB = static_cast<int>(blocks.size(0));

  // Only the Kimi-K3 shape is instantiated; the caller falls back elsewhere.
  TVM_FFI_ICHECK_EQ(H, 7168) << "attn_res_fwd_v2: only H=7168 is instantiated, got " << H;
  TVM_FFI_ICHECK(KB >= 1 && KB <= 8) << "attn_res_fwd_v2: KB=" << KB << " must be in [1, 8]";
  TVM_FFI_ICHECK(T >= 1 && T <= 16384) << "attn_res_fwd_v2: T=" << T << " must be in [1, 16384]";
  TVM_FFI_ICHECK_EQ(blocks.size(1), T) << "attn_res_fwd_v2: blocks/prefix token mismatch";
  TVM_FFI_ICHECK_EQ(blocks.size(2), H) << "attn_res_fwd_v2: blocks/prefix hidden mismatch";
  TVM_FFI_ICHECK_EQ(output.numel(), prefix.numel())
      << "attn_res_fwd_v2: output must be [T, H]";

  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(prefix.dtype()), bfloat16_code)
      << "attn_res_fwd_v2: prefix must be bf16";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(blocks.dtype()), bfloat16_code)
      << "attn_res_fwd_v2: blocks must be bf16";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(res_weight.dtype()), bfloat16_code)
      << "attn_res_fwd_v2: res_weight must be bf16";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(rms_weight.dtype()), bfloat16_code)
      << "attn_res_fwd_v2: rms_weight must be bf16";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(out_norm_weight.dtype()), bfloat16_code)
      << "attn_res_fwd_v2: out_norm_weight must be bf16";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(output.dtype()), bfloat16_code)
      << "attn_res_fwd_v2: output must be bf16";
  TVM_FFI_ICHECK_EQ(res_weight.numel(), H) << "attn_res_fwd_v2: res_weight must have H elements";
  TVM_FFI_ICHECK_EQ(rms_weight.numel(), H) << "attn_res_fwd_v2: rms_weight must have H elements";
  TVM_FFI_ICHECK_EQ(out_norm_weight.numel(), H)
      << "attn_res_fwd_v2: out_norm_weight must have H elements";

  CHECK_CONTIGUOUS(prefix);
  CHECK_CONTIGUOUS(res_weight);
  CHECK_CONTIGUOUS(rms_weight);
  CHECK_CONTIGUOUS(out_norm_weight);
  CHECK_CONTIGUOUS(output);

  // Blocks may be a leading-dim slice of a larger buffer; only the rows must
  // be dense and 16-byte aligned (cp.async.bulk source alignment).
  const int64_t block_stride_r = blocks.stride(0);
  const int64_t block_stride_m = blocks.stride(1);
  TVM_FFI_ICHECK_EQ(blocks.stride(2), 1) << "attn_res_fwd_v2: blocks rows must be dense";
  TVM_FFI_ICHECK(block_stride_m % 8 == 0 && block_stride_r % 8 == 0)
      << "attn_res_fwd_v2: blocks strides must be 16-byte aligned";
  TVM_FFI_ICHECK(block_stride_m <= INT32_MAX && block_stride_r <= INT32_MAX)
      << "attn_res_fwd_v2: blocks strides exceed int32";

  const int num_sm = attn_res_fwd_grid_size(prefix.device().device_id);
  TVM_FFI_ICHECK_GT(num_sm, 0) << "attn_res_fwd_v2: could not query SM count";

  run_attn_res_fwd_online_v2(
      reinterpret_cast<const bf16_t*>(blocks.data_ptr()),
      reinterpret_cast<bf16_t*>(prefix.data_ptr()), delta_ptr,
      reinterpret_cast<const bf16_t*>(res_weight.data_ptr()),
      reinterpret_cast<const bf16_t*>(rms_weight.data_ptr()),
      reinterpret_cast<const bf16_t*>(out_norm_weight.data_ptr()),
      reinterpret_cast<bf16_t*>(output.data_ptr()), KB, T, /*B=*/1,
      static_cast<int>(block_stride_m), static_cast<int>(block_stride_r),
      static_cast<float>(rms_eps), static_cast<float>(out_norm_eps), num_sm,
      enable_pdl, get_stream(prefix.device()));

  cudaError_t error = cudaGetLastError();
  TVM_FFI_ICHECK(error == cudaSuccess)
      << "attn_res_fwd_v2 launch failed: " << cudaGetErrorString(error);
}

void attn_res_fwd_v2(TensorView prefix, TensorView blocks, TensorView res_weight,
                     TensorView rms_weight, TensorView out_norm_weight, TensorView output,
                     double rms_eps, double out_norm_eps, bool enable_pdl) {
  attn_res_fwd_v2_impl(prefix, nullptr, blocks, res_weight, rms_weight, out_norm_weight,
                       output, rms_eps, out_norm_eps, enable_pdl);
}

// Variant with the fused residual accumulate: prefix += delta (bf16, written
// back), then the updated prefix is the last mix candidate.
void attn_res_fwd_v2_delta(TensorView prefix, TensorView delta, TensorView blocks,
                           TensorView res_weight, TensorView rms_weight,
                           TensorView out_norm_weight, TensorView output, double rms_eps,
                           double out_norm_eps, bool enable_pdl) {
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(delta.dtype()), bfloat16_code)
      << "attn_res_fwd_v2: delta must be bf16";
  TVM_FFI_ICHECK_EQ(delta.numel(), prefix.numel()) << "attn_res_fwd_v2: delta must be [T, H]";
  CHECK_CONTIGUOUS(delta);
  attn_res_fwd_v2_impl(prefix, reinterpret_cast<const bf16_t*>(delta.data_ptr()), blocks,
                       res_weight, rms_weight, out_norm_weight, output, rms_eps,
                       out_norm_eps, enable_pdl);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(attn_res_fwd_v2, attn_res_fwd_v2);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(attn_res_fwd_v2_delta, attn_res_fwd_v2_delta);
