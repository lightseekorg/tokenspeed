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
 * tvm_ffi binding for the Blackwell (sm_100a/sm_103a) Attention-Residual forward
 * kernel. Wraps the raw ``run_attn_res_fwd_tma`` launcher declared in
 * ``attn_res/attn_res.cuh``. Candidate order is ``block_residual[0..K-1]``
 * followed by ``layer_residual`` (N = K + 1). Caller allocates all outputs.
 */
#include <cstdint>

#include "attn_res/attn_res.cuh"
#include "tvm_ffi_utils.h"


using tvm::ffi::Optional;
// SM count for the persistent-grid launch used by run_attn_res_fwd_tma. Keep
// the host definition beside the TVM FFI binding.
int attn_res_fwd_grid_size(int dev) {
  static int cached_num_sm[64] = {};
  if (dev >= 0 && dev < 64 && cached_num_sm[dev] > 0) {
    return cached_num_sm[dev];
  }
  int n = 0;
  cudaError_t err = cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
  if (err != cudaSuccess || n <= 0) return 0;
  if (dev >= 0 && dev < 64) {
    cached_num_sm[dev] = n;
  }
  return n;
}

// layer_residual : bf16 [T, B, H]  (the current residual stream)
// block_residual : bf16 [K, T, B, H]  (K periodic snapshots; K = N - 1)
// res_weight     : bf16 [H]  (scorer projection weight)
// rms_weight     : bf16 [H]  (RMSNorm weight)
// output         : bf16 [T, B, H]
// rsigma/probs/logits : fp32 [N, T, B]  (kernel scratch/aux outputs)
static void attn_res_fwd_impl(TensorView layer_residual, TensorView block_residual,
                              TensorView res_weight, TensorView rms_weight,
                              const bf16_t* delta_ptr, const bf16_t* out_norm_ptr,
                              TensorView output, TensorView rsigma, TensorView probs,
                              TensorView logits, int num_blocks, double rms_eps,
                              bool use_v2) {
  cudaSetDevice(layer_residual.device().device_id);

  TVM_FFI_ICHECK_EQ(layer_residual.ndim(), 3) << "attn_res_fwd: layer_residual must be [T, B, H]";
  TVM_FFI_ICHECK_EQ(block_residual.ndim(), 4) << "attn_res_fwd: block_residual must be [K, T, B, H]";

  const int T = static_cast<int>(layer_residual.size(0));
  const int B = static_cast<int>(layer_residual.size(1));
  const int H = static_cast<int>(layer_residual.size(2));
  const int N = num_blocks + 1;

  // Packed-forward contract.
  TVM_FFI_ICHECK(num_blocks >= 0 && num_blocks <= block_residual.size(0))
      << "attn_res_fwd: num_blocks=" << num_blocks << " must be in [0, "
      << block_residual.size(0) << "]";

  TVM_FFI_ICHECK_EQ(B, 1) << "attn_res_fwd: only B=1 supported, got " << B;
  TVM_FFI_ICHECK(N >= 1 && N <= 12) << "attn_res_fwd: N=" << N << " must be in [1, 12]";
  TVM_FFI_ICHECK(T >= 1 && T <= 16384) << "attn_res_fwd: T=" << T << " must be in [1, 16384]";
  TVM_FFI_ICHECK(H >= 4096 && H <= 8192 && H % 1024 == 0)
      << "attn_res_fwd: H=" << H << " must be a multiple of 1024 in [4096, 8192]";

  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(layer_residual.dtype()), bfloat16_code)
      << "attn_res_fwd: layer_residual must be bf16";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(block_residual.dtype()), bfloat16_code)
      << "attn_res_fwd: block_residual must be bf16";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(res_weight.dtype()), bfloat16_code)
      << "attn_res_fwd: res_weight must be bf16";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(rms_weight.dtype()), bfloat16_code)
      << "attn_res_fwd: rms_weight must be bf16";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(output.dtype()), bfloat16_code)
      << "attn_res_fwd: output must be bf16";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(rsigma.dtype()), float32_code)
      << "attn_res_fwd: rsigma must be fp32";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(probs.dtype()), float32_code)
      << "attn_res_fwd: probs must be fp32";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(logits.dtype()), float32_code)
      << "attn_res_fwd: logits must be fp32";
  TVM_FFI_ICHECK_EQ(res_weight.numel(), H) << "attn_res_fwd: res_weight must have H elements";
  TVM_FFI_ICHECK_EQ(rms_weight.numel(), H) << "attn_res_fwd: rms_weight must have H elements";

  // The kernel's TMA descriptors ignore strides, so inputs must be packed.
  CHECK_CONTIGUOUS(layer_residual);
  CHECK_CONTIGUOUS(block_residual);
  CHECK_CONTIGUOUS(res_weight);
  CHECK_CONTIGUOUS(rms_weight);

  const bf16_t* block_ptr =
      N > 1 ? reinterpret_cast<const bf16_t*>(block_residual.data_ptr()) : nullptr;

  if (use_v2) {
    TVM_FFI_ICHECK_EQ(H, 7168) << "attn_res_fwd_v2: H must be 7168";
    run_attn_res_fwd_online_v2(
        block_ptr, reinterpret_cast<bf16_t*>(layer_residual.data_ptr()), delta_ptr,
        reinterpret_cast<const bf16_t*>(res_weight.data_ptr()),
        reinterpret_cast<const bf16_t*>(rms_weight.data_ptr()), out_norm_ptr,
        reinterpret_cast<bf16_t*>(output.data_ptr()), N, T, H, T * B * H,
        static_cast<float>(rms_eps),
        attn_res_fwd_grid_size(layer_residual.device().device_id),
        get_stream(layer_residual.device()));
  } else {
    run_attn_res_fwd_tma(block_ptr, reinterpret_cast<const bf16_t*>(layer_residual.data_ptr()),
                       reinterpret_cast<const bf16_t*>(res_weight.data_ptr()),
                       reinterpret_cast<const bf16_t*>(rms_weight.data_ptr()), out_norm_ptr,
                       reinterpret_cast<bf16_t*>(output.data_ptr()),
                       reinterpret_cast<float*>(rsigma.data_ptr()),
                       reinterpret_cast<float*>(probs.data_ptr()),
                       reinterpret_cast<float*>(logits.data_ptr()), N, T, B, H,
                       static_cast<float>(rms_eps), get_stream(layer_residual.device()));
  }
}

void attn_res_fwd(TensorView layer_residual, TensorView block_residual, TensorView res_weight,
                  TensorView rms_weight, TensorView output, TensorView rsigma, TensorView probs,
                  TensorView logits, double rms_eps) {
  attn_res_fwd_impl(layer_residual, block_residual, res_weight, rms_weight, nullptr, nullptr,
                    output, rsigma, probs, logits, block_residual.size(0), rms_eps, false);
}

// Variant with the following RMSNorm fused into the epilogue:
// output = rmsnorm(mix) * out_norm_weight (same eps as the candidate norms).
void attn_res_fwd_out_norm(TensorView layer_residual, TensorView block_residual,
                           TensorView res_weight, TensorView rms_weight,
                           TensorView out_norm_weight, TensorView output, TensorView rsigma,
                           TensorView probs, TensorView logits, double rms_eps) {
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(out_norm_weight.dtype()), bfloat16_code)
      << "attn_res_fwd: out_norm_weight must be bf16";
  TVM_FFI_ICHECK_EQ(out_norm_weight.numel(), layer_residual.size(layer_residual.ndim() - 1))
      << "attn_res_fwd: out_norm_weight must have H elements";
  CHECK_CONTIGUOUS(out_norm_weight);
  attn_res_fwd_impl(layer_residual, block_residual, res_weight, rms_weight, nullptr,
                    reinterpret_cast<const bf16_t*>(out_norm_weight.data_ptr()), output, rsigma,
                    probs, logits, block_residual.size(0), rms_eps, false);
}

static const bf16_t* attn_res_delta_ptr(Optional<TensorView> delta,
                                          TensorView layer_residual) {
  if (!delta.has_value()) return nullptr;
  TensorView value = delta.value();
  TVM_FFI_ICHECK_EQ(value.ndim(), 3) << "attn_res_fwd_v2: delta must be [T, B, H]";
  TVM_FFI_ICHECK_EQ(value.size(0), layer_residual.size(0))
      << "attn_res_fwd_v2: delta T must match layer_residual";
  TVM_FFI_ICHECK_EQ(value.size(1), layer_residual.size(1))
      << "attn_res_fwd_v2: delta B must match layer_residual";
  TVM_FFI_ICHECK_EQ(value.size(2), layer_residual.size(2))
      << "attn_res_fwd_v2: delta H must match layer_residual";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(value.dtype()), bfloat16_code)
      << "attn_res_fwd_v2: delta must be bf16";
  CHECK_CONTIGUOUS(value);
  return reinterpret_cast<const bf16_t*>(value.data_ptr());
}

void attn_res_fwd_v2(TensorView layer_residual, Optional<TensorView> delta,
                     TensorView block_residual, TensorView res_weight,
                     TensorView rms_weight, TensorView output, TensorView rsigma,
                     TensorView probs, TensorView logits, int64_t num_blocks,
                     double rms_eps) {
  attn_res_fwd_impl(layer_residual, block_residual, res_weight, rms_weight,
                    attn_res_delta_ptr(delta, layer_residual), nullptr, output,
                    rsigma, probs, logits, static_cast<int>(num_blocks), rms_eps, true);
}

void attn_res_fwd_v2_out_norm(
    TensorView layer_residual, Optional<TensorView> delta, TensorView block_residual,
    TensorView res_weight, TensorView rms_weight, TensorView out_norm_weight,
    TensorView output, TensorView rsigma, TensorView probs, TensorView logits,
    int64_t num_blocks, double rms_eps) {
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(out_norm_weight.dtype()), bfloat16_code)
      << "attn_res_fwd_v2: out_norm_weight must be bf16";
  TVM_FFI_ICHECK_EQ(out_norm_weight.numel(), layer_residual.size(2))
      << "attn_res_fwd_v2: out_norm_weight must have H elements";
  CHECK_CONTIGUOUS(out_norm_weight);
  attn_res_fwd_impl(
      layer_residual, block_residual, res_weight, rms_weight,
      attn_res_delta_ptr(delta, layer_residual),
      reinterpret_cast<const bf16_t*>(out_norm_weight.data_ptr()), output,
      rsigma, probs, logits, static_cast<int>(num_blocks), rms_eps, true);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(attn_res_fwd, attn_res_fwd);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(attn_res_fwd_out_norm, attn_res_fwd_out_norm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(attn_res_fwd_v2, attn_res_fwd_v2);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(attn_res_fwd_v2_out_norm, attn_res_fwd_v2_out_norm);
