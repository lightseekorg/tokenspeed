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

// The ATen reduction wrapper follows PyTorch v2.9.0 Reduce.cuh; see
// full_vocab_scoring_LICENSE for its BSD license.

#include "full_vocab_scoring_topk.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/cub.cuh>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/cuda/Sort.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

std::vector<at::Tensor> selected_topk(at::Tensor input, at::Tensor active,
                                      int64_t k) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kFloat &&
              input.dim() == 2 && input.is_contiguous());
  TORCH_CHECK(active.is_cuda() && active.scalar_type() == at::kBool &&
              active.is_contiguous() && active.numel() == input.size(0));
  TORCH_CHECK(active.device() == input.device());
  TORCH_CHECK(k > 0 && k <= 128 && input.size(1) >= 1024);
  c10::cuda::CUDAGuard guard(input.device());
  auto values = at::empty({input.size(0), k}, input.options());
  auto indices =
      at::empty({input.size(0), k}, input.options().dtype(at::kLong));
  if (!input.size(0))
    return {values, indices};
  TORCH_CHECK(input.numel() * sizeof(float) < INT32_MAX,
              "SC top-k requires byte offsets below 2^31");
  auto base = at::cuda::detail::getTensorInfo<const float, uint32_t>(input);
  auto out = at::cuda::detail::getTensorInfo<float, uint32_t>(values);
  auto ids = at::cuda::detail::getTensorInfo<int64_t, uint32_t>(indices);
  base.sizes[1] = out.sizes[1] = ids.sizes[1] = 1;
  sc_topk::ActiveInputInfo<float, uint32_t> info(base, active.data_ptr<bool>());
  sc_topk::mbtopk::launch<float, uint32_t, 2>(info, input.size(1), k, true,
                                              input.size(0), 1, out, 1, ids, 1);
  at::native::sortKeyValueInplace(values, indices, 1, true);
  return {values, indices};
}

__global__ void build_head_mask(const int64_t *ids, const bool *active,
                                uint32_t *mask, int64_t words, int k) {
  int64_t row = blockIdx.x;
  if (!active[row])
    return;
  for (int64_t word = threadIdx.x; word < words; word += blockDim.x)
    mask[row * words + word] = 0;
  __syncthreads();
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    int64_t id = ids[row * k + i];
    CUDA_KERNEL_ASSERT(id >= 0 && id < words * 32);
    atomicOr(mask + row * words + (id >> 5), uint32_t(1) << (id & 31));
  }
}

struct MaskMax {
  const uint32_t *mask;
  int64_t words;
  int64_t row = 0;
  __device__ float reduce(float a, float b, int64_t index) const {
    if (mask &&
        (mask[row * words + (index >> 5)] & (uint32_t(1) << (index & 31))))
      b = -INFINITY;
    return combine(a, b);
  }
  __device__ float combine(float a, float b) const {
    return (isnan(a) || a > b) ? a : b;
  }
  __device__ float project(float a) const { return a; }
  __device__ float inactive_value(int64_t) const { return 0.0f; }
  __device__ float warp_shfl_down(float a, int offset) const {
    return WARP_SHFL_DOWN(a, offset);
  }
  __device__ float translate_idx(float a, int64_t) const { return a; }
};
struct MaskExpSum {
  const uint32_t *mask;
  const float *maximum;
  const float *log_z;
  const float *row_maximum;
  int64_t words;
  int64_t row = 0;
  __device__ float reduce(float a, float b, int64_t index) const {
    if (mask[row * words + (index >> 5)] & (uint32_t(1) << (index & 31)))
      b = -INFINITY;
    float m = maximum[row];
    if (isinf(m))
      m = 0.0f;
    return a + expf(__fsub_rn(b, m));
  }
  __device__ float combine(float a, float b) const { return a + b; }
  __device__ float project(float a) const {
    float m = maximum[row];
    if (isinf(m))
      m = 0.0f;
    // ATen projects only the final reduction result. Keep its CUDA logf and
    // the two separately rounded operations from log().add_().sub_().
    float normalizer = __fadd_rn(row_maximum[row], log_z[row]);
    return __fsub_rn(__fadd_rn(logf(a), m), normalizer);
  }
  __device__ float inactive_value(int64_t index) const {
    return __fsub_rn(-INFINITY, __fadd_rn(row_maximum[index], log_z[index]));
  }
  __device__ float warp_shfl_down(float a, int offset) const {
    return WARP_SHFL_DOWN(a, offset);
  }
  __device__ float translate_idx(float a, int64_t) const { return a; }
};

template <typename R>
__global__ void selected_reduce(R reduction, const bool *active) {
  // The unchanged ATen config maps one output row to each blockIdx.x.
  int64_t row = blockIdx.x;
  if (!active[row]) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.y == 0)
      reinterpret_cast<float *>(const_cast<char *>(reduction.dst[0]))[row] =
          reduction.ops.inactive_value(row);
    return;
  }
  reduction.ops.row = row;
  reduction.template run<1>();
}

template <typename Ops, typename Identity>
at::Tensor reduce_rows(at::Tensor input, at::Tensor active, Ops ops,
                       Identity ident) {
  auto output = at::empty({input.size(0)}, input.options());
  auto iter = at::native::make_reduction("selected_sc_reduce", output, input,
                                         {1}, false, at::kFloat);
  TORCH_CHECK(iter.can_use_32bit_indexing(),
              "SC reduction currently requires a 32-bit ATen iterator");
  using arg_t = Identity;
  auto config = at::native::setReduceConfig<arg_t, float, 4, 4>(iter);
  TORCH_CHECK(config.output_vec_size == 1 && config.step_output == 1 &&
                  config.output_mult[0] == 0 && config.output_mult[1] == 0,
              "SC reduction requires one row per CTA group");
  at::DataPtr buffer, semaphores;
  if (config.should_global_reduce()) {
    auto &alloc = *c10::cuda::CUDACachingAllocator::get();
    buffer = alloc.allocate(config.global_memory_size());
    semaphores = alloc.allocate(config.semaphore_size());
    AT_CUDA_CHECK(cudaMemsetAsync(semaphores.get(), 0, config.semaphore_size(),
                                  at::cuda::getCurrentCUDAStream()));
  }
  auto output_calc = at::native::make_output_calculator<uint32_t>(iter);
  auto input_calc = at::native::make_input_calculator<uint32_t>(iter);
  using R = at::native::ReduceOp<float, Ops, uint32_t, float, 4, 4>;
  R reduction(ops, config, input_calc, output_calc,
              (const char *)iter.data_ptr(1), (char *)iter.data_ptr(0),
              std::nullopt, nullptr, buffer.get(), (int *)semaphores.get(),
              ident, 1, 0);
  reduction.accumulate = iter.should_accumulate();
  reduction.final_output = iter.is_final_output();
  selected_reduce<<<config.grid(), config.block(), config.shared_memory_size(),
                    at::cuda::getCurrentCUDAStream()>>>(
      reduction, active.data_ptr<bool>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

struct FirstMaximum {
  using Pair = thrust::pair<float, int64_t>;
  const float *candidate;
  int64_t candidate_stride;
  int64_t row = 0;
  __device__ Pair combine(Pair a, Pair b) const {
    bool an = isnan(a.first), bn = isnan(b.first);
    if (an || bn)
      return an && (!bn || a.second < b.second) ? a : b;
    if (a.first == b.first)
      return a.second < b.second ? a : b;
    return a.first > b.first ? a : b;
  }
  __device__ Pair reduce(Pair a, float b, int64_t index) const {
    return combine(a, Pair(b, index));
  }
  __device__ float project(Pair a) const { return a.first; }
  __device__ float inactive_value(int64_t index) const {
    return candidate[index * candidate_stride];
  }
  __device__ Pair warp_shfl_down(Pair a, int offset) const {
    return Pair(WARP_SHFL_DOWN(a.first, offset),
                WARP_SHFL_DOWN(a.second, offset));
  }
  __device__ Pair translate_idx(Pair a, int64_t) const { return a; }
};

__global__ void maximum_fallback_flags(const float *values, bool *fallback,
                                       int64_t rows, int64_t stride) {
  int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < rows) {
    float candidate = values[row * stride];
    fallback[row] = candidate == 0.0f || !isfinite(candidate);
  }
}

at::Tensor maximum_from_topk(at::Tensor logits, at::Tensor values) {
  c10::cuda::CUDAGuard guard(logits.device());
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kFloat &&
              logits.dim() == 2 && logits.is_contiguous());
  TORCH_CHECK(values.device() == logits.device() &&
              values.scalar_type() == at::kFloat && values.dim() == 2 &&
              values.size(0) == logits.size(0) && values.size(1) > 0);
  auto fallback =
      at::empty({logits.size(0)}, logits.options().dtype(at::kBool));
  maximum_fallback_flags<<<(logits.size(0) + 127) / 128, 128, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
      values.data_ptr<float>(), fallback.data_ptr<bool>(), logits.size(0),
      values.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // Equal zero and non-finite values retain torch.max's first-index semantics.
  // Other rows store Top-K's already selected maximum without reading logits.
  return reduce_rows(
      logits, fallback,
      FirstMaximum{values.data_ptr<float>(), values.stride(0), 0},
      FirstMaximum::Pair(-INFINITY, INT64_MAX));
}

at::Tensor selected_tail(at::Tensor shifted, at::Tensor ids, at::Tensor log_z,
                         at::Tensor row_maximum, at::Tensor active) {
  c10::cuda::CUDAGuard guard(shifted.device());
  TORCH_CHECK(shifted.is_cuda() && shifted.scalar_type() == at::kFloat &&
              shifted.dim() == 2 && shifted.is_contiguous());
  TORCH_CHECK(ids.scalar_type() == at::kLong && ids.is_contiguous() &&
              ids.size(0) == shifted.size(0));
  TORCH_CHECK(log_z.scalar_type() == at::kFloat &&
              log_z.numel() == shifted.size(0));
  TORCH_CHECK(row_maximum.scalar_type() == at::kFloat && row_maximum.is_contiguous() &&
              row_maximum.numel() == shifted.size(0));
  TORCH_CHECK(active.scalar_type() == at::kBool && active.is_contiguous() &&
              active.numel() == shifted.size(0));
  TORCH_CHECK(ids.device() == shifted.device() &&
              log_z.device() == shifted.device() &&
              row_maximum.device() == shifted.device() &&
              active.device() == shifted.device());
  int64_t words = (shifted.size(1) + 31) / 32;
  auto mask =
      at::empty({shifted.size(0), words}, ids.options().dtype(at::kInt));
  build_head_mask<<<shifted.size(0), 128, 0,
                    at::cuda::getCurrentCUDAStream()>>>(
      ids.data_ptr<int64_t>(), active.data_ptr<bool>(),
      reinterpret_cast<uint32_t *>(mask.data_ptr<int>()), words, ids.size(1));
  auto maximum = reduce_rows(
      shifted, active,
      MaskMax{reinterpret_cast<const uint32_t *>(mask.data_ptr<int>()), words,
              0},
      -INFINITY);
  return reduce_rows(
      shifted, active,
      MaskExpSum{reinterpret_cast<const uint32_t *>(mask.data_ptr<int>()),
                 maximum.data_ptr<float>(), log_z.data_ptr<float>(),
                 row_maximum.data_ptr<float>(), words, 0},
      0.0f);
}

TORCH_LIBRARY(tokenspeed_full_vocab, m) {
  m.def("topk(Tensor logits, Tensor active, int k) -> Tensor[]",
        &selected_topk);
  m.def("maximum(Tensor logits, Tensor values) -> Tensor", &maximum_from_topk);
  m.def(
      "tail(Tensor shifted, Tensor ids, Tensor log_z, Tensor row_maximum, Tensor active) -> Tensor",
      &selected_tail);
}
