/*
 * Copyright (c) 2026 LightSeek Foundation
 *
 * Vendored from flashinfer/sampling.cuh (Apache-2.0), with the kernel
 * template split on input vs output dtype so bf16/fp16 input folds the
 * upcast into the kernel's load. Output is always fp32.
 */

#include <flashinfer/allocator.h>
#include <flashinfer/math.cuh>
#include <flashinfer/sampling.cuh>
#include <flashinfer/utils.cuh>
#include <flashinfer/vec_dtypes.cuh>

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

namespace tokenspeed {

using flashinfer::AlignedAllocator;
using flashinfer::ceil_div;
using flashinfer::round_up;
using flashinfer::vec_t;
using flashinfer::sampling::Float2SoftmaxReduceOp;
using flashinfer::sampling::OnlineSoftmaxTempStorage;
using flashinfer::sampling::PartialSoftmaxResult;

template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, typename DTypeIn,
          typename DTypeOut, bool CACHE_INPUT>
__global__ void OnlineSoftmaxFusedKernel(DTypeIn* logits, DTypeOut* output,
                                         float* temperature_arr, float temperature_val,
                                         uint32_t d) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  float temperature = temperature_arr == nullptr ? temperature_val : temperature_arr[bx];
  const float inv_temp = (temperature == 0.f) ? 0.f : 1.f / temperature;

  using TempStorage = OnlineSoftmaxTempStorage<BLOCK_THREADS>;
  extern __shared__ __align__(alignof(TempStorage)) uint8_t smem[];
  auto& temp_storage = reinterpret_cast<TempStorage&>(smem);

  float* smem_vec_base = nullptr;
  if constexpr (CACHE_INPUT) {
    constexpr size_t vec_alignment = alignof(vec_t<float, VEC_SIZE>);
    size_t aligned_offset = round_up(sizeof(TempStorage), vec_alignment);
    smem_vec_base = reinterpret_cast<float*>(smem + aligned_offset);
  }

  vec_t<DTypeIn, VEC_SIZE> in_vec;
  float scaled[VEC_SIZE];

  float running_max = -cuda::std::numeric_limits<float>::infinity();
  float threadlocal_running_denominator = 0.0f;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  // Pass 1: max + denominator.
#pragma unroll 2
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    in_vec.fill(-cuda::std::numeric_limits<DTypeIn>::infinity());
    bool in_bounds = (i * BLOCK_THREADS + tx) * VEC_SIZE < d;
    if (in_bounds) {
      in_vec.cast_load(logits + bx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
    }

#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      scaled[j] = static_cast<float>(in_vec[j]) * inv_temp;
    }

    if constexpr (CACHE_INPUT) {
      if (in_bounds) {
        vec_t<float, VEC_SIZE> cache_vec;
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
          cache_vec[j] = scaled[j];
        }
        cache_vec.store(smem_vec_base + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }
    }

    float thread_max = -cuda::std::numeric_limits<float>::infinity();
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      thread_max = max(thread_max, scaled[j]);
    }
    float block_max = cub::BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                          .Reduce(thread_max, MaxReduceOp{});
    if (tx == 0) {
      temp_storage.shared_state.max_val = block_max;
    }
    __syncthreads();
    block_max = temp_storage.shared_state.max_val;
    if (!isinf(block_max)) {
      float threadlocal_sum = 0.0f;
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        threadlocal_sum += __expf(scaled[j] - block_max);
      }
      float new_max = max(running_max, block_max);
      threadlocal_running_denominator =
          threadlocal_running_denominator * __expf(running_max - new_max) +
          threadlocal_sum * __expf(block_max - new_max);
      running_max = new_max;
    }
  }

  float running_denominator =
      cub::BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
          .Sum(threadlocal_running_denominator);
  if (tx == 0) {
    temp_storage.shared_state.denominator = running_denominator;
  }
  __syncthreads();
  running_denominator = temp_storage.shared_state.denominator;

  const float final_max = running_max;
  const float inv_denominator = 1.0f / running_denominator;

  // Pass 2: normalize and store.
  vec_t<DTypeOut, VEC_SIZE> prob_vec;
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    bool in_bounds = (i * BLOCK_THREADS + tx) * VEC_SIZE < d;
    if constexpr (CACHE_INPUT) {
      if (in_bounds) {
        vec_t<float, VEC_SIZE> cache_vec;
        cache_vec.load(smem_vec_base + (i * BLOCK_THREADS + tx) * VEC_SIZE);
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
          scaled[j] = cache_vec[j];
        }
      }
    } else {
      if (in_bounds) {
        in_vec.cast_load(logits + bx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
          scaled[j] = static_cast<float>(in_vec[j]) * inv_temp;
        }
      }
    }

#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      float p = __expf(scaled[j] - final_max) * inv_denominator;
      prob_vec[j] = static_cast<DTypeOut>(p);
    }

    if (in_bounds) {
      prob_vec.cast_store(output + bx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, typename DTypeIn>
__global__ void OnlineSoftmaxMapKernel(DTypeIn* logits,
                                       PartialSoftmaxResult* partial_results,
                                       float* temperature_arr, float temperature_val,
                                       uint32_t d, uint32_t num_slices) {
  const uint32_t bx = blockIdx.x;
  const uint32_t by = blockIdx.y;  // slice index
  const uint32_t tx = threadIdx.x;
  float temperature = temperature_arr == nullptr ? temperature_val : temperature_arr[bx];
  const float inv_temp = (temperature == 0.f) ? 0.f : 1.f / temperature;

  const uint32_t vec_alignment_elems = alignof(vec_t<DTypeIn, VEC_SIZE>) / sizeof(DTypeIn);
  const uint32_t slice_stride = round_up(ceil_div(d, num_slices), vec_alignment_elems);
  const uint32_t slice_start = by * slice_stride;
  if (slice_start >= d) return;
  const uint32_t slice_size = min((by + 1) * slice_stride, d) - slice_start;

  using TempStorage = OnlineSoftmaxTempStorage<BLOCK_THREADS>;
  extern __shared__ __align__(alignof(TempStorage)) uint8_t smem[];
  auto& temp_storage = reinterpret_cast<TempStorage&>(smem);

  vec_t<DTypeIn, VEC_SIZE> in_vec;
  float scaled[VEC_SIZE];
  float running_max = -cuda::std::numeric_limits<float>::infinity();
  float threadlocal_running_denominator = 0.0f;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

#pragma unroll 2
  for (uint32_t i = 0; i < ceil_div(slice_size, BLOCK_THREADS * VEC_SIZE); ++i) {
    in_vec.fill(-cuda::std::numeric_limits<DTypeIn>::infinity());
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < slice_size) {
      in_vec.cast_load(logits + bx * d + slice_start + (i * BLOCK_THREADS + tx) * VEC_SIZE);
    }

    float thread_max = -cuda::std::numeric_limits<float>::infinity();
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      scaled[j] = static_cast<float>(in_vec[j]) * inv_temp;
      thread_max = max(thread_max, scaled[j]);
    }

    float block_max = cub::BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                          .Reduce(thread_max, MaxReduceOp{});
    if (tx == 0) {
      temp_storage.shared_state.max_val = block_max;
    }
    __syncthreads();
    block_max = temp_storage.shared_state.max_val;

    if (!isinf(block_max)) {
      float threadlocal_sum = 0.0f;
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        threadlocal_sum += __expf(scaled[j] - block_max);
      }
      float new_max = max(running_max, block_max);
      threadlocal_running_denominator =
          threadlocal_running_denominator * __expf(running_max - new_max) +
          threadlocal_sum * __expf(block_max - new_max);
      running_max = new_max;
    }
  }

  float running_denominator =
      cub::BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
          .Sum(threadlocal_running_denominator);
  if (tx == 0) {
    temp_storage.shared_state.denominator = running_denominator;
    partial_results[bx * num_slices + by] = {running_max, running_denominator};
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, typename DTypeIn, typename DTypeOut>
__global__ void OnlineSoftmaxReduceKernel(DTypeIn* logits, DTypeOut* output,
                                          PartialSoftmaxResult* partial_results,
                                          float* temperature_arr, float temperature_val,
                                          uint32_t d, uint32_t num_slices) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  float temperature = temperature_arr == nullptr ? temperature_val : temperature_arr[bx];
  const float inv_temp = (temperature == 0.f) ? 0.f : 1.f / temperature;

  using TempStorage = OnlineSoftmaxTempStorage<BLOCK_THREADS>;
  extern __shared__ __align__(alignof(TempStorage)) uint8_t smem[];
  auto& temp_storage = reinterpret_cast<TempStorage&>(smem);

  const Float2SoftmaxReduceOp reduce_op;
  float2 thread_aggregate = make_float2(-cuda::std::numeric_limits<float>::infinity(), 0.0f);

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  for (uint32_t i = tx; i < num_slices; i += BLOCK_THREADS) {
    PartialSoftmaxResult partial = partial_results[bx * num_slices + i];
    float2 partial_pair = make_float2(partial.max_val, partial.denominator);
    thread_aggregate = reduce_op(thread_aggregate, partial_pair);
  }

  float2 block_result = cub::BlockReduce<float2, BLOCK_THREADS>(temp_storage.block_prim.reduce_pair)
                            .Reduce(thread_aggregate, reduce_op);

  if (tx == 0) {
    temp_storage.shared_state.max_val = block_result.x;
    temp_storage.shared_state.denominator = block_result.y;
  }
  __syncthreads();

  const float final_max = temp_storage.shared_state.max_val;
  const float inv_denominator = 1.0f / temp_storage.shared_state.denominator;

  vec_t<DTypeIn, VEC_SIZE> in_vec;
  vec_t<DTypeOut, VEC_SIZE> prob_vec;
  float scaled[VEC_SIZE];

  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    in_vec.fill(-cuda::std::numeric_limits<DTypeIn>::infinity());
    bool in_bounds = (i * BLOCK_THREADS + tx) * VEC_SIZE < d;
    if (in_bounds) {
      in_vec.cast_load(logits + bx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
    }

#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      scaled[j] = static_cast<float>(in_vec[j]) * inv_temp;
      float p = __expf(scaled[j] - final_max) * inv_denominator;
      prob_vec[j] = static_cast<DTypeOut>(p);
    }

    if (in_bounds) {
      prob_vec.cast_store(output + bx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename DTypeIn, typename DTypeOut>
cudaError_t OnlineSoftmax(DTypeIn* logits, DTypeOut* output, uint32_t batch_size, uint32_t d,
                          float* temperature_arr, float temperature_val, void* workspace_buffer,
                          size_t workspace_buffer_size_in_bytes, bool enable_pdl,
                          cudaStream_t stream = 0) {
  constexpr uint32_t SMALL_BATCH_THRESHOLD = 128;
  constexpr uint32_t LARGE_VOCAB_THRESHOLD = 24576;
  constexpr uint32_t DEFAULT_SLICE_SIZE = 8192;

  // Size vec by max(in, out) so store transactions stay <= 16 bytes.
  constexpr size_t kMaxDtypeSize =
      sizeof(DTypeIn) > sizeof(DTypeOut) ? sizeof(DTypeIn) : sizeof(DTypeOut);
  const uint32_t vec_size = std::gcd(16 / kMaxDtypeSize, d);
  auto compute_capacity = flashinfer::GetCudaComputeCapability();

  DISPATCH_COMPUTE_CAP_NUM_THREADS(
      compute_capacity, BLOCK_THREADS, {DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
        if (batch_size <= SMALL_BATCH_THRESHOLD && d >= LARGE_VOCAB_THRESHOLD) {
          uint32_t num_slices = ceil_div(d, DEFAULT_SLICE_SIZE);
          const size_t partial_buffer_size = batch_size * num_slices * sizeof(PartialSoftmaxResult);
          if (workspace_buffer_size_in_bytes < partial_buffer_size) {
            return cudaErrorInvalidValue;
          }

          AlignedAllocator allocator(workspace_buffer, workspace_buffer_size_in_bytes);
          auto partial_results = allocator.aligned_alloc<PartialSoftmaxResult>(
              partial_buffer_size, alignof(PartialSoftmaxResult), "softmax_workspace");

          dim3 phase1_nblks(batch_size, num_slices);
          dim3 phase1_nthrs(BLOCK_THREADS);
          size_t smem_size = sizeof(OnlineSoftmaxTempStorage<BLOCK_THREADS>);

          auto phase1_kernel = OnlineSoftmaxMapKernel<BLOCK_THREADS, VEC_SIZE, DTypeIn>;
          FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
              phase1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
          if (enable_pdl) {
            cudaLaunchAttribute attribute[1];
            attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attribute[0].val.programmaticStreamSerializationAllowed = 1;
            cudaLaunchConfig_t config{
                phase1_nblks, phase1_nthrs, smem_size, stream, attribute, 1};
            FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, phase1_kernel, logits, partial_results,
                                                    temperature_arr, temperature_val, d,
                                                    num_slices));
          } else {
            void* args[] = {&logits, &partial_results, &temperature_arr,
                            &temperature_val, &d, &num_slices};
            FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)phase1_kernel, phase1_nblks, phase1_nthrs,
                                                  args, smem_size, stream));
          }

          dim3 phase2_nblks(batch_size);
          dim3 phase2_nthrs(BLOCK_THREADS);
          auto phase2_kernel = OnlineSoftmaxReduceKernel<BLOCK_THREADS, VEC_SIZE, DTypeIn, DTypeOut>;
          FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
              phase2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
          if (enable_pdl) {
            cudaLaunchAttribute attribute[1];
            attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attribute[0].val.programmaticStreamSerializationAllowed = 1;
            cudaLaunchConfig_t config{
                phase2_nblks, phase2_nthrs, smem_size, stream, attribute, 1};
            FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, phase2_kernel, logits, output,
                                                    partial_results, temperature_arr,
                                                    temperature_val, d, num_slices));
          } else {
            void* args[] = {&logits, &output, &partial_results, &temperature_arr,
                            &temperature_val, &d, &num_slices};
            FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)phase2_kernel, phase2_nblks, phase2_nthrs,
                                                  args, smem_size, stream));
          }
        } else {
          int device, smem_max;
          FLASHINFER_CUDA_CALL(cudaGetDevice(&device));
          FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
              &smem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
          uint32_t cache_threshold = 0;
          if (smem_max > (int)sizeof(OnlineSoftmaxTempStorage<BLOCK_THREADS>)) {
            cache_threshold =
                (smem_max - sizeof(OnlineSoftmaxTempStorage<BLOCK_THREADS>)) / sizeof(float) -
                VEC_SIZE;
          }
          const bool cache_input = d <= cache_threshold;

          dim3 nblks(batch_size);
          dim3 nthrs(BLOCK_THREADS);

          const size_t smem_logits_bytes = (round_up(d, VEC_SIZE) + VEC_SIZE) * sizeof(float);
          uint32_t smem_size = sizeof(OnlineSoftmaxTempStorage<BLOCK_THREADS>) +
                               (cache_input ? smem_logits_bytes : 0);

          DISPATCH_SOFTMAX_CACHE_INPUT(cache_input, CACHE_INPUT, {
            auto kernel =
                OnlineSoftmaxFusedKernel<BLOCK_THREADS, VEC_SIZE, DTypeIn, DTypeOut, CACHE_INPUT>;
            FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            if (enable_pdl) {
              cudaLaunchAttribute attribute[1];
              attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
              attribute[0].val.programmaticStreamSerializationAllowed = 1;
              cudaLaunchConfig_t config{nblks, nthrs, smem_size, stream, attribute, 1};
              FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, logits, output,
                                                      temperature_arr, temperature_val, d));
            } else {
              void* args[] = {&logits, &output, &temperature_arr, &temperature_val, &d};
              FLASHINFER_CUDA_CALL(
                  cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
            }
          });
        }
      })});
  return cudaSuccess;
}

}  // namespace tokenspeed

void softmax(TensorView workspace_buffer, TensorView logits, TensorView output,
             Optional<TensorView> maybe_temperature_arr, double temperature_val,
             bool enable_pdl) {
  CHECK_INPUT(workspace_buffer);
  CHECK_INPUT(logits);
  CHECK_INPUT(output);
  CHECK_DIM(2, logits);
  CHECK_DIM(2, output);
  CHECK_SHAPE(logits, output);

  TVM_FFI_ICHECK(output.dtype() == dl_float32)
      << "softmax output must be fp32, got " << output.dtype().code << " "
      << output.dtype().bits;

  if (maybe_temperature_arr.has_value()) {
    CHECK_INPUT(maybe_temperature_arr.value());
    TVM_FFI_ICHECK(maybe_temperature_arr.value().dtype() == dl_float32)
        << "temperature tensor must be fp32";
  }

  unsigned int batch_size = logits.size(0);
  unsigned int vocab_size = logits.size(1);

  cudaSetDevice(logits.device().device_id);
  auto stream = get_stream(logits.device());

  bool ok = DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(logits.dtype(), c_type, [&] {
    cudaError_t status = tokenspeed::OnlineSoftmax<c_type, float>(
        static_cast<c_type*>(logits.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        batch_size,
        vocab_size,
        maybe_temperature_arr.has_value()
            ? static_cast<float*>(maybe_temperature_arr.value().data_ptr())
            : nullptr,
        static_cast<float>(temperature_val),
        workspace_buffer.data_ptr(),
        get_element_size(workspace_buffer) * workspace_buffer.size(0),
        enable_pdl,
        stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "OnlineSoftmax launch failed: " << cudaGetErrorString(status);
    return true;
  });
  TVM_FFI_ICHECK(ok) << "softmax: unsupported input dtype.";
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(softmax, softmax);
