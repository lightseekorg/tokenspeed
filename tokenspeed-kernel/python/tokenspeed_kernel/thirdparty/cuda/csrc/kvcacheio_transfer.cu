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

// KV-cache transfer kernel.
// TVM-FFI integration for tokenspeed_kernel.

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "tvm_ffi_utils.h"

#define WARP_SIZE 32

namespace {

inline void check_cuda(cudaError_t err, const char* msg) {
  TVM_FFI_ICHECK(err == cudaSuccess) << msg << ": " << cudaGetErrorString(err);
}

inline int64_t div_up(int64_t x, int64_t y) {
  TVM_FFI_ICHECK_GT(y, 0);
  return (x + y - 1) / y;
}

inline const uintptr_t* as_ptr_table(TensorView t) {
  return static_cast<const uintptr_t*>(t.data_ptr());
}

inline void check_indices(TensorView src_indices, TensorView dst_indices) {
  CHECK_CUDA(src_indices);
  CHECK_CUDA(dst_indices);
  CHECK_INPUT_TYPE(src_indices, dl_int64);
  CHECK_INPUT_TYPE(dst_indices, dl_int64);
  CHECK_DIM(1, src_indices);
  CHECK_DIM(1, dst_indices);
  TVM_FFI_ICHECK_EQ(src_indices.numel(), dst_indices.numel())
      << "Source and destination indices must have the same length";
}

inline std::vector<int64_t> indices_to_host(TensorView indices) {
  CHECK_DIM(1, indices);
  CHECK_INPUT_TYPE(indices, dl_int64);

  std::vector<int64_t> host(indices.numel());
  if (host.empty()) {
    return host;
  }

  if (indices.device().device_type == kDLCPU) {
    std::memcpy(host.data(), indices.data_ptr(), host.size() * sizeof(int64_t));
    return host;
  }

  TVM_FFI_ICHECK_EQ(indices.device().device_type, kDLCUDA)
      << "indices must be either CPU or CUDA tensor";
  check_cuda(
      cudaMemcpy(host.data(), indices.data_ptr(), host.size() * sizeof(int64_t), cudaMemcpyDeviceToHost),
      "Failed to copy indices to host");
  return host;
}

inline void copy_async_bytes(void* dst, const void* src, size_t num_bytes, cudaStream_t stream) {
  if (num_bytes == 0) {
    return;
  }
  check_cuda(cudaMemcpyAsync(dst, src, num_bytes, cudaMemcpyDefault, stream), "cudaMemcpyAsync failed");
}

inline void copy_token_span(
    TensorView src,
    TensorView dst,
    int64_t src_index,
    int64_t dst_index,
    int64_t num_tokens,
    cudaStream_t stream) {
  TVM_FFI_ICHECK_GE(src.dim(), 1);
  TVM_FFI_ICHECK_GE(dst.dim(), 1);

  const int64_t src_stride_bytes = src.stride(0) * get_element_size(src);
  const int64_t dst_stride_bytes = dst.stride(0) * get_element_size(dst);
  TVM_FFI_ICHECK_EQ(src_stride_bytes, dst_stride_bytes)
      << "Source and destination token stride bytes must match for direct copy";

  const size_t copy_bytes = static_cast<size_t>(num_tokens * src_stride_bytes);
  const char* src_ptr = static_cast<const char*>(src.data_ptr()) + src_index * src_stride_bytes;
  char* dst_ptr = static_cast<char*>(dst.data_ptr()) + dst_index * dst_stride_bytes;
  copy_async_bytes(dst_ptr, src_ptr, copy_bytes, stream);
}

__device__ __forceinline__ void transfer_item_warp(
    int32_t lane_id,
    const void* src_addr,
    void* dst_addr,
    int64_t item_size_bytes) {
  const uint64_t* __restrict__ src = static_cast<const uint64_t*>(src_addr);
  uint64_t* __restrict__ dst = static_cast<uint64_t*>(dst_addr);
  const int total_chunks = static_cast<int>(item_size_bytes / sizeof(uint64_t));

#pragma unroll
  for (int j = lane_id; j < total_chunks; j += WARP_SIZE) {
    dst[j] = src[j];
  }
}

template <typename T>
__device__ __forceinline__ T* get_global_offset_lf(
    T* base,
    const uintptr_t* __restrict__ /*unused*/,
    int64_t layer_id,
    int64_t layer_dim,
    int64_t page_id,
    int64_t item_size_bytes) {
  return base + layer_id * layer_dim + page_id * item_size_bytes;
}

template <typename T>
__device__ __forceinline__ T* get_global_offset_pf(
    T* base,
    const uintptr_t* __restrict__ /*unused*/,
    int64_t layer_id,
    int64_t page_dim,
    int64_t page_id,
    int64_t item_size_bytes) {
  return base + page_id * page_dim + layer_id * item_size_bytes;
}

template <typename T>
__device__ __forceinline__ T* get_global_offset_lf_tbl(
    T* /*unused*/,
    const uintptr_t* __restrict__ layer_base_tbl,
    int64_t layer_id,
    int64_t /*unused*/,
    int64_t page_id,
    int64_t item_size_bytes) {
  return reinterpret_cast<T*>(layer_base_tbl[layer_id]) + page_id * item_size_bytes;
}

template <typename T>
__device__ __forceinline__ T* get_global_offset_per_head_lf(
    T* base,
    const uintptr_t* __restrict__ /*unused*/,
    int64_t layer_id,
    int64_t layer_dim,
    int64_t page_id,
    int64_t item_size_bytes,
    int64_t head_id,
    int64_t head_num,
    int64_t /*unused*/) {
  return base + layer_id * layer_dim + page_id * item_size_bytes + item_size_bytes / head_num * head_id;
}

template <typename T>
__device__ __forceinline__ T* get_global_offset_per_head_lf_tbl(
    T* /*unused*/,
    const uintptr_t* __restrict__ layer_base_tbl,
    int64_t layer_id,
    int64_t /*unused*/,
    int64_t page_id,
    int64_t item_size_bytes,
    int64_t head_id,
    int64_t head_num,
    int64_t /*unused*/) {
  return reinterpret_cast<T*>(layer_base_tbl[layer_id]) + page_id * item_size_bytes +
         item_size_bytes / head_num * head_id;
}

template <typename T>
__device__ __forceinline__ T* get_global_offset_ph(
    T* base,
    const uintptr_t* __restrict__ /*unused*/,
    int64_t layer_id,
    int64_t page_dim,
    int64_t page_id,
    int64_t item_size_bytes,
    int64_t head_id,
    int64_t head_num,
    int64_t page_size) {
  // page head layout: [page_num, head_num, page_size, layer_num, head_dim]
  return base + page_id / page_size * page_size * page_dim +  // page_num dimension offset
         page_dim / head_num * head_id * page_size +           // head_num dimension offset
         page_id % page_size * page_dim / head_num +           // page_size dimension offset
         layer_id * item_size_bytes / head_num;                // layer_num dimension offset
}

template <auto SrcOffsetFn, auto DstOffsetFn>
__global__ void transfer_page_head_kernel_impl(
    const void* __restrict__ src_k,
    void* __restrict__ dst_k,
    const void* __restrict__ src_v,
    void* __restrict__ dst_v,
    const int64_t* __restrict__ src_indices,
    const int64_t* __restrict__ dst_indices,
    int64_t start_layer_id,
    int64_t num_layers_to_process,
    int64_t num_items,
    int64_t items_per_warp,
    int64_t item_size_bytes,
    int64_t src_layout_dim,
    int64_t dst_layout_dim,
    const uintptr_t* __restrict__ src_k_layer_tbl,
    const uintptr_t* __restrict__ dst_k_layer_tbl,
    const uintptr_t* __restrict__ src_v_layer_tbl,
    const uintptr_t* __restrict__ dst_v_layer_tbl,
    int64_t page_size,
    int64_t head_num) {
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t lane_id = tid % WARP_SIZE;
  int32_t warp_id = tid / WARP_SIZE;
  const int64_t head_size_bytes = item_size_bytes / head_num;

  for (int i = 0; i < items_per_warp; ++i) {
    int64_t item_id = warp_id * items_per_warp + i;
    if (item_id >= num_items) {
      break;
    }
    const int64_t src_page_id = src_indices[item_id];
    const int64_t dst_page_id = dst_indices[item_id];

    for (int64_t layer_id = start_layer_id; layer_id < start_layer_id + num_layers_to_process; ++layer_id) {
      for (int64_t head_id = 0; head_id < head_num; ++head_id) {
        const char* src_k_ptr = SrcOffsetFn(
            static_cast<const char*>(src_k),
            src_k_layer_tbl,
            layer_id,
            src_layout_dim,
            src_page_id,
            item_size_bytes,
            head_id,
            head_num,
            page_size);
        char* dst_k_ptr = DstOffsetFn(
            static_cast<char*>(dst_k),
            dst_k_layer_tbl,
            layer_id,
            dst_layout_dim,
            dst_page_id,
            item_size_bytes,
            head_id,
            head_num,
            page_size);
        transfer_item_warp(lane_id, src_k_ptr, dst_k_ptr, head_size_bytes);

        const char* src_v_ptr = SrcOffsetFn(
            static_cast<const char*>(src_v),
            src_v_layer_tbl,
            layer_id,
            src_layout_dim,
            src_page_id,
            item_size_bytes,
            head_id,
            head_num,
            page_size);
        char* dst_v_ptr = DstOffsetFn(
            static_cast<char*>(dst_v),
            dst_v_layer_tbl,
            layer_id,
            dst_layout_dim,
            dst_page_id,
            item_size_bytes,
            head_id,
            head_num,
            page_size);
        transfer_item_warp(lane_id, src_v_ptr, dst_v_ptr, head_size_bytes);
      }
    }
  }
}

template <auto SrcOffsetFn, auto DstOffsetFn, bool IsMLA>
__global__ void transfer_kernel_impl(
    const void* __restrict__ src_k,
    void* __restrict__ dst_k,
    const void* __restrict__ src_v,
    void* __restrict__ dst_v,
    const int64_t* __restrict__ src_indices,
    const int64_t* __restrict__ dst_indices,
    int64_t start_layer_id,
    int64_t num_layers_to_process,
    int64_t num_items,
    int64_t items_per_warp,
    int64_t item_size_bytes,
    int64_t src_layout_dim,
    int64_t dst_layout_dim,
    const uintptr_t* __restrict__ src_k_layer_tbl,
    const uintptr_t* __restrict__ dst_k_layer_tbl,
    const uintptr_t* __restrict__ src_v_layer_tbl,
    const uintptr_t* __restrict__ dst_v_layer_tbl) {
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t lane_id = tid % WARP_SIZE;
  int32_t warp_id = tid / WARP_SIZE;

  for (int i = 0; i < items_per_warp; ++i) {
    int64_t item_id = warp_id * items_per_warp + i;
    if (item_id >= num_items) {
      break;
    }
    const int64_t src_page_id = src_indices[item_id];
    const int64_t dst_page_id = dst_indices[item_id];

    for (int64_t layer_id = start_layer_id; layer_id < start_layer_id + num_layers_to_process; ++layer_id) {
      const char* src_ptr = SrcOffsetFn(
          static_cast<const char*>(src_k), src_k_layer_tbl, layer_id, src_layout_dim, src_page_id, item_size_bytes);
      char* dst_ptr = DstOffsetFn(
          static_cast<char*>(dst_k), dst_k_layer_tbl, layer_id, dst_layout_dim, dst_page_id, item_size_bytes);
      transfer_item_warp(lane_id, src_ptr, dst_ptr, item_size_bytes);

      if constexpr (!IsMLA) {
        const char* src_v_ptr = SrcOffsetFn(
            static_cast<const char*>(src_v), src_v_layer_tbl, layer_id, src_layout_dim, src_page_id, item_size_bytes);
        char* dst_v_ptr = DstOffsetFn(
            static_cast<char*>(dst_v), dst_v_layer_tbl, layer_id, dst_layout_dim, dst_page_id, item_size_bytes);
        transfer_item_warp(lane_id, src_v_ptr, dst_v_ptr, item_size_bytes);
      }
    }
  }
}

template <auto SrcOffsetFn, auto DstOffsetFn, bool IsMLA, bool PageHeadLayout = false>
void transfer_kv_launcher(
    const void* src_k,
    void* dst_k,
    const void* src_v,
    void* dst_v,
    TensorView src_indices,
    TensorView dst_indices,
    int64_t start_layer_id,
    int64_t num_layers_to_process,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t dst_layout_dim,
    const uintptr_t* src_k_layers,
    const uintptr_t* dst_k_layers,
    const uintptr_t* src_v_layers,
    const uintptr_t* dst_v_layers,
    int64_t block_quota,
    int64_t num_warps_per_block,
    int64_t page_size = 16,
    int64_t head_num = 1) {
  check_indices(src_indices, dst_indices);
  TVM_FFI_ICHECK_EQ(item_size % 8, 0) << "Item byte size must be divisible by 8";
  TVM_FFI_ICHECK_GT(block_quota, 0);
  TVM_FFI_ICHECK_GT(num_warps_per_block, 0);

  const int64_t num_items = src_indices.numel();
  if (num_items == 0 || num_layers_to_process == 0) {
    return;
  }

  const int64_t items_per_warp = div_up(num_items, block_quota * num_warps_per_block);
  const int32_t num_blocks = static_cast<int32_t>(div_up(num_items, items_per_warp * num_warps_per_block));
  const int32_t threads_per_block = static_cast<int32_t>(num_warps_per_block * WARP_SIZE);
  const dim3 grid_dim(num_blocks, 1, 1);

  const cudaStream_t stream = get_current_stream();
  if constexpr (PageHeadLayout) {
    transfer_page_head_kernel_impl<SrcOffsetFn, DstOffsetFn><<<grid_dim, threads_per_block, 0, stream>>>(
        src_k,
        dst_k,
        src_v,
        dst_v,
        static_cast<const int64_t*>(src_indices.data_ptr()),
        static_cast<const int64_t*>(dst_indices.data_ptr()),
        start_layer_id,
        num_layers_to_process,
        num_items,
        items_per_warp,
        item_size,
        src_layout_dim,
        dst_layout_dim,
        src_k_layers,
        dst_k_layers,
        src_v_layers,
        dst_v_layers,
        page_size,
        head_num);
  } else {
    transfer_kernel_impl<SrcOffsetFn, DstOffsetFn, IsMLA><<<grid_dim, threads_per_block, 0, stream>>>(
        src_k,
        dst_k,
        src_v,
        dst_v,
        static_cast<const int64_t*>(src_indices.data_ptr()),
        static_cast<const int64_t*>(dst_indices.data_ptr()),
        start_layer_id,
        num_layers_to_process,
        num_items,
        items_per_warp,
        item_size,
        src_layout_dim,
        dst_layout_dim,
        src_k_layers,
        dst_k_layers,
        src_v_layers,
        dst_v_layers);
  }

  check_cuda(cudaGetLastError(), "transfer_kv kernel launch failed");
}

}  // namespace

void transfer_kv_per_layer(
    TensorView src_k,
    TensorView dst_k,
    TensorView src_v,
    TensorView dst_v,
    TensorView src_indices,
    TensorView dst_indices,
    int64_t item_size,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  transfer_kv_launcher<get_global_offset_lf<const char>, get_global_offset_lf<char>, false>(
      src_k.data_ptr(),
      dst_k.data_ptr(),
      src_v.data_ptr(),
      dst_v.data_ptr(),
      src_indices,
      dst_indices,
      0,
      1,
      item_size,
      0,
      0,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      block_quota,
      num_warps_per_block);
}

void transfer_kv_per_layer_pf_lf(
    TensorView src_k,
    TensorView dst_k,
    TensorView src_v,
    TensorView dst_v,
    TensorView src_indices,
    TensorView dst_indices,
    int64_t layer_id,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  transfer_kv_launcher<get_global_offset_pf<const char>, get_global_offset_lf<char>, false>(
      src_k.data_ptr(),
      dst_k.data_ptr(),
      src_v.data_ptr(),
      dst_v.data_ptr(),
      src_indices,
      dst_indices,
      layer_id,
      1,
      item_size,
      src_layout_dim,
      0,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      block_quota,
      num_warps_per_block);
}

void transfer_kv_per_layer_ph_lf(
    TensorView src_k,
    TensorView dst_k,
    TensorView src_v,
    TensorView dst_v,
    TensorView src_indices,
    TensorView dst_indices,
    int64_t layer_id,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t page_size,
    int64_t head_num,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  transfer_kv_launcher<get_global_offset_ph<const char>, get_global_offset_per_head_lf<char>, false, true>(
      src_k.data_ptr(),
      dst_k.data_ptr(),
      src_v.data_ptr(),
      dst_v.data_ptr(),
      src_indices,
      dst_indices,
      layer_id,
      1,
      item_size,
      src_layout_dim,
      0,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      block_quota,
      num_warps_per_block,
      page_size,
      head_num);
}

void transfer_kv_all_layer(
    TensorView src_k_layers,
    TensorView dst_k_layers,
    TensorView src_v_layers,
    TensorView dst_v_layers,
    TensorView src_indices,
    TensorView dst_indices,
    int64_t item_size,
    int64_t num_layers,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  TVM_FFI_ICHECK_EQ(num_layers, src_k_layers.size(0))
      << "Number of layers in source k tensor does not match num_layers";
  transfer_kv_launcher<get_global_offset_lf_tbl<const char>, get_global_offset_lf_tbl<char>, false>(
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      src_indices,
      dst_indices,
      0,
      num_layers,
      item_size,
      0,
      0,
      as_ptr_table(src_k_layers),
      as_ptr_table(dst_k_layers),
      as_ptr_table(src_v_layers),
      as_ptr_table(dst_v_layers),
      block_quota,
      num_warps_per_block);
}

void transfer_kv_all_layer_lf_pf(
    TensorView src_k_layers,
    TensorView dst_k,
    TensorView src_v_layers,
    TensorView dst_v,
    TensorView src_indices,
    TensorView dst_indices,
    int64_t item_size,
    int64_t dst_layout_dim,
    int64_t num_layers,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  TVM_FFI_ICHECK_EQ(num_layers, src_k_layers.size(0))
      << "Number of layers in source k tensor does not match num_layers";
  transfer_kv_launcher<get_global_offset_lf_tbl<const char>, get_global_offset_pf<char>, false>(
      nullptr,
      dst_k.data_ptr(),
      nullptr,
      dst_v.data_ptr(),
      src_indices,
      dst_indices,
      0,
      num_layers,
      item_size,
      0,
      dst_layout_dim,
      as_ptr_table(src_k_layers),
      nullptr,
      as_ptr_table(src_v_layers),
      nullptr,
      block_quota,
      num_warps_per_block);
}

void transfer_kv_all_layer_lf_ph(
    TensorView src_k_layers,
    TensorView dst_k,
    TensorView src_v_layers,
    TensorView dst_v,
    TensorView src_indices,
    TensorView dst_indices,
    int64_t item_size,
    int64_t dst_layout_dim,
    int64_t num_layers,
    int64_t page_size,
    int64_t head_num,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  TVM_FFI_ICHECK_EQ(num_layers, src_k_layers.size(0))
      << "Number of layers in source k tensor does not match num_layers";
  transfer_kv_launcher<get_global_offset_per_head_lf_tbl<const char>, get_global_offset_ph<char>, false, true>(
      nullptr,
      dst_k.data_ptr(),
      nullptr,
      dst_v.data_ptr(),
      src_indices,
      dst_indices,
      0,
      num_layers,
      item_size,
      0,
      dst_layout_dim,
      as_ptr_table(src_k_layers),
      nullptr,
      as_ptr_table(src_v_layers),
      nullptr,
      block_quota,
      num_warps_per_block,
      page_size,
      head_num);
}

void transfer_kv_per_layer_mla(
    TensorView src,
    TensorView dst,
    TensorView src_indices,
    TensorView dst_indices,
    int64_t item_size,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  transfer_kv_launcher<get_global_offset_lf<const char>, get_global_offset_lf<char>, true>(
      src.data_ptr(),
      dst.data_ptr(),
      nullptr,
      nullptr,
      src_indices,
      dst_indices,
      0,
      1,
      item_size,
      0,
      0,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      block_quota,
      num_warps_per_block);
}

void transfer_kv_per_layer_mla_pf_lf(
    TensorView src,
    TensorView dst,
    TensorView src_indices,
    TensorView dst_indices,
    int64_t layer_id,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  transfer_kv_launcher<get_global_offset_pf<const char>, get_global_offset_lf<char>, true>(
      src.data_ptr(),
      dst.data_ptr(),
      nullptr,
      nullptr,
      src_indices,
      dst_indices,
      layer_id,
      1,
      item_size,
      src_layout_dim,
      0,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      block_quota,
      num_warps_per_block);
}

void transfer_kv_all_layer_mla(
    TensorView src_layers,
    TensorView dst_layers,
    TensorView src_indices,
    TensorView dst_indices,
    int64_t item_size,
    int64_t num_layers,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  TVM_FFI_ICHECK_EQ(num_layers, src_layers.size(0)) << "Number of layers in source tensor does not match num_layers";
  transfer_kv_launcher<get_global_offset_lf_tbl<const char>, get_global_offset_lf_tbl<char>, true>(
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      src_indices,
      dst_indices,
      0,
      num_layers,
      item_size,
      0,
      0,
      as_ptr_table(src_layers),
      as_ptr_table(dst_layers),
      nullptr,
      nullptr,
      block_quota,
      num_warps_per_block);
}

void transfer_kv_all_layer_mla_lf_pf(
    TensorView src_layers,
    TensorView dst,
    TensorView src_indices,
    TensorView dst_indices,
    int64_t item_size,
    int64_t dst_layout_dim,
    int64_t num_layers,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  TVM_FFI_ICHECK_EQ(num_layers, src_layers.size(0)) << "Number of layers in source tensor does not match num_layers";
  transfer_kv_launcher<get_global_offset_lf_tbl<const char>, get_global_offset_pf<char>, true>(
      nullptr,
      dst.data_ptr(),
      nullptr,
      nullptr,
      src_indices,
      dst_indices,
      0,
      num_layers,
      item_size,
      0,
      dst_layout_dim,
      as_ptr_table(src_layers),
      nullptr,
      nullptr,
      nullptr,
      block_quota,
      num_warps_per_block);
}

void transfer_kv_direct(
    const std::vector<TensorView>& src_layers,
    std::vector<TensorView> dst_layers,
    TensorView src_indices,
    TensorView dst_indices,
    int64_t page_size) {
  TVM_FFI_ICHECK_EQ(src_layers.size(), dst_layers.size())
      << "Source and destination layers must have the same number of layers";
  TVM_FFI_ICHECK_EQ(src_indices.numel(), dst_indices.numel())
      << "Source and destination indices must have the same length";
  TVM_FFI_ICHECK_GT(page_size, 0) << "Page size must be positive";
  TVM_FFI_ICHECK_EQ(src_indices.numel() % page_size, 0)
      << "Source indices size must be divisible by page size";

  const auto src_indices_host = indices_to_host(src_indices);
  const auto dst_indices_host = indices_to_host(dst_indices);
  const int64_t num_indices = static_cast<int64_t>(src_indices_host.size());
  const int64_t num_layers = static_cast<int64_t>(src_layers.size());
  const cudaStream_t stream = get_current_stream();

  int64_t start_index = 0;
  int64_t end_index = 0;
  for (int64_t i = 0; i < num_indices; ++i) {
    if (i < num_indices - 1) {
      const int64_t src_diff = src_indices_host[i + 1] - src_indices_host[i];
      const int64_t dst_diff = dst_indices_host[i + 1] - dst_indices_host[i];
      if (src_diff == 1 && dst_diff == 1) {
        continue;
      }
      end_index = i + 1;
    } else {
      end_index = num_indices;
    }

    const int64_t src_index = src_indices_host[start_index];
    const int64_t dst_index = dst_indices_host[start_index];
    const int64_t num_tokens = end_index - start_index;

    for (int64_t j = 0; j < num_layers; ++j) {
      copy_token_span(src_layers[j], dst_layers[j], src_index, dst_index, num_tokens, stream);
    }
    start_index = end_index;
  }
}
