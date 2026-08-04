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

void transfer_kv_per_layer(
    TensorView src_k,
    TensorView dst_k,
    TensorView src_v,
    TensorView dst_v,
    TensorView src_indices,
    TensorView dst_indices,
    int64_t item_size,
    int64_t block_quota,
    int64_t num_warps_per_block);

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
    int64_t num_warps_per_block);

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
    int64_t num_warps_per_block);

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
    int64_t num_warps_per_block);

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
    int64_t num_warps_per_block);

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
    int64_t num_warps_per_block);

void transfer_kv_per_layer_mla(
    TensorView src,
    TensorView dst,
    TensorView src_indices,
    TensorView dst_indices,
    int64_t item_size,
    int64_t block_quota,
    int64_t num_warps_per_block);

void transfer_kv_per_layer_mla_pf_lf(
    TensorView src,
    TensorView dst,
    TensorView src_indices,
    TensorView dst_indices,
    int64_t layer_id,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t block_quota,
    int64_t num_warps_per_block);

void transfer_kv_all_layer_mla(
    TensorView src_layers,
    TensorView dst_layers,
    TensorView src_indices,
    TensorView dst_indices,
    int64_t item_size,
    int64_t num_layers,
    int64_t block_quota,
    int64_t num_warps_per_block);

void transfer_kv_all_layer_mla_lf_pf(
    TensorView src_layers,
    TensorView dst,
    TensorView src_indices,
    TensorView dst_indices,
    int64_t item_size,
    int64_t dst_layout_dim,
    int64_t num_layers,
    int64_t block_quota,
    int64_t num_warps_per_block);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(transfer_kv_per_layer, transfer_kv_per_layer);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(transfer_kv_per_layer_pf_lf, transfer_kv_per_layer_pf_lf);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(transfer_kv_per_layer_ph_lf, transfer_kv_per_layer_ph_lf);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(transfer_kv_all_layer, transfer_kv_all_layer);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(transfer_kv_all_layer_lf_pf, transfer_kv_all_layer_lf_pf);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(transfer_kv_all_layer_lf_ph, transfer_kv_all_layer_lf_ph);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(transfer_kv_per_layer_mla, transfer_kv_per_layer_mla);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(transfer_kv_per_layer_mla_pf_lf, transfer_kv_per_layer_mla_pf_lf);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(transfer_kv_all_layer_mla, transfer_kv_all_layer_mla);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(transfer_kv_all_layer_mla_lf_pf, transfer_kv_all_layer_mla_lf_pf);
