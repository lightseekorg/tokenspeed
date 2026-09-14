# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Stable imports for DSV4 Triton kernels and helpers."""

from tokenspeed_kernel.ops.attention.dsv4._triton.attention import (  # noqa: F401
    _dsv4_dequantize_selected_cache_rows_kernel,
    _dsv4_dequantize_selected_cache_segment,
    _dsv4_sparse_attention_kernel,
    triton_dsv4_decode,
    triton_dsv4_prefill,
)
from tokenspeed_kernel.ops.attention.dsv4._triton.cache import (  # noqa: F401
    _dsv4_dequantize_and_gather_k_kernel,
    _dsv4_gather_indexer_mxfp4_cache_kernel,
    _dsv4_gather_launch_config,
    _dsv4_indexer_mxfp4_cache_write_kernel,
    _dsv4_qnorm_rope_kv_insert_kernel,
    dsv4_dequantize_and_gather_k_cache,
    dsv4_gather_indexer_mxfp4_cache,
    triton_dsv4_swa_cache_insert,
    write_dsv4_indexer_mxfp4_cache_cuda,
)
from tokenspeed_kernel.ops.attention.dsv4._triton.common import (  # noqa: F401
    DEEPSEEK_V4_FP8_MAX,
    DEEPSEEK_V4_FP8_QUANT_BLOCK,
    DEEPSEEK_V4_HEAD_DIM,
    DEEPSEEK_V4_INDEXER_DIM,
    DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM,
    DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES,
    DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
    DEEPSEEK_V4_NOPE_DIM,
    DEEPSEEK_V4_ROPE_DIM,
    DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT,
    DEEPSEEK_V4_SWA_SCALE_DIM,
    DEEPSEEK_V4_SWA_TOKEN_STRIDE,
    _as_int32_block_table,
    _dsv4_mxfp4_e2m1_nibble,
)
from tokenspeed_kernel.ops.attention.dsv4._triton.compress import (  # noqa: F401
    _dsv4_fused_csa_indexer_fp8_cache_kernel,
    _dsv4_fused_csa_indexer_mxfp4_cache_kernel,
    _dsv4_fused_sparse_compress_cache_kernel,
    _dsv4_save_compressor_state_kernel,
    _wide_compress_launch_supported,
    dsv4_fused_csa_indexer_mxfp4_cache_insert,
    dsv4_fused_sparse_compress_cache_insert,
    dsv4_save_compressor_state,
    triton_dsv4_csa_indexer_fp8_cache_insert,
)
from tokenspeed_kernel.ops.attention.dsv4._triton.indexer import (  # noqa: F401
    triton_dsv4_decode_topk_mxfp4,
    triton_dsv4_plan,
    triton_dsv4_prefill_topk_mxfp4,
)
from tokenspeed_kernel.ops.attention.dsv4._triton.indices import (  # noqa: F401
    _dsv4_build_dense_prefill_local_compressed_indices_kernel,
    _dsv4_combine_dense_swa_indices_kernel,
    _dsv4_combine_topk_swa_indices_kernel,
    _dsv4_compute_global_topk_indices_and_lens_kernel,
    _dsv4_decode_dense_compressed_indices_and_lens_kernel,
    _dsv4_decode_swa_indices_and_lens_kernel,
    dsv4_build_dense_prefill_local_compressed_indices,
    dsv4_combine_dense_swa_indices,
    dsv4_combine_topk_swa_indices,
    dsv4_compute_global_topk_indices_and_lens,
    dsv4_decode_dense_compressed_indices_and_lens,
    dsv4_decode_swa_indices_and_lens,
)
from tokenspeed_kernel.ops.attention.dsv4._triton.metadata import (  # noqa: F401
    _dsv4_compact_compressed_slot_mapping_kernel,
    _dsv4_compressed_slot_mapping_kernel,
    _dsv4_group_slot_mapping_kernel,
    _dsv4_indexer_decode_metadata_kernel,
    _dsv4_validate_active_cache_pages_kernel,
    dsv4_compact_compressed_slot_mapping,
    dsv4_compressed_slot_mapping,
    dsv4_group_slot_mapping,
    dsv4_indexer_decode_metadata_compute,
    dsv4_validate_active_cache_pages,
)
from tokenspeed_kernel.ops.attention.dsv4._triton.quantization import (  # noqa: F401
    _dsv4_fused_inv_rope_fp8_quant_per_head,
    dsv4_fused_inv_rope_fp8_quant,
)
from tokenspeed_kernel.ops.attention.dsv4._triton.query import (  # noqa: F401
    _dsv4_fused_indexer_q_rope_hadamard_mxfp4_kernel,
    _dsv4_fused_indexer_q_rope_hadamard_mxfp4_serial_four_block_kernel,
    _dsv4_indexer_q_cuda_capability,
    _dsv4_serial_four_block_indexer_q_supported,
    _dsv4_use_serial_four_block_indexer_q,
    _log_serial_four_block_indexer_q_selection,
    dsv4_fused_indexer_q_rope_hadamard_mxfp4,
)

__all__ = [
    "dsv4_build_dense_prefill_local_compressed_indices",
    "dsv4_combine_dense_swa_indices",
    "dsv4_combine_topk_swa_indices",
    "dsv4_compact_compressed_slot_mapping",
    "dsv4_compressed_slot_mapping",
    "dsv4_compute_global_topk_indices_and_lens",
    "dsv4_decode_dense_compressed_indices_and_lens",
    "dsv4_decode_swa_indices_and_lens",
    "dsv4_dequantize_and_gather_k_cache",
    "dsv4_fused_csa_indexer_mxfp4_cache_insert",
    "dsv4_fused_indexer_q_rope_hadamard_mxfp4",
    "dsv4_fused_sparse_compress_cache_insert",
    "dsv4_gather_indexer_mxfp4_cache",
    "dsv4_group_slot_mapping",
    "dsv4_indexer_decode_metadata_compute",
    "dsv4_save_compressor_state",
    "dsv4_validate_active_cache_pages",
    "triton_dsv4_csa_indexer_fp8_cache_insert",
    "triton_dsv4_prefill",
    "triton_dsv4_swa_cache_insert",
    "write_dsv4_indexer_mxfp4_cache_cuda",
]
