# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Public surface for the SM100 CuTe-DSL sparse attention kernels."""

from __future__ import annotations

# FP4 block-score indexer.
# Returns per-(Hq, kv_block, q) max scores; topK selection + q2k construction
# remain caller-owned downstream steps.
from .fp4_indexer_interface import fp4_indexer_block_scores

# Sparse attention forward / decode.
from .interface import (
    SparseDecodePagedAttentionWrapper,
    sparse_atten_func,
    sparse_atten_nvfp4_kv_func,
    sparse_decode_atten_func,
)

# NVFP4 quantization helpers used to feed the FP4 indexer / NVFP4 attention
# (quantize.py).
from .quantize import (
    Nvfp4QuantizedTensor,
    dequantize_nvfp4_128x4_to_bf16,
    nvfp4_global_scale_from_amax,
    quantize_bf16_to_nvfp4_128x4,
    quantize_kv_bf16_to_nvfp4_128x4,
    swizzle_nvfp4_scale_to_128x4,
)

# CSR + schedule construction.
from .sparse_index_utils import build_k2q_csr

# SM100 fused CSR builder.
from .src.sm100.prepare_k2q_csr import SparseK2qCsrBuilderSm100

__all__ = [
    # attention
    "sparse_atten_func",
    "sparse_atten_nvfp4_kv_func",
    "sparse_decode_atten_func",
    "SparseDecodePagedAttentionWrapper",
    # indexing / CSR
    "fp4_indexer_block_scores",
    "build_k2q_csr",
    "SparseK2qCsrBuilderSm100",
    # nvfp4 quantization helpers
    "Nvfp4QuantizedTensor",
    "quantize_bf16_to_nvfp4_128x4",
    "quantize_kv_bf16_to_nvfp4_128x4",
    "dequantize_nvfp4_128x4_to_bf16",
    "swizzle_nvfp4_scale_to_128x4",
    "nvfp4_global_scale_from_amax",
]
