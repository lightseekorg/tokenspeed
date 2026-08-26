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

"""Fused single-launch Gluon MXFP4 MoE family for gfx950 (CDNA4).

Every approach in this package fuses routing/GEMM/activation/quantization into
as few kernel launches as possible and communicates through
``RaggedTensorMetadata`` plus gather/scatter index tensors:

* ``moe``: model-facing fused-MoE entry points and the dispatch policy.
* ``gemm_api`` / ``launch`` / ``tuning``: per-GEMM entries over the pipelined
  ragged GEMM, its launch marshaling, and its block heuristics.
* ``pipelined_program`` / ``pipelined_kernel``: the pipelined ragged GEMM
  device code (program aggregates, tile runners, kernel entry).
* ``medium_decode`` / ``warp_decode``: the M=8/16 direct-load body and the
  M<=4 two-stage warp-decode kernels.
* ``routing``: single-launch route kernels producing full ragged metadata.
* ``quantize``: FP8 and dynamic-MXFP4 activation quantization.

The staged "package" pipeline (separate launches per stage, precomputed top-k
between them) lives directly under ``mxfp4/``; see ``mxfp4/README.md`` for the
split.
"""

# Importing _common first sets AMDGCN_COALESCE_BUFFER_LOAD_I8 before any
# kernel module triggers Gluon compilation.
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused._common import (
    _extract_gluon_raw_s,
    _extract_gluon_raw_w,
    _extract_gluon_raw_w_unshuffled,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.gemm_api import (
    gluon_mxfp_combine,
    gluon_mxfp_dispatch_swiglu,
    gluon_mxfp_ragged_matmul,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.launch import (
    assert_no_spills,
    last_kernel_profile,
    static_profile,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.moe import (
    gluon_mxfp4_fp8_precomputed_situ,
    gluon_mxfp_dynamic_mxfp4_fused_moe,
    gluon_mxfp_fused_moe,
    gluon_mxfp_precomputed_mxfp4_fused_moe,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.quantize import (
    _quantize_mxfp4_activation,
    fp8_quantize,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.routing import (
    FUSED_ROUTE_MAX_M,
    GLUON_ROUTE_DTYPES,
    GLUON_ROUTE_MAX_E,
    GLUON_ROUTE_MAX_G,
    SMALLM_MAX_M,
    _biased_grouped_topk_reference,
    default_biased_grouped_route,
    default_biased_route,
    default_grouped_route,
    default_packed_topk_route,
    default_route,
    default_scaled_route,
    gluon_biased_grouped_fused_route,
    gluon_biased_grouped_route_supported,
    gluon_fused_route,
    gluon_precomputed_topk_flat_m1_route,
    gluon_precomputed_topk_fused_route,
    gluon_precomputed_topk_route_supported,
    gluon_route_supported,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.warp_decode import (
    WARP_DECODE_MAX_M,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale_layout import (
    MXFP4_BLOCK,
)

__all__ = [
    # Compatibility surface consumed by mxfp4/moe.py, weight preprocessing,
    # and the unit tests.
    "_biased_grouped_topk_reference",
    "_extract_gluon_raw_s",
    "_extract_gluon_raw_w",
    "_extract_gluon_raw_w_unshuffled",
    "_quantize_mxfp4_activation",
    "FUSED_ROUTE_MAX_M",
    "GLUON_ROUTE_DTYPES",
    "GLUON_ROUTE_MAX_E",
    "GLUON_ROUTE_MAX_G",
    "MXFP4_BLOCK",
    "SMALLM_MAX_M",
    "WARP_DECODE_MAX_M",
    "assert_no_spills",
    "default_biased_grouped_route",
    "default_biased_route",
    "default_grouped_route",
    "default_packed_topk_route",
    "default_route",
    "default_scaled_route",
    "fp8_quantize",
    "gluon_biased_grouped_fused_route",
    "gluon_biased_grouped_route_supported",
    "gluon_fused_route",
    "gluon_mxfp_combine",
    "gluon_mxfp_dispatch_swiglu",
    "gluon_mxfp4_fp8_precomputed_situ",
    "gluon_mxfp_dynamic_mxfp4_fused_moe",
    "gluon_mxfp_fused_moe",
    "gluon_mxfp_precomputed_mxfp4_fused_moe",
    "gluon_mxfp_ragged_matmul",
    "gluon_precomputed_topk_flat_m1_route",
    "gluon_precomputed_topk_fused_route",
    "gluon_precomputed_topk_route_supported",
    "gluon_route_supported",
    "last_kernel_profile",
    "static_profile",
]
