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

from tokenspeed_kernel.profiling import bootstrap_profiling_from_env

bootstrap_profiling_from_env()

from tokenspeed_kernel.ops.activation import add3, situ_and_mul
from tokenspeed_kernel.ops.attention import (
    GdnCheckpointLayout,
    GdnChunkPrefillResult,
    MLAQueryProjection,
    attn_merge_state,
    dsa_decode,
    dsa_decode_topk,
    dsa_plan,
    dsa_prefill,
    dsa_prefill_topk,
    gdn_chunk_prefill,
    gdn_decode_mtp,
    gdn_decode_step,
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_plan,
    mha_prefill,
    mla_decode_with_kvcache,
    mla_extend_with_kvcache,
    mla_normalize_project_query,
    mla_prefill,
    mla_project_value,
    mla_use_absorbed_extend,
    msa_decode_with_kvcache,
    msa_extend_with_kvcache,
    rel_mha_decode_with_kvcache,
    rel_mha_extend_with_kvcache,
    rel_mha_plan,
    rel_mha_prefill,
)
from tokenspeed_kernel.ops.gemm import (
    bmm,
    kimi3_latent_projection,
    kimi3_latent_projection_add3,
    kimi3_mla_qkv_gate_projection,
    kimi3_qkvfab_projection,
    kimi3_router_projection,
    kimi3_shared_down_projection,
    kimi3_shared_situ_projection,
    mm,
    rmsnorm_linear_add,
)
from tokenspeed_kernel.ops.moe import (
    moe_apply,
    moe_plan,
    moe_process_weights,
    moe_sigmoid_bias_topk,
    moe_softmax_topk,
    native_latent_moe_available,
)
from tokenspeed_kernel.ops.quantization import (
    quantize_fp8,
    quantize_fp8_with_scale,
    quantize_mxfp4,
    quantize_mxfp8,
    quantize_nvfp4,
)
from tokenspeed_kernel.ops.sampling import argmax
from tokenspeed_kernel.ops.transform import hadamard_transform
from tokenspeed_kernel.selection import NoKernelFoundError

__all__ = [
    # exceptions
    "NoKernelFoundError",
    # gemm
    "bmm",
    "kimi3_latent_projection",
    "kimi3_mla_qkv_gate_projection",
    "kimi3_latent_projection_add3",
    "kimi3_qkvfab_projection",
    "kimi3_router_projection",
    "kimi3_shared_down_projection",
    "kimi3_shared_situ_projection",
    "mm",
    "rmsnorm_linear_add",
    # attention
    "mha_plan",
    "mha_prefill",
    "mha_extend_with_kvcache",
    "mha_decode_with_kvcache",
    "rel_mha_prefill",
    "rel_mha_extend_with_kvcache",
    "rel_mha_decode_with_kvcache",
    "rel_mha_plan",
    "mla_prefill",
    "mla_extend_with_kvcache",
    "mla_decode_with_kvcache",
    "mla_use_absorbed_extend",
    "mla_normalize_project_query",
    "mla_project_value",
    "MLAQueryProjection",
    "dsa_prefill",
    "dsa_decode",
    "dsa_prefill_topk",
    "dsa_decode_topk",
    "dsa_plan",
    "msa_decode_with_kvcache",
    "msa_extend_with_kvcache",
    "attn_merge_state",
    "gdn_chunk_prefill",
    "gdn_decode_step",
    "gdn_decode_mtp",
    "GdnCheckpointLayout",
    "GdnChunkPrefillResult",
    # activation
    "add3",
    "situ_and_mul",
    # moe
    "native_latent_moe_available",
    "moe_apply",
    "moe_plan",
    "moe_process_weights",
    "moe_sigmoid_bias_topk",
    "moe_softmax_topk",
    # quantization
    "quantize_fp8",
    "quantize_fp8_with_scale",
    "quantize_mxfp8",
    "quantize_nvfp4",
    "quantize_mxfp4",
    # sampling
    "argmax",
    # transform
    "hadamard_transform",
]
