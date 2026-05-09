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

"""tokenspeed_kernel third-party CUDA kernel wrappers."""

from tokenspeed_kernel.thirdparty.cuda.activation import (
    silu_and_mul_fuse_block_quant,
    silu_and_mul_fuse_nvfp4_quant,
)
from tokenspeed_kernel.thirdparty.cuda.dsv3_gemm import dsv3_router_gemm
from tokenspeed_kernel.thirdparty.cuda.fp32_router_gemm import fp32_router_gemm
from tokenspeed_kernel.thirdparty.cuda.marlin import gptq_marlin_repack
from tokenspeed_kernel.thirdparty.cuda.moe import moe_finalize_fuse_shared
from tokenspeed_kernel.thirdparty.cuda.rmsnorm import rmsnorm_fused_parallel
from tokenspeed_kernel.thirdparty.cuda.rope import apply_rope_with_cos_sin_cache_inplace
from tokenspeed_kernel.thirdparty.cuda.routing import (
    hash_softplus_sqrt_topk_flash,
    routing_flash,
    softplus_sqrt_topk_flash,
)
from tokenspeed_kernel.thirdparty.cuda.sampling_chain import (
    chain_speculative_sampling_target_only,
    verify_chain_greedy,
)

__all__ = [
    "apply_rope_with_cos_sin_cache_inplace",
    "chain_speculative_sampling_target_only",
    "dsv3_router_gemm",
    "fp32_router_gemm",
    "gptq_marlin_repack",
    "hash_softplus_sqrt_topk_flash",
    "moe_finalize_fuse_shared",
    "rmsnorm_fused_parallel",
    "routing_flash",
    "silu_and_mul_fuse_block_quant",
    "silu_and_mul_fuse_nvfp4_quant",
    "softplus_sqrt_topk_flash",
    "verify_chain_greedy",
]
