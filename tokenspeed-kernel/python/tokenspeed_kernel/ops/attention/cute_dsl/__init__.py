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

"""CuTe DSL attention kernels (NVIDIA Blackwell)."""

from tokenspeed_kernel.ops.attention.cute_dsl.dsa_topk import (
    cute_dsl_decode_topk,
    has_cute_dsl_decode_topk,
)
from tokenspeed_kernel.registry import error_fn

try:
    from tokenspeed_kernel.ops.attention.cute_dsl import rel_mha
    from tokenspeed_kernel.ops.attention.cute_dsl.rel_mha import (
        CuBlocksToBatchKernel,
        CuSeqlensToBlocksKernel,
        FlashAttentionDecodeSm100Bias,
        FlashAttentionForwardCombine,
        FlashAttentionForwardSm100,
        ShearingBias,
        create_mxfp8_scale_factor_tensor,
        rel_decode,
        rel_decode_v2,
        rel_extend,
    )

    HAS_REL_MHA = True
except ImportError:
    # rel_mha needs tokenspeed-fa4 + cutlass-dsl; degrade instead of failing the import.
    rel_mha = None
    rel_decode = rel_decode_v2 = rel_extend = None
    CuBlocksToBatchKernel = error_fn
    CuSeqlensToBlocksKernel = error_fn
    FlashAttentionDecodeSm100Bias = error_fn
    FlashAttentionForwardCombine = error_fn
    FlashAttentionForwardSm100 = error_fn
    ShearingBias = error_fn
    create_mxfp8_scale_factor_tensor = error_fn
    HAS_REL_MHA = False

__all__ = [
    "CuBlocksToBatchKernel",
    "CuSeqlensToBlocksKernel",
    "FlashAttentionDecodeSm100Bias",
    "FlashAttentionForwardCombine",
    "FlashAttentionForwardSm100",
    "HAS_REL_MHA",
    "ShearingBias",
    "create_mxfp8_scale_factor_tensor",
    "cute_dsl_decode_topk",
    "has_cute_dsl_decode_topk",
    "rel_decode",
    "rel_decode_v2",
    "rel_extend",
    "rel_mha",
]
