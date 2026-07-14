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

"""Standalone Gluon bf16 MoE GEMM package for AMD MI350X (gfx950).

Two-stage unquantized (bf16 activation / bf16 weight) fused MoE in Gluon:

  Stage 1 (``stage1_kernel.py``): gate + up GEMM with fused SwiGLU.
  Stage 2 (``stage2_kernel.py``): down GEMM + routed-weight scale, written
     to a ``[num_tokens, topk, D]`` scratch, then reduced over topk.

  Block-align: ``moe_align_fused.py`` (decode, single-workgroup) and
     ``moe_align_device.py`` (prefill) produce the ``sorted_token_ids`` /
     ``sorted_expert_ids`` / ``sorted_weights`` contract the kernels consume.

  End-to-end (``moe.py``): ``gluon_bf16_moe(...)`` (decode split-K path for
     small M, XCD-remap prefill path otherwise).

Reference shape: DeepSeek-V3 (E=256, D=7168, I=256, topk=8).
"""

from tokenspeed_kernel_amd.ops.moe.gluon_bf16_moe.moe import gluon_bf16_moe
from tokenspeed_kernel_amd.ops.moe.gluon_bf16_moe.moe_align_device import (
    moe_align_block_size_device,
)
from tokenspeed_kernel_amd.ops.moe.gluon_bf16_moe.moe_align_fused import (
    moe_align_block_size_fused,
)
from tokenspeed_kernel_amd.ops.moe.gluon_bf16_moe.stage1_kernel import (
    gluon_bf16_moe_stage1_kernel,
    invoke_stage1,
)
from tokenspeed_kernel_amd.ops.moe.gluon_bf16_moe.stage1_splitk_kernel import (
    auto_split_k,
    invoke_stage1_splitk,
)
from tokenspeed_kernel_amd.ops.moe.gluon_bf16_moe.stage2_decode_kernel import (
    invoke_stage2_decode,
)
from tokenspeed_kernel_amd.ops.moe.gluon_bf16_moe.stage2_kernel import (
    gluon_bf16_moe_reduce_kernel,
    gluon_bf16_moe_stage2_kernel,
    invoke_stage2,
)

__all__ = [
    "gluon_bf16_moe",
    "invoke_stage1",
    "invoke_stage1_splitk",
    "auto_split_k",
    "invoke_stage2",
    "invoke_stage2_decode",
    "moe_align_block_size_device",
    "moe_align_block_size_fused",
    "gluon_bf16_moe_stage1_kernel",
    "gluon_bf16_moe_stage2_kernel",
    "gluon_bf16_moe_reduce_kernel",
]
