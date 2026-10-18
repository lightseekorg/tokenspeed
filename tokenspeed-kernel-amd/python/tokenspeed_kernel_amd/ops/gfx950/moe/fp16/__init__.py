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

"""Standalone Gluon 16-bit floating-point MoE package for gfx950.

The current implementation specializes BF16 activations and weights. It uses a
two-stage unquantized fused MoE in Gluon:

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

from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.moe import gluon_bf16_moe

__all__ = ["gluon_bf16_moe"]
