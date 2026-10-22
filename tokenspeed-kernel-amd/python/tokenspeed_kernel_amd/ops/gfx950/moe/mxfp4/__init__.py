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


"""Gluon MXFP4-weight MoE kernels for AMD CDNA4/gfx950.

The package contains BF16, FP8, and dynamically quantized MXFP4 activation
paths. It keeps the production stages behind stage-specific modules: prefill
stage 1/2 and decode stage 1/2, while ``routing.py`` exposes the fused dense
top-k helpers used by decode and package prefill.

The ``fused/`` subpackage holds the fused single-launch MoE family (pipelined
ragged GEMM, warp/medium decode, ragged-metadata routing, activation
quantization, and the model-facing dispatch policy). See ``README.md`` for
how the staged files here and ``fused/`` relate.
"""

from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.preprocess import (
    attach_prefill_aliases,
)

__all__ = ["attach_prefill_aliases"]
