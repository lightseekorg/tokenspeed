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

"""PyTorch reference kernels: the ground truth kernel tests compare against.

Modules mirror ``tokenspeed_kernel.ops``: ``<family>.py``, and
``attention/<variant>.py`` for attention. Every reference registers with
``solution="torch"`` so the numerics and benchmark tools find it. A reference
registers under an operator's mode only when it takes the arguments that
operator's dispatcher passes its kernels; one with a simpler interface (dense
instead of paged K/V, dequantized weights) registers under its own mode
naming what it computes, such as ``moe.experts``, so no dispatcher can select
it. References register at ``Priority.REFERENCE`` unless they are an
operator's intended portable fallback. Helpers that are not references
themselves (dequantizers, rounding models) are plain functions here.

Reference modules import only torch, the registry, signatures and each other,
never ``tokenspeed_kernel.ops``, so importing this package has no cycles and
needs no GPU.
"""

import tokenspeed_kernel.numerics.torch.activation  # noqa: F401
import tokenspeed_kernel.numerics.torch.attention.dsa  # noqa: F401
import tokenspeed_kernel.numerics.torch.attention.dsv4  # noqa: F401
import tokenspeed_kernel.numerics.torch.attention.dsv41  # noqa: F401
import tokenspeed_kernel.numerics.torch.attention.gdn  # noqa: F401
import tokenspeed_kernel.numerics.torch.attention.kda  # noqa: F401
import tokenspeed_kernel.numerics.torch.attention.kpool  # noqa: F401
import tokenspeed_kernel.numerics.torch.attention.mamba2  # noqa: F401
import tokenspeed_kernel.numerics.torch.attention.mha  # noqa: F401
import tokenspeed_kernel.numerics.torch.attention.mla  # noqa: F401
import tokenspeed_kernel.numerics.torch.attention.msa  # noqa: F401
import tokenspeed_kernel.numerics.torch.attention.prologue  # noqa: F401
import tokenspeed_kernel.numerics.torch.attention.qsa  # noqa: F401
import tokenspeed_kernel.numerics.torch.attention.rmha  # noqa: F401
import tokenspeed_kernel.numerics.torch.attention.tree  # noqa: F401
import tokenspeed_kernel.numerics.torch.communication  # noqa: F401
import tokenspeed_kernel.numerics.torch.conv  # noqa: F401
import tokenspeed_kernel.numerics.torch.embedding  # noqa: F401
import tokenspeed_kernel.numerics.torch.gemm  # noqa: F401
import tokenspeed_kernel.numerics.torch.kvcache  # noqa: F401
import tokenspeed_kernel.numerics.torch.layernorm  # noqa: F401
import tokenspeed_kernel.numerics.torch.metadata  # noqa: F401
import tokenspeed_kernel.numerics.torch.moe  # noqa: F401
import tokenspeed_kernel.numerics.torch.ple  # noqa: F401
import tokenspeed_kernel.numerics.torch.quantization  # noqa: F401
import tokenspeed_kernel.numerics.torch.residual  # noqa: F401
import tokenspeed_kernel.numerics.torch.sampling  # noqa: F401
import tokenspeed_kernel.numerics.torch.transform  # noqa: F401
