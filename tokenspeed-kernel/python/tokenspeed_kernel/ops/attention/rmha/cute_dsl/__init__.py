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

"""SM100 FMHA kernels with relative bias and MXFP8 (rel_mha).

Two CuteDSL kernels, their prepass helpers, and the runtime-facing rel_mha
operator modules, plus a compatibility layer over the installed
``tokenspeed-fa4`` distribution:

- ``flash_fwd_sm100_bias``: prefill/extend kernel — fused relative bias,
  MXFP8 Q/K, paged KV (page_size 128, interleaved UE8M0 scale layout).
- ``flash_fwd_sm100_bias_decode``: split-KV decode kernel — relative bias,
  MXFP8 Q/K with BF16 V, dense fixed-length KV.
- ``shearing_bias``: bias prepass — shears compact per-row relative
  logits into the padded tile layout the attention kernels consume
  (batch and varlen configurations).
- ``cu_blocks_kernels``: varlen prepass helpers — block prefix sums over
  ``cu_seqlens`` and the block-to-batch index map the varlen shear uses.
- ``flash_fwd_combine``: vendored split-KV combine kernel.
- ``fmha_bias_helper``: cutlass-dsl compatibility shims plus the APIs
  missing from the installed FA4 version; the kernels import everything
  FA4-related through it.
- ``rel_decode``/``rel_decode_v2``/``rel_extend``: operator wrappers the
  ops layer routes through (compile caches, paging glue, warmup hooks).

Importing this package requires ``tokenspeed-fa4`` (and cutlass-dsl);
``cute_dsl/__init__.py`` guards its re-exports for environments without it.
"""

from . import fmha_bias_helper  # isort: skip  (FA4 mount precedes the kernels)
from .cu_blocks_kernels import CuBlocksToBatchKernel, CuSeqlensToBlocksKernel
from .flash_fwd_combine import FlashAttentionForwardCombine
from .flash_fwd_sm100_bias import FlashAttentionForwardSm100
from .flash_fwd_sm100_bias_decode import (
    FlashAttentionDecodeSm100Bias,
    create_mxfp8_scale_factor_tensor,
)
from .shearing_bias import ShearingBias

from . import rel_decode, rel_decode_v2, rel_extend  # isort: skip

__all__ = [
    "CuBlocksToBatchKernel",
    "CuSeqlensToBlocksKernel",
    "FlashAttentionDecodeSm100Bias",
    "FlashAttentionForwardCombine",
    "FlashAttentionForwardSm100",
    "ShearingBias",
    "create_mxfp8_scale_factor_tensor",
    "fmha_bias_helper",
    "rel_decode",
    "rel_decode_v2",
    "rel_extend",
]
