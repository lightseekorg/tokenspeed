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

"""Triton kernels for segment-grouped LoRA matmuls.

Adapted from sglang ``python/sglang/srt/lora/triton_ops/`` (Apache-2.0):
https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/lora/triton_ops.
sglang's kernels in turn descend from the Punica S-LoRA design
(https://github.com/punica-ai/punica).  Each batch is a sequence of
segments where each segment uses a single adapter; the kernels fuse the
per-segment GEMMs into a single launch and keep per-segment state
(rank, scaling) on-device.  See each kernel module for file-level
provenance.
"""

from tokenspeed_kernel.ops.lora.triton.lora_expand import lora_expand_fwd
from tokenspeed_kernel.ops.lora.triton.lora_expand_grouped_v2 import (
    lora_expand_grouped_v2_fwd,
)
from tokenspeed_kernel.ops.lora.triton.lora_expand_prefill import (
    lora_expand_prefill_fwd,
)
from tokenspeed_kernel.ops.lora.triton.lora_gate_up_expand import (
    lora_gate_up_expand_fwd,
)
from tokenspeed_kernel.ops.lora.triton.lora_qkv_expand import lora_qkv_expand_fwd
from tokenspeed_kernel.ops.lora.triton.lora_shrink import lora_shrink_fwd
from tokenspeed_kernel.ops.lora.triton.lora_shrink_prefill import (
    lora_shrink_prefill_fwd,
)

__all__ = [
    "lora_shrink_fwd",
    "lora_shrink_prefill_fwd",
    "lora_expand_fwd",
    "lora_expand_grouped_v2_fwd",
    "lora_qkv_expand_fwd",
    "lora_gate_up_expand_fwd",
    "lora_expand_prefill_fwd",
]
