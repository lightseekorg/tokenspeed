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

Adapted from sglang's S-LoRA / Punica style kernels.  Each batch is a
sequence of segments where each segment uses a single adapter; the kernels
fuse the per-segment GEMMs into a single launch and keep per-segment state
(rank, scaling) on-device.
"""

from tokenspeed.runtime.lora.triton_ops.qkv_lora_b import qkv_lora_b_fwd
from tokenspeed.runtime.lora.triton_ops.sgemm_lora_a import sgemm_lora_a_fwd
from tokenspeed.runtime.lora.triton_ops.sgemm_lora_b import sgemm_lora_b_fwd

__all__ = ["sgemm_lora_a_fwd", "sgemm_lora_b_fwd", "qkv_lora_b_fwd"]
