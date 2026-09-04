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

"""Per-kernel page geometry, in one place.

``kernel_page_size`` is the token span of one attention-kernel page — a property of the kernel implementation, not of the scheduler.
"""

# FlashMLA decode kernels read the KV cache with a fixed 64-token page stride.
FLASH_MLA_PAGE_SIZE = 64

# DeepGEMM fp8_paged_mqa_logits lays out sparse-decode KV as 64-token pages
# (block-split FP8 payload + FP32 scales).
DSA_SPARSE_PAGE_SIZE = 64

# DeepSeek V4 kernels consume compressed-KV pages spanning this many raw
# tokens (256 / compress_ratio rows per page). The V4 cache spec and the
# model's default prefix granularity both derive from it.
DEEPSEEK_V4_PAGE_SIZE = 256

# trtllm-gen paged MLA kernels support exactly these page sizes.
TRTLLM_MLA_SUPPORTED_PAGE_SIZES = (32, 64)

# tokenspeed-mla (CuteDSL) paged decode supports exactly these page sizes.
TOKENSPEED_MLA_SUPPORTED_PAGE_SIZES = (32, 64)

# Flexible paged kernel: a chosen default, not a hardware constant.
MLA_PAGE_SIZE = 64

# The MSA sparse-attention kernels register exactly one supported page size
# (see tokenspeed_kernel/ops/attention/msa.py: page_size {128}); this is a
# kernel constraint, not a chosen default.
MSA_PAGE_SIZE = 128
TRTLLM_MHA_PAGE_SIZE = 64

# Constrained kernels: default within the supported set.
TRTLLM_MLA_DEFAULT_PAGE_SIZE = 64
TOKENSPEED_MLA_DEFAULT_PAGE_SIZE = 64
