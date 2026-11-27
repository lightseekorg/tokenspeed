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

"""Block-size constants shared by sampling dispatch and Triton kernels."""

from __future__ import annotations

# Columns of one block of ``selected_token_logprobs``'s softmax denominator:
# a row's denominator adds its block sums in order, which the shard form
# (:func:`vocab_shard_logprob_blocks`) keeps.
SELECTED_LOGPROBS_BLOCK = 1024

# Columns of a shard each draft-stats block reduces, and the fp32 partials it writes.
VOCAB_SHARD_DRAFT_BLOCK = 2048
VOCAB_SHARD_DRAFT_FIELDS = 8

__all__ = [
    "SELECTED_LOGPROBS_BLOCK",
    "VOCAB_SHARD_DRAFT_BLOCK",
    "VOCAB_SHARD_DRAFT_FIELDS",
]
