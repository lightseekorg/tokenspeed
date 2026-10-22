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

"""Ascend communication buffer preparation."""


def pad_token_shards(tensor, counts):
    """Pack [sum(counts), hidden] rows into equal zero-padded rank shards.

    Args:
        tensor: Rank-major source rows on an NPU.
        counts: Live row count of each rank, in collective group order.

    Returns:
        A contiguous tensor of shape [len(counts) * max(counts), hidden].
    """
    from tokenspeed_kernel_npu.ops.token_shards import pad_token_shards as pack

    return pack(tensor, counts)


def sum_token_shards(tensor, shards):
    """Sum rank-major FP32 [shards*rows,hidden] into BF16 [rows,hidden].

    Args:
        tensor: Contiguous gathered partial projections on an NPU.
        shards: Number of equal shards, in collective group order.

    Returns:
        The FP32 sum, rounded once to a contiguous BF16 tensor.
    """
    from tokenspeed_kernel_npu.ops.token_shards import sum_token_shards as reduce

    return reduce(tensor, shards)
