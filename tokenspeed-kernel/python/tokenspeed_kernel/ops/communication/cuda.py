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

"""NVIDIA CUDA communication kernels with caller-managed persistent state.

The Lamport A2A APIs exchange TP4 BF16 channel shards directly, optionally
quantizing received 128-element groups for a prepared FP8 GEMM. Callers must
construct the state collectively before CUDA-graph capture, serialize calls
and consumers on one stream, pad empty owners so every peer participates, and
select any fallback outside this module. Returned buffers are borrowed until
the next call unless an explicit destination is supplied.
"""

from tokenspeed_kernel.ops.communication._cuda.lamport_a2a import (
    TokenSpeedA2ALamportState,
    tokenspeed_a2a_lamport,
    tokenspeed_a2a_lamport_fp8_quantize,
)

__all__ = [
    "TokenSpeedA2ALamportState",
    "tokenspeed_a2a_lamport",
    "tokenspeed_a2a_lamport_fp8_quantize",
]
