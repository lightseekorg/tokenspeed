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

"""Vendored TensorRT-LLM CuTe DSL single-pass multi-CTA radix top-k kernels.

The kernel bodies under :mod:`top_k` and :mod:`utils` are copied unchanged from
NVIDIA TensorRT-LLM (Apache-2.0). :mod:`runner` carries the class-level runners
that compile and launch them, with a self-contained scratch arena replacing the
TensorRT-LLM native memory-buffer utility.
"""

from .runner import (
    CuteDSLTopKDecodeSinglePassMultiCTAClusterRunner,
    CuteDSLTopKDecodeSinglePassMultiCTARunner,
)

__all__ = [
    "CuteDSLTopKDecodeSinglePassMultiCTARunner",
    "CuteDSLTopKDecodeSinglePassMultiCTAClusterRunner",
]
