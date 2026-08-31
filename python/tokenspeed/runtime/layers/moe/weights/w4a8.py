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

"""INT4 group-128 pack-quantized MoE weights for compressed-tensors W4A8.

Checkpoint layout matches the mxint4 path (``weight_packed`` int32 + per-group
``weight_scale``), so the buffers are the same; only ``group_size`` differs
(128 vs 32). The Hopper apply kernel repacks them into CUTLASS nibbles.
"""

from __future__ import annotations

from tokenspeed.runtime.layers.moe.weights.mxint4 import create_mxint4_weight_pair

create_w4a8_weight_pair = create_mxint4_weight_pair

__all__ = ["create_w4a8_weight_pair"]
