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

from __future__ import annotations

# Backend registration (side-effect imports)
import tokenspeed_kernel.ops.attention.triton.dsa  # noqa: F401
import tokenspeed_kernel.ops.attention.triton.dsa_topk  # noqa: F401
import tokenspeed_kernel.ops.attention.triton.gated_delta_rule  # noqa: F401
import tokenspeed_kernel.ops.attention.triton.merge_state  # noqa: F401
import tokenspeed_kernel.ops.attention.triton.mha_decode  # noqa: F401
import tokenspeed_kernel.ops.attention.triton.mha_prefill  # noqa: F401
import tokenspeed_kernel.ops.attention.triton.minimax_sparse_attention  # noqa: F401
import tokenspeed_kernel.ops.attention.triton.mla_decode  # noqa: F401
import tokenspeed_kernel.ops.attention.triton.mla_prefill  # noqa: F401
import tokenspeed_kernel.ops.attention.triton.rel_mha  # noqa: F401
