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

from tokenspeed_kernel.profiling import bootstrap_profiling_from_env

bootstrap_profiling_from_env()

from tokenspeed_kernel.ops.attention import (
    mla_decode_with_kvcache,
    mha_decode_with_kvcache,
    mha_prefill,
    mha_prefill_with_kvcache,
)
from tokenspeed_kernel.ops.gemm import mm
from tokenspeed_kernel.ops.moe import (
    moe_combine,
    moe_dispatch,
    moe_experts,
    moe_fused,
    moe_route,
)

__all__ = [
    "mm",
    "moe_route",
    "moe_dispatch",
    "moe_experts",
    "moe_combine",
    "moe_fused",
    "mha_prefill",
    "mha_prefill_with_kvcache",
    "mha_decode_with_kvcache",
    "mla_decode_with_kvcache",
]
