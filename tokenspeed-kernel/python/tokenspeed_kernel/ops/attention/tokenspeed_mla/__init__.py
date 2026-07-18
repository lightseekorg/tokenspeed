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

"""TokenSpeed MLA kernels exposed through tokenspeed-kernel."""

from tokenspeed_kernel.registry import error_fn

try:
    from tokenspeed_mla import (
        get_num_sm,
        mla_kv_pack_quantize_fp8,
        tokenspeed_mla_decode,
        tokenspeed_mla_prefill,
        warmup_compile_prefill,
    )
except ImportError:
    get_num_sm = error_fn
    mla_kv_pack_quantize_fp8 = error_fn
    tokenspeed_mla_decode = error_fn
    tokenspeed_mla_prefill = error_fn
    warmup_compile_prefill = error_fn

__all__ = [
    "get_num_sm",
    "mla_kv_pack_quantize_fp8",
    "tokenspeed_mla_decode",
    "tokenspeed_mla_prefill",
    "warmup_compile_prefill",
]
