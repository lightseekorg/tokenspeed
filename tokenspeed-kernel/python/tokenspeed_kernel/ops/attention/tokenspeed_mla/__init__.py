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

from functools import wraps

from tokenspeed_kernel.ops.attention.tokenspeed_mla.fallback import (
    mla_kv_pack_quantize_fp8 as _fallback_mla_kv_pack_quantize_fp8,
)
from tokenspeed_kernel.platform import current_platform, pdl_enabled
from tokenspeed_kernel.registry import error_fn


def _with_pdl_default(kernel, enable_pdl_position):
    @wraps(kernel)
    def wrapped(*args, **kwargs):
        if len(args) <= enable_pdl_position and "enable_pdl" not in kwargs:
            kwargs["enable_pdl"] = pdl_enabled()
        return kernel(*args, **kwargs)

    return wrapped


get_num_sm = error_fn
tokenspeed_mla_decode = error_fn
tokenspeed_mla_prefill = error_fn
warmup_compile_prefill = error_fn
mla_kv_pack_quantize_fp8 = _fallback_mla_kv_pack_quantize_fp8

if current_platform().is_cdna4:
    from tokenspeed_kernel_amd.ops.gfx950.attention.mla.kv_pack import (
        gluon_mla_kv_pack_quantize_fp8_gfx950 as mla_kv_pack_quantize_fp8,
    )
elif current_platform().is_nvidia:
    try:
        from tokenspeed_mla import (
            get_num_sm,
        )
        from tokenspeed_mla import mla_kv_pack_quantize_fp8 as _mla_kv_pack_quantize_fp8
        from tokenspeed_mla import tokenspeed_mla_decode as _tokenspeed_mla_decode
        from tokenspeed_mla import tokenspeed_mla_prefill as _tokenspeed_mla_prefill
        from tokenspeed_mla import warmup_compile_prefill as _warmup_compile_prefill
    except ImportError:
        pass
    else:
        mla_kv_pack_quantize_fp8 = _with_pdl_default(_mla_kv_pack_quantize_fp8, 8)
        tokenspeed_mla_decode = _with_pdl_default(_tokenspeed_mla_decode, 13)
        tokenspeed_mla_prefill = _with_pdl_default(_tokenspeed_mla_prefill, 12)
        warmup_compile_prefill = _with_pdl_default(_warmup_compile_prefill, 3)

__all__ = [
    "get_num_sm",
    "mla_kv_pack_quantize_fp8",
    "tokenspeed_mla_decode",
    "tokenspeed_mla_prefill",
    "warmup_compile_prefill",
]
