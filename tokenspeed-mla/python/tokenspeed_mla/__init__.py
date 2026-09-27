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

"""Speed-of-light TokenSpeed MLA kernels for Blackwell SM100 and SM103."""

try:
    from tokenspeed_mla.mla_decode import (
        tokenspeed_mla_decode,
    )
    from tokenspeed_mla.mla_kv_pack_quantize_fp8 import (
        mla_kv_pack_quantize_fp8,
    )
    from tokenspeed_mla.mla_prefill import (
        tokenspeed_mla_prefill,
        warmup_compile_prefill,
    )
    from tokenspeed_mla.utils import get_num_sm
except ImportError as exc:
    _IMPORT_ERROR = exc

    def _unavailable(*args, **kwargs):
        raise ImportError(
            "tokenspeed_mla requires PyTorch, CUDA bindings, and NVIDIA CuTe DSL "
            "runtime dependencies."
        ) from _IMPORT_ERROR

    tokenspeed_mla_decode = _unavailable
    tokenspeed_mla_prefill = _unavailable
    mla_kv_pack_quantize_fp8 = _unavailable
    get_num_sm = _unavailable
    warmup_compile_prefill = _unavailable

__all__ = [
    "tokenspeed_mla_decode",
    "tokenspeed_mla_prefill",
    "mla_kv_pack_quantize_fp8",
    "get_num_sm",
    "warmup_compile_prefill",
]
