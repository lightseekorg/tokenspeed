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

from tokenspeed_kernel.op_profile import OpProfile, wrap_op_for_profiling
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import KernelSpec

if current_platform().is_hopper_plus:
    from flash_mla import (
        flash_mla_sparse_fwd,
        flash_mla_with_kvcache,
        get_mla_metadata,
    )

    # Preserve the vendor's signature, optional arguments and return values.
    flash_mla_with_kvcache = wrap_op_for_profiling(
        flash_mla_with_kvcache,
        spec=KernelSpec(name="flashmla_decode", family="attention", mode="mla_decode"),
        profile=OpProfile(
            cost_rule="flashmla_decode.v1",
            scalar_args=("head_dim_v", "causal", "is_fp8_kvcache"),
            integer_tensor_args=("cache_seqlens",),
        ),
    )

    __all__ = [
        "flash_mla_sparse_fwd",
        "flash_mla_with_kvcache",
        "get_mla_metadata",
    ]
else:
    __all__ = []
