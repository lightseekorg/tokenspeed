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

"""Optional FlashInfer SM120 MLA entry; keep the vendor import here."""

from functools import cache

import torch


@cache
def sm120_bf16_mla_runner(device_index: int, stream_id: int, has_global: bool):
    # The wrapper owns scratch. Separate stream identities must never share it.
    # Construction is lazy so importing the kernel registry needs no FlashInfer.
    from flashinfer.mla import SparseMLASm120Wrapper

    return SparseMLASm120Wrapper(
        d_v=512,
        kv_scale_format="ue8m0_g32",
        kv_cache_format="fp8",
        extra_kv_fp4=has_global,
        compute_precision="bf16",
        device=torch.device("cuda", device_index),
        backend="sparse",
    )
