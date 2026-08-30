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

"""DeepSeek V4 attention kernels for AMD GFX950."""

from tokenspeed_kernel_amd.ops.gfx950.attention.dsv4.decode import (
    gluon_dsv4_decode_split_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.dsv4.indexer import (
    gluon_dsv4_decode_topk_mxfp4_gfx950,
    gluon_dsv4_plan_gfx950,
    gluon_dsv4_prefill_topk_mxfp4_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.dsv4.prefill import (
    gluon_dsv4_prefill_gfx950,
)

__all__ = [
    "gluon_dsv4_decode_topk_mxfp4_gfx950",
    "gluon_dsv4_prefill_topk_mxfp4_gfx950",
    "gluon_dsv4_decode_split_gfx950",
    "gluon_dsv4_plan_gfx950",
    "gluon_dsv4_prefill_gfx950",
]
