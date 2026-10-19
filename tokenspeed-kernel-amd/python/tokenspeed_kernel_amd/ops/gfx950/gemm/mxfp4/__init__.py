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

"""Dense MXFP4 (dynamic MXFP4 activation) linears for gfx950."""

from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp4.mm import (
    gluon_mxfp4_linear_num_counters_gfx950,
    launch_gluon_mxfp4_linear_gfx950,
    launch_gluon_mxfp4_swiglu_linear_gfx950,
    preshuffle_mxfp4_weight,
)

__all__ = [
    "gluon_mxfp4_linear_num_counters_gfx950",
    "launch_gluon_mxfp4_linear_gfx950",
    "launch_gluon_mxfp4_swiglu_linear_gfx950",
    "preshuffle_mxfp4_weight",
]
