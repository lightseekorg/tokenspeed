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

"""AMD GEMM kernels."""

from __future__ import annotations

from tokenspeed_kernel_amd.ops.gemm.mm_a16w16_gfx950 import (  # noqa: F401
    gluon_mm_a16w16_gfx950,
    gluon_mm_a16w16_mfma_lds_mediumm_gfx950,
    gluon_mm_a16w16_mfma_lds_smallm_gfx950,
    gluon_mm_a16w16_warp_reduce_smallm_gfx950,
)
from tokenspeed_kernel_amd.ops.gemm.mm_a16w16_largem_gfx950 import (  # noqa: F401
    gluon_mm_a16w16_largem_gfx950,
)
