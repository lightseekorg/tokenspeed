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


"""Gluon A4W4 MoE kernels for AMD CDNA4/gfx950.

The package keeps the four production stages behind stage-specific modules:
prefill stage 1/2 and decode stage 1/2.  ``moe.py`` provides the small-M
end-to-end decode entry point, while ``routing.py`` exposes the fused decode
top-k helpers.
"""

from tokenspeed_kernel_amd.ops.moe.gluon_a4w4_gfx950.decode_stage1 import (
    invoke_stage1_mxfp4_mfma_decode_gluon,
)
from tokenspeed_kernel_amd.ops.moe.gluon_a4w4_gfx950.decode_stage2 import (
    invoke_stage2_mxfp4_mfma_decode_gluon,
)
from tokenspeed_kernel_amd.ops.moe.gluon_a4w4_gfx950.moe import gluon_mxfp4_moe_decode
from tokenspeed_kernel_amd.ops.moe.gluon_a4w4_gfx950.prefill_stage1 import (
    invoke_gluon_mxfp4_moe_stage1,
)
from tokenspeed_kernel_amd.ops.moe.gluon_a4w4_gfx950.prefill_stage2 import (
    invoke_gluon_mxfp4_moe_stage2_1x2,
)
from tokenspeed_kernel_amd.ops.moe.gluon_a4w4_gfx950.preprocess import (
    attach_prefill_aliases,
)
from tokenspeed_kernel_amd.ops.moe.gluon_a4w4_gfx950.routing import (
    invoke_sigmoid_bias_topk_route_gluon,
    invoke_softmax_topk_route_gluon,
)
from tokenspeed_kernel_amd.ops.moe.gluon_a4w4_gfx950.scale import (
    gather_package_cdna4_scale,
)

__all__ = [
    "attach_prefill_aliases",
    "gather_package_cdna4_scale",
    "gluon_mxfp4_moe_decode",
    "invoke_gluon_mxfp4_moe_stage1",
    "invoke_gluon_mxfp4_moe_stage2_1x2",
    "invoke_sigmoid_bias_topk_route_gluon",
    "invoke_softmax_topk_route_gluon",
    "invoke_stage1_mxfp4_mfma_decode_gluon",
    "invoke_stage2_mxfp4_mfma_decode_gluon",
]
