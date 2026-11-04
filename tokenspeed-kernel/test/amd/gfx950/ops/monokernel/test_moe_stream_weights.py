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

"""CPU test of the monokernel MoE stream's host weight contract.

``moe_stream_weights`` must read a processed MoE layer (built with the
production weight prep) in place, with the shapes and strides the codegen
test compiles the stream for.
"""

import pytest
import torch
from moe_stream_helpers import DIVISIBLE, PHYS, make_experts
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import (
    _extract_gluon_raw_s,
    _extract_gluon_raw_w,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale_layout import (
    pack_cdna4_mxfp4_scale_lanes,
    swizzle_cdna4_mxfp4_scale,
    unswizzle_cdna4_mxfp4_scale,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.moe_stream import (
    moe_stream_weights,
)

EXPERTS = 4


@pytest.fixture(scope="module")
def experts():
    return make_experts(EXPERTS, "cpu", 7)


def test_weights_are_read_in_place(experts):
    module, _ = experts
    weights = moe_stream_weights(module, scale_lanes=False)
    w13 = _extract_gluon_raw_w(module.w13_weight_triton_tensor)
    s13 = _extract_gluon_raw_s(module.w13_precision_config.b_mx_scale)
    w2 = _extract_gluon_raw_w(module.w2_weight_triton_tensor)
    s2 = _extract_gluon_raw_s(module.w2_precision_config.b_mx_scale)
    assert weights.w13.data_ptr() == w13.data_ptr()
    assert weights.w2.data_ptr() == w2.data_ptr()
    assert weights.s13.data_ptr() == s13.data_ptr()
    assert weights.s2.data_ptr() == s2.data_ptr()
    assert weights.s13.dtype == weights.s2.dtype == torch.int32
    assert weights.s13_lanes is None
    assert (weights.n_phys1, weights.n_phys2) == (PHYS["N_PHYS1"], PHYS["N_PHYS2"])
    assert (weights.stride_s13lin, weights.stride_s2lin) == (1, 1)
    for name in DIVISIBLE - {"stride_xsnb"}:
        assert getattr(weights, name) % 16 == 0, name


def test_lane_scales_come_from_the_swizzled_scales(experts):
    module, w13_scale = experts
    weights = moe_stream_weights(module, scale_lanes=True)
    expected = pack_cdna4_mxfp4_scale_lanes(w13_scale)
    lanes = weights.s13_lanes.view(torch.uint8)[: expected.numel()]
    assert torch.equal(lanes, expected.reshape(-1))


def test_unswizzle_inverts_the_swizzle():
    generator = torch.Generator().manual_seed(3)
    for shape in ((2, 1024, 128), (2, 4096, 16), (1, 48, 20)):
        scale = torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)
        swizzled = swizzle_cdna4_mxfp4_scale(scale)
        assert torch.equal(unswizzle_cdna4_mxfp4_scale(swizzled, *shape[1:]), scale)


def test_rejects_other_layouts(experts):
    module, _ = experts
    narrow = torch.nn.Module()
    narrow.w13_weight_triton_tensor = _extract_gluon_raw_w(
        module.w13_weight_triton_tensor
    )[:, :, :512]
    narrow.w13_precision_config = module.w13_precision_config
    narrow.w2_weight_triton_tensor = module.w2_weight_triton_tensor
    narrow.w2_precision_config = module.w2_precision_config
    with pytest.raises(ValueError, match="gate/up weights"):
        moe_stream_weights(narrow, scale_lanes=False)
