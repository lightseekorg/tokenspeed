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


"""CPU test of the gfx950 MXFP4 MoE stream's host weight contract:
``moe_stream_weights`` reads a processed MoE layer (built with the production
weight prep) in place, with the shapes and strides the codegen test compiles
the kernel for, and rejects other layouts."""

import pytest
import torch

pytest.importorskip("tokenspeed_kernel_amd", reason="AMD kernel package is optional")

from moe_stream_helpers import DIVISIBLE, PHYS, make_experts  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import (  # noqa: E402
    _extract_gluon_raw_s,
    _extract_gluon_raw_w,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.moe_stream import (  # noqa: E402
    moe_stream_weights,
)

EXPERTS = 4


@pytest.fixture(scope="module")
def experts():
    return make_experts(EXPERTS, "cpu", 7)


def _layer(w13, s13, w2, s2):
    # A layer exposing the four processed tensors the direct decode reads.
    module = torch.nn.Module()
    module.w13_weight_triton_tensor = w13
    module.w13_precision_config = s13
    module.w2_weight_triton_tensor = w2
    module.w2_precision_config = s2
    return module


def test_weights_are_read_in_place(experts):
    weights = moe_stream_weights(experts)
    w13 = _extract_gluon_raw_w(experts.w13_weight_triton_tensor)
    s13 = _extract_gluon_raw_s(experts.w13_precision_config.b_mx_scale)
    w2 = _extract_gluon_raw_w(experts.w2_weight_triton_tensor)
    s2 = _extract_gluon_raw_s(experts.w2_precision_config.b_mx_scale)
    assert weights.w13.data_ptr() == w13.data_ptr()
    assert weights.w2.data_ptr() == w2.data_ptr()
    assert weights.s13.data_ptr() == s13.data_ptr()
    assert weights.s2.data_ptr() == s2.data_ptr()
    assert weights.s13.dtype == weights.s2.dtype == torch.uint8
    assert (weights.n_phys1, weights.n_phys2) == (PHYS["N_PHYS1"], PHYS["N_PHYS2"])
    assert (weights.stride_s13lin, weights.stride_s2lin) == (1, 1)
    for name in DIVISIBLE - {"stride_xq", "stride_xsnb"}:
        assert getattr(weights, name) % 16 == 0, name


def test_rejects_a_narrower_gate_up_slice(experts):
    narrow = _layer(
        _extract_gluon_raw_w(experts.w13_weight_triton_tensor)[:, :, :512],
        experts.w13_precision_config,
        experts.w2_weight_triton_tensor,
        experts.w2_precision_config,
    )
    with pytest.raises(ValueError, match="gate/up weights"):
        moe_stream_weights(narrow)


def test_rejects_unshuffled_weights(experts):
    # The same bytes without the gdot128 tile order the kernel addresses.
    plain = _layer(
        _extract_gluon_raw_w(experts.w13_weight_triton_tensor).clone(),
        experts.w13_precision_config,
        experts.w2_weight_triton_tensor,
        experts.w2_precision_config,
    )
    with pytest.raises(ValueError, match="gdot128-shuffled"):
        moe_stream_weights(plain)


def _shuffled_meta(shape):
    # A gdot128-shuffled weight of `shape` with no storage, for sizes the
    # host check must refuse without allocating them.
    weight = torch.empty(shape, dtype=torch.uint8, device="meta")
    weight.is_shuffled_for_gluon_dot = True
    weight.gluon_dot_block_k_pk = 128
    weight.gluon_dot_block_n = 128
    return weight


@pytest.mark.parametrize("num_experts, fits", [(1023, True), (1024, False)])
def test_refuses_weights_past_the_int32_buffer_window(experts, num_experts, fits):
    # 1024 gate/up experts span 2**31 bytes, past the 0x7FFFFFF0 bytes the
    # kernel's int32 offsets and buffer descriptors reach.
    w13 = _extract_gluon_raw_w(experts.w13_weight_triton_tensor)
    w2 = _extract_gluon_raw_w(experts.w2_weight_triton_tensor)
    layer = _layer(
        _shuffled_meta((num_experts, *w13.shape[1:])),
        experts.w13_precision_config,
        _shuffled_meta((num_experts, *w2.shape[1:])),
        experts.w2_precision_config,
    )
    if fits:
        moe_stream_weights(layer)
    else:
        with pytest.raises(ValueError, match="gate/up weights must span"):
            moe_stream_weights(layer)
