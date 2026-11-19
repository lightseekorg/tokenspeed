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

import sys
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel_npu.ops.w4a4 import (
    FlatQuantStage,
    W4A4MoePlan,
    pack_int4,
    prepare_flatquant_stage,
    prepare_w4a4_grouped_plan,
    prepare_w4a4_moe_weights,
    unpack_int4,
)


def test_signed_nibble_order_and_range():
    values = torch.arange(-8, 8, dtype=torch.int8).reshape(2, 8)
    packed = pack_int4(values)
    assert packed.flatten().tolist() == [-19088744, 1985229328]  # FEDCBA98, 76543210
    assert torch.equal(unpack_int4(packed), values)
    with pytest.raises(ValueError, match="range|\\[-8,7\\]"):
        pack_int4(torch.full((8,), 8, dtype=torch.int8))


def test_grouped_plan_shares_weights_and_encodes_scales_without_numeric_cast():
    matrix = torch.eye(8).bfloat16()
    stage = FlatQuantStage(matrix, matrix, 0.875)
    scale13 = torch.tensor([[0.004, 0.5], [1.25, 2.0]])
    scale2 = scale13[:, :1].contiguous()
    state = W4A4MoePlan(
        torch.empty(2, 2, 8, dtype=torch.int32),
        torch.empty(2, 1, 8, dtype=torch.int32),
        scale13,
        scale2,
        stage,
        stage,
        True,
    )
    plan = prepare_w4a4_grouped_plan(state)
    assert plan.weights is state
    for encoded, expected in ((plan.scale13, scale13), (plan.scale2, scale2)):
        assert encoded.dtype == torch.uint64
        bits = encoded.view(torch.int64).squeeze(1).to(torch.int32)
        torch.testing.assert_close(bits.view(torch.float32), expected, rtol=0, atol=0)


@pytest.mark.parametrize("transposed", (False, True))
def test_prepare_preserves_layout_scales_and_matrix_orientation(
    monkeypatch, transposed
):
    torch.manual_seed(42)
    e, h, i = 2, 64, 64
    logical13 = torch.randint(-8, 8, (e, h, 2 * i), dtype=torch.int8)
    logical2 = torch.randint(-8, 8, (e, i, h), dtype=torch.int8)
    if transposed:
        w13 = pack_int4(logical13.transpose(1, 2))
        w2 = pack_int4(logical2.transpose(1, 2))
    else:
        w13 = pack_int4(logical13).transpose(1, 2).contiguous()
        w2 = pack_int4(logical2).transpose(1, 2).contiguous()
    left = torch.randn(8, 8).bfloat16()
    first = prepare_flatquant_stage(
        left, torch.randn(8, 8).bfloat16(), torch.tensor([0.8])
    )
    second = prepare_flatquant_stage(
        torch.randn(8, 8).bfloat16(),
        torch.randn(8, 8).bfloat16(),
        torch.tensor([0.9]),
    )

    def cast(tensor, fmt):
        assert tensor.is_contiguous()
        assert fmt == 29
        return tensor.clone()

    def convert(tensor):
        rows, total_cols = tensor.shape
        # Simulate NZ's expert-major physical storage, not its logical ND view.
        values = tensor.reshape(rows, e, total_cols // e).permute(1, 0, 2)
        return pack_int4(values).reshape(rows, total_cols // 8)

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            get_npu_format=lambda x: 2,
            npu_format_cast=cast,
            npu_convert_weight_to_int4pack=convert,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tokenspeed_kernel_npu.ops.gemm",
        SimpleNamespace(prepare_weight_nz=lambda x: cast(x, 29)),
    )
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(config=SimpleNamespace(allow_internal_format=False)),
        raising=False,
    )
    s13, s2 = torch.rand(e, 2 * i) + 0.1, torch.rand(e, h) + 0.1
    plan = prepare_w4a4_moe_weights(
        w13, w2, s13, s2, first, second, transposed=transposed
    )
    result13, result2 = unpack_int4(plan.w13), unpack_int4(plan.w2)
    if transposed:
        result13, result2 = result13.transpose(1, 2), result2.transpose(1, 2)
    assert torch.equal(result13, logical13)
    assert torch.equal(result2, logical2)
    grouped = prepare_w4a4_grouped_plan(plan)
    assert not grouped.weights.transposed
    assert torch.equal(unpack_int4(grouped.weights.w13), logical13)
    assert torch.equal(unpack_int4(grouped.weights.w2), logical2)
    if not transposed:
        assert grouped.weights is plan
    assert torch.equal(plan.first.left, left.t())
    assert torch.equal(plan.scale13, s13)
    assert torch.equal(plan.scale2, s2)


@pytest.mark.parametrize("transposed", (False, True))
@pytest.mark.parametrize("h,i", ((128, 128), (512, 1024)))
@pytest.mark.parametrize("activation_range", (1, 7))
def test_real_nz_weights_match_integer_gmm(transposed, h, i, activation_range):
    torch_npu = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires Ascend")
    torch.npu.set_device(0)
    torch.manual_seed(19)
    e = 2
    w13 = torch.randint(-8, 8, (e, h, 2 * i), dtype=torch.int8)
    w2 = torch.randint(-8, 8, (e, i, h), dtype=torch.int8)

    def source(weight):
        return (
            pack_int4(weight.transpose(1, 2))
            if transposed
            else pack_int4(weight).transpose(1, 2).contiguous()
        ).npu()

    def stage(width):
        return prepare_flatquant_stage(
            torch.eye(8, dtype=torch.bfloat16).npu(),
            torch.eye(width // 8, dtype=torch.bfloat16).npu(),
            torch.tensor([0.9]).npu(),
        )

    plan = prepare_w4a4_moe_weights(
        source(w13),
        source(w2),
        torch.ones(e, 2 * i).npu(),
        torch.ones(e, h).npu(),
        stage(h),
        stage(i),
        transposed=transposed,
    )
    grouped = prepare_w4a4_grouped_plan(plan)
    for nz, weight, scale in (
        (grouped.weights.w13, w13, grouped.scale13),
        (grouped.weights.w2, w2, grouped.scale2),
    ):
        x = torch.randint(
            -activation_range,
            activation_range + 1,
            (8, weight.shape[1]),
            dtype=torch.int8,
        )
        packed_x = pack_int4(x).npu()
        assert torch_npu.get_npu_format(nz) == 29
        actual = torch_npu.npu_grouped_matmul(
            [packed_x],
            [nz],
            scale=[scale],
            per_token_scale=[torch.ones(8).npu()],
            group_list=torch.tensor([3, 5], dtype=torch.int64).npu(),
            split_item=3,
            output_dtype=torch.bfloat16,
            group_type=0,
            group_list_type=1,
        )[0]
        accum = torch.cat(
            (x[:3].int() @ weight[0].int(), x[3:].int() @ weight[1].int())
        )
        # The 910B official INT4 GMM's dequantized BF16 output rounds via
        # FP16. Keep that test-oracle behavior explicit, not a loose tolerance.
        # Small accumulators also prove layout without double-rounding effects.
        expected = accum.half().bfloat16()
        if activation_range == 1:
            assert torch.equal(expected, accum.bfloat16())
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
