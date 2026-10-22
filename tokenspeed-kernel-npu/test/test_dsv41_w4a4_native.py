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

"""Live graph counts, padding and the existing INT4 dequantization contract."""

import pytest
import torch
import torch_npu
from tokenspeed_kernel.selection import kernel_override
from tokenspeed_kernel_npu.ops.dsv41_w4a4_native import available, grouped_w4a4


@pytest.mark.skipif(not available(), reason="Build the native grouped W4A4 kernel")
@pytest.mark.parametrize("k,n", [(5120, 4608), (2304, 5120)])
def test_grouped_w4a4_changing_graph(k, n):
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.set_num_threads(8)
    torch.manual_seed(127)
    experts, capacity = 24, 1152
    codes = torch.randint(-8, 8, (experts, n, k), dtype=torch.int8)
    weight = torch.stack(
        [torch_npu.npu_convert_weight_to_int4pack(w.int().npu()) for w in codes]
    )
    weight = torch_npu.npu_format_cast(weight, 29, customize_dtype=3)
    scale = torch.rand(experts, n) * 0.025
    scale[:, ::127] = 0
    scale_words = scale.view(torch.int32).long().view(torch.uint64).unsqueeze(1).npu()
    # Fixpipe consumes the high 19 bits of the FP32 scale word, then rounds
    # its scaled integer accumulator to FP16 before per-token scaling.
    effective_scale = (scale.view(torch.int32) & -8192).view(torch.float32)
    x = torch.randint(-8, 8, (capacity, k), dtype=torch.int32)
    packed_x = torch_npu.npu_convert_weight_to_int4pack(x.npu())
    token_scale = torch.rand(capacity, device="npu") * 0.1
    counts = torch.zeros(experts, dtype=torch.int64, device="npu")
    out = torch.full((capacity, n), 123, dtype=torch.bfloat16, device="npu")

    def run():
        return grouped_w4a4(packed_x, weight, scale_words, token_scale, counts, out)

    run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream(device=0)):
        run()
    cases = [torch.zeros(experts, dtype=torch.int64) for _ in range(5)]
    cases[1][-1] = 1
    cases[2][1], cases[2][9], cases[2][23] = 3, 2, 4
    cases[3][17] = 65
    cases[4][:] = 3
    for current in cases:
        x.random_(-8, 8)
        packed_x.copy_(torch_npu.npu_convert_weight_to_int4pack(x.npu()))
        token_scale.uniform_(0.001, 0.1)
        counts.copy_(current)
        out.fill_(123)
        graph.replay()
        torch.npu.synchronize()
        actual, per_token = out.cpu(), token_scale.cpu()
        start = 0
        for expert, size in enumerate(current.tolist()):
            end = start + size
            if size:
                # Small integer products and their sum are exactly representable
                # in FP32; no CPU integer-matmul dependency is needed.
                integer_dot = x[start:end].float() @ codes[expert].float().T
                channel_scaled = (integer_dot * effective_scale[expert]).half().float()
                expected = (channel_scaled * per_token[start:end, None]).bfloat16()
                torch.testing.assert_close(actual[start:end], expected, atol=0, rtol=0)
            start = end
        assert torch.all(actual[start:] == 123)
        native = torch_npu.npu_grouped_matmul(
            [packed_x],
            [weight.transpose(1, 2)],
            scale=[scale_words],
            per_token_scale=[token_scale],
            group_list=counts,
            split_item=3,
            group_type=0,
            group_list_type=1,
            output_dtype=torch.bfloat16,
        )[0]
        torch.testing.assert_close(out[:start], native[:start], atol=0, rtol=0)

    # Exercise the production FlatQuant -> grouped-linear dispatch, including
    # nonidentity transforms, against the previous complete execution path.
    import flash_ops  # noqa: F401
    from tokenspeed_kernel_npu.ops.dsv41_moe import (
        AscendGroupedLinear,
        FlatQuantParameters,
    )

    left_size, right_size = (64, 80) if k == 5120 else (48, 48)
    left = torch.linalg.qr(torch.randn(left_size, left_size)).Q.bfloat16().npu()
    right = torch.linalg.qr(torch.randn(right_size, right_size)).Q.bfloat16().npu()
    transform = FlatQuantParameters(left, right, 1.0)
    with kernel_override("gemm", "grouped_w4a4", "ascend_c_grouped_w4a4"):
        linear = AscendGroupedLinear("w4a4", weight, scale_words, transform, k, n)
    values = torch.randn(capacity, k, device="npu", dtype=torch.bfloat16)
    actual = linear(values, counts)
    quantized, per_token = torch.ops.custom.flat_quant(
        values.reshape(capacity, left_size, right_size),
        left,
        right,
        group_list=counts,
        group_list_type=1,
        clip_ratio=1.0,
    )
    expected = torch_npu.npu_grouped_matmul(
        [quantized.flatten(1)],
        [weight.transpose(1, 2)],
        scale=[scale_words],
        per_token_scale=[per_token],
        group_list=counts,
        split_item=3,
        group_type=0,
        group_list_type=1,
        output_dtype=torch.bfloat16,
    )[0]
    torch.testing.assert_close(actual[:start], expected[:start], atol=0, rtol=0)
    with kernel_override("gemm", "grouped_w4a4", "cann_grouped_w4a4"):
        cann_linear = AscendGroupedLinear("w4a4", weight, scale_words, transform, k, n)
        # A later override must not change an already prepared layer.
        actual = linear(values, counts)
        expected = cann_linear(values, counts)
        torch.testing.assert_close(actual[:start], expected[:start], atol=0, rtol=0)
