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

"""FP32 logits from unchanged BF16 operands, including native replay updates."""

import pytest
import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel.ops.gemm import dsv4_linear_fp32
from tokenspeed_kernel_npu.ops.bf16_linear_fp32 import bf16_linear_fp32


@pytest.mark.parametrize("m", [1, 6, 12, 48, 96, 192, 384])
@pytest.mark.parametrize("strided", [False, True])
def test_short_projection_fp64_reference_and_output_owner(m, strided):
    from tokenspeed_kernel.ops.gemm import mm

    torch.npu.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(518)
    x = torch.randn(m, 288, dtype=torch.bfloat16)
    weight = torch.randn(5120, 288, dtype=torch.bfloat16) * 0.02
    reference = (x.double() @ weight.double().T).float()
    device_x, device_weight = x.npu(), weight.npu()
    storage = torch.full((m, 5128), 71, device="npu", dtype=torch.bfloat16)
    out = storage[:, :5120] if strided else None
    if strided:
        device_x = torch.cat((device_x, device_x), dim=1)[:, :288]
        device_weight = torch.cat((device_weight, device_weight), dim=1)[:, :288]
    actual = mm(device_x, device_weight, out=out, solution="triton_ascend")
    torch.testing.assert_close(actual.cpu().float(), reference, rtol=0.004, atol=2e-5)
    torch.testing.assert_close(device_x.cpu(), x, atol=0, rtol=0)
    torch.testing.assert_close(device_weight.cpu(), weight, atol=0, rtol=0)
    if out is not None:
        assert actual is out
        torch.testing.assert_close(
            storage[:, 5120:], torch.full_like(storage[:, 5120:], 71)
        )


@pytest.mark.parametrize("m", [6, 192])
def test_short_projection_live_graph_and_bias(m):
    from tokenspeed_kernel.ops.gemm import mm

    torch.npu.set_device(0)
    x = torch.arange(m * 288).remainder(7).bfloat16().reshape(m, 288).npu()
    weight = torch.arange(5120 * 288).remainder(5).bfloat16().reshape(5120, 288).npu()
    bias = torch.ones(5120, device="npu", dtype=torch.bfloat16)
    mm(x, weight, bias=bias, solution="triton_ascend")
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(
        graph, stream=torch.npu.Stream(device=torch.npu.current_device())
    ):
        actual = mm(x, weight, bias=bias, solution="triton_ascend")
    for _ in range(3):
        x.add_(1)
        weight.add_(1)
        graph.replay()
        torch.npu.synchronize()
        expected = (x.cpu().float() @ weight.cpu().float().T).bfloat16() + bias.cpu()
        torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


def test_short_projection_runtime_rows_reuse_compilation():
    from tokenspeed_kernel.ops.gemm import mm
    from tokenspeed_kernel_npu.ops.bf16_linear_fp32 import _bf16_linear_fp32_kernel
    from utils import assert_no_triton_compile

    torch.npu.set_device(0)
    x = torch.ones(384, 288, device="npu", dtype=torch.bfloat16)
    weight = torch.ones(5120, 288, device="npu", dtype=torch.bfloat16)
    mm(x[:129], weight, solution="triton_ascend")
    with assert_no_triton_compile(_bf16_linear_fp32_kernel):
        for rows in (130, 192, 256, 384):
            actual = mm(x[:rows], weight, solution="triton_ascend")
            torch.testing.assert_close(
                actual, torch.full_like(actual, 288), atol=0, rtol=0
            )


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 129, 257),
        (6, 128, 5120),
        (6, 384, 5120),
        (7, 384, 5120),
        (8, 32320, 5120),
        (48, 1024, 5120),
        (96, 384, 5120),
        (192, 384, 5120),
        (0, 384, 5120),
    ],
)
@pytest.mark.parametrize("strided", [False, True])
def test_fp64_reference_and_input_ownership(m, n, k, strided):
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    torch.manual_seed(825)
    x = torch.randn((m, k), dtype=torch.bfloat16)
    w = torch.randn((n, k), dtype=torch.bfloat16) * 0.02
    expected = (x.double() @ w.double().T).float()
    actual_x, actual_w = x.npu(), w.npu()
    if strided:
        actual_x = actual_x.T.contiguous().T
        actual_w = actual_w.T.contiguous().T
    result = dsv4_linear_fp32(actual_x, actual_w, override=None, solution=None)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result.cpu(), expected, rtol=2e-6, atol=2e-5)
    torch.testing.assert_close(actual_x.cpu(), x, rtol=0, atol=0)
    torch.testing.assert_close(actual_w.cpu(), w, rtol=0, atol=0)


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("m,n,k", [(6, 69, 33), (6, 384, 5120), (96, 384, 5120)])
def test_mutable_native_graph_reads_live_weights(strided, m, n, k):
    torch.npu.set_device(0)
    x = torch.arange(m * k).remainder(7).bfloat16().reshape(2, m // 2, k).npu()
    w = torch.arange(n * k).remainder(5).bfloat16().reshape(n, k).npu()
    if strided:
        w = w.T.contiguous().T
    dsv4_linear_fp32(x, w, override=None, solution=None)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(
        graph,
        stream=torch.npu.Stream(device=torch.npu.current_device()),
        auto_dispatch_capture=True,
    ):
        result = dsv4_linear_fp32(x, w, override=None, solution=None)
    for _ in range(4):
        x.add_(1)
        w.add_(1)
        graph.replay()
        torch.npu.synchronize()
        expected = x.cpu().float() @ w.cpu().float().T
        torch.testing.assert_close(result.cpu(), expected, atol=0, rtol=0)


def test_nonfinite_inputs():
    torch.npu.set_device(0)
    x = torch.ones((4, 64), dtype=torch.bfloat16)
    x[0, 0] = float("nan")
    x[1, 1] = float("inf")
    x[2, 2] = -float("inf")
    w = torch.ones((129, 64), dtype=torch.bfloat16)
    expected = x.float() @ w.float().T
    result = dsv4_linear_fp32(x.npu(), w.npu(), override=None, solution=None)
    torch.testing.assert_close(result.cpu(), expected, atol=0, rtol=0, equal_nan=True)


def test_never_rounds_fp32_operands_to_bf16():
    torch.npu.set_device(0)
    x = torch.ones((2, 32), device="npu", dtype=torch.bfloat16)
    w = torch.randn((24, 32), device="npu", dtype=torch.float32)
    with pytest.raises(ValueError, match="never quantizes"):
        bf16_linear_fp32(x, w)
    # The common dispatcher keeps its FP32 fallback for FP32 weights.
    actual = dsv4_linear_fp32(x, w, override=None, solution=None)
    expected = x.float() @ w.T
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_runtime_rows_and_columns_reuse_compilation():
    from tokenspeed_kernel_npu.ops.bf16_linear_fp32 import _bf16_linear_fp32_kernel
    from utils import assert_no_triton_compile

    torch.npu.set_device(0)
    x = torch.ones((16, 257), device="npu", dtype=torch.bfloat16)
    w = torch.ones((384, 257), device="npu", dtype=torch.bfloat16)
    bf16_linear_fp32(x[:1], w[:320])
    with assert_no_triton_compile(_bf16_linear_fp32_kernel):
        for rows, columns in ((2, 336), (3, 352), (7, 368), (16, 384)):
            actual = bf16_linear_fp32(x[:rows], w[:columns])
            torch.testing.assert_close(
                actual, torch.full_like(actual, 257), atol=0, rtol=0
            )


def test_dspark_and_router_use_unchanged_bf16_weights():
    from types import SimpleNamespace

    from tokenspeed.runtime.models.deepseek_v41 import DeepseekV41MoE
    from tokenspeed.runtime.models.deepseek_v41_dspark import DeepseekV41DSparkModel

    torch.npu.set_device(0)
    torch.manual_seed(716)
    hidden = torch.randn(192, 5120, device="npu", dtype=torch.bfloat16)
    head = torch.randn(384, 5120, device="npu", dtype=torch.bfloat16) * 0.02
    reference = (hidden.cpu().double() @ head.cpu().double().T).float()
    draft = object.__new__(DeepseekV41DSparkModel)
    actual = draft.local_base_logits(hidden, head)
    torch.testing.assert_close(actual.cpu(), reference, rtol=2e-6, atol=2e-5)
    bias = torch.randn(384, device="npu")
    moe = SimpleNamespace(
        gate=SimpleNamespace(weight=head, bias_vl=None, e_score_correction_bias=bias)
    )
    logits, actual_bias, a, b = DeepseekV41MoE._routing_inputs(moe, hidden, None)
    assert actual_bias is bias and a is None and b is None
    torch.testing.assert_close(logits.cpu(), reference, rtol=2e-6, atol=2e-5)
    # Verify the top-6 expert decisions, not just aggregate projection error.
    torch.testing.assert_close(
        logits.topk(6).indices.cpu(), reference.topk(6).indices, atol=0, rtol=0
    )


@pytest.mark.parametrize("rows", [1, 6, 32, 96, 192])
@pytest.mark.parametrize("shard", [0, 1])
def test_strided_k_shard_fp64_reference(rows, shard):
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(781)
    x = torch.randn(rows, 2048, dtype=torch.bfloat16)
    storage = torch.randn(4096, 1024, dtype=torch.bfloat16) * 0.01
    w = storage.T[:, shard * 2048 : (shard + 1) * 2048]
    device_storage = storage.npu()
    actual = dsv4_linear_fp32(
        x.npu(),
        device_storage.T[:, shard * 2048 : (shard + 1) * 2048],
        override=None,
        solution=None,
    )
    expected = (x.double() @ w.double().T).float()
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-5)
    torch.testing.assert_close(device_storage.cpu(), storage, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [6, 192])
def test_k_shard_graph_reads_mutable_operand_views(rows):
    torch.npu.set_device(0)
    x = torch.arange(rows * 2048).remainder(7).bfloat16().reshape(rows, 2048).npu()
    storage = (
        torch.arange(4096 * 1024).remainder(5).bfloat16().reshape(4096, 1024).npu()
    )
    weight = storage.T[:, 2048:]
    dsv4_linear_fp32(x, weight, override=None, solution=None)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(
        graph, stream=torch.npu.Stream(device=torch.npu.current_device())
    ):
        actual = dsv4_linear_fp32(x, weight, override=None, solution=None)
    for _ in range(3):
        x.add_(1)
        weight.add_(1)
        graph.replay()
        torch.npu.synchronize()
        expected = x.cpu().float() @ weight.cpu().float().T
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_strided_projection_runtime_dimensions_reuse_compilation():
    from tokenspeed_kernel_npu.ops.grouped_bf16_projection import _grouped_projection
    from utils import assert_no_triton_compile

    torch.npu.set_device(0)
    x = torch.ones(16, 2048, device="npu", dtype=torch.bfloat16)
    w = torch.ones(4096, 1024, device="npu", dtype=torch.bfloat16).T[:, 2048:]
    dsv4_linear_fp32(x[:1], w[:512], override=None, solution=None)
    with assert_no_triton_compile(_grouped_projection):
        for rows, columns in ((2, 528), (3, 544), (7, 560), (16, 576)):
            actual = dsv4_linear_fp32(
                x[:rows], w[:columns], override=None, solution=None
            )
            torch.testing.assert_close(
                actual, torch.full_like(actual, 2048), rtol=0, atol=0
            )


@pytest.mark.parametrize("rows", [1, 6, 32, 96, 192])
@pytest.mark.parametrize("columns", [1600, 3200, 6400])
def test_wide_projection_fp64_reference(rows, columns):
    from tokenspeed_kernel.ops.gemm import mm

    torch.npu.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(831)
    x = torch.randn(rows, 6144, dtype=torch.bfloat16)
    weight = torch.randn(columns, 6144, dtype=torch.bfloat16) * 0.01
    device_weight = weight.npu()
    storage = torch.full((rows, columns + 8), 71, device="npu", dtype=torch.bfloat16)
    out = storage[:, :columns]
    actual = mm(x.npu(), device_weight, out=out, solution="triton_ascend")
    expected = (x.double() @ weight.double().T).float()
    assert actual is out
    torch.testing.assert_close(actual.cpu().float(), expected, rtol=0.004, atol=2e-5)
    torch.testing.assert_close(device_weight.cpu(), weight, rtol=0, atol=0)
    torch.testing.assert_close(
        storage[:, columns:], torch.full_like(storage[:, columns:], 71), rtol=0, atol=0
    )


@pytest.mark.parametrize("rows", [6, 192])
def test_wide_projection_mutable_graph(rows):
    from tokenspeed_kernel.ops.gemm import mm

    torch.npu.set_device(0)
    x = torch.arange(rows * 6144).remainder(3).bfloat16().reshape(rows, 6144).npu()
    weight = torch.arange(3200 * 6144).remainder(2).bfloat16().reshape(3200, 6144).npu()
    mm(x, weight, solution="triton_ascend")
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(
        graph, stream=torch.npu.Stream(device=torch.npu.current_device())
    ):
        actual = mm(x, weight, solution="triton_ascend")
    for _ in range(3):
        x.add_(1)
        weight.add_(1)
        graph.replay()
        torch.npu.synchronize()
        expected = (x.cpu().float() @ weight.cpu().float().T).bfloat16()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_wide_projection_runtime_rows_reuse_compilation():
    from tokenspeed_kernel.ops.gemm import mm
    from tokenspeed_kernel_npu.ops.bf16_linear_fp32 import _bf16_linear_fp32_kernel
    from utils import assert_no_triton_compile

    torch.npu.set_device(0)
    x = torch.ones(192, 6144, device="npu", dtype=torch.bfloat16)
    weight = torch.ones(3200, 6144, device="npu", dtype=torch.bfloat16)
    mm(x[:65], weight, solution="triton_ascend")
    with assert_no_triton_compile(_bf16_linear_fp32_kernel):
        for rows in (66, 96, 128, 192):
            actual = mm(x[:rows], weight, solution="triton_ascend")
            torch.testing.assert_close(
                actual, torch.full_like(actual, 6144), rtol=0, atol=0
            )
