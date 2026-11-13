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

"""Real Fp8LinearMethod weight loading, packed quantization and replay."""

import pytest
import torch
from tokenspeed_kernel.ops.activation.triton import sigmoid_mul
from tokenspeed_kernel.ops.layernorm import gated_rmsnorm
from tokenspeed_kernel.ops.quantization import output_fp8_quant_pack, quantize_fp8
from tokenspeed_kernel.platform import pdl_enabled

from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod
from tokenspeed.runtime.layers.linear import RowParallelLinear
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.utils.env import global_server_args_dict

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="requires SM100 or SM103",
)


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("rows", [1, 32, 128])
def test_loaded_linear_and_graph(monkeypatch, serialized, rows):
    pytest.importorskip("deep_gemm")
    monkeypatch.setitem(global_server_args_dict, "dense_gemm_backend", "deepgemm")
    config = Fp8Config(
        is_checkpoint_fp8_serialized=serialized, weight_block_size=[128, 128]
    )
    method = Fp8LinearMethod(config)
    layer = torch.nn.Module()
    n, k = 256, 640
    method.create_weights(layer, k, [n], k, n, torch.bfloat16, weight_loader=None)
    layer = layer.cuda()
    torch.manual_seed(31)
    original = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.03
    if serialized:
        weight, scales = quantize_fp8(
            original, granularity="block", block_size=[128, 128]
        )
        layer.weight.data.copy_(weight)
        layer.weight_scale_inv.data.copy_(scales)
    else:
        layer.weight.data.copy_(original)
    method.process_weights_after_loading(layer)
    # A later global option change must not reinterpret prepared weight storage.
    monkeypatch.setitem(global_server_args_dict, "dense_gemm_backend", "auto")
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16) * 0.1
    actual = method.apply(layer, x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = method.apply(layer, x)
    x.normal_().mul_(0.2)
    graph.replay()
    expected = method.apply(layer, x)
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    reference = x.float() @ original.float().t()
    error = (captured.float() - reference).norm() / reference.norm()
    assert error < 0.06
    assert actual.dtype == torch.bfloat16
    assert layer.weight_scale_inv.dtype == torch.int32


@pytest.mark.parametrize("rows", [4, 32])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("pdl", [False, True])
def test_output_projection_graph(monkeypatch, rows, normalize, pdl):
    monkeypatch.setitem(global_server_args_dict, "dense_gemm_backend", "deepgemm")
    torch.manual_seed(43)
    width, head = 6144, 128 if normalize else 256
    config = Fp8Config(
        is_checkpoint_fp8_serialized=False,
        weight_block_size=[128, 128],
    )
    layer = RowParallelLinear(
        width,
        2560,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=config,
        prefix="projection",
    ).cuda()
    with torch.no_grad():
        layer.weight.normal_(std=0.03)
    layer.quant_method.process_weights_after_loading(layer)
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    backing = torch.randn(
        rows, width // head, 2, head, device="cuda", dtype=torch.bfloat16
    )
    z = (
        backing[:, :, 1, :]
        if not normalize
        else torch.randn(rows, width * 2, device="cuda", dtype=torch.bfloat16)[
            :, width:
        ]
    )
    weight = (
        torch.ones(head, device="cuda", dtype=torch.bfloat16) if normalize else None
    )

    def fused(value):
        codes, scales = output_fp8_quant_pack(
            value,
            z,
            weight,
            head_dim=head,
            eps=1e-6,
            sigmoid_gate=True,
            enable_pdl=pdl,
        )
        return layer(codes, scale=scales)[0]

    old_pdl = pdl_enabled()
    pdl_enabled(pdl)
    try:
        fused(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = fused(x * 1.0)
        for magnitude in (0.01, 1.0, 16.0):
            x.normal_().mul_(magnitude)
            graph.replay()
            if normalize:
                value = gated_rmsnorm(
                    x,
                    weight,
                    z,
                    eps=1e-6,
                    group_size=head,
                    norm_before_gate=True,
                    sigmoid_gate=True,
                    fp8_scale=None,
                    weights_independent=True,
                )
            else:
                value = sigmoid_mul(x.clone(), z)
            expected = layer(value)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        pdl_enabled(old_pdl)
