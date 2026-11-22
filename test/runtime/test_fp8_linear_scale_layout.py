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

"""Block-FP8 linears quantize activation scales into their GEMM's layout."""

import os
import sys

import pytest
import torch
from torch.nn.parameter import Parameter
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM10x GPU for the CuTe-DSL block-FP8 GEMM",
)

N, K = 4096, 2048


@pytest.fixture
def linear(monkeypatch):
    from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod
    from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "dense_gemm_backend", "auto")
    torch.manual_seed(0)
    layer = torch.nn.Module()
    weight = (torch.randn(N, K, device="cuda") * 0.02).to(torch.float8_e4m3fn)
    scales = torch.rand(N // 128, K // 128, device="cuda") * 0.02 + 0.001
    layer.weight = Parameter(weight, requires_grad=False)
    layer.weight_scale_inv = Parameter(scales, requires_grad=False)
    method = Fp8LinearMethod(
        Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
    )
    method.process_weights_after_loading(layer)
    return method, layer


def _torch_copies(fn) -> list[str]:
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [
        event.name
        for event in prof.events()
        if event.device_type.name == "CUDA"
        and ("copy" in event.name.lower() or "memcpy" in event.name.lower())
    ]


@pytest.mark.parametrize("m", [17, 64, 1000])
def test_large_m_linear_launches_no_scale_copy(linear, m: int) -> None:
    method, layer = linear
    x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
    method.apply(layer, x)
    assert _torch_copies(lambda: method.apply(layer, x)) == []


@pytest.mark.parametrize("m", [1, 4, 16, 17, 64, 1000])
def test_linear_matches_canonical_scales(linear, m: int) -> None:
    from tokenspeed_kernel.ops.gemm import mm, mm_activation_scale_layout
    from tokenspeed_kernel.ops.quantization import quantize_fp8

    method, layer = linear
    x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
    kwargs = dict(B_scales=layer.weight_scale_inv, block_size=[128, 128])
    layout = mm_activation_scale_layout(x, layer.weight, **kwargs)
    a, canonical = quantize_fp8(x, granularity="token_group", group_size=128)
    scales = canonical.t().contiguous().t() if layout == "mn_major" else canonical
    expected = mm(
        a,
        layer.weight,
        A_scales=scales,
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        **kwargs,
    )
    assert torch.equal(method.apply(layer, x), expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
