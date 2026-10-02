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

"""Per-tensor FP8 linears (Fp8LinearMethod without a weight block) on CPU.

A forward pass of one and of several tokens with dynamic input scales: the
input goes through the real ``fp8_utils`` per-token quantization wrappers and
the real ``tokenspeed_kernel.mm`` dispatcher (traits, format signature,
selection). Only the TRT-LLM activation quantizer and the selected kernel's
body are replaced by CPU stand-ins that keep their contracts.
"""

import pytest
import tokenspeed_kernel.ops.gemm as gemm
import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import SelectedKernel

from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia, reason="the TRT-LLM quantizer is NVIDIA's"
)

FP8 = torch.float8_e4m3fn


@pytest.fixture(params=["M1", "M"])
def trtllm_quantizer(request, monkeypatch):
    """torch.ops.tensorrt_llm.quantize_e4m3_activation on CPU: amax / 448 per
    token, its scales shaped [M, 1] or [M]."""

    def quantize(x):
        scale = x.float().abs().amax(-1, keepdim=True) / 448.0
        codes = (x.float() / scale).clamp(-448, 448).to(FP8)
        return codes, scale if request.param == "M1" else scale.squeeze(-1)

    monkeypatch.setattr(
        torch.ops.tensorrt_llm, "quantize_e4m3_activation", quantize, raising=False
    )


@pytest.fixture
def selected_kernel(monkeypatch):
    """The dispatcher's own selection, with the selected kernel run on CPU:
    scales of one dimension or none become columns, as triton_scaled_mm takes
    them. Records the scale shapes the kernel receives."""
    calls = []
    select = gemm.select_kernel

    def selected(*args, **kwargs):
        kernel = select(*args, **kwargs)

        def run(A, B, A_scales, B_scales, out_dtype, **_):
            calls.append((tuple(A_scales.shape), tuple(B_scales.shape)))
            a = A_scales.reshape(-1, 1) if A_scales.dim() <= 1 else A_scales
            b = B_scales.reshape(-1, 1) if B_scales.dim() <= 1 else B_scales
            return ((A.double() * a.double()) @ (B.double() * b.double().T)).to(
                out_dtype
            )

        return SelectedKernel(kernel.name, run)

    monkeypatch.setattr(gemm, "select_kernel", selected)
    return calls


def _layer(activation_scheme):
    layer = ReplicatedLinear(
        16,
        8,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=Fp8Config(
            is_checkpoint_fp8_serialized=True, activation_scheme=activation_scheme
        ),
        prefix="model.layers.0.mlp.down_proj",
    )
    generator = torch.Generator().manual_seed(0)
    codes = (torch.randn(8, 16, generator=generator) * 64).clamp(-448, 448).to(FP8)
    layer.weight.data.copy_(codes)
    layer.weight_scale.data.fill_(2.0**-7)
    if activation_scheme == "static":
        layer.input_scale.data.fill_(2.0**-6)
    layer.quant_method.process_weights_after_loading(layer)
    return layer, codes.double() * 2.0**-7


@pytest.mark.parametrize("m", [1, 3])
def test_dynamic_input_is_quantized_per_token(trtllm_quantizer, selected_kernel, m):
    layer, weight = _layer("dynamic")
    x = (torch.randn(m, 16, generator=torch.Generator().manual_seed(m)) * 3).to(
        torch.bfloat16
    )
    out = layer.quant_method.apply(layer, x)
    assert selected_kernel == [((m, 1), (8, 1))]
    scale = x.float().abs().amax(-1, keepdim=True) / 448.0
    codes = (x.float() / scale).clamp(-448, 448).to(FP8)
    reference = (codes.double() * scale.double()) @ weight.T
    assert torch.equal(out, reference.to(torch.bfloat16))
