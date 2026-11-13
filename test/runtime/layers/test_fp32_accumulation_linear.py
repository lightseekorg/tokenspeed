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

import pytest
import torch

from tokenspeed.runtime.models.inkling2_attention import FP32AccumulationLinearMethod


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("use_out", (False, True))
@pytest.mark.parametrize("use_bias", (False, True))
def test_fp32_accumulation_linear(device, dtype, use_out, use_bias):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU required")
    torch.manual_seed(71)
    layer = torch.nn.Linear(1536, 64, bias=use_bias, device=device, dtype=dtype)
    inputs = torch.randn(24, 1536, device=device, dtype=dtype).repeat(4, 1)
    method = FP32AccumulationLinearMethod()
    if device == "cuda":
        expected = torch.mm(inputs, layer.weight.T, out_dtype=torch.float32)
    else:
        expected = inputs.float() @ layer.weight.float().T
    if layer.bias is not None:
        expected = expected + layer.bias.float()
    expected = expected.to(dtype)
    with torch.no_grad():
        if use_out:
            out = torch.empty(96, 66, device=device, dtype=dtype)[:, :64]
            actual = method.apply_into(layer, inputs, layer.bias, None, dtype, out)
            assert actual is out
        else:
            actual = method.apply(layer, inputs, layer.bias)
    if device == "cuda" and not use_bias:
        # The GEMM rounds its own FP32 sum, whose K order may differ from the
        # reference's: within one output step.
        step = 2**-7 if dtype == torch.bfloat16 else 2**-10
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=step, atol=1e-5
        )
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    repeated = actual.reshape(4, 24, 64)
    for row in repeated[1:]:
        torch.testing.assert_close(row, repeated[0], rtol=0, atol=0)


@pytest.mark.parametrize("rows", (1, 8, 24))
def test_decode_rows_accumulate_in_fp32_and_round_once(rows):
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    torch.manual_seed(5)
    layer = torch.nn.Linear(4608, 512, bias=False, device="cuda", dtype=torch.bfloat16)
    inputs = torch.randn(rows, 4608, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        actual = FP32AccumulationLinearMethod().apply(layer, inputs, None)
    expected = (inputs.double() @ layer.weight.double().T).to(torch.bfloat16)
    assert actual.dtype == torch.bfloat16
    # One BF16 step of the exact result; near zero, the FP32 accumulation order
    # over 4,608 products moves the sum by up to about 1e-5 (cancellation).
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2**-7, atol=2e-5)


@pytest.mark.parametrize("rows", (33, 300, 2061))
@pytest.mark.parametrize("width", ((4608, 3792), (1536, 4608)))
def test_prefill_projections_accumulate_in_fp32_and_round_once(rows, width):
    """Prefill-size Inkling2 projections (attention input, output) round
    the FP32 sum once to BF16."""
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    torch.manual_seed(rows)
    inner, outer = width
    layer = torch.nn.Linear(
        inner, outer, bias=False, device="cuda", dtype=torch.bfloat16
    )
    inputs = torch.randn(rows, inner, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        actual = FP32AccumulationLinearMethod().apply(layer, inputs, None)
    expected = (inputs.double() @ layer.weight.double().T).to(torch.bfloat16)
    assert actual.dtype == torch.bfloat16
    # As for decode rows: one BF16 step, and FP32 order near zero.
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2**-7, atol=2e-5)


@pytest.mark.parametrize("rows", (24, 300, 2061))
@pytest.mark.parametrize("use_bias", (False, True))
def test_apply_into_writes_the_bits_of_apply(rows, use_bias):
    """A destination (such as a symmetric buffer a reduce-scatter reads)
    receives the bits ``apply`` returns, at decode and prefill sizes."""
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    torch.manual_seed(rows)
    layer = torch.nn.Linear(
        1536, 4608, bias=use_bias, device="cuda", dtype=torch.bfloat16
    )
    inputs = torch.randn(rows, 1536, device="cuda", dtype=torch.bfloat16)
    method = FP32AccumulationLinearMethod()
    out = torch.full((rows, 4608), float("nan"), device="cuda").bfloat16()
    with torch.no_grad():
        expected = method.apply(layer, inputs, layer.bias)
        actual = method.apply_into(layer, inputs, layer.bias, None, torch.bfloat16, out)
    assert actual is out
    assert torch.equal(out, expected)
