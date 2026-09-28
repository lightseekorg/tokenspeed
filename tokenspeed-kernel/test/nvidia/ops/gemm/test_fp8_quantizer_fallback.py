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
from tokenspeed_kernel.ops import quantization
from tokenspeed_kernel.ops.gemm import flashinfer as gemm_flashinfer
from tokenspeed_kernel.ops.gemm import mm
from tokenspeed_kernel.ops.quantization import trtllm as quantization_trtllm
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_hopper, reason="requires SM90"
)


@pytest.mark.parametrize("rows,groups", [(1, 1), (3, 4), (4, 4), (5, 40), (40, 40)])
@pytest.mark.parametrize("encoding", ["float32", "ue8m0"])
def test_native_scale_padding_adapter(monkeypatch, rows, groups, encoding):
    """Check the adapter on CPU, independently of the native quantizer's math."""
    x = torch.empty(rows, groups * 128, dtype=torch.bfloat16)
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    padded_rows = (rows + 3) // 4 * 4
    canonical = torch.exp2(torch.arange(rows * groups).reshape(rows, groups) % 8)
    native = torch.full((groups * padded_rows + 32,), -1.0)
    native[: groups * padded_rows].view(groups, padded_rows)[:, :rows] = canonical.T

    def quantize(input, use_ue8m0):
        assert input is x
        assert use_ue8m0 == (encoding == "ue8m0")
        return q, native

    monkeypatch.setattr(torch.ops.trtllm, "fp8_quantize_1x128", quantize)
    actual_q, scales = quantization_trtllm.trtllm_quantize_fp8_with_scale(
        x, granularity="token_group", group_size=128, scale_encoding=encoding
    )
    assert actual_q is q
    assert scales.is_contiguous()
    if encoding == "ue8m0":
        assert scales.dtype == torch.uint8
        scales = scales.view(torch.float8_e8m0fnu).float()
    torch.testing.assert_close(scales, canonical, atol=0, rtol=0)


@pytest.mark.parametrize(
    "m,k", [(1, 128), (3, 512), (4, 512), (5, 5120), (40, 5120), (1024, 5120)]
)
@pytest.mark.parametrize("encoding", ["float32", "ue8m0"])
def test_installed_quantizer_preserves_canonical_scales(device, m, k, encoding):
    # Distinct row/group scales catch transposition even for square [40, 40].
    x = torch.arange(1, m * (k // 128) + 1, device=device, dtype=torch.float32)
    x = x.reshape(m, k // 128).repeat_interleave(128, dim=1).to(torch.bfloat16)
    q, scales = quantization.quantize_fp8(
        x,
        granularity="token_group",
        group_size=128,
        scale_encoding=encoding,
        solution="trtllm",
    )
    assert scales.shape == (m, k // 128)
    assert scales.is_contiguous()
    expected = x.float().reshape(m, -1, 128).abs().amax(-1) / 448.0
    if encoding == "ue8m0":
        assert scales.dtype == torch.uint8
        scales = scales.view(torch.float8_e8m0fnu).float()
        expected = torch.exp2(torch.ceil(torch.log2(expected)))
    else:
        assert scales.dtype == torch.float32
    torch.testing.assert_close(scales, expected, atol=1e-8, rtol=1e-6)
    reconstructed = q.float() * scales.repeat_interleave(128, dim=1)
    torch.testing.assert_close(reconstructed, x.float(), atol=1e-5, rtol=0.0625)


@pytest.mark.parametrize("m", [4, 8, 40, 1024])
def test_prepacked_installed_quantizer_layout(device, m):
    torch.manual_seed(1)
    x = torch.randn(m, 5120, device=device, dtype=torch.bfloat16)
    q, scales = gemm_flashinfer.flashinfer_fp8_blockscale_quantize_prepacked(x, 128)
    assert scales.shape == (40, m)
    assert scales.is_contiguous()
    expected = x.float().reshape(m, 40, 128).abs().amax(-1) / 448.0
    torch.testing.assert_close(scales, expected.T, atol=1e-8, rtol=1e-6)
    torch.testing.assert_close(
        q.float() * scales.T.repeat_interleave(128, dim=1),
        x.float(),
        atol=1e-5,
        rtol=0.0625,
    )


@pytest.mark.parametrize("m", [1, 3, 8, 40, 1024])
def test_public_quantizer_feeds_model_block_fp8_gemm_and_graph(device, m):
    # Replicated q_a projection: K=5120, N=1024, including startup's M=1024.
    torch.manual_seed(2)
    x = torch.randn(m, 5120, device=device, dtype=torch.bfloat16)
    weight = (torch.randn(1024, 5120, device=device) * 0.125).to(torch.float8_e4m3fn)
    weight_scales = torch.rand(8, 40, device=device) + 0.5
    out = torch.empty(m, 1024, dtype=torch.bfloat16, device=device)

    def run():
        q, scales = quantization.quantize_fp8(
            x, granularity="token_group", group_size=128, scale_encoding="float32"
        )
        result = mm(
            q,
            weight,
            A_scales=scales,
            B_scales=weight_scales,
            out_dtype=torch.bfloat16,
            quant="mxfp8",
            block_size=[128, 128],
            out=out,
        )
        assert result is out
        return q, scales

    def check(q, scales):
        activation = q.float() * scales.repeat_interleave(128, dim=1)
        dequantized_weight = weight.float() * weight_scales.repeat_interleave(
            128, dim=0
        ).repeat_interleave(128, dim=1)
        expected = (activation @ dequantized_weight.T).to(torch.bfloat16)
        torch.testing.assert_close(out, expected, atol=0.01, rtol=0.0078125)

    q, scales = run()
    check(q, scales)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, scales = run()
    x.normal_()
    graph.replay()
    check(q, scales)
    # The quantizer's graph outputs must reflect the refreshed input, not warmup.
    expected_scales = x.float().reshape(m, 40, 128).abs().amax(-1) / 448.0
    torch.testing.assert_close(scales, expected_scales, atol=1e-8, rtol=1e-6)
