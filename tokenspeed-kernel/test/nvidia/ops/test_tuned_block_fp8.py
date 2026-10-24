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


from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.gemm import mm
from tokenspeed_kernel.ops.quantization import quantize_fp8
from tokenspeed_kernel.ops.tuning import autotune

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires Blackwell",
)


@pytest.mark.parametrize("rows", [1, 16, 24, 33, 64, 129, 512])
def test_tuned_block_fp8_matches_existing_and_replays(rows, tmp_path):
    from flashinfer.autotuner import AutoTuner

    torch.manual_seed(42)
    x = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    values, scales = quantize_fp8(
        x, granularity="token_group", group_size=128, solution="triton"
    )
    codes, weight_scales = quantize_fp8(
        weight, granularity="block", block_size=(128, 128), solution="triton"
    )
    original_codes = codes.view(torch.uint8).clone()
    original_scales = weight_scales.clone()
    output = torch.empty(rows, 256, device="cuda", dtype=torch.bfloat16)

    def run(solution, destination):
        return mm(
            values,
            codes,
            A_scales=scales,
            B_scales=weight_scales,
            out=destination,
            out_dtype=torch.bfloat16,
            quant="mxfp8",
            block_size=[128, 128],
            solution=solution,
        )

    expected = run("triton" if 17 <= rows <= 32 else "flashinfer", None)
    with autotune(tune_mode=True, tuning_buckets=(rows,), round_up=True):
        actual = run(None, output)
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    tuned_output = actual.clone()
    saved = tmp_path / "tactics.json"
    tuner = AutoTuner.get()
    tuner.save_configs(str(saved))
    tuner.clear_cache()
    assert tuner.load_configs(str(saved))
    torch.testing.assert_close(run(None, output), tuned_output, rtol=0, atol=0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run(None, output)
    x.normal_()
    fresh_values, fresh_scales = quantize_fp8(
        x, granularity="token_group", group_size=128, solution="triton"
    )
    values.copy_(fresh_values)
    scales.copy_(fresh_scales)
    graph.replay()
    expected = run("triton" if 17 <= rows <= 32 else "flashinfer", None)
    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(codes.view(torch.uint8), original_codes, rtol=0, atol=0)
    torch.testing.assert_close(weight_scales, original_scales, rtol=0, atol=0)


def test_tuned_block_fp8_empty_output():
    from tokenspeed_kernel.thirdparty.trtllm_blockwise.tuning import (
        tuned_blockwise_gemm,
    )

    a = torch.empty(0, 128, device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.empty(128, 128, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.empty(0, 1, device="cuda")
    out = torch.empty(0, 128, device="cuda", dtype=torch.bfloat16)
    result = tuned_blockwise_gemm(
        a, b, scales, torch.ones(1, 1, device="cuda"), out, allow_existing=True
    )
    assert result is out


def test_tuned_block_fp8_preserves_strided_output():
    x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    q, qs = quantize_fp8(
        x, granularity="token_group", group_size=128, solution="triton"
    )
    b, bs = quantize_fp8(
        w, granularity="block", block_size=(128, 128), solution="triton"
    )
    storage = torch.full((64, 512), 7.0, device="cuda", dtype=torch.bfloat16)
    out = storage[:, :256]
    result = mm(
        q,
        b,
        A_scales=qs,
        B_scales=bs,
        out=out,
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=[128, 128],
    )
    expected = mm(
        q,
        b,
        A_scales=qs,
        B_scales=bs,
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=[128, 128],
        solution="flashinfer",
    )
    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert torch.all(storage[:, 256:] == 7)


def test_unprepared_auto_retains_existing_serving_path():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import torch
from tokenspeed_kernel.compile_monitor import mark_serving
from tokenspeed_kernel.ops.gemm import mm
from tokenspeed_kernel.ops.quantization import quantize_fp8
x = torch.randn(64,512,device='cuda',dtype=torch.bfloat16)
w = torch.randn(256,512,device='cuda',dtype=torch.bfloat16)
a,sa = quantize_fp8(x,granularity='token_group',group_size=128,solution='triton')
b,sb = quantize_fp8(w,granularity='block',block_size=(128,128),solution='triton')
kw = dict(A_scales=sa,B_scales=sb,out_dtype=torch.bfloat16,quant='mxfp8',block_size=[128,128])
expected = mm(a,b,solution='flashinfer',**kw)
mark_serving()
actual = mm(a,b,**kw)
torch.testing.assert_close(actual,expected,rtol=0,atol=0)
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
