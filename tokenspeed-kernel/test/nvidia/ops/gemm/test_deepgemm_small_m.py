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

"""Native small-M tactics, global state restoration, cache and graph replay."""

import pytest
import torch

dg = pytest.importorskip("deep_gemm")
pytest.importorskip("flashinfer")
from flashinfer.autotuner import AutoTuner
from tokenspeed_kernel.ops.gemm.deepgemm import prepare_deepgemm_fp8_weight
from tokenspeed_kernel.ops.quantization import quantize_fp8
from tokenspeed_kernel.ops.tuning import autotune
from tokenspeed_kernel.thirdparty.deepgemm_small_m import (
    _runner,
    block_fp8_mm_tuned,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="requires SM100/SM103",
)


def _dequantize(codes, packed):
    shifts = torch.arange(4, device=packed.device, dtype=torch.int32) * 8
    exponents = (packed[..., None] >> shifts) & 255
    scales = (exponents << 23).contiguous().view(torch.float32)
    scales = scales.reshape(codes.shape[0], -1)[:, : codes.shape[1] // 128]
    return codes.float() * scales.repeat_interleave(128, 1)


def _relative_l2(actual, reference):
    return ((actual.float() - reference).norm() / reference.norm()).item()


@pytest.mark.parametrize("pdl", [False, True])
def test_tactics_and_graph(tmp_path, pdl, set_matmul_precision):
    set_matmul_precision("ieee")
    torch.manual_seed(81)
    n, k = 2560, 6144
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    b, sb = quantize_fp8(weight, granularity="block", block_size=[128, 128])
    b, sb = prepare_deepgemm_fp8_weight(b, sb)
    a, sa = quantize_fp8(
        torch.randn(128, k, device="cuda", dtype=torch.bfloat16),
        granularity="token_group",
        group_size=128,
        scale_encoding="packed_ue8m0",
        enable_pdl=pdl,
    )
    num_sms, old_pdl = dg.get_num_sms(), dg.get_pdl()
    runner = _runner(pdl, num_sms)
    tuner = AutoTuner.get()
    tuner.clear_cache()
    try:
        # A real prefill call discovers the decode shapes with canonical
        # packed strides; their scale layout differs from a[:4]/sa[:4].
        out = torch.empty(128, n, device="cuda", dtype=torch.bfloat16)
        with autotune(tune_mode=True, tuning_buckets=None, round_up=None):
            block_fp8_mm_tuned(a, b, sa, sb, out, enable_pdl=pdl)
        assert dg.get_num_sms() == num_sms
        cache_path = str(tmp_path / "tactics.json")
        tuner.save_configs(cache_path)
        tuner.clear_cache()
        assert tuner.load_configs(cache_path)
        weight_fp32 = _dequantize(b, sb)
        for rows in (1, 2, 4, 8, 16, 32):
            x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
            a, sa = quantize_fp8(
                x,
                granularity="token_group",
                group_size=128,
                scale_encoding="packed_ue8m0",
                enable_pdl=pdl,
            )
            out = torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)
            inputs = [a, b, sa, sb, out]
            reference = runner(inputs, tactic=num_sms).clone()
            reference_fp32 = _dequantize(a, sa) @ weight_fp32.t()
            for tactic in runner.get_valid_tactics(inputs, None):
                result = runner(inputs, tactic=tactic)
                # FP32 split-K can round a BF16 tie to the adjacent value.
                torch.testing.assert_close(result, reference, rtol=0.008, atol=0.002)
                assert _relative_l2(result, reference_fp32) < 0.002
                assert dg.get_num_sms() == num_sms

            def run():
                return block_fp8_mm_tuned(*inputs, enable_pdl=pdl)

            def run_split():
                return runner(inputs, tactic=-4)

            # Cover cache selection and a split-K graph even on a device
            # whose tuner happens to prefer the native single GEMM.
            for execute in (run, run_split):
                execute()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with autotune(tune_mode=True, tuning_buckets=None, round_up=None):
                    with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
                        execute()
                for magnitude in (0.01, 1.0, 16.0):
                    x.normal_().mul_(magnitude)
                    aq, sq = quantize_fp8(
                        x,
                        granularity="token_group",
                        group_size=128,
                        scale_encoding="packed_ue8m0",
                        enable_pdl=pdl,
                    )
                    a.copy_(aq)
                    sa.copy_(sq)
                    graph.replay()
                    actual = out.clone()
                    expected = runner(inputs, tactic=num_sms)
                    reference_fp32 = _dequantize(a, sa) @ weight_fp32.t()
                    torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.002)
                    assert _relative_l2(actual, reference_fp32) < 0.002
                    assert dg.get_num_sms() == num_sms
    finally:
        dg.set_num_sms(num_sms)
        dg.set_pdl(old_pdl)
        tuner.clear_cache()


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("rows", [4, 128])
@pytest.mark.parametrize("split_k", [False, True])
def test_quant_pack_to_gemm_graph_dependency(pdl, rows, split_k):
    from tokenspeed_kernel.ops.gemm import mm
    from tokenspeed_kernel.platform import pdl_enabled

    previous_pdl = pdl_enabled()
    old_dg_pdl = dg.get_pdl()
    pdl_enabled(pdl)
    try:
        width, n = 6144, 2560
        x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, width, device="cuda", dtype=torch.bfloat16) * 0.02
        b, sb = quantize_fp8(weight, granularity="block", block_size=[128, 128])
        b, sb = prepare_deepgemm_fp8_weight(b, sb)
        out = torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)

        def run(enabled):
            a, sa = quantize_fp8(
                x * 3,
                granularity="token_group",
                group_size=128,
                scale_encoding="packed_ue8m0",
                enable_pdl=enabled,
            )
            if split_k:
                return _runner(enabled, dg.get_num_sms())(
                    [a, b, sa, sb, out], tactic=-4
                )
            return mm(
                a,
                b,
                quant="mxfp8",
                A_scales=sa,
                B_scales=sb,
                block_size=[128, 128],
                out_dtype=torch.bfloat16,
                out=out,
                solution="deepgemm",
            )

        run(pdl)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
            run(pdl)
        for magnitude in (0.01, 1.0, 16.0):
            x.normal_().mul_(magnitude)
            graph.replay()
            actual = out.clone()
            pdl_enabled(False)
            expected = run(False)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            pdl_enabled(pdl)
    finally:
        dg.set_pdl(old_dg_pdl)
        pdl_enabled(previous_pdl)
