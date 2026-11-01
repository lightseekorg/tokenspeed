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

"""FP8 tactic correctness, cache persistence and graph replay."""

import pytest
import torch
from flashinfer.autotuner import AutoTuner
from tokenspeed_kernel.ops.gemm.flashinfer import (
    _FP8_BLOCKSCALE_RUNNERS,
    _FP8_BLOCKSCALE_TUNING,
    flashinfer_fp8_blockscale_tuned,
)
from tokenspeed_kernel.ops.quantization import quantize_fp8
from tokenspeed_kernel.ops.tuning import autotune
from tokenspeed_kernel.platform import pdl_enabled

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires SM10x",
)


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("m", [1, 8, 16, 128, 4096])
def test_tactics_cache_and_graph(tmp_path, pdl, m):
    old = pdl_enabled()
    pdl_enabled(pdl)
    tuner = AutoTuner.get()
    tuner.clear_cache()
    try:
        a, sa = quantize_fp8(
            torch.randn(m, 512, device="cuda", dtype=torch.bfloat16),
            granularity="token_group",
            group_size=128,
            scale_encoding="float32",
            enable_pdl=pdl,
        )
        b, sb = quantize_fp8(
            torch.randn(512, 512, device="cuda", dtype=torch.bfloat16),
            granularity="block",
            block_size=[128, 128],
        )
        out = torch.empty(m, 512, device="cuda", dtype=torch.bfloat16)
        inputs = [a, b, sa, sb, out]
        runner = _FP8_BLOCKSCALE_RUNNERS[pdl]
        ref = runner(inputs, tactic=1).clone()
        torch.testing.assert_close(runner(inputs, tactic=2), ref, rtol=0.01, atol=0.01)
        with autotune(tune_mode=True, tuning_buckets=None, round_up=None):
            flashinfer_fp8_blockscale_tuned(*inputs)
        _, tactic = tuner.choose_one(
            "tokenspeed_fp8_blockscale_gemm", [runner], _FP8_BLOCKSCALE_TUNING, inputs
        )
        assert tactic in (1, 2)
        if m >= 128:
            decode_inputs = [a[:8], b, sa[:8], sb, out[:8]]
            _, decode_tactic = tuner.choose_one(
                "tokenspeed_fp8_blockscale_gemm",
                [runner],
                _FP8_BLOCKSCALE_TUNING,
                decode_inputs,
            )
            assert decode_tactic in (1, 2)
        path = str(tmp_path / "cache.json")
        tuner.save_configs(path)
        tuner.clear_cache()
        assert tuner.load_configs(path)
        _, restored = tuner.choose_one(
            "tokenspeed_fp8_blockscale_gemm", [runner], _FP8_BLOCKSCALE_TUNING, inputs
        )
        assert restored == tactic
        for cached in (True, False):
            if not cached:
                tuner.clear_cache()
            flashinfer_fp8_blockscale_tuned(*inputs)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            # Even an enclosing tuning context must not profile inside capture.
            with autotune(tune_mode=True, tuning_buckets=None, round_up=None):
                with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
                    flashinfer_fp8_blockscale_tuned(*inputs)
            for _ in range(3):
                a.copy_(torch.randn_like(a, dtype=torch.float32).to(a.dtype))
                graph.replay()
                actual = out.clone()
                expected = runner(inputs, tactic=tactic if cached else 1)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        pdl_enabled(old)
        tuner.clear_cache()
