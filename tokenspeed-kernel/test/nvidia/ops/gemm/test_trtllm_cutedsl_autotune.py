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

import ast
import json
from pathlib import Path

import pytest
import torch
from flashinfer.autotuner import AutoTuner
from tokenspeed_kernel.ops.gemm import mm
from tokenspeed_kernel.ops.tuning import (
    autotune,
    load_autotune_cache,
    save_autotune_cache,
)
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_blackwell, reason="Blackwell only"
)


def _inputs(m, n, k):
    torch.manual_seed(1639)
    a = torch.randn((m, k), device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn((n, k), device="cuda").to(torch.float8_e4m3fn)
    sa = torch.rand((m, k // 128), device="cuda") * 0.03 + 0.01
    sb = torch.rand((n // 128, k // 128), device="cuda") * 0.03 + 0.01
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    return [a, b, sa, sb, out]


def _reference(inputs):
    a, b, sa, sb, _ = inputs
    return (
        (a.float() * sa.repeat_interleave(128, 1))
        @ (b.float() * sb.repeat_interleave(128, 0).repeat_interleave(128, 1)).T
    ).to(torch.bfloat16)


def _mm(inputs):
    a, b, sa, sb, out = inputs
    return mm(
        a,
        b,
        A_scales=sa,
        B_scales=sb,
        quant="mxfp8",
        block_size=[128, 128],
        out_dtype=torch.bfloat16,
        out=out,
        solution="trtllm_cutedsl",
    )


@torch.inference_mode()
def test_startup_autotune_persists_fp8_tactics(tmp_path, monkeypatch):
    """Startup tuning must reach this GEMM and persist real tile choices."""
    inputs = _inputs(64, 384, 256)
    with autotune(tune_mode=True, tuning_buckets=(32, 64), round_up=False):
        actual = _mm(inputs)
    torch.testing.assert_close(actual, _reference(inputs), atol=1e-4, rtol=1e-2)
    path = str(tmp_path / "fp8-configs.json")
    assert save_autotune_cache(path, None, 0)
    entries = {
        key: value
        for key, value in json.loads(Path(path).read_text()).items()
        if "trtllm_cutedsl_fp8_blockscale" in key
    }
    assert len(entries) >= 2, "FP8 tile choices were not saved by startup autotune"
    # Emulate a cache with different winners at 32 and 64 rows. Both choices
    # are valid; the test checks preparation after restart, not timing noise.
    configs = json.loads(Path(path).read_text())
    for key in entries:
        m = ast.literal_eval(key)[2][0][0]
        configs[key][1] = (
            [False, [64, 128], [1, 1]] if m == 32 else [True, [256, 128], [2, 1]]
        )
    Path(path).write_text(json.dumps(configs))
    from tokenspeed_kernel.thirdparty.trtllm_blockwise import _launch as launcher

    tuner = AutoTuner.get()
    tuner.clear_cache()
    monkeypatch.setattr(launcher, "_compiled", {})
    assert load_autotune_cache(path, None, 0)

    def unexpected_profile(*args, **kwargs):
        pytest.fail("Reloaded FP8 tactics must not be profiled again")

    monkeypatch.setattr(tuner, "_profile_single_kernel", unexpected_profile)
    with autotune(tune_mode=True, tuning_buckets=(32, 64), round_up=False):
        _mm(inputs)

    def unexpected_compile(*args, **kwargs):
        pytest.fail("A new M must reuse the compiled dynamic-shape tactic")

    monkeypatch.setattr(launcher, "_compile", unexpected_compile)
    from tokenspeed_kernel.thirdparty.trtllm_blockwise import runner as runner_module

    selected = []

    def record_tactic(*args):
        selected.append(args[-1])
        return launcher._run(*args)

    monkeypatch.setattr(runner_module, "_run", record_tactic)
    for m in (63, 64, 65):
        tensors = _inputs(m, 384, 256)
        _mm(tensors)
        assert selected[-1] == (
            (False, (64, 128), (1, 1)) if m < 64 else (True, (256, 128), (2, 1))
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = _mm(tensors)
        tensors[0].copy_((-tensors[0].float()).to(tensors[0].dtype))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, _reference(tensors), atol=1e-4, rtol=1e-2)

    tensors = _inputs(33, 384, 256)
    tuner.clear_cache()
    _mm(tensors)
    assert selected[-1] == -1
    for key in entries:
        configs[key][1] = [False, [128, 128], [1, 1]]
    Path(path).write_text(json.dumps(configs))
    assert load_autotune_cache(path, None, 0)
    _mm(tensors)
    assert selected[-1] == (False, (128, 128), (1, 1))


@torch.inference_mode()
def test_fp8_all_tactics_preserve_values_and_graph(monkeypatch):
    """Every candidate must implement tail rows, scale blocks and graph replay."""
    from tokenspeed_kernel.thirdparty.trtllm_blockwise import _launch as launcher
    from tokenspeed_kernel.thirdparty.trtllm_blockwise.runner import BlockwiseGemmRunner

    monkeypatch.setattr(launcher, "_compiled", {})
    runner = BlockwiseGemmRunner()
    for m, n, k in ((1, 384, 256), (17, 384, 256), (129, 512, 512), (257, 640, 256)):
        inputs = _inputs(m, n, k)
        expected = _reference(inputs)
        tactics = runner.get_valid_tactics(inputs, None)
        assert (False, (64, 128), (1, 1)) in tactics
        assert (True, (128, 128), (2, 2)) in tactics
        assert (True, (256, 128), (2, 1)) in tactics
        for tactic in tactics:
            actual = runner(inputs, tactic=tactic)
            assert actual.data_ptr() == inputs[-1].data_ptr()
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-2)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                runner(inputs, tactic=tactic)
            inputs[0].copy_((-inputs[0].float()).to(inputs[0].dtype))
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, -expected, atol=1e-4, rtol=1e-2)
            inputs[0].copy_((-inputs[0].float()).to(inputs[0].dtype))


@torch.inference_mode()
def test_prepare_after_one_shape_warms_all_fallbacks(monkeypatch):
    """An eager small-M call must not make prepare skip the larger graph tiles."""
    from tokenspeed_kernel.thirdparty import trtllm_blockwise as adapter
    from tokenspeed_kernel.thirdparty.trtllm_blockwise import _launch as launcher

    AutoTuner.get().clear_cache()
    monkeypatch.setattr(launcher, "_compiled", {})
    adapter.blockwise_gemm(*_inputs(1, 256, 128))
    adapter.prepare(torch.device("cuda", 0))
    inputs = _inputs(129, 256, 128)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = adapter.blockwise_gemm(*inputs)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, _reference(inputs), atol=1e-4, rtol=1e-2)
