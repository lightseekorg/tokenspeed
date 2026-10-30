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
from tokenspeed_kernel.ops.gemm import mm
from tokenspeed_kernel.ops.tuning import (
    autotune,
    load_autotune_cache,
    save_autotune_cache,
)
from tokenspeed_kernel.platform import current_platform

if current_platform().is_blackwell:
    from flashinfer.autotuner import AutoTuner

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
    # are valid; the test checks cache reuse after restart, not timing noise.
    configs = json.loads(Path(path).read_text())
    for key in entries:
        m = ast.literal_eval(key)[2][0][0]
        configs[key][1] = (
            [False, [64, 128], [1, 1]] if m == 32 else [True, [128, 128], [2, 1]]
        )
    Path(path).write_text(json.dumps(configs))
    from tokenspeed_kernel.thirdparty.trtllm_blockwise import _launch as launcher

    tuner = AutoTuner.get()
    tuner.clear_cache()
    monkeypatch.setattr(launcher, "_compiled", {})
    assert load_autotune_cache(path, None, 0)
    assert not launcher._compiled

    def unexpected_profile(*args, **kwargs):
        pytest.fail("Reloaded FP8 tactics must not be profiled again")

    monkeypatch.setattr(tuner, "_profile_single_kernel", unexpected_profile)
    with autotune(tune_mode=True, tuning_buckets=(32, 64), round_up=False):
        _mm(inputs)
    from tokenspeed_kernel.thirdparty.trtllm_blockwise import runner as runner_module

    selected = []

    def record_tactic(*args):
        selected.append(args[-1])
        return launcher._run(*args)

    monkeypatch.setattr(runner_module, "_run", record_tactic)
    for m in (32, 63, 64, 65):
        tensors = _inputs(m, 384, 256)
        _mm(tensors)
        assert selected[-1] == (
            (False, (64, 128), (1, 1)) if m <= 32 else (True, (128, 128), (2, 1))
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = _mm(tensors)
        tensors[0].copy_((-tensors[0].float()).to(tensors[0].dtype))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, _reference(tensors), atol=1e-4, rtol=1e-2)

    assert {tactic for _, tactic in launcher._compiled} == {
        (False, (64, 128), (1, 1)),
        (True, (128, 128), (2, 1)),
    }
    tensors = _inputs(33, 384, 256)
    tuner.clear_cache()
    _mm(tensors)
    assert selected[-1] == -1
    for key in entries:
        configs[key][1] = [False, [128, 128], [1, 1]]
    Path(path).write_text(json.dumps(configs))
    assert load_autotune_cache(path, None, 0)
    actual = _mm(tensors)
    assert selected[-1] == (False, (128, 128), (1, 1))
    torch.testing.assert_close(actual, _reference(tensors), atol=1e-4, rtol=1e-2)


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
        torch.testing.assert_close(runner(inputs), expected, atol=1e-4, rtol=1e-2)
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
def test_fallback_compiles_only_selected_tactic_and_reuses_it(monkeypatch):
    """Each fallback compiles on demand and supports new shapes during capture."""
    from tokenspeed_kernel.thirdparty import trtllm_blockwise as adapter
    from tokenspeed_kernel.thirdparty.trtllm_blockwise import _launch as launcher

    AutoTuner.get().clear_cache()
    monkeypatch.setattr(launcher, "_compiled", {})
    expected_tactics = set()
    for warmup_m, capture_m, tactic in (
        (1, 64, (False, (64, 128), (1, 1))),
        (65, 128, (False, (128, 128), (1, 1))),
        (129, 257, (True, (256, 128), (2, 1))),
    ):
        warmup = _inputs(warmup_m, 256, 128)
        actual = adapter.blockwise_gemm(*warmup)
        torch.testing.assert_close(actual, _reference(warmup), atol=1e-4, rtol=1e-2)
        expected_tactics.add((warmup[0].device.index, tactic))
        assert launcher._compiled.keys() == expected_tactics
        compiled = launcher._compiled.copy()

        inputs = _inputs(capture_m, 384, 256)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = adapter.blockwise_gemm(*inputs)
        inputs[0].copy_((-inputs[0].float()).to(inputs[0].dtype))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, _reference(inputs), atol=1e-4, rtol=1e-2)
        assert launcher._compiled == compiled


@torch.inference_mode()
def test_loaded_tactics_compile_lazily_and_match_hybrid_buckets(tmp_path, monkeypatch):
    """Loaded tactics compile on first use and work with eager and graph execution."""
    from tokenspeed_kernel.thirdparty.trtllm_blockwise import _launch as launcher
    from tokenspeed_kernel.thirdparty.trtllm_blockwise import runner as runner_module

    tuner = AutoTuner.get()
    tuner.clear_cache()
    monkeypatch.setenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "0")
    monkeypatch.setattr(launcher, "_compiled", {})
    extras = (*runner_module._CACHE_EXTRAS, torch.cuda.current_device())
    lower = (False, (64, 128), (1, 1))
    upper = (True, (128, 128), (2, 1))
    configs = {}
    for bucket, tactic in ((256, lower), (512, lower), (768, upper), (8192, lower)):
        shapes = ((bucket, 256), (384, 256), (bucket, 2), (3, 2), (bucket, 384))
        key = str(
            ("trtllm_cutedsl_fp8_blockscale", "BlockwiseGemmRunner", shapes, extras)
        )
        configs[key] = ["BlockwiseGemmRunner", tactic]
        if bucket == 512:
            key_512 = key
    path = tmp_path / "hybrid-configs.json"
    path.write_text(json.dumps(configs))
    compiled = []
    original_compile = launcher._compile

    def record_compile(*args):
        compiled.append(args[-1])
        return original_compile(*args)

    monkeypatch.setattr(launcher, "_compile", record_compile)
    assert load_autotune_cache(str(path), None, 0)
    assert not compiled

    selected = []
    original_run = runner_module._run

    def record_run(*args):
        selected.append(args[-1])
        return original_run(*args)

    monkeypatch.setattr(runner_module, "_run", record_run)
    for m, expected in ((300, lower), (700, upper), (800, upper), (8193, lower)):
        inputs = _inputs(m, 384, 256)
        actual = _mm(inputs)
        assert selected[-1] == expected
        torch.testing.assert_close(actual, _reference(inputs), atol=1e-4, rtol=1e-2)
    assert compiled == [lower, upper]

    # With the exact bucket missing, equidistant cached buckets prefer smaller M.
    del tuner._file_configs[key_512]
    inputs = _inputs(512, 384, 256)
    actual = _mm(inputs)
    assert selected[-1] == lower
    torch.testing.assert_close(actual, _reference(inputs), atol=1e-4, rtol=1e-2)
    inputs = _inputs(700, 384, 256)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _mm(inputs)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, _reference(inputs), atol=1e-4, rtol=1e-2)
    tuner.clear_cache()


@torch.inference_mode()
def test_nearest_tactic_from_in_memory_cache(monkeypatch):
    """Nearest-M reuse also works before the tuning cache is saved."""
    from tokenspeed_kernel.thirdparty.trtllm_blockwise import runner as runner_module

    tuner = AutoTuner.get()
    tuner.clear_cache()
    monkeypatch.setenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "0")
    lower = (False, (64, 128), (1, 1))
    upper = (True, (128, 128), (2, 1))
    extras = (*runner_module._CACHE_EXTRAS, torch.cuda.current_device())
    for m, tactic in ((64, lower), (256, upper)):
        shapes = ((m, 256), (384, 256), (m, 2), (3, 2), (m, 384))
        key = AutoTuner._get_cache_key(
            "trtllm_cutedsl_fp8_blockscale",
            runner_module._runner,
            shapes,
            runner_module._runner.tuning_config,
            extras,
        )
        tuner.profiling_cache[key] = (tactic, None)

    selected = []
    original_run = runner_module._run

    def record_run(*args):
        selected.append(args[-1])
        return original_run(*args)

    monkeypatch.setattr(runner_module, "_run", record_run)
    for m, n, expected in ((112, 384, lower), (800, 384, upper), (800, 512, -1)):
        inputs = _inputs(m, n, 256)
        actual = _mm(inputs)
        assert selected[-1] == expected
        torch.testing.assert_close(actual, _reference(inputs), atol=1e-4, rtol=1e-2)

    from flashinfer.autotuner import autotune as flashinfer_autotune

    with flashinfer_autotune(
        tune_mode=False, skip_ops={"trtllm_cutedsl_fp8_blockscale"}
    ):
        _mm(_inputs(800, 384, 256))
    assert selected[-1] == -1
    tuner.clear_cache()


@pytest.mark.parametrize("l2_ratio,expected_groups", [(1, 1), (5, 6), (12, 13)])
@torch.inference_mode()
def test_autotune_rotates_and_releases_profile_buffers(
    monkeypatch, l2_ratio, expected_groups
):
    """Each timed graph visits every L2-sized group and releases it."""
    import gc
    import weakref
    from collections import Counter, defaultdict

    from flashinfer.autotuner import is_in_profile_measurement
    from tokenspeed_kernel.thirdparty.trtllm_blockwise import runner as runner_module

    tuner = AutoTuner.get()
    tuner.clear_cache()
    monkeypatch.setenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "0")
    monkeypatch.setattr(tuner, "repeat", 7)
    inputs = _inputs(32, 384, 256)
    group_bytes = sum(tensor.numel() * tensor.element_size() for tensor in inputs)
    l2_bytes = l2_ratio * group_bytes + (expected_groups > l2_ratio)
    monkeypatch.setattr(
        tuner, "_get_l2_cache_size_in_bytes", lambda device_id=None: l2_bytes
    )
    original_repeat = tuner.repeat
    original_addresses = {tensor.data_ptr() for tensor in inputs}
    observed = defaultdict(list)
    temporary_refs = []
    original_run = runner_module._run

    def record_profile(a, b, sa, sb, out, tactic):
        tensors = (a, b, sa, sb, out)
        if is_in_profile_measurement() and torch.cuda.is_current_stream_capturing():
            observed[tactic].append(tuple(tensor.data_ptr() for tensor in tensors))
            temporary_refs.extend(
                weakref.ref(tensor)
                for tensor in tensors
                if tensor.data_ptr() not in original_addresses
            )
        return original_run(a, b, sa, sb, out, tactic)

    monkeypatch.setattr(runner_module, "_run", record_profile)
    with autotune(tune_mode=True, tuning_buckets=(32,), round_up=False):
        actual = _mm(inputs)
    assert tuner.repeat == original_repeat
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, _reference(inputs), atol=1e-4, rtol=1e-2)
    assert observed
    calls_per_group = max(1, (original_repeat + expected_groups - 1) // expected_groups)
    for addresses in observed.values():
        assert len(addresses) == expected_groups * calls_per_group
        assert all(
            set(Counter(row[index] for row in addresses).values()) == {calls_per_group}
            for index in range(5)
        )
    gc.collect()
    assert temporary_refs and all(ref() is None for ref in temporary_refs)
    tuner.clear_cache()
