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
import tokenspeed_kernel.benchmark.cli as benchmark_cli
import tokenspeed_kernel.benchmark.runner as benchmark_runner_module
import tokenspeed_kernel.numerics.gemm  # noqa: F401
import torch
from tokenspeed_kernel.benchmark.config import BenchmarkConfig
from tokenspeed_kernel.benchmark.report import format_report
from tokenspeed_kernel.benchmark.result import export_results, import_results
from tokenspeed_kernel.benchmark.runner import BenchmarkRunner
from tokenspeed_kernel.benchmark.throughput import ThroughputCalculator
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.profiling import ProfilingConfig
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec
from tokenspeed_kernel.signature import format_signatures

pytestmark = [
    pytest.mark.usefixtures("fresh_registry"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]

_TEST_GEMM_TRAITS = {"b_layout": frozenset({"KN"})}
_TEST_SHAPES = [{"M": 8, "N": 8, "K": 8}]


def _torch_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
) -> torch.Tensor:
    _ = A_scales, B_scales, alpha, block_size
    return (A @ B).to(out_dtype)


def _register_test_gemm_kernels() -> None:
    registry = KernelRegistry.get()
    dtype = torch.float32

    ref_spec = KernelSpec(
        name="test_gemm_reference",
        family="gemm",
        mode="mm",
        solution="reference",
        format_signatures=format_signatures(("a", "b"), "dense", {dtype}),
        traits=_TEST_GEMM_TRAITS,
        priority=0,
    )
    fast_spec = KernelSpec(
        name="test_gemm_fast",
        family="gemm",
        mode="mm",
        solution="triton",
        format_signatures=format_signatures(("a", "b"), "dense", {dtype}),
        traits=_TEST_GEMM_TRAITS,
        priority=10,
    )

    registry.register(ref_spec, _torch_mm)
    registry.register(fast_spec, _torch_mm)


@pytest.fixture
def setup_gemm_case():
    Platform.reset()
    _register_test_gemm_kernels()

    yield
    Platform.reset()


def test_benchmark_kernel_returns_result(setup_gemm_case):
    runner = BenchmarkRunner(
        BenchmarkConfig(
            warmup_iters=0,
            bench_iters=3,
        )
    )
    results = runner.benchmark_kernel(
        "test_gemm_fast",
        shapes=_TEST_SHAPES,
        dtype=torch.float32,
    )

    assert len(results) == 1
    result = results[0]
    assert result.kernel_name == "test_gemm_fast"
    assert result.tflops is not None
    assert result.bandwidth_gb_s is not None
    assert result.numerics_passed is True
    assert result.max_abs_diff is not None
    assert result.max_rel_diff is not None


def test_benchmark_kernel_supports_cpu_wall_time(setup_gemm_case):
    runner = BenchmarkRunner(
        BenchmarkConfig(
            warmup_iters=0,
            bench_iters=3,
            use_cuda_events=False,
        )
    )
    results = runner.benchmark_kernel(
        "test_gemm_fast",
        shapes=_TEST_SHAPES,
        dtype=torch.float32,
    )

    assert len(results) == 1
    result = results[0]
    assert result.kernel_name == "test_gemm_fast"
    assert result.median_latency_us >= 0.0
    assert result.max_latency_us >= result.min_latency_us
    assert result.numerics_passed is True


def test_benchmark_kernel_can_disable_verification(setup_gemm_case):
    runner = BenchmarkRunner(
        BenchmarkConfig(
            warmup_iters=0,
            bench_iters=3,
            verify=False,
        )
    )
    results = runner.benchmark_kernel(
        "test_gemm_fast",
        shapes=_TEST_SHAPES,
        dtype=torch.float32,
    )

    assert len(results) == 1
    result = results[0]
    assert result.numerics_passed is None
    assert result.max_abs_diff is None
    assert result.max_rel_diff is None


def test_benchmark_op_includes_reference(setup_gemm_case):
    runner = BenchmarkRunner(
        BenchmarkConfig(
            warmup_iters=0,
            bench_iters=2,
        )
    )
    results = runner.benchmark_op(
        "gemm",
        "mm",
        shapes=_TEST_SHAPES,
        dtype=torch.float32,
    )

    names = {result.kernel_name for result in results}
    assert names == {"test_gemm_reference", "test_gemm_fast"}


def test_report_format_contains_expected_columns(setup_gemm_case):
    runner = BenchmarkRunner(
        BenchmarkConfig(
            warmup_iters=0,
            bench_iters=2,
        )
    )
    results = runner.benchmark_op(
        "gemm",
        "mm",
        shapes=_TEST_SHAPES,
        dtype=torch.float32,
    )
    report = format_report(results)

    assert "Kernel" in report
    assert "p50 (us)" in report
    assert "TFLOPs" in report
    assert "test_gemm_fast" in report
    assert "Numerics" not in report


def test_export_import_roundtrip(tmp_path, setup_gemm_case):
    runner = BenchmarkRunner(
        BenchmarkConfig(
            warmup_iters=0,
            bench_iters=2,
        )
    )
    results = runner.benchmark_op(
        "gemm",
        "mm",
        shapes=_TEST_SHAPES,
        dtype=torch.float32,
    )

    export_path = tmp_path / "bench_results.json"
    export_results(results, export_path)
    loaded = import_results(export_path)

    assert len(loaded) == len(results)
    assert loaded[0].kernel_name == results[0].kernel_name
    assert loaded[0].shape_params == results[0].shape_params
    assert loaded[0].numerics_passed == results[0].numerics_passed
    assert loaded[0].max_abs_diff == results[0].max_abs_diff
    assert loaded[0].max_rel_diff == results[0].max_rel_diff


def test_attention_decode_throughput_with_explicit_kv_heads():
    shape = {
        "batch": 2,
        "seq_len": 4,
        "num_q_heads": 8,
        "num_kv_heads": 2,
        "head_dim": 16,
    }

    tflops, bandwidth = ThroughputCalculator.compute(
        "attention",
        "decode",
        shape,
        latency_us=1000.0,
        dtype=torch.float16,
    )

    assert tflops == pytest.approx(4.096e-6)
    assert bandwidth == pytest.approx(0.002048)


def test_attention_decode_throughput_with_shape_aliases():
    shape = {
        "batch_size": 1,
        "max_seq_len": 2,
        "heads": 4,
        "head_dim": 8,
    }

    tflops, bandwidth = ThroughputCalculator.compute(
        "attn",
        "decode",
        shape,
        latency_us=1000.0,
        dtype=torch.float16,
    )

    assert tflops == pytest.approx(2.56e-7)
    assert bandwidth == pytest.approx(0.000384)


def test_attention_decode_throughput_missing_required_shape_returns_none():
    shape = {
        "batch": 2,
        "seq_len": 4,
        "num_q_heads": 8,
    }

    tflops, bandwidth = ThroughputCalculator.compute(
        "attention",
        "decode",
        shape,
        latency_us=1000.0,
        dtype=torch.float16,
    )

    assert tflops is None
    assert bandwidth is None


def test_benchmark_config_rejects_invalid_proton_data():
    cfg = BenchmarkConfig(
        proton_profile=True,
        proton_config=ProfilingConfig(data="invalid"),
    )

    with pytest.raises(ValueError, match="proton_config.data"):
        cfg.validate()


def test_benchmark_kernel_uses_profiling_context_when_enabled(
    setup_gemm_case, monkeypatch
):
    configs: list[ProfilingConfig] = []
    trace: list[str] = []

    class _Ctx:
        def __enter__(self):
            trace.append("enter")
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb
            trace.append("exit")

    def _fake_profiling(cfg: ProfilingConfig):
        configs.append(cfg)
        return _Ctx()

    monkeypatch.setattr(benchmark_runner_module, "profiling", _fake_profiling)

    runner = BenchmarkRunner(
        BenchmarkConfig(
            warmup_iters=0,
            bench_iters=1,
            verify=False,
            proton_profile=True,
        )
    )
    results = runner.benchmark_kernel(
        "test_gemm_fast",
        shapes=_TEST_SHAPES,
        dtype=torch.float32,
    )

    assert len(results) == 1
    assert trace == ["enter", "exit"]
    assert len(configs) == 1
    assert configs[0].output == "bench_test_gemm_fast"
    assert configs[0].data == "trace"


def test_benchmark_op_profiles_once_when_enabled(setup_gemm_case, monkeypatch):
    configs: list[ProfilingConfig] = []
    trace: list[str] = []

    class _Ctx:
        def __enter__(self):
            trace.append("enter")
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb
            trace.append("exit")

    def _fake_profiling(cfg: ProfilingConfig):
        configs.append(cfg)
        return _Ctx()

    monkeypatch.setattr(benchmark_runner_module, "profiling", _fake_profiling)

    runner = BenchmarkRunner(
        BenchmarkConfig(
            warmup_iters=0,
            bench_iters=1,
            verify=False,
            proton_profile=True,
        )
    )
    results = runner.benchmark_op(
        "gemm",
        "mm",
        shapes=_TEST_SHAPES,
        dtype=torch.float32,
    )

    assert {result.kernel_name for result in results} == {
        "test_gemm_reference",
        "test_gemm_fast",
    }
    assert trace == ["enter", "exit"]
    assert len(configs) == 1
    assert configs[0].output == "bench_gemm_mm"
    assert configs[0].data == "trace"


def test_benchmark_kernel_uses_explicit_proton_config(setup_gemm_case, monkeypatch):
    configs: list[ProfilingConfig] = []

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb

    def _fake_profiling(cfg: ProfilingConfig):
        configs.append(cfg)
        return _Ctx()

    monkeypatch.setattr(benchmark_runner_module, "profiling", _fake_profiling)

    custom_cfg = ProfilingConfig(
        output="custom_profile",
        data="tree",
        backend="cupti",
        mode="pcsampling",
        hook="triton",
        output_format="chrome_trace",
    )
    runner = BenchmarkRunner(
        BenchmarkConfig(
            warmup_iters=0,
            bench_iters=1,
            verify=False,
            proton_profile=True,
            proton_config=custom_cfg,
        )
    )
    runner.benchmark_kernel(
        "test_gemm_fast",
        shapes=_TEST_SHAPES,
        dtype=torch.float32,
    )

    assert len(configs) == 1
    assert configs[0] is custom_cfg


def test_benchmark_cli_proton_flag_sets_profile_mode(monkeypatch):
    class _FakeRunner:
        last_config: BenchmarkConfig | None = None
        last_call: tuple | None = None

        def __init__(self, config: BenchmarkConfig):
            config.validate()
            type(self).last_config = config

        def benchmark_kernel(self, *args, **kwargs):
            type(self).last_call = ("kernel", args, kwargs)
            return []

        def benchmark_op(self, *args, **kwargs):
            type(self).last_call = ("op", args, kwargs)
            return []

        def benchmark_all(self, *args, **kwargs):
            type(self).last_call = ("all", args, kwargs)
            return []

    monkeypatch.setattr(benchmark_cli, "BenchmarkRunner", _FakeRunner)
    monkeypatch.setattr(benchmark_cli, "load_builtin_kernels", lambda: None)
    monkeypatch.setattr(benchmark_cli, "format_report", lambda _results: "ok")

    rc = benchmark_cli.main(["--op", "gemm.mm", "--proton"])

    assert rc == 0
    assert _FakeRunner.last_config is not None
    assert _FakeRunner.last_config.proton_profile is True
    assert _FakeRunner.last_config.proton_config is None
    assert _FakeRunner.last_call is not None
    assert _FakeRunner.last_call[0] == "op"


def test_benchmark_cli_builds_proton_config_from_flags(monkeypatch):
    class _FakeRunner:
        last_config: BenchmarkConfig | None = None

        def __init__(self, config: BenchmarkConfig):
            config.validate()
            type(self).last_config = config

        def benchmark_kernel(self, *args, **kwargs):
            _ = args, kwargs
            return []

        def benchmark_op(self, *args, **kwargs):
            _ = args, kwargs
            return []

        def benchmark_all(self, *args, **kwargs):
            _ = args, kwargs
            return []

    monkeypatch.setattr(benchmark_cli, "BenchmarkRunner", _FakeRunner)
    monkeypatch.setattr(benchmark_cli, "load_builtin_kernels", lambda: None)
    monkeypatch.setattr(benchmark_cli, "format_report", lambda _results: "ok")

    rc = benchmark_cli.main(
        [
            "--op",
            "gemm.mm",
            "--proton-output",
            "bench_cli",
            "--proton-data",
            "tree",
            "--proton-backend",
            "cupti",
            "--proton-mode",
            "pcsampling",
            "--proton-hook",
            "none",
            "--proton-output-format",
            "chrome_trace",
        ]
    )

    assert rc == 0
    assert _FakeRunner.last_config is not None
    assert _FakeRunner.last_config.proton_profile is True
    assert _FakeRunner.last_config.proton_config is not None
    proton_cfg = _FakeRunner.last_config.proton_config
    assert proton_cfg.output == "bench_cli"
    assert proton_cfg.data == "tree"
    assert proton_cfg.backend == "cupti"
    assert proton_cfg.mode == "pcsampling"
    assert proton_cfg.hook is None
    assert proton_cfg.output_format == "chrome_trace"
