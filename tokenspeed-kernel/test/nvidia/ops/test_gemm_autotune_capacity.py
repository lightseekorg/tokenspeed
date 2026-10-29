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

"""Real GEMM calls verify shared startup coverage and cache reuse."""

import pytest
import torch
from tokenspeed_kernel.ops.gemm import mm, prepare_nvfp4_a16_weights
from tokenspeed_kernel.ops.tuning import (
    autotune,
    get_autotune_max_num_tokens,
    set_autotune_max_num_tokens,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.msa.cute.quantize import swizzle_nvfp4_scale_to_128x4

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="capacity regressions require FlashInfer SM100/103 GEMMs",
)


def _packed_ones(rows, cols):
    packed = torch.full((rows, cols // 2), 0x22, dtype=torch.uint8, device="cuda")
    scale = torch.ones(rows, cols // 16, dtype=torch.float8_e4m3fn, device="cuda")
    return packed, swizzle_nvfp4_scale_to_128x4(scale, rows=rows, cols=cols // 16)


def _run(kind, rows):
    n, k = 128, 256
    unit = torch.ones(1, device="cuda")
    if kind in ("fp8", "mxfp8"):
        x = torch.ones(rows, k, device="cuda", dtype=torch.float8_e4m3fn)
        weight = torch.ones(n, k, device="cuda", dtype=x.dtype)
        if kind == "fp8":
            return mm(
                x,
                weight.t(),
                A_scales=unit,
                B_scales=unit,
                out_dtype=torch.bfloat16,
                quant="fp8",
                override="flashinfer_mm_fp8_tensor_scaled",
            )
        return mm(
            x,
            weight,
            A_scales=torch.full((rows, k // 32), 127, dtype=torch.uint8, device="cuda"),
            B_scales=torch.full((n, k // 32), 127, dtype=torch.uint8, device="cuda"),
            out_dtype=torch.bfloat16,
            quant="mxfp8",
            block_size=[1, 32],
            override="flashinfer_mm_mxfp8",
        )

    weight, weight_scale = _packed_ones(2 * n if kind == "swiglu" else n, k)
    if kind == "w4a16":
        weight, weight_scale, alpha = prepare_nvfp4_a16_weights(
            weight, weight_scale, alpha=unit
        )
        return mm(
            torch.ones(rows, k, device="cuda", dtype=torch.bfloat16),
            weight,
            B_scales=weight_scale,
            alpha=alpha,
            out_dtype=torch.bfloat16,
            quant="nvfp4_a16",
            override="flashinfer_cute_dsl_mm_nvfp4_a16",
        )

    x, scale = _packed_ones(rows, k)
    if kind == "nvfp4":
        return mm(
            x,
            weight.t(),
            A_scales=scale,
            B_scales=weight_scale.t(),
            alpha=unit,
            out_dtype=torch.bfloat16,
            quant="nvfp4",
            override="flashinfer_mm_nvfp4",
        )
    assert kind == "swiglu"
    from tokenspeed_kernel.ops.gemm._cute_dsl.nvfp4_gemm_swiglu import (
        nvfp4_gemm_swiglu_nvfp4_quant,
    )

    return nvfp4_gemm_swiglu_nvfp4_quant(
        x, scale, weight, weight_scale, unit / k, unit, enable_pdl=False
    )


def _rows(tuner, op):
    return {
        key.nearest_profile[0][-2]
        for key in tuner.profiling_cache
        if key.custom_op == op
    }


@pytest.fixture
def isolated_tuner(monkeypatch, caplog):
    from flashinfer.autotuner import AutoTuner
    from flashinfer.jit.core import logger as jit_logger

    tuner = AutoTuner()
    monkeypatch.setattr(AutoTuner, "_instance", tuner)
    previous = get_autotune_max_num_tokens()
    set_autotune_max_num_tokens(8193)
    jit_logger.addHandler(caplog.handler)
    try:
        yield tuner
    finally:
        jit_logger.removeHandler(caplog.handler)
        set_autotune_max_num_tokens(previous)


@pytest.mark.parametrize(
    "kind,op",
    [
        ("mxfp8", "mxfp8_gemm"),
        ("fp8", "fp8_gemm"),
        ("nvfp4", "fp4_gemm"),
        ("w4a16", "bf16_fp4_cute_dsl_sm100_gemm"),
        ("swiglu", "nvfp4_gemm_swiglu_nvfp4_quant"),
    ],
)
def test_gemm_capacity(isolated_tuner, tmp_path, caplog, kind, op):
    tuner = isolated_tuner
    with autotune(tune_mode=True, tuning_buckets=None, round_up=None):
        _run(kind, 1)
    assert {1, 4, 512, 768, 8192, 16384} <= _rows(tuner, op)

    if kind == "mxfp8":
        # Exercise the shared persistence path once: retain the maximum
        # bucket and verify that a cached M1 call fills an interior hole.
        for key in list(tuner.profiling_cache):
            if key.custom_op == op and key.nearest_profile[0][-2] == 768:
                del tuner.profiling_cache[key]
        path = str(tmp_path / "cache.json")
        tuner.save_configs(path)
        tuner.clear_cache()
        tuner.load_configs(path)
        with autotune(tune_mode=True, tuning_buckets=None, round_up=None):
            _run(kind, 1)
        assert _rows(tuner, op) == {768}
        tuner.save_configs(path)
        tuner.clear_cache()
        tuner.load_configs(path)

    before = dict(tuner.profiling_cache)
    caplog.clear()
    for rows in (513, 8193):
        _run(kind, rows)
    torch.cuda.synchronize()
    assert tuner.profiling_cache == before
    assert not any("No tuned config covers" in r.getMessage() for r in caplog.records)


def test_tuning_controls_and_cleanup(isolated_tuner):
    from flashinfer.autotuner import autotune as native_autotune

    tuner = isolated_tuner
    with autotune(tune_mode=False, tuning_buckets=None, round_up=None):
        _run("fp8", 1)
    assert not tuner.profiling_cache

    with pytest.raises(RuntimeError, match="leave startup"), autotune(
        tune_mode=True, tuning_buckets=(1, 4), round_up=True
    ):
        _run("fp8", 1)
        raise RuntimeError("leave startup")
    assert _rows(tuner, "fp8_gemm") == {1, 4}
    assert not tuner.is_tuning_mode

    tuner.clear_cache()
    with native_autotune(tune_mode=True):
        _run("fp8", 1)
    assert _rows(tuner, "fp8_gemm") == {1}
