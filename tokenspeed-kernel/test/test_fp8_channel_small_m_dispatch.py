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

"""The small-M FP8 GEMM's selection, dispatch and launch arguments, on CPU.

On a synthetic SM100 / SM103 / SM90 platform: which ``gemm.mm`` kernel serves
per-token x per-channel FP8 rows, what it receives for ``quant="fp8"``
without ``A_scales`` (FP8 codes and ``[M, 1]`` scales, from the Triton
quantizer up to 64 rows and from TRT-LLM's token quantizer above, as
``quantize_fp8`` selects them by row count), and the scale layouts and tile
constexprs the small-M wrapper launches with. Also how the functions that
write that quantizer's codes from an RMSNorm or a SwiGLU launch their kernels.
Kernel bodies and launches are recorders; the numerics run on the GPU
(``test/nvidia/ops/gemm/test_fp8_channel_small_m.py``,
``test_fp8_unscaled_consumers.py`` and ``layernorm/test_rmsnorm_token_fp8.py``).
"""

from __future__ import annotations

import dataclasses

import pytest
import tokenspeed_kernel.ops.gemm as gemm
import tokenspeed_kernel.ops.layernorm.triton as triton_layernorm
import tokenspeed_kernel.ops.quantization as quantization
import torch
from tokenspeed_kernel.ops.gemm import triton_fp8_channel
from tokenspeed_kernel.ops.quantization import (
    quantize_fp8,
    rmsnorm_quantize_fp8,
    silu_and_mul_quantize_fp8,
)
from tokenspeed_kernel.ops.quantization import triton as quant_triton
from tokenspeed_kernel.platform import ArchVersion, Platform, pdl_enabled
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import SelectedKernel

FP8 = torch.float8_e4m3fn
SMALL_M = "triton_mm_fp8_channel_small_m"
PER_TOKEN = "triton_quantize_fp8_per_token"


@pytest.fixture
def arch(b200_platform):
    """Switch to a synthetic NVIDIA platform of the given arch; restored after."""
    original = Platform.get()

    def switch(major: int, minor: int) -> None:
        Platform.override(
            dataclasses.replace(b200_platform, arch_version=ArchVersion(major, minor))
        )
        KernelRegistry.get().clear_cache()

    switch(10, 0)
    yield switch
    Platform.override(original)
    KernelRegistry.get().clear_cache()


def _stand_in_quantizer(monkeypatch, quantized):
    """TRT-LLM's per-token quantizer (codes, FP32 [M, 1] scales) and the Triton
    one, recorded as ("trtllm" | "triton", rows, the Triton one's enable_pdl)."""

    def quantize(x):
        scale = (x.float().abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-12)
        return (x.float() / scale).clamp(-448, 448).to(FP8), scale

    def quantize_e4m3_activation(x):
        quantized.append(("trtllm", x, None))
        return quantize(x)

    def fp8_quantize_per_token(x, enable_pdl):
        quantized.append(("triton", x, enable_pdl))
        return quantize(x)

    monkeypatch.setattr(
        torch.ops.tensorrt_llm,
        "quantize_e4m3_activation",
        quantize_e4m3_activation,
        raising=False,
    )
    monkeypatch.setattr(quant_triton, "fp8_quantize_per_token", fp8_quantize_per_token)


def _recording_kernels(monkeypatch, calls):
    select = gemm.select_kernel

    def selected(*args, **kwargs):
        kernel = select(*args, **kwargs)

        def run(A, B, A_scales, B_scales, out_dtype, **extra):
            calls.append(dict(kernel=kernel.name, A=A, A_scales=A_scales))
            return torch.zeros(A.shape[0], B.shape[1], dtype=out_dtype)

        return SelectedKernel(kernel.name, run)

    monkeypatch.setattr(gemm, "select_kernel", selected)


def _weight(n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A ``[K, N]`` column-major FP8 weight and its FP32 ``[N, 1]`` scales."""
    return torch.zeros(n, k, dtype=FP8).t(), torch.ones(n, 1)


@pytest.mark.parametrize("major,minor", [(10, 0), (10, 3)])
@pytest.mark.parametrize("m", [1, 8, 9, 64])
def test_decode_rows_reach_the_small_m_gemm_quantized_by_triton(
    arch, monkeypatch, major, minor, m
):
    arch(major, minor)
    calls, quantized = [], []
    _stand_in_quantizer(monkeypatch, quantized)
    _recording_kernels(monkeypatch, calls)
    weight, scales = _weight(768, 4096)
    x = torch.randn(m, 4096).to(torch.bfloat16)
    gemm.mm(x, weight, B_scales=scales, out_dtype=torch.bfloat16, quant="fp8")
    (call,) = calls
    assert call["kernel"] == SMALL_M
    ((source, rows, enable_pdl),) = quantized
    assert source == "triton" and rows is x and enable_pdl == pdl_enabled()
    assert call["A"].dtype == FP8 and tuple(call["A_scales"].shape) == (m, 1)


@pytest.mark.parametrize(
    "major,minor,m,k,expected,quantizer",
    [
        (10, 0, 65, 4096, "torch_mm_fp8_scaled", "trtllm"),
        (10, 3, 65, 4096, "triton_mm_fp8_scaled", "trtllm"),
        (10, 0, 8, 1000, "triton_mm_fp8_scaled", "triton"),
    ],
)
def test_other_kernels_receive_rows_quantized_per_token(
    arch, require, monkeypatch, major, minor, m, k, expected, quantizer
):
    arch(major, minor)
    if quantizer == "trtllm":
        require("quantization", "fp8_with_scale", "trtllm", torch.bfloat16, "x")
    calls, quantized = [], []
    _stand_in_quantizer(monkeypatch, quantized)
    _recording_kernels(monkeypatch, calls)
    weight, scales = _weight(768, k)
    x = torch.randn(m, k).to(torch.bfloat16)
    gemm.mm(x, weight, B_scales=scales, out_dtype=torch.bfloat16, quant="fp8")
    (call,) = calls
    assert call["kernel"] == expected
    ((source, rows, _),) = quantized
    assert source == quantizer and rows is x
    assert call["A"].dtype == FP8 and tuple(call["A_scales"].shape) == (m, 1)


@pytest.mark.parametrize("major,minor", [(9, 0), (12, 0)])
def test_no_small_m_gemm_outside_sm100_to_sm103(arch, major, minor):
    arch(major, minor)
    names = {
        spec.name
        for spec in KernelRegistry.get().get_for_operator(
            "gemm", "mm", platform=Platform.get()
        )
    }
    assert SMALL_M not in names


def test_prequantized_rows_keep_their_scales(arch, monkeypatch):
    calls, quantized = [], []
    _stand_in_quantizer(monkeypatch, quantized)
    _recording_kernels(monkeypatch, calls)
    weight, scales = _weight(768, 4096)
    codes = torch.zeros(4, 4096, dtype=FP8)
    a_scales = torch.ones(4, 1)
    gemm.mm(
        codes,
        weight,
        A_scales=a_scales,
        B_scales=scales,
        out_dtype=torch.bfloat16,
        quant="fp8",
    )
    (call,) = calls
    assert call["kernel"] == SMALL_M
    assert call["A"] is codes and call["A_scales"] is a_scales
    assert not quantized


def test_online_signature_names_the_quantized_activation():
    weight, scales = _weight(256, 512)
    x = torch.empty(4, 512, dtype=torch.bfloat16)
    signature = gemm._gemm_format_signature(
        x, weight, None, scales, torch.bfloat16, "fp8", None
    )
    a = signature.format_for("a")
    assert a.format == "scaled-fp8" and a.storage_dtype == FP8
    assert a.scale.granularity == "channel" and a.scale.storage_dtype == torch.float32


class _LaunchRecorder:
    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append(dict(grid=grid, args=args, kwargs=kwargs))

        return launch


@pytest.mark.parametrize(
    "m,a_layout,b_layout",
    [
        (1, "M1", "N1"),
        (3, "M", "N1"),
        (5, "T", "T"),
        (16, "M1", "N1"),
        (17, "M1", "N1"),
        (33, "M1", "T"),
        (64, "M", "N1"),
    ],
)
# (N, K): 8-way tensor-parallel shards of Llama-3.1-8B (fused Q/K/V, output,
# gate/up) and Qwen3-30B-A3B (output, N = 2,048), and a synthetic partial
# block.
@pytest.mark.parametrize(
    "n,k", [(768, 4096), (4096, 512), (3584, 4096), (2048, 512), (200, 384)]
)
def test_launch_arguments(arch, monkeypatch, m, a_layout, b_layout, n, k):
    """Scale layouts become the per-row / per-column constexprs; BLOCK_M
    buckets M to 16, 32 or 64; the tile is the measured table's for the
    measured shards and the rule's otherwise."""
    recorder = _LaunchRecorder()
    monkeypatch.setattr(triton_fp8_channel, "_fp8_channel_small_m_kernel", recorder)
    weight = torch.zeros(n, k, dtype=FP8)
    b_scales = torch.ones(n, 1) if b_layout == "N1" else torch.ones(1)
    a = torch.zeros(m, k, dtype=FP8)
    a_scales = {"M1": torch.ones(m, 1), "M": torch.ones(m), "T": torch.ones(1, 1)}[
        a_layout
    ]
    out = triton_fp8_channel.triton_mm_fp8_channel_small_m(
        a, weight.t(), a_scales, b_scales, torch.bfloat16
    )
    assert tuple(out.shape) == (m, n) and out.dtype == torch.bfloat16
    (launch,) = recorder.launches
    kwargs = launch["kwargs"]
    assert "QUANTIZE_A" not in kwargs
    assert kwargs["A_SCALE_PER_ROW"] is (a_scales.numel() > 1)
    assert kwargs["B_SCALE_PER_COL"] is (b_layout == "N1")
    block_m = max(16, 1 << (m - 1).bit_length())
    assert kwargs["BLOCK_M"] == block_m
    tile = (
        kwargs["BLOCK_N"],
        kwargs["BLOCK_K"],
        kwargs["num_warps"],
        kwargs["num_stages"],
    )
    measured = triton_fp8_channel._SMALL_M_TILES.get((n, k, block_m))
    if measured is not None:
        assert tile == measured
    assert launch["grid"] == ((n + kwargs["BLOCK_N"] - 1) // kwargs["BLOCK_N"],)
    assert k % kwargs["BLOCK_K"] == 0
    # M, N, K and the strides are runtime arguments.
    assert launch["args"][5:11] == (m, n, k, k, k, n)


def test_tiles_follow_the_rule_unless_measured(monkeypatch):
    """The rule fitted to the GB200 sweep (1-16 | 32 | 64 rows), a measured
    ``_SMALL_M_TILES`` entry taking precedence for its shape, and an entry
    whose BLOCK_K does not divide K refused."""
    config = triton_fp8_channel._small_m_config
    # Llama-3.1-8B shards at 8-way tensor parallelism: Q/K/V (768, 4096),
    # output (4096, 512), gate/up (3584, 4096), down (4096, 1792).
    assert config(1, 768, 4096) == (16, 16, 512, 4, 4)
    assert config(32, 768, 4096) == (32, 16, 512, 4, 3)
    assert config(64, 768, 4096) == (64, 64, 256, 8, 3)
    assert config(8, 4096, 512) == (16, 64, 256, 4, 4)
    assert config(17, 3584, 4096) == (32, 64, 512, 4, 3)
    assert config(1, 4096, 1792) == (16, 64, 256, 4, 4)
    # Qwen3-30B-A3B shards (same parallelism) at the rule's edges: K = 2,048
    # takes K tiles of 512, N = 2,048 keeps 16 weight rows per program.
    assert config(1, 768, 2048) == (16, 16, 512, 4, 4)
    assert config(1, 2048, 512) == (16, 16, 256, 4, 4)
    assert config(40, 4096, 384) == (64, 64, 128, 8, 3)
    monkeypatch.setitem(
        triton_fp8_channel._SMALL_M_TILES, (768, 4096, 16), (16, 256, 4, 3)
    )
    assert config(1, 768, 4096) == (16, 16, 256, 4, 3)
    assert config(32, 768, 4096) == (32, 16, 512, 4, 3)
    # The kernel loads whole K tiles, so an entry whose BLOCK_K does not
    # divide K is refused instead of reading past the rows.
    monkeypatch.setitem(
        triton_fp8_channel._SMALL_M_TILES, (4096, 1792, 16), (64, 512, 4, 4)
    )
    with pytest.raises(ValueError, match="does not divide K 1792"):
        config(1, 4096, 1792)


@pytest.mark.parametrize(
    "a,a_scales,b,kwargs,match",
    [
        ("fp8", None, "kn", {}, "FP8 A needs A_scales"),
        ("bf16", None, "kn", {}, "must be FP8 E4M3"),
        ("fp8", "m1", "nk", {}, "column-major"),
        ("fp8", "m1", "kn", {"alpha": torch.ones(1)}, "no alpha"),
        ("fp8_65", "m1", "kn", {}, "up to 64 rows"),
    ],
)
def test_wrapper_refusals(a, a_scales, b, kwargs, match):
    m = 65 if a == "fp8_65" else 4
    x = torch.zeros(m, 256, dtype=torch.bfloat16 if a == "bf16" else FP8)
    weight = torch.zeros(128, 256, dtype=FP8)
    b_arg = weight.t() if b == "kn" else weight.t().contiguous()
    scales = torch.ones(m, 1) if a_scales else None
    with pytest.raises(ValueError, match=match):
        triton_fp8_channel.triton_mm_fp8_channel_small_m(
            x, b_arg, scales, torch.ones(128, 1), torch.bfloat16, **kwargs
        )


def _quantizer_selections(monkeypatch):
    """quantize_fp8's kernel selections as (name, traits); the kernels do not run."""
    selections = []
    select = quantization.select_kernel

    def selected(*args, **kwargs):
        kernel = select(*args, **kwargs)
        selections.append((kernel.name, kwargs["traits"]))
        return SelectedKernel(kernel.name, lambda x, **_: (x, None))

    monkeypatch.setattr(quantization, "select_kernel", selected)
    return selections


@pytest.mark.parametrize("major,minor", [(9, 0), (10, 0), (10, 3)])
@pytest.mark.parametrize("shape", [(1, 4096), (64, 4096), (65, 4096), (1, 4, 4096)])
def test_token_quantization_takes_the_triton_quantizer_up_to_64_rows(
    arch, monkeypatch, major, minor, shape
):
    """quantize_fp8 states the rows of a 2-D input as ``m``. The Triton
    per-token quantizer declares ``m_max`` 64 on SM100 and newer, so it takes
    1 to 64 rows there, and neither 65 rows nor a 3-D input (no ``m``)."""
    arch(major, minor)
    selections = _quantizer_selections(monkeypatch)
    quantize_fp8(torch.empty(shape, dtype=torch.bfloat16), granularity="token")
    ((name, traits),) = selections
    assert traits.get("m") == (shape[0] if len(shape) == 2 else None)
    expected = major >= 10 and len(shape) == 2 and shape[0] <= 64
    assert (name == PER_TOKEN) is expected


@pytest.mark.parametrize(
    "granularity,group_size,scale_encoding,solution",
    [
        ("token_group", 128, "float32", None),
        # Only TRT-LLM's quantizer serves 128-wide groups with UE8M0 scales.
        ("token_group", 128, "ue8m0", "trtllm"),
        ("token_group", 32, "ue8m0", None),
        ("block", None, "float32", None),
    ],
)
def test_other_granularities_ignore_the_row_count(
    arch, require, monkeypatch, granularity, group_size, scale_encoding, solution
):
    """No other quantization.fp8_with_scale kernel bounds rows: a 2-D input
    (with ``m``) selects what the same row as a 3-D input (no ``m``) does."""
    if solution is not None:
        require("quantization", "fp8_with_scale", solution, torch.bfloat16, "x")
    selections = _quantizer_selections(monkeypatch)
    for shape in [(1, 4096), (1, 1, 4096)]:
        quantize_fp8(
            torch.empty(shape, dtype=torch.bfloat16),
            granularity=granularity,
            group_size=group_size,
            block_size=(128, 128) if granularity == "block" else None,
            scale_encoding=scale_encoding,
        )
    (with_m, traits), (without_m, _) = selections
    assert traits["m"] == 1 and with_m == without_m


@pytest.mark.parametrize("enable_pdl", [False, True])
def test_the_triton_quantizer_launches_as_asked(arch, monkeypatch, enable_pdl):
    """quantize_fp8's ``enable_pdl`` reaches the launch; one program per row."""
    recorder = _LaunchRecorder()
    monkeypatch.setattr(quant_triton, "_fp8_token_quantize_kernel", recorder)
    x = torch.empty(8, 4096, dtype=torch.bfloat16)
    codes, scales = quantize_fp8(x, granularity="token", enable_pdl=enable_pdl)
    (launch,) = recorder.launches
    assert launch["grid"] == (8,)
    assert launch["kwargs"]["ENABLE_PDL"] is enable_pdl
    assert launch["kwargs"].get("launch_pdl", False) is enable_pdl
    assert codes.dtype == FP8 and tuple(scales.shape) == (8, 1)


def _fused_fp8_calls(enable_pdl):
    """One rmsnorm_quantize_fp8 call with every option, and one
    silu_and_mul_quantize_fp8 call, on 8 rows."""
    x = torch.empty(8, 4096, dtype=torch.bfloat16)
    weight = torch.empty(4096, dtype=torch.bfloat16)

    def norm():
        return rmsnorm_quantize_fp8(
            x,
            weight,
            1e-6,
            residual=torch.empty_like(x),
            round_residual_sum_bf16=True,
            x_scale=0.5,
            residual_scale=2.0,
            enable_pdl=enable_pdl,
        )

    def swiglu():
        return silu_and_mul_quantize_fp8(
            x, torch.ones(8, 1), torch.ones(4096, 1), enable_pdl=enable_pdl
        )

    return norm, swiglu


@pytest.mark.parametrize("enable_pdl", [False, True])
def test_fused_fp8_quantizers_launch_as_asked(arch, monkeypatch, enable_pdl):
    """rmsnorm_quantize_fp8 and silu_and_mul_quantize_fp8 launch their Triton
    kernels once each, one program per row, with the caller's ``enable_pdl``
    and every option the call uses."""
    launches = {"norm": _LaunchRecorder(), "swiglu": _LaunchRecorder()}
    monkeypatch.setattr(triton_layernorm, "_rmsnorm_kernel", launches["norm"])
    monkeypatch.setattr(
        quant_triton, "_silu_and_mul_quantize_fp8_kernel", launches["swiglu"]
    )
    norm, swiglu = _fused_fp8_calls(enable_pdl)
    out, residual_sum, codes, scales = norm()
    act_codes, act_scales = swiglu()
    (norm_launch,) = launches["norm"].launches
    (swiglu_launch,) = launches["swiglu"].launches
    for launch in (norm_launch, swiglu_launch):
        assert launch["grid"] == (8,)
        assert launch["kwargs"]["ENABLE_PDL"] is enable_pdl
        assert launch["kwargs"].get("launch_pdl", False) is enable_pdl
    norm_kwargs = norm_launch["kwargs"]
    assert norm_kwargs["TOKEN_FP8_OUT"] and norm_kwargs["fp8_out_ptr"] is codes
    assert norm_kwargs["fp8_scale_ptr"] is scales
    assert norm_kwargs["HAS_RESIDUAL"] and norm_kwargs["ROUND_RESIDUAL_SUM_BF16"]
    assert norm_kwargs["SCALE_INPUTS_BF16"]
    assert out.shape == residual_sum.shape == (8, 4096)
    assert codes.dtype == FP8 and tuple(codes.shape) == (8, 4096)
    assert act_codes.dtype == FP8 and tuple(act_codes.shape) == (8, 2048)
    assert tuple(scales.shape) == tuple(act_scales.shape) == (8, 1)


@pytest.mark.parametrize("intermediate_size", [16384, 16385])
def test_silu_and_mul_fp8_refuses_a_half_wider_than_16384(
    arch, monkeypatch, intermediate_size
):
    """One program holds a SwiGLU half of at most 16,384 columns. Wider, the
    call raises ValueError before any launch."""
    recorder = _LaunchRecorder()
    monkeypatch.setattr(quant_triton, "_silu_and_mul_quantize_fp8_kernel", recorder)
    x = torch.empty(2, 2 * intermediate_size, dtype=torch.bfloat16)
    args = (x, torch.ones(2, 1), torch.ones(2 * intermediate_size, 1))
    if intermediate_size > 16384:
        with pytest.raises(ValueError, match="I=16385 exceeds"):
            silu_and_mul_quantize_fp8(*args, enable_pdl=False)
        assert not recorder.launches
    else:
        silu_and_mul_quantize_fp8(*args, enable_pdl=False)
        assert len(recorder.launches) == 1
