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

"""compressed-tensors FP8 W8A8 (llm-compressor FP8 / FP8_DYNAMIC) on CPU.

The linear scheme's parameters and loading across tensor parallelism, the FP8
GEMM call it makes where one is registered (SM100, SM103) and its BF16
dequantization at load where none is (SM90, SM120). The configs are the
installed compressed-tensors package's own serialization. The FP8 GEMM call
runs the real quantization dispatch and the real ``tokenspeed_kernel.ops.gemm.mm``
dispatcher; only the TRT-LLM activation quantizer, the static quantizer's
Triton launch and the selected kernel's body are CPU stand-ins that keep their
contracts.
"""

import dataclasses
import logging
import os
import sys
from types import SimpleNamespace

import numpy
import pytest
import tokenspeed_kernel.ops.gemm as gemm
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    preset_name_to_scheme,
)
from tokenspeed_kernel.ops.quantization import triton as quant_triton
from tokenspeed_kernel.platform import (
    ArchVersion,
    Platform,
    _get_cuda_sm_features,
    current_platform,
)
from tokenspeed_kernel.selection import SelectedKernel

from tokenspeed.runtime.layers import dense
from tokenspeed.runtime.layers.dense import UnquantizedLinearMethod
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.quantization import QUANTIZATION_METHODS
from tokenspeed.runtime.layers.quantization.compressed_tensors import (
    compressed_tensors as ct,
)
from tokenspeed.runtime.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Fp8,
)
from tokenspeed.runtime.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a8_fp8 as w8a8,
)
from tokenspeed.runtime.model_loader.weight_utils import get_quant_config

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(
    est_time=15,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="the FP8 kernels it selects are NVIDIA's",
)

FP8 = torch.float8_e4m3fn
FP8_ARGS = dict(num_bits=8, type="float", symmetric=True)


def _config(weights, inputs, *, fmt="float-quantized", ignore=("lm_head",)):
    """config.json quantization_config as llm-compressor saves it."""
    scheme = QuantizationScheme(
        targets=["Linear"], weights=weights, input_activations=inputs, format=fmt
    )
    raw = QuantizationConfig(
        config_groups={"group_0": scheme},
        format=fmt,
        quantization_status="compressed",
        ignore=list(ignore),
    ).model_dump(mode="json", exclude=["quant_method"])
    return {"quant_method": "compressed-tensors", "version": "0.18.0", **raw}


FP8_DYNAMIC = {
    name: getattr(preset_name_to_scheme("FP8_DYNAMIC", ["Linear"]), name)
    for name in ("weights", "input_activations")
}
CHANNEL_DYNAMIC = _config(FP8_DYNAMIC["weights"], FP8_DYNAMIC["input_activations"])
TENSOR_DYNAMIC = _config(
    QuantizationArgs(strategy="tensor", dynamic=False, **FP8_ARGS),
    FP8_DYNAMIC["input_activations"],
)
TENSOR_STATIC = _config(
    *(
        getattr(preset_name_to_scheme("FP8", ["Linear"]), name)
        for name in ("weights", "input_activations")
    )
)
CHANNEL_STATIC = _config(
    FP8_DYNAMIC["weights"],
    preset_name_to_scheme("FP8", ["Linear"]).input_activations,
)


def _parse(raw):
    return ct.CompressedTensorsConfig.from_config(raw)


@pytest.fixture
def platform():
    """Switch the synthetic platform's arch; restored afterwards."""
    original = current_platform()

    def switch(major, minor):
        arch = ArchVersion(major, minor)
        Platform.override(
            dataclasses.replace(
                original, arch_version=arch, sm_features=_get_cuda_sm_features(arch)
            )
        )

    switch(10, 0)
    w8a8._warn_once.cache_clear()
    yield switch
    Platform.override(original)
    w8a8._warn_once.cache_clear()


def bf16_exact(x64):
    """Exact float64 values rounded once to BF16, to nearest even, on the
    bit pattern: of the 52 mantissa bits keep 7."""
    bits = x64.view(torch.int64)
    bits = (bits + (1 << 44) - 1 + ((bits >> 45) & 1)) & ~((1 << 45) - 1)
    return bits.view(torch.float64).to(torch.bfloat16)


def _fp8_checkpoint(rows, cols, *, scale_dtype=torch.bfloat16, per_tensor=False):
    """(FP8 codes, scales) of a random matrix; for E4M3 scales the magnitudes
    keep every scale a normal E4M3 value."""
    generator = torch.Generator().manual_seed(rows * 1000 + cols)
    weight = torch.randn(rows, cols, generator=generator) * 2.0 ** torch.randint(
        -4, 3, (rows, 1), generator=generator
    )
    if scale_dtype == FP8:
        weight = weight * 64.0
    amax = weight.abs().amax() if per_tensor else weight.abs().amax(-1, keepdim=True)
    scale = (amax / 448.0).reshape(
        1 if per_tensor else rows, *(() if per_tensor else (1,))
    )
    scale = scale.to(scale_dtype)
    codes = (weight / scale.float()).clamp(-448, 448).to(FP8)
    return codes, scale


# ---------------------------------------------------------------- schemes
@pytest.mark.parametrize(
    "raw,strategy,static",
    [
        (CHANNEL_DYNAMIC, "channel", False),
        (TENSOR_DYNAMIC, "tensor", False),
        (TENSOR_STATIC, "tensor", True),
        (
            _config(
                **{
                    "weights": FP8_DYNAMIC["weights"],
                    "inputs": FP8_DYNAMIC["input_activations"],
                    "fmt": "naive-quantized",
                }
            ),
            "channel",
            False,
        ),
    ],
    ids=["fp8-dynamic", "tensor-weights", "fp8-static", "naive-quantized"],
)
def test_fp8_w8a8_scheme_is_selected(platform, raw, strategy, static):
    config = _parse(raw)
    layer = ColumnParallelLinear(
        16,
        8,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=config,
        prefix="model.layers.0.mlp.up_proj",
    )
    assert type(layer.quant_method) is ct.CompressedTensorsLinearMethod
    assert type(layer.scheme) is CompressedTensorsW8A8Fp8
    assert (layer.scheme.strategy, layer.scheme.is_static_input_scheme) == (
        strategy,
        static,
    )
    assert QUANTIZATION_METHODS["compressed-tensors"] is ct.CompressedTensorsConfig
    model_config = SimpleNamespace(
        quantization="compressed-tensors",
        hf_config=SimpleNamespace(quantization_config=raw),
    )
    assert type(get_quant_config(model_config, None)) is ct.CompressedTensorsConfig


def test_fp8_w8a8_needs_hopper(platform):
    platform(8, 9)
    with pytest.raises(RuntimeError, match="Min capability: 90"):
        ColumnParallelLinear(
            16,
            8,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=_parse(CHANNEL_DYNAMIC),
            prefix="x.up_proj",
        )


@pytest.mark.parametrize(
    "raw,match",
    [
        (
            _config(FP8_DYNAMIC["weights"], None, fmt="naive-quantized"),
            r"FP8 weight-only \(W8A16\) linears are not supported",
        ),
        (
            _config(
                *(
                    getattr(preset_name_to_scheme("FP8_BLOCK", ["Linear"]), name)
                    for name in ("weights", "input_activations")
                )
            ),
            "No compressed-tensors compatible scheme was found for weights",
        ),
    ],
    ids=["w8a16", "fp8-block"],
)
def test_other_fp8_schemes_are_refused_with_the_reason(platform, raw, match):
    with pytest.raises(NotImplementedError, match=match):
        ColumnParallelLinear(
            16,
            8,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=_parse(raw),
            prefix="x.up_proj",
        )


def test_ignored_and_unmatched_layers(platform):
    config = _parse(
        _config(
            FP8_DYNAMIC["weights"],
            FP8_DYNAMIC["input_activations"],
            ignore=[
                "lm_head",
                "re:.*mlp.gate$",
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.k_proj",
                "model.layers.0.self_attn.v_proj",
            ],
        )
    )
    for prefix in ("lm_head", "model.layers.3.mlp.gate"):
        layer = ReplicatedLinear(
            16,
            8,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=config,
            prefix=prefix,
        )
        assert type(layer.quant_method) is UnquantizedLinearMethod
        assert layer.weight.dtype == torch.bfloat16
    qkv = QKVParallelLinear(
        16,
        4,
        2,
        1,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=config,
        prefix="model.layers.0.self_attn.qkv_proj",
        tp_rank=0,
        tp_size=1,
    )
    assert type(qkv.quant_method) is UnquantizedLinearMethod
    other = QKVParallelLinear(
        16,
        4,
        2,
        1,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=config,
        prefix="model.layers.1.self_attn.qkv_proj",
        tp_rank=0,
        tp_size=1,
    )
    assert type(other.quant_method) is ct.CompressedTensorsLinearMethod
    regex_only = _parse(
        dict(
            CHANNEL_DYNAMIC,
            config_groups={
                "group_0": {
                    **CHANNEL_DYNAMIC["config_groups"]["group_0"],
                    "targets": ["re:.*experts.*"],
                }
            },
        )
    )
    with pytest.raises(ValueError, match="Unable to find matching target"):
        ColumnParallelLinear(
            16,
            8,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=regex_only,
            prefix="model.layers.0.mlp.up_proj",
        )


def test_linear_target_names_every_linear_layer(platform):
    """compressed-tensors matches the ``Linear`` target by a module's classes:
    every LinearBase, whatever its own class name. An ``ignore`` entry naming
    one of a layer's classes leaves it unquantized."""

    class Router(ReplicatedLinear):
        pass

    def router(config):
        return Router(
            16,
            8,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=config,
            prefix="model.layers.3.mlp.router",
        )

    layer = router(_parse(CHANNEL_DYNAMIC))
    assert type(layer.quant_method) is ct.CompressedTensorsLinearMethod
    assert type(layer.scheme) is CompressedTensorsW8A8Fp8
    for ignored in ("Router", "ReplicatedLinear"):
        raw = _config(
            FP8_DYNAMIC["weights"],
            FP8_DYNAMIC["input_activations"],
            ignore=["lm_head", ignored],
        )
        layer = router(_parse(raw))
        assert type(layer.quant_method) is UnquantizedLinearMethod
        assert layer.weight.dtype == torch.bfloat16


def test_float_weights_of_other_widths_are_not_fp8(platform):
    four_bits = dict(num_bits=4, type="float", symmetric=True)
    raw = _config(
        QuantizationArgs(strategy="channel", dynamic=False, **four_bits),
        QuantizationArgs(strategy="token", dynamic=True, **four_bits),
    )
    with pytest.raises(NotImplementedError, match="No compressed-tensors compatible"):
        ColumnParallelLinear(
            16,
            8,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=_parse(raw),
            prefix="x.up_proj",
        )


# ---------------------------------------------------------------- parameters and loading
@pytest.mark.parametrize("scale_dtype", [torch.bfloat16, torch.float32, FP8])
@pytest.mark.parametrize("tp_rank", [0, 1])
def test_channel_scales_follow_their_rows(platform, scale_dtype, tp_rank):
    """Column parallel: weight rows and their scales split together; row
    parallel: the input columns split, every rank keeps all output scales.
    BF16, FP32 and E4M3 scales become FP32 exactly."""
    config = _parse(CHANNEL_DYNAMIC)
    group = object()
    column = ColumnParallelLinear(
        6,
        8,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=config,
        prefix="m.up_proj",
        tp_rank=tp_rank,
        tp_size=2,
        tp_group=group,
    )
    row = RowParallelLinear(
        8,
        6,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=config,
        prefix="m.down_proj",
        tp_rank=tp_rank,
        tp_size=2,
        tp_group=group,
    )
    for layer, (rows, cols) in ((column, (8, 6)), (row, (6, 8))):
        assert layer.weight.dtype == FP8 and layer.weight_scale.dtype == torch.float32
        assert torch.isnan(layer.weight_scale).all()
        assert layer.input_scale is None
        codes, scale = _fp8_checkpoint(rows, cols, scale_dtype=scale_dtype)
        layer.weight_loader_v2(layer.weight, codes)
        layer.weight_loader_v2(layer.weight_scale, scale)
        if layer is column:
            part = slice(4 * tp_rank, 4 * tp_rank + 4)
            assert torch.equal(
                layer.weight.view(torch.uint8), codes[part].view(torch.uint8)
            )
            assert torch.equal(layer.weight_scale, scale[part].float())
        else:
            part = slice(4 * tp_rank, 4 * tp_rank + 4)
            assert torch.equal(
                layer.weight.view(torch.uint8), codes[:, part].view(torch.uint8)
            )
            assert torch.equal(layer.weight_scale, scale.float())
    assert tuple(column.weight_scale.shape) == (4, 1)


def test_fused_linears_load_each_shard(platform):
    """QKV and gate/up: each shard's rows and per-channel scales land at the
    shard's offset; per-tensor scales keep one value per shard."""
    channel, tensor = _parse(CHANNEL_DYNAMIC), _parse(TENSOR_DYNAMIC)
    qkv = QKVParallelLinear(
        16,
        4,
        2,
        1,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=channel,
        prefix="m.self_attn.qkv_proj",
        tp_rank=0,
        tp_size=1,
    )
    sizes = {"q": 8, "k": 4, "v": 4}
    start = 0
    for shard, size in sizes.items():
        codes, scale = _fp8_checkpoint(size, 16)
        qkv.weight_loader_v2(qkv.weight, codes, shard)
        qkv.weight_loader_v2(qkv.weight_scale, scale, shard)
        assert torch.equal(
            qkv.weight[start : start + size].view(torch.uint8), codes.view(torch.uint8)
        )
        assert torch.equal(qkv.weight_scale[start : start + size], scale.float())
        start += size
    merged = MergedColumnParallelLinear(
        16,
        [6, 6],
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=tensor,
        prefix="m.mlp.gate_up_proj",
        tp_rank=0,
        tp_size=1,
    )
    assert tuple(merged.weight_scale.shape) == (2,)
    values = []
    for index in range(2):
        codes, scale = _fp8_checkpoint(6, 16, per_tensor=True)
        merged.weight_loader_v2(merged.weight, codes, index)
        merged.weight_loader_v2(merged.weight_scale, scale, index)
        values.append(scale.float().item())
    assert merged.weight_scale.tolist() == values


# ---------------------------------------------------------------- run: FP8 GEMM or BF16
@pytest.mark.parametrize(
    "vendor,arch", [("amd", (12, 5)), ("nvidia", (12, 0)), ("nvidia", (9, 0))]
)
def test_unqualified_platforms_keep_dense_fp8_fallback(vendor, arch):
    original = current_platform()
    Platform.override(
        dataclasses.replace(original, vendor=vendor, arch_version=ArchVersion(*arch))
    )
    try:
        assert not w8a8.fp8_channel_gemm_available()
        layer, _, _ = _loaded(_parse(CHANNEL_DYNAMIC))
        layer.quant_method.process_weights_after_loading(layer)
        assert layer.scheme.dequantized
        assert layer.weight.dtype == torch.bfloat16
        assert layer.weight_scale is None
    finally:
        Platform.override(original)


@pytest.mark.parametrize("raw", [CHANNEL_DYNAMIC, TENSOR_DYNAMIC])
def test_gfx950_loading_retains_fp8_weights_and_fp32_channel_scales(raw):
    original = current_platform()
    Platform.override(
        dataclasses.replace(
            original,
            vendor="amd",
            arch_version=ArchVersion(9, 5),
            runtime_features=frozenset(),
        )
    )
    try:
        assert w8a8.fp8_channel_gemm_available()
        layer, codes, scale = _loaded(_parse(raw))
        layer.quant_method.process_weights_after_loading(layer)
        assert not layer.scheme.dequantized
        assert layer.weight.dtype == FP8
        assert layer.weight_scale.dtype == torch.float32
        torch.testing.assert_close(
            layer.weight.t().float(), codes.float(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            layer.weight_scale, scale.float().expand(8, 1), rtol=0, atol=0
        )
    finally:
        Platform.override(original)


def _loaded(config, rows=8, cols=16, scale_dtype=torch.bfloat16):
    layer = ColumnParallelLinear(
        cols,
        rows,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=config,
        prefix="m.up_proj",
    )
    per_tensor = config.target_scheme_map["Linear"]["weights"].strategy == "tensor"
    codes, scale = _fp8_checkpoint(
        rows, cols, scale_dtype=scale_dtype, per_tensor=per_tensor
    )
    layer.weight_loader_v2(layer.weight, codes)
    layer.weight_loader_v2(layer.weight_scale, scale)
    if layer.input_scale is not None:
        layer.weight_loader_v2(layer.input_scale, torch.tensor([0.25]))
    return layer, codes, scale


def _stand_ins(monkeypatch, calls, op_scale_shape="M1"):
    """The TRT-LLM activation quantizer (scales [M, 1] or [M]) and the Triton
    one mm runs up to 64 rows (scales [M, 1]), the static quantizer's Triton
    launch (it returns the codes of x over the scale) and the body of the
    kernel the real mm dispatcher selects (scales of one dimension or none
    become columns, as triton_scaled_mm takes them)."""

    def quantize_e4m3_activation(x):
        scale = x.float().abs().amax(-1, keepdim=True) / 448.0
        codes = (x.float() / scale).clamp(-448, 448).to(FP8)
        return codes, scale if op_scale_shape == "M1" else scale.squeeze(-1)

    def fp8_quantize_per_token(x, enable_pdl):
        scale = x.float().abs().amax(-1, keepdim=True) / 448.0
        return (x.float() / scale).clamp(-448, 448).to(FP8), scale

    def static_quant(x, scale=None, enable_pdl=None, **_):
        assert x.is_contiguous() and scale.numel() == 1
        return (x.float() / scale.float()).clamp(-448, 448).to(FP8)

    select = gemm.select_kernel

    def selected(*args, **kwargs):
        kernel = select(*args, **kwargs)

        def run(A, B, A_scales, B_scales, out_dtype, **extra):
            calls.append(
                dict(
                    kernel=kernel.name,
                    A=A,
                    B=B,
                    A_scales=A_scales,
                    B_scales=B_scales,
                    out_dtype=out_dtype,
                )
            )
            a = A_scales.reshape(-1, 1) if A_scales.dim() <= 1 else A_scales
            b = B_scales.reshape(-1, 1) if B_scales.dim() <= 1 else B_scales
            product = (A.double() * a.double()) @ (B.double() * b.double().T)
            return product.to(out_dtype)

        return SelectedKernel(kernel.name, run)

    monkeypatch.setattr(
        torch.ops.tensorrt_llm,
        "quantize_e4m3_activation",
        quantize_e4m3_activation,
        raising=False,
    )
    monkeypatch.setattr(quant_triton, "fp8_quantize_per_token", fp8_quantize_per_token)
    monkeypatch.setattr(quant_triton, "fp8_quantize", static_quant)
    monkeypatch.setattr(gemm, "select_kernel", selected)


@pytest.mark.parametrize("tokens", [1, 3])
@pytest.mark.parametrize("op_scale_shape", ["M1", "M"])
@pytest.mark.parametrize("arch", [(10, 0), (10, 3)])
@pytest.mark.parametrize(
    "raw",
    [CHANNEL_DYNAMIC, TENSOR_DYNAMIC, TENSOR_STATIC],
    ids=["channel", "tensor", "static"],
)
def test_blackwell_runs_the_fp8_gemm(
    platform, monkeypatch, arch, raw, op_scale_shape, tokens
):
    platform(*arch)
    assert w8a8.fp8_channel_gemm_available()
    layer, codes, scale = _loaded(_parse(raw))
    layer.quant_method.process_weights_after_loading(layer)
    assert not layer.scheme.dequantized
    assert layer.weight.dtype == FP8 and tuple(layer.weight.shape) == (16, 8)
    assert torch.equal(layer.weight.t().view(torch.uint8), codes.view(torch.uint8))
    assert layer.weight_scale.dtype == torch.float32
    assert torch.equal(layer.weight_scale, scale.float().expand(8, 1))
    calls = []
    _stand_ins(monkeypatch, calls, op_scale_shape)
    x = torch.randn(tokens, 16).to(torch.bfloat16)
    out = layer.quant_method.apply(layer, x)
    (call,) = calls
    assert call["kernel"] == "triton_mm_fp8_scaled"
    assert call["out_dtype"] == torch.bfloat16 and call["A"].dtype == FP8
    if layer.input_scale is not None:
        # The static input scale: one 0-d value for all shards, which mm
        # hands to the kernel as its [1, 1] view.
        assert tuple(layer.input_scale.shape) == ()
        assert call["A_scales"].data_ptr() == layer.input_scale.data_ptr()
        assert tuple(call["A_scales"].shape) == (1, 1)
        assert call["A_scales"].item() == 0.25
        input_scale = torch.tensor([[0.25]], dtype=torch.float64)
    else:
        input_scale = (x.float().abs().amax(-1, keepdim=True) / 448.0).double()
        assert call["A_scales"].dtype == torch.float32
        assert tuple(call["A_scales"].shape) == (tokens, 1)
        assert torch.equal(call["A_scales"].double(), input_scale)
    assert call["B"] is layer.weight and call["B_scales"] is layer.weight_scale
    expected_codes = (x.float() / input_scale.float()).clamp(-448, 448).to(FP8)
    assert torch.equal(call["A"].view(torch.uint8), expected_codes.view(torch.uint8))
    weight = (
        codes.double() * scale.double().reshape(-1, 1)
        if scale.numel() > 1
        else codes.double() * scale.double()
    )
    reference = (expected_codes.double() * input_scale) @ weight.T
    assert torch.equal(out, reference.to(torch.bfloat16))


@pytest.mark.parametrize("tokens", [1, 8, 9, 64, 65])
@pytest.mark.parametrize("arch", [(10, 0), (10, 3)])
def test_decode_rows_reach_the_small_m_gemm_quantized_by_triton(
    platform, monkeypatch, arch, tokens
):
    """With K in whole 128-element tiles, up to 64 rows of a dynamic FP8
    linear go to the small-M FP8 GEMM as FP8 codes and scales from the Triton
    per-token quantizer; 65 rows go to the next kernel quantized by TRT-LLM's.
    Both equal the FP64 product of per-token codes."""
    platform(*arch)
    layer, codes, scale = _loaded(_parse(CHANNEL_DYNAMIC), rows=32, cols=256)
    layer.quant_method.process_weights_after_loading(layer)
    calls, quantized = [], []

    def quantizer(name):
        def quantize(x, enable_pdl=None):
            quantized.append((name, x))
            x_scale = x.float().abs().amax(-1, keepdim=True) / 448.0
            return (x.float() / x_scale).clamp(-448, 448).to(FP8), x_scale

        return quantize

    select = gemm.select_kernel

    def selected(*args, **kwargs):
        kernel = select(*args, **kwargs)

        def run(A, B, A_scales, B_scales, out_dtype, **extra):
            calls.append(dict(kernel=kernel.name, A=A, A_scales=A_scales))
            product = (A.double() * A_scales.double()) @ (
                B.double() * B_scales.double().T
            )
            return product.to(out_dtype)

        return SelectedKernel(kernel.name, run)

    monkeypatch.setattr(
        torch.ops.tensorrt_llm,
        "quantize_e4m3_activation",
        quantizer("trtllm"),
        raising=False,
    )
    monkeypatch.setattr(quant_triton, "fp8_quantize_per_token", quantizer("triton"))
    monkeypatch.setattr(gemm, "select_kernel", selected)
    x = torch.randn(tokens, 256).to(torch.bfloat16)
    out = layer.quant_method.apply(layer, x)
    (call,) = calls
    ((source, rows),) = quantized
    assert call["A"].dtype == FP8 and rows.data_ptr() == x.data_ptr()
    if tokens <= 64:
        assert call["kernel"] == "triton_mm_fp8_channel_small_m"
        assert source == "triton"
    else:
        assert call["kernel"] != "triton_mm_fp8_channel_small_m"
        assert source == "trtllm"
    x_scale = (x.float().abs().amax(-1, keepdim=True) / 448.0).double()
    x_codes = (x.float() / x_scale.float()).clamp(-448, 448).to(FP8)
    weight = codes.double() * scale.double().reshape(-1, 1)
    reference = (x_codes.double() * x_scale) @ weight.T
    assert torch.equal(out, reference.to(torch.bfloat16))


@pytest.mark.parametrize(
    "arch,scale_dtype",
    [
        ((9, 0), torch.bfloat16),
        ((9, 0), torch.float32),
        ((9, 0), FP8),
        ((12, 0), torch.bfloat16),
    ],
)
def test_hopper_dequantizes_at_load_and_says_so(
    platform, monkeypatch, caplog, arch, scale_dtype
):
    """SM90 has no FP8 GEMM with channel scales; SM120 has one, but not the
    TRT-LLM per-token quantizer of its input."""
    platform(*arch)
    assert not w8a8.fp8_channel_gemm_available()
    caplog.set_level(logging.WARNING, logger=w8a8.__name__)
    layer, codes, scale = _loaded(_parse(CHANNEL_DYNAMIC), scale_dtype=scale_dtype)
    layer.quant_method.process_weights_after_loading(layer)
    other, _, _ = _loaded(_parse(CHANNEL_DYNAMIC))
    other.quant_method.process_weights_after_loading(other)
    assert layer.scheme.dequantized
    assert layer.weight.dtype == torch.bfloat16 and tuple(layer.weight.shape) == (8, 16)
    assert torch.equal(layer.weight, bf16_exact(codes.double() * scale.double()))
    assert layer.weight_scale is None and layer.input_scale is None
    warnings = [r.getMessage() for r in caplog.records]
    assert warnings == [
        "compressed-tensors FP8 W8A8 linears are dequantized to bfloat16 at load: "
        "no FP8 GEMM with per-channel weight scales and per-token FP8 input "
        f"quantization is registered for nvidia {arch[0]}.{arch[1]}"
    ]
    seen = []
    monkeypatch.setattr(
        dense.UnquantizedLinearMethod,
        "apply",
        lambda self, layer, x, bias=None: seen.append((layer, x, bias)) or "bf16",
    )
    x = torch.randn(2, 16).to(torch.bfloat16)
    assert layer.quant_method.apply(layer, x) == "bf16"
    assert seen == [(layer, x, None)]


@pytest.mark.parametrize("arch", [(10, 0), (10, 3)])
def test_the_unscaled_linear_runs_a_vendor_gemm_with_unit_scales(
    platform, monkeypatch, arch
):
    """fp8_unscaled_linear: the per-tensor FP8 GEMM the registry selects, with
    both scales the one unit tensor, writing BF16; a vendor kernel serves it
    on NVIDIA arch 10."""
    platform(*arch)
    assert w8a8.fp8_unscaled_gemm_available()
    calls = []
    select = gemm.select_kernel

    def selected(*args, **kwargs):
        kernel = select(*args, **kwargs)

        def run(A, B, A_scales, B_scales, out_dtype, **extra):
            calls.append(dict(kernel=kernel.name, A_scales=A_scales, B_scales=B_scales))
            return (A.double() @ B.double()).to(out_dtype)

        return SelectedKernel(kernel.name, run)

    monkeypatch.setattr(gemm, "select_kernel", selected)
    codes, _ = _fp8_checkpoint(3, 16)
    weight, _ = _fp8_checkpoint(8, 16)
    unit = torch.ones(1)
    out = w8a8.fp8_unscaled_linear(codes, weight.t(), unit)
    (call,) = calls
    assert not call["kernel"].startswith(("triton", "torch_mm_fp8_scaled_"))
    assert call["A_scales"].item() == 1 and call["B_scales"].item() == 1
    assert out.dtype == torch.bfloat16
    exact = codes.double() @ weight.double().t()
    assert torch.equal(out, exact.to(torch.bfloat16))


def test_the_unscaled_linear_rejects_a_kn_contiguous_weight():
    """Selection keys the weight layout on stride(-1) == 1, which no vendor
    per-tensor FP8 GEMM accepts: a KN-contiguous weight would quietly run the
    Triton GEMM with unit scales instead of raising."""
    codes, _ = _fp8_checkpoint(3, 16)
    weight, _ = _fp8_checkpoint(8, 16)
    with pytest.raises(ValueError, match="a transposed weight has"):
        w8a8.fp8_unscaled_linear(codes, weight.t().contiguous(), torch.ones(1))


def test_dequantization_rounds_the_exact_product_once():
    """1.125 times 15087843 * 2**-24 lies just below the BF16 midpoint
    1 + 3 * 2**-8; rounded once it is 1 + 2**-7 (through FP32, 1 + 2**-6)."""
    code = torch.tensor([[1.125]]).to(FP8)
    scale = torch.tensor([[15087843 * 2.0**-24]])
    value = w8a8.dequantize_fp8_per_channel(code, scale)
    assert value.dtype == torch.bfloat16 and value.item() == 1 + 2**-7
    assert (code.float() * scale).bfloat16().item() == 1 + 2**-6


def test_fp16_dequantization_rounds_once_too():
    """FP16 model dtype: every FP8 code times FP32 scales spanning FP16's
    normal and subnormal range equals NumPy's direct FP64 -> FP16 rounding.
    Torch's FP64 -> FP16 cast goes through FP32: 1.375 times 9833379 * 2**-23
    (exactly 1.61181642...) becomes 1.611328125 there, 1.6123046875 here."""
    codes = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(FP8)
    codes = codes[torch.isfinite(codes.float())].reshape(-1, 1)
    generator = torch.Generator().manual_seed(5)
    scales = torch.rand(1, 2048, generator=generator, dtype=torch.float64)
    scales = scales * 2.0 ** torch.randint(-34, 2, (1, 2048), generator=generator)
    exact = codes.double() * scales.float().double()
    value = w8a8._round_once(exact, torch.float16)
    reference = torch.from_numpy(exact.numpy().astype(numpy.float16))
    assert value.dtype == torch.float16
    assert torch.equal(value.view(torch.int16), reference.view(torch.int16))
    code = torch.tensor([[1.375]]).to(FP8)
    scale = torch.tensor([[9833379 * 2.0**-23]])
    value = w8a8.dequantize_fp8_per_channel(code, scale, torch.float16)
    assert value.item() == 1.6123046875
    assert (code.double() * scale.double()).half().item() == 1.611328125


@pytest.mark.parametrize("value", [float("nan"), 0.0, -1.0])
def test_scales_must_be_present_finite_and_positive(platform, value):
    layer, _, scale = _loaded(_parse(CHANNEL_DYNAMIC))
    layer.weight_scale.data[3] = value
    with pytest.raises(
        ValueError,
        match="m.up_proj.weight_scale: compressed-tensors FP8 scales must be finite and positive",
    ):
        layer.quant_method.process_weights_after_loading(layer)


def test_kv_cache_scheme_is_kept_and_reported_as_not_applied(caplog):
    """Nothing consumes the checkpoint's KV-cache scheme yet, so parsing it
    says so once; the linears are quantized as usual."""
    caplog.set_level(logging.WARNING, logger=ct.__name__)
    scheme = {"num_bits": 8, "type": "float", "strategy": "tensor", "dynamic": False}
    config = _parse(dict(CHANNEL_DYNAMIC, kv_cache_scheme=scheme))
    assert config.kv_cache_scheme == scheme
    assert [r.getMessage() for r in caplog.records] == [
        f"compressed-tensors kv_cache_scheme {scheme} is not applied: the KV "
        "cache dtype follows --kv-cache-dtype, which keeps the model dtype by "
        "default."
    ]
    caplog.clear()
    assert _parse(CHANNEL_DYNAMIC).kv_cache_scheme is None
    assert not caplog.records


# quantization_config fields of mconcat/GLM-5.1-FP8-Dynamic's config.json: every
# Linear but the routers, norms and lm_head is FP8, kv_b_proj included.
GLM_5_1_FP8_DYNAMIC = {
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
    "format": "float-quantized",
    "config_groups": {
        "group_fp8": {
            "format": "float-quantized",
            "targets": ["Linear"],
            "weights": {
                "num_bits": 8,
                "type": "float",
                "symmetric": True,
                "strategy": "channel",
                "dynamic": False,
            },
            "input_activations": {
                "num_bits": 8,
                "type": "float",
                "symmetric": True,
                "strategy": "token",
                "dynamic": True,
            },
        }
    },
    "ignore": [
        "lm_head",
        "model.embed_tokens",
        "re:.*\\.mlp\\.gate$",
        "re:.*norm.*",
        "re:.*layernorm.*",
        "re:.*\\.indexer\\.k_norm.*",
    ],
}


def _mla_attention(config, q_lora_rank):
    """A tiny DeepSeek-V3-shaped MLA attention layer (q_lora_rank None is the
    Moonlight layout: q_proj and kv_a_proj_with_mqa instead of the fused
    q/kv latent projection)."""
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3AttentionMLA

    return DeepseekV3AttentionMLA(
        config=SimpleNamespace(rms_norm_eps=1e-6),
        mapping=Mapping(rank=0, world_size=1),
        hidden_size=64,
        num_heads=4,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=32,
        quant_config=config,
        layer_id=0,
        prefix="model.layers.0.self_attn",
        reduce_attn_results=False,
    )


@pytest.mark.parametrize(
    "raw,q_lora_rank",
    [
        pytest.param(GLM_5_1_FP8_DYNAMIC, 32, id="glm-5.1-fp8-dynamic-dsv3"),
        pytest.param(GLM_5_1_FP8_DYNAMIC, None, id="glm-5.1-fp8-dynamic-moonlight"),
        pytest.param(CHANNEL_DYNAMIC, 32, id="fp8-dynamic-dsv3"),
        pytest.param(CHANNEL_DYNAMIC, None, id="fp8-dynamic-moonlight"),
        # The DeepSeek V3 layout refuses static input scales earlier, at its
        # fused q/kv latent projection; see
        # test_fused_qkv_a_refuses_per_tensor_and_static_scales.
        pytest.param(TENSOR_STATIC, None, id="fp8-static-moonlight"),
    ],
)
def test_mla_kv_b_proj_in_fp8_is_refused(platform, monkeypatch, raw, q_lora_rank):
    """MLA models absorb kv_b_proj into w_kc/w_vc while loading, before the
    FP8 scales are applied, so an FP8 kv_b_proj is refused when the layer is
    built. A checkpoint that keeps kv_b_proj unquantized still builds, with
    the other projections on the FP8 scheme."""
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "tp_batch_invariant", "none")
    with pytest.raises(
        ValueError,
        match="model.layers.0.self_attn.kv_b_proj: compressed-tensors quantized "
        r"weights on an MLA kv_b_proj are not supported yet "
        r"\(CompressedTensorsW8A8Fp8\)",
    ):
        _mla_attention(_parse(raw), q_lora_rank)
    keep = dict(raw, ignore=[*raw["ignore"], "re:.*kv_b_proj$"])
    attn = _mla_attention(_parse(keep), q_lora_rank)
    assert type(attn.kv_b_proj.quant_method) is UnquantizedLinearMethod
    first = attn.q_b_proj if q_lora_rank else attn.q_proj
    assert type(first.quant_method) is ct.CompressedTensorsLinearMethod
    assert first.weight.dtype == FP8


def test_mla_kv_b_proj_is_refused_for_every_scheme(platform):
    """The absorption reads kv_b_proj.weight before any scheme's post-load
    fix-up, not only FP8's, so a pack-quantized kv_b_proj (weight_packed and
    no weight) is refused by name as well."""
    raw = _config(
        QuantizationArgs(
            num_bits=4,
            type="int",
            symmetric=True,
            strategy="group",
            group_size=128,
            dynamic=False,
        ),
        None,
        fmt="pack-quantized",
    )
    with pytest.raises(
        ValueError,
        match="model.layers.0.self_attn.kv_b_proj: compressed-tensors quantized "
        r"weights on an MLA kv_b_proj are not supported yet "
        r"\(CompressedTensorsWNA16\)",
    ):
        ReplicatedLinear(
            64,
            32,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=_parse(raw),
            prefix="model.layers.0.self_attn.kv_b_proj",
        )


def _fused_qkv_a(config):
    """The fused q/kv latent projection of _mla_attention's DeepSeek V3
    layout: q_lora_rank 32, then kv_lora_rank 32 plus 8 rope rows."""
    from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3FusedQkvAProjWithMqa

    return DeepseekV3FusedQkvAProjWithMqa(
        64,
        32 + 32 + 8,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=config,
        prefix="model.layers.0.self_attn.fused_qkv_a_proj_with_mqa",
    )


@pytest.mark.parametrize(
    "raw,combination",
    [
        (TENSOR_DYNAMIC, "per-tensor weight scales with dynamic activations"),
        (TENSOR_STATIC, "per-tensor weight scales with static input scales"),
        (CHANNEL_STATIC, "per-channel weight scales with static input scales"),
    ],
    ids=["tensor-dynamic", "tensor-static", "channel-static"],
)
def test_fused_qkv_a_refuses_per_tensor_and_static_scales(
    platform, monkeypatch, raw, combination
):
    """DeepSeek V3 loads q_a_proj and kv_a_proj_with_mqa into one fused linear
    at row offsets: per-channel weight scales follow their rows, but a
    per-tensor weight scale or a static input scale has one slot for both
    layers. Those schemes are refused when the layer is built, which the
    attention layer does before kv_b_proj."""
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "tp_batch_invariant", "none")
    match = (
        "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa: compressed-tensors "
        f"FP8 {combination} are not supported on the fused q_a_proj and "
        "kv_a_proj_with_mqa linear.* Per-channel weight scales with dynamic "
        "per-token activations are supported"
    )
    with pytest.raises(ValueError, match=match):
        _fused_qkv_a(_parse(raw))
    with pytest.raises(ValueError, match=match):
        _mla_attention(_parse(raw), 32)


@pytest.mark.parametrize(
    "raw",
    [GLM_5_1_FP8_DYNAMIC, CHANNEL_DYNAMIC],
    ids=["glm-5.1-fp8-dynamic", "fp8-dynamic"],
)
def test_fused_qkv_a_loads_per_channel_dynamic_scales(platform, raw):
    """Per-channel weights with dynamic activations: each layer's rows and
    scales land at the DeepSeek V3 loader's row offset (begin_size)."""
    layer = _fused_qkv_a(_parse(raw))
    assert type(layer.scheme) is CompressedTensorsW8A8Fp8
    q_a, kv_a = _fp8_checkpoint(32, 64), _fp8_checkpoint(40, 64)
    for (codes, scale), begin_size in ((q_a, 0), (kv_a, 32)):
        layer.weight_loader(layer.weight, codes, begin_size=begin_size)
        layer.weight_loader(layer.weight_scale, scale, begin_size=begin_size)
    layer.quant_method.process_weights_after_loading(layer)
    codes = torch.cat([q_a[0], kv_a[0]])
    assert torch.equal(layer.weight.t().view(torch.uint8), codes.view(torch.uint8))
    assert torch.equal(layer.weight_scale, torch.cat([q_a[1], kv_a[1]]).float())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
