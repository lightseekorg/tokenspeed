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
runs the real per-token quantization wrappers and the real
``tokenspeed_kernel.mm`` dispatcher; only the TRT-LLM activation quantizer, the
static quantizer's Triton launch and the selected kernel's body are CPU
stand-ins that keep their contracts.
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
from tokenspeed_kernel.ops.gemm import fp8_utils
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
    """The TRT-LLM activation quantizer (scales [M, 1] or [M]), the static
    quantizer's Triton launch (it hands back the scale it was given) and the
    body of the kernel the real mm dispatcher selects (scales of one dimension
    or none become columns, as triton_scaled_mm takes them)."""

    def quantize_e4m3_activation(x):
        scale = x.float().abs().amax(-1, keepdim=True) / 448.0
        codes = (x.float() / scale).clamp(-448, 448).to(FP8)
        return codes, scale if op_scale_shape == "M1" else scale.squeeze(-1)

    def static_quant(x, x_s, repeat_scale=False):
        assert x.is_contiguous() and x_s.numel() == 1 and not repeat_scale
        return (x.float() / x_s.float()).clamp(-448, 448).to(FP8), x_s

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
    monkeypatch.setattr(fp8_utils, "static_quant_fp8", static_quant)
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
