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


"""compressed-tensors FP8 W8A8 experts on the fp8_channel MoE weight kind.

The config's MoE weight kind, the logical layout and its scale loader across
tensor and expert parallelism (through the shared MoE checkpoint loader), and
the plan an MoE layer of such experts makes. The configs are the installed
compressed-tensors package's own serialization.
"""

import dataclasses
import os
import sys

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    preset_name_to_scheme,
)
from tokenspeed_kernel.platform import (
    ArchVersion,
    Platform,
    _get_cuda_sm_features,
    current_platform,
)
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import NoKernelFoundError

from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.loader import build_moe_checkpoint_loader
from tokenspeed.runtime.layers.moe.schema import ExpertCheckpointSchema
from tokenspeed.runtime.layers.moe.types import MoELayerSpec
from tokenspeed.runtime.layers.moe.weights import create_layer_weights
from tokenspeed.runtime.layers.quantization.compressed_tensors import (
    compressed_tensors as ct,
)
from tokenspeed.runtime.utils.env import global_server_args_dict

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(
    est_time=15,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="the FP8 channel MoE kernel is NVIDIA's",
)

FP8 = torch.float8_e4m3fn
FP8_ARGS = dict(num_bits=8, type="float", symmetric=True)
PREFIX = "model.layers.1.mlp.experts"


def _config(weights, inputs):
    """config.json quantization_config as llm-compressor saves it."""
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=weights,
        input_activations=inputs,
        format="float-quantized",
    )
    raw = QuantizationConfig(
        config_groups={"group_0": scheme},
        format="float-quantized",
        quantization_status="compressed",
        ignore=["lm_head"],
    ).model_dump(mode="json", exclude=["quant_method"])
    return {"quant_method": "compressed-tensors", "version": "0.18.0", **raw}


def _preset(name):
    scheme = preset_name_to_scheme(name, ["Linear"])
    return scheme.weights, scheme.input_activations


CHANNEL_DYNAMIC = _config(*_preset("FP8_DYNAMIC"))
TENSOR_DYNAMIC = _config(
    QuantizationArgs(strategy="tensor", dynamic=False, **FP8_ARGS),
    _preset("FP8_DYNAMIC")[1],
)
TENSOR_STATIC = _config(*_preset("FP8"))
FP4_CHANNEL_DYNAMIC = _config(
    QuantizationArgs(num_bits=4, type="float", strategy="channel", symmetric=True),
    QuantizationArgs(
        num_bits=4, type="float", strategy="token", dynamic=True, symmetric=True
    ),
)


def _parse(raw):
    return ct.CompressedTensorsConfig.from_config(raw)


def _fp8_checkpoint(rows, cols, *, scale_dtype=torch.bfloat16):
    """(FP8 codes, [rows, 1] scales) of a random matrix; for E4M3 scales the
    magnitudes keep every scale a normal E4M3 value."""
    generator = torch.Generator().manual_seed(rows * 1000 + cols)
    weight = torch.randn(rows, cols, generator=generator) * 2.0 ** torch.randint(
        -4, 3, (rows, 1), generator=generator
    )
    if scale_dtype == FP8:
        weight = weight * 64.0
    scale = (weight.abs().amax(-1, keepdim=True) / 448.0).to(scale_dtype)
    codes = (weight / scale.float()).clamp(-448, 448).to(FP8)
    return codes, scale


def test_moe_dtype_of_fp8_experts():
    assert _parse(CHANNEL_DYNAMIC).moe_weight_dtype(PREFIX) == "fp8_channel"
    assert _parse(TENSOR_DYNAMIC).moe_weight_dtype(PREFIX) == "fp8_channel"
    with pytest.raises(ValueError, match="static input scale are not supported"):
        _parse(TENSOR_STATIC).moe_weight_dtype(PREFIX)
    # 4-bit floats are not FP8.
    with pytest.raises(ValueError, match="unsupported compressed-tensors MoE scheme"):
        _parse(FP4_CHANNEL_DYNAMIC).moe_weight_dtype(PREFIX)


def _experts(
    tp_rank=0, tp_size=1, ep_rank=0, ep_size=1, experts=4, hidden=16, intermediate=24
):
    spec = MoELayerSpec(
        top_k=2,
        num_experts=experts,
        num_local_experts=experts // ep_size,
        hidden_size=hidden,
        intermediate_size=intermediate,
        activation="silu",
        tp_rank=tp_rank,
        tp_size=tp_size,
        ep_rank=ep_rank,
        ep_size=ep_size,
        prefix=PREFIX,
        a2a_backend="none",
    )
    module = torch.nn.Module()
    create_layer_weights(spec, module, "fp8_channel", _parse(CHANNEL_DYNAMIC))
    return module


@pytest.mark.parametrize("scale_dtype", [torch.bfloat16, torch.float32, FP8])
@pytest.mark.parametrize(
    "layout",
    [(0, 2, 0, 1), (1, 2, 0, 1), (0, 1, 1, 2)],
    ids=["tp2-rank0", "tp2-rank1", "ep2-rank1"],
)
def test_fp8_channel_experts_load_logical_tensors(scale_dtype, layout):
    """Through the shared MoE checkpoint loader: FP8 weights in w13 (gate
    rows, then up rows) and w2, an FP32 scale per output row; w2's scales are
    the hidden channels, whole on every rank."""
    tp_rank, tp_size, ep_rank, ep_size = layout
    module = _experts(tp_rank, tp_size, ep_rank, ep_size)
    ispp, local = 24 // tp_size, 4 // ep_size
    shapes = {name: (p.dtype, tuple(p.shape)) for name, p in module.named_parameters()}
    assert shapes == {
        "w13_weight": (FP8, (local, 2 * ispp, 16)),
        "w2_weight": (FP8, (local, 16, ispp)),
        "w13_weight_scale": (torch.float32, (local, 2 * ispp)),
        "w2_weight_scale": (torch.float32, (local, 16)),
    }
    assert torch.equal(module.w13_weight_scale, torch.ones(local, 2 * ispp))
    params = {f"{PREFIX}.{n}": p for n, p in module.named_parameters()}
    loader = build_moe_checkpoint_loader(
        params_dict=params,
        expert_schema=ExpertCheckpointSchema(
            gate_proj_name="gate_proj",
            down_proj_name="down_proj",
            up_proj_name="up_proj",
        ),
        num_experts=4,
        ep_rank=ep_rank,
        ep_size=ep_size,
    )
    checkpoint = {}
    for expert in range(4):
        for projection, (rows, cols) in (
            ("gate_proj", (24, 16)),
            ("up_proj", (24, 16)),
            ("down_proj", (16, 24)),
        ):
            codes, scale = _fp8_checkpoint(rows + expert, cols, scale_dtype=scale_dtype)
            codes, scale = codes[:rows], scale[:rows]
            name = f"{PREFIX}.{expert}.{projection}"
            checkpoint[name] = (codes, scale)
            if loader.matches(f"{name}.weight") and expert // local == ep_rank:
                loader.load(f"{name}.weight", codes)
                loader.load(f"{name}.weight_scale", scale)
    part = slice(ispp * tp_rank, ispp * (tp_rank + 1))
    for local_id in range(local):
        prefix = f"{PREFIX}.{ep_rank * local + local_id}"
        gate, up, down = (
            checkpoint[f"{prefix}.{p}"] for p in ("gate_proj", "up_proj", "down_proj")
        )
        w13 = module.w13_weight[local_id].view(torch.uint8)
        assert torch.equal(w13[:ispp], gate[0][part].view(torch.uint8))
        assert torch.equal(w13[ispp:], up[0][part].view(torch.uint8))
        assert torch.equal(
            module.w13_weight_scale[local_id, :ispp], gate[1][part, 0].float()
        )
        assert torch.equal(
            module.w13_weight_scale[local_id, ispp:], up[1][part, 0].float()
        )
        assert torch.equal(
            module.w2_weight[local_id].view(torch.uint8),
            down[0][:, part].view(torch.uint8),
        )
        assert torch.equal(module.w2_weight_scale[local_id], down[1][:, 0].float())


def test_fp8_channel_per_tensor_scales_fill_their_rows():
    module = _experts(tp_rank=1, tp_size=2)
    load = module.w13_weight_scale.weight_loader
    load(
        module.w13_weight_scale,
        torch.tensor([0.5], dtype=torch.bfloat16),
        shard_id="w1",
        local_expert_id=2,
    )
    load(
        module.w13_weight_scale, torch.tensor([0.25]), shard_id="w3", local_expert_id=2
    )
    load(
        module.w2_weight_scale,
        torch.tensor([0.125], dtype=FP8),
        shard_id="w2",
        local_expert_id=2,
    )
    assert module.w13_weight_scale[2].tolist() == [0.5] * 12 + [0.25] * 12
    assert module.w2_weight_scale[2].tolist() == [0.125] * 16
    with pytest.raises(ValueError, match="w2 scales per output channel: expected 16"):
        load(
            module.w2_weight_scale, torch.ones(15, 1), shard_id="w2", local_expert_id=0
        )


@pytest.mark.parametrize("presharded", [False, True])
def test_fp8_channel_scales_of_the_wrong_length_are_refused(presharded):
    """Gate and up scales: one per checkpoint row (TP2 x 12 here, or this
    rank's 12 when presharded) or one per tensor; the count is checked
    before the rank's rows are cut, so a short or long tensor and a
    one-row slice cannot pass for a per-tensor scale."""
    module = _experts(tp_rank=1, tp_size=2)
    load = module.w13_weight_scale.weight_loader
    rows = 12 if presharded else 24
    scale = torch.arange(1, rows + 1, dtype=torch.float32).reshape(-1, 1)
    load(
        module.w13_weight_scale,
        scale,
        shard_id="w3",
        local_expert_id=1,
        use_presharded_weights=presharded,
    )
    part = scale[:, 0] if presharded else scale[12:, 0]
    assert torch.equal(module.w13_weight_scale[1, 12:], part)
    for wrong in (rows - 1, rows + 1, 13 if not presharded else 2):
        for shard in ("w1", "w3"):
            with pytest.raises(
                ValueError, match=f"{shard} scales per output channel: expected {rows}"
            ):
                load(
                    module.w13_weight_scale,
                    torch.ones(wrong, 1),
                    shard_id=shard,
                    local_expert_id=0,
                    use_presharded_weights=presharded,
                )


@pytest.fixture
def nvidia_arch(monkeypatch):
    """Plan on a synthetic NVIDIA arch; the host's platform and the
    registry's cached selections are restored afterwards."""
    original = current_platform()
    registry = KernelRegistry.get()
    monkeypatch.setitem(global_server_args_dict, "moe_mxfp4_fp8_activation", False)
    monkeypatch.setitem(global_server_args_dict, "ep_num_redundant_experts", 0)

    def switch(major, minor):
        arch = ArchVersion(major, minor)
        Platform.override(
            dataclasses.replace(
                original,
                vendor="nvidia",
                arch_version=arch,
                sm_features=_get_cuda_sm_features(arch),
            )
        )
        registry.clear_cache()

    yield switch
    Platform.override(original)
    registry.clear_cache()


def _moe_layer(tp_size, intermediate_size=512):
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return MoELayer(
            top_k=2,
            num_experts=4,
            hidden_size=256,
            intermediate_size=intermediate_size,
            quant_config=_parse(CHANNEL_DYNAMIC),
            layer_index=1,
            prefix="model.layers.1.mlp",
            tp_rank=0,
            tp_size=tp_size,
        )
    finally:
        torch.set_default_dtype(default_dtype)


@pytest.mark.parametrize("arch", [(9, 0), (10, 0), (10, 3)])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_fp8_channel_moe_layer_plans_the_triton_kernel(nvidia_arch, arch, tp_size):
    """An MoE layer of compressed-tensors FP8_DYNAMIC experts takes the
    fp8_channel weights and plans the Triton FP8 channel MoE on SM90 to
    SM103, with the tensors it reads."""
    nvidia_arch(*arch)
    layer = _moe_layer(tp_size)
    ispp = 512 // tp_size
    assert layer._quant_kind == "fp8_channel"
    assert layer.plan["apply_kernel_name"] == "triton_fp8_channel_precomputed_moe_apply"
    shapes = {
        name: (p.dtype, tuple(p.shape))
        for name, p in layer.named_parameters()
        if name.startswith(("w13", "w2"))
    }
    assert shapes == {
        "w13_weight": (FP8, (4, 2 * ispp, 256)),
        "w2_weight": (FP8, (4, 256, ispp)),
        "w13_weight_scale": (torch.float32, (4, 2 * ispp)),
        "w2_weight_scale": (torch.float32, (4, 256)),
    }


def test_fp8_channel_moe_layer_has_no_kernel_before_sm90(nvidia_arch):
    nvidia_arch(8, 9)
    with pytest.raises(NoKernelFoundError):
        _moe_layer(1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
