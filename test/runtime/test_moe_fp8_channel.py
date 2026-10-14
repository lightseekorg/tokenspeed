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
from types import SimpleNamespace

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

from tokenspeed.runtime.configs.load_config import LoadConfig
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.loader import build_moe_checkpoint_loader
from tokenspeed.runtime.layers.moe.schema import ExpertCheckpointSchema
from tokenspeed.runtime.layers.moe.types import MoELayerSpec
from tokenspeed.runtime.layers.moe.weights import create_layer_weights
from tokenspeed.runtime.layers.moe.weights.fp8_channel import check_fp8_channel_scales
from tokenspeed.runtime.layers.quantization.compressed_tensors import (
    compressed_tensors as ct,
)
from tokenspeed.runtime.layers.quantization.utils import should_ignore_quant_layer
from tokenspeed.runtime.model_loader import loader
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
    assert torch.isnan(module.w13_weight_scale).all()
    assert torch.isnan(module.w2_weight_scale).all()
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
    check_fp8_channel_scales(module, PREFIX)


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


def _load_all_scales(module, skip=()):
    """Fill every expert's gate, up and down scales but the ``(shard,
    expert)`` pairs in ``skip``."""
    load = module.w13_weight_scale.weight_loader
    experts = module.w13_weight_scale.shape[0]
    for expert in range(experts):
        for shard in ("w1", "w3", "w2"):
            if (shard, expert) in skip:
                continue
            param = module.w2_weight_scale if shard == "w2" else module.w13_weight_scale
            load(param, torch.tensor([0.5]), shard_id=shard, local_expert_id=expert)


def test_fp8_channel_scales_the_checkpoint_did_not_fill_are_refused():
    """Scales start as NaN, so a row no checkpoint tensor filled is refused
    after loading instead of running with scale 1."""
    module = _experts()
    _load_all_scales(module)
    check_fp8_channel_scales(module, PREFIX)
    for skipped, name in (
        (("w3", 2), "w13_weight_scale"),
        (("w2", 0), "w2_weight_scale"),
    ):
        module = _experts()
        _load_all_scales(module, skip={skipped})
        with pytest.raises(ValueError, match=f"{PREFIX}: {name} has .* did not load"):
            check_fp8_channel_scales(module, PREFIX)
    module = _experts()
    _load_all_scales(module)
    module.w2_weight_scale.data[1, 3] = float("inf")
    with pytest.raises(ValueError, match="local expert 1, row 3"):
        check_fp8_channel_scales(module, PREFIX)


@pytest.mark.parametrize("shape", [(4, 48, 1), (4, 16, 1), (4, 48), (48, 2)], ids=str)
def test_fp8_channel_stacked_expert_scales_are_refused(shape):
    """A checkpoint that stores all experts' scales in one tensor (for example
    ``experts.w13_weight_scale [experts, 2 * intermediate, 1]``) reaches the
    loader without shard_id and local_expert_id; it is refused with the
    parameter's name instead of failing with a TypeError."""
    module = _experts()
    for name in ("w13_weight_scale", "w2_weight_scale"):
        param = getattr(module, name)
        with pytest.raises(ValueError, match=f"{PREFIX}.{name}: .*not supported"):
            param.weight_loader(param, torch.ones(shape))
        with pytest.raises(ValueError, match="stacked or fused expert scales"):
            param.weight_loader(
                param,
                torch.ones(shape),
                shard_id="w2" if name == "w2_weight_scale" else "w1",
                local_expert_id=0,
            )
    assert torch.isnan(module.w13_weight_scale).all()


# quantization_config fields of public checkpoints' config.json.
# RedHatAI/Inkling-FP8-dynamic: the experts of layer 2 stay BF16 and are
# ignored by an expert-name regex, while the MoE block itself is quantized.
INKLING_FP8_DYNAMIC = {
    "config_groups": {
        "config_group_0": {
            "format": "float-quantized",
            "input_activations": {
                "dynamic": True,
                "num_bits": 8,
                "strategy": "token",
                "symmetric": True,
                "type": "float",
            },
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "dynamic": False,
                "num_bits": 8,
                "observer": "memoryless_minmax",
                "strategy": "channel",
                "symmetric": True,
                "type": "float",
            },
        }
    },
    "format": "float-quantized",
    "ignore": [
        "model.llm.unembed",
        "model.llm.embed",
        "re:.*norm.*",
        "re:.*bias$",
        "re:.*\\.attn$",
        "re:.*\\.attn\\..*",
        "re:.*sconv$",
        "re:.*\\.gate$",
        "re:.*global_scale$",
        "re:.*shared_experts.*",
        "re:model\\.layers\\.[01]\\.mlp\\..*",
        "re:model\\.layers\\.2\\.mlp\\.experts\\..*",
        "re:model\\.visual\\..*",
        "re:model\\.audio\\..*",
    ],
    "kv_cache_scheme": None,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
    "transform_config": {},
}

# RedHatAI/Qwen3-30B-A3B-FP8-dynamic: only the routers and lm_head are ignored.
QWEN3_30B_A3B_FP8_DYNAMIC = {
    "config_groups": {
        "group_0": {
            "input_activations": {
                "dynamic": True,
                "num_bits": 8,
                "strategy": "token",
                "symmetric": True,
                "type": "float",
            },
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "dynamic": False,
                "num_bits": 8,
                "observer": "mse",
                "strategy": "channel",
                "symmetric": True,
                "type": "float",
            },
        }
    },
    "format": "float-quantized",
    "ignore": [f"model.layers.{i}.mlp.gate" for i in range(48)] + ["lm_head"],
    "kv_cache_scheme": None,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
}


def test_public_fp8_dynamic_moe_configs():
    """Blocks the ignore list covers whole stay unquantized, and the other
    blocks take fp8_channel."""
    inkling = _parse(INKLING_FP8_DYNAMIC)
    # Layers 0 and 1 are dense MLPs; the routed experts start at layer 2.
    for prefix in ("model.layers.3.mlp", "model.layers.3.mlp.experts"):
        assert inkling.moe_weight_dtype(prefix) == "fp8_channel"
    assert inkling.moe_weight_dtype("model.layers.65.mlp") == "fp8_channel"
    assert should_ignore_quant_layer(
        "model.layers.1.mlp.up_proj", inkling.ignored_layers
    )
    assert not should_ignore_quant_layer("model.layers.2.mlp", inkling.ignored_layers)

    qwen3 = _parse(QWEN3_30B_A3B_FP8_DYNAMIC)
    for layer in range(48):
        prefix = f"model.layers.{layer}.mlp"
        assert qwen3.moe_weight_dtype(prefix) == "fp8_channel"
        assert should_ignore_quant_layer(f"{prefix}.gate", qwen3.ignored_layers)
        assert not should_ignore_quant_layer(prefix, qwen3.ignored_layers)


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
def test_fp8_channel_moe_layer_plans_its_kernel(nvidia_arch, arch, tp_size):
    """An MoE layer of compressed-tensors FP8_DYNAMIC experts takes the
    fp8_channel weights and plans FlashInfer's TRT-LLM FP8 per-channel MoE on
    SM100 and SM103 and the Triton FP8 channel MoE on SM90, with the tensors
    they read."""
    nvidia_arch(*arch)
    layer = _moe_layer(tp_size)
    ispp = 512 // tp_size
    assert layer._quant_kind == "fp8_channel"
    assert layer.plan["apply_kernel_name"] == (
        "triton_fp8_channel_precomputed_moe_apply"
        if arch == (9, 0)
        else "flashinfer_trtllm_fp8_channel_moe_apply"
    )
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


def test_fp8_channel_moe_layer_refuses_unloaded_scales(nvidia_arch):
    """MoELayer checks the expert scales before the kernel's post-load step."""
    nvidia_arch(10, 0)
    layer = _moe_layer(1)
    with pytest.raises(ValueError, match="model.layers.1.mlp: w13_weight_scale has"):
        layer.process_weights_after_loading(layer)


def test_fp8_channel_moe_layer_takes_a_dummy_load(nvidia_arch, monkeypatch):
    """``--load-format dummy`` sets the NaN scales to 1 before post-processing
    checks them; a real load that leaves a row unfilled is still refused."""
    nvidia_arch(9, 0)
    layer = _moe_layer(1)
    monkeypatch.setattr(loader, "_initialize_model", lambda *args: layer)
    loader.DummyModelLoader(LoadConfig()).load_model(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        device_config=SimpleNamespace(device="cpu"),
    )
    assert layer._weights_processed
    layer = _moe_layer(1)
    _load_all_scales(layer, skip={("w3", 2)})
    with pytest.raises(ValueError, match="w13_weight_scale .* local expert 2, row 512"):
        layer.process_weights_after_loading(layer)


def test_fp8_channel_moe_layer_has_no_kernel_before_sm90(nvidia_arch):
    nvidia_arch(8, 9)
    with pytest.raises(NoKernelFoundError):
        _moe_layer(1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
