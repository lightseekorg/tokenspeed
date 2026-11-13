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

"""CPU coverage for manifest admission and exact checkpoint shard loading."""

import copy
import json
from functools import partial

import pytest
import torch
from transformers import PretrainedConfig

from tokenspeed.runtime.configs.load_config import LoadConfig
from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.layers.quantization import get_quantization_config
from tokenspeed.runtime.layers.quantization.inkling2 import (
    Inkling2QuantConfig,
    load_inkling2_expert_tensor,
)
from tokenspeed.runtime.model_loader.weight_utils import get_quant_config


def _group(compute: str, names: list[str]) -> dict:
    group = {
        "compute_dtype": compute,
        "output_dtype": {"nvfp4": "uint8", "mxfp8": "float8_e4m3fn"}.get(
            compute, compute
        ),
        "tensors": names,
        "weight_transform": "cast",
        "weight_block_shape": [],
        "required_activation_quantization": None,
    }
    if compute == "nvfp4":
        group.update(
            weight_transform="nvfp4_nearest",
            weight_block_shape=[1, 16],
            weight_scale_format="amax_denominator",
            required_activation_quantization={
                "format": "nvfp4_e2m1",
                "block_size": 16,
                "scale_dtype": "float8_e4m3fn",
                "rounding": "nearest",
                "amax_type": "per_token",
                "scale_selection_rule": "fouroversix_mae",
            },
        )
    elif compute == "mxfp8":
        group.update(
            weight_transform="mxfp8_nearest",
            weight_block_shape=[32, 32],
            required_activation_quantization={
                "format": "mxfp8_e4m3",
                "block_size": 32,
                "scale_dtype": "ue8m0",
                "rounding": "nearest",
            },
        )
    return group


@pytest.fixture
def manifest() -> dict:
    output = _group("bfloat16", ["model.llm.unembed.weight"])
    output["fwd_output_dtype"] = "float32"
    return {
        "quant_method": "inkling2",
        "model": {
            "parameter_groups": [
                _group("float32", ["model.llm.layers.0.mlp.gate.weight"]),
                _group(
                    "nvfp4",
                    [
                        f"model.llm.layers.0.mlp.experts.{leaf}_weight"
                        for leaf in ("w13", "w2")
                    ],
                ),
                _group(
                    "mxfp8",
                    [
                        f"model.llm.layers.1.mlp.experts.{leaf}_weight"
                        for leaf in ("w13", "w2")
                    ],
                ),
                output,
            ]
        },
        "mtp": {"parameter_groups": [_group("bfloat16", ["fuse_weight"])]},
    }


def test_manifest_preserves_mixed_storage_and_compute(manifest: dict) -> None:
    config = get_quantization_config("inkling2").from_config(manifest)
    nvfp4 = "model.llm.layers.0.mlp.experts.w13_weight"
    mxfp8 = "model.llm.layers.1.mlp.experts.w13_weight"
    assert config.precision_for_tensor(nvfp4).compute_dtype == "nvfp4"
    assert config.precision_for_tensor(mxfp8).compute_dtype == "mxfp8"
    assert config.tensor_dtypes[nvfp4] == torch.uint8
    assert config.tensor_dtypes[nvfp4 + "_scale"] == torch.float8_e4m3fn
    assert config.tensor_dtypes[nvfp4 + "_amax"] == torch.float32
    assert config.tensor_dtypes[nvfp4 + "_outer_scale_denominator"] == torch.float32
    assert config.tensor_dtypes[mxfp8] == torch.float8_e4m3fn
    assert config.tensor_dtypes[mxfp8 + "_scale"] == torch.uint8
    assert config.tensor_dtypes["model.llm.layers.0.mlp.gate.weight"] == torch.float32
    assert config.unembed_output_dtype == torch.float32
    assert (
        config.precision_for_tensor("model.llm.unembed.weight").storage_dtype
        == torch.bfloat16
    )


def test_manifest_rejects_a_quantized_mtp_section(manifest: dict) -> None:
    manifest["mtp"]["parameter_groups"][0] = _group("float32", ["fuse_weight"])
    with pytest.raises(ValueError, match="MTP requires BF16"):
        Inkling2QuantConfig.from_config(manifest)


@pytest.mark.parametrize(
    ("compute", "field", "value"),
    [
        ("nvfp4", "weight_transform", "cast"),
        ("nvfp4", "weight_block_shape", [1, 32]),
        ("nvfp4", "weight_scale_format", "scale"),
        ("nvfp4", "output_dtype", "bfloat16"),
        ("nvfp4", "required_activation_quantization", None),
        ("mxfp8", "weight_transform", "mxfp8_roundtrip_nearest"),
        ("mxfp8", "weight_block_shape", [128, 128]),
        ("mxfp8", "weight_scale_format", "amax_denominator"),
        ("mxfp8", "output_dtype", "bfloat16"),
        ("mxfp8", "required_activation_quantization", None),
        ("float32", "weight_block_shape", [1, 16]),
        ("bfloat16", "weight_transform", "nvfp4_nearest"),
        ("bfloat16", "unknown_behavior", True),
    ],
)
def test_rejects_precision_changes(compute: str, field: str, value: object) -> None:
    group = _group(compute, ["model.llm.layers.0.mlp.experts.w13_weight"])
    group[field] = value
    with pytest.raises(ValueError):
        Inkling2QuantConfig.from_config({"model": {"parameter_groups": [group]}})


def test_rejects_nvfp4_scale_rule_change() -> None:
    group = _group("nvfp4", ["model.llm.layers.0.mlp.experts.w13_weight"])
    group["required_activation_quantization"]["scale_selection_rule"] = "max"
    with pytest.raises(ValueError, match="activation precision"):
        Inkling2QuantConfig.from_config({"model": {"parameter_groups": [group]}})


@pytest.mark.parametrize("reverse", [False, True])
def test_rejects_duplicate_auxiliary_precision(reverse: bool) -> None:
    name = "model.llm.layers.0.mlp.experts.w13_weight"
    groups = [_group("nvfp4", [name]), _group("float32", [name + "_scale"])]
    if reverse:
        groups.reverse()
    with pytest.raises(ValueError, match="Repeated precision"):
        Inkling2QuantConfig.from_config({"model": {"parameter_groups": groups}})


def test_rejects_non_bf16_mtp(manifest: dict) -> None:
    manifest["mtp"]["parameter_groups"][0] = _group("float32", ["fuse_weight"])
    with pytest.raises(ValueError, match="MTP requires BF16"):
        Inkling2QuantConfig.from_config(manifest)


def test_tensor_validation_rejects_cast_and_undeclared_parameter(
    manifest: dict,
) -> None:
    config = Inkling2QuantConfig.from_config(manifest)
    with pytest.raises(ValueError, match="expected torch.uint8"):
        config.validate_tensor(
            "model.llm.layers.0.mlp.experts.w13_weight",
            torch.ones(4, dtype=torch.bfloat16),
        )
    with pytest.raises(ValueError, match="No checkpoint precision"):
        config.validate_tensor("missing.weight", torch.ones(4))


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e4m3fn, torch.float32])
def test_expert_shard_preserves_storage_bytes(dtype: torch.dtype) -> None:
    loaded = torch.arange(4 * 8 * 16).view(4, 8, 16).to(dtype)
    target = torch.empty(2, 8, 16, dtype=dtype)
    load_inkling2_expert_tensor(
        target,
        loaded,
        ep_rank=1,
        ep_size=2,
        expert_axis=True,
        slot_experts=None,
    )
    assert torch.equal(target.view(torch.uint8), loaded[2:].view(torch.uint8))


def test_expert_placement_fills_every_slot_from_its_checkpoint_expert() -> None:
    loaded = torch.arange(4 * 8 * 16).view(4, 8, 16).to(torch.uint8)
    target = torch.empty(3, 8, 16, dtype=torch.uint8)
    load = partial(
        load_inkling2_expert_tensor,
        ep_rank=1,
        ep_size=2,
        expert_axis=True,
        slot_experts=(1, 2, 0, 3, 0, 3),
    )
    load(target, loaded)
    assert torch.equal(target, loaded[[3, 0, 3]])
    # A tensor without every placed expert (such as a pre-sharded one) is refused.
    with pytest.raises(ValueError, match="shape mismatch"):
        load(target, loaded[:2])


def test_replicated_denominator() -> None:
    denominator = torch.empty(1, dtype=torch.float32)
    load_inkling2_expert_tensor(
        denominator,
        torch.tensor([1536.0], dtype=torch.float32),
        ep_rank=1,
        ep_size=2,
        expert_axis=False,
        slot_experts=None,
    )
    assert denominator.item() == 1536.0


@pytest.mark.parametrize(("rows", "width"), [(64, 64), (64, 32)])
def test_mxfp8_block_manifest_keeps_rowwise_checkpoint_scales(
    manifest: dict, rows: int, width: int
) -> None:
    config = Inkling2QuantConfig.from_config(manifest)
    name = "model.llm.layers.1.mlp.experts.w13_weight"
    weight = torch.ones(4, rows, width, dtype=torch.float8_e4m3fn)
    scales = (
        torch.arange(4 * rows * (width // 32), dtype=torch.int64)
        .to(torch.uint8)
        .view(4, rows, width // 32)
    )
    config.validate_tensor(name, weight)
    config.validate_tensor(name + "_scale", scales)
    target = torch.empty(2, rows, width // 32, dtype=torch.uint8)
    load_inkling2_expert_tensor(
        target,
        scales,
        ep_rank=1,
        ep_size=2,
        expert_axis=True,
        slot_experts=None,
    )
    assert torch.equal(target, scales[2:])


@pytest.mark.parametrize("bad_dtype", [False, True])
def test_expert_loader_rejects_cast_or_bad_shape(bad_dtype: bool) -> None:
    target = torch.empty(2, 8, 16, dtype=torch.uint8)
    loaded = (
        torch.ones(4, 8, 16, dtype=torch.bfloat16)
        if bad_dtype
        else torch.ones(3, 8, 16, dtype=torch.uint8)
    )
    with pytest.raises(ValueError, match="mismatch"):
        load_inkling2_expert_tensor(
            target,
            loaded,
            ep_rank=0,
            ep_size=2,
            expert_axis=True,
            slot_experts=None,
        )


def _loader_model(path: str, inline: dict | None) -> ModelConfig:
    # The parser needs checkpoint metadata only, without execution or hardware setup.
    model = ModelConfig.__new__(ModelConfig)
    model.model_path = path
    model.revision = None
    model.quantization = "inkling2"
    model.hf_config = PretrainedConfig()
    if inline is not None:
        model.hf_config.quantization_config = inline
    return model


@pytest.mark.parametrize("inline", [None, {"quant_method": "inkling2"}])
def test_external_manifest_detection_and_loading(
    tmp_path, manifest: dict, inline
) -> None:
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(manifest))
    model = _loader_model(str(tmp_path), inline)
    detected = model._parse_quant_hf_config()
    assert detected == manifest
    config = get_quant_config(model, LoadConfig())
    assert config.tensor_precisions == (
        Inkling2QuantConfig.from_config(manifest).tensor_precisions
    )
    assert config.unembed_output_dtype == torch.float32


def test_external_manifest_cannot_replace_inline_method(
    tmp_path, manifest: dict
) -> None:
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(manifest))
    inline = {"quant_method": "fp8", "activation_scheme": "dynamic"}
    model = _loader_model(str(tmp_path), inline)
    assert model._parse_quant_hf_config() == inline


def test_empty_inline_config_is_unquantized_only_for_the_inkling2_draft(
    tmp_path, manifest: dict
) -> None:
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(manifest))
    model = _loader_model(str(tmp_path), {})
    model.quantization = None
    model.hf_config.architectures = ["Inkling2ForConditionalGenerationMTP"]
    model._verify_quantization()
    assert model.quantization is None
    model.hf_config.architectures = ["LlamaForCausalLM"]
    with pytest.raises(ValueError, match="Unknown quantization method"):
        model._verify_quantization()


def test_token_only_manifest_rejects_missing_precision(tmp_path) -> None:
    incomplete = {"quant_method": "inkling2"}
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(incomplete))
    model = _loader_model(str(tmp_path), incomplete)
    with pytest.raises(ValueError, match="model.parameter_groups"):
        get_quant_config(model, LoadConfig())


def test_external_manifest_rejects_method_mismatch(tmp_path, manifest: dict) -> None:
    external = copy.deepcopy(manifest)
    external["quant_method"] = "modelopt"
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(external))
    model = _loader_model(str(tmp_path), {"quant_method": "inkling2"})
    with pytest.raises(ValueError, match="full checkpoint precision"):
        get_quant_config(model, LoadConfig())
