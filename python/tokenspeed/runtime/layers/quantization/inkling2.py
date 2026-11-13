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

"""Checkpoint precision contracts for Inkling2.

The manifest names individual tensors, including FP32 model parameters. NVFP4
retains packed E2M1 weights, FP8 per-16 scales, an FP32 amax per expert and an
FP32 outer-scale denominator. Its input quantization selects the lower MAE of
the four/six candidates per block, with a per-token outer scale. The generic
ModelOpt NVFP4 configuration does not express this arithmetic.

MXFP8 retains FP8 values and UE8M0 scale bytes. Checkpoint scales have shape
[experts, output_rows, input_width / 32], including when the manifest declares
a [32, 32] weight block. These scales are already expanded over output rows;
loading must preserve their shape and bytes. Neither format permits a
load-time conversion to BF16. This module validates precision and shards
bytes; matching execution kernels belong behind the tokenspeed-kernel boundary.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import torch

from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig


@dataclass(frozen=True)
class Inkling2TensorPrecision:
    """Storage and computation for one tensor."""

    compute_dtype: str
    storage_dtype: torch.dtype


_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "uint8": torch.uint8,
}
_GROUP_KEYS = {
    "tensors",
    "output_dtype",
    "compute_dtype",
    "weight_transform",
    "weight_block_shape",
    "required_activation_quantization",
    "weight_scale_format",
    "fwd_output_dtype",
}
_REQUIRED_GROUP_KEYS = _GROUP_KEYS - {"weight_scale_format", "fwd_output_dtype"}
_NVFP4_ACTIVATION = {
    "format": "nvfp4_e2m1",
    "block_size": 16,
    "scale_dtype": "float8_e4m3fn",
    "rounding": "nearest",
    "amax_type": "per_token",
    "scale_selection_rule": "fouroversix_mae",
}
_MXFP8_ACTIVATION = {
    "format": "mxfp8_e4m3",
    "block_size": 32,
    "scale_dtype": "ue8m0",
    "rounding": "nearest",
}


def is_inkling2_manifest_stub(quant_config: dict[str, Any]) -> bool:
    """Whether an inline config names Inkling2 but leaves its manifest to hf_quant_config.json."""
    return (
        quant_config.get("quant_method") == "inkling2" and "model" not in quant_config
    )


def read_inkling2_manifest(folder: str) -> dict[str, Any]:
    """Read the full precision manifest from a checkpoint's hf_quant_config.json.

    Args:
        folder: Local checkpoint folder holding its JSON files.

    Returns:
        The manifest. Another quantization method there raises ValueError.
    """
    with open(os.path.join(folder, "hf_quant_config.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("quant_method") != "inkling2":
        raise ValueError("Inkling2 requires its full checkpoint precision manifest")
    return manifest


class Inkling2QuantConfig(QuantizationConfig):
    """Read and validate the original mixed-precision checkpoint manifest.

    Args:
        config: Full quantization JSON containing model.parameter_groups and,
            optionally, a BF16-only mtp.parameter_groups section.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(ignored_layers=[], exclude_modules=[])
        if config.get("quant_method", "inkling2") != "inkling2":
            raise ValueError("Expected inkling2 checkpoint precision")
        model = config.get("model")
        if not isinstance(model, dict) or not isinstance(
            model.get("parameter_groups"), list
        ):
            raise ValueError("Inkling2 requires model.parameter_groups")
        if not model["parameter_groups"]:
            raise ValueError("Inkling2 requires non-empty parameter_groups")
        self.tensor_dtypes: dict[str, torch.dtype] = {}
        self.tensor_precisions: dict[str, Inkling2TensorPrecision] = {}
        self.unembed_output_dtype = torch.bfloat16
        for group in model["parameter_groups"]:
            self._add_group(group)
        if "mtp" in config:
            draft = Inkling2QuantConfig({"model": config["mtp"]})
            if draft.unembed_output_dtype != torch.bfloat16 or any(
                dtype != torch.bfloat16 for dtype in draft.tensor_dtypes.values()
            ):
                raise ValueError("Inkling2 MTP requires BF16 checkpoint precision")

    def _add_group(self, group: dict[str, Any]) -> None:
        if not isinstance(group, dict):
            raise ValueError("Inkling2 precision group must be a mapping")
        if unsupported := group.keys() - _GROUP_KEYS:
            raise ValueError(f"Unsupported Inkling2 precision fields: {unsupported}")
        if missing := _REQUIRED_GROUP_KEYS - group.keys():
            raise ValueError(f"Missing Inkling2 precision fields: {missing}")
        names = group["tensors"]
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) or not name for name in names)
        ):
            raise ValueError("Precision tensors must be a non-empty list of names")
        compute = group["compute_dtype"]
        if compute not in ("float32", "bfloat16", "mxfp8", "nvfp4"):
            raise ValueError(f"Unsupported computation dtype: {compute}")
        expected_storage = {"mxfp8": "float8_e4m3fn", "nvfp4": "uint8"}.get(
            compute, compute
        )
        if group["output_dtype"] != expected_storage:
            raise ValueError(f"Unexpected storage dtype for {compute}")
        if "fwd_output_dtype" in group:
            if (
                compute != "bfloat16"
                or group["fwd_output_dtype"] != "float32"
                or names != ["model.llm.unembed.weight"]
            ):
                raise ValueError("Unsupported Inkling2 forward output precision")
            self.unembed_output_dtype = torch.float32
        if "weight_scale_format" in group and compute != "nvfp4":
            raise ValueError("Only NVFP4 weights use an outer scale")
        auxiliary: dict[str, torch.dtype] = {}
        if compute == "nvfp4":
            if group["required_activation_quantization"] != _NVFP4_ACTIVATION:
                raise ValueError("Unsupported NVFP4 activation precision")
            if (
                group["weight_transform"] != "nvfp4_nearest"
                or group["weight_block_shape"] != [1, 16]
                or group.get("weight_scale_format") != "amax_denominator"
            ):
                raise ValueError("Unsupported NVFP4 weight precision")
            auxiliary = {
                "_scale": torch.float8_e4m3fn,
                "_amax": torch.float32,
                "_outer_scale_denominator": torch.float32,
            }
        elif compute == "mxfp8":
            if group["required_activation_quantization"] != _MXFP8_ACTIVATION:
                raise ValueError("Unsupported MXFP8 activation precision")
            if group["weight_transform"] != "mxfp8_nearest" or group[
                "weight_block_shape"
            ] not in ([1, 32], [32, 32]):
                raise ValueError("Unsupported MXFP8 weight precision")
            auxiliary = {"_scale": torch.uint8}
        elif (
            group["required_activation_quantization"] is not None
            or group["weight_transform"] != "cast"
            or group["weight_block_shape"] != []
        ):
            raise ValueError("Dense precision must not request quantization")
        precision = Inkling2TensorPrecision(
            compute_dtype=compute, storage_dtype=_DTYPES[expected_storage]
        )
        for name in names:
            if auxiliary and not name.endswith(
                (".w13_weight", ".w2_weight", ".shared_w13_weight", ".shared_w2_weight")
            ):
                raise ValueError(f"Quantization requires expert weights: {name}")
            entries = {name: precision.storage_dtype} | {
                name + suffix: dtype for suffix, dtype in auxiliary.items()
            }
            if duplicates := self.tensor_dtypes.keys() & entries.keys():
                raise ValueError(f"Repeated precision entry: {duplicates}")
            self.tensor_dtypes.update(entries)
            self.tensor_precisions[name] = precision

    @classmethod
    def get_name(cls) -> str:
        return "inkling2"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Backend selection admits the actual device and the execution contract.
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["hf_quant_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Inkling2QuantConfig:
        return cls(config)

    def get_scaled_act_names(self) -> list[str]:
        return []

    def precision_for_tensor(self, name: str) -> Inkling2TensorPrecision:
        """Return the declared primary tensor precision, or reject an unknown name."""
        if name not in self.tensor_precisions:
            raise ValueError(f"No checkpoint precision declared for {name}")
        return self.tensor_precisions[name]

    def validate_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """Reject undeclared tensors or storage dtype changes before copying bytes."""
        if name not in self.tensor_dtypes:
            raise ValueError(f"No checkpoint precision declared for {name}")
        expected = self.tensor_dtypes[name]
        if tensor.dtype != expected:
            raise ValueError(f"{name}: expected {expected}, got {tensor.dtype}")


def load_inkling2_expert_tensor(
    param: torch.Tensor,
    loaded: torch.Tensor,
    *,
    ep_rank: int,
    ep_size: int,
    expert_axis: bool,
    slot_experts: tuple[int, ...] | None,
) -> None:
    """Copy an exact checkpoint shard without dtype conversion.

    Args:
        param: Destination tensor for the rank-local shard.
        loaded: Full checkpoint tensor, with the destination storage dtype.
        ep_rank: Contiguous expert shard rank.
        ep_size: Expert shard count.
        expert_axis: Whether axis zero contains experts. False for a scalar
            outer-scale denominator, which must remain replicated.
        slot_experts: Under an expert placement, the checkpoint expert held
            by every physical slot of the layer, rank after rank (replicas
            repeat an expert; every expert has a slot); ``ep_rank``'s slots
            are loaded. None loads the contiguous ``ep_rank`` shard.

    Returns:
        None. The destination is updated in place.
    """
    if slot_experts is not None and (
        not expert_axis or len(slot_experts) != param.shape[0] * ep_size
    ):
        raise ValueError("Expert placement must name one expert per slot")
    expected = list(param.shape)
    if expert_axis:
        if not expected:
            raise ValueError("An expert tensor requires an expert axis")
        # A placement holds every checkpoint expert at least once.
        expected[0] = (
            expected[0] * ep_size if slot_experts is None else max(slot_experts) + 1
        )
    if tuple(loaded.shape) != tuple(expected):
        raise ValueError(f"Expert tensor shape mismatch: {loaded.shape} != {expected}")
    if loaded.dtype != param.dtype:
        raise ValueError(
            f"Expert tensor dtype mismatch: {loaded.dtype} != {param.dtype}"
        )
    shard = loaded
    if slot_experts is not None:
        local = param.shape[0]
        shard = shard[list(slot_experts[ep_rank * local : (ep_rank + 1) * local])]
    elif expert_axis:
        shard = shard.chunk(ep_size, dim=0)[ep_rank]
    with torch.no_grad():
        param.copy_(shard)
