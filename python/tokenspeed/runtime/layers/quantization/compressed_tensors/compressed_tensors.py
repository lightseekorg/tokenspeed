# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
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

import logging
from contextlib import suppress
from typing import Any, Literal, NamedTuple, cast

import torch
from compressed_tensors.config import (
    CompressionFormat,
    SparsityCompressionConfig,
    SparsityStructure,
)
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from pydantic import BaseModel

from tokenspeed.runtime.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizationConfig,
)
from tokenspeed.runtime.layers.quantization.compressed_tensors.gptq_marlin_moe import (
    is_activation_quantization_format,
)
from tokenspeed.runtime.layers.quantization.compressed_tensors.schemes.compressed_tensors_scheme import (
    CompressedTensorsScheme,
)
from tokenspeed.runtime.layers.quantization.utils import find_matched_target

# ruff: noqa: F821


logger = logging.getLogger(__name__)

__all__ = ["CompressedTensorsLinearMethod"]

SPARSITY_CONFIG_NAME: Literal["sparsity_config"] = "sparsity_config"
QUANTIZATION_SCHEME_MAP_TYPE = dict[str, dict[str, QuantizationArgs] | None]
WNA16_SUPPORTED_BITS = [4, 8]


class DeviceCapability(NamedTuple):
    major: int
    minor: int

    def as_version_str(self) -> str:
        return f"{self.major}.{self.minor}"

    def to_int(self) -> int:
        """
        Express device capability as an integer ``<major><minor>``.

        It is assumed that the minor version is always a single digit.
        """
        if not 0 <= self.minor < 10:
            raise ValueError(f"Invalid device capability minor version: {self.minor}.")
        return self.major * 10 + self.minor


class CompressedTensorsConfig(QuantizationConfig):
    DeepSeekFP8Config = None

    def __init__(
        self,
        target_scheme_map: dict[str, Any],
        ignore: list[str],
        quant_format: str,
        sparsity_scheme_map: dict[str, SparsityCompressionConfig],
        sparsity_ignore_list: list[str],
        kv_cache_scheme: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        packed_modules_mapping: dict[str, list[str]] | None = None,
    ):
        super().__init__(ignored_layers=ignore)
        self.quant_format = quant_format
        # Map from [target -> scheme]
        self.target_scheme_map = target_scheme_map
        self.kv_cache_scheme = kv_cache_scheme
        self.sparsity_scheme_map = sparsity_scheme_map
        self.sparsity_ignore_list = sparsity_ignore_list
        self.config = config
        _packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
        self.packed_modules_mapping = packed_modules_mapping or _packed_modules_mapping
        # True when any config_group is packed INT4 weights + dynamic FP8
        # activations (GLM-5.3 W4A8). Kimi compressed-tensors MXFP4 / INT4
        # group-32 checkpoints are weight-only, so this stays False there.
        self.is_w4a8_fp8 = any(
            self._is_wint4afp8(
                (scheme or {}).get("weights"),
                (scheme or {}).get("input_activations"),
                (scheme or {}).get("format"),
            )
            for scheme in self.target_scheme_map.values()
        )
        self.use_dynamic_mxfp4_activations = False

    def get_linear_method(self) -> CompressedTensorsLinearMethod:
        return CompressedTensorsLinearMethod(self)

    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def get_name(self) -> str:
        return "compressed_tensors"

    @property
    def weight_block_size(self) -> list[int] | None:
        """Block size for block-FP8 dense layers in a mixed-precision config.

        A mixed-precision checkpoint may have no ``Linear`` target, or may use
        that target for group-wise INT4 experts that have no block structure.
        Prefer any target that declares ``block_structure`` so GLM-5 DSA
        padding and fused QKV-A scale sharding still see ``[128, 128]``.
        """
        targets = list(self.target_scheme_map)
        if "Linear" in self.target_scheme_map:
            targets = ["Linear"] + [t for t in targets if t != "Linear"]
        for target in targets:
            scheme = self.target_scheme_map.get(target)
            if not scheme:
                continue
            block_structure = getattr(scheme.get("weights"), "block_structure", None)
            if block_structure:
                return list(block_structure)
        return None

    def _scheme_dict_for_name(self, layer_name: str) -> dict[str, Any] | None:
        """Return the config_group scheme that matches ``layer_name``, if any."""
        if not self.target_scheme_map:
            return None
        stub = torch.nn.Module()
        try:
            matched_target = find_matched_target(
                layer_name=layer_name,
                module=stub,
                targets=self.target_scheme_map.keys(),
                fused_mapping=self.packed_modules_mapping,
            )
        except ValueError:
            return None
        return self.target_scheme_map.get(matched_target)

    def _moe_scheme_dict(self, prefix: str = "") -> dict[str, Any] | None:
        """Scheme for routed experts under ``prefix``, not a catch-all Linear."""
        candidates: list[str] = []
        if prefix:
            if not prefix.endswith(".experts"):
                candidates.append(f"{prefix}.experts.0.gate_proj")
            candidates.extend(
                (
                    f"{prefix}.0.gate_proj",
                    f"{prefix}.experts.0.down_proj",
                    prefix,
                )
            )
        for name in candidates:
            scheme = self._scheme_dict_for_name(name)
            if scheme is not None:
                return scheme
        return self.target_scheme_map.get("Linear")

    def moe_weight_dtype(self, prefix: str = "") -> str:
        # Container format: resolve the *matched* routed-expert scheme, not
        # ``target_scheme_map["Linear"]``. Mixed-precision GLM-5.3 W4A8 puts
        # INT4 group-128 + FP8 activations on the expert regex and block-FP8
        # on attention / shared experts; Kimi still keys everything as Linear.
        scheme = self._moe_scheme_dict(prefix)
        if scheme is None:
            raise ValueError(
                "unsupported compressed-tensors MoE scheme for kernel "
                f"selection: no matching target for {prefix!r}"
            )
        weight_quant = scheme.get("weights")
        input_quant = scheme.get("input_activations")
        scheme_format = scheme.get("format")
        if self._is_wint4afp8(weight_quant, input_quant, scheme_format):
            return "w4a8"
        is_4bit_group32 = (
            weight_quant is not None
            and weight_quant.num_bits == 4
            and weight_quant.strategy == QuantizationStrategy.GROUP.value
            and weight_quant.group_size == 32
            and not weight_quant.actorder
        )
        if is_4bit_group32:
            if weight_quant.type == QuantizationType.INT and (
                self._is_wNa16_group_channel(weight_quant, input_quant)
            ):
                return "mxint4"
            if weight_quant.type == QuantizationType.FLOAT:
                return "mxfp4"
        raise ValueError(
            f"unsupported compressed-tensors MoE scheme for kernel selection: "
            f"{weight_quant}"
        )

    def moe_group_size(self, prefix: str = "") -> int:
        """Group size of the matched routed-expert weight scheme."""
        scheme = self._moe_scheme_dict(prefix)
        weight_quant = None if scheme is None else scheme.get("weights")
        group_size = getattr(weight_quant, "group_size", None)
        if not group_size:
            raise ValueError(
                f"compressed-tensors MoE scheme for {prefix!r} has no group_size"
            )
        return int(group_size)

    def get_scaled_act_names(self) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CompressedTensorsConfig:
        ignore: list[str] = cast(list[str], config.get("ignore", []))
        quant_format = cast(str, config.get("format"))
        target_scheme_map = cls._quantization_scheme_map_from_config(config=config)
        sparsity_scheme_map, sparsity_ignore_list = cls._parse_sparsity_config(
            config=config
        )
        packed_modules_mapping = config.get("packed_modules_mapping", {})

        return cls(
            target_scheme_map=target_scheme_map,
            ignore=ignore,
            quant_format=quant_format,
            sparsity_scheme_map=sparsity_scheme_map,
            sparsity_ignore_list=sparsity_ignore_list,
            config=config,
            packed_modules_mapping=packed_modules_mapping,
        )

    @classmethod
    def _parse_sparsity_config(
        cls, config: dict[str, Any]
    ) -> tuple[dict[str, SparsityCompressionConfig], list[str]]:
        """
        :param config: The `quantization_config` dictionary from config.json
        :return: A tuple with two elements
            1. A dictionary mapping target layer names to their corresponding
                sparsity_config
            2. A list of layer names to ignore for sparsity
        """
        if not (sparsity_config := config.get(SPARSITY_CONFIG_NAME)):
            return dict(), []

        sparsity_config = SparsityCompressionConfig.model_validate(sparsity_config)
        sparse_scheme_map: dict[str, SparsityCompressionConfig] = {
            target: sparsity_config for target in sparsity_config.targets or list()
        }
        sparsity_ignore_list = sparsity_config.ignore or list()
        return sparse_scheme_map, sparsity_ignore_list

    @classmethod
    def _quantization_scheme_map_from_config(
        cls, config: dict[str, Any]
    ) -> QUANTIZATION_SCHEME_MAP_TYPE:
        """
        :param config: The `quantization_config` dictionary from config.json
        :return: A dictionary mapping target layer names to their corresponding
            quantization_args for weights and input activations
        """
        target_scheme_map: dict[str, Any] = dict()
        quant_format = cast(str, config.get("format"))

        # The quant_config has multiple config_groups, each containing
        # an input_activations key with details about how the activations are
        # quantized, a weights key indicating how the weights are quantized,
        # and a list of targets under the `targets` key, dictating which
        # layers are impacted by the quantization details. The quantization
        # details follow the structure defined by the QuantizationArgs
        # pydantic model, which is used to verify the structure of the
        # quant_config and also store the details for later use.

        config_groups = config.get("config_groups", dict())
        for _, quant_config in config_groups.items():
            targets = quant_config.get("targets")
            for target in targets:
                target_scheme_map[target] = {}
                target_scheme_map[target]["weights"] = QuantizationArgs.model_validate(
                    quant_config.get("weights")
                )

                target_scheme_map[target]["input_activations"] = None
                # A config_group may carry its own format. When several groups
                # disagree, compressed-tensors sets the top-level format to
                # "mixed-precision" (or keeps the first group's format) and the
                # real format lives on each group, so the per-group value must
                # win for activation-quant detection.
                group_format = quant_config.get("format")
                target_scheme_map[target]["format"] = group_format
                act_quant_format = is_activation_quantization_format(
                    group_format if group_format is not None else quant_format
                )

                if act_quant_format:
                    input_activations = quant_config.get("input_activations")
                    # When the format admits activation quant but the group
                    # omits input_activations: valid for w8a16fp8 (FLOAT
                    # weights) and pack-quantized weight-only INT (WNA16).
                    if not input_activations:
                        weight_type = target_scheme_map[target]["weights"].type
                        if weight_type == QuantizationType.INT:
                            pass
                        elif (
                            target_scheme_map[target]["weights"].type
                            != QuantizationType.FLOAT
                        ):
                            raise ValueError(
                                "Activation quantization config is missing input_activations."
                            )
                    else:
                        target_scheme_map[target]["input_activations"] = (
                            QuantizationArgs.model_validate(  # noqa: E501
                                quant_config.get("input_activations")
                            )
                        )
        return target_scheme_map

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def _check_scheme_supported(self, min_capability: int, error: bool = True) -> bool:
        from tokenspeed_kernel.platform import current_platform

        platform = current_platform()
        capability_tuple = DeviceCapability(
            platform.arch_version.major, platform.arch_version.minor
        )

        if capability_tuple is not None:
            capability = capability_tuple.to_int()
            supported = capability >= min_capability
            if error and not supported:
                raise RuntimeError(
                    "Quantization scheme is not supported for ",
                    f"the current GPU. Min capability: {min_capability}. ",
                    f"Current capability: {capability}.",
                )
            return supported
        else:
            return False

    def _is_fp8_w8a8(self, weight_quant: BaseModel, input_quant: BaseModel) -> bool:
        # Confirm weights and activations quantized.
        if weight_quant is None or input_quant is None:
            return False

        # Confirm weight scheme is supported.
        is_floating_point = (
            weight_quant.type == QuantizationType.FLOAT
            and input_quant.type == QuantizationType.FLOAT
        )
        is_symmetric_weight = weight_quant.symmetric
        is_static_weight = not weight_quant.dynamic
        strategy = getattr(weight_quant.strategy, "value", weight_quant.strategy)
        is_per_tensor_or_channel_weight = strategy in {
            QuantizationStrategy.TENSOR,
            QuantizationStrategy.CHANNEL,
            getattr(QuantizationStrategy, "BLOCK", "block"),
            getattr(QuantizationStrategy.TENSOR, "value", "tensor"),
            getattr(QuantizationStrategy.CHANNEL, "value", "channel"),
            "block",
        }
        if not (
            is_floating_point
            and is_symmetric_weight
            and is_static_weight
            and is_per_tensor_or_channel_weight
        ):
            return False

        # Dynamic quantization is always supported if weights supported.
        if input_quant.dynamic:
            return True

        # Confirm activation scheme is supported.
        is_symmetric_activation = input_quant.symmetric
        is_per_tensor_activation = input_quant.strategy == QuantizationStrategy.TENSOR
        return is_symmetric_activation and is_per_tensor_activation

    def _is_fp8_w8a16(self, weight_quant: BaseModel, input_quant: BaseModel) -> bool:
        # Confirm weights quantized.
        if weight_quant is None:
            return False

        # Confirm we have floating points.
        if weight_quant.type != QuantizationType.FLOAT:
            return False

        # Confirm weight scheme is supported.
        is_symmetric_weight = weight_quant.symmetric
        is_static_weight = not weight_quant.dynamic
        is_per_tensor_or_channel_weight = weight_quant.strategy in [
            QuantizationStrategy.TENSOR,
            QuantizationStrategy.CHANNEL,
        ]
        if not (
            is_symmetric_weight
            and is_static_weight  # noqa: SIM103
            and is_per_tensor_or_channel_weight
        ):
            return False

        # All conditions satisfied.
        return True

    def _is_wNa16_group_channel(
        self, weight_quant: BaseModel, input_quant: BaseModel
    ) -> bool:
        input_quant_none = input_quant is None
        is_symmetric = weight_quant.symmetric
        is_channel_group = (
            weight_quant.strategy == QuantizationStrategy.CHANNEL.value
            or weight_quant.strategy == QuantizationStrategy.GROUP.value
        )
        is_static = not weight_quant.dynamic

        return is_channel_group and input_quant_none and is_symmetric and is_static

    def _is_wint4afp8(
        self,
        weight_quant: BaseModel | None,
        input_quant: BaseModel | None,
        format: str | None = None,
    ) -> bool:
        """Detect W4A8: packed INT4 weights + 8-bit dynamic per-token activations."""
        if weight_quant is None or input_quant is None:
            return False
        quant_format = format if format is not None else self.quant_format
        return (
            quant_format == CompressionFormat.pack_quantized.value
            and weight_quant.num_bits == 4
            and weight_quant.type == QuantizationType.INT
            and weight_quant.symmetric
            and not weight_quant.dynamic
            and input_quant.num_bits == 8
            and input_quant.type in [QuantizationType.FLOAT, QuantizationType.INT]
            and bool(input_quant.dynamic)
        )

    def _get_scheme_from_parts(
        self,
        weight_quant: BaseModel,
        input_quant: BaseModel,
        format: str | None = None,
    ) -> CompressedTensorsScheme:
        quant_format = format if format is not None else self.quant_format

        # Detect If Mixed Precision
        if self._is_wNa16_group_channel(weight_quant, input_quant):
            if (
                quant_format == CompressionFormat.pack_quantized.value
                and weight_quant.num_bits in WNA16_SUPPORTED_BITS
            ):
                from tokenspeed.runtime.layers.quantization.compressed_tensors.schemes import (
                    CompressedTensorsWNA16,
                )

                return CompressedTensorsWNA16(
                    num_bits=weight_quant.num_bits,
                    strategy=weight_quant.strategy,
                    group_size=weight_quant.group_size,
                    actorder=weight_quant.actorder,
                )
            else:
                raise ImportError(
                    "Other method (CompressedTensorsW4A16Sparse24) is not supported now"
                )

        if is_activation_quantization_format(quant_format):
            if self._is_fp8_w8a8(weight_quant, input_quant):
                is_fp8_w8a8_supported = self._check_scheme_supported(
                    CompressedTensorsW8A8Fp8.get_min_capability(), error=False
                )
                if is_fp8_w8a8_supported:
                    return CompressedTensorsW8A8Fp8(
                        strategy=weight_quant.strategy,
                        is_static_input_scheme=(
                            input_quant and not input_quant.dynamic
                        ),
                    )
                else:
                    # note: input_quant will be present for converted models;
                    # will be ignored during inference post loading
                    return CompressedTensorsW8A16Fp8(
                        strategy=weight_quant.strategy,
                        is_static_input_scheme=not input_quant.dynamic,
                    )

            # note: input_quant can be None
            if self._is_fp8_w8a16(weight_quant, input_quant):
                is_static_input_scheme = input_quant and not input_quant.dynamic
                return CompressedTensorsW8A16Fp8(
                    strategy=weight_quant.strategy,
                    is_static_input_scheme=is_static_input_scheme,
                )

        raise NotImplementedError("No compressed-tensors compatible scheme was found.")

    def get_scheme(
        self, layer: torch.nn.Module, layer_name: str | None = None
    ) -> CompressedTensorsScheme | None:
        """
        compressed-tensors supports non uniform in the following way:

        targets of config_groups: There can be N config_groups which each
            have a quantization scheme. Each config_group has a list of targets
            which can be a full layer_name, a regex for a layer_name, or
            an nn.Module name.

        Detect whether a layer_name is found in any target and
        use the quantization scheme corresponding to the matched target
        to select the CompressedTensorsScheme used for infernece.
        """

        # Find the "target" in the compressed-tensors config
        # that our layer conforms to.
        # so we do not have to re-write these functions
        # need to make accelerate optional in ct to do this

        # Will be empty for models with only sparsity
        weight_quant = input_quant = None
        scheme_format = None
        if self.target_scheme_map:
            matched_target = find_matched_target(
                layer_name=layer_name,
                module=layer,
                targets=self.target_scheme_map.keys(),
                fused_mapping=self.packed_modules_mapping,
            )

            scheme_dict = self.target_scheme_map[matched_target]
            weight_quant = scheme_dict.get("weights")
            input_quant = scheme_dict.get("input_activations")
            scheme_format = scheme_dict.get("format")

        # Find the sparsity scheme of the layer
        # assume that fused layers inerhit first component's sparsity scheme
        sparsity_targets = self.sparsity_scheme_map.keys() - set(
            self.sparsity_ignore_list
        )
        sparsity_scheme: SparsityCompressionConfig | None = None
        with suppress(ValueError):
            matched_target = find_matched_target(
                layer_name=layer_name,
                module=layer,
                targets=sparsity_targets,
                fused_mapping=self.packed_modules_mapping,
            )
            sparsity_scheme = self.sparsity_scheme_map[matched_target]

        if self.supports_cutlass_24(
            weight_quant=weight_quant,
            input_quant=input_quant,
            sparsity_scheme=sparsity_scheme,
        ):
            raise ImportError("CompressedTensors24 is not supported now")
        elif weight_quant is None:
            logger.warning(
                "Acceleration for non-quantized schemes is "
                "not supported by Compressed Tensors. "
                "Falling back to UnquantizedLinearMethod"
            )
            return None

        else:
            # Find the quant_scheme
            scheme = self._get_scheme_from_parts(  # type: ignore
                weight_quant=weight_quant,
                input_quant=input_quant,
                format=scheme_format,
            )

        # Raise error if device does not support the scheme
        # (e.g. fp8 needs ada lovelace)
        self._check_scheme_supported(scheme.get_min_capability())
        logger.debug("Using scheme: %s for %s", scheme.__class__.__name__, layer_name)
        return scheme

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> LinearMethodBase:
        """Select a linear method for ``prefix``.

        Mixed-precision compressed-tensors checkpoints (GLM-5.3 W4A8) keep
        attention / shared-expert projections as block-FP8 and routed experts
        as packed INT4. TokenSpeed has no CompressedTensorsW8A8Fp8 scheme, so
        block-FP8 linears reuse :class:`Fp8LinearMethod`. Packed INT4 dense
        linears keep :class:`CompressedTensorsLinearMethod` (WNA16).
        """
        from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
        from tokenspeed.runtime.layers.quantization.utils import (
            should_ignore_quant_layer,
        )

        def _unquantized():
            from tokenspeed.runtime.layers.dense import UnquantizedLinearMethod

            return UnquantizedLinearMethod()

        if should_ignore_quant_layer(
            prefix, self.ignored_layers, self.packed_modules_mapping
        ):
            return _unquantized()

        try:
            matched_target = find_matched_target(
                layer_name=prefix,
                module=layer,
                targets=self.target_scheme_map.keys(),
                fused_mapping=self.packed_modules_mapping,
            )
        except ValueError:
            return _unquantized()

        scheme_dict = self.target_scheme_map[matched_target]
        weight_quant = scheme_dict.get("weights")
        input_quant = scheme_dict.get("input_activations")
        scheme_format = scheme_dict.get("format")
        if weight_quant is None:
            return _unquantized()

        if self._is_wint4afp8(weight_quant, input_quant, scheme_format):
            raise NotImplementedError(
                "Dense W4A8 compressed-tensors linears are not supported; "
                f"{prefix!r} matched packed INT4 + FP8 activations. Routed "
                "experts use moe_weight_dtype() instead."
            )

        if self._is_fp8_w8a8(weight_quant, input_quant) or self._is_fp8_w8a16(
            weight_quant, input_quant
        ):
            from tokenspeed.runtime.layers.dense import Fp8LinearMethod

            block_structure = getattr(weight_quant, "block_structure", None)
            activation_scheme = (
                "dynamic" if (input_quant is None or input_quant.dynamic) else "static"
            )
            if block_structure is not None:
                activation_scheme = "dynamic"
            fp8_config = Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme=activation_scheme,
                ignored_layers=self.ignored_layers,
                weight_block_size=(
                    list(block_structure) if block_structure is not None else None
                ),
            )
            return Fp8LinearMethod(fp8_config)

        layer.scheme = self._get_scheme_from_parts(
            weight_quant=weight_quant,
            input_quant=input_quant,
            format=scheme_format,
        )
        self._check_scheme_supported(layer.scheme.get_min_capability())
        return CompressedTensorsLinearMethod(self)

    def get_cache_scale(self, name: str) -> str | None:
        """
        Check whether the param name matches the format for k/v cache scales
        in compressed-tensors. If this is the case, return its equivalent
        param name expected by TokenSpeed

        :param name: param name
        :return: matching param name for KV cache scale in TokenSpeed
        """
        if name.endswith(".output_scale") and ".k_proj" in name:
            return name.replace(".k_proj.output_scale", ".attn.k_scale")
        if name.endswith(".output_scale") and ".v_proj" in name:
            return name.replace(".v_proj.output_scale", ".attn.v_scale")
        # If no matches, return None
        return None

    @staticmethod
    def supports_cutlass_24(
        weight_quant: QuantizationArgs | None,
        input_quant: QuantizationArgs | None,
        sparsity_scheme: SparsityCompressionConfig | None = None,
    ) -> bool:
        """
        Check if the layer is supported by the Cutlass 2:4 Kernel
        Conditions:
            - Overarching condition: Sparsity Structure is 2:4
            - Unquantized cases are supported
            - Weight only quantization is not-supported
            - Supported weight quantization strategies are TENSOR and CHANNEL
            - Supported input quantization strategies are TENSOR and TOKEN
            - Only 8 bit quantization is supported

        :return: True if the layer is supported by the Cutlass 2:4 Kernel
            False otherwise
        """
        if sparsity_scheme is None:
            return False

        is_valid_sparsity_structure: bool = (
            sparsity_scheme.sparsity_structure == SparsityStructure.TWO_FOUR.value
        )

        valid_compressors = {
            CompressionFormat.dense.value,
            CompressionFormat.sparse_24_bitmask.value,
        }

        is_valid_sparsity = (
            is_valid_sparsity_structure and sparsity_scheme.format in valid_compressors
        )

        if not is_valid_sparsity:
            return False

        # Unquantized cases are supported
        if weight_quant is None and input_quant is None:
            return True

        # Weight only quantization is not-supported
        if weight_quant is not None and input_quant is None:
            return False

        supported_weight_quant_strategies = [
            QuantizationStrategy.TENSOR.value,
            QuantizationStrategy.CHANNEL.value,
        ]

        if weight_quant is None or input_quant is None:
            raise RuntimeError("Quantization args should be populated at this point.")
        if weight_quant.strategy not in supported_weight_quant_strategies:
            return False

        supported_input_quant_strategies = [
            QuantizationStrategy.TENSOR.value,
            QuantizationStrategy.TOKEN.value,
        ]

        if input_quant.strategy not in supported_input_quant_strategies:
            return False

        return weight_quant.num_bits == input_quant.num_bits == 8


class CompressedTensorsLinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: CompressedTensorsConfig):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """
        Use the CompressedTensorsScheme associated with each layer to create
        the necessary parameters for the layer. See LinearMethodBase for param
        details
        """
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.scheme.create_weights(
            layer=layer,
            input_size=input_size,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=weight_loader,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ):
        """
        Use the output of create_weights and the CompressedTensorsScheme
        associated with the layer to apply the forward pass with the
        layer input.  See LinearMethodBase for param details

        """

        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, x, bias=bias)
