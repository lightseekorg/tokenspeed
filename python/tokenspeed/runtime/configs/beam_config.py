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

"""Checkpoint-owned Beam configuration."""

from __future__ import annotations

import copy
import math
from typing import Any

from transformers.configuration_utils import PretrainedConfig

BEAM_REQUIRED_CONFIG_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_act",
    "max_position_embeddings",
    "rms_norm_eps",
    "tie_word_embeddings",
    "rope_parameters",
    "attention_bias",
    "sliding_window",
    "layer_types",
    "mlp_layer_types",
    "moe_intermediate_size",
    "num_experts",
    "num_experts_per_tok",
    "norm_topk_prob",
    "n_group",
    "topk_group",
    "routed_scaling_factor",
    "final_logit_softcapping",
)
# Fields a Beam config must not set: architecture values Beam fixes or derives
# (a config that sets one is refused, not interpreted as an override), and the
# context aliases runtime context discovery checks before the canonical length
# that sizes Beam's rotary cache.
BEAM_OBSOLETE_CONFIG_FIELDS = frozenset(
    {
        "rope_theta",
        "no_rope_layers",
        "ssmax",
        "ssmax_offset",
        "ssmax_divide_by_log_offset",
        "ssmax_scale_mode",
        "mlp_only_layers",
        "decoder_sparse_step",
        "scoring_func",
        "topk_method",
        "enable_sandwich_norm",
        "self_attn_depth_scaled_residual",
        "mlp_depth_scaled_residual",
        "use_paramfree_qk_norm",
        "router_dtype",
        "attention_output_gate",
        "attention_output_gate_bias",
        "attention_output_gate_floor",
        "attention_output_gate_temperature",
        "num_local_experts",
        "lm_head_dtype",
        "residual_in_fp32",
        "num_nextn_predict_layers",
        "router_aux_loss_coef",
        "moe_precision_policy",
        "moe_output_scale",
        "raw_lm_head_dtype",
        "raw_residual_in_fp32",
        "resolved_lm_head_dtype",
        "lm_head_dtype_resolved",
        "resolved_residual_in_fp32",
        "residual_in_fp32_resolved",
        "max_sequence_length",
        "seq_length",
        "max_seq_len",
        "model_max_length",
    }
)


class BeamConfig(PretrainedConfig):
    """Strict Beam schema with read-only adapters for shared runtime layers.

    Removed numerical switches are rejected, not interpreted as overrides.
    Runtime adapter properties never become serialized checkpoint fields.
    """

    model_type = "beam"
    constrained_decoding = "always"
    keys_to_ignore_at_inference = ["past_key_values"]
    has_no_defaults_at_init = True

    # The shared DeepSeek-V3 MoE gate keeps its FP32 correction bias.
    topk_method = "noaux_tc"

    def __init__(self, **kwargs: Any) -> None:
        values = copy.deepcopy(kwargs)
        if values.pop("model_type", self.model_type) != self.model_type:
            raise ValueError("model_type must be 'beam'")
        if values.get("architectures") != ["BeamForCausalLM"]:
            raise ValueError("architectures must be ['BeamForCausalLM']")
        obsolete = sorted(BEAM_OBSOLETE_CONFIG_FIELDS.intersection(values))
        if obsolete:
            raise ValueError(f"obsolete or unsupported Beam config fields: {obsolete}")
        missing = sorted(set(BEAM_REQUIRED_CONFIG_FIELDS) - values.keys())
        if missing:
            raise ValueError(f"missing required Beam config fields: {missing}")
        architecture = {key: values.pop(key) for key in BEAM_REQUIRED_CONFIG_FIELDS}
        super().__init__(
            tie_word_embeddings=architecture.pop("tie_word_embeddings"), **values
        )
        for key, value in architecture.items():
            setattr(self, key, value)
        self._validate_architecture()

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any):
        if config_dict.get("model_type") != cls.model_type:
            raise ValueError("model_type must be 'beam'")
        if BEAM_OBSOLETE_CONFIG_FIELDS.intersection(kwargs):
            raise ValueError("Beam precision cannot be overridden by legacy fields")
        loaded = super().from_dict(config_dict, **kwargs)
        config = loaded[0] if isinstance(loaded, tuple) else loaded
        config._validate_architecture()
        return loaded

    def validate(self) -> None:
        # Transformers' generic validator assumes every nested RoPE entry is
        # a dictionary. Beam intentionally represents full-attention NoPE by
        # null; validate the canonical schema without rewriting that metadata.
        self._validate_architecture()

    def update(self, config_dict: dict[str, Any]) -> None:
        if BEAM_OBSOLETE_CONFIG_FIELDS.intersection(config_dict):
            raise ValueError("Beam precision cannot be overridden by legacy fields")
        super().update(config_dict)
        self._validate_architecture()

    @property
    def rope_theta(self) -> float:
        return self.rope_parameters["sliding_attention"]["rope_theta"]

    @property
    def no_rope_layers(self) -> list[int]:
        return [int(kind == "sliding_attention") for kind in self.layer_types]

    @property
    def cache_layer_types(self) -> list[str]:
        """Per-layer KV cache group labels, read by the cache recipes only.

        The hybrid slab layout gives every page as many layer slots as the
        largest cache group has layers. With one sliding group (43 layers)
        a full-attention page would carry 43 slots for 9 layers, so a long
        context would pay 43 layers of KV per token. Splitting the sliding
        layers round-robin into sub-groups no larger than the full group
        (43 sliding + 9 full -> 5 sub-groups of at most 9) keeps every slot
        of a full-attention page in use. All sub-groups share the one
        window, so what each layer keeps and evicts is unchanged. A stack
        whose sliding layers already fit one group keeps its labels.
        """
        sliding = self.layer_types.count("sliding_attention")
        full = len(self.layer_types) - sliding
        subgroups = -(-sliding // full) if full else 1
        if subgroups <= 1:
            return list(self.layer_types)
        labels: list[str] = []
        rank = 0
        for kind in self.layer_types:
            if kind == "sliding_attention":
                labels.append(f"sliding_attention_{rank % subgroups}")
                rank += 1
            else:
                labels.append(kind)
        return labels

    @property
    def mlp_only_layers(self) -> list[int]:
        return [i for i, kind in enumerate(self.mlp_layer_types) if kind == "dense"]

    def _validate_architecture(self) -> None:
        if self.model_type != "beam" or self.architectures != ["BeamForCausalLM"]:
            raise ValueError("Beam requires its canonical model identity")
        if BEAM_OBSOLETE_CONFIG_FIELDS.intersection(self.__dict__):
            raise ValueError("obsolete or unsupported Beam config fields")
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
            "moe_intermediate_size",
            "num_experts",
            "num_experts_per_tok",
            "n_group",
            "topk_group",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.head_dim % 2:
            raise ValueError("Beam requires an even head_dim")
        if self.hidden_act != "silu" or self.tie_word_embeddings or self.attention_bias:
            raise ValueError(
                "Beam requires silu, untied embeddings and no attention bias"
            )
        if getattr(self, "attention_dropout", 0.0) != 0.0:
            raise ValueError("Beam serving requires attention_dropout=0")
        if not isinstance(self.norm_topk_prob, bool):
            raise TypeError("norm_topk_prob must be a boolean")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.num_experts % self.n_group or self.num_experts // self.n_group < 2:
            raise ValueError(
                "expert groups must divide experts and contain at least two"
            )
        if self.topk_group > self.n_group:
            raise ValueError("topk_group must not exceed n_group")
        if self.num_experts_per_tok > self.topk_group * (
            self.num_experts // self.n_group
        ):
            raise ValueError("num_experts_per_tok exceeds the selected expert groups")
        for name, allowed in (
            ("layer_types", {"sliding_attention", "full_attention"}),
            ("mlp_layer_types", {"dense", "sparse"}),
        ):
            value = getattr(self, name)
            if not isinstance(value, list) or len(value) != self.num_hidden_layers:
                raise ValueError(f"{name} must contain one entry per layer")
            if any(kind not in allowed for kind in value):
                raise ValueError(f"unsupported {name}")
        if "sliding_attention" in self.layer_types and (
            isinstance(self.sliding_window, bool)
            or not isinstance(self.sliding_window, int)
            or self.sliding_window < 1
        ):
            raise ValueError("sliding_window must be a positive integer")
        for name in ("rms_norm_eps", "routed_scaling_factor"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        cap = self.final_logit_softcapping
        if cap is not None and (
            isinstance(cap, bool)
            or not isinstance(cap, (int, float))
            or not math.isfinite(cap)
            or cap <= 0
        ):
            raise ValueError(
                "final_logit_softcapping must be None or finite and positive"
            )
        rope = self.rope_parameters
        if not isinstance(rope, dict) or set(rope) != {
            "sliding_attention",
            "full_attention",
        }:
            raise ValueError(
                "rope_parameters requires sliding_attention and full_attention"
            )
        if rope["full_attention"] is not None:
            raise ValueError("full_attention must use NoPE")
        sliding = rope["sliding_attention"]
        if not isinstance(sliding, dict) or sliding.get("rope_type") not in {
            "default",
            "linear",
        }:
            raise ValueError("sliding RoPE requires default or linear rope_type")
        theta = sliding.get("rope_theta")
        if (
            isinstance(theta, bool)
            or not isinstance(theta, (int, float))
            or not math.isfinite(theta)
            or theta <= 0
        ):
            raise ValueError("sliding rope_theta must be finite and positive")
        if sliding["rope_type"] == "linear":
            factor = sliding.get("factor")
            if (
                isinstance(factor, bool)
                or not isinstance(factor, (int, float))
                or not math.isfinite(factor)
                or factor <= 0
            ):
                raise ValueError("linear RoPE factor must be finite and positive")
        elif "factor" in sliding:
            raise ValueError("default RoPE does not accept a scaling factor")

    def validate_parallelism(self, tensor_parallel_size: int, ep_size: int) -> None:
        """Reject unsupported parallel layouts before allocating parameters."""
        # Beam accepts expert parallelism over whole experts; its MoE layers
        # decide which expert paths can run it.
        if (
            isinstance(ep_size, bool)
            or not isinstance(ep_size, int)
            or ep_size < 1
            or self.num_experts % ep_size
        ):
            raise ValueError(
                f"ep_size must be a positive integer dividing num_experts="
                f"{self.num_experts}, got {ep_size!r}"
            )
        if (
            isinstance(tensor_parallel_size, bool)
            or not isinstance(tensor_parallel_size, int)
            or tensor_parallel_size < 1
        ):
            raise ValueError("tensor_parallel_size must be a positive integer")
        for field in ("num_attention_heads", "num_key_value_heads"):
            heads = getattr(self, field)
            if heads % tensor_parallel_size != 0:
                raise ValueError(
                    f"tensor_parallel_size={tensor_parallel_size} must divide "
                    f"{field}={heads}"
                )


__all__ = ["BEAM_REQUIRED_CONFIG_FIELDS", "BEAM_OBSOLETE_CONFIG_FIELDS", "BeamConfig"]
