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

"""DeepSeek V4 model configuration."""

from collections.abc import Mapping, Sequence
from typing import Any

from tokenspeed.runtime.configs.base_config import BaseConfig, TextConfigBase

DEEPSEEK_V4_LAYER_TYPES = (
    "sliding_attention",
    "compressed_sparse_attention",
    "heavily_compressed_attention",
)

_COMPRESS_RATIO_TO_LAYER_TYPE = {
    0: "sliding_attention",
    1: "sliding_attention",
    4: "compressed_sparse_attention",
    128: "heavily_compressed_attention",
}

DEEPSEEK_V4_MLP_LAYER_TYPES = ("hash_moe", "moe")
_DEEPSEEK_V4_COMPRESS_RATIOS_BY_LAYER_TYPE = {
    "compressed_sparse_attention": 4,
    "heavily_compressed_attention": 128,
}


def get_deepseek_v4_compress_ratio(config: BaseConfig, layer_index: int) -> int:
    """Resolve one layer's compression ratio from the canonical V4 schedule.

    Draft layers (MTP / DSpark) address global ``layer_index`` values at or
    past ``num_hidden_layers``; they reuse the sliding-window schedule.
    """
    layer_types = getattr(config, "layer_types", None)
    if not isinstance(layer_types, Sequence) or isinstance(layer_types, str):
        raise TypeError("DeepSeek V4 `layer_types` must be a sequence.")
    if layer_index < 0:
        raise IndexError(
            "DeepSeek V4 layer index out of range: "
            f"index={layer_index}, layers={len(layer_types)}"
        )
    if layer_index >= len(layer_types):
        # Draft layers (MTP / DSpark) run the sliding-window schedule.
        return 1

    layer_type = layer_types[layer_index]
    if layer_type == "sliding_attention":
        return 1

    compress_rates = getattr(config, "compress_rates", None)
    if not isinstance(compress_rates, Mapping):
        raise TypeError("DeepSeek V4 `compress_rates` must be a mapping.")
    if layer_type not in _DEEPSEEK_V4_COMPRESS_RATIOS_BY_LAYER_TYPE:
        raise ValueError(f"Unknown DeepSeek V4 attention layer type: {layer_type!r}")
    if layer_type not in compress_rates:
        raise ValueError(
            f"DeepSeek V4 layer type {layer_type!r} has no compression rate."
        )

    ratio = int(compress_rates[layer_type])
    expected_ratio = _DEEPSEEK_V4_COMPRESS_RATIOS_BY_LAYER_TYPE[layer_type]
    if ratio != expected_ratio:
        raise ValueError(
            f"Unsupported DeepSeek V4 compress_ratio={ratio}; "
            f"expected {expected_ratio} for {layer_type!r}."
        )
    return ratio


class DeepseekV4Config(TextConfigBase):
    r"""
    DeepSeek-V4 checkpoint configuration.

    V4 differs from :class:`DeepseekV3Config` in its attention schedule: each
    layer runs a sliding-window branch plus one of two compressed branches
    (``compressed_sparse_attention`` at ratio 4 or ``heavily_compressed_attention``
    at ratio 128), and the MLP schedule mixes frozen ``hash_moe`` bootstrap layers
    with standard routed ``moe``. RoPE is keyed by *rope-type* labels
    (``main`` / ``compress``) rather than by ``layer_types``.
    """

    model_type = "deepseek_v4"
    keys_to_ignore_at_inference = ["past_key_values"]

    # `num_local_experts` is the standard MoE attribute name; `intermediate_size`
    # is what generic MLP code reads for the shared-expert width, but V4 only
    # ships `moe_intermediate_size`, so route the read through.
    attribute_map = {
        "num_local_experts": "n_routed_experts",
        "intermediate_size": "moe_intermediate_size",
    }

    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    # V4 ships EP only (no `base_model_tp_plan`): MoE parallelism routes on the
    # gate, runs routed experts as a grouped-GEMM sharded along the expert axis,
    # and wraps the experts module with `moe_tp_experts`. Main attention stays
    # replicated (shared-KV MQA), as does the shared MLP.
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.self_attn.compressor.indexer.q_b_proj": "colwise",
        "layers.*.self_attn.compressor.indexer.scorer.weights_proj": "colwise",
        "layers.*.self_attn.compressor.indexer.scorer": "all_reduce",
    }

    vocab_size: int = 129280
    hidden_size: int = 4096
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    q_lora_rank: int = 1024
    num_experts_per_tok: int = 6
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    scoring_func: str = "sqrtsoftplus"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    max_position_embeddings: int = 1048576
    rope_theta: float | int = 10000.0

    layer_types: list[str] | None = None
    compress_rates: dict | None = None
    compress_rope_theta: float | int = 160000.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1.0e-6
    mlp_layer_types: list[str] | None = None
    swiglu_limit: float = 10.0
    sliding_window: int = 128
    o_groups: int = 8
    o_lora_rank: int = 1024
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    num_nextn_predict_layers: int = 1

    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0

    hidden_act: str = "silu"
    initializer_range: float = 0.02
    rms_norm_eps: float = 1.0e-6
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    tie_word_embeddings: bool = False
    rope_parameters: dict | None = None
    partial_rotary_factor: float | None = None
    attention_bias: bool = False
    mlp_bias: bool = False
    attention_dropout: float = 0.0

    # Class-level defaults (not dataclass fields).
    default_partial_rotary_factor = 64 / 512  # qk_rope_head_dim (64) / head_dim (512)
    default_compress_rates = _DEEPSEEK_V4_COMPRESS_RATIOS_BY_LAYER_TYPE
    default_num_hash_layers = 3
    _rope_type_labels = ("main", "compress")

    def standardize_rope_params(self) -> None:
        """Normalize flat or rope-type-keyed RoPE parameters for V4.

        Legacy checkpoints provide one flat scaling dictionary, which applies
        only to compressed attention. Native V4 configs instead key parameters
        by the ``main`` and ``compress`` rope-type labels. Normalize both forms
        here so :class:`BaseConfig` can run the V4 validator exactly once after
        the final structure has been built.
        """
        rope_parameters = self.rope_parameters or {}
        if not isinstance(rope_parameters, dict):
            raise TypeError("`rope_parameters` must be a dictionary.")

        has_main = "main" in rope_parameters
        has_compress = "compress" in rope_parameters
        if has_main != has_compress:
            raise ValueError(
                "`rope_parameters` must contain both `main` and `compress`, "
                "or use the legacy flat format."
            )

        if has_main:
            if not isinstance(rope_parameters["main"], dict) or not isinstance(
                rope_parameters["compress"], dict
            ):
                raise TypeError(
                    "`rope_parameters.main` and `rope_parameters.compress` "
                    "must be dictionaries."
                )
            # Do not mutate dictionaries owned by the checkpoint loader/caller.
            main = dict(rope_parameters["main"])
            compress = dict(rope_parameters["compress"])
        else:
            # Flat scaling parameters apply only to compressed attention.
            main = {
                "rope_type": "default",
                "rope_theta": self.rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
            }
            compress = {
                **rope_parameters,
                "rope_theta": self.compress_rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
            }

        for parameters, default_theta in (
            (main, self.rope_theta),
            (compress, self.compress_rope_theta),
        ):
            parameters.setdefault("rope_type", parameters.get("type", "default"))
            parameters.setdefault("rope_theta", default_theta)
            parameters.setdefault("partial_rotary_factor", self.partial_rotary_factor)
            if parameters["rope_type"] in {
                "deepseek_yarn",
                "llama3",
                "longrope",
                "yarn",
            }:
                # Preserve the missing-field convention for YaRN and do not
                # invent the pre-extension training length required by Llama
                # 3 or LongRoPE. The latter two must fail validation when it
                # was not explicitly supplied.
                original_max = getattr(self, "original_max_position_embeddings", None)
                if original_max is not None:
                    parameters.setdefault(
                        "original_max_position_embeddings", original_max
                    )

        if compress["rope_type"] in {"deepseek_yarn", "yarn"}:
            compress.setdefault("attn_factor", 1.0)

        self.rope_parameters = {"main": main, "compress": compress}

    def validate_rope(self) -> None:
        """Validate the rope-type-keyed ``rope_parameters`` sub-dicts.

        V4 keys ``rope_parameters`` by rope-type label (``main`` / ``compress``)
        rather than by ``layer_types``, so the base ``validate_rope`` (which
        treats a non-``layer_types``-subset dict as a single global parameter
        set) would misread the nesting. Iterate the labels directly instead.
        """
        rope_parameters_dict = getattr(self, "rope_parameters", None) or {}
        ignore_keys = self.ignore_keys_at_rope_validation
        for rope_type_label in self._rope_type_labels:
            rope_parameters = rope_parameters_dict.get(rope_type_label)
            if not isinstance(rope_parameters, dict):
                continue
            rope_type = rope_parameters.get(
                "rope_type", rope_parameters.get("type", "default")
            )
            rope_parameters["rope_type"] = rope_type
            validator_type = "yarn" if rope_type == "deepseek_yarn" else rope_type
            validator = getattr(
                self, f"_validate_{validator_type}_rope_parameters", None
            )
            if validator is None:
                raise ValueError(f"Unsupported DeepSeek V4 rope type: {rope_type!r}")
            validator(rope_parameters, ignore_keys=ignore_keys)

    def validate_layer_type(self) -> None:
        """Narrow the global layer-type check to the V4 attention/MLP types."""
        if self.num_hidden_layers is None:
            return
        for name, types, allowed in (
            ("layer_types", self.layer_types, DEEPSEEK_V4_LAYER_TYPES),
            ("mlp_layer_types", self.mlp_layer_types, DEEPSEEK_V4_MLP_LAYER_TYPES),
        ):
            if types is None:
                continue
            if len(types) != self.num_hidden_layers:
                raise ValueError(
                    f"`num_hidden_layers` ({self.num_hidden_layers}) must equal "
                    f"`len({name})` ({len(types)})."
                )
            bad = [t for t in types if t not in allowed]
            if bad:
                raise ValueError(
                    f"`{name}` entries must be one of {allowed} for DeepSeek-V4; "
                    f"got {bad}."
                )

    def __post_init__(self, **kwargs: Any) -> None:
        # Strip legacy V4 kwargs (V3-flavoured names that older checkpoints
        # still ship) before the parent sees them, then fold each into the
        # modern field below.
        legacy_compress_ratios = kwargs.pop("compress_ratios", None)
        legacy_compress_rate_csa = kwargs.pop("compress_rate_csa", None)
        legacy_compress_rate_hca = kwargs.pop("compress_rate_hca", None)
        legacy_num_hash_layers = kwargs.pop("num_hash_layers", None)
        legacy_qk_rope_head_dim = kwargs.pop("qk_rope_head_dim", None)

        # V4 RoPE standardization runs inside the parent finalizer, so resolve
        # the rotary width before delegating to it.
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.partial_rotary_factor is None:
            self.partial_rotary_factor = (
                legacy_qk_rope_head_dim / self.head_dim
                if legacy_qk_rope_head_dim is not None
                else self.default_partial_rotary_factor
            )
        self.qk_rope_head_dim = int(self.head_dim * self.partial_rotary_factor)

        n = self.num_hidden_layers
        # `compress_rates`: dict, default per attention type. Legacy scalar
        # overrides fold in.
        self.compress_rates = dict(
            self.compress_rates
            if self.compress_rates is not None
            else self.default_compress_rates
        )
        if legacy_compress_rate_csa is not None:
            self.compress_rates["compressed_sparse_attention"] = (
                legacy_compress_rate_csa
            )
        if legacy_compress_rate_hca is not None:
            self.compress_rates["heavily_compressed_attention"] = (
                legacy_compress_rate_hca
            )

        # `layer_types`: explicit > legacy `compress_ratios` per-layer ints
        # (0/1/4/128) > V4-Pro default (2x HCA bootstrap + CSA/HCA interleave).
        if self.layer_types is None and legacy_compress_ratios is not None:
            invalid_ratios = [
                ratio
                for ratio in legacy_compress_ratios
                if ratio not in _COMPRESS_RATIO_TO_LAYER_TYPE
            ]
            if invalid_ratios:
                raise ValueError(
                    "Legacy DeepSeek V4 `compress_ratios` entries must be "
                    f"0, 1, 4, or 128; got {invalid_ratios}."
                )
            self.layer_types = [
                _COMPRESS_RATIO_TO_LAYER_TYPE[r] for r in legacy_compress_ratios
            ]
        if self.layer_types is None:
            interleave = [
                (
                    "compressed_sparse_attention"
                    if i % 2
                    else "heavily_compressed_attention"
                )
                for i in range(max(n - 2, 0))
            ]
            self.layer_types = ["heavily_compressed_attention"] * min(n, 2) + interleave
        self.layer_types = list(self.layer_types[:n])

        # `mlp_layer_types`: first `num_hash_layers` hash_moe, rest moe.
        if self.mlp_layer_types is None:
            n_hash = (
                legacy_num_hash_layers
                if legacy_num_hash_layers is not None
                else self.default_num_hash_layers
            )
            self.mlp_layer_types = ["hash_moe"] * min(n, n_hash) + ["moe"] * max(
                0, n - n_hash
            )
        self.mlp_layer_types = list(self.mlp_layer_types[:n])

        super().__post_init__(**kwargs)

        for layer_index in range(n):
            get_deepseek_v4_compress_ratio(self, layer_index)


__all__ = ["DeepseekV4Config", "get_deepseek_v4_compress_ratio"]
