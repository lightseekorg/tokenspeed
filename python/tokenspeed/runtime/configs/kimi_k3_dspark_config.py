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

"""Config for the MLA-native Kimi-K3 DSpark draft model.

The reference checkpoint is `Inferact/Kimi-K3-DSpark
<https://huggingface.co/Inferact/Kimi-K3-DSpark>`_: five dense MLA layers that
draft seven tokens per block from target hidden states tapped at five K3 layers.

The draft's attention is *not* K3's. K3 runs NoPE MLA with an output gate; this
draft is a DeepSeek-V3-style RoPE MLA with YaRN scaling. Only the 576-element
latent KV shape is shared, which is what lets the draft's pages sit alongside
the target's rather than in a second format.
"""

from __future__ import annotations

from typing import Any

from tokenspeed.runtime.configs.base_config import TextConfigBase

# The checkpoint ships an embed_tokens copy of the frozen target embedding and
# a training-only confidence head. Neither is instantiated at serving time.
K3_DSPARK_SKIPPED_WEIGHT_PREFIXES = ("embed_tokens.", "lm_head.", "confidence_head.")

SUPPORTED_MARKOV_HEAD_TYPES = ("vanilla",)

SUPPORTED_AUX_HIDDEN_STREAMS = ("prefix", "attn_res")


class KimiK3DSparkConfig(TextConfigBase):
    """Draft-side config for `K3DSparkModel`."""

    model_type = "k3_dspark"

    hidden_size: int = 7168
    intermediate_size: int = 14336
    num_hidden_layers: int = 5
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    mla_use_nope: bool = False
    mla_use_output_gate: bool = False
    vocab_size: int = 163840
    draft_vocab_size: int | None = None
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"
    max_position_embeddings: int = 1048576
    rope_theta: float = 50000.0
    rope_parameters: dict[str, Any] | None = None
    num_target_layers: int = 5
    target_hidden_size: int = 7168
    target_num_hidden_layers: int = 93
    target_layer_ids: list[int] | None = None
    fc_norm: bool = False
    aux_hidden_stream: str = "prefix"
    mask_token_id: int | None = None
    markov_rank: int = 256
    markov_head_type: str = "vanilla"
    enable_confidence_head: bool = True
    confidence_head_with_markov: bool = True

    def __post_init__(self, **kwargs) -> None:
        # MLA
        self.q_lora_rank = int(self.q_lora_rank)
        self.kv_lora_rank = int(self.kv_lora_rank)
        self.qk_nope_head_dim = int(self.qk_nope_head_dim)
        self.qk_rope_head_dim = int(self.qk_rope_head_dim)
        self.v_head_dim = int(self.v_head_dim)
        self.mla_use_nope = bool(self.mla_use_nope)
        self.mla_use_output_gate = bool(self.mla_use_output_gate)

        self.draft_vocab_size = int(
            self.draft_vocab_size
            if self.draft_vocab_size is not None
            else self.vocab_size
        )
        self.rope_theta = float(self.rope_theta)
        self.num_target_layers = int(self.num_target_layers)
        self.target_hidden_size = int(self.target_hidden_size)
        self.target_num_hidden_layers = int(self.target_num_hidden_layers)
        self.target_layer_ids = [int(x) for x in (self.target_layer_ids or [])]
        self.fc_norm = bool(self.fc_norm)
        self.aux_hidden_stream = str(self.aux_hidden_stream).lower()
        self.mask_token_id = (
            self.mask_token_id
            if self.mask_token_id is None
            else int(self.mask_token_id)
        )
        self.markov_rank = int(self.markov_rank)
        self.markov_head_type = str(self.markov_head_type).lower()
        self.enable_confidence_head = bool(self.enable_confidence_head)
        self.confidence_head_with_markov = bool(self.confidence_head_with_markov)
        if not self.architectures:
            self.architectures = ["K3DSparkModel"]

        self.vocab_size = int(self.vocab_size)
        self.hidden_size = int(self.hidden_size)
        self.intermediate_size = int(self.intermediate_size)
        self.num_hidden_layers = int(self.num_hidden_layers)
        self.num_attention_heads = int(self.num_attention_heads)
        self.num_key_value_heads = int(self.num_key_value_heads)
        self.hidden_act = str(self.hidden_act)
        self.max_position_embeddings = int(self.max_position_embeddings)
        self.rms_norm_eps = float(self.rms_norm_eps)
        self.head_dim = self.qk_rope_head_dim

        super().__post_init__(**kwargs)

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def kv_latent_dim(self) -> int:
        """Width of one cached latent row: [c_KV_norm | k_PE_RoPE]."""
        return self.kv_lora_rank + self.qk_rope_head_dim

    def rope_scaling_dict(self) -> dict[str, Any] | None:
        """YaRN parameters in the shape ``get_rope`` expects, or None."""
        params = self.rope_parameters or {}
        if str(params.get("rope_type", "default")).lower() not in (
            "yarn",
            "deepseek_yarn",
        ):
            return None
        scaling = {
            key: params[key]
            for key in (
                "factor",
                "original_max_position_embeddings",
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
            )
            if key in params
        }
        scaling["rope_type"] = "deepseek_yarn"
        return scaling


def k3_dspark_inactive_features(config: KimiK3DSparkConfig) -> list[str]:
    """Optional checkpoint features this build loads but does not act on.

    Issue #879 asks for these to be stated rather than silently dropped: a user
    who trained a confidence head is entitled to know the scheduler is ignoring
    it and every request is verifying the full block.
    """
    inactive = []
    if config.enable_confidence_head:
        inactive.append(
            "confidence_head: present in the checkpoint but unused. Verify is "
            "static (the full block is verified every step); confidence-scheduled "
            "ragged verify is not implemented."
        )
    return inactive


def validate_k3_dspark_config(config: KimiK3DSparkConfig, target_config=None) -> None:
    """Fail a mis-specified draft at startup rather than at the first token.

    Every check here corresponds to a silent numerical corruption: a wrong tap
    count feeds `context_proj` a mis-shaped concat, a wrong latent width writes
    the KV cache with a stride that decodes to noise, and a missing mask token
    makes the draft attend to whatever id happens to be zero.
    """
    if config.num_hidden_layers <= 0:
        raise ValueError(
            f"K3 DSpark draft needs at least one layer; got {config.num_hidden_layers}."
        )
    if len(config.target_layer_ids) != config.num_target_layers:
        raise ValueError(
            f"K3 DSpark target_layer_ids has {len(config.target_layer_ids)} entries "
            f"but num_target_layers={config.num_target_layers}. context_proj expects "
            f"exactly num_target_layers x hidden_size inputs."
        )
    if not config.target_layer_ids:
        raise ValueError("K3 DSpark draft requires a non-empty target_layer_ids.")
    if sorted(config.target_layer_ids) != config.target_layer_ids:
        raise ValueError(
            f"K3 DSpark target_layer_ids must be ascending; got "
            f"{config.target_layer_ids}. The concat order is positional, so a "
            f"permuted list silently feeds context_proj the wrong features."
        )
    if len(set(config.target_layer_ids)) != len(config.target_layer_ids):
        raise ValueError(
            f"K3 DSpark target_layer_ids must be distinct; got {config.target_layer_ids}."
        )
    max_tap = max(config.target_layer_ids)
    if max_tap >= config.target_num_hidden_layers:
        raise ValueError(
            f"K3 DSpark target_layer_ids max {max_tap} is out of range for a "
            f"{config.target_num_hidden_layers}-layer target."
        )
    if config.mask_token_id is None:
        raise ValueError(
            "K3 DSpark draft config must define mask_token_id; the block's "
            "non-anchor positions are all mask tokens."
        )
    if not 0 <= config.mask_token_id < config.vocab_size:
        raise ValueError(
            f"K3 DSpark mask_token_id={config.mask_token_id} is outside "
            f"[0, {config.vocab_size})."
        )
    if config.markov_head_type not in SUPPORTED_MARKOV_HEAD_TYPES:
        raise ValueError(
            f"Unsupported K3 DSpark markov_head_type={config.markov_head_type!r}; "
            f"supported: {SUPPORTED_MARKOV_HEAD_TYPES}."
        )
    if config.aux_hidden_stream not in SUPPORTED_AUX_HIDDEN_STREAMS:
        raise ValueError(
            f"Unsupported K3 DSpark aux_hidden_stream={config.aux_hidden_stream!r}; "
            f"supported: {SUPPORTED_AUX_HIDDEN_STREAMS}."
        )
    if config.markov_rank <= 0:
        raise ValueError(
            "K3 DSpark requires markov_rank > 0; the Markov head supplies the "
            "entire intra-block token dependence."
        )
    if config.num_attention_heads % 1 or config.num_attention_heads <= 0:
        raise ValueError(
            f"K3 DSpark num_attention_heads must be positive; got "
            f"{config.num_attention_heads}."
        )
    if config.mla_use_nope:
        raise ValueError(
            "K3 DSpark draft attention is RoPE MLA (mla_use_nope=false). The "
            "checkpoint was trained with rotary keys; running it NoPE discards "
            "position information the draft depends on."
        )
    if config.mla_use_output_gate:
        raise ValueError(
            "K3 DSpark draft attention has no output gate "
            "(mla_use_output_gate=false); the checkpoint ships no gate weights."
        )
    if config.q_lora_rank <= 0:
        raise ValueError(
            "K3 DSpark draft requires a q-LoRA rank; the checkpoint stores "
            "q_a_proj/q_b_proj rather than a single q_proj."
        )

    if target_config is None:
        return
    target_hidden = getattr(target_config, "hidden_size", None)
    if target_hidden is not None and int(target_hidden) != config.target_hidden_size:
        raise ValueError(
            f"K3 DSpark target_hidden_size={config.target_hidden_size} does not "
            f"match the loaded target's hidden_size={int(target_hidden)}."
        )
    target_layers = getattr(target_config, "num_hidden_layers", None)
    if (
        target_layers is not None
        and int(target_layers) != config.target_num_hidden_layers
    ):
        raise ValueError(
            f"K3 DSpark target_num_hidden_layers={config.target_num_hidden_layers} "
            f"does not match the loaded target's {int(target_layers)} layers."
        )
    target_vocab = getattr(target_config, "vocab_size", None)
    if target_vocab is not None and int(target_vocab) != config.vocab_size:
        raise ValueError(
            f"K3 DSpark vocab_size={config.vocab_size} does not match the loaded "
            f"target's {int(target_vocab)}. The draft samples through the "
            f"target's lm_head, so the vocabularies must be identical."
        )
