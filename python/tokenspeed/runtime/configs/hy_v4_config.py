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

"""Configuration for HY V4 (``HYV4ForCausalLM``).

HY V4 is a MoE causal language model built on Multi-head Latent Attention (MLA)
with DeepSeek-style sparse attention (DSA), plus four features that are not in
the GLM/DSA family: a gated MLA output projection, per-head learnable attention
sinks, independent Hyper-Connections (iHC) instead of a single residual stream,
and a clamped SwiGLU on the routed experts. The checkpoint is a pure text model,
so this class is the top-level config with no nested ``text_config``.
"""

from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig

_SPARSE_ATTENTION_LAYER_TYPE = "deepseek_sparse_attention"
_SPARSE_ATTENTION_ALIASES = ("sparse_attention", "sparse")


def _normalized_rope_fields(fields: dict | None) -> dict:
    """Copy of ``fields`` with the legacy ``type`` key spelled ``rope_type``."""
    return {
        ("rope_type" if key == "type" else key): value
        for key, value in (fields or {}).items()
    }


class HYV4Config(PretrainedConfig):
    """HY V4 text configuration.

    Field defaults mirror ``tencent/Hy4-preview``. ``mlp_layer_types`` and
    ``indexer_types`` are declared per layer (not derived from
    ``first_k_dense_replace`` / ``index_topk_freq``), because the released
    checkpoint pins an explicit vector for each; the derivation below only fills
    in the pattern the reference config would have produced when a checkpoint
    omits the field.
    """

    model_type = "hy_v4"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "n_routed_experts"}

    def __init__(
        self,
        vocab_size: int = 120832,
        hidden_size: int = 6144,
        intermediate_size: int = 18432,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 78,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        head_dim: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1048576,
        initializer_range: float = 0.006,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int | None = 120002,
        bos_token_id: int | None = 120000,
        eos_token_id: int | list[int] | None = 120025,
        tie_word_embeddings: bool = False,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        n_routed_experts: int = 256,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 8,
        routed_scaling_factor: float = 2.827,
        norm_topk_prob: bool = True,
        n_group: int = 1,
        topk_group: int = 1,
        q_lora_rank: int = 2048,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 192,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 256,
        mlp_layer_types: list[str] | None = None,
        layer_types: list[str] | None = None,
        index_topk: int = 2048,
        index_head_dim: int = 128,
        index_n_heads: int = 32,
        indexer_types: list[str] | None = None,
        enable_lm_head_fp32: bool = True,
        enable_ihc: bool = True,
        hc_mult: int = 4,
        hc_magnitude: float = 2.0,
        hc_eps: float = 1e-6,
        gated_mla: bool = True,
        gating_type: str = "elementwise",
        learnable_sink: bool = True,
        learnable_sink_init: float = 0.0,
        swiglu_limit: float = 10.0,
        rope_parameters: dict | None = None,
        num_nextn_predict_layers: int = 1,
        mtp_loss_factor: float = 0.1,
        use_dsa: bool = True,
        use_mla: bool = True,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.n_group = n_group
        self.topk_group = topk_group
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.enable_lm_head_fp32 = enable_lm_head_fp32
        self.enable_ihc = enable_ihc
        self.hc_mult = hc_mult
        self.hc_magnitude = hc_magnitude
        self.hc_eps = hc_eps
        self.gated_mla = gated_mla
        self.gating_type = gating_type
        self.learnable_sink = learnable_sink
        self.learnable_sink_init = learnable_sink_init
        self.swiglu_limit = swiglu_limit
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_loss_factor = mtp_loss_factor
        self.use_dsa = use_dsa
        self.use_mla = use_mla
        # Read by the shared DeepSeek MoE gate: HY V4 routes with sigmoid scores
        # plus the ``e_score_correction_bias`` the checkpoint ships.
        self.topk_method = "noaux_tc"
        self.scoring_func = "sigmoid"
        # Shared DeepSeek plumbing reads this; which layers are MoE is decided by
        # ``mlp_layer_types``, so every layer is eligible.
        self.moe_layer_freq = 1
        # Read by the shared GLM-DSA indexer: HY V4's indexer RoPE is
        # interleaved (Megatron layout). NeoX rotation would drop the relative
        # position dependence and corrupt the sparse top-k selection.
        self.indexer_rope_interleave = True

        if gating_type not in ("elementwise", "headwise"):
            raise ValueError(
                f"HYV4 MLA gate granularity must be 'elementwise' or 'headwise', "
                f"got {gating_type!r}"
            )
        if hc_mult <= 1:
            raise ValueError(
                f"HYV4 iHC requires more than one residual channel, got {hc_mult}"
            )
        if enable_ihc and not learnable_sink:
            # Both are architectural; a checkpoint that disables only the sink
            # would silently change attention, so require them together.
            raise ValueError("HYV4 iHC requires the learnable attention sink")

        # The released checkpoint declares both vectors explicitly. Derive the
        # same pattern only when a checkpoint omits them, matching the reference
        # config's own defaults (one dense layer, then sparse; a full indexer on
        # layer 0 and every fourth layer after it).
        self.mlp_layer_types = (
            list(mlp_layer_types)
            if mlp_layer_types is not None
            else ["dense"] * min(1, num_hidden_layers)
            + ["sparse"] * max(num_hidden_layers - 1, 0)
        )
        self.indexer_types = (
            list(indexer_types)
            if indexer_types is not None
            else [
                "full" if layer_id == 0 or (layer_id - 1) % 4 == 0 else "shared"
                for layer_id in range(num_hidden_layers)
            ]
        )
        if len(self.mlp_layer_types) != num_hidden_layers:
            raise ValueError(
                "mlp_layer_types must contain one entry per hidden layer: "
                f"expected {num_hidden_layers}, got {len(self.mlp_layer_types)}"
            )
        if len(self.indexer_types) != num_hidden_layers:
            raise ValueError(
                "indexer_types must contain one entry per hidden layer: "
                f"expected {num_hidden_layers}, got {len(self.indexer_types)}"
            )
        num_dense = self.mlp_layer_types.count("dense")
        if self.mlp_layer_types != ["dense"] * num_dense + ["sparse"] * (
            num_hidden_layers - num_dense
        ):
            # The shared DeepSeek MoE plumbing takes the dense layers to be the
            # leading ``first_k_dense_replace`` layers.
            raise ValueError(
                "HYV4 mlp_layer_types must list the 'dense' layers first and only "
                f"'sparse' layers after them, got {self.mlp_layer_types}"
            )
        self.layer_types = (
            [
                (
                    _SPARSE_ATTENTION_LAYER_TYPE
                    if layer_type in _SPARSE_ATTENTION_ALIASES
                    else layer_type
                )
                for layer_type in layer_types
            ]
            if layer_types is not None
            else [_SPARSE_ATTENTION_LAYER_TYPE] * num_hidden_layers
        )
        if len(self.layer_types) != num_hidden_layers:
            raise ValueError(
                "layer_types must contain one entry per hidden layer: "
                f"expected {num_hidden_layers}, got {len(self.layer_types)}"
            )
        unsupported = sorted(
            set(self.layer_types) - {_SPARSE_ATTENTION_LAYER_TYPE}, key=str
        )
        if unsupported:
            # Every layer is built as DSA; a dense-MLA layer would silently run
            # sparse attention once the context passes ``index_topk``.
            raise ValueError(
                f"HYV4 supports only sparse attention layers, got {unsupported}"
            )

        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        # RoPE covers only the rope slice, so ``head_dim`` points at it and the
        # shared rotary embedding computes the right frequencies unmodified.
        if head_dim is not None and head_dim != qk_rope_head_dim:
            raise ValueError(
                f"HYV4 head_dim must equal qk_rope_head_dim ({qk_rope_head_dim}), "
                f"got {head_dim}"
            )
        self.head_dim = qk_rope_head_dim

        # ``rope_scaling`` is the legacy spelling of ``rope_parameters``.
        # ``PretrainedConfig`` would let a non-empty one replace the validated
        # dict, so the two are merged and checked here.
        rope_fields = _normalized_rope_fields(rope_parameters)
        legacy_fields = _normalized_rope_fields(kwargs.pop("rope_scaling", None))
        for key, value in legacy_fields.items():
            if rope_fields.setdefault(key, value) != value:
                raise ValueError(
                    f"HYV4 rope_scaling and rope_parameters disagree on {key!r}: "
                    f"{value!r} vs {rope_fields[key]!r}"
                )
        rope_fields.setdefault("rope_type", "default")
        rope_fields.setdefault("rope_theta", 10000000)
        if rope_fields["rope_type"] != "default":
            raise ValueError(
                f"HYV4 uses plain RoPE; got rope_type {rope_fields['rope_type']!r}"
            )
        self.rope_parameters = rope_fields
        self.rope_theta = float(rope_fields["rope_theta"])
        # ``enable_lm_head_fp32`` surfaces as ``head_dtype`` so the logits
        # processor accumulates the head GEMM in FP32 with a BF16 weight.
        self.head_dtype: str | None = "float32" if enable_lm_head_fp32 else None

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def rope_scaling(self) -> dict | None:
        """DeepSeek-family consumers read ``rope_scaling``; HY V4 has none."""
        return None

    def is_moe_layer(self, layer_id: int) -> bool:
        return self.mlp_layer_types[layer_id] == "sparse"

    @property
    def first_k_dense_replace(self) -> int:
        """Dense-layer prefix length, derived from the per-layer MLP vector.

        The shared MoE plumbing reads this field; HY V4 pins the vector instead,
        so recover the prefix length from it rather than duplicating the count.
        """
        return sum(1 for layer_type in self.mlp_layer_types if layer_type == "dense")
