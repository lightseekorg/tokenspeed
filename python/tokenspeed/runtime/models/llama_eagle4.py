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

"""EAGLE-4 draft: an EAGLE-3 draft with per-aux fc norms, a normed output fed
back as the next hidden, a sliding-window draft layer (all config-driven in
the EAGLE-3 model) and a low-rank candidate LM head. Its layer may also merge
its two inputs through ``eh_proj`` (``attn_norm``), attend without rotary
embeddings (``use_rope`` false) and route a latent squared-ReLU MoE, the
Nemotron-H block (``moe_architecture`` ``nemotron_latent_relu2``)."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import torch
from tokenspeed_kernel.ops.sampling.triton.candidate_head import gather_dot
from tokenspeed_kernel.thirdparty.cuda.eagle4_candidate import (
    eagle4_candidate_supports,
    eagle4_candidate_tail,
    eagle4_shard_topk,
)
from torch import nn
from transformers import LlamaConfig

from tokenspeed.runtime.distributed.comm_ops import all_gather
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    LogitsProcessorOutput,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.llama_eagle3 import (
    Eagle3DecoderLayer,
    Eagle3LlamaModel,
    LlamaAttention,
    LlamaForCausalLMEagle3,
    _draft_window_left,
)
from tokenspeed.runtime.models.nemotron_h import (
    NemotronHMoE,
    expert_checkpoint_loader,
    load_weight,
)
from tokenspeed.runtime.utils import add_prefix

# The one MoE an EAGLE-4 draft layer may carry: Nemotron-H's latent squared-ReLU block.
LATENT_MOE_ARCHITECTURE = "nemotron_latent_relu2"

# Draft config keys beyond LlamaConfig's; a key in neither set fails, so none is dropped silently.
_STRUCTURAL_KEYS = frozenset(
    {
        # aux-hidden-state input
        "draft_vocab_size",
        "target_hidden_size",
        "num_aux_hidden_states",
        "eagle_config",
        "share_draft_embedding_from_target",
        "fc_norm",
        "enable_fc_norm",
        "bias",
        # input fusion
        "input_fuse_type",
        "attn_norm",
        # attention
        "use_rope",
        "position_embedding_type",
        "partial_rotary_factor",
        "rope_is_neox_style",
        "sliding_window",
        "use_sliding_window",
        "layer_types",
        # output norms
        "norm_output",
        "enable_post_mlp_norm",
        # MoE
        "enable_moe",
        "moe_architecture",
        "n_routed_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "n_shared_experts",
        "moe_shared_expert_intermediate_size",
        "moe_latent_size",
        "n_group",
        "topk_group",
        "norm_topk_prob",
        "routed_scaling_factor",
        "scoring_func",
        "topk_method",
        "expert_bias_update_rate",
        "moe_router_bias_update_rate",
        # candidate LM head
        "efficient_lm_head_config",
        # retired structures, accepted only when off
        "enable_2hs",
        "enable_defa",
    }
)
# Training hyperparameters and provenance: no effect on the served structure.
_NON_STRUCTURAL_KEYS = frozenset(
    {
        "_torchspec_version",
        "load_balance_loss_alpha",
        "seq_aux_loss_alpha",
        "aux_loss_alpha",
        "moe_seq_aux_loss_coeff",
        "expert_parallel_size",
        "moe_expert_parallel_size",
        "moe_initialization",
        "moe_mtp_layer_index",
        "moe_mtp_source_revision",
        # The latent MoE routes in FP32 as trained (BF16 weights, FP32 accumulate, output and bias).
        "moe_router_dtype",
        "rope_interleave",
        "max_window_layers",
        "quantization_config",
        "torch_dtype",
    }
)
# The draft attention uses head_dim for q, k and v; a config may carry these only at that value.
_DERIVED_HEAD_DIM_KEYS = frozenset({"v_head_dim", "swa_head_dim", "swa_v_head_dim"})
_EFFICIENT_LM_HEAD_KEYS = frozenset({"type", "rank", "num_candidates", "smooth_factor"})
# The sizes and routing of a latent MoE, which its config must state.
_LATENT_MOE_KEYS = (
    "moe_latent_size",
    "n_routed_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "n_shared_experts",
    "moe_shared_expert_intermediate_size",
    "n_group",
    "topk_group",
    "norm_topk_prob",
    "routed_scaling_factor",
)
# Names the trainers give the draft layer; the model registers it as midlayer.
_LAYER_PREFIXES = (
    "model.layers.0.",
    "layers.0.",
    "model.midlayers.0.",
    "midlayers.0.",
    "model.midlayer.",
    "midlayer.",
)
_STACKED_PARAMS = (
    (".qkv_proj", ".q_proj", "q"),
    (".qkv_proj", ".k_proj", "k"),
    (".qkv_proj", ".v_proj", "v"),
    (".gate_up_proj", ".gate_proj", 0),
    (".gate_up_proj", ".up_proj", 1),
)


def _expect(config: LlamaConfig, name: str, allowed: tuple) -> None:
    """If the config sets ``name``, its value must be one of ``allowed``."""
    if not hasattr(config, name):
        return
    value = getattr(config, name)
    # `is` for booleans: 1 == True must not pass as a boolean flag.
    if isinstance(value, bool) or any(isinstance(a, bool) for a in allowed):
        ok = any(value is a for a in allowed)
    else:
        ok = value in allowed
    if not ok:
        raise ValueError(
            f"EAGLE-4 draft config {name}={value!r} is not supported "
            f"(expected one of {list(allowed)})"
        )


def validate_draft_config(config: LlamaConfig) -> None:
    """Refuse an EAGLE-4 draft config whose structure this model does not build."""
    standard = set(type(config)().to_dict())
    unknown = sorted(
        set(config.to_dict())
        - standard
        - _DERIVED_HEAD_DIM_KEYS
        - _STRUCTURAL_KEYS
        - _NON_STRUCTURAL_KEYS
    )
    if unknown:
        raise ValueError(
            f"EAGLE-4 draft config has keys this model does not implement: {unknown}"
        )
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    for name in sorted(_DERIVED_HEAD_DIM_KEYS):
        value = getattr(config, name, None)
        if value and value != head_dim:
            raise ValueError(
                f"EAGLE-4 draft config {name}={value} differs from head_dim={head_dim}"
            )
    for name in (
        "fc_norm",
        "enable_fc_norm",
        "attn_norm",
        "use_rope",
        "norm_output",
        "enable_moe",
    ):
        _expect(config, name, (True, False))
    # Not built here: the post-MLP norm and a draft-owned embedding.
    _expect(config, "enable_post_mlp_norm", (False,))
    _expect(config, "share_draft_embedding_from_target", (True,))
    for name in ("bias", "attention_bias", "mlp_bias", "enable_2hs", "enable_defa"):
        _expect(config, name, (False,))
    _expect(config, "rope_is_neox_style", (True,))
    _expect(config, "partial_rotary_factor", (None, 0, 1))
    _expect(config, "input_fuse_type", ("concat",))
    _expect(config, "num_aux_hidden_states", (3,))
    # The candidate head scores the target vocabulary.
    _expect(config, "draft_vocab_size", (None, config.vocab_size))
    # fc_norm is read (enable_fc_norm is its alias); the fc input is three draft-width aux states.
    _expect(config, "enable_fc_norm", (getattr(config, "fc_norm", False),))
    _expect(config, "target_hidden_size", (config.hidden_size,))
    # The draft layer's window comes from layer_types alone.
    use_sliding_window = getattr(config, "use_sliding_window", None)
    window_without_types = use_sliding_window or (
        use_sliding_window is None and getattr(config, "sliding_window", None)
    )
    if getattr(config, "layer_types", None) is None and window_without_types:
        raise ValueError(
            "EAGLE-4 draft config sets a sliding window without layer_types"
        )

    eagle_config = getattr(config, "eagle_config", None) or {}
    extra = set(eagle_config) - {
        "use_aux_hidden_state",
        "eagle_aux_hidden_state_layer_ids",
    }
    layer_ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")
    if extra or (layer_ids is not None and len(layer_ids) != 3):
        raise ValueError(
            f"EAGLE-4 draft eagle_config={eagle_config!r} is not supported: the fc "
            "input takes exactly three aux hidden states"
        )

    # The type names how the shipped head weights were factored; the loader requires those weights.
    head = getattr(config, "efficient_lm_head_config", None)
    if (
        not isinstance(head, dict)
        or head.get("type") not in ("low_rank", "weighted_low_rank")
        or set(head) - _EFFICIENT_LM_HEAD_KEYS
    ):
        raise ValueError(
            f"EAGLE-4 draft efficient_lm_head_config={head!r} is not supported"
        )

    if not getattr(config, "enable_moe", False):
        _expect(config, "hidden_act", ("silu",))
        return
    _expect(config, "moe_architecture", (LATENT_MOE_ARCHITECTURE,))
    _expect(config, "hidden_act", ("relu2",))
    # Nemotron-H routing: sigmoid, top-k over bias-corrected scores, renormalized unbiased scores.
    _expect(config, "scoring_func", ("sigmoid",))
    _expect(config, "topk_method", ("noaux_tc",))
    _expect(config, "norm_topk_prob", (True,))
    _expect(config, "n_shared_experts", (1,))
    missing = [
        name
        for name in ("moe_architecture", *_LATENT_MOE_KEYS)
        if getattr(config, name, None) is None
    ]
    if missing:
        raise ValueError(f"EAGLE-4 latent MoE draft config is missing {missing}")


def draft_uses_rope(config: LlamaConfig) -> bool:
    """Whether the draft layer rotates q and k: not when ``use_rope`` is false,
    ``position_embedding_type`` is "none" or ``partial_rotary_factor`` is 0."""
    return not (
        getattr(config, "use_rope", True) is False
        or getattr(config, "position_embedding_type", None) == "none"
        or getattr(config, "partial_rotary_factor", None) == 0
    )


def _module_name(checkpoint_name: str) -> str:
    """A draft checkpoint tensor's name in the module tree."""
    for prefix in _LAYER_PREFIXES:
        if checkpoint_name.startswith(prefix):
            return "model.midlayer." + checkpoint_name[len(prefix) :]
    if checkpoint_name.startswith(("model.", "lm_head.", "efficient_lm_head")):
        return checkpoint_name
    return "model." + checkpoint_name


@dataclasses.dataclass(frozen=True)
class LatentMoEConfig:
    """The fields ``NemotronHMoE`` reads, from a latent-MoE draft config."""

    hidden_size: int
    mlp_hidden_act: str
    mlp_bias: bool
    moe_latent_size: int
    n_routed_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    moe_shared_expert_intermediate_size: int
    n_group: int
    topk_group: int
    routed_scaling_factor: float
    norm_topk_prob: bool

    @classmethod
    def from_draft_config(cls, config: LlamaConfig) -> LatentMoEConfig:
        """Read a config ``validate_draft_config`` accepted (one shared expert)."""
        return cls(
            hidden_size=config.hidden_size,
            mlp_hidden_act=config.hidden_act,
            mlp_bias=False,
            moe_latent_size=config.moe_latent_size,
            n_routed_experts=config.n_routed_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            moe_intermediate_size=config.moe_intermediate_size,
            moe_shared_expert_intermediate_size=(
                config.moe_shared_expert_intermediate_size
            ),
            n_group=config.n_group,
            topk_group=config.topk_group,
            routed_scaling_factor=float(config.routed_scaling_factor),
            norm_topk_prob=config.norm_topk_prob,
        )


class LatentMoEMLP(NemotronHMoE):
    """The latent MoE of an EAGLE-4 draft layer: the Nemotron-H block in BF16.

    Every rank holds every expert and splits their intermediate size over the
    dense TP group, so the routed and shared halves are partial sums over the
    group a dense MLP's down projection reduces over, and the draft layer
    reduces their sum the same way.
    """

    def __init__(
        self, config: LlamaConfig, mapping: Mapping, layer_index: int, prefix: str
    ) -> None:
        if (
            mapping.moe.ep_size != 1
            or mapping.moe.tp_size != mapping.dense.tp_size
            or mapping.moe.tp_group != mapping.dense.tp_group
        ):
            raise NotImplementedError(
                "a latent MoE draft splits its experts over the dense TP group "
                f"without expert parallelism; got moe tp={mapping.moe.tp_size} "
                f"ep={mapping.moe.ep_size}, dense tp={mapping.dense.tp_size}"
            )
        super().__init__(
            LatentMoEConfig.from_draft_config(config),
            mapping,
            layer_index,
            None,
            prefix,
            torch.cuda.Stream(),
        )
        # The draft layer's post-attention norm quantizes nothing for it.
        self.gateup_unquanted = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_scale: torch.Tensor | None,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        """This rank's partial output, routed plus shared; ``block_scale`` is
        None since the layer quantizes nothing ahead of an unquantized MLP."""
        first, second = super().forward(hidden_states, None, ctx)
        return first if second is None else first + second


class Eagle4Attention(LlamaAttention):
    """The draft attention, without rotary embeddings when the config says so."""

    def resolve_rotary_emb(self, config: LlamaConfig) -> nn.Module | None:
        return super().resolve_rotary_emb(config) if draft_uses_rope(config) else None


class Eagle4DecoderLayer(Eagle3DecoderLayer):
    """The EAGLE-3 draft layer plus the EAGLE-4 structures its config selects:
    ``attn_norm`` (``eh_proj`` maps ``[norm(embeds) || norm(hidden)]`` to the
    residual, whose norm feeds q/k/v), NoPE and the latent MoE."""

    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, layer_id, mapping, quant_config, prefix)
        self.eh_proj: ReplicatedLinear | None = None
        self.pre_attention_layernorm: RMSNorm | None = None
        if getattr(config, "attn_norm", False):
            self.eh_proj = ReplicatedLinear(
                2 * config.hidden_size,
                config.hidden_size,
                bias=False,
                quant_config=None,
                prefix=add_prefix("eh_proj", prefix),
            )
            self.pre_attention_layernorm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

    def resolve_attn(self, prefix: str) -> nn.Module:
        config = self._eagle3_config
        return Eagle4Attention(
            config,
            self._eagle3_mapping,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=self.layer_id,
            quant_config=self._eagle3_quant_config,
            prefix=add_prefix("self_attn", prefix),
            qkv_input_size=(
                config.hidden_size
                if getattr(config, "attn_norm", False)
                else 2 * config.hidden_size
            ),
            sliding_window_size=_draft_window_left(config),
        )

    def resolve_mlp(self, prefix: str) -> nn.Module:
        config = self._eagle3_config
        if getattr(config, "enable_moe", False):
            return LatentMoEMLP(
                config, self._eagle3_mapping, self.layer_id, add_prefix("mlp", prefix)
            )
        return super().resolve_mlp(prefix)

    def _fuse_input(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.eh_proj is None:
            return hidden_states, residual
        residual, _ = self.eh_proj(hidden_states)
        if residual.shape[0] == 0:
            return residual, residual
        return self.pre_attention_layernorm(residual), residual


class Eagle4LlamaModel(Eagle3LlamaModel):
    """The EAGLE-3 draft model body over EAGLE-4 decoder layers."""

    layer_cls = Eagle4DecoderLayer


@dataclasses.dataclass(frozen=True)
class CandidateHeads:
    """The three weights of the low-rank candidate head."""

    lm_head: ParallelLMHead  # exact logits of the candidates
    scores_head: ParallelLMHead  # rank-r token embeddings that pick them
    down_proj: ReplicatedLinear  # hidden -> rank r
    num_candidates: int


@dataclasses.dataclass
class Eagle4LogitsOutput(LogitsProcessorOutput):
    """The candidate head's output: logits column ``j`` of a row is the token
    ``candidate_ids[row, j]``; ``draft_topk`` holds each row's best draft
    probabilities and their token ids (``[rows, draft_top_k]`` each)."""

    candidate_ids: torch.Tensor | None = None
    draft_topk: tuple[torch.Tensor, torch.Tensor] | None = None


class Eagle4CandidateLogitsProcessor(LogitsProcessor):
    """Each sampled row's logits over its ``num_candidates`` best tokens.

    EAGLE-4's candidate head: rank-r scores over this rank's vocab shard
    (bf16), the shard's top ``num_candidates`` (unsorted), their exact logits
    from the draft LM head (bf16), then under TP the top ``num_candidates`` of
    the rank-major gathered pools; logits widen to fp32. Output column ``j``
    of a row is the token ``candidate_ids[row, j]``. With ``draft_top_k`` the
    output also carries each row's best ``draft_top_k`` softmax probabilities
    over its candidates and their tokens (``draft_topk``).

    The fused kernels (one shard top-k launch, the gather-dot, one packed
    all-gather of logits and ids, one launch for the rest) reproduce those
    torch ops bit for bit; the model refuses a head they do not cover.
    """

    def __init__(
        self, config: LlamaConfig, *, tp_rank: int, tp_size: int, tp_group
    ) -> None:
        super().__init__(
            config,
            do_argmax=True,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            dp_lm_head_tp=False,
        )
        self._candidate_ids: torch.Tensor | None = None
        self._draft_topk: tuple[torch.Tensor, torch.Tensor] | None = None
        # Best probabilities per row each forward also returns (draft_topk; 0: none).
        self.draft_top_k = 0

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: CandidateHeads,
        logits_metadata: LogitsMetadata,
        embedding_bias: torch.Tensor | None = None,
        plan=None,
        *,
        require_full_vocab: bool,
    ) -> torch.Tensor:
        if require_full_vocab or plan is not None or embedding_bias is not None:
            raise RuntimeError("the candidate head scores sampled draft rows only")
        nc = lm_head.num_candidates
        weight = lm_head.lm_head.weight
        hidden = hidden_states.to(weight.dtype)
        low_rank, _ = lm_head.down_proj(hidden)
        scores = torch.matmul(low_rank, lm_head.scores_head.weight.T)
        rows = hidden.shape[0]
        vocab_start = lm_head.lm_head.shard_indices.org_vocab_start_index
        # Per rank: nc bf16 logits, then nc int64 ids as raw bytes, so one all-gather moves both.
        block = torch.empty((rows, 5 * nc), dtype=weight.dtype, device=hidden.device)
        global_ids = block.view(torch.int64)[:, nc // 4 : nc // 4 + nc]
        eagle4_shard_topk(scores, nc, vocab_start, global_ids)
        gather_dot(weight, global_ids, hidden, vocab_start, out=block[:, :nc])
        if self.tp_size > 1:
            block = all_gather(block, self.tp_group)
        # Rank r's candidates: columns r * 5 * nc onward of the gathered rows.
        ranks, width = block.shape[1] // (5 * nc), 5 * nc
        cand_logits = block.as_strided(
            (ranks, rows, nc), (width, block.stride(0), 1), block.storage_offset()
        )
        ids = block.view(torch.int64)
        cand_ids = ids.as_strided(
            (ranks, rows, nc),
            (width // 4, ids.stride(0), 1),
            ids.storage_offset() + nc // 4,
        )

        top_k = self.draft_top_k
        logits = torch.empty((rows, nc), dtype=torch.float32, device=hidden.device)
        tokens = (
            torch.empty((rows, nc), dtype=torch.int64, device=hidden.device)
            if ranks > 1
            else cand_ids[0]
        )
        draft_topk = (
            (
                torch.empty((rows, top_k), dtype=torch.float32, device=hidden.device),
                torch.empty((rows, top_k), dtype=torch.int64, device=hidden.device),
            )
            if top_k
            else None
        )
        eagle4_candidate_tail(
            cand_logits, cand_ids, top_k, logits, tokens, *(draft_topk or (None, None))
        )
        self._candidate_ids = tokens
        self._draft_topk = draft_topk
        return logits

    def _dist_argmax_eligible(self) -> bool:
        # The gathered candidate logits are whole on every rank, so argmax stays local.
        return False

    def _argmax(self, logits: torch.Tensor) -> torch.Tensor:
        best = super()._argmax(logits)
        return self._candidate_ids.gather(-1, best.view(-1, 1).long()).view(-1)

    def forward(self, *args, **kwargs) -> Eagle4LogitsOutput:
        output = super().forward(*args, **kwargs)
        return Eagle4LogitsOutput(
            **vars(output),
            candidate_ids=self._candidate_ids,
            draft_topk=self._draft_topk,
        )


class LlamaForCausalLMEagle4(LlamaForCausalLMEagle3):
    """The EAGLE-4 draft: an EAGLE-3 draft with the low-rank candidate head."""

    model_cls = Eagle4LlamaModel

    def __init__(
        self,
        config: LlamaConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        validate_draft_config(config)
        super().__init__(config, mapping, quant_config, prefix)
        head = config.efficient_lm_head_config
        self.efficient_lm_head_down_proj = ReplicatedLinear(
            config.hidden_size,
            head["rank"],
            bias=False,
            prefix=add_prefix("efficient_lm_head_down_proj", prefix),
        )
        self.efficient_lm_head = ParallelLMHead(
            config.vocab_size,
            head["rank"],
            tp_rank=mapping.lm_head.tp_rank,
            tp_size=mapping.lm_head.tp_size,
            tp_group=mapping.lm_head.tp_group,
            prefix=add_prefix("efficient_lm_head", prefix),
        )
        self.num_candidates = head["num_candidates"]
        # A padded shard's zero rows could outscore every real token's.
        shard_vocab = config.vocab_size // mapping.lm_head.tp_size
        if any(
            h.weight.shape[0] * mapping.lm_head.tp_size != config.vocab_size
            for h in (self.lm_head, self.efficient_lm_head)
        ):
            raise ValueError(
                "the candidate head scores unpadded vocab shards: vocab_size "
                f"{config.vocab_size} must split into {mapping.lm_head.tp_size} "
                "shards the LM heads do not pad"
            )
        if not eagle4_candidate_supports(
            self.lm_head.weight.dtype,
            shard_vocab,
            self.num_candidates,
            mapping.lm_head.tp_size,
        ):
            raise ValueError(
                f"the candidate head kernels do not cover {self.num_candidates} "
                f"candidates over {mapping.lm_head.tp_size} {self.lm_head.weight.dtype} "
                f"vocab shards of {shard_vocab}"
            )
        self.logits_processor = Eagle4CandidateLogitsProcessor(
            config,
            tp_rank=mapping.lm_head.tp_rank,
            tp_size=mapping.lm_head.tp_size,
            tp_group=mapping.lm_head.tp_group,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Load the draft checkpoint: every tensor must reach a parameter and
        every parameter not shared from the target must be filled.

        A latent MoE loads like the Nemotron-H block: ``experts.N.up_proj``
        fills all of the non-gated w13 and every expert must arrive.
        """
        params = dict(self.named_parameters())
        latent_moe = isinstance(self.model.midlayer.mlp, LatentMoEMLP)
        experts = (
            expert_checkpoint_loader(params, self.config, self.mapping)
            if latent_moe
            else None
        )
        expert_tensors: set[str] = set()
        # (parameter, shard): a fused parameter is filled once each of its shards arrived.
        loaded: set[tuple[str, str | int | None]] = set()
        for checkpoint_name, tensor in weights:
            name = _module_name(checkpoint_name)
            if experts is not None and name.startswith("model.midlayer.mlp."):
                if experts.matches(name):
                    if name in expert_tensors:
                        raise ValueError(f"draft checkpoint repeats {name!r}")
                    expert_tensors.add(name)
                else:
                    loaded.add((name, None))
                load_weight(params, experts, name, tensor)
                continue
            shard_id = None
            for param_suffix, weight_suffix, stacked_id in _STACKED_PARAMS:
                if weight_suffix in name:
                    name = name.replace(weight_suffix, param_suffix)
                    shard_id = stacked_id
                    break
            if name not in params:
                raise KeyError(
                    f"draft checkpoint tensor {checkpoint_name!r} has no parameter"
                )
            param = params[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if shard_id is None:
                weight_loader(param, tensor)
            else:
                weight_loader(param, tensor, shard_id)
            loaded.add((name, shard_id))
        # set_embed_and_head later supplies these from the target (head only without a draft one).
        shared = {"model.embed_tokens.weight"}
        if self.load_lm_head_from_target:
            shared.add("lm_head.weight")
        required = set()
        for name in params:
            if name in shared or (
                latent_moe and name.startswith("model.midlayer.mlp.experts.")
            ):
                continue
            shards = [s for suffix, _, s in _STACKED_PARAMS if suffix in name]
            required.update((name, shard) for shard in shards or [None])
        missing = sorted(f"{name} ({shard})" for name, shard in required - loaded)
        if missing:
            raise ValueError(f"draft checkpoint lacks {missing}")
        if latent_moe and len(expert_tensors) != 2 * self.config.n_routed_experts:
            raise ValueError(
                f"draft checkpoint has {len(expert_tensors)} expert tensors, expected "
                f"up_proj and down_proj of {self.config.n_routed_experts} experts"
            )

    def exit_logits(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        ctx: ForwardContext,
    ) -> LogitsProcessorOutput:
        heads = CandidateHeads(
            lm_head=self.lm_head,
            scores_head=self.efficient_lm_head,
            down_proj=self.efficient_lm_head_down_proj,
            num_candidates=self.num_candidates,
        )
        return self.logits_processor(
            input_ids,
            hidden_states,
            heads,
            LogitsMetadata.from_forward_context(ctx),
            aux_hidden_states,
        )


EntryClass = [LlamaForCausalLMEagle4]
